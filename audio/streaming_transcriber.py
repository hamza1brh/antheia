from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Tuple

import numpy as np

logging.getLogger("faster_whisper").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

# ------------------ Repeat-guard knobs  ------------------
REPEAT_SENT_THRESHOLD = 3       # same committed sentence seen >= 3 times recently
REPEAT_HYP_STALL = 3            # hypothesis identical for the last 3 iterations
REPEAT_NGRAM_N = 6              # n-gram length to compare between commit/hyp tail
REPEAT_JUMP_SEC = 0.35          # skip-ahead when stuck (seconds)

# ------------------ Language-probe knobs (fast, stable switching) --------
LANG_MIN_WINDOW_SEC = 1.5       # duration of tail audio used for detection
LANG_SWITCH_DELTA   = 0.15      # new_prob must exceed current by at least this
LANG_SWITCH_PROB    = 0.70      # absolute confidence required to accept a language
LANG_DWELL_ITERS    = 2         # consecutive detections needed to confirm switch
LANG_COOLDOWN_SEC   = 10.0      # cooldown after a switch before another switch

# ------------------------------------------------------------------------------
#                           Faster-Whisper Backend
# ------------------------------------------------------------------------------
class FasterWhisperBackend:
    """
    Thin wrapper around faster-whisper with streaming-friendly defaults.
    - Low-latency: greedy, no word timestamps
    - Safety: temperature fallback + anti-hallucination thresholds
    - Robust language detection across faster-whisper versions
    """
    sep = ""  

    def __init__(self, model_size: str = "small", device: str = "cuda",
                 language: Optional[str] = None):
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        self._detected_lang: Optional[str] = None
        self._detected_prob: float = 0.0
        self._lang_locked: bool = language is not None

        # Live defaults + built-in guards against silence hallucinations.
        self.kw = dict(
            word_timestamps=False,             # big speed win in live path
            condition_on_previous_text=True,   # prompt on; toggled off briefly when stuck/switching
            beam_size=1,                       # greedy for latency
            temperature=[0.0, 0.2, 0.4],       # fallback schedule to escape low-confidence stalls
            compression_ratio_threshold=2.4,   # suppress repetitive junk
            log_prob_threshold=-1.0,           # suppress low-confidence text
            no_speech_threshold=0.6,           # skip output on strong silence
            task="transcribe",
        )

    # returns may be (lang, prob_float), (lang, probs_dict), or probs_dict
    def maybe_detect_language(self, audio_16k_mono: np.ndarray):
        if self._lang_locked or audio_16k_mono.shape[0] < 3 * 16000:
            return
        res = self.model.detect_language(audio_16k_mono)
        lang = None
        conf = 0.0
        if isinstance(res, tuple) and len(res) == 2:
            lang = res[0]
            second = res[1]
            conf = float(second.get(lang, 0.0)) if isinstance(second, dict) else float(second)
        elif isinstance(res, dict) and res:
            lang = max(res, key=res.get)
            conf = float(res[lang])
        if lang and (conf >= LANG_SWITCH_PROB or self._detected_lang is None):
            self._detected_lang, self._detected_prob = lang, conf
            self.language = lang
            self._lang_locked = True

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> List[Any]:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=init_prompt,
            **self.kw,
        )
        return list(segments)

    @staticmethod
    def ts_words(segments) -> List[Tuple[float, float, str]]:
        """
        Convert segments to (start, end, token/text) units for local-agreement.
        With word_timestamps=False, fall back to using whole segment text.
        """
        out: List[Tuple[float, float, str]] = []
        for seg in segments:
            if getattr(seg, "no_speech_prob", 0.0) >= 0.99:
                # strong silence; skip
                continue
            words = getattr(seg, "words", None)
            if words:
                for w in words or []:
                    out.append((w.start, w.end, w.word))
            else:
                out.append((seg.start, seg.end, seg.text))
        return out

    @staticmethod
    def segments_end_ts(segments) -> List[float]:
        return [s.end for s in segments]


# ------------------------------------------------------------------------------
#                             Local Agreement Buffer
# ------------------------------------------------------------------------------
class HypothesisBuffer:
    """
    Commits only the stable prefix shared with the previous hypothesis (local agreement).
    """
    def __init__(self):
        self.commited_in_buffer: List[Tuple[float, float, str]] = []
        self.buffer: List[Tuple[float, float, str]] = []
        self.new: List[Tuple[float, float, str]] = []
        self.last_commited_time: float = 0.0

    def insert(self, new_words: List[Tuple[float, float, str]], offset: float):
        # to absolute stream time, drop before last commit (tiny tolerance)
        shifted = [(a + offset, b + offset, t) for a, b, t in new_words]
        self.new = [(a, b, t) for (a, b, t) in shifted if a > self.last_commited_time - 0.1]

        # short n-gram overlap guard (remove repeated head matching committed tail)
        if self.new and self.commited_in_buffer:
            cn = len(self.commited_in_buffer)
            nn = len(self.new)
            for n in range(1, min(min(cn, nn), 5) + 1):
                tail = " ".join(self.commited_in_buffer[-j][2] for j in range(1, n + 1))
                head = " ".join(self.new[j - 1][2] for j in range(1, n + 1))
                if tail == head:
                    for _ in range(n):
                        self.new.pop(0)
                    break

    def flush(self) -> List[Tuple[float, float, str]]:
        commit: List[Tuple[float, float, str]] = []
        while self.new and self.buffer:
            na, nb, nt = self.new[0]
            _, _, bt = self.buffer[0]
            if nt == bt:
                commit.append((na, nb, nt))
                self.last_commited_time = nb
                self.new.pop(0)
                self.buffer.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time_sec: float):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time_sec:
            self.commited_in_buffer.pop(0)

    def complete(self) -> List[Tuple[float, float, str]]:
        return self.buffer


# ------------------------------------------------------------------------------
#                            Online ASR Processor
# ------------------------------------------------------------------------------
class OnlineASRProcessor:
    """
    Single-stream processor with:
      - Local Agreement stable commit
      - Segment-aware trimming
      - Repeat watchdog (suppresses loops, realigns buffer, temp relaxed decode)
      - Fast language switching via tail-probe + hysteresis (no external VAD)
    """
    SAMPLING_RATE = 16000

    def __init__(self, model_size: str = "small", device: str = "cuda",
                 language: Optional[str] = None,
                 buffer_trimming_sec: float = 12.0,
                 lang_probe_sec: float = 3.5,
                 word_timestamps_live: bool = False):
        self.asr = FasterWhisperBackend(model_size=model_size, device=device, language=language)
        if word_timestamps_live:
            self.asr.kw["word_timestamps"] = True  # optional, slower

        # audio buffer and time bookkeeping
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0.0
        self.buffer_trimming_sec = buffer_trimming_sec

        # transcript state
        self.transcript_buffer = HypothesisBuffer()
        self.commited: List[Tuple[float, float, str]] = []

        # language detection state
        self.lang_probe_sec = lang_probe_sec        # legacy first-probe budget
        self._lang_locked_once = False              # legacy gate for first detect
        self._lang_current_prob = 0.0
        self._lang_dwell_count = 0
        self._last_lang_switch_time = -1e9          # long ago

        # repeat watchdog
        self._recent_hyps: List[str] = []
        self._recent_commits: List[str] = []
        self._unstick_cooldown = 0

    # ---------------------- Public API ----------------------
    def allow_next_lang_probe(self):
        """Optional: caller may allow another initial-probe style detect."""
        self._lang_locked_once = False

    def insert_audio_chunk(self, audio: np.ndarray):
        if audio is None or audio.size == 0:
            return
        self.audio_buffer = np.append(self.audio_buffer, audio.astype(np.float32, copy=False))

        # legacy "first big probe" for bootstrapping (3.5s by default)
        if (not self._lang_locked_once
                and self.audio_buffer.size >= int(self.lang_probe_sec * self.SAMPLING_RATE)):
            self.asr.maybe_detect_language(self.audio_buffer)
            if self.asr._lang_locked:
                self._lang_locked_once = True
                self._lang_current_prob = getattr(self.asr, "_detected_prob", 0.0)

        # fast tail-based probing with hysteresis (reacts to actual switches)
        self._maybe_lang_probe_tail()

    def process_iter(self) -> Tuple[Optional[str], str]:
        if self.audio_buffer.size == 0:
            return None, ""

        prompt, _ = self._build_prompt()
        segments = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
        tsw = self.asr.ts_words(segments)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        committed_words = self.transcript_buffer.flush()
        self.commited.extend(committed_words)

        committed_text = self._join_words(committed_words) if committed_words else None
        hypothesis_text = self._join_words(self.transcript_buffer.complete())

        # histories
        self._recent_hyps.append(hypothesis_text or "")
        if len(self._recent_hyps) > 6:
            self._recent_hyps.pop(0)
        if committed_text:
            ct = committed_text.strip()
            if ct:
                self._recent_commits.append(ct)
                if len(self._recent_commits) > 8:
                    self._recent_commits.pop(0)

        # cooldown restore after an unstick
        if self._unstick_cooldown > 0:
            self._unstick_cooldown -= 1
            if self._unstick_cooldown == 0:
                # restore fast defaults
                self.asr.kw["beam_size"] = 1
                self.asr.kw["temperature"] = [0.0, 0.2, 0.4]
                self.asr.kw["condition_on_previous_text"] = True

        # repeat/loop detection → suppress emission + realign
        if self._repeating_now(committed_text, hypothesis_text):
            if committed_text and self._recent_commits:
                self._recent_commits.pop()  # don't emit the repeated line
            self._force_realign(segments)
            return None, ""  # swallow this tick

        # segment-aware trimming
        if (self.audio_buffer.size / self.SAMPLING_RATE) > self.buffer_trimming_sec:
            self._chunk_completed_segment(segments)

        # after any language switch, we temporarily disabled conditioning; restore here after one pass
        if self.asr.kw.get("condition_on_previous_text") is False and self._unstick_cooldown == 0:
            self.asr.kw["condition_on_previous_text"] = True

        return committed_text, hypothesis_text

    def finish(self) -> Tuple[Optional[str], str]:
        hyp = self._join_words(self.transcript_buffer.complete())
        self.buffer_time_offset += self.audio_buffer.size / self.SAMPLING_RATE
        return None, hyp

    # ---------------------- Internals ----------------------
    def _tail_audio(self, seconds: float) -> np.ndarray:
        n = int(seconds * self.SAMPLING_RATE)
        if self.audio_buffer.size <= n:
            return self.audio_buffer
        return self.audio_buffer[-n:]

    def _maybe_lang_probe_tail(self):
        """
        Fast, stable language switching:
          - probe on the tail (LANG_MIN_WINDOW_SEC)
          - require absolute prob and margin over current
          - dwell across a couple iterations
          - cooldown after a switch to avoid flapping
          - one safe iteration with no previous-text conditioning
        """
        now = time.time()
        if (now - self._last_lang_switch_time) < LANG_COOLDOWN_SEC:
            return
        if self.audio_buffer.size < int(LANG_MIN_WINDOW_SEC * self.SAMPLING_RATE):
            return

        tail = self._tail_audio(LANG_MIN_WINDOW_SEC)
        res = self.asr.model.detect_language(tail)

        if isinstance(res, tuple) and len(res) == 2:
            new_lang = res[0]
            probs = res[1] if isinstance(res[1], dict) else {res[0]: float(res[1])}
        elif isinstance(res, dict) and res:
            new_lang = max(res, key=res.get)
            probs = res
        else:
            return

        new_prob = float(probs.get(new_lang, 0.0))
        cur_lang = self.asr.language
        cur_prob = self._lang_current_prob if cur_lang else 0.0

        if new_prob >= LANG_SWITCH_PROB and (cur_lang is None or new_lang != cur_lang) and (new_prob - cur_prob) >= LANG_SWITCH_DELTA:
            self._lang_dwell_count += 1
            if self._lang_dwell_count >= LANG_DWELL_ITERS:
                # commit switch
                self.asr.language = new_lang
                self._lang_current_prob = new_prob
                self._last_lang_switch_time = now
                self._lang_dwell_count = 0
                # guard against prompt bias for one iteration
                self.asr.kw["condition_on_previous_text"] = False
                # temperature schedule is already a list; keep as-is
        else:
            self._lang_dwell_count = 0

    def _build_prompt(self) -> Tuple[str, str]:
        # Prompt = ~200 chars of scrolled-away committed text (outside current buffer)
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1
        prompt_words = [t for _, _, t in self.commited[:k]]
        prompt = []
        total_chars = 0
        while prompt_words and total_chars < 200:
            x = prompt_words.pop(-1)
            total_chars += len(x) + 1
            prompt.append(x)
        non_prompt = [t for _, _, t in self.commited[k:]]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(non_prompt)

    def _chunk_completed_segment(self, segments: List[Any]):
        if not self.commited:
            return
        ends = self.asr.segments_end_ts(segments)
        last_commit_t = self.commited[-1][1]
        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > last_commit_t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= last_commit_t:
                self._chunk_at(e)

    def _chunk_at(self, time_sec: float):
        # drop committed words <= time_sec
        self.transcript_buffer.pop_commited(time_sec)
        # cut audio buffer from left at absolute time
        cut_seconds = time_sec - self.buffer_time_offset
        cut_samples = int(cut_seconds * self.SAMPLING_RATE)
        if cut_samples <= 0:
            return
        self.audio_buffer = self.audio_buffer[cut_samples:]
        self.buffer_time_offset = time_sec

    @staticmethod
    def _join_words(words: List[Tuple[float, float, str]]) -> str:
        return "".join(w for _, _, w in words)

    # -------------------- Repeat detection & realign --------------------
    def _norm(self, s: str) -> str:
        return " ".join((s or "").lower().strip().split())

    def _same_ngram(self, a: str, b: str, n: int = REPEAT_NGRAM_N) -> bool:
        ax, bx = self._norm(a).split(), self._norm(b).split()
        if len(ax) < n or len(bx) < n:
            return False
        return " ".join(ax[-n:]) == " ".join(bx[-n:])

    def _repeating_now(self, committed_text: Optional[str], hypothesis_text: str) -> bool:
        # 1) hypothesis stalled across last REPEAT_HYP_STALL iterations
        if len(self._recent_hyps) >= REPEAT_HYP_STALL:
            hh = [self._norm(h) for h in self._recent_hyps[-REPEAT_HYP_STALL:]]
            if hh[0] and all(h == hh[0] for h in hh[1:]):
                return True

        # 2) same committed sentence appears >= threshold in recent history
        if committed_text:
            c = self._norm(committed_text)
            if c:
                cnt = sum(1 for x in self._recent_commits if self._norm(x) == c)
                if cnt + 1 >= REPEAT_SENT_THRESHOLD:  # include current
                    return True

        # 3) repeated tail n-gram between last commit and current hypothesis
        if committed_text and hypothesis_text and self._same_ngram(committed_text, hypothesis_text):
            return True

        return False

    def _force_realign(self, segments: List[Any]):
        """
        Suppress immediate output and:
          - reset hypothesis buffer (clear local-agreement state)
          - trim audio at a safe cut point:
              pref: last completed segment end <= last commit time
              else: tiny fixed jump REPEAT_JUMP_SEC
          - temporarily relax decoding (beam↑, no prev-text) to escape loop
        """
        # Reset hypothesis buffer
        self.transcript_buffer = HypothesisBuffer()

        # Prefer trimming at a completed segment end behind last commit
        cut_at: Optional[float] = None
        if self.commited:
            ends = self.asr.segments_end_ts(segments) or []
            if len(ends) > 1:
                last_commit_t = self.commited[-1][1]
                e = ends[-2] + self.buffer_time_offset
                while len(ends) > 2 and e > last_commit_t:
                    ends.pop(-1)
                    e = ends[-2] + self.buffer_time_offset
                if e <= last_commit_t:
                    cut_at = e

        # Fallback: fixed small jump
        if cut_at is None:
            cut_at = self.buffer_time_offset + REPEAT_JUMP_SEC

        self._chunk_at(cut_at)

        # Relax decoding for the next couple of iterations
        self.asr.kw["beam_size"] = 2
        self.asr.kw["temperature"] = [0.3]      # brief nudge; restore to schedule after cooldown
        self.asr.kw["condition_on_previous_text"] = False
        self._unstick_cooldown = 2
