from __future__ import annotations
import logging
import queue
import threading
import time
from typing import Optional
import numpy as np
import soundcard as sc

from .streaming_transcriber import OnlineASRProcessor
from .system_audio_capture import get_system_loopback, record_chunk

logging.getLogger("faster_whisper").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

# ---- TUNABLES ----
SAMPLE_RATE = 16000
CAPTURE_CHUNK_SEC = 0.5           # device read granularity (fast I/O)
PROCESS_INTERVAL_SEC = 2          # step cadence; 1.5 for lower latency, 2.5–3.0 for stability
TRIM_WINDOW_SEC = 12.0            # rolling context inside processor
LANG_PROBE_SEC = 3.5              # seconds of audio before first language detect
MODEL_SIZE = "small"              # faster-whisper model size for ASR

VAD_ENABLED = False              
FRAMES = int(SAMPLE_RATE * CAPTURE_CHUNK_SEC)

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)))

# ---- Stream runner ----
class SystemStream:
    def __init__(self, model_size="small", device_type="cuda",
                 language: Optional[str] = None):
        self.device = get_system_loopback()
        if self.device is None:
            raise RuntimeError(
                "No system loopback device found. "
                "Windows: ensure WASAPI loopback; Linux: use PulseAudio/PipeWire monitor."
            )
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self.proc = OnlineASRProcessor(
            model_size=model_size,
            device=device_type,
            language=language,  # None => auto-detect once
            buffer_trimming_sec=TRIM_WINDOW_SEC,
            lang_probe_sec=LANG_PROBE_SEC,
            # new fast-path toggles in the processor:
            word_timestamps_live=False,   # big speed win
        )
        self.running = False

    def capture_loop(self):
        self.running = True
        while self.running:
            data = record_chunk(self.device, FRAMES)  
            # simple RMS energy gate 
            if data.size == 0 or rms(data) < 0.002:
                continue
            try:
                self.audio_q.put_nowait(data)
            except queue.Full:
                _ = self.audio_q.get_nowait()
                self.audio_q.put_nowait(data)

    def transcribe_loop(self):
        self.running = True
        acc: list[np.ndarray] = []
        last_proc = time.time()
        while self.running:
            try:
                chunk = self.audio_q.get(timeout=0.2)
                acc.append(chunk)
            except queue.Empty:
                pass

            now = time.time()
            if acc and (now - last_proc) >= PROCESS_INTERVAL_SEC:
                audio = np.concatenate(acc)
                acc = []
                last_proc = now

                self.proc.insert_audio_chunk(audio)
                committed, hypothesis = self.proc.process_iter()

                if committed:  
                    print(f"them: {committed}", flush=True)

    def stop(self):
        self.running = False

def main():
    try:
        import torch
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device_type = "cpu"

    model_size = MODEL_SIZE if device_type == "cuda" else "tiny"
    print(f"🎮 device={device_type} | 🧠 model={model_size} | 16kHz | step={PROCESS_INTERVAL_SEC}s | trim={TRIM_WINDOW_SEC}s")
    # print("🎚  external VAD: disabled")

    stream = SystemStream(model_size=model_size, device_type=device_type, language=None)

    t_cap = threading.Thread(target=stream.capture_loop, daemon=True)
    t_asr = threading.Thread(target=stream.transcribe_loop, daemon=True)
    t_cap.start(); t_asr.start()

    print("listening to SYSTEM audio only… (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stream.stop()
        t_cap.join(timeout=1.0); t_asr.join(timeout=1.0)
        _, hyp = stream.proc.finish()
        if hyp:
            print(f"them(hyp): {hyp}")

if __name__ == "__main__":
    main()
