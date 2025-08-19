# Antheia (Ἀνθεία) Audio Service

A real-time audio transcription service that captures system audio and microphone input.

## Features

- Real-time system audio capture via loopback
- Microphone audio capture
- Speech transcription using Whisper (Faster-Whisper)
- Real-time audio visualization
- Modular architecture for separate audio processing

## Usage

```bash
python -m audio.streaming_main
```

You should see output like:

```
🎮 device=cuda | 🧠 model=small | 16kHz | step=2s | trim=12s
🎚  external VAD: disabled
listening to SYSTEM audio only… (Ctrl+C to stop)
```

Transcribed text will appear in real time as system audio is detected.

---

## ⚙️ Configuration

Tune parameters in `audio/streaming_main.py`:

- `SAMPLE_RATE`: Audio sample rate (default: 16000)
- `CAPTURE_CHUNK_SEC`: Audio chunk size in seconds
- `PROCESS_INTERVAL_SEC`: How often to process audio (lower = lower latency)
- `TRIM_WINDOW_SEC`: Rolling context window for the ASR
- `LANG_PROBE_SEC`: Seconds of audio before first language detect
- `SIZE`: Model size ("tiny", "small", etc.)

---

## 🛠️ Project Structure

```
audio/
	streaming_main.py         # Entry point, main logic, tunables
	system_stream.py          # SystemStream: streaming pipeline logic
	system_audio_capture.py   # System audio capture utilities
	streaming_transcriber.py  # OnlineASRProcessor: streaming ASR logic
```

---

## 🧠 How It Works (Technical)

- **SystemStream**: Orchestrates audio capture and transcription in separate threads.
- **OnlineASRProcessor**: Handles streaming ASR, context prompts, repeat-guard, and language switching.
- **Prompt Engineering**: Supplies the ASR model with up to 200 characters of prior context for better accuracy.
- **Repeat/Loop Guard**: Detects and suppresses repeated or stuck outputs.

---

## 🏆 Why Antheia ?

- **No microphone required**: Captures everything you hear, not just what you say.
- **Streaming-first**: Designed for live, low-latency use cases.
- **Extensible**: Easy to adapt for custom output, UI, or downstream processing.

---

## 🙏 Credits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [python-soundcard](https://github.com/bastibe/python-soundcard)
- [whisper_streaming](https://github.com/ufal/whisper_streaming): - This project is heavily inspired by and reuses code from the [Whisper-Streaming](https://github.com/ufal/whisper_streaming) project by Dominik Macháček, Raj Dabre, and Ondřej Bojar ([paper](https://aclanthology.org/2023.ijcnlp-demo.3/)). - Please cite their work if you use this project in academic or derivative works:

      	> Macháček, D., Dabre, R., & Bojar, O. (2023). Turning Whisper into Real-Time Transcription System. In Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations (pp. 17–24). Association for Computational Linguistics. [PDF](https://aclanthology.org/2023.ijcnlp-demo.3.pdf)

      	[BibTeX](https://aclanthology.org/2023.ijcnlp-demo.3.bib)

---
