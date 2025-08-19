import soundcard as sc
import numpy as np

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono

def get_system_loopback():
    """
    Return a loopback/monitor device for the default speaker.
    - Windows (WASAPI): a loopback input usually appears that matches the speaker name.
    - Linux (PulseAudio/PipeWire): 'Monitor of ...' source.
    """
    spk = sc.default_speaker()
    mics = sc.all_microphones(include_loopback=True)
    # best: exact match to default speaker name
    for m in mics:
        if spk and spk.name and m.name and spk.name in m.name:
            return m
    # fallbacks
    for m in mics:
        name = (m.name or "").lower()
        if any(k in name for k in ("loopback", "monitor", "speaker", "output", "what u hear")):
            return m
    return None

def record_chunk(device, frames: int) -> np.ndarray:
    """
    Record a chunk from the loopback device at 16 kHz.
    Returns mono float32 in [-1, 1], length = frames.
    """
    if device is None:
        return np.zeros(frames, dtype=np.float32)
    # soundcard resamples to requested samplerate internally
    with device.recorder(samplerate=SAMPLE_RATE, channels=2, blocksize=frames) as rec:
        data = rec.record(numframes=frames)  # float32, shape (frames, 2)
    if data.ndim == 2:
        data = data.mean(axis=1)  # downmix to mono
    return data.astype(np.float32)
