"""mix(): place clips on a timeline, sum them, limit the peak, optionally write a WAV."""
import wave as _wave
import numpy as np
from .osc import SR

def _write_wav(path, x):
    pcm = np.rint(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with _wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

def mix(tracks, path=None, duration=None, master=0.9):
    """Sum clips placed at times. tracks = [(start_seconds, samples), (start, samples, gain), ...].
    duration: total length in seconds (default: end of the last clip; longer clips are cut).
    If the summed peak exceeds `master` everything is scaled down (never up).
    path: also write a 16-bit mono WAV at 44100 Hz. Returns the float32 array."""
    if len(tracks) == 0:
        raise ValueError("tracks is empty; give [(start_seconds, samples), ...]")
    placed, end = [], 0
    for i, tr in enumerate(tracks):
        if len(tr) not in (2, 3):
            raise ValueError(f"track {i} must be (start, samples) or (start, samples, gain)")
        start, x = float(tr[0]), np.asarray(tr[1], dtype=np.float64)
        gain = float(tr[2]) if len(tr) == 3 else 1.0
        if x.ndim != 1:
            raise ValueError(f"track {i}: samples must be a 1-D array")
        if start < 0:
            raise ValueError(f"track {i}: start must be >= 0 seconds")
        i0 = int(round(start * SR))
        placed.append((i0, x * gain))
        end = max(end, i0 + len(x))
    n = int(round(duration * SR)) if duration is not None else end
    if n <= 0:
        raise ValueError("mix would be empty")
    out = np.zeros(n)
    for i0, x in placed:
        if i0 < n:
            seg = x[:n - i0]
            out[i0:i0 + len(seg)] += seg
    peak = float(np.abs(out).max())
    if peak > master:
        out *= master / peak
    out = out.astype(np.float32)
    if path is not None:
        _write_wav(path, out)
    return out
