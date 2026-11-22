"""Narration voices.
engine='formant': a stylised 'blip voice' built from NumPy math (a pitched saw wave pushed through
  vowel-formant resonators, noise bursts for consonants). It is rhythmic, babble-like and NOT
  intelligible as words (think Animalese / Speak&Spell).
engine='espeak': optional add-on that calls the espeak-ng program (NOT NumPy-generated) for
  understandable, still robotic, speech."""
import hashlib, os, shutil, subprocess, tempfile, wave, zlib
from math import gcd
import numpy as np
from scipy.signal import lfilter, butter, resample_poly
from .osc import SR, oscillator

_VOWELS = {"a": (730, 1090, 2440), "e": (530, 1840, 2480), "i": (270, 2290, 3010),
           "o": (570, 840, 2410), "u": (300, 870, 2240), "y": (270, 2290, 3010)}
_LIQUID = {"l": (350, 1500, 3000), "r": (400, 1300, 1700), "w": (300, 700, 2200),
           "j": (270, 2290, 3010)}
_NASAL = {"m": (250, 1000, 2200), "n": (250, 1400, 2400)}
_FRIC = {"s": (5000, 0.35, 0.07), "z": (5000, 0.30, 0.07), "f": (3000, 0.25, 0.06),
         "v": (3000, 0.20, 0.06), "h": (1200, 0.20, 0.06), "x": (4000, 0.30, 0.08)}  # hp Hz, amp, dur
_PLOSIVE = {"p": 0.0, "t": 0.0, "k": 0.0, "c": 0.0, "q": 0.0, "b": 1.0, "d": 1.0, "g": 1.0}
_PAUSE = {" ": 0.05, ",": 0.15, ";": 0.15, ":": 0.15, ".": 0.30, "?": 0.30, "!": 0.30, "-": 0.05}
_BW, _AMP = (80, 100, 140), (1.0, 0.55, 0.25)

def _resonator(x, f, bw):
    r = np.exp(-np.pi * bw / SR)
    th = 2.0 * np.pi * f / SR
    b0 = (1.0 - r) * abs(1.0 - r * np.exp(-2j * th))     # unit gain at the resonance peak
    return lfilter([b0], [1.0, -2.0 * r * np.cos(th), r * r], x)

def _env(n, a=0.004, r=0.012):
    na, nr = min(int(a * SR), n // 2), min(int(r * SR), n // 2)
    e = np.ones(n)
    if na:
        e[:na] = np.linspace(0.0, 1.0, na, endpoint=False)
    if nr:
        e[n - nr:] = np.linspace(1.0, 0.0, nr)
    return e

def _voiced(f0, dur, formants):
    n = max(8, int(dur * SR))
    src = lfilter([1.0, -0.9], [1.0], oscillator("saw", f0, n))   # pre-emphasis: lifts F2/F3 (saw rolls off 1/k)
    y = sum(g * _resonator(src, f, bw) for f, bw, g in zip(formants, _BW, _AMP))
    return y / max(float(np.abs(y).max()), 1e-9) * _env(n)

def _noise(rng, dur, hp_hz, amp, decay=None):
    n = max(8, int(dur * SR))
    x = rng.uniform(-1.0, 1.0, n)
    b, a = butter(2, hp_hz / (SR / 2.0), "high")
    x = lfilter(b, a, x)
    x = x / max(float(np.abs(x).max()), 1e-9) * amp
    if decay:
        x = x * np.exp(-np.arange(n) / SR / decay)
    return x * _env(n, 0.002, 0.008)

def _formant_voice(text, pitch, speed):
    rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
    s = text.lower()
    question = s.rstrip().endswith("?")
    N = max(len(s), 1)
    parts = []
    for i, ch in enumerate(s):
        p = i / N
        semis = -3.0 * p + ((ord(ch) * 7) % 5 - 2) * 0.8              # declination + per-letter wobble
        if question and p > 0.8:
            semis += 6.0 * (p - 0.8) / 0.2                              # rising question intonation
        f0 = pitch * 2.0 ** (semis / 12.0)
        if ch.isdigit():
            ch = "aeiou"[int(ch) % 5]
        if ch in _VOWELS:
            parts.append(_voiced(f0, 0.09 / speed, _VOWELS[ch]))
        elif ch in _LIQUID:
            parts.append(0.8 * _voiced(f0, 0.06 / speed, _LIQUID[ch]))
        elif ch in _NASAL:
            parts.append(0.7 * _voiced(f0, 0.07 / speed, _NASAL[ch]))
        elif ch in _FRIC:
            hp, amp, dur = _FRIC[ch]
            parts.append(_noise(rng, dur / speed, hp, amp))
        elif ch in _PLOSIVE:
            burst = _noise(rng, 0.03 / speed, 1500, 0.5, decay=0.01)
            if _PLOSIVE[ch] > 0:
                hum = 0.5 * _voiced(f0, 0.03 / speed, (400, 1200, 2400))
                m = max(len(burst), len(hum))
                burst = np.pad(burst, (0, m - len(burst))) + np.pad(hum, (0, m - len(hum)))
            parts.append(burst)
        elif ch in _PAUSE:
            parts.append(np.zeros(int(_PAUSE[ch] / speed * SR)))
    if not parts:
        raise ValueError("text has no speakable characters (letters, digits or punctuation)")
    return np.concatenate(parts)

def _espeak(text, pitch, speed):
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        raise RuntimeError("espeak-ng not found (Colab: !apt-get install -y espeak-ng)")
    p = int(min(max(round(50 + 25 * np.log2(pitch / 140.0)), 0), 99))
    wpm = max(80, int(round(175 * speed)))
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "v.wav")
        subprocess.run([exe, "-s", str(wpm), "-p", str(p), "-w", out, "--", text],
                       check=True, capture_output=True)
        with wave.open(out, "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    if sw != 2:
        raise RuntimeError("unexpected espeak sample width")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x

def _read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise RuntimeError("unexpected WAV sample width")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x

def _piper_paths(model):
    exe = os.environ.get("PIPER_BIN") or shutil.which("piper")
    model = model or os.environ.get("PIPER_MODEL")
    if not exe or not os.path.exists(exe):
        raise RuntimeError("piper binary not found: set os.environ['PIPER_BIN'] to the 'piper' executable")
    if not model or not os.path.exists(model):
        raise RuntimeError("piper voice not found: pass model='/path/voice.onnx' or set os.environ['PIPER_MODEL']")
    stem = model[:-5] if model.endswith(".onnx") else model
    cfg = next((c for c in (model + ".json", stem + "_onnx.json", stem + ".json") if os.path.exists(c)), None)
    if cfg is None:
        raise RuntimeError(f"no config json next to {model!r} (expected {model}.json or {stem}_onnx.json)")
    return exe, model, cfg

def _piper(text, speed, model):
    """Neural TTS via the piper program (NOT NumPy-generated). Results are cached on disk because
    piper adds random noise (so two runs differ) and loads its model on every launch (~3 s)."""
    exe, model, cfg = _piper_paths(model)
    key = hashlib.sha1(f"{text}|{speed:.4f}|{os.path.abspath(model)}|{os.path.getsize(model)}".encode()).hexdigest()
    cache = os.environ.get("SUFVID_CACHE") or os.path.join(tempfile.gettempdir(), "sufvid_voice")
    os.makedirs(cache, exist_ok=True)
    hit = os.path.join(cache, key + ".npy")
    if os.path.exists(hit):
        return np.load(hit)
    env = dict(os.environ)
    libdir = os.path.dirname(os.path.abspath(exe))
    env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "v.wav")
        r = subprocess.run([exe, "--model", model, "--config", cfg, "--length_scale", f"{1.0 / speed:.4f}",
                            "--output_file", out], input=text.encode("utf-8"), capture_output=True, env=env)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("piper failed:\n" + r.stderr.decode(errors="replace")[-800:])
        x = _read_wav(out)
    np.save(hit, x)
    return x

def voice(text, pitch=170.0, speed=1.0, vol=0.6, engine="formant", model=None):
    """Speak `text`; returns a float32 array (peak = vol). Duration = len(x) / 44100.
    pitch: base Hz (formant) / mapped to espeak -p; ignored by piper. speed: 1.0 normal, 2.0 twice as fast.
    engine='piper': neural voice; needs os.environ['PIPER_BIN'] (the piper executable) and
    model='x.onnx' or os.environ['PIPER_MODEL']. Cached on disk, so repeated lines are instant.
    Same text + settings always give the same samples."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if pitch <= 0 or speed <= 0:
        raise ValueError("pitch and speed must be > 0")
    if engine == "formant":
        x = _formant_voice(text, pitch, speed)
    elif engine == "espeak":
        x = _espeak(text, pitch, speed)
    elif engine == "piper":
        x = _piper(text, speed, model)
    else:
        raise KeyError(f"unknown engine {engine!r}; choose 'formant', 'espeak' or 'piper'")
    peak = float(np.abs(x).max())
    if peak <= 0:
        raise ValueError("voice produced silence")
    return (x / peak * vol).astype(np.float32)
