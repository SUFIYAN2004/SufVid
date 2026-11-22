"""Sound effects built from the oscillators in osc.py. sfx(name) -> float32 array."""
import numpy as np
from .osc import SR, synth
from .mixer import mix

def _t(n):
    return np.arange(n) / SR

def _finish(x, peak, fade=0.02):
    """Normalise to `peak` and fade the last `fade` seconds so nothing ends on a click."""
    x = np.asarray(x, dtype=np.float64)
    x = x / max(float(np.abs(x).max()), 1e-9) * peak
    k = min(int(fade * SR), len(x))
    x[len(x) - k:] *= np.linspace(1.0, 0.0, k)
    return x.astype(np.float32)

def _click(seed):
    n = int(0.03 * SR)
    x = np.random.default_rng(seed).uniform(-1, 1, n) * np.exp(-_t(n) / 0.003)
    return np.diff(x, prepend=0.0)                       # crude high-pass => crisp click

def _key(seed):                                          # typewriter tick, for type_text
    tick = synth("triangle", 1100, 0.04, vol=0.5, slide_to=500, adsr=(0.0005, 0.01, 0.0, 0.02))
    return mix([(0.0, tick), (0.0, _click(seed), 0.6)])

def _pop(seed):
    return synth("sine", 650, 0.12, vol=0.8, slide_to=160, adsr=(0.002, 0.04, 0.0, 0.05))

def _blip(seed):
    return synth("square", 880, 0.07, vol=0.4, adsr=(0.002, 0.02, 0.6, 0.02))

def _ding(seed):
    n = int(1.2 * SR)
    t = _t(n)
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / d)
            for f, a, d in [(1318.5, 1.0, 0.45), (1318.5 * 2.76, 0.35, 0.18), (1318.5 * 5.4, 0.15, 0.08)])
    return y * np.minimum(1.0, t / 0.002)

def _success(seed):
    notes = ["C5", "E5", "G5", "C6"]
    tr = [(0.08 * i, synth("triangle", f, 0.14 if i < 3 else 0.5, vol=0.6,
                           adsr=(0.004, 0.04, 0.6, 0.06))) for i, f in enumerate(notes)]
    return mix(tr)

def _error(seed):
    buzz = synth("saw", 170, 0.13, vol=0.6, slide_to=120, adsr=(0.004, 0.02, 0.8, 0.03))
    return mix([(0.0, buzz), (0.17, buzz)])

def _whoosh(seed):
    n = int(0.7 * SR)
    x = np.random.default_rng(seed).uniform(-1, 1, n)
    u = np.linspace(0.0, 1.0, n)
    fc = 300.0 * (6000.0 / 300.0) ** u                    # cutoff sweeps up 300 -> 6000 Hz
    alpha = 1.0 - np.exp(-2.0 * np.pi * fc / SR)
    for _ in range(2):                                     # two one-pole low-passes in series
        y, s = np.empty(n), 0.0
        for i in range(n):
            s += alpha[i] * (x[i] - s)
            y[i] = s
        x = y
    return x * np.sin(np.pi * u) ** 2

SFX = {"click": _click, "key": _key, "pop": _pop, "blip": _blip, "ding": _ding,
       "success": _success, "error": _error, "whoosh": _whoosh}

def sfx(name, vol=0.8, seed=0):
    """Sound effect by name: click, key, pop, blip, ding, success, error, whoosh.
    Returns a float32 array whose peak is `vol` (deterministic for a given seed)."""
    if name not in SFX:
        raise KeyError(f"unknown sfx {name!r}; choose from {sorted(SFX)}")
    return _finish(SFX[name](seed), vol)
