"""Oscillators, envelopes, note names and the synth() entry point. Mono float32, SR = 44100."""
import numpy as np

SR = 44100

_NOTES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
WAVES = ("sine", "square", "saw", "triangle", "noise")

def note_freq(name):
    """'A4' -> 440.0 Hz. Also 'C#5', 'Bb3'."""
    s = name.strip()
    err = ValueError(f"bad note {name!r}; use e.g. 'A4', 'C#5', 'Bb3'")
    if len(s) < 2 or s[0].upper() not in _NOTES:
        raise err
    i, acc = 1, 0
    while i < len(s) and s[i] in "#b":
        acc += 1 if s[i] == "#" else -1
        i += 1
    try:
        octave = int(s[i:])
    except ValueError:
        raise err
    midi = 12 * (octave + 1) + _NOTES[s[0].upper()] + acc
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)

def oscillator(wave, freq, n, seed=0):
    """n samples of a unit-amplitude wave. freq: Hz, scalar or array of length n (for glides)."""
    if wave not in WAVES:
        raise KeyError(f"unknown wave {wave!r}; choose from {sorted(WAVES)}")
    if wave == "noise":
        return np.random.default_rng(seed).uniform(-1.0, 1.0, n)
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    ph = np.cumsum(f) / SR                      # phase in cycles
    if wave == "sine":
        return np.sin(2.0 * np.pi * ph)
    frac = ph % 1.0
    if wave == "square":
        return np.where(frac < 0.5, 1.0, -1.0)
    if wave == "saw":
        return 2.0 * frac - 1.0
    return 1.0 - 4.0 * np.abs(frac - 0.5)        # triangle

def adsr_env(n, a, d, s, r):
    """Envelope of exactly n samples. a, d, r in seconds, s = sustain level 0..1.
    The release occupies the LAST r seconds; a+d+r is shrunk if the note is too short."""
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    tot = na + nd + nr
    if tot > n:
        k = n / tot
        na, nd, nr = int(na * k), int(nd * k), int(nr * k)
    ns = n - na - nd - nr
    return np.concatenate([
        np.linspace(0.0, 1.0, na, endpoint=False),
        np.linspace(1.0, s, nd, endpoint=False),
        np.full(ns, float(s)),
        np.linspace(s, 0.0, nr),
    ])

def synth(wave="sine", freq=440.0, dur=0.5, vol=0.5, adsr=(0.01, 0.08, 0.7, 0.1),
          slide_to=None, vibrato=0.0, seed=0):
    """One note as a float32 array. wave: sine|square|saw|triangle|noise.
    freq / slide_to: Hz or a note name like 'A4' (slide_to glides exponentially to that pitch).
    adsr = (attack s, decay s, sustain level 0..1, release s). vibrato = depth in semitones
    (5.5 Hz). Naive square/saw alias a little at high pitch: fine for tutorials."""
    if wave not in WAVES:
        raise KeyError(f"unknown wave {wave!r}; choose from {sorted(WAVES)}")
    if dur <= 0:
        raise ValueError("dur must be > 0 seconds")
    hz = lambda v: note_freq(v) if isinstance(v, str) else float(v)
    f0 = hz(freq)
    if f0 <= 0:
        raise ValueError("freq must be > 0")
    n = max(1, int(round(dur * SR)))
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    f = np.full(n, f0)
    if slide_to is not None:
        f1 = hz(slide_to)
        if f1 <= 0:
            raise ValueError("slide_to must be > 0")
        f = f0 * (f1 / f0) ** u
    if vibrato:
        f = f * 2.0 ** (vibrato * np.sin(2.0 * np.pi * 5.5 * u * dur) / 12.0)
    a, d, s, r = adsr
    env = adsr_env(n, a, d, s, max(r, 0.004))        # never end on a click
    return (vol * oscillator(wave, f, n, seed) * env).astype(np.float32)
