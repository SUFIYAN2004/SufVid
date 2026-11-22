import sys, shutil, subprocess, wave, os
for m in [k for k in sys.modules if k.startswith("Sufvid")]:
    del sys.modules[m]
import numpy as np
import Sufvid
from Sufvid import synth, sfx, voice, mix, render_video, circle
from Sufvid.audio.osc import SR, note_freq, adsr_env
from Sufvid.audio.sfx import SFX

print("public API size:", len(Sufvid.__all__), "/ 40")
assert len(Sufvid.__all__) < 40

def raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")

def spectrum(x):
    return np.fft.rfftfreq(len(x), 1.0 / SR), np.abs(np.fft.rfft(x))

def mag_at(x, hz):                                   # magnitude near one frequency (1 s clips -> 1 Hz bins)
    f, m = spectrum(x)
    return m[(f > hz - 3) & (f < hz + 3)].max()

def band(x, lo, hi):                                 # fraction of spectral energy in [lo, hi]
    f, m = spectrum(x)
    e = m ** 2
    return e[(f >= lo) & (f <= hi)].sum() / e.sum()

# ---------------- notes ----------------
assert abs(note_freq("A4") - 440.0) < 1e-9
assert abs(note_freq("C4") - 261.6256) < 1e-3
assert abs(note_freq("Bb3") - 233.0819) < 1e-3 and abs(note_freq("C#5") - 554.3653) < 1e-3
raises(ValueError, lambda: note_freq("H4"))
raises(ValueError, lambda: note_freq("A"))

# ---------------- envelope ----------------
e = adsr_env(SR, 0.1, 0.2, 0.5, 0.3)
assert len(e) == SR and e[0] == 0 and abs(e.max() - 1) < 1e-3 and e[-1] == 0
assert abs(e[int(0.5 * SR)] - 0.5) < 1e-9                         # sustain level
assert len(adsr_env(100, 1.0, 1.0, 0.5, 1.0)) == 100              # too-short note is squeezed, not broken

# ---------------- synth: the spectrum proves the waveform ----------------
flat = (0.0, 0.0, 1.0, 0.004)
sine = synth("sine", 440, 1.0, vol=1.0, adsr=flat)
assert sine.dtype == np.float32 and len(sine) == SR
f, m = spectrum(sine)
assert abs(f[np.argmax(m)] - 440) < 1.5                           # pitch is right
assert band(sine, 430, 450) > 0.99                                # and it is a pure tone
sq = synth("square", 440, 1.0, vol=1.0, adsr=flat)
assert 0.28 < mag_at(sq, 1320) / mag_at(sq, 440) < 0.38           # square: 3rd harmonic ~ 1/3
assert mag_at(sq, 880) < 0.03 * mag_at(sq, 440)                   # even harmonics absent
sw = synth("saw", 440, 1.0, vol=1.0, adsr=flat)
assert 0.4 < mag_at(sw, 880) / mag_at(sw, 440) < 0.6              # saw: 2nd harmonic ~ 1/2
tri = synth("triangle", 440, 1.0, vol=1.0, adsr=flat)
assert 0.07 < mag_at(tri, 1320) / mag_at(tri, 440) < 0.16         # triangle: 3rd ~ 1/9
nz = synth("noise", 440, 1.0, vol=1.0, adsr=flat)
assert abs(nz.mean()) < 0.02 and abs(band(nz, 0, 5000) / band(nz, 5000, 10000) - 1) < 0.15   # flat spectrum
assert np.array_equal(synth("noise", 1, 0.2, seed=3), synth("noise", 1, 0.2, seed=3))
assert not np.array_equal(synth("noise", 1, 0.2, seed=3), synth("noise", 1, 0.2, seed=4))
assert abs(synth("sine", "A4", 0.3)[:100]).max() == abs(synth("sine", 440.0, 0.3)[:100]).max()
g = synth("sine", 440, 1.0, vol=1.0, adsr=flat, slide_to=880)     # glide 440 -> 880
f1, m1 = spectrum(g[: SR // 10]); f2, m2 = spectrum(g[-SR // 10:])
assert 430 < f1[np.argmax(m1)] < 490 and 800 < f2[np.argmax(m2)] < 900
assert not np.array_equal(synth("sine", 440, 0.5), synth("sine", 440, 0.5, vibrato=0.5))
s0 = synth("sine", 300, 0.2, vol=0.5)
assert s0[0] == 0 and abs(s0[-1]) < 1e-3 and np.abs(s0).max() <= 0.5 + 1e-6      # no click at the ends
raises(KeyError, lambda: synth("zigzag"))
raises(ValueError, lambda: synth("sine", dur=0))
raises(ValueError, lambda: synth("sine", freq=-5))

# ---------------- sfx ----------------
for name in SFX:
    x = sfx(name, vol=0.7)
    assert x.dtype == np.float32 and np.isfinite(x).all(), name
    assert abs(np.abs(x).max() - 0.7) < 1e-4, name
    assert 0.02 < len(x) / SR < 1.5, (name, len(x) / SR)
    assert abs(x[-1]) < 1e-3, name                                # ends silently
    assert np.array_equal(x, sfx(name, vol=0.7)), name            # deterministic
    assert (np.abs(x) > 0.05).sum() > 100, name                   # not a near-silent blip
d = sfx("ding", vol=1.0)
assert abs(spectrum(d)[0][np.argmax(spectrum(d)[1])] - 1318.5) < 8   # a ding is an E6
wh = sfx("whoosh", vol=1.0)
h = len(wh) // 3
c1 = (lambda f, m: (f * m).sum() / m.sum())(*spectrum(wh[:h]))
c2 = (lambda f, m: (f * m).sum() / m.sum())(*spectrum(wh[-h:]))
assert c2 > 1.5 * c1, (c1, c2)                                    # whoosh brightens as it sweeps up
raises(KeyError, lambda: sfx("boom"))

# ---------------- voice (blip / formant) ----------------
va, vi = voice("a", vol=0.6), voice("i", vol=0.6)
assert abs(np.abs(va).max() - 0.6) < 1e-4 and va.dtype == np.float32
assert band(va, 600, 900) > band(vi, 600, 900)                    # /a/ has its energy near F1 = 730
assert band(vi, 2000, 2600) > band(va, 2000, 2600)                # /i/ has its energy near F2 = 2290
t1 = voice("hello world, this is a test.")
assert np.array_equal(t1, voice("hello world, this is a test."))  # deterministic
assert not np.array_equal(t1, voice("hello world, this is a test!"))
assert np.isfinite(t1).all() and 1.0 < len(t1) / SR < 4.0
t2 = voice("hello world, this is a test.", speed=2.0)
assert abs(len(t2) / len(t1) - 0.5) < 0.03                        # speed 2 = half as long
assert len(voice("a. a")) > len(voice("a a"))                     # a full stop is a longer pause
assert len(voice("say 42 now")) > len(voice("say now"))           # digits are voiced too
raises(ValueError, lambda: voice(""))
raises(ValueError, lambda: voice("###"))
raises(ValueError, lambda: voice("hi", speed=0))
raises(KeyError, lambda: voice("hi", engine="magic"))
if shutil.which("espeak-ng") or shutil.which("espeak"):
    es = voice("hello world", engine="espeak")
    assert 0.5 < len(es) / SR < 3.0 and np.isfinite(es).all()
    print("  espeak-ng found: engine='espeak' works,", f"{len(es) / SR:.2f}s")
else:
    print("  espeak-ng not installed: espeak test skipped (optional add-on)")

# ---------------- mixer ----------------
a, b = np.ones(SR) * 0.3, np.ones(SR) * 0.4
y = mix([(0.0, a), (0.5, b)])
assert len(y) == int(1.5 * SR)
assert abs(y[int(0.25 * SR)] - 0.3) < 1e-6 and abs(y[int(0.75 * SR)] - 0.7) < 1e-6 and abs(y[int(1.25 * SR)] - 0.4) < 1e-6
y = mix([(0.0, a), (0.5, b, 0.5)], duration=2.0)                  # gain + padding
assert len(y) == 2 * SR and abs(y[int(0.75 * SR)] - 0.5) < 1e-6 and y[-1] == 0
y = mix([(0.0, a), (0.0, a)], master=0.5)                         # limiter scales down, never up
assert abs(np.abs(y).max() - 0.5) < 1e-6
y = mix([(0.0, a * 0.1)], master=0.9)
assert abs(np.abs(y).max() - 0.03) < 1e-6
assert len(mix([(0.5, a)], duration=0.75)) == int(0.75 * SR)       # long clips are cut
raises(ValueError, lambda: mix([]))
raises(ValueError, lambda: mix([(-1.0, a)]))
raises(ValueError, lambda: mix([(0.0, np.zeros((2, 2)))]))
raises(ValueError, lambda: mix([(0.0,)]))
clip = sfx("ding", vol=0.8)
y = mix([(0.2, clip)], path="/tmp/_t5.wav")                        # WAV round trip
with wave.open("/tmp/_t5.wav", "rb") as w:
    assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR) and w.getnframes() == len(y)
    back = np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32767.0
assert np.abs(back - y).max() < 1.0 / 32767 + 1e-9

# ---------------- muxing through the ffmpeg pipe ----------------
if shutil.which("ffmpeg"):
    def decode_audio(p):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.float32)
    def count_video(p):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                              "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", p],
                             capture_output=True, check=True).stdout
        return int(out.strip())
    scene = {"size": (160, 90), "bg": "#000", "shapes": [circle((0, 0), 2, fill="#ff0000", stroke=None)]}
    tone = mix([(0.0, synth("sine", 1000, 1.0, vol=0.5, adsr=(0.0, 0.0, 1.0, 0.01)))], path="/tmp/_tone1.wav")
    # 1) audio shorter than video: video decides length, audio is padded with silence
    render_video(scene, duration=2.0, fps=10, path="/tmp/_m1.mp4", workers=2, audio="/tmp/_tone1.wav")
    au = decode_audio("/tmp/_m1.mp4")
    assert count_video("/tmp/_m1.mp4") == 20
    assert abs(len(au) / SR - 2.0) < 0.15, len(au) / SR
    assert np.abs(au[: SR // 2]).max() > 0.3 and np.abs(au[int(1.3 * SR):]).max() < 0.01
    assert abs(spectrum(au[: SR // 2])[0][np.argmax(spectrum(au[: SR // 2])[1])] - 1000) < 10   # pitch survived AAC
    # 2) duration=None: derived from the audio length
    render_video(scene, fps=10, path="/tmp/_m2.mp4", workers=1, audio="/tmp/_tone1.wav")
    assert count_video("/tmp/_m2.mp4") == 10
    # 3) audio longer than video: truncated to the video
    long = mix([(0.0, synth("sine", 500, 3.0, vol=0.5))], path="/tmp/_tone3.wav")
    render_video(scene, duration=1.0, fps=10, path="/tmp/_m3.mp4", workers=2, audio="/tmp/_tone3.wav")
    assert count_video("/tmp/_m3.mp4") == 10 and abs(len(decode_audio("/tmp/_m3.mp4")) / SR - 1.0) < 0.15
    # 4) no audio -> no audio stream (old behaviour intact); missing file -> clear error
    render_video(scene, duration=0.5, fps=10, path="/tmp/_m4.mp4", workers=1)
    assert b"audio" not in subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                                           "-of", "csv=p=0", "/tmp/_m4.mp4"], capture_output=True).stdout
    raises(FileNotFoundError, lambda: render_video(scene, duration=1, path="/tmp/_x.mp4", audio="/tmp/nope.wav"))
    raises(ValueError, lambda: render_video(scene, path="/tmp/_x.mp4"))
else:
    print("ffmpeg missing: mux tests skipped")

print("all Phase 5 tests passed")
