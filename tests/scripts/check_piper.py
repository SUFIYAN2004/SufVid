import os, shutil, time
import numpy as np
from Sufvid import voice
from Sufvid.audio.osc import SR

def raises(exc, fn):
    try: fn()
    except exc: return
    raise AssertionError(f"expected {exc.__name__}")

if os.environ.get("PIPER_BIN") and os.environ.get("PIPER_MODEL"):
    shutil.rmtree("/tmp/sufvid_voice", ignore_errors=True)
    t0 = time.perf_counter(); a = voice("Watch the curve. All math!", engine="piper"); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); b = voice("Watch the curve. All math!", engine="piper"); warm = time.perf_counter() - t0
    assert a.dtype == np.float32 and np.isfinite(a).all() and abs(np.abs(a).max() - 0.6) < 1e-4
    assert 1.5 < len(a) / SR < 3.5, len(a) / SR                         # 44.1 kHz after resampling from 22.05 kHz
    assert np.array_equal(a, b) and warm < 0.2 * cold + 0.05            # cache: identical and fast
    f = np.fft.rfftfreq(len(a), 1 / SR); e = np.abs(np.fft.rfft(a)) ** 2
    assert e[(f > 80) & (f < 5000)].sum() / e.sum() > 0.97              # it is speech-band audio, not noise/clicks
    assert e[f > 10000].sum() / e.sum() < 0.01                          # nothing above 10 kHz (22 kHz source)
    long_ = voice("Watch the curve. All math! And now the second line is a lot longer.", engine="piper")
    assert len(long_) > 1.4 * len(a)                                    # more words, more time
    fast = voice("Watch the curve. All math!", engine="piper", speed=2.0)
    assert 0.35 < len(fast) / len(a) < 0.7, len(fast) / len(a)          # speed 2 ~ half as long
    rms = np.sqrt(np.convolve(a.astype(float) ** 2, np.ones(441) / 441, "same"))
    assert (rms < 0.01).mean() > 0.05 and (rms > 0.05).mean() > 0.3     # real pauses between words + real speech
    print(f"piper OK: {len(a)/SR:.2f}s, cold {cold:.2f}s, cached {warm*1000:.0f} ms, speed2 ratio {len(fast)/len(a):.2f}")
    raises(RuntimeError, lambda: voice("hi", engine="piper", model="/nope.onnx"))
    old = os.environ["PIPER_BIN"]; os.environ["PIPER_BIN"] = "/nope/piper"
    raises(RuntimeError, lambda: voice("hi", engine="piper")); os.environ["PIPER_BIN"] = old
    # other engines untouched
    assert len(voice("hello")) > 0
    raises(KeyError, lambda: voice("hi", engine="magic"))
    # run from a different cwd with the config named ..._onnx.json (as in your upload)
    os.chdir("/tmp"); shutil.rmtree("/tmp/sufvid_voice", ignore_errors=True)
    assert len(voice("Testing from another folder.", engine="piper")) > SR
    print("piper tests passed")
else:
    print("PIPER_BIN / PIPER_MODEL not set: piper tests skipped")
