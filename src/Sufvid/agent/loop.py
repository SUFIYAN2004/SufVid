"""generate -> check -> preview -> render -> read the error -> fix. The LLM is any function prompt -> reply."""
import contextlib, io, os, re, traceback
import numpy as np
from ..anim.timeline import play, timeline_duration
from ..render.frame import render_frame
from ..render.video import render_video, _wav_seconds
from ..audio.mixer import mix
from .check import check_scene
from .docs import build_docs

class _Fail(Exception):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage, self.message = stage, message

def _extract_code(reply):
    """Pull the program out of an LLM reply: the longest ```python block, else the whole reply."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", reply, re.S)
    return max(blocks, key=len) if blocks else reply

def _explain(exc, code):
    """Short, LLM-friendly description of an exception, pointing at the offending program line."""
    where = ""
    for fr in reversed(traceback.extract_tb(exc.__traceback__)):
        if fr.filename == "<agent>":
            lines = code.splitlines()
            src = lines[fr.lineno - 1].strip() if 0 < fr.lineno <= len(lines) else ""
            where = f" at program line {fr.lineno}: {src}"
            break
    if isinstance(exc, SyntaxError):
        where = f" at program line {exc.lineno}: {(exc.text or '').strip()}"
    msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
    return f"{type(exc).__name__}: {msg}{where}"

def run_program(code):
    """Execute a program and return its namespace. Raises _Fail(stage, message). (exec of untrusted code!)"""
    ns, out = {"__name__": "__sufvid_program__"}, io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(code, "<agent>", "exec"), ns)
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        raise _Fail("exec", _explain(e, code))
    ns["__stdout__"] = out.getvalue()
    return ns

def _contract(ns, wav_path="agent_audio.wav"):
    """-> (scene_or_fn, timeline, duration, audio_path). Raises _Fail('contract', ...).
    `audio` may be the array returned by mix(...) (written to wav_path here) or the path of a WAV."""
    audio = ns.get("audio")
    if isinstance(audio, np.ndarray) and audio.ndim == 1 and audio.size:
        mix([(0.0, audio)], path=wav_path, master=1.0)
        audio = wav_path
    elif audio is not None and (not isinstance(audio, str) or not os.path.exists(audio)):
        raise _Fail("contract", "`audio` must be the array returned by mix(...) or the path of an existing WAV file")
    timeline, duration = ns.get("timeline"), ns.get("duration")
    if "scene_at" in ns:
        scene = ns["scene_at"]
        if not callable(scene):
            raise _Fail("contract", "`scene_at` must be a function t -> scene dict")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise _Fail("contract", "with scene_at(t) you must also define duration = seconds (a number > 0)")
        first = scene(0.0)
    elif "scene" in ns:
        scene = first = ns["scene"]
        if not isinstance(scene, dict):
            raise _Fail("contract", "`scene` must be a dict with 'size' and 'shapes'")
    else:
        raise _Fail("contract", "the program must define `scene` (a dict) or `scene_at(t)` (a function) - see the contract")
    if not isinstance(first, dict) or "size" not in first or "shapes" not in first:
        raise _Fail("contract", "a scene must be a dict with 'size': (width, height) and 'shapes': [...]")
    if first["size"][0] % 2 or first["size"][1] % 2:
        raise _Fail("contract", "width and height of 'size' must be even numbers")
    if timeline is not None and not isinstance(timeline, list):
        raise _Fail("contract", "`timeline` must be a list of animations")
    if duration is None:
        if timeline is None and audio is None:
            raise _Fail("contract", "define a `timeline`, a `duration`, or an `audio` so the video has a length")
        parts = []
        if timeline is not None:
            try:
                parts.append(timeline_duration(first, timeline) + 0.5)
            except (KeyError, ValueError):
                parts.append(1.0)          # a broken timeline is reported (clearly) by check_scene next
        if audio is not None:
            parts.append(_wav_seconds(audio))
        duration = max(parts)
    return scene, timeline, float(duration), audio

def _preview(scene, timeline, duration, fractions=(0.0, 0.5, 1.0)):
    """Render a few real frames at full size; catches drawing errors cheaply and blank videos."""
    frames = []
    for f in fractions:
        t = f * duration
        try:
            sc = scene(t) if callable(scene) else scene
            if timeline:
                sc = play(sc, timeline, t)
            frames.append(render_frame(sc, t))
        except Exception as e:
            raise _Fail("preview", f"rendering the frame at t={t:.2f}s failed: {type(e).__name__}: "
                                   f"{e.args[0] if isinstance(e, KeyError) and e.args else e}")
    flat = lambda fr: float(fr.reshape(-1, 3).std(axis=0).max()) < 1e-6      # per channel: a coloured background alone is "flat"
    if all(flat(fr) for fr in frames):
        raise _Fail("preview", "every sampled frame is a flat background: nothing is visible. Check that "
                               "shapes are inside the frame and not hidden by fade_in/draw that never runs")
    if flat(frames[-1]):
        raise _Fail("preview", f"the last frame (t={duration:.2f}s) is empty: everything has disappeared "
                               "(a fade_out at the end?). Keep something visible at the end")

def _prompt(task, history):
    p = build_docs() + f"\n\nTASK:\n{task}\n"
    if history:
        h = history[-1]
        p += (f"\nYOUR PREVIOUS PROGRAM:\n```python\n{h['code'].strip()}\n```\n"
              f"IT FAILED at stage '{h['stage']}':\n{h['message']}\n"
              "Fix exactly this problem and return the complete corrected program in one ```python block.\n")
    return p

def agent_loop(task, generate, max_tries=4, path="agent_out.mp4", size=None, fps=30, verbose=True):
    """Ask an LLM for a Sufvid program, check it, render it, and feed every failure back until it works.
    task: plain-English description of the video.
    generate: ANY function prompt:str -> reply:str (wrap your LLM client; the reply may contain prose
      and a ```python block).
    Stages: exec -> contract -> check (check_scene) -> preview (3 real frames) -> render (mp4).
    size: optional (w, h) override for the final render only (the checks always run at the program's own size).
      Text and stroke sizes are fixed PIXELS, so a smaller size makes them look bigger relative to the layout:
      use it for quick smoke tests, not for judging the final look.
    Returns {'ok', 'video', 'code', 'attempts', 'history', 'render'}; history = failed attempts."""
    history = []
    for k in range(1, max_tries + 1):
        reply = generate(_prompt(task, history))
        if not isinstance(reply, str):
            raise TypeError("generate(prompt) must return a string")
        code = _extract_code(reply)
        stage = "exec"
        try:
            ns = run_program(code)
            stage = "contract"
            scene, timeline, duration, audio = _contract(ns)
            stage = "check"
            problems = check_scene(scene, timeline, duration)
            errors = [m for m in problems if m.startswith("error")]
            if errors:
                raise _Fail("check", "\n".join(problems))
            stage = "preview"
            _preview(scene, timeline, duration)
            stage = "render"
            target = scene
            if size is not None:
                target = (lambda t, f=scene: {**f(t), "size": tuple(size)}) if callable(scene) \
                    else {**scene, "size": tuple(size)}
            stats = render_video(target, duration=duration, fps=fps, path=path, timeline=timeline, audio=audio)
        except _Fail as f:
            history.append({"attempt": k, "stage": f.stage, "message": f.message, "code": code})
            if verbose:
                print(f"[attempt {k}] failed at {f.stage}: {f.message.splitlines()[0][:160]}")
            continue
        except Exception as e:
            history.append({"attempt": k, "stage": stage, "message": _explain(e, code), "code": code})
            if verbose:
                print(f"[attempt {k}] failed at {stage}: {history[-1]['message'][:160]}")
            continue
        if verbose:
            print(f"[attempt {k}] ok: {stats['frames']} frames -> {path}" +
                  ("".join(f"\n  {m}" for m in problems) if problems else ""))
        return {"ok": True, "video": path, "code": code, "attempts": k, "history": history, "render": stats}
    return {"ok": False, "video": None, "code": history[-1]["code"] if history else "",
            "attempts": max_tries, "history": history, "render": None}
