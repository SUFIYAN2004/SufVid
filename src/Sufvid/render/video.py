import os, shutil, subprocess, time, wave
import multiprocessing as mp
from .frame import render_frame
from ..anim.timeline import play, timeline_duration

_STATE = {}   # worker plumbing only (set once per process); rendering itself is pure

def _init(scene, timeline, fps):
    _STATE["scene"], _STATE["timeline"], _STATE["fps"] = scene, timeline, fps

def _work(i):
    sc, tl, fps = _STATE["scene"], _STATE["timeline"], _STATE["fps"]
    t = i / fps
    scene = sc(t) if callable(sc) else sc
    if tl is not None:
        scene = play(scene, tl, t)
    return render_frame(scene, t).tobytes()

def _wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()

def render_video(scene, duration=None, fps=30, path="out.mp4", workers=2, crf=18,
                 preset="veryfast", timeline=None, tail=0.5, audio=None):
    """Render frames t = i/fps and stream them into ffmpeg (no temp files).
    scene: a scene dict, or a pure function t -> scene dict.
    timeline: optional animation list applied with play().
    audio: optional path to a WAV (e.g. from mix(..., path=...)); it is muxed as AAC. The audio is
    padded with silence or cut so it matches the video length exactly.
    If duration is None it is max(timeline_duration + tail, audio length).
    Returns {'frames', 'seconds', 'fps', 'path'}."""
    first = scene(0.0) if callable(scene) else scene
    a_dur = None
    if audio is not None:
        if not os.path.exists(audio):
            raise FileNotFoundError(f"audio file not found: {audio!r}")
        a_dur = _wav_seconds(audio)
    if duration is None:
        cands = []
        if timeline is not None:
            cands.append(timeline_duration(first, timeline) + tail)
        if a_dur is not None:
            cands.append(a_dur)
        if not cands:
            raise ValueError("give duration=, or a timeline or audio to derive it from")
        duration = max(cands)
    W, H = first["size"]
    if W % 2 or H % 2:
        raise ValueError("width and height must be even for yuv420p")
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise RuntimeError("ffmpeg not found (Colab: !apt-get install -y ffmpeg)")
    n = int(round(duration * fps))
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio is not None:
        cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-af", "apad",
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    t0 = time.perf_counter()
    try:
        if workers <= 1:
            _init(scene, timeline, fps)
            for i in range(n):
                proc.stdin.write(_work(i))
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(workers, initializer=_init, initargs=(scene, timeline, fps)) as pool:
                for fb in pool.imap(_work, range(n), chunksize=2):   # ordered
                    proc.stdin.write(fb)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    except BaseException:
        proc.kill()
        raise
    err = proc.stderr.read().decode(errors="replace")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg failed:\n" + err)
    dt = time.perf_counter() - t0
    return {"frames": n, "seconds": dt, "fps": n / dt, "path": path}
