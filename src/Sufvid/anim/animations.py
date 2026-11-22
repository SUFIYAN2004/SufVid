import numpy as np
from ..core.easing import EASINGS

def _anim(kind, target, at, dur, easing, **extra):
    if easing not in EASINGS:
        raise KeyError(f"unknown easing {easing!r}; choose from {sorted(EASINGS)}")
    a = {"type": kind, "target": target, "at": at, "dur": dur, "easing": easing}
    a.update(extra)
    return a

def fade_in(target, at=None, dur=0.5, easing="smooth"):
    """Opacity 0 -> 1. The shape is hidden before the animation starts."""
    return _anim("fade_in", target, at, dur, easing, to=1.0)

def fade_out(target, at=None, dur=0.5, easing="smooth"):
    """Opacity 1 -> 0 (stays hidden afterwards)."""
    return _anim("fade_out", target, at, dur, easing, to=0.0)

def draw(target, at=None, dur=1.0, easing="smooth"):
    """Progress 0 -> 1 on a line, arrow or graph (hidden before it starts)."""
    return _anim("draw", target, at, dur, easing, to=1.0)

def move_to(target, to, at=None, dur=1.0, easing="smooth"):
    """Move the shape's anchor to world point `to`, starting from wherever it is."""
    return _anim("move_to", target, at, dur, easing, to=tuple(float(v) for v in to))

def follow_path(target, path, at=None, dur=1.0, easing="linear"):
    """Move the anchor along `path`: an (N, 2) array of world points, or a function u in [0,1] -> (x, y)."""
    if not callable(path):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
            raise ValueError("path must be an (N, 2) array with N >= 2, or a function u -> (x, y)")
    return _anim("follow_path", target, at, dur, easing, path=path)

def type_text(target, at=None, dur=None, cps=20.0, cursor=True):
    """Reveal a text shape character by character. dur defaults to len(text)/cps."""
    return _anim("type_text", target, at, dur, "linear", to=1.0, cps=float(cps), cursor=cursor)

def wait(dur=1.0, at=None):
    """Do nothing for dur seconds (pushes the sequence forward / extends the video)."""
    return {"type": "wait", "target": None, "at": at, "dur": dur, "easing": "linear"}
