"""Pre-render lint. check_scene() finds mistakes an LLM (or a human) makes BEFORE spending minutes rendering."""
import numpy as np
from ..shapes.text import get_font
from ..shapes.mathtext import _tex_mask
from ..anim.timeline import play, _schedule
from ..render.frame import DRAWERS

_LABELS = ("text", "mathtext")
_NEEDS_PROGRESS = ("line", "arrow", "graph")

def _bbox(s, scale):
    """World-space (x0, y0, x1, y1) of a shape as drawn, or None if it has no extent."""
    t = s["type"]
    if t == "text":
        if not s.get("text"):
            return None
        px = max(1, int(round(s["size"])))
        l, top, r, b = get_font(s.get("font", "sans"), px).getbbox(s["text"], anchor=s.get("anchor", "mm"))
        x, y = s["pos"]
        return (x + l / scale, y - b / scale, x + r / scale, y - top / scale)
    if t == "mathtext":
        px = max(4, int(round(s["size"])))
        w, h = _tex_mask(s["tex"], px).size
        a = s.get("anchor", "mm")
        ox = {"l": 0, "m": w / 2, "r": w}[a[0]]
        oy = {"t": 0, "m": h / 2, "b": h}[a[1]]
        x, y = s["pos"]
        return (x - ox / scale, y - (h - oy) / scale, x + (w - ox) / scale, y + oy / scale)
    if t == "circle":
        (x, y), r = s["center"], s["r"]
        return (x - r, y - r, x + r, y + r)
    if t == "rect":
        (x, y), (w, h) = s["center"], s["size"]
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
    if t in ("line", "arrow"):
        (x0, y0), (x1, y1) = s["p0"], s["p1"]
        p = s.get("progress", 1.0)
        qx, qy = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
        return (min(x0, qx), min(y0, qy), max(x0, qx), max(y0, qy))
    if t == "polygon":
        pts = np.asarray(s["points"], dtype=float)
        return (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()) if len(pts) else None
    if t == "points":
        xy, r = s["xy"], s["r"]
        return (xy[:, 0].min() - r, xy[:, 1].min() - r, xy[:, 0].max() + r, xy[:, 1].max() + r) if len(xy) else None
    if t == "heatmap":
        return tuple(s["rect"])
    if t in ("axes", "graph"):
        (x, y), (w, h) = s["box"]["pos"], s["box"]["size"]
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
    return None

def _visible(s):
    if float(s.get("opacity", 1.0)) <= 0.05:
        return False
    if s["type"] in _NEEDS_PROGRESS and float(s.get("progress", 1.0)) <= 0.0:
        return False
    if s["type"] == "text" and not s.get("text"):
        return False
    return True

def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def _inter(a, b):
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))

def _name(s):
    ident = f"'{s['id']}'" if "id" in s else ""
    if s["type"] == "text":
        return f"text {ident} \"{s['text'].rstrip('|')[:30]}\"".replace("  ", " ")
    if s["type"] == "mathtext":
        return f"mathtext {ident} {s['tex'][:30]!r}".replace("  ", " ")
    return f"{s['type']} {ident}".strip()

def _sample_times(scene, timeline, duration):
    ts = {0.0}
    if timeline:
        for a in _schedule(scene, timeline):
            ts.update([a["start"], a["end"]])
    if duration:
        ts.add(float(duration))
    ts = sorted(ts)
    ts = sorted(set(ts + [(a + b) / 2 for a, b in zip(ts, ts[1:])]))
    if len(ts) > 80:
        ts = [ts[i] for i in np.linspace(0, len(ts) - 1, 80).astype(int)]
    return ts

def check_scene(scene, timeline=None, duration=None, times=None):
    """Lint a scene BEFORE rendering. Returns a list of strings starting with 'error:' or 'warning:'
    (empty list = clean). scene: a scene dict, or a function t -> scene dict (then give duration=).
    Errors: unknown shape type / animation target, draw/type_text on the wrong shape type, text or math
    overlapping other text or math, text cut off by the frame edge, any visible shape completely outside
    the frame. Warnings: tiny fonts, animations that start after `duration`.
    Overlaps are evaluated at every animation start/end and midpoint, so moving or typed text is covered."""
    msgs, seen = [], set()
    def add(level, m, key=None):
        full = f"{level}: {m}"
        if key is not None:                       # one report per problem (first moment it appears), not one per sample time
            if key in seen:
                return
            seen.add(key)
        if full not in msgs:
            msgs.append(full)

    fn = scene if callable(scene) else None
    if fn is not None and not times:
        if not duration:
            return ["error: a scene function needs duration=seconds so it can be checked"]
        times = [duration * i / 4 for i in range(5)]
    try:
        first = fn(times[0]) if fn else scene
    except Exception as e:
        return [f"error: scene function raised {type(e).__name__}: {e} (at t={times[0]})"]
    if not isinstance(first, dict) or "shapes" not in first or "size" not in first:
        return ["error: a scene must be a dict with 'size' (width, height) and 'shapes' (a list)"]

    W, H = first["size"]
    view = first.get("view", {})
    vh = float(view.get("height", 9.0))
    cx, cy = view.get("center", (0.0, 0.0))
    vw = vh * W / H
    frame = (cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)
    scale = H / vh                                   # output pixels per world unit

    if timeline:
        try:
            sched = _schedule(first, timeline)
        except (KeyError, ValueError) as e:
            return [f"error: {e.args[0] if e.args else e}"]
        if duration:
            for a in sched:
                if a["type"] != "wait" and a["start"] >= duration - 1e-9:
                    add("warning", f"{a['type']}({a['target']!r}) starts at {a['start']:.2f}s, "
                                   f"after the video ends ({duration:.2f}s)")
    else:
        try:
            _schedule(first, [])
        except ValueError as e:
            return [f"error: {e}"]

    for i, s in enumerate(first["shapes"]):
        if s.get("type") not in DRAWERS:
            add("error", f"shape #{i} has unknown type {s.get('type')!r}; known: {sorted(DRAWERS)}")
        elif s["type"] == "text" and s["size"] < H / 40:
            add("warning", f"{_name(s)} has font size {s['size']} px, which is hard to read at {W}x{H}; use >= {int(H / 40)}")
    if any(m.startswith("error") for m in msgs):
        return msgs

    for t in (times if times else _sample_times(first, timeline, duration)):
        try:
            sc = fn(t) if fn else first
            sc = play(sc, timeline, t) if timeline else sc
        except Exception as e:
            add("error", f"evaluating the scene at t={t:.2f}s raised {type(e).__name__}: {e}")
            break
        items = []
        for idx, s in enumerate(sc["shapes"]):
            if s.get("type") not in DRAWERS or not _visible(s):
                continue
            b = _bbox(s, scale)
            if b is not None:
                items.append((idx, s, b))
        for idx, s, b in items:                                  # frame edge
            ix = _inter(b, frame)
            if ix[2] < ix[0] or ix[3] < ix[1]:
                add("error", f"{_name(s)} is completely outside the frame at t={t:.2f}s "
                             f"(frame spans x {frame[0]:.1f}..{frame[2]:.1f}, y {frame[1]:.1f}..{frame[3]:.1f})",
                    key=("out", idx))
            elif s["type"] in _LABELS and _area(b) > 0 and 1 - _area(ix) / _area(b) > 0.02:
                add("error", f"{_name(s)} is cut off by the frame edge at t={t:.2f}s; move it inward or shrink it",
                    key=("cut", idx))
        labels = [(idx, s, b) for idx, s, b in items if s["type"] in _LABELS]
        for i in range(len(labels)):                             # label vs label
            for j in range(i + 1, len(labels)):
                (ia, sa, ba), (ib, sb, bb) = labels[i], labels[j]
                ix = _inter(ba, bb)
                if ix[2] <= ix[0] or ix[3] <= ix[1]:
                    continue
                small = min(_area(ba), _area(bb))
                if small > 0 and _area(ix) / small > 0.15:
                    add("error", f"{_name(sa)} overlaps {_name(sb)} at t={t:.2f}s; move one of them "
                                 f"(boxes: x {ba[0]:.1f}..{ba[2]:.1f} y {ba[1]:.1f}..{ba[3]:.1f} vs "
                                 f"x {bb[0]:.1f}..{bb[2]:.1f} y {bb[1]:.1f}..{bb[3]:.1f})",
                        key=("overlap", ia, ib))
    return msgs
