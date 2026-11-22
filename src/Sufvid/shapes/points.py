import numpy as np
from ..core.transform import to_px, scale_len
from ._common import col, stroke_of

def points(xy, labels=None, palette=("#ff5555", "#6290ff"), r=0.07,
           stroke="white", stroke_w=1.5, **extra):
    """Scatter dots. xy: (N, 2) world coords. labels: (N,) ints indexing into palette."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2)")
    lab = np.zeros(len(xy), dtype=int) if labels is None else np.asarray(labels).astype(int)
    if lab.shape != (len(xy),):
        raise ValueError("labels must have shape (N,)")
    if len(lab) and (lab.min() < 0 or lab.max() >= len(palette)):
        raise ValueError(f"labels must be in 0..{len(palette) - 1} (palette has {len(palette)} colors)")
    s = {"type": "points", "xy": xy, "labels": lab, "palette": tuple(palette), "r": float(r),
         "stroke": stroke, "stroke_w": stroke_w}
    s.update(extra)
    return s

def draw_points(d, view, s, t):
    op = s.get("opacity", 1.0)
    fills = [col({"c": c, "opacity": op}, "c") for c in s["palette"]]
    stroke, w = stroke_of(view, s)
    r = scale_len(view, s["r"])
    X, Y = to_px(view, s["xy"][:, 0], s["xy"][:, 1])
    for x, y, k in zip(X.tolist(), Y.tolist(), s["labels"].tolist()):
        d.ellipse([x - r, y - r, x + r, y + r], fill=fills[k], outline=stroke, width=w)
