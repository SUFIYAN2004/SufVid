import numpy as np
from ..core.transform import to_px
from ..core.interpolate import clamp
from ._common import col, wpx
from .axes import axes_point

def graph(ax, fn=None, xs=None, ys=None, n=400, color="#ffd60a", w=4, progress=1.0, **extra):
    """Curve on an axes. Give a vectorized fn(x) (sampled over the x range) or xs and ys arrays.
    progress in [0,1] draws only the first part of the curve."""
    box = ax["box"] if "box" in ax else ax
    if fn is not None:
        xs = np.linspace(box["x_range"][0], box["x_range"][1], n)
        with np.errstate(all="ignore"):
            ys = fn(xs)
    xs = np.asarray(xs, dtype=float)
    ys = np.broadcast_to(np.asarray(ys, dtype=float), xs.shape).copy()
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError("xs and ys must be 1-D arrays of the same length")
    s = {"type": "graph", "box": dict(box), "xs": xs, "ys": ys,
         "color": color, "w": w, "progress": progress}
    s.update(extra)
    return s

def draw_graph(d, view, s, t):
    c = col(s, "color")
    p = clamp(s.get("progress", 1.0))
    if c is None or p <= 0:
        return
    xs, ys, b = s["xs"], s["ys"], s["box"]
    k = max(2, int(round(p * (len(xs) - 1))) + 1)
    xs, ys = xs[:k], ys[:k]
    (x0, x1), (y0, y1) = b["x_range"], b["y_range"]
    ok = np.isfinite(ys) & np.isfinite(xs) & (ys >= y0) & (ys <= y1) & (xs >= x0) & (xs <= x1)
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return
    wx, wy = axes_point(b, xs, ys)
    X, Y = to_px(view, wx, wy)
    pts = np.column_stack([X, Y])
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if run.size >= 2:
            d.line(list(map(tuple, pts[run].tolist())), fill=c,
                   width=wpx(view, s["w"]), joint="curve")
