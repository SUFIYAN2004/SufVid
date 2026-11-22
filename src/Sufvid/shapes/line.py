from ..core.transform import to_px
from ..core.interpolate import clamp
from ._common import col, wpx

def line(p0, p1, color="white", w=3, progress=1.0, **extra):
    """Straight line from p0 to p1 (world). progress in [0,1] draws only the first part."""
    s = {"type": "line", "p0": tuple(p0), "p1": tuple(p1),
         "color": color, "w": w, "progress": progress}
    s.update(extra)
    return s

def draw_line(d, view, s, t):
    c = col(s, "color")
    p = clamp(s.get("progress", 1.0))
    if c is None or p <= 0:
        return
    (x0, y0), (x1, y1) = s["p0"], s["p1"]
    q = (x0 + (x1 - x0) * p, y0 + (y1 - y0) * p)
    d.line([to_px(view, x0, y0), to_px(view, *q)], fill=c, width=wpx(view, s["w"]))
