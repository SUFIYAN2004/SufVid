import math
from ..core.transform import to_px
from ..core.interpolate import clamp
from ._common import col, wpx

def arrow(p0, p1, color="white", w=4, head=0.35, progress=1.0, **extra):
    """Arrow from p0 to p1 (world). head = head length in world units."""
    s = {"type": "arrow", "p0": tuple(p0), "p1": tuple(p1), "color": color,
         "w": w, "head": head, "progress": progress}
    s.update(extra)
    return s

def draw_arrow(d, view, s, t):
    c = col(s, "color")
    p = clamp(s.get("progress", 1.0))
    if c is None or p <= 0:
        return
    (x0, y0), (x1, y1) = s["p0"], s["p1"]
    qx, qy = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
    L = math.hypot(qx - x0, qy - y0)
    if L < 1e-9:
        return
    ux, uy = (qx - x0) / L, (qy - y0) / L
    hl = min(float(s.get("head", 0.35)), L)
    hw = 0.55 * hl
    bx, by = qx - ux * hl, qy - uy * hl
    if L - hl > 1e-9:
        d.line([to_px(view, x0, y0), to_px(view, bx, by)], fill=c,
               width=wpx(view, s.get("w", 4)))
    d.polygon([to_px(view, qx, qy),
               to_px(view, bx - uy * hw, by + ux * hw),
               to_px(view, bx + uy * hw, by - ux * hw)], fill=c)
