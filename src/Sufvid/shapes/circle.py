from ..core.transform import to_px, scale_len
from ._common import col, stroke_of

def circle(center=(0.0, 0.0), r=1.0, fill=None, stroke="white", stroke_w=3, **extra):
    """Circle. center/r in world units; stroke_w in output pixels. extra: opacity."""
    s = {"type": "circle", "center": tuple(center), "r": float(r),
         "fill": fill, "stroke": stroke, "stroke_w": stroke_w}
    s.update(extra)
    return s

def draw_circle(d, view, shape, t):
    x, y = to_px(view, *shape["center"])
    r = scale_len(view, shape["r"])
    stroke, w = stroke_of(view, shape)
    d.ellipse([x - r, y - r, x + r, y + r], fill=col(shape, "fill"),
              outline=stroke, width=w)
