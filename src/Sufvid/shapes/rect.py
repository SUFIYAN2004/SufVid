from ..core.transform import to_px, scale_len
from ._common import col, stroke_of

def rect(center=(0.0, 0.0), size=(2.0, 1.0), fill=None, stroke="white",
         stroke_w=3, radius=0.0, **extra):
    """Axis-aligned rectangle. center/size/radius in world units (radius rounds corners)."""
    s = {"type": "rect", "center": tuple(center), "size": tuple(size), "fill": fill,
         "stroke": stroke, "stroke_w": stroke_w, "radius": radius}
    s.update(extra)
    return s

def draw_rect(d, view, s, t):
    cx, cy = s["center"]
    w, h = s["size"]
    x0, y0 = to_px(view, cx - w / 2, cy + h / 2)
    x1, y1 = to_px(view, cx + w / 2, cy - h / 2)
    stroke, sw = stroke_of(view, s)
    r = scale_len(view, s.get("radius", 0.0))
    if r > 0:
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=col(s, "fill"),
                            outline=stroke, width=sw)
    else:
        d.rectangle([x0, y0, x1, y1], fill=col(s, "fill"), outline=stroke, width=sw)
