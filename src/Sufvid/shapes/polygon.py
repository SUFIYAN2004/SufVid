from ..core.transform import to_px
from ._common import col, stroke_of

def polygon(points, fill=None, stroke="white", stroke_w=3, closed=True, **extra):
    """Polygon / polyline through world points. closed=False draws an open path."""
    s = {"type": "polygon", "points": [tuple(p) for p in points], "fill": fill,
         "stroke": stroke, "stroke_w": stroke_w, "closed": closed}
    s.update(extra)
    return s

def draw_polygon(d, view, s, t):
    pts = [to_px(view, x, y) for x, y in s["points"]]
    closed = s.get("closed", True)
    fill = col(s, "fill")
    if fill is not None and closed and len(pts) >= 3:
        d.polygon(pts, fill=fill)
    stroke, sw = stroke_of(view, s)
    if stroke is not None and len(pts) >= 2:
        path = pts + [pts[0]] if closed else pts
        d.line(path, fill=stroke, width=sw, joint="curve")
