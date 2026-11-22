from ..core.colors import parse_color

def col(shape, key, default=None):
    c = shape.get(key, default)
    if c is None:
        return None
    r, g, b, a = parse_color(c)
    op = min(max(float(shape.get("opacity", 1.0)), 0.0), 1.0)
    return (r, g, b, int(round(a * op)))

def wpx(view, w):
    return max(1, int(round(w * view["ss"])))

def stroke_of(view, shape):
    c = col(shape, "stroke")
    w = int(round(shape.get("stroke_w", 3) * view["ss"]))
    if c is None or w <= 0:
        return None, 1
    return c, w
