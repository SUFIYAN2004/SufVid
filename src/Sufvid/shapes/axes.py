import math
from ..core.transform import to_px, scale_len
from ..core.interpolate import clamp
from ._common import col, wpx
from .text import draw_label

def axes(x_range=(-5.0, 5.0), y_range=(-3.0, 3.0), pos=(0.0, 0.0), size=(10.0, 6.0),
         x_step=1.0, y_step=1.0, grid=True, labels=True, label_size=20,
         color="#8b98a5", **extra):
    """Plot box. Data ranges map onto a box of `size` world units centered at `pos`."""
    box = {"x_range": tuple(x_range), "y_range": tuple(y_range),
           "pos": tuple(pos), "size": tuple(size)}
    s = {"type": "axes", "box": box, "x_step": x_step, "y_step": y_step,
         "grid": grid, "labels": labels, "label_size": label_size, "color": color}
    s.update(extra)
    return s

def axes_point(a, x, y):
    """Data coords -> world coords. `a` is an axes shape (or its box). Works on arrays."""
    b = a["box"] if "box" in a else a
    (x0, x1), (y0, y1) = b["x_range"], b["y_range"]
    (px, py), (w, h) = b["pos"], b["size"]
    return (px - w / 2 + (x - x0) / (x1 - x0) * w,
            py - h / 2 + (y - y0) / (y1 - y0) * h)

def _ticks(lo, hi, step):
    k0 = math.ceil(lo / step - 1e-9)
    k1 = math.floor(hi / step + 1e-9)
    return [k * step for k in range(k0, k1 + 1)]

def draw_axes(d, view, s, t):
    c = col(s, "color")
    if c is None:
        return
    b = s["box"]
    (x0, x1), (y0, y1) = b["x_range"], b["y_range"]
    P = lambda x, y: to_px(view, *axes_point(b, x, y))
    xt, yt = _ticks(x0, x1, s["x_step"]), _ticks(y0, y1, s["y_step"])
    if s.get("grid", True):
        gc = (c[0], c[1], c[2], int(c[3] * 0.22))
        for x in xt:
            d.line([P(x, y0), P(x, y1)], fill=gc, width=wpx(view, 1))
        for y in yt:
            d.line([P(x0, y), P(x1, y)], fill=gc, width=wpx(view, 1))
    ay, ax_ = clamp(0.0, y0, y1), clamp(0.0, x0, x1)
    lw = wpx(view, 2.5)
    d.line([P(x0, ay), P(x1, ay)], fill=c, width=lw)
    d.line([P(ax_, y0), P(ax_, y1)], fill=c, width=lw)
    tl = scale_len(view, 0.08)
    for x in xt:
        px, py = P(x, ay)
        d.line([(px, py - tl), (px, py + tl)], fill=c, width=wpx(view, 2))
    for y in yt:
        px, py = P(ax_, y)
        d.line([(px - tl, py), (px + tl, py)], fill=c, width=wpx(view, 2))
    if s.get("labels", True):
        ls = s.get("label_size", 20)
        for x in xt:
            wx, wy = axes_point(b, x, ay)
            draw_label(d, view, (wx, wy - 0.15), f"{x:g}", ls, c, "mt")
        for y in yt:
            if abs(y) < 1e-12:
                continue
            wx, wy = axes_point(b, ax_, y)
            draw_label(d, view, (wx - 0.15, wy), f"{y:g}", ls, c, "rm")
