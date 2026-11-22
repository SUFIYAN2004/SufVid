import numpy as np
from PIL import Image
from ..core.transform import to_px
from ..core.colors import parse_color
from ..core.interpolate import clamp

def heatmap(data, rect=(-4.0, -3.0, 4.0, 3.0), colors=("#ff5555", "#6290ff"),
            index=0.0, alpha=0.45, vmin=0.0, vmax=1.0, **extra):
    """Color a 2-D array over a world rectangle rect=(x0, y0, x1, y1).
    data[row, col]: row 0 is the LOWEST y, col 0 the lowest x.
    A 3-D array (frames, rows, cols) is blended between frames at fractional `index`.
    Values vmin..vmax map from colors[0] to colors[1]; alpha is the fill opacity."""
    data = np.asarray(data, dtype=float)
    if data.ndim not in (2, 3) or min(data.shape) < 1:
        raise ValueError("heatmap data must be a 2-D (rows, cols) or 3-D (frames, rows, cols) array")
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        raise ValueError("rect must be (x0, y0, x1, y1) with x1 > x0 and y1 > y0")
    if vmax <= vmin:
        raise ValueError("vmax must be > vmin")
    s = {"type": "heatmap", "data": data, "rect": (x0, y0, x1, y1), "colors": tuple(colors),
         "index": index, "alpha": alpha, "vmin": vmin, "vmax": vmax}
    s.update(extra)
    return s

def _frame(data, index):
    if data.ndim == 2:
        return data
    T = data.shape[0]
    i = clamp(float(index), 0.0, T - 1.0)
    i0 = int(np.floor(i))
    i1 = min(i0 + 1, T - 1)
    r = i - i0
    return data[i0] * (1.0 - r) + data[i1] * r

def draw_heatmap(d, view, s, t):
    a = clamp(float(s["alpha"])) * clamp(float(s.get("opacity", 1.0)))
    if a <= 0:
        return
    x0, y0, x1, y1 = s["rect"]
    px0, py0 = to_px(view, x0, y1)
    px1, py1 = to_px(view, x1, y0)
    w, h = int(round(px1 - px0)), int(round(py1 - py0))
    if w < 1 or h < 1:
        return
    z = _frame(s["data"], s["index"])
    u = np.clip((z - s["vmin"]) / (s["vmax"] - s["vmin"]), 0.0, 1.0)[::-1]
    c0 = np.array(parse_color(s["colors"][0])[:3], dtype=float)
    c1 = np.array(parse_color(s["colors"][1])[:3], dtype=float)
    rgb = c0 * (1.0 - u[..., None]) + c1 * u[..., None]
    rgba = np.empty(u.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = np.rint(rgb)
    rgba[..., 3] = int(round(255 * a))
    im = Image.fromarray(rgba).resize((w, h), Image.Resampling.BILINEAR)
    view["img"].paste(im, (int(round(px0)), int(round(py0))), im)
