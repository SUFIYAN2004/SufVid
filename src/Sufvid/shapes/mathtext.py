import io
from functools import lru_cache
from PIL import Image, ImageOps
from ..core.transform import to_px
from ._common import col

@lru_cache(maxsize=256)
def _tex_mask(tex, px):
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    s = tex if "$" in tex else "$" + tex + "$"
    buf = io.BytesIO()
    math_to_image(s, buf, prop=FontProperties(size=px), dpi=72, format="png")
    buf.seek(0)
    ink = ImageOps.invert(Image.open(buf).convert("L"))
    pad = Image.new("L", (ink.width + 4, ink.height + 4), 0)
    pad.paste(ink, (2, 2))
    return pad

def mathtext(tex, pos=(0.0, 0.0), size=44, color="white", anchor="mm", **extra):
    """Math expression, e.g. mathtext(r'\frac{a}{b}'). size = pixel size at output resolution."""
    s = {"type": "mathtext", "tex": tex, "pos": tuple(pos), "size": size,
         "color": color, "anchor": anchor}
    s.update(extra)
    return s

def draw_mathtext(d, view, s, t):
    c = col(s, "color")
    if c is None or c[3] == 0:
        return
    px = max(4, int(round(s["size"] * view["ss"])))
    mask = _tex_mask(s["tex"], px)
    if c[3] < 255:
        k = c[3] / 255.0
        mask = mask.point(lambda v: int(v * k))
    x, y = to_px(view, *s["pos"])
    w, h = mask.size
    a = s.get("anchor", "mm")
    ox = {"l": 0, "m": w / 2, "r": w}[a[0]]
    oy = {"t": 0, "m": h / 2, "b": h}[a[1]]
    d.bitmap((int(round(x - ox)), int(round(y - oy))), mask, fill=c)
