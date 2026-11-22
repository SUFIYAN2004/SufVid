import os
from functools import lru_cache
from PIL import ImageFont
from ..core.transform import to_px
from ._common import col

_NAMES = {"sans": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "mono": "DejaVuSansMono.ttf"}

@lru_cache(maxsize=None)
def _font_path(kind):
    fname = _NAMES[kind]
    cands = []
    try:
        import matplotlib
        cands.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", fname))
    except Exception:
        pass
    cands.append("/usr/share/fonts/truetype/dejavu/" + fname)
    for c in cands:
        if os.path.exists(c):
            return c
    return None

@lru_cache(maxsize=64)
def get_font(kind, px):
    """Cached PIL font of the given kind ('sans'|'bold'|'mono') and pixel size."""
    path = _font_path(kind)
    if path:
        return ImageFont.truetype(path, px)
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()

def text(string, pos=(0.0, 0.0), size=36, color="white", anchor="mm", font="sans", **extra):
    """Single-line text. size = font pixel size at output resolution.
    anchor: two letters, horizontal l/m/r + vertical t/m/b (e.g. 'mm', 'lm', 'mt')."""
    s = {"type": "text", "text": string, "pos": tuple(pos), "size": size,
         "color": color, "anchor": anchor, "font": font}
    s.update(extra)
    return s

def draw_label(d, view, xy, string, size, color, anchor="mm", font="sans"):
    x, y = to_px(view, *xy)
    px = max(1, int(round(size * view["ss"])))
    d.text((x, y), string, font=get_font(font, px), fill=color, anchor=anchor)

def draw_text(d, view, s, t):
    c = col(s, "color")
    if c is None or not s["text"]:
        return
    draw_label(d, view, s["pos"], s["text"], s["size"], c,
               s.get("anchor", "mm"), s.get("font", "sans"))
