from .interpolate import clamp

NAMED = {
    "white": (255, 255, 255, 255), "black": (0, 0, 0, 255),
    "red": (255, 85, 85, 255), "green": (80, 250, 123, 255),
    "blue": (98, 144, 255, 255), "yellow": (255, 214, 10, 255),
    "orange": (255, 159, 64, 255), "purple": (189, 147, 249, 255),
    "gray": (140, 150, 160, 255), "transparent": (0, 0, 0, 0),
}

def parse_color(c):
    """Return an (r, g, b, a) int tuple."""
    if isinstance(c, str):
        if c in NAMED:
            return NAMED[c]
        h = c.lstrip("#")
        if len(h) in (3, 4):
            h = "".join(ch * 2 for ch in h)
        if len(h) == 6:
            h += "ff"
        if len(h) != 8:
            raise ValueError(f"bad color {c!r}")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4, 6))
    t = tuple(int(round(v)) for v in c)
    if len(t) == 3:
        t += (255,)
    if len(t) != 4:
        raise ValueError(f"bad color {c!r}")
    return t

def with_alpha(c, a):
    """Same color with alpha a in [0,1]."""
    r, g, b, _ = parse_color(c)
    return (r, g, b, int(round(255 * clamp(a))))

def mix_colors(c1, c2, u):
    """Blend two colors (including alpha) by u in [0,1]."""
    a, b = parse_color(c1), parse_color(c2)
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))
