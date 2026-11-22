from .interpolate import clamp

def linear(u): return u
def smooth(u): return u * u * (3 - 2 * u)
def ease_in(u): return u ** 3
def ease_out(u): return 1 - (1 - u) ** 3
def ease_in_out(u): return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2

def bounce_out(u):
    n1, d1 = 7.5625, 2.75
    if u < 1 / d1:
        return n1 * u * u
    if u < 2 / d1:
        u -= 1.5 / d1
        return n1 * u * u + 0.75
    if u < 2.5 / d1:
        u -= 2.25 / d1
        return n1 * u * u + 0.9375
    u -= 2.625 / d1
    return n1 * u * u + 0.984375

EASINGS = {
    "linear": linear, "smooth": smooth, "ease_in": ease_in,
    "ease_out": ease_out, "ease_in_out": ease_in_out, "bounce_out": bounce_out,
}

def ease(name, u):
    """Apply the named easing to u (clamped to [0,1])."""
    if name not in EASINGS:
        raise KeyError(f"unknown easing {name!r}; choose from {sorted(EASINGS)}")
    return EASINGS[name](clamp(u))
