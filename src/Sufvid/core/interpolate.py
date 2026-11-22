def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x

def lerp(a, b, u):
    """Linear interpolation between numbers a and b."""
    return a + (b - a) * u

def lerp_vec(a, b, u):
    """Linear interpolation between equal-length tuples/lists."""
    return tuple(x + (y - x) * u for x, y in zip(a, b))

def remap(t, t0, t1):
    """Map time t in [t0, t1] to progress u in [0, 1] (clamped)."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return clamp((t - t0) / (t1 - t0))
