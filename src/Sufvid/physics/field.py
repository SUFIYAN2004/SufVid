import numpy as np

def _efield(p, ch):
    d = p - ch[:, :2]
    r3 = np.maximum((d * d).sum(1) ** 1.5, 1e-12)
    return (ch[:, 2:3] * d / r3[:, None]).sum(0)

def _trace(p0, ch, sign, bounds, step, max_steps, stop_r):
    xmin, xmax, ymin, ymax = bounds
    pts = [p0]
    p = p0.copy()
    for _ in range(max_steps):
        e = _efield(p, ch)
        m = np.hypot(e[0], e[1])
        if m < 1e-12:
            break
        pm = p + 0.5 * step * sign * e / m
        e2 = _efield(pm, ch)
        m2 = np.hypot(e2[0], e2[1])
        if m2 < 1e-12:
            break
        p = p + step * sign * e2 / m2
        pts.append(p.copy())
        dist = np.hypot(ch[:, 0] - p[0], ch[:, 1] - p[1])
        j = int(np.argmin(dist))
        if dist[j] < stop_r:
            pts[-1] = ch[j, :2].copy()
            break
        if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax):
            break
    return np.array(pts)

def field_lines(charges, bounds=(-5.0, 5.0, -3.0, 3.0), n_per_charge=12, step=0.04,
                max_steps=2000, stop_r=0.12):
    """Electric field lines for point charges [(x, y, q), ...] as a list of (N, 2) arrays.
    Seeds ring the positive charges (or the negative ones if there are none); the strongest
    source gets n_per_charge lines, others in proportion to |q|. Lines end on an opposite
    charge (snapped onto it) or just outside bounds=(xmin, xmax, ymin, ymax)."""
    ch = np.asarray(charges, dtype=float)
    if ch.ndim != 2 or ch.shape[1] != 3 or len(ch) == 0:
        raise ValueError("charges must be a non-empty list of (x, y, q)")
    if (ch[:, 2] == 0).any():
        raise ValueError("charges must be non-zero")
    pos = ch[:, 2] > 0
    sources, sign = (ch[pos], 1.0) if pos.any() else (ch, -1.0)
    qmax = np.abs(sources[:, 2]).max()
    r_seed = 2.0 * stop_r
    lines = []
    for cx, cy, q in sources:
        n = max(2, int(round(n_per_charge * abs(q) / qmax)))
        for k in range(n):
            a = 2.0 * np.pi * k / n
            p0 = np.array([cx + r_seed * np.cos(a), cy + r_seed * np.sin(a)])
            tr = _trace(p0, ch, sign, bounds, step, max_steps, stop_r)
            lines.append(np.vstack([[cx, cy], tr]))
    return lines
