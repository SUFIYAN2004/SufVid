import numpy as np
from scipy.integrate import solve_ivp

_DEFAULTS = {
    "projectile": dict(v0=20.0, angle=45.0, g=9.81, drag=0.0, x0=0.0, y0=0.0),
    "pendulum": dict(L=1.0, theta0=1.0, omega0=0.0, g=9.81, damping=0.0, pivot=(0.0, 0.0)),
    "spring": dict(k=10.0, m=1.0, x0=1.0, v0=0.0, damping=0.0, rest=0.0, anchor=(0.0, 0.0)),
}

def _projectile(p):
    g, k = p["g"], p["drag"]
    th = np.radians(p["angle"])
    y0 = [p["x0"], p["y0"], p["v0"] * np.cos(th), p["v0"] * np.sin(th)]
    def rhs(t, s):
        return [s[2], s[3], -k * s[2], -g - k * s[3]]
    def ground(t, s):
        return s[1]
    ground.terminal, ground.direction = True, -1
    def out(Y):
        return {"pos": np.column_stack([Y[0], Y[1]]), "vel": np.column_stack([Y[2], Y[3]]),
                "energy": 0.5 * (Y[2] ** 2 + Y[3] ** 2) + g * Y[1]}
    return rhs, y0, None, [ground], out

def _pendulum(p):
    L, g, b = p["L"], p["g"], p["damping"]
    px, py = p["pivot"]
    def rhs(t, s):
        return [s[1], -(g / L) * np.sin(s[0]) - b * s[1]]
    def out(Y):
        th, w = Y
        return {"pos": np.column_stack([px + L * np.sin(th), py - L * np.cos(th)]),
                "theta": th, "omega": w,
                "energy": 0.5 * L ** 2 * w ** 2 + g * L * (1.0 - np.cos(th))}
    return rhs, [p["theta0"], p["omega0"]], 2 * 2 * np.pi * np.sqrt(L / g), None, out

def _spring(p):
    k, m, c = p["k"], p["m"], p["damping"]
    ax, ay = p["anchor"]
    def rhs(t, s):
        return [s[1], -(k / m) * s[0] - (c / m) * s[1]]
    def out(Y):
        x, v = Y
        return {"pos": np.column_stack([ax + p["rest"] + x, np.full_like(x, ay)]),
                "x": x, "v": v, "energy": 0.5 * m * v ** 2 + 0.5 * k * x ** 2}
    return rhs, [p["x0"], p["v0"]], 2 * 2 * np.pi * np.sqrt(m / k), None, out

_SYSTEMS = {"projectile": _projectile, "pendulum": _pendulum, "spring": _spring}

def simulate(kind, T=None, n=300, **params):
    """Integrate a physical system and sample it at n evenly spaced times in [0, T].
    kind: 'projectile' (v0, angle[deg], g, drag, x0, y0), 'pendulum' (L, theta0[rad], omega0,
    g, damping, pivot), 'spring' (k, m, x0, v0, damping, rest, anchor).
    T=None: projectile runs until it lands; others run two small-angle periods.
    Returns {'t', 'pos' (n,2), 'energy', ...extras, 'kind', 'params'} (energy per unit mass,
    except spring which uses m)."""
    if kind not in _SYSTEMS:
        raise KeyError(f"unknown kind {kind!r}; choose from {sorted(_SYSTEMS)}")
    p = dict(_DEFAULTS[kind])
    bad = [k for k in params if k not in p]
    if bad:
        raise KeyError(f"unknown parameter(s) {bad} for {kind!r}; valid: {sorted(p)}")
    p.update(params)
    if n < 2:
        raise ValueError("n must be >= 2")
    rhs, y0, T_def, events, out = _SYSTEMS[kind](p)
    t_end = T if T is not None else T_def
    until_event = t_end is None
    if not until_event and t_end <= 0:
        raise ValueError("T must be > 0")
    span_end = 1000.0 if until_event else float(t_end)
    sol = solve_ivp(rhs, (0.0, span_end), y0, method="DOP853", rtol=1e-10, atol=1e-12,
                    dense_output=True, events=events if until_event else None)
    if until_event:
        if len(sol.t_events[0]) == 0:
            raise ValueError("the projectile never lands; pass T=seconds")
        t_end = float(sol.t[-1])
    t = np.linspace(0.0, t_end, n)
    d = out(sol.sol(t))
    d.update(t=t, kind=kind, params=p)
    return d
