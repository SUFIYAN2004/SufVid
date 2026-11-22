from ..core.easing import ease
from ..core.interpolate import lerp, lerp_vec, remap

_DRAWABLE = ("line", "arrow", "graph")

def _anchor(s):
    if "center" in s: return tuple(s["center"])
    if "pos" in s: return tuple(s["pos"])
    if "p0" in s: return tuple(s["p0"])
    if "points" in s:
        n = len(s["points"])
        return (sum(p[0] for p in s["points"]) / n, sum(p[1] for p in s["points"]) / n)
    if "box" in s: return tuple(s["box"]["pos"])
    raise ValueError(f"shape type {s.get('type')!r} has no position to move")

def _translate(s, dx, dy):
    s = dict(s)
    sh = lambda p: (p[0] + dx, p[1] + dy)
    if "center" in s: s["center"] = sh(s["center"])
    elif "pos" in s: s["pos"] = sh(s["pos"])
    elif "p0" in s:
        s["p0"], s["p1"] = sh(s["p0"]), sh(s["p1"])
    elif "points" in s: s["points"] = [sh(p) for p in s["points"]]
    elif "box" in s:
        b = dict(s["box"]); b["pos"] = sh(b["pos"]); s["box"] = b
    return s

def _path_point(path, u):
    if callable(path):
        x, y = path(u)
        return (float(x), float(y))
    n = len(path)
    f = u * (n - 1)
    i = min(int(f), n - 2)
    r = f - i
    p0, p1 = path[i], path[i + 1]
    return (float(p0[0] + (p1[0] - p0[0]) * r), float(p0[1] + (p1[1] - p0[1]) * r))

def _check_target(kind, shape):
    if kind == "draw" and shape["type"] not in _DRAWABLE:
        raise ValueError(f"draw() works on {_DRAWABLE}, not {shape['type']!r}; use fade_in instead")
    if kind == "type_text" and shape["type"] != "text":
        raise ValueError(f"type_text() works on 'text' shapes, not {shape['type']!r}")
    if kind in ("move_to", "follow_path"):
        _anchor(shape)

def _schedule(scene, timeline):
    by_id = {}
    for s in scene["shapes"]:
        if "id" in s:
            if s["id"] in by_id:
                raise ValueError(f"duplicate shape id {s['id']!r}")
            by_id[s["id"]] = s
    out, cursor, prev_start = [], 0.0, 0.0
    for a in timeline:
        a = dict(a)
        kind = a["type"]
        shape = None
        if kind != "wait":
            shape = by_id.get(a["target"])
            if shape is None:
                raise KeyError(f"no shape with id {a['target']!r}; ids in scene: {sorted(map(str, by_id))}")
            _check_target(kind, shape)
        dur = a["dur"]
        if dur is None:
            if a["cps"] <= 0:
                raise ValueError("cps must be > 0")
            dur = len(shape["text"]) / a["cps"]
        if dur < 0:
            raise ValueError("dur must be >= 0")
        at = a["at"]
        if at is None: start = cursor
        elif at == "with": start = prev_start
        elif isinstance(at, (int, float)): start = float(at)
        else: raise ValueError(f"bad 'at' {at!r}: use None, 'with' or seconds")
        a.update(dur=float(dur), start=start, end=start + float(dur))
        out.append(a)
        cursor = max(cursor, a["end"])
        prev_start = start
    return out

def timeline_duration(scene, timeline):
    """Time (seconds) at which the last animation ends."""
    sched = _schedule(scene, timeline)
    return max((a["end"] for a in sched), default=0.0)

def _by_kind(anims, kinds):
    return sorted((a for a in anims if a["type"] in kinds), key=lambda a: a["start"])

def _fold_scalar(anims, v, t):
    for a in anims:
        if t >= a["end"]:
            v = a["to"]
        elif t > a["start"]:
            return lerp(v, a["to"], ease(a["easing"], remap(t, a["start"], a["end"])))
        else:
            break
    return v

def _fold_pos(anims, p, t):
    for a in anims:
        if t >= a["end"]:
            p = a["to"] if a["type"] == "move_to" else _path_point(a["path"], 1.0)
        elif t > a["start"]:
            u = ease(a["easing"], remap(t, a["start"], a["end"]))
            return lerp_vec(p, a["to"], u) if a["type"] == "move_to" else _path_point(a["path"], u)
        else:
            break
    return p

def _apply(shape, anims, t):
    s = dict(shape)
    fades = _by_kind(anims, ("fade_in", "fade_out"))
    if fades:
        v0 = 0.0 if fades[0]["type"] == "fade_in" else 1.0
        s["opacity"] = float(shape.get("opacity", 1.0)) * _fold_scalar(fades, v0, t)
    draws = _by_kind(anims, ("draw",))
    if draws:
        s["progress"] = float(shape.get("progress", 1.0)) * _fold_scalar(draws, 0.0, t)
    typ = _by_kind(anims, ("type_text",))
    if typ:
        full = shape["text"]
        f = _fold_scalar(typ, 0.0, t)
        k = min(len(full), int(f * len(full) + 1e-9))
        typing = any(a["cursor"] and a["start"] <= t < a["end"] for a in typ)
        s["text"] = full[:k] + ("|" if typing else "")
    moves = _by_kind(anims, ("move_to", "follow_path"))
    if moves:
        base = _anchor(shape)
        p = _fold_pos(moves, base, t)
        s = _translate(s, p[0] - base[0], p[1] - base[1])
    return s

def play(scene, timeline, t):
    """Pure: return a copy of `scene` with all animations evaluated at time t."""
    per_target = {}
    for a in _schedule(scene, timeline):
        if a["type"] != "wait":
            per_target.setdefault(a["target"], []).append(a)
    shapes = []
    for s in scene["shapes"]:
        anims = per_target.get(s.get("id"))
        shapes.append(_apply(s, anims, t) if anims else s)
    return {**scene, "shapes": shapes}
