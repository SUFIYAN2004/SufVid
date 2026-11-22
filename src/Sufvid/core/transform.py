def make_view(scene, ss=1):
    W, H = scene["size"]
    v = scene.get("view", {})
    height = float(v.get("height", 9.0))
    cx, cy = v.get("center", (0.0, 0.0))
    return {"W": W * ss, "H": H * ss, "ss": ss,
            "scale": H * ss / height, "cx": cx, "cy": cy}

def to_px(view, x, y):
    return ((x - view["cx"]) * view["scale"] + view["W"] / 2,
            view["H"] / 2 - (y - view["cy"]) * view["scale"])

def scale_len(view, length):
    return length * view["scale"]
