from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw
from ..core.colors import parse_color
from ..core.transform import make_view
from ..shapes.circle import draw_circle
from ..shapes.line import draw_line
from ..shapes.rect import draw_rect
from ..shapes.arrow import draw_arrow
from ..shapes.polygon import draw_polygon
from ..shapes.text import draw_text
from ..shapes.mathtext import draw_mathtext
from ..shapes.axes import draw_axes
from ..shapes.graph import draw_graph
from ..shapes.heatmap import draw_heatmap
from ..shapes.points import draw_points

SS = 2

DRAWERS = {
    "circle": draw_circle, "line": draw_line, "rect": draw_rect, "arrow": draw_arrow,
    "polygon": draw_polygon, "text": draw_text, "mathtext": draw_mathtext,
    "axes": draw_axes, "graph": draw_graph, "heatmap": draw_heatmap, "points": draw_points,
}

@lru_cache(maxsize=8)
def _background(w, h, rgb):
    return Image.new("RGB", (w, h), rgb)

def render_frame(scene, t, ss=SS):
    """Pure function: (scene, t) -> uint8 array of shape (H, W, 3). Draw order = list order."""
    W, H = scene["size"]
    view = make_view(scene, ss)
    bg = parse_color(scene.get("bg", "#0f1419"))[:3]
    img = _background(W * ss, H * ss, bg).copy()
    view["img"] = img
    d = ImageDraw.Draw(img, "RGBA")
    for shape in scene["shapes"]:
        fn = DRAWERS.get(shape["type"])
        if fn is None:
            raise KeyError(f"unknown shape type {shape['type']!r}; known: {sorted(DRAWERS)}")
        fn(d, view, shape, t)
    return np.array(img.reduce(ss))

def save_png(frame, path):
    """Write a (H, W, 3) uint8 array to a PNG file."""
    Image.fromarray(frame).save(path)
