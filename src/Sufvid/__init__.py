__version__ = "0.1.0"

from .core.colors import with_alpha, mix_colors
from .core.interpolate import lerp, remap, clamp
from .core.easing import ease
from .shapes.circle import circle
from .shapes.line import line
from .shapes.rect import rect
from .shapes.arrow import arrow
from .shapes.polygon import polygon
from .shapes.text import text
from .shapes.mathtext import mathtext
from .shapes.axes import axes, axes_point
from .shapes.graph import graph
from .shapes.heatmap import heatmap
from .shapes.points import points
from .anim.animations import fade_in, fade_out, move_to, draw, type_text, follow_path, wait
from .anim.timeline import play, timeline_duration
from .physics.ode import simulate
from .physics.field import field_lines
from .nn.mlp import train_mlp
from .audio.osc import synth
from .audio.sfx import sfx
from .audio.voice import voice
from .audio.mixer import mix
from .render.frame import render_frame, save_png
from .render.video import render_video
from .agent.check import check_scene
from .agent.loop import agent_loop

__all__ = [
    "with_alpha", "mix_colors", "lerp", "remap", "clamp", "ease",
    "circle", "line", "rect", "arrow", "polygon", "text", "mathtext",
    "axes", "axes_point", "graph", "heatmap", "points",
    "fade_in", "fade_out", "move_to", "draw", "type_text", "follow_path", "wait",
    "play", "timeline_duration",
    "simulate", "field_lines", "train_mlp",
    "synth", "sfx", "voice", "mix",
    "render_frame", "save_png", "render_video",
    "check_scene", "agent_loop",
]
