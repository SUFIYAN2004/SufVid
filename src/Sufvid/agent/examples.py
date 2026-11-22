"""Eight complete example programs. Every one is executed by the test-suite (lint + render)."""

EXAMPLES = {}

EXAMPLES["shapes_and_timeline"] = r"""
from Sufvid import *
import numpy as np

scene = {"size": (1280, 720), "bg": "#101820", "view": {"height": 9.0}, "shapes": [
    text("Shapes and timelines", pos=(0, 3.8), size=46, font="bold", id="title"),
    circle((-4, 0), 1.0, fill=with_alpha("#6290ff", 0.4), stroke="#6290ff", id="c"),
    rect((0, 0), (2.0, 2.0), fill=with_alpha("#50fa7b", 0.4), stroke="#50fa7b", radius=0.2, id="r"),
    arrow((3, -1), (5.5, 1.5), color="#ffd60a", id="a"),
    text("fade_in, draw, move_to", pos=(0, -3.4), size=30, color="#8b98a5", id="cap"),
]}
timeline = [
    fade_in("title"),
    fade_in("c"), fade_in("r", at="with"),
    draw("a", dur=0.8),
    fade_in("cap"),
    move_to("c", (-2.6, 1.6), dur=1.0),
    wait(0.5),
]
"""

EXAMPLES["typing_and_math"] = r"""
from Sufvid import *
import numpy as np

scene = {"size": (1280, 720), "bg": "#101820", "view": {"height": 9.0}, "shapes": [
    text("The quadratic formula", pos=(-7, 3.5), size=44, font="bold", anchor="lm", id="title"),
    mathtext(r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}", pos=(0, 0.8), size=72, color="#ffd60a", id="f"),
    line((-5, -0.8), (5, -0.8), color="#8b98a5", id="rule"),
    text("The discriminant b^2 - 4ac decides how many real roots exist.",
         pos=(-6.3, -1.8), size=28, color="#c9d1d9", anchor="lm", id="note"),
]}
timeline = [type_text("title", cps=24), fade_in("f", dur=0.8), draw("rule", dur=0.6), type_text("note", cps=40)]
"""

EXAMPLES["graph_and_ball"] = r"""
from Sufvid import *
import numpy as np

damped = lambda x: np.exp(-np.abs(x) / 6) * np.sin(x)
ax = axes((-7, 7), (-1.5, 1.5), pos=(0, -0.5), size=(13, 5), x_step=1, y_step=0.5, id="ax")
curve = graph(ax, damped, color="#ffd60a", w=5, id="curve")
path = np.column_stack(axes_point(ax, curve["xs"], curve["ys"]))      # the ball rides the SAME samples
scene = {"size": (1280, 720), "bg": "#101820", "view": {"height": 9.0}, "shapes": [
    text("Damped oscillation", pos=(-7, 3.8), size=42, font="bold", anchor="lm", id="title"),
    mathtext(r"y=e^{-|x|/6}\,\sin(x)", pos=(5, 3.7), size=44, color="#ffd60a", id="formula"),
    ax, curve,
    circle(tuple(path[0]), 0.16, fill="#ffd60a", stroke="white", stroke_w=2, id="ball"),
]}
timeline = [
    fade_in("title"), fade_in("ax", at="with"), fade_in("formula", at="with"),
    fade_in("ball", dur=0.3),
    draw("curve", dur=3.5, easing="linear"),
    follow_path("ball", path, at="with", dur=3.5, easing="linear"),
]
"""

EXAMPLES["projectile_real_data"] = r"""
from Sufvid import *
import numpy as np

vac = simulate("projectile", v0=20, angle=50, n=200)               # real ODE solutions, not formulas
drg = simulate("projectile", v0=20, angle=50, drag=0.15, n=200)
ax = axes((0, 45), (0, 14), pos=(0, -0.4), size=(13, 5.4), x_step=5, y_step=2, id="ax")
path = lambda r: np.column_stack(axes_point(ax, r["pos"][:, 0], r["pos"][:, 1]))
pv, pd_ = path(vac), path(drg)
tv, td = float(vac["t"][-1]), float(drg["t"][-1])                  # flight times in seconds
scene = {"size": (1280, 720), "bg": "#101820", "view": {"height": 9.0}, "shapes": [
    text("Projectile: vacuum vs air drag", pos=(-7, 3.9), size=40, font="bold", anchor="lm", id="title"),
    ax,
    graph(ax, xs=vac["pos"][:, 0], ys=vac["pos"][:, 1], color="#ff9f40", w=4, id="cv"),
    graph(ax, xs=drg["pos"][:, 0], ys=drg["pos"][:, 1], color="#6290ff", w=4, id="cd"),
    circle(tuple(pv[0]), 0.14, fill="#ff9f40", stroke="white", stroke_w=2, id="bv"),
    circle(tuple(pd_[0]), 0.14, fill="#6290ff", stroke="white", stroke_w=2, id="bd"),
    text(f"vacuum: range {vac['pos'][-1, 0]:.1f} m, flight {tv:.2f} s", pos=(0, -3.75), size=26,
         color="#ff9f40", id="r1"),
    text(f"drag k=0.15: range {drg['pos'][-1, 0]:.1f} m, flight {td:.2f} s", pos=(0, -4.15), size=26,
         color="#6290ff", id="r2"),
]}
timeline = [
    fade_in("title"), fade_in("ax", at="with"),
    fade_in("bv", at=1.0, dur=0.2), fade_in("bd", at=1.0, dur=0.2),
    draw("cv", at=1.2, dur=tv, easing="linear"), follow_path("bv", pv, at=1.2, dur=tv, easing="linear"),
    draw("cd", at=1.2, dur=td, easing="linear"), follow_path("bd", pd_, at=1.2, dur=td, easing="linear"),
    fade_in("r1", at=1.2 + tv, dur=0.4), fade_in("r2", at=1.2 + tv, dur=0.4),
]
"""

EXAMPLES["pendulum_scene_function"] = r"""
from Sufvid import *
import numpy as np

TH0, T = 2.2, 6.0
sim = simulate("pendulum", L=1.0, theta0=TH0, T=T, n=600)           # nonlinear pendulum via solve_ivp
PIV, S = (-4.2, 2.6), 2.8
ax = axes((0, T), (-2.5, 2.5), pos=(3.2, -0.2), size=(8.0, 4.4), x_step=1, y_step=1)
curve = graph(ax, xs=sim["t"], ys=sim["theta"], color="#ff9f40", w=4)
duration = T

def scene_at(t):
    th = float(np.interp(t, sim["t"], sim["theta"]))
    bob = (PIV[0] + S * np.sin(th), PIV[1] - S * np.cos(th))
    e = float(np.interp(t, sim["t"], sim["energy"])) / sim["energy"][0]
    p = min(t / T, 1.0)
    return {"size": (1280, 720), "bg": "#101820", "view": {"height": 9.0}, "shapes": [
        text("A real nonlinear pendulum", pos=(-7.5, 3.95), size=38, font="bold", anchor="lm"),
        line(PIV, bob, color="#ff9f40", w=4),
        circle(PIV, 0.08, fill="white", stroke=None),
        circle(bob, 0.26, fill="#ff9f40", stroke="white", stroke_w=2),
        ax, {**curve, "progress": p},
        circle(axes_point(ax, min(t, T), th), 0.11, fill="#ff9f40", stroke="white", stroke_w=2),
        text(f"energy / E0 = {e:.6f}", pos=(3.2, 2.7), size=28, color="#8b98a5", font="mono"),
        text(f"theta = {th:+.2f} rad", pos=(-4.2, -3.2), size=28, color="#ff9f40", font="mono"),
    ]}
"""

EXAMPLES["nn_decision_boundary"] = r"""
from Sufvid import *
import numpy as np

res = train_mlp("moons", n=240, hidden=(8,), epochs=300, snapshots=30, res=40, seed=0)   # real training
EPOCHS, S = len(res["loss"]) - 1, len(res["snap_epochs"])
bx0, bx1, by0, by1 = res["bounds"]
k = 6.0 / max(bx1 - bx0, by1 - by0)
axn = axes((bx0, bx1), (by0, by1), pos=(-3.6, -0.2), size=((bx1 - bx0) * k, (by1 - by0) * k),
           x_step=1, y_step=1, label_size=18)
(wx0, wy0), (wx1, wy1) = axes_point(axn, bx0, by0), axes_point(axn, bx1, by1)
heat = heatmap(res["grids"], rect=(wx0, wy0, wx1, wy1), alpha=0.5)
pts = points(np.column_stack(axes_point(axn, res["X"][:, 0], res["X"][:, 1])), labels=res["y"], r=0.07)
duration = 5.0

def scene_at(t):
    s = min(t / 4.0, 1.0) * (S - 1)                                   # position among the snapshots
    ep = float(np.interp(s, np.arange(S), res["snap_epochs"]))
    i = int(ep)
    return {"size": (1280, 720), "bg": "#101820", "view": {"height": 9.0}, "shapes": [
        {**heat, "index": s}, axn, pts,
        text("A tiny neural net learns a boundary", pos=(-7.5, 3.95), size=36, font="bold", anchor="lm"),
        text(f"epoch    {i:4d}", pos=(4.6, 0.8), size=30, font="mono"),
        text(f"loss     {res['loss'][i]:.3f}", pos=(4.6, 0.0), size=30, font="mono", color="#ffd60a"),
        text(f"accuracy {res['acc'][i] * 100:5.1f}%", pos=(4.6, -0.8), size=30, font="mono", color="#50fa7b"),
    ]}
"""

EXAMPLES["dipole_field_lines"] = r"""
from Sufvid import *
import numpy as np

charges = [(-2.0, 0.0, 1.0), (2.0, 0.0, -1.0)]
lines = field_lines(charges, bounds=(-7, 7, -3.3, 3.3), n_per_charge=18)
shapes = [text("Electric field of a dipole", pos=(0, 3.9), size=42, font="bold", id="title")]
for l in lines:
    shapes.append(polygon(l, stroke="#9db4ff", stroke_w=2, closed=False))
    if len(l) > 30:                                                    # tiny arrowhead = field direction
        m = len(l) // 2
        shapes.append(arrow(tuple(l[m]), tuple(l[m + 4]), color="#9db4ff", w=2, head=0.2))
for x, y, q in charges:
    shapes.append(circle((x, y), 0.3, fill="#ff5555" if q > 0 else "#6290ff", stroke="white", stroke_w=2))
    shapes.append(text("+" if q > 0 else "-", pos=(x, y), size=32, font="bold"))
scene = {"size": (1280, 720), "bg": "#101820", "view": {"height": 9.0}, "shapes": shapes}
timeline = [fade_in("title", dur=0.8), wait(2.0)]
"""

EXAMPLES["narrated_with_sound"] = r"""
from Sufvid import *
import numpy as np

TITLE = "Sound is just math"
CPS, T0 = 20.0, 0.3
T_LINE, T_MOVE, D_MOVE = 2.0, 3.2, 2.0
voice1 = voice("watch the dot", pitch=190)
D1 = len(voice1) / 44100.0                                             # clip length in seconds

scene = {"size": (1280, 720), "bg": "#101820", "view": {"height": 9.0}, "shapes": [
    text(TITLE, pos=(-7, 3.8), size=46, font="bold", anchor="lm", id="title"),
    circle((-5, -0.5), 0.4, fill="#ffd60a", stroke="white", id="dot"),
    text("watch the dot", pos=(0, -3.2), size=34, color="#50fa7b", id="cap"),
]}
timeline = [
    type_text("title", at=T0, cps=CPS),
    fade_in("dot", at=T_LINE, dur=0.4),
    fade_in("cap", at=T_LINE, dur=0.3), fade_out("cap", at=T_LINE + D1, dur=0.3),
    move_to("dot", (5, -0.5), at=T_MOVE, dur=D_MOVE, easing="ease_in_out"),
]
tracks = [(T_LINE, voice1), (T_LINE, sfx("pop", vol=0.6)), (T_MOVE, sfx("whoosh", vol=0.5)),
          (T_MOVE + D_MOVE, sfx("success", vol=0.6))]
for i, ch in enumerate(TITLE):                                         # one key tick per typed character
    if ch != " ":
        tracks.append((T0 + (i + 1) / CPS, sfx("key", vol=0.4, seed=i)))
audio = mix(tracks, duration=T_MOVE + D_MOVE + 0.8)
"""
