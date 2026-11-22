import sys, os, subprocess, shutil, json, copy
for m in [k for k in sys.modules if k.startswith("Sufvid")]:
    del sys.modules[m]
import numpy as np
import Sufvid
from Sufvid import *
from Sufvid.agent.check import _bbox
from Sufvid.agent.docs import build_docs, CONVENTIONS
from Sufvid.agent.examples import EXAMPLES
from Sufvid.agent.loop import run_program, _contract, _preview, _extract_code, _Fail

print("public API size:", len(Sufvid.__all__), "/ 40")
assert len(Sufvid.__all__) < 40

def raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")

SIZE = (1280, 720)
def sc(*shapes, size=SIZE):
    return {"size": size, "bg": "#000", "view": {"height": 9.0}, "shapes": list(shapes)}
errs = lambda m: [x for x in m if x.startswith("error")]
warns = lambda m: [x for x in m if x.startswith("warning")]

# ================= bounding boxes agree with what is actually drawn =================
def ink_box(shape):
    f = render_frame(sc(shape), 0.0)
    ys, xs = np.where(f.max(axis=2) > 40)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

def to_px(b):                                            # world bbox -> pixel box at 1280x720 (80 px per unit)
    return (640 + b[0] * 80, 360 - b[3] * 80, 640 + b[2] * 80, 360 - b[1] * 80)

cases = [text("Hello, World", pos=(0, 0), size=48, anchor="mm"),
         text("Typography gjpqy", pos=(-3, 1), size=40, anchor="lm", font="bold"),
         text("right side", pos=(5, -2), size=30, anchor="rb", font="mono"),
         text("top", pos=(1, 2), size=36, anchor="mt"),
         mathtext(r"\frac{a}{b}", pos=(0, 0), size=70, anchor="mm"),
         mathtext(r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}", pos=(2, 1), size=60, anchor="lm"),
         mathtext(r"e^{i\pi}+1=0", pos=(-2, -1), size=50, anchor="rt")]
for c in cases:
    ink, box = ink_box(c), to_px(_bbox(c, 80.0))
    # the box must CONTAIN the ink (no missed overlaps) ...
    assert box[0] <= ink[0] + 2 and box[1] <= ink[1] + 2 and box[2] >= ink[2] - 2 and box[3] >= ink[3] - 2, (c.get("text", c.get("tex")), ink, box)
    # ... and not be absurdly larger than it (no false alarms)
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    assert area(box) < 1.7 * area(ink), (c.get("text", c.get("tex")), area(box) / area(ink))
b = _bbox(circle((1, 2), 0.5), 80.0); assert b == (0.5, 1.5, 1.5, 2.5)
b = _bbox(rect((0, 0), (4, 2)), 80.0); assert b == (-2, -1, 2, 1)
b = _bbox(line((0, 0), (4, 2), progress=0.5), 80.0); assert b == (0, 0, 2, 1)
assert _bbox(text("", pos=(0, 0)), 80.0) is None

# ================= check_scene =================
clean = sc(text("Title", pos=(0, 3.8), size=44, id="t"), circle((0, 0), 1, id="c"),
           text("caption", pos=(0, -3.5), size=30))
before = repr(clean)
assert check_scene(clean, [fade_in("t"), move_to("c", (3, 0))]) == []
assert repr(clean) == before, "check_scene mutated the scene"

m = check_scene(sc(circle(id="a")), [fade_in("zzz")])                         # unknown id, lists the valid ones
assert len(m) == 1 and m[0].startswith("error") and "zzz" in m[0] and "'a'" in m[0], m
m = check_scene(sc(circle(id="a")), [draw("a")])
assert errs(m) and "draw() works on" in m[0], m
m = check_scene(sc(circle(id="a")), [type_text("a")])
assert errs(m) and "type_text" in m[0], m
m = check_scene(sc(circle(id="a"), circle(id="a")))
assert errs(m) and "duplicate" in m[0], m
m = check_scene(sc({"type": "blob"}))
assert errs(m) and "unknown type" in m[0] and "circle" in m[0], m
assert errs(check_scene({"shapes": []}))

# overlaps: static, moving, typed, math-vs-text
m = check_scene(sc(text("Hello", pos=(0, 0), size=40, id="a"), text("World", pos=(0.1, 0.05), size=40, id="b")))
assert errs(m) and "overlaps" in m[0] and "'a'" in m[0] and "'b'" in m[0], m
assert check_scene(sc(text("Hello", pos=(0, 0), size=40), text("World", pos=(0, 2), size=40))) == []
m = check_scene(sc(text("Hello", pos=(0, 0), size=40, id="a"), text("World", pos=(0, 2), size=40, id="b")),
                [wait(1), move_to("b", (0, 0), dur=1)])
assert errs(m) and "overlaps" in m[0], m
t_hit = float(m[0].split("at t=")[1].split("s")[0])
assert t_hit > 1.0, t_hit                                                        # only once it has moved into place
assert len(m) == 1, m                                                            # reported once, not once per sample time
m = check_scene(sc(text("This is a long typed line of text", pos=(-6, 0), size=40, anchor="lm", id="a"),
                   text("XXXXX", pos=(1.0, 0), size=40, id="b")), [type_text("a", cps=10)])
assert errs(m) and "overlaps" in m[0], m                                          # grows into B while typing
assert float(m[0].split("at t=")[1].split("s")[0]) > 2.0, m                        # ... only once enough has been typed
m = check_scene(sc(mathtext(r"x^2", pos=(0, 0), size=60, id="m"), text("abc", pos=(0, 0), size=40, id="t")))
assert errs(m) and "overlaps" in m[0], m
m = check_scene(sc(text("Hello", pos=(0, 0), size=40, id="a", opacity=0.0), text("World", pos=(0, 0), size=40)))
assert m == [], m                                                                  # an invisible shape cannot overlap

# frame edge
m = check_scene(sc(text("Hello", pos=(7.9, 0), size=40)))
assert errs(m) and "cut off" in m[0], m
m = check_scene(sc(text("Hello", pos=(20, 0), size=40)))
assert errs(m) and "completely outside" in m[0], m
m = check_scene(sc(circle((30, 0), 1, id="c")))
assert errs(m) and "completely outside" in m[0], m
assert check_scene(sc(circle((30, 0), 1, opacity=0.0))) == []
m = check_scene(sc(circle((30, 0), 1, id="c")), [fade_in("c")])
assert errs(m), m                                                                  # becomes visible later
assert check_scene(sc(rect((8, 0), (6, 6)))) == []                                 # big shapes may bleed off an edge
assert check_scene(sc(text("Hello", pos=(0, 0), size=40, id="a")), [move_to("a", (20, 0), dur=1)]) != []
# warnings do not block
m = check_scene(sc(text("tiny", pos=(0, 0), size=10)))
assert m and not errs(m) and "font size" in m[0], m
m = check_scene(sc(circle(id="c")), [fade_in("c", at=5)], duration=3)
assert warns(m) and "after the video ends" in m[0] and not errs(m), m
# scene functions
f = lambda t: sc(text("Hello", pos=(0, 0), size=40), text("World", pos=(0.1 * t, 0), size=40))
assert errs(check_scene(f))                                                         # needs duration
m = check_scene(f, duration=10.0)
assert errs(m) and "overlaps" in m[0], m
assert check_scene(lambda t: sc(text("A", pos=(0, 0)), text("B", pos=(0, 3 + t))), duration=0.5) == []
m = check_scene(lambda t: 1 / 0, duration=1.0)
assert errs(m) and "ZeroDivisionError" in m[0], m

# ================= docs =================
doc = build_docs()
for name in Sufvid.__all__:
    assert f"- {name}(" in doc, name                                                  # every public function documented
for name in EXAMPLES:
    assert f"### example: {name}" in doc, name
assert "from Sufvid import *" in doc and "scene_at" in doc and "check_scene" in doc
assert "engine" in doc.split("- voice(")[1].split("\n")[0]                           # signature comes from the real code
assert len(build_docs(False)) < len(doc) < 40000
assert "```python" in CONVENTIONS

# ================= program execution helpers =================
assert _extract_code("sure!\n```python\nx = 1\n```\nbye") == "x = 1\n"
assert _extract_code("```\nx = 2\n```") == "x = 2\n" and _extract_code("y = 3") == "y = 3"
assert _extract_code("```python\na=1\n```\n```python\nlonger = 12345\n```") == "longer = 12345\n"

def stage_of(code):
    try:
        ns = run_program(code)
        _contract(ns)
    except _Fail as f:
        return f.stage, f.message
    return "ok", ""

st, msg = stage_of("from Sufvid import *\nx = 1\ny = undefined_name + 1\n")
assert st == "exec" and "NameError" in msg and "program line 3" in msg and "undefined_name" in msg, msg
st, msg = stage_of("from Sufvid import *\nscene = {'size': (10, 10\n")
assert st == "exec" and "SyntaxError" in msg and "line" in msg, msg
st, msg = stage_of("from Sufvid import *\nx = simulate('rocket')\n")
assert st == "exec" and "KeyError" in msg and "projectile" in msg, msg           # library errors reach the model verbatim
assert stage_of("x = 1")[0] == "contract"
assert stage_of("scene = 5")[0] == "contract"
assert stage_of("scene = {'size': (101, 100), 'shapes': []}\nduration = 1")[0] == "contract"
assert stage_of("scene = {'size': (100, 100), 'shapes': []}")[0] == "contract"      # no length given
assert stage_of("scene_at = lambda t: {'size': (100, 100), 'shapes': []}")[0] == "contract"   # no duration
assert stage_of("scene = {'size': (100, 100), 'shapes': []}\nduration = 2\naudio = 'nope.wav'")[0] == "contract"
assert stage_of("scene = {'size': (100, 100), 'shapes': []}\nduration = 2")[0] == "ok"

# ================= the examples: every one lints clean, previews, and has a visible last frame =================
for name, code in EXAMPLES.items():
    ns = run_program(code)
    scene, tl, dur, audio = _contract(ns)
    assert check_scene(scene, tl, dur) == [], (name, check_scene(scene, tl, dur))
    _preview(scene, tl, dur)
print(f"  {len(EXAMPLES)} example programs: lint clean + preview OK")

# ================= the agent loop, driven by scripted 'LLMs' that fail in specific ways =================
def scripted(replies):
    prompts, it = [], iter(replies)
    def gen(p):
        prompts.append(p)
        return next(it)
    return gen, prompts

BAD_ID = "from Sufvid import *\nscene = {'size': (1280, 720), 'shapes': [circle((0, 0), 1, id='ball')]}\ntimeline = [fade_in('bal')]\n"
OVERLAP = ("from Sufvid import *\nscene = {'size': (1280, 720), 'shapes': [text('Hello there', pos=(0, 0), size=44, id='a'),"
           " text('General Kenobi', pos=(0.3, 0.1), size=44, id='b')]}\ntimeline = [fade_in('a'), fade_in('b')]\n")
GOOD = ("from Sufvid import *\nscene = {'size': (1280, 720), 'bg': '#101820', 'shapes': [text('Hello', pos=(0, 2), size=44, id='a'),"
        " circle((-3, -1), 0.5, fill='#ffd60a', id='c')]}\ntimeline = [fade_in('a'), move_to('c', (3, -1), dur=1.0)]\n")

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames",
                          "-of", "json", path], capture_output=True, check=True).stdout
    return json.loads(out)["streams"]

gen, prompts = scripted(["Sure! Here you go:\n```python\n" + BAD_ID + "```\nHope it helps.",
                         "```python\n" + OVERLAP + "```", "```python\n" + GOOD + "```"])
r = agent_loop("Say hello and move a dot.", gen, max_tries=4, path="/tmp/_a1.mp4", size=(320, 180), fps=5, verbose=False)
assert r["ok"] and r["attempts"] == 3 and len(r["history"]) == 2 and os.path.getsize("/tmp/_a1.mp4") > 0
assert r["history"][0]["stage"] == "check" and r["history"][1]["stage"] == "check"
assert "Say hello and move a dot." in prompts[0] and "from Sufvid import *" in prompts[0]    # docs + task in the first prompt
assert "bal" in prompts[1] and "ids in scene" in prompts[1] and "fade_in('bal')" in prompts[1]   # the error AND the old code come back
assert "overlaps" in prompts[2] and "General Kenobi" in prompts[2] and "Hello there" in prompts[2]
assert "bal" not in prompts[2].split("YOUR PREVIOUS PROGRAM")[1].split("IT FAILED")[0]            # it shows the latest program only
v = [s for s in probe("/tmp/_a1.mp4") if s["codec_type"] == "video"][0]
assert (v["width"], v["height"]) == (320, 180) and int(v["nb_frames"]) == 10, v                   # size override; (1.5 s timeline + 0.5 s tail) * 5 fps
assert r["code"].strip() == GOOD.strip() and r["render"]["frames"] == 10

# exec error -> fed back with line number, then fixed
gen, prompts = scripted(["from Sufvid import *\nx = 1\nscene = oops\n", GOOD])
r = agent_loop("t", gen, path="/tmp/_a2.mp4", size=(320, 180), fps=5, verbose=False)
assert r["ok"] and r["attempts"] == 2 and r["history"][0]["stage"] == "exec"
assert "NameError" in prompts[1] and "program line 3: scene = oops" in prompts[1]

# preview catches a blank video that lint cannot see
BLANK = "from Sufvid import *\nscene = {'size': (1280, 720), 'shapes': []}\nduration = 1.0\n"
FADED = ("from Sufvid import *\nscene = {'size': (1280, 720), 'shapes': [circle((0, 0), 1, id='c')]}\n"
         "timeline = [fade_in('c', dur=0.3), fade_out('c', dur=0.3)]\n")
gen, prompts = scripted([BLANK, FADED, GOOD])
r = agent_loop("t", gen, path="/tmp/_a3.mp4", size=(320, 180), fps=5, verbose=False)
assert r["ok"] and r["attempts"] == 3
assert [h["stage"] for h in r["history"]] == ["preview", "preview"]
assert "nothing is visible" in r["history"][0]["message"] and "last frame" in r["history"][1]["message"]

# warnings do not block; contract failure is reported
TINY = "from Sufvid import *\nscene = {'size': (1280, 720), 'shapes': [text('small', pos=(0, 0), size=10)]}\nduration = 0.6\n"
gen, _ = scripted([TINY])
r = agent_loop("t", gen, path="/tmp/_a4.mp4", size=(320, 180), fps=5, verbose=False)
assert r["ok"] and r["attempts"] == 1 and r["history"] == []
gen, prompts = scripted(["x = 1", GOOD])
r = agent_loop("t", gen, path="/tmp/_a5.mp4", size=(320, 180), fps=5, verbose=False)
assert r["ok"] and r["history"][0]["stage"] == "contract" and "define `scene`" in prompts[1]

# gives up cleanly
gen, _ = scripted([BAD_ID] * 3)
r = agent_loop("t", gen, max_tries=3, path="/tmp/_a6.mp4", verbose=False)
assert not r["ok"] and r["video"] is None and r["attempts"] == 3 and len(r["history"]) == 3 and r["code"] == BAD_ID
raises(TypeError, lambda: agent_loop("t", lambda p: 42, verbose=False))

# scene functions + audio arrays go through the real renderer and the real mux
SCENEFN = ("from Sufvid import *\nduration = 1.0\ndef scene_at(t):\n    return {'size': (1280, 720), 'shapes': "
           "[circle((-3 + 6 * t, 0), 0.5, fill='#ff5555', stroke=None)]}\naudio = mix([(0.0, synth('sine', 440, 0.8))])\n")
gen, _ = scripted([SCENEFN])
r = agent_loop("t", gen, path="/tmp/_a7.mp4", size=(320, 180), fps=5, verbose=False)
assert r["ok"], r["history"]
kinds = sorted(s["codec_type"] for s in probe("/tmp/_a7.mp4"))
assert kinds == ["audio", "video"], kinds

# three full examples end to end (scene+timeline, scene function, and sound)
for name in ("shapes_and_timeline", "nn_decision_boundary", "narrated_with_sound"):
    gen, _ = scripted(["```python\n" + EXAMPLES[name] + "\n```"])
    r = agent_loop(name, gen, path=f"/tmp/_ex_{name}.mp4", size=(320, 180), fps=5, verbose=False)
    assert r["ok"] and r["attempts"] == 1, (name, r["history"])
    kinds = sorted(s["codec_type"] for s in probe(f"/tmp/_ex_{name}.mp4"))
    assert kinds == (["audio", "video"] if name == "narrated_with_sound" else ["video"]), (name, kinds)
print("  agent loop: 12 scripted scenarios OK (fix-from-error, blank video, give-up, audio, scene functions)")

print("all Phase 6 tests passed")
