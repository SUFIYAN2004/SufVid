"""LLM-facing documentation, generated from the real public API so it can never go stale."""
import inspect
from functools import lru_cache

CONVENTIONS = """\
You write ONE short Python program that makes a narrated math/physics tutorial video with the library
`Sufvid`. Start with `from Sufvid import *` and `import numpy as np` (math and scipy are also allowed). No classes,
no files except the optional WAV, no other imports, no network.

PROGRAM CONTRACT - the program must define:
  scene     : a dict {"size": (1280, 720), "bg": "#101820", "view": {"height": 9.0}, "shapes": [ ... ]}
              OR scene_at(t): a pure function returning such a dict for time t (use this when values
              depend on t, e.g. a heatmap index or a simulation read-out) - then also set `duration`.
  timeline  : optional list of animations (fade_in, draw, move_to, ...). Needs ids on the shapes it targets.
  duration  : optional seconds. If missing, it is the end of the timeline + 0.5 s (or the audio length).
  audio     : optional soundtrack: the array returned by mix([...]) (or a WAV path). It is muxed into the mp4.
The harness renders the video itself - never call render_video yourself.

COORDINATES AND UNITS
  World units, y points UP, origin at the screen centre. At 1280x720 with view height 9 the visible area
  is x in [-8, 8], y in [-4.5, 4.5]. Keep everything inside it.
  Stroke widths and text/mathtext `size` are in OUTPUT PIXELS: title 40-48, body 26-34, axis labels >= 20.
  Text is one line only. Use anchor="lm" (left, middle) for text that will be typed with type_text.

SHAPES, IDS AND ANIMATION
  Give an id= to every shape that an animation targets. A shape targeted by fade_in, draw or type_text is
  hidden until that animation starts. at=None starts after the previous animation, at="with" starts with
  the previous one, at=2.5 starts at 2.5 s. Do not overlap two animations of the same kind on one shape.
  draw() works only on line, arrow and graph. type_text() only on text. To make a curve and a dot move
  together, draw(curve, at=T, dur=D, easing="linear") and follow_path(dot, path, at=T, dur=D,
  easing="linear") where path = np.column_stack(axes_point(ax, curve["xs"], curve["ys"])).

DATA AND SOUND
  Never invent numbers: compute them (simulate, train_mlp, field_lines, numpy) and draw the result.
  Put sound on the same clock as the animation: reuse the same time constants for sfx/voice cues, then
  audio = mix([(start_seconds, samples), ...]). voice() defaults to a stylised blip voice.

LAYOUT RULES (check_scene enforces them before rendering)
  Text and mathtext must not overlap each other at ANY moment (also while moving or typing), must not be
  cut off by the frame edge, and every visible shape must be at least partly on screen.
  Reply with a single ```python code block and nothing else."""

@lru_cache(maxsize=1)
def _api_text():
    import Sufvid
    out = []
    for name in Sufvid.__all__:
        obj = getattr(Sufvid, name)
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = "(...)"
        doc = " ".join((inspect.getdoc(obj) or "").split())
        out.append(f"- {name}{sig}\n    {doc[:420]}")
    return "\n".join(out)

def build_docs(examples=True):
    """The full prompt prefix for an LLM: contract + API reference + working example programs."""
    parts = [CONVENTIONS, "\nAPI (all of it comes from `from Sufvid import *`):", _api_text()]
    if examples:
        from .examples import EXAMPLES
        parts.append("\nWORKING EXAMPLE PROGRAMS (each one passes check_scene and renders):")
        for name, code in EXAMPLES.items():
            parts.append(f"\n### example: {name}\n```python\n{code.strip()}\n```")
    return "\n".join(parts)
