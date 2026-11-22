# Sufvid

Narrated math and physics tutorial videos from **plain functions**: no classes, no Manim, no AI video generators.
Everything is drawn with Pillow, computed with NumPy/SciPy, encoded through an ffmpeg pipe, and the sound
(music, effects, a stylised blip voice) is synthesised from math waves.

```python
import Sufvid
from Sufvid import *
import numpy as np

ax = axes((-7, 7), (-1.5, 1.5), size=(13, 5), id="ax")
curve = graph(ax, lambda x: np.exp(-abs(x) / 6) * np.sin(x), id="curve")
scene = {"size": (1280, 720), "bg": "#101820", "shapes": [
    text("Hello, Sufvid", pos=(0, 3.8), size=46, font="bold", id="title"), ax, curve]}
timeline = [fade_in("title"), fade_in("ax"), draw("curve", dur=2.0)]
render_video(scene, timeline=timeline, path="hello.mp4")
```

## Install

Needs Python 3.10+ and `ffmpeg` on your PATH (Colab/Ubuntu: `apt-get install -y ffmpeg`).

```bash
pip install git+https://github.com/YOUR_NAME/YOUR_REPO.git     # straight from GitHub
# or, to hack on it:
git clone https://github.com/YOUR_NAME/YOUR_REPO.git && cd YOUR_REPO && pip install -e ".[test]"
```

## Tests

```bash
pytest -q          # runs against the installed package
```

## Using it with an LLM

`Sufvid.agent_loop(task, generate)` asks any LLM for a program, lints it (`check_scene`), renders it, and feeds
every error back until it works. The prompt text the model sees is `Sufvid.agent.docs.build_docs()`.
See `examples/demo_agent_real_llm.py`.

## Optional neural voice

`voice(text, engine="piper")` uses the external Piper program (not NumPy-generated): set `PIPER_BIN` to the
`piper` executable and `PIPER_MODEL` to a `.onnx` voice. Voices have their own licences: check before publishing.

## Rules this library keeps

No OOP; a scene is data (dicts/lists); `render_frame(scene, t)` is a pure function; animations are functions of
time; the public API (`Sufvid.__all__`) stays under 40 functions.
