import numpy as np
from IPython.display import Audio, Image as IPImage, Video, display
from Sufvid import *
from Sufvid.audio.osc import SR

# ---------- timing: every cue comes from these numbers ----------
TITLE = "Sound demo: every sound is a formula"
CPS, T_TYPE = 22.0, 0.3                                  # typing speed, start
T_WHOOSH = 2.0                                           # axes + formula appear
T_V1 = 2.7                                               # first voice line
T_DRAW, D_DRAW = 3.4, 3.5                                # curve is drawn
T_END = T_DRAW + D_DRAW                                  # 6.9 s: curve finished
LEN = 8.5                                                # soundtrack length (s)

# ---------- 1) music: a chiptune loop made only from synth() ----------
def music(seconds, bpm=96):
    beat = 60.0 / bpm
    chords = [("A2", "A3", "C4", "E4"), ("F2", "F3", "A3", "C4"),
              ("C3", "C4", "E4", "G4"), ("G2", "G3", "B3", "D4")]
    hat = np.diff(synth("noise", 440, 0.05, vol=0.5, adsr=(0.001, 0.01, 0.2, 0.02)), prepend=0.0)
    kick = synth("sine", 140, 0.14, vol=0.8, slide_to=45, adsr=(0.001, 0.04, 0.0, 0.06))
    tracks = []
    for b in range(int(seconds / beat) + 1):
        ch, tb = chords[(b // 4) % 4], b * beat
        if b % 2 == 0:
            tracks.append((tb, synth("square", ch[0], beat * 0.9, vol=0.30, adsr=(0.005, 0.05, 0.6, 0.05))))
        for k in range(2):
            tracks.append((tb + k * beat / 2, synth("triangle", ch[1 + (2 * b + k) % 3], beat * 0.48,
                                                    vol=0.35, adsr=(0.003, 0.03, 0.5, 0.03))))
        tracks.append((tb, kick))
        tracks.append((tb + beat / 2, hat, 0.5))
    m = mix(tracks, duration=seconds, master=0.8)
    return m * np.minimum(1.0, np.linspace(1.0, 0.0, len(m)) * seconds)      # fade out over the last second

bgm = music(LEN)

# ---------- 2) voice cues (blip voice; pass engine='espeak' for real words) ----------
v1 = voice("watch the curve", pitch=190)
v2 = voice("all math!", pitch=210, speed=1.1)
D_V1, T_V2 = len(v1) / SR, T_END + 0.2

# ---------- 3) soundtrack: sound effects placed from the SAME numbers that drive the animation ----------
tracks = [(0.0, bgm, 0.22), (T_V1, v1), (T_V2, v2), (T_WHOOSH, sfx("whoosh", vol=0.6)),
          (T_END, sfx("success", vol=0.7))]
for i, ch in enumerate(TITLE):                           # a key tick when each typed character appears
    if ch != " ":
        tracks.append((T_TYPE + (i + 1) / CPS, sfx("key", vol=0.5, seed=i), 0.7))
for k in range(-2, 3):                                   # a blip each time the curve crosses zero (x = k*pi)
    tracks.append((T_DRAW + D_DRAW * (k * np.pi + 7.0) / 14.0, sfx("blip", vol=0.5), 0.8))
audio = mix(tracks, path="phase5_audio.wav", duration=LEN)
print(f"soundtrack {len(audio) / SR:.2f}s, peak {np.abs(audio).max():.2f}, voice 1 = {D_V1:.2f}s")

print("music:");  display(Audio(bgm, rate=SR))              # listen to the parts on their own first
print("voice:");  display(Audio(v1, rate=SR))
print("whoosh:"); display(Audio(sfx("whoosh"), rate=SR))
print("mix:");    display(Audio(audio, rate=SR))

# ---------- 4) visuals (same objects as the Phase 3 demo) ----------
damped = lambda x: np.exp(-np.abs(x) / 6) * np.sin(x)
ax = axes((-7, 7), (-1.5, 1.5), pos=(0, -0.6), size=(13, 5), x_step=1, y_step=0.5, id="ax")
curve = graph(ax, damped, color="#ffd60a", w=5, id="curve")
path = np.column_stack(axes_point(ax, curve["xs"], curve["ys"]))
scene = {"size": (1280, 720), "bg": "#101820", "view": {"height": 9.0}, "shapes": [
    text(TITLE, pos=(-7.0, 3.8), size=40, font="bold", anchor="lm", id="title"),
    ax, curve,
    circle(tuple(path[0]), 0.16, fill="#ffd60a", stroke="white", stroke_w=2, id="ball"),
    mathtext(r"y=e^{-|x|/6}\,\sin(x)", pos=(0, 2.7), size=48, color="#ffd60a", id="formula"),
    text("watch the curve", pos=(0, -4.0), size=34, color="#50fa7b", id="cap1"),
    text("all math!", pos=(0, -4.0), size=34, color="#50fa7b", id="cap2"),
]}
timeline = [
    type_text("title", at=T_TYPE, cps=CPS),
    fade_in("ax", at=T_WHOOSH, dur=0.6), fade_in("formula", at=T_WHOOSH, dur=0.6),
    fade_in("cap1", at=T_V1, dur=0.15), fade_out("cap1", at=T_V1 + D_V1, dur=0.3),
    fade_in("ball", at=T_DRAW - 0.2, dur=0.2),
    draw("curve", at=T_DRAW, dur=D_DRAW, easing="linear"),
    follow_path("ball", path, at=T_DRAW, dur=D_DRAW, easing="linear"),
    fade_in("cap2", at=T_V2, dur=0.15),
]

s = render_video(scene, timeline=timeline, audio="phase5_audio.wav", fps=30,
                 path="phase5.mp4", workers=2)                  # duration = max(timeline + tail, audio)
print(f"{s['frames']} frames in {s['seconds']:.1f}s -> {s['fps']:.1f} fps")
display(Video("phase5.mp4", embed=True, width=800))
