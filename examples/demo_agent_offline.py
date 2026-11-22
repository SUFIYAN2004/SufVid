from Sufvid import *
from Sufvid.agent.examples import EXAMPLES

# An offline stand-in for an LLM: two typical mistakes, then a correct program.
typo = EXAMPLES["projectile_real_data"].replace('fade_in("bv", at=1.0, dur=0.2)', 'fade_in("ball_v", at=1.0, dur=0.2)')   # unknown id
clash = EXAMPLES["projectile_real_data"].replace('pos=(0, -4.15)', 'pos=(0, -3.8)')                                  # labels collide
replies = iter([f"```python\n{typo}\n```", f"```python\n{clash}\n```", f"```python\n{EXAMPLES['projectile_real_data']}\n```"])
fake_llm = lambda prompt: next(replies)

result = agent_loop("Compare a projectile in vacuum and with air drag, using real simulation data.",
                    fake_llm, max_tries=4, path="phase6_demo.mp4")
print("ok:", result["ok"], "| attempts:", result["attempts"], "| stages that failed:", [h["stage"] for h in result["history"]])
for h in result["history"]:
    print("--- attempt", h["attempt"], "feedback the model would receive:\n" + h["message"])
if result["ok"]:
    print(f"{result['render']['frames']} frames at {result['render']['fps']:.1f} fps")
