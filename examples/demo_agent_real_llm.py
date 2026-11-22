# Plug a real LLM into agent_loop. Needs:  pip install anthropic   and   os.environ["ANTHROPIC_API_KEY"] = "..."
# (not run here: no API key / network in my sandbox - the loop itself is fully tested with scripted replies)
import anthropic
from Sufvid import agent_loop

client = anthropic.Anthropic()

def generate(prompt):                       # ANY function  prompt:str -> reply:str  works
    msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=6000,
                                 messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text

result = agent_loop("A 30-second tutorial: a spring-mass oscillator, showing position vs time from a real "
                    "simulation, with a typed title, a ding when it finishes, and a short voice line.",
                    generate, max_tries=5, path="llm_video.mp4")
print(result["ok"], result["attempts"])
