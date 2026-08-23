"""A terminal chat agent that remembers what it learns between sessions.

Run it with:  python3 agent.py

It keeps a plain-JSON memory file (.agent_memory.json) next to this script.
The agent decides on its own when something is worth keeping and calls the
`remember` tool; those lessons are fed back into the system prompt on every
future run, so the agent gets better at your specific work over time.
"""

import json
import os
import sys
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent
MEMORY_PATH = ROOT / ".agent_memory.json"
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")
MAX_LESSONS = 100

TOOLS = [
    {
        "name": "remember",
        "description": (
            "Save a durable lesson so future conversations start smarter. Use this "
            "when you learn a stable preference, a correction to something you got "
            "wrong, a fact about this NFL analytics project, or a rule of thumb "
            "worth reusing. Do not save one-off chatter or anything already stored."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lesson": {
                    "type": "string",
                    "description": "One self-contained sentence, phrased so it makes sense with no other context.",
                }
            },
            "required": ["lesson"],
        },
    }
]


def load_env():
    """Read KEY=VALUE pairs out of a local .env file without extra dependencies."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_lessons():
    if not MEMORY_PATH.exists():
        return []
    return json.loads(MEMORY_PATH.read_text()).get("lessons", [])


def save_lessons(lessons):
    MEMORY_PATH.write_text(json.dumps({"lessons": lessons[-MAX_LESSONS:]}, indent=2))


def build_system_prompt(lessons):
    prompt = (
        "You are a research assistant for Brandon's NFL analytics project, a Python "
        "repo with a Streamlit app, team-rating models, and a draft simulator.\n"
        "Be direct and concrete. Say when you are unsure rather than guessing at "
        "numbers. When you learn something durable, call the `remember` tool."
    )
    if lessons:
        numbered = "\n".join("%d. %s" % (i + 1, l) for i, l in enumerate(lessons))
        prompt += "\n\nWhat you have learned in past sessions:\n" + numbered
    return prompt


def run_turn(client, messages, lessons):
    """Stream one assistant turn, looping until it stops calling tools."""
    while True:
        with client.messages.stream(
            model=MODEL,
            max_tokens=2000,
            system=build_system_prompt(lessons),
            tools=TOOLS,
            messages=messages,
        ) as stream:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
            reply = stream.get_final_message()
        print()

        messages.append({"role": "assistant", "content": reply.content})
        tool_calls = [b for b in reply.content if b.type == "tool_use"]
        if not tool_calls:
            return

        results = []
        for call in tool_calls:
            lesson = call.input["lesson"].strip()
            if lesson and lesson not in lessons:
                lessons.append(lesson)
                save_lessons(lessons)
                print("  [learned] %s" % lesson)
            results.append(
                {"type": "tool_result", "tool_use_id": call.id, "content": "saved"}
            )
        messages.append({"role": "user", "content": results})


def handle_command(text, lessons):
    """Return True if the input was a slash command and was handled here."""
    if text == "/memory":
        if not lessons:
            print("  (nothing learned yet)")
        for i, lesson in enumerate(lessons, 1):
            print("  %d. %s" % (i, lesson))
        return True

    if text.startswith("/teach "):
        lesson = text[len("/teach "):].strip()
        if lesson:
            lessons.append(lesson)
            save_lessons(lessons)
            print("  [learned] %s" % lesson)
        return True

    if text.startswith("/forget "):
        try:
            index = int(text.split()[1]) - 1
            print("  [forgot] %s" % lessons.pop(index))
            save_lessons(lessons)
        except (ValueError, IndexError):
            print("  usage: /forget <number from /memory>")
        return True

    return False


def main():
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "No ANTHROPIC_API_KEY found.\n"
            "Create a .env file in this folder containing:\n"
            "  ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic()
    lessons = load_lessons()
    messages = []

    print("NFL analytics agent (%s) - %d lessons remembered." % (MODEL, len(lessons)))
    print("Commands: /memory  /teach <lesson>  /forget <n>  /quit\n")

    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not text:
            continue
        if text in ("/quit", "/exit"):
            return
        if handle_command(text, lessons):
            continue

        messages.append({"role": "user", "content": text})
        print("\nagent > ", end="", flush=True)
        run_turn(client, messages, lessons)
        print()


if __name__ == "__main__":
    main()
