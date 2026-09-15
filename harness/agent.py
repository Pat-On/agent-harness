import os
import json

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from harness.tools import registry
from harness.memory import load_agents_md
from harness.config import MODEL, STEP_BUDGET

from harness.sandbox import Sandbox
from harness.tools.bash import set_sandbox

# load variables from .env (e.g. OPENAI_API_KEY) into the process environment
load_dotenv()

# pick the provider at startup: if a Kimi key is configured, use Kimi's
# OpenAI-compatible endpoint instead of talking to OpenAI directly
if os.getenv("KIMI_API_KEY"):
    # MODEL = "kimi-k2.6"
    # Kimi exposes an OpenAI-compatible API, so the same OpenAI SDK works -
    # just point it at Kimi's base_url with a Kimi key instead
    client = OpenAI(
        api_key=os.getenv("KIMI_API_KEY"), base_url=os.getenv("KIMI_BASE_URL")
    )
    EXTRA_BODY = {
        "thinking": {"type": "disabled"}
    }  # disabled to not consume to much tokens

else:
    # MODEL = "gpt-4o-mini"  # cheap and small
    # # reads OPENAI_API_KEY from the environment automatically
    client = OpenAI()
    EXTRA_BODY = {}  # no provider-specific extras needed for plain OpenAI


# Maximum number of tool-call rounds per user turn. When hit, the harness
# forces the model to summarize what it did and hand control back to the
# user, instead of letting the loop run indefinitely.
# STEP_BUDGET = 25

# The synthetic system message injected when the step budget is exceeded.
# It tells the model why it's being asked to stop and what shape its
# response should take.
BUDGET_HIT_MESSAGE = """\
You've reached the step budget for this turn (25 tool calls). Do not make
any more tool calls. Instead, respond directly to the user with:

1. What you accomplished in this turn.
2. What remains to be done.
3. What the user should ask next to continue the work.

Your response will be the final message for this turn. The user will
reply to it and you can continue from there.
"""

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text()

def run():
    """Run the loop till the user quiets"""

    # Step 1: start the sandbox and wire it into the bash tool.            
    # This creates the Docker container, bind-mounts the workspace, and
    # cleans up any orphan containers from prior crashed sessions.
    sandbox = Sandbox()
    sandbox.start()
    set_sandbox(sandbox)


    # conversation history - entire memory of the agent
    # every turn we append to it and send the whole thing to the model
    # (there's no server-side session state - the full transcript IS the state)

    print("Agent ready. Type 'quite' or 'exit' to leave. \n")


    try: 
        # Load AGENTS.md and assemble the initial message list.
        # The first system message is the harness's prompt; the second is the
        # project's accumulated memory.
        agents_md = load_agents_md()

        # messages = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": agents_md},
        ]

        while True:

            try:
                # block here until the user types something and hits enter
                user_input = input("you > ").strip()
            except KeyboardInterrupt:
                # Ctrl+C while waiting for input - exit quietly instead of a traceback
                print("\nGoodbye")
                break

            if user_input in {"quit", "exit"}:
                print("\nGoodbye")
                break

            # ignore empty input (blank line) instead of sending it to the model
            if not user_input:
                continue

            # 3. Append the user's message to the history
            messages.append({"role": "user", "content": user_input})

            ###### ReAct dispatch loop ######
            # Full ReAct dispatch loop — replaces the single-round dispatch
            step_count = 0
            while True:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=registry.get_schemas(),
                )
                message = response.choices[0].message

                if not message.tool_calls:
                    break

                if step_count >= STEP_BUDGET:
                    messages.append({"role": "system", "content": BUDGET_HIT_MESSAGE})
                    # tool_choice="none" forces a text reply — the model literally
                    # cannot request another tool call on this response, which is
                    # what makes BUDGET_HIT_MESSAGE's instructions binding rather
                    # than just a request it could ignore.
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=registry.get_schemas(),
                        tool_choice="none",
                    )
                    message = response.choices[0].message
                    break

                messages.append(message)

                for call in message.tool_calls:
                    arguments = json.loads(call.function.arguments)
                    result = registry.dispatch(call.function.name, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        }
                    )

                step_count += 1

            # After the loop: `message.content` should have real text. If it
            # doesn't, something unexpected happened (rare API edge case or a
            # bug in the loop termination logic). Raise loudly rather than
            # silently substituting a placeholder — silent fallbacks hide real
            # problems and were exactly the 3.3 None-content workaround we're
            # now removing.
            if not message.content:
                raise RuntimeError(
                    "Loop terminated but message.content is empty. "
                    "This shouldn't happen — check the API response and the "
                    "termination logic."
                )
            
            assistant_text = message.content

            messages.append({"role": "assistant", "content": assistant_text})
            print(f"\nagent > {assistant_text}\n")
    finally:
        # Step 2: tear down the sandbox no matter how run() exits.
        # This runs on normal exit, on exception, on user Ctrl-C — but
        # NOT on hard crashes (SIGKILL, power loss). The orphan cleanup
        # in Sandbox.start() handles those on the next session start.
        print("Stopping sandbox...")
        sandbox.stop()
        print("Sandbox stopped.")

if __name__ == "__main__":
    run()
