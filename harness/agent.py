import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from harness.tools import registry

# load variables from .env (e.g. OPENAI_API_KEY) into the process environment
load_dotenv()

# pick the provider at startup: if a Kimi key is configured, use Kimi's
# OpenAI-compatible endpoint instead of talking to OpenAI directly
if os.getenv("KIMI_API_KEY"):
    MODEL = "kimi-k2.6"
    # Kimi exposes an OpenAI-compatible API, so the same OpenAI SDK works -
    # just point it at Kimi's base_url with a Kimi key instead
    client = OpenAI(
        api_key=os.getenv("KIMI_API_KEY"), base_url=os.getenv("KIMI_BASE_URL")
    )
    EXTRA_BODY = {
        "thinking": {"type": "disabled"}
    }  # disabled to not consume to much tokens

else:
    MODEL = "gpt-4o-mini"  # cheap and small
    # # reads OPENAI_API_KEY from the environment automatically
    client = OpenAI()
    EXTRA_BODY = {}  # no provider-specific extras needed for plain OpenAI


SYSTEM_PROMPT = """
You are a coding assistant running in a terminal, helping a developer with software engineering tasks.

Be concise. Prefer short, direct answers over long ones. When the user asks for code, return the code with
minimal explanation unless they ask for more.

When returning code, use fenced code blocks and specify the language.

You have access to five filesystem tools — read, write, list, mkdir, delete — operating on a workspace directory. 
Use them whenever a task involves reading, modifying, or organizing files. Paths are relative to
the workspace root. Prefer reading and writing real files over describing them in conversation.
"""


def run():
    """Run the loop till the user quiets"""

    # conversation history - entire memory of the agent
    # every turn we append to it and send the whole thing to the model
    # (there's no server-side session state - the full transcript IS the state)

    print("Agent ready. Type 'quite' or 'exit' to leave. \n")

    # messages = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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

        # record the user's turn before calling the model
        messages.append({"role": "user", "content": user_input})

        # send the whole conversation so far - the model has no memory of its own
        # extra_body carries provider-specific options not in the standard OpenAI
        # API (e.g. Kimi's "disable thinking"); it's just {} for plain OpenAI
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            extra_body=EXTRA_BODY,
            # tells the model which tools it can call (read/write/list/mkdir/delete);
            # schemas are generated from the registry, so this list updates on its
            # own as tools are added or removed - nothing here to keep in sync by hand
            tools = registry.get_schemas()
        )

        message = response.choices[0].message                
        print(f"\n **** DEBUG **** \n")
        print(f"\n **** DEBUG **** message   > {message}\n")
        print(f"\n **** DEBUG **** \n")
        print(f"\n **** DEBUG **** message   > {message.tool_calls}\n")
        print(f"\n **** DEBUG **** \n")
        
        # If the model asked for a tool call, handle it before producing    
        # the user-facing reply. Minimum-viable dispatch: one round only.
        if message.tool_calls:
            # Step 1: record the model's tool-call message in history so the
            # upcoming tool-result messages have something to reference.
            messages.append(message)

            # Step 2: run each requested tool and append its result to history,
            # using the matching tool_call_id so the model can pair them up.
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                result = registry.dispatch(call.function.name, arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

            # Step 3: re-call the model now that the tool results are in
            # context. This second call produces the model's final text reply.
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=registry.get_schemas(),
            )
            message = response.choices[0].message

        # By here, `message` is the model's final text response for this turn —
        # either from the first call (no tools needed) or the second (after dispatch).
        assistant_text = message.content 
        messages.append({"role": "assistant", "content": assistant_text})

        print(f"\nagent > {assistant_text}\n")


if __name__ == "__main__":
    run()
