import os
from dotenv import load_dotenv
from openai import OpenAI

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

You do not currently have access to any tools — you cannot read files, run commands,
or modify anything on the user's system. If the user asks you to do something that would require a tool,
say so plainly and suggest they describe the relevant content directly.
"""

# SYSTEM_PROMPT = """
# You are a coding assistant running in a terminal, helping a developer with software engineering tasks.

# When returning code, use fenced code blocks and specify the language.

# You do not currently have access to any tools — you cannot read files, run commands,
# or modify anything on the user's system. If the user asks you to do something that would require a tool,
# say so plainly and suggest they describe the relevant content directly.
# """


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
            model=MODEL, messages=messages, extra_body=EXTRA_BODY
        )

        # pull just the reply text out of the response object
        # (see docs/openai_chat_completion_response.md for the full shape)
        assistant_message = response.choices[0].message.content

        # you need to save user message but as well the ML response
        # otherwise the next request would forget the agent's own reply
        messages.append({"role": "assistant", "content": assistant_message})

        print(f"\nagent > {assistant_message}\n")


if __name__ == "__main__":
    run()
