import os
from dotenv import load_dotenv
from openai import OpenAI


# load variables from .env (e.g. OPENAI_API_KEY) into the process environment
load_dotenv()

MODEL = "gpt-4o-mini" # cheap and small

# reads OPENAI_API_KEY from the environment automatically
client = OpenAI()

def run():
    """Run the loop till the user quiets"""

    # conversation history - entire memory of the agent
    # every turn we append to it and send the whole thing to the model
    # (there's no server-side session state - the full transcript IS the state)

    print("Agent ready. Type 'quite' or 'exit' to leave. \n")

    messages = []

    while True:

        # block here until the user types something and hits enter
        user_input = input("you > ").strip()

        if user_input in {"quit", "exit"}:
            print("Goodbye")
            break

        # ignore empty input (blank line) instead of sending it to the model
        if not user_input:
            continue

        # record the user's turn before calling the model
        messages.append({
            "role": "user",
            "content": user_input
        })

        # send the whole conversation so far - the model has no memory of its own
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )

        # pull just the reply text out of the response object
        # (see docs/openai_chat_completion_response.md for the full shape)
        assistant_message = response.choices[0].message.content

        # you need to save user message but as well the ML response
        # otherwise the next request would forget the agent's own reply
        messages.append({"role":"assistant", "content": assistant_message })

        print(f"\nagent > {assistant_message}\n")

if __name__ == "__main__":
    run()
