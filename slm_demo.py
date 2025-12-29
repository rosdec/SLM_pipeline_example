import json
import re
from llama_cpp import Llama

# ===============================
# 1. LOAD MODEL
# ===============================

llm = Llama(
    model_path="./models/functiongemma-270m-it-q4_k_m.gguf",
    n_ctx=4096,
    n_threads=8,
    temperature=0.0,
    top_p=1.0,
    repeat_penalty=1.1,
    verbose=False
)

# ===============================
# 2. REAL FUNCTIONS
# ===============================

def turn_on_light(room: str) -> str:
    return f"Lights in the {room} are now ON."

def turn_off_light(room: str) -> str:
    return f"Lights in the {room} are now OFF."

def set_timer(seconds: int) -> str:
    return f"Timer set for {seconds} seconds."

FUNCTIONS = {
    "turn_on_light": turn_on_light,
    "turn_off_light": turn_off_light,
    "set_timer": set_timer,
}

# ===============================
# 3. FEW-SHOT + HARD DELIMITERS
# ===============================

SYSTEM_PROMPT = """
You are a function router.

You MUST output a function call wrapped in <json> tags.
Inside the tags, output VALID JSON only.
No text outside the tags.

Available functions:

turn_on_light(room: string)
set_timer(seconds: integer)

EXAMPLES:

User: Turn on the kitchen lights
<json>
{"name":"turn_on_light","arguments":{"room":"kitchen"}}
</json>

User: Set a timer for 10 seconds
<json>
{"name":"set_timer","arguments":{"seconds":10}}
</json>
"""

def build_prompt(user_input: str) -> str:
    return f"""
{SYSTEM_PROMPT}

User: {user_input}
<json>
"""

# ===============================
# 4. MODEL CALL
# ===============================

def generate(prompt: str) -> str:
    output = llm(
        prompt,
        max_tokens=128,
        stop=["</json>"]
    )
    return output["choices"][0]["text"]

# ===============================
# 5. JSON EXTRACTION
# ===============================

def extract_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON:\n{text}") from e

# ===============================
# 6. MAIN HANDLER
# ===============================

def handle_user_request(user_input: str) -> str:
    prompt = build_prompt(user_input)

    raw = generate(prompt)

    call = extract_json(raw)

    print(raw)

    name = call.get("name")
    args = call.get("arguments", {})

    if name not in FUNCTIONS:
        raise RuntimeError(f"Unknown function: {name}")

    result = FUNCTIONS[name](**args)

    return result

# ===============================
# 7. INTERACTIVE LOOP
# ===============================

if __name__ == "__main__":
    print("FunctionGemma (local, llama.cpp)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User> ").strip()
        if user_input.lower() == "exit":
            break

        try:
            response = handle_user_request(user_input)
            print(f"Assistant> {response}\n")
        except Exception as e:
            print(f"ERROR: {e}\n")
