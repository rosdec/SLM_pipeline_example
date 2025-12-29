"""
Local-first agentic pipeline using Ollama.

Models:
- all-minilm        -> intent detection (embeddings)
- phi3:mini        -> planning / reasoning
- function-gemma:2b -> tool execution (best-effort, prompt-constrained)

This is an EXPLORATORY implementation.
Expect occasional non-JSON outputs from Function Gemma.
"""

import ollama
import numpy as np
import json
import time
import sys
import json_repair


def clean_json_response(response: str) -> str:
    """Remove markdown formatting from JSON response."""
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    return response.strip()

# -----------------------------
# Functions definition
# -----------------------------

def restart_service(service, env) -> str:
    """
    Restart the specified service for the given environment. 
    Args: service: The name of the service to restart, env: the environment
    Returns: None
    """

    print(f"[ACTION] Restarting {service} in {env}")


def scale_service(service, replicas)-> str:
    """
    Scale up the specified service to a given number of replicas
    Args: service: The name of the service to restart, replicas: the number (integer) of replicas
    Returns: None
    """
    
    print(f"[ACTION] Scaling {service} to {replicas} replicas")


def open_ticket(summary, severity)-> str:
    """
    Opens a ticket with a given summary and a specified severity level
    Args: summary: reason for the ticket, severity: severity level for the ticket LOW, HIGH or SEVERE
    Returns: None
    """
    
    print(f"[ACTION] Opening ticket: {summary} ({severity})")

FUNCTIONS = {
    "restart_service": restart_service,
    "scale_service": scale_service,
    "open_ticket": open_ticket
}


# -----------------------------
# Intent detection (MiniLM)
# -----------------------------
INTENTS = {
    "incident": "System incident or outage",
    "question": "User question",
    "noise": "Irrelevant input"
}

print("[INIT] Loading intent embeddings (MiniLM)...")
intent_texts = list(INTENTS.values())
intent_embeddings = [
    ollama.embeddings(model="all-minilm", prompt=text)["embedding"]
    for text in intent_texts
]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect_intent(text: str) -> str:
    emb = ollama.embeddings(
        model="all-minilm",
        prompt=text
    )["embedding"]

    scores = [cosine_sim(emb, e) for e in intent_embeddings]
    return list(INTENTS.keys())[int(np.argmax(scores))]

# -----------------------------
# Planner (Phi-3 Mini)
# -----------------------------

def plan(alert: str) -> dict:
    prompt = f"""
You are an incident response planner.

Given an alert, produce a plan with ordered steps containing the function to call and the parameters.
Do not explain just return the steps in the plan.
Return the plan as a JSON array of steps each step contains one of the available actions.

Available actions:
- restart_service(service, env)
- scale_service(service, replicas)
- open_ticket(summary, severity)

Alert:
{alert}
"""

    response = ollama.generate(
        model="phi3:mini",
        prompt=prompt,
        options={
            "temperature": 0
        }
    )["response"]

    try:
        response = clean_json_response(response)

        return json_repair.loads(response)
    except json.JSONDecodeError:
        print("[ERROR] Planner returned non-JSON:")
        print(response)
        sys.exit(1)


# -----------------------------
# Tool execution (Function Gemma)
# -----------------------------

def call_function(step: dict) -> dict | None:
    messages = [
        {
            'role': 'user', 
            'content': f'Select the correct function for the following step: {step}'
        }
    ]
    print(f"[STEP]\n {step}")

    response = ollama.chat(
        model='functiongemma',
        messages=messages,
        tools=[open_ticket, scale_service, restart_service] # Pass your function here
    )    
    
    print(f"[FUNCTION]\n{response}")
    
    return response


# -----------------------------
# Agent manager (orchestrator)
# -----------------------------

def handle_input(text: str):
    print("\n==============================")
    print(f"[INPUT] {text}")

    intent = detect_intent(text)
    print(f"[INTENT] {intent}")

    if intent != "incident":
        print("[INFO] Not an incident. Stopping.")
        return

    plan_json = plan(text)
    print("[PLAN]")
    print(json.dumps(plan_json, indent=2))

    for step in plan_json:
        call = call_function(step)
        if call['message']['role'] == 'assistant' and 'tool_calls' in call['message']:
            tool_call = call['message']['tool_calls'][0]
            print(f"Gemma wants to call: {tool_call.function.name} with args: {tool_call.function.arguments}")
            fn = FUNCTIONS.get(tool_call.function.name)
            
            if not fn:
                print(f"[WARN] Unknown function: {tool_call.function.name}")
                continue

            fn(**tool_call.function.arguments)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    test_alert = "Master service has been returning 500 errors in production for 10 minutes"

    start = time.time()
    handle_input(test_alert)
    end = time.time()

    print(f"\n[TOTAL LATENCY] {end - start:.2f}s")
