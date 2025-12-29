"""
Local-first HR triage agent using Ollama.

Models:
- all-minilm        -> intent detection (embeddings)
- phi3:mini        -> HR response planning
- function-gemma:2b -> constrained function execution

Use case:
- Private employee reports
- Burnout, harassment, policy violations
- GDPR / labor-law sensitive data
"""

import ollama
import numpy as np
import json
import time
import sys
import json_repair


# -----------------------------
# Utilities
# -----------------------------

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
# HR Functions (Tools)
# -----------------------------

def open_hr_case(employee_id, category, risk_level):
    """
    Open a confidential HR case.
    category: harassment | burnout | policy_violation | performance
    risk_level: LOW | MEDIUM | HIGH
    """
    print(f"[ACTION] Open HR case for employee={employee_id}, category={category}, risk={risk_level}")


def notify_legal(case_id):
    """
    Notify legal department about a sensitive HR case.
    """
    print(f"[ACTION] Notify Legal for HR case={case_id}")


def schedule_hr_meeting(employee_id, urgency):
    """
    Schedule an HR follow-up meeting.
    urgency: LOW | MEDIUM | HIGH
    """
    print(f"[ACTION] Schedule HR meeting for employee={employee_id}, urgency={urgency}")


FUNCTIONS = {
    "open_hr_case": open_hr_case,
    "notify_legal": notify_legal,
    "schedule_hr_meeting": schedule_hr_meeting
}


# -----------------------------
# Intent Detection (MiniLM)
# -----------------------------

INTENTS = {
    "harassment": "Harassment, intimidation, or misconduct at work",
    "burnout": "Employee burnout, stress, or mental health risk",
    "policy_violation": "Internal policy or ethics violation",
    "performance": "Performance or role-related concern",
    "noise": "Irrelevant or non-HR related input"
}

print("[INIT] Loading HR intent embeddings (MiniLM)...")
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

def plan_hr_response(report: str, intent: str) -> list:
    prompt = f"""
You are an HR response planner.

Given a confidential employee report and its classified intent,
produce an ordered plan of actions.

Do not explain just return the steps in the plan.
Return the plan as a JSON array of steps each step contains one of the available actions.


Available actions:
- open_hr_case(employee_id, category, risk_level)
- notify_legal(case_id)
- schedule_hr_meeting(employee_id, urgency)

Employee report:
{report}

Detected intent:
{intent}
"""

    response = ollama.generate(
        model="phi3:mini",
        prompt=prompt,
        options={"temperature": 0}
    )["response"]

    try:
        response = clean_json_response(response)
        return json_repair.loads(response)
    except json.JSONDecodeError:
        print("[ERROR] Planner returned non-JSON output:")
        print(response)
        sys.exit(1)


# -----------------------------
# Tool Execution (Function Gemma)
# -----------------------------

def call_function(step: dict):
    messages = [
        {
            "role": "user",
            "content": f"Select the correct function and arguments for the following HR action step:\n{step}"
        }
    ]

    print(f"[STEP]\n{step}")

    response = ollama.chat(
        model="functiongemma",
        messages=messages,
        tools=[open_hr_case, notify_legal, schedule_hr_meeting]
    )

    return response


# -----------------------------
# Agent Orchestrator
# -----------------------------

def handle_employee_report(text: str):
    print("\n==============================")
    print(f"[EMPLOYEE REPORT]\n{text}")

    intent = detect_intent(text)
    print(f"[INTENT] {intent}")

    if intent == "noise":
        print("[INFO] Not an HR-relevant report. Stopping.")
        return

    plan = plan_hr_response(text, intent)

    print("[HR PLAN]")
    print(json.dumps(plan, indent=2))

    for step in plan:
        response = call_function(step)

        if response["message"]["role"] == "assistant" and "tool_calls" in response["message"]:
            tool_call = response["message"]["tool_calls"][0]
            fn = FUNCTIONS.get(tool_call.function.name)

            if not fn:
                print(f"[WARN] Unknown function: {tool_call.function.name}")
                continue

            print(
                f"[EXECUTE] {tool_call.function.name} "
                f"args={tool_call.function.arguments}"
            )

            fn(**tool_call.function.arguments)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    test_report = """
        My Eployee ID is 12345: I have been working excessive hours for months. 
        My manager threatens retaliation if I raise concerns, 
        and I am feeling mentally exhausted and unsafe.
    """

    start = time.time()
    handle_employee_report(test_report)
    end = time.time()

    print(f"\n[TOTAL LATENCY] {end - start:.2f}s")
