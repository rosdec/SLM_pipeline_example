"""
HR function tools for the triage agent.

These functions represent the available actions that can be taken
in response to HR reports.
"""


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


# Registry of available functions
FUNCTIONS = {
    "open_hr_case": open_hr_case,
    "notify_legal": notify_legal,
    "schedule_hr_meeting": schedule_hr_meeting
}


# List of tool functions for Ollama function calling
TOOLS = [open_hr_case, notify_legal, schedule_hr_meeting]
