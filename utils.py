"""
Utility functions for HR triage agent.
"""


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
