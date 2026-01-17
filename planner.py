"""
HR response planner using Phi-3 Mini.

Generates an ordered plan of actions to handle employee reports.
"""

import json
import sys
import ollama
import json_repair
from utils import clean_json_response


class Planner:
    """Plans HR responses using Phi-3 Mini."""
    
    def __init__(self):
        """
        Initialize the HR planner.
        
        Args:
        """
   
    def plan(self, report: str, intent: str) -> list:
        """
        Plan HR response actions for a given report.
        
        Args:
            report: The employee report text
            intent: The detected intent category
            
        Returns:
            A list of action steps to execute
        """
        prompt = f"""
You are an HR response planner.

Given a confidential employee report and its classified intent,
produce an ordered plan of actions.

Do not explain just return the steps in the plan.
Return the plan as a JSON array of action steps. Each step should describe what needs to be done.


Available types of actions you can recommend:
- Creating a formal HR case to document and track the issue (specify employee, category, and severity)
- Escalating the matter to the legal department (specify case reference)
- Arranging a confidential meeting with HR (specify employee and priority level)

Categories: harassment, burnout, policy_violation, performance
Priority/Risk levels: LOW, MEDIUM, HIGH

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
