"""
HR response planner using Phi-3 Mini.

Generates an ordered plan of actions to handle employee reports.
"""

import json
import sys
import ollama
import json_repair
from utils import clean_json_response


class HRPlanner:
    """Plans HR responses using Phi-3 Mini."""
    
    def __init__(self, model: str = "phi3:mini"):
        """
        Initialize the HR planner.
        
        Args:
            model: The planning model to use (default: phi3:mini)
        """
        self.model = model
    
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
            model=self.model,
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
