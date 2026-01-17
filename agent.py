"""
HR triage agent orchestrator.

Coordinates the intent detection, planning, and execution pipeline.
"""

import json
from intent_detector import IntentDetector
from planner import Planner
from executor import ToolExecutor


class HRTriageAgent:
    """Main orchestrator for HR triage workflow."""
    
    def __init__(self):
        """Initialize the HR triage agent with all components."""
        self.intent_detector = IntentDetector()
        self.planner = Planner()
        self.executor = ToolExecutor()
    
    def handle_report(self, text: str):
        """
        Handle an employee report through the full pipeline.
        
        Args:
            text: The employee report text
        """
        print("\n==============================")
        print(f"[EMPLOYEE REPORT]\n{text}")
        
        # Step 1: Detect intent
        intent = self.intent_detector.detect(text)
        print(f"[INTENT] {intent}")
        
        if intent == "noise":
            print("[INFO] Not an HR-relevant report. Stopping.")
            return
        
        # Step 2: Plan response
        plan = self.planner.plan(text, intent)
        
        print("[HR PLAN]")
        print(json.dumps(plan, indent=2))
        
        # Step 3: Execute plan
        for step in plan:
            self.executor.execute_step(step)
