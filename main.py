"""
Local-first HR triage agent using Ollama.

Main entry point for the HR triage system.

Models:
- all-minilm        -> intent detection (embeddings)
- phi3:mini        -> HR response planning
- function-gemma:2b -> constrained function execution

Use case:
- Private employee reports
- Burnout, harassment, policy violations
- GDPR / labor-law sensitive data
"""

import time
from agent import HRTriageAgent


def main():
    """Main entry point."""
    # Initialize the agent
    agent = HRTriageAgent()
    
    # Test report
    test_report = """
        My Eployee ID is 12345: I have been working excessive hours for months. 
        My manager threatens retaliation if I raise concerns, 
        and I am feeling mentally exhausted and unsafe.
    """
    
    # Process the report
    start = time.time()
    agent.handle_report(test_report)
    end = time.time()
    
    print(f"\n[TOTAL LATENCY] {end - start:.2f}s")


if __name__ == "__main__":
    main()
