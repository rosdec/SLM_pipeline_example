# SLM Pipeline - Local-first HR Triage System

A local-first agentic pipeline using Ollama small language models for HR report management. This exploratory implementation uses multiple specialized models for intent detection, planning, and tool execution.

## Overview

This pipeline demonstrates a multi-model approach to automated HR report management:

- **all-minilm** → Intent detection using embeddings
- **phi3:mini** → Planning and reasoning
- **functiongemma** → Tool execution with function calling

The system receives confidential employee reports, classifies them, generates a remediation plan, and executes actions through simulated function calls.

## Features

- **Intent Detection**: Classifies input as harassment, burnout, policy violation, or noise
- **Automated Planning**: Generates step-by-step HR response plans
- **Function Execution**: Calls appropriate tools based on the plan
- **Available Actions**:
  - `open_hr_case(employee_id, category, risk_level)` - Open a confidential HR case
  - `notify_legal(case_id)` - Notify the legal department about sensitive cases
  - `schedule_hr_meeting(employee_id, urgency)` - Schedule follow-up HR meetings

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running

## Installation

### 1. Set up Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama Models

Make sure Ollama is installed and running, then pull the required models:

```bash
ollama pull all-minilm
ollama pull phi3:mini
ollama pull functiongemma
```

Verify installation:
```bash
ollama ls
```

Expected output:
```
NAME                    ID              SIZE      MODIFIED   
all-minilm:latest       1b226e2802db    45 MB     ...    
phi3:mini               4f2222927938    2.2 GB    ...    
functiongemma:latest    7c19b650567a    300 MB    ...
```

## Usage

Run the pipeline:

```bash
python slm_pipeline.py
```

The default test scenario simulates an HR report:
```
"My Employee ID is 12345: I have been working excessive hours for months. My manager threatens retaliation if I raise concerns, and I am feeling mentally exhausted and unsafe."
```

### Example Output

```
[INIT] Loading intent embeddings (MiniLM)...
==============================
[INPUT] My Employee ID is 12345: I have been working excessive hours for months. My manager threatens retaliation if I raise concerns, and I am feeling mentally exhausted and unsafe.
[INTENT] burnout
[PLAN]
[
  {
    "action": "open_hr_case",
    "employee_id": "12345",
    "category": "burnout",
    "risk_level": "HIGH"
  },
  {
    "action": "schedule_hr_meeting",
    "employee_id": "12345",
    "urgency": "HIGH"
  }
]
[STEP]
 {'action': 'open_hr_case', 'employee_id': '12345', 'category': 'burnout', 'risk_level': 'HIGH'}
[ACTION] Open HR case for employee=12345, category=burnout, risk=HIGH
...
[TOTAL LATENCY] X.XXs
```

## Project Structure

```
slm/
├── slm_pipeline.py      # Main pipeline implementation
├── requirements.txt     # Python dependencies
├── models/              # (Model storage directory)
└── README.md           # This file
```

## Architecture

1. **Intent Detection**: Uses MiniLM embeddings to classify input via cosine similarity
2. **Planning**: Phi-3 Mini generates structured JSON plans with ordered steps
3. **Execution**: Function Gemma selects and calls appropriate functions based on plan steps

## Notes

- This is an **exploratory implementation** for demonstration purposes
- Function calls are simulated (print statements) - replace with actual implementations
- Function Gemma may occasionally produce non-JSON outputs; error handling is included
- All models run locally via Ollama for privacy and offline capability

## Customization

### Adding New Functions

1. Define the function in `slm_pipeline.py`
2. Add to the `FUNCTIONS` dictionary
3. Include in the planner's available actions
4. Add to Function Gemma's tools list

### Modifying Test Scenarios

Edit the `test_alert` variable in the `__main__` block:

```python
if __name__ == "__main__":
    test_alert = "Your custom HR report here"
    handle_input(test_alert)
```

## Dependencies

- `ollama>=0.1.0` - Ollama Python client
- `numpy>=1.24.0` - Numerical operations for embeddings
- `json-repair>=0.25.0` - Robust JSON parsing

## License

This is an exploratory project for demonstration purposes.

## Troubleshooting

**Models not found**: Ensure all three models are installed via `ollama pull`

**Ollama not running**: Start Ollama service before running the pipeline

**JSON parsing errors**: Function Gemma may produce non-JSON output; adjust temperature or retry logic as needed
