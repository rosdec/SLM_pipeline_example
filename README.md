# SLM Pipeline - Local-first Agentic System

A local-first agentic pipeline using Ollama small language models for incident response automation. This exploratory implementation uses multiple specialized models for intent detection, planning, and tool execution.

## Overview

This pipeline demonstrates a multi-model approach to automated incident response:

- **all-minilm** → Intent detection using embeddings
- **phi3:mini** → Planning and reasoning
- **functiongemma** → Tool execution with function calling

The system receives incident alerts, classifies them, generates a remediation plan, and executes actions through simulated function calls.

## Features

- **Intent Detection**: Classifies input as incident, question, or noise
- **Automated Planning**: Generates step-by-step remediation plans
- **Function Execution**: Calls appropriate tools based on the plan
- **Available Actions**:
  - `restart_service(service, env)` - Restart a service in an environment
  - `scale_service(service, replicas)` - Scale a service to N replicas
  - `open_ticket(summary, severity)` - Create support tickets

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

The default test scenario simulates a production incident:
```
"Master service has been returning 500 errors in production for 10 minutes"
```

### Example Output

```
[INIT] Loading intent embeddings (MiniLM)...
==============================
[INPUT] Master service has been returning 500 errors in production for 10 minutes
[INTENT] incident
[PLAN]
[
  {
    "action": "restart_service",
    "service": "master",
    "env": "production"
  },
  {
    "action": "open_ticket",
    "summary": "Master service 500 errors",
    "severity": "HIGH"
  }
]
[STEP]
 {'action': 'restart_service', 'service': 'master', 'env': 'production'}
[ACTION] Restarting master in production
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
    test_alert = "Your custom incident alert here"
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
