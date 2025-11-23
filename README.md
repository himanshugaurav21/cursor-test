# cursor-test

This project demonstrates how to log OpenAI API calls to MLflow for tracking, monitoring, and analysis.

## Features

- 🔍 Track all OpenAI API calls with detailed metrics
- 💰 Calculate and log estimated costs per call
- ⏱️ Monitor API latency and response times
- 📊 Log prompt tokens, completion tokens, and total tokens
- 📝 Save prompts and responses as MLflow artifacts
- 🏷️ Tag runs for easy filtering and analysis
- ⚠️ Error tracking and logging

## Installation

1. Install required dependencies:
```bash
pip install openai mlflow
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

Run the example script:
```bash
python cursortest.py
```

This will:
- Make several example OpenAI API calls
- Log all details (tokens, cost, latency, etc.) to MLflow
- Save prompts and responses as artifacts

## View Results in MLflow UI

After running the script, view your logged experiments:
```bash
mlflow ui
```

Then open your browser to `http://127.0.0.1:5000` to view:
- All API calls with timestamps
- Token usage and estimated costs
- Latency metrics
- Prompts and responses
- Model parameters

## Logged Metrics

Each OpenAI call logs:
- **Parameters**: model, temperature, max_tokens, finish_reason
- **Metrics**: latency, prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd
- **Artifacts**: prompt.txt (input), response.txt (output)
- **Tags**: status, model_family

## Code Example

```python
from cursortest import log_openai_call

messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

response = log_openai_call(
    messages=messages,
    model="gpt-4o-mini",
    temperature=0.7,
    run_name="my_test_run"
)
```

## MLflow Experiment Structure

All runs are logged to the `openai-logging` experiment. Data is stored in the local `./mlruns` directory by default. 
