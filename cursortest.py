import os
import time
from datetime import datetime
import mlflow
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set MLflow tracking URI (optional - defaults to local mlruns folder)
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("openai-logging")


def log_openai_call(
    messages,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=None,
    run_name=None
):
    """
    Make an OpenAI API call and log all details to MLflow.
    
    Args:
        messages: List of message dicts for the OpenAI chat completion
        model: OpenAI model to use
        temperature: Temperature parameter
        max_tokens: Maximum tokens to generate
        run_name: Optional name for the MLflow run
    
    Returns:
        The response from OpenAI
    """
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("model", model)
        mlflow.log_param("temperature", temperature)
        if max_tokens:
            mlflow.log_param("max_tokens", max_tokens)
        mlflow.log_param("num_messages", len(messages))
        
        # Log input messages as text artifact
        prompt_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in messages
        ])
        mlflow.log_text(prompt_text, "prompt.txt")
        
        # Make the OpenAI API call and track timing
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            latency = time.time() - start_time
            
            # Log metrics
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("prompt_tokens", response.usage.prompt_tokens)
            mlflow.log_metric("completion_tokens", response.usage.completion_tokens)
            mlflow.log_metric("total_tokens", response.usage.total_tokens)
            
            # Calculate approximate cost (prices as of 2024)
            cost_per_1k_prompt = {
                "gpt-4": 0.03,
                "gpt-4-turbo": 0.01,
                "gpt-4o": 0.005,
                "gpt-4o-mini": 0.00015,
                "gpt-3.5-turbo": 0.0005
            }
            cost_per_1k_completion = {
                "gpt-4": 0.06,
                "gpt-4-turbo": 0.03,
                "gpt-4o": 0.015,
                "gpt-4o-mini": 0.0006,
                "gpt-3.5-turbo": 0.0015
            }
            
            # Get base model name for cost calculation
            base_model = model.split("-")[0:2]
            if len(base_model) > 1:
                base_model = "-".join(base_model)
            else:
                base_model = base_model[0]
            
            if model in cost_per_1k_prompt:
                prompt_cost = (response.usage.prompt_tokens / 1000) * cost_per_1k_prompt[model]
                completion_cost = (response.usage.completion_tokens / 1000) * cost_per_1k_completion[model]
                total_cost = prompt_cost + completion_cost
                mlflow.log_metric("estimated_cost_usd", total_cost)
            
            # Log response details
            mlflow.log_param("finish_reason", response.choices[0].finish_reason)
            mlflow.log_param("response_id", response.id)
            mlflow.log_param("created_at", datetime.fromtimestamp(response.created).isoformat())
            
            # Log response content as artifact
            response_text = response.choices[0].message.content
            mlflow.log_text(response_text, "response.txt")
            
            # Log tags for easier filtering
            mlflow.set_tag("status", "success")
            mlflow.set_tag("model_family", model.split("-")[0])
            
            print(f"✓ Successfully logged OpenAI call to MLflow (Run ID: {mlflow.active_run().info.run_id})")
            
            return response
            
        except Exception as e:
            latency = time.time() - start_time
            mlflow.log_metric("latency_seconds", latency)
            mlflow.set_tag("status", "error")
            mlflow.log_param("error_message", str(e))
            mlflow.log_text(str(e), "error.txt")
            print(f"✗ Error logged to MLflow: {e}")
            raise


def example_usage():
    """Example demonstrating how to use the logging function."""
    
    # Example 1: Simple question
    print("\n=== Example 1: Simple Question ===")
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = log_openai_call(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.7,
        run_name="simple_question"
    )
    print(f"Response: {response.choices[0].message.content}\n")
    
    # Example 2: Multi-turn conversation
    print("\n=== Example 2: Multi-turn Conversation ===")
    messages = [
        {"role": "system", "content": "You are a helpful Python programming assistant."},
        {"role": "user", "content": "How do I read a CSV file in Python?"},
    ]
    response = log_openai_call(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=500,
        run_name="python_help"
    )
    print(f"Response: {response.choices[0].message.content[:200]}...\n")
    
    # Example 3: Batch processing with loop
    print("\n=== Example 3: Batch Processing ===")
    questions = [
        "What is 2+2?",
        "Name a color.",
        "What year is it?"
    ]
    
    for i, question in enumerate(questions):
        messages = [{"role": "user", "content": question}]
        response = log_openai_call(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0.3,
            run_name=f"batch_question_{i+1}"
        )
        print(f"Q{i+1}: {question}")
        print(f"A{i+1}: {response.choices[0].message.content}\n")
    
    print("\n" + "="*60)
    print("All calls have been logged to MLflow!")
    print("View results by running: mlflow ui")
    print("="*60)


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
    else:
        example_usage()
