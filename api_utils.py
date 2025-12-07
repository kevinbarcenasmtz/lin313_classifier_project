import openai
from typing import Tuple, Dict


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """
    Get pricing information for OpenAI models.
    
    Pricing is per 1M tokens (as of 2024). Prices may vary by region and time.
    For most accurate pricing, refer to: https://openai.com/pricing
    
    Args:
        model_name: Name of the OpenAI model (e.g., 'gpt-4-turbo', 'gpt-3.5-turbo')
        
    Returns:
        Dict with 'input' and 'output' keys representing cost per 1M tokens
        
    Raises:
        ValueError: If model_name is not supported
    """
    model_name_lower = model_name.lower()
    
    # Pricing per 1M tokens (as of 2024)
    pricing = {
        'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
        'gpt-4': {'input': 30.00, 'output': 60.00},
        'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
        'gpt-4.1-nano': {'input': 0.15, 'output': 0.60}
    }
    
    # Match model name (case-insensitive, handle variations)
    if 'gpt-4-turbo' in model_name_lower or model_name_lower == 'gpt-4-turbo':
        return pricing['gpt-4-turbo']
    elif 'gpt-4' in model_name_lower and 'turbo' not in model_name_lower and 'nano' not in model_name_lower:
        return pricing['gpt-4']
    elif 'gpt-3.5-turbo' in model_name_lower:
        return pricing['gpt-3.5-turbo']
    elif 'gpt-4.1-nano' in model_name_lower or 'nano' in model_name_lower:
        return pricing['gpt-4.1-nano']
    else:
        # Default to gpt-4-turbo if unknown, but warn
        import warnings
        warnings.warn(f"Unknown model '{model_name}', defaulting to gpt-4-turbo pricing")
        return pricing['gpt-4-turbo']


def estimate_api_cost(
    model_name: str,
    num_test_tweets: int,
    num_ablations: int,
    few_shot_examples: int,
    avg_tweet_length: int = 100,
    avg_output_tokens: int = 20
) -> Dict[str, float]:
    """
    Estimate API cost for running classification experiments.
    
    Args:
        model_name: OpenAI model name
        num_test_tweets: Number of test tweets to classify
        num_ablations: Number of different configurations to run
        few_shot_examples: Number of few-shot examples per prompt
        avg_tweet_length: Average tweet length in tokens
        avg_output_tokens: Average output tokens per response
        
    Returns:
        Dict with cost breakdown including total_cost, input_cost, output_cost, etc.
    """
    pricing = get_model_pricing(model_name)
    
    system_tokens = 50
    instruction_tokens = 100
    few_shot_tokens = (avg_tweet_length + 50) * few_shot_examples
    target_tweet_tokens = avg_tweet_length
    
    input_tokens_per_request = system_tokens + instruction_tokens + few_shot_tokens + target_tweet_tokens
    output_tokens_per_request = avg_output_tokens
    
    total_input_tokens = input_tokens_per_request * num_test_tweets * num_ablations
    total_output_tokens = output_tokens_per_request * num_test_tweets * num_ablations
    
    input_cost = (total_input_tokens / 1_000_000) * pricing['input']
    output_cost = (total_output_tokens / 1_000_000) * pricing['output']
    total_cost = input_cost + output_cost
    
    cost_per_ablation = total_cost / num_ablations if num_ablations > 0 else 0
    
    return {
        'total_cost': total_cost,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'cost_per_ablation': cost_per_ablation,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'tokens_per_request_input': input_tokens_per_request,
        'tokens_per_request_output': output_tokens_per_request
    }


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate an OpenAI API key by checking format and making a test API call.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not api_key:
        return False, "API key cannot be empty"
    
    if not api_key.startswith("sk-"):
        return False, "Invalid format: API key must start with 'sk-'"
    
    if len(api_key) < 20:
        return False, "Invalid format: API key too short"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        # Use a lightweight model for validation to minimize cost
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model for validation
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return True, "API key is valid and connected"
    except openai.AuthenticationError:
        return False, "Authentication failed: Invalid API key"
    except openai.RateLimitError:
        return False, "Rate limit exceeded: Please try again later"
    except openai.APIError as e:
        return False, f"API error: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

