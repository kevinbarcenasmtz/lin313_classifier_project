import openai
from typing import Tuple, Dict


def get_model_pricing(model_name: str) -> Dict[str, float]:
    pricing = {
        'gpt-4.1-nano': {'input': 0.15, 'output': 0.60}
    }
    
    if 'gpt-4.1-nano' in model_name.lower():
        return pricing['gpt-4.1-nano']
    return pricing['gpt-4.1-nano']


def estimate_api_cost(
    model_name: str,
    num_test_tweets: int,
    num_ablations: int,
    few_shot_examples: int,
    avg_tweet_length: int = 100,
    avg_output_tokens: int = 20
) -> Dict:
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
    if not api_key:
        return False, "API key cannot be empty"
    
    if not api_key.startswith("sk-"):
        return False, "Invalid format: API key must start with 'sk-'"
    
    if len(api_key) < 20:
        return False, "Invalid format: API key too short"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
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

