import time
import re
import openai
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

from constants import CLASS_NAMES, CLASS_LABELS
from api_utils import get_model_pricing


def split_train_test(df: pd.DataFrame, test_size: float = 0.35, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df['stratify_key'] = df[CLASS_LABELS].apply(lambda x: ''.join(map(str, x)), axis=1)
    
    # Count occurrences of each stratify_key
    key_counts = df['stratify_key'].value_counts()
    
    # Group rare combinations (those with < 2 examples) into a single group
    # This ensures stratification can work
    rare_keys = key_counts[key_counts < 2].index
    if len(rare_keys) > 0:
        df.loc[df['stratify_key'].isin(rare_keys), 'stratify_key'] = 'rare_combination'
    
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=df['stratify_key']
        )
    except ValueError:
        # If stratification still fails (e.g., all examples are rare), use non-stratified split
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed
        )
    
    train_df = train_df.drop(columns=['stratify_key']).reset_index(drop=True)
    test_df = test_df.drop(columns=['stratify_key']).reset_index(drop=True)
    
    return train_df, test_df


def format_pool_examples(pool: Dict) -> List[Dict]:
    formatted = []
    for example in pool['examples']:
        formatted.append({
            'text': example['tweet_text'],
            'labels': example['labels']
        })
    return formatted


def build_classification_prompt(few_shot_examples: List[Dict], target_tweet: str) -> List[Dict]:
    system_message = {
        "role": "system",
        "content": "You are a classifier for LGBT+phobic content in Mexican Spanish tweets. You will classify tweets into one or more categories."
    }
    
    examples_text = "Here are some examples:\n\n"
    for example in few_shot_examples:
        text = example['text'].replace('"', '\\"')
        labels = example['labels']
        examples_text += f'Tweet: "{text}"\nLabels: {labels}\n\n'
    
    escaped_tweet = target_tweet.replace('"', '\\"')
    
    instruction = f"""Now classify this tweet into one or more of these categories:
- Gayphobia
- Lesbophobia
- Biphobia
- Transphobia
- Other

Tweet: "{escaped_tweet}"

Answer with only the applicable label names, separated by commas.
If none apply, answer "None"."""
    
    user_content = examples_text + instruction
    
    user_message = {
        "role": "user",
        "content": user_content
    }
    
    return [system_message, user_message]


def parse_llm_response(response_text: str) -> Tuple[List[int], str, bool]:
    if not response_text:
        return [0, 0, 0, 0, 0], "None", False
    
    normalized = response_text.lower().strip()
    
    if normalized in ['none', 'n/a', 'no labels', 'no', '']:
        return [0, 0, 0, 0, 0], "None", True
    
    binary_vector = [0, 0, 0, 0, 0]
    found_labels = []
    
    label_patterns = {
        'G': [r'\bgayphobia\b', r'\bgay-phobia\b', r'\bgay\s+phobia\b'],
        'L': [r'\blesbophobia\b', r'\blesbo-phobia\b', r'\blesbo\s+phobia\b'],
        'B': [r'\bbiphobia\b', r'\bbi-phobia\b', r'\bbi\s+phobia\b'],
        'T': [r'\btransphobia\b', r'\btrans-phobia\b', r'\btrans\s+phobia\b'],
        'O': [r'\bother\b']
    }
    
    for label_idx, (label_key, patterns) in enumerate(label_patterns.items()):
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                binary_vector[label_idx] = 1
                found_labels.append(CLASS_NAMES[label_idx])
                break
    
    if found_labels:
        parsed_labels = ', '.join(found_labels)
    else:
        parsed_labels = "None"
    
    is_valid = len(found_labels) > 0 or normalized in ['none', 'n/a', 'no labels', 'no', '']
    
    return binary_vector, parsed_labels, is_valid


def make_api_call_with_retry(
    client,
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3
) -> Tuple[str, Dict]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response_text = response.choices[0].message.content
            return response_text, {
                'success': True,
                'attempt': attempt + 1,
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return None, {
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'attempt': attempt + 1
                }
        except openai.APIError as e:
            if attempt == max_retries - 1:
                return None, {
                    'success': False,
                    'error': str(e),
                    'attempt': attempt + 1
                }
            wait_time = 2 ** attempt
            time.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                return None, {
                    'success': False,
                    'error': str(e),
                    'attempt': attempt + 1
                }
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    
    return None, {'success': False, 'error': 'Max retries exceeded'}


def calculate_cost_so_far(api_calls: int, tokens_used: Dict, model_name: str) -> float:
    pricing = get_model_pricing(model_name)
    
    input_cost = (tokens_used.get('input_tokens', 0) / 1_000_000) * pricing['input']
    output_cost = (tokens_used.get('output_tokens', 0) / 1_000_000) * pricing['output']
    
    return input_cost + output_cost


def save_checkpoint(results: List[Dict], config_name: str, metadata: Dict):
    st.session_state.experiment_results = {
        'results': results,
        'config': config_name,
        'metadata': metadata
    }


def create_progress_display(progress_container):
    """Create empty containers for progress display that will be updated in place."""
    with progress_container:
        progress_bar = st.empty()
        metrics_container = st.empty()
        eta_container = st.empty()
    return progress_bar, metrics_container, eta_container


def update_progress(
    progress_bar,
    metrics_container,
    eta_container,
    current: int,
    total: int,
    cost_so_far: float,
    success_count: int,
    error_count: int,
    start_time: float
):
    """Update progress display in place (only update every 5 tweets to reduce UI updates)."""
    if current % 5 != 0 and current != total:
        return  # Only update every 5 tweets to reduce UI overhead
    
    tweet_progress = current / total if total > 0 else 0
    progress_bar.progress(tweet_progress, text=f"Processing tweet {current}/{total}")
    
    # Update metrics in a single container to avoid creating multiple widgets
    with metrics_container.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Progress", f"{current}/{total}")
        with col2:
            st.metric("Cost So Far", f"${cost_so_far:.4f}")
        with col3:
            st.metric("Success", success_count)
        with col4:
            st.metric("Errors", error_count)
    
    if current > 0 and start_time:
        elapsed = time.time() - start_time
        avg_time_per_tweet = elapsed / current
        remaining_tweets = total - current
        eta_seconds = remaining_tweets * avg_time_per_tweet
        if eta_seconds > 0:
            eta_minutes = eta_seconds / 60
            eta_container.caption(f"Estimated time remaining: {eta_minutes:.1f} minutes")


def run_classification_experiment(
    test_df: pd.DataFrame,
    few_shot_examples: List[Dict],
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    config_name: str,
    progress_container
):
    """
    Run classification experiment with single few-shot configuration.
    
    Args:
        test_df: Test dataframe
        few_shot_examples: List of formatted few-shot examples
        api_key: OpenAI API key
        model_name: Model name
        temperature: Temperature setting
        max_tokens: Max tokens for response
        config_name: Configuration name (e.g., '15-shot')
        progress_container: Streamlit container for progress display
    """
    results = []
    client = openai.OpenAI(api_key=api_key)
    
    start_time = time.time()
    total_api_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    errors = []
    success_count = 0
    error_count = 0
    
    # Create progress display containers
    progress_bar, metrics_container, eta_container = create_progress_display(progress_container)
    
    for idx, row in test_df.iterrows():
        messages = build_classification_prompt(few_shot_examples, row['tweet_text'])
        
        response_text, api_metadata = make_api_call_with_retry(
            client, messages, model_name, temperature, max_tokens
        )
        
        total_api_calls += 1
        
        if api_metadata.get('success'):
            total_input_tokens += api_metadata.get('input_tokens', 0)
            total_output_tokens += api_metadata.get('output_tokens', 0)
            
            pred_vector, labels, is_valid = parse_llm_response(response_text)
            
            result = {
                'tweet_id': int(row['id']),
                'tweet_text': row['tweet_text'],
                'true_labels': row['label_vector'],
                'true_labels_str': row['labels'],
                'pred_labels': pred_vector,
                'pred_labels_str': labels,
                'is_valid': is_valid,
                'raw_response': response_text
            }
            results.append(result)
            success_count += 1
        else:
            error_info = {
                'tweet_id': int(row['id']),
                'error': api_metadata.get('error', 'Unknown error'),
                'attempt': api_metadata.get('attempt', 0)
            }
            errors.append(error_info)
            error_count += 1
            
            result = {
                'tweet_id': int(row['id']),
                'tweet_text': row['tweet_text'],
                'true_labels': row['label_vector'],
                'true_labels_str': row['labels'],
                'pred_labels': [0, 0, 0, 0, 0],
                'pred_labels_str': "Error",
                'is_valid': False,
                'raw_response': None,
                'error': api_metadata.get('error', 'Unknown error')
            }
            results.append(result)
        
        current_tweet = idx + 1
        tokens_used = {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens
        }
        cost_so_far = calculate_cost_so_far(total_api_calls, tokens_used, model_name)
        
        # Update progress (only updates every 5 tweets to reduce UI overhead)
        update_progress(
            progress_bar,
            metrics_container,
            eta_container,
            current_tweet,
            len(test_df),
            cost_so_far,
            success_count,
            error_count,
            start_time
        )
        
        if current_tweet % 50 == 0:
            metadata = {
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'model': model_name,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'test_set_size': len(test_df),
                'total_api_calls': total_api_calls,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'errors': errors
            }
            save_checkpoint(results, config_name, metadata)
    
    # Final progress update
    tokens_used = {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens
    }
    cost_so_far = calculate_cost_so_far(total_api_calls, tokens_used, model_name)
    update_progress(
        progress_bar,
        metrics_container,
        eta_container,
        len(test_df),
        len(test_df),
        cost_so_far,
        success_count,
        error_count,
        start_time
    )
    
    end_time = time.time()
    total_cost = calculate_cost_so_far(total_api_calls, {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens
    }, model_name)
    
    metadata = {
        'start_time': datetime.fromtimestamp(start_time).isoformat(),
        'end_time': datetime.fromtimestamp(end_time).isoformat(),
        'model': model_name,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'test_set_size': len(test_df),
        'total_api_calls': total_api_calls,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_cost': total_cost,
        'errors': errors
    }
    
    save_checkpoint(results, config_name, metadata)
    
    st.session_state.experiment_results = {
        'results': results,
        'config': config_name,
        'metadata': metadata
    }
    st.session_state.experiment_complete = True

