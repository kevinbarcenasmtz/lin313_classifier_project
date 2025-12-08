import time
import re
import openai
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split

from constants import (
    CLASS_NAMES,
    CLASS_LABELS,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_MAX_RETRIES
)
from api_utils import get_model_pricing
from exceptions import APIError
from logger_config import logger
from validation import (
    validate_tweet_text,
    validate_label_vector,
    validate_api_response,
    validate_experiment_config
)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets with stratified sampling.
    
    Attempts to maintain label distribution across splits. If stratification
    fails (e.g., rare label combinations), falls back to random split.
    
    Args:
        df: DataFrame with label columns (G, L, B, T, O)
        test_size: Fraction of data to use for testing (default: 0.35)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df) DataFrames
    """
    df = df.copy()
    df['stratify_key'] = df[CLASS_LABELS].apply(lambda x: ''.join(map(str, x)), axis=1)
    
    # Count occurrences of each stratify_key
    key_counts = df['stratify_key'].value_counts()
    
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
        logger.warning("Stratified split failed, falling back to random split")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed
        )
    
    train_df = train_df.drop(columns=['stratify_key']).reset_index(drop=True)
    test_df = test_df.drop(columns=['stratify_key']).reset_index(drop=True)
    
    return train_df, test_df


def format_pool_examples(pool: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Format few-shot pool examples for use in prompts.
    
    Extracts tweet text and labels from pool structure into a simpler
    format suitable for prompt construction.
    
    Args:
        pool: Dict with 'examples' key containing list of example dicts
        
    Returns:
        List of dicts with 'text' and 'labels' keys
    """
    formatted = []
    for example in pool['examples']:
        formatted.append({
            'text': example['tweet_text'],
            'labels': example['labels']
        })
    return formatted


def build_classification_prompt(
    few_shot_examples: List[Dict[str, str]],
    target_tweet: str
) -> List[Dict[str, str]]:
    """
    Build a classification prompt for the LLM.
    
    Constructs a prompt with system message, few-shot examples, and
    the target tweet to classify.
    
    Args:
        few_shot_examples: List of example dicts with 'text' and 'labels' keys
        target_tweet: The tweet text to classify
        
    Returns:
        List of message dicts for OpenAI API (system + user messages)
    """
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
    """
    Parse LLM response text into binary label vector.
    
    Extracts label names from LLM response and converts to binary vector
    [G, L, B, T, O]. Handles variations in label naming and "None" responses.
    
    Args:
        response_text: Raw text response from LLM
        
    Returns:
        Tuple of (binary_vector, labels_string, is_valid)
        - binary_vector: List of 5 integers [0 or 1 for each class]
        - labels_string: Comma-separated label names or "None"
        - is_valid: True if response was successfully parsed
    """
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
    client: Any,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Make OpenAI API call with exponential backoff retry logic.
    
    Retries on rate limit errors and API errors with exponential backoff.
    Returns response text and metadata including token usage.
    
    Args:
        client: OpenAI client instance
        messages: List of message dicts for the API
        model: Model name (e.g., 'gpt-4-turbo')
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens in response
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (response_text, metadata_dict)
        - response_text: Response content or None if failed
        - metadata_dict: Dict with 'success', 'attempt', token counts, 'error' if failed
    """
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
                logger.warning(f"Rate limit exceeded, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Rate limit exceeded after {max_retries} attempts")
                return None, {
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'attempt': attempt + 1
                }
        except openai.APIError as e:
            if attempt == max_retries - 1:
                logger.error(f"API error after {max_retries} attempts: {str(e)}")
                return None, {
                    'success': False,
                    'error': str(e),
                    'attempt': attempt + 1
                }
            wait_time = 2 ** attempt
            logger.warning(f"API error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Unexpected error after {max_retries} attempts: {str(e)}", exc_info=True)
                return None, {
                    'success': False,
                    'error': str(e),
                    'attempt': attempt + 1
                }
            wait_time = 2 ** attempt
            logger.warning(f"Unexpected error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(wait_time)
    
    # After all retries exhausted, return error info
    logger.error(f"Max retries ({max_retries}) exceeded for API call")
    return None, {
        'success': False,
        'error': 'Max retries exceeded',
        'attempt': max_retries
    }


def calculate_cost_so_far(
    api_calls: int,
    tokens_used: Dict[str, int],
    model_name: str
) -> float:
    """
    Calculate total API cost based on token usage.
    
    Args:
        api_calls: Number of API calls made (currently unused but kept for future use)
        tokens_used: Dict with 'input_tokens' and 'output_tokens' keys
        model_name: Model name for pricing lookup
        
    Returns:
        Total cost in USD
    """
    pricing = get_model_pricing(model_name)
    
    input_cost = (tokens_used.get('input_tokens', 0) / 1_000_000) * pricing['input']
    output_cost = (tokens_used.get('output_tokens', 0) / 1_000_000) * pricing['output']
    
    return input_cost + output_cost


def save_checkpoint(
    results: List[Dict[str, Any]],
    config_name: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Save experiment results to Streamlit session state.
    
    Args:
        results: List of result dicts from classification
        config_name: Configuration name (e.g., '15-shot')
        metadata: Dict with experiment metadata (cost, tokens, errors, etc.)
    """
    st.session_state.experiment_results = {
        'results': results,
        'config': config_name,
        'metadata': metadata
    }


def create_progress_display(progress_container: Any) -> Tuple[Any, Any, Any]:
    """
    Create empty containers for progress display.
    
    Args:
        progress_container: Streamlit container for progress UI
        
    Returns:
        Tuple of (progress_bar, metrics_container, eta_container) Streamlit widgets
    """
    with progress_container:
        progress_bar = st.empty()
        metrics_container = st.empty()
        eta_container = st.empty()
    return progress_bar, metrics_container, eta_container


def update_progress(
    progress_bar: Any,
    metrics_container: Any,
    eta_container: Any,
    current: int,
    total: int,
    cost_so_far: float,
    success_count: int,
    error_count: int,
    start_time: float
) -> None:
    """
    Update progress display in place.
    
    Only updates every 5 tweets to reduce UI overhead. Always updates on final tweet.
    
    Args:
        progress_bar: Streamlit progress bar widget
        metrics_container: Streamlit container for metrics
        eta_container: Streamlit container for ETA
        current: Current tweet number (1-indexed)
        total: Total number of tweets
        cost_so_far: Cumulative API cost
        success_count: Number of successful API calls
        error_count: Number of failed API calls
        start_time: Experiment start time (timestamp)
    """
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
    few_shot_examples: List[Dict[str, str]],
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    config_name: str,
    progress_container: Any
) -> None:
    """
    Run classification experiment with single few-shot configuration.
    
    Processes all test tweets, tracks progress, costs, and errors. Saves
    results to Streamlit session state. Creates checkpoints every 50 tweets.
    
    Args:
        test_df: Test dataframe with 'id', 'tweet_text', 'label_vector', 'labels' columns
        few_shot_examples: List of formatted few-shot examples with 'text' and 'labels' keys
        api_key: OpenAI API key
        model_name: Model name (e.g., 'gpt-4-turbo')
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens for response
        config_name: Configuration name (e.g., '15-shot')
        progress_container: Streamlit container for progress display
        
    Side Effects:
        Updates Streamlit session state with experiment results and metadata
    """
    # Store few-shot examples used in this experiment for later predictions
    st.session_state.few_shot_examples_used = few_shot_examples
    
    # Validate experiment configuration
    # Allow wider range for evaluation contexts (1-shot is valid for systematic evaluation)
    try:
        validate_experiment_config(
            model_name, 
            temperature, 
            max_tokens, 
            len(few_shot_examples),
            min_few_shot_examples=1,  # Allow 1-shot for evaluation
            max_few_shot_examples=50  # Allow higher for future experiments
        )
    except Exception as e:
        logger.error(f"Invalid experiment configuration: {str(e)}")
        raise
    
    logger.info(f"Starting classification experiment: {config_name}, {len(test_df)} test tweets")
    logger.info(f"Model: {model_name}, Temperature: {temperature}, Max tokens: {max_tokens}")
    
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
        # Validate tweet text
        try:
            validate_tweet_text(row['tweet_text'])
        except Exception as e:
            logger.warning(f"Invalid tweet text at index {idx}: {str(e)}")
            # Continue with invalid tweet but mark as error
            results.append({
                'tweet_id': int(row['id']),
                'tweet_text': row['tweet_text'],
                'true_labels': row['label_vector'],
                'true_labels_str': row['labels'],
                'pred_labels': [0, 0, 0, 0, 0],
                'pred_labels_str': "Validation Error",
                'is_valid': False,
                'raw_response': None,
                'error': str(e)
            })
            error_count += 1
            continue
        
        messages = build_classification_prompt(few_shot_examples, row['tweet_text'])
        
        response_text, api_metadata = make_api_call_with_retry(
            client, messages, model_name, temperature, max_tokens
        )
        
        total_api_calls += 1
        
        if api_metadata.get('success'):
            total_input_tokens += api_metadata.get('input_tokens', 0)
            total_output_tokens += api_metadata.get('output_tokens', 0)
            
            # Validate API response
            try:
                validate_api_response(response_text)
            except Exception as e:
                logger.warning(f"Invalid API response for tweet {row['id']}: {str(e)}")
                response_text = None
            
            if response_text:
                pred_vector, labels, is_valid = parse_llm_response(response_text)
            else:
                pred_vector, labels, is_valid = [0, 0, 0, 0, 0], "Validation Error", False
            
            # Validate predicted label vector
            try:
                validate_label_vector(pred_vector)
            except Exception as e:
                logger.warning(f"Invalid predicted label vector for tweet {row['id']}: {str(e)}")
                pred_vector = [0, 0, 0, 0, 0]
                is_valid = False
            
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
            if is_valid:
                success_count += 1
            else:
                error_count += 1
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
            logger.info(f"Checkpoint: {current_tweet}/{len(test_df)} tweets processed, cost: ${cost_so_far:.4f}")
            metadata = {
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'model': model_name,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'test_set_size': len(test_df),
                'few_shot_examples': len(few_shot_examples),
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
    elapsed_time = end_time - start_time
    total_cost = calculate_cost_so_far(total_api_calls, {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens
    }, model_name)
    
    logger.info(f"Experiment completed: {config_name}")
    logger.info(f"Total: {total_api_calls} API calls, {success_count} success, {error_count} errors")
    logger.info(f"Cost: ${total_cost:.4f}, Time: {elapsed_time:.1f}s")
    
    metadata = {
        'start_time': datetime.fromtimestamp(start_time).isoformat(),
        'end_time': datetime.fromtimestamp(end_time).isoformat(),
        'model': model_name,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'test_set_size': len(test_df),
        'few_shot_examples': len(few_shot_examples),
        'few_shot_examples': len(few_shot_examples),
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

