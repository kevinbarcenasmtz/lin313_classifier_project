"""
Systematic shot count evaluation module.

Provides functionality to run experiments across multiple shot counts
with multiple trials, aggregate results, and generate comparison tables.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json

from constants import (
    DEFAULT_RANDOM_SEED,
    EVALUATION_SHOT_COUNTS,
    DEFAULT_NUM_TRIALS,
    EVALUATION_RANDOM_SEEDS
)
from few_shot import create_single_few_shot_pool
from classification import (
    format_pool_examples,
    run_classification_experiment
)
from metrics import (
    extract_predictions_from_results,
    calculate_metrics,
    calculate_per_class_metrics
)
from logger_config import logger

# Import classification functions
from classification import (
    build_classification_prompt,
    make_api_call_with_retry,
    parse_llm_response,
    calculate_cost_so_far
)
from validation import (
    validate_tweet_text,
    validate_api_response,
    validate_label_vector,
    validate_experiment_config
)
import openai
import time


@dataclass
class ShotCountConfig:
    """Configuration for a single shot count experiment."""
    shot_count: int
    trial_number: int
    random_seed: int
    config_name: str = field(init=False)
    
    def __post_init__(self):
        self.config_name = f"{self.shot_count}-shot-trial{self.trial_number}"


@dataclass
class EvaluationResults:
    """Results from systematic shot count evaluation."""
    configs: List[ShotCountConfig]
    results: Dict[str, List[Dict[str, Any]]]  # {shot_count: [trial1_results, trial2_results]}
    aggregated_metrics: pd.DataFrame
    total_cost: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            'configs': [
                {
                    'shot_count': c.shot_count,
                    'trial_number': c.trial_number,
                    'random_seed': c.random_seed,
                    'config_name': c.config_name
                }
                for c in self.configs
            ],
            'results': {
                str(k): v for k, v in self.results.items()
            },
            'aggregated_metrics': self.aggregated_metrics.to_dict('records'),
            'total_cost': self.total_cost,
            'metadata': self.metadata
        }


def run_shot_count_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    shot_counts: List[int] = None,
    num_trials: int = None,
    api_key: str = None,
    model_name: str = "gpt-4-turbo",
    temperature: float = 0.1,
    max_tokens: int = 50,
    random_seeds: List[int] = None,
    progress_callback: Optional[Callable] = None
) -> EvaluationResults:
    """
    Run systematic evaluation across multiple shot counts with multiple trials.
    
    Args:
        train_df: Training dataframe for creating few-shot pools
        test_df: Test dataframe for evaluation
        shot_counts: List of shot counts to evaluate (default: EVALUATION_SHOT_COUNTS)
        num_trials: Number of trials per shot count (default: DEFAULT_NUM_TRIALS)
        api_key: OpenAI API key
        model_name: Model name to use
        temperature: Temperature setting
        max_tokens: Max tokens for responses
        random_seeds: List of random seeds for trials (default: EVALUATION_RANDOM_SEEDS)
        progress_callback: Optional callback function(current, total, message)
        
    Returns:
        EvaluationResults with aggregated metrics and all trial results
    """
    if shot_counts is None:
        shot_counts = EVALUATION_SHOT_COUNTS
    if num_trials is None:
        num_trials = DEFAULT_NUM_TRIALS
    if random_seeds is None:
        random_seeds = EVALUATION_RANDOM_SEEDS[:num_trials]
    
    if len(random_seeds) < num_trials:
        # Generate additional seeds if needed
        random_seeds = random_seeds + [
            DEFAULT_RANDOM_SEED + i for i in range(len(random_seeds), num_trials)
        ]
    
    logger.info(f"Starting systematic evaluation: {len(shot_counts)} shot counts, {num_trials} trials each")
    logger.info(f"Shot counts: {shot_counts}, Random seeds: {random_seeds}")
    
    configs = []
    all_results = {}
    total_cost = 0.0
    
    total_experiments = len(shot_counts) * num_trials
    current_experiment = 0
    
    for shot_count in shot_counts:
        shot_results = []
        
        for trial in range(num_trials):
            current_experiment += 1
            random_seed = random_seeds[trial]
            
            config = ShotCountConfig(
                shot_count=shot_count,
                trial_number=trial + 1,
                random_seed=random_seed
            )
            configs.append(config)
            
            if progress_callback:
                progress_callback(
                    current_experiment,
                    total_experiments,
                    f"Running {config.config_name}..."
                )
            
            logger.info(f"Running experiment: {config.config_name} (seed: {random_seed})")
            
            # Create few-shot pool for this configuration
            pool = create_single_few_shot_pool(
                train_df,
                shot_count,
                random_seed=random_seed
            )
            few_shot_examples = format_pool_examples(pool)
            
            # Run experiment using standalone classification function
            try:
                # Check if we're in Streamlit context
                try:
                    import streamlit as st
                    in_streamlit = True
                except ImportError:
                    in_streamlit = False
                
                if in_streamlit:
                    # Use Streamlit version with progress container
                    progress_container = st.empty()
                    from classification import run_classification_experiment
                    run_classification_experiment(
                        test_df=test_df,
                        few_shot_examples=few_shot_examples,
                        api_key=api_key,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        config_name=config.config_name,
                        progress_container=progress_container
                    )
                    # Extract from session state
                    if 'experiment_results' in st.session_state:
                        experiment_result = {
                            'results': st.session_state.experiment_results.get('results', []),
                            'metadata': st.session_state.experiment_results.get('metadata', {})
                        }
                    else:
                        raise ValueError("No results in session state")
                else:
                    # Use standalone version
                    experiment_result = run_classification_experiment_standalone(
                        test_df=test_df,
                        few_shot_examples=few_shot_examples,
                        api_key=api_key,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        config_name=config.config_name
                    )
                
                trial_result = {
                    'config': config.config_name,
                    'shot_count': shot_count,
                    'trial': trial + 1,
                    'random_seed': random_seed,
                    'results': experiment_result.get('results', []),
                    'metadata': experiment_result.get('metadata', {}),
                    'metrics': None  # Will compute below
                }
                
                # Calculate metrics for this trial
                results_list = trial_result['results']
                if results_list:
                    true_labels, pred_labels = extract_predictions_from_results(results_list)
                    metrics = calculate_metrics(true_labels, pred_labels)
                    per_class = calculate_per_class_metrics(true_labels, pred_labels)
                    metrics['per_class_metrics'] = per_class
                    trial_result['metrics'] = metrics
                    
                    # Track cost
                    trial_cost = trial_result['metadata'].get('total_cost', 0)
                    total_cost += trial_cost
                
                shot_results.append(trial_result)
                    
            except Exception as e:
                logger.error(f"Error running experiment {config.config_name}: {str(e)}", exc_info=True)
                shot_results.append({
                    'config': config.config_name,
                    'shot_count': shot_count,
                    'trial': trial + 1,
                    'random_seed': random_seed,
                    'error': str(e),
                    'metrics': None
                })
        
        all_results[str(shot_count)] = shot_results
    
    # Aggregate metrics across trials
    aggregated_df = aggregate_metrics(all_results)
    
    metadata = {
        'start_time': datetime.now().isoformat(),
        'shot_counts': shot_counts,
        'num_trials': num_trials,
        'random_seeds': random_seeds,
        'model_name': model_name,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'test_set_size': len(test_df),
        'total_experiments': total_experiments,
        'total_cost': total_cost
    }
    
    return EvaluationResults(
        configs=configs,
        results=all_results,
        aggregated_metrics=aggregated_df,
        total_cost=total_cost,
        metadata=metadata
    )


def aggregate_metrics(results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Aggregate metrics across trials for each shot count.
    
    Creates a table matching assignment format:
    | Shot Count | Accuracy Run 1 | F1 Run 1 | Accuracy Run 2 | F1 Run 2 | Avg Accuracy | Avg F1 |
    
    Args:
        results: Dict mapping shot_count (str) to list of trial results
        
    Returns:
        DataFrame with aggregated metrics
    """
    rows = []
    
    for shot_count_str in sorted(results.keys(), key=int):
        shot_count = int(shot_count_str)
        trials = results[shot_count_str]
        
        # Extract metrics for each trial
        trial_metrics = []
        for trial in trials:
            if trial.get('metrics'):
                trial_metrics.append(trial['metrics'])
        
        if not trial_metrics:
            # No valid metrics, create empty row
            rows.append({
                'Shot Count': shot_count,
                'Accuracy Run 1': 'N/A',
                'F1 Run 1': 'N/A',
                'Accuracy Run 2': 'N/A',
                'F1 Run 2': 'N/A',
                'Avg Accuracy': 'N/A',
                'Avg F1': 'N/A'
            })
            continue
        
        # Fill in individual trial metrics
        accuracy_run1 = trial_metrics[0].get('accuracy', 0) * 100 if len(trial_metrics) > 0 else None
        f1_run1 = trial_metrics[0].get('f1_macro', 0) * 100 if len(trial_metrics) > 0 else None
        
        accuracy_run2 = trial_metrics[1].get('accuracy', 0) * 100 if len(trial_metrics) > 1 else None
        f1_run2 = trial_metrics[1].get('f1_macro', 0) * 100 if len(trial_metrics) > 1 else None
        
        # Calculate averages
        avg_accuracy = np.mean([m.get('accuracy', 0) * 100 for m in trial_metrics]) if trial_metrics else None
        avg_f1 = np.mean([m.get('f1_macro', 0) * 100 for m in trial_metrics]) if trial_metrics else None
        
        rows.append({
            'Shot Count': shot_count,
            'Accuracy Run 1': f"{accuracy_run1:.2f}%" if accuracy_run1 is not None else 'N/A',
            'F1 Run 1': f"{f1_run1:.2f}%" if f1_run1 is not None else 'N/A',
            'Accuracy Run 2': f"{accuracy_run2:.2f}%" if accuracy_run2 is not None else 'N/A',
            'F1 Run 2': f"{f1_run2:.2f}%" if f1_run2 is not None else 'N/A',
            'Avg Accuracy': f"{avg_accuracy:.2f}%" if avg_accuracy is not None else 'N/A',
            'Avg F1': f"{avg_f1:.2f}%" if avg_f1 is not None else 'N/A'
        })
    
    return pd.DataFrame(rows)


def save_evaluation_results(results: EvaluationResults, filepath: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: EvaluationResults to save
        filepath: Path to save JSON file
    """
    data = results.to_dict()
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved evaluation results to {filepath}")


def run_classification_experiment_standalone(
    test_df: pd.DataFrame,
    few_shot_examples: List[Dict[str, str]],
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    config_name: str
) -> Dict[str, Any]:
    """
    Run classification experiment standalone (without Streamlit dependencies).
    
    Returns results directly instead of storing in session state.
    
    Args:
        test_df: Test dataframe
        few_shot_examples: List of formatted few-shot examples
        api_key: OpenAI API key
        model_name: Model name
        temperature: Temperature setting
        max_tokens: Max tokens for response
        config_name: Configuration name
        
    Returns:
        Dict with 'results' and 'metadata' keys
    """
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
    
    results = []
    client = openai.OpenAI(api_key=api_key)
    
    start_time = time.time()
    total_api_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    errors = []
    success_count = 0
    error_count = 0
    
    for idx, row in test_df.iterrows():
        # Validate tweet text
        try:
            validate_tweet_text(row['tweet_text'])
        except Exception as e:
            logger.warning(f"Invalid tweet text at index {idx}: {str(e)}")
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
        
        # Log progress every 50 tweets
        if (idx + 1) % 50 == 0:
            tokens_used = {
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens
            }
            cost_so_far = calculate_cost_so_far(total_api_calls, tokens_used, model_name)
            logger.info(f"Checkpoint: {idx + 1}/{len(test_df)} tweets processed, cost: ${cost_so_far:.4f}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens_used = {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens
    }
    total_cost = calculate_cost_so_far(total_api_calls, tokens_used, model_name)
    
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
        'total_api_calls': total_api_calls,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_cost': total_cost,
        'errors': errors
    }
    
    return {
        'results': results,
        'metadata': metadata
    }


def load_evaluation_results(filepath: str) -> EvaluationResults:
    """
    Load evaluation results from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        EvaluationResults object
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Reconstruct configs
    configs = [
        ShotCountConfig(
            shot_count=c['shot_count'],
            trial_number=c['trial_number'],
            random_seed=c['random_seed']
        )
        for c in data['configs']
    ]
    
    # Reconstruct aggregated metrics DataFrame
    aggregated_df = pd.DataFrame(data['aggregated_metrics'])
    
    return EvaluationResults(
        configs=configs,
        results=data['results'],
        aggregated_metrics=aggregated_df,
        total_cost=data['total_cost'],
        metadata=data['metadata']
    )

