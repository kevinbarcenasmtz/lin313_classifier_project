"""
Input validation functions for user inputs, API responses, and data integrity.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from constants import (
    CLASS_LABELS,
    MIN_FEW_SHOT_EXAMPLES,
    MAX_FEW_SHOT_EXAMPLES,
    SUPPORTED_MODELS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS
)
from exceptions import ValidationError
from logger_config import logger


def validate_model_name(model_name: str) -> bool:
    """
    Validate that model name is supported.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        True if model is supported
        
    Raises:
        ValidationError: If model is not supported
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValidationError(
            f"Unsupported model: {model_name}. Supported models: {SUPPORTED_MODELS}"
        )
    return True


def validate_few_shot_examples(
    n_examples: int,
    min_examples: Optional[int] = None,
    max_examples: Optional[int] = None
) -> bool:
    """
    Validate number of few-shot examples is within allowed range.
    
    Args:
        n_examples: Number of few-shot examples
        min_examples: Optional minimum (defaults to MIN_FEW_SHOT_EXAMPLES)
        max_examples: Optional maximum (defaults to MAX_FEW_SHOT_EXAMPLES)
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If number is out of range
    """
    if not isinstance(n_examples, int):
        raise ValidationError(f"Few-shot examples must be an integer, got {type(n_examples)}")
    
    min_val = min_examples if min_examples is not None else MIN_FEW_SHOT_EXAMPLES
    max_val = max_examples if max_examples is not None else MAX_FEW_SHOT_EXAMPLES
    
    if n_examples < min_val or n_examples > max_val:
        raise ValidationError(
            f"Few-shot examples must be between {min_val} and {max_val}, got {n_examples}"
        )
    
    if n_examples < 1:
        raise ValidationError(f"Few-shot examples must be at least 1, got {n_examples}")
    
    return True


def validate_temperature(temperature: float) -> bool:
    """
    Validate temperature setting is within valid range.
    
    Args:
        temperature: Temperature value
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If temperature is out of range
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError(f"Temperature must be a number, got {type(temperature)}")
    if temperature < 0.0 or temperature > 2.0:
        raise ValidationError(f"Temperature must be between 0.0 and 2.0, got {temperature}")
    return True


def validate_max_tokens(max_tokens: int) -> bool:
    """
    Validate max tokens is within reasonable range.
    
    Args:
        max_tokens: Maximum tokens value
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If max_tokens is out of range
    """
    if not isinstance(max_tokens, int):
        raise ValidationError(f"Max tokens must be an integer, got {type(max_tokens)}")
    if max_tokens < 1 or max_tokens > 4096:
        raise ValidationError(f"Max tokens must be between 1 and 4096, got {max_tokens}")
    return True


def validate_tweet_text(tweet_text: str) -> bool:
    """
    Validate tweet text is not empty and within reasonable length.
    
    Args:
        tweet_text: Tweet text to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If tweet text is invalid
    """
    if not isinstance(tweet_text, str):
        raise ValidationError(f"Tweet text must be a string, got {type(tweet_text)}")
    if not tweet_text.strip():
        raise ValidationError("Tweet text cannot be empty")
    if len(tweet_text) > 10000:  # Reasonable upper limit
        raise ValidationError(f"Tweet text too long: {len(tweet_text)} characters (max 10000)")
    return True


def validate_label_vector(label_vector: List[int]) -> bool:
    """
    Validate label vector has correct format.
    
    Args:
        label_vector: List of 5 integers (0 or 1)
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If label vector is invalid
    """
    if not isinstance(label_vector, list):
        raise ValidationError(f"Label vector must be a list, got {type(label_vector)}")
    if len(label_vector) != len(CLASS_LABELS):
        raise ValidationError(
            f"Label vector must have {len(CLASS_LABELS)} elements, got {len(label_vector)}"
        )
    for i, val in enumerate(label_vector):
        if not isinstance(val, int):
            raise ValidationError(f"Label vector element {i} must be an integer, got {type(val)}")
        if val not in [0, 1]:
            raise ValidationError(f"Label vector element {i} must be 0 or 1, got {val}")
    return True


def validate_api_response(response_text: Optional[str]) -> bool:
    """
    Validate API response is not None and is a string.
    
    Args:
        response_text: API response text
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If response is invalid
    """
    if response_text is None:
        raise ValidationError("API response is None")
    if not isinstance(response_text, str):
        raise ValidationError(f"API response must be a string, got {type(response_text)}")
    return True


def validate_dataframe_structure(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame has required columns for classification.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If DataFrame structure is invalid
    """
    required_cols = ['id', 'tweet_text', 'G', 'L', 'B', 'T', 'O']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValidationError(f"DataFrame missing required columns: {missing_cols}")
    
    if len(df) == 0:
        raise ValidationError("DataFrame is empty")
    
    # Check for required columns in processed DataFrame
    processed_cols = ['label_vector', 'labels']
    missing_processed = [col for col in processed_cols if col not in df.columns]
    if missing_processed:
        logger.warning(f"DataFrame missing processed columns (may need prepare_label_vectors): {missing_processed}")
    
    return True


def validate_experiment_config(
    model_name: str,
    temperature: float,
    max_tokens: int,
    n_few_shot_examples: int,
    min_few_shot_examples: Optional[int] = None,
    max_few_shot_examples: Optional[int] = None
) -> bool:
    """
    Validate complete experiment configuration.
    
    Args:
        model_name: Model name
        temperature: Temperature setting
        max_tokens: Max tokens
        n_few_shot_examples: Number of few-shot examples
        min_few_shot_examples: Optional minimum for few-shot examples (for evaluation contexts)
        max_few_shot_examples: Optional maximum for few-shot examples (for evaluation contexts)
        
    Returns:
        True if all valid
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    validate_model_name(model_name)
    validate_temperature(temperature)
    validate_max_tokens(max_tokens)
    validate_few_shot_examples(
        n_few_shot_examples,
        min_examples=min_few_shot_examples,
        max_examples=max_few_shot_examples
    )
    return True

