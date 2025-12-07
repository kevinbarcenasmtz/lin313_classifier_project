"""
Constants used throughout the application.
"""

# Classification labels
CLASS_NAMES = {
    0: 'Gayphobia',
    1: 'Lesbophobia',
    2: 'Biphobia',
    3: 'Transphobia',
    4: 'Other'
}

CLASS_LABELS = ['G', 'L', 'B', 'T', 'O']

# Dataset configuration
DEFAULT_TEST_SIZE = 0.35  # Fraction of data to use for testing
DEFAULT_RANDOM_SEED = 42  # Random seed for reproducibility
EXPECTED_DATASET_SIZE = 862  # Expected number of training tweets in HOMO-MEX dataset

# Few-shot learning configuration
DEFAULT_FEW_SHOT_EXAMPLES = 15  # Default number of few-shot examples
MIN_FEW_SHOT_EXAMPLES = 5  # Minimum number of few-shot examples
MAX_FEW_SHOT_EXAMPLES = 20  # Maximum number of few-shot examples
MAX_BIPHOBIA_EXAMPLES = 3  # Maximum examples to use from Biphobia class (to preserve test set)

# API configuration
DEFAULT_MAX_RETRIES = 3  # Maximum number of API retry attempts
DEFAULT_TEMPERATURE = 0.1  # Default temperature for LLM (lower = more consistent)
DEFAULT_MAX_TOKENS = 50  # Default max tokens for LLM responses

# Model options
SUPPORTED_MODELS = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4.1-nano"]
DEFAULT_MODEL = "gpt-4-turbo"

# Cost estimation defaults
AVG_TWEET_LENGTH_TOKENS = 100  # Average tweet length in tokens
AVG_OUTPUT_TOKENS = 20  # Average output tokens per response
SYSTEM_TOKENS = 50  # System message tokens
INSTRUCTION_TOKENS = 100  # Instruction tokens in prompt

# Systematic evaluation configuration
EVALUATION_SHOT_COUNTS = [1, 5, 10, 15, 20]  # Shot counts to evaluate
DEFAULT_NUM_TRIALS = 2  # Default number of trials per shot count
EVALUATION_RANDOM_SEEDS = [42, 123]  # Random seeds for trials (one per trial)

