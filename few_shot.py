import random
import numpy as np
import pandas as pd
import math
from datetime import datetime
from typing import Dict, Any, List
from constants import CLASS_LABELS, DEFAULT_RANDOM_SEED, MAX_BIPHOBIA_EXAMPLES


def create_single_few_shot_pool(
    df: pd.DataFrame,
    n_examples: int,
    random_seed: int = DEFAULT_RANDOM_SEED
) -> Dict[str, Any]:
    """
    Create a single few-shot pool with n_examples total examples.
    
    Args:
        df: Training dataframe with label columns
        n_examples: Total number of examples (5-20)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict with 'examples', 'class_distribution', and 'metadata' keys
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    examples_per_class = math.ceil(n_examples / 5)
    
    class_data = {}
    for label in CLASS_LABELS:
        class_data[label] = df[df[label] == 1].copy()
    
    pool_examples = []
    class_distribution = {}
    
    for label in CLASS_LABELS:
        available = class_data[label]
        
        # For Biphobia, limit to max examples to preserve test set integrity
        if label == 'B':
            n_samples = min(examples_per_class, MAX_BIPHOBIA_EXAMPLES, len(available))
        else:
            n_samples = min(examples_per_class, len(available))
        
        if n_samples > 0:
            sampled = available.sample(n=n_samples, random_state=random_seed)
            
            for _, row in sampled.iterrows():
                example = {
                    'tweet_id': int(row['id']),
                    'tweet_text': row['tweet_text'],
                    'label_vector': row['label_vector'],
                    'labels': row['labels'],
                    'class_indices': row['class_indices']
                }
                pool_examples.append(example)
            
            class_distribution[label] = n_samples
        else:
            class_distribution[label] = 0
    
    # Shuffle the examples
    random.Random(random_seed).shuffle(pool_examples)
    
    # If we have more examples than requested, trim to n_examples
    if len(pool_examples) > n_examples:
        pool_examples = pool_examples[:n_examples]
    
    return {
        'examples': pool_examples,
        'class_distribution': class_distribution,
        'metadata': {
            'total': len(pool_examples),
            'created_at': datetime.now().isoformat(),
            'random_seed': random_seed,
            'n_examples': n_examples
        }
    }

