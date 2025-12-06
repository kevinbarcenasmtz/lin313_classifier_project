import random
import numpy as np
import pandas as pd
import math
from datetime import datetime
from typing import Dict
from constants import CLASS_LABELS


def create_single_few_shot_pool(df: pd.DataFrame, n_examples: int, random_seed: int = 42) -> Dict:
    """
    Create a single few-shot pool with n_examples total examples.
    
    Args:
        df: Training dataframe with label columns
        n_examples: Total number of examples (5-20)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict with same structure as pools from create_few_shot_pools()
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
        
        # For Biphobia, limit to max 3 examples to preserve test set integrity
        if label == 'B':
            n_samples = min(examples_per_class, 3, len(available))
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


def create_few_shot_pools(df: pd.DataFrame, random_seed: int = 42) -> Dict:
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    pools = {}
    configs = [
        ('5_shot', 1),
        ('10_shot', 2),
        ('15_shot', 3),
        ('20_shot', 4)
    ]
    
    class_data = {}
    for label in CLASS_LABELS:
        class_data[label] = df[df[label] == 1].copy()
    
    for pool_idx, (pool_name, examples_per_class) in enumerate(configs):
        pool_examples = []
        class_distribution = {}
        pool_random_state = random_seed + pool_idx * 1000
        
        for label in CLASS_LABELS:
            available = class_data[label]
            
            if label == 'B':
                n_samples = min(examples_per_class, 3, len(available))
            else:
                n_samples = min(examples_per_class, len(available))
            
            if n_samples > 0:
                sampled = available.sample(n=n_samples, random_state=pool_random_state)
                
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
        
        random.Random(pool_random_state).shuffle(pool_examples)
        
        pools[pool_name] = {
            'examples': pool_examples,
            'class_distribution': class_distribution,
            'metadata': {
                'total': len(pool_examples),
                'created_at': datetime.now().isoformat(),
                'random_seed': random_seed
            }
        }
    
    return pools

