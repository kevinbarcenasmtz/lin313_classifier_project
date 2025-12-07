import pandas as pd
import streamlit as st
from typing import Dict, List, Any
from constants import CLASS_LABELS, EXPECTED_DATASET_SIZE
from exceptions import DataLoadingError, ValidationError
from logger_config import logger


@st.cache_data
def load_homo_mex_dataset(file_path: str = "data/Annotated LGBTQ+ Phobia Tweets.xlsx") -> pd.DataFrame:
    """
    Load HOMO-MEX dataset from Excel file.
    
    Validates required columns and data integrity. Handles column name
    variations (e.g., 'tuit' -> 'tweet_text').
    
    Args:
        file_path: Path to Excel file with dataset
        
    Returns:
        DataFrame with columns: id, tweet_text, G, L, B, T, O
        
    Raises:
        DataLoadingError: If required columns are missing or data is invalid
    """
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_excel(file_path)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    if 'tuit' in df.columns:
        df = df.rename(columns={'tuit': 'tweet_text'})
    
    required_cols = ['id', 'tweet_text', 'G', 'L', 'B', 'T', 'O']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise DataLoadingError(f"Missing required columns: {missing_cols}")
    
    if df['tweet_text'].isnull().any():
        raise DataLoadingError("Found missing tweet texts in the dataset")
    
    if df[['G', 'L', 'B', 'T', 'O']].isnull().any().any():
        raise DataLoadingError("Found missing labels in the dataset")
    
    return df[required_cols].copy()


def prepare_label_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare label vectors and formatted label strings from binary columns.
    
    Creates 'label_vector' (list of 5 integers), 'labels' (comma-separated string),
    and 'class_indices' (list of active class indices) columns.
    
    Args:
        df: DataFrame with G, L, B, T, O binary columns
        
    Returns:
        DataFrame with added 'label_vector', 'labels', and 'class_indices' columns
    """
    df = df.copy()
    df['label_vector'] = df[['G', 'L', 'B', 'T', 'O']].values.tolist()
    
    def format_labels(row):
        labels = []
        if row['G'] == 1:
            labels.append('Gayphobia')
        if row['L'] == 1:
            labels.append('Lesbophobia')
        if row['B'] == 1:
            labels.append('Biphobia')
        if row['T'] == 1:
            labels.append('Transphobia')
        if row['O'] == 1:
            labels.append('Other')
        return ', '.join(labels) if labels else 'None'
    
    df['labels'] = df.apply(format_labels, axis=1)
    
    def get_class_indices(row):
        indices = []
        for i, label in enumerate(CLASS_LABELS):
            if row[label] == 1:
                indices.append(i)
        return indices
    
    df['class_indices'] = df.apply(get_class_indices, axis=1)
    
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate dataset structure and label counts.
    
    Checks dataset size, label counts against expected values, and counts
    multi-label examples.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dict with 'valid', 'errors', 'warnings', and 'stats' keys
        - valid: Boolean indicating if dataset is valid
        - errors: List of error messages
        - warnings: List of warning messages (e.g., count mismatches)
        - stats: Dict with label_counts, expected_counts, multi_label_count, total_rows
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if len(df) != EXPECTED_DATASET_SIZE:
        warning_msg = f"Expected {EXPECTED_DATASET_SIZE} rows, found {len(df)}"
        results['warnings'].append(warning_msg)
        logger.warning(warning_msg)
    
    expected_counts = {'G': 714, 'L': 72, 'B': 10, 'T': 79, 'O': 64}
    actual_counts = {label: int(df[label].sum()) for label in CLASS_LABELS}
    
    results['stats']['label_counts'] = actual_counts
    results['stats']['expected_counts'] = expected_counts
    
    for label, expected in expected_counts.items():
        actual = actual_counts[label]
        if actual != expected:
            warning_msg = f"Label {label}: Expected {expected}, found {actual}"
            results['warnings'].append(warning_msg)
            logger.warning(warning_msg)
    
    multi_label_count = (df[CLASS_LABELS].sum(axis=1) > 1).sum()
    results['stats']['multi_label_count'] = int(multi_label_count)
    results['stats']['total_rows'] = len(df)
    
    logger.info(f"Dataset validation complete: {len(df)} rows, {multi_label_count} multi-label examples")
    if results['warnings']:
        logger.warning(f"Validation produced {len(results['warnings'])} warnings")
    
    return results

