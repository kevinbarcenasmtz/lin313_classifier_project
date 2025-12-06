import pandas as pd
import streamlit as st
from constants import CLASS_LABELS


@st.cache_data
def load_homo_mex_dataset(file_path: str = "data/Annotated LGBTQ+ Phobia Tweets.xlsx") -> pd.DataFrame:
    df = pd.read_excel(file_path)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    if 'tuit' in df.columns:
        df = df.rename(columns={'tuit': 'tweet_text'})
    
    required_cols = ['id', 'tweet_text', 'G', 'L', 'B', 'T', 'O']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df['tweet_text'].isnull().any():
        raise ValueError("Found missing tweet texts in the dataset")
    
    if df[['G', 'L', 'B', 'T', 'O']].isnull().any().any():
        raise ValueError("Found missing labels in the dataset")
    
    return df[required_cols].copy()


def prepare_label_vectors(df: pd.DataFrame) -> pd.DataFrame:
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


def validate_dataset(df: pd.DataFrame) -> dict:
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if len(df) != 862:
        results['warnings'].append(f"Expected 862 rows, found {len(df)}")
    
    expected_counts = {'G': 714, 'L': 72, 'B': 10, 'T': 79, 'O': 64}
    actual_counts = {label: int(df[label].sum()) for label in CLASS_LABELS}
    
    results['stats']['label_counts'] = actual_counts
    results['stats']['expected_counts'] = expected_counts
    
    for label, expected in expected_counts.items():
        actual = actual_counts[label]
        if actual != expected:
            results['warnings'].append(
                f"Label {label}: Expected {expected}, found {actual}"
            )
    
    multi_label_count = (df[CLASS_LABELS].sum(axis=1) > 1).sum()
    results['stats']['multi_label_count'] = int(multi_label_count)
    results['stats']['total_rows'] = len(df)
    
    return results

