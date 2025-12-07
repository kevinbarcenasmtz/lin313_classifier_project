import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.metrics import (
    multilabel_confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from constants import CLASS_LABELS, CLASS_NAMES

BERT_BASELINE = {
    'accuracy': 0.7815,
    'precision_macro': 0.9354,
    'recall_macro': 0.7815,
    'f1_macro': 0.7396,
    'per_class': {
        'G': {'precision': 0.95, 'recall': 0.82, 'f1': 0.88},
        'L': {'precision': 0.85, 'recall': 0.72, 'f1': 0.78},
        'B': {'precision': 0.42, 'recall': 0.40, 'f1': 0.42},
        'T': {'precision': 0.88, 'recall': 0.75, 'f1': 0.81},
        'O': {'precision': 0.90, 'recall': 0.68, 'f1': 0.78}
    }
}


def extract_predictions_from_results(results: List[Dict]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Extract true_labels and pred_labels from results list.
    
    Args:
        results: List of result dictionaries from classification experiment
        
    Returns:
        Tuple of (true_labels, pred_labels) as lists of lists
    """
    true_labels = []
    pred_labels = []
    
    for result in results:
        if result.get('is_valid', True):
            true_labels.append(result['true_labels'])
            pred_labels.append(result['pred_labels'])
    
    return true_labels, pred_labels


def calculate_per_class_metrics(
    true_labels: List[List[int]],
    pred_labels: List[List[int]]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, F1 for each class (G, L, B, T, O).
    
    Args:
        true_labels: List of true label vectors
        pred_labels: List of predicted label vectors
        
    Returns:
        Dict with class names as keys, metrics as values
    """
    if not true_labels or not pred_labels:
        return {label: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for label in CLASS_LABELS}
    
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    
    per_class_metrics = {}
    
    for idx, label in enumerate(CLASS_LABELS):
        true_binary = y_true[:, idx]
        pred_binary = y_pred[:, idx]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_binary, pred_binary, average='binary', zero_division=0
        )
        
        per_class_metrics[label] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    return per_class_metrics


def calculate_metrics(
    true_labels: List[List[int]],
    pred_labels: List[List[int]]
) -> Dict[str, float]:
    """
    Calculate per-class and overall metrics for multi-label classification.
    
    Args:
        true_labels: List of true label vectors
        pred_labels: List of predicted label vectors
        
    Returns:
        Dict with accuracy, precision (macro/micro), recall (macro/micro), F1 (macro/micro)
    """
    if not true_labels or not pred_labels:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'precision_micro': 0.0,
            'recall_macro': 0.0,
            'recall_micro': 0.0,
            'f1_macro': 0.0,
            'f1_micro': 0.0
        }
    
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'precision_micro': float(precision_micro),
        'recall_macro': float(recall_macro),
        'recall_micro': float(recall_micro),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro)
    }


def calculate_confusion_matrices(true_labels: List[List[int]], pred_labels: List[List[int]]) -> Dict[str, np.ndarray]:
    """
    For multi-label: Create 5 separate binary confusion matrices (one per class).
    
    Args:
        true_labels: List of true label vectors
        pred_labels: List of predicted label vectors
        
    Returns:
        Dict: {'G': matrix, 'L': matrix, 'B': matrix, 'T': matrix, 'O': matrix}
        Each matrix is 2x2: [TN, FP], [FN, TP]
    """
    if not true_labels or not pred_labels:
        return {label: np.array([[0, 0], [0, 0]]) for label in CLASS_LABELS}
    
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    
    result = {}
    for idx, label in enumerate(CLASS_LABELS):
        result[label] = confusion_matrices[idx]
    
    return result


def analyze_label_cooccurrence(
    true_labels: List[List[int]],
    pred_labels: List[List[int]]
) -> Dict[str, int]:
    """
    Analyze which label combinations are confused.
    
    Args:
        true_labels: List of true label vectors
        pred_labels: List of predicted label vectors
        
    Returns:
        Summary statistics about label co-occurrence errors
    """
    if not true_labels or not pred_labels:
        return {
            'correct_multi_label': 0,
            'missed_labels': 0,
            'extra_labels': 0,
            'exact_matches': 0,
            'total': 0
        }
    
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    
    exact_matches = 0
    correct_multi_label = 0
    missed_labels = 0
    extra_labels = 0
    
    for i in range(len(y_true)):
        true_vec = y_true[i]
        pred_vec = y_pred[i]
        
        if np.array_equal(true_vec, pred_vec):
            exact_matches += 1
            if np.sum(true_vec) > 1:
                correct_multi_label += 1
        else:
            true_sum = np.sum(true_vec)
            pred_sum = np.sum(pred_vec)
            
            if pred_sum < true_sum:
                missed_labels += 1
            elif pred_sum > true_sum:
                extra_labels += 1
    
    return {
        'correct_multi_label': correct_multi_label,
        'missed_labels': missed_labels,
        'extra_labels': extra_labels,
        'exact_matches': exact_matches,
        'total': len(y_true)
    }

