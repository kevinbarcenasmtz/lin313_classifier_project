"""
Tests for metrics module.
"""

import pytest
import numpy as np
from metrics import (
    calculate_metrics,
    calculate_per_class_metrics,
    calculate_confusion_matrices,
    analyze_label_cooccurrence,
    extract_predictions_from_results
)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        true_labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
        pred_labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
        
        metrics = calculate_metrics(true_labels, pred_labels)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision_macro'] == 1.0
        assert metrics['recall_macro'] == 1.0
        assert metrics['f1_macro'] == 1.0
    
    def test_no_predictions(self):
        """Test metrics when no labels are predicted."""
        true_labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        pred_labels = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        
        metrics = calculate_metrics(true_labels, pred_labels)
        
        assert metrics['accuracy'] == 0.0
        assert metrics['recall_macro'] == 0.0
        assert metrics['f1_macro'] == 0.0
    
    def test_empty_inputs(self):
        """Test metrics with empty inputs."""
        metrics = calculate_metrics([], [])
        
        assert metrics['accuracy'] == 0.0
        assert metrics['precision_macro'] == 0.0
        assert metrics['recall_macro'] == 0.0
        assert metrics['f1_macro'] == 0.0
    
    def test_multi_label_predictions(self):
        """Test metrics with multi-label predictions."""
        true_labels = [[1, 0, 0, 1, 0], [0, 1, 0, 0, 0]]
        pred_labels = [[1, 0, 0, 1, 0], [0, 1, 0, 0, 0]]
        
        metrics = calculate_metrics(true_labels, pred_labels)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['f1_macro'] == 1.0


class TestCalculatePerClassMetrics:
    """Tests for calculate_per_class_metrics function."""
    
    def test_per_class_perfect(self):
        """Test per-class metrics with perfect predictions."""
        true_labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        pred_labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        
        per_class = calculate_per_class_metrics(true_labels, pred_labels)
        
        assert per_class['G']['precision'] == 1.0
        assert per_class['G']['recall'] == 1.0
        assert per_class['G']['f1'] == 1.0
        assert per_class['L']['precision'] == 1.0
    
    def test_per_class_empty(self):
        """Test per-class metrics with empty inputs."""
        per_class = calculate_per_class_metrics([], [])
        
        for label in ['G', 'L', 'B', 'T', 'O']:
            assert per_class[label]['precision'] == 0.0
            assert per_class[label]['recall'] == 0.0
            assert per_class[label]['f1'] == 0.0


class TestCalculateConfusionMatrices:
    """Tests for calculate_confusion_matrices function."""
    
    def test_confusion_matrix_structure(self):
        """Test that confusion matrices have correct structure."""
        true_labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
        pred_labels = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        
        matrices = calculate_confusion_matrices(true_labels, pred_labels)
        
        assert len(matrices) == 5
        for label in ['G', 'L', 'B', 'T', 'O']:
            assert label in matrices
            assert matrices[label].shape == (2, 2)
    
    def test_confusion_matrix_values(self):
        """Test confusion matrix values for a simple case."""
        true_labels = [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        pred_labels = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        
        matrices = calculate_confusion_matrices(true_labels, pred_labels)
        g_matrix = matrices['G']
        
        # TP=1, FN=1, TN=1, FP=0
        assert g_matrix[1, 1] == 1  # TP
        assert g_matrix[1, 0] == 1  # FN
        assert g_matrix[0, 0] == 1  # TN
        assert g_matrix[0, 1] == 0  # FP


class TestAnalyzeLabelCooccurrence:
    """Tests for analyze_label_cooccurrence function."""
    
    def test_exact_matches(self):
        """Test co-occurrence analysis with exact matches."""
        true_labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 1, 0]]
        pred_labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 1, 0]]
        
        analysis = analyze_label_cooccurrence(true_labels, pred_labels)
        
        assert analysis['exact_matches'] == 3
        assert analysis['correct_multi_label'] == 1
        assert analysis['missed_labels'] == 0
        assert analysis['extra_labels'] == 0
    
    def test_missed_labels(self):
        """Test co-occurrence analysis with missed labels."""
        true_labels = [[1, 0, 0, 1, 0], [0, 1, 0, 0, 0]]
        pred_labels = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        
        analysis = analyze_label_cooccurrence(true_labels, pred_labels)
        
        assert analysis['missed_labels'] == 2
        assert analysis['exact_matches'] == 0
    
    def test_extra_labels(self):
        """Test co-occurrence analysis with extra labels."""
        true_labels = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        pred_labels = [[1, 0, 0, 1, 0], [0, 1, 0, 0, 0]]
        
        analysis = analyze_label_cooccurrence(true_labels, pred_labels)
        
        assert analysis['extra_labels'] == 2
        assert analysis['exact_matches'] == 0


class TestExtractPredictionsFromResults:
    """Tests for extract_predictions_from_results function."""
    
    def test_extract_valid_results(self):
        """Test extracting predictions from valid results."""
        results = [
            {'true_labels': [1, 0, 0, 0, 0], 'pred_labels': [1, 0, 0, 0, 0], 'is_valid': True},
            {'true_labels': [0, 1, 0, 0, 0], 'pred_labels': [0, 1, 0, 0, 0], 'is_valid': True},
            {'true_labels': [0, 0, 0, 0, 0], 'pred_labels': [0, 0, 0, 0, 0], 'is_valid': False}
        ]
        
        true_labels, pred_labels = extract_predictions_from_results(results)
        
        assert len(true_labels) == 2
        assert len(pred_labels) == 2
        assert true_labels[0] == [1, 0, 0, 0, 0]
        assert pred_labels[0] == [1, 0, 0, 0, 0]
    
    def test_extract_empty_results(self):
        """Test extracting from empty results."""
        true_labels, pred_labels = extract_predictions_from_results([])
        
        assert len(true_labels) == 0
        assert len(pred_labels) == 0

