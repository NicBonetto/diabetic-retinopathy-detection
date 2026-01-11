import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metrics import (
    calculate_metrics,
    calculate_metrics_with_probs,
    print_metrics_report
)


class TestBasicMetrics:
    """Test suite for basic metric calculations."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision_macro'] == 1.0
        assert metrics['recall_macro'] == 1.0
        assert metrics['f1_macro'] == 1.0
        assert metrics['kappa'] == 1.0
    
    def test_all_wrong_predictions(self):
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.0
    
    def test_random_predictions(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 5, 100)
        y_pred = np.random.randint(0, 5, 100)
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Accuracy should be around 20% (1/5 classes)
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision_macro'] <= 1
        assert 0 <= metrics['recall_macro'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1
    
    def test_binary_classification(self):
        """Test metrics work with binary classification."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 'per_class' in metrics
        assert len(metrics['per_class']['precision']) == 2
    
    def test_multiclass_classification(self):
        """Test metrics with 5-class problem."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 2, 4, 1, 1, 2, 3, 3])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 0 <= metrics['accuracy'] <= 1
        assert len(metrics['per_class']['precision']) == 5
        assert len(metrics['per_class']['recall']) == 5
        assert len(metrics['per_class']['f1']) == 5


class TestPerClassMetrics:
    """Test suite for per-class metrics."""
    
    def test_per_class_precision(self):
        """Test per-class precision calculation."""
        # Class 0: 2/2 correct (precision = 1.0)
        # Class 1: 1/2 correct (precision = 0.5)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        precisions = metrics['per_class']['precision']
        assert abs(precisions[0] - 2/3) < 1e-6  # Class 0 perfect
        assert precisions[1] == 1.0  # Class 1 (1 TP, 0 FP)
    
    def test_per_class_recall(self):
        """Test per-class recall calculation."""
        # Class 0: 2/2 recalled (recall = 1.0)
        # Class 1: 1/2 recalled (recall = 0.5)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        recalls = metrics['per_class']['recall']
        assert recalls[0] == 1.0  # Class 0 perfect
        assert recalls[1] == 0.5  # Class 1 (1 TP, 1 FN)
    
    def test_per_class_support(self):
        """Test per-class support (number of samples)."""
        y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        support = metrics['per_class']['support']
        assert support[0] == 3  # 3 samples of class 0
        assert support[1] == 2  # 2 samples of class 1
        assert support[2] == 4  # 4 samples of class 2


class TestClinicalMetrics:
    """Test suite for clinical metrics (sensitivity, specificity)."""
    
    def test_sensitivity_calculation(self):
        """Test sensitivity (recall) for each class."""
        # Perfect predictions
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        sensitivities = metrics['per_class']['sensitivity']
        # All classes perfectly predicted
        assert all(s == 1.0 for s in sensitivities)
    
    def test_specificity_calculation(self):
        """Test specificity for each class."""
        # Perfect predictions
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        specificities = metrics['per_class']['specificity']
        # All classes perfectly predicted
        assert all(s == 1.0 for s in specificities)
    
    def test_sensitivity_vs_recall(self):
        """Test that sensitivity equals recall."""
        y_true = np.array([0, 1, 2, 3, 4] * 10)
        y_pred = np.random.randint(0, 5, 50)
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Sensitivity should equal recall for each class
        sensitivities = metrics['per_class']['sensitivity']
        recalls = metrics['per_class']['recall']
        
        for sens, rec in zip(sensitivities, recalls):
            assert abs(sens - rec) < 1e-10
    
    def test_cohens_kappa(self):
        """Test Cohen's kappa calculation."""
        # Perfect agreement
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['kappa'] == 1.0
        
        # No agreement (all wrong)
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['kappa'] <= 0  # Worse than random

        # Systematically wrong
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])

        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['kappa'] < 0


class TestConfusionMatrix:
    """Test suite for confusion matrix."""
    
    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        cm = np.array(metrics['confusion_matrix'])
        
        # Should be 3x3 for 3 classes
        assert cm.shape == (3, 3)
    
    def test_confusion_matrix_diagonal_perfect(self):
        """Test confusion matrix diagonal for perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        cm = np.array(metrics['confusion_matrix'])
        
        # Diagonal should have all samples
        assert cm[0, 0] == 2  # 2 samples of class 0
        assert cm[1, 1] == 2  # 2 samples of class 1
        assert cm[2, 2] == 2  # 2 samples of class 2
        
        # Off-diagonal should be zero
        assert np.sum(cm) - np.trace(cm) == 0
    
    def test_confusion_matrix_misclassifications(self):
        """Test confusion matrix captures misclassifications."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        
        metrics = calculate_metrics(y_true, y_pred)
        cm = np.array(metrics['confusion_matrix'])
        
        # cm[true_class, pred_class]
        assert cm[0, 0] == 1  # One 0 predicted as 0
        assert cm[0, 1] == 1  # One 0 predicted as 1
        assert cm[1, 1] == 1  # One 1 predicted as 1
        assert cm[1, 0] == 1  # One 1 predicted as 0


class TestMetricsWithProbabilities:
    """Test suite for metrics that use probability scores."""
    
    def test_auc_calculation(self):
        """Test AUC calculation with probabilities."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        # Perfect probability predictions
        y_probs = np.array([
            [1.0, 0.0, 0.0],  # Confident class 0
            [1.0, 0.0, 0.0],  # Confident class 0
            [0.0, 1.0, 0.0],  # Confident class 1
            [0.0, 1.0, 0.0],  # Confident class 1
            [0.0, 0.0, 1.0],  # Confident class 2
            [0.0, 0.0, 1.0],  # Confident class 2
        ])
        
        metrics = calculate_metrics_with_probs(y_true, y_pred, y_probs)
        
        # Perfect predictions should have AUC = 1.0
        assert 'auc_macro' in metrics
        assert metrics['auc_macro'] == 1.0
        assert all(auc == 1.0 for auc in metrics['per_class']['auc'])
    
    def test_auc_with_uncertainty(self):
        """Test AUC with uncertain predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        # Uncertain probabilities
        y_probs = np.array([
            [0.6, 0.4],  # Somewhat confident class 0
            [0.3, 0.7],  # Somewhat confident class 1
            [0.8, 0.2],  # More confident class 0
            [0.4, 0.6],  # Less confident class 1
        ])
        
        metrics = calculate_metrics_with_probs(y_true, y_pred, y_probs)
        
        # AUC should still be 1.0 (all predictions correct)
        assert metrics['auc_macro'] == 1.0
    
    def test_auc_with_five_classes(self):
        """Test AUC with 5-class DR problem."""
        np.random.seed(42)
        n_samples = 50
        y_true = np.random.randint(0, 5, n_samples)
        y_pred = y_true  # Perfect predictions
        
        # Create perfect probability matrix
        y_probs = np.zeros((n_samples, 5))
        for i, label in enumerate(y_true):
            y_probs[i, label] = 1.0
        
        metrics = calculate_metrics_with_probs(y_true, y_pred, y_probs)
        
        assert 'auc_macro' in metrics
        assert metrics['auc_macro'] == 1.0
        assert len(metrics['per_class']['auc']) == 5


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_class_predictions(self):
        """Test with predictions all from one class."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 0, 0, 0, 0])  # All predicted as class 0
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Should complete without error
        assert metrics['accuracy'] == 0.2  # 1/5 correct
        assert 'per_class' in metrics
    
    def test_missing_classes_in_predictions(self):
        """Test when some classes never predicted."""
        y_true = np.array([0, 1, 2, 3, 4] * 2)
        y_pred = np.array([0, 0, 0, 0, 0] * 2)  # Only class 0 predicted
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Should handle gracefully (some metrics may be 0 or NaN)
        assert 'per_class' in metrics
        assert len(metrics['per_class']['precision']) == 5
    
    def test_empty_predictions(self):
        """Test with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        
        # This might raise an error or return special values
        # Both are acceptable behaviors
        try:
            metrics = calculate_metrics(y_true, y_pred)
            # If it doesn't crash, check that it returns something
            assert metrics is not None
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise error for empty input
            assert True
    
    def test_mismatched_array_lengths(self):
        """Test with mismatched y_true and y_pred lengths."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])  # Different length
        
        # Should raise an error
        with pytest.raises((ValueError, IndexError)):
            calculate_metrics(y_true, y_pred)
    
    def test_single_sample(self):
        """Test with just one sample."""
        y_true = np.array([0])
        y_pred = np.array([0])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert 'per_class' in metrics
    
    def test_class_with_zero_support(self):
        """Test when a class has zero samples in y_true."""
        # Only classes 0 and 1 present
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Should only have metrics for present classes
        assert len(metrics['per_class']['precision']) == 2


class TestMetricsConsistency:
    """Test internal consistency of metrics."""
    
    def test_macro_vs_per_class_consistency(self):
        """Test that macro metrics match average of per-class metrics."""
        y_true = np.array([0, 1, 2, 0, 1, 2] * 10)
        y_pred = np.array([0, 1, 2, 0, 1, 2] * 10)
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Macro should be average of per-class
        per_class_precision = metrics['per_class']['precision']
        assert abs(metrics['precision_macro'] - np.mean(per_class_precision)) < 1e-10
        
        per_class_recall = metrics['per_class']['recall']
        assert abs(metrics['recall_macro'] - np.mean(per_class_recall)) < 1e-10
    
    def test_weighted_vs_accuracy_consistency(self):
        """Test relationship between weighted metrics and accuracy."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # For perfect predictions, weighted recall should equal accuracy
        assert abs(metrics['recall_weighted'] - metrics['accuracy']) < 1e-10
    
    def test_precision_recall_f1_relationship(self):
        """Test that F1 is harmonic mean of precision and recall."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Check for each class
        for i in range(3):
            p = metrics['per_class']['precision'][i]
            r = metrics['per_class']['recall'][i]
            f1 = metrics['per_class']['f1'][i]
            
            if p + r > 0:
                expected_f1 = 2 * (p * r) / (p + r)
                assert abs(f1 - expected_f1) < 1e-10


class TestMetricsReporting:
    """Test metrics reporting functionality."""
    
    def test_print_metrics_report_executes(self, capsys):
        """Test that print_metrics_report runs without error."""
        y_true = np.array([0, 1, 2, 3, 4] * 5)
        y_pred = np.array([0, 1, 2, 3, 4] * 5)
        
        metrics = calculate_metrics(y_true, y_pred)
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        
        # Should not raise an error
        print_metrics_report(metrics, class_names)
        
        # Check that something was printed
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert 'CLASSIFICATION METRICS REPORT' in captured.out
    
    def test_print_metrics_with_default_names(self, capsys):
        """Test print_metrics_report with default class names."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Should use default names
        print_metrics_report(metrics)
        
        captured = capsys.readouterr()
        assert 'Class 0' in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
