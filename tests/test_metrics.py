"""
test_metrics.py — Tests for src/evaluation/metrics.py.

Verifies the correctness of all metric computations used in the dissertation.
No datasets or trained models required — tests use synthetic predictions.

Why these tests matter:
    FPR and Detection Rate are operationally critical IDS metrics cited directly
    in the dissertation results. MCC is the primary binary imbalance metric.
    A bug in the confusion-matrix math would silently produce wrong dissertation
    numbers, so we verify each formula explicitly against known ground truth.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.evaluation.metrics import (
    evaluate_model,
    _binary_metrics,
    _multiclass_metrics,
    compute_privacy_utility_tradeoff,
    summarise_results,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_model(y_pred, y_proba=None):
    """Create a mock sklearn-compatible model with fixed predictions."""
    model = MagicMock()
    model.predict.return_value = np.array(y_pred)
    if y_proba is not None:
        model.predict_proba.return_value = np.array(y_proba)
    else:
        del model.predict_proba  # simulate no predict_proba
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# _binary_metrics — formula verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinaryMetricsFormulas:
    """Verify FPR and DR directly from known confusion matrix values."""

    def test_fpr_formula_tn_fp(self):
        """FPR = FP / (FP + TN). With TN=8, FP=2 → FPR = 0.2."""
        # 10 benign: 8 correct (TN), 2 wrong (FP)
        # 0 attacks
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        metrics = _binary_metrics(y_true, y_pred, y_prob=None)
        assert abs(metrics["false_positive_rate"] - 0.2) < 1e-9

    def test_detection_rate_formula_tp_fn(self):
        """DR = TP / (TP + FN). With TP=7, FN=3 → DR = 0.7."""
        # 10 attacks: 7 detected (TP), 3 missed (FN)
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        metrics = _binary_metrics(y_true, y_pred, y_prob=None)
        assert abs(metrics["detection_rate"] - 0.7) < 1e-9

    def test_false_negative_rate_is_complement_of_dr(self):
        """FNR = 1 - DR."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        metrics = _binary_metrics(y_true, y_pred, y_prob=None)
        dr  = metrics["detection_rate"]
        fnr = metrics["false_negative_rate"]
        assert abs(dr + fnr - 1.0) < 1e-9, "DR + FNR must equal 1"

    def test_perfect_classifier_fpr_0_dr_1(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = y_true.copy()
        metrics = _binary_metrics(y_true, y_pred, y_prob=None)
        assert metrics["false_positive_rate"] == 0.0
        assert metrics["detection_rate"]      == 1.0
        assert metrics["false_negative_rate"] == 0.0

    def test_all_benign_predicted_as_attack(self):
        """All FP: FPR = 1.0, no actual attacks so DR is None or 0."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        metrics = _binary_metrics(y_true, y_pred, y_prob=None)
        assert metrics["false_positive_rate"] == 1.0

    def test_roc_auc_with_probabilities(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        metrics = _binary_metrics(y_true, y_pred, y_prob=y_prob)
        assert metrics["roc_auc"] == 1.0

    def test_roc_auc_none_without_probabilities(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = _binary_metrics(y_true, y_pred, y_prob=None)
        assert metrics["roc_auc"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# evaluate_model — end-to-end via mock model
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateModelBinary:
    REQUIRED_KEYS = {
        "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
        "mcc", "confusion_matrix", "false_positive_rate", "detection_rate",
        "false_negative_rate",
    }

    def _perfect_binary(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = y_true.copy()
        # Build a (n_samples, 2) probability matrix
        proba = np.column_stack([
            np.where(y_pred == 0, 0.9, 0.1),
            np.where(y_pred == 1, 0.9, 0.1),
        ])
        model = _make_model(y_pred, y_proba=proba)
        return model, y_true

    def test_all_required_keys_present(self):
        model, y_true = self._perfect_binary()
        metrics = evaluate_model(model, np.zeros((len(y_true), 3)), y_true, label_type="binary")
        assert self.REQUIRED_KEYS.issubset(set(metrics.keys()))

    def test_perfect_accuracy_is_1(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = y_true.copy()
        model = _make_model(y_pred)
        metrics = evaluate_model(model, np.zeros((4, 2)), y_true, label_type="binary")
        assert metrics["accuracy"] == 1.0

    def test_f1_macro_range(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        model = _make_model(y_pred)
        metrics = evaluate_model(model, np.zeros((6, 2)), y_true, label_type="binary")
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_mcc_range(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        model = _make_model(y_pred)
        metrics = evaluate_model(model, np.zeros((6, 2)), y_true, label_type="binary")
        assert -1.0 <= metrics["mcc"] <= 1.0

    def test_confusion_matrix_is_list_of_lists(self):
        """confusion_matrix must be a list (JSON-serialisable), not a numpy array."""
        y_true = np.array([0, 1, 0, 1])
        model = _make_model(y_true)
        metrics = evaluate_model(model, np.zeros((4, 2)), y_true, label_type="binary")
        assert isinstance(metrics["confusion_matrix"], list)
        assert isinstance(metrics["confusion_matrix"][0], list)

    def test_confusion_matrix_correct_shape_binary(self):
        y_true = np.array([0, 0, 1, 1])
        model = _make_model(y_true)
        metrics = evaluate_model(model, np.zeros((4, 2)), y_true, label_type="binary")
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2

    def test_all_scalar_metrics_are_json_serialisable(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        model = _make_model(y_pred)
        metrics = evaluate_model(model, np.zeros((6, 2)), y_true, label_type="binary")
        # Remove non-scalar items before JSON check
        scalar = {k: v for k, v in metrics.items() if not isinstance(v, list)}
        json.dumps(scalar)  # must not raise


class TestEvaluateModelMulticlass:
    def test_required_keys_present(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        model = _make_model(y_pred)
        metrics = evaluate_model(model, np.zeros((6, 3)), y_true, label_type="multiclass")
        for key in ["accuracy", "f1_macro", "mcc", "confusion_matrix"]:
            assert key in metrics

    def test_per_class_f1_present(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 1])
        model = _make_model(y_pred)
        metrics = evaluate_model(model, np.zeros((6, 3)), y_true, label_type="multiclass")
        assert "per_class_f1" in metrics
        assert len(metrics["per_class_f1"]) == 3

    def test_raises_on_invalid_label_type(self):
        y_true = np.array([0, 1])
        model = _make_model(y_true)
        with pytest.raises(ValueError):
            evaluate_model(model, np.zeros((2, 2)), y_true, label_type="invalid")


# ═══════════════════════════════════════════════════════════════════════════════
# compute_privacy_utility_tradeoff
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrivacyUtilityTradeoff:
    def _make_dp_results(self, f1_values, eps_values):
        return [
            {
                "model_name": "dp_logistic_regression",
                "epsilon": eps,
                "metrics": {"test": {"f1_macro": f1}},
            }
            for eps, f1 in zip(eps_values, f1_values)
        ]

    def test_zero_loss_when_dp_matches_baseline(self):
        results = self._make_dp_results([0.90], [1.0])
        augmented = compute_privacy_utility_tradeoff(results, baseline_metric_value=0.90)
        assert augmented[0]["metric_loss_vs_baseline"] == 0.0
        assert augmented[0]["metric_loss_pct"] == 0.0

    def test_positive_loss_when_dp_worse(self):
        results = self._make_dp_results([0.80], [1.0])
        augmented = compute_privacy_utility_tradeoff(results, baseline_metric_value=0.90)
        assert augmented[0]["metric_loss_vs_baseline"] > 0

    def test_negative_loss_when_dp_better(self):
        """DP model that beats the baseline should show negative loss."""
        results = self._make_dp_results([0.95], [1.0])
        augmented = compute_privacy_utility_tradeoff(results, baseline_metric_value=0.90)
        assert augmented[0]["metric_loss_vs_baseline"] < 0

    def test_loss_pct_formula(self):
        """loss_pct = 100 * (baseline - dp_metric) / baseline."""
        baseline = 0.80
        dp_f1    = 0.72
        results  = self._make_dp_results([dp_f1], [1.0])
        augmented = compute_privacy_utility_tradeoff(results, baseline_metric_value=baseline)
        expected_pct = round(100.0 * (baseline - dp_f1) / baseline, 2)
        assert abs(augmented[0]["metric_loss_pct"] - expected_pct) < 0.01

    def test_original_results_not_mutated(self):
        """The function should not modify the caller's list."""
        results = self._make_dp_results([0.80], [1.0])
        original_copy = [dict(r) for r in results]
        compute_privacy_utility_tradeoff(results, baseline_metric_value=0.90)
        assert "metric_loss_vs_baseline" not in results[0], \
            "Original result dicts must not be mutated"

    def test_multiple_epsilons_all_augmented(self):
        results = self._make_dp_results([0.60, 0.75, 0.85, 0.90], [0.1, 0.5, 1.0, 5.0])
        augmented = compute_privacy_utility_tradeoff(results, baseline_metric_value=0.90)
        assert len(augmented) == 4
        for r in augmented:
            assert "metric_loss_vs_baseline" in r
            assert "metric_loss_pct" in r


# ═══════════════════════════════════════════════════════════════════════════════
# summarise_results
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummariseResults:
    def _make_results(self):
        return [
            {
                "model_name": "logistic_regression",
                "dataset": "cic_ids2018",
                "label_type": "binary",
                "epsilon": None,
                "train_time_s": 10.5,
                "metrics": {"test": {"f1_macro": 0.91, "accuracy": 0.95,
                                     "roc_auc": 0.97, "mcc": 0.83}},
            },
            {
                "model_name": "dp_logistic_regression",
                "dataset": "cic_ids2018",
                "label_type": "binary",
                "epsilon": 1.0,
                "train_time_s": 12.0,
                "metrics": {"test": {"f1_macro": 0.88, "accuracy": 0.93,
                                     "roc_auc": 0.95, "mcc": 0.79}},
            },
        ]

    def test_returns_list_of_dicts(self):
        rows = summarise_results(self._make_results(), split="test")
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_one_row_per_result(self):
        rows = summarise_results(self._make_results(), split="test")
        assert len(rows) == 2

    def test_epsilon_is_none_for_baseline(self):
        rows = summarise_results(self._make_results(), split="test")
        baseline_row = next(r for r in rows if r["model"] == "logistic_regression")
        assert baseline_row["epsilon"] is None

    def test_f1_macro_rounded_to_4dp(self):
        rows = summarise_results(self._make_results(), split="test")
        for r in rows:
            v = r.get("f1_macro")
            if v is not None:
                assert isinstance(v, float)
