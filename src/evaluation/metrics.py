"""
metrics.py — Model evaluation and metric computation.

This module is model-agnostic: it only works with predictions and ground-truth
labels. Both baseline.py and dp_model.py call evaluate_model() identically,
which ensures the comparison between private and non-private models is fair —
the same metric functions with the same averaging strategies applied to both.

Metrics computed:
    Binary classification:
        accuracy, precision, recall, f1 (macro + weighted), roc_auc,
        matthews_corrcoef (MCC), false_positive_rate, detection_rate,
        confusion_matrix

    Multiclass classification:
        accuracy, precision (macro), recall (macro), f1 (macro + weighted),
        roc_auc (OvR, macro), matthews_corrcoef, confusion_matrix,
        per_class_f1 (dict of class_name → f1 score)

Why these metrics for IDS:
    - F1 macro    : treats all attack categories equally; avoids metric
                    inflation from the large benign class
    - ROC-AUC     : threshold-independent; shows discrimination ability
    - MCC         : best single metric for binary imbalanced classification;
                    mathematically equivalent to a Phi correlation coefficient
    - FPR / DR    : operationally meaningful for IDS operators; FPR = false
                    alarm rate, DR = detection rate (True Positive Rate)
    - Accuracy    : included for completeness but should not be the headline
                    metric on imbalanced data

Usage:
    from src.evaluation.metrics import evaluate_model

    metrics = evaluate_model(model, X_test, y_test, label_type="binary")
    # returns: { "accuracy": 0.95, "f1_macro": 0.87, "roc_auc": 0.96, ... }
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Core evaluation function
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    label_type: str = "binary",
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute the full evaluation metric suite for a fitted classifier.

    Calls model.predict() and model.predict_proba() internally. Both
    sklearn and diffprivlib models expose these methods so this function
    works identically for baseline and DP models.

    Args:
        model:       Fitted sklearn-compatible model (baseline or DP).
        X:           Feature matrix for the split being evaluated (numpy array).
        y_true:      Ground-truth integer labels.
        label_type:  'binary' or 'multiclass'. Controls which metric variants
                     are computed and which averaging strategy is used.
        class_names: Optional list of class name strings in the same order as
                     integer label encoding. Used for per-class F1 reporting.
                     If None, classes are named by their integer code.

    Returns:
        Dict with metric names as keys and scalar or structured values.
        All scalar metrics are Python floats (not numpy scalars) so the dict
        can be serialised directly to JSON.
    """
    if label_type not in ("binary", "multiclass"):
        raise ValueError(f"label_type must be 'binary' or 'multiclass', got '{label_type}'")

    # ── Generate predictions ──────────────────────────────────────────────────
    y_pred = model.predict(X)

    # predict_proba may not exist for all models; handle gracefully
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)
        except Exception as exc:
            log.warning("predict_proba failed (%s) — ROC-AUC will be None", exc)

    # ── Shared metrics ────────────────────────────────────────────────────────
    metrics: dict[str, Any] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_macro"]     = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    metrics["f1_weighted"]  = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
    metrics["recall_macro"]    = float(recall_score(y_true, y_pred,    average="macro",    zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # Confusion matrix stored as nested list for JSON serialisability
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # ── Binary-specific metrics ───────────────────────────────────────────────
    if label_type == "binary":
        metrics.update(_binary_metrics(y_true, y_pred, y_prob))

    # ── Multiclass-specific metrics ───────────────────────────────────────────
    else:
        metrics.update(_multiclass_metrics(y_true, y_pred, y_prob, class_names))

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Binary and multiclass metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> dict[str, Any]:
    """
    Compute metrics specific to binary classification.

    False Positive Rate (FPR) and Detection Rate (DR) are particularly
    meaningful for IDS evaluation:
        FPR = FP / (FP + TN)  — fraction of benign flows flagged as attacks
        DR  = TP / (TP + FN)  — fraction of actual attacks detected

    In operational IDS terms: low FPR prevents alert fatigue; high DR
    ensures attacks are not silently missed.
    """
    extra: dict[str, Any] = {}

    # ROC-AUC using positive class probabilities
    if y_prob is not None:
        try:
            # y_prob[:, 1] = probability of the attack class (class 1)
            prob_positive = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            extra["roc_auc"] = float(roc_auc_score(y_true, prob_positive))
        except Exception as exc:
            log.warning("Binary ROC-AUC computation failed: %s", exc)
            extra["roc_auc"] = None
    else:
        extra["roc_auc"] = None

    # Derive FPR and DR from the confusion matrix
    # For binary: CM = [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        extra["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        extra["detection_rate"]      = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        extra["false_negative_rate"] = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    else:
        # Fallback if confusion matrix is not 2×2 (e.g., only one class present)
        extra["false_positive_rate"] = None
        extra["detection_rate"]      = None
        extra["false_negative_rate"] = None

    return extra


def _multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    class_names: list[str] | None,
) -> dict[str, Any]:
    """
    Compute metrics specific to multiclass classification.

    ROC-AUC uses One-vs-Rest (OvR) strategy with macro averaging.
    Per-class F1 scores are useful in the dissertation to show which
    attack categories are hardest to detect under privacy noise.
    """
    extra: dict[str, Any] = {}

    # ROC-AUC (OvR, macro) — requires probability scores
    if y_prob is not None:
        try:
            extra["roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except Exception as exc:
            log.warning("Multiclass ROC-AUC computation failed: %s", exc)
            extra["roc_auc"] = None
    else:
        extra["roc_auc"] = None

    # Per-class F1 scores — keyed by class name or integer string
    classes = np.unique(y_true)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    if class_names and len(class_names) == len(per_class_f1):
        extra["per_class_f1"] = {
            name: float(score)
            for name, score in zip(class_names, per_class_f1)
        }
    else:
        extra["per_class_f1"] = {
            str(cls): float(score)
            for cls, score in zip(classes, per_class_f1)
        }

    return extra


# ═══════════════════════════════════════════════════════════════════════════════
# Tradeoff comparison helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_privacy_utility_tradeoff(
    dp_results: list[dict],
    baseline_metric_value: float,
    metric_key: str = "f1_macro",
    split: str = "test",
) -> list[dict]:
    """
    Augment DP experiment result dicts with privacy cost columns.

    Adds two comparison fields to each DP result:
        metric_loss_vs_baseline : absolute drop from the non-private baseline
        metric_loss_pct         : percentage drop from the non-private baseline

    These are used directly to populate the privacy–utility tradeoff table
    in the dissertation and the Streamlit comparison page.

    Args:
        dp_results:            List of result dicts from dp_model.run_epsilon_sweep().
        baseline_metric_value: The test-split metric value from the matching
                               non-private model (e.g. LR baseline F1 macro).
        metric_key:            Which metric to compare (default 'f1_macro').
        split:                 Which split's metrics to use (default 'test').

    Returns:
        The same list with two additional keys added to each result dict:
            'metric_loss_vs_baseline' and 'metric_loss_pct'.
    """
    augmented = []
    for result in dp_results:
        r = dict(result)   # shallow copy — do not mutate caller's list
        dp_metric = r["metrics"][split].get(metric_key)

        if dp_metric is not None and baseline_metric_value > 0:
            loss     = baseline_metric_value - dp_metric
            loss_pct = 100.0 * loss / baseline_metric_value
        else:
            loss, loss_pct = None, None

        r["metric_loss_vs_baseline"] = round(loss, 4)     if loss     is not None else None
        r["metric_loss_pct"]         = round(loss_pct, 2) if loss_pct is not None else None
        augmented.append(r)

    return augmented


def summarise_results(results: list[dict], split: str = "test") -> list[dict]:
    """
    Flatten a list of result dicts into a list of plain row dicts suitable
    for display as a Streamlit dataframe or export as a CSV/LaTeX table.

    Each output row has: model_name, dataset, epsilon, and all metric scalars
    from the given split.

    Args:
        results: List of result dicts from train_baseline() or run_epsilon_sweep().
        split:   Which split to extract metrics from ('val' or 'test').

    Returns:
        List of flat dicts, one per result row.
    """
    rows = []
    for r in results:
        row = {
            "model":   r.get("model_name", ""),
            "dataset": r.get("dataset", ""),
            "label":   r.get("label_type", ""),
            "epsilon": r.get("epsilon"),      # None for non-private models
        }
        # Flatten scalar metrics from the chosen split
        split_metrics = r.get("metrics", {}).get(split, {})
        for k, v in split_metrics.items():
            # Skip non-scalar metrics (confusion matrix, per_class_f1)
            if isinstance(v, (int, float)) or v is None:
                row[k] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)
    return rows


def print_classification_report(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str] | None = None,
) -> str:
    """
    Return a formatted sklearn classification report string.

    Useful for logging during training and for pasting into a dissertation
    appendix. Reports precision, recall, F1, and support per class.

    Args:
        model:       Fitted sklearn-compatible model.
        X:           Feature matrix.
        y_true:      Ground-truth labels.
        class_names: Optional class name labels for display.

    Returns:
        Multi-line string classification report.
    """
    y_pred = model.predict(X)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    return report
