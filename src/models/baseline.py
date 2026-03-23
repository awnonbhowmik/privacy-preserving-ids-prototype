"""
baseline.py — Standard (non-private) classifier training and persistence.

Implements three baseline models for network intrusion detection:
    1. Logistic Regression  — simple linear reference; serves as the direct
                              comparison anchor for the DP-LR experiments.
    2. Random Forest        — primary tree-based baseline; strong default
                              performance on tabular IDS data; produces
                              feature importances for dissertation analysis.
    3. XGBoost              — performance ceiling; shows the best achievable
                              accuracy without any privacy constraints.

All models:
  - Use class_weight='balanced' to handle severe class imbalance without
    modifying the training data (no SMOTE required at baseline stage).
  - Are trained on the same fixed train split from the subset Parquet.
  - Are saved to results/models/ as .joblib files so they do not need to
    be re-trained for each Streamlit session.

Usage:
    from src.models.baseline import train_all_baselines, load_baseline_model

    results = train_all_baselines("cic_ids2018", label="binary")
    rf_model = load_baseline_model("cic_ids2018", "random_forest", label="binary")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import shutil

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

_gpu_available = shutil.which("nvidia-smi") is not None

from src.utils.config import PATHS, MODEL_CFG, SAMPLING_CFG
from src.utils.logger import get_logger
from src.data.sampler import get_split_arrays
from src.evaluation.metrics import evaluate_model

log = get_logger(__name__)

# ── Model name constants ───────────────────────────────────────────────────────
LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST       = "random_forest"
XGBOOST             = "xgboost"

BASELINE_MODELS = [LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST]


# ── Model factory ──────────────────────────────────────────────────────────────

def _build_logistic_regression(seed: int) -> LogisticRegression:
    """
    Construct a Logistic Regression classifier with settings appropriate
    for large IDS feature spaces.

    solver='saga' is chosen because it supports L1/L2 regularisation,
    scales to large datasets, and uses a stochastic gradient approach
    that is closest in spirit to the DP-LR optimiser (making the
    baseline/DP comparison methodologically consistent).

    max_iter=1000 is needed because IDS features span many orders of
    magnitude and convergence takes more iterations than the default 100.
    """
    return LogisticRegression(
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        C=1.0,               # inverse regularisation strength (default)
        random_state=seed,
        n_jobs=MODEL_CFG.get("n_jobs", -1),
    )


def _build_random_forest(seed: int) -> RandomForestClassifier:
    """
    Construct a Random Forest classifier.

    n_estimators and max_depth are pulled from config.yaml so they can be
    adjusted without touching code. class_weight='balanced' applies
    inverse-frequency weighting to each tree's bootstrap sample.
    """
    return RandomForestClassifier(
        n_estimators=MODEL_CFG.get("n_estimators", 100),
        max_depth=MODEL_CFG.get("max_depth", 10),
        class_weight="balanced",
        random_state=seed,
        n_jobs=MODEL_CFG.get("n_jobs", -1),
    )


def _build_xgboost(seed: int, n_classes: int) -> XGBClassifier:
    """
    Construct an XGBoost classifier.

    For binary classification use 'binary:logistic'; for multiclass use
    'multi:softprob'. scale_pos_weight is set for binary only — for
    multiclass, XGBoost handles imbalance via sample_weight at fit time,
    but we approximate it here via the eval_metric setting.
    """
    device_kwargs = {"device": "cuda", "tree_method": "hist"} if _gpu_available else {}
    if n_classes == 2:
        return XGBClassifier(
            n_estimators=MODEL_CFG.get("n_estimators", 100),
            max_depth=MODEL_CFG.get("max_depth", 10),
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed,
            n_jobs=MODEL_CFG.get("n_jobs", -1),
            verbosity=0,
            **device_kwargs,
        )
    else:
        return XGBClassifier(
            n_estimators=MODEL_CFG.get("n_estimators", 100),
            max_depth=MODEL_CFG.get("max_depth", 10),
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=seed,
            n_jobs=MODEL_CFG.get("n_jobs", -1),
            verbosity=0,
            **device_kwargs,
        )


# ── Model path helpers ─────────────────────────────────────────────────────────

def _model_path(dataset_name: str, model_name: str, label: str) -> Path:
    """Construct the .joblib save path for a trained baseline model."""
    return PATHS["models"] / f"{dataset_name}_{model_name}_{label}.joblib"


# ── Train / load API ───────────────────────────────────────────────────────────

def train_baseline(
    dataset_name: str,
    model_name: str,
    label: str = "binary",
    force: bool = False,
) -> dict[str, Any]:
    """
    Train a single baseline model and return its evaluation results.

    If a saved model already exists on disk and force=False, the existing
    model is loaded and re-evaluated rather than retrained. This avoids
    expensive retraining during Streamlit sessions.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        model_name:   One of LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST.
        label:        'binary' or 'multiclass'.
        force:        Re-train even if a saved model exists.

    Returns:
        Dict with structure:
        {
          "model_name":   str,
          "dataset":      str,
          "label_type":   str,
          "train_time_s": float,
          "metrics": {
              "val":  { metric_name: value, ... },
              "test": { metric_name: value, ... },
          }
        }

    Raises:
        ValueError: If model_name is not one of the supported options.
    """
    if model_name not in BASELINE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {BASELINE_MODELS}"
        )

    saved_path = _model_path(dataset_name, model_name, label)
    seed = SAMPLING_CFG["random_seed"]

    # ── Load arrays ───────────────────────────────────────────────────────────
    splits = get_split_arrays(dataset_name, label=label, scale=True)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    n_classes = len(np.unique(y_train))

    # ── Load or train ─────────────────────────────────────────────────────────
    if saved_path.exists() and not force:
        log.info(
            "Loading saved %s model for '%s' (label=%s)",
            model_name, dataset_name, label
        )
        model = joblib.load(saved_path)
        train_time = 0.0   # not re-measured for loaded models
    else:
        log.info(
            "Training %s on '%s' (label=%s) — %d train samples, %d features",
            model_name, dataset_name, label, X_train.shape[0], X_train.shape[1]
        )

        if model_name == LOGISTIC_REGRESSION:
            model = _build_logistic_regression(seed)
        elif model_name == RANDOM_FOREST:
            model = _build_random_forest(seed)
        elif model_name == XGBOOST:
            model = _build_xgboost(seed, n_classes)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        log.info("Training complete in %.1f seconds", train_time)

        # Persist model to disk
        PATHS["models"].mkdir(parents=True, exist_ok=True)
        joblib.dump(model, saved_path)
        log.info("Model saved to: %s", saved_path)

    # ── Evaluate on val and test ──────────────────────────────────────────────
    log.info("Evaluating %s on val split...", model_name)
    val_metrics  = evaluate_model(model, X_val,  y_val,  label_type=label)

    log.info("Evaluating %s on test split...", model_name)
    test_metrics = evaluate_model(model, X_test, y_test, label_type=label)

    result = {
        "model_name":   model_name,
        "dataset":      dataset_name,
        "label_type":   label,
        "train_time_s": round(train_time, 2),
        "epsilon":      None,   # None signals this is a non-private model
        "metrics": {
            "val":  val_metrics,
            "test": test_metrics,
        },
    }

    log.info(
        "%s | %s | test F1=%.4f | AUC=%.4f | MCC=%.4f",
        model_name, dataset_name,
        test_metrics.get("f1_macro", 0),
        test_metrics.get("roc_auc", 0),
        test_metrics.get("mcc", 0),
    )

    return result


def train_all_baselines(
    dataset_name: str,
    label: str = "binary",
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    Train all three baseline models and return a list of result dicts.

    Convenience wrapper that calls train_baseline() for each model in
    BASELINE_MODELS and collects the results. Used by the Streamlit
    training page to run all baselines with a single button click.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        label:        'binary' or 'multiclass'.
        force:        Re-train all models even if saved versions exist.

    Returns:
        List of result dicts (one per model) as returned by train_baseline().
    """
    log.info("=" * 70)
    log.info(
        "Training all baselines for '%s' (label=%s)", dataset_name, label
    )
    log.info("=" * 70)

    results = []
    for model_name in BASELINE_MODELS:
        result = train_baseline(dataset_name, model_name, label=label, force=force)
        results.append(result)

    log.info(
        "All baselines complete for '%s'. Summary:", dataset_name
    )
    for r in results:
        log.info(
            "  %-22s | test F1=%.4f | AUC=%.4f",
            r["model_name"],
            r["metrics"]["test"].get("f1_macro", 0),
            r["metrics"]["test"].get("roc_auc", 0),
        )

    return results


def load_baseline_model(
    dataset_name: str,
    model_name: str,
    label: str = "binary",
) -> Any:
    """
    Load a previously saved baseline model from disk.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        model_name:   One of LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST.
        label:        'binary' or 'multiclass'.

    Returns:
        Fitted sklearn-compatible model object.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    saved_path = _model_path(dataset_name, model_name, label)

    if not saved_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {saved_path}. "
            f"Run train_baseline('{dataset_name}', '{model_name}', label='{label}') first."
        )

    log.info("Loading baseline model: %s", saved_path)
    return joblib.load(saved_path)


def get_feature_importances(
    dataset_name: str,
    label: str = "binary",
) -> dict[str, float] | None:
    """
    Return feature importances from the saved Random Forest model.

    Feature importances are the mean decrease in impurity (Gini importance)
    across all trees. Useful for dissertation discussion of which network
    flow features are most discriminative for intrusion detection.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        label:        'binary' or 'multiclass'.

    Returns:
        Dict mapping feature name to importance score, sorted descending.
        Returns None if the Random Forest model has not been trained yet.
    """
    from src.data.loader import load_subset

    try:
        model = load_baseline_model(dataset_name, RANDOM_FOREST, label)
    except FileNotFoundError:
        log.warning("Random Forest model not found — cannot return feature importances.")
        return None

    df = load_subset(dataset_name)
    _NON_FEATURE_COLS = {
        "label_raw", "label_binary", "label_multiclass", "dataset_source", "split"
    }
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]

    importances = dict(zip(feature_cols, model.feature_importances_))
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
