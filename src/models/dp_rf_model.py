"""
dp_rf_model.py — Differentially Private Random Forest training.

Implements a DP Random Forest using diffprivlib's RandomForestClassifier,
which is based on the "Differentially Private Random Decision Forests" algorithm.

Privacy mechanism
-----------------
Unlike a standard Random Forest (which chooses splits that maximise information
gain from the data), this DP-RF:

    1. Builds trees with **random split selection** — feature and threshold are
       chosen uniformly at random, not by optimising over the training data.
       This means the tree *structure* does not consume any privacy budget.

    2. Applies the **PermuteAndFlip mechanism** at each leaf to determine the
       predicted class label. This is the only step that uses training labels,
       and it is where the ε budget is spent.

Guarantee: **pure ε-Differential Privacy** (no δ term).
This is a *stronger* formal guarantee than DP-SGD's (ε, δ)-DP because there
is no failure probability δ. The same ε value provides a stricter bound on
information leakage.

Comparison with other DP models in this codebase
-------------------------------------------------
| Model   | Library     | DP type       | Model family       |
|---------|-------------|---------------|--------------------|
| DP-LR   | diffprivlib | pure ε-DP     | Linear             |
| DP-RF   | diffprivlib | pure ε-DP     | Tree ensemble      |
| DP-SGD  | Opacus      | (ε, δ)-DP     | Neural network     |

Why not DP-XGBoost?
-------------------
XGBoost uses **data-driven split selection**: the best split threshold is
found by computing gradient/hessian statistics across all training samples.
Making this step differentially private (via the exponential mechanism or
noisy histogram methods) is research-level work with no established pip
package. Papers like DPBoost (2019) and DP-GBDT address this problem, but
no production-ready library exists. diffprivlib's DP-RF avoids the problem
entirely by randomising splits rather than optimising them.

Feature bounds
--------------
diffprivlib requires explicit min/max bounds per feature for the PermuteAndFlip
mechanism. After StandardScaler, features have approximately zero mean and unit
variance. We compute empirical per-feature bounds from the training split (99th
percentile expanded by 10%) to avoid clipping legitimate extreme values while
keeping the bounds tight enough not to inflate noise unnecessarily.

Usage:
    from src.models.dp_rf_model import train_dp_rf_model, run_dp_rf_sweep

    result  = train_dp_rf_model("cic_ids2018", epsilon=1.0, label="binary")
    results = run_dp_rf_sweep("cic_ids2018", label="binary")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.utils.config import PATHS, PRIVACY_CFG, SAMPLING_CFG
from src.utils.logger import get_logger
from src.data.sampler import get_split_arrays
from src.evaluation.metrics import evaluate_model

log = get_logger(__name__)

DP_RF_MODEL_NAME = "dp_random_forest"

# Number of trees. Using 10 (diffprivlib default) rather than 100 because
# more trees do not improve privacy — the ε budget covers the whole forest —
# but they increase training time significantly.
_N_ESTIMATORS = 10
_MAX_DEPTH = 5  # Limits tree size; deeper trees do not add privacy cost here
                # but keeping it shallow improves generalisation under noise.


# ═══════════════════════════════════════════════════════════════════════════════
# Feature bounds estimation
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_feature_bounds(
    X_train: np.ndarray,
    percentile: float = 99.0,
    buffer: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-feature min/max bounds for diffprivlib's DP-RF.

    The PermuteAndFlip mechanism needs explicit feature bounds to generate
    random split thresholds within a valid range. After StandardScaler,
    features are approximately N(0,1); the 99th percentile covers most of
    the distribution without being dominated by extreme outliers.

    A 10% buffer is added to each bound to prevent clipping at the edges.
    This slightly widens the noise scale but avoids underestimating the range.

    Args:
        X_train:    Scaled training feature matrix (n_samples × n_features).
        percentile: Upper percentile for computing per-feature maximum.
        buffer:     Fractional buffer added to each bound (default 10%).

    Returns:
        Tuple (lower_bounds, upper_bounds), each shape (n_features,).
    """
    lo = np.percentile(X_train, 100.0 - percentile, axis=0)
    hi = np.percentile(X_train, percentile, axis=0)

    # Expand each bound outward by the buffer fraction of the feature range
    margin = (hi - lo) * buffer
    lo -= margin
    hi += margin

    log.debug(
        "Feature bounds computed: lo in [%.2f, %.2f], hi in [%.2f, %.2f]",
        lo.min(), lo.max(), hi.min(), hi.max(),
    )
    return lo, hi


# ═══════════════════════════════════════════════════════════════════════════════
# Path helper
# ═══════════════════════════════════════════════════════════════════════════════

def _dp_rf_model_path(dataset_name: str, epsilon: float, label: str) -> Path:
    eps_str = f"{epsilon:.3f}".replace(".", "_")
    return PATHS["models"] / f"dp_rf_{dataset_name}_{label}_eps{eps_str}.joblib"


# ═══════════════════════════════════════════════════════════════════════════════
# Single-epsilon training
# ═══════════════════════════════════════════════════════════════════════════════

def train_dp_rf_model(
    dataset_name: str,
    epsilon: float,
    label: str = "binary",
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Train one Differentially Private Random Forest at a given epsilon.

    Args:
        dataset_name: One of "cic_ids2018", "unsw_nb15".
        epsilon:      Privacy budget. Same values used as DP-LR and DP-SGD
                      for direct comparison.
        label:        "binary" or "multiclass".
        bounds:       Optional pre-computed (lower, upper) feature bounds.
                      If None, bounds are estimated from X_train. Pass the
                      same bounds across all ε values so the only variable
                      is ε, not the bounds.
        force:        Re-train even if model file already exists.

    Returns:
        Result dict with the same schema as train_dp_model() and
        train_dp_sgd_model() for consistent downstream handling.
    """
    model_path = _dp_rf_model_path(dataset_name, epsilon, label)

    # Load cached result if available
    results_path = (
        PATHS["results"]
        / f"{dataset_name}_dp_rf_sweep_{label}.json"
    )
    if model_path.exists() and not force:
        log.info(
            "DP-RF model already exists for eps=%.3f — skipping. "
            "Pass force=True to retrain.",
            epsilon,
        )
        # Load and return cached result from the sweep JSON if it exists
        import json
        if results_path.exists():
            with open(results_path) as f:
                cached = json.load(f)
            for r in cached:
                if abs(r.get("epsilon", -1) - epsilon) < 1e-9:
                    return r
        # No cached result dict — fall through and re-evaluate
        pass

    seed = int(SAMPLING_CFG.get("random_seed", 42))
    splits = get_split_arrays(dataset_name, label=label)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    if bounds is None:
        lo, hi = estimate_feature_bounds(X_train)
    else:
        lo, hi = bounds

    n_classes = len(np.unique(y_train))
    classes   = np.arange(n_classes)

    log.info(
        "Training DP-RF on '%s' — eps=%.3f, label=%s, "
        "%d train samples, %d features",
        dataset_name, epsilon, label, len(X_train), X_train.shape[1],
    )
    log.info(
        "DP-RF config: n_estimators=%d, max_depth=%d, "
        "pure epsilon-DP (no delta)",
        _N_ESTIMATORS, _MAX_DEPTH,
    )

    from diffprivlib.models import RandomForestClassifier as DPRFClassifier

    t0 = time.time()
    model = DPRFClassifier(
        n_estimators=_N_ESTIMATORS,
        epsilon=epsilon,
        max_depth=_MAX_DEPTH,
        bounds=(lo, hi),
        classes=classes,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    log.info("DP-RF training complete — eps=%.3f, %.1fs", epsilon, train_time)

    # Save model
    PATHS["models"].mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    log.info("DP-RF model saved to: %s", model_path)

    # Evaluate on val and test splits
    log.info("Evaluating DP-RF (eps=%.3f) on val split...", epsilon)
    val_metrics  = evaluate_model(model, X_val,  y_val,  label_type=label)
    log.info("Evaluating DP-RF (eps=%.3f) on test split...", epsilon)
    test_metrics = evaluate_model(model, X_test, y_test, label_type=label)

    log.info(
        "DP-RF | eps=%.3f | %s | test F1=%.4f | AUC=%s | MCC=%.4f",
        epsilon, dataset_name,
        test_metrics.get("f1_macro", 0),
        f"{test_metrics.get('roc_auc', 0):.4f}" if test_metrics.get("roc_auc") else "—",
        test_metrics.get("mcc", 0),
    )

    return {
        "model_name":    DP_RF_MODEL_NAME,
        "dataset":       dataset_name,
        "label_type":    label,
        "epsilon":       epsilon,
        "delta":         None,  # pure ε-DP — no delta
        "dp_type":       "pure_epsilon",  # distinguishes from (ε,δ) models
        "n_estimators":  _N_ESTIMATORS,
        "max_depth":     _MAX_DEPTH,
        "train_time_s":  round(train_time, 2),
        "metrics": {
            "val":  val_metrics,
            "test": test_metrics,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Epsilon sweep
# ═══════════════════════════════════════════════════════════════════════════════

def run_dp_rf_sweep(
    dataset_name: str,
    label: str = "binary",
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    Train DP-RF at every epsilon value in config.yaml and save results.

    Feature bounds are computed once from the training split and reused
    across all ε values so that the only variable between runs is ε.

    Args:
        dataset_name: One of "cic_ids2018", "unsw_nb15".
        label:        "binary" or "multiclass".
        force:        Re-train models even if they already exist.

    Returns:
        List of result dicts (one per ε value), also saved to JSON.
    """
    from src.evaluation.results_store import save_results

    epsilon_values = PRIVACY_CFG.get("epsilon_values", [])
    if not epsilon_values:
        raise ValueError("No epsilon_values found in config.yaml privacy section.")

    log.info("=" * 70)
    log.info(
        "Starting DP-RF epsilon sweep for '%s' (label=%s) — %d values: %s",
        dataset_name, label, len(epsilon_values), epsilon_values,
    )
    log.info("=" * 70)

    # Compute bounds once (same training data for all ε)
    splits = get_split_arrays(dataset_name, label=label)
    X_train, _ = splits["train"]
    bounds = estimate_feature_bounds(X_train)
    log.info("Feature bounds pre-computed from training split.")

    results: list[dict] = []
    for epsilon in epsilon_values:
        log.info("-" * 50)
        log.info("Running DP-RF with eps = %s", epsilon)
        result = train_dp_rf_model(
            dataset_name, epsilon,
            label=label,
            bounds=bounds,
            force=force,
        )
        results.append(result)
        m = result["metrics"]["test"]
        log.info(
            "DP-RF | eps=%.3f | %s | test F1=%.4f | AUC=%s",
            epsilon, dataset_name,
            m.get("f1_macro", 0),
            f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") else "—",
        )

    save_results(results, dataset_name, "dp_rf_sweep", label=label)

    log.info("=" * 70)
    log.info(
        "DP-RF sweep complete for '%s'. Summary (test F1 macro):", dataset_name
    )
    for r in results:
        log.info(
            "  eps=%5.2f -> F1=%.4f | AUC=%s | MCC=%.4f",
            r["epsilon"],
            r["metrics"]["test"].get("f1_macro", 0),
            f"{r['metrics']['test'].get('roc_auc', 0):.4f}"
            if r["metrics"]["test"].get("roc_auc") else "—",
            r["metrics"]["test"].get("mcc", 0),
        )
    log.info("=" * 70)

    return results
