"""
dp_model.py — Differentially Private Logistic Regression training.

Implements the privacy-preserving variant of the experiment using IBM's
diffprivlib library. diffprivlib.models.LogisticRegression is a drop-in
replacement for sklearn's equivalent, with two additional parameters:

    epsilon   : privacy budget (eps). Smaller = stronger privacy = more noise.
    data_norm : L2 norm bound on feature vectors. Used internally to
                calibrate the Gaussian mechanism noise scale. Must be
                set manually because the DP guarantee requires knowing the
                sensitivity of the data upfront.

The key design principle: this module is intentionally symmetric with
baseline.py. Both expose train_<model>() and the same result dict schema.
This means evaluation and result storage code can treat them identically —
the only distinguishing field is `epsilon` (None for baselines, float for DP).

Epsilon sweep:
    The primary experiment iterates over epsilon_values from config.yaml
    (default: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]) and records metrics at
    each value. This produces the privacy–utility tradeoff curve that is
    the central empirical result of the dissertation.

Data norm:
    data_norm must be set to an upper bound on the L2 norm of the
    (scaled) feature vectors. After StandardScaler, most vectors have
    L2 norm ≈ sqrt(n_features). We compute this empirically from the
    training set and add a small buffer (5%) for safety. Setting it too
    low clips inputs and degrades accuracy; setting it too high adds more
    noise than necessary.

Usage:
    from src.models.dp_model import train_dp_model, run_epsilon_sweep

    # Single run at a fixed epsilon
    result = train_dp_model("cic_ids2018", epsilon=1.0, label="binary")

    # Full epsilon sweep (all values from config.yaml)
    results = run_epsilon_sweep("cic_ids2018", label="binary")
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

DP_MODEL_NAME = "dp_logistic_regression"


# ── Data norm estimation ───────────────────────────────────────────────────────

def estimate_data_norm(X_train: np.ndarray, buffer: float = 0.05) -> float:
    """
    Estimate the L2 norm bound required by diffprivlib's LogisticRegression.

    diffprivlib requires an explicit upper bound on the L2 norm of each
    training sample. This is used to calibrate the noise added to the
    gradient during differentially private optimisation. Setting it correctly
    is important:
        - Too low  → inputs are clipped, accuracy degrades unnecessarily.
        - Too high → more noise is added than required, degrading accuracy.

    After StandardScaler, feature vectors have mean ≈ 0 and std ≈ 1, so the
    L2 norm of a d-dimensional vector is approximately sqrt(d). We compute
    the empirical 99th percentile across training samples and add a small
    buffer to ensure the bound is not violated by edge-case samples.

    Args:
        X_train: Scaled training feature matrix (n_samples × n_features).
        buffer:  Fractional buffer added to the empirical 99th percentile
                 (default 0.05 = 5%). Ensures edge cases stay within bound.

    Returns:
        Float data_norm value ready to pass to DPLogisticRegression.
    """
    # Compute L2 norm of each training sample
    norms = np.linalg.norm(X_train, axis=1)
    # Use 99th percentile rather than max to be robust to outliers
    norm_bound = float(np.percentile(norms, 99))
    # Add buffer to avoid clipping legitimate samples
    data_norm = norm_bound * (1.0 + buffer)

    log.info(
        "Data norm estimation — median: %.2f, 99th pctile: %.2f, bound (with buffer): %.2f",
        float(np.median(norms)), norm_bound, data_norm
    )
    return data_norm


# ── Model path helper ──────────────────────────────────────────────────────────

def _dp_model_path(dataset_name: str, epsilon: float, label: str) -> Path:
    """
    Construct the .joblib save path for a trained DP model at a given epsilon.

    Epsilon is formatted to three decimal places to avoid ambiguous filenames
    (e.g. 1.0 and 1.00 would otherwise collide on case-insensitive filesystems).
    """
    eps_str = f"{epsilon:.3f}".replace(".", "_")
    return PATHS["models"] / f"{dataset_name}_{DP_MODEL_NAME}_eps{eps_str}_{label}.joblib"


# ═══════════════════════════════════════════════════════════════════════════════
# Core training function
# ═══════════════════════════════════════════════════════════════════════════════

def train_dp_model(
    dataset_name: str,
    epsilon: float,
    label: str = "binary",
    data_norm: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Train a Differentially Private Logistic Regression at a given epsilon.

    The DP guarantee provided is (epsilon, delta)-DP where delta defaults
    to 1e-5 inside diffprivlib. Epsilon controls the privacy budget:
    lower epsilon → stronger privacy → more noise → lower accuracy.

    The model is saved to disk at results/models/ after training, keyed by
    dataset name, epsilon value, and label type. If the saved model exists
    and force=False, it is loaded and re-evaluated without retraining.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        epsilon:      Privacy budget eps > 0. Typical dissertation values:
                      [0.1, 0.5, 1.0, 2.0, 5.0, 10.0].
        label:        'binary' or 'multiclass'.
        data_norm:    L2 norm bound for training samples. If None, estimated
                      automatically from training data via estimate_data_norm().
        force:        Re-train even if a saved model exists.

    Returns:
        Result dict with same schema as baseline.train_baseline():
        {
          "model_name":   "dp_logistic_regression",
          "dataset":      str,
          "label_type":   str,
          "epsilon":      float,     ← key difference from baseline results
          "train_time_s": float,
          "data_norm":    float,
          "metrics": {
              "val":  { metric_name: value, ... },
              "test": { metric_name: value, ... },
          }
        }

    Raises:
        ImportError: If diffprivlib is not installed.
        ValueError:  If epsilon <= 0.
    """
    try:
        from diffprivlib.models import LogisticRegression as DPLogisticRegression
    except ImportError:
        raise ImportError(
            "diffprivlib is required for DP experiments. "
            "Install it with: pip install diffprivlib"
        )

    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    saved_path = _dp_model_path(dataset_name, epsilon, label)
    seed = SAMPLING_CFG["random_seed"]

    # ── Load arrays ───────────────────────────────────────────────────────────
    splits = get_split_arrays(dataset_name, label=label, scale=True)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    # ── Estimate or use provided data norm ────────────────────────────────────
    if data_norm is None:
        data_norm = estimate_data_norm(X_train)

    # ── Load or train ─────────────────────────────────────────────────────────
    if saved_path.exists() and not force:
        log.info(
            "Loading saved DP-LR model for '%s' (eps=%.2f, label=%s)",
            dataset_name, epsilon, label
        )
        model = joblib.load(saved_path)
        train_time = 0.0
    else:
        log.info(
            "Training DP-LR on '%s' — eps=%.2f, data_norm=%.2f, "
            "label=%s, %d train samples, %d features",
            dataset_name, epsilon, data_norm, label,
            X_train.shape[0], X_train.shape[1]
        )

        # ── Construct DP Logistic Regression ──────────────────────────────────
        # diffprivlib's LogisticRegression wraps sklearn's LR with a
        # Differentially Private SGD optimiser. The noise scale is derived
        # from epsilon and data_norm automatically.
        #
        # solver: 'lbfgs' is the only solver supported by diffprivlib LR.
        # max_iter: may need to be higher than non-private LR because the
        #           noisy gradient updates converge more slowly.
        model = DPLogisticRegression(
            epsilon=epsilon,
            data_norm=data_norm,
            max_iter=200,
            random_state=seed,
        )

        t0 = time.perf_counter()

        # diffprivlib LR does not support class_weight='balanced' natively.
        # We approximate class balancing by passing sample_weight instead.
        sample_weights = _compute_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        train_time = time.perf_counter() - t0
        log.info(
            "DP-LR training complete — eps=%.2f, %.1f seconds",
            epsilon, train_time
        )

        # Persist model
        PATHS["models"].mkdir(parents=True, exist_ok=True)
        joblib.dump(model, saved_path)
        log.info("DP model saved to: %s", saved_path)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info("Evaluating DP-LR (eps=%.2f) on val split...", epsilon)
    val_metrics  = evaluate_model(model, X_val,  y_val,  label_type=label)

    log.info("Evaluating DP-LR (eps=%.2f) on test split...", epsilon)
    test_metrics = evaluate_model(model, X_test, y_test, label_type=label)

    result = {
        "model_name":   DP_MODEL_NAME,
        "dataset":      dataset_name,
        "label_type":   label,
        "epsilon":      epsilon,
        "data_norm":    round(data_norm, 4),
        "train_time_s": round(train_time, 2),
        "metrics": {
            "val":  val_metrics,
            "test": test_metrics,
        },
    }

    log.info(
        "DP-LR | eps=%.2f | %s | test F1=%.4f | AUC=%s | MCC=%.4f",
        epsilon,
        dataset_name,
        test_metrics.get("f1_macro", 0),
        f"{test_metrics['roc_auc']:.4f}" if test_metrics.get("roc_auc") is not None else "N/A",
        test_metrics.get("mcc", 0),
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Epsilon sweep
# ═══════════════════════════════════════════════════════════════════════════════

def run_epsilon_sweep(
    dataset_name: str,
    label: str = "binary",
    epsilon_values: list[float] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    Train DP-LR at each epsilon value and return all results.

    This is the central experiment of the dissertation: by holding everything
    fixed except epsilon, the results isolate the effect of the privacy budget
    on model utility. The returned list is passed directly to
    results_store.save_results() and metrics.compute_privacy_utility_tradeoff().

    A single data_norm is estimated once from the training split and reused
    across all epsilon values. This ensures the only variable changing between
    runs is the noise magnitude (which is a function of epsilon and data_norm),
    making the comparison clean.

    Args:
        dataset_name:   Key matching config.yaml 'datasets' entry.
        label:          'binary' or 'multiclass'.
        epsilon_values: List of epsilon values to sweep. Defaults to
                        privacy.epsilon_values from config.yaml.
        force:          Re-train all epsilon models even if saved.

    Returns:
        List of result dicts (one per epsilon), ordered by ascending epsilon.
    """
    if epsilon_values is None:
        epsilon_values = PRIVACY_CFG["epsilon_values"]

    # Sort ascending so results are in a natural order
    epsilon_values = sorted(epsilon_values)

    log.info("=" * 70)
    log.info(
        "Starting epsilon sweep for '%s' (label=%s) — %d values: %s",
        dataset_name, label, len(epsilon_values), epsilon_values
    )
    log.info("=" * 70)

    # ── Pre-compute data norm once for the full sweep ─────────────────────────
    # Loading arrays here so we can estimate data_norm before the loop.
    splits = get_split_arrays(dataset_name, label=label, scale=True)
    X_train, _ = splits["train"]
    data_norm = estimate_data_norm(X_train)

    # ── Run sweep ─────────────────────────────────────────────────────────────
    all_results = []
    for eps in epsilon_values:
        log.info("-" * 50)
        log.info("Running DP-LR with eps = %.2f", eps)
        result = train_dp_model(
            dataset_name=dataset_name,
            epsilon=eps,
            label=label,
            data_norm=data_norm,
            force=force,
        )
        all_results.append(result)

    # ── Log sweep summary ─────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Epsilon sweep complete for '%s'. Summary (test F1 macro):", dataset_name)
    for r in all_results:
        log.info(
            "  eps=%5.2f → F1=%.4f | AUC=%s | MCC=%.4f",
            r["epsilon"],
            r["metrics"]["test"].get("f1_macro", 0),
            f"{r['metrics']['test']['roc_auc']:.4f}"
            if r["metrics"]["test"].get("roc_auc") is not None else "N/A",
            r["metrics"]["test"].get("mcc", 0),
        )

    return all_results


def load_dp_model(
    dataset_name: str,
    epsilon: float,
    label: str = "binary",
) -> Any:
    """
    Load a previously trained DP model from disk.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        epsilon:      The epsilon value the model was trained at.
        label:        'binary' or 'multiclass'.

    Returns:
        Fitted diffprivlib LogisticRegression model.

    Raises:
        FileNotFoundError: If the model has not been trained at this epsilon.
    """
    saved_path = _dp_model_path(dataset_name, epsilon, label)

    if not saved_path.exists():
        raise FileNotFoundError(
            f"No DP model found at {saved_path}. "
            f"Run train_dp_model('{dataset_name}', epsilon={epsilon}) first."
        )

    log.info("Loading DP model (eps=%.2f): %s", epsilon, saved_path)
    return joblib.load(saved_path)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute inverse-frequency sample weights to approximate class balancing.

    diffprivlib's LogisticRegression does not support class_weight='balanced'
    as a constructor parameter. Passing sample_weight to .fit() achieves the
    same effect: each sample is weighted by the inverse frequency of its class.

    Formula: weight_i = n_samples / (n_classes * count(class_i))
    This matches sklearn's 'balanced' class_weight computation exactly.

    Args:
        y: Integer label array (training labels).

    Returns:
        Float array of per-sample weights, same length as y.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)

    # Map class → weight
    class_weight = {
        cls: n_samples / (n_classes * cnt)
        for cls, cnt in zip(classes, counts)
    }

    sample_weights = np.array([class_weight[label] for label in y])
    log.debug(
        "Sample weights computed — class weights: %s",
        {int(k): round(v, 3) for k, v in class_weight.items()}
    )
    return sample_weights
