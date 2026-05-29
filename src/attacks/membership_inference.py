"""
membership_inference.py — Black-box membership inference attack evaluation.

Implements a confidence-based black-box MIA that empirically measures whether
a target IDS model leaks information about its training records.

Attack threat model (black-box):
    The adversary queries the model with predict_proba() and observes the
    probability vector. No access to model weights or training data is assumed.
    This is the most realistic and easiest-to-justify setting for a dissertation.

How differential privacy is expected to help:
    DP training (diffprivlib, Opacus) bounds the maximum change in any model
    output caused by adding or removing a single training record. Models trained
    with tighter privacy (lower ε) should show weaker membership signals and
    therefore lower attack ROC-AUC. This module empirically verifies that claim.

Pipeline for one target model:
    1. Extract attack features from model.predict_proba() on training records
       (members, label=1) and held-out test records (non-members, label=0).
    2. Build a balanced attack dataset: n members + n non-members.
    3. Split into attack train / attack test (70/30), fit a logistic regression
       meta-classifier on the attack train features.
    4. Evaluate on attack test: accuracy, F1, ROC-AUC, advantage.
    5. Save structured results to results/{dataset}_mia_{label}.json.

Usage:
    from src.attacks.membership_inference import run_mia_on_model, run_mia_sweep

    result = run_mia_on_model(
        model, X_train, y_train, X_test, y_test,
        target_model_name="logistic_regression",
        dataset="cic_ids2018",
        label_type="binary",
    )

    all_results = run_mia_sweep("cic_ids2018", label="binary")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.config import PATHS, SAMPLING_CFG, DP_SGD_CFG
from src.utils.logger import get_logger

log = get_logger(__name__)

_ATTACK_RESULT_TYPE = "mia"
_DEFAULT_N_ATTACK_SAMPLES = 5000
_ATTACK_TEST_RATIO = 0.30
# Only include raw per-class probabilities as attack features when n_classes is small.
# For large multiclass problems the raw probability vector dominates and the feature
# vector becomes very wide without adding proportional signal.
_N_CLASSES_PROB_CUTOFF = 20


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_attack_features(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """
    Extract black-box attack features from model outputs.

    Features (when predict_proba is available):
        max_conf   : max predicted probability — high for members (model is
                     more confident on records it was trained on)
        entropy    : prediction entropy — low for members
        loss       : cross-entropy loss for the true label — low for members
        correct    : 1 if prediction matches true label — higher for members

    When n_classes <= _N_CLASSES_PROB_CUTOFF, the full per-class probability
    vector is appended. It carries the complete signal for small output spaces.

    Falls back to correctness-only (shape (n, 1)) when predict_proba is absent
    or raises an exception.

    Args:
        model:  Fitted sklearn-compatible classifier.
        X:      Feature matrix (n_samples × n_features).
        y_true: True integer labels aligned with X rows.

    Returns:
        2D float64 array of shape (n_samples, n_attack_features).
    """
    y_pred = model.predict(X)
    correct = (y_pred == y_true).astype(np.float64)

    if not hasattr(model, "predict_proba"):
        return correct.reshape(-1, 1)

    try:
        y_prob = model.predict_proba(X)
    except Exception as exc:
        log.warning("predict_proba failed (%s) — using correctness-only features", exc)
        return correct.reshape(-1, 1)

    # Clip to avoid log(0) in entropy and loss
    y_prob_clipped = np.clip(y_prob, 1e-10, 1.0)

    max_conf = y_prob.max(axis=1)
    entropy = -np.sum(y_prob_clipped * np.log(y_prob_clipped), axis=1)

    n_samples = len(X)
    n_classes = y_prob.shape[1]
    y_idx = np.clip(y_true.astype(int), 0, n_classes - 1)
    true_class_prob = y_prob_clipped[np.arange(n_samples), y_idx]
    loss = -np.log(true_class_prob)

    base = np.column_stack([max_conf, entropy, loss, correct])
    if n_classes <= _N_CLASSES_PROB_CUTOFF:
        return np.column_stack([base, y_prob])
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# Attack dataset construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_attack_dataset(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples_per_class: int = _DEFAULT_N_ATTACK_SAMPLES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a balanced attack dataset for membership inference evaluation.

    Members (label=1) are sampled from the model's actual training set.
    Non-members (label=0) are sampled from the held-out test set.
    Both sets are capped at n_samples_per_class and balanced so the attack
    classifier starts at a 50% base rate.

    The same seed is used for all target models so that the member/non-member
    record selection is identical across baseline, DP-LR, and DP-SGD models.
    This makes the attack ROC-AUC values directly comparable.

    Args:
        model:               Fitted target classifier.
        X_train, y_train:    Training-split data used to fit the target model.
        X_test, y_test:      Held-out test split (never seen during target training).
        n_samples_per_class: Maximum members AND non-members in the dataset.
        seed:                Random seed for reproducible sampling.

    Returns:
        Tuple (X_attack, y_attack) where y_attack is binary: 1=member, 0=non-member.
    """
    rng = np.random.default_rng(seed)

    n_balanced = min(n_samples_per_class, len(X_train), len(X_test))

    member_idx = rng.choice(len(X_train), size=n_balanced, replace=False)
    non_member_idx = rng.choice(len(X_test), size=n_balanced, replace=False)

    feat_members = extract_attack_features(
        model, X_train[member_idx], y_train[member_idx]
    )
    feat_non_members = extract_attack_features(
        model, X_test[non_member_idx], y_test[non_member_idx]
    )

    X_attack = np.vstack([feat_members, feat_non_members])
    y_attack = np.concatenate([
        np.ones(n_balanced, dtype=int),
        np.zeros(n_balanced, dtype=int),
    ])

    log.info(
        "Attack dataset built: %d members + %d non-members = %d samples",
        n_balanced, n_balanced, len(y_attack),
    )
    return X_attack, y_attack


# ═══════════════════════════════════════════════════════════════════════════════
# Attack classifier
# ═══════════════════════════════════════════════════════════════════════════════

def run_attack_classifier(
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train and evaluate a logistic regression membership-inference meta-classifier.

    Splits the attack dataset 70/30, normalises attack features (fit on attack
    train only), fits a logistic regression, and evaluates on attack test.

    Args:
        X_attack: Attack feature matrix (n_attack_samples × n_attack_features).
        y_attack: Membership labels — 1=member, 0=non-member.
        seed:     Random seed for reproducible train/test split.

    Returns:
        Dict with:
            attack_accuracy, attack_precision, attack_recall, attack_f1,
            attack_roc_auc, attack_advantage (= AUC − 0.5),
            tpr_at_fpr_0_01, tpr_at_fpr_0_10,
            n_attack_train, n_attack_test.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_attack, y_attack,
        test_size=_ATTACK_TEST_RATIO,
        stratify=y_attack,
        random_state=seed,
    )

    # Normalise attack features; fit scaler on attack train only to avoid leakage
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = SKLogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    y_prob_pos = clf.predict_proba(X_te)[:, 1]

    attack_auc = float(roc_auc_score(y_te, y_prob_pos))
    fpr_curve, tpr_curve, _ = roc_curve(y_te, y_prob_pos)

    def _tpr_at_fpr(target: float) -> float:
        idxs = np.where(fpr_curve <= target)[0]
        return float(tpr_curve[idxs[-1]]) if len(idxs) > 0 else 0.0

    return {
        "attack_accuracy":  float(accuracy_score(y_te, y_pred)),
        "attack_precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "attack_recall":    float(recall_score(y_te, y_pred, zero_division=0)),
        "attack_f1":        float(f1_score(y_te, y_pred, zero_division=0)),
        "attack_roc_auc":   attack_auc,
        "attack_advantage": round(attack_auc - 0.5, 4),
        "tpr_at_fpr_0_01":  _tpr_at_fpr(0.01),
        "tpr_at_fpr_0_10":  _tpr_at_fpr(0.10),
        "n_attack_train":   int(len(y_tr)),
        "n_attack_test":    int(len(y_te)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Single-model MIA
# ═══════════════════════════════════════════════════════════════════════════════

def run_mia_on_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    target_model_name: str,
    dataset: str,
    label_type: str,
    epsilon: float | None = None,
    delta: float | None = None,
    n_attack_samples: int = _DEFAULT_N_ATTACK_SAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run a full black-box membership inference attack against one target model.

    Args:
        model:             Fitted sklearn-compatible target classifier.
        X_train, y_train:  Training data used to fit the target model.
        X_test, y_test:    Held-out test split (non-members).
        target_model_name: Internal model identifier (e.g. "logistic_regression").
        dataset:           Dataset name for result metadata.
        label_type:        "binary" or "multiclass".
        epsilon:           DP budget (None for non-private models).
        delta:             DP delta (None for non-private models).
        n_attack_samples:  Members + non-members per class.
        seed:              Random seed (should match the dataset-level seed).

    Returns:
        Result dict with metadata + attack metrics + target model test metrics.
    """
    log.info(
        "MIA on '%s' (ε=%s) for '%s' (%s)",
        target_model_name, epsilon, dataset, label_type,
    )

    # Compute target model's own test-split performance metrics
    target_metrics: dict[str, Any] = {}
    try:
        from src.evaluation.metrics import evaluate_model
        full_metrics = evaluate_model(model, X_test, y_test, label_type=label_type)
        # Keep only JSON-serialisable scalar values
        target_metrics = {
            k: v for k, v in full_metrics.items()
            if isinstance(v, (int, float)) or v is None
        }
    except Exception as exc:
        log.warning("Could not compute target model test metrics: %s", exc)

    X_attack, y_attack = build_attack_dataset(
        model, X_train, y_train, X_test, y_test,
        n_samples_per_class=n_attack_samples,
        seed=seed,
    )

    n_per_class = len(y_attack) // 2
    attack_metrics = run_attack_classifier(X_attack, y_attack, seed=seed)

    result: dict[str, Any] = {
        "dataset":       dataset,
        "label_type":    label_type,
        "target_model":  target_model_name,
        "epsilon":       epsilon,
        "delta":         delta,
        "attack_type":   "black_box_confidence",
        "n_members":     n_per_class,
        "n_non_members": n_per_class,
        "seed":          seed,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        **attack_metrics,
        **{f"target_{k}": v for k, v in target_metrics.items()},
    }

    log.info(
        "MIA complete — attack_roc_auc=%.3f, advantage=%.3f",
        result["attack_roc_auc"], result["attack_advantage"],
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Full sweep across all saved models
# ═══════════════════════════════════════════════════════════════════════════════

def run_mia_sweep(
    dataset_name: str,
    label: str = "binary",
    n_attack_samples: int = _DEFAULT_N_ATTACK_SAMPLES,
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    Run membership inference evaluation across every trained model on disk.

    Loads the dataset splits once, then iterates over all models returned by
    list_trained_models() (baselines, DP-LR, DP-SGD). Skips models whose MIA
    result already exists unless force=True.

    Args:
        dataset_name:     One of "cic_ids2018", "unsw_nb15".
        label:            "binary" or "multiclass".
        n_attack_samples: Max members / non-members per model.
        force:            Re-run even if a result already exists.

    Returns:
        Complete list of MIA result dicts for this dataset + label.
    """
    from src.data.sampler import get_split_arrays
    from src.models.model_utils import list_trained_models

    log.info("Loading split arrays for '%s' (label=%s)", dataset_name, label)
    splits = get_split_arrays(dataset_name, label=label)
    X_train, y_train = splits["train"]
    X_test, y_test   = splits["test"]

    seed = int(SAMPLING_CFG.get("random_seed", 42))

    existing = load_mia_results(dataset_name, label)
    existing_keys = {(r["target_model"], r.get("epsilon")) for r in existing}

    trained = list_trained_models(dataset_name, label)
    if not trained:
        log.warning(
            "No trained models found for '%s' (label=%s). "
            "Run training pipelines first.",
            dataset_name, label,
        )
        return []

    new_results: list[dict] = []

    for info in trained:
        model_name = info["model_name"]
        epsilon    = info.get("epsilon")
        key        = (model_name, epsilon)

        if key in existing_keys and not force:
            log.info("Skipping '%s' ε=%s — result already exists.", model_name, epsilon)
            continue

        try:
            model = joblib.load(info["path"])
        except Exception as exc:
            log.error("Could not load %s: %s — skipping.", info["path"], exc)
            continue

        # Determine delta from config for DP models
        delta: float | None = None
        if epsilon is not None and model_name == "dp_sgd":
            delta = float(DP_SGD_CFG.get("delta", 1e-5))

        try:
            result = run_mia_on_model(
                model, X_train, y_train, X_test, y_test,
                target_model_name=model_name,
                dataset=dataset_name,
                label_type=label,
                epsilon=epsilon,
                delta=delta,
                n_attack_samples=n_attack_samples,
                seed=seed,
            )
            new_results.append(result)
        except Exception as exc:
            log.error("MIA failed for '%s' ε=%s: %s", model_name, epsilon, exc)
            continue

    all_results = existing + new_results
    if new_results:
        save_mia_results(all_results, dataset_name, label)
        log.info(
            "MIA sweep done — %d new results, %d total.",
            len(new_results), len(all_results),
        )
    else:
        log.info("No new MIA results — all models already evaluated.")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Result persistence
# ═══════════════════════════════════════════════════════════════════════════════

def _mia_path(dataset_name: str, label: str) -> Path:
    return PATHS["results"] / f"{dataset_name}_{_ATTACK_RESULT_TYPE}_{label}.json"


def save_mia_results(
    results: list[dict[str, Any]],
    dataset_name: str,
    label: str = "binary",
) -> Path:
    """
    Save MIA results to JSON with upsert semantics (keyed by target_model + ε).

    Also writes a companion CSV for easy import into LaTeX / spreadsheets.

    Args:
        results:      List of attack result dicts to persist.
        dataset_name: Dataset key used to construct the filename.
        label:        "binary" or "multiclass".

    Returns:
        Path to the written JSON file.
    """
    path = _mia_path(dataset_name, label)
    PATHS["results"].mkdir(parents=True, exist_ok=True)

    # Merge with any results already on disk
    existing: dict[tuple, dict] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for r in json.load(fh):
                    existing[(r["target_model"], r.get("epsilon"))] = r
        except (json.JSONDecodeError, KeyError):
            log.warning("Could not parse existing MIA results file — overwriting.")

    for r in results:
        existing[(r["target_model"], r.get("epsilon"))] = r

    merged = list(existing.values())
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, default=_json_serialise)
    log.info("Saved %d MIA result(s) to %s", len(merged), path.name)

    # Companion CSV
    df = build_mia_table(merged)
    if not df.empty:
        csv_path = PATHS["results"] / f"{dataset_name}_{_ATTACK_RESULT_TYPE}_{label}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        log.info("MIA summary CSV → %s", csv_path.name)

    return path


def load_mia_results(
    dataset_name: str,
    label: str = "binary",
) -> list[dict[str, Any]]:
    """
    Load MIA results from JSON, or return [] if the file does not exist.
    """
    path = _mia_path(dataset_name, label)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    log.info("Loaded %d MIA result(s) from %s", len(results), path.name)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Table builders for display and export
# ═══════════════════════════════════════════════════════════════════════════════

def build_mia_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Build a flat DataFrame of MIA metrics suitable for Streamlit or CSV export.

    Columns: Model | ε | attack_roc_auc | attack_advantage | attack_accuracy |
             attack_f1 | tpr_at_fpr_0_01 | tpr_at_fpr_0_10 |
             target_f1_macro | target_roc_auc | n_members

    Rows are sorted by (Model, ε ascending), with non-private models first.
    """
    attack_keys = [
        "attack_accuracy", "attack_precision", "attack_recall", "attack_f1",
        "attack_roc_auc", "attack_advantage", "tpr_at_fpr_0_01", "tpr_at_fpr_0_10",
    ]
    target_keys = [
        "target_accuracy", "target_f1_macro", "target_roc_auc",
        "target_false_positive_rate", "target_detection_rate",
    ]

    rows = []
    for r in results:
        row: dict[str, Any] = {
            "Model":     _display_name(r.get("target_model", "")),
            "ε":         r.get("epsilon"),
            "n_members": r.get("n_members"),
        }
        for k in attack_keys:
            v = r.get(k)
            row[k] = round(v, 4) if isinstance(v, float) else v
        for k in target_keys:
            v = r.get(k)
            row[k] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["_sort_eps"] = df["ε"].apply(lambda x: -1.0 if x is None else float(x))
    df = (
        df.sort_values(["Model", "_sort_eps"])
        .drop(columns=["_sort_eps"])
        .reset_index(drop=True)
    )
    return df


def build_combined_dissertation_table(
    dataset_name: str,
    label: str = "binary",
) -> pd.DataFrame:
    """
    Build the combined privacy-utility-risk table for the dissertation.

    Columns: Model | ε | F1 macro | ROC-AUC (IDS) | FPR | Detection Rate |
             Attack ROC-AUC | Attack Advantage

    This table lets the dissertation show that lower ε (tighter DP) reduces
    both IDS performance and membership-inference attack success simultaneously,
    supporting the privacy-utility tradeoff narrative.
    """
    results = load_mia_results(dataset_name, label)
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        rows.append({
            "Model":            _display_name(r.get("target_model", "")),
            "ε":                r.get("epsilon"),
            "F1 macro":         _safe_round(r.get("target_f1_macro")),
            "ROC-AUC (IDS)":    _safe_round(r.get("target_roc_auc")),
            "FPR":              _safe_round(r.get("target_false_positive_rate")),
            "Detection Rate":   _safe_round(r.get("target_detection_rate")),
            "Attack ROC-AUC":   _safe_round(r.get("attack_roc_auc")),
            "Attack Advantage": _safe_round(r.get("attack_advantage")),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["_sort_eps"] = df["ε"].apply(lambda x: -1.0 if x is None else float(x))
    df = (
        df.sort_values(["Model", "_sort_eps"])
        .drop(columns=["_sort_eps"])
        .reset_index(drop=True)
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _display_name(model_name: str) -> str:
    return {
        "logistic_regression":    "Logistic Regression",
        "random_forest":          "Random Forest",
        "xgboost":                "XGBoost",
        "dp_logistic_regression": "DP-LR",
        "dp_random_forest":       "DP-RF",
        "dp_sgd":                 "DP-SGD",
    }.get(model_name, model_name)


def _safe_round(v: Any, places: int = 4) -> Any:
    return round(v, places) if isinstance(v, float) else v


def _json_serialise(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
