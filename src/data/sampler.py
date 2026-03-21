"""
sampler.py — Stratified subset creation and train/val/test splitting.

This module operates on Stage-A processed Parquet files produced by
preprocessor.py. It never reads raw CSVs.

Pipeline (Stage B):
    1. Load processed Parquet (data/processed/{dataset_name}.parquet)
    2. Draw a stratified sample by label_multiclass → subset DataFrame
    3. Stratified train/val/test split → add 'split' column
    4. Save to data/subsets/{dataset_name}_subset.parquet

The subset Parquet is the single fixed file used by all model training and
evaluation experiments. By keeping it fixed, the only variable changing
between the baseline model and DP experiments is the model itself — which
makes the privacy-utility comparison clean and methodologically sound.

Usage:
    from src.data.sampler import run_sampling_pipeline, get_split_arrays

    run_sampling_pipeline("cic_ids2018")
    splits = get_split_arrays("cic_ids2018", label="binary")
    X_train, y_train = splits["train"]
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.config import PATHS, SAMPLING_CFG
from src.utils.logger import get_logger
from src.data.loader import load_processed, load_subset
from src.data.label_mapping import encode_multiclass_labels

log = get_logger(__name__)

# ── Columns that are metadata / labels, not model features ────────────────────
_NON_FEATURE_COLS: frozenset[str] = frozenset({
    "label_raw",
    "label_binary",
    "label_multiclass",
    "dataset_source",
    "split",
})


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns that are model features (excludes metadata/labels)."""
    return [col for col in df.columns if col not in _NON_FEATURE_COLS]


# ═══════════════════════════════════════════════════════════════════════════════
# Core sampling functions
# ═══════════════════════════════════════════════════════════════════════════════

def create_stratified_subset(
    dataset_name: str,
    n_samples: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Draw a stratified sample from the processed Parquet for a dataset.

    Stratification is performed on 'label_multiclass' so each attack category
    is proportionally represented in the subset. This is important for
    cybersecurity datasets which have very uneven class sizes — without
    stratification, rare attack types would be underrepresented or absent.

    If a minority class has fewer rows than its proportional quota, all its
    rows are included and the deficit is filled from the majority class. This
    means the returned DataFrame may be slightly smaller than n_samples when
    minority classes are very rare.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        n_samples:    Total target rows in the subset.
                      Defaults to sampling.subset_size from config.yaml.
        seed:         Random seed. Defaults to sampling.random_seed.

    Returns:
        Stratified sample DataFrame without the 'split' column.
        Call add_train_val_test_split() next.
    """
    n_samples = n_samples or SAMPLING_CFG["subset_size"]
    seed      = seed      or SAMPLING_CFG["random_seed"]

    log.info(
        "Creating stratified subset for '%s': target=%d rows, seed=%d",
        dataset_name, n_samples, seed
    )

    df = load_processed(dataset_name)
    total_rows = len(df)
    log.info("Loaded processed dataset: %d rows", total_rows)

    if n_samples >= total_rows:
        log.warning(
            "Requested n_samples (%d) >= total rows (%d). "
            "Returning full dataset without sampling.",
            n_samples, total_rows
        )
        return df.reset_index(drop=True)

    # ── Stratified sampling per class ─────────────────────────────────────────
    # Compute per-class quota proportional to the class size in the full dataset.
    class_counts   = df["label_multiclass"].value_counts()
    class_fractions = class_counts / total_rows
    class_quotas    = (class_fractions * n_samples).astype(int)

    sampled_parts: list[pd.DataFrame] = []
    for class_name, quota in class_quotas.items():
        class_df = df[df["label_multiclass"] == class_name]
        actual_quota = min(quota, len(class_df))  # can't sample more than exists

        sampled = class_df.sample(n=actual_quota, random_state=seed)
        sampled_parts.append(sampled)

        log.debug(
            "  Class '%s': quota=%d, available=%d, sampled=%d",
            class_name, quota, len(class_df), actual_quota
        )

    subset = pd.concat(sampled_parts, ignore_index=True)

    # Shuffle so class groups are not contiguous in the final file
    subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    log.info(
        "Subset created: %d rows (target was %d). "
        "Class distribution:\n%s",
        len(subset),
        n_samples,
        subset["label_multiclass"].value_counts().to_string()
    )

    return subset


def add_train_val_test_split(
    df: pd.DataFrame,
    test_size: float | None = None,
    val_size: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Add a 'split' column assigning each row to 'train', 'val', or 'test'.

    All splits are stratified on 'label_multiclass' to preserve class
    proportions in every partition. This is critical for imbalanced datasets
    where a random split might place all instances of a rare class in test.

    Strategy (two-step stratified split):
        Step 1: Split off test set (e.g. 20% of total).
        Step 2: Split remainder into train and val (e.g. 75%/25% of remainder
                gives 60%/20% of total when test_size=val_size=0.20).

    Default split proportions from config.yaml:
        test_size = val_size = sampling.test_split (default 0.20)
        → 60% train / 20% val / 20% test

    Args:
        df:        Subset DataFrame from create_stratified_subset().
        test_size: Fraction for test set. Defaults to sampling.test_split.
        val_size:  Fraction for validation set. Defaults to test_size.
        seed:      Random seed. Defaults to sampling.random_seed.

    Returns:
        DataFrame with a new 'split' column (values: 'train', 'val', 'test').
    """
    test_size = test_size or SAMPLING_CFG["test_split"]
    val_size  = val_size  or test_size
    seed      = seed      or SAMPLING_CFG["random_seed"]

    log.info(
        "Splitting into train/val/test — test=%.0f%%, val=%.0f%%, train≈%.0f%%",
        test_size * 100, val_size * 100, (1 - test_size - val_size) * 100
    )

    # Stratify by label_multiclass throughout
    strat_col = df["label_multiclass"].astype(str)

    # Step 1: carve out test set
    idx_trainval, idx_test = train_test_split(
        df.index,
        test_size=test_size,
        stratify=strat_col,
        random_state=seed,
    )

    # Step 2: split remaining into train and val
    # val fraction relative to the train+val pool:
    #   val_size_relative = val_size / (1 - test_size)
    val_size_relative = val_size / (1.0 - test_size)

    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size_relative,
        stratify=strat_col.loc[idx_trainval],
        random_state=seed,
    )

    # Assign split labels
    df = df.copy()
    df["split"] = pd.Categorical(
        [""] * len(df),
        categories=["train", "val", "test"]
    )
    df.loc[idx_train, "split"] = "train"
    df.loc[idx_val,   "split"] = "val"
    df.loc[idx_test,  "split"] = "test"

    # Verify split sizes
    split_counts = df["split"].value_counts()
    log.info(
        "Split sizes — train: %d, val: %d, test: %d",
        split_counts.get("train", 0),
        split_counts.get("val", 0),
        split_counts.get("test", 0),
    )

    return df


def save_subset(df: pd.DataFrame, dataset_name: str) -> Path:
    """
    Write a subset DataFrame (including 'split' column) to Parquet.

    Output path: data/subsets/{dataset_name}_subset.parquet

    Args:
        df:           DataFrame with 'split' column present.
        dataset_name: Used to construct the output filename.

    Returns:
        Path to the written Parquet file.
    """
    if "split" not in df.columns:
        raise ValueError(
            "DataFrame does not have a 'split' column. "
            "Call add_train_val_test_split() before save_subset()."
        )

    subset_path = PATHS["subsets"] / f"{dataset_name}_subset.parquet"

    log.info("Writing subset Parquet to: %s", subset_path)
    df.to_parquet(subset_path, index=False, compression="snappy")
    log.info("Subset written (%.1f MB)", subset_path.stat().st_size / 1e6)

    return subset_path


# ═══════════════════════════════════════════════════════════════════════════════
# Array preparation for model training
# ═══════════════════════════════════════════════════════════════════════════════

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to the training split only.

    SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic
    examples for minority classes by interpolating between existing samples
    in feature space. It is applied AFTER the train/val/test split so that
    synthetic samples never leak into the validation or test sets.

    This is offered as a secondary class-imbalance strategy alongside the
    primary approach of class_weight='balanced' in sklearn estimators. Using
    both gives a natural comparison point in the dissertation.

    Important: SMOTE must be applied AFTER StandardScaler (i.e., on the
    already-scaled X_train) to ensure interpolation happens in a normalised
    feature space.

    Args:
        X_train: Scaled training feature matrix (n_samples × n_features).
        y_train: Training integer labels (binary or multiclass encoded).
        seed:    Random seed for reproducibility.

    Returns:
        Tuple (X_resampled, y_resampled) with minority classes augmented.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "Install it with: pip install imbalanced-learn"
        )

    seed = seed or SAMPLING_CFG["random_seed"]
    log.info(
        "Applying SMOTE to training split — original shape: %s",
        X_train.shape
    )

    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    log.info(
        "SMOTE complete — resampled shape: %s (added %d synthetic samples)",
        X_res.shape,
        len(X_res) - len(X_train)
    )
    return X_res, y_res


def get_split_arrays(
    dataset_name: str,
    label: str = "binary",
    scale: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load subset Parquet and return ready-to-use numpy arrays per split.

    This is the primary entry point for model training code. It handles:
    - Identifying feature columns vs label/metadata columns
    - Fitting StandardScaler on train only and applying to val and test
      (prevents data leakage from val/test statistics into training)
    - Integer-encoding multiclass labels via LabelEncoder (train-fitted)
    - Saving the fitted scaler and encoder to results/models/ for inference

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        label:        'binary'     → uses label_binary column (0/1 int)
                      'multiclass' → uses label_multiclass column (integer
                                     encoded via LabelEncoder)
        scale:        If True (default), apply StandardScaler. Set to False
                      only for tree-based models that are scale-invariant,
                      though scaling does not hurt them either.

    Returns:
        Dict with keys 'train', 'val', 'test'. Each value is a tuple (X, y)
        where X is a float64 numpy array and y is an int numpy array.

    Example:
        splits = get_split_arrays("cic_ids2018", label="binary")
        X_train, y_train = splits["train"]
        X_val,   y_val   = splits["val"]
        X_test,  y_test  = splits["test"]
    """
    if label not in ("binary", "multiclass"):
        raise ValueError(f"label must be 'binary' or 'multiclass', got '{label}'")

    df = load_subset(dataset_name)

    feature_cols = _get_feature_columns(df)
    log.info(
        "Preparing arrays for '%s' — %d feature columns, label='%s'",
        dataset_name, len(feature_cols), label
    )

    # ── Separate splits ───────────────────────────────────────────────────────
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    X_train_raw = train_df[feature_cols].values.astype(np.float64)
    X_val_raw   = val_df[feature_cols].values.astype(np.float64)
    X_test_raw  = test_df[feature_cols].values.astype(np.float64)

    # ── StandardScaler (fit on train only) ────────────────────────────────────
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val   = scaler.transform(X_val_raw)
        X_test  = scaler.transform(X_test_raw)

        # Persist scaler for inference / Streamlit demo
        scaler_path = PATHS["models"] / f"{dataset_name}_scaler.joblib"
        PATHS["models"].mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        log.info("Scaler saved to: %s", scaler_path)
    else:
        X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw

    # ── Label extraction ──────────────────────────────────────────────────────
    if label == "binary":
        y_train = train_df["label_binary"].values.astype(np.int32)
        y_val   = val_df["label_binary"].values.astype(np.int32)
        y_test  = test_df["label_binary"].values.astype(np.int32)

    else:  # multiclass — integer-encode via LabelEncoder (fit on train only)
        y_train_enc, encoder = encode_multiclass_labels(train_df["label_multiclass"])
        y_val_enc,   _       = encode_multiclass_labels(val_df["label_multiclass"],   encoder)
        y_test_enc,  _       = encode_multiclass_labels(test_df["label_multiclass"],  encoder)

        y_train, y_val, y_test = y_train_enc, y_val_enc, y_test_enc

        # Persist encoder so class names can be recovered at evaluation time
        encoder_path = PATHS["models"] / f"{dataset_name}_label_encoder.joblib"
        joblib.dump(encoder, encoder_path)
        log.info("Label encoder saved to: %s", encoder_path)

    log.info(
        "Arrays ready — train: %s, val: %s, test: %s",
        X_train.shape, X_val.shape, X_test.shape
    )

    return {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_sampling_pipeline(
    dataset_name: str,
    force: bool = False,
) -> None:
    """
    Orchestrate the full processed → subset pipeline for one dataset.

    This is the top-level function to call from the Streamlit UI or a script
    after run_preprocessing_pipeline() has produced the processed Parquet.

    Execution order:
        1. Check if subset Parquet already exists — skip unless force=True.
        2. create_stratified_subset()
        3. add_train_val_test_split()
        4. save_subset()
        5. Log final class distribution per split for verification.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        force:        Re-run even if subset Parquet already exists.

    Returns:
        None. Side effect: writes {dataset_name}_subset.parquet to data/subsets/.
    """
    subset_path = PATHS["subsets"] / f"{dataset_name}_subset.parquet"

    if subset_path.exists() and not force:
        log.info(
            "Subset Parquet already exists for '%s'. "
            "Pass force=True to re-create. Skipping.",
            dataset_name
        )
        return

    log.info("=" * 70)
    log.info("Starting sampling pipeline for: %s", dataset_name)
    log.info("=" * 70)

    subset = create_stratified_subset(dataset_name)
    subset = add_train_val_test_split(subset)
    save_subset(subset, dataset_name)

    # ── Verify class distribution per split ───────────────────────────────────
    log.info("Class distribution per split:")
    for split_name in ["train", "val", "test"]:
        split_df = subset[subset["split"] == split_name]
        dist = split_df["label_multiclass"].value_counts()
        log.info("  [%s] %d rows:\n%s", split_name, len(split_df), dist.to_string())

    log.info("Sampling pipeline complete for '%s'", dataset_name)
