"""
model_utils.py — Model registry and pipeline status utilities.

Provides a single source of truth for what has been trained and what still
needs to run. Used by both CLI scripts and Streamlit pages to:
    - Check whether a processed Parquet / subset / trained model exists.
    - List all saved models for a dataset.
    - Load a model or scaler by name without knowing the exact filename.
    - Report a full pipeline status dict that drives the Streamlit UI.

No training logic lives here — this module only reads from disk.
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

from src.utils.config import PATHS, PRIVACY_CFG
from src.utils.logger import get_logger
from src.models.baseline import BASELINE_MODELS, _model_path as _baseline_path
from src.models.dp_model  import _dp_model_path, DP_MODEL_NAME

log = get_logger(__name__)

SUPPORTED_DATASETS = ["cic_ids2018", "unsw_nb15"]
SUPPORTED_LABELS   = ["binary", "multiclass"]


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline status
# ═══════════════════════════════════════════════════════════════════════════════

def get_pipeline_status(dataset_name: str, label: str = "binary") -> dict[str, Any]:
    """
    Return a dict describing what pipeline stages have been completed.

    Used by Streamlit pages to decide which UI controls to enable and to
    show informative status messages guiding the user through the pipeline.

    Args:
        dataset_name: One of SUPPORTED_DATASETS.
        label:        'binary' or 'multiclass'.

    Returns:
        Dict with boolean flags for each pipeline stage:
        {
          "raw_files_exist":      bool,   # CSVs present in data/raw/
          "processed_exists":     bool,   # Stage-A Parquet exists
          "subset_exists":        bool,   # Stage-B subset Parquet exists
          "baselines_trained":    dict,   # {model_name: bool} for each baseline
          "dp_trained":           dict,   # {epsilon: bool} for each ε value
          "results_exist": {
              "baselines": bool,
              "dp_sweep":  bool,
              "combined":  bool,
          }
        }
    """
    raw_dir        = PATHS["raw_data"] / dataset_name
    processed_path = PATHS["processed_data"] / f"{dataset_name}.parquet"
    subset_path    = PATHS["subsets"]        / f"{dataset_name}_subset.parquet"

    raw_files = list(raw_dir.glob("*.csv")) if raw_dir.exists() else []

    # Baseline model status
    baselines_trained = {
        model: _baseline_path(dataset_name, model, label).exists()
        for model in BASELINE_MODELS
    }

    # DP model status per epsilon value
    epsilon_values = PRIVACY_CFG.get("epsilon_values", [])
    dp_trained = {
        eps: _dp_model_path(dataset_name, eps, label).exists()
        for eps in epsilon_values
    }

    # Results JSON status
    results_dir = PATHS["results"]
    results_exist = {
        "baselines": (results_dir / f"{dataset_name}_baselines_{label}.json").exists(),
        "dp_sweep":  (results_dir / f"{dataset_name}_dp_sweep_{label}.json").exists(),
        "combined":  (results_dir / f"{dataset_name}_combined_{label}.json").exists(),
    }

    return {
        "raw_files_exist":   len(raw_files) > 0,
        "raw_file_count":    len(raw_files),
        "processed_exists":  processed_path.exists(),
        "subset_exists":     subset_path.exists(),
        "baselines_trained": baselines_trained,
        "dp_trained":        dp_trained,
        "results_exist":     results_exist,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(
    dataset_name: str,
    model_name: str,
    label: str = "binary",
    epsilon: float | None = None,
) -> Any:
    """
    Load any trained model by name.

    Handles both baseline models (epsilon=None) and DP models (epsilon=float)
    through a unified interface, avoiding the need for callers to know the
    exact path convention.

    Args:
        dataset_name: One of SUPPORTED_DATASETS.
        model_name:   For baselines: one of BASELINE_MODELS.
                      For DP:       DP_MODEL_NAME ('dp_logistic_regression').
        label:        'binary' or 'multiclass'.
        epsilon:      Required when model_name is DP_MODEL_NAME.

    Returns:
        Fitted sklearn-compatible model.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
        ValueError:        If epsilon is required but not provided.
    """
    if model_name == DP_MODEL_NAME:
        if epsilon is None:
            raise ValueError("epsilon must be provided when loading a DP model.")
        path = _dp_model_path(dataset_name, epsilon, label)
    else:
        path = _baseline_path(dataset_name, model_name, label)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train it first."
        )

    log.info("Loading model from %s", path)
    return joblib.load(path)


def load_scaler(dataset_name: str) -> Any:
    """
    Load the StandardScaler fitted for this dataset's training split.

    The scaler is saved by sampler.get_split_arrays() and is needed when
    running inference on new data or when displaying feature-space statistics.

    Args:
        dataset_name: One of SUPPORTED_DATASETS.

    Returns:
        Fitted sklearn StandardScaler.

    Raises:
        FileNotFoundError: If the scaler has not been created yet.
    """
    scaler_path = PATHS["models"] / f"{dataset_name}_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Run sampler.run_sampling_pipeline() first."
        )
    return joblib.load(scaler_path)


def list_trained_models(
    dataset_name: str,
    label: str = "binary",
) -> list[dict[str, Any]]:
    """
    Return a list of dicts describing every trained model found on disk.

    Each dict has keys: model_name, epsilon, label, path, size_kb.
    Ordered: baselines first, then DP models by ascending epsilon.

    Args:
        dataset_name: One of SUPPORTED_DATASETS.
        label:        'binary' or 'multiclass'.

    Returns:
        List of model descriptor dicts.
    """
    found = []

    # Check baseline models
    for model_name in BASELINE_MODELS:
        path = _baseline_path(dataset_name, model_name, label)
        if path.exists():
            found.append({
                "model_name": model_name,
                "epsilon":    None,
                "label":      label,
                "path":       str(path),
                "size_kb":    round(path.stat().st_size / 1024, 1),
            })

    # Check DP models across all configured epsilon values
    for eps in sorted(PRIVACY_CFG.get("epsilon_values", [])):
        path = _dp_model_path(dataset_name, eps, label)
        if path.exists():
            found.append({
                "model_name": DP_MODEL_NAME,
                "epsilon":    eps,
                "label":      label,
                "path":       str(path),
                "size_kb":    round(path.stat().st_size / 1024, 1),
            })

    log.info(
        "Found %d trained model(s) for '%s' (label=%s)",
        len(found), dataset_name, label
    )
    return found


def get_subset_stats(dataset_name: str) -> dict[str, Any] | None:
    """
    Return basic statistics about the working subset for display in the UI.

    Args:
        dataset_name: One of SUPPORTED_DATASETS.

    Returns:
        Dict with row counts per split and class distribution, or None if
        the subset Parquet does not exist.
    """
    import pandas as pd

    subset_path = PATHS["subsets"] / f"{dataset_name}_subset.parquet"
    if not subset_path.exists():
        return None

    df = pd.read_parquet(subset_path, columns=["split", "label_multiclass", "label_binary"])

    stats: dict[str, Any] = {
        "total_rows": len(df),
        "n_features": None,   # computed below
        "split_counts": df["split"].value_counts().to_dict(),
        "class_distribution": df["label_multiclass"].value_counts().to_dict(),
        "binary_distribution": df["label_binary"].value_counts().to_dict(),
    }

    # Count features separately (avoid reading all columns above)
    full_df = pd.read_parquet(subset_path, columns=None)
    _NON_FEAT = {"label_raw", "label_binary", "label_multiclass", "dataset_source", "split"}
    stats["n_features"] = sum(1 for c in full_df.columns if c not in _NON_FEAT)

    return stats
