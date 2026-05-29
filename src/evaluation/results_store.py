"""
results_store.py — Persist and reload experiment results.

Every training run (baseline or DP) produces a result dict. This module
saves those dicts to JSON files under results/ so that:
    1. Streamlit pages can display results without re-running experiments.
    2. Results survive session resets and kernel restarts.
    3. You can iterate on the Streamlit UI without re-training models.

File naming convention:
    results/{dataset_name}_baselines_{label}.json       — all baseline results
    results/{dataset_name}_dp_sweep_{label}.json        — full epsilon sweep
    results/{dataset_name}_combined_{label}.json        — merged for comparison

All JSON files store a list of result dicts. Appending a new result to an
existing file merges by (model_name, epsilon) key — re-running one epsilon
updates only that entry without wiping other results.

Usage:
    from src.evaluation.results_store import save_results, load_results, build_comparison_table

    save_results(baseline_results, "cic_ids2018", "baselines", label="binary")
    save_results(dp_results,       "cic_ids2018", "dp_sweep",  label="binary")

    all_results = load_results("cic_ids2018", "combined", label="binary")
    df = build_comparison_table(all_results, split="test")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import PATHS
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Path helpers ───────────────────────────────────────────────────────────────

def _results_path(dataset_name: str, result_type: str, label: str) -> Path:
    """Construct the JSON path for a results file."""
    return PATHS["results"] / f"{dataset_name}_{result_type}_{label}.json"


def _result_key(result: dict) -> tuple:
    """
    Unique key for a result dict used to detect duplicates during upsert.

    A result is uniquely identified by (model_name, epsilon).
    Non-private baselines have epsilon=None; DP models have a float epsilon.
    """
    return (result.get("model_name", ""), result.get("epsilon"))


# ═══════════════════════════════════════════════════════════════════════════════
# Save and load
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(
    results: list[dict[str, Any]],
    dataset_name: str,
    result_type: str,
    label: str = "binary",
) -> Path:
    """
    Save (or upsert) a list of result dicts to a JSON file.

    If the JSON file already exists, existing results are loaded and any
    result in the new list is merged by its (model_name, epsilon) key.
    This means re-running a single epsilon does not erase other results.

    Args:
        results:     List of result dicts from train_baseline() or
                     run_epsilon_sweep().
        dataset_name: Key matching config.yaml 'datasets' entry.
        result_type: Short label for the file: 'baselines', 'dp_sweep', etc.
        label:       'binary' or 'multiclass'.

    Returns:
        Path to the written JSON file.
    """
    path = _results_path(dataset_name, result_type, label)

    # Load existing results if file exists (for upsert behaviour)
    existing: dict[tuple, dict] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            existing = {_result_key(r): r for r in loaded}
            log.info("Loaded %d existing result(s) from %s for merge", len(existing), path.name)
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Could not parse existing results file (%s) — overwriting.", exc)

    # Upsert: new results overwrite existing entries with the same key
    for r in results:
        existing[_result_key(r)] = r

    merged = list(existing.values())

    PATHS["results"].mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, default=_json_serialise)

    log.info(
        "Saved %d result(s) to %s (total in file: %d)",
        len(results), path.name, len(merged)
    )
    return path


def load_results(
    dataset_name: str,
    result_type: str,
    label: str = "binary",
) -> list[dict[str, Any]]:
    """
    Load a results JSON file and return a list of result dicts.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        result_type: 'baselines', 'dp_sweep', or 'combined'.
        label:        'binary' or 'multiclass'.

    Returns:
        List of result dicts. Empty list if the file does not exist.
    """
    path = _results_path(dataset_name, result_type, label)

    if not path.exists():
        log.warning(
            "Results file not found: %s — returning empty list. "
            "Run the training pipeline first.",
            path
        )
        return []

    with open(path, "r", encoding="utf-8") as fh:
        results = json.load(fh)

    log.info("Loaded %d result(s) from %s", len(results), path.name)
    return results


def save_combined_results(
    dataset_name: str,
    label: str = "binary",
) -> Path:
    """
    Merge baseline, DP-LR sweep, and DP-SGD sweep results into a single
    combined JSON file.

    Reads baselines, dp_sweep, and dp_sgd_sweep JSON files for the given
    dataset and label, merges them, and writes a combined JSON. This single
    file is used by the Streamlit comparison page to build all summary tables
    and tradeoff plots.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        label:        'binary' or 'multiclass'.

    Returns:
        Path to the written combined JSON file.
    """
    baselines  = load_results(dataset_name, "baselines",     label)
    dp_sweep   = load_results(dataset_name, "dp_sweep",      label)
    dp_rf      = load_results(dataset_name, "dp_rf_sweep",   label)
    dp_sgd     = load_results(dataset_name, "dp_sgd_sweep",  label)
    combined   = baselines + dp_sweep + dp_rf + dp_sgd

    if not combined:
        log.warning(
            "No results to combine for '%s' (label=%s). "
            "Run training pipelines first.",
            dataset_name, label
        )

    path = save_results(combined, dataset_name, "combined", label)
    log.info(
        "Combined results written (%d baseline + %d DP-LR + %d DP-RF + %d DP-SGD = %d total)",
        len(baselines), len(dp_sweep), len(dp_rf), len(dp_sgd), len(combined)
    )
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Table builders for display and export
# ═══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(
    results: list[dict[str, Any]],
    split: str = "test",
    metric_keys: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a flat comparison DataFrame from a list of result dicts.

    This is the primary function called by the Streamlit comparison page.
    It produces a DataFrame where each row is one experimental run and
    columns are the key metrics. The DataFrame can be displayed directly
    with st.dataframe() or exported to CSV / LaTeX.

    Column order is designed for dissertation table readability:
        Model | Dataset | ε | Accuracy | F1 (macro) | AUC | MCC | FPR | DR

    Args:
        results:     List of result dicts (baseline and/or DP).
        split:       Which split to extract metrics from ('val' or 'test').
        metric_keys: Specific metrics to include. If None, a default
                     dissertation-friendly selection is used.

    Returns:
        pd.DataFrame with one row per result, sorted by (dataset, epsilon).
    """
    if metric_keys is None:
        metric_keys = [
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "roc_auc",
            "mcc",
            "false_positive_rate",
            "detection_rate",
        ]

    rows = []
    for r in results:
        split_metrics = r.get("metrics", {}).get(split, {})

        row = {
            "Model":   _display_name(r.get("model_name", "")),
            "Dataset": r.get("dataset", ""),
            "Label":   r.get("label_type", ""),
            "ε":       r.get("epsilon"),          # None for non-private models
        }

        for key in metric_keys:
            val = split_metrics.get(key)
            # Round floats to 4 decimal places for clean display
            row[key] = round(val, 4) if isinstance(val, float) else val

        # Training time
        t = r.get("train_time_s")
        row["Train Time (s)"] = round(t, 1) if isinstance(t, float) else t

        # Privacy cost columns (present only in augmented dp results)
        if "metric_loss_pct" in r:
            row["F1 loss vs LR (%)"] = r["metric_loss_pct"]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort: non-private baselines first (ε=None), then DP by ascending ε
    df["_sort_eps"] = df["ε"].apply(lambda x: -1.0 if x is None else float(x))
    df = df.sort_values(["Dataset", "_sort_eps"]).drop(columns=["_sort_eps"])
    df = df.reset_index(drop=True)

    return df


def build_tradeoff_table(
    dataset_name: str,
    label: str = "binary",
    metric_key: str = "f1_macro",
    split: str = "test",
) -> pd.DataFrame:
    """
    Build the privacy–utility tradeoff table for a single dataset.

    Loads the combined results, identifies the non-private LR baseline
    metric value, then computes absolute and percentage loss at each epsilon.
    This table maps directly to Table 2 in the dissertation results chapter.

    Columns:
        ε | F1 macro (test) | F1 loss vs LR | F1 loss % | AUC | MCC

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        label:        'binary' or 'multiclass'.
        metric_key:   Primary metric for tradeoff computation (default 'f1_macro').
        split:        Which split to use (default 'test').

    Returns:
        pd.DataFrame ready for st.dataframe() or export.
    """
    from src.evaluation.metrics import compute_privacy_utility_tradeoff

    all_results = load_results(dataset_name, "combined", label)
    if not all_results:
        log.warning("No combined results found for '%s'. Returning empty table.", dataset_name)
        return pd.DataFrame()

    # Separate baselines and DP results
    dp_results   = [r for r in all_results if r.get("epsilon") is not None]
    lr_baselines = [
        r for r in all_results
        if r.get("epsilon") is None and r.get("model_name") == "logistic_regression"
    ]

    if not lr_baselines:
        log.warning("No LR baseline result found — tradeoff percentages will be None.")
        baseline_val = None
    else:
        baseline_val = lr_baselines[0]["metrics"][split].get(metric_key)

    # Augment DP results with loss columns
    augmented = compute_privacy_utility_tradeoff(
        dp_results,
        baseline_metric_value=baseline_val or 0.0,
        metric_key=metric_key,
        split=split,
    )

    # Build display rows
    rows = []
    for r in sorted(augmented, key=lambda x: x["epsilon"]):
        m = r["metrics"][split]
        rows.append({
            "ε":                  r["epsilon"],
            f"{metric_key}":      round(m.get(metric_key, 0), 4),
            "AUC":                round(m["roc_auc"], 4) if m.get("roc_auc") else None,
            "MCC":                round(m.get("mcc", 0), 4),
            "Loss vs LR (abs)":   r.get("metric_loss_vs_baseline"),
            "Loss vs LR (%)":     r.get("metric_loss_pct"),
        })

    return pd.DataFrame(rows)


def export_to_csv(
    df: pd.DataFrame,
    filename: str,
) -> Path:
    """
    Write a results DataFrame to CSV in the results/ directory.

    Useful for generating dissertation appendix tables and feeding data
    into LaTeX or a separate plotting script.

    Args:
        df:       DataFrame to export.
        filename: Output filename (e.g. 'cic_ids2018_tradeoff_table.csv').

    Returns:
        Path to the written CSV.
    """
    output_path = PATHS["results"] / filename
    df.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Exported %d rows to %s", len(df), output_path)
    return output_path


# ── Internal utilities ─────────────────────────────────────────────────────────

def _display_name(model_name: str) -> str:
    """
    Convert an internal model name to a human-readable display string
    suitable for dissertation tables and Streamlit UI.
    """
    display_map = {
        "logistic_regression":    "Logistic Regression",
        "random_forest":          "Random Forest",
        "xgboost":                "XGBoost",
        "dp_logistic_regression": "DP-LR",
        "dp_random_forest":       "DP-RF",
        "dp_sgd":                 "DP-SGD",
    }
    return display_map.get(model_name, model_name)


def _json_serialise(obj: Any) -> Any:
    """
    Custom JSON serialiser for types that the standard library does not handle.

    Converts numpy scalars and arrays to native Python types so that result
    dicts containing numpy values can be written to JSON without error.
    """
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
