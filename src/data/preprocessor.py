"""
preprocessor.py — Cleaning, normalisation, and Stage A pipeline orchestration.

Pipeline stages handled here (in execution order):
    1.  Normalise column names          (strip whitespace, lowercase, underscores)
    2.  Drop intra-chunk duplicate rows
    3.  Replace ±Inf with NaN           (CICFlowMeter ratio columns)
    4.  Normalise labels                (delegated to label_mapping.py)
    5.  Add dataset_source column
    6.  Coerce feature columns to float32
    --- (per-chunk work ends here — chunks accumulated then concatenated) ---
    7.  Drop columns with >60% NaN      (global pass on full dataset)
    8.  Impute remaining NaN            (median for numeric, mode for categorical)
    9.  Global deduplication
    10. Save to data/processed/{dataset_name}.parquet
    11. Save dropped column list to data/processed/{dataset_name}_dropped_cols.json

No sampling or train/test splitting happens here — that lives in sampler.py.

Memory note:
    Chunks are accumulated in a Python list, then pd.concat() is called once.
    For the full CIC-IDS2018 dataset (~80 float32 columns, ~16 M rows) this
    may require ~5–8 GB RAM. If that is a constraint, reduce chunk_size so
    fewer rows are held per chunk, or process day-files individually and
    concatenate the Parquet files at the end.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import PATHS, get_dataset_config
from src.utils.logger import get_logger
from src.data.label_mapping import apply_cic_label_mapping, apply_unsw_label_mapping
from src.data.loader import list_raw_files, read_csv_chunked

log = get_logger(__name__)

# ── Protected columns ─────────────────────────────────────────────────────────
# These columns must never be cast to float32 or imputed as numeric values.
# They are metadata / label columns added by the pipeline.
_PROTECTED_COLS: frozenset[str] = frozenset({
    "label_raw",
    "label_binary",
    "label_multiclass",
    "dataset_source",
    "split",
})


# ═══════════════════════════════════════════════════════════════════════════════
# Per-chunk transformation functions
# Each function is a pure transformation: DataFrame in → DataFrame out.
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise all column names to lowercase with underscores.

    CIC-IDS2018 raw headers often have leading/trailing spaces and mixed
    capitalisation (e.g. ' Flow Duration', 'Bwd Packet Length Max').
    This step ensures all downstream code uses a consistent naming convention.

    Transformation applied:
        ' Flow Duration' → 'flow_duration'
        'Bwd Packet Length Max' → 'bwd_packet_length_max'

    Args:
        df: Raw chunk DataFrame straight from the CSV reader.

    Returns:
        DataFrame with cleaned column names. Column data is unchanged.
    """
    original_names = df.columns.tolist()

    df.columns = (
        df.columns
        .str.strip()                                     # remove surrounding whitespace
        .str.lower()                                     # to lowercase
        .str.replace(r"\s+", "_", regex=True)           # spaces → underscores
        .str.replace(r"[^\w]", "_", regex=True)         # non-word chars → underscores
        .str.replace(r"_+", "_", regex=True)            # collapse consecutive underscores
        .str.strip("_")                                  # trim leading/trailing underscores
    )

    # Log any name changes at debug level for traceability
    changed = [
        (old, new) for old, new in zip(original_names, df.columns) if old != new
    ]
    if changed:
        log.debug("Renamed %d column(s): %s", len(changed), changed[:5])

    return df


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully-duplicate rows within a chunk.

    Exact duplicates (all columns identical) are dropped, keeping the first
    occurrence. Note that duplicates spanning chunk boundaries are not caught
    here — a global deduplication pass runs after all chunks are concatenated
    (step 9 in the orchestrator).

    Args:
        df: Chunk DataFrame with normalised column names.

    Returns:
        DataFrame with intra-chunk duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        log.debug("Dropped %d duplicate row(s) in chunk", dropped)
    return df


def replace_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace +Inf and -Inf with NaN across all numeric columns.

    CICFlowMeter (used to generate CIC-IDS2018) computes many features as
    ratios (e.g. bytes-per-packet). Division-by-zero flows produce Inf values
    that standard sklearn estimators cannot handle. Replacing with NaN lets
    them be treated uniformly by the imputation step that follows.

    Args:
        df: Chunk DataFrame (column names already normalised).

    Returns:
        DataFrame with np.inf and -np.inf replaced by np.nan in numeric cols.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols].values).sum()
    if inf_count > 0:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        log.debug("Replaced %d Inf value(s) with NaN", inf_count)
    return df


def normalize_labels(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Apply the shared label taxonomy to a raw chunk.

    Delegates to label_mapping.py based on dataset_name. After this step:
        - label_raw        contains the original label string (preserved for audit)
        - label_multiclass contains the normalised attack category
        - label_binary     contains 0 (benign) or 1 (attack) as int8
        - original label column(s) are dropped

    Args:
        df:           Chunk DataFrame with normalised column names.
        dataset_name: Key matching config.yaml 'datasets' entry.

    Returns:
        DataFrame with the three standard label columns and the original
        label column(s) removed.

    Raises:
        ValueError: If dataset_name is not one of the supported values.
    """
    ds_cfg = get_dataset_config(dataset_name)
    # After normalize_column_names, the raw label column name is lowercased
    raw_label_col = ds_cfg["label_column"].strip().lower().replace(" ", "_")

    if dataset_name == "cic_ids2018":
        df = apply_cic_label_mapping(df, raw_col=raw_label_col)

    elif dataset_name == "unsw_nb15":
        # UNSW-NB15 requires both 'label' (binary) and 'attack_cat' (category)
        df = apply_unsw_label_mapping(df)

    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            "Add a label mapping branch for this dataset in preprocessor.py "
            "and a map in label_mapping.py."
        )

    return df


def add_dataset_source(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Append a 'dataset_source' column for provenance tracking.

    This column is useful when:
    - Displaying results across both datasets in the Streamlit app.
    - Writing result tables that compare CIC-IDS2018 vs UNSW-NB15 outcomes.

    Args:
        df:           DataFrame chunk.
        dataset_name: Key matching config.yaml 'datasets' entry.

    Returns:
        DataFrame with a new 'dataset_source' column (dtype: category).
    """
    df = df.copy()
    df["dataset_source"] = pd.Categorical([dataset_name] * len(df))
    return df


def coerce_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all non-protected columns to float32 to halve memory usage.

    float32 provides ~7 significant decimal digits, which is more than
    sufficient for the feature magnitudes in network flow datasets.
    float64 is the pandas default but wastes 50% of memory unnecessarily.

    Columns that cannot be coerced to a number (e.g. IP address strings,
    categorical protocol names) are converted to NaN via errors='coerce'.
    These NaN-heavy columns will be caught by drop_high_null_columns in the
    global pass.

    Protected metadata and label columns are excluded from casting.

    Args:
        df: DataFrame chunk after label normalisation.

    Returns:
        DataFrame with numeric feature columns cast to float32.
    """
    feature_cols = [col for col in df.columns if col not in _PROTECTED_COLS]

    cast_count = 0
    for col in feature_cols:
        if df[col].dtype == object:
            # Try numeric conversion; non-parseable values become NaN
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            cast_count += 1
        elif df[col].dtype != np.float32:
            df[col] = df[col].astype("float32")
            cast_count += 1

    log.debug("Coerced %d column(s) to float32", cast_count)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Global-pass functions (operate on fully-concatenated DataFrame)
# ═══════════════════════════════════════════════════════════════════════════════

def drop_high_null_columns(
    df: pd.DataFrame,
    threshold: float = 0.6,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns where more than `threshold` fraction of values are NaN.

    Called on the fully-concatenated DataFrame (not per-chunk) so that null
    rates reflect the full dataset rather than one chunk's composition.

    A threshold of 0.6 means a column must have >60% NaN to be dropped.
    This tolerates features that are sparsely populated for certain attack
    types but still informative overall.

    The list of dropped columns is also returned so sampler.py can apply
    the same exclusions consistently when building subsets.

    Args:
        df:        Full concatenated DataFrame after chunk processing.
        threshold: NaN fraction above which a column is dropped (default 0.6).

    Returns:
        Tuple of (cleaned DataFrame, list of dropped column names).
    """
    feature_cols = [col for col in df.columns if col not in _PROTECTED_COLS]
    null_fractions = df[feature_cols].isnull().mean()
    cols_to_drop = null_fractions[null_fractions > threshold].index.tolist()

    if cols_to_drop:
        log.info(
            "Dropping %d high-null column(s) (threshold=%.0f%%): %s",
            len(cols_to_drop),
            threshold * 100,
            cols_to_drop,
        )
        df = df.drop(columns=cols_to_drop)
    else:
        log.info("No high-null columns found at threshold=%.0f%%", threshold * 100)

    return df, cols_to_drop


def impute_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
) -> pd.DataFrame:
    """
    Fill remaining NaN values after infinite replacement and column dropping.

    Strategy by column type:
    - Numeric (float32):  fill with column median (robust to IDS outliers).
    - Object / category:  fill with column mode (most frequent value).

    Note on leakage: this imputation is computed on the full Stage-A dataset
    before the train/val/test split. For a more rigorous experimental setup,
    fit imputation parameters on the training split only and apply to val/test.
    The sklearn Pipeline in the model training step handles this correctly for
    the actual experiments; Stage-A imputation is a coarser pre-processing pass.

    Args:
        df:               Full concatenated DataFrame.
        numeric_strategy: 'median' (default) or 'mean'.

    Returns:
        DataFrame with no remaining NaN values in feature columns.
    """
    feature_cols = [col for col in df.columns if col not in _PROTECTED_COLS]

    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    object_cols  = df[feature_cols].select_dtypes(include=["object", "category"]).columns

    # ── Numeric imputation ─────────────────────────────────────────────────────
    total_numeric_nans = df[numeric_cols].isnull().sum().sum()
    if total_numeric_nans > 0:
        if numeric_strategy == "median":
            fill_values = df[numeric_cols].median()
        else:
            fill_values = df[numeric_cols].mean()

        df[numeric_cols] = df[numeric_cols].fillna(fill_values)
        log.info(
            "Imputed %d NaN(s) in numeric columns using %s",
            total_numeric_nans,
            numeric_strategy,
        )

    # ── Categorical / object imputation ────────────────────────────────────────
    for col in object_cols:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            mode_series = df[col].mode()
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series.iloc[0])
                log.debug("Imputed %d NaN(s) in '%s' with mode '%s'", nan_count, col, mode_series.iloc[0])

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_preprocessing_pipeline(
    dataset_name: str,
    chunk_size: int = 50_000,
    force: bool = False,
) -> None:
    """
    Orchestrate the full raw CSV → processed Parquet pipeline for one dataset.

    This is the top-level function to call from the Streamlit UI or a script.
    It is idempotent: if the processed Parquet already exists and force=False,
    the function logs a skip message and returns immediately.

    Execution order:
        Per CSV file (chunked):
            1. normalize_column_names
            2. drop_duplicate_rows
            3. replace_infinite_values
            4. normalize_labels       (adds label_* cols, drops original label col)
            5. add_dataset_source     (adds dataset_source col)
            6. coerce_feature_dtypes  (feature cols → float32)
            7. Accumulate chunk in list
        After all chunks collected:
            8. pd.concat() all chunks
            9. drop_high_null_columns (global pass)
           10. impute_missing_values  (global pass)
           11. drop_duplicates()      (global cross-chunk dedup)
           12. Save to data/processed/{dataset_name}.parquet
           13. Save dropped column names to
               data/processed/{dataset_name}_dropped_cols.json

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
                      Must be "cic_ids2018" or "unsw_nb15".
        chunk_size:   Rows per CSV reading chunk. Default 50 000.
                      Increase for faster I/O; decrease if RAM is limited.
        force:        If True, re-run even when the processed Parquet exists.
                      Default False prevents accidental re-processing.

    Returns:
        None. Side effects:
            - Writes {dataset_name}.parquet to data/processed/
            - Writes {dataset_name}_dropped_cols.json to data/processed/
    """
    output_parquet = PATHS["processed_data"] / f"{dataset_name}.parquet"
    dropped_cols_path = PATHS["processed_data"] / f"{dataset_name}_dropped_cols.json"

    # ── Idempotency check ─────────────────────────────────────────────────────
    if output_parquet.exists() and not force:
        log.info(
            "Processed Parquet already exists for '%s'. "
            "Pass force=True to re-run. Skipping.",
            dataset_name,
        )
        return

    log.info("=" * 70)
    log.info("Starting preprocessing pipeline for: %s", dataset_name)
    log.info("=" * 70)

    # ── Locate raw files ──────────────────────────────────────────────────────
    raw_files = list_raw_files(dataset_name)
    log.info("Processing %d CSV file(s)", len(raw_files))

    # ── Per-chunk processing pass ─────────────────────────────────────────────
    all_chunks: list[pd.DataFrame] = []
    total_rows_raw = 0
    total_rows_after_dedup = 0

    for file_path in raw_files:
        log.info("Processing file: %s", file_path.name)
        file_chunks: list[pd.DataFrame] = []

        for chunk in read_csv_chunked(file_path, dataset_name, chunk_size):
            total_rows_raw += len(chunk)

            # Apply per-chunk transformations in order
            chunk = normalize_column_names(chunk)
            chunk = drop_duplicate_rows(chunk)
            chunk = replace_infinite_values(chunk)
            chunk = normalize_labels(chunk, dataset_name)
            chunk = add_dataset_source(chunk, dataset_name)
            chunk = coerce_feature_dtypes(chunk)

            file_chunks.append(chunk)
            total_rows_after_dedup += len(chunk)

        all_chunks.extend(file_chunks)
        log.info("File '%s' processed — %d chunks accumulated", file_path.name, len(file_chunks))

    if not all_chunks:
        raise RuntimeError(
            f"No data was read for dataset '{dataset_name}'. "
            "Check that the raw CSV files are present and non-empty."
        )

    # ── Concatenate all chunks ────────────────────────────────────────────────
    log.info("Concatenating %d chunk(s) into a single DataFrame...", len(all_chunks))
    df = pd.concat(all_chunks, ignore_index=True)
    log.info("Concatenated shape: %d rows × %d columns", *df.shape)

    # Free the chunk list from memory — we only need df from here
    del all_chunks

    # ── Global deduplication (catches cross-chunk duplicates) ─────────────────
    before_global_dedup = len(df)
    df = df.drop_duplicates()
    cross_chunk_dupes = before_global_dedup - len(df)
    if cross_chunk_dupes:
        log.info("Global dedup removed %d cross-chunk duplicate row(s)", cross_chunk_dupes)

    # ── Drop high-null columns ────────────────────────────────────────────────
    df, dropped_cols = drop_high_null_columns(df, threshold=0.6)

    # ── Impute remaining NaN values ───────────────────────────────────────────
    df = impute_missing_values(df, numeric_strategy="median")

    # ── Log final shape and class distribution ─────────────────────────────────
    log.info("Final cleaned shape: %d rows × %d columns", *df.shape)
    log.info("Raw rows read: %d | After intra-chunk dedup: %d | After global dedup: %d",
             total_rows_raw, total_rows_after_dedup, len(df))

    if "label_multiclass" in df.columns:
        dist = df["label_multiclass"].value_counts()
        log.info("Class distribution:\n%s", dist.to_string())

    # ── Save processed Parquet ────────────────────────────────────────────────
    log.info("Writing processed Parquet to: %s", output_parquet)
    df.to_parquet(output_parquet, index=False, compression="snappy")
    log.info("Parquet written successfully (%.1f MB)", output_parquet.stat().st_size / 1e6)

    # ── Save dropped column list for downstream consistency ───────────────────
    with open(dropped_cols_path, "w", encoding="utf-8") as fh:
        json.dump(dropped_cols, fh, indent=2)
    log.info("Dropped column list written to: %s", dropped_cols_path)

    log.info("Preprocessing pipeline complete for '%s'", dataset_name)
