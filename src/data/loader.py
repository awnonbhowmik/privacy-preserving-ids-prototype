"""
loader.py — Raw CSV discovery and chunked reading.

Responsibilities:
- Locate raw CSV files for a given dataset (handles multi-file datasets)
- Stream CSV data in fixed-size chunks so large files never occupy full RAM
- Load already-processed Parquet files for downstream use

This module has no cleaning or transformation logic — that lives in
preprocessor.py. The separation makes each layer independently testable.

Dataset directory layout expected under data/raw/:
    data/raw/
        cic_ids2018/
            Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
            Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
            ...
        unsw_nb15/
            UNSW-NB15_1.csv
            UNSW-NB15_2.csv
            UNSW-NB15_3.csv
            UNSW-NB15_4.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from src.utils.config import PATHS, get_dataset_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# Columns known to be almost entirely empty in CIC-IDS2018 raw files.
# Excluding them at read time saves memory during the chunked pass.
# This list may need adjusting depending on which days of data are present.
_CIC_USELESS_COLS: list[str] = []  # populated dynamically if needed

# UNSW-NB15_1-4.csv are headerless; these are the 49 official column names
# from the NUSW-NB15_features.csv features description file.
UNSW_NB15_COLUMNS: list[str] = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]


def list_raw_files(dataset_name: str) -> list[Path]:
    """
    Return a sorted list of raw CSV paths for the given dataset.

    Each dataset's CSVs must be placed in a subdirectory named after the
    dataset key (e.g. data/raw/cic_ids2018/ or data/raw/unsw_nb15/).

    CIC-IDS2018 ships as one CSV per capture day; UNSW-NB15 ships as four
    numbered files. Both cases work by globbing *.csv in the subdirectory.

    Args:
        dataset_name: Key matching a top-level entry in config.yaml 'datasets'
                      (e.g. "cic_ids2018" or "unsw_nb15").

    Returns:
        Sorted list of Path objects pointing to CSV files.

    Raises:
        FileNotFoundError: If the dataset subdirectory does not exist, or if
                           no CSV files are found within it.
    """
    dataset_dir = PATHS["raw_data"] / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {dataset_dir}\n"
            f"Create the directory and place the {dataset_name} CSV files inside it."
        )

    files = sorted(dataset_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {dataset_dir}\n"
            f"Ensure {dataset_name} CSV files are placed directly in that directory."
        )

    log.info(
        "Found %d raw CSV file(s) for '%s': %s",
        len(files),
        dataset_name,
        [f.name for f in files],
    )
    return files


def read_csv_chunked(
    filepath: Path,
    dataset_name: str,
    chunk_size: int = 50_000,
) -> Iterator[pd.DataFrame]:
    """
    Yield successive DataFrame chunks from a single CSV file.

    Applies dataset-specific read settings (encoding, separator) from
    config.yaml. The yielded chunks are completely raw — no cleaning has been
    applied. Use preprocessor.py to clean each chunk as it arrives.

    Memory note: with ~80 float columns and chunk_size=50 000, each chunk is
    roughly 30–50 MB before any dtype optimisation. Increase chunk_size on
    machines with more RAM to speed up I/O.

    Args:
        filepath:     Path to a single CSV file.
        dataset_name: Key matching config.yaml 'datasets' entry.
        chunk_size:   Number of rows per yielded chunk. Default 50 000.

    Yields:
        pd.DataFrame chunks with original (uncleaned) column names and dtypes.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If chunk_size is not a positive integer.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")

    ds_cfg = get_dataset_config(dataset_name)
    encoding   = ds_cfg.get("encoding", "utf-8")
    separator  = ds_cfg.get("separator", ",")
    has_header = ds_cfg.get("has_header", True)

    # For headerless files, supply known column names and skip the header row.
    if not has_header and dataset_name == "unsw_nb15":
        read_kwargs: dict = dict(
            header=None,
            names=UNSW_NB15_COLUMNS,
        )
    else:
        read_kwargs = {}

    log.info(
        "Reading '%s' in chunks of %d rows (encoding=%s, has_header=%s)",
        filepath.name,
        chunk_size,
        encoding,
        has_header,
    )

    try:
        reader = pd.read_csv(
            filepath,
            encoding=encoding,
            sep=separator,
            chunksize=chunk_size,
            low_memory=False,      # avoid mixed-type inference warnings
            on_bad_lines="warn",   # log malformed rows instead of crashing
            **read_kwargs,
        )

        chunk_index = 0
        for chunk in reader:
            chunk_index += 1
            log.debug(
                "  [%s] chunk %d — %d rows, %d cols",
                filepath.name,
                chunk_index,
                len(chunk),
                len(chunk.columns),
            )
            yield chunk

        log.info("Finished reading '%s' (%d chunks)", filepath.name, chunk_index)

    except UnicodeDecodeError as exc:
        # CIC-IDS2018 files are latin-1; UNSW-NB15 is UTF-8.
        # A UnicodeDecodeError here usually means the wrong encoding is set.
        raise UnicodeDecodeError(
            exc.encoding, exc.object, exc.start, exc.end,
            f"Encoding error reading {filepath.name}. "
            f"Check the 'encoding' setting for '{dataset_name}' in config.yaml. "
            f"Current value: '{encoding}'"
        ) from exc


def load_processed(dataset_name: str) -> pd.DataFrame:
    """
    Load a previously saved Stage-A processed Parquet file into memory.

    Parquet is columnar and compressed, so this is much faster than re-reading
    the raw CSVs. Use this when you need the full cleaned dataset (e.g. to
    inspect class distributions or check data quality).

    For model training, use load_subset() instead — the subset is much smaller.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.

    Returns:
        DataFrame with the Stage A output schema:
        - all float32 feature columns
        - label_raw, label_binary (int8), label_multiclass (category)
        - dataset_source (category)

    Raises:
        FileNotFoundError: If the processed Parquet does not exist yet.
                           Run preprocessor.run_preprocessing_pipeline() first.
    """
    parquet_path = PATHS["processed_data"] / f"{dataset_name}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed Parquet not found: {parquet_path}\n"
            "Run preprocessor.run_preprocessing_pipeline('{dataset_name}') first."
        )

    log.info("Loading processed Parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info(
        "Loaded processed data — %d rows, %d columns", len(df), len(df.columns)
    )
    return df


def load_subset(dataset_name: str) -> pd.DataFrame:
    """
    Load the stratified working subset Parquet for a dataset.

    The subset Parquet is the file used for all model training and evaluation.
    It is smaller than the full processed file and includes a 'split' column
    that assigns each row to 'train', 'val', or 'test'.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.

    Returns:
        DataFrame with Stage B schema:
        - all float32 feature columns
        - label_raw, label_binary (int8), label_multiclass (category)
        - dataset_source (category)
        - split (category): 'train', 'val', or 'test'

    Raises:
        FileNotFoundError: If the subset Parquet does not exist yet.
                           Run sampler.run_sampling_pipeline() first.
    """
    subset_path = PATHS["subsets"] / f"{dataset_name}_subset.parquet"

    if not subset_path.exists():
        raise FileNotFoundError(
            f"Subset Parquet not found: {subset_path}\n"
            "Run sampler.run_sampling_pipeline('{dataset_name}') first."
        )

    log.info("Loading subset Parquet: %s", subset_path)
    df = pd.read_parquet(subset_path)
    log.info(
        "Loaded subset — %d rows, %d columns, splits: %s",
        len(df),
        len(df.columns),
        df["split"].value_counts().to_dict() if "split" in df.columns else "no split col",
    )
    return df
