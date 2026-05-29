"""
setup_data.py — One-command data setup from zip files or raw CSVs.

Run this whenever you copy new dataset files into the data/ directory.
The script is fully idempotent: each stage is skipped if its output already exists.

Supported input layouts
-----------------------
Option A — ZIP files in data/:
    data/CSE-CIC-IDS2018 Cleaned.zip     (any filename containing "CIC" or "IDS2018")
    data/UNSW-NB15 dataset.zip           (any filename containing "UNSW" or "NB15")

Option B — Raw CSVs already extracted:
    data/raw/cic_ids2018/*.csv           (any .csv files, any filenames)
    data/raw/unsw_nb15/UNSW-NB15_1.csv  (must be named UNSW-NB15_1.csv through _4.csv)

You can mix both: if raw CSVs already exist for one dataset the extraction step is
skipped for that one regardless of whether a zip file is also present.

Usage
-----
    # Auto-detect everything and run all missing stages:
    python scripts/setup_data.py

    # Show current pipeline status without running anything:
    python scripts/setup_data.py --status

    # Force re-run all stages (re-extracts, re-preprocesses, re-samples):
    python scripts/setup_data.py --force

    # Run only specific stage(s):
    python scripts/setup_data.py --only extract
    python scripts/setup_data.py --only preprocess
    python scripts/setup_data.py --only sample

    # Run for one dataset only:
    python scripts/setup_data.py --dataset cic_ids2018
    python scripts/setup_data.py --dataset unsw_nb15

Known dataset quirks (handled automatically)
---------------------------------------------
CIC-IDS2018 ("Cleaned" version):
  - latin-1 encoding (NOT utf-8 — the raw files use Windows code page 1252)
  - Column names have leading/trailing spaces → normalised to snake_case
  - ±Inf values in ratio-based features (e.g. Flow Byts/s) → replaced with NaN
  - Label column named "Label" (capital L)
  - The "Cleaned" version has no Timestamp column.
    If you use the *original* (uncleaned) CICFlowMeter output, the Timestamp
    column ("15/02/2018 09:00:00" format) cannot be parsed as a number so it
    becomes all-NaN and is automatically dropped by drop_high_null_columns().
  - "Infilteration" label spelling typo in the raw data → mapped to "Infiltration"
    in label_mapping.py

UNSW-NB15:
  - No header row — 49 column names are assigned by src/data/loader.py
    (UNSW_NB15_COLUMNS list, matching the official UNSW feature specification)
  - UNSW-NB15_1.csv starts with a UTF-8 BOM (\\ufeff) at byte 0.
    pandas reads this as part of the first cell value; since the first column
    is srcip (an IP address string like "59.166.0.0") it is non-numeric,
    coerced to NaN, and dropped. The BOM causes no functional issue.
  - 5 non-numeric columns are always dropped:
      srcip, dstip  — IP address strings
      proto         — protocol name ("tcp", "udp", ...)
      state         — connection state ("FIN", "INT", ...)
      service       — application service ("http", "ftp", ...)
    These are dropped because they cannot be represented as float32 without
    one-hot encoding, which is out of scope for this prototype.
  - label column is lowercase "label" (integer 0/1)
  - attack_cat column contains the attack category string (may be NaN/empty
    for benign rows → mapped to "BENIGN")
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

# Bootstrap sys.path so src/ is importable regardless of working directory
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.utils.config import PATHS
from src.utils.logger import get_logger

log = get_logger("setup_data")

# ── Known zip filename patterns ────────────────────────────────────────────────
# Any zip file in data/ whose name contains one of these tokens (case-insensitive)
# is recognised as the corresponding dataset.
_CIC_ZIP_TOKENS  = ["cic", "ids2018", "cse-cic"]
_UNSW_ZIP_TOKENS = ["unsw", "nb15"]

# The only UNSW-NB15 CSV files that contain the flow features used in this study.
# The zip also contains per-day Argus/PCAP files and training/testing subsets;
# only these 4 main files are needed.
_UNSW_NEEDED_CSV_PATHS = {
    "UNSW-NB15 dataset/CSV Files/UNSW-NB15_1.csv",
    "UNSW-NB15 dataset/CSV Files/UNSW-NB15_2.csv",
    "UNSW-NB15 dataset/CSV Files/UNSW-NB15_3.csv",
    "UNSW-NB15 dataset/CSV Files/UNSW-NB15_4.csv",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Zip detection helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _find_zip(tokens: list[str]) -> Path | None:
    """Return the first .zip in data/ whose name matches any token (case-insensitive)."""
    data_dir = _ROOT / "data"
    for f in data_dir.glob("*.zip"):
        name_lower = f.name.lower()
        if any(tok in name_lower for tok in tokens):
            return f
    return None


def _raw_csvs_exist(dataset_name: str) -> bool:
    raw_dir = PATHS["raw_data"] / dataset_name
    if not raw_dir.exists():
        return False
    csvs = list(raw_dir.glob("*.csv"))
    if dataset_name == "unsw_nb15":
        return any(f.name.startswith("UNSW-NB15_") for f in csvs)
    return len(csvs) > 0


def _processed_exists(dataset_name: str) -> bool:
    return (PATHS["processed_data"] / f"{dataset_name}.parquet").exists()


def _subset_exists(dataset_name: str) -> bool:
    return (PATHS["subsets"] / f"{dataset_name}_subset.parquet").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_dataset(dataset_name: str, force: bool = False) -> bool:
    """
    Extract CSVs from a detected zip file into data/raw/{dataset_name}/.

    Returns True if extraction was performed, False if skipped.
    """
    if _raw_csvs_exist(dataset_name) and not force:
        log.info("[%s] Raw CSVs already present — skipping extraction.", dataset_name)
        return False

    tokens = _CIC_ZIP_TOKENS if dataset_name == "cic_ids2018" else _UNSW_ZIP_TOKENS
    zip_path = _find_zip(tokens)
    if zip_path is None:
        log.warning(
            "[%s] No zip file found in data/ (looked for tokens: %s). "
            "Place the zip there and re-run, or extract manually to data/raw/%s/.",
            dataset_name, tokens, dataset_name,
        )
        return False

    dest = PATHS["raw_data"] / dataset_name
    dest.mkdir(parents=True, exist_ok=True)

    log.info("[%s] Extracting from %s → %s", dataset_name, zip_path.name, dest)
    t0 = time.time()

    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()

        if dataset_name == "cic_ids2018":
            csv_members = [m for m in members if m.filename.endswith(".csv")]
        else:
            csv_members = [m for m in members if m.filename in _UNSW_NEEDED_CSV_PATHS]

        if not csv_members:
            log.error(
                "[%s] No expected CSV files found inside %s. "
                "Check that the zip contains the correct dataset.",
                dataset_name, zip_path.name,
            )
            return False

        for m in csv_members:
            out_path = dest / Path(m.filename).name
            if out_path.exists() and not force:
                log.info("  skip (exists): %s", out_path.name)
                continue
            log.info("  extracting: %s  (%.0f MB)", out_path.name, m.file_size / 1e6)
            with z.open(m) as src, open(out_path, "wb") as dst:
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)

    elapsed = time.time() - t0
    n = len(csv_members)
    log.info("[%s] Extracted %d CSV file(s) in %.0fs.", dataset_name, n, elapsed)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing and sampling
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_dataset(dataset_name: str, force: bool = False) -> bool:
    if _processed_exists(dataset_name) and not force:
        parquet = PATHS["processed_data"] / f"{dataset_name}.parquet"
        log.info(
            "[%s] Processed Parquet already exists (%.1f MB) — skipping Stage A.",
            dataset_name, parquet.stat().st_size / 1e6,
        )
        return False

    if not _raw_csvs_exist(dataset_name):
        log.error(
            "[%s] No raw CSVs found. Run extraction first or place CSVs in data/raw/%s/.",
            dataset_name, dataset_name,
        )
        return False

    from src.data.preprocessor import run_preprocessing_pipeline
    t0 = time.time()
    log.info("[%s] Running Stage A preprocessing...", dataset_name)
    run_preprocessing_pipeline(dataset_name, force=force)
    log.info("[%s] Stage A done in %.1f min.", dataset_name, (time.time() - t0) / 60)
    return True


def sample_dataset(dataset_name: str, force: bool = False) -> bool:
    if _subset_exists(dataset_name) and not force:
        subset = PATHS["subsets"] / f"{dataset_name}_subset.parquet"
        log.info(
            "[%s] Subset Parquet already exists (%.1f MB) — skipping Stage B.",
            dataset_name, subset.stat().st_size / 1e6,
        )
        return False

    if not _processed_exists(dataset_name):
        log.error(
            "[%s] Processed Parquet not found. Run preprocessing first.",
            dataset_name,
        )
        return False

    from src.data.sampler import run_sampling_pipeline
    t0 = time.time()
    log.info("[%s] Running Stage B sampling...", dataset_name)
    run_sampling_pipeline(dataset_name, force=force)
    log.info("[%s] Stage B done in %.0fs.", dataset_name, time.time() - t0)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Status report
# ═══════════════════════════════════════════════════════════════════════════════

def print_status(datasets: list[str]) -> None:
    """Print a concise pipeline readiness table."""
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  Dataset Setup Status                                           │")
    print("├────────────────┬───────────┬──────────────┬────────────────────┤")
    print("│ Dataset        │ Raw CSVs  │ Preprocessed │ Subset ready       │")
    print("├────────────────┼───────────┼──────────────┼────────────────────┤")

    for ds in datasets:
        raw   = "✅" if _raw_csvs_exist(ds)     else "❌ missing"
        proc  = "✅" if _processed_exists(ds)   else "❌ run setup"
        sub   = "✅" if _subset_exists(ds)       else "❌ run setup"
        n_csv = len(list((PATHS["raw_data"] / ds).glob("*.csv"))) if _raw_csvs_exist(ds) else 0
        raw_str = f"✅ ({n_csv} files)" if _raw_csvs_exist(ds) else "❌ missing"
        if _processed_exists(ds):
            mb = (PATHS["processed_data"] / f"{ds}.parquet").stat().st_size / 1e6
            proc_str = f"✅ ({mb:.0f} MB)"
        else:
            proc_str = "❌ run setup"
        if _subset_exists(ds):
            mb2 = (PATHS["subsets"] / f"{ds}_subset.parquet").stat().st_size / 1e6
            sub_str = f"✅ ({mb2:.0f} MB)"
        else:
            sub_str = "❌ run setup"
        print(f"│ {ds:<14s} │ {raw_str:<9s} │ {proc_str:<12s} │ {sub_str:<18s} │")

    print("└────────────────┴───────────┴──────────────┴────────────────────┘")

    # Check for zip files
    print()
    cic_zip  = _find_zip(_CIC_ZIP_TOKENS)
    unsw_zip = _find_zip(_UNSW_ZIP_TOKENS)
    print(f"  CIC-IDS2018 zip:  {cic_zip.name if cic_zip else '(not found in data/)'}")
    print(f"  UNSW-NB15 zip:    {unsw_zip.name if unsw_zip else '(not found in data/)'}")

    # Check for models
    n_models = len(list(PATHS["models"].glob("*.joblib")))
    n_results = len(list(PATHS["results"].glob("*.json")))
    print(f"\n  Trained models:  {n_models} .joblib files in results/models/")
    print(f"  Result files:    {n_results} .json files in results/")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command dataset setup (extraction + preprocessing + sampling)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=["cic_ids2018", "unsw_nb15", "both"],
        default="both",
        help="Which dataset to set up (default: both)",
    )
    parser.add_argument(
        "--only",
        choices=["extract", "preprocess", "sample"],
        default=None,
        help="Run only one stage instead of all three",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages even if outputs already exist",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print pipeline status and exit without running anything",
    )

    args = parser.parse_args()

    datasets = (
        ["cic_ids2018", "unsw_nb15"] if args.dataset == "both"
        else [args.dataset]
    )

    if args.status:
        print_status(datasets)
        return

    print_status(datasets)

    for ds in datasets:
        log.info("=" * 60)
        log.info("Setting up: %s", ds)
        log.info("=" * 60)

        run_all = args.only is None

        if run_all or args.only == "extract":
            extract_dataset(ds, force=args.force)

        if run_all or args.only == "preprocess":
            preprocess_dataset(ds, force=args.force)

        if run_all or args.only == "sample":
            sample_dataset(ds, force=args.force)

    print_status(datasets)
    log.info("Setup complete. Next step: python scripts/train_baseline.py --dataset <name> --label binary")


if __name__ == "__main__":
    main()
