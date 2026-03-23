"""
train_baseline.py — CLI script to train all baseline models for a dataset.

This script is a thin wrapper over src/models/baseline.py. It handles
argument parsing and result persistence so you can run experiments from
the terminal without opening the Streamlit app.

Usage:
    # Activate the virtual environment first
    # Linux/macOS:  source .venv/bin/activate
    # Windows:      .venv\\Scripts\\activate

    # Train all baselines for CIC-IDS2018 (binary classification)
    python scripts/train_baseline.py --dataset cic_ids2018

    # Train all baselines for UNSW-NB15 (multiclass)
    python scripts/train_baseline.py --dataset unsw_nb15 --label multiclass

    # Force re-training even if saved models exist
    python scripts/train_baseline.py --dataset cic_ids2018 --force

    # Train a single model only
    python scripts/train_baseline.py --dataset cic_ids2018 --model random_forest

Prerequisites:
    The subset Parquet must exist. If it does not, run the preprocessing
    and sampling pipelines first:
        python scripts/preprocess.py --dataset cic_ids2018
    Or trigger them from the Streamlit Data Overview page.
"""

import argparse
import sys
from pathlib import Path

# ── Ensure project root is on the Python path ─────────────────────────────────
# This allows `from src.xxx import yyy` to work when running the script
# from any working directory, not just the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.models.baseline import (
    train_baseline,
    train_all_baselines,
    BASELINE_MODELS,
)
from src.evaluation.results_store import save_results

log = get_logger("train_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline intrusion detection classifiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cic_ids2018", "unsw_nb15"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--label",
        default="binary",
        choices=["binary", "multiclass"],
        help="Label type for classification (default: binary).",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=BASELINE_MODELS,
        help="Train only this model. Omit to train all three baselines.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if a saved model already exists on disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("Baseline Training")
    log.info("  Dataset : %s", args.dataset)
    log.info("  Label   : %s", args.label)
    log.info("  Model   : %s", args.model or "all")
    log.info("  Force   : %s", args.force)
    log.info("=" * 60)

    if args.model:
        # Train a single specified model
        results = [
            train_baseline(
                args.dataset,
                args.model,
                label=args.label,
                force=args.force,
            )
        ]
    else:
        # Train all three baselines
        results = train_all_baselines(
            args.dataset,
            label=args.label,
            force=args.force,
        )

    # Persist results to JSON for later use by evaluate.py and Streamlit
    save_results(results, args.dataset, "baselines", label=args.label)

    log.info("=" * 60)
    log.info("Training complete. Results saved.")
    log.info(
        "Next step: run  python scripts/train_private.py --dataset %s",
        args.dataset
    )


if __name__ == "__main__":
    main()
