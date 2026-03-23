"""
train_private.py — CLI script to run the DP-LR epsilon sweep for a dataset.

Trains a Differentially Private Logistic Regression model at each epsilon
value defined in config.yaml (privacy.epsilon_values). Results are saved
to results/{dataset}_dp_sweep_{label}.json.

Usage:
    source .venv/bin/activate

    # Full epsilon sweep on CIC-IDS2018 (binary)
    python scripts/train_private.py --dataset cic_ids2018

    # Sweep on UNSW-NB15 with multiclass labels
    python scripts/train_private.py --dataset unsw_nb15 --label multiclass

    # Single epsilon value
    python scripts/train_private.py --dataset cic_ids2018 --epsilon 1.0

    # Custom epsilon list
    python scripts/train_private.py --dataset cic_ids2018 --epsilon 0.5 1.0 5.0

    # Force re-train all epsilon models
    python scripts/train_private.py --dataset cic_ids2018 --force

Prerequisites:
    Subset Parquet must exist. Baseline models are not required, but
    running train_baseline.py first is recommended so evaluate.py can
    produce the comparison table in one shot.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import PRIVACY_CFG
from src.models.dp_model import run_epsilon_sweep, train_dp_model
from src.evaluation.results_store import save_results, save_combined_results

log = get_logger("train_private")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Differentially Private Logistic Regression (epsilon sweep).",
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
        help="Label type (default: binary).",
    )
    parser.add_argument(
        "--epsilon",
        nargs="+",
        type=float,
        default=None,
        metavar="eps",
        help=(
            "One or more epsilon values to train at. "
            "If omitted, uses the full list from config.yaml "
            f"(currently: {PRIVACY_CFG['epsilon_values']})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if saved DP models already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epsilon_values = args.epsilon or PRIVACY_CFG["epsilon_values"]

    log.info("=" * 60)
    log.info("Differentially Private Training (eps sweep)")
    log.info("  Dataset  : %s", args.dataset)
    log.info("  Label    : %s", args.label)
    log.info("  Epsilons : %s", epsilon_values)
    log.info("  Force    : %s", args.force)
    log.info("=" * 60)

    # Run the epsilon sweep
    dp_results = run_epsilon_sweep(
        dataset_name=args.dataset,
        label=args.label,
        epsilon_values=sorted(epsilon_values),
        force=args.force,
    )

    # Save DP sweep results
    save_results(dp_results, args.dataset, "dp_sweep", label=args.label)

    # Merge with any existing baseline results into combined JSON
    save_combined_results(args.dataset, label=args.label)

    log.info("=" * 60)
    log.info("DP sweep complete. Combined results saved.")
    log.info(
        "Next step: run  python scripts/evaluate.py --dataset %s",
        args.dataset
    )


if __name__ == "__main__":
    main()
