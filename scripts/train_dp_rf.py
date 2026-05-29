"""
train_dp_rf.py — CLI for the DP Random Forest epsilon sweep.

Trains diffprivlib's RandomForestClassifier (pure ε-DP) across the epsilon
values defined in config.yaml and saves results to results/.

Usage
-----
    # Full sweep (all 13 epsilon values from config.yaml):
    python scripts/train_dp_rf.py --dataset cic_ids2018 --label binary
    python scripts/train_dp_rf.py --dataset unsw_nb15   --label binary

    # Specific epsilon values:
    python scripts/train_dp_rf.py --dataset cic_ids2018 --epsilon 0.1 1.0 10.0

    # Force re-train (overwrite saved models):
    python scripts/train_dp_rf.py --dataset cic_ids2018 --label binary --force

    # Multiclass labels:
    python scripts/train_dp_rf.py --dataset cic_ids2018 --label multiclass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.dp_rf_model import run_dp_rf_sweep, train_dp_rf_model
from src.evaluation.results_store import save_results, save_combined_results
from src.utils.logger import get_logger

log = get_logger("train_dp_rf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DP Random Forest (diffprivlib, pure ε-DP) epsilon sweep"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cic_ids2018", "unsw_nb15"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--label",
        default="binary",
        choices=["binary", "multiclass"],
        help="Label type (default: binary)",
    )
    parser.add_argument(
        "--epsilon",
        nargs="+",
        type=float,
        default=None,
        metavar="EPS",
        help="Specific epsilon values (default: all values from config.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if model files already exist",
    )

    args = parser.parse_args()

    if args.epsilon:
        # Train specific epsilon values one by one
        from src.data.sampler import get_split_arrays
        from src.models.dp_rf_model import estimate_feature_bounds
        splits = get_split_arrays(args.dataset, label=args.label)
        X_train, _ = splits["train"]
        bounds = estimate_feature_bounds(X_train)

        results = []
        for eps in sorted(args.epsilon):
            log.info("Training DP-RF at eps=%.3f...", eps)
            r = train_dp_rf_model(
                args.dataset, eps,
                label=args.label,
                bounds=bounds,
                force=args.force,
            )
            results.append(r)
            m = r["metrics"]["test"]
            log.info(
                "  eps=%.3f  F1=%.4f  AUC=%s",
                eps, m.get("f1_macro", 0),
                f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") else "—",
            )
        save_results(results, args.dataset, "dp_rf_sweep", label=args.label)
    else:
        results = run_dp_rf_sweep(
            args.dataset, label=args.label, force=args.force
        )

    # Update combined results JSON
    save_combined_results(args.dataset, label=args.label)

    log.info("DP-RF sweep complete — %d models. Results saved.", len(results))


if __name__ == "__main__":
    main()
