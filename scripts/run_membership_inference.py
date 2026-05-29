"""
run_membership_inference.py — Phase 3 MIA evaluation CLI.

Run membership inference attack evaluation against all trained IDS models for
a given dataset and label type.

Examples
--------
# Full sweep (all saved models, default 5 000 samples per model):
    python scripts/run_membership_inference.py --dataset cic_ids2018 --label binary

# Custom sample size, force re-run, print summary table:
    python scripts/run_membership_inference.py --dataset cic_ids2018 --label binary \\
        --n-samples 2000 --force --summary

# Export summary and combined dissertation table to CSV:
    python scripts/run_membership_inference.py --dataset cic_ids2018 --label binary \\
        --export-csv

# Validation dataset:
    python scripts/run_membership_inference.py --dataset unsw_nb15 --label binary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks.membership_inference import (
    build_combined_dissertation_table,
    build_mia_table,
    load_mia_results,
    run_mia_sweep,
)
from src.evaluation.results_store import export_to_csv
from src.utils.logger import get_logger

log = get_logger("run_membership_inference")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Membership Inference Attack Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default="cic_ids2018",
        choices=["cic_ids2018", "unsw_nb15"],
        help="Dataset to evaluate (default: cic_ids2018)",
    )
    parser.add_argument(
        "--label",
        default="binary",
        choices=["binary", "multiclass"],
        help="Label type (default: binary)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        metavar="N",
        help="Members AND non-members per target model (default: 5000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run attack even if result already exists for that model/ε",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary table after sweep completes",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export MIA summary and combined dissertation table to CSV",
    )

    args = parser.parse_args()

    log.info(
        "MIA evaluation starting — dataset=%s  label=%s  n_samples=%d  force=%s",
        args.dataset, args.label, args.n_samples, args.force,
    )

    results = run_mia_sweep(
        dataset_name=args.dataset,
        label=args.label,
        n_attack_samples=args.n_samples,
        force=args.force,
    )

    if not results:
        log.warning(
            "No results produced. "
            "Ensure models are trained first (scripts/train_baseline.py, "
            "scripts/train_private.py, scripts/train_dp_sgd.py)."
        )
        return

    log.info("Sweep done — %d models evaluated.", len(results))

    if args.summary or args.export_csv:
        mia_df = build_mia_table(results)
        combined_df = build_combined_dissertation_table(args.dataset, args.label)

        if not mia_df.empty:
            print("\n=== Membership Inference Attack Results ===")
            print(mia_df.to_string(index=False))

        if not combined_df.empty:
            print("\n=== Combined Dissertation Table (IDS performance + attack risk) ===")
            print(combined_df.to_string(index=False))

        if args.export_csv:
            if not mia_df.empty:
                out = export_to_csv(
                    mia_df,
                    f"{args.dataset}_mia_summary_{args.label}.csv",
                )
                log.info("MIA table exported → %s", out)

            if not combined_df.empty:
                out2 = export_to_csv(
                    combined_df,
                    f"{args.dataset}_combined_dissertation_table_{args.label}.csv",
                )
                log.info("Combined dissertation table exported → %s", out2)


if __name__ == "__main__":
    main()
