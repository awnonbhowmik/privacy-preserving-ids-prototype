"""
train_dp_sgd.py — CLI to run the DP-SGD (Opacus) epsilon sweep.

Mirrors the interface of scripts/train_private.py exactly. Trains an MLP with
DP-SGD (via Opacus) at each epsilon value defined in config.yaml under the
dp_sgd section. Results are saved to results/{dataset}_dp_sgd_sweep_{label}.json
and the combined file is updated.

Usage:
    # Full epsilon sweep on CIC-IDS2018 (binary)
    python scripts/train_dp_sgd.py --dataset cic_ids2018

    # Sweep on UNSW-NB15 with multiclass labels
    python scripts/train_dp_sgd.py --dataset unsw_nb15 --label multiclass

    # Single epsilon value
    python scripts/train_dp_sgd.py --dataset cic_ids2018 --epsilon 1.0

    # Custom epsilon list
    python scripts/train_dp_sgd.py --dataset cic_ids2018 --epsilon 0.5 1.0 5.0

    # Force re-train all epsilon models
    python scripts/train_dp_sgd.py --dataset cic_ids2018 --force

Prerequisites:
    Subset Parquet must exist (run Data Overview -> Stage B first).
    torch and opacus must be installed: pip install torch opacus
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import DP_SGD_CFG
from src.models.dp_sgd_model import run_dp_sgd_sweep, train_dp_sgd_model
from src.evaluation.results_store import save_results, load_results

log = get_logger("train_dp_sgd")


def parse_args() -> argparse.Namespace:
    default_eps = DP_SGD_CFG.get("epsilon_values", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

    parser = argparse.ArgumentParser(
        description="Train Differentially Private Neural Network via DP-SGD (epsilon sweep).",
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
            "If omitted, uses the full list from config.yaml dp_sgd section "
            f"(currently: {default_eps})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if saved DP-SGD models already exist.",
    )
    return parser.parse_args()


def _print_summary(results: list[dict]) -> None:
    """Print a formatted summary table after the sweep completes."""
    if not results:
        return

    header = f"{'eps':>8} | {'F1 macro':>10} | {'AUC':>10} | {'MCC':>10} | {'Time (s)':>10}"
    print("\n" + "=" * len(header))
    print("DP-SGD Sweep Summary (test split)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        m   = r["metrics"]["test"]
        eps = r["epsilon"]
        f1  = m.get("f1_macro", 0)
        auc = m.get("roc_auc")
        mcc = m.get("mcc", 0)
        t   = r.get("train_time_s", 0)

        auc_str = f"{auc:.4f}" if auc is not None else "     N/A"
        print(f"{eps:>8.2f} | {f1:>10.4f} | {auc_str:>10} | {mcc:>10.4f} | {t:>10.1f}")

    print("=" * len(header))


def main() -> None:
    args = parse_args()
    epsilon_values = args.epsilon or DP_SGD_CFG.get("epsilon_values", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

    log.info("=" * 60)
    log.info("Differentially Private Neural Network Training via DP-SGD (eps sweep)")
    log.info("  Dataset  : %s", args.dataset)
    log.info("  Label    : %s", args.label)
    log.info("  Epsilons : %s", sorted(epsilon_values))
    log.info("  Force    : %s", args.force)
    log.info("=" * 60)

    # Run the epsilon sweep
    dp_sgd_results = run_dp_sgd_sweep(
        dataset_name=args.dataset,
        label=args.label,
        epsilon_values=sorted(epsilon_values),
        force=args.force,
    )

    # Save DP-SGD sweep results
    save_results(dp_sgd_results, args.dataset, "dp_sgd_sweep", label=args.label)

    # Merge baselines + dp_sweep + dp_sgd_sweep into combined JSON
    baselines = load_results(args.dataset, "baselines",  label=args.label)
    dp_sweep  = load_results(args.dataset, "dp_sweep",   label=args.label)
    combined  = baselines + dp_sweep + dp_sgd_results
    save_results(combined, args.dataset, "combined", label=args.label)

    log.info("=" * 60)
    log.info("DP-SGD sweep complete. Results saved.")
    log.info(
        "Next step: run  python scripts/evaluate.py --dataset %s",
        args.dataset,
    )

    # Print human-readable summary table
    _print_summary(dp_sgd_results)


if __name__ == "__main__":
    main()
