"""
evaluate.py — CLI script to generate evaluation reports from saved results.

Loads saved result JSON files, builds comparison tables, computes the
privacy–utility tradeoff, and exports CSV files to results/. Also prints
a formatted summary to stdout suitable for copying into a dissertation draft.

Usage:
    # Linux/macOS:  source .venv/bin/activate
    # Windows:      .venv\\Scripts\\activate

    # Full evaluation report for CIC-IDS2018 (binary)
    python scripts/evaluate.py --dataset cic_ids2018

    # Multiclass evaluation
    python scripts/evaluate.py --dataset cic_ids2018 --label multiclass

    # Export tables to CSV for LaTeX import
    python scripts/evaluate.py --dataset cic_ids2018 --export-csv

    # Print detailed per-model classification report
    python scripts/evaluate.py --dataset cic_ids2018 --report

Prerequisites:
    Run train_baseline.py and train_private.py first so their result JSON
    files exist in results/.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.utils.logger import get_logger
from src.evaluation.results_store import (
    load_results,
    save_combined_results,
    build_comparison_table,
    build_tradeoff_table,
    export_to_csv,
)
from src.evaluation.metrics import (
    print_classification_report,
    summarise_results,
)
from src.models.model_utils import load_model, list_trained_models
from src.data.sampler import get_split_arrays

log = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation reports from saved training results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cic_ids2018", "unsw_nb15"],
    )
    parser.add_argument(
        "--label",
        default="binary",
        choices=["binary", "multiclass"],
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Which data split to report on (default: test).",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export comparison and tradeoff tables to CSV in results/.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print full sklearn classification report for each baseline model.",
    )
    return parser.parse_args()


def print_table(title: str, df: pd.DataFrame) -> None:
    """Print a DataFrame as a formatted console table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("=" * 70)
    if df.empty:
        print("  (no data)")
    else:
        print(df.to_string(index=False))
    print()


def main() -> None:
    args = parse_args()

    log.info("Generating evaluation report: dataset=%s, label=%s, split=%s",
             args.dataset, args.label, args.split)

    # ── Merge combined results if not already done ────────────────────────────
    save_combined_results(args.dataset, label=args.label)

    all_results = load_results(args.dataset, "combined", label=args.label)
    if not all_results:
        log.error(
            "No results found for '%s'. "
            "Run train_baseline.py and train_private.py first.",
            args.dataset,
        )
        sys.exit(1)

    # ── Table 1: Baseline model comparison ───────────────────────────────────
    baseline_results = [r for r in all_results if r.get("epsilon") is None]
    comparison_df = build_comparison_table(baseline_results, split=args.split)
    print_table(
        f"Table 1 — Baseline Model Comparison ({args.dataset}, {args.split})",
        comparison_df,
    )

    # ── Table 2: Privacy–utility tradeoff ────────────────────────────────────
    tradeoff_df = build_tradeoff_table(
        args.dataset, label=args.label, split=args.split
    )
    print_table(
        f"Table 2 — Privacy–Utility Tradeoff [{args.dataset}] (vs LR baseline)",
        tradeoff_df,
    )

    # ── Table 3: All results together ─────────────────────────────────────────
    full_df = build_comparison_table(all_results, split=args.split)
    print_table(
        f"Table 3 — Full Results ({args.dataset}, {args.split})",
        full_df,
    )

    # ── Highlight key findings ────────────────────────────────────────────────
    _print_key_findings(all_results, args.split)

    # ── Optional: export to CSV ───────────────────────────────────────────────
    if args.export_csv:
        path1 = export_to_csv(
            comparison_df,
            f"{args.dataset}_table1_baselines_{args.label}.csv"
        )
        path2 = export_to_csv(
            tradeoff_df,
            f"{args.dataset}_table2_tradeoff_{args.label}.csv"
        )
        path3 = export_to_csv(
            full_df,
            f"{args.dataset}_table3_full_{args.label}.csv"
        )
        log.info("CSVs exported:\n  %s\n  %s\n  %s", path1, path2, path3)

    # ── Optional: per-model classification reports ────────────────────────────
    if args.report:
        _print_classification_reports(args.dataset, args.label, args.split)


def _print_key_findings(results: list, split: str) -> None:
    """Print a concise 'key findings' summary for dissertation reference."""
    baseline_results = [r for r in results if r.get("epsilon") is None]
    dp_results = sorted(
        [r for r in results if r.get("epsilon") is not None],
        key=lambda x: x["epsilon"]
    )

    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)

    if baseline_results:
        best_baseline = max(
            baseline_results,
            key=lambda r: r["metrics"][split].get("f1_macro", 0)
        )
        print(
            f"\n  Best non-private model : {best_baseline['model_name']}"
            f"  (F1={best_baseline['metrics'][split].get('f1_macro', 0):.4f})"
        )
        lr = next((r for r in baseline_results if r["model_name"] == "logistic_regression"), None)
        if lr:
            lr_f1 = lr["metrics"][split].get("f1_macro", 0)
            print(f"  LR baseline (anchor)   : F1={lr_f1:.4f}")

            if dp_results:
                # Find which epsilon gets within 5% of LR baseline
                for r in dp_results:
                    dp_f1 = r["metrics"][split].get("f1_macro", 0)
                    loss_pct = 100 * (lr_f1 - dp_f1) / lr_f1 if lr_f1 > 0 else None
                    within = loss_pct is not None and loss_pct <= 5.0
                    marker = " ← within 5% of baseline" if within else ""
                    print(
                        f"  DP-LR ε={r['epsilon']:5.2f}          : "
                        f"F1={dp_f1:.4f}  (loss={loss_pct:.1f}%){marker}"
                    )
    print()


def _print_classification_reports(dataset: str, label: str, split: str) -> None:
    """Load each baseline model and print its sklearn classification report."""
    from src.models.baseline import BASELINE_MODELS

    log.info("Loading test data for classification reports...")
    splits = get_split_arrays(dataset, label=label, scale=True)
    X, y = splits[split]

    for model_name in BASELINE_MODELS:
        try:
            model = load_model(dataset, model_name, label=label)
            report = print_classification_report(model, X, y)
            print(f"\n--- {model_name} ({split} split) ---")
            print(report)
        except FileNotFoundError:
            log.warning("Model not found: %s — skipping.", model_name)


if __name__ == "__main__":
    main()
