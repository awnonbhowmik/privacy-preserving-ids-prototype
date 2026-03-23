"""
home.py — Home page content for the Privacy-Preserving IDS Streamlit app.
"""
import streamlit as st
import pandas as pd
from src.models.model_utils import get_pipeline_status, get_subset_stats, SUPPORTED_DATASETS
from src.evaluation.results_store import load_results

# ── Hero banner ───────────────────────────────────────────────────────────────
st.title("🛡️ Privacy-Preserving Intrusion Detection System")
st.markdown(
    "> A dissertation prototype demonstrating how **Differential Privacy (DP)** can be "
    "applied to network intrusion detection while quantifying the **privacy–utility tradeoff**."
)

st.divider()

# ── Cross-dataset key metrics summary ─────────────────────────────────────────
st.subheader("Research Summary")

DS_DISPLAY = {"cic_ids2018": "CIC-IDS2018", "unsw_nb15": "UNSW-NB15"}

summary_rows = []
for ds in SUPPORTED_DATASETS:
    for lbl in ["binary", "multiclass"]:
        combined = load_results(ds, "combined", label=lbl)
        if not combined:
            continue
        baselines = [r for r in combined if r.get("epsilon") is None]
        dp_lr     = [r for r in combined if r.get("epsilon") is not None and r.get("model_name") == "dp_logistic_regression"]
        dp_nn     = [r for r in combined if r.get("epsilon") is not None and r.get("model_name") == "dp_sgd"]
        if baselines:
            best_bl = max(baselines, key=lambda r: r["metrics"]["test"].get("f1_macro", 0))
            best_f1 = best_bl["metrics"]["test"].get("f1_macro", 0)
            best_model = best_bl["model_name"].replace("_", " ").title()
        else:
            best_f1, best_model = None, "—"

        min_ok_eps_nn = None
        if dp_nn and best_f1:
            for r in sorted(dp_nn, key=lambda r: r["epsilon"]):
                f1 = r["metrics"]["test"].get("f1_macro", 0)
                if best_f1 > 0 and (best_f1 - f1) / best_f1 < 0.10:
                    min_ok_eps_nn = r["epsilon"]
                    break

        if best_f1 is not None:
            summary_rows.append({
                "Dataset": DS_DISPLAY[ds],
                "Label": lbl.title(),
                "Best Baseline": best_model,
                "Best F1": round(best_f1, 4),
                "DP-LR trained": len(dp_lr),
                "DP-SGD trained": len(dp_nn),
                "DP-SGD ε ≤ 10% loss": min_ok_eps_nn if min_ok_eps_nn else "—",
            })

if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
else:
    st.info("Train models to see the research summary here.")

st.divider()

# ── Pipeline status overview ──────────────────────────────────────────────────
st.subheader("Pipeline Status")
st.caption("Complete each stage in order. Expand a dataset to see details.")

dataset    = st.session_state.get("global_dataset", "cic_ids2018")
label_type = st.session_state.get("global_label", "binary")

for ds in SUPPORTED_DATASETS:
    ds_display = DS_DISPLAY[ds]
    status = get_pipeline_status(ds, label=label_type)
    stats  = get_subset_stats(ds)

    checks = [
        status["raw_files_exist"],
        status["processed_exists"],
        status["subset_exists"],
        any(status["baselines_trained"].values()),
        any(status["dp_trained"].values()),
    ]
    n_done = sum(checks)
    pct    = n_done / len(checks)

    with st.expander(
        f"**{ds_display}**  —  {n_done}/{len(checks)} stages complete",
        expanded=(ds == dataset),
    ):
        st.progress(pct)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("##### Data Pipeline")
            n = status["raw_file_count"]
            st.markdown(
                f"{'✅' if status['raw_files_exist'] else '❌'} Raw CSVs ({n} file{'s' if n != 1 else ''})\n\n"
                f"{'✅' if status['processed_exists'] else '❌'} Processed Parquet\n\n"
                f"{'✅' if status['subset_exists'] else '❌'} Working subset"
            )
            if stats:
                st.caption(f"{stats['total_rows']:,} rows · {stats['n_features']} features")

        with col2:
            st.markdown("##### Baseline Models")
            for model_name, trained in status["baselines_trained"].items():
                icon = "✅" if trained else "⏳"
                display = model_name.replace("_", " ").title()
                st.markdown(f"{icon} {display}")

        with col3:
            st.markdown("##### DP-LR Sweep")
            dp_done  = sum(status["dp_trained"].values())
            dp_total = len(status["dp_trained"])
            st.progress(dp_done / max(dp_total, 1), text=f"{dp_done}/{dp_total} ε values")
            res = status["results_exist"]
            st.markdown(
                f"{'✅' if res['baselines'] else '❌'} Baseline results\n\n"
                f"{'✅' if res['dp_sweep'] else '❌'} DP sweep results\n\n"
                f"{'✅' if res['combined'] else '❌'} Combined results"
            )

        with col4:
            st.markdown("##### DP-SGD Sweep")
            nn_done  = sum(status.get("dp_sgd_trained", {}).values())
            nn_total = len(status.get("dp_sgd_trained", {}))
            st.progress(nn_done / max(nn_total, 1), text=f"{nn_done}/{nn_total} ε values")
            dp_sgd_res = status["results_exist"].get("dp_sgd_sweep", False)
            st.markdown(f"{'✅' if dp_sgd_res else '❌'} DP-SGD results")

st.divider()

# ── Methodology cards ─────────────────────────────────────────────────────────
st.subheader("Methodology Overview")

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("#### 1️⃣  Baseline Models")
        st.markdown(
            "Three non-private classifiers trained on the same stratified split:\n\n"
            "- Logistic Regression\n"
            "- Random Forest\n"
            "- XGBoost\n\n"
            "These define the **upper-bound** performance ceiling."
        )
with c2:
    with st.container(border=True):
        st.markdown("#### 2️⃣  DP-LR (diffprivlib)")
        st.markdown(
            "Differentially Private Logistic Regression using IBM's diffprivlib "
            "(objective perturbation). Sweeps ε ∈ {0.01 … 10.0}.\n\n"
            "Low computational cost, interpretable privacy guarantee."
        )
with c3:
    with st.container(border=True):
        st.markdown("#### 3️⃣  DP-SGD (Opacus)")
        st.markdown(
            "2-layer MLP (128→64) trained with **DP-SGD** via Opacus. "
            "Per-sample gradient clipping + Gaussian noise injection each step.\n\n"
            "GPU-accelerated. RDP accountant tracks actual ε spent."
        )

st.divider()

# ── Quick start ───────────────────────────────────────────────────────────────
st.subheader("Quick Start")
with st.expander("Show CLI commands"):
    st.code(
        """# 1. Preprocess raw CSVs into Parquet  (one-time)
python scripts/preprocess.py --dataset cic_ids2018

# 2. Train all baseline models
python scripts/train_baseline.py --dataset cic_ids2018

# 3. DP-LR epsilon sweep
python scripts/train_private.py  --dataset cic_ids2018

# 4. DP-SGD sweep  (requires torch + opacus + CUDA)
python scripts/train_dp_sgd.py --dataset cic_ids2018

# 5. Export comparison tables
python scripts/evaluate.py --dataset cic_ids2018 --export-csv
""",
        language="bash",
    )
