"""
home.py — Home page for the Privacy-Preserving IDS Streamlit app.
"""
import streamlit as st
import pandas as pd
from src.models.model_utils import get_pipeline_status, get_subset_stats, SUPPORTED_DATASETS
from src.evaluation.results_store import load_results
from src.utils.config import PATHS

DS_DISPLAY = {"cic_ids2018": "CIC-IDS2018", "unsw_nb15": "UNSW-NB15"}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.title("🛡️ Privacy-Preserving Intrusion Detection System")

col_hero, col_rq = st.columns([2, 3])
with col_hero:
    st.markdown(
        "A **Doctor of Computer Science** dissertation prototype evaluating how "
        "Differential Privacy can be embedded in machine-learning-based "
        "network intrusion detection without sacrificing operational utility."
    )
with col_rq:
    st.info(
        "**Research Question:** How can Differential Privacy be effectively integrated "
        "into ML-based cybersecurity threat-detection systems to achieve an optimal "
        "balance between data privacy and analytic utility?"
    )

st.divider()

# ── Key results summary ───────────────────────────────────────────────────────
st.subheader("Experiment Results at a Glance")

summary_rows = []
for ds in SUPPORTED_DATASETS:
    for lbl in ["binary", "multiclass"]:
        combined = load_results(ds, "combined", label=lbl)
        if not combined:
            continue

        baselines = [r for r in combined if r.get("epsilon") is None]
        dp_lr     = sorted(
            [r for r in combined if r.get("model_name") == "dp_logistic_regression"],
            key=lambda r: r["epsilon"],
        )
        dp_rf     = sorted(
            [r for r in combined if r.get("model_name") == "dp_random_forest"],
            key=lambda r: r["epsilon"],
        )
        dp_nn     = sorted(
            [r for r in combined if r.get("model_name") == "dp_sgd"],
            key=lambda r: r["epsilon"],
        )

        if not baselines:
            continue

        best_bl   = max(baselines, key=lambda r: r["metrics"]["test"].get("f1_macro", 0))
        best_f1   = round(best_bl["metrics"]["test"].get("f1_macro", 0), 4)
        best_name = best_bl["model_name"].replace("_", " ").title()

        def _min_eps(dp_list, threshold=0.10):
            if not dp_list or not best_f1:
                return "—"
            for r in dp_list:
                f1 = r["metrics"]["test"].get("f1_macro", 0)
                if best_f1 > 0 and (best_f1 - f1) / best_f1 < threshold:
                    return r["epsilon"]
            return "—"

        summary_rows.append({
            "Dataset":           DS_DISPLAY[ds],
            "Label":             lbl.title(),
            "Best Baseline":     best_name,
            "Best F1":           best_f1,
            "DP-LR (13 ε)":     len(dp_lr),
            "DP-RF (13 ε)":     len(dp_rf) if dp_rf else "—",
            "DP-SGD (13 ε)":    len(dp_nn),
            "Min ε (≤10% loss)": _min_eps(dp_nn),
        })

if summary_rows:
    df = pd.DataFrame(summary_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        "Min ε (≤10% loss) = smallest privacy budget where DP-SGD stays within 10% of the best baseline F1. "
        "Lower ε = stronger privacy."
    )
else:
    st.info(
        "No results yet. Complete the pipeline to see experiment results here. "
        "Start at **Data Overview** → **Train Models** → **Compare Results**."
    )

st.divider()

# ── Pipeline status ───────────────────────────────────────────────────────────
st.subheader("Pipeline Status")
st.caption("Complete each stage in order. All stages must be green before training.")

label_type = st.session_state.get("global_label", "binary")
dataset    = st.session_state.get("global_dataset", "cic_ids2018")

for ds in SUPPORTED_DATASETS:
    status = get_pipeline_status(ds, label=label_type)
    stats  = get_subset_stats(ds)

    # Count all completed stages
    dp_lr_done   = sum(status["dp_trained"].values())
    dp_rf_done   = sum(status.get("dp_rf_trained", {}).values())
    dp_sgd_done  = sum(status.get("dp_sgd_trained", {}).values())
    dp_total     = len(status["dp_trained"])

    checks = [
        status["raw_files_exist"],
        status["processed_exists"],
        status["subset_exists"],
        any(status["baselines_trained"].values()),
        dp_lr_done > 0,
        dp_sgd_done > 0,
    ]
    n_done = sum(checks)
    pct    = n_done / len(checks)

    with st.expander(
        f"**{DS_DISPLAY[ds]}** — {n_done}/{len(checks)} stages complete",
        expanded=(ds == dataset),
    ):
        st.progress(pct)
        st.markdown("")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Data Pipeline**")
            n = status["raw_file_count"]
            st.markdown(
                f"{'✅' if status['raw_files_exist'] else '❌'} Raw CSVs ({n})\n\n"
                f"{'✅' if status['processed_exists'] else '❌'} Preprocessed\n\n"
                f"{'✅' if status['subset_exists'] else '❌'} Subset (300k rows)"
            )
            if stats:
                st.caption(f"{stats['total_rows']:,} rows · {stats['n_features']} features · 60/20/20 split")

        with c2:
            st.markdown("**Baselines (non-private)**")
            for model_name, trained in status["baselines_trained"].items():
                icon = "✅" if trained else "⏳"
                st.markdown(f"{icon} {model_name.replace('_', ' ').title()}")
            res = status["results_exist"]
            st.markdown(f"{'✅' if res['baselines'] else '❌'} Results saved")

        with c3:
            st.markdown("**DP Models (diffprivlib)**")
            st.markdown(f"{'✅' if dp_lr_done == dp_total else '🔄'} DP-LR: {dp_lr_done}/{dp_total} ε values")
            st.markdown(f"{'✅' if dp_rf_done == dp_total else '🔄'} DP-RF: {dp_rf_done}/{dp_total} ε values")
            st.caption("Pure ε-DP · no failure probability δ")

        with c4:
            st.markdown("**DP-SGD (Opacus)**")
            st.markdown(f"{'✅' if dp_sgd_done == dp_total else '🔄'} {dp_sgd_done}/{dp_total} ε values trained")
            dp_sgd_res = status["results_exist"].get("dp_sgd_sweep", False)
            mia_res    = (PATHS["results"] / f"{ds}_mia_{label_type}.json").exists()
            st.markdown(f"{'✅' if dp_sgd_res else '❌'} Results saved")
            st.markdown(f"{'✅' if mia_res else '❌'} MIA evaluation")

st.divider()

# ── Methodology cards ─────────────────────────────────────────────────────────
st.subheader("Experiment Design")
st.caption(
    "Four model families trained on the same fixed dataset split. "
    "The only variable between experiments is the model type and privacy budget ε."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.markdown("#### Baselines")
        st.markdown("**No privacy mechanism**")
        st.markdown(
            "- Logistic Regression\n"
            "- Random Forest\n"
            "- XGBoost (GPU)\n\n"
            "Performance **ceiling** — the best achievable results without DP."
        )

with c2:
    with st.container(border=True):
        st.markdown("#### DP-LR")
        st.markdown("**Pure ε-DP** · diffprivlib")
        st.markdown(
            "Differentially Private Logistic Regression (objective perturbation). "
            "13 ε values: 0.01 → 10.0.\n\n"
            "Linear model · low compute · no δ."
        )

with c3:
    with st.container(border=True):
        st.markdown("#### DP-RF")
        st.markdown("**Pure ε-DP** · diffprivlib")
        st.markdown(
            "DP Random Forest — random split selection + PermuteAndFlip mechanism. "
            "Same 13 ε values.\n\n"
            "Tree ensemble · stronger guarantee than DP-SGD."
        )

with c4:
    with st.container(border=True):
        st.markdown("#### DP-SGD")
        st.markdown("**(ε, δ)-DP** · Opacus")
        st.markdown(
            "2-layer MLP (128→64) trained with per-sample gradient clipping + "
            "Gaussian noise injection (RDP accounting). "
            "Same 13 ε values.\n\n"
            "Neural network · GPU-accelerated."
        )

st.divider()

# ── Phase 3 banner ────────────────────────────────────────────────────────────
st.subheader("Phase 3 — Privacy Risk Evaluation")
with st.container(border=True):
    ph1, ph2 = st.columns([3, 1])
    with ph1:
        st.markdown(
            "Beyond accuracy metrics, this prototype empirically evaluates "
            "**membership inference attack (MIA)** resistance: does DP actually reduce "
            "an adversary's ability to determine whether a specific network flow record "
            "was used to train the model?\n\n"
            "**Attack ROC-AUC near 0.5** = attacker cannot distinguish members from "
            "non-members = strong privacy protection."
        )
    with ph2:
        st.page_link("pages/4_Attack_Evaluation.py", label="View Attack Results", icon="🔍")

st.divider()

# ── Quick start ───────────────────────────────────────────────────────────────
st.subheader("Quick Start")
with st.expander("Show CLI commands (run from project root)"):
    st.markdown("**Step 1 — Dataset setup (one command, idempotent):**")
    st.code("python scripts/setup_data.py", language="bash")

    st.markdown("**Steps 2–6 — Training pipeline:**")
    st.code(
        """# 2. Train baselines (LR, RF, XGBoost)
python scripts/train_baseline.py --dataset cic_ids2018 --label binary

# 3. DP-LR sweep (pure ε-DP, fast ~2 min)
python scripts/train_private.py --dataset cic_ids2018 --label binary

# 3b. DP-RF sweep (pure ε-DP, fast ~1 min)
python scripts/train_dp_rf.py --dataset cic_ids2018 --label binary

# 4. DP-SGD sweep (~7 hours CPU / ~30 min GPU)
python scripts/train_dp_sgd.py --dataset cic_ids2018 --label binary

# 5. Phase 3 — Membership Inference Attack evaluation
python scripts/run_membership_inference.py --dataset cic_ids2018 --label binary --export-csv

# 6. Streamlit dashboard
python -m streamlit run app/main.py""",
        language="bash",
    )
    st.markdown("Run tests (125 tests, ~7 seconds, no dataset required):")
    st.code("python -m pytest tests/ -v", language="bash")
