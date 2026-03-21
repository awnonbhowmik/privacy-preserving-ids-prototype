"""
2_Train_Models.py — Baseline and DP model training controls.

Allows the user to:
  - Train all three baseline models (LR, RF, XGBoost)
  - Run the DP-LR epsilon sweep
  - See training status for each model
  - View a live results summary table after training
"""

from app._path_setup import setup_path
setup_path()

import streamlit as st
import pandas as pd

from src.models.model_utils import (
    get_pipeline_status,
    list_trained_models,
    SUPPORTED_DATASETS,
)
from src.models.baseline import BASELINE_MODELS
from src.utils.config import PRIVACY_CFG

st.set_page_config(page_title="Train Models", page_icon="🤖", layout="wide")
st.title("Train Models")

# ── Dataset and label selectors ───────────────────────────────────────────────
col_ds, col_lb, _ = st.columns([2, 2, 4])
with col_ds:
    dataset = st.selectbox(
        "Dataset",
        SUPPORTED_DATASETS,
        format_func=lambda x: {"cic_ids2018": "CIC-IDS2018", "unsw_nb15": "UNSW-NB15"}[x],
        index=SUPPORTED_DATASETS.index(st.session_state.get("global_dataset", "cic_ids2018")),
        key="train_dataset",
    )
with col_lb:
    label_type = st.selectbox(
        "Label type",
        ["binary", "multiclass"],
        index=["binary", "multiclass"].index(st.session_state.get("global_label", "binary")),
        key="train_label",
    )

status = get_pipeline_status(dataset, label=label_type)

# Guard: subset must exist before training
if not status["subset_exists"]:
    st.error(
        "No working subset found for this dataset. "
        "Go to **Data Overview** and run the preprocessing and sampling pipelines first."
    )
    st.stop()

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Baseline Models
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Baseline Models")
st.caption(
    "Trains Logistic Regression, Random Forest, and XGBoost on the fixed "
    "train split. Results are saved to `results/` and can be loaded without retraining."
)

# Training status indicators
st.markdown("**Current status:**")
status_cols = st.columns(len(BASELINE_MODELS))
for i, model_name in enumerate(BASELINE_MODELS):
    trained = status["baselines_trained"].get(model_name, False)
    with status_cols[i]:
        display = model_name.replace("_", " ").title()
        if trained:
            st.success(f"✅ {display}")
        else:
            st.warning(f"⏳ {display}")

col_btn1, col_btn2, _ = st.columns([2, 2, 4])
with col_btn1:
    run_baselines = st.button(
        "Train All Baselines",
        type="primary",
        key="btn_train_all",
        help="Trains LR, RF, and XGBoost. Skips models already saved unless Force is on.",
    )
with col_btn2:
    force_train = st.checkbox("Force re-train", key="force_baseline", value=False)

if run_baselines:
    with st.spinner("Training baseline models... (may take 1–3 minutes)"):
        try:
            from src.models.baseline import train_all_baselines
            from src.evaluation.results_store import save_results

            results = train_all_baselines(dataset, label=label_type, force=force_train)
            save_results(results, dataset, "baselines", label=label_type)

            st.success(f"Trained {len(results)} baseline model(s) successfully.")

            # Show quick results table
            rows = []
            for r in results:
                m = r["metrics"]["test"]
                rows.append({
                    "Model":       r["model_name"].replace("_", " ").title(),
                    "Train Time":  f"{r['train_time_s']:.1f}s",
                    "F1 (macro)":  f"{m.get('f1_macro', 0):.4f}",
                    "AUC":         f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") else "—",
                    "MCC":         f"{m.get('mcc', 0):.4f}",
                    "FPR":         f"{m.get('false_positive_rate', 0):.4f}" if m.get("false_positive_rate") is not None else "—",
                    "DR":          f"{m.get('detection_rate', 0):.4f}" if m.get("detection_rate") is not None else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.exception(exc)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Differential Privacy Experiment
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Differential Privacy Experiment (ε sweep)")
st.caption(
    "Trains DP-Logistic Regression at each ε value from `config.yaml`. "
    "Smaller ε = stronger privacy = more noise = lower accuracy."
)

epsilon_values = PRIVACY_CFG.get("epsilon_values", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

# Status per epsilon
dp_done = sum(status["dp_trained"].values())
dp_total = len(status["dp_trained"])
st.markdown(f"**ε sweep status:** {dp_done} / {dp_total} models trained")

eps_status_cols = st.columns(min(len(epsilon_values), 6))
for i, eps in enumerate(sorted(epsilon_values)):
    trained = status["dp_trained"].get(eps, False)
    with eps_status_cols[i % 6]:
        icon = "✅" if trained else "⏳"
        st.markdown(f"{icon} **ε={eps}**")

# Custom epsilon selector
st.markdown("**Select ε values to train:**")
selected_eps = st.multiselect(
    "Epsilon values",
    options=sorted(epsilon_values),
    default=sorted(epsilon_values),
    key="selected_eps",
    label_visibility="collapsed",
)

col_btn3, col_btn4, _ = st.columns([2, 2, 4])
with col_btn3:
    run_dp = st.button(
        "Run ε Sweep",
        type="primary",
        key="btn_run_dp",
        disabled=not selected_eps,
    )
with col_btn4:
    force_dp = st.checkbox("Force re-train DP", key="force_dp", value=False)

if run_dp and selected_eps:
    progress_bar = st.progress(0, text="Starting DP training...")
    results_placeholder = st.empty()

    try:
        from src.models.dp_model import train_dp_model, estimate_data_norm
        from src.data.sampler import get_split_arrays
        from src.evaluation.results_store import save_results, save_combined_results

        # Pre-compute data_norm once for the full sweep
        splits = get_split_arrays(dataset, label=label_type, scale=True)
        X_train, _ = splits["train"]
        data_norm = estimate_data_norm(X_train)

        dp_results = []
        for i, eps in enumerate(sorted(selected_eps)):
            progress_bar.progress(
                (i) / len(selected_eps),
                text=f"Training DP-LR at ε={eps}..."
            )
            result = train_dp_model(
                dataset, eps,
                label=label_type,
                data_norm=data_norm,
                force=force_dp,
            )
            dp_results.append(result)

            # Update live results table
            rows = []
            for r in dp_results:
                m = r["metrics"]["test"]
                rows.append({
                    "ε":           r["epsilon"],
                    "F1 (macro)":  f"{m.get('f1_macro', 0):.4f}",
                    "AUC":         f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") else "—",
                    "MCC":         f"{m.get('mcc', 0):.4f}",
                    "Train Time":  f"{r['train_time_s']:.1f}s",
                })
            results_placeholder.dataframe(
                pd.DataFrame(rows), use_container_width=False, hide_index=True
            )

        progress_bar.progress(1.0, text="ε sweep complete.")

        save_results(dp_results, dataset, "dp_sweep", label=label_type)
        save_combined_results(dataset, label=label_type)
        st.success(f"ε sweep complete. {len(dp_results)} DP model(s) trained and saved.")

    except Exception as exc:
        st.error(f"DP training failed: {exc}")
        st.exception(exc)

st.divider()

# ── Trained models on disk ─────────────────────────────────────────────────────
st.subheader("Models on Disk")
trained_models = list_trained_models(dataset, label=label_type)
if trained_models:
    df_models = pd.DataFrame(trained_models).drop(columns=["path"])
    df_models["epsilon"] = df_models["epsilon"].fillna("—")
    df_models.columns = [c.replace("_", " ").title() for c in df_models.columns]
    st.dataframe(df_models, use_container_width=True, hide_index=True)
else:
    st.info("No trained models found yet for this dataset/label combination.")
