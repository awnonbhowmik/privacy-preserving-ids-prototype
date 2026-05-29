"""
2_Train_Models.py — Baseline and DP model training controls.

Allows the user to:
  - Train all three baseline models (LR, RF, XGBoost)
  - Run the DP-LR epsilon sweep
  - See training status for each model
  - View a live results summary table after training
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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
from src.utils.config import PRIVACY_CFG, DP_SGD_CFG

st.title("Train Models")
st.markdown(
    "Train non-private baseline models and differentially private models on the fixed "
    "training split. All models use the same preprocessed data so their performance is "
    "directly comparable."
)

from app.glossary import render_glossary
render_glossary(sections=["dp"])

st.divider()

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
st.subheader("Baseline Models (Non-Private)")
st.caption(
    "Trains Logistic Regression, Random Forest, and XGBoost on the fixed train split "
    "**without any privacy mechanism**. These are the performance ceilings — the best "
    "achievable detection rates when privacy is not a constraint. "
    "Results are saved to `results/` and can be loaded without retraining."
)
with st.expander("About each baseline model"):
    st.markdown(
        "**Logistic Regression (LR)** — Linear classifier. Fast, interpretable, "
        "and used as the DP-LR comparison target because diffprivlib implements "
        "a differentially private version of the same algorithm.\n\n"
        "**Random Forest (RF)** — Ensemble of decision trees. Handles non-linear "
        "decision boundaries, provides feature importances, and is typically the "
        "strongest non-deep-learning baseline on tabular IDS data.\n\n"
        "**XGBoost** — Gradient-boosted trees. GPU-accelerated when CUDA is available. "
        "Typically the best-performing non-private model, making it the hardest "
        "performance ceiling for the DP models to match. "
        "No DP-XGBoost library exists — the XGBoost library itself has no DP support "
        "because data-driven split selection (its core advantage) is fundamentally "
        "difficult to privatise. DP-RF is the closest DP equivalent."
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
st.subheader("DP-LR Experiment — ε Sweep (diffprivlib)")
st.caption(
    "Trains Differentially Private Logistic Regression at each ε value. "
    "Uses IBM diffprivlib, which adds calibrated noise to the objective function. "
    "**Smaller ε = stronger privacy guarantee = more noise = lower accuracy.** "
    "The sweep across ε values produces the privacy–utility tradeoff curve shown on the Compare Results page."
)
with st.expander("How DP-LR privacy works"):
    st.markdown(
        "Logistic regression minimises a loss function over the training data. "
        "diffprivlib adds **Gaussian noise** to the gradient proportional to the "
        "sensitivity of the gradient (bounded by `data_norm`) divided by ε. "
        "A smaller ε means more noise, which prevents the model from memorising "
        "any individual training record — at the cost of accuracy.\n\n"
        "The **data norm** is estimated from the 99th percentile of training sample "
        "L2 norms to bound how much any single sample can shift the gradient."
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

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2b: DP-RF — Differentially Private Random Forest (diffprivlib)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("DP-RF — Differentially Private Random Forest (diffprivlib)")
st.caption(
    "Trains a DP Random Forest using IBM diffprivlib. "
    "Uses **random split selection** (not data-driven) and the PermuteAndFlip mechanism "
    "for leaf predictions. Provides **pure ε-DP** — a stronger formal guarantee than "
    "DP-SGD's (ε, δ)-DP because there is no failure probability δ."
)
with st.expander("How DP-RF differs from regular Random Forest and from DP-SGD"):
    st.markdown(
        "**Standard Random Forest** selects the best split threshold by computing "
        "information gain across all training samples — this leaks information about "
        "individual records.\n\n"
        "**DP-RF (diffprivlib)** instead:\n"
        "1. Chooses split features and thresholds **uniformly at random** — the tree "
        "structure uses no information from the data, so it costs zero privacy budget.\n"
        "2. At each leaf, applies the **PermuteAndFlip mechanism** to the class labels "
        "of records reaching that leaf. This is where the ε budget is spent.\n\n"
        "**Why not DP-XGBoost?** XGBoost uses data-driven split selection "
        "(gradient/hessian statistics). Making that step private (via the exponential "
        "mechanism or noisy histograms) is research-level work with no established "
        "pip package. diffprivlib's DP-RF avoids the problem by randomising splits "
        "rather than optimising them.\n\n"
        "**Privacy type comparison:**\n"
        "- DP-LR: pure ε-DP (no δ)\n"
        "- DP-RF: pure ε-DP (no δ) — same as DP-LR, stronger than DP-SGD\n"
        "- DP-SGD: (ε, δ)-DP — δ=1e-5 failure probability\n"
    )

dp_rf_eps = PRIVACY_CFG.get("epsilon_values", [])
dp_rf_done  = sum(status.get("dp_rf_trained", {}).values())
dp_rf_total = len(status.get("dp_rf_trained", {}))
st.markdown(f"**DP-RF ε sweep status:** {dp_rf_done} / {dp_rf_total} models trained")

rf_status_cols = st.columns(min(len(dp_rf_eps), 6))
for i, eps in enumerate(sorted(dp_rf_eps)):
    trained = status.get("dp_rf_trained", {}).get(eps, False)
    with rf_status_cols[i % 6]:
        st.markdown(f"{'✅' if trained else '⏳'} **ε={eps}**")

st.markdown("**Select ε values to train:**")
selected_rf_eps = st.multiselect(
    "DP-RF epsilon values",
    options=sorted(dp_rf_eps),
    default=sorted(dp_rf_eps),
    key="selected_rf_eps",
    label_visibility="collapsed",
)

col_rf1, col_rf2, _ = st.columns([2, 2, 4])
with col_rf1:
    run_dp_rf = st.button(
        "Train DP-RF Models",
        type="primary",
        key="btn_run_dp_rf",
        disabled=not selected_rf_eps,
    )
with col_rf2:
    force_dp_rf = st.checkbox("Force re-train DP-RF", key="force_dp_rf", value=False)

if run_dp_rf and selected_rf_eps:
    rf_progress = st.progress(0, text="Starting DP-RF training...")
    rf_placeholder = st.empty()
    try:
        from src.models.dp_rf_model import train_dp_rf_model, estimate_feature_bounds
        from src.data.sampler import get_split_arrays
        from src.evaluation.results_store import save_results, save_combined_results

        splits   = get_split_arrays(dataset, label=label_type)
        X_train, _ = splits["train"]
        bounds   = estimate_feature_bounds(X_train)

        rf_results = []
        for i, eps in enumerate(sorted(selected_rf_eps)):
            rf_progress.progress(i / len(selected_rf_eps), text=f"Training DP-RF at ε={eps}...")
            r = train_dp_rf_model(
                dataset, eps, label=label_type, bounds=bounds, force=force_dp_rf
            )
            rf_results.append(r)
            rows = []
            for res in rf_results:
                m = res["metrics"]["test"]
                rows.append({
                    "ε":           res["epsilon"],
                    "F1 (macro)":  f"{m.get('f1_macro', 0):.4f}",
                    "AUC":         f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") else "—",
                    "MCC":         f"{m.get('mcc', 0):.4f}",
                    "Train Time":  f"{res['train_time_s']:.1f}s",
                })
            rf_placeholder.dataframe(pd.DataFrame(rows), use_container_width=False, hide_index=True)

        rf_progress.progress(1.0, text="DP-RF sweep complete.")
        save_results(rf_results, dataset, "dp_rf_sweep", label=label_type)
        save_combined_results(dataset, label=label_type)
        st.success(f"DP-RF sweep complete. {len(rf_results)} model(s) trained and saved.")

    except Exception as exc:
        st.error(f"DP-RF training failed: {exc}")
        st.exception(exc)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Phase 2 — DP-SGD (Opacus)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("DP-SGD Neural Network — ε Sweep (Opacus)")
st.caption(
    "Trains a 2-hidden-layer MLP (input → 128 → 64 → output) with "
    "Differentially Private SGD via PyTorch Opacus at each ε value. "
    "More expressive than DP-LR but requires more data and epochs to converge under noise."
)
with st.expander("How DP-SGD privacy works"):
    st.markdown(
        "**Per-sample gradient clipping**: During each mini-batch, the gradient for "
        "every individual training sample is computed separately and clipped to a "
        "maximum L2 norm (`max_grad_norm = 1.0`). This bounds the **sensitivity** — "
        "how much any one record can change the gradient update.\n\n"
        "**Gaussian noise addition**: After clipping, Gaussian noise is added to the "
        "summed gradients. The noise scale is calibrated so that the total privacy cost "
        "across all epochs satisfies the (ε, δ)-DP guarantee.\n\n"
        "**RDP Accounting (Rényi DP)**: Opacus tracks the cumulative privacy cost "
        "across all gradient steps using the Rényi DP accountant and stops training "
        "when the target ε is reached. The **actual ε** spent may be ≤ target ε.\n\n"
        "**Class imbalance**: A `WeightedRandomSampler` is used so attack samples are "
        "seen more frequently during training. The weights are inversely proportional "
        "to class frequency. This is compatible with DP accounting."
    )

dp_sgd_eps_values = DP_SGD_CFG.get("epsilon_values", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

# Status per epsilon
dp_sgd_done  = sum(status.get("dp_sgd_trained", {}).values())
dp_sgd_total = len(status.get("dp_sgd_trained", {}))
st.markdown(f"**DP-SGD ε sweep status:** {dp_sgd_done} / {dp_sgd_total} models trained")

nn_status_cols = st.columns(min(len(dp_sgd_eps_values), 6))
for i, eps in enumerate(sorted(dp_sgd_eps_values)):
    trained = status.get("dp_sgd_trained", {}).get(eps, False)
    with nn_status_cols[i % 6]:
        icon = "✅" if trained else "⏳"
        st.markdown(f"{icon} **ε={eps}**")

# Custom epsilon selector
st.markdown("**Select ε values to train:**")
selected_nn_eps = st.multiselect(
    "DP-SGD epsilon values",
    options=sorted(dp_sgd_eps_values),
    default=sorted(dp_sgd_eps_values),
    key="selected_nn_eps",
    label_visibility="collapsed",
)

col_btn5, col_btn6, _ = st.columns([2, 2, 4])
with col_btn5:
    run_dp_nn = st.button(
        "Train DP-SGD Models",
        type="primary",
        key="btn_run_dp_nn",
        disabled=not selected_nn_eps,
    )
with col_btn6:
    force_dp_nn = st.checkbox("Force re-train DP-SGD", key="force_dp_nn", value=False)

if run_dp_nn and selected_nn_eps:
    nn_progress_bar      = st.progress(0, text="Starting DP-SGD training...")
    nn_results_placeholder = st.empty()

    try:
        from src.models.dp_sgd_model import train_dp_sgd_model
        from src.evaluation.results_store import save_results, load_results

        dp_sgd_results = []
        for i, eps in enumerate(sorted(selected_nn_eps)):
            nn_progress_bar.progress(
                i / len(selected_nn_eps),
                text=f"Training DP-SGD at ε={eps}... (epoch logs in terminal)"
            )
            result = train_dp_sgd_model(
                dataset, eps,
                label=label_type,
                force=force_dp_nn,
            )
            dp_sgd_results.append(result)

            # Live results table
            rows = []
            for r in dp_sgd_results:
                m = r["metrics"]["test"]
                rows.append({
                    "ε":              r["epsilon"],
                    "Actual ε":       r.get("actual_epsilon", "—"),
                    "F1 (macro)":     f"{m.get('f1_macro', 0):.4f}",
                    "AUC":            f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") else "—",
                    "MCC":            f"{m.get('mcc', 0):.4f}",
                    "Train Time (s)": f"{r['train_time_s']:.1f}",
                })
            nn_results_placeholder.dataframe(
                pd.DataFrame(rows), use_container_width=False, hide_index=True
            )

        nn_progress_bar.progress(1.0, text="DP-SGD sweep complete.")

        # Save dp_sgd_sweep results and update combined
        save_results(dp_sgd_results, dataset, "dp_sgd_sweep", label=label_type)
        baselines = load_results(dataset, "baselines",  label=label_type)
        dp_sweep  = load_results(dataset, "dp_sweep",   label=label_type)
        save_results(baselines + dp_sweep + dp_sgd_results, dataset, "combined", label=label_type)

        st.success(f"DP-SGD sweep complete. {len(dp_sgd_results)} model(s) trained and saved.")

    except ImportError:
        st.error(
            "PyTorch and Opacus are required for DP-SGD training. "
            "Install them with:  `pip install torch opacus`"
        )
    except Exception as exc:
        st.error(f"DP-SGD training failed: {exc}")
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
