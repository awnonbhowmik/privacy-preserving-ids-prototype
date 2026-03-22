"""
3_Compare_Results.py — Privacy–utility tradeoff analysis and comparison tables.

Shows:
  - Baseline model comparison table (Table 1)
  - Privacy–utility tradeoff curve (the central dissertation figure)
  - Privacy–utility tradeoff table (Table 2)
  - Confusion matrix viewer
  - Feature importance (Random Forest)
  - Export controls
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
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.models.model_utils import (
    get_pipeline_status,
    list_trained_models,
    SUPPORTED_DATASETS,
)
from src.evaluation.results_store import (
    load_results,
    save_combined_results,
    build_comparison_table,
    build_tradeoff_table,
    export_to_csv,
)
from src.utils.config import PRIVACY_CFG

st.set_page_config(page_title="Compare Results", page_icon="📈", layout="wide")
st.title("Compare Results")

# ── Dataset and label selectors ───────────────────────────────────────────────
col_ds, col_lb, col_split, _ = st.columns([2, 2, 2, 2])
with col_ds:
    dataset = st.selectbox(
        "Dataset",
        SUPPORTED_DATASETS,
        format_func=lambda x: {"cic_ids2018": "CIC-IDS2018", "unsw_nb15": "UNSW-NB15"}[x],
        index=SUPPORTED_DATASETS.index(st.session_state.get("global_dataset", "cic_ids2018")),
        key="compare_dataset",
    )
with col_lb:
    label_type = st.selectbox(
        "Label type",
        ["binary", "multiclass"],
        index=["binary", "multiclass"].index(st.session_state.get("global_label", "binary")),
        key="compare_label",
    )
with col_split:
    eval_split = st.selectbox("Evaluate on", ["test", "val"], key="compare_split")

status = get_pipeline_status(dataset, label=label_type)

# Merge combined results silently before loading
save_combined_results(dataset, label=label_type)
all_results = load_results(dataset, "combined", label=label_type)

if not all_results:
    st.warning(
        "No results found for this dataset. "
        "Go to **Train Models** and run the baseline and DP training first."
    )
    st.stop()

baseline_results = [r for r in all_results if r.get("epsilon") is None]
dp_results       = sorted(
    [r for r in all_results if r.get("epsilon") is not None],
    key=lambda r: r["epsilon"],
)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Baseline Comparison Table
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Table 1 — Baseline Model Comparison")
st.caption(f"Evaluated on the **{eval_split}** split. All models trained on the same fixed train split.")

if baseline_results:
    baseline_df = build_comparison_table(baseline_results, split=eval_split)

    # Style: highlight best value in each numeric column
    numeric_cols = baseline_df.select_dtypes(include=[float, int]).columns.tolist()
    styled = (
        baseline_df.style
        .highlight_max(subset=numeric_cols, color="#d4edda")
        .format({c: "{:.4f}" for c in numeric_cols}, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("No baseline results found. Train baselines on the Train Models page.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Privacy–Utility Tradeoff Curve (the key dissertation figure)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Privacy–Utility Tradeoff Curve")
st.caption(
    "The central result: F1 score vs. privacy budget ε. "
    "Horizontal reference lines show non-private baseline performance."
)

metric_key = st.selectbox(
    "Metric",
    ["f1_macro", "roc_auc", "mcc", "detection_rate"],
    format_func=lambda x: {
        "f1_macro": "F1 Score (macro)",
        "roc_auc": "ROC-AUC",
        "mcc": "Matthews Correlation Coefficient",
        "detection_rate": "Detection Rate (Recall)",
    }[x],
    key="tradeoff_metric",
)

if dp_results:
    # ── Build DP curve data ───────────────────────────────────────────────────
    eps_vals  = [r["epsilon"] for r in dp_results]
    dp_metric = [r["metrics"][eval_split].get(metric_key) for r in dp_results]

    fig = go.Figure()

    # DP curve
    fig.add_trace(go.Scatter(
        x=eps_vals,
        y=dp_metric,
        mode="lines+markers",
        name="DP-LR",
        line=dict(color="#e74c3c", width=2.5),
        marker=dict(size=9, symbol="circle"),
    ))

    # Baseline reference lines
    baseline_colors = {
        "logistic_regression": ("#2ecc71", "LR (non-private)"),
        "random_forest":       ("#3498db", "Random Forest"),
        "xgboost":             ("#9b59b6", "XGBoost"),
    }
    for r in baseline_results:
        model_name = r.get("model_name", "")
        val = r["metrics"][eval_split].get(metric_key)
        if val is None:
            continue
        color, label = baseline_colors.get(model_name, ("#95a5a6", model_name))
        fig.add_hline(
            y=val,
            line_dash="dash",
            line_color=color,
            line_width=1.5,
            annotation_text=f" {label}: {val:.3f}",
            annotation_position="right",
            annotation_font_size=11,
        )

    fig.update_layout(
        xaxis_title="Privacy Budget (ε) — lower = stronger privacy",
        yaxis_title=metric_key.replace("_", " ").title(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=450,
        margin=dict(r=200, l=60, t=30, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No DP results found. Run the ε sweep on the Train Models page.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Privacy–Utility Tradeoff Table
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Table 2 — Privacy–Utility Tradeoff")
st.caption("Shows F1 loss relative to the non-private Logistic Regression baseline.")

if dp_results:
    tradeoff_df = build_tradeoff_table(
        dataset, label=label_type, metric_key=metric_key, split=eval_split
    )
    if not tradeoff_df.empty:
        # Highlight the row where loss becomes acceptable (< 5%)
        def _highlight_loss(row):
            val = row.get("Loss vs LR (%)")
            if val is None:
                return [""] * len(row)
            if abs(val) <= 5.0:
                return ["background-color: #d4edda"] * len(row)
            elif abs(val) <= 15.0:
                return ["background-color: #fff3cd"] * len(row)
            else:
                return ["background-color: #f8d7da"] * len(row)

        st.dataframe(
            tradeoff_df.style.apply(_highlight_loss, axis=1),
            use_container_width=False,
            hide_index=True,
        )
        st.caption(
            "🟢 < 5% loss from baseline  |  🟡 5–15% loss  |  🔴 > 15% loss"
        )
else:
    st.info("No DP results available yet.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Confusion Matrix Viewer
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Confusion Matrix")

# Build selection options from available results
all_model_options = []
for r in baseline_results:
    label = r["model_name"].replace("_", " ").title()
    all_model_options.append((label, r))
for r in dp_results:
    label = f"DP-LR ε={r['epsilon']}"
    all_model_options.append((label, r))

if all_model_options:
    selected_label = st.selectbox(
        "Select model",
        options=[x[0] for x in all_model_options],
        key="cm_model_select",
    )
    selected_result = next(r for lbl, r in all_model_options if lbl == selected_label)
    cm = selected_result["metrics"][eval_split].get("confusion_matrix")

    if cm:
        cm_arr = np.array(cm)
        n = cm_arr.shape[0]

        if n == 2:
            axis_labels = ["Benign (0)", "Attack (1)"]
        else:
            axis_labels = [str(i) for i in range(n)]

        fig_cm = px.imshow(
            cm_arr,
            x=axis_labels,
            y=axis_labels,
            color_continuous_scale="Blues",
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
        )
        fig_cm.update_layout(
            height=350,
            margin=dict(l=60, r=20, t=30, b=60),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=False)
    else:
        st.info("Confusion matrix not available for this result.")
else:
    st.info("Train models first to view confusion matrices.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Feature Importance (Random Forest only)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Feature Importance (Random Forest)")

if status["baselines_trained"].get("random_forest", False):
    from src.models.baseline import get_feature_importances

    importances = get_feature_importances(dataset, label=label_type)
    if importances:
        top_n = st.slider("Show top N features", min_value=5, max_value=30, value=15, key="fi_top_n")
        top_features = list(importances.items())[:top_n]
        fi_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])

        fig_fi = px.bar(
            fi_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(
            height=max(300, top_n * 22),
            margin=dict(l=0, r=10, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.info("Train the Random Forest baseline to view feature importances.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Export
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Export Tables")

col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    if st.button("Export Baseline Table (CSV)", key="export_baseline"):
        if baseline_results:
            df = build_comparison_table(baseline_results, split=eval_split)
            path = export_to_csv(df, f"{dataset}_table1_baselines_{label_type}.csv")
            st.success(f"Exported to `{path.name}`")
        else:
            st.warning("No baseline results to export.")

with col_e2:
    if st.button("Export Tradeoff Table (CSV)", key="export_tradeoff"):
        df = build_tradeoff_table(dataset, label=label_type, metric_key=metric_key, split=eval_split)
        if not df.empty:
            path = export_to_csv(df, f"{dataset}_table2_tradeoff_{label_type}.csv")
            st.success(f"Exported to `{path.name}`")
        else:
            st.warning("No tradeoff data to export.")

with col_e3:
    if st.button("Export All Results (CSV)", key="export_all"):
        if all_results:
            df = build_comparison_table(all_results, split=eval_split)
            path = export_to_csv(df, f"{dataset}_all_results_{label_type}.csv")
            st.success(f"Exported to `{path.name}`")
        else:
            st.warning("No results to export.")
