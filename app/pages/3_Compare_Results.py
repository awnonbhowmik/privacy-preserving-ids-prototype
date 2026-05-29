"""
3_Compare_Results.py — Privacy–utility tradeoff analysis and comparison tables.

Shows:
  - Executive summary metrics
  - Baseline model comparison (table + grouped bar chart + radar chart)
  - Privacy–utility tradeoff curve (the central dissertation figure)
  - Multi-metric tradeoff table (Table 2)
  - DP-LR vs DP-SGD head-to-head table (Table 3)
  - Privacy budget analysis (actual ε vs target ε for DP-SGD)
  - Detection analysis: FPR vs Detection Rate
  - Confusion matrix viewer (raw + normalised)
  - Feature importance (Random Forest)
  - Full results table
  - Key Findings (auto-derived: best model, privacy cost, min viable ε, phase transition, head-to-head, noise floor)
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
from plotly.subplots import make_subplots

from src.models.model_utils import (
    get_pipeline_status,
    SUPPORTED_DATASETS,
)
from src.evaluation.results_store import (
    load_results,
    save_results,
    save_combined_results,
    build_comparison_table,
    build_tradeoff_table,
    export_to_csv,
)

st.title("📈 Compare Results")
st.markdown(
    "The central dissertation page. Compares non-private baseline classifiers against "
    "differentially private models (DP-LR and DP-SGD) across all ε values. "
    "The key research question: **how much detection performance must be sacrificed to "
    "achieve a meaningful privacy guarantee?**"
)

from app.glossary import render_glossary
render_glossary(sections=["ids", "dp"])

st.divider()

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

# Merge baselines + dp_sweep + dp_sgd_sweep → combined
_baselines_r    = load_results(dataset, "baselines",       label=label_type)
_dp_sweep_r     = load_results(dataset, "dp_sweep",        label=label_type)
_dp_sgd_r       = load_results(dataset, "dp_sgd_sweep", label=label_type)
_combined_fresh = _baselines_r + _dp_sweep_r + _dp_sgd_r
if _combined_fresh:
    save_results(_combined_fresh, dataset, "combined", label=label_type)
else:
    save_combined_results(dataset, label=label_type)
all_results = load_results(dataset, "combined", label=label_type)

if not all_results:
    st.warning(
        "No results found for this dataset. "
        "Go to **Train Models** and run the baseline and DP training first."
    )
    st.stop()

baseline_results = [r for r in all_results if r.get("epsilon") is None]
dp_lr_results    = sorted(
    [r for r in all_results
     if r.get("epsilon") is not None and r.get("model_name") == "dp_logistic_regression"],
    key=lambda r: r["epsilon"],
)
dp_rf_results    = sorted(
    [r for r in all_results
     if r.get("epsilon") is not None and r.get("model_name") == "dp_random_forest"],
    key=lambda r: r["epsilon"],
)
dp_nn_results    = sorted(
    [r for r in all_results
     if r.get("epsilon") is not None and r.get("model_name") == "dp_sgd"],
    key=lambda r: r["epsilon"],
)
dp_results = dp_lr_results + dp_rf_results + dp_nn_results

# ── Metric selector (shared across sections) ──────────────────────────────────
metric_key = st.selectbox(
    "Primary metric",
    ["f1_macro", "roc_auc", "mcc", "detection_rate"],
    format_func=lambda x: {
        "f1_macro":       "F1 Score (macro)",
        "roc_auc":        "ROC-AUC",
        "mcc":            "Matthews Correlation Coefficient",
        "detection_rate": "Detection Rate (Recall)",
    }[x],
    key="tradeoff_metric",
)
METRIC_LABEL = {
    "f1_macro":       "F1 Score (macro)",
    "roc_auc":        "ROC-AUC",
    "mcc":            "MCC",
    "detection_rate": "Detection Rate",
}[metric_key]

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 0: Executive Summary
# ═══════════════════════════════════════════════════════════════════════════════
if baseline_results or dp_results:
    st.subheader("Executive Summary")

    # Best baseline
    best_bl   = max(baseline_results, key=lambda r: r["metrics"][eval_split].get(metric_key, 0)) if baseline_results else None
    best_bl_v = best_bl["metrics"][eval_split].get(metric_key, 0) if best_bl else None

    # Best DP-RF
    best_rf   = max(dp_rf_results, key=lambda r: r["metrics"][eval_split].get(metric_key, 0)) if dp_rf_results else None
    best_rf_v = best_rf["metrics"][eval_split].get(metric_key, 0) if best_rf else None

    # Best DP-SGD
    best_nn   = max(dp_nn_results, key=lambda r: r["metrics"][eval_split].get(metric_key, 0)) if dp_nn_results else None
    best_nn_v = best_nn["metrics"][eval_split].get(metric_key, 0) if best_nn else None

    # Min ε (DP-SGD) within 5% of best baseline
    min_ok_eps = None
    if dp_nn_results and best_bl_v and best_bl_v > 0:
        for r in sorted(dp_nn_results, key=lambda r: r["epsilon"]):
            v = r["metrics"][eval_split].get(metric_key, 0)
            if (best_bl_v - v) / best_bl_v < 0.05:
                min_ok_eps = r["epsilon"]
                break

    # Best DP-LR
    best_lr   = max(dp_lr_results, key=lambda r: r["metrics"][eval_split].get(metric_key, 0)) if dp_lr_results else None
    best_lr_v = best_lr["metrics"][eval_split].get(metric_key, 0) if best_lr else None

    cols = st.columns(5)
    with cols[0]:
        with st.container(border=True):
            st.metric(
                f"Baseline ({METRIC_LABEL})",
                f"{best_bl_v:.4f}" if best_bl_v is not None else "—",
                help=(
                    "Best non-private baseline model — the performance ceiling with no privacy constraint. "
                    "All DP models are compared against this value."
                    + (f" Model: {best_bl['model_name'].replace('_', ' ').title()}" if best_bl else "")
                ),
            )
            if best_bl:
                st.caption(f"{best_bl['model_name'].replace('_', ' ').title()} · no DP")
    with cols[1]:
        with st.container(border=True):
            delta_lr = None
            if best_lr_v is not None and best_bl_v is not None and best_bl_v > 0:
                delta_lr = f"{100*(best_lr_v - best_bl_v)/best_bl_v:+.1f}% vs baseline"
            st.metric(
                f"DP-LR ({METRIC_LABEL})",
                f"{best_lr_v:.4f}" if best_lr_v is not None else "—",
                delta=delta_lr,
                help="Differentially Private Logistic Regression (IBM diffprivlib). Pure ε-DP (no δ).",
            )
            if best_lr:
                st.caption(f"ε={best_lr['epsilon']} · pure ε-DP")
    with cols[2]:
        with st.container(border=True):
            delta_rf = None
            if best_rf_v is not None and best_bl_v is not None and best_bl_v > 0:
                delta_rf = f"{100*(best_rf_v - best_bl_v)/best_bl_v:+.1f}% vs baseline"
            st.metric(
                f"DP-RF ({METRIC_LABEL})",
                f"{best_rf_v:.4f}" if best_rf_v is not None else "—",
                delta=delta_rf,
                help="DP Random Forest (diffprivlib). Random splits + PermuteAndFlip. Pure ε-DP (stronger than DP-SGD).",
            )
            if best_rf:
                st.caption(f"ε={best_rf['epsilon']} · pure ε-DP")
            elif not dp_rf_results:
                st.caption("Run: train_dp_rf.py")
    with cols[3]:
        with st.container(border=True):
            delta_nn = None
            if best_nn_v is not None and best_bl_v is not None and best_bl_v > 0:
                delta_nn = f"{100*(best_nn_v - best_bl_v)/best_bl_v:+.1f}% vs baseline"
            st.metric(
                f"DP-SGD ({METRIC_LABEL})",
                f"{best_nn_v:.4f}" if best_nn_v is not None else "—",
                delta=delta_nn,
                help="DP-SGD MLP (PyTorch Opacus). (ε,δ)-DP with δ=1e-5. Per-sample gradient clipping.",
            )
            if best_nn:
                st.caption(f"ε={best_nn['epsilon']} · (ε,δ)-DP")
    with cols[4]:
        with st.container(border=True):
            st.metric(
                "Min ε ≤ 5% F1 loss (DP-SGD)",
                str(min_ok_eps) if min_ok_eps is not None else "—",
                help=(
                    "The smallest (strongest) privacy budget ε at which DP-SGD still "
                    "achieves within 5% of the best non-private baseline. "
                    "This is the dissertation's recommended operating point: "
                    "maximum privacy protection while keeping detection performance acceptable."
                ),
            )
            st.caption("Recommended DP operating point")

    st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Baseline Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Table 1 — Baseline Model Comparison (Non-Private)")
st.caption(
    f"Evaluated on the **{eval_split}** split. All models trained on the same fixed train split. "
    "**Green = highest value in column, Red = lowest.** "
    "These results are the privacy-free performance ceilings — "
    "the best IDS detection rates achievable without any privacy constraint."
)
with st.expander("How to read this table"):
    st.markdown(
        "- **F1 (macro)**: Primary headline metric. Penalises models that sacrifice recall (missed attacks) or precision (false alarms).\n"
        "- **ROC-AUC**: Threshold-independent. A model with AUC = 0.97 separates attack from benign traffic well across all operating points.\n"
        "- **MCC**: Accounts for both classes fairly. Best single metric for imbalanced binary classification.\n"
        "- **FPR**: False alarm rate. An operational IDS must keep this low to avoid alert fatigue.\n"
        "- **Detection Rate**: Fraction of real attacks caught. The primary success criterion for an IDS.\n"
        "- **Train Time**: How long the model took to fit. Relevant for operational deployment.\n\n"
        "A good IDS model has: high F1 + high Detection Rate + low FPR."
    )

if baseline_results:
    baseline_df = build_comparison_table(baseline_results, split=eval_split)
    numeric_cols = baseline_df.select_dtypes(include=[float, int]).columns.tolist()
    styled = (
        baseline_df.style
        .highlight_max(subset=numeric_cols, props="background-color: #d4edda; color: #1a202c")
        .highlight_min(subset=numeric_cols, props="background-color: #f8d7da; color: #1a202c")
        .format(precision=4, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Grouped bar chart: all baselines, all metrics ─────────────────────────
    st.markdown("**Baseline Performance — Multi-Metric Comparison**")
    _metric_cols = ["f1_macro", "roc_auc", "mcc", "detection_rate", "false_positive_rate"]
    _metric_labels = {
        "f1_macro":           "F1 (macro)",
        "roc_auc":            "ROC-AUC",
        "mcc":                "MCC",
        "detection_rate":     "Detection Rate",
        "false_positive_rate":"False Positive Rate",
    }
    bar_rows = []
    for r in baseline_results:
        m = r["metrics"][eval_split]
        model = r["model_name"].replace("_", " ").title()
        for mk in _metric_cols:
            v = m.get(mk)
            if v is not None:
                bar_rows.append({"Model": model, "Metric": _metric_labels[mk], "Value": round(v, 4)})
    if bar_rows:
        bar_df = pd.DataFrame(bar_rows)
        fig_bar = px.bar(
            bar_df,
            x="Metric", y="Value", color="Model",
            barmode="group",
            text="Value",
            color_discrete_sequence=["#3498db", "#2ecc71", "#9b59b6"],
        )
        fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_bar.update_layout(
            height=380,
            yaxis=dict(range=[0, 1.1]),
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Radar / spider chart for baselines ────────────────────────────────────
    st.markdown("**Baseline Radar Chart (multi-metric profile)**")
    radar_metrics = ["f1_macro", "roc_auc", "mcc", "detection_rate"]
    radar_labels  = ["F1 (macro)", "ROC-AUC", "MCC", "Detection Rate"]
    radar_colors  = ["#3498db", "#2ecc71", "#9b59b6"]

    fig_radar = go.Figure()
    for idx, r in enumerate(baseline_results):
        m = r["metrics"][eval_split]
        vals = [m.get(mk, 0) or 0 for mk in radar_metrics]
        vals_closed = vals + [vals[0]]
        labels_closed = radar_labels + [radar_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            name=r["model_name"].replace("_", " ").title(),
            line_color=radar_colors[idx % len(radar_colors)],
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=380,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Multi-metric performance heatmap ──────────────────────────────────────
    st.markdown("**Performance Heatmap — All Metrics × All Baselines**")
    st.caption("Green = high score. Note: FPR lower is better — interpret that column inversely.")
    _hm_metrics = ["f1_macro", "f1_weighted", "precision_macro", "recall_macro",
                   "roc_auc", "mcc", "detection_rate", "false_positive_rate"]
    _hm_labels  = ["F1 (macro)", "F1 (weighted)", "Precision", "Recall",
                   "ROC-AUC", "MCC", "Detection Rate", "FPR"]
    _hm_data, _hm_models = [], []
    for r in baseline_results:
        m = r["metrics"][eval_split]
        _hm_models.append(r["model_name"].replace("_", " ").title())
        _hm_data.append([m.get(mk) for mk in _hm_metrics])
    if _hm_data:
        _hm_arr = np.array(
            [[v if v is not None else np.nan for v in row] for row in _hm_data],
            dtype=float,
        )
        _hm_text = np.where(
            np.isnan(_hm_arr), "",
            np.vectorize(lambda v: f"{v:.3f}")(_hm_arr),
        )
        fig_hm = px.imshow(
            _hm_arr,
            x=_hm_labels,
            y=_hm_models,
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            labels=dict(x="Metric", y="Model", color="Score"),
        )
        fig_hm.update_traces(text=_hm_text, texttemplate="%{text}", textfont_size=11)
        fig_hm.update_layout(
            height=max(220, len(_hm_models) * 70 + 100),
            margin=dict(l=150, r=20, t=30, b=80),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

else:
    st.info("No baseline results found. Train baselines on the Train Models page.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Privacy–Utility Tradeoff Curve
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Privacy–Utility Tradeoff Curve")
st.caption(
    "The central dissertation result: classifier performance vs. privacy budget ε. "
    "**Smaller ε = stronger privacy = more Gaussian noise injected into gradients = lower accuracy.** "
    "Dotted horizontal lines show the non-private baselines as performance ceilings."
)
with st.expander("How to read this chart"):
    st.markdown(
        "- The **x-axis (log scale)** shows the privacy budget ε. Moving left = stronger privacy.\n"
        "- The **y-axis** shows the selected performance metric (default: F1 macro).\n"
        "- **DP-LR** (red line): Logistic regression with DP via diffprivlib.\n"
        "- **DP-SGD** (orange line): Neural network MLP with DP via Opacus.\n"
        "- **Dotted horizontal lines**: Non-private baseline performance (no DP). These are the ceilings.\n"
        "- **Key question**: At what ε does the DP model come within 5% of the non-private baseline? "
        "That ε is the recommended operating point for the dissertation.\n\n"
        "A steep drop at low ε indicates the noise floor is the binding constraint. "
        "A flat curve means the model is robust to privacy noise."
    )

if dp_lr_results or dp_rf_results or dp_nn_results:
    fig = go.Figure()

    # ── DP-LR curve ───────────────────────────────────────────────────────────
    if dp_lr_results:
        fig.add_trace(go.Scatter(
            x=[r["epsilon"] for r in dp_lr_results],
            y=[r["metrics"][eval_split].get(metric_key) for r in dp_lr_results],
            mode="lines+markers",
            name="DP-LR (diffprivlib, pure ε-DP)",
            line=dict(color="#e74c3c", width=2.5),
            marker=dict(size=9, symbol="circle"),
            hovertemplate="ε=%{x}<br>" + METRIC_LABEL + "=%{y:.4f}<extra>DP-LR</extra>",
        ))

    # ── DP-RF curve ───────────────────────────────────────────────────────────
    if dp_rf_results:
        fig.add_trace(go.Scatter(
            x=[r["epsilon"] for r in dp_rf_results],
            y=[r["metrics"][eval_split].get(metric_key) for r in dp_rf_results],
            mode="lines+markers",
            name="DP-RF (diffprivlib, pure ε-DP)",
            line=dict(color="#27ae60", width=2.5),
            marker=dict(size=9, symbol="square"),
            hovertemplate="ε=%{x}<br>" + METRIC_LABEL + "=%{y:.4f}<extra>DP-RF</extra>",
        ))

    # ── DP-SGD curve ───────────────────────────────────────────────────────────
    if dp_nn_results:
        fig.add_trace(go.Scatter(
            x=[r["epsilon"] for r in dp_nn_results],
            y=[r["metrics"][eval_split].get(metric_key) for r in dp_nn_results],
            mode="lines+markers",
            name="DP-SGD / MLP (Opacus, (ε,δ)-DP)",
            line=dict(color="#f39c12", width=2.5),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="ε=%{x}<br>" + METRIC_LABEL + "=%{y:.4f}<extra>DP-SGD</extra>",
        ))

    # ── Baseline reference lines ──────────────────────────────────────────────
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
        color, lbl = baseline_colors.get(model_name, ("#95a5a6", model_name))
        fig.add_hline(
            y=val,
            line_dash="dot",
            line_color=color,
            line_width=1.5,
            annotation_text=f" {lbl}: {val:.3f}",
            annotation_position="right",
            annotation_font_size=11,
        )

    fig.update_layout(
        xaxis_title="Privacy Budget (ε) — lower = stronger privacy",
        yaxis_title=METRIC_LABEL,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=480,
        margin=dict(r=200, l=60, t=40, b=60),
        hovermode="x unified",
        xaxis=dict(type="log", title="Privacy Budget (ε) — log scale"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "X-axis shown on **log scale** to spread low-ε values. "
        "Dotted lines = non-private baseline ceilings."
    )
else:
    st.info("No DP results found. Run the ε sweep on the Train Models page.")

# ── Multi-metric tradeoff subplots ────────────────────────────────────────────
if dp_lr_results or dp_rf_results or dp_nn_results:
    st.markdown("**Privacy–Utility Tradeoff — F1, AUC and MCC simultaneously**")
    st.caption("All three metrics plotted against ε to show whether the tradeoff is metric-dependent.")
    _sub_metrics = ["f1_macro",        "roc_auc",  "mcc"]
    _sub_labels  = ["F1 Score (macro)", "ROC-AUC",  "MCC"]
    fig_sub = make_subplots(
        rows=1, cols=3,
        subplot_titles=_sub_labels,
        shared_yaxes=False,
    )
    for _ci, (_mk2, _) in enumerate(zip(_sub_metrics, _sub_labels), start=1):
        if dp_lr_results:
            fig_sub.add_trace(go.Scatter(
                x=[r["epsilon"] for r in dp_lr_results],
                y=[r["metrics"][eval_split].get(_mk2) for r in dp_lr_results],
                mode="lines+markers",
                name="DP-LR",
                showlegend=(_ci == 1),
                legendgroup="DP-LR",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=6),
            ), row=1, col=_ci)
        if dp_rf_results:
            fig_sub.add_trace(go.Scatter(
                x=[r["epsilon"] for r in dp_rf_results],
                y=[r["metrics"][eval_split].get(_mk2) for r in dp_rf_results],
                mode="lines+markers",
                name="DP-RF",
                showlegend=(_ci == 1),
                legendgroup="DP-RF",
                line=dict(color="#27ae60", width=2),
                marker=dict(size=6, symbol="square"),
            ), row=1, col=_ci)
        if dp_nn_results:
            fig_sub.add_trace(go.Scatter(
                x=[r["epsilon"] for r in dp_nn_results],
                y=[r["metrics"][eval_split].get(_mk2) for r in dp_nn_results],
                mode="lines+markers",
                name="DP-SGD",
                showlegend=(_ci == 1),
                legendgroup="DP-SGD",
                line=dict(color="#f39c12", width=2),
                marker=dict(size=6, symbol="diamond"),
            ), row=1, col=_ci)
    fig_sub.update_xaxes(type="log", title_text="ε (log scale)")
    fig_sub.update_yaxes(range=[0, 1.05])
    fig_sub.update_layout(
        height=360,
        margin=dict(l=50, r=30, t=50, b=60),
        legend=dict(orientation="h", y=-0.2, x=0.35),
    )
    st.plotly_chart(fig_sub, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Privacy–Utility Tradeoff Table
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Table 2 — Privacy–Utility Tradeoff")
st.caption(
    "Performance at each ε value vs the non-private Logistic Regression baseline. "
    "**Green ≤ 5% loss** (acceptable), **Yellow 5–15%** (noticeable but manageable), **Red > 15%** (significant)."
)
with st.expander("How to read this table"):
    st.markdown(
        "Each row is one DP model at one ε value. "
        "**Loss vs LR (%)** shows how much F1 dropped compared to the non-private LR baseline. "
        "The colouring directly maps to the dissertation's acceptability thresholds:\n\n"
        "- **Green (≤ 5% loss)**: The privacy cost is negligible — DP protection at very low utility cost.\n"
        "- **Yellow (5–15% loss)**: Meaningful tradeoff — privacy is gained at a noticeable but defensible cost.\n"
        "- **Red (> 15% loss)**: High privacy cost — the model is not yet useful enough for operational deployment.\n\n"
        "The dissertation aims to identify the lowest ε that stays in the green band."
    )

if dp_lr_results or dp_rf_results or dp_nn_results:
    tradeoff_df = build_tradeoff_table(
        dataset, label=label_type, metric_key=metric_key, split=eval_split
    )
    if not tradeoff_df.empty:
        def _highlight_loss(row):
            val = row.get("Loss vs LR (%)")
            if val is None:
                return [""] * len(row)
            if abs(val) <= 5.0:
                return ["background-color: #d4edda; color: #1a202c"] * len(row)
            elif abs(val) <= 15.0:
                return ["background-color: #fff3cd; color: #1a202c"] * len(row)
            else:
                return ["background-color: #f8d7da; color: #1a202c"] * len(row)

        st.dataframe(
            tradeoff_df.style.apply(_highlight_loss, axis=1),
            use_container_width=False,
            hide_index=True,
        )
        st.caption("🟢 ≤ 5% loss  |  🟡 5–15% loss  |  🔴 > 15% loss  (vs non-private LR)")
else:
    st.info("No DP results available yet.")

# ── Epsilon degradation heatmap ───────────────────────────────────────────────
if (dp_lr_results or dp_rf_results or dp_nn_results) and baseline_results:
    st.markdown("**Epsilon Degradation Heatmap — % Drop vs Best Baseline**")
    st.caption(
        "Each cell shows the percentage drop in that metric vs. the best non-private baseline. "
        "Green = low loss, Red = high loss."
    )
    _deg_metrics = ["f1_macro", "roc_auc", "mcc", "detection_rate"]
    _deg_labels  = ["F1 (macro)", "ROC-AUC", "MCC", "Detection Rate"]
    _bl_ref = {
        mk: max((r["metrics"][eval_split].get(mk) or 0) for r in baseline_results)
        for mk in _deg_metrics
    }
    _deg_rows, _deg_ylabels = [], []
    for r in dp_lr_results:
        m = r["metrics"][eval_split]
        _deg_rows.append([
            100 * (_bl_ref[mk] - (m.get(mk) or 0)) / _bl_ref[mk] if _bl_ref[mk] > 0 else 0
            for mk in _deg_metrics
        ])
        _deg_ylabels.append(f"DP-LR  ε={r['epsilon']}")
    for r in dp_rf_results:
        m = r["metrics"][eval_split]
        _deg_rows.append([
            100 * (_bl_ref[mk] - (m.get(mk) or 0)) / _bl_ref[mk] if _bl_ref[mk] > 0 else 0
            for mk in _deg_metrics
        ])
        _deg_ylabels.append(f"DP-RF  ε={r['epsilon']}")
    for r in dp_nn_results:
        m = r["metrics"][eval_split]
        _deg_rows.append([
            100 * (_bl_ref[mk] - (m.get(mk) or 0)) / _bl_ref[mk] if _bl_ref[mk] > 0 else 0
            for mk in _deg_metrics
        ])
        _deg_ylabels.append(f"DP-SGD ε={r['epsilon']}")
    if _deg_rows:
        _deg_arr  = np.array(_deg_rows, dtype=float)
        _deg_text = np.vectorize(lambda v: f"{v:.1f}%")(_deg_arr)
        fig_deg = px.imshow(
            _deg_arr,
            x=_deg_labels,
            y=_deg_ylabels,
            color_continuous_scale="RdYlGn_r",
            zmin=0,
            labels=dict(x="Metric", y="Model / ε", color="% Drop"),
        )
        fig_deg.update_traces(text=_deg_text, texttemplate="%{text}", textfont_size=10)
        fig_deg.update_layout(
            height=max(320, len(_deg_ylabels) * 22 + 100),
            margin=dict(l=160, r=20, t=30, b=80),
            coloraxis_colorbar=dict(title="% Drop"),
        )
        st.plotly_chart(fig_deg, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3b: DP-LR vs DP-SGD Head-to-Head
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Table 3 — DP-LR vs DP-SGD Head-to-Head")
st.caption(
    "Side-by-side comparison at each shared ε value. "
    "**DP-LR** = differentially private logistic regression (linear model, diffprivlib). "
    "**DP-SGD** = differentially private neural network (2-layer MLP, Opacus). "
    "🟢 = higher primary metric at this ε. 🔴 = lower."
)
with st.expander("Why compare DP-LR vs DP-SGD?"):
    st.markdown(
        "Both models offer (ε, δ)-differential privacy, but they differ in expressiveness and "
        "how noise is applied:\n\n"
        "- **DP-LR** is a linear model. Simpler, faster, more theoretically tractable. "
        "Noise is added directly to the objective.\n"
        "- **DP-SGD (MLP)** is a non-linear neural network. More expressive, potentially better "
        "at capturing complex attack patterns, but needs more training and is more sensitive to "
        "noise at low ε.\n\n"
        "At **high ε** (weak privacy), DP-SGD often wins because it can fit the data better. "
        "At **low ε** (strong privacy), DP-LR sometimes wins because the simpler model is less "
        "disrupted by heavy noise. This tradeoff is the dissertation's key empirical finding."
    )

if dp_lr_results and dp_nn_results:
    lr_by_eps = {r["epsilon"]: r["metrics"][eval_split] for r in dp_lr_results}
    nn_by_eps = {r["epsilon"]: r["metrics"][eval_split] for r in dp_nn_results}
    shared_eps = sorted(set(lr_by_eps) & set(nn_by_eps))

    if shared_eps:
        h2h_rows = []
        for eps in shared_eps:
            lr_m = lr_by_eps[eps]
            nn_m = nn_by_eps[eps]
            f1_lr  = lr_m.get(metric_key, 0) or 0
            f1_nn  = nn_m.get(metric_key, 0) or 0
            auc_lr = lr_m.get("roc_auc")
            auc_nn = nn_m.get("roc_auc")
            mcc_lr = lr_m.get("mcc", 0) or 0
            mcc_nn = nn_m.get("mcc", 0) or 0
            fpr_lr = lr_m.get("false_positive_rate")
            fpr_nn = nn_m.get("false_positive_rate")

            h2h_rows.append({
                "ε":           eps,
                f"DP-LR {METRIC_LABEL}": round(f1_lr, 4),
                f"DP-SGD {METRIC_LABEL}": round(f1_nn, 4),
                "DP-LR AUC":   round(auc_lr, 4) if auc_lr is not None else None,
                "DP-SGD AUC":   round(auc_nn, 4) if auc_nn is not None else None,
                "DP-LR MCC":   round(mcc_lr, 4),
                "DP-SGD MCC":   round(mcc_nn, 4),
                "DP-LR FPR":   round(fpr_lr, 4) if fpr_lr is not None else None,
                "DP-SGD FPR":   round(fpr_nn, 4) if fpr_nn is not None else None,
                "Winner":      "DP-SGD" if f1_nn > f1_lr else ("DP-LR" if f1_lr > f1_nn else "Tie"),
            })

        h2h_df = pd.DataFrame(h2h_rows)

        def _highlight_winner(row):
            styles = [""] * len(row)
            winner = row.get("Winner", "")
            cols   = list(h2h_df.columns)
            lr_idx = cols.index(f"DP-LR {METRIC_LABEL}")
            nn_idx = cols.index(f"DP-SGD {METRIC_LABEL}")
            if winner == "DP-SGD":
                styles[nn_idx] = "background-color: #d4edda; color: #1a202c"
                styles[lr_idx] = "background-color: #f8d7da; color: #1a202c"
            elif winner == "DP-LR":
                styles[lr_idx] = "background-color: #d4edda; color: #1a202c"
                styles[nn_idx] = "background-color: #f8d7da; color: #1a202c"
            return styles

        st.dataframe(
            h2h_df.style.apply(_highlight_winner, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("🟢 = higher primary metric at this ε  |  🔴 = lower")

        # ── Winner summary bar ─────────────────────────────────────────────────
        win_counts = h2h_df["Winner"].value_counts().to_dict()
        wc1, wc2, wc3, wc4 = st.columns([1, 1, 1, 2])
        wc1.metric("DP-SGD wins", win_counts.get("DP-SGD", 0))
        wc2.metric("DP-LR wins", win_counts.get("DP-LR", 0))
        wc3.metric("Ties",       win_counts.get("Tie", 0))
        with wc4:
            if st.button("Export Table 3 (Head-to-Head)", key="export_h2h"):
                path = export_to_csv(
                    h2h_df,
                    f"{dataset}_table3_h2h_{label_type}.csv",
                )
                st.success(f"Saved: `{path.name}`")
    else:
        st.info("No shared ε values between DP-LR and DP-SGD results yet.")
elif dp_lr_results:
    st.info("Train DP-SGD on the Train Models page to see the head-to-head comparison.")
else:
    st.info("Train both DP-LR and DP-SGD to see the head-to-head comparison.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Privacy Budget Analysis (DP-SGD)
# ═══════════════════════════════════════════════════════════════════════════════
if dp_nn_results:
    st.subheader("Privacy Budget Analysis — DP-SGD (Opacus RDP Accountant)")
    st.caption(
        "Opacus tracks the **actual ε** consumed across all training epochs via "
        "Rényi Differential Privacy (RDP) accounting. "
        "Training stops when the target ε is exhausted, so actual ε ≤ target ε."
    )
    with st.expander("What is RDP accounting and why does actual ε ≤ target ε?"):
        st.markdown(
            "Every gradient update in DP-SGD consumes a small fraction of the total privacy budget. "
            "The Rényi DP (RDP) accountant tracks the cumulative privacy cost using "
            "tighter composition bounds than naïve sequential composition.\n\n"
            "**Target ε** is the maximum privacy cost you are willing to accept. "
            "**Actual ε** is what Opacus reports at the end of training. "
            "Actual ε ≤ target ε because:\n"
            "- Training may stop before using the full budget (epoch limit reached first).\n"
            "- The RDP→(ε,δ)-DP conversion is conservative.\n\n"
            "A model with actual ε = 0.95 when target was 1.0 still provides a **(0.95, δ)-DP** guarantee, "
            "which is slightly stronger than requested."
        )

    budget_rows = []
    for r in dp_nn_results:
        target  = r["epsilon"]
        actual  = r.get("actual_epsilon", None)
        f1_val  = r["metrics"][eval_split].get("f1_macro", 0)
        auc_val = r["metrics"][eval_split].get("roc_auc")
        budget_rows.append({
            "Target ε":   target,
            "Actual ε":   round(actual, 4) if actual is not None else "—",
            "Budget used (%)": round(100 * actual / target, 1) if actual is not None else "—",
            "F1 (macro)": round(f1_val, 4),
            "ROC-AUC":    round(auc_val, 4) if auc_val else "—",
            "Train time (s)": round(r.get("train_time_s", 0), 1),
        })
    budget_df = pd.DataFrame(budget_rows)
    st.dataframe(budget_df, use_container_width=False, hide_index=True)

    # Budget utilisation bar
    valid_rows = [r for r in dp_nn_results if r.get("actual_epsilon") is not None]
    if valid_rows:
        butil_df = pd.DataFrame({
            "Target ε":      [r["epsilon"] for r in valid_rows],
            "Actual ε":      [r["actual_epsilon"] for r in valid_rows],
            "Unused budget": [r["epsilon"] - r["actual_epsilon"] for r in valid_rows],
        })
        fig_budget = go.Figure()
        fig_budget.add_bar(
            x=[str(e) for e in butil_df["Target ε"]],
            y=butil_df["Actual ε"],
            name="Actual ε spent",
            marker_color="#e74c3c",
        )
        fig_budget.add_bar(
            x=[str(e) for e in butil_df["Target ε"]],
            y=butil_df["Unused budget"],
            name="Unused budget",
            marker_color="#95a5a6",
        )
        fig_budget.update_layout(
            barmode="stack",
            xaxis_title="Target ε",
            yaxis_title="Privacy Budget (ε)",
            height=320,
            margin=dict(l=0, r=0, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_budget, use_container_width=True)
        st.caption(
            "Stacked bar shows how much of the target ε budget was actually consumed. "
            "Red = spent, grey = unused. "
            "At high ε, more epochs fit within the budget, so utilisation is higher."
        )

    # ── Training time analysis ────────────────────────────────────────────────
    st.markdown("**Training Time Analysis — DP-SGD**")
    _time_rows = [
        {
            "ε":              r["epsilon"],
            "Train time (s)": round(r.get("train_time_s", 0), 1),
            "F1 (macro)":     round(r["metrics"][eval_split].get("f1_macro", 0), 4),
        }
        for r in dp_nn_results
    ]
    if _time_rows:
        _time_df = pd.DataFrame(_time_rows)
        _tc1, _tc2 = st.columns(2)
        with _tc1:
            st.markdown("*Time per ε value*")
            fig_time = px.bar(
                _time_df, x="ε", y="Train time (s)",
                color="Train time (s)",
                color_continuous_scale="Oranges",
                text="Train time (s)",
            )
            fig_time.update_traces(texttemplate="%{text:.0f}s", textposition="outside")
            fig_time.update_layout(
                height=300, margin=dict(l=0, r=0, t=20, b=40),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_time, use_container_width=True)
        with _tc2:
            st.markdown("*Utility vs. Computation*")
            fig_tc = px.scatter(
                _time_df, x="Train time (s)", y="F1 (macro)",
                text="ε",
                color="ε",
                color_continuous_scale="Viridis",
            )
            fig_tc.update_traces(
                textposition="top center", textfont_size=9, marker_size=10,
            )
            fig_tc.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=40))
            st.plotly_chart(fig_tc, use_container_width=True)
        _total_s = _time_df["Train time (s)"].sum()
        _avg_s   = _total_s / len(_time_df)
        st.caption(
            f"Total sweep time: **{_total_s/60:.1f} min** "
            f"across {len(_time_df)} models (~{_avg_s:.0f}s average per ε)."
        )

    st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Detection Analysis — FPR vs Detection Rate
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Detection Analysis — False Positive Rate vs. Detection Rate")
st.caption(
    "Each point is one model. **Ideal position: bottom-right** (low FPR, high DR). "
    "Annotated by model type and ε value. "
    "FPR = fraction of benign traffic falsely flagged. DR = fraction of real attacks caught."
)
with st.expander("Why FPR vs Detection Rate matters for IDS"):
    st.markdown(
        "Accuracy alone is a poor metric for intrusion detection systems because the dataset "
        "is heavily imbalanced — most traffic is benign. A model that labels everything as "
        "benign achieves high accuracy but zero detection rate.\n\n"
        "**False Positive Rate (FPR = FP / (FP + TN))**\n"
        "In a real-world IDS, every false positive is an alert that a security analyst must "
        "investigate. If FPR is 1%, in a network handling 1 million flows per hour that is "
        "10,000 false alarms per hour — untenable. Low FPR is an **operational necessity**.\n\n"
        "**Detection Rate (DR = TP / (TP + FN))**\n"
        "Every false negative is an attack that slipped through undetected. "
        "Low DR means the IDS is failing its primary purpose. High DR is the **security objective**.\n\n"
        "**The ideal model** sits at the bottom-right of this chart: near-zero FPR with near-perfect DR. "
        "DP models typically move up and to the right (higher FPR, lower DR) as ε decreases."
    )

fpr_dr_rows = []
for r in baseline_results:
    m   = r["metrics"][eval_split]
    fpr = m.get("false_positive_rate")
    dr  = m.get("detection_rate")
    if fpr is not None and dr is not None:
        fpr_dr_rows.append({
            "Model":  r["model_name"].replace("_", " ").title(),
            "FPR":    fpr,
            "DR":     dr,
            "Type":   "Baseline",
            "ε":      "—",
            "Label":  r["model_name"].replace("_", " ").title(),
        })

for r in dp_lr_results:
    m   = r["metrics"][eval_split]
    fpr = m.get("false_positive_rate")
    dr  = m.get("detection_rate")
    if fpr is not None and dr is not None:
        fpr_dr_rows.append({
            "Model":  f"DP-LR ε={r['epsilon']}",
            "FPR":    fpr,
            "DR":     dr,
            "Type":   "DP-LR",
            "ε":      r["epsilon"],
            "Label":  f"ε={r['epsilon']}",
        })

for r in dp_rf_results:
    m   = r["metrics"][eval_split]
    fpr = m.get("false_positive_rate")
    dr  = m.get("detection_rate")
    if fpr is not None and dr is not None:
        fpr_dr_rows.append({
            "Model":  f"DP-RF ε={r['epsilon']}",
            "FPR":    fpr,
            "DR":     dr,
            "Type":   "DP-RF",
            "ε":      r["epsilon"],
            "Label":  f"ε={r['epsilon']}",
        })

for r in dp_nn_results:
    m   = r["metrics"][eval_split]
    fpr = m.get("false_positive_rate")
    dr  = m.get("detection_rate")
    if fpr is not None and dr is not None:
        fpr_dr_rows.append({
            "Model":  f"DP-SGD ε={r['epsilon']}",
            "FPR":    fpr,
            "DR":     dr,
            "Type":   "DP-SGD",
            "ε":      r["epsilon"],
            "Label":  f"ε={r['epsilon']}",
        })

if fpr_dr_rows:
    fpr_df = pd.DataFrame(fpr_dr_rows)
    color_map  = {"Baseline": "#3498db", "DP-LR": "#e74c3c", "DP-RF": "#27ae60", "DP-SGD": "#f39c12"}
    symbol_map = {"Baseline": "square",  "DP-LR": "circle",  "DP-RF": "square",  "DP-SGD": "diamond"}

    fig_fpr = go.Figure()
    for type_name, group in fpr_df.groupby("Type"):
        fig_fpr.add_trace(go.Scatter(
            x=group["FPR"],
            y=group["DR"],
            mode="markers+text",
            name=type_name,
            marker=dict(
                size=12,
                color=color_map.get(type_name, "#95a5a6"),
                symbol=symbol_map.get(type_name, "circle"),
                line=dict(width=1, color="white"),
            ),
            text=group["Label"],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "FPR: %{x:.4f}<br>"
                "Detection Rate: %{y:.4f}<extra></extra>"
            ),
        ))

    # Ideal point annotation
    fig_fpr.add_annotation(
        x=0, y=1,
        text="Ideal",
        showarrow=True,
        arrowhead=2,
        ax=40, ay=40,
        font=dict(color="green", size=12),
    )
    fig_fpr.update_layout(
        xaxis_title="False Positive Rate (FPR) — lower is better →",
        yaxis_title="Detection Rate (Recall) — higher is better →",
        height=480,
        margin=dict(l=60, r=40, t=30, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_fpr, use_container_width=True)

    # Summary table
    fpr_summary = fpr_df[["Model", "Type", "FPR", "DR"]].copy()
    fpr_summary["FPR"] = fpr_summary["FPR"].round(4)
    fpr_summary["DR"]  = fpr_summary["DR"].round(4)
    fpr_summary["FPR/DR ratio"] = (fpr_summary["FPR"] / fpr_summary["DR"].clip(lower=1e-9)).round(4)
    st.dataframe(
        fpr_summary.sort_values("FPR/DR ratio"),
        use_container_width=False,
        hide_index=True,
    )
    st.caption("Lower FPR/DR ratio = better detection efficiency.")
else:
    st.info("FPR and Detection Rate metrics not available. Train models to see this analysis.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5b: Precision vs. Recall Scatter
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Precision vs. Recall")
st.caption(
    "Each point is one model. Ideal: top-right (high precision, high recall). "
    "Shows whether DP models trade precision for recall or vice versa."
)
_pr_rows = []
for r in baseline_results:
    m = r["metrics"][eval_split]
    p, rc = m.get("precision_macro"), m.get("recall_macro")
    if p is not None and rc is not None:
        _pr_rows.append({"Label": r["model_name"].replace("_", " ").title(),
                         "Precision": p, "Recall": rc, "Type": "Baseline"})
for r in dp_lr_results:
    m = r["metrics"][eval_split]
    p, rc = m.get("precision_macro"), m.get("recall_macro")
    if p is not None and rc is not None:
        _pr_rows.append({"Label": f"ε={r['epsilon']}",
                         "Precision": p, "Recall": rc, "Type": "DP-LR"})
for r in dp_rf_results:
    m = r["metrics"][eval_split]
    p, rc = m.get("precision_macro"), m.get("recall_macro")
    if p is not None and rc is not None:
        _pr_rows.append({"Label": f"ε={r['epsilon']}",
                         "Precision": p, "Recall": rc, "Type": "DP-RF"})
for r in dp_nn_results:
    m = r["metrics"][eval_split]
    p, rc = m.get("precision_macro"), m.get("recall_macro")
    if p is not None and rc is not None:
        _pr_rows.append({"Label": f"ε={r['epsilon']}",
                         "Precision": p, "Recall": rc, "Type": "DP-SGD"})

if _pr_rows:
    _pr_df = pd.DataFrame(_pr_rows)
    _pr_color = {"Baseline": "#3498db", "DP-LR": "#e74c3c", "DP-RF": "#27ae60", "DP-SGD": "#f39c12"}
    _pr_sym   = {"Baseline": "square",  "DP-LR": "circle",  "DP-RF": "square",  "DP-SGD": "diamond"}
    fig_pr = go.Figure()
    for _type, _grp in _pr_df.groupby("Type"):
        _ts = str(_type)
        fig_pr.add_trace(go.Scatter(
            x=_grp["Recall"], y=_grp["Precision"],
            mode="markers+text",
            name=_ts,
            marker=dict(size=11, color=_pr_color.get(_ts, "#95a5a6"),
                        symbol=_pr_sym.get(_ts, "circle"),
                        line=dict(width=1, color="white")),
            text=_grp["Label"],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate="<b>%{text}</b><br>Precision: %{y:.4f}<br>Recall: %{x:.4f}<extra></extra>",
        ))
    fig_pr.update_layout(
        xaxis=dict(title="Recall — higher is better", range=[0, 1.05]),
        yaxis=dict(title="Precision — higher is better", range=[0, 1.05]),
        height=460,
        margin=dict(l=60, r=40, t=30, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_pr, use_container_width=True)
else:
    st.info("Train models to see the precision–recall scatter.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Confusion Matrix Viewer
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Confusion Matrix")
st.caption(
    "Shows exactly where the model makes errors. "
    "Each cell is a count of actual vs predicted class combinations. "
    "For IDS: **False Positives = false alarms** (benign flagged as attack), "
    "**False Negatives = missed attacks** (attack classified as benign)."
)
with st.expander("How to read the confusion matrix"):
    st.markdown(
        "For binary classification (Benign vs Attack):\n\n"
        "| | **Predicted: Benign** | **Predicted: Attack** |\n"
        "|---|---|---|\n"
        "| **Actual: Benign** | ✅ True Negative (TN) | ❌ False Positive (FP) |\n"
        "| **Actual: Attack** | ❌ False Negative (FN) | ✅ True Positive (TP) |\n\n"
        "- **TN**: Benign traffic correctly passed through. Good.\n"
        "- **FP**: Benign traffic falsely flagged. Causes alert fatigue.\n"
        "- **FN**: Attack that slipped through undetected. Security failure.\n"
        "- **TP**: Attack correctly detected. The primary success metric.\n\n"
        "Tick **Normalise** to see row-percentages (fraction of each actual class that was predicted correctly or not)."
    )

all_model_options = []
for r in baseline_results:
    lbl = r["model_name"].replace("_", " ").title()
    all_model_options.append((lbl, r))
for r in dp_lr_results:
    lbl = f"DP-LR ε={r['epsilon']}"
    all_model_options.append((lbl, r))
for r in dp_nn_results:
    lbl = f"DP-SGD ε={r['epsilon']}"
    all_model_options.append((lbl, r))

if all_model_options:
    cm_col1, cm_col2 = st.columns([2, 1])
    with cm_col1:
        selected_label = st.selectbox(
            "Select model",
            options=[x[0] for x in all_model_options],
            key="cm_model_select",
        )
    with cm_col2:
        normalise_cm = st.checkbox("Normalise (% of actual)", value=False, key="cm_normalise")

    selected_result = next(r for lbl, r in all_model_options if lbl == selected_label)
    cm = selected_result["metrics"][eval_split].get("confusion_matrix")

    if cm:
        cm_arr = np.array(cm, dtype=float)
        n = cm_arr.shape[0]

        if n == 2:
            axis_labels = ["Benign (0)", "Attack (1)"]
        else:
            axis_labels = [str(i) for i in range(n)]

        display_arr = cm_arr
        fmt = "%d"
        colorscale = "Blues"
        if normalise_cm:
            row_sums = cm_arr.sum(axis=1, keepdims=True)
            display_arr = cm_arr / np.where(row_sums == 0, 1, row_sums)
            fmt = ".2%"

        fig_cm = px.imshow(
            display_arr,
            x=axis_labels,
            y=axis_labels,
            color_continuous_scale=colorscale,
            text_auto=fmt,
            labels=dict(x="Predicted", y="Actual"),
            zmin=0, zmax=1 if normalise_cm else None,
        )
        fig_cm.update_layout(
            height=400,
            margin=dict(l=80, r=20, t=30, b=80),
            coloraxis_showscale=True,
        )
        st.plotly_chart(fig_cm, use_container_width=False)

        if n == 2 and not normalise_cm:
            tn, fp, fn, tp = int(cm_arr[0,0]), int(cm_arr[0,1]), int(cm_arr[1,0]), int(cm_arr[1,1])
            total = tn + fp + fn + tp
            cmc1, cmc2, cmc3, cmc4 = st.columns(4)
            cmc1.metric("True Negatives (TN)",  f"{tn:,}",  f"{100*tn/total:.1f}%",
                        help="Benign traffic correctly classified as benign. Good — no false alarm.")
            cmc2.metric("False Positives (FP)", f"{fp:,}",  f"{100*fp/total:.1f}%",
                        help="Benign traffic incorrectly flagged as attacks. Causes alert fatigue. Lower is better.")
            cmc3.metric("False Negatives (FN)", f"{fn:,}",  f"{100*fn/total:.1f}%",
                        help="Actual attacks that slipped through undetected (classified as benign). Security failure. Lower is better.")
            cmc4.metric("True Positives (TP)",  f"{tp:,}",  f"{100*tp/total:.1f}%",
                        help="Attacks correctly detected and flagged. The primary success criterion. Higher is better.")
    else:
        st.info("Confusion matrix not available for this result.")
else:
    st.info("Train models first to view confusion matrices.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Feature Importance (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Feature Importance — Random Forest")

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
            text="Importance",
        )
        fig_fi.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_fi.update_layout(
            height=max(320, top_n * 24),
            margin=dict(l=0, r=80, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption(
            "Random Forest feature importances (mean decrease in impurity). "
            "These are the most discriminative network traffic features for intrusion detection."
        )
else:
    st.info("Train the Random Forest baseline to view feature importances.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Full Results Table
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("All Results — Full Table")
with st.expander("Show all model results"):
    if all_results:
        full_df = build_comparison_table(all_results, split=eval_split)
        numeric_cols = full_df.select_dtypes(include=[float, int]).columns.tolist()
        st.dataframe(
            full_df.style
            .highlight_max(subset=numeric_cols, props="background-color: #d4edda; color: #1a202c")
            .format(precision=4, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Key Findings
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Key Findings")
st.caption(
    "Automatically derived from the loaded results. "
    "Findings update when you change dataset, label type, or evaluation split."
)

if all_results:
    # ── Pre-compute values needed across all finding cards ─────────────────
    _split = eval_split
    _mk    = "f1_macro"  # anchor metric for findings (always F1 for comparability)

    # Best across everything
    _all_with_metric = [r for r in all_results if r["metrics"][_split].get(_mk) is not None]
    _best_overall    = max(_all_with_metric, key=lambda r: r["metrics"][_split][_mk]) if _all_with_metric else None

    # Baseline
    _bl_f1s = {r["model_name"]: r["metrics"][_split].get(_mk, 0) for r in baseline_results}
    _best_bl_name = max(_bl_f1s, key=_bl_f1s.get) if _bl_f1s else None
    _best_bl_f1   = _bl_f1s[_best_bl_name] if _best_bl_name else None

    # DP-SGD sorted by epsilon
    _nn_sorted = sorted(dp_nn_results, key=lambda r: r["epsilon"])
    _nn_f1s    = [(r["epsilon"], r["metrics"][_split].get(_mk, 0)) for r in _nn_sorted]

    # DP-LR sorted by epsilon
    _lr_sorted = sorted(dp_lr_results, key=lambda r: r["epsilon"])
    _lr_f1s    = [(r["epsilon"], r["metrics"][_split].get(_mk, 0)) for r in _lr_sorted]

    # Min ε within 5% of best baseline (DP-SGD)
    _min_5pct, _min_10pct = None, None
    if _best_bl_f1 and _best_bl_f1 > 0:
        for eps, f1 in _nn_f1s:
            loss = (_best_bl_f1 - f1) / _best_bl_f1
            if _min_5pct  is None and loss < 0.05:  _min_5pct  = eps
            if _min_10pct is None and loss < 0.10:  _min_10pct = eps

    # Biggest single-step F1 gain in DP-SGD sweep (phase transition)
    _phase_eps, _phase_gain = None, 0.0
    for i in range(1, len(_nn_f1s)):
        gain = _nn_f1s[i][1] - _nn_f1s[i-1][1]
        if gain > _phase_gain:
            _phase_gain = gain
            _phase_eps  = (_nn_f1s[i-1][0], _nn_f1s[i][0])

    # DP-SGD vs DP-LR winner per shared epsilon
    _shared_eps = {eps for eps, _ in _nn_f1s} & {eps for eps, _ in _lr_f1s}
    _nn_wins, _lr_wins, _ties = 0, 0, 0
    for eps in _shared_eps:
        nn_v = next((f1 for e, f1 in _nn_f1s if e == eps), None)
        lr_v = next((f1 for e, f1 in _lr_f1s if e == eps), None)
        if nn_v is None or lr_v is None:
            continue
        if nn_v > lr_v + 0.001:   _nn_wins += 1
        elif lr_v > nn_v + 0.001: _lr_wins += 1
        else:                      _ties    += 1

    # Privacy cost: best DP-SGD at max eps vs best baseline
    _nn_max_eps_f1 = _nn_f1s[-1][1] if _nn_f1s else None
    _privacy_cost  = None
    if _nn_max_eps_f1 is not None and _best_bl_f1 and _best_bl_f1 > 0:
        _privacy_cost = 100 * (_best_bl_f1 - _nn_max_eps_f1) / _best_bl_f1

    # ── Layout: 2-column grid of finding cards ─────────────────────────────
    fc1, fc2 = st.columns(2)

    # Card 1 — Best Model Overall
    with fc1:
        with st.container(border=True):
            st.markdown("#### 🏆 Best Model Overall")
            if _best_overall:
                _mn  = _best_overall["model_name"].replace("_", " ").title()
                _f1v = _best_overall["metrics"][_split][_mk]
                _eps = _best_overall.get("epsilon")
                _eps_str = f" at ε={_eps}" if _eps is not None else ""
                st.markdown(
                    f"**{_mn}{_eps_str}** achieves the highest F1 of "
                    f"**{_f1v:.4f}** on the {_split} split."
                )
                if _best_bl_f1 and _best_overall.get("epsilon") is not None:
                    _gap = _f1v - _best_bl_f1
                    _sign = "above" if _gap >= 0 else "below"
                    st.markdown(
                        f"This is **{abs(_gap):.4f} {_sign}** the best non-private "
                        f"baseline ({_best_bl_name.replace('_', ' ').title()}: {_best_bl_f1:.4f})."
                    )
                elif _best_bl_f1:
                    st.markdown(
                        f"Best non-private baseline "
                        f"({_best_bl_name.replace('_', ' ').title()}): **{_best_bl_f1:.4f}**"
                    )

    # Card 2 — Privacy Cost of DP
    with fc2:
        with st.container(border=True):
            st.markdown("#### 🔒 Privacy Cost of DP-SGD")
            if _privacy_cost is not None:
                _max_nn_eps = _nn_f1s[-1][0]
                if _privacy_cost <= 2.0:
                    _verdict = "Negligible — DP-SGD at maximum ε is practically lossless."
                    _colour  = "green"
                elif _privacy_cost <= 5.0:
                    _verdict = "Low — strong utility preserved even under DP."
                    _colour  = "green"
                elif _privacy_cost <= 15.0:
                    _verdict = "Moderate — meaningful but acceptable tradeoff."
                    _colour  = "orange"
                else:
                    _verdict = "High — significant utility loss under DP."
                    _colour  = "red"
                st.markdown(
                    f"At ε={_max_nn_eps}, DP-SGD loses **{_privacy_cost:.1f}%** F1 "
                    f"vs the best baseline.  \n:{_colour}[{_verdict}]"
                )
            else:
                st.info("Train DP-SGD models to see privacy cost.")

    fc3, fc4 = st.columns(2)

    # Card 3 — Minimum Viable Epsilon
    with fc3:
        with st.container(border=True):
            st.markdown("#### ⚡ Minimum Viable Epsilon (DP-SGD)")
            if _min_5pct is not None:
                st.markdown(
                    f"DP-SGD reaches within **5% of baseline F1** at "
                    f"**ε = {_min_5pct}**."
                )
            elif _min_10pct is not None:
                st.markdown(
                    f"DP-SGD does not reach within 5% of baseline F1 at any tested ε.  \n"
                    f"Nearest: within **10%** at **ε = {_min_10pct}**."
                )
            else:
                st.markdown(
                    "DP-SGD does not come within 10% of baseline F1 at any tested ε.  \n"
                    "The noise floor is the binding constraint."
                )
            if _min_10pct is not None and _best_bl_f1:
                _f1_at_10 = next(
                    (f1 for e, f1 in _nn_f1s if e == _min_10pct), None
                )
                if _f1_at_10:
                    st.caption(
                        f"At ε={_min_10pct}: F1={_f1_at_10:.4f} "
                        f"(baseline: {_best_bl_f1:.4f})"
                    )

    # Card 4 — Phase Transition
    with fc4:
        with st.container(border=True):
            st.markdown("#### 📈 Sharpest Privacy–Utility Transition")
            if _phase_eps and _phase_gain > 0.005:
                _f1_before = next((f1 for e, f1 in _nn_f1s if e == _phase_eps[0]), None)
                _f1_after  = next((f1 for e, f1 in _nn_f1s if e == _phase_eps[1]), None)
                st.markdown(
                    f"The largest single-step F1 gain occurs between "
                    f"**ε={_phase_eps[0]}** and **ε={_phase_eps[1]}**: "
                    f"**+{_phase_gain:.4f}**."
                )
                if _f1_before is not None and _f1_after is not None:
                    st.caption(
                        f"F1 jumps from {_f1_before:.4f} → {_f1_after:.4f}. "
                        "This is the critical threshold where noise becomes manageable."
                    )
            elif _nn_f1s:
                st.markdown("F1 increases gradually across the full ε range — no sharp transition.")
            else:
                st.info("Train DP-SGD to see transition analysis.")

    fc5, fc6 = st.columns(2)

    # Card 5 — DP-SGD vs DP-LR head-to-head
    with fc5:
        with st.container(border=True):
            st.markdown("#### ⚔️ DP-SGD vs DP-LR (Head-to-Head)")
            if _shared_eps:
                _total = _nn_wins + _lr_wins + _ties
                if _nn_wins > _lr_wins:
                    _winner = "DP-SGD"
                    _icon   = "🟢"
                elif _lr_wins > _nn_wins:
                    _winner = "DP-LR"
                    _icon   = "🟡"
                else:
                    _winner = "tied"
                    _icon   = "⚪"
                st.markdown(
                    f"Across {_total} shared ε values:  \n"
                    f"**DP-SGD wins: {_nn_wins}** | "
                    f"**DP-LR wins: {_lr_wins}** | "
                    f"**Tied: {_ties}**  \n"
                    f"{_icon} Overall winner: **{_winner}**"
                )
                # Low / high epsilon breakdown
                _low_eps  = [e for e in _shared_eps if e <= 0.5]
                _high_eps = [e for e in _shared_eps if e >= 1.0]
                _nn_low_w = sum(
                    1 for e in _low_eps
                    if (next((f for ep, f in _nn_f1s if ep == e), 0) or 0) >
                       (next((f for ep, f in _lr_f1s if ep == e), 0) or 0) + 0.001
                )
                _nn_hi_w  = sum(
                    1 for e in _high_eps
                    if (next((f for ep, f in _nn_f1s if ep == e), 0) or 0) >
                       (next((f for ep, f in _lr_f1s if ep == e), 0) or 0) + 0.001
                )
                if _low_eps:
                    st.caption(
                        f"Low privacy regime (ε≤0.5): DP-SGD wins {_nn_low_w}/{len(_low_eps)}. "
                        f"High privacy regime (ε≥1.0): DP-SGD wins {_nn_hi_w}/{len(_high_eps)}."
                    )
            else:
                st.info("Train both DP-LR and DP-SGD to compare.")

    # Card 6 — Noise Floor Assessment
    with fc6:
        with st.container(border=True):
            st.markdown("#### 🔬 Noise Floor Assessment")
            if len(_nn_f1s) >= 3:
                # Compare F1 at lowest 2 epsilons vs highest 2
                _range = _nn_f1s[-1][1] - _nn_f1s[0][1]
                st.markdown(
                    f"F1 range across all ε: "
                    f"**{_nn_f1s[0][1]:.4f}** (ε={_nn_f1s[0][0]}) "
                    f"→ **{_nn_f1s[-1][1]:.4f}** (ε={_nn_f1s[-1][0]})  \n"
                    f"Total gain: **+{_range:.4f}**"
                )
                if _range < 0.05:
                    st.caption(
                        "Narrow F1 range — the model is relatively robust to privacy noise, "
                        "suggesting the noise floor is not the primary bottleneck."
                    )
                elif _range < 0.20:
                    st.caption(
                        "Moderate F1 range — privacy noise has a measurable but bounded effect."
                    )
                else:
                    st.caption(
                        "Wide F1 range — strong noise floor effect at low ε; "
                        "significant gains as privacy budget relaxes."
                    )
                # Plateau detection: last 3 epsilons
                if len(_nn_f1s) >= 4:
                    _tail_gains = [
                        abs(_nn_f1s[i][1] - _nn_f1s[i-1][1])
                        for i in range(len(_nn_f1s)-3, len(_nn_f1s))
                    ]
                    if max(_tail_gains) < 0.005:
                        st.caption(
                            f"⚠️ Performance plateaus at high ε "
                            f"(ε≥{_nn_f1s[-3][0]}), suggesting convergence — "
                            "further relaxing privacy yields diminishing returns."
                        )
            else:
                st.info("Train DP-SGD across more ε values for noise floor analysis.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Export
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Export Tables")

col_e1, col_e2, col_e3, col_e4 = st.columns(4)
with col_e1:
    if st.button("Export Baseline Table", key="export_baseline"):
        if baseline_results:
            df = build_comparison_table(baseline_results, split=eval_split)
            path = export_to_csv(df, f"{dataset}_table1_baselines_{label_type}.csv")
            st.success(f"Saved: `{path.name}`")
        else:
            st.warning("No baseline results.")

with col_e2:
    if st.button("Export Tradeoff Table", key="export_tradeoff"):
        df = build_tradeoff_table(dataset, label=label_type, metric_key=metric_key, split=eval_split)
        if not df.empty:
            path = export_to_csv(df, f"{dataset}_table2_tradeoff_{label_type}.csv")
            st.success(f"Saved: `{path.name}`")
        else:
            st.warning("No tradeoff data.")

with col_e3:
    if st.button("Export All Results", key="export_all"):
        if all_results:
            df = build_comparison_table(all_results, split=eval_split)
            path = export_to_csv(df, f"{dataset}_all_results_{label_type}.csv")
            st.success(f"Saved: `{path.name}`")
        else:
            st.warning("No results.")

with col_e4:
    if dp_nn_results and st.button("Export DP-SGD Budget Analysis", key="export_budget"):
        _budget_rows = []
        for _r in dp_nn_results:
            _target = _r["epsilon"]
            _actual = _r.get("actual_epsilon")
            _f1     = _r["metrics"][eval_split].get("f1_macro", 0)
            _auc    = _r["metrics"][eval_split].get("roc_auc")
            _budget_rows.append({
                "Target ε":        _target,
                "Actual ε":        round(_actual, 4) if _actual else "—",
                "Budget used (%)": round(100 * _actual / _target, 1) if _actual else "—",
                "F1 (macro)":      round(_f1, 4),
                "ROC-AUC":         round(_auc, 4) if _auc else "—",
                "Train time (s)":  round(_r.get("train_time_s", 0), 1),
            })
        if _budget_rows:
            df_b = pd.DataFrame(_budget_rows)
            path = export_to_csv(df_b, f"{dataset}_dpnn_budget_{label_type}.csv")
            st.success(f"Saved: `{path.name}`")
