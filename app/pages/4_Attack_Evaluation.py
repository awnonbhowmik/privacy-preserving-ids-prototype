"""
4_Attack_Evaluation.py — Phase 3: Membership Inference Attack Evaluation.

Answers the dissertation question:
    Does differential privacy reduce susceptibility to membership inference
    attacks, and by how much as ε varies?

Key visualisations:
    1. Attack ROC-AUC vs ε — shows privacy protection improving as ε decreases.
    2. Attack Advantage vs ε — how far above random guessing (0.5 AUC) the
       attacker is for each model.
    3. Combined table — ε / IDS F1 / IDS ROC-AUC / Attack ROC-AUC / Advantage.
    4. Per-model attack metric breakdown.

How to interpret:
    - attack_roc_auc ≈ 0.5 : attacker cannot distinguish members from
      non-members — close to random guessing, strong privacy protection.
    - attack_roc_auc >> 0.5 : attacker distinguishes members, model leaks
      membership information.
    - Lower ε models should trend toward attack_roc_auc ≈ 0.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

st.set_page_config(
    page_title="Attack Evaluation — Privacy-Preserving IDS",
    page_icon="🔍",
    layout="wide",
)

try:
    from app._path_setup import setup_path
    setup_path()
except ImportError:
    pass

from src.attacks.membership_inference import (
    build_combined_dissertation_table,
    build_mia_table,
    load_mia_results,
)
from src.models.model_utils import SUPPORTED_DATASETS

# ── Page header ───────────────────────────────────────────────────────────────
st.title("Phase 3 — Membership Inference Attack Evaluation")
st.markdown(
    "Empirically measures whether Differential Privacy reduces the ability of "
    "a black-box attacker to infer whether a specific record was part of the model's training set. "
    "This evaluates the **privacy protection** side of the dissertation's argument — "
    "not just detection performance loss, but actual reduction in information leakage."
)

from app.glossary import render_glossary
render_glossary(sections=["mia"])

with st.expander("Attack setup and methodology", expanded=False):
    st.markdown(
        """
### What question does this evaluate?

> *Can an adversary determine whether a specific network flow record was used to train the IDS model,
> based only on the model's predicted probability scores?*

If yes: the model leaks training membership information — a privacy violation.
If no (near-random): the model provides plausible deniability about its training data.

---

### How the attack works (black-box)

1. **Members**: Sample records from the training set. The model was trained on these.
2. **Non-members**: Sample records from the held-out test set. The model never saw these during training.
3. **Attack features** (computed from `predict_proba()` output):
   - `max_conf` — maximum predicted probability: high for members (model is more confident on training data)
   - `entropy` — prediction uncertainty: low for members
   - `loss` — cross-entropy loss for the true label: low for members
   - `correct` — 1 if prediction is correct: higher for members
4. **Meta-classifier**: A logistic regression trained to predict "is this record a member?" from the attack features.
5. **Evaluate**: Attack ROC-AUC and advantage are computed on a held-out attack test set.

---

### What the metrics mean

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Attack ROC-AUC** | 0.5 | Ideal — attacker cannot distinguish members from non-members |
| **Attack ROC-AUC** | 1.0 | Worst case — perfect membership inference |
| **Attack Advantage** | 0 | No information leaked above random guessing |
| **Attack Advantage** | 0.1 | Attacker has 10 percentage-point edge over random — moderate leak |
| **TPR @ FPR=0.01** | Low | Attacker correctly identifies few members when false-positive rate is constrained to 1% |

**Expected result**: As ε decreases (stronger DP), Attack ROC-AUC should trend toward 0.5.
"""
    )

st.divider()

# ── Dataset / label selectors ─────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    dataset = st.selectbox(
        "Dataset",
        SUPPORTED_DATASETS,
        format_func=lambda x: {"cic_ids2018": "CIC-IDS2018", "unsw_nb15": "UNSW-NB15"}[x],
        key="atk_dataset",
    )
with col2:
    label = st.selectbox(
        "Label type",
        ["binary", "multiclass"],
        key="atk_label",
    )

# ── Load results ──────────────────────────────────────────────────────────────
results = load_mia_results(dataset, label)

if not results:
    st.warning(
        "No membership inference results found for this dataset/label combination. "
        "Run the attack evaluation first:"
        "\n\n```\npython scripts/run_membership_inference.py "
        f"--dataset {dataset} --label {label}\n```"
    )
    st.stop()

mia_df = build_mia_table(results)
combined_df = build_combined_dissertation_table(dataset, label)

# ── Summary cards ─────────────────────────────────────────────────────────────
st.subheader("Summary")
metrics_cols = st.columns(4)

non_private = [r for r in results if r.get("epsilon") is None]
dp_models = [r for r in results if r.get("epsilon") is not None]

with metrics_cols[0]:
    if non_private:
        avg_np_auc = sum(r["attack_roc_auc"] for r in non_private) / len(non_private)
        st.metric(
            "Non-private avg attack AUC",
            f"{avg_np_auc:.3f}",
            help=(
                "Average membership inference attack ROC-AUC across all non-private baseline models. "
                "This is the privacy risk without DP. "
                "Values > 0.5 indicate the model leaks training membership. "
                "0.5 = random guessing (ideal). 1.0 = perfect attack."
            ),
        )

with metrics_cols[1]:
    if dp_models:
        best_dp = min(dp_models, key=lambda r: r["attack_roc_auc"])
        st.metric(
            "Best DP attack AUC",
            f"{best_dp['attack_roc_auc']:.3f}",
            delta=f"ε={best_dp['epsilon']}",
            delta_color="off",
            help=(
                "Lowest (best for privacy) attack ROC-AUC across all DP models. "
                "This is the strongest privacy protection achieved. "
                "A value near 0.5 means the attacker cannot distinguish training members from non-members."
            ),
        )

with metrics_cols[2]:
    if non_private and dp_models:
        best_dp_auc = min(r["attack_roc_auc"] for r in dp_models)
        reduction = avg_np_auc - best_dp_auc
        st.metric(
            "Attack AUC reduction (DP effect)",
            f"{reduction:.3f}",
            help=(
                "How much DP reduced the attacker's ROC-AUC at the tightest ε. "
                "A larger reduction shows DP is empirically reducing membership leakage, "
                "not just providing a theoretical guarantee."
            ),
        )

with metrics_cols[3]:
    below_threshold = [r for r in dp_models if r["attack_roc_auc"] < 0.55]
    if below_threshold:
        min_eps_safe = min(r["epsilon"] for r in below_threshold)
        st.metric(
            "Min ε with near-random attack (AUC < 0.55)",
            f"ε={min_eps_safe}",
            help=(
                "The tightest (strongest) ε at which the attacker's ROC-AUC falls below 0.55 — "
                "close to the 0.5 random-guessing baseline. "
                "At this ε, DP is providing near-complete membership-inference protection."
            ),
        )

st.divider()

# ── Plot: Attack ROC-AUC vs ε ─────────────────────────────────────────────────
try:
    import plotly.graph_objects as go

    st.subheader("Attack ROC-AUC vs Privacy Budget (ε)")
    st.caption(
        "Each line shows the attacker's discrimination ability (ROC-AUC) as a function of ε. "
        "**Lower is better for privacy.** The green dashed line at 0.5 is random guessing — the ideal. "
        "Red dotted lines show the non-private baseline risk level. "
        "The dissertation claim: as ε decreases, attack AUC should trend toward 0.5."
    )

    dp_results_by_model: dict[str, list] = {}
    for r in results:
        if r.get("epsilon") is not None:
            name = r["target_model"]
            dp_results_by_model.setdefault(name, []).append(r)

    fig = go.Figure()

    colours = {
        "dp_logistic_regression": "#1f77b4",
        "dp_sgd": "#ff7f0e",
    }

    for model_name, model_results in dp_results_by_model.items():
        model_results_sorted = sorted(model_results, key=lambda r: r["epsilon"])
        eps_vals = [r["epsilon"] for r in model_results_sorted]
        auc_vals = [r["attack_roc_auc"] for r in model_results_sorted]
        display = {
            "dp_logistic_regression": "DP-LR",
            "dp_sgd": "DP-SGD",
        }.get(model_name, model_name)

        fig.add_trace(go.Scatter(
            x=eps_vals,
            y=auc_vals,
            mode="lines+markers",
            name=display,
            line=dict(color=colours.get(model_name), width=2),
            marker=dict(size=7),
        ))

    # Baseline: random guessing reference
    all_eps = [r["epsilon"] for r in results if r.get("epsilon") is not None]
    if all_eps:
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="green",
            annotation_text="Random guessing (AUC=0.5)",
            annotation_position="bottom right",
        )

    # Non-private baselines as horizontal dotted lines
    for r in non_private:
        display = {
            "logistic_regression": "LR (non-private)",
            "random_forest": "RF (non-private)",
            "xgboost": "XGBoost (non-private)",
        }.get(r["target_model"], r["target_model"])
        fig.add_hline(
            y=r["attack_roc_auc"],
            line_dash="dot",
            line_color="red",
            opacity=0.5,
            annotation_text=display,
            annotation_position="top right",
        )

    fig.update_layout(
        xaxis_title="Privacy budget ε (log scale)",
        yaxis_title="Attack ROC-AUC",
        yaxis=dict(range=[0.45, 1.0]),
        xaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

except ImportError:
    st.info("Install plotly for interactive charts: `pip install plotly`")

# ── Plot: Attack Advantage vs ε ───────────────────────────────────────────────
try:
    import plotly.graph_objects as go2

    st.subheader("Attack Advantage vs ε")
    st.caption(
        "Advantage = Attack ROC-AUC − 0.5. "
        "**0 = no information leakage (ideal).** Positive values mean the attacker has a measurable edge over random guessing. "
        "The green dashed line at 0 is the ideal target. "
        "DP models should show advantage trending toward 0 as ε decreases."
    )

    fig2 = go2.Figure()

    for model_name, model_results in dp_results_by_model.items():
        model_results_sorted = sorted(model_results, key=lambda r: r["epsilon"])
        display = {
            "dp_logistic_regression": "DP-LR",
            "dp_sgd": "DP-SGD",
        }.get(model_name, model_name)
        fig2.add_trace(go2.Scatter(
            x=[r["epsilon"] for r in model_results_sorted],
            y=[r["attack_advantage"] for r in model_results_sorted],
            mode="lines+markers",
            name=display,
            line=dict(color=colours.get(model_name), width=2),
            marker=dict(size=7),
        ))

    fig2.add_hline(y=0.0, line_dash="dash", line_color="green",
                   annotation_text="Zero advantage (ideal)", annotation_position="bottom right")
    fig2.update_layout(
        xaxis_title="Privacy budget ε (log scale)",
        yaxis_title="Attack Advantage (AUC − 0.5)",
        xaxis_type="log",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig2, use_container_width=True)

except ImportError:
    pass

st.divider()

# ── Combined dissertation table ───────────────────────────────────────────────
st.subheader("Combined Dissertation Table — IDS Performance + Attack Risk")
st.caption(
    "The dissertation's unified privacy-utility-risk table. "
    "Each row is one model at one ε value. "
    "**Left columns**: IDS performance (F1, AUC, FPR, DR). "
    "**Right columns**: Attack success (Attack ROC-AUC, Advantage). "
    "Use this to support the claim that lower ε simultaneously reduces detection performance "
    "and membership-inference attack success — confirming DP is working as intended."
)
with st.expander("How to read this table for your dissertation defence"):
    st.markdown(
        "A good dissertation result shows:\n\n"
        "1. **Non-private baselines**: High F1 + High AUC + Low FPR + High DR + **High attack AUC** (privacy risk).\n"
        "2. **DP models at low ε**: Moderate F1 + Moderate AUC + **Low attack AUC (near 0.5)**. "
        "Privacy protection achieved, some utility cost.\n"
        "3. **DP models at high ε**: High F1 + High AUC + **Moderate attack AUC**. "
        "Good utility, partial privacy protection.\n\n"
        "The **(ε, F1 macro, Attack ROC-AUC)** triple is the core empirical result. "
        "**Colour coding**: Redder attack AUC = more privacy risk. Greener = better privacy protection."
    )

if not combined_df.empty:
    st.dataframe(
        combined_df.style.background_gradient(
            subset=["Attack ROC-AUC", "Attack Advantage"],
            cmap="RdYlGn_r",
            vmin=0.5, vmax=0.8,
        ).set_properties(
            subset=["Attack ROC-AUC", "Attack Advantage"],
            **{"color": "#1a202c"},
        ).format({
            "F1 macro": "{:.4f}",
            "ROC-AUC (IDS)": "{:.4f}",
            "FPR": "{:.4f}",
            "Detection Rate": "{:.4f}",
            "Attack ROC-AUC": "{:.4f}",
            "Attack Advantage": "{:.4f}",
        }, na_rep="—"),
        use_container_width=True,
    )

    if st.button("Export combined table to CSV", key="export_combined"):
        from src.evaluation.results_store import export_to_csv
        path = export_to_csv(
            combined_df,
            f"{dataset}_combined_dissertation_table_{label}.csv",
        )
        st.success(f"Exported to {path}")

# ── Full attack metrics table ─────────────────────────────────────────────────
with st.expander("Full attack metrics table", expanded=False):
    if not mia_df.empty:
        st.dataframe(mia_df, use_container_width=True)

        if st.button("Export MIA table to CSV", key="export_mia"):
            from src.evaluation.results_store import export_to_csv
            path = export_to_csv(
                mia_df,
                f"{dataset}_mia_summary_{label}.csv",
            )
            st.success(f"Exported to {path}")

# ── Run attack button ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Run Attack Evaluation")

with st.form("mia_form"):
    n_samples = st.slider(
        "Attack samples per model (members + non-members each)",
        min_value=500, max_value=10000, value=5000, step=500,
    )
    force = st.checkbox("Force re-run (overwrite existing results)", value=False)
    submitted = st.form_submit_button("Run MIA sweep")

if submitted:
    from src.attacks.membership_inference import run_mia_sweep
    with st.spinner("Running membership inference evaluation…"):
        new_results = run_mia_sweep(
            dataset_name=dataset,
            label=label,
            n_attack_samples=n_samples,
            force=force,
        )
    if new_results:
        st.success(f"Done — {len(new_results)} models evaluated.")
        st.rerun()
    else:
        st.warning("No new results generated. Check logs for errors.")
