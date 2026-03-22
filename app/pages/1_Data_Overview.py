"""
1_Data_Overview.py — Dataset status, preprocessing controls, and distribution charts.

Allows the user to:
  - Inspect which raw CSV files are present
  - Run the preprocessing pipeline (Stage A: raw → Parquet)
  - Run the sampling pipeline (Stage B: Parquet → subset)
  - View class distribution charts for the working subset
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
import plotly.express as px
import plotly.graph_objects as go

from src.models.model_utils import get_pipeline_status, get_subset_stats, SUPPORTED_DATASETS
from src.utils.config import PATHS

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")
st.title("Data Overview")

# ── Dataset selector ──────────────────────────────────────────────────────────
dataset = st.selectbox(
    "Dataset",
    SUPPORTED_DATASETS,
    format_func=lambda x: {"cic_ids2018": "CIC-IDS2018", "unsw_nb15": "UNSW-NB15"}[x],
    index=SUPPORTED_DATASETS.index(st.session_state.get("global_dataset", "cic_ids2018")),
    key="data_overview_dataset",
)

status = get_pipeline_status(dataset)

# ── Stage A: Preprocessing ───────────────────────────────────────────────────
st.subheader("Stage A — Preprocessing")

raw_dir = PATHS["raw_data"] / dataset
raw_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []

if not raw_files:
    st.error(
        f"No CSV files found in `data/raw/{dataset}/`. "
        "Place the raw dataset CSVs there before continuing."
    )
else:
    st.success(f"Found **{len(raw_files)} CSV file(s)** in `data/raw/{dataset}/`")
    with st.expander("Show file list"):
        for f in raw_files:
            size_mb = f.stat().st_size / 1e6
            st.text(f"  {f.name}  ({size_mb:.1f} MB)")

if status["processed_exists"]:
    processed_path = PATHS["processed_data"] / f"{dataset}.parquet"
    size_mb = processed_path.stat().st_size / 1e6
    st.success(
        f"Processed Parquet exists (`data/processed/{dataset}.parquet`, "
        f"{size_mb:.1f} MB)"
    )
else:
    st.warning("Processed Parquet does not exist yet.")
    if raw_files:
        if st.button("Run Preprocessing Pipeline", type="primary", key="run_preproc"):
            with st.spinner("Preprocessing... this may take several minutes for large datasets."):
                try:
                    from src.data.preprocessor import run_preprocessing_pipeline
                    run_preprocessing_pipeline(dataset, force=False)
                    st.success("Preprocessing complete.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Preprocessing failed: {exc}")

st.divider()

# ── Stage B: Sampling ────────────────────────────────────────────────────────
st.subheader("Stage B — Sampling")

if not status["processed_exists"]:
    st.info("Complete Stage A first.")
elif status["subset_exists"]:
    subset_path = PATHS["subsets"] / f"{dataset}_subset.parquet"
    size_mb = subset_path.stat().st_size / 1e6
    st.success(
        f"Subset Parquet exists (`data/subsets/{dataset}_subset.parquet`, "
        f"{size_mb:.1f} MB)"
    )

    col_force, _ = st.columns([1, 3])
    with col_force:
        if st.button("Re-create Subset", key="recreate_subset"):
            with st.spinner("Creating subset..."):
                try:
                    from src.data.sampler import run_sampling_pipeline
                    run_sampling_pipeline(dataset, force=True)
                    st.success("Subset re-created.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Sampling failed: {exc}")
else:
    st.warning("Working subset does not exist yet.")
    if st.button("Create Stratified Subset", type="primary", key="run_sampling"):
        with st.spinner("Creating subset..."):
            try:
                from src.data.sampler import run_sampling_pipeline
                run_sampling_pipeline(dataset, force=False)
                st.success("Subset created.")
                st.rerun()
            except Exception as exc:
                st.error(f"Sampling failed: {exc}")

st.divider()

# ── Subset statistics and charts ─────────────────────────────────────────────
st.subheader("Dataset Statistics")

stats = get_subset_stats(dataset)

if stats is None:
    st.info("No subset found. Complete Stage B to see statistics.")
else:
    # ── Summary metrics row ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",  f"{stats['total_rows']:,}")
    c2.metric("Features",    stats["n_features"])
    c3.metric("Train Split", f"{stats['split_counts'].get('train', 0):,}")
    c4.metric("Test Split",  f"{stats['split_counts'].get('test', 0):,}")

    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    # ── Class distribution bar chart (multiclass) ─────────────────────────────
    with col_chart1:
        st.markdown("**Attack Category Distribution**")
        class_dist = stats["class_distribution"]
        df_dist = (
            pd.DataFrame.from_dict(class_dist, orient="index", columns=["count"])
            .reset_index()
            .rename(columns={"index": "Category"})
            .sort_values("count", ascending=True)
        )
        fig = px.bar(
            df_dist,
            x="count",
            y="Category",
            orientation="h",
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={"count": "Number of Samples"},
        )
        fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Binary class pie chart ────────────────────────────────────────────────
    with col_chart2:
        st.markdown("**Binary Label Balance**")
        bin_dist = stats["binary_distribution"]
        labels_map = {0: "Benign (0)", 1: "Attack (1)"}
        labels = [labels_map.get(k, str(k)) for k in bin_dist.keys()]
        values = list(bin_dist.values())
        fig2 = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=["#2ecc71", "#e74c3c"],
        )])
        fig2.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Split breakdown table ─────────────────────────────────────────────────
    st.markdown("**Split Breakdown**")
    split_df = pd.DataFrame([
        {"Split": k, "Rows": f"{v:,}", "Fraction": f"{100*v/stats['total_rows']:.1f}%"}
        for k, v in stats["split_counts"].items()
    ])
    st.dataframe(split_df, use_container_width=False, hide_index=True)
