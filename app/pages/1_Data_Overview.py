"""
1_Data_Overview.py — Dataset status, preprocessing controls, and distribution charts.

Allows the user to:
  - Inspect which raw CSV files are present
  - Run the preprocessing pipeline (Stage A: raw → Parquet)
  - Run the sampling pipeline (Stage B: Parquet → subset)
  - View class distribution charts for the working subset
  - View feature statistics and split summary
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

from src.models.model_utils import get_pipeline_status, get_subset_stats, SUPPORTED_DATASETS
from src.utils.config import PATHS

st.title("📊 Data Overview")
st.markdown(
    "This page shows the status of the two IDS datasets used in the dissertation, "
    "controls for running the preprocessing pipeline, and summary statistics for the "
    "working data subset that all models are trained and evaluated on."
)

from app.glossary import render_glossary
render_glossary(sections=["data"])

st.divider()

# ── Dataset selector ──────────────────────────────────────────────────────────
dataset = st.selectbox(
    "Dataset",
    SUPPORTED_DATASETS,
    format_func=lambda x: {"cic_ids2018": "CIC-IDS2018", "unsw_nb15": "UNSW-NB15"}[x],
    index=SUPPORTED_DATASETS.index(st.session_state.get("global_dataset", "cic_ids2018")),
    key="data_overview_dataset",
)

status = get_pipeline_status(dataset)

# ── Stage A: Preprocessing ────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### Stage A — Raw CSV → Processed Parquet")
    st.caption(
        "Reads all raw CSV files, normalises column names, drops duplicates, "
        "replaces infinite values with NaN, maps raw labels to the dissertation taxonomy, "
        "and saves a single cleaned Parquet file. Run once per dataset."
    )

    raw_dir = PATHS["raw_data"] / dataset
    raw_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []

    col_status, col_action = st.columns([3, 1])
    with col_status:
        if not raw_files:
            st.error(
                f"No CSV files found in `data/raw/{dataset}/`. "
                "Place the raw dataset CSVs there before continuing."
            )
        else:
            st.success(f"Found **{len(raw_files)} CSV file(s)** in `data/raw/{dataset}/`")

        if status["processed_exists"]:
            processed_path = PATHS["processed_data"] / f"{dataset}.parquet"
            size_mb = processed_path.stat().st_size / 1e6
            st.success(f"Processed Parquet: `data/processed/{dataset}.parquet` ({size_mb:.1f} MB)")
        else:
            st.warning("Processed Parquet does not exist yet.")

        with st.expander("Show raw file list"):
            for f in raw_files:
                size_mb = f.stat().st_size / 1e6
                st.text(f"  {f.name}  ({size_mb:.1f} MB)")

    with col_action:
        if not status["processed_exists"] and raw_files:
            if st.button("▶ Run Preprocessing", type="primary", key="run_preproc"):
                with st.spinner("Preprocessing... this may take several minutes."):
                    try:
                        from src.data.preprocessor import run_preprocessing_pipeline
                        run_preprocessing_pipeline(dataset, force=False)
                        st.success("Done.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed: {exc}")

st.divider()

# ── Stage B: Sampling ─────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### Stage B — Processed Parquet → Stratified Subset")
    st.caption(
        "Draws a stratified sample of ~300,000 rows (proportional by attack category) "
        "and assigns each row to train / val / test (60% / 20% / 20%) using stratified splitting. "
        "This fixed subset is the single data source used by all models in the experiment — "
        "ensuring fair comparison between baseline and DP models."
    )

    if not status["processed_exists"]:
        st.info("Complete Stage A first.")
    elif status["subset_exists"]:
        subset_path = PATHS["subsets"] / f"{dataset}_subset.parquet"
        size_mb = subset_path.stat().st_size / 1e6
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            st.success(
                f"Subset exists: `data/subsets/{dataset}_subset.parquet` ({size_mb:.1f} MB)"
            )
        with col_s2:
            if st.button("Re-create Subset", key="recreate_subset"):
                with st.spinner("Creating subset..."):
                    try:
                        from src.data.sampler import run_sampling_pipeline
                        run_sampling_pipeline(dataset, force=True)
                        st.success("Done.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
    else:
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            st.warning("Working subset does not exist yet.")
        with col_s2:
            if st.button("▶ Create Subset", type="primary", key="run_sampling"):
                with st.spinner("Creating subset..."):
                    try:
                        from src.data.sampler import run_sampling_pipeline
                        run_sampling_pipeline(dataset, force=False)
                        st.success("Done.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed: {exc}")

st.divider()

# ── Statistics and charts ─────────────────────────────────────────────────────
st.subheader("Dataset Statistics")

stats = get_subset_stats(dataset)

if stats is None:
    st.info("No subset found. Complete Stage B to see statistics.")
else:
    # ── Key metrics row ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Rows",   f"{stats['total_rows']:,}")
    c2.metric("Features",     stats["n_features"])
    c3.metric("Train",        f"{stats['split_counts'].get('train', 0):,}")
    c4.metric("Validation",   f"{stats['split_counts'].get('val', 0):,}")
    c5.metric("Test",         f"{stats['split_counts'].get('test', 0):,}")

    st.markdown("---")

    # ── Row 1: class distributions ────────────────────────────────────────────
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**Attack Category Distribution (multiclass)**")
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
            color="count",
            color_continuous_scale="Blues",
            labels={"count": "Samples"},
            text="count",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(
            showlegend=False,
            height=max(300, len(df_dist) * 28),
            margin=dict(l=0, r=80, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.markdown("**Binary Label Balance**")
        bin_dist = stats["binary_distribution"]
        labels_map = {0: "Benign (0)", 1: "Attack (1)"}
        labels = [labels_map.get(k, str(k)) for k in bin_dist.keys()]
        values = list(bin_dist.values())
        total  = sum(values)
        fig2 = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker_colors=["#2ecc71", "#e74c3c"],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        )])
        fig2.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            annotations=[dict(
                text=f"{total:,}<br>total",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False,
            )],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Split breakdown ─────────────────────────────────────────────────
    col_split1, col_split2 = st.columns(2)

    with col_split1:
        st.markdown("**Train / Val / Test Split**")
        split_counts = stats["split_counts"]
        split_labels = list(split_counts.keys())
        split_values = list(split_counts.values())
        split_colors = ["#3498db", "#f39c12", "#2ecc71"]
        fig3 = go.Figure(go.Bar(
            x=split_labels,
            y=split_values,
            text=[f"{v:,}<br>({100*v/stats['total_rows']:.1f}%)" for v in split_values],
            textposition="outside",
            marker_color=split_colors[:len(split_labels)],
        ))
        fig3.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=20),
            yaxis_title="Rows",
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_split2:
        st.markdown("**Split Summary Table**")
        split_df = pd.DataFrame([
            {
                "Split":    k,
                "Rows":     f"{v:,}",
                "Fraction": f"{100*v/stats['total_rows']:.1f}%",
                "Purpose":  {"train": "Model training", "val": "Hyperparameter tuning / early stopping", "test": "Final evaluation (held-out)"}.get(k, ""),
            }
            for k, v in split_counts.items()
        ])
        st.dataframe(split_df, use_container_width=True, hide_index=True)

    # ── Class imbalance summary card ─────────────────────────────────────────
    st.markdown("---")
    with st.container(border=True):
        st.markdown("**Class Imbalance Summary**")
        bin_dist = stats["binary_distribution"]
        total    = sum(bin_dist.values())
        n_benign = bin_dist.get(0, 0)
        n_attack = bin_dist.get(1, 0)
        if n_benign > 0 and n_attack > 0:
            ratio = max(n_benign, n_attack) / min(n_benign, n_attack)
            majority_class = "Benign" if n_benign >= n_attack else "Attack"
            st.markdown(
                f"Majority class: **{majority_class}** | "
                f"Imbalance ratio: **{ratio:.1f}:1** | "
                f"Benign: **{100*n_benign/total:.1f}%** | "
                f"Attack: **{100*n_attack/total:.1f}%**"
            )
            if ratio > 5:
                st.warning(
                    f"High imbalance ({ratio:.1f}:1). "
                    "Models use class-weighted training (WeightedRandomSampler for DP-SGD, "
                    "class_weight='balanced' for baselines)."
                )
            else:
                st.success(f"Moderate imbalance ({ratio:.1f}:1). Class weighting applied.")

    # ── Class distribution per split ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("Class Distribution per Split")
    st.caption("Shows whether the stratified sampling preserved class proportions across train/val/test.")

    @st.cache_data(show_spinner=False)
    def _load_split_class_dist(ds: str) -> pd.DataFrame:
        import pandas as _pd
        from src.utils.config import PATHS as _PATHS
        _path = _PATHS["subsets"] / f"{ds}_subset.parquet"
        if not _path.exists():
            return _pd.DataFrame()
        _df = _pd.read_parquet(_path, columns=["split", "label_binary", "label_multiclass"])
        return _df

    _raw_df = _load_split_class_dist(dataset)
    if not _raw_df.empty:
        _spl_col1, _spl_col2 = st.columns(2)

        with _spl_col1:
            st.markdown("**Binary label counts per split**")
            _bin_grp = (
                _raw_df.groupby(["split", "label_binary"])
                .size().reset_index(name="count")
            )
            _bin_grp["label_binary"] = _bin_grp["label_binary"].map({0: "Benign", 1: "Attack"})
            fig_spl = px.bar(
                _bin_grp, x="split", y="count", color="label_binary",
                barmode="stack",
                color_discrete_map={"Benign": "#2ecc71", "Attack": "#e74c3c"},
                text="count",
                category_orders={"split": ["train", "val", "test"]},
            )
            fig_spl.update_traces(texttemplate="%{text:,}", textposition="inside")
            fig_spl.update_layout(
                height=320, margin=dict(l=0, r=0, t=20, b=40),
                xaxis_title="Split", yaxis_title="Rows",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_spl, use_container_width=True)

        with _spl_col2:
            st.markdown("**Multiclass category counts per split**")
            _mc_grp = (
                _raw_df.groupby(["split", "label_multiclass"])
                .size().reset_index(name="count")
            )
            fig_mc = px.bar(
                _mc_grp, x="split", y="count", color="label_multiclass",
                barmode="stack",
                text="count",
                category_orders={"split": ["train", "val", "test"]},
            )
            fig_mc.update_traces(texttemplate="%{text:,}", textposition="inside",
                                  textfont_size=9)
            fig_mc.update_layout(
                height=320, margin=dict(l=0, r=0, t=20, b=40),
                xaxis_title="Split", yaxis_title="Rows",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            font=dict(size=9)),
            )
            st.plotly_chart(fig_mc, use_container_width=True)

    # ── Feature statistics ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Feature Statistics")
    st.caption("Descriptive statistics for all numeric features in the working subset (train split only).")

    @st.cache_data(show_spinner="Loading feature statistics…")
    def _load_feature_stats(ds: str) -> pd.DataFrame:
        import pandas as _pd
        from src.utils.config import PATHS as _PATHS
        _path = _PATHS["subsets"] / f"{ds}_subset.parquet"
        if not _path.exists():
            return _pd.DataFrame()
        _NON_FEAT = {"label_raw", "label_binary", "label_multiclass",
                     "dataset_source", "split"}
        _df = _pd.read_parquet(_path)
        _train = _df[_df["split"] == "train"]
        _feat_cols = [c for c in _train.columns if c not in _NON_FEAT]
        _desc = _train[_feat_cols].describe().T
        _desc = _desc.rename(columns={"mean": "Mean", "std": "Std Dev",
                                       "min": "Min", "max": "Max",
                                       "50%": "Median"})
        _desc["CV (%)"] = (_desc["Std Dev"] / _desc["Mean"].abs().clip(lower=1e-9) * 100).round(1)
        return _desc[["Mean", "Std Dev", "Median", "Min", "Max", "CV (%)"]].round(4)

    _feat_stats = _load_feature_stats(dataset)
    if not _feat_stats.empty:
        _top_n_feat = st.slider("Show top N features by coefficient of variation",
                                min_value=10, max_value=min(50, len(_feat_stats)),
                                value=20, key="feat_stats_n")
        _top_feat = _feat_stats.nlargest(_top_n_feat, "CV (%)")
        st.dataframe(
            _top_feat.style.background_gradient(subset=["CV (%)"], cmap="YlOrRd")
                           .format(precision=4, na_rep="—"),
            use_container_width=True,
        )
        st.caption(
            "Sorted by coefficient of variation (CV = Std/Mean × 100). "
            "High-CV features vary most relative to their mean — "
            "these tend to be most discriminative for intrusion detection."
        )

        # Variance bar chart for top features
        _vc1, _vc2 = st.columns(2)
        with _vc1:
            st.markdown("**Top features by Std Dev**")
            _top_std = _feat_stats.nlargest(15, "Std Dev").reset_index()
            _top_std.columns = ["Feature"] + list(_top_std.columns[1:])
            fig_std = px.bar(
                _top_std.sort_values("Std Dev"),
                x="Std Dev", y="Feature", orientation="h",
                color="Std Dev", color_continuous_scale="Blues",
            )
            fig_std.update_layout(height=380, margin=dict(l=0, r=40, t=20, b=20),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_std, use_container_width=True)

        with _vc2:
            st.markdown("**Mean value distribution (log scale)**")
            _mean_vals = _feat_stats["Mean"].abs().clip(lower=1e-9)
            fig_hist = px.histogram(
                x=np.log10(_mean_vals.values),
                nbins=30,
                labels={"x": "log₁₀(|Mean|)"},
            )
            fig_hist.update_layout(height=380, margin=dict(l=0, r=20, t=20, b=40),
                                   xaxis_title="log₁₀(|Mean feature value|)",
                                   yaxis_title="Number of features")
            st.plotly_chart(fig_hist, use_container_width=True)
