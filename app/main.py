"""
main.py — Streamlit application entry point and home page.

Run with:
    streamlit run app/main.py

The home page shows a pipeline status overview for both datasets so the
user can see at a glance which stages have been completed and what to run next.
"""

from app._path_setup import setup_path
setup_path()

import streamlit as st
from src.models.model_utils import get_pipeline_status, SUPPORTED_DATASETS

st.set_page_config(
    page_title="Privacy-Preserving IDS",
    page_icon="🛡️",
    layout="wide",
)

# ── Sidebar — global controls ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    dataset = st.selectbox(
        "Dataset",
        SUPPORTED_DATASETS,
        format_func=lambda x: {"cic_ids2018": "CIC-IDS2018", "unsw_nb15": "UNSW-NB15"}[x],
        key="global_dataset",
    )
    label_type = st.selectbox(
        "Label type",
        ["binary", "multiclass"],
        key="global_label",
    )
    st.caption(
        "These selections are used as defaults across all pages. "
        "You can change them per-page too."
    )

# ── Main content ──────────────────────────────────────────────────────────────
st.title("Privacy-Preserving Intrusion Detection System")
st.markdown(
    "A dissertation prototype demonstrating **Differential Privacy** for "
    "network intrusion detection. Navigate using the sidebar pages."
)

st.divider()

# ── Pipeline status overview ──────────────────────────────────────────────────
st.subheader("Pipeline Status")
st.caption(
    "Shows which stages have been completed for each dataset. "
    "Complete stages in order from top to bottom."
)

STATUS_ICON = {True: "✅", False: "❌"}

for ds in SUPPORTED_DATASETS:
    ds_display = {"cic_ids2018": "CIC-IDS2018", "unsw_nb15": "UNSW-NB15"}[ds]
    status = get_pipeline_status(ds, label=label_type)

    with st.expander(f"**{ds_display}**", expanded=(ds == dataset)):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Data Pipeline**")
            raw = STATUS_ICON[status["raw_files_exist"]]
            proc = STATUS_ICON[status["processed_exists"]]
            sub = STATUS_ICON[status["subset_exists"]]
            n = status["raw_file_count"]
            st.markdown(
                f"{raw} Raw CSVs found ({n} file{'s' if n != 1 else ''})\n\n"
                f"{proc} Processed Parquet\n\n"
                f"{sub} Working subset"
            )

        with col2:
            st.markdown("**Baseline Models**")
            for model_name, trained in status["baselines_trained"].items():
                icon = STATUS_ICON[trained]
                display = model_name.replace("_", " ").title()
                st.markdown(f"{icon} {display}")

        with col3:
            st.markdown("**DP Experiments**")
            dp_done = sum(status["dp_trained"].values())
            dp_total = len(status["dp_trained"])
            st.markdown(f"ε sweep: {dp_done}/{dp_total} trained")
            res = status["results_exist"]
            st.markdown(
                f"{STATUS_ICON[res['baselines']]} Baseline results saved\n\n"
                f"{STATUS_ICON[res['dp_sweep']]} DP sweep results saved\n\n"
                f"{STATUS_ICON[res['combined']]} Combined results ready"
            )

st.divider()

# ── Quick-start guide ─────────────────────────────────────────────────────────
st.subheader("Quick Start")
st.markdown(
    """
    Follow these steps in order:

    1. **Place raw CSVs** in `data/raw/cic_ids2018/` and `data/raw/unsw_nb15/`
    2. **Data Overview page** — run preprocessing and sampling pipeline
    3. **Train Models page** — train baseline classifiers and the DP sweep
    4. **Compare Results page** — view the privacy–utility tradeoff analysis

    Or run everything from the terminal:
    ```bash
    source .venv/bin/activate

    python scripts/train_baseline.py --dataset cic_ids2018
    python scripts/train_private.py  --dataset cic_ids2018
    python scripts/evaluate.py       --dataset cic_ids2018 --export-csv
    ```
    """
)
