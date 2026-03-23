"""
main.py — Streamlit navigation entry point.

Run with:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from app._path_setup import setup_path
setup_path()

import streamlit as st
from src.models.model_utils import SUPPORTED_DATASETS

st.set_page_config(
    page_title="Privacy-Preserving IDS",
    page_icon="🛡️",
    layout="wide",
)

# ── Sidebar — global controls (shown on every page) ───────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Global Settings")
    st.selectbox(
        "Dataset",
        SUPPORTED_DATASETS,
        format_func=lambda x: {"cic_ids2018": "CIC-IDS2018", "unsw_nb15": "UNSW-NB15"}[x],
        key="global_dataset",
    )
    st.selectbox(
        "Label type",
        ["binary", "multiclass"],
        key="global_label",
    )
    st.caption("Default dataset/label used across all pages.")

# ── Navigation ────────────────────────────────────────────────────────────────
pg = st.navigation([
    st.Page("home.py",                        title="Home",           icon="🛡️",  default=True),
    st.Page("pages/1_Data_Overview.py",        title="Data Overview",  icon="📊"),
    st.Page("pages/2_Train_Models.py",         title="Train Models",   icon="🧠"),
    st.Page("pages/3_Compare_Results.py",      title="Compare Results",icon="📈"),
])

pg.run()
