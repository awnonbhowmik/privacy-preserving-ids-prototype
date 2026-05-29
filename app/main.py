"""
main.py — Streamlit navigation entry point.

Run with:
    python -m streamlit run app/main.py
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Global dataset / label settings
    st.markdown("### ⚙️ Settings")
    st.selectbox(
        "Dataset",
        SUPPORTED_DATASETS,
        format_func=lambda x: {
            "cic_ids2018": "CIC-IDS2018  (primary)",
            "unsw_nb15":   "UNSW-NB15    (validation)",
        }[x],
        key="global_dataset",
        help="CIC-IDS2018 is the primary dataset. UNSW-NB15 validates that findings generalise.",
    )
    st.selectbox(
        "Label type",
        ["binary", "multiclass"],
        format_func=lambda x: {
            "binary":     "Binary  (benign vs attack)",
            "multiclass": "Multiclass  (attack categories)",
        }[x],
        key="global_label",
        help="Binary: 0=benign, 1=attack. Multiclass: individual attack category labels.",
    )

    st.divider()

    # Quick page guide in sidebar
    st.markdown(
        "<small>"
        "**Page guide**<br>"
        "📊 Data — inspect datasets<br>"
        "🧠 Train — run model sweeps<br>"
        "📈 Compare — tradeoff curves<br>"
        "🔍 Attack — MIA evaluation"
        "</small>",
        unsafe_allow_html=True,
    )

# ── Navigation ────────────────────────────────────────────────────────────────
pg = st.navigation([
    st.Page("home.py",                      title="Home",              icon="🛡️", default=True),
    st.Page("pages/1_Data_Overview.py",     title="Data Overview",     icon="📊"),
    st.Page("pages/2_Train_Models.py",      title="Train Models",      icon="🧠"),
    st.Page("pages/3_Compare_Results.py",   title="Compare Results",   icon="📈"),
    st.Page("pages/4_Attack_Evaluation.py", title="Attack Evaluation",  icon="🔍"),
])

pg.run()
