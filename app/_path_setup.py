"""
_path_setup.py — Ensure the project root is on sys.path.

Import this at the top of every Streamlit page file before any src.* imports.
This allows pages to resolve `from src.xxx import yyy` correctly regardless of
how Streamlit resolves the working directory.

Usage (first two lines of every page):
    from app._path_setup import setup_path
    setup_path()
"""

import sys
from pathlib import Path


def setup_path() -> Path:
    """
    Insert the project root directory into sys.path[0] if not already present.

    Returns the resolved project root Path for reference.
    """
    # app/_path_setup.py → parent is app/ → parent is project root
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root
