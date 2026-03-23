"""
config.py — Project-wide configuration loader.

Reads config.yaml once at import time and exposes typed constants that every
other module can import directly. No module should open config.yaml itself —
always import from here so there is a single source of truth.

Usage:
    from src.utils.config import PATHS, DATASET_CFG, SAMPLING_CFG

    raw_dir = PATHS["raw_data"]          # pathlib.Path
    seed    = SAMPLING_CFG["random_seed"]  # int
"""

import yaml
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Locate project root and config file ────────────────────────────────────────
# This file lives at src/utils/config.py → two parents up = project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE  = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    """
    Parse config.yaml and return its contents as a plain Python dict.

    Raises:
        FileNotFoundError: If config.yaml is missing from the project root.
        yaml.YAMLError:    If the file is malformed.
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {CONFIG_FILE}. "
            "Ensure you are running from the project root."
        )
    with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    log.debug("Loaded config from %s", CONFIG_FILE)
    return cfg


def ensure_directories(paths: dict[str, Path]) -> None:
    """
    Create all output directories if they do not already exist.

    Called once at startup so that downstream code never has to check whether
    a directory exists before writing a file.

    Args:
        paths: Dict of {name: Path} as produced by building PATHS below.
    """
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        log.debug("Ensured directory exists: %s (%s)", name, path)


def get_dataset_config(dataset_name: str) -> dict:
    """
    Return the config block for a single dataset, with validation.

    Args:
        dataset_name: Key matching a top-level entry in config.yaml 'datasets'
                      section (e.g. "cic_ids2018" or "unsw_nb15").

    Returns:
        Dict with keys: name, label_column, encoding, separator.

    Raises:
        KeyError: If dataset_name is not present in config.yaml.
    """
    if dataset_name not in DATASET_CFG:
        valid = list(DATASET_CFG.keys())
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options are: {valid}"
        )
    return DATASET_CFG[dataset_name]


# ── Load config at import time ─────────────────────────────────────────────────
cfg = load_config()

# ── Resolved absolute paths ────────────────────────────────────────────────────
# All paths in config.yaml are relative to the project root. We resolve them
# to absolute Path objects here so downstream code never has to think about CWD.
PATHS: dict[str, Path] = {
    key: PROJECT_ROOT / value
    for key, value in cfg["paths"].items()
}

# ── Typed config sections ──────────────────────────────────────────────────────
DATASET_CFG:    dict = cfg["datasets"]
SAMPLING_CFG:   dict = cfg["sampling"]
PRIVACY_CFG:    dict = cfg["privacy"]
MODEL_CFG:      dict = cfg["models"]
DP_SGD_CFG:     dict = cfg.get("dp_sgd", {})

# ── Create output directories on import ───────────────────────────────────────
# This means any module that imports config.py will automatically ensure that
# data/processed/, data/subsets/, results/, etc. exist.
ensure_directories(PATHS)
