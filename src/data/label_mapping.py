"""
label_mapping.py — Authoritative label taxonomy for all datasets.

This module owns the label normalisation logic. It is deliberately kept
separate from preprocessing logic so the dissertation can point to one file
as the single source of truth for the threat taxonomy used across experiments.

Design:
- CIC-IDS2018 and UNSW-NB15 use different label schemas. We map both to a
  shared multiclass taxonomy and a common binary label (0=benign, 1=attack).
- All raw label strings are lowercased before lookup so mapping is case-insensitive.
- Unknown labels that do not match any map entry are assigned "Unknown" and
  logged as warnings so data quality issues surface during preprocessing.

Taxonomy (multiclass):
    BENIGN, DoS, DDoS, BruteForce, Botnet, Infiltration, WebAttack,
    Fuzzing, Reconnaissance, Exploits, Malware, Other, Unknown
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Binary label constant ──────────────────────────────────────────────────────
BENIGN_CLASS   = "BENIGN"
BENIGN_BINARY  = 0
ATTACK_BINARY  = 1

# ── CIC-IDS2018 label map ──────────────────────────────────────────────────────
# Keys are raw label strings, lowercased. Values are normalised taxonomy strings.
# Source: CIC-IDS2018 dataset documentation and prior IDS literature.
CIC_IDS2018_LABEL_MAP: dict[str, str] = {
    "benign":                        BENIGN_CLASS,
    # DoS family
    "dos attacks-hulk":              "DoS",
    "dos attacks-slowhttptest":      "DoS",
    "dos attacks-slowloris":         "DoS",
    "dos attacks-goldeneye":         "DoS",
    # DDoS family
    "ddos attack-hoic":              "DDoS",
    "ddos attack-loic-udp":          "DDoS",
    "ddos attacks-loic-http":        "DDoS",
    # Brute force family
    "ssh-bruteforce":                "BruteForce",
    "ftp-bruteforce":                "BruteForce",
    # Other attack types
    "bot":                           "Botnet",
    "infilteration":                 "Infiltration",  # note: typo in original dataset
    # Web attacks
    "sql injection":                 "WebAttack",
    "brute force -web":              "WebAttack",
    "brute force -xss":              "WebAttack",
}

# ── UNSW-NB15 label map ────────────────────────────────────────────────────────
# UNSW-NB15 uses the 'attack_cat' column for category labels.
# An empty or NaN attack_cat with label==0 indicates normal (benign) traffic.
# Source: Moustafa & Slay (2015), UNSW-NB15 dataset documentation.
UNSW_NB15_LABEL_MAP: dict[str, str] = {
    "":               BENIGN_CLASS,   # empty attack_cat = normal traffic
    "normal":         BENIGN_CLASS,
    # Mapped to shared taxonomy where possible
    "dos":            "DoS",
    "fuzzers":        "Fuzzing",
    "reconnaissance": "Reconnaissance",
    "exploits":       "Exploits",
    # Malware family
    "backdoors":      "Malware",
    "shellcode":      "Malware",
    "worms":          "Malware",
    # Grouped under Other (no clean taxonomy match)
    "analysis":       "Other",
    "generic":        "Other",
}


def map_cic_label(raw_label: str) -> str:
    """
    Map a single raw CIC-IDS2018 label string to the normalised taxonomy.

    Args:
        raw_label: Raw string from the 'Label' column (any casing).

    Returns:
        Normalised taxonomy string. Returns "Unknown" if the label is not
        in the map and logs a warning so data quality issues are surfaced.
    """
    key = raw_label.strip().lower()
    result = CIC_IDS2018_LABEL_MAP.get(key)
    if result is None:
        log.warning("Unmapped CIC-IDS2018 label: '%s' → assigned 'Unknown'", raw_label)
        return "Unknown"
    return result


def map_unsw_label(attack_cat: str) -> str:
    """
    Map a single raw UNSW-NB15 attack_cat string to the normalised taxonomy.

    Args:
        attack_cat: Raw string from the 'attack_cat' column (may be empty).

    Returns:
        Normalised taxonomy string. Returns "Unknown" for unrecognised values.
    """
    key = str(attack_cat).strip().lower()
    result = UNSW_NB15_LABEL_MAP.get(key)
    if result is None:
        log.warning("Unmapped UNSW-NB15 attack_cat: '%s' → assigned 'Unknown'", attack_cat)
        return "Unknown"
    return result


def to_binary(multiclass_label: str) -> int:
    """
    Convert a normalised multiclass label to a binary integer.

    Args:
        multiclass_label: Normalised label string from the taxonomy.

    Returns:
        0 if label is BENIGN, 1 for all attack classes.
    """
    return BENIGN_BINARY if multiclass_label == BENIGN_CLASS else ATTACK_BINARY


def apply_cic_label_mapping(df: pd.DataFrame, raw_col: str) -> pd.DataFrame:
    """
    Vectorised label normalisation for a CIC-IDS2018 DataFrame chunk.

    Adds three columns:
        label_raw        — original label string (preserved for audit)
        label_multiclass — normalised attack category string
        label_binary     — 0 (benign) or 1 (attack) as int8

    Drops the original label column after mapping.

    Args:
        df:      DataFrame chunk containing the raw label column.
        raw_col: Name of the raw label column (after column name normalisation
                 this will be 'label').

    Returns:
        DataFrame with label_raw, label_multiclass, label_binary added and
        the original label column dropped.
    """
    if raw_col not in df.columns:
        raise KeyError(
            f"Expected label column '{raw_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Preserve raw label
    df = df.copy()
    df["label_raw"] = df[raw_col].astype(str).str.strip()

    # Vectorised map — faster than .apply(map_cic_label) for large chunks
    df["label_multiclass"] = (
        df["label_raw"]
        .str.lower()
        .map(CIC_IDS2018_LABEL_MAP)
        .fillna("Unknown")          # unmapped labels default to Unknown
        .astype("category")
    )

    df["label_binary"] = (
        df["label_multiclass"]
        .apply(to_binary)
        .astype("int8")
    )

    # Drop original label column
    df = df.drop(columns=[raw_col])

    return df


def apply_unsw_label_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised label normalisation for a UNSW-NB15 DataFrame chunk.

    UNSW-NB15 has two label columns:
        'label'      — binary integer (0=normal, 1=attack)
        'attack_cat' — string attack category (may be empty/NaN)

    After normalisation these are replaced by the standard three-column schema.

    Args:
        df: DataFrame chunk containing 'label' and 'attack_cat' columns
            (these names are post column-name normalisation).

    Returns:
        DataFrame with label_raw, label_multiclass, label_binary columns and
        the original 'label' and 'attack_cat' columns dropped.
    """
    df = df.copy()

    # attack_cat may be NaN for normal rows — fill with empty string before mapping
    attack_cat_col = df["attack_cat"].fillna("").astype(str).str.strip()

    df["label_raw"] = attack_cat_col

    df["label_multiclass"] = (
        attack_cat_col
        .str.lower()
        .map(UNSW_NB15_LABEL_MAP)
        .fillna("Unknown")
        .astype("category")
    )

    # Use the existing binary column — already 0/1 integers
    df["label_binary"] = df["label"].astype("int8")

    # Drop original label columns
    df = df.drop(columns=["label", "attack_cat"], errors="ignore")

    return df


def encode_multiclass_labels(
    labels: pd.Series,
    encoder: LabelEncoder | None = None,
) -> tuple[np.ndarray, LabelEncoder]:
    """
    Integer-encode a multiclass label Series using sklearn LabelEncoder.

    Used by sampler.get_split_arrays() to produce numpy arrays for sklearn
    model training. The encoder is returned so it can be saved to disk and
    applied consistently to val/test splits.

    Args:
        labels:  pd.Series of normalised label_multiclass strings.
        encoder: An already-fitted LabelEncoder to apply (for val/test splits).
                 Pass None to fit a new encoder (for training split).

    Returns:
        Tuple of (encoded integer array, fitted LabelEncoder).
    """
    if encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(labels.astype(str))
        log.info(
            "LabelEncoder fitted. Classes: %s",
            list(encoder.classes_)
        )
    else:
        encoded = encoder.transform(labels.astype(str))

    return encoded.astype(np.int32), encoder
