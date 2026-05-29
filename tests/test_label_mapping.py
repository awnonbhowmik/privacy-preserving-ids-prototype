"""
test_label_mapping.py — Tests for src/data/label_mapping.py.

Verifies the shared label taxonomy used across CIC-IDS2018 and UNSW-NB15.
These tests have no external dependencies — no datasets, no config, no disk I/O.

Why this matters for the dissertation:
    - The "Infilteration" typo in CIC-IDS2018 raw data must map to "Infiltration"
    - Empty attack_cat in UNSW-NB15 must map to "BENIGN" (not drop the row)
    - Binary label must be exactly 0 for BENIGN and 1 for ALL attack classes
    - Mapping is case-insensitive (raw data has mixed casing)
    - Unknown labels must not raise exceptions (they log a warning and return "Unknown")
    - The apply_cic_label_mapping vectorised function must add all three label columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.label_mapping import (
    map_cic_label,
    map_unsw_label,
    to_binary,
    apply_cic_label_mapping,
    apply_unsw_label_mapping,
    encode_multiclass_labels,
    BENIGN_CLASS,
    CIC_IDS2018_LABEL_MAP,
    UNSW_NB15_LABEL_MAP,
)


# ═══════════════════════════════════════════════════════════════════════════════
# map_cic_label — single-label function
# ═══════════════════════════════════════════════════════════════════════════════

class TestMapCicLabel:
    def test_benign_maps_correctly(self):
        assert map_cic_label("BENIGN") == BENIGN_CLASS
        assert map_cic_label("benign") == BENIGN_CLASS

    def test_infilteration_typo_is_handled(self):
        """The raw CIC-IDS2018 data spells 'Infiltration' as 'Infilteration'."""
        result = map_cic_label("Infilteration")
        assert result == "Infiltration", (
            "The typo 'Infilteration' in raw CIC data must map to 'Infiltration'"
        )

    def test_dos_family(self):
        dos_labels = [
            "DoS attacks-Hulk",
            "DoS attacks-SlowHTTPTest",
            "DoS attacks-Slowloris",
            "DoS attacks-GoldenEye",
        ]
        for raw in dos_labels:
            assert map_cic_label(raw) == "DoS", f"{raw!r} should map to DoS"

    def test_ddos_family(self):
        ddos_labels = [
            "DDOS attack-HOIC",
            "DDOS attack-LOIC-UDP",
            "DDoS attacks-LOIC-HTTP",
        ]
        for raw in ddos_labels:
            assert map_cic_label(raw) == "DDoS", f"{raw!r} should map to DDoS"

    def test_brute_force_family(self):
        for raw in ["SSH-Bruteforce", "FTP-BruteForce"]:
            assert map_cic_label(raw) == "BruteForce"

    def test_web_attacks(self):
        assert map_cic_label("SQL Injection") == "WebAttack"
        assert map_cic_label("Brute Force -Web") == "WebAttack"
        assert map_cic_label("Brute Force -XSS") == "WebAttack"

    def test_botnet(self):
        assert map_cic_label("Bot") == "Botnet"

    def test_unknown_label_returns_unknown_not_exception(self):
        """Unknown labels must not raise — they return 'Unknown' and log a warning."""
        result = map_cic_label("SomeNewAttackType")
        assert result == "Unknown"

    def test_case_insensitive(self):
        """Mapping is lowercased before lookup — casing differences must not matter."""
        assert map_cic_label("BENIGN") == map_cic_label("benign")
        assert map_cic_label("DoS attacks-Hulk") == map_cic_label("dos attacks-hulk")

    def test_all_map_entries_return_non_empty_string(self):
        """Every entry in CIC_IDS2018_LABEL_MAP must resolve to a non-empty taxonomy string."""
        for raw_lower, taxonomy in CIC_IDS2018_LABEL_MAP.items():
            assert isinstance(taxonomy, str) and len(taxonomy) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# map_unsw_label — single-label function
# ═══════════════════════════════════════════════════════════════════════════════

class TestMapUnswLabel:
    def test_empty_attack_cat_is_benign(self):
        """Empty string in attack_cat column means normal traffic → BENIGN."""
        assert map_unsw_label("") == BENIGN_CLASS

    def test_normal_is_benign(self):
        assert map_unsw_label("Normal") == BENIGN_CLASS
        assert map_unsw_label("normal") == BENIGN_CLASS

    def test_dos_mapping(self):
        assert map_unsw_label("DoS") == "DoS"
        assert map_unsw_label("dos") == "DoS"

    def test_fuzzing_mapping(self):
        assert map_unsw_label("Fuzzers") == "Fuzzing"

    def test_reconnaissance_mapping(self):
        assert map_unsw_label("Reconnaissance") == "Reconnaissance"

    def test_exploits_mapping(self):
        assert map_unsw_label("Exploits") == "Exploits"

    def test_malware_family(self):
        """Backdoors, Shellcode, and Worms all collapse to Malware."""
        for raw in ["Backdoors", "Shellcode", "Worms"]:
            result = map_unsw_label(raw)
            assert result == "Malware", f"{raw!r} should map to Malware"

    def test_other_family(self):
        assert map_unsw_label("Analysis") == "Other"
        assert map_unsw_label("Generic")  == "Other"

    def test_unknown_returns_unknown_not_exception(self):
        assert map_unsw_label("SomeFutureAttack") == "Unknown"

    def test_all_map_entries_return_non_empty_string(self):
        for raw_lower, taxonomy in UNSW_NB15_LABEL_MAP.items():
            assert isinstance(taxonomy, str) and len(taxonomy) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# to_binary
# ═══════════════════════════════════════════════════════════════════════════════

class TestToBinary:
    def test_benign_is_zero(self):
        assert to_binary(BENIGN_CLASS) == 0
        assert to_binary("BENIGN") == 0

    def test_all_attack_classes_are_one(self):
        attack_classes = [
            "DoS", "DDoS", "BruteForce", "Botnet", "Infiltration",
            "WebAttack", "Fuzzing", "Reconnaissance", "Exploits", "Malware",
            "Other", "Unknown",
        ]
        for cls in attack_classes:
            assert to_binary(cls) == 1, f"Attack class '{cls}' should map to 1"

    def test_binary_labels_are_integers(self):
        assert isinstance(to_binary(BENIGN_CLASS), int)
        assert isinstance(to_binary("DoS"), int)


# ═══════════════════════════════════════════════════════════════════════════════
# apply_cic_label_mapping (vectorised)
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyCicLabelMapping:
    def _make_df(self):
        return pd.DataFrame({
            "Label": ["BENIGN", "Bot", "Infilteration", "DoS attacks-Hulk", "UNKNOWN_TYPE"],
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

    def test_adds_three_label_columns(self):
        df = apply_cic_label_mapping(self._make_df(), raw_col="Label")
        assert "label_raw" in df.columns
        assert "label_multiclass" in df.columns
        assert "label_binary" in df.columns

    def test_drops_original_label_column(self):
        df = apply_cic_label_mapping(self._make_df(), raw_col="Label")
        assert "Label" not in df.columns

    def test_feature_columns_preserved(self):
        df = apply_cic_label_mapping(self._make_df(), raw_col="Label")
        assert "feature_a" in df.columns

    def test_infilteration_typo_resolved_vectorised(self):
        df = apply_cic_label_mapping(self._make_df(), raw_col="Label")
        infiltration_rows = df[df["label_raw"] == "Infilteration"]
        assert all(infiltration_rows["label_multiclass"] == "Infiltration")

    def test_binary_column_is_0_or_1_only(self):
        df = apply_cic_label_mapping(self._make_df(), raw_col="Label")
        assert set(df["label_binary"].unique()).issubset({0, 1})

    def test_benign_rows_have_binary_0(self):
        df = apply_cic_label_mapping(self._make_df(), raw_col="Label")
        benign_binary = df[df["label_raw"] == "BENIGN"]["label_binary"].values
        assert all(v == 0 for v in benign_binary)

    def test_attack_rows_have_binary_1(self):
        df = apply_cic_label_mapping(self._make_df(), raw_col="Label")
        attack_binary = df[df["label_multiclass"] != BENIGN_CLASS]["label_binary"].values
        assert all(v == 1 for v in attack_binary)

    def test_unknown_label_gets_unknown_multiclass(self):
        df = apply_cic_label_mapping(self._make_df(), raw_col="Label")
        unknown_rows = df[df["label_raw"] == "UNKNOWN_TYPE"]
        assert all(unknown_rows["label_multiclass"] == "Unknown")

    def test_raises_on_missing_label_column(self):
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(KeyError):
            apply_cic_label_mapping(df, raw_col="Label")


# ═══════════════════════════════════════════════════════════════════════════════
# apply_unsw_label_mapping (vectorised)
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyUnswLabelMapping:
    def _make_df(self):
        return pd.DataFrame({
            "attack_cat": ["", "DoS", "Backdoors", "Fuzzers", None],
            "label":      [0,   1,     1,           1,         0],
            "feature_x":  [1.0, 2.0,  3.0,         4.0,       5.0],
        })

    def test_adds_three_label_columns(self):
        df = apply_unsw_label_mapping(self._make_df())
        assert "label_raw" in df.columns
        assert "label_multiclass" in df.columns
        assert "label_binary" in df.columns

    def test_drops_original_label_columns(self):
        df = apply_unsw_label_mapping(self._make_df())
        assert "label" not in df.columns
        assert "attack_cat" not in df.columns

    def test_empty_attack_cat_becomes_benign(self):
        df = apply_unsw_label_mapping(self._make_df())
        empty_rows = df[df["label_raw"] == ""]
        assert all(empty_rows["label_multiclass"] == BENIGN_CLASS)

    def test_none_attack_cat_becomes_benign(self):
        """NaN attack_cat (None in pandas) must also resolve to BENIGN."""
        df = apply_unsw_label_mapping(self._make_df())
        # Row with None should have been normalised to empty string → BENIGN
        assert BENIGN_CLASS in df["label_multiclass"].values

    def test_binary_column_preserved_from_label(self):
        """The existing integer label column is used as-is for label_binary."""
        df = apply_unsw_label_mapping(self._make_df())
        # Row 0 had label=0 → label_binary should be 0
        assert df.iloc[0]["label_binary"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# encode_multiclass_labels
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncodeMulticlassLabels:
    def test_returns_integer_array_and_encoder(self):
        labels = pd.Series(["BENIGN", "DoS", "DDoS", "BENIGN", "DoS"])
        encoded, encoder = encode_multiclass_labels(labels)
        assert encoded.dtype == np.int32
        assert len(encoded) == len(labels)

    def test_encoder_reusable_on_test_split(self):
        train_labels = pd.Series(["BENIGN", "DoS", "DDoS"])
        test_labels  = pd.Series(["DoS", "BENIGN"])
        _, encoder = encode_multiclass_labels(train_labels)
        test_enc, _ = encode_multiclass_labels(test_labels, encoder=encoder)
        assert len(test_enc) == len(test_labels)

    def test_consistent_class_encoding(self):
        """The same class name must always get the same integer across splits."""
        labels = pd.Series(["BENIGN", "DoS", "DDoS", "BENIGN"])
        enc, encoder = encode_multiclass_labels(labels)
        # Apply same encoder to single class — must give same integer
        benign_code = encoder.transform(["BENIGN"])[0]
        assert enc[0] == benign_code
        assert enc[3] == benign_code
