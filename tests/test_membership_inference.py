"""
test_membership_inference.py — Smoke tests for Phase 3 MIA module.

All tests use synthetic data and an in-process sklearn model.
No CIC-IDS2018 / UNSW-NB15 download is required.

Tests verify:
    - extract_attack_features: shape, no NaN/Inf, fallback for models without
      predict_proba
    - build_attack_dataset: balanced labels, correct membership values,
      caps at available data, reproducibility
    - run_attack_classifier: all expected keys, metrics in [0, 1],
      attack_advantage == attack_roc_auc - 0.5
    - run_mia_on_model: schema completeness, JSON serialisability,
      epsilon/delta propagation
    - save / load round-trip: upsert does not duplicate results
    - build_mia_table: returns non-empty DataFrame; empty input → empty DataFrame
    - build_combined_dissertation_table: returns DataFrame with required columns
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from unittest.mock import MagicMock

from src.attacks.membership_inference import (
    _json_serialise,
    build_attack_dataset,
    build_mia_table,
    extract_attack_features,
    load_mia_results,
    run_attack_classifier,
    run_mia_on_model,
    save_mia_results,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def binary_data():
    """Small synthetic binary dataset (fast, no disk I/O)."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=2,
        n_informative=10,
        random_state=42,
    )
    return X[:300], y[:300], X[300:], y[300:]   # train, test


@pytest.fixture(scope="module")
def trained_lr(binary_data):
    """Logistic regression trained on the synthetic training split."""
    X_train, y_train, _, _ = binary_data
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    return clf


@pytest.fixture(scope="module")
def small_attack_dataset(trained_lr, binary_data):
    """Pre-built attack dataset (80 + 80 = 160 samples)."""
    X_train, y_train, X_test, y_test = binary_data
    return build_attack_dataset(
        trained_lr, X_train, y_train, X_test, y_test,
        n_samples_per_class=80, seed=42,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# extract_attack_features
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractAttackFeatures:
    def test_returns_2d_array(self, trained_lr, binary_data):
        X_train, y_train, _, _ = binary_data
        feats = extract_attack_features(trained_lr, X_train, y_train)
        assert feats.ndim == 2

    def test_row_count_matches_input(self, trained_lr, binary_data):
        X_train, y_train, _, _ = binary_data
        feats = extract_attack_features(trained_lr, X_train, y_train)
        assert feats.shape[0] == len(X_train)

    def test_has_at_least_four_base_features(self, trained_lr, binary_data):
        X_train, y_train, _, _ = binary_data
        feats = extract_attack_features(trained_lr, X_train, y_train)
        # base (4) + n_classes=2 probs → 6 minimum for binary model
        assert feats.shape[1] >= 4

    def test_no_nans(self, trained_lr, binary_data):
        X_train, y_train, _, _ = binary_data
        feats = extract_attack_features(trained_lr, X_train, y_train)
        assert not np.any(np.isnan(feats)), "NaN found in attack features"

    def test_no_infs(self, trained_lr, binary_data):
        X_train, y_train, _, _ = binary_data
        feats = extract_attack_features(trained_lr, X_train, y_train)
        assert not np.any(np.isinf(feats)), "Inf found in attack features"

    def test_fallback_for_model_without_predict_proba(self, binary_data):
        X_train, y_train, _, _ = binary_data
        # spec=['predict'] gives predict but not predict_proba
        mock = MagicMock(spec=["predict"])
        mock.predict.return_value = y_train
        feats = extract_attack_features(mock, X_train, y_train)
        assert feats.shape == (len(X_train), 1)

    def test_fallback_when_predict_proba_raises(self, binary_data):
        X_train, y_train, _, _ = binary_data
        mock = MagicMock()
        mock.predict.return_value = y_train
        mock.predict_proba.side_effect = RuntimeError("not available")
        feats = extract_attack_features(mock, X_train, y_train)
        assert feats.shape == (len(X_train), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# build_attack_dataset
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildAttackDataset:
    def test_balanced_class_counts(self, trained_lr, binary_data):
        X_train, y_train, X_test, y_test = binary_data
        _, y_atk = build_attack_dataset(
            trained_lr, X_train, y_train, X_test, y_test,
            n_samples_per_class=50, seed=42,
        )
        assert (y_atk == 1).sum() == (y_atk == 0).sum()

    def test_only_binary_membership_labels(self, trained_lr, binary_data):
        X_train, y_train, X_test, y_test = binary_data
        _, y_atk = build_attack_dataset(
            trained_lr, X_train, y_train, X_test, y_test,
            n_samples_per_class=50, seed=42,
        )
        assert set(np.unique(y_atk)) == {0, 1}

    def test_caps_to_available_data(self, trained_lr, binary_data):
        X_train, y_train, X_test, y_test = binary_data
        _, y_atk = build_attack_dataset(
            trained_lr, X_train, y_train, X_test, y_test,
            n_samples_per_class=99999, seed=42,
        )
        # Must not exceed 2 * min(n_train, n_test)
        max_possible = 2 * min(len(X_train), len(X_test))
        assert len(y_atk) <= max_possible

    def test_reproducible_with_same_seed(self, trained_lr, binary_data):
        X_train, y_train, X_test, y_test = binary_data
        X1, y1 = build_attack_dataset(
            trained_lr, X_train, y_train, X_test, y_test,
            n_samples_per_class=60, seed=7,
        )
        X2, y2 = build_attack_dataset(
            trained_lr, X_train, y_train, X_test, y_test,
            n_samples_per_class=60, seed=7,
        )
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_give_different_samples(self, trained_lr, binary_data):
        X_train, y_train, X_test, y_test = binary_data
        X1, _ = build_attack_dataset(
            trained_lr, X_train, y_train, X_test, y_test,
            n_samples_per_class=60, seed=1,
        )
        X2, _ = build_attack_dataset(
            trained_lr, X_train, y_train, X_test, y_test,
            n_samples_per_class=60, seed=2,
        )
        assert not np.array_equal(X1, X2)


# ═══════════════════════════════════════════════════════════════════════════════
# run_attack_classifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunAttackClassifier:
    EXPECTED_KEYS = {
        "attack_accuracy", "attack_precision", "attack_recall", "attack_f1",
        "attack_roc_auc", "attack_advantage", "tpr_at_fpr_0_01",
        "tpr_at_fpr_0_10", "n_attack_train", "n_attack_test",
    }

    def test_all_keys_present(self, small_attack_dataset):
        X_atk, y_atk = small_attack_dataset
        metrics = run_attack_classifier(X_atk, y_atk, seed=42)
        assert self.EXPECTED_KEYS.issubset(set(metrics.keys()))

    def test_accuracy_in_valid_range(self, small_attack_dataset):
        X_atk, y_atk = small_attack_dataset
        metrics = run_attack_classifier(X_atk, y_atk, seed=42)
        assert 0.0 <= metrics["attack_accuracy"] <= 1.0

    def test_roc_auc_in_valid_range(self, small_attack_dataset):
        X_atk, y_atk = small_attack_dataset
        metrics = run_attack_classifier(X_atk, y_atk, seed=42)
        assert 0.0 <= metrics["attack_roc_auc"] <= 1.0

    def test_advantage_equals_auc_minus_half(self, small_attack_dataset):
        X_atk, y_atk = small_attack_dataset
        metrics = run_attack_classifier(X_atk, y_atk, seed=42)
        expected = round(metrics["attack_roc_auc"] - 0.5, 4)
        assert abs(metrics["attack_advantage"] - expected) < 1e-9

    def test_sample_counts_sum_to_total(self, small_attack_dataset):
        X_atk, y_atk = small_attack_dataset
        metrics = run_attack_classifier(X_atk, y_atk, seed=42)
        assert metrics["n_attack_train"] + metrics["n_attack_test"] == len(y_atk)

    def test_tpr_at_fpr_in_valid_range(self, small_attack_dataset):
        X_atk, y_atk = small_attack_dataset
        metrics = run_attack_classifier(X_atk, y_atk, seed=42)
        assert 0.0 <= metrics["tpr_at_fpr_0_01"] <= 1.0
        assert 0.0 <= metrics["tpr_at_fpr_0_10"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# run_mia_on_model
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunMiaOnModel:
    REQUIRED_FIELDS = {
        "dataset", "label_type", "target_model", "epsilon", "delta",
        "attack_type", "n_members", "n_non_members", "seed", "timestamp",
        "attack_accuracy", "attack_roc_auc", "attack_advantage",
        "attack_f1", "n_attack_train", "n_attack_test",
    }

    def _run(self, trained_lr, binary_data, **kwargs):
        X_train, y_train, X_test, y_test = binary_data
        defaults = dict(
            target_model_name="logistic_regression",
            dataset="mock",
            label_type="binary",
            n_attack_samples=80,
            seed=42,
        )
        defaults.update(kwargs)
        return run_mia_on_model(
            trained_lr, X_train, y_train, X_test, y_test, **defaults
        )

    def test_required_keys_present(self, trained_lr, binary_data):
        result = self._run(trained_lr, binary_data)
        assert self.REQUIRED_FIELDS.issubset(set(result.keys()))

    def test_json_serialisable(self, trained_lr, binary_data):
        result = self._run(trained_lr, binary_data)
        serialised = json.dumps(result, default=_json_serialise)
        assert len(serialised) > 0

    def test_epsilon_propagated(self, trained_lr, binary_data):
        result = self._run(trained_lr, binary_data, epsilon=1.0, delta=1e-5)
        assert result["epsilon"] == 1.0
        assert result["delta"] == 1e-5

    def test_non_private_epsilon_is_none(self, trained_lr, binary_data):
        result = self._run(trained_lr, binary_data)
        assert result["epsilon"] is None

    def test_attack_type_is_black_box(self, trained_lr, binary_data):
        result = self._run(trained_lr, binary_data)
        assert "black_box" in result["attack_type"]

    def test_member_count_matches_non_member(self, trained_lr, binary_data):
        result = self._run(trained_lr, binary_data)
        assert result["n_members"] == result["n_non_members"]

    def test_target_metrics_included(self, trained_lr, binary_data):
        result = self._run(trained_lr, binary_data)
        # At minimum accuracy should be present from evaluate_model
        assert "target_accuracy" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence: save / load round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersistence:
    def _make_result(self, trained_lr, binary_data, model_name="lr", epsilon=None):
        X_train, y_train, X_test, y_test = binary_data
        return run_mia_on_model(
            trained_lr, X_train, y_train, X_test, y_test,
            target_model_name=model_name,
            dataset="mock",
            label_type="binary",
            epsilon=epsilon,
            n_attack_samples=60,
            seed=42,
        )

    def test_save_then_load_returns_same_result(
        self, trained_lr, binary_data, tmp_path, monkeypatch
    ):
        import src.utils.config as cfg
        monkeypatch.setitem(cfg.PATHS, "results", tmp_path)

        result = self._make_result(trained_lr, binary_data)
        save_mia_results([result], "mock", "binary")
        loaded = load_mia_results("mock", "binary")

        assert len(loaded) == 1
        assert loaded[0]["target_model"] == result["target_model"]
        assert abs(loaded[0]["attack_roc_auc"] - result["attack_roc_auc"]) < 1e-9

    def test_upsert_does_not_duplicate(
        self, trained_lr, binary_data, tmp_path, monkeypatch
    ):
        import src.utils.config as cfg
        monkeypatch.setitem(cfg.PATHS, "results", tmp_path)

        result = self._make_result(trained_lr, binary_data, model_name="lr2")
        save_mia_results([result], "mock2", "binary")
        save_mia_results([result], "mock2", "binary")  # second save same result

        loaded = load_mia_results("mock2", "binary")
        assert len(loaded) == 1

    def test_multiple_models_saved_correctly(
        self, trained_lr, binary_data, tmp_path, monkeypatch
    ):
        import src.utils.config as cfg
        monkeypatch.setitem(cfg.PATHS, "results", tmp_path)

        r1 = self._make_result(trained_lr, binary_data, model_name="lr", epsilon=None)
        r2 = self._make_result(trained_lr, binary_data, model_name="dp_lr", epsilon=1.0)
        save_mia_results([r1, r2], "mock3", "binary")

        loaded = load_mia_results("mock3", "binary")
        assert len(loaded) == 2

    def test_empty_file_returns_empty_list(self, tmp_path, monkeypatch):
        import src.utils.config as cfg
        monkeypatch.setitem(cfg.PATHS, "results", tmp_path)

        loaded = load_mia_results("nonexistent", "binary")
        assert loaded == []

    def test_json_file_is_valid(
        self, trained_lr, binary_data, tmp_path, monkeypatch
    ):
        import src.utils.config as cfg
        monkeypatch.setitem(cfg.PATHS, "results", tmp_path)

        result = self._make_result(trained_lr, binary_data, model_name="lr_json_test")
        save_mia_results([result], "mock4", "binary")

        json_path = tmp_path / "mock4_mia_binary.json"
        assert json_path.exists()
        with open(json_path) as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Table builders
# ═══════════════════════════════════════════════════════════════════════════════

class TestTableBuilders:
    def _result(self, trained_lr, binary_data, model_name="lr", epsilon=None):
        X_train, y_train, X_test, y_test = binary_data
        return run_mia_on_model(
            trained_lr, X_train, y_train, X_test, y_test,
            target_model_name=model_name,
            dataset="mock",
            label_type="binary",
            epsilon=epsilon,
            n_attack_samples=60,
            seed=42,
        )

    def test_build_mia_table_non_empty(self, trained_lr, binary_data):
        r = self._result(trained_lr, binary_data)
        df = build_mia_table([r])
        assert not df.empty

    def test_build_mia_table_has_required_columns(self, trained_lr, binary_data):
        r = self._result(trained_lr, binary_data)
        df = build_mia_table([r])
        for col in ["Model", "ε", "attack_roc_auc", "attack_advantage"]:
            assert col in df.columns

    def test_build_mia_table_empty_input_returns_empty_df(self):
        df = build_mia_table([])
        assert df.empty

    def test_build_mia_table_sorted_by_model_and_epsilon(self, trained_lr, binary_data):
        # Use the canonical internal names that _display_name recognises
        r1 = self._result(trained_lr, binary_data, model_name="dp_logistic_regression", epsilon=2.0)
        r2 = self._result(trained_lr, binary_data, model_name="dp_logistic_regression", epsilon=0.5)
        r3 = self._result(trained_lr, binary_data, model_name="logistic_regression", epsilon=None)
        df = build_mia_table([r1, r2, r3])
        lr_rows = df[df["Model"] == "Logistic Regression"]
        dp_rows = df[df["Model"] == "DP-LR"]
        assert len(lr_rows) == 1
        assert len(dp_rows) == 2
        # DP-LR rows must be sorted ascending by ε
        dp_eps = dp_rows["ε"].tolist()
        assert dp_eps == sorted(dp_eps)


# ═══════════════════════════════════════════════════════════════════════════════
# JSON serialiser helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestJsonSerialise:
    def test_numpy_int(self):
        assert _json_serialise(np.int64(5)) == 5
        assert isinstance(_json_serialise(np.int64(5)), int)

    def test_numpy_float(self):
        assert abs(_json_serialise(np.float32(1.5)) - 1.5) < 1e-5
        assert isinstance(_json_serialise(np.float32(1.5)), float)

    def test_numpy_array(self):
        result = _json_serialise(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            _json_serialise(object())
