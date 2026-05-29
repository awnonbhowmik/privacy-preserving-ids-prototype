"""
test_dp_rf_model.py — Tests for the DP Random Forest module.

All tests use synthetic data. No CIC-IDS2018 or UNSW-NB15 download required.

Tests cover:
    - estimate_feature_bounds: shape, validity, buffer behaviour
    - train_dp_rf_model: result schema, JSON serialisability, pure ε-DP contract
    - run_dp_rf_sweep: multi-epsilon results, epsilon coverage, JSON save
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path
from sklearn.datasets import make_classification
from unittest.mock import patch, MagicMock

from src.models.dp_rf_model import (
    estimate_feature_bounds,
    train_dp_rf_model,
    run_dp_rf_sweep,
    DP_RF_MODEL_NAME,
    _dp_rf_model_path,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(
        n_samples=400, n_features=15, n_classes=2,
        n_informative=8, random_state=42,
    )
    return X[:300], y[:300], X[300:], y[300:]  # X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def synthetic_splits(binary_data):
    """Mock get_split_arrays return value using synthetic data."""
    X_train, y_train, X_test, y_test = binary_data
    n_val = len(X_test) // 2
    return {
        "train": (X_train, y_train),
        "val":   (X_test[:n_val], y_test[:n_val]),
        "test":  (X_test[n_val:], y_test[n_val:]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# estimate_feature_bounds
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimateFeatureBounds:
    def test_returns_two_arrays(self, binary_data):
        X_train, _, _, _ = binary_data
        lo, hi = estimate_feature_bounds(X_train)
        assert isinstance(lo, np.ndarray)
        assert isinstance(hi, np.ndarray)

    def test_shape_matches_n_features(self, binary_data):
        X_train, _, _, _ = binary_data
        lo, hi = estimate_feature_bounds(X_train)
        assert lo.shape == (X_train.shape[1],)
        assert hi.shape == (X_train.shape[1],)

    def test_lower_strictly_less_than_upper(self, binary_data):
        X_train, _, _, _ = binary_data
        lo, hi = estimate_feature_bounds(X_train)
        assert np.all(lo < hi), "Every lower bound must be < upper bound"

    def test_buffer_widens_bounds_vs_raw_percentile(self, binary_data):
        X_train, _, _, _ = binary_data
        lo_buf, hi_buf = estimate_feature_bounds(X_train, buffer=0.10)
        lo_raw, hi_raw = estimate_feature_bounds(X_train, buffer=0.0)
        # With buffer, bounds should be at least as wide
        assert np.all(lo_buf <= lo_raw)
        assert np.all(hi_buf >= hi_raw)

    def test_no_nans(self, binary_data):
        X_train, _, _, _ = binary_data
        lo, hi = estimate_feature_bounds(X_train)
        assert not np.any(np.isnan(lo))
        assert not np.any(np.isnan(hi))

    def test_zero_buffer_returns_percentile_bounds(self, binary_data):
        X_train, _, _, _ = binary_data
        lo, hi = estimate_feature_bounds(X_train, percentile=99.0, buffer=0.0)
        expected_lo = np.percentile(X_train, 1.0, axis=0)
        expected_hi = np.percentile(X_train, 99.0, axis=0)
        np.testing.assert_allclose(lo, expected_lo, rtol=1e-5)
        np.testing.assert_allclose(hi, expected_hi, rtol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# train_dp_rf_model (with mocked data pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainDpRfModel:
    """Tests for train_dp_rf_model using mocked get_split_arrays."""

    REQUIRED_KEYS = {
        "model_name", "dataset", "label_type", "epsilon", "delta",
        "dp_type", "n_estimators", "max_depth", "train_time_s", "metrics",
    }

    def _train(self, synthetic_splits, tmp_path, epsilon=1.0, **kwargs):
        with patch("src.models.dp_rf_model.get_split_arrays", return_value=synthetic_splits), \
             patch("src.models.dp_rf_model.PATHS", {"models": tmp_path, "results": tmp_path}):
            return train_dp_rf_model(
                "mock_dataset", epsilon, label="binary",
                force=True, **kwargs
            )

    def test_required_keys_present(self, synthetic_splits, tmp_path):
        r = self._train(synthetic_splits, tmp_path)
        assert self.REQUIRED_KEYS.issubset(set(r.keys()))

    def test_model_name_correct(self, synthetic_splits, tmp_path):
        r = self._train(synthetic_splits, tmp_path)
        assert r["model_name"] == DP_RF_MODEL_NAME

    def test_pure_epsilon_dp_contract(self, synthetic_splits, tmp_path):
        """DP-RF must have delta=None (pure ε-DP, no failure probability)."""
        r = self._train(synthetic_splits, tmp_path)
        assert r["delta"] is None, "DP-RF must have delta=None (pure ε-DP)"
        assert r["dp_type"] == "pure_epsilon"

    def test_epsilon_propagated(self, synthetic_splits, tmp_path):
        for eps in [0.1, 1.0, 5.0]:
            r = self._train(synthetic_splits, tmp_path, epsilon=eps)
            assert r["epsilon"] == eps

    def test_metrics_schema(self, synthetic_splits, tmp_path):
        r = self._train(synthetic_splits, tmp_path)
        assert "val" in r["metrics"]
        assert "test" in r["metrics"]
        for split in ("val", "test"):
            assert "f1_macro" in r["metrics"][split]
            assert "accuracy" in r["metrics"][split]

    def test_json_serialisable(self, synthetic_splits, tmp_path):
        r = self._train(synthetic_splits, tmp_path)
        from src.attacks.membership_inference import _json_serialise
        s = json.dumps(r, default=_json_serialise)
        assert len(s) > 0

    def test_train_time_is_positive(self, synthetic_splits, tmp_path):
        r = self._train(synthetic_splits, tmp_path)
        assert r["train_time_s"] > 0

    def test_model_file_saved(self, synthetic_splits, tmp_path):
        self._train(synthetic_splits, tmp_path, epsilon=2.0)
        saved = list(tmp_path.glob("dp_rf_*.joblib"))
        assert len(saved) >= 1, "A .joblib model file should be saved"

    def test_predict_proba_available(self, synthetic_splits, tmp_path):
        """DP-RF must support predict_proba for MIA compatibility."""
        import joblib
        self._train(synthetic_splits, tmp_path, epsilon=1.0)
        model_files = list(tmp_path.glob("dp_rf_*.joblib"))
        if model_files:
            model = joblib.load(model_files[0])
            assert hasattr(model, "predict_proba"), "DP-RF must expose predict_proba"
            X_test, _ = synthetic_splits["test"]
            proba = model.predict_proba(X_test)
            assert proba.shape[1] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# run_dp_rf_sweep
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunDpRfSweep:
    """Tests for run_dp_rf_sweep over multiple epsilon values."""

    TEST_EPS = [0.5, 1.0, 2.0]

    def _sweep(self, synthetic_splits, tmp_path):
        with patch("src.models.dp_rf_model.get_split_arrays", return_value=synthetic_splits), \
             patch("src.models.dp_rf_model.PATHS", {"models": tmp_path, "results": tmp_path}), \
             patch("src.models.dp_rf_model.PRIVACY_CFG", {"epsilon_values": self.TEST_EPS}), \
             patch("src.models.dp_rf_model.SAMPLING_CFG", {"random_seed": 42}), \
             patch("src.evaluation.results_store.PATHS", {"results": tmp_path}):
            return run_dp_rf_sweep("mock_dataset", label="binary", force=True)

    def test_returns_list(self, synthetic_splits, tmp_path):
        results = self._sweep(synthetic_splits, tmp_path)
        assert isinstance(results, list)

    def test_one_result_per_epsilon(self, synthetic_splits, tmp_path):
        results = self._sweep(synthetic_splits, tmp_path)
        assert len(results) == len(self.TEST_EPS)

    def test_all_epsilon_values_present(self, synthetic_splits, tmp_path):
        results = self._sweep(synthetic_splits, tmp_path)
        found_eps = {r["epsilon"] for r in results}
        assert found_eps == set(self.TEST_EPS)

    def test_all_results_have_pure_epsilon_dp(self, synthetic_splits, tmp_path):
        results = self._sweep(synthetic_splits, tmp_path)
        for r in results:
            assert r["delta"] is None
            assert r["dp_type"] == "pure_epsilon"

    def test_json_file_written(self, synthetic_splits, tmp_path):
        import src.utils.config as cfg
        original = cfg.PATHS.copy()
        cfg.PATHS["results"] = tmp_path
        try:
            self._sweep(synthetic_splits, tmp_path)
        finally:
            cfg.PATHS.update(original)
        # The sweep saves via save_results — file should exist
        json_files = list(tmp_path.glob("*dp_rf*.json"))
        assert len(json_files) >= 1, "Sweep should write a JSON result file"
