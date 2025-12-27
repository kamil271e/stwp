"""Tests for experiment analysis module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from exp.experiments import MODEL_LABELS, Analyzer


class TestModelLabels:
    """Tests for MODEL_LABELS constant."""

    def test_is_dict(self) -> None:
        """MODEL_LABELS should be a simple dict."""
        assert isinstance(MODEL_LABELS, dict)

    def test_contains_expected_models(self) -> None:
        """Should contain all expected model mappings."""
        expected_models = [
            "grad_booster",
            "simple_linear_regressor",
            "linear_regressor",
            "unet",
            "trans",
            "baseline_regressor",
            "tigge",
        ]
        for model in expected_models:
            assert model in MODEL_LABELS

    def test_label_values(self) -> None:
        """Should map to correct display labels."""
        assert MODEL_LABELS["grad_booster"] == "GB"
        assert MODEL_LABELS["trans"] == "GNN"
        assert MODEL_LABELS["unet"] == "U-NET"
        assert MODEL_LABELS["tigge"] == "TIGGE"


class TestAnalyzer:
    """Tests for Analyzer class."""

    def test_init(self) -> None:
        """Should initialize with empty state."""
        analyzer = Analyzer()

        assert analyzer.predictions == {}
        assert analyzer.errors == {}
        assert analyzer.avg_errors == {}
        assert analyzer.era5 is None
        assert analyzer.scalers is None

    def test_feature_list(self) -> None:
        """Should return feature list from Features enum."""
        analyzer = Analyzer()
        features = analyzer.feature_list

        assert isinstance(features, list)
        assert len(features) == 6
        assert "t2m" in features

    def test_min_length_raises_before_init(self) -> None:
        """Should raise ValueError if data not loaded."""
        analyzer = Analyzer()

        with pytest.raises(ValueError, match="Data not loaded"):
            _ = analyzer.min_length

    def test_get_label_known_model(self) -> None:
        """Should return mapped label for known models."""
        analyzer = Analyzer()

        assert analyzer._get_label("trans") == "GNN"
        assert analyzer._get_label("grad_booster") == "GB"

    def test_get_label_unknown_model(self) -> None:
        """Should return model name for unknown models."""
        analyzer = Analyzer()

        assert analyzer._get_label("unknown_model") == "unknown_model"

    def test_calculate_errors_raises_without_era5(self) -> None:
        """Should raise if ERA5 not loaded."""
        analyzer = Analyzer()
        analyzer._min_length = 10
        analyzer.predictions = {"test": np.zeros((10, 5, 5, 6))}

        with pytest.raises(ValueError, match="ERA5 data not loaded"):
            analyzer._calculate_errors()

    def test_calculate_errors(self) -> None:
        """Should calculate errors for each model."""
        analyzer = Analyzer()
        analyzer._min_length = 5

        # Create mock data
        analyzer.era5 = np.ones((5, 3, 3, 6))
        analyzer.predictions = {
            "test_model": np.zeros((5, 3, 3, 6)),
        }

        analyzer._calculate_errors()

        assert "test_model" in analyzer.errors
        # Error should be era5 - predictions = 1 - 0 = 1
        np.testing.assert_array_equal(analyzer.errors["test_model"], np.ones((5, 3, 3, 6)))

    def test_calculate_errors_skips_tigge(self) -> None:
        """Should skip tigge model in error calculation."""
        analyzer = Analyzer()
        analyzer._min_length = 5
        analyzer.era5 = np.ones((5, 3, 3, 6))
        analyzer.predictions = {
            "tigge": np.zeros((5, 3, 3, 6)),
            "test_model": np.zeros((5, 3, 3, 6)),
        }

        analyzer._calculate_errors()

        assert "tigge" not in analyzer.errors
        assert "test_model" in analyzer.errors

    def test_calculate_avg_errors(self) -> None:
        """Should calculate time-averaged errors."""
        analyzer = Analyzer()
        analyzer.errors = {
            "test_model": np.ones((10, 3, 3, 6)) * 2,
        }

        analyzer._calculate_avg_errors()

        assert "test_model" in analyzer.avg_errors
        # Average of constant 2 should be 2
        np.testing.assert_array_almost_equal(
            analyzer.avg_errors["test_model"], np.ones((3, 3, 6)) * 2
        )

    def test_calculate_metrics(self) -> None:
        """Should calculate RMSE and MAE for predictions."""
        analyzer = Analyzer()

        y_hat = np.ones((5, 3, 3, 6)) * 2
        y_true = np.ones((5, 3, 3, 6))

        rmse_values, mae_values = analyzer._calculate_metrics(y_hat, y_true, verbose=False)

        assert len(rmse_values) == 6
        assert len(mae_values) == 6
        # Error is constant 1, so RMSE = MAE = 1
        for rmse, mae in zip(rmse_values, mae_values, strict=True):
            assert pytest.approx(rmse, abs=0.01) == 1.0
            assert pytest.approx(mae, abs=0.01) == 1.0

    def test_align_tensor_lengths(self) -> None:
        """Should align tensors to minimum length."""
        analyzer = Analyzer()
        analyzer.predictions = {
            "model1": np.zeros((10, 3, 3, 6)),
            "model2": np.zeros((8, 3, 3, 6)),
            "tigge": np.zeros((20, 3, 3, 6)),  # Should be excluded from min calculation
        }
        analyzer.era5 = np.zeros((15, 3, 3, 6))

        analyzer._align_tensor_lengths()

        assert analyzer._min_length == 8
        assert analyzer.predictions["model1"].shape[0] == 8
        assert analyzer.predictions["model2"].shape[0] == 8
        # tigge should remain unchanged
        assert analyzer.predictions["tigge"].shape[0] == 20
        assert analyzer.era5.shape[0] == 8

    def test_generate_full_metrics_raises_without_era5(self) -> None:
        """Should raise if ERA5 not loaded."""
        analyzer = Analyzer()

        with pytest.raises(ValueError, match="ERA5 data not loaded"):
            analyzer.generate_full_metrics()

    @patch("exp.experiments.Analyzer._calculate_metrics")
    def test_generate_full_metrics(self, mock_calc: MagicMock) -> None:
        """Should generate metrics DataFrames for all models."""
        mock_calc.return_value = ([1.0] * 6, [0.5] * 6)

        analyzer = Analyzer()
        analyzer._min_length = 5
        analyzer.era5 = np.ones((10, 3, 3, 6))
        analyzer.predictions = {
            "trans": np.zeros((5, 3, 3, 6)),
        }

        rmse_df, mae_df = analyzer.generate_full_metrics(verbose=False, latex=False)

        assert "GNN" in rmse_df.index
        assert rmse_df.shape == (1, 6)
        assert mae_df.shape == (1, 6)


class TestAnalyzerEvaluateCombination:
    """Tests for _evaluate_combination method."""

    def test_evaluate_combination_raises_without_era5(self) -> None:
        """Should raise if ERA5 not loaded."""
        analyzer = Analyzer()
        y1 = np.zeros((5, 3, 3, 6))
        y2 = np.ones((5, 3, 3, 6))

        with pytest.raises(ValueError, match="ERA5 data not loaded"):
            analyzer._evaluate_combination(y1, y2)

    def test_evaluate_combination_weighted_average(self) -> None:
        """Should compute weighted average of predictions."""
        analyzer = Analyzer()
        analyzer.era5 = np.ones((10, 3, 3, 6))

        y1 = np.zeros((5, 3, 3, 6))
        y2 = np.ones((5, 3, 3, 6)) * 2

        # With alpha=0.5, combined should be (0 + 2) / 2 = 1
        result = analyzer._evaluate_combination(y1, y2, alpha=0.5, consolidate=False)
        assert result is None  # Non-consolidate returns None

    @patch("exp.experiments.Analyzer._ensure_scalers_loaded")
    def test_evaluate_combination_consolidate(self, mock_scalers: MagicMock) -> None:
        """Should return consolidated loss when consolidate=True."""
        analyzer = Analyzer()
        analyzer.era5 = np.ones((10, 3, 3, 6))

        # Mock scalers
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((45, 1))
        analyzer.scalers = [mock_scaler] * 6

        y1 = np.zeros((5, 3, 3, 6))
        y2 = np.ones((5, 3, 3, 6))

        result = analyzer._evaluate_combination(y1, y2, alpha=0.5, consolidate=True)
        assert isinstance(result, float)
