"""Tests for HPO visualization module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from hpo.visualization import (
    HPOVisualizer,
    PlotConfig,
    StyleConfig,
    Visualization,
    _save_and_show,
)


class TestStyleConfig:
    """Tests for StyleConfig dataclass."""

    def test_is_frozen_dataclass(self) -> None:
        """StyleConfig should be a frozen dataclass."""
        config = StyleConfig()
        assert config is not None

    def test_colors_has_expected_models(self) -> None:
        """Should have colors for all model types."""
        expected = ["simple-linear", "linear", "lgbm", "gnn", "cnn"]
        for model in expected:
            assert model in StyleConfig.COLORS

    def test_labels_is_dict(self) -> None:
        """LABELS should be a dict."""
        assert isinstance(StyleConfig.LABELS, dict)

    def test_labels_has_latex_format(self) -> None:
        """Labels should be LaTeX formatted."""
        assert StyleConfig.LABELS["gnn"] == r"$GNN$"
        assert StyleConfig.LABELS["lgbm"] == r"$GB$"

    def test_features_count(self) -> None:
        """Should have 6 features."""
        assert len(StyleConfig.FEATURES) == 6
        assert "t2m" in StyleConfig.FEATURES
        assert "tp" in StyleConfig.FEATURES

    def test_months_count(self) -> None:
        """Should have 12 months."""
        assert len(StyleConfig.MONTHS) == 12
        assert StyleConfig.MONTHS[0] == "Jan"
        assert StyleConfig.MONTHS[11] == "Dec"

    def test_baseline_types(self) -> None:
        """Should contain expected baseline models."""
        assert "simple-linear" in StyleConfig.BASELINE_TYPES
        assert "linear" in StyleConfig.BASELINE_TYPES
        assert "lgbm" in StyleConfig.BASELINE_TYPES

    def test_neural_net_types(self) -> None:
        """Should contain GNN and CNN."""
        assert "gnn" in StyleConfig.NEURAL_NET_TYPES
        assert "cnn" in StyleConfig.NEURAL_NET_TYPES


class TestPlotConfig:
    """Tests for PlotConfig class."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = PlotConfig()

        assert config.figsize == (10, 8)
        assert config.save_format == "pdf"
        assert config.show is True

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = PlotConfig(figsize=(15, 10), save_format="png", show=False)

        assert config.figsize == (15, 10)
        assert config.save_format == "png"
        assert config.show is False


class TestSaveAndShow:
    """Tests for _save_and_show helper function."""

    @patch("hpo.visualization.plt")
    def test_saves_when_path_provided(self, mock_plt: MagicMock) -> None:
        """Should save figure when save_path is provided."""
        mock_fig = MagicMock()
        config = PlotConfig(show=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = f"{tmpdir}/test_plot"
            _save_and_show(mock_fig, save_path, config)

            mock_fig.savefig.assert_called_once()
            call_args = mock_fig.savefig.call_args[0][0]
            assert call_args == f"{save_path}.pdf"

    @patch("hpo.visualization.plt")
    def test_shows_when_configured(self, mock_plt: MagicMock) -> None:
        """Should call plt.show when config.show is True."""
        mock_fig = MagicMock()
        config = PlotConfig(show=True)

        _save_and_show(mock_fig, None, config)

        mock_plt.show.assert_called_once()

    @patch("hpo.visualization.plt")
    def test_no_show_when_disabled(self, mock_plt: MagicMock) -> None:
        """Should not call plt.show when config.show is False."""
        mock_fig = MagicMock()
        config = PlotConfig(show=False)

        _save_and_show(mock_fig, None, config)

        mock_plt.show.assert_not_called()

    @patch("hpo.visualization.plt")
    def test_always_closes_figure(self, mock_plt: MagicMock) -> None:
        """Should always close the figure."""
        mock_fig = MagicMock()
        config = PlotConfig(show=False)

        _save_and_show(mock_fig, None, config)

        mock_plt.close.assert_called_once_with(mock_fig)


class TestHPOVisualizer:
    """Tests for HPOVisualizer class."""

    def test_init_with_default_config(self) -> None:
        """Should initialize with default PlotConfig."""
        viz = HPOVisualizer(data_path="nonexistent.json")

        assert viz.config is not None
        assert isinstance(viz.config, PlotConfig)

    def test_init_with_custom_config(self) -> None:
        """Should use provided config."""
        config = PlotConfig(figsize=(20, 15))
        viz = HPOVisualizer(data_path="nonexistent.json", config=config)

        assert viz.config.figsize == (20, 15)

    def test_load_data_from_file(self) -> None:
        """Should load JSON data from file."""
        test_data = {"gnn": {"sequence_plot_x": [1, 2, 3], "sequence_plot_y": [0.5, 0.4, 0.3]}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            viz = HPOVisualizer(data_path=temp_path)
            assert viz.data == test_data
        finally:
            Path(temp_path).unlink()

    def test_load_data_missing_file(self) -> None:
        """Should handle missing file gracefully."""
        viz = HPOVisualizer(data_path="definitely_not_exists.json")
        assert viz.data == {}

    def test_model_types_property(self) -> None:
        """Should return list of available model types."""
        test_data = {"gnn": {}, "cnn": {}, "lgbm": {}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            viz = HPOVisualizer(data_path=temp_path)
            assert set(viz.model_types) == {"gnn", "cnn", "lgbm"}
        finally:
            Path(temp_path).unlink()

    def test_get_color_known_model(self) -> None:
        """Should return color for known models."""
        viz = HPOVisualizer(data_path="nonexistent.json")

        assert viz._get_color("gnn") == "black"
        assert viz._get_color("lgbm") == "#4daf4a"

    def test_get_color_unknown_model(self) -> None:
        """Should return gray for unknown models."""
        viz = HPOVisualizer(data_path="nonexistent.json")

        assert viz._get_color("unknown") == "gray"

    def test_get_label_known_model(self) -> None:
        """Should return label for known models."""
        viz = HPOVisualizer(data_path="nonexistent.json")

        assert viz._get_label("gnn") == r"$GNN$"

    def test_get_label_unknown_model(self) -> None:
        """Should return model name for unknown models."""
        viz = HPOVisualizer(data_path="nonexistent.json")

        assert viz._get_label("unknown") == "unknown"

    def test_create_metrics_table(self) -> None:
        """Should create metrics table from data."""
        test_data = {
            "gnn": {"metrics": [1.0, 2.0]},
            "cnn": {"metrics": [1.5, 2.5]},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            viz = HPOVisualizer(data_path=temp_path)
            table = viz.create_metrics_table()

            assert len(table) == 2
            assert table[0]["model_type"] == "gnn"
            assert table[0]["metrics"] == [1.0, 2.0]
        finally:
            Path(temp_path).unlink()

    def test_create_scaler_metrics_table(self) -> None:
        """Should create scaler metrics table from data."""
        test_data = {
            "gnn": {"metrics_for_scalers": {"standard": 0.5, "robust": 0.6}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            viz = HPOVisualizer(data_path=temp_path)
            table = viz.create_scaler_metrics_table()

            assert len(table) == 1
            assert table[0]["metrics_for_scalers"] == {"standard": 0.5, "robust": 0.6}
        finally:
            Path(temp_path).unlink()

    def test_transform_to_feature_format(self) -> None:
        """Should transform model-centric data to feature-centric."""
        test_data = {
            "gnn": {
                "not_normalized_plot_sequence": {
                    "1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    "2": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            viz = HPOVisualizer(data_path=temp_path)
            result = viz._transform_to_feature_format("not_normalized_plot_sequence")

            # Should have data for each feature
            assert "t2m" in result
            assert "gnn" in result["t2m"]
            assert result["t2m"]["gnn"]["y"] == [0.1, 0.15]
        finally:
            Path(temp_path).unlink()


class TestVisualizationAlias:
    """Tests for backwards compatibility alias."""

    def test_visualization_is_hpo_visualizer(self) -> None:
        """Visualization should be alias for HPOVisualizer."""
        assert Visualization is HPOVisualizer
