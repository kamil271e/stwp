"""Visualization module for HPO results.

Provides plotting utilities for analyzing hyperparameter optimization
results across different model types.
"""

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from stwp.data.processor import DataProcessor
from stwp.utils.visualization import draw_poland

logger = logging.getLogger(__name__)

plt.style.use("ggplot")


@dataclass(frozen=True)
class StyleConfig:
    """Styling constants for plots."""

    COLORS: ClassVar[dict[str, str]] = {
        "simple-linear": "#377eb8",
        "linear": "#ff7f00",
        "lgbm": "#4daf4a",
        "gnn": "black",
        "cnn": "#a65628",
    }

    LABELS: ClassVar[dict[str, str]] = {
        "lgbm": r"$GB$",
        "simple-linear": r"$SLR$",
        "linear": r"$LR$",
        "gnn": r"$GNN$",
        "cnn": r"$U-NET$",
    }

    FEATURES: ClassVar[tuple[str, ...]] = ("t2m", "sp", "tcc", "u10", "v10", "tp")

    MONTHS: ClassVar[tuple[str, ...]] = (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    )

    BASELINE_TYPES: ClassVar[tuple[str, ...]] = ("simple-linear", "linear", "lgbm")
    NEURAL_NET_TYPES: ClassVar[tuple[str, ...]] = ("gnn", "cnn")


class SaveFormat(StrEnum):
    """Supported save formats for plots."""

    PDF = "pdf"
    PNG = "png"
    SVG = "svg"


class PlotConfig:
    """Configuration for plot generation."""

    def __init__(
        self,
        figsize: tuple[int, int] = (10, 8),
        save_format: SaveFormat = SaveFormat.PDF,
        show: bool = True,
    ) -> None:
        self.figsize = figsize
        self.save_format = save_format
        self.show = show


def _save_and_show(
    fig: plt.Figure,
    save_path: str | None,
    config: PlotConfig,
) -> None:
    """Save figure and optionally display it."""
    if save_path:
        fig.savefig(f"{save_path}.{config.save_format.value}", bbox_inches="tight")
    if config.show:
        plt.show()
    plt.close(fig)


class HPOVisualizer:
    """Visualizer for HPO experiment results.

    Loads results from JSON files and generates various plots for
    analyzing model performance across different hyperparameters.
    """

    def __init__(
        self,
        data_path: str = "modelsplots.json",
        config: PlotConfig | None = None,
    ) -> None:
        """Initialize visualizer.

        Args:
            data_path: Path to JSON file containing plot data.
            config: Plot configuration options.
        """
        self.data_path = Path(data_path)
        self.config = config or PlotConfig()
        self.data: dict[str, Any] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load plot data from JSON file."""
        if self.data_path.exists():
            with open(self.data_path) as f:
                self.data = json.load(f)
        else:
            logger.warning(f"Data file not found: {self.data_path}")

    @property
    def model_types(self) -> list[str]:
        """Available model types in loaded data."""
        return list(self.data.keys())

    def _get_color(self, model_type: str) -> str:
        return StyleConfig.COLORS.get(model_type, "gray")

    def _get_label(self, model_type: str) -> str:
        return StyleConfig.LABELS.get(model_type, model_type)

    # -------------------------------------------------------------------------
    # Metrics Tables
    # -------------------------------------------------------------------------

    def create_metrics_table(self) -> list[dict[str, Any]]:
        """Create table of metrics for all model types.

        Returns:
            List of dictionaries with model metrics.
        """
        table = [
            {"model_type": model, "metrics": self.data[model].get("metrics", [])}
            for model in self.data
        ]
        print(tabulate(table, tablefmt="fancy_grid"))
        return table

    def create_scaler_metrics_table(self) -> list[dict[str, Any]]:
        """Create table of scaler comparison metrics.

        Returns:
            List of dictionaries with scaler metrics per model.
        """
        return [
            {
                "model_type": model,
                "metrics_for_scalers": self.data[model].get("metrics_for_scalers", {}),
            }
            for model in self.data
        ]

    # -------------------------------------------------------------------------
    # Sequence Length Plots
    # -------------------------------------------------------------------------

    def plot_sequence_rmse(
        self,
        combined: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plot RMSE vs sequence length.

        Args:
            combined: If True, show all models on one plot.
            save_path: Optional path to save the figure.
        """
        if combined:
            self._plot_sequence_combined(save_path)
        else:
            self._plot_sequence_grid(save_path)

    def _plot_sequence_combined(self, save_path: str | None) -> None:
        fig, ax = plt.subplots(figsize=self.config.figsize)

        for model in self.data:
            x = self.data[model].get("sequence_plot_x", [])
            y = self.data[model].get("sequence_plot_y", [])
            if x and y:
                ax.plot(x, y, "-o", label=self._get_label(model), color=self._get_color(model))

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(r"$\tilde{\mathcal{L}}_{RMSE}$")
        ax.set_title("Input Sequence Length")
        ax.legend()

        _save_and_show(fig, save_path, self.config)

    def _plot_sequence_grid(self, save_path: str | None) -> None:
        num_models = len(self.data)
        num_cols = 2
        num_rows = (num_models + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=self.config.figsize)
        axes = np.atleast_2d(axes)

        for i, model in enumerate(self.data):
            row, col = i // num_cols, i % num_cols
            ax = axes[row, col]

            x = self.data[model].get("sequence_plot_x", [])
            y = self.data[model].get("sequence_plot_y", [])
            if x and y:
                ax.plot(x, y, "-o", color=self._get_color(model))
                ax.set_title(self._get_label(model))
                ax.set_xlabel("Sequence Length")
                ax.set_ylabel(r"$\tilde{\mathcal{L}}_{RMSE}$")

        plt.tight_layout()
        _save_and_show(fig, save_path, self.config)

    def plot_sequence_time(
        self,
        combined: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plot training time vs sequence length.

        Args:
            combined: If True, show all models on one plot.
            save_path: Optional path to save the figure.
        """
        if combined:
            self._plot_time_combined("sequence", save_path)
        else:
            self._plot_time_grid("sequence", save_path)

    # -------------------------------------------------------------------------
    # Forecast Horizon Plots
    # -------------------------------------------------------------------------

    def plot_fh_rmse(
        self,
        combined: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plot RMSE vs forecast horizon.

        Args:
            combined: If True, show all models on one plot.
            save_path: Optional path to save the figure.
        """
        if combined:
            self._plot_fh_combined(save_path)
        else:
            self._plot_fh_grid(save_path)

    def _plot_fh_combined(self, save_path: str | None) -> None:
        fig, ax = plt.subplots(figsize=self.config.figsize)

        for model in self.data:
            x = self.data[model].get("fh_plot_x", [])
            y = self.data[model].get("fh_plot_y", [])
            if x and y:
                ax.plot(x, y, "-o", label=self._get_label(model), color=self._get_color(model))

        ax.set_xlabel("Number of Predicted Steps")
        ax.set_ylabel(r"$\tilde{\mathcal{L}}_{RMSE}$")
        ax.set_title("Predicted Steps")
        ax.legend()

        _save_and_show(fig, save_path, self.config)

    def _plot_fh_grid(self, save_path: str | None) -> None:
        num_models = len(self.data)
        num_cols = 2
        num_rows = (num_models + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=self.config.figsize)
        axes = np.atleast_2d(axes)

        for i, model in enumerate(self.data):
            row, col = i // num_cols, i % num_cols
            ax = axes[row, col]

            x = self.data[model].get("fh_plot_x", [])
            y = self.data[model].get("fh_plot_y", [])
            if x and y:
                ax.plot(x, y, "-o", color=self._get_color(model))
                ax.set_title(self._get_label(model))
                ax.set_xlabel("Forecast Horizon")
                ax.set_ylabel(r"$\tilde{\mathcal{L}}_{RMSE}$")

        plt.tight_layout()
        _save_and_show(fig, save_path, self.config)

    def plot_fh_time(
        self,
        combined: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plot training time vs forecast horizon.

        Args:
            combined: If True, show all models on one plot.
            save_path: Optional path to save the figure.
        """
        if combined:
            self._plot_time_combined("fh", save_path)
        else:
            self._plot_time_grid("fh", save_path)

    # -------------------------------------------------------------------------
    # Time Plots (shared implementation)
    # -------------------------------------------------------------------------

    def _plot_time_combined(
        self,
        metric_type: str,
        save_path: str | None,
    ) -> None:
        """Plot training time for baselines and neural nets separately."""
        x_key = f"{metric_type}_plot_x"
        time_key = f"{metric_type}_plot_time"
        xlabel = "Sequence Length" if metric_type == "sequence" else "Number of Predicted Steps"

        # Baselines plot
        fig, ax = plt.subplots(figsize=self.config.figsize)
        for model in self.data:
            if model in StyleConfig.BASELINE_TYPES:
                x = self.data[model].get(x_key, [])
                times = self.data[model].get(time_key, [])
                if x and times:
                    ax.plot(
                        x, times, "-o", label=self._get_label(model), color=self._get_color(model)
                    )

        ax.set_title(f"Training Time (Baselines) - {metric_type.title()}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time [s]")
        ax.legend()

        baseline_path = f"baselines_{save_path}" if save_path else None
        _save_and_show(fig, baseline_path, self.config)

        # Neural nets plot
        fig, ax = plt.subplots(figsize=self.config.figsize)
        for model in self.data:
            if model in StyleConfig.NEURAL_NET_TYPES:
                x = self.data[model].get(x_key, [])
                times = self.data[model].get(time_key, [])
                if x and times:
                    ax.plot(
                        x, times, "-o", label=self._get_label(model), color=self._get_color(model)
                    )

        ax.set_title(f"Training Time (Neural Nets) - {metric_type.title()}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time [s]")
        ax.legend()

        nets_path = f"nets_{save_path}" if save_path else None
        _save_and_show(fig, nets_path, self.config)

    def _plot_time_grid(
        self,
        metric_type: str,
        save_path: str | None,
    ) -> None:
        x_key = f"{metric_type}_plot_x"
        time_key = f"{metric_type}_plot_time"
        xlabel = "Sequence Length" if metric_type == "sequence" else "Forecast Horizon"

        num_models = len(self.data)
        num_cols = 2
        num_rows = (num_models + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=self.config.figsize)
        axes = np.atleast_2d(axes)

        for i, model in enumerate(self.data):
            row, col = i // num_cols, i % num_cols
            ax = axes[row, col]

            x = self.data[model].get(x_key, [])
            times = self.data[model].get(time_key, [])
            if x and times:
                ax.plot(x, times, "-o", color=self._get_color(model))
                ax.set_title(self._get_label(model))
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Time [s]")

        plt.tight_layout()
        _save_and_show(fig, save_path, self.config)

    # -------------------------------------------------------------------------
    # Feature-wise Plots
    # -------------------------------------------------------------------------

    def plot_raw_metrics_by_feature(
        self,
        metric_type: str = "sequence",
        combined: bool = False,
        save_path: str | None = None,
    ) -> None:
        """Plot raw (non-normalized) metrics per feature.

        Args:
            metric_type: "sequence" or "fh" for forecast horizon.
            combined: If True, show all features in one grid.
            save_path: Optional path to save the figure.
        """
        data_key = f"not_normalized_plot_{metric_type}"
        xlabel = "Sequence Length" if metric_type == "sequence" else "Number of Predicted Steps"
        title_prefix = "Input Sequence" if metric_type == "sequence" else "Predicted Steps"

        if combined:
            self._plot_features_grid(data_key, xlabel, title_prefix, save_path)
        else:
            self._plot_features_separate(data_key, xlabel, title_prefix, save_path)

    def _plot_features_grid(
        self,
        data_key: str,
        xlabel: str,
        title_prefix: str,
        save_path: str | None,
    ) -> None:
        """Plot all features in a grid, comparing models."""
        num_features = len(StyleConfig.FEATURES)
        num_cols = 2
        num_rows = (num_features + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 18))
        axes = np.atleast_2d(axes)

        # Transform data from per-model to per-feature format
        feature_data = self._transform_to_feature_format(data_key)

        for j, feature in enumerate(StyleConfig.FEATURES):
            row, col = j // num_cols, j % num_cols
            ax = axes[row, col]

            for model, values in feature_data.get(feature, {}).items():
                if values["y"]:
                    ax.plot(
                        values["x"],
                        values["y"],
                        "-o",
                        label=self._get_label(model),
                        color=self._get_color(model),
                    )

            ax.set_title(f"{title_prefix} - {feature}", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(r"$\mathcal{L}_{RMSE}$", fontsize=8)
            ax.legend(fontsize=8)

        plt.tight_layout()
        _save_and_show(fig, save_path, self.config)

    def _plot_features_separate(
        self,
        data_key: str,
        xlabel: str,
        title_prefix: str,
        save_path: str | None,
    ) -> None:
        """Plot each feature in a separate figure."""
        feature_data = self._transform_to_feature_format(data_key)

        for feature in StyleConfig.FEATURES:
            fig, ax = plt.subplots(figsize=self.config.figsize)

            for model, values in feature_data.get(feature, {}).items():
                if values["y"]:
                    ax.plot(values["x"], values["y"], "-o", label=self._get_label(model))

            ax.set_title(f"{title_prefix} - {feature}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\mathcal{L}_{RMSE}$")
            ax.legend()

            feature_path = f"{feature}_{save_path}" if save_path else None
            _save_and_show(fig, feature_path, self.config)

    def _transform_to_feature_format(
        self,
        data_key: str,
    ) -> dict[str, dict[str, dict[str, list[Any]]]]:
        """Transform data from model-centric to feature-centric format.

        Returns:
            Dict mapping feature -> model -> {x: [...], y: [...]}
        """
        result: dict[str, dict[str, dict[str, list[Any]]]] = {
            feature: {} for feature in StyleConfig.FEATURES
        }

        for model in self.data:
            raw_data = self.data[model].get(data_key, {})
            if not raw_data:
                continue

            x_values = list(raw_data.keys())

            for i, feature in enumerate(StyleConfig.FEATURES):
                y_values = []
                for key in x_values:
                    values = raw_data[key]
                    if isinstance(values, list) and i < len(values):
                        y_values.append(values[i])

                result[feature][model] = {"x": x_values, "y": y_values}

        return result

    # -------------------------------------------------------------------------
    # Monthly Error Plot
    # -------------------------------------------------------------------------

    def plot_monthly_errors(self, save_path: str | None = None) -> None:
        """Plot monthly prediction errors as grouped bar chart.

        Args:
            save_path: Optional path to save the figure.
        """
        fig, ax = plt.subplots(figsize=(20, 8))

        bar_width = 0.15
        models_with_data = []

        for model in self.data:
            month_errors = self.data[model].get("month_error", {})
            if len(month_errors) == 12:
                models_with_data.append(model)

        x_positions = np.arange(12)

        for i, model in enumerate(models_with_data):
            month_errors = self.data[model]["month_error"]
            values = list(month_errors.values())
            ax.bar(
                x_positions + bar_width * i,
                values,
                bar_width,
                label=self._get_label(model),
                color=self._get_color(model),
            )

        ax.set_xticks(x_positions + bar_width * (len(models_with_data) - 1) / 2)
        ax.set_xticklabels(StyleConfig.MONTHS)
        ax.set_title("Monthly Prediction Errors")
        ax.set_xlabel("Month")
        ax.set_ylabel(r"$\tilde{\mathcal{L}}_{RMSE}$")
        ax.legend()

        _save_and_show(fig, save_path, self.config)

    # -------------------------------------------------------------------------
    # GNN-specific Plots
    # -------------------------------------------------------------------------

    def plot_gnn_alpha(self, save_path: str | None = None) -> None:
        """Plot GNN alpha parameter study results.

        Args:
            save_path: Optional path to save the figure.
        """
        if "gnn" not in self.data:
            logger.warning("No GNN data available")
            return

        fig, ax = plt.subplots(figsize=self.config.figsize)

        x = self.data["gnn"].get("gnn_alpha_plot_x", [])
        y = self.data["gnn"].get("gnn_alpha_plot_y", [])

        if x and y:
            ax.plot(x, y, "-o", label=self._get_label("gnn"), color=self._get_color("gnn"))
            ax.set_xlabel("Alpha")
            ax.set_ylabel(r"$\tilde{\mathcal{L}}_{RMSE}$")
            ax.set_title("Alpha for GNN and TIGGE Mix")
            ax.legend()

        _save_and_show(fig, save_path, self.config)

    def plot_gnn_layers(self, save_path: str | None = None) -> None:
        """Plot GNN layer count study results.

        Args:
            save_path: Optional path to save the figure.
        """
        if "gnn" not in self.data:
            logger.warning("No GNN data available")
            return

        fig, ax = plt.subplots(figsize=self.config.figsize)

        x = self.data["gnn"].get("gnn_cell_plot_x", [])
        y = self.data["gnn"].get("gnn_cell_plot_y", [])

        if x and y:
            ax.plot(x, y, "-o", label=self._get_label("gnn"), color=self._get_color("gnn"))
            ax.set_xlabel("Number of Graph Layers")
            ax.set_ylabel(r"$\tilde{\mathcal{L}}_{RMSE}$")
            ax.set_title("GNN Graph Layers")
            ax.legend()

        _save_and_show(fig, save_path, self.config)

    # -------------------------------------------------------------------------
    # Error Maps
    # -------------------------------------------------------------------------

    def plot_error_maps(self) -> None:
        """Plot spatial error maps for each model type."""
        lat_span, lon_span, spatial_limits = DataProcessor.get_spatial_info()
        spatial = {
            "lat_span": lat_span,
            "lon_span": lon_span,
            "spatial_limits": spatial_limits,
        }

        for model in self.data:
            error_map_path = Path(f"./{model}/error_maps.npy")
            if not error_map_path.exists():
                continue

            error_maps = np.load(error_map_path)

            fig, axs = plt.subplots(
                len(StyleConfig.FEATURES),
                figsize=(10, 12),
                subplot_kw={"projection": ccrs.Mercator(central_longitude=40)},
            )

            for j, feature in enumerate(StyleConfig.FEATURES):
                ax = axs[j]
                title = rf"$|(X - \hat{{X}})^2|_{{{feature}}}$"
                draw_poland(ax, error_maps[j], title, "binary", **spatial)

            fig.suptitle(f"{model} error maps", x=0.7, y=0.95, weight="bold")
            plt.show()


# Backwards compatibility alias
Visualization = HPOVisualizer
