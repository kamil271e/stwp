"""Experiment analysis module for comparing model predictions.

This module provides utilities for loading predictions, calculating errors,
and generating visualizations comparing different weather prediction models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import CenteredNorm
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error

from stwp.data.processor import DataProcessor
from stwp.features import Features
from stwp.models.gnn.processor import NNDataProcessor
from stwp.utils.visualization import draw_poland

logger = logging.getLogger(__name__)

plt.style.use("ggplot")


MODEL_LABELS: dict[str, str] = {
    "grad_booster": "GB",
    "simple_linear_regressor": "SLR",
    "linear_regressor": "LR",
    "unet": "U-NET",
    "trans": "GNN",
    "baseline_regressor": "NAIVE",
    "tigge": "TIGGE",
}


class Analyzer:
    """Analyzer for comparing weather prediction models.

    Loads prediction tensors from various models, compares them against
    ERA5 ground truth, and generates analysis visualizations.
    """

    DEFAULT_PRED_PATH = Path("../data/pred/")
    DEFAULT_ERA5_PATH = Path("../data/input/data2021-small.grib")
    DEFAULT_NN_DATA_PATH = Path("../data/input/data2019-2021-small.grib")
    DEFAULT_ANALYSIS_PATH = Path("../data/analysis/")

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self.predictions: dict[str, NDArray[np.floating[Any]]] = {}
        self.errors: dict[str, NDArray[np.floating[Any]]] = {}
        self.avg_errors: dict[str, NDArray[np.floating[Any]]] = {}
        self.era5: NDArray[np.floating[Any]] | None = None
        self.scalers: list[Any] | None = None
        self._nn_processor: NNDataProcessor | None = None
        self._min_length: int | None = None

    @property
    def feature_list(self) -> list[str]:
        """Feature names used in analysis."""
        return Features.as_list()

    @property
    def min_length(self) -> int:
        """Minimum sample length across all loaded data."""
        if self._min_length is None:
            raise ValueError("Data not loaded. Call init() first.")
        return self._min_length

    def _get_label(self, model_name: str) -> str:
        """Get display label for a model."""
        return MODEL_LABELS.get(model_name, model_name)

    def init(self) -> None:
        """Initialize analyzer by loading data and calculating errors."""
        self.load_predictions()
        self.load_era5()
        self._calculate_errors()

    def _align_tensor_lengths(self) -> None:
        """Align all tensors to the same minimum length."""
        self._min_length = min(
            tensor.shape[0] for name, tensor in self.predictions.items() if name != "tigge"
        )
        for model, tensor in self.predictions.items():
            if model != "tigge":
                self.predictions[model] = tensor[-self._min_length :]
        if self.era5 is not None:
            self.era5 = self.era5[-self._min_length :]

    def load_predictions(self, path: Path | str = DEFAULT_PRED_PATH) -> None:
        """Load prediction tensors from directory.

        Args:
            path: Directory containing prediction .npy files.
        """
        path = Path(path)
        for filename in os.listdir(path):
            filepath = path / filename
            if filepath.is_file() and filename.endswith(".npy"):
                tensor = np.load(filepath)
                model_name = filename.split("_2024")[0]

                if "tigge" in filename:
                    self.predictions[model_name] = tensor
                else:
                    # Extract first forecast horizon
                    self.predictions[model_name] = tensor[..., 0]

        self._min_length = min(
            tensor.shape[0] for name, tensor in self.predictions.items() if name != "tigge"
        )

    def load_era5(self, path: Path | str = DEFAULT_ERA5_PATH) -> None:
        """Load ERA5 ground truth data.

        Args:
            path: Path to ERA5 GRIB file.
        """
        processor = DataProcessor(path=str(path))
        self.era5 = processor.data
        if self._min_length is not None:
            self._min_length = min(self.era5.shape[0], self._min_length)

    def _ensure_scalers_loaded(self) -> None:
        """Lazily load scalers from NNDataProcessor."""
        if self._nn_processor is None:
            self._nn_processor = NNDataProcessor(path=str(self.DEFAULT_NN_DATA_PATH))
            self._nn_processor.preprocess()
            self.scalers = self._nn_processor.scalers

    def _calculate_errors(self) -> None:
        """Calculate prediction errors for all models."""
        if self.era5 is None:
            raise ValueError("ERA5 data not loaded")

        era5_slice = self.era5[-self.min_length :]
        for model, predictions in self.predictions.items():
            if model == "tigge":
                continue

            pred_slice = predictions[-self.min_length :]
            self.errors[model] = np.zeros_like(era5_slice)

            for i, _feature in enumerate(self.feature_list):
                self.errors[model][..., i] = era5_slice[..., i] - pred_slice[..., i]

    def _calculate_avg_errors(self) -> None:
        """Calculate average errors across time for spatial analysis."""
        for model, error_tensor in self.errors.items():
            avg_tensor = np.zeros(error_tensor.shape[1:])
            for i in range(len(self.feature_list)):
                avg_tensor[..., i] = np.mean(error_tensor[..., i], axis=0)
            self.avg_errors[model] = avg_tensor

    def _calculate_metrics(
        self,
        y_hat: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        verbose: bool = True,
    ) -> tuple[list[float], list[float]]:
        """Calculate RMSE and MAE for each feature.

        Args:
            y_hat: Predictions
            y_true: Ground truth
            verbose: Whether to print metrics

        Returns:
            Tuple of (rmse_per_feature, mae_per_feature)
        """
        rmse_values, mae_values = [], []

        for i, feature in enumerate(self.feature_list):
            y_true_flat = y_true[..., i].reshape(-1, 1)
            y_hat_flat = y_hat[..., i].reshape(-1, 1)

            rmse = np.sqrt(mean_squared_error(y_hat_flat, y_true_flat))
            mae = mean_absolute_error(y_hat_flat, y_true_flat)

            if verbose:
                logger.info(f"RMSE for {feature}: {rmse:.3f}; MAE for {feature}: {mae:.3f}")

            rmse_values.append(rmse)
            mae_values.append(mae)

        return rmse_values, mae_values

    # -------------------------------------------------------------------------
    # Metrics Generation
    # -------------------------------------------------------------------------

    def generate_full_metrics(
        self,
        verbose: bool = False,
        latex: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate RMSE and MAE metrics for all models.

        Args:
            verbose: Whether to print per-model metrics
            latex: Whether to print LaTeX formatted tables

        Returns:
            Tuple of (rmse_dataframe, mae_dataframe)
        """
        if self.era5 is None:
            raise ValueError("ERA5 data not loaded")

        rmse_results, mae_results = [], []
        models = []

        for model, predictions in self.predictions.items():
            if "tigge" in model:
                # TIGGE data has different temporal alignment
                rmse, mae = self._calculate_metrics(predictions, self.era5[1::2], verbose=verbose)
            else:
                rmse, mae = self._calculate_metrics(
                    predictions[-self.min_length :],
                    self.era5[-self.min_length :],
                    verbose=verbose,
                )

            if verbose:
                logger.info(f"Model: {model}\n")

            models.append(self._get_label(model))
            rmse_results.append(rmse)
            mae_results.append(mae)

        rmse_df = pd.DataFrame(
            np.array(rmse_results),
            columns=self.feature_list,
            index=models,
        )
        mae_df = pd.DataFrame(
            np.array(mae_results),
            columns=self.feature_list,
            index=models,
        )

        if latex:
            print(rmse_df.to_latex(float_format="%.3f", caption="RMSE Results", label="tab:rmse"))
            print(mae_df.to_latex(float_format="%.3f", caption="MAE Results", label="tab:mae"))

        return rmse_df, mae_df

    # -------------------------------------------------------------------------
    # Correlation Analysis
    # -------------------------------------------------------------------------

    def plot_error_correlation_matrix(self, save: bool = False) -> None:
        """Plot correlation matrix of prediction errors across models.

        Args:
            save: Whether to save the figure.
        """
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        divider = fig.add_axes([1.05, 0.15, 0.02, 0.8])
        vmin, vmax = 1.0, -1.0

        for i, (feature, ax) in enumerate(zip(self.feature_list, axs.flatten(), strict=False)):
            ax.set_title(feature)

            df_err = pd.DataFrame(
                {
                    self._get_label(model): self.errors[model][..., i].reshape(-1)
                    for model in self.errors
                }
            )

            corr = df_err.corr()
            vmin = min(vmin, corr.values.min())
            vmax = max(vmax, corr.values.max())

            sns.heatmap(
                corr,
                square=True,
                cmap="RdYlGn",
                annot=True,
                ax=ax,
                annot_kws={"fontsize": 10},
                fmt=".2f",
                cbar=(i == 0),
                cbar_ax=None if i else divider,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        cbar = fig.colorbar(axs[-1, -1].collections[0], cax=divider, pad=0.02)
        cbar.set_ticks(np.linspace(vmin, vmax, 6))
        fig.subplots_adjust(right=0.8, hspace=0.5)
        plt.tight_layout()

        if save:
            plt.savefig(self.DEFAULT_ANALYSIS_PATH / "err_corr_matrix.pdf")
        plt.show()

    def plot_prediction_correlation_matrix(self, save: bool = False) -> None:
        """Plot correlation matrix of predictions across models.

        Args:
            save: Whether to save the figure.
        """
        filtered = {k: v for k, v in self.predictions.items() if k != "tigge"}

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        divider = fig.add_axes([1.05, 0.05, 0.02, 0.8])
        vmin, vmax = 1.0, -1.0

        for i, (feature, ax) in enumerate(zip(self.feature_list, axs.flatten(), strict=False)):
            ax.set_title(feature)

            df_pred = pd.DataFrame(
                {self._get_label(model): filtered[model][..., i].reshape(-1) for model in filtered}
            )

            corr = df_pred.corr()
            vmin = min(vmin, corr.values.min())
            vmax = max(vmax, corr.values.max())

            sns.heatmap(
                corr,
                square=True,
                cmap="RdYlGn",
                annot=True,
                ax=ax,
                annot_kws={"fontsize": 8},
                cbar=(i == 0),
                cbar_ax=None if i else divider,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        cbar = fig.colorbar(axs[-1, -1].collections[0], cax=divider, pad=0.02)
        cbar.set_ticks(np.linspace(vmin, vmax, 6))
        plt.tight_layout()

        if save:
            plt.savefig(self.DEFAULT_ANALYSIS_PATH / "pred_corr_matrix.pdf")
        plt.show()

    # -------------------------------------------------------------------------
    # Model Combination Analysis
    # -------------------------------------------------------------------------

    def analyze_tigge_combination(
        self,
        verbose: bool = True,
        plot: bool = False,
        save: bool = False,
    ) -> None:
        """Analyze optimal combination of GNN and TIGGE predictions.

        Args:
            verbose: Whether to print metrics for each alpha
            plot: Whether to plot alpha vs loss curve
            save: Whether to save the plot
        """
        y_trans = self.predictions["trans"][1:][1::2]
        y_tigge = self.predictions["tigge"]
        alphas = np.arange(0, 1.1, 0.1)

        if verbose:
            for alpha in alphas:
                logger.info(f"Alpha: {alpha}")
                self._evaluate_combination(y_trans, y_tigge, alpha=alpha)

        if plot:
            losses = np.array(
                [
                    self._evaluate_combination(y_trans, y_tigge, alpha=a, consolidate=True)
                    for a in alphas
                ]
            )

            plt.figure()
            plt.plot(alphas, losses, "-o")
            plt.title("Combined models")
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\tilde{\mathcal{L}}_{RMSE}$")

            if save:
                plt.savefig(self.DEFAULT_ANALYSIS_PATH / "alpha_loss.pdf")
            plt.show()

    def _evaluate_combination(
        self,
        y1: NDArray[np.floating[Any]],
        y2: NDArray[np.floating[Any]],
        alpha: float = 0.5,
        consolidate: bool = False,
    ) -> float | None:
        """Evaluate weighted combination of two predictions.

        Args:
            y1: First prediction tensor
            y2: Second prediction tensor
            alpha: Weight for y1 (y2 gets 1-alpha)
            consolidate: If True, return single normalized loss

        Returns:
            Consolidated loss if consolidate=True, else None
        """
        if self.era5 is None:
            raise ValueError("ERA5 data not loaded")

        y_combined = alpha * y1 + (1 - alpha) * y2

        if consolidate:
            self._ensure_scalers_loaded()
            if self.scalers is None:
                raise ValueError("Scalers not loaded")

            losses = np.zeros(len(self.feature_list))
            for i in range(len(self.feature_list)):
                y_norm = self.scalers[i].transform(y_combined[..., i].reshape(-1, 1))
                era5_norm = self.scalers[i].transform(self.era5[1::2][..., i].reshape(-1, 1))
                losses[i] = np.sqrt(mean_squared_error(y_norm, era5_norm))
            return float(np.mean(losses))

        self._calculate_metrics(y_combined, self.era5[1::2])
        return None

    # -------------------------------------------------------------------------
    # Distribution Analysis
    # -------------------------------------------------------------------------

    def plot_feature_distributions(
        self,
        use_histplot: bool = True,
        save: bool = False,
        stats: bool = True,
    ) -> pd.DataFrame | None:
        """Plot distribution of each feature in ERA5 data.

        Args:
            use_histplot: Use histplot (True) or distplot (False)
            save: Whether to save the figure
            stats: Whether to calculate and print statistics

        Returns:
            Statistics DataFrame if stats=True
        """
        if self.era5 is None:
            raise ValueError("ERA5 data not loaded")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        stats_df = None
        if stats:
            stats_df = pd.DataFrame(
                index=self.feature_list,
                columns=["Mean", "Standard Deviation", "Skewness", "Kurtosis"],
            )

        for i, (feature, ax) in enumerate(zip(self.feature_list, axes.flatten(), strict=False)):
            data = self.era5[..., i].flatten()

            if use_histplot:
                sns.histplot(data, kde=True, color="maroon", ax=ax)
            else:
                sns.distplot(data, kde=True, color="maroon", ax=ax)

            ax.set_title(feature)

            if stats and stats_df is not None:
                stats_df.loc[feature] = [
                    np.mean(data),
                    np.std(data),
                    pd.Series(data).skew(),
                    pd.Series(data).kurtosis(),
                ]

        if stats and stats_df is not None:
            print(stats_df.to_latex())

        plt.tight_layout()
        if save:
            plt.savefig(self.DEFAULT_ANALYSIS_PATH / "feature_dist.pdf")
        plt.show()

        return stats_df

    # -------------------------------------------------------------------------
    # Spatial Error Maps
    # -------------------------------------------------------------------------

    def generate_error_maps(self, save: bool = False) -> None:
        """Generate spatial error maps for all models.

        Args:
            save: Whether to save the figure.
        """
        self._calculate_avg_errors()

        num_models = len(self.avg_errors)
        num_features = len(self.feature_list)

        lat_span, lon_span, spatial_limits = DataProcessor.get_spatial_info()
        spatial = {
            "lat_span": lat_span,
            "lon_span": lon_span,
            "spatial_limits": spatial_limits,
        }

        fig, axes = plt.subplots(
            num_features,
            num_models,
            figsize=(15, 15),
            subplot_kw={"projection": ccrs.Mercator(central_longitude=40)},
        )

        for j, model in enumerate(self.avg_errors.keys()):
            # Add model title
            ax_title = fig.add_subplot(num_features, num_models, j + 1)
            ax_title.set_title(self._get_label(model), fontsize=12, y=1.05)
            ax_title.axis("off")

            for i, feature in enumerate(self.feature_list):
                error_map = self.avg_errors[model][..., i]
                title = rf"$(Y - \hat{{Y}})_{{{feature}}}$"
                axes[i, j].axis("off")
                draw_poland(
                    axes[i, j],
                    error_map,
                    title,
                    plt.cm.coolwarm,
                    norm=CenteredNorm(),
                    **spatial,
                )

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if save:
            plt.savefig(self.DEFAULT_ANALYSIS_PATH / "error_maps.pdf")
        plt.show()
