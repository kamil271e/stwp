"""Trainer for GNN weather prediction models."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from stwp.config import Config
from stwp.data.processor import DataProcessor
from stwp.features import Features
from stwp.models.gnn.callbacks import CkptCallback, EarlyStoppingCallback, LRAdjustCallback
from stwp.models.gnn.gnn_module import ArchitectureType, GNNModule
from stwp.models.gnn.processor import NNDataProcessor
from stwp.utils.encoding import trig_decode
from stwp.utils.visualization import draw_poland

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for GNN weather prediction models."""

    def __init__(
        self,
        architecture: ArchitectureType = ArchitectureType.TRANSFORMER,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.5,
        subset: int | None = None,
        spatial_mapping: bool = True,
        additional_encodings: bool = True,
        test_shuffle: bool = True,
    ):
        """Initialize the trainer.

        Args:
            architecture: GNN architecture type
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: LR decay factor
            subset: Optional subset size
            spatial_mapping: Whether to use spatial mapping
            additional_encodings: Whether to use additional encodings
            test_shuffle: Whether to shuffle test data
        """
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.feature_list: list[str] | None = None
        self.features: int | None = None
        self.constants: int | None = None
        self.edge_index: torch.Tensor | None = None
        self.edge_weights: torch.Tensor | None = None
        self.edge_attr: torch.Tensor | None = None
        self.scalers: list[Any] | None = None
        self.train_size: int | None = None
        self.val_size: int | None = None
        self.test_size: int | None = None
        self.spatial_mapping = spatial_mapping
        self.subset = subset
        self.latitude: int = 0
        self.longitude: int = 0

        self.cfg = Config
        self.nn_proc = NNDataProcessor(
            additional_encodings=additional_encodings, test_shuffle=test_shuffle
        )
        self.init_data_process()

        self.model: GNNModule | None = None
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.init_architecture()

        self.lr = lr
        self.gamma = gamma
        self.criterion = torch.nn.L1Loss()
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_callback: LRAdjustCallback | None = None
        self.ckpt_callback: CkptCallback | None = None
        self.early_stop_callback: EarlyStoppingCallback | None = None
        self.init_train_details()

    def update_config(self, config: Any) -> None:
        """Update configuration.

        Args:
            config: New configuration
        """
        self.cfg = config
        self.init_architecture()
        self.update_data_process()
        self.init_train_details()

    def init_data_process(self) -> None:
        """Initialize data processing."""
        self.nn_proc.preprocess(subset=self.subset)
        self.train_loader = self.nn_proc.train_loader
        self.val_loader = self.nn_proc.val_loader
        self.test_loader = self.nn_proc.test_loader
        self.feature_list = self.nn_proc.feature_list
        self.features = len(self.feature_list)
        (_, self.latitude, self.longitude, self.features) = self.nn_proc.get_shapes()
        self.constants = self.nn_proc.num_spatial_constants + self.nn_proc.num_temporal_constants
        self.edge_index = self.nn_proc.edge_index
        self.edge_weights = self.nn_proc.edge_weights
        self.edge_attr = self.nn_proc.edge_attr
        self.scalers = self.nn_proc.scalers
        if self.train_loader is None or self.val_loader is None or self.test_loader is None:
            raise ValueError("Data loaders not properly initialized")
        self.train_size = len(self.train_loader)
        self.val_size = len(self.val_loader)
        self.test_size = len(self.test_loader)
        self.spatial_mapping = self.spatial_mapping
        if self.subset is None:
            self.subset = self.train_size

    def update_data_process(self) -> None:
        """Update data processing with new configuration."""
        self.nn_proc.update(self.cfg)
        self.init_data_process()

    def init_architecture(self) -> None:
        """Initialize model architecture."""
        if self.edge_attr is None or self.features is None:
            raise ValueError("Data must be processed before initializing architecture")
        self.model = GNNModule(
            architecture=self.architecture,
            input_features=self.features,
            output_features=self.features,
            edge_dim=self.edge_attr.size(-1),
            hidden_dim=self.hidden_dim,
            input_t_dim=self.nn_proc.num_temporal_constants,
            input_s_dim=self.nn_proc.num_spatial_constants,
            input_size=self.cfg.input_size,
            fh=self.cfg.forecast_horizon,
            num_graph_cells=self.cfg.graph_cells,
        ).to(self.cfg.device)

    def init_train_details(self) -> None:
        """Initialize training details."""
        if self.model is None:
            raise ValueError("Model must be initialized before training details")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.lr_callback = LRAdjustCallback(self.optimizer, gamma=self.gamma)
        self.ckpt_callback = CkptCallback(self.model)
        self.early_stop_callback = EarlyStoppingCallback()

    def load_model(self, path: str) -> None:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file
        """
        if self.model is None:
            raise ValueError("Model must be initialized before loading")
        self.model.load_state_dict(torch.load(path))

    def train(self, num_epochs: int = 50, verbose: bool = False) -> None:
        """Train the model.

        Args:
            num_epochs: Number of training epochs
            verbose: Whether to print training progress
        """
        if (
            self.model is None
            or self.train_loader is None
            or self.val_loader is None
            or self.optimizer is None
            or self.subset is None
            or self.val_size is None
            or self.lr_callback is None
            or self.ckpt_callback is None
            or self.early_stop_callback is None
        ):
            raise ValueError("Trainer not properly initialized")
        start = time.time()

        val_loss_list = []
        train_loss_list = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                y_hat = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.time, batch.pos
                )
                batch_y = batch.y

                if self.spatial_mapping:
                    y_hat = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(y_hat))
                    batch_y = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(batch.y))

                loss = self.criterion(y_hat, batch_y)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / (self.subset * self.cfg.batch_size)
            train_loss_list.append(avg_loss)
            last_lr = self.optimizer.param_groups[0]["lr"]

            if verbose:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.5f}, lr: {last_lr}"
                )

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in self.val_loader:
                    y_hat = self.model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.time,
                        batch.pos,
                    )
                    batch_y = batch.y

                    if self.spatial_mapping:
                        y_hat = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(y_hat))
                        batch_y = self._ensure_numpy(
                            self.nn_proc.map_latitude_longitude_span(batch.y)
                        )

                    loss = self.criterion(y_hat, batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / (min(self.subset, self.val_size) * self.cfg.batch_size)
            val_loss_list.append(avg_val_loss)

            if verbose:
                logger.info(f"Val Loss: {avg_val_loss:.5f}")

            self.lr_callback.step(avg_val_loss)
            self.ckpt_callback.step(avg_val_loss)
            self.early_stop_callback.step(avg_val_loss)
            if self.early_stop_callback.early_stop:
                break

        end = time.time()
        if verbose:
            logger.info(f"Training completed in {end - start:.2f} seconds")
            self.plot_loss(val_loss_list, train_loss_list)

    @staticmethod
    def plot_loss(val_loss_list: list[float], train_loss_list: list[float]) -> None:
        """Plot training and validation loss.

        Args:
            val_loss_list: Validation losses
            train_loss_list: Training losses
        """
        x = np.arange(1, len(train_loss_list) + 1)
        plt.figure(figsize=(20, 7))
        plt.plot(x, train_loss_list, label="train loss")
        plt.plot(x, val_loss_list, label="val loss")
        plt.title("Loss plot")
        plt.legend()
        plt.show()

    def predict(
        self,
        X: torch.Tensor,
        y_tensor: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        inverse_norm: bool = True,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Make predictions.

        Args:
            X: Input features
            y_tensor: Target values
            edge_index: Edge indices
            edge_attr: Edge attributes
            s: Spatial features
            t: Temporal features
            inverse_norm: Whether to inverse normalize

        Returns:
            Tuple of (y, y_hat) arrays
        """
        if self.model is None or self.features is None or self.scalers is None:
            raise ValueError("Model not properly initialized")
        y_reshaped = y_tensor.reshape(
            (-1, self.latitude, self.longitude, self.features, self.cfg.forecast_horizon)
        )
        y_hat_tensor = self.model(X, edge_index, edge_attr, t, s).reshape(
            (-1, self.latitude, self.longitude, self.features, self.cfg.forecast_horizon)
        )

        y: NDArray[Any] = y_reshaped.cpu().detach().numpy()
        y_hat: NDArray[Any] = y_hat_tensor.cpu().detach().numpy()

        if inverse_norm:
            y_shape = (self.latitude, self.longitude, self.cfg.forecast_horizon)
            for i in range(self.features):
                for j in range(y_hat.shape[0]):
                    yi = y[j, ..., i, :].copy().reshape(-1, 1)
                    yhat_i = y_hat[j, ..., i, :].copy().reshape(-1, 1)

                    y[j, ..., i, :] = self.scalers[i].inverse_transform(yi).reshape(y_shape)
                    y_hat[j, ..., i, :] = self.scalers[i].inverse_transform(yhat_i).reshape(y_shape)
        if inverse_norm:
            y_hat = self.clip_total_cloud_cover(y_hat)
        return y, y_hat

    def plot_predictions(
        self,
        data_type: str = "test",
        pretty: bool = False,
        save: bool = False,
    ) -> None:
        """Plot predictions.

        Args:
            data_type: Type of data (train, test, val)
            pretty: Whether to use pretty map plots
            save: Whether to save figures
        """
        if self.train_loader is None or self.test_loader is None or self.val_loader is None:
            raise ValueError("Data loaders not initialized")
        if self.features is None or self.feature_list is None:
            raise ValueError("Features not initialized")
        if data_type == "train":
            sample = next(iter(self.train_loader))
        elif data_type == "test":
            sample = next(iter(self.test_loader))
        elif data_type == "val":
            sample = next(iter(self.val_loader))
        else:
            raise ValueError("Invalid type: (train, test, val)")

        spatial: dict[str, Any] = {}
        if pretty:
            lat_span, lon_span, spatial_limits = DataProcessor.get_spatial_info()
            spatial = {
                "lat_span": lat_span,
                "lon_span": lon_span,
                "spatial_limits": spatial_limits,
            }

        X, y_sample = sample.x, sample.y
        y, y_hat = self.predict(
            X, y_sample, sample.edge_index, sample.edge_attr, sample.pos, sample.time
        )
        latitude, longitude = self.latitude, self.longitude

        if self.spatial_mapping:
            y_hat = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(y_hat, flat=False))
            y = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(y, flat=False))
            latitude, longitude = y_hat.shape[1:3]

        for i in range(self.cfg.batch_size):
            if pretty:
                fig, axs = plt.subplots(
                    self.features,
                    3 * self.cfg.forecast_horizon,
                    figsize=(10 * self.cfg.forecast_horizon, 3 * self.features),
                    subplot_kw={"projection": ccrs.Mercator(central_longitude=40)},
                )
            else:
                fig, ax = plt.subplots(
                    self.features,
                    3 * self.cfg.forecast_horizon,
                    figsize=(10 * self.cfg.forecast_horizon, 3 * self.features),
                )

            for j, feature_name in enumerate(self.feature_list):
                for k in range(3 * self.cfg.forecast_horizon):
                    ts = k // 3
                    current_ax = axs[j, k] if pretty else ax[j, k]

                    if k % 3 == 0:
                        title = rf"$Y^{{t+{ts + 1}}}_{{{feature_name}}}$"
                        value = y[i, ..., j, ts]
                        cmap = plt.cm.coolwarm  # type: ignore[attr-defined]
                    elif k % 3 == 1:
                        title = rf"$\hat{{Y}}^{{t+{ts + 1}}}_{{{feature_name}}}$"
                        value = y_hat[i, ..., j, ts]
                        cmap = plt.cm.coolwarm  # type: ignore[attr-defined]
                    else:
                        title = rf"$|Y - \hat{{Y}}|^{{t+{ts + 1}}}_{{{feature_name}}}$"
                        value = np.abs(y[i, ..., j, ts] - y_hat[i, ..., j, ts])
                        cmap = "binary"

                    if pretty:
                        draw_poland(current_ax, value, title, cmap, **spatial)
                    else:
                        pl = current_ax.imshow(value.reshape(latitude, longitude), cmap=cmap)
                        current_ax.set_title(title)
                        current_ax.axis("off")
                        fig.colorbar(pl, ax=current_ax, fraction=0.15)

        plt.tight_layout()
        if save:
            plt.savefig(f"../data/analysis/{self.architecture}_{data_type}.pdf")
        self.calculate_metrics(y_hat, y)

    def plot_error_heatmap(self, data_type: str = "test", pretty: bool = False) -> None:
        """Plot error heatmap.

        Args:
            data_type: Type of data (train, test, val)
            pretty: Whether to use pretty map plots
        """
        if self.train_loader is None or self.test_loader is None or self.val_loader is None:
            raise ValueError("Data loaders not initialized")
        if self.features is None or self.feature_list is None:
            raise ValueError("Features not initialized")
        if data_type == "train":
            sample = next(iter(self.train_loader))
        elif data_type == "test":
            sample = next(iter(self.test_loader))
        elif data_type == "val":
            sample = next(iter(self.val_loader))
        else:
            raise ValueError("Invalid type: (train, test, val)")

        spatial: dict[str, Any] = {}
        if pretty:
            lat_span, lon_span, spatial_limits = DataProcessor.get_spatial_info()
            spatial = {
                "lat_span": lat_span,
                "lon_span": lon_span,
                "spatial_limits": spatial_limits,
            }

        X, y_sample = sample.x, sample.y
        y, y_hat = self.predict(
            X, y_sample, sample.edge_index, sample.edge_attr, sample.pos, sample.time
        )
        latitude, longitude = self.latitude, self.longitude

        if self.spatial_mapping:
            y_hat = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(y_hat, flat=False))
            y = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(y, flat=False))
            latitude, longitude = y_hat.shape[1:3]

        for i in range(self.cfg.batch_size):
            if pretty:
                fig, axs = plt.subplots(
                    self.features,
                    self.cfg.forecast_horizon,
                    figsize=(10 * self.cfg.forecast_horizon, self.features),
                    subplot_kw={"projection": ccrs.Mercator(central_longitude=40)},
                )
            else:
                fig, ax = plt.subplots(
                    self.features,
                    self.cfg.forecast_horizon,
                    figsize=(10 * self.cfg.forecast_horizon, self.features),
                )

            for j, feature_name in enumerate(self.feature_list):
                for k in range(self.cfg.forecast_horizon):
                    ts = k
                    current_ax = axs[j] if pretty else ax[j]

                    title = rf"$|X - \hat{{X}}|_{{{feature_name},t+{ts + 1}}}$"
                    value = np.abs(y[i, ..., j, ts] - y_hat[i, ..., j, ts])
                    cmap = "binary"

                    if pretty:
                        draw_poland(current_ax, value, title, cmap, **spatial)
                    else:
                        pl = current_ax.imshow(value.reshape(latitude, longitude), cmap=cmap)
                        current_ax.set_title(title)
                        current_ax.axis("off")
                        fig.colorbar(pl, ax=current_ax, fraction=0.15)

    def evaluate(
        self,
        data_type: str = "test",
        verbose: bool = True,
        inverse_norm: bool = True,
        begin: int | None = None,
        end: int | None = None,
    ) -> tuple[tuple[list[float], list[float]] | None, NDArray[Any]]:
        """Evaluate model on data.

        Args:
            data_type: Type of data (train, test, val)
            verbose: Whether to print metrics
            inverse_norm: Whether to inverse normalize
            begin: Start day of year filter
            end: End day of year filter

        Returns:
            Tuple of (metrics, y_hat)
        """
        if self.train_loader is None or self.test_loader is None or self.val_loader is None:
            raise ValueError("Data loaders not initialized")
        if self.features is None:
            raise ValueError("Features not initialized")
        if data_type == "train":
            loader = self.train_loader
        elif data_type == "test":
            loader = self.test_loader
        elif data_type == "val":
            loader = self.val_loader
        else:
            raise ValueError("Invalid type: (train, test, val)")

        y = np.empty((0, self.latitude, self.longitude, self.features, self.cfg.forecast_horizon))
        y_hat = np.empty(
            (0, self.latitude, self.longitude, self.features, self.cfg.forecast_horizon)
        )
        for batch in loader:
            if begin is not None and end is not None:
                v_sin = batch.time[0].item()
                v_cos = batch.time[1].item()
                ts = trig_decode(v_sin, v_cos, 366)
                if begin > ts or end < ts:
                    continue
            y_i, y_hat_i = self.predict(
                batch.x,
                batch.y,
                batch.edge_index,
                batch.edge_attr,
                batch.pos,
                batch.time,
                inverse_norm=inverse_norm,
            )
            y = np.concatenate((y, y_i), axis=0)
            y_hat = np.concatenate((y_hat, y_hat_i), axis=0)

        if self.spatial_mapping:
            y_hat = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(y_hat, flat=False))
            y = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(y, flat=False))
        try:
            return self.calculate_metrics(y_hat, y, verbose=verbose), y_hat
        except ValueError as e:
            logger.exception(f"Error calculating metrics: {e}")
            return None, y_hat

    def autoreg_evaluate(
        self,
        data_type: str = "test",
        fh: int = 2,
        verbose: bool = True,
        inverse_norm: bool = True,
    ) -> tuple[tuple[list[float], list[float]], NDArray[Any]]:
        """Evaluate with autoregressive prediction.

        Args:
            data_type: Type of data (train, test, val)
            fh: Forecast horizon
            verbose: Whether to print metrics
            inverse_norm: Whether to inverse normalize

        Returns:
            Tuple of (metrics, y_hat)
        """
        self.cfg.batch_size = 1
        self.cfg.forecast_horizon = fh
        self.update_data_process()
        self.cfg.forecast_horizon = 1

        if self.train_loader is None or self.test_loader is None or self.val_loader is None:
            raise ValueError("Data loaders not initialized")
        if self.features is None:
            raise ValueError("Features not initialized")
        if data_type == "train":
            loader = self.train_loader
        elif data_type == "test":
            loader = self.test_loader
        elif data_type == "val":
            loader = self.val_loader
        else:
            raise ValueError("Invalid type: (train, test, val)")

        y_accum = torch.empty((0, self.latitude, self.longitude, self.features, fh)).to(
            self.cfg.device
        )
        y_hat_accum = torch.empty((0, self.latitude, self.longitude, self.features, fh)).to(
            self.cfg.device
        )
        y_shape = (self.latitude * self.longitude, self.features, 1)

        for batch in loader:
            y_hat_autoreg_i = torch.zeros_like(batch.y)
            y_i = torch.zeros_like(batch.y)
            for t in range(fh):
                input_batch = batch.clone()
                input_batch.y = input_batch.y[..., t : t + 1]
                if t == 0:
                    y_it, y_hat_it = self.predict(
                        input_batch.x,
                        input_batch.y,
                        input_batch.edge_index,
                        input_batch.edge_attr,
                        input_batch.pos,
                        input_batch.time,
                        inverse_norm=inverse_norm,
                    )
                else:
                    input_batch.x = torch.cat(
                        (input_batch.x[..., :-t], y_hat_autoreg_i[..., :t]), dim=-1
                    )
                    y_it, y_hat_it = self.predict(
                        input_batch.x,
                        input_batch.y,
                        input_batch.edge_index,
                        input_batch.edge_attr,
                        input_batch.pos,
                        input_batch.time,
                        inverse_norm=inverse_norm,
                    )
                y_hat_i_tensor = torch.from_numpy(y_hat_it).to(self.cfg.device)
                y_it_tensor = torch.from_numpy(y_it).to(self.cfg.device)
                y_hat_autoreg_i[..., t : t + 1] = y_hat_i_tensor.reshape(y_shape)
                y_i[..., t : t + 1] = y_it_tensor.reshape(y_shape)

            y_accum = torch.cat(
                (y_accum, y_i.reshape(1, self.latitude, self.longitude, self.features, fh)),
                dim=0,
            )
            y_hat_accum = torch.cat(
                (
                    y_hat_accum,
                    y_hat_autoreg_i.reshape(1, self.latitude, self.longitude, self.features, fh),
                ),
                dim=0,
            )

        y_hat: NDArray[Any] = y_hat_accum.cpu().detach().numpy()
        y: NDArray[Any] = y_accum.cpu().detach().numpy()

        if self.spatial_mapping:
            y_hat = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(y_hat, flat=False))
            y = self._ensure_numpy(self.nn_proc.map_latitude_longitude_span(y, flat=False))

        self.cfg.forecast_horizon = 1
        return self.calculate_metrics(y_hat, y, verbose=verbose), y_hat

    def calculate_metrics(
        self,
        y_hat: NDArray[Any],
        y: NDArray[Any],
        verbose: bool = False,
    ) -> tuple[list[float], list[float]]:
        """Calculate RMSE and MAE metrics.

        Args:
            y_hat: Predictions
            y: Ground truth
            verbose: Whether to print metrics

        Returns:
            Tuple of (rmse_features, mae_features)
        """
        if self.feature_list is None:
            raise ValueError("Feature list not initialized")
        rmse_features = []
        mae_features = []
        for i, feature_name in enumerate(self.feature_list):
            y_fi = y[..., i, :].reshape(-1, 1)
            y_hat_fi = y_hat[..., i, :].reshape(-1, 1)
            rmse = np.sqrt(mean_squared_error(y_hat_fi, y_fi))
            mae = mean_absolute_error(y_hat_fi, y_fi)
            if verbose:
                logger.info(
                    f"RMSE for {feature_name}: {rmse:.4f}; MAE for {feature_name}: {mae:.4f}"
                )
            rmse_features.append(rmse)
            mae_features.append(mae)
        return rmse_features, mae_features

    def save_prediction_tensor(
        self,
        y_hat: NDArray[Any] | torch.Tensor,
        path: str | None = None,
    ) -> None:
        """Save predictions to file.

        Args:
            y_hat: Predictions
            path: Output path
        """
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.cpu().detach().numpy()
        elif not isinstance(y_hat, np.ndarray):
            raise ValueError("Input y_hat should be either a PyTorch Tensor or a NumPy array.")
        if path is None:
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            path = f"../data/pred/{self.architecture}_{t}.npy"
        np.save(path, y_hat)

    def calculate_model_params(self) -> None:
        """Print number of model parameters."""
        if self.model is None:
            raise ValueError("Model not initialized")
        params = 0
        for p in self.model.parameters():
            params += p.reshape(-1).shape[0]
        logger.info(f"Model parameters: {params}")

    @staticmethod
    def clip_total_cloud_cover(y_hat: NDArray[Any], idx: int = Features.TCC_IDX) -> NDArray[Any]:
        """Clip total cloud cover to valid range.

        Args:
            y_hat: Predictions
            idx: Index of TCC feature

        Returns:
            Clipped predictions
        """
        y_hat[..., idx, :] = np.clip(y_hat[..., idx, :], 0, 1)
        return y_hat

    def get_model(self) -> GNNModule:
        """Get the model.

        Returns:
            GNN model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        return self.model

    def predict_to_json(
        self,
        X: Any | None = None,
        path: str | None = None,
        which_sequence: int = 0,
    ) -> dict[str, Any]:
        """Export predictions to JSON.

        Args:
            X: Input data (uses test sample if None)
            path: Output path (if None, doesn't save to file)
            which_sequence: Which sequence to use from test loader (0-3 for 6h intervals)

        Returns:
            Dictionary with predictions keyed by lat/lon/feature/timestamp
        """
        if self.test_loader is None or self.features is None or self.feature_list is None:
            raise ValueError("Trainer not properly initialized")
        if X is None:
            for i, data in enumerate(self.test_loader):
                if i == which_sequence:
                    X = data
                    break

        if X is None:
            raise ValueError("No data found in test loader")

        _, y_hat = self.predict(X.x, X.y, X.edge_index, X.edge_attr, X.pos, X.time)
        y_hat = y_hat.reshape((self.latitude, self.longitude, self.features, -1))
        lat_span, lon_span, _ = DataProcessor.get_spatial_info()
        lat_span_list = list(lat_span[:, 0])
        lon_span_list = list(lon_span[0, :])

        json_data: dict[str, Any] = {}

        # Decode prediction time
        prediction_day = trig_decode(X.time[0].item(), X.time[1].item(), 365)
        prediction_hour = trig_decode(X.time[2].item(), X.time[3].item(), 24)
        prediction_date = datetime(year=2024, month=1, day=1, hour=prediction_hour) + timedelta(
            days=prediction_day
        )

        for i, lat in enumerate(lat_span_list):
            lat_key = str(lat)
            json_data[lat_key] = {}
            for j, lon in enumerate(lon_span_list):
                lon_key = str(lon)
                json_data[lat_key][lon_key] = {}
                for k, feature in enumerate(self.feature_list):
                    json_data[lat_key][lon_key][feature] = {}
                    for ts in range(y_hat.shape[-1]):
                        t = prediction_date + timedelta(hours=6 * (ts + 1))
                        t_key = t.strftime("%Y-%m-%dT%H:%M:%S")
                        json_data[lat_key][lon_key][feature][t_key] = float(y_hat[i, j, k, ts])

        if path is not None:
            with open(path, "w") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

        return json_data

    def _ensure_numpy(self, data: NDArray[Any] | torch.Tensor) -> NDArray[Any]:
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        return data
