"""Neural network data processor for GNN models."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch_geometric.data as data
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import shuffle
from torch_geometric.loader import DataLoader

from stwp.config import Config
from stwp.data.processor import DataProcessor

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NNDataProcessor:
    """Data processor for neural network models."""

    def __init__(
        self,
        spatial_encoding: bool = False,
        temporal_encoding: bool = False,
        additional_encodings: bool = False,
        test_shuffle: bool = True,
        path: str | None = None,
    ):
        """Initialize the processor.

        Args:
            spatial_encoding: Whether to add spatial encodings
            temporal_encoding: Whether to add temporal encodings
            additional_encodings: Add both spatial and temporal encodings
            test_shuffle: Whether to shuffle test data
            path: Path to GRIB data file
        """
        data_path = path if path is not None else Config.data_path

        self.data_proc = DataProcessor(
            path=data_path,
            spatial_encoding=spatial_encoding,
            temporal_encoding=temporal_encoding,
            additional_encodings=additional_encodings,
        )
        self.raw_data = self.data_proc.data
        self.temporal_data = self.data_proc.temporal_data
        self.spatial_data = self.data_proc.spatial_data
        self.feature_list = self.data_proc.feature_list
        (
            self.num_samples,
            self.num_latitudes,
            self.num_longitudes,
            self.num_features,
        ) = self.raw_data.shape

        self.spatial_encoding = spatial_encoding or additional_encodings
        self.temporal_encoding = temporal_encoding or additional_encodings
        self.num_spatial_constants = self.data_proc.num_spatial_constants
        self.num_temporal_constants = self.data_proc.num_temporal_constants

        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.test_shuffle = test_shuffle

        self.train_size: int | None = None
        self.val_size: int | None = None
        self.test_size: int | None = None
        self.scaler: Any = None
        self.scalers: list[Any] | None = None
        self.edge_weights: torch.Tensor | None = None
        self.edge_index: torch.Tensor | None = None
        self.edge_attr: torch.Tensor | None = None
        self.cfg = Config

    def update(self, c: Any) -> None:
        """Update configuration.

        Args:
            c: New configuration
        """
        self.data_proc.upload_data(self.raw_data)
        self.cfg = c

    def preprocess(self, subset: int | None = None) -> None:
        """Preprocess data for training.

        Args:
            subset: Optional subset size
        """
        self.edge_index, self.edge_weights, self.edge_attr = self.create_edges()
        X_train, X_test, y_train, y_test = self.train_val_test_split()
        X, y = self.fit_transform_scalers(
            X_train, X_test, y_train, y_test, scaler_type=self.cfg.scalar_type
        )
        self.train_loader, self.val_loader, self.test_loader = self.get_loaders(
            X, y, subset, test_shuffle=self.test_shuffle
        )

    def train_val_test_split(
        self,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Split data into train, validation and test sets.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X, y = self.data_proc.preprocess(self.cfg.input_size, self.cfg.forecast_horizon)

        self.num_samples = X.shape[0]
        self.train_size = int(self.num_samples * self.cfg.train_ratio)
        self.val_size = self.train_size
        self.test_size = self.num_samples - self.train_size - self.val_size

        X = X.reshape(
            -1,
            self.num_latitudes * self.num_longitudes * self.cfg.input_size,
            self.num_features,
        )
        y = y.reshape(
            -1,
            self.num_latitudes * self.num_longitudes * self.cfg.forecast_horizon,
            self.num_features,
        )

        return self.data_proc.train_val_test_split(X, y, split_type=3)

    def fit_transform_scalers(
        self,
        X_train: NDArray[Any],
        X_test: NDArray[Any],
        y_train: NDArray[Any],
        y_test: NDArray[Any],
        scaler_type: str = "standard",
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Fit and transform scalers on data.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            scaler_type: Type of scaler to use

        Returns:
            Tuple of (X, y) scaled arrays
        """
        scaler_map = {
            "min_max": MinMaxScaler,
            "standard": StandardScaler,
            "max_abs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        if scaler_type not in scaler_map:
            raise ValueError(f"{scaler_type} scaler not implemented")

        self.scaler = scaler_map[scaler_type]()
        self.scalers = [copy.deepcopy(self.scaler) for _ in range(self.num_features)]

        Xi_shape = self.num_latitudes * self.num_longitudes * self.cfg.input_size
        yi_shape = self.num_latitudes * self.num_longitudes * self.cfg.forecast_horizon

        for i in range(self.num_features):
            X_train_i = X_train[..., i].reshape(-1, 1)
            X_test_i = X_test[..., i].reshape(-1, 1)
            y_train_i = y_train[..., i].reshape(-1, 1)
            y_test_i = y_test[..., i].reshape(-1, 1)

            self.scalers[i].fit(X_train_i)
            X_train[..., i] = (
                self.scalers[i].transform(X_train_i).reshape((self.train_size, Xi_shape))
            )
            X_test[..., i] = (
                self.scalers[i]
                .transform(X_test_i)
                .reshape((self.test_size + self.val_size, Xi_shape))
            )
            y_train[..., i] = (
                self.scalers[i].transform(y_train_i).reshape((self.train_size, yi_shape))
            )
            y_test[..., i] = (
                self.scalers[i]
                .transform(y_test_i)
                .reshape((self.test_size + self.val_size, yi_shape))
            )

        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        X = X.reshape(
            -1,
            self.num_latitudes * self.num_longitudes,
            self.cfg.input_size,
            self.num_features,
        )
        y = y.reshape(
            -1,
            self.num_latitudes * self.num_longitudes,
            self.cfg.forecast_horizon,
            self.num_features,
        )
        X = X.transpose((0, 1, 3, 2))
        y = y.transpose((0, 1, 3, 2))

        return X, y

    def inverse_transform_scalers(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Inverse transform scaled data.

        Args:
            X: Scaled features
            y: Scaled targets

        Returns:
            Tuple of (X, y) original scale arrays
        """
        X = X.transpose((0, 1, 3, 2))
        y = y.transpose((0, 1, 3, 2))
        for i in range(self.num_features):
            X_i = X[..., i].reshape(-1, 1)
            y_i = y[..., i].reshape(-1, 1)
            X[..., i] = (
                self.scalers[i]
                .inverse_transform(X_i)
                .reshape(
                    (
                        X.shape[0],
                        self.num_latitudes * self.num_longitudes,
                        self.cfg.input_size,
                    )
                )
            )
            y[..., i] = (
                self.scalers[i]
                .inverse_transform(y_i)
                .reshape(
                    (
                        y.shape[0],
                        self.num_latitudes * self.num_longitudes,
                        self.cfg.forecast_horizon,
                    )
                )
            )
        X = X.reshape(
            -1,
            self.num_latitudes * self.num_longitudes,
            self.cfg.input_size,
            self.num_features,
        )
        y = y.reshape(
            -1,
            self.num_latitudes * self.num_longitudes,
            self.cfg.forecast_horizon,
            self.num_features,
        )
        X = X.transpose((0, 1, 3, 2))
        y = y.transpose((0, 1, 3, 2))
        return X, y

    def create_edges(
        self,
        r: int | None = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """Create graph edges.

        Args:
            r: Neighbourhood radius

        Returns:
            Tuple of (edge_index, edge_weights, edge_attr)
        """
        if r is None:
            r = self.cfg.r

        def node_index(i: int, j: int, num_cols: int) -> int:
            return i * num_cols + j

        u = 0.5  # edge aggregation unit
        edge_index = []
        edge_attr = []
        _, indices = DataProcessor.count_neighbours(radius=r)

        for la in range(self.num_latitudes):
            for lo in range(self.num_longitudes):
                for i, j in indices:
                    if -1 < la + i < self.num_latitudes and -1 < lo + j < self.num_longitudes:
                        edge_index.append(
                            [
                                node_index(la, lo, self.num_longitudes),
                                node_index(la + i, lo + j, self.num_longitudes),
                            ]
                        )
                        edge_attr.append([u * i, u * j, np.sqrt((u * i) ** 2 + (u * j) ** 2)])

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.int64).t().to(self.cfg.device)
        edge_weights = None
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32).to(self.cfg.device)

        return edge_index_tensor, edge_weights, edge_attr_tensor

    def get_loaders(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        subset: int | None = None,
        test_shuffle: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders.

        Args:
            X: Features
            y: Targets
            subset: Optional subset size
            test_shuffle: Whether to shuffle test data

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        dataset = []
        for i in range(X.shape[0]):
            Xi = torch.from_numpy(X[i].astype("float32")).to(self.cfg.device)
            yi = torch.from_numpy(y[i].astype("float32")).to(self.cfg.device)
            if self.temporal_encoding:
                ti = torch.from_numpy(self.temporal_data[i].astype("float32")).to(self.cfg.device)
            else:
                ti = None
            if self.spatial_encoding:
                si = torch.from_numpy(self.spatial_data.astype("float32")).to(self.cfg.device)
            else:
                si = None
            g = data.Data(
                x=Xi,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=yi,
                pos=si,
                time=ti,
            )
            g = g.to(self.cfg.device)
            dataset.append(g)

        train_dataset = dataset[: self.train_size]
        val_dataset = dataset[self.train_size : self.train_size + self.val_size]
        test_dataset = dataset[-self.test_size :]

        train_dataset = shuffle(train_dataset, random_state=self.cfg.random_state)
        val_dataset = shuffle(val_dataset, random_state=self.cfg.random_state)
        if test_shuffle:
            test_dataset = shuffle(test_dataset, random_state=self.cfg.random_state)

        if subset is not None:
            train_dataset = train_dataset[: subset * self.cfg.batch_size]
            val_dataset = val_dataset[: subset * self.cfg.batch_size]
            test_dataset = test_dataset[: subset * self.cfg.batch_size]

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size)
        return train_loader, val_loader, test_loader

    def get_shapes(self) -> tuple[int, int, int, int]:
        """Get data shapes.

        Returns:
            Tuple of (num_samples, num_latitudes, num_longitudes, num_features)
        """
        return (
            self.num_samples,
            self.num_latitudes,
            self.num_longitudes,
            self.num_features,
        )

    def map_latitude_longitude_span(
        self,
        input_tensor: NDArray[Any] | torch.Tensor,
        old_span: tuple[int, int] = (32, 48),
        new_span: tuple[int, int] = (25, 45),
        flat: bool = True,
    ) -> NDArray[Any] | torch.Tensor:
        """Map latitude-longitude span.

        Args:
            input_tensor: Input tensor
            old_span: Original span (lat, lon)
            new_span: Target span (lat, lon)
            flat: Whether input is flattened

        Returns:
            Mapped tensor
        """
        if flat:
            batch_size = int(input_tensor.shape[0] / self.num_latitudes / self.num_longitudes)
            input_tensor = input_tensor.reshape(
                (
                    batch_size,
                    self.num_latitudes,
                    self.num_longitudes,
                    self.num_features,
                    input_tensor.shape[-1],
                )
            )

        old_lat, old_lon = old_span
        new_lat, new_lon = new_span

        lat_diff = old_lat - new_lat
        left_lat = lat_diff // 2
        right_lat = new_lat + lat_diff - left_lat - 1

        lon_diff = old_lon - new_lon

        mapped_tensor = input_tensor[:, left_lat:right_lat, lon_diff:]
        mapped_tensor.reshape(-1, self.num_features, mapped_tensor.shape[4])

        return mapped_tensor
