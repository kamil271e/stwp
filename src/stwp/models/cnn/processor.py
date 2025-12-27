"""CNN data processor."""

from typing import Any

import numpy as np

from stwp.config import Config
from stwp.models.gnn.processor import NNDataProcessor


class CNNDataProcessor(NNDataProcessor):
    """Data processor for CNN models."""

    def __init__(
        self,
        spatial_encoding: bool = False,
        temporal_encoding: bool = False,
        additional_encodings: bool = False,
        test_shuffle: bool = True,
    ) -> None:
        """Initialize the processor.

        Args:
            spatial_encoding: Whether to add spatial encodings
            temporal_encoding: Whether to add temporal encodings
            additional_encodings: Add both spatial and temporal encodings
            test_shuffle: Whether to shuffle test data
        """
        super().__init__(
            spatial_encoding=spatial_encoding,
            temporal_encoding=temporal_encoding,
            additional_encodings=additional_encodings,
            test_shuffle=test_shuffle,
        )

    def preprocess(self, subset: int | None = None) -> None:
        """Preprocess data for CNN training.

        Args:
            subset: Optional subset size
        """
        X_train, X_test, y_train, y_test = self.train_val_test_split()
        X, y = self.fit_transform_scalers(
            X_train, X_test, y_train, y_test, scaler_type=self.cfg.scalar_type
        )
        X = X.transpose((0, 1, 3, 2))
        y = y.transpose((0, 1, 3, 2))
        X = X.reshape(
            -1,
            self.num_latitudes,
            self.num_longitudes,
            self.cfg.input_size,
            self.num_features,
        )
        y = y.reshape(
            -1, self.num_latitudes, self.num_longitudes, Config.forecast_horizon, self.num_features
        )
        self.train_loader, self.val_loader, self.test_loader = self.get_loaders(
            X, y, subset, test_shuffle=self.test_shuffle
        )

    def map_latitude_longitude_span(
        self,
        input_tensor: np.ndarray | Any,
        old_span: tuple[int, int] = (32, 48),
        new_span: tuple[int, int] = (25, 45),
        flat: bool = False,
    ) -> np.ndarray | Any:
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
            input_tensor = input_tensor.reshape(
                (
                    self.cfg.batch_size,
                    self.num_features,
                    self.num_latitudes,
                    self.num_longitudes,
                    1,
                )
            )

        old_lat, old_lon = old_span
        new_lat, new_lon = new_span

        lat_diff = old_lat - new_lat
        left_lat = lat_diff // 2
        right_lat = new_lat + lat_diff - left_lat - 1

        lon_diff = old_lon - new_lon

        if len(input_tensor.shape) == 4:
            mapped_tensor = input_tensor[:, :, left_lat:right_lat, lon_diff:]
        else:
            mapped_tensor = input_tensor[:, left_lat:right_lat, lon_diff:, ...]

        return mapped_tensor
