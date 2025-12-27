"""Data processor for weather data loading and preprocessing."""

import copy
from datetime import datetime
from typing import Any

import cfgrib
import numpy as np
from sklearn.utils import shuffle

from stwp.config import Config as Config
from stwp.data.download import SMALL_AREA
from stwp.features import Features
from stwp.utils.encoding import TrigFunc, datetime64_to_datetime, get_day_of_year, trig_encode


class DataProcessor:
    """Process GRIB weather data for ML models."""

    def __init__(
        self,
        spatial_encoding: bool = False,
        temporal_encoding: bool = False,
        additional_encodings: bool = False,
        path: str | None = None,
    ):
        """Initialize the data processor.

        Args:
            spatial_encoding: Whether to add spatial encodings
            temporal_encoding: Whether to add temporal encodings
            additional_encodings: Add both spatial and temporal encodings
            path: Path to GRIB data file
        """
        self.num_spatial_constants, self.num_temporal_constants = 0, 0
        self.temporal_encoding = temporal_encoding or additional_encodings
        self.spatial_encoding = spatial_encoding or additional_encodings

        data_path = path if path is not None else Config.data_path

        (
            self.data,
            self.feature_list,
            self.temporal_data,
            self.spatial_data,
        ) = self.load_data(
            path=data_path,
            spatial_encoding=self.spatial_encoding,
            temporal_encoding=self.temporal_encoding,
        )
        self.raw_data = copy.deepcopy(self.data)
        self.raw_temporal_data = copy.deepcopy(self.temporal_data)
        self.samples, self.latitude, self.longitude, self.num_features = self.data.shape
        self.neighbours: int | None = None
        self.sequence_length: int | None = None

    def upload_data(self, data: np.ndarray) -> None:
        """Upload custom data array.

        Args:
            data: Data array to use
        """
        self.data = data

    def create_autoregressive_sequences(
        self,
        input_size: int | None = None,
        fh: int | None = None,
    ) -> None:
        """Create autoregressive sequences for time series forecasting.

        Args:
            input_size: Number of input timesteps
            fh: Forecast horizon
        """
        input_size = input_size if input_size is not None else Config.input_size
        fh = fh if fh is not None else Config.forecast_horizon

        self.sequence_length = input_size + fh
        sequences = np.empty(
            (
                self.samples - self.sequence_length + 1,
                self.sequence_length,
                self.latitude,
                self.longitude,
                self.num_features,
            )
        )
        for i in range(self.samples - self.sequence_length + 1):
            sequences[i] = self.raw_data[i : i + self.sequence_length]
        sequences = sequences.transpose((0, 2, 3, 1, 4))
        self.data = sequences

        if self.temporal_encoding and self.raw_temporal_data is not None:
            time_sequences = np.empty(
                (
                    self.samples - self.sequence_length + 1,
                    self.num_temporal_constants,
                )
            )
            for i in range(self.samples - self.sequence_length + 1):
                time_sequences[i] = self.raw_temporal_data[i + input_size]
            self.temporal_data = time_sequences

        if self.spatial_encoding and self.spatial_data is not None:
            spatial_sequences = np.empty(
                (
                    self.samples - self.sequence_length + 1,
                    self.latitude * self.longitude,
                    self.num_spatial_constants,
                )
            )
            for i in range(self.samples - self.sequence_length + 1):
                spatial_sequences[i] = self.spatial_data[0]
            self.spatial_data = spatial_sequences

    def create_neighbours(self, radius: int) -> None:
        """Create neighbour feature tensors.

        Args:
            radius: Neighbourhood radius
        """
        if self.sequence_length is None:
            raise ValueError("sequence_length must be set before creating neighbours")
        self.neighbours, indices = self.count_neighbours(radius=radius)
        neigh_data = np.empty(
            (
                self.samples,
                self.latitude,
                self.longitude,
                self.neighbours + 1,
                self.sequence_length,
                self.num_features + self.num_spatial_constants + self.num_temporal_constants,
            )
        )
        neigh_data[..., 0, :, :] = self.data

        for n in range(1, self.neighbours + 1):
            i, j = indices[n - 1]
            for s in range(self.samples):
                for la in range(self.latitude):
                    for lo in range(self.longitude):
                        if -1 < la + i < self.latitude and -1 < lo + j < self.longitude:
                            neigh_data[s, la, lo, n] = self.data[s, la + i, lo + j]
                        else:
                            neigh_data[s, la, lo, n] = self.data[s, la, lo]

        self.data = neigh_data

    def preprocess(
        self,
        input_size: int | None = None,
        fh: int | None = None,
        r: int | None = None,
        use_neighbours: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess data for training.

        Args:
            input_size: Number of input timesteps
            fh: Forecast horizon
            r: Neighbourhood radius
            use_neighbours: Whether to use neighbour features

        Returns:
            Tuple of (X, y) arrays
        """
        input_size = input_size if input_size is not None else Config.input_size
        fh = fh if fh is not None else Config.forecast_horizon
        r = r if r is not None else Config.r

        self.create_autoregressive_sequences(input_size, fh)
        if use_neighbours:
            self.create_neighbours(radius=r)
            y = self.data[..., 0, -fh:, : self.num_features]
        else:
            y = self.data[..., -fh:, : self.num_features]
        X = self.data[..., :input_size, :]
        return X, y

    def load_data(
        self,
        path: str | None = None,
        spatial_encoding: bool = False,
        temporal_encoding: bool = False,
    ) -> tuple[np.ndarray, list[str], np.ndarray | None, np.ndarray | None]:
        """Load data from GRIB file.

        Args:
            path: Path to GRIB file
            spatial_encoding: Whether to add spatial encodings
            temporal_encoding: Whether to add temporal encodings

        Returns:
            Tuple of (data, feature_list, temporal_data, spatial_data)
        """
        data_path = path if path is not None else Config.data_path
        grib_data = cfgrib.open_datasets(data_path)
        surface = grib_data[0]
        hybrid = grib_data[1]

        t2m = surface.t2m.to_numpy() - 273.15  # K -> C
        sp = surface.sp.to_numpy() / 100  # Pa -> hPa
        tcc = surface.tcc.to_numpy()
        u10 = surface.u10.to_numpy()
        v10 = surface.v10.to_numpy()
        tp = hybrid.tp.to_numpy() * 1000  # m -> mm
        if tp.ndim >= 4:
            tp = tp.reshape((-1,) + hybrid.tp.shape[2:])

        data = np.stack((t2m, sp, tcc, u10, v10, tp), axis=-1)
        feature_list = Features.as_list()

        spatial_data: np.ndarray | None = None
        if spatial_encoding:
            lsm = surface.lsm.to_numpy()
            z = surface.z.to_numpy()
            z = (z - z.mean()) / z.std()
            stacked = np.stack([lsm, z], axis=-1)

            spatial_encodings = np.empty(data.shape[1:-1] + (4,))

            latitudes = np.array(surface.latitude)
            longitudes = np.array(surface.longitude)
            for i, lat in enumerate(latitudes):
                for j, lon in enumerate(longitudes):
                    for idx, v in enumerate(
                        [
                            trig_encode(lat, 180, TrigFunc.SIN),
                            trig_encode(lat, 180, TrigFunc.COS),
                            trig_encode(lon, 360, TrigFunc.SIN),
                            trig_encode(lon, 360, TrigFunc.COS),
                        ]
                    ):
                        spatial_encodings[i, j, idx] = v

            spatial_data = np.concatenate([stacked[0], spatial_encodings], axis=-1)
            self.num_spatial_constants = spatial_data.shape[-1]
            spatial_data = spatial_data.reshape(
                (1, len(latitudes) * len(longitudes), self.num_spatial_constants)
            )

        temporal_data: np.ndarray | None = None
        if temporal_encoding:
            dt = surface.time.to_numpy()
            dt = np.fromiter((datetime64_to_datetime(ti) for ti in dt), dtype=datetime)

            temporal_data = np.empty((data.shape[0], 4))

            for t in range(data.shape[0]):
                for idx, v in enumerate(
                    [
                        trig_encode(get_day_of_year(dt[t]), 366, TrigFunc.SIN),
                        trig_encode(get_day_of_year(dt[t]), 366, TrigFunc.COS),
                        trig_encode(dt[t].hour, 24, TrigFunc.SIN),
                        trig_encode(dt[t].hour, 24, TrigFunc.COS),
                    ]
                ):
                    temporal_data[t, idx] = v

            self.num_temporal_constants = temporal_data.shape[-1]

        return data, feature_list, temporal_data, spatial_data

    @staticmethod
    def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        split_ratio: float | None = None,
        split_type: int = 2,
        test_shuffle: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets.

        Args:
            X: Input features
            y: Target values
            split_ratio: Ratio of training data
            split_type: Type of split (1, 2, or 3)
            test_shuffle: Whether to shuffle test data

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return DataProcessor.train_val_test_split(X, y, split_ratio, split_type, test_shuffle)

    @staticmethod
    def train_val_test_split(
        X: np.ndarray,
        y: np.ndarray,
        split_ratio: float | None = None,
        split_type: int = 1,
        test_shuffle: bool = False,
        random_state: int = 42,
    ) -> tuple[Any, ...]:
        """Split data into train, validation, and test sets.

        Split types:
        - 0: X_train (2020), X_val (2021), X_test (2022)
        - 1: X_train (2020), X_test (2021)
        - 2: X_train (2020), X_test (2022)
        - 3: X_train (2020), X_test (2021-2022)

        Args:
            X: Input features
            y: Target values
            split_ratio: Ratio of training data
            split_type: Type of split
            test_shuffle: Whether to shuffle test data
            random_state: Random seed for shuffling

        Returns:
            Tuple of arrays depending on split_type
        """
        split_ratio = split_ratio if split_ratio is not None else Config.train_ratio
        train_samples = int(split_ratio * len(X))

        if split_type == 0:
            X_train, X_val, X_test = (
                X[:train_samples],
                X[train_samples : 2 * train_samples],
                X[2 * train_samples :],
            )
            y_train, y_val, y_test = (
                y[:train_samples],
                y[train_samples : 2 * train_samples],
                y[2 * train_samples :],
            )
            X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
            X_val, y_val = shuffle(X_val, y_val, random_state=random_state)
            if test_shuffle:
                X_test, y_test = shuffle(X_test, y_test, random_state=random_state)

            return X_train, X_val, X_test, y_train, y_val, y_test

        elif split_type == 1:
            X_train, X_test = X[:train_samples], X[train_samples : 2 * train_samples]
            y_train, y_test = y[:train_samples], y[train_samples : 2 * train_samples]

        elif split_type == 2:
            X_train, X_test = X[:train_samples], X[2 * train_samples :]
            y_train, y_test = y[:train_samples], y[2 * train_samples :]

        else:
            X_train, X_test = X[:train_samples], X[train_samples:]
            y_train, y_test = y[:train_samples], y[train_samples:]

        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

        if test_shuffle:
            X_test, y_test = shuffle(X_test, y_test, random_state=random_state)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def count_neighbours(radius: int) -> tuple[int, list[tuple[int, int]]]:
        """Count neighbours within a radius.

        Args:
            radius: Neighbourhood radius

        Returns:
            Tuple of (count, indices)
        """
        count = 0
        indices: list[tuple[int, int]] = []
        if radius < 0:
            return count, indices

        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if x == 0 and y == 0:
                    continue
                distance = (x**2 + y**2) ** 0.5
                if distance <= radius:
                    count += 1
                    indices.append((x, y))
        return count, indices

    @staticmethod
    def get_spatial_info(
        area: tuple[float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Get spatial information for plotting.

        Args:
            area: Geographic area (north, west, south, east)

        Returns:
            Tuple of (lat_span, lon_span, spatial_limits)
        """
        res = 0.25
        if area is None:
            north, west, south, east = SMALL_AREA
        else:
            north, west, south, east = area
        spatial_limits = [west, east, south, north]
        we_span_1d = np.arange(west, east + res, res)
        ns_span_1d = np.arange(north, south - res, -res)
        lon_span = np.array([we_span_1d for _ in range(len(ns_span_1d))])
        lat_span = np.array([ns_span_1d for _ in range(len(we_span_1d))]).T
        return lat_span, lon_span, spatial_limits
