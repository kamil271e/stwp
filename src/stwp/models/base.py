"""Baseline regressor model."""

from __future__ import annotations

import copy
import os
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from stwp.data.processor import DataProcessor
from stwp.features import Features
from stwp.utils.visualization import draw_poland

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Scaler:
    MIN_MAX: type[MinMaxScaler] = MinMaxScaler
    STANDARD: type[StandardScaler] = StandardScaler
    MAX_ABS: type[MaxAbsScaler] = MaxAbsScaler
    ROBUST: type[RobustScaler] = RobustScaler


class RegressorType(StrEnum):
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"


# emprically determined
def get_radius(neighbours: int) -> int:
    if neighbours <= 5:
        return 1
    elif neighbours <= 13:
        return 2
    else:
        return 3


# TODO: magic numbers handling
class BaselineRegressor:
    """Baseline regressor using mean prediction."""

    def __init__(
        self,
        X_shape: tuple[int, ...],
        fh: int,
        feature_list: list[str],
        scaler_type: type[StandardScaler]
        | type[MinMaxScaler]
        | type[MaxAbsScaler]
        | type[RobustScaler] = Scaler.STANDARD,
    ):
        """Initialize the baseline regressor.

        Args:
            X_shape: Shape of input data
            fh: Forecast horizon
            feature_list: List of feature names
            scaler_type: Type of scaler to use
        """
        # TODO: I don't understand this '5' check here
        if len(X_shape) > 5:
            (
                _,
                self.latitude,
                self.longitude,
                self.neighbours,
                self.input_state,
                self.num_features,
            ) = X_shape
        else:
            (
                _,
                self.latitude,
                self.longitude,
                self.input_state,
                self.num_features,
            ) = X_shape
            self.neighbours = 1

        self.fh = fh
        self.feature_list = feature_list
        self.num_spatial_constants = self.num_features - len(self.feature_list)
        self.num_features = self.num_features - self.num_spatial_constants

        self.scaler = scaler_type()
        self.model = DummyRegressor(strategy="constant", constant=0)
        self.models = [copy.deepcopy(self.model) for _ in range(self.num_features)]
        self.scalers = [copy.deepcopy(self.scaler) for _ in range(self.num_features)]

    def train(
        self,
        X_train: NDArray[np.floating[Any]],
        y_train: NDArray[np.floating[Any]],
        normalize: bool = False,
    ) -> None:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            normalize: Whether to normalize targets
        """
        if (
            len(str(self.__class__).split(".")) < 4
        ):  # BaselineRegressor # TODO: i dont understand this '4'
            y_mean = np.mean(y_train, axis=0)
            self.model.constant = y_mean

        X = X_train.reshape(
            -1,
            self.neighbours * self.input_state * (self.num_features + self.num_spatial_constants),
        )
        for i in range(self.num_features):
            yi = y_train[..., 0, i].reshape(-1, 1)
            if normalize:
                self.scalers[i].fit(yi)
            self.models[i].fit(X, yi)

    def get_rmse(
        self,
        y_hat: NDArray[np.floating[Any]],
        y_test: NDArray[np.floating[Any]],
        normalize: bool = False,
        begin: int | None = None,
        end: int | None = None,
    ) -> list[float]:
        """Calculate RMSE for each feature.

        Args:
            y_hat: Predictions
            y_test: Ground truth
            normalize: Whether to normalize before calculating
            begin: Start index
            end: End index

        Returns:
            List of RMSE values per feature
        """
        rmse_features = []
        if begin is not None and end is not None:
            y_hat = y_hat[begin:end]
            y_test = y_test[begin:end]
        for i in range(self.num_features):
            y_hat_i = y_hat[..., i].reshape(-1, 1)
            y_test_i = y_test[..., i].reshape(-1, 1)
            if normalize:
                y_test_i = self.scalers[i].transform(y_test_i)
                y_hat_i = self.scalers[i].transform(y_hat_i)
            err = np.sqrt(mean_squared_error(y_hat_i, y_test_i))
            rmse_features.append(err)
        return rmse_features

    def get_mae(
        self,
        y_hat: NDArray[np.floating[Any]],
        y_test: NDArray[np.floating[Any]],
        normalize: bool = False,
    ) -> list[float]:
        """Calculate MAE for each feature.

        Args:
            y_hat: Predictions
            y_test: Ground truth
            normalize: Whether to normalize before calculating

        Returns:
            List of MAE values per feature
        """
        mae_features = []
        for i in range(self.num_features):
            y_hat_i = y_hat[..., i].reshape(-1, 1)
            y_test_i = y_test[..., i].reshape(-1, 1)
            if normalize:
                y_test_i = self.scalers[i].transform(y_test_i)
                y_hat_i = self.scalers[i].transform(y_hat_i)
            err = mean_absolute_error(y_hat_i, y_test_i)
            mae_features.append(err)
        return mae_features

    def evaluate(
        self,
        y_hat: NDArray[np.floating[Any]],
        y_test: NDArray[np.floating[Any]],
    ) -> tuple[list[float], list[float]]:
        """Evaluate predictions.

        Args:
            y_hat: Predictions
            y_test: Ground truth

        Returns:
            Tuple of (rmse_scores, mae_scores)
        """
        return self.get_rmse(y_hat, y_test), self.get_mae(y_hat, y_test)

    def plot_predictions(
        self,
        y_hat: NDArray[np.floating[Any]],
        y_test: NDArray[np.floating[Any]],
        max_samples: int,
        pretty: bool = False,
        save: bool = False,
    ) -> None:
        """Plot predictions vs ground truth.

        Args:
            y_hat: Predictions
            y_test: Ground truth
            max_samples: Maximum number of samples to plot
            pretty: Whether to use cartopy for pretty maps
            save: Whether to save figures
        """
        spatial: dict[str, Any] = {}
        if pretty:
            lat_span, lon_span, spatial_limits = DataProcessor.get_spatial_info()
            spatial = {
                "lat_span": lat_span,
                "lon_span": lon_span,
                "spatial_limits": spatial_limits,
            }

        for i in range(max_samples):
            y_test_sample, y_hat_sample = y_test[i], y_hat[i]
            if pretty:
                fig, axs = plt.subplots(
                    self.num_features,
                    3 * self.fh,
                    figsize=(10 * self.fh, 3 * self.num_features),
                    subplot_kw={"projection": ccrs.Mercator(central_longitude=40)},
                )
            else:
                fig, ax = plt.subplots(
                    self.num_features,
                    3 * self.fh,
                    figsize=(10 * self.fh, 3 * self.num_features),
                )

            for j in range(self.num_features):
                cur_feature = self.feature_list[j]
                y_test_sample_feature_j = y_test_sample[..., j].reshape(-1, 1)
                y_hat_sample_feature_j = y_hat_sample[..., j].reshape(-1, 1)
                mse = mean_squared_error(y_test_sample_feature_j, y_hat_sample_feature_j)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_sample_feature_j, y_hat_sample_feature_j)
                std = np.std(y_test_sample_feature_j)
                sqrt_n = np.sqrt(y_test_sample_feature_j.shape[0])
                print(f"{cur_feature} => RMSE:  {rmse}; MAE: {mae}; SE: {std / sqrt_n}")

                for k in range(3 * self.fh):
                    ts = k // 3
                    current_ax = axs[j, k] if pretty else ax[j, k]

                    if k % 3 == 0:
                        title = rf"$Y^{{t+{ts + 1}}}_{{{cur_feature}}}$"
                        value = y_test[i, ..., ts, j]
                        cmap = plt.cm.coolwarm  # type: ignore[attr-defined]
                    elif k % 3 == 1:
                        title = rf"$\hat{{Y}}^{{t+{ts + 1}}}_{{{cur_feature}}}$"
                        value = y_hat[i, ..., ts, j]
                        cmap = plt.cm.coolwarm  # type: ignore[attr-defined]
                    else:
                        title = rf"$|Y - \hat{{Y}}|^{{t+{ts + 1}}}_{{{cur_feature}}}$"
                        value = np.abs(y_test[i, ..., ts, j] - y_hat[i, ..., ts, j])
                        cmap = "binary"

                    if pretty:
                        draw_poland(current_ax, value, title, cmap, **spatial)
                    else:
                        pl = current_ax.imshow(value, cmap=cmap)
                        current_ax.set_title(title)
                        current_ax.axis("off")
                        fig.colorbar(pl, ax=current_ax, fraction=0.15)

            plt.tight_layout()
            if save:
                name = str(self.__class__).split(".")[-2]
                plt.savefig(f"../data/analysis/{name}_{i}.png")
            plt.show()

    def _predict(
        self,
        X_test: NDArray[np.floating[Any]],
        y_test: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Make predictions.

        Args:
            X_test: Test features
            y_test: Test targets (for shape reference)

        Returns:
            Predictions array
        """
        if (
            len(str(self.__class__).split(".")) < 4
        ):  # BaselineRegressor # TODO: i dont understand this '4'
            y_mean = np.tile(self.model.constant, (y_test.shape[0], 1, 1, 1, 1))
            return y_mean

        X = X_test.reshape(
            -1,
            self.neighbours * self.input_state * (self.num_features + self.num_spatial_constants),
        )
        y_hat: NDArray[np.floating[Any]]
        if self.fh == 1:
            y_hat_list = []
            for i in range(self.num_features):
                y_hat_i = (
                    self.models[i].predict(X).reshape(-1, self.latitude, self.longitude, self.fh)
                )
                y_hat_list.append(y_hat_i)
            y_hat = np.array(y_hat_list).transpose((1, 2, 3, 4, 0))
        else:
            y_hat = self.predict_autoreg(X_test, y_test)
        y_hat = self.clip_total_cloud_cover(y_hat)
        return y_hat

    def predict_and_evaluate(
        self,
        X_test: NDArray[np.floating[Any]],
        y_test: NDArray[np.floating[Any]],
        plot: bool = True,
        max_samples: int = 5,
    ) -> NDArray[np.floating[Any]]:
        """Make predictions and evaluate.

        Args:
            X_test: Test features
            y_test: Test targets
            plot: Whether to plot predictions
            max_samples: Maximum samples to plot

        Returns:
            Predictions array
        """
        y_hat = self._predict(X_test, y_test)
        if plot:
            self.plot_predictions(y_hat, y_test, max_samples=max_samples)
        rmse_scores, mae_scores = self.evaluate(y_hat, y_test)
        print("=======================================")
        print("Evaluation metrics for entire test set:")
        print("=======================================")

        sqrt_n = np.sqrt(y_test.shape[0] * self.latitude * self.longitude * self.fh)
        for i in range(self.num_features):
            print(
                f"{self.feature_list[i]} => RMSE: {rmse_scores[i]};  "
                f"MAE: {mae_scores[i]}; SE: {np.std(y_test[..., i]) / sqrt_n}"
            )

        return y_hat

    def predict_autoreg(
        self,
        X_test: NDArray[np.floating[Any]],
        y_test: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Autoregressive prediction.

        Prediction in an autoregressive manner:
        Depends on forecasting horizon parameter, each model prediction
        becomes a part of an input for next timestamp prediction.

        Args:
            X_test: Test features
            y_test: Test targets (for shape reference)

        Returns:
            Predictions array
        """
        y_hat = np.empty(y_test.shape)
        num_samples = X_test.shape[0]
        for i in range(num_samples):
            Xi = X_test[i]
            Yik = np.empty(
                (
                    self.latitude,
                    self.longitude,
                    self.neighbours,
                    self.fh,
                    self.num_features,
                )
            )
            for k in range(-1, self.fh - 1):
                Xik = Xi
                if k > -1:
                    if self.fh - self.input_state < 2:
                        autoreg_start = 0
                    else:
                        autoreg_start = max(0, k - self.input_state + 1)

                    if self.neighbours > 1:
                        Yik[..., k, :] = self.extend(y_hat[i, ..., k, :])
                        Xik = np.concatenate(
                            (Xi[..., k + 1 :, :], Yik[..., autoreg_start : k + 1, :]),
                            axis=-2,
                        )
                    else:
                        Xik = np.concatenate(
                            (
                                Xi[..., k + 1 :, :],
                                y_hat[i, ..., autoreg_start : k + 1, :],
                            ),
                            axis=-2,
                        )
                for j in range(self.num_features):
                    y_hat[i, ..., k + 1, j] = (
                        self.models[j]
                        .predict(
                            Xik.reshape(
                                -1,
                                self.neighbours * self.input_state * self.num_features,
                            )
                        )
                        .reshape(1, self.latitude, self.longitude)
                    )
        return y_hat

    def extend(self, Y: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Extend data sample to include neighbours.

        Args:
            Y: Input data

        Returns:
            Extended data with neighbours
        """
        radius = get_radius(self.neighbours)
        _, indices = DataProcessor.count_neighbours(radius=radius)
        Y_out = np.empty((self.latitude, self.longitude, self.neighbours, self.num_features))
        Y_out[..., 0, :] = Y
        for n in range(1, self.neighbours):
            i, j = indices[n - 1]
            for lo in range(self.longitude):
                for la in range(self.latitude):
                    if -1 < la + i < self.latitude and -1 < lo + j < self.longitude:
                        Y_out[la, lo, n] = Y[la + i, lo + j]
                    else:
                        Y_out[la, lo, n] = Y[la, lo]
        return Y_out

    def save_prediction_tensor(
        self,
        y_hat: NDArray[np.floating[Any]],
        path: str | None = None,
    ) -> None:
        """Save predictions to file.

        Args:
            y_hat: Predictions to save
            path: Output path
        """
        if path is None:
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            name = str(self.__class__).split(".")[-2]
            path = f"../data/pred/{name}_{t}.npy"

        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("")
        np.save(path, y_hat.transpose(0, 1, 2, 4, 3))

    @staticmethod
    def clip_total_cloud_cover(
        y_hat: NDArray[np.floating[Any]],
        idx: int = Features.TCC_IDX,
    ) -> NDArray[np.floating[Any]]:
        """Clip total cloud cover to valid range.

        Args:
            y_hat: Predictions
            idx: Index of total cloud cover feature

        Returns:
            Clipped predictions
        """
        y_hat[..., idx] = np.clip(y_hat[..., idx], 0, 1)
        return y_hat
