"""Simple linear regression models for weather prediction."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from stwp.data.processor import DataProcessor
from stwp.models.base import BaselineRegressor, RegressorType, Scaler, get_radius

if TYPE_CHECKING:
    from numpy.typing import NDArray

from typing import TypeAlias

LinearModel: TypeAlias = ElasticNet | Lasso | LinearRegression | Ridge


class SimpleLinearRegressor(BaselineRegressor):
    """Simple linear regressor where each model operates on single feature.

    Model M_i takes X_fi as an input - models have no access to different features.
    """

    def __init__(
        self,
        X_shape: tuple[int, ...],
        fh: int,
        feature_list: list[str],
        regressor_type: RegressorType = RegressorType.LINEAR,
        alpha: float = 1.0,
        scaler_type: type = Scaler.STANDARD,
    ):
        """Initialize the simple linear regressor.

        Args:
            X_shape: Shape of input data
            fh: Forecast horizon
            feature_list: List of feature names
            regressor_type: Type of regressor (linear, ridge, lasso, elastic_net)
            alpha: Regularization strength
            scaler_type: Type of scaler
        """
        super().__init__(X_shape, fh, feature_list, scaler_type=scaler_type)

        regressor_map: dict[RegressorType, Any] = {
            RegressorType.LINEAR: LinearRegression(),
            RegressorType.RIDGE: Ridge(alpha=alpha),
            RegressorType.LASSO: Lasso(alpha=alpha),
            RegressorType.ELASTIC_NET: ElasticNet(alpha=alpha),
        }

        if regressor_type not in regressor_map:
            raise ValueError(f"{regressor_type} regressor not implemented")

        self.model: LinearModel = regressor_map[regressor_type]
        self.models: list[LinearModel] = [
            copy.deepcopy(self.model) for _ in range(self.num_features)
        ]

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
        for i in range(self.num_features):
            Xi = X_train[..., i].reshape(-1, self.neighbours * self.input_state)
            yi = y_train[..., 0, i].reshape(-1, 1)
            if normalize:
                self.scalers[i].fit(yi)
            self.models[i].fit(Xi, yi)

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
        if self.fh == 1:
            y_hat = []
            for i in range(self.num_features):
                Xi = X_test[..., i].reshape(-1, self.neighbours * self.input_state)
                y_hat_i = (
                    self.models[i].predict(Xi).reshape(-1, self.latitude, self.longitude, self.fh)
                )
                y_hat.append(y_hat_i)
            y_hat = np.array(y_hat).transpose((1, 2, 3, 4, 0))
        else:
            y_hat = self.predict_autoreg(X_test, y_test)
        return y_hat

    def predict_autoreg(
        self,
        X_test: NDArray[np.floating[Any]],
        y_test: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Autoregressive prediction.

        Args:
            X_test: Test features
            y_test: Test targets (for shape reference)

        Returns:
            Predictions array
        """
        y_hat = np.empty(y_test.shape)
        num_samples = X_test.shape[0]
        for i in range(num_samples):
            for j in range(self.num_features):
                y_hat_ij = np.zeros(y_test[i].shape[:-1])
                Xij = X_test[i, ..., j].reshape(-1, self.neighbours * self.input_state)
                y_hat_ij[..., 0] = (
                    self.models[j].predict(Xij).reshape(1, self.latitude, self.longitude)
                )
                for k in range(self.fh - 1):
                    if self.fh - self.input_state < 2:
                        autoreg_start = 0
                    else:
                        autoreg_start = max(0, k - self.input_state + 1)

                    if self.neighbours > 1:
                        Xij = np.concatenate(
                            (
                                X_test[i, ..., k + 1 :, j],
                                self.extend(y_hat_ij[..., autoreg_start : k + 1]),
                            ),
                            axis=3,
                        )
                    else:
                        Xij = np.concatenate(
                            (
                                X_test[i, ..., k + 1 :, j],
                                y_hat_ij[..., autoreg_start : k + 1],
                            ),
                            axis=2,
                        )
                    Xij = Xij.reshape(-1, self.neighbours * self.input_state)
                    y_hat_ij[..., k + 1] = (
                        self.models[j].predict(Xij).reshape(1, self.latitude, self.longitude)
                    )
                y_hat[i, ..., j] = y_hat_ij
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
        Y_out = np.empty((self.latitude, self.longitude, self.neighbours, Y.shape[-1]))
        Y_out[..., 0, :] = Y.reshape((self.latitude, self.longitude, -1))
        for n in range(1, self.neighbours):
            i, j = indices[n - 1]
            for lo in range(self.longitude):
                for la in range(self.latitude):
                    if -1 < la + i < self.latitude and -1 < lo + j < self.longitude:
                        Y_out[la, lo, n] = Y[la + i, lo + j]
                    else:
                        Y_out[la, lo, n] = Y[la, lo]
        return Y_out
