"""Exponential smoothing models for weather prediction."""

from typing import TYPE_CHECKING, Any

import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing

from stwp.models.base import BaselineRegressor
from stwp.features import Features
from enum import StrEnum

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Smoothing configuration
DEFAULT_SMOOTHING_PARAM: float = 0.8
SMOOTHING_PARAMS: dict[str, float] = {
    Features.T2M: 0.4,
    Features.TCC: 0.6,
}

class SmoothingType(StrEnum):
    SIMPLE = "simple"
    HOLT = "holt"
    SEASONAL = "seasonal"

class InitializationMethod(StrEnum):
    ESTIMATED = "estimated"
    HEURISTIC = "heuristic"
    LEGACY_HEURISTIC = "legacy-heuristic"
    KNOWN = "known"


class SmoothingPredictor(BaselineRegressor):
    """Exponential smoothing predictor for weather data."""

    def __init__(
        self,
        X_shape: tuple[int, ...],
        fh: int,
        feature_list: list[str],
        smoothing_type: SmoothingType = SmoothingType.SIMPLE,
    ):
        """Initialize the smoothing predictor.

        Args:
            X_shape: Shape of input data
            fh: Forecast horizon
            feature_list: List of feature names
            smoothing_type: Type of smoothing to use
        """
        super().__init__(X_shape, fh, feature_list)

        if smoothing_type == SmoothingType.SIMPLE:
            self.type = smoothing_type
        elif smoothing_type in (SmoothingType.HOLT, SmoothingType.SEASONAL):
            raise DeprecationWarning("holt and seasonal smoothing are not supported")
        else:
            raise ValueError(f"Smoothing type {smoothing_type} not implemented")

        self.params = [
            SMOOTHING_PARAMS.get(fname, DEFAULT_SMOOTHING_PARAM)
            for fname in self.feature_list
        ]

    def train(
        self,
        X_train: NDArray[np.floating[Any]],
        y_train: NDArray[np.floating[Any]],
        normalized: bool = False,
    ) -> None:
        """Training not needed for exponential smoothing.

        Args:
            X_train: Training features (unused)
            y_train: Training targets (unused)
            normalized: Whether to normalize (unused)
        """
        pass

    def _predict(
        self,
        X_test: NDArray[np.floating[Any]],
        y_test: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Make predictions using exponential smoothing.

        Args:
            X_test: Test features
            y_test: Test targets (for shape reference)

        Returns:
            Predictions array
        """
        X = X_test.reshape(-1, self.latitude, self.longitude, self.input_state, self.num_features)
        y_hat = []
        for i in range(X.shape[0]):
            y_hat_i = []
            for j in range(self.num_features):
                ylat = []
                for lat in range(X.shape[1]):
                    ylon = []
                    for lon in range(X.shape[2]):
                        if self.type == SmoothingType.SIMPLE:
                            forecast = (
                                SimpleExpSmoothing(
                                    X[i, lat, lon, :, j],
                                    initialization_method=InitializationMethod.KNOWN,
                                    initial_level=X[i, lat, lon, 0, j],
                                )
                                .fit(smoothing_level=self.params[j], optimized=False)
                                .forecast(self.fh)
                            )
                        else:
                            raise ValueError(f"Unknown smoothing type: {self.type}")
                        ylon.append(forecast)
                    ylat.append(ylon)
                y_hat_i.append(ylat)
            y_hat.append(y_hat_i)

        y_hat_arr = (
            np.array(y_hat)
            .reshape((X.shape[0], self.num_features, self.latitude, self.longitude, self.fh))
            .transpose((0, 2, 3, 4, 1))
        )
        return y_hat_arr
