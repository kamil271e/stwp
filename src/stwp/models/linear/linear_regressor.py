"""Linear regression models for weather prediction."""

import copy
from typing import Any

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from stwp.models.base import BaselineRegressor, Scaler, RegressorType

LinearModel = ElasticNet | Lasso | LinearRegression | Ridge


class LinearRegressor(BaselineRegressor):
    """Linear regression model for weather prediction."""

    def __init__(
        self,
        X_shape: tuple[int, ...],
        fh: int,
        feature_list: list[str],
        regressor_type: RegressorType = RegressorType.LINEAR,
        alpha: float = 1.0,
        scaler_type: Scaler = Scaler.STANDARD,
    ):
        """Initialize the linear regressor.

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
        self.models: list[LinearModel] = [copy.deepcopy(self.model) for _ in range(self.num_features)]