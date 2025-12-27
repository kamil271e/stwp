"""Gradient boosting models for weather prediction."""

import copy
from typing import Any

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor

from stwp.models.base import BaselineRegressor


class GradBooster(BaselineRegressor):
    """Gradient boosting regressor for weather prediction."""

    def __init__(
        self,
        X_shape: tuple[int, ...],
        fh: int,
        feature_list: list[str],
        booster: str = "lgb",
        scaler_type: str = "standard",
        **kwargs: Any,
    ):
        """Initialize the gradient booster.

        Args:
            X_shape: Shape of input data
            fh: Forecast horizon
            feature_list: List of feature names
            booster: Type of booster (lgb, xgb, cat, ada)
            scaler_type: Type of scaler
            **kwargs: Additional arguments for the booster
        """
        super().__init__(X_shape, fh, feature_list, scaler_type=scaler_type)

        booster_map: dict[str, Any] = {
            "lgb": LGBMRegressor(verbose=-1, n_jobs=-1, **kwargs),
            "xgb": XGBRegressor(n_jobs=-1, **kwargs),
            "cat": CatBoostRegressor(verbose=0, thread_count=-1, **kwargs),
            "ada": AdaBoostRegressor(**kwargs),
        }

        if booster not in booster_map:
            raise ValueError(f"{booster} booster not implemented")

        self.model = booster_map[booster]
        self.models = [copy.deepcopy(self.model) for _ in range(self.num_features)]
