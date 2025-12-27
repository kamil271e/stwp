"""Gradient boosting models for weather prediction."""

import copy
from enum import StrEnum
from typing import Any

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor

from stwp.models.base import BaselineRegressor, Scaler


class BoosterType(StrEnum):
    LGB = "lgb"
    XGB = "xgb"
    CAT = "cat"
    ADA = "ada"


class GradBooster(BaselineRegressor):
    """Gradient boosting regressor for weather prediction."""

    def __init__(
        self,
        X_shape: tuple[int, ...],
        fh: int,
        feature_list: list[str],
        booster_type: BoosterType = BoosterType.LGB,
        scaler_type: type = Scaler.STANDARD,
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

        booster_map: dict[BoosterType, Any] = {
            BoosterType.LGB: LGBMRegressor(verbose=-1, n_jobs=-1, **kwargs),
            BoosterType.XGB: XGBRegressor(n_jobs=-1, **kwargs),
            BoosterType.CAT: CatBoostRegressor(verbose=0, thread_count=-1, **kwargs),
            BoosterType.ADA: AdaBoostRegressor(**kwargs),
        }

        if booster_type not in booster_map:
            raise ValueError(f"{booster_type} booster not implemented")

        self.model = booster_map[booster_type]
        self.models = [copy.deepcopy(self.model) for _ in range(self.num_features)]
