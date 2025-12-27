"""Hyperparameter Optimization module for weather prediction models.

This module provides a unified interface for running HPO studies across
different model types (linear, gradient boosting, GNN, CNN).
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import optuna
import torch

from stwp.config import Config
from stwp.data.processor import DataProcessor
from stwp.models.base import RegressorType
from stwp.models.boosting.grad_booster import GradBooster
from stwp.models.cnn.trainer import Trainer as CNNTrainer
from stwp.models.gnn.gnn_module import ArchitectureType
from stwp.models.gnn.trainer import Trainer as GNNTrainer
from stwp.models.linear.linear_regressor import LinearRegressor
from stwp.models.linear.simple_linear_regressor import SimpleLinearRegressor
from stwp.utils.progress import print_progress_bar

logger = logging.getLogger(__name__)


class ModelType(StrEnum):
    """Supported model types for HPO."""

    SIMPLE_LINEAR = "simple-linear"
    LINEAR = "linear"
    LGBM = "lgbm"
    GNN = "gnn"
    CNN = "cnn"


class InvalidModelTypeError(ValueError):
    """Raised when an unsupported model type is specified."""

    VALID_TYPES = [m.value for m in ModelType]

    def __init__(self, model_type: str) -> None:
        super().__init__(f"Invalid model type: '{model_type}'. Valid types: {self.VALID_TYPES}")


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""

    model_type: ModelType
    n_trials: int = 50
    use_neighbours: bool = False
    sequence_length: int = 1
    forecast_horizon: int = 1
    sequence_n_trials: int = 15
    fh_n_trials: int = 15
    num_epochs: int = 3
    max_alpha: float = 10.0
    subset: int | None = None

    # Linear model defaults
    sequence_alpha: float = 5.0
    sequence_regressor: RegressorType = RegressorType.RIDGE

    # GNN defaults
    gnn_hidden_dim: int = 32
    gnn_lr: float = 1e-3
    gnn_architecture: ArchitectureType = ArchitectureType.TRANSFORMER

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HPOConfig:
        """Create config from dictionary."""
        model_type = ModelType(data.pop("model_type", data.pop("baseline_type", "gnn")))
        return cls(model_type=model_type, **data)


@dataclass
class StudyResults:
    """Results from an HPO study."""

    best_value: int
    x_values: list[int] = field(default_factory=list)
    y_values: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    raw_metrics: dict[int, Any] = field(default_factory=dict)


@dataclass
class HPOResults:
    """Complete results from all HPO studies."""

    sequence_results: StudyResults | None = None
    fh_results: StudyResults | None = None
    params: dict[str, Any] = field(default_factory=dict)
    scaler_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    monthly_errors: dict[str, float] = field(default_factory=dict)
    layer_results: StudyResults | None = None
    error_maps: list[Any] = field(default_factory=list)


class ModelTrainer(Protocol):
    """Protocol for model trainers."""

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> None:
        """Train the model."""
        ...

    def predict(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        ...

    def get_rmse(
        self, y_hat: np.ndarray, y_test: np.ndarray, normalize: bool = True, **kwargs: Any
    ) -> np.ndarray:
        """Calculate RMSE metrics."""
        ...

    def save_prediction_tensor(self, y_hat: np.ndarray, path: str | None = None) -> None:
        """Save predictions to file."""
        ...


class BaseModelHandler(ABC):
    """Base class for model-specific HPO handling."""

    def __init__(self, config: HPOConfig, processor: DataProcessor) -> None:
        self.config = config
        self.processor = processor
        self.feature_list = processor.feature_list

    @abstractmethod
    def create_model(
        self,
        X_shape: tuple[int, ...],
        forecast_horizon: int,
        **params: Any,
    ) -> Any:
        """Create a model instance with given parameters."""
        ...

    @abstractmethod
    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters for Optuna trial."""
        ...

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        X_shape: tuple[int, ...],
        forecast_horizon: int,
        normalize: bool = True,
        **params: Any,
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        """Train model and return (mean_rmse, rmse_values, raw_rmse)."""
        model = self.create_model(X_shape, forecast_horizon, **params)
        model.train(X_train, y_train, normalize=normalize)
        y_hat = model._predict(X_test, y_test)
        rmse_values = model.get_rmse(y_hat, y_test, normalize=normalize)
        raw_rmse = model.get_rmse(y_hat, y_test, normalize=False)
        return float(np.mean(rmse_values)), rmse_values, raw_rmse


class LinearModelHandler(BaseModelHandler):
    """Handler for linear regression models."""

    REGRESSOR_TYPES: list[str] = [
        RegressorType.LASSO,
        RegressorType.RIDGE,
        RegressorType.ELASTIC_NET,
    ]

    def __init__(
        self,
        config: HPOConfig,
        processor: DataProcessor,
        simple: bool = False,
    ) -> None:
        super().__init__(config, processor)
        self.simple = simple
        self.model_class = SimpleLinearRegressor if simple else LinearRegressor

    def create_model(
        self,
        X_shape: tuple[int, ...],
        forecast_horizon: int,
        **params: Any,
    ) -> LinearRegressor | SimpleLinearRegressor:
        defaults = {
            "regressor_type": self.config.sequence_regressor,
            "alpha": self.config.sequence_alpha,
        }
        defaults.update(params)
        return self.model_class(X_shape, forecast_horizon, self.feature_list, **defaults)

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", 0.1, self.config.max_alpha, log=True),
            "regressor_type": trial.suggest_categorical("regressor_type", self.REGRESSOR_TYPES),
        }


class GradBoostHandler(BaseModelHandler):
    """Handler for gradient boosting models."""

    def create_model(
        self,
        X_shape: tuple[int, ...],
        forecast_horizon: int,
        **params: Any,
    ) -> GradBooster:
        return GradBooster(X_shape, forecast_horizon, self.feature_list, **params)

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 0.5, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 0.5, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 25, 50),
        }


class NeuralNetHandler(BaseModelHandler):
    """Handler for neural network models (GNN/CNN)."""

    def __init__(
        self,
        config: HPOConfig,
        processor: DataProcessor,
        is_gnn: bool = True,
    ) -> None:
        super().__init__(config, processor)
        self.is_gnn = is_gnn
        self._trainer: GNNTrainer | CNNTrainer | None = None

    @property
    def trainer(self) -> GNNTrainer | CNNTrainer:
        """Lazily initialize trainer."""
        if self._trainer is None:
            if self.is_gnn:
                self._trainer = GNNTrainer(
                    architecture=self.config.gnn_architecture,
                    hidden_dim=self.config.gnn_hidden_dim,
                    lr=self.config.gnn_lr,
                    subset=self.config.subset,
                )
            else:
                self._trainer = CNNTrainer(
                    subset=self.config.subset,
                    test_shuffle=False,
                )
        return self._trainer

    def create_model(
        self,
        X_shape: tuple[int, ...],
        forecast_horizon: int,
        **params: Any,
    ) -> GNNTrainer | CNNTrainer:
        return self.trainer

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        logger.info("HPO not implemented for neural networks")
        return {}

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        X_shape: tuple[int, ...],
        forecast_horizon: int,
        normalize: bool = True,
        input_size: int = 1,
        graph_cells: int = 9,
        verbose: bool = True,
        **params: Any,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Train neural network and evaluate."""
        Config.forecast_horizon = forecast_horizon
        Config.input_size = input_size
        if self.is_gnn:
            Config.graph_cells = graph_cells

        self.trainer.update_config(Config)
        self.trainer.train(num_epochs=self.config.num_epochs)

        rmse_norm, y_hat_norm = self.trainer.evaluate("test", verbose=verbose, inverse_norm=False)
        rmse_raw, y_hat_real = self.trainer.evaluate("test", verbose=verbose)

        return float(np.mean(rmse_norm[0])), rmse_norm[0], rmse_raw[0]

    def save_artifacts(
        self,
        output_dir: Path,
        suffix: str,
        y_hat_norm: np.ndarray,
        y_hat_real: np.ndarray,
    ) -> None:
        """Save model state and predictions."""
        output_dir.mkdir(exist_ok=True)
        model_type = "gnn" if self.is_gnn else "cnn"

        torch.save(
            self.trainer.model.state_dict(),
            output_dir / f"model_state_{model_type}_{suffix}.pt",
        )
        self.trainer.save_prediction_tensor(
            y_hat_norm,
            str(output_dir / f"prediction_tensor_{model_type}_{suffix}_norm.pt"),
        )
        self.trainer.save_prediction_tensor(
            y_hat_real,
            str(output_dir / f"prediction_tensor_{model_type}_{suffix}_real.pt"),
        )


def create_handler(config: HPOConfig, processor: DataProcessor) -> BaseModelHandler:
    """Factory function to create appropriate model handler."""
    handlers: dict[ModelType, type[BaseModelHandler] | tuple[type, dict[str, Any]]] = {
        ModelType.SIMPLE_LINEAR: (LinearModelHandler, {"simple": True}),
        ModelType.LINEAR: (LinearModelHandler, {"simple": False}),
        ModelType.LGBM: GradBoostHandler,
        ModelType.GNN: (NeuralNetHandler, {"is_gnn": True}),
        ModelType.CNN: (NeuralNetHandler, {"is_gnn": False}),
    }

    handler_info = handlers.get(config.model_type)
    if handler_info is None:
        raise InvalidModelTypeError(str(config.model_type))

    if isinstance(handler_info, tuple):
        handler_class, kwargs = handler_info
        return handler_class(config, processor, **kwargs)
    return handler_info(config, processor)


class HPO:
    """Hyperparameter Optimization orchestrator.

    Provides a unified interface for running sequence length, forecast horizon,
    and hyperparameter optimization studies across different model types.
    """

    SCALERS = ("standard", "min_max", "max_abs", "robust")
    MONTHS = {
        1: ("January", 1, 31),
        2: ("February", 32, 59),
        3: ("March", 60, 90),
        4: ("April", 91, 120),
        5: ("May", 121, 151),
        6: ("June", 152, 181),
        7: ("July", 182, 212),
        8: ("August", 213, 243),
        9: ("September", 244, 273),
        10: ("October", 274, 304),
        11: ("November", 305, 334),
        12: ("December", 335, 365),
    }

    def __init__(self, config: HPOConfig | dict[str, Any]) -> None:
        """Initialize HPO with configuration.

        Args:
            config: HPOConfig instance or dict with configuration values.
        """
        if isinstance(config, dict):
            config = HPOConfig.from_dict(config)
        self.config = config

        self.processor = DataProcessor()
        self.data = self.processor.data
        self.handler = create_handler(config, self.processor)

        self.results = HPOResults()
        self._best_sequence = config.sequence_length
        self._best_fh = config.forecast_horizon
        self._best_layer = 5
        self._best_params: dict[str, Any] = {}

        self._output_dir = Path(f"./{config.model_type.value}")
        self._verbosity = False

    @property
    def best_sequence(self) -> int:
        return self._best_sequence

    @property
    def best_fh(self) -> int:
        return self._best_fh

    @property
    def best_params(self) -> dict[str, Any]:
        return self._best_params

    def _is_neural_net(self) -> bool:
        return self.config.model_type in (ModelType.GNN, ModelType.CNN)

    def _prepare_data(
        self, sequence_length: int, forecast_horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        self.processor.upload_data(self.data)
        return self.processor.preprocess(
            sequence_length, forecast_horizon, self.config.use_neighbours
        )

    def determine_best_sequence(self) -> StudyResults:
        """Find optimal input sequence length.

        Returns:
            StudyResults with sequence optimization data.
        """
        results = StudyResults(best_value=1)
        best_rmse = float("inf")

        print_progress_bar(
            0,
            self.config.sequence_n_trials + 1,
            prefix="Sequence Progress:",
            suffix="Complete",
            length=50,
        )

        for s in range(1, self.config.sequence_n_trials + 1):
            X, y = self._prepare_data(s, self.config.forecast_horizon)
            X_train, X_test, y_train, y_test = self.processor.train_val_test_split(X, y)

            start_time = time.time()

            if self._is_neural_net():
                mean_rmse, rmse_norm, rmse_raw = self.handler.train_and_evaluate(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X.shape,
                    self.config.forecast_horizon,
                    input_size=s,
                    graph_cells=self._best_layer,
                )
                # Save neural net artifacts
                if isinstance(self.handler, NeuralNetHandler):
                    _, y_hat_norm = self.handler.trainer.evaluate(
                        "test", verbose=False, inverse_norm=False
                    )
                    _, y_hat_real = self.handler.trainer.evaluate("test", verbose=False)
                    self.handler.save_artifacts(self._output_dir, f"s{s}", y_hat_norm, y_hat_real)
            else:
                mean_rmse, _, rmse_raw = self.handler.train_and_evaluate(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X.shape,
                    self.config.forecast_horizon,
                )

            elapsed = time.time() - start_time

            results.x_values.append(s)
            results.y_values.append(mean_rmse)
            results.times.append(elapsed)
            results.raw_metrics[s] = rmse_raw.tolist() if hasattr(rmse_raw, "tolist") else rmse_raw

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                results.best_value = s

            print_progress_bar(
                s,
                self.config.sequence_n_trials + 1,
                prefix="Sequence Progress:",
                suffix="Complete",
                length=50,
            )

        self._best_sequence = results.best_value
        self.results.sequence_results = results
        return results

    def determine_best_fh(self) -> StudyResults:
        """Find optimal forecast horizon.

        Returns:
            StudyResults with forecast horizon optimization data.
        """
        results = StudyResults(best_value=1)
        best_rmse = float("inf")

        print_progress_bar(
            0,
            self.config.fh_n_trials + 1,
            prefix="Forecast Horizon Progress:",
            suffix="Complete",
            length=50,
        )

        for fh in range(1, self.config.fh_n_trials + 1):
            X, y = self._prepare_data(self._best_sequence, fh)
            X_train, X_test, y_train, y_test = self.processor.train_val_test_split(X, y)

            start_time = time.time()

            if self._is_neural_net():
                mean_rmse, rmse_norm, rmse_raw = self.handler.train_and_evaluate(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X.shape,
                    fh,
                    input_size=self._best_sequence,
                    graph_cells=self._best_layer,
                )
                if isinstance(self.handler, NeuralNetHandler):
                    _, y_hat_norm = self.handler.trainer.evaluate(
                        "test", verbose=False, inverse_norm=False
                    )
                    _, y_hat_real = self.handler.trainer.evaluate("test", verbose=False)
                    self.handler.save_artifacts(
                        self._output_dir, f"fh_{fh}", y_hat_norm, y_hat_real
                    )
            else:
                mean_rmse, _, rmse_raw = self.handler.train_and_evaluate(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X.shape,
                    fh,
                    **self._best_params,
                )

            elapsed = time.time() - start_time

            results.x_values.append(fh)
            results.y_values.append(mean_rmse)
            results.times.append(elapsed)
            results.raw_metrics[fh] = rmse_raw.tolist() if hasattr(rmse_raw, "tolist") else rmse_raw

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                results.best_value = fh

            print_progress_bar(
                fh,
                self.config.fh_n_trials + 1,
                prefix="Forecast Horizon Progress:",
                suffix="Complete",
                length=50,
            )

        self._best_fh = results.best_value
        self.results.fh_results = results
        return results

    def run_param_study(self) -> dict[str, Any]:
        """Run Optuna study for hyperparameter optimization.

        Returns:
            Best parameters found.
        """
        if self._is_neural_net():
            logger.info("Parameter HPO not implemented for neural networks")
            return {}

        def objective(trial: optuna.Trial) -> float:
            X, y = self._prepare_data(self._best_sequence, self.config.forecast_horizon)
            X_train, X_val, _, y_train, y_val, _ = self.processor.train_val_test_split(
                X, y, split_type=0
            )
            params = self.handler.suggest_params(trial)
            mean_rmse, _, _ = self.handler.train_and_evaluate(
                X_train,
                X_val,
                y_train,
                y_val,
                X.shape,
                self.config.forecast_horizon,
                **params,
            )
            return mean_rmse

        if not self._verbosity:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.n_trials)

        self._best_params = study.best_params
        self.results.params = study.best_params
        logger.info("Parameter study finished")
        return study.best_params

    def run_full_study(self) -> HPOResults:
        """Run complete HPO pipeline: sequence -> params -> forecast horizon.

        Returns:
            Complete HPOResults with all study data.
        """
        self.determine_best_sequence()
        if not self._is_neural_net():
            self.run_param_study()
        self.determine_best_fh()
        self.save_results()
        return self.results

    def test_scalers(self, model_path: str | None = None) -> dict[str, dict[str, float]]:
        """Test different scaler types.

        Args:
            model_path: Optional path to pre-trained model.

        Returns:
            Dictionary mapping scaler names to their metrics.
        """
        scaler_metrics: dict[str, dict[str, float]] = {}

        for scaler in self.SCALERS:
            X, y = self._prepare_data(self._best_sequence, self._best_fh)
            X_train, X_test, y_train, y_test = self.processor.train_val_test_split(X, y)

            start_time = time.time()

            if self._is_neural_net():
                Config.scaler_type = scaler
                mean_rmse, _, _ = self.handler.train_and_evaluate(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X.shape,
                    self._best_fh,
                    input_size=self._best_sequence,
                    graph_cells=self._best_layer,
                )
            else:
                handler = create_handler(self.config, self.processor)
                model = handler.create_model(
                    X.shape,
                    self._best_fh,
                    scaler_type=scaler,
                    **self._best_params,
                )
                model.train(X_train, y_train, normalize=True)
                y_hat = model._predict(X_test, y_test)
                rmse = model.get_rmse(y_hat, y_test, normalize=True)
                mean_rmse = float(np.mean(rmse))

            elapsed = time.time() - start_time
            scaler_metrics[scaler] = {"rmse": mean_rmse, "execution_time": elapsed}

        self.results.scaler_metrics = scaler_metrics
        return scaler_metrics

    def monthly_error(self, model_path: str | None = None) -> dict[str, float]:
        """Calculate prediction error by month.

        Args:
            model_path: Optional path to pre-trained model.

        Returns:
            Dictionary mapping month names to RMSE values.
        """
        monthly_errors: dict[str, float] = {}

        X, y = self._prepare_data(self._best_sequence, self._best_fh)
        X_train, X_test, y_train, y_test = self.processor.train_val_test_split(
            X, y, split_type=2, test_shuffle=False
        )

        if self._is_neural_net():
            if model_path:
                self.handler.trainer.load_model(model_path)
            else:
                Config.forecast_horizon = self._best_fh
                Config.input_size = self._best_sequence
                if self.config.model_type == ModelType.GNN:
                    Config.graph_cells = self._best_layer
                self.handler.trainer.update_config(Config)
                self.handler.trainer.train(num_epochs=self.config.num_epochs)

            for _month_num, (name, start_day, end_day) in self.MONTHS.items():
                rmse, _ = self.handler.trainer.evaluate(
                    "test",
                    verbose=False,
                    inverse_norm=False,
                    begin=start_day,
                    end=end_day,
                )
                monthly_errors[name] = float(np.mean(rmse[0]))
        else:
            model = self.handler.create_model(X.shape, self._best_fh, **self._best_params)
            model.train(X_train, y_train, normalize=True)
            y_hat = model._predict(X_test, y_test)

            for month_num, (name, start_day, end_day) in self.MONTHS.items():
                begin, end = self._get_month_indices(month_num, start_day, end_day)
                rmse = model.get_rmse(y_hat, y_test, normalize=True, begin=begin, end=end)
                monthly_errors[name] = float(np.mean(rmse))

        self.results.monthly_errors = monthly_errors
        return monthly_errors

    def _get_month_indices(self, month: int, start_day: int, end_day: int) -> tuple[int, int]:
        """Convert month day range to data indices (4 samples per day)."""
        samples_per_day = 4
        if month == 1:
            return start_day - 1, self.MONTHS[2][1] * samples_per_day
        elif month == 12:
            return start_day * samples_per_day + 1, end_day * samples_per_day + 1
        else:
            next_month_start = self.MONTHS[month + 1][1]
            return start_day * samples_per_day + 1, next_month_start * samples_per_day + 1

    def gnn_layer_study(self, models: list[str] | None = None) -> StudyResults:
        """Study effect of GNN layer count.

        Args:
            models: Optional list of pre-trained model paths.

        Returns:
            StudyResults with layer optimization data.
        """
        if self.config.model_type != ModelType.GNN:
            raise ValueError("Layer study only available for GNN models")

        results = StudyResults(best_value=2)
        best_rmse = float("inf")

        layer_range = range(2, 10)

        for i, cell_count in enumerate(layer_range):
            if models:
                Config.graph_cells = cell_count
                self.handler.trainer.update_config(Config)
                self.handler.trainer.load_model(models[i])
            else:
                Config.forecast_horizon = self.config.forecast_horizon
                Config.input_size = self._best_sequence
                Config.graph_cells = cell_count
                self.handler.trainer.update_config(Config)
                self.handler.trainer.train(num_epochs=self.config.num_epochs)

            rmse_norm, y_hat_norm = self.handler.trainer.evaluate(
                "test", verbose=False, inverse_norm=False
            )
            _, y_hat_real = self.handler.trainer.evaluate("test", verbose=False)
            mean_rmse = float(np.mean(rmse_norm[0]))

            results.x_values.append(cell_count)
            results.y_values.append(mean_rmse)

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                results.best_value = cell_count

            if not models:
                self.handler.save_artifacts(
                    self._output_dir, f"cell_{cell_count}", y_hat_norm, y_hat_real
                )

        self._best_layer = results.best_value
        self.results.layer_results = results
        return results

    def save_results(self) -> None:
        """Save all results to JSON files."""
        # Save parameters
        params_file = f"{self.config.model_type.value}-params.json"
        with open(params_file, "w") as f:
            json.dump(self._best_params, f, indent=2)

        # Save plots data
        self._save_plots_data()

        # Save error maps if available
        self._output_dir.mkdir(exist_ok=True)
        if self.results.error_maps:
            np.save(self._output_dir / "error_maps.npy", self.results.error_maps)

    def _save_plots_data(self) -> None:
        """Save plot data to modelsplots.json."""
        plots_file = Path("modelsplots.json")
        data: dict[str, Any] = {}

        if plots_file.exists():
            with open(plots_file) as f:
                data = json.load(f)

        model_key = self.config.model_type.value
        seq = self.results.sequence_results
        fh = self.results.fh_results
        layer = self.results.layer_results

        data[model_key] = {
            "sequence_plot_x": seq.x_values if seq else [],
            "sequence_plot_y": seq.y_values if seq else [],
            "sequence_plot_time": seq.times if seq else [],
            "fh_plot_x": fh.x_values if fh else [],
            "fh_plot_y": fh.y_values if fh else [],
            "fh_plot_time": fh.times if fh else [],
            "metrics": [],
            "metrics_for_scalers": self.results.scaler_metrics,
            "not_normalized_plot_sequence": seq.raw_metrics if seq else {},
            "not_normalized_plot_fh": fh.raw_metrics if fh else {},
            "month_error": self.results.monthly_errors,
            "gnn_alpha_plot_x": [],
            "gnn_alpha_plot_y": [],
            "gnn_cell_plot_x": layer.x_values if layer else [],
            "gnn_cell_plot_y": layer.y_values if layer else [],
        }

        with open(plots_file, "w") as f:
            json.dump(data, f, indent=2)

    def report(self) -> None:
        """Print summary report of HPO results."""
        logger.info(f"Best sequence length: {self._best_sequence}")
        logger.info(f"Best forecast horizon: {self._best_fh}")
        if self._best_params:
            logger.info("Best parameters:")
            for key, value in self._best_params.items():
                logger.info(f"  {key}: {value}")
