"""Tests for HPO module."""

from unittest.mock import MagicMock, patch

from hpo.hpo import (
    HPO,
    GradBoostHandler,
    HPOConfig,
    HPOResults,
    InvalidModelTypeError,
    LinearModelHandler,
    ModelType,
    NeuralNetHandler,
    StudyResults,
    create_handler,
)


class TestModelType:
    """Tests for ModelType enum."""

    def test_values(self) -> None:
        """Should have expected model types."""
        assert ModelType.SIMPLE_LINEAR.value == "simple-linear"
        assert ModelType.LINEAR.value == "linear"
        assert ModelType.LGBM.value == "lgbm"
        assert ModelType.GNN.value == "gnn"
        assert ModelType.CNN.value == "cnn"

    def test_is_str_enum(self) -> None:
        """Should be usable as string via .value."""
        assert ModelType.GNN.value == "gnn"
        assert "gnn" in str(ModelType.GNN).lower()


class TestInvalidModelTypeError:
    """Tests for InvalidModelTypeError exception."""

    def test_message_contains_invalid_type(self) -> None:
        """Should include the invalid type in error message."""
        error = InvalidModelTypeError("invalid_model")
        assert "invalid_model" in str(error)

    def test_message_contains_valid_types(self) -> None:
        """Should list valid types in error message."""
        error = InvalidModelTypeError("invalid_model")
        for model_type in ModelType:
            assert model_type.value in str(error)


class TestHPOConfig:
    """Tests for HPOConfig dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = HPOConfig(model_type=ModelType.GNN)

        assert config.n_trials == 50
        assert config.use_neighbours is False
        assert config.sequence_length == 1
        assert config.forecast_horizon == 1
        assert config.num_epochs == 3
        assert config.max_alpha == 10.0
        assert config.subset is None

    def test_from_dict(self) -> None:
        """Should create config from dictionary."""
        data = {
            "model_type": "gnn",
            "n_trials": 100,
            "num_epochs": 5,
        }
        config = HPOConfig.from_dict(data)

        assert config.model_type == ModelType.GNN
        assert config.n_trials == 100
        assert config.num_epochs == 5

    def test_from_dict_with_baseline_type(self) -> None:
        """Should handle legacy 'baseline_type' key."""
        data = {"baseline_type": "linear"}
        config = HPOConfig.from_dict(data)

        assert config.model_type == ModelType.LINEAR


class TestStudyResults:
    """Tests for StudyResults dataclass."""

    def test_default_values(self) -> None:
        """Should initialize with empty lists."""
        results = StudyResults(best_value=1)

        assert results.best_value == 1
        assert results.x_values == []
        assert results.y_values == []
        assert results.times == []
        assert results.raw_metrics == {}

    def test_mutable_lists(self) -> None:
        """Should allow appending to lists."""
        results = StudyResults(best_value=1)
        results.x_values.append(1)
        results.y_values.append(0.5)

        assert results.x_values == [1]
        assert results.y_values == [0.5]


class TestHPOResults:
    """Tests for HPOResults dataclass."""

    def test_default_values(self) -> None:
        """Should initialize with None/empty values."""
        results = HPOResults()

        assert results.sequence_results is None
        assert results.fh_results is None
        assert results.params == {}
        assert results.scaler_metrics == {}
        assert results.monthly_errors == {}
        assert results.layer_results is None
        assert results.error_maps == []


class TestLinearModelHandler:
    """Tests for LinearModelHandler."""

    @patch("hpo.hpo.DataProcessor")
    def test_suggest_params(self, mock_processor: MagicMock) -> None:
        """Should suggest alpha and regressor_type."""
        config = HPOConfig(model_type=ModelType.LINEAR)
        handler = LinearModelHandler(config, mock_processor, simple=False)

        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 1.0
        mock_trial.suggest_categorical.return_value = "ridge"

        params = handler.suggest_params(mock_trial)

        assert "alpha" in params
        assert "regressor_type" in params
        mock_trial.suggest_float.assert_called_once()
        mock_trial.suggest_categorical.assert_called_once()

    @patch("hpo.hpo.DataProcessor")
    def test_simple_vs_full(self, mock_processor: MagicMock) -> None:
        """Should use correct model class based on simple flag."""
        config = HPOConfig(model_type=ModelType.SIMPLE_LINEAR)

        simple_handler = LinearModelHandler(config, mock_processor, simple=True)
        full_handler = LinearModelHandler(config, mock_processor, simple=False)

        assert simple_handler.simple is True
        assert full_handler.simple is False


class TestGradBoostHandler:
    """Tests for GradBoostHandler."""

    @patch("hpo.hpo.DataProcessor")
    def test_suggest_params(self, mock_processor: MagicMock) -> None:
        """Should suggest LGBM hyperparameters."""
        config = HPOConfig(model_type=ModelType.LGBM)
        handler = GradBoostHandler(config, mock_processor)

        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value = 100
        mock_trial.suggest_float.return_value = 0.1

        params = handler.suggest_params(mock_trial)

        expected_params = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "reg_lambda",
            "reg_alpha",
            "num_leaves",
        ]
        for param in expected_params:
            assert param in params


class TestNeuralNetHandler:
    """Tests for NeuralNetHandler."""

    @patch("hpo.hpo.DataProcessor")
    def test_is_gnn_flag(self, mock_processor: MagicMock) -> None:
        """Should track GNN vs CNN type."""
        config = HPOConfig(model_type=ModelType.GNN)

        gnn_handler = NeuralNetHandler(config, mock_processor, is_gnn=True)
        cnn_handler = NeuralNetHandler(config, mock_processor, is_gnn=False)

        assert gnn_handler.is_gnn is True
        assert cnn_handler.is_gnn is False

    @patch("hpo.hpo.DataProcessor")
    def test_suggest_params_returns_empty(self, mock_processor: MagicMock) -> None:
        """HPO not implemented for neural nets, should return empty dict."""
        config = HPOConfig(model_type=ModelType.GNN)
        handler = NeuralNetHandler(config, mock_processor, is_gnn=True)

        mock_trial = MagicMock()
        params = handler.suggest_params(mock_trial)

        assert params == {}


class TestCreateHandler:
    """Tests for create_handler factory function."""

    @patch("hpo.hpo.DataProcessor")
    def test_creates_linear_handler(self, mock_processor: MagicMock) -> None:
        """Should create LinearModelHandler for linear types."""
        config = HPOConfig(model_type=ModelType.LINEAR)
        handler = create_handler(config, mock_processor)

        assert isinstance(handler, LinearModelHandler)
        assert handler.simple is False

    @patch("hpo.hpo.DataProcessor")
    def test_creates_simple_linear_handler(self, mock_processor: MagicMock) -> None:
        """Should create LinearModelHandler with simple=True."""
        config = HPOConfig(model_type=ModelType.SIMPLE_LINEAR)
        handler = create_handler(config, mock_processor)

        assert isinstance(handler, LinearModelHandler)
        assert handler.simple is True

    @patch("hpo.hpo.DataProcessor")
    def test_creates_gradboost_handler(self, mock_processor: MagicMock) -> None:
        """Should create GradBoostHandler for LGBM."""
        config = HPOConfig(model_type=ModelType.LGBM)
        handler = create_handler(config, mock_processor)

        assert isinstance(handler, GradBoostHandler)

    @patch("hpo.hpo.DataProcessor")
    def test_creates_gnn_handler(self, mock_processor: MagicMock) -> None:
        """Should create NeuralNetHandler for GNN."""
        config = HPOConfig(model_type=ModelType.GNN)
        handler = create_handler(config, mock_processor)

        assert isinstance(handler, NeuralNetHandler)
        assert handler.is_gnn is True

    @patch("hpo.hpo.DataProcessor")
    def test_creates_cnn_handler(self, mock_processor: MagicMock) -> None:
        """Should create NeuralNetHandler for CNN."""
        config = HPOConfig(model_type=ModelType.CNN)
        handler = create_handler(config, mock_processor)

        assert isinstance(handler, NeuralNetHandler)
        assert handler.is_gnn is False


class TestHPO:
    """Tests for HPO orchestrator class."""

    @patch("hpo.hpo.create_handler")
    @patch("hpo.hpo.DataProcessor")
    def test_init_with_config(
        self, mock_processor_cls: MagicMock, mock_create_handler: MagicMock
    ) -> None:
        """Should initialize with HPOConfig."""
        config = HPOConfig(model_type=ModelType.GNN)
        hpo = HPO(config)

        assert hpo.config == config
        assert hpo._best_sequence == config.sequence_length
        assert hpo._best_fh == config.forecast_horizon

    @patch("hpo.hpo.create_handler")
    @patch("hpo.hpo.DataProcessor")
    def test_init_with_dict(
        self, mock_processor_cls: MagicMock, mock_create_handler: MagicMock
    ) -> None:
        """Should accept dict config and convert to HPOConfig."""
        hpo = HPO({"model_type": "lgbm", "n_trials": 10})

        assert hpo.config.model_type == ModelType.LGBM
        assert hpo.config.n_trials == 10

    @patch("hpo.hpo.create_handler")
    @patch("hpo.hpo.DataProcessor")
    def test_is_neural_net(
        self, mock_processor_cls: MagicMock, mock_create_handler: MagicMock
    ) -> None:
        """Should correctly identify neural net model types."""
        gnn_hpo = HPO(HPOConfig(model_type=ModelType.GNN))
        cnn_hpo = HPO(HPOConfig(model_type=ModelType.CNN))
        lgbm_hpo = HPO(HPOConfig(model_type=ModelType.LGBM))

        assert gnn_hpo._is_neural_net() is True
        assert cnn_hpo._is_neural_net() is True
        assert lgbm_hpo._is_neural_net() is False

    @patch("hpo.hpo.create_handler")
    @patch("hpo.hpo.DataProcessor")
    def test_best_properties(
        self, mock_processor_cls: MagicMock, mock_create_handler: MagicMock
    ) -> None:
        """Should expose best values as properties."""
        hpo = HPO(HPOConfig(model_type=ModelType.GNN))
        hpo._best_sequence = 5
        hpo._best_fh = 3
        hpo._best_params = {"lr": 0.01}

        assert hpo.best_sequence == 5
        assert hpo.best_fh == 3
        assert hpo.best_params == {"lr": 0.01}

    def test_months_constant(self) -> None:
        """Should have correct month ranges."""
        assert HPO.MONTHS[1] == ("January", 1, 31)
        assert HPO.MONTHS[12] == ("December", 335, 365)
        assert len(HPO.MONTHS) == 12

    def test_scalers_constant(self) -> None:
        """Should have expected scaler types."""
        expected = ("standard", "min_max", "max_abs", "robust")
        assert expected == HPO.SCALERS

    @patch("hpo.hpo.create_handler")
    @patch("hpo.hpo.DataProcessor")
    def test_get_month_indices(
        self, mock_processor_cls: MagicMock, mock_create_handler: MagicMock
    ) -> None:
        """Should convert month ranges to sample indices."""
        hpo = HPO(HPOConfig(model_type=ModelType.GNN))

        # January is special case
        jan_start, jan_end = hpo._get_month_indices(1, 1, 31)
        assert jan_start == 0

        # Regular month
        feb_start, feb_end = hpo._get_month_indices(2, 32, 59)
        samples_per_day = 4
        assert feb_start == 32 * samples_per_day + 1

        # December is special case
        dec_start, dec_end = hpo._get_month_indices(12, 335, 365)
        assert dec_end == 365 * samples_per_day + 1
