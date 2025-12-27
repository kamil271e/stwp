"""Tests for configuration module."""

import os

import torch

from stwp.config import Config, _Config


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        # Clear env vars first
        for key in list(os.environ.keys()):
            if key.startswith("STWP_"):
                del os.environ[key]

        config = _Config()

        assert config.train_ratio == 0.333
        assert config.batch_size == 8
        assert config.forecast_horizon == 1
        assert config.input_size == 5
        assert config.random_state == 42
        assert config.input_dims == (32, 48)
        assert config.output_dims == (25, 45)
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8888

    def test_env_var_override(self) -> None:
        """Test that environment variables override defaults."""
        os.environ["STWP_BATCH_SIZE"] = "16"
        os.environ["STWP_TRAIN_RATIO"] = "0.5"
        os.environ["STWP_FORECAST_HORIZON"] = "3"

        config = _Config()

        assert config.batch_size == 16
        assert config.train_ratio == 0.5
        assert config.forecast_horizon == 3

        # Cleanup
        del os.environ["STWP_BATCH_SIZE"]
        del os.environ["STWP_TRAIN_RATIO"]
        del os.environ["STWP_FORECAST_HORIZON"]

    def test_device_selection(self) -> None:
        """Test device selection logic."""
        os.environ["STWP_DEVICE"] = "cpu"
        config = _Config()
        assert config.device == torch.device("cpu")
        del os.environ["STWP_DEVICE"]

    def test_api_config(self) -> None:
        """Test API configuration."""
        os.environ["STWP_API_HOST"] = "127.0.0.1"
        os.environ["STWP_API_PORT"] = "9000"

        config = _Config()

        assert config.api_host == "127.0.0.1"
        assert config.api_port == 9000

        # Cleanup
        del os.environ["STWP_API_HOST"]
        del os.environ["STWP_API_PORT"]

    def test_default_instance(self) -> None:
        """Test that Config is a default instance of _Config."""
        assert isinstance(Config, _Config)
