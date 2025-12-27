"""Pytest fixtures for STWP tests."""

import os
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from stwp.config import Config
from stwp.features import Features


@pytest.fixture
def mock_config() -> Config:
    """Create a mock configuration for testing."""
    os.environ["STWP_DEVICE"] = "cpu"
    os.environ["STWP_DATA_PATH"] = "test_data.grib"
    os.environ["STWP_TRAIN_RATIO"] = "0.333"
    os.environ["STWP_BATCH_SIZE"] = "2"
    os.environ["STWP_FORECAST_HORIZON"] = "1"

    config = Config()

    # Clean up env vars
    for key in [
        "STWP_DEVICE",
        "STWP_DATA_PATH",
        "STWP_TRAIN_RATIO",
        "STWP_BATCH_SIZE",
        "STWP_FORECAST_HORIZON",
    ]:
        os.environ.pop(key, None)

    return config


@pytest.fixture
def sample_weather_data() -> dict[str, Any]:
    """Create sample weather prediction data for testing."""
    return {
        "55.0": {
            "14.0": {
                "t2m": {"2024-01-01T06:00:00": 280.5, "2024-01-01T12:00:00": 282.0},
                "sp": {"2024-01-01T06:00:00": 101325.0, "2024-01-01T12:00:00": 101320.0},
                "tcc": {"2024-01-01T06:00:00": 0.5, "2024-01-01T12:00:00": 0.7},
                "tp": {"2024-01-01T06:00:00": 0.0, "2024-01-01T12:00:00": 0.5},
                "u10": {"2024-01-01T06:00:00": 2.5, "2024-01-01T12:00:00": 3.0},
                "v10": {"2024-01-01T06:00:00": 1.5, "2024-01-01T12:00:00": 2.0},
            },
            "14.25": {
                "t2m": {"2024-01-01T06:00:00": 281.0, "2024-01-01T12:00:00": 282.5},
                "sp": {"2024-01-01T06:00:00": 101320.0, "2024-01-01T12:00:00": 101315.0},
                "tcc": {"2024-01-01T06:00:00": 0.4, "2024-01-01T12:00:00": 0.6},
                "tp": {"2024-01-01T06:00:00": 0.0, "2024-01-01T12:00:00": 0.3},
                "u10": {"2024-01-01T06:00:00": 2.3, "2024-01-01T12:00:00": 2.8},
                "v10": {"2024-01-01T06:00:00": 1.3, "2024-01-01T12:00:00": 1.8},
            },
        },
        "54.75": {
            "14.0": {
                "t2m": {"2024-01-01T06:00:00": 279.5, "2024-01-01T12:00:00": 281.0},
                "sp": {"2024-01-01T06:00:00": 101330.0, "2024-01-01T12:00:00": 101325.0},
                "tcc": {"2024-01-01T06:00:00": 0.6, "2024-01-01T12:00:00": 0.8},
                "tp": {"2024-01-01T06:00:00": 0.1, "2024-01-01T12:00:00": 0.6},
                "u10": {"2024-01-01T06:00:00": 2.7, "2024-01-01T12:00:00": 3.2},
                "v10": {"2024-01-01T06:00:00": 1.7, "2024-01-01T12:00:00": 2.2},
            },
            "14.25": {
                "t2m": {"2024-01-01T06:00:00": 280.0, "2024-01-01T12:00:00": 281.5},
                "sp": {"2024-01-01T06:00:00": 101325.0, "2024-01-01T12:00:00": 101320.0},
                "tcc": {"2024-01-01T06:00:00": 0.55, "2024-01-01T12:00:00": 0.75},
                "tp": {"2024-01-01T06:00:00": 0.05, "2024-01-01T12:00:00": 0.4},
                "u10": {"2024-01-01T06:00:00": 2.6, "2024-01-01T12:00:00": 3.1},
                "v10": {"2024-01-01T06:00:00": 1.6, "2024-01-01T12:00:00": 2.1},
            },
        },
    }


@pytest.fixture
def sample_array() -> np.ndarray:
    """Create a sample numpy array for testing."""
    return np.random.rand(25, 45, 6)


@pytest.fixture
def mock_trainer() -> MagicMock:
    """Create a mock trainer for testing."""
    trainer = MagicMock()
    trainer.predict_to_json.return_value = {
        "55.0": {
            "14.0": {
                Features.T2M: {"2024-01-01T06:00:00": 280.5},
            }
        }
    }
    return trainer


@pytest.fixture
def feature_list() -> list[str]:
    """Return the standard feature list."""
    return Features.as_list()


@pytest.fixture
def sample_prediction_array() -> np.ndarray:
    """Create a sample prediction array with correct shape."""
    # Shape: (batch, lat, lon, timesteps, features)
    return np.random.rand(10, 25, 45, 1, Features.COUNT)
