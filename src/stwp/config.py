"""Unified configuration with environment variable support.

This is the primary configuration class for STWP. It supports configuration
via environment variables with the STWP_ prefix.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import torch


def _get_device() -> torch.device:
    """Determine the best available device."""
    device_str = os.getenv("STWP_DEVICE")
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class _Config:
    """Configuration for STWP models and training."""

    # Device configuration
    device: torch.device = field(default_factory=_get_device)

    # Data paths
    data_path: str = field(default_factory=lambda: os.getenv("STWP_DATA_PATH", "data/input.grib"))
    model_path: str = field(default_factory=lambda: os.getenv("STWP_MODEL_PATH", "data/gnn_fh5.pt"))

    # Training configuration
    train_ratio: float = field(
        default_factory=lambda: float(os.getenv("STWP_TRAIN_RATIO", "0.333"))
    )
    batch_size: int = field(default_factory=lambda: int(os.getenv("STWP_BATCH_SIZE", "8")))
    random_state: int = field(default_factory=lambda: int(os.getenv("STWP_RANDOM_STATE", "42")))

    # Model configuration
    forecast_horizon: int = field(
        default_factory=lambda: int(os.getenv("STWP_FORECAST_HORIZON", "1"))
    )
    input_size: int = field(default_factory=lambda: int(os.getenv("STWP_INPUT_SIZE", "5")))
    r: int = field(default_factory=lambda: int(os.getenv("STWP_R", "2")))
    graph_cells: int = field(default_factory=lambda: int(os.getenv("STWP_GRAPH_CELLS", "9")))
    scaler_type: Literal["standard", "min_max", "max_abs", "robust"] = field(
        default_factory=lambda: os.getenv("STWP_SCALER_TYPE", "standard")  # type: ignore
    )

    # Grid dimensions
    input_dims: tuple[int, int] = field(
        default_factory=lambda: (
            int(os.getenv("STWP_INPUT_DIMS_H", "32")),
            int(os.getenv("STWP_INPUT_DIMS_W", "48")),
        )
    )
    output_dims: tuple[int, int] = field(
        default_factory=lambda: (
            int(os.getenv("STWP_OUTPUT_DIMS_H", "25")),
            int(os.getenv("STWP_OUTPUT_DIMS_W", "45")),
        )
    )

    # API configuration
    api_host: str = field(default_factory=lambda: os.getenv("STWP_API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("STWP_API_PORT", "8888")))


# Default configuration instance - use this throughout the codebase
Config = _Config()
