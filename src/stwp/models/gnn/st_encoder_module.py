"""Spatio-temporal encoder module for GNN."""

import torch
import torch.nn as nn

from stwp.config import Config


class SpatioTemporalEncoder(nn.Module):
    """Encoder for combining spatial and temporal features."""

    def __init__(
        self,
        input_X_dim: int,
        input_t_dim: int,
        input_s_dim: int,
        output_dim: int,
        hidden: int = 16,
    ):
        """Initialize the encoder.

        Args:
            input_X_dim: Input feature dimension
            input_t_dim: Temporal feature dimension
            input_s_dim: Spatial feature dimension
            output_dim: Output dimension
            hidden: Hidden layer dimension
        """
        super().__init__()
        self.input_X_dim = input_X_dim
        self.lat, self.lon = Config.input_dims
        self.mlp_embedder = nn.Linear(input_X_dim, hidden)
        self.temporal_embedder = nn.Linear(input_t_dim, self.lat * self.lon)
        self.mlp_decoder = nn.Linear(hidden + 1 + input_s_dim, output_dim)

    def forward(
        self,
        X: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            X: Input features
            t: Temporal features
            s: Spatial features

        Returns:
            Encoded output
        """
        batch_size = s.shape[0]
        X = X.reshape((batch_size, self.lat * self.lon, self.input_X_dim))
        X = self.mlp_embedder(X).relu()
        t = self.temporal_embedder(t.reshape(batch_size, 1, -1)).relu().permute((0, 2, 1))
        concat = torch.cat((X, t, s), dim=-1)
        output = self.mlp_decoder(concat)
        output = output.reshape((-1, self.input_X_dim))
        return output
