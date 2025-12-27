"""Graph Neural Network module for weather prediction."""

from enum import StrEnum

import torch
import torch.nn as nn
from torch_geometric.nn.conv import CGConv, GATConv, GENConv, PDNConv, TransformerConv

from stwp.config import Config
from stwp.models.gnn.st_encoder_module import SpatioTemporalEncoder


class ArchitectureType(StrEnum):
    TRANSFORMER = "transformer"
    CGC = "cgc"
    GAT = "gat"
    GEN = "gen"
    PDN = "pdn"


class GNNModule(torch.nn.Module):
    """Graph Neural Network for spatio-temporal weather prediction."""

    def __init__(
        self,
        input_features: int,
        output_features: int,
        edge_dim: int,
        hidden_dim: int,
        input_t_dim: int = 4,
        input_s_dim: int = 6,
        input_size: int | None = None,
        fh: int | None = None,
        architecture: ArchitectureType = ArchitectureType.TRANSFORMER,
        num_graph_cells: int | None = None,
    ):
        """Initialize the GNN module.

        Args:
            input_features: Number of input features
            output_features: Number of output features
            edge_dim: Edge attribute dimension
            hidden_dim: Hidden layer dimension
            input_t_dim: Temporal input dimension
            input_s_dim: Spatial input dimension
            input_size: Number of input timesteps
            fh: Forecast horizon
            arch: Architecture type (trans, cgc, gat, gen, pdn)
            num_graph_cells: Number of graph convolution cells
        """
        super().__init__()

        input_size = input_size if input_size is not None else Config.input_size
        fh = fh if fh is not None else Config.forecast_horizon
        num_graph_cells = num_graph_cells if num_graph_cells is not None else Config.graph_cells

        self.mlp_embedder = nn.Linear(input_features * input_size, hidden_dim)
        self.st_encoder = SpatioTemporalEncoder(hidden_dim, input_t_dim, input_s_dim, hidden_dim)
        self.layer_norm_embed = nn.LayerNorm(hidden_dim)
        self.gnns: nn.ModuleList | None = None
        self._choose_graph_cells(architecture, hidden_dim, edge_dim, num_graph_cells)
        self.mlp_decoder = nn.Linear(hidden_dim, output_features * fh)
        self.fh = fh

    def _choose_graph_cells(
        self,
        architecture: ArchitectureType,
        hidden_dim: int,
        edge_dim: int,
        num_graph_cells: int,
    ) -> None:
        """Initialize graph convolution layers based on architecture.

        Args:
            arch: Architecture type
            hidden_dim: Hidden dimension
            edge_dim: Edge attribute dimension
            num_graph_cells: Number of graph cells
        """
        if architecture == ArchitectureType.TRANSFORMER:
            self.gnns = nn.ModuleList(
                [
                    TransformerConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        edge_dim=edge_dim,
                    )
                    for _ in range(num_graph_cells)
                ]
            )
        elif architecture == ArchitectureType.CGC:
            self.gnns = nn.ModuleList(
                [CGConv(hidden_dim, edge_dim, aggr="mean") for _ in range(num_graph_cells)]
            )
        elif architecture == ArchitectureType.GAT:
            self.gnns = nn.ModuleList(
                [
                    GATConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        edge_dim=edge_dim,
                    )
                    for _ in range(num_graph_cells)
                ]
            )
        elif architecture == ArchitectureType.GEN:
            self.gnns = nn.ModuleList(
                [
                    GENConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        edge_dim=edge_dim,
                        num_layers=num_graph_cells,
                    )
                ]
            )
        elif architecture == ArchitectureType.PDN:
            self.gnns = nn.ModuleList(
                [
                    PDNConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        edge_dim=edge_dim,
                        hidden_channels=4 * edge_dim,
                    )
                    for _ in range(num_graph_cells)
                ]
            )
        else:
            raise NotImplementedError(f"Architecture {architecture} not implemented")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            t: Temporal features
            s: Spatial features

        Returns:
            Predictions
        """
        x = x.view(-1, x.size(-2) * x.size(-1))
        x = self.mlp_embedder(x)
        x = self.st_encoder(x, t, s)
        x = self.layer_norm_embed(x).relu()
        if self.gnns is not None:
            for gnn in self.gnns:
                x = x + gnn(x, edge_index, edge_attr).relu()
        x = self.mlp_decoder(x)
        return x.view(x.size(0), x.size(1) // self.fh, self.fh)
