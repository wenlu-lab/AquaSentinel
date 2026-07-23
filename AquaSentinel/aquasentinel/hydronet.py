"""HydroNet expert.

Spatiotemporal graph neural network for modeling hydraulic dependencies
in urban wastewater systems (Guo and Wang, SIGSPATIAL 2024). The model
couples a graph convolution that respects the directed flow topology
with a GRU over time, so upstream states inform downstream forecasts.

Unified expert interface:
    forward(x: [B, T, N, F], a_hat: [N, N]) -> y: [B, N, F]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FlowAwareGraphConv(nn.Module):
    """Two-branch propagation: normalized adjacency + learned edge gate.

    The learned gate lets the model modulate how strongly each hydraulic
    dependency contributes, which is useful when pipe attributes make
    the nominal adjacency an imperfect proxy for influence.
    """

    def __init__(self, c_in: int, c_out: int, num_nodes_hint: int = 0) -> None:
        super().__init__()
        self.lin_self = nn.Linear(c_in, c_out)
        self.lin_neigh = nn.Linear(c_in, c_out)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        neigh = torch.einsum("mn,bnc->bmc", a_hat, x)
        g = torch.sigmoid(self.gate)
        return torch.relu(self.lin_self(x) + g * self.lin_neigh(neigh))


class HydroNet(nn.Module):
    def __init__(
        self,
        num_features: int = 2,
        hidden: int = 48,
        gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(num_features, hidden)
        self.gnns = nn.ModuleList(
            [FlowAwareGraphConv(hidden, hidden) for _ in range(gnn_layers)]
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_features)
        )

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        h = self.encoder(x)                       # [B, T, N, H]
        # Spatial propagation applied per timestep.
        frames = []
        for step in range(t):
            hs = h[:, step]
            for gnn in self.gnns:
                hs = gnn(hs, a_hat) + hs          # residual
            frames.append(hs)
        h = torch.stack(frames, dim=1)            # [B, T, N, H]
        # Temporal recurrence per node.
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        _, last = self.gru(h)                     # [1, B*N, H]
        out = last.squeeze(0).reshape(b, n, -1)
        return self.head(out)                     # [B, N, F]
