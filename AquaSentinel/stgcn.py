"""STGCN expert: Spatio-Temporal Graph Convolutional Network.

Compact implementation of the sandwich ST-Conv block design
(temporal gated conv -> spatial graph conv -> temporal gated conv)
from Yu, Yin, and Zhu (IJCAI 2018), adapted to the unified expert
interface used by the MoE ensemble:

    forward(x: [B, T, N, F], a_hat: [N, N]) -> y: [B, N, F]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalGatedConv(nn.Module):
    """1D causal convolution along time with a GLU gate."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kernel, 1))
        self.res = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N]
        res = self.res(x)[:, :, self.kernel - 1:, :]
        p, q = self.conv(x).chunk(2, dim=1)
        return (p + res) * torch.sigmoid(q)


class GraphConv(nn.Module):
    """First-order Chebyshev graph convolution with normalized adjacency."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.theta = nn.Linear(c_in, c_out)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N] -> graph conv over N
        x = x.permute(0, 2, 3, 1)                 # [B, T, N, C]
        x = torch.einsum("mn,btnc->btmc", a_hat, x)
        x = torch.relu(self.theta(x))
        return x.permute(0, 3, 1, 2)              # [B, C, T, N]


class STConvBlock(nn.Module):
    def __init__(self, c_in: int, c_spatial: int, c_out: int, kernel: int = 3):
        super().__init__()
        self.t1 = TemporalGatedConv(c_in, c_out, kernel)
        self.g = GraphConv(c_out, c_spatial)
        self.t2 = TemporalGatedConv(c_spatial, c_out, kernel)
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        x = self.t1(x)
        x = self.g(x, a_hat)
        x = self.t2(x)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class STGCN(nn.Module):
    def __init__(
        self,
        num_features: int = 2,
        hidden: int = 32,
        spatial_hidden: int = 16,
        kernel: int = 3,
        window: int = 12,
    ) -> None:
        super().__init__()
        self.block1 = STConvBlock(num_features, spatial_hidden, hidden, kernel)
        self.block2 = STConvBlock(hidden, spatial_hidden, hidden, kernel)
        remaining = window - 4 * (kernel - 1)
        assert remaining >= 1, "window too short for the chosen kernel"
        self.out_conv = nn.Conv2d(hidden, hidden, kernel_size=(remaining, 1))
        self.head = nn.Linear(hidden, num_features)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F] -> [B, F, T, N]
        h = x.permute(0, 3, 1, 2)
        h = self.block1(h, a_hat)
        h = self.block2(h, a_hat)
        h = torch.relu(self.out_conv(h))          # [B, C, 1, N]
        h = h.squeeze(2).permute(0, 2, 1)         # [B, N, C]
        return self.head(h)                       # [B, N, F]
