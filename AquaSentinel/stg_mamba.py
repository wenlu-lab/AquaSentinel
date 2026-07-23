"""STG-Mamba expert: spatial-temporal graph learning via selective state
space models (Li et al., 2024). Compact implementation retaining the
core mechanism: a selective SSM scan along time whose (dt, B, C)
parameters are input-dependent, combined with graph convolution for
spatial propagation.

Unified expert interface:
    forward(x: [B, T, N, F], a_hat: [N, N]) -> y: [B, N, F]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SelectiveSSM(nn.Module):
    """Minimal S6-style selective scan over the time dimension."""

    def __init__(self, hidden: int, state: int = 16) -> None:
        super().__init__()
        self.hidden = hidden
        self.state = state
        # Log-parameterized negative-real diagonal state matrix A.
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, state + 1).float()).repeat(hidden, 1)
        )
        # Input-dependent (selective) parameters.
        self.to_dt = nn.Linear(hidden, hidden)
        self.to_b = nn.Linear(hidden, state)
        self.to_c = nn.Linear(hidden, state)
        self.d = nn.Parameter(torch.ones(hidden))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [L, T, H] where L folds batch and nodes
        l, t, h = u.shape
        a = -torch.exp(self.a_log)                            # [H, S]
        dt = torch.nn.functional.softplus(self.to_dt(u))      # [L, T, H]
        b = self.to_b(u)                                      # [L, T, S]
        c = self.to_c(u)                                      # [L, T, S]

        state = torch.zeros(l, h, self.state, device=u.device)
        outs = []
        for step in range(t):
            dt_s = dt[:, step].unsqueeze(-1)                  # [L, H, 1]
            da = torch.exp(dt_s * a[None])                    # discretized A
            db = dt_s * b[:, step].unsqueeze(1)               # [L, H, S]
            state = da * state + db * u[:, step].unsqueeze(-1)
            y = (state * c[:, step].unsqueeze(1)).sum(-1)     # [L, H]
            outs.append(y + self.d * u[:, step])
        return torch.stack(outs, dim=1)                       # [L, T, H]


class STGMambaBlock(nn.Module):
    def __init__(self, hidden: int, state: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.in_proj = nn.Linear(hidden, 2 * hidden)
        self.ssm = SelectiveSSM(hidden, state)
        self.out_proj = nn.Linear(hidden, hidden)
        self.gconv = nn.Linear(hidden, hidden)

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # h: [B, T, N, H]
        b, t, n, d = h.shape
        z = self.norm(h)
        u, gate = self.in_proj(z).chunk(2, dim=-1)
        u = u.permute(0, 2, 1, 3).reshape(b * n, t, d)
        y = self.ssm(u).reshape(b, n, t, d).permute(0, 2, 1, 3)
        y = self.out_proj(y * torch.nn.functional.silu(gate))
        h = h + y
        # Spatial propagation via graph convolution (Kalman-style update
        # of node states with neighborhood information).
        h = h + torch.relu(self.gconv(torch.einsum("mn,btnh->btmh", a_hat, h)))
        return h


class STGMamba(nn.Module):
    def __init__(
        self,
        num_features: int = 2,
        hidden: int = 32,
        state: int = 16,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(num_features, hidden)
        self.blocks = nn.ModuleList(
            [STGMambaBlock(hidden, state) for _ in range(blocks)]
        )
        self.head = nn.Linear(hidden, num_features)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for blk in self.blocks:
            h = blk(h, a_hat)
        return self.head(h[:, -1])                            # [B, N, F]
