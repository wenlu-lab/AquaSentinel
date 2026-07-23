"""CaST expert: causal-lens spatiotemporal forecasting (Xia et al.,
NeurIPS 2023). Compact implementation retaining the core idea of
disentangling the temporal signal into an invariant (causal) component
and an environment-dependent component, with an environment codebook
selected by attention, plus a spatial mixing layer.

Unified expert interface:
    forward(x: [B, T, N, F], a_hat: [N, N]) -> y: [B, N, F]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnvironmentCodebook(nn.Module):
    """Learned bank of environment prototypes queried by soft attention."""

    def __init__(self, hidden: int, num_envs: int = 8) -> None:
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_envs, hidden) * 0.02)
        self.query = nn.Linear(hidden, hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, H] -> environment representation of same shape
        q = self.query(h)                                     # [B, N, H]
        logits = torch.einsum("bnh,eh->bne", q, self.codebook)
        w = torch.softmax(logits / q.shape[-1] ** 0.5, dim=-1)
        return torch.einsum("bne,eh->bnh", w, self.codebook)


class CaST(nn.Module):
    def __init__(
        self,
        num_features: int = 2,
        hidden: int = 32,
        num_envs: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(num_features, hidden, batch_first=True)
        # Disentangle into causal (invariant) and environment branches.
        self.causal_proj = nn.Linear(hidden, hidden)
        self.env_proj = nn.Linear(hidden, hidden)
        self.envs = EnvironmentCodebook(hidden, num_envs)
        self.spatial = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_features),
        )

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        _, last = self.encoder(seq)
        h = last.squeeze(0).reshape(b, n, -1)                 # [B, N, H]

        causal = torch.relu(self.causal_proj(h))
        env = self.envs(torch.relu(self.env_proj(h)))

        # Spatial mixing of the causal branch over the graph.
        causal = causal + torch.relu(
            self.spatial(torch.einsum("mn,bnh->bmh", a_hat, causal))
        )
        return self.head(torch.cat([causal, env], dim=-1))    # [B, N, F]
