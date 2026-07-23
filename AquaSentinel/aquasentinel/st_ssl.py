"""ST-SSL expert: spatio-temporal self-supervised learning (Ji et al.,
AAAI 2023). Compact implementation retaining the core mechanisms: a
graph-convolutional spatiotemporal encoder trained jointly with an
auxiliary self-supervised contrastive objective computed from an
adaptively augmented (perturbed) view of the input.

The auxiliary loss is exposed via `ssl_loss()` computed on the most
recent forward pass; the training loop may add it to the forecasting
loss with a small weight.

Unified expert interface:
    forward(x: [B, T, N, F], a_hat: [N, N]) -> y: [B, N, F]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class STEncoder(nn.Module):
    def __init__(self, num_features: int, hidden: int) -> None:
        super().__init__()
        self.inp = nn.Linear(num_features, hidden)
        self.gc1 = nn.Linear(hidden, hidden)
        self.gc2 = nn.Linear(hidden, hidden)
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        h = torch.relu(self.inp(x))
        h = torch.relu(self.gc1(torch.einsum("mn,btnh->btmh", a_hat, h))) + h
        h = torch.relu(self.gc2(torch.einsum("mn,btnh->btmh", a_hat, h))) + h
        seq = h.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        _, last = self.temporal(seq)
        return last.squeeze(0).reshape(b, n, -1)              # [B, N, H]


class STSSL(nn.Module):
    def __init__(
        self,
        num_features: int = 2,
        hidden: int = 32,
        aug_drop: float = 0.1,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = STEncoder(num_features, hidden)
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, num_features)
        self.aug_drop = aug_drop
        self.temperature = temperature
        self._last_ssl: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, a_hat)
        if self.training:
            # Augmented view: random feature masking (data-level perturbation).
            x_aug = F.dropout(x, p=self.aug_drop)
            z_aug = self.encoder(x_aug, a_hat)
            self._last_ssl = self._contrastive(self.proj(z), self.proj(z_aug))
        return self.head(z)                                   # [B, N, F]

    def _contrastive(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE over node representations within the batch."""
        z1 = F.normalize(z1.reshape(-1, z1.shape[-1]), dim=-1)
        z2 = F.normalize(z2.reshape(-1, z2.shape[-1]), dim=-1)
        logits = z1 @ z2.t() / self.temperature
        labels = torch.arange(z1.shape[0], device=z1.device)
        return F.cross_entropy(logits, labels)

    def ssl_loss(self) -> torch.Tensor:
        """Auxiliary self-supervised loss from the latest training forward."""
        if self._last_ssl is None:
            return torch.tensor(0.0)
        return self._last_ssl
