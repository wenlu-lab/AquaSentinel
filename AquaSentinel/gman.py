"""GMAN expert: Graph Multi-Attention Network (Zheng et al., AAAI 2020).

Compact implementation retaining the core mechanisms: spatial
self-attention across nodes, temporal self-attention across timesteps,
gated fusion of the two branches, and a learned spatio-temporal
embedding added to the input representation.

Unified expert interface:
    forward(x: [B, T, N, F], a_hat: [N, N]) -> y: [B, N, F]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, hidden: int, heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, N, H] -> attention over N
        b, t, n, d = h.shape
        q = h.reshape(b * t, n, d)
        out, _ = self.attn(q, q, q)
        return out.reshape(b, t, n, d)


class TemporalAttention(nn.Module):
    def __init__(self, hidden: int, heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, N, H] -> attention over T
        b, t, n, d = h.shape
        q = h.permute(0, 2, 1, 3).reshape(b * n, t, d)
        out, _ = self.attn(q, q, q)
        return out.reshape(b, n, t, d).permute(0, 2, 1, 3)


class GatedFusion(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.ws = nn.Linear(hidden, hidden)
        self.wt = nn.Linear(hidden, hidden)

    def forward(self, hs: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.ws(hs) + self.wt(ht))
        return z * hs + (1.0 - z) * ht


class STAttBlock(nn.Module):
    def __init__(self, hidden: int, heads: int = 4) -> None:
        super().__init__()
        self.sa = SpatialAttention(hidden, heads)
        self.ta = TemporalAttention(hidden, heads)
        self.fusion = GatedFusion(hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(self.sa(h), self.ta(h))
        return self.norm(h + fused)


class GMAN(nn.Module):
    def __init__(
        self,
        num_features: int = 2,
        num_nodes: int = 23,
        window: int = 12,
        hidden: int = 32,
        blocks: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(num_features, hidden)
        # Learned spatio-temporal embedding (replaces node2vec + time-of-day
        # one-hot from the original formulation with trainable tables).
        self.node_emb = nn.Parameter(torch.randn(num_nodes, hidden) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(window, hidden) * 0.02)
        self.blocks = nn.ModuleList([STAttBlock(hidden, heads) for _ in range(blocks)])
        self.head = nn.Linear(hidden, num_features)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        h = self.encoder(x)
        h = h + self.node_emb[None, None, :, :] + self.time_emb[None, :t, None, :]
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, -1])                # [B, N, F]
