"""Mixture-of-Experts ensemble over spatiotemporal forecasting models.

Combines the six expert models with a gating scheme driven by
exponentially smoothed recent loss (Eq. 8):

    y_hat_t = sum_m w_m(t) * f_m(X_{t-T:t}, A)
    w_m(t)  = exp(-lambda * L_m(t)) / sum_j exp(-lambda * L_j(t))

where L_m(t) is expert m's exponentially weighted prediction error,
updated online whenever ground truth becomes available. Experts whose
recent predictions are accurate therefore receive larger weights,
keeping the ensemble stable under sensor noise and regime changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .cast import CaST
from .gman import GMAN
from .hydronet import HydroNet
from .st_ssl import STSSL
from .stg_mamba import STGMamba
from .stgcn import STGCN


@dataclass
class MoEConfig:
    lam: float = 5.0          # lambda: gating temperature on smoothed loss
    ema_alpha: float = 0.1    # smoothing factor for the per-expert loss EMA


def build_experts(
    num_features: int, num_nodes: int, window: int
) -> Dict[str, nn.Module]:
    """Instantiate the six expert models with the unified interface."""
    return {
        "CaST": CaST(num_features=num_features),
        "GMAN": GMAN(num_features=num_features, num_nodes=num_nodes,
                     window=window),
        "ST-SSL": STSSL(num_features=num_features),
        "STG-MAMBA": STGMamba(num_features=num_features),
        "STGCN": STGCN(num_features=num_features, window=window),
        "HydroNet": HydroNet(num_features=num_features),
    }


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        experts: Dict[str, nn.Module],
        cfg: Optional[MoEConfig] = None,
    ) -> None:
        super().__init__()
        self.names: List[str] = list(experts.keys())
        self.experts = nn.ModuleDict(experts)
        self.cfg = cfg or MoEConfig()
        # Exponentially smoothed loss L_m(t) per expert.
        self.register_buffer(
            "smoothed_loss", torch.zeros(len(self.names)), persistent=True
        )
        self._last_preds: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def weights(self) -> torch.Tensor:
        """Current gating weights w_m(t) (Eq. 8, second line)."""
        return torch.softmax(-self.cfg.lam * self.smoothed_loss, dim=0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        """Weighted ensemble prediction (Eq. 8, first line).

        Args:
            x: input window [B, T, N, F].
            a_hat: normalized adjacency [N, N].
        Returns:
            y_hat: [B, N, F].
        """
        preds = torch.stack(
            [self.experts[name](x, a_hat) for name in self.names], dim=0
        )                                                    # [M, B, N, F]
        self._last_preds = preds
        w = self.weights().view(-1, 1, 1, 1)
        return (w * preds).sum(dim=0)

    @torch.no_grad()
    def update_weights(self, y_true: torch.Tensor) -> None:
        """Update per-expert smoothed loss once ground truth arrives.

        Must be called after `forward` for the same window, with y_true
        of shape [B, N, F].
        """
        if self._last_preds is None:
            return
        a = self.cfg.ema_alpha
        errs = (self._last_preds - y_true.unsqueeze(0)).abs().mean(dim=(1, 2, 3))
        self.smoothed_loss.mul_(1.0 - a).add_(a * errs)
        self._last_preds = None

    # ------------------------------------------------------------------ #
    def state_summary(self) -> Dict[str, float]:
        """Human-readable weight report used by the LLM agent context."""
        w = self.weights().tolist()
        return {name: round(wi, 4) for name, wi in zip(self.names, w)}
