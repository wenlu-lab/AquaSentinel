"""RTCA: Real-Time Cumulative Anomaly detection.

Dual-threshold monitoring per node (Eq. 9-14):

  * Real-time relative error
        e_RT(v) = |y - y_hat| / (y_hat + eps)
  * Cumulative error, mean of e_RT over a sliding window W
  * Adaptive thresholds from exponential moving statistics
        mu    <- (1 - alpha) mu + alpha e_RT
        sigma <- EMA of squared deviation
        tau_RT = mu + k1 * sigma,  tau_C = mu + k2 * sigma
  * Anomaly confirmed only when BOTH thresholds are exceeded for T
    consecutive timesteps, which suppresses transient fluctuations
    while remaining sensitive to persistent deviations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np


@dataclass
class RTCAConfig:
    eps: float = 1e-6         # numerical stabilizer in the relative error
    window: int = 6           # W: cumulative error window (timesteps)
    alpha: float = 0.05       # EMA factor for adaptive statistics
    k1: float = 2.5           # real-time threshold multiplier
    k2: float = 3.0           # cumulative threshold multiplier
    persistence: int = 3      # T: consecutive steps required to confirm
    warmup: int = 24          # steps before thresholds become active


@dataclass
class NodeState:
    mu: float = 0.0
    var: float = 1e-4
    history: Deque[float] = field(default_factory=deque)
    consecutive: int = 0
    steps_seen: int = 0


@dataclass
class NodeReport:
    node_id: str
    e_rt: float
    e_cum: float
    tau_rt: float
    tau_cum: float
    exceeded: bool
    confirmed: bool
    confidence: float


class RTCADetector:
    """Stateful per-node anomaly detector over the full network."""

    def __init__(self, node_ids: List[str], cfg: RTCAConfig | None = None) -> None:
        self.cfg = cfg or RTCAConfig()
        self.states: Dict[str, NodeState] = {v: NodeState() for v in node_ids}

    # ------------------------------------------------------------------ #
    def step(
        self, actual: Dict[str, float], predicted: Dict[str, float]
    ) -> Dict[str, NodeReport]:
        """Process one timestep of (actual, predicted) values per node.

        Returns a per-node report; `confirmed=True` marks nodes whose
        anomaly has persisted for T consecutive steps (Eq. 14).
        """
        cfg = self.cfg
        reports: Dict[str, NodeReport] = {}

        for v, y in actual.items():
            st = self.states[v]
            y_hat = predicted[v]

            # Eq. 9: instantaneous relative error.
            e_rt = abs(y - y_hat) / (abs(y_hat) + cfg.eps)

            # Eq. 10: cumulative error over window W.
            st.history.append(e_rt)
            if len(st.history) > cfg.window:
                st.history.popleft()
            e_cum = float(np.mean(st.history))

            # Eq. 11-13: adaptive thresholds via EMA statistics.
            sigma = st.var ** 0.5
            tau_rt = st.mu + cfg.k1 * sigma
            tau_cum = st.mu + cfg.k2 * sigma

            in_warmup = st.steps_seen < cfg.warmup
            exceeded = (not in_warmup) and (e_rt > tau_rt) and (e_cum > tau_cum)

            # Eq. 14: persistence-based confirmation.
            st.consecutive = st.consecutive + 1 if exceeded else 0
            confirmed = st.consecutive >= cfg.persistence

            # Update statistics AFTER thresholding, and only from
            # non-exceeding steps: the EMA statistics characterize
            # NORMAL behavior, so they are frozen while the error is
            # above threshold. Otherwise the anomalous regime would
            # inflate the thresholds and mask its own confirmation.
            if not exceeded:
                st.mu = (1 - cfg.alpha) * st.mu + cfg.alpha * e_rt
                st.var = (1 - cfg.alpha) * st.var + cfg.alpha * (e_rt - st.mu) ** 2
            st.steps_seen += 1

            # Detection confidence grows with persistence and margin.
            margin = 0.0 if tau_rt <= 0 else min(1.0, e_rt / (tau_rt + cfg.eps) - 1.0)
            confidence = min(
                1.0, 0.5 * (st.consecutive / cfg.persistence) + 0.5 * max(0.0, margin)
            ) if exceeded else 0.0

            reports[v] = NodeReport(
                node_id=v, e_rt=e_rt, e_cum=e_cum,
                tau_rt=tau_rt, tau_cum=tau_cum,
                exceeded=exceeded, confirmed=confirmed,
                confidence=confidence,
            )
        return reports

    # ------------------------------------------------------------------ #
    def anomalous_nodes(self, reports: Dict[str, NodeReport]) -> List[str]:
        return [v for v, r in reports.items() if r.confirmed]

    def reset(self) -> None:
        for v in self.states:
            self.states[v] = NodeState()
