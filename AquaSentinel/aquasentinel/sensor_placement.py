"""Strategic sparse sensor deployment.

Implements the node importance score
    Score(v) = alpha * C_B(v) + beta * Hydraulic(v) + gamma * Risk(v)
and the constrained selection
    S* = argmax_{S, |S| <= k} sum Score(v)  s.t.  d(u, v) >= d_min
solved with a greedy algorithm that respects the minimum pairwise
hop-distance constraint to guarantee spatial spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .graph import PipelineNetwork


@dataclass
class PlacementConfig:
    alpha: float = 0.5      # weight of betweenness centrality
    beta: float = 0.3       # weight of hydraulic significance
    gamma: float = 0.2      # weight of risk factor
    k: int = 6              # sensor budget (20-30% of nodes in practice)
    d_min: int = 2          # minimum pairwise hop distance between sensors


def hydraulic_significance(
    mean_flow: Dict[str, float],
    pressure_range: Dict[str, float],
) -> Dict[str, float]:
    """Hydraulic(v) = mean flow volume * pressure (or depth) dynamic range.

    Both inputs come from historical records or a calibrated hydraulic
    model. Values are min-max normalized before combination.
    """
    keys = list(mean_flow.keys())
    q = np.array([mean_flow[k] for k in keys], dtype=np.float64)
    p = np.array([pressure_range[k] for k in keys], dtype=np.float64)

    def _norm(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

    h = _norm(q) * _norm(p)
    return dict(zip(keys, h.tolist()))


def node_scores(
    network: PipelineNetwork,
    hydraulic: Dict[str, float],
    risk: Dict[str, float],
    cfg: PlacementConfig,
) -> Dict[str, float]:
    """Combined importance score for every node (Eq. 3)."""
    cb = network.betweenness()
    scores = {}
    for v in network.node_ids:
        scores[v] = (
            cfg.alpha * cb.get(v, 0.0)
            + cfg.beta * hydraulic.get(v, 0.0)
            + cfg.gamma * risk.get(v, 0.0)
        )
    return scores


def select_sensor_nodes(
    network: PipelineNetwork,
    scores: Dict[str, float],
    cfg: PlacementConfig,
) -> List[str]:
    """Greedy solution of the constrained placement problem (Eq. 4).

    Nodes are visited in descending score order; a candidate is accepted
    only if it keeps at least d_min hops from every already-selected
    sensor, which enforces uniform spatial distribution and avoids
    clustered blind spots.
    """
    selected: List[str] = []
    for v in sorted(scores, key=scores.get, reverse=True):
        if len(selected) >= cfg.k:
            break
        ok = True
        for u in selected:
            d = network.shortest_hop_distance(u, v)
            if d is not None and d < cfg.d_min:
                ok = False
                break
        if ok:
            selected.append(v)
    return selected


def deploy(
    network: PipelineNetwork,
    mean_flow: Dict[str, float],
    pressure_range: Dict[str, float],
    risk: Optional[Dict[str, float]] = None,
    cfg: Optional[PlacementConfig] = None,
) -> List[str]:
    """End-to-end placement: scores then constrained greedy selection."""
    cfg = cfg or PlacementConfig()
    risk = risk or {v: 0.0 for v in network.node_ids}
    hyd = hydraulic_significance(mean_flow, pressure_range)
    scores = node_scores(network, hyd, risk, cfg)
    return select_sensor_nodes(network, scores, cfg)
