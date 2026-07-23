"""Causal flow-based leak localization.

Given the set of anomalous nodes A produced by RTCA, the source nodes
are those with no anomalous upstream neighbor (Eq. 15):

    v* = { v in A : Upstream(v) ∩ A = ∅ }

Because anomalies propagate downstream with the flow, the most upstream
anomalous node marks the origin. The leaking pipe segment is then
localized between each source node and its nearest NORMAL upstream
neighbor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from .graph import PipelineNetwork


@dataclass
class LeakHypothesis:
    source_node: str
    upstream_normal_node: Optional[str]
    suspected_segment: Optional[Tuple[str, str]]

    def describe(self) -> str:
        if self.suspected_segment:
            u, v = self.suspected_segment
            return (f"Leak suspected on pipe segment {u} -> {v} "
                    f"(anomaly source node: {self.source_node}).")
        return (f"Leak suspected at or immediately upstream of node "
                f"{self.source_node} (no instrumented upstream neighbor).")


def find_source_nodes(
    network: PipelineNetwork, anomalous: Set[str]
) -> List[str]:
    """Source nodes: anomalous nodes with no anomalous ancestor (Eq. 15)."""
    return [
        v for v in anomalous
        if not (network.upstream(v) & anomalous)
    ]


def localize(
    network: PipelineNetwork, anomalous: Set[str]
) -> List[LeakHypothesis]:
    """Full localization: sources plus suspected pipe segments."""
    hypotheses: List[LeakHypothesis] = []
    for src in find_source_nodes(network, anomalous):
        # Nearest normal immediate-upstream neighbor bounds the segment.
        normal_up = [
            u for u in network.immediate_upstream(src) if u not in anomalous
        ]
        if normal_up:
            u = normal_up[0]
            hypotheses.append(
                LeakHypothesis(
                    source_node=src,
                    upstream_normal_node=u,
                    suspected_segment=network.edge_between(u, src),
                )
            )
        else:
            hypotheses.append(
                LeakHypothesis(
                    source_node=src,
                    upstream_normal_node=None,
                    suspected_segment=None,
                )
            )
    return hypotheses
