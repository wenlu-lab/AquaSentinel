"""Severity classification and maintenance prioritization.

Implements the rule-based severity assignment (Eq. 17)

    Critical  if Conf(v) > 0.9 and e_RT > 0.30
    Major     if Conf(v) > 0.7 and e_RT > 0.15
    Minor     otherwise

and the maintenance priority score (Eq. 18)

    Priority(v) = Conf(v) * C_B(v) * Impact(v)

where C_B is betweenness centrality and Impact reflects the downstream
population of the node (share of the network affected if it fails).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .graph import PipelineNetwork
from .rtca import NodeReport


@dataclass
class SeverityAssessment:
    node_id: str
    severity: str      # "Critical" | "Major" | "Minor"
    confidence: float
    e_rt: float
    priority: float


def classify_severity(report: NodeReport) -> str:
    """Eq. 17."""
    if report.confidence > 0.9 and report.e_rt > 0.30:
        return "Critical"
    if report.confidence > 0.7 and report.e_rt > 0.15:
        return "Major"
    return "Minor"


def impact_factor(network: PipelineNetwork, node_id: str) -> float:
    """Impact(v): fraction of the network downstream of v."""
    n = max(network.num_nodes - 1, 1)
    return len(network.downstream(node_id)) / n


def assess(
    network: PipelineNetwork, reports: Dict[str, NodeReport]
) -> List[SeverityAssessment]:
    """Assess every confirmed-anomalous node, sorted by priority."""
    cb = network.betweenness()
    out: List[SeverityAssessment] = []
    for v, r in reports.items():
        if not r.confirmed:
            continue
        priority = r.confidence * cb.get(v, 0.0) * impact_factor(network, v)
        out.append(
            SeverityAssessment(
                node_id=v,
                severity=classify_severity(r),
                confidence=r.confidence,
                e_rt=r.e_rt,
                priority=priority,
            )
        )
    out.sort(key=lambda s: s.priority, reverse=True)
    return out
