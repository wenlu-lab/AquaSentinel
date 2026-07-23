"""Pipeline network graph utilities.

Represents the water pipeline network G = (V, E) with directed edges
following the flow direction. Provides adjacency matrices for the
spatiotemporal models, betweenness centrality for sensor placement,
and upstream/downstream topology queries for causal leak localization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


@dataclass
class PipeAttributes:
    """Physical attributes of a pipe (edge) used by the hydraulic model."""

    length_ft: float = 100.0          # L: pipe length
    diameter_ft: float = 1.0          # D: pipe diameter
    roughness: float = 130.0          # C: Hazen-Williams roughness coefficient


@dataclass
class PipelineNetwork:
    """Directed pipeline network with hydraulic attributes.

    Nodes are identified by string IDs (e.g. manhole IDs). Edges are
    directed (u -> v) along the nominal flow direction.
    """

    node_ids: List[str]
    edges: List[Tuple[str, str]]
    pipe_attrs: Dict[Tuple[str, str], PipeAttributes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._index = {nid: i for i, nid in enumerate(self.node_ids)}
        self._g = nx.DiGraph()
        self._g.add_nodes_from(self.node_ids)
        self._g.add_edges_from(self.edges)
        for e in self.edges:
            self.pipe_attrs.setdefault(e, PipeAttributes())

    # ------------------------------------------------------------------ #
    # Basic accessors
    # ------------------------------------------------------------------ #
    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    def index_of(self, node_id: str) -> int:
        return self._index[node_id]

    def id_of(self, index: int) -> str:
        return self.node_ids[index]

    # ------------------------------------------------------------------ #
    # Matrices for learning models
    # ------------------------------------------------------------------ #
    def adjacency(self, symmetric: bool = True, self_loops: bool = True) -> np.ndarray:
        """Dense adjacency matrix A in node-index order."""
        n = self.num_nodes
        a = np.zeros((n, n), dtype=np.float32)
        for u, v in self.edges:
            a[self._index[u], self._index[v]] = 1.0
            if symmetric:
                a[self._index[v], self._index[u]] = 1.0
        if self_loops:
            a += np.eye(n, dtype=np.float32)
        return a

    def normalized_adjacency(self) -> np.ndarray:
        """Symmetrically normalized adjacency D^{-1/2} (A + I) D^{-1/2}."""
        a = self.adjacency(symmetric=True, self_loops=True)
        d = a.sum(axis=1)
        d_inv_sqrt = np.power(np.maximum(d, 1e-12), -0.5)
        return (a * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]

    # ------------------------------------------------------------------ #
    # Topology queries used by placement and localization
    # ------------------------------------------------------------------ #
    def betweenness(self) -> Dict[str, float]:
        """Betweenness centrality C_B(v) over the directed graph."""
        return nx.betweenness_centrality(self._g, normalized=True)

    def upstream(self, node_id: str) -> Set[str]:
        """All ancestors of a node along the flow direction."""
        return nx.ancestors(self._g, node_id)

    def downstream(self, node_id: str) -> Set[str]:
        """All descendants of a node along the flow direction."""
        return nx.descendants(self._g, node_id)

    def immediate_upstream(self, node_id: str) -> List[str]:
        return list(self._g.predecessors(node_id))

    def shortest_hop_distance(self, u: str, v: str) -> Optional[int]:
        """Undirected hop distance, used for the d_min placement constraint."""
        ug = self._g.to_undirected()
        try:
            return nx.shortest_path_length(ug, u, v)
        except nx.NetworkXNoPath:
            return None

    def edge_between(self, u: str, v: str) -> Optional[Tuple[str, str]]:
        if self._g.has_edge(u, v):
            return (u, v)
        if self._g.has_edge(v, u):
            return (v, u)
        return None

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(cls, path: str) -> "PipelineNetwork":
        """Load a network description from a JSON file.

        Expected schema:
        {
          "nodes": ["J1", "J2", ...],
          "edges": [
            {"from": "J1", "to": "J2",
             "length_ft": 120.0, "diameter_ft": 1.0, "roughness": 130.0},
            ...
          ]
        }
        """
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        edges = [(e["from"], e["to"]) for e in spec["edges"]]
        attrs = {
            (e["from"], e["to"]): PipeAttributes(
                length_ft=float(e.get("length_ft", 100.0)),
                diameter_ft=float(e.get("diameter_ft", 1.0)),
                roughness=float(e.get("roughness", 130.0)),
            )
            for e in spec["edges"]
        }
        return cls(node_ids=list(spec["nodes"]), edges=edges, pipe_attrs=attrs)
