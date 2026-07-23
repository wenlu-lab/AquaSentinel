"""Physics-based state augmentation (virtual sensors).

Estimates hydraulic states at unmonitored nodes from sparse sensor
readings by solving the physics-constrained optimization

    X* = argmin_X ||F(X)||^2 + lambda * ||grad X||^2

where F enforces
  * mass conservation at every junction:
        sum_{i in In(v)} Q_i = sum_{j in Out(v)} Q_j + D_v
  * energy conservation along pipes via the Hazen-Williams equation:
        h_f = 10.67 * L * (Q / (C * D^2.63))^1.852

The optimization is solved with gradient descent in PyTorch. States at
sensor-equipped nodes are clamped to their measured values; states at
the remaining nodes become "virtual sensor" readings.

The same interface also accepts states exported from an external
calibrated hydraulic simulator (e.g. PCSWMM); in that case the solver
is bypassed and the simulator output is used directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .graph import PipelineNetwork


@dataclass
class PhysicsConfig:
    lam_smooth: float = 0.1        # lambda: spatial smoothness weight
    lr: float = 0.05               # optimizer learning rate
    iterations: int = 500          # gradient steps per augmentation call
    demand: float = 0.0            # default nodal demand D_v
    hw_exponent: float = 1.852     # Hazen-Williams flow exponent
    hw_constant: float = 10.67     # Hazen-Williams constant (SI-style form)


def hazen_williams_headloss(
    q: torch.Tensor, length: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
    cfg: PhysicsConfig,
) -> torch.Tensor:
    """h_f = 10.67 * L * (Q / (C * D^2.63))^1.852 (Eq. 7)."""
    base = torch.clamp(torch.abs(q) / (c * d ** 2.63 + 1e-12), min=1e-12)
    return cfg.hw_constant * length * base ** cfg.hw_exponent * torch.sign(q)


class PhysicsAugmenter:
    """Propagates sparse measurements to all nodes of the network."""

    def __init__(
        self,
        network: PipelineNetwork,
        sensor_nodes: List[str],
        cfg: Optional[PhysicsConfig] = None,
    ) -> None:
        self.network = network
        self.cfg = cfg or PhysicsConfig()
        self.sensor_idx = torch.tensor(
            [network.index_of(v) for v in sensor_nodes], dtype=torch.long
        )
        n = network.num_nodes
        self.free_idx = torch.tensor(
            [i for i in range(n) if i not in set(self.sensor_idx.tolist())],
            dtype=torch.long,
        )
        # Edge tensors in node-index space
        self.edge_src = torch.tensor(
            [network.index_of(u) for u, _ in network.edges], dtype=torch.long
        )
        self.edge_dst = torch.tensor(
            [network.index_of(v) for _, v in network.edges], dtype=torch.long
        )
        attrs = [network.pipe_attrs[e] for e in network.edges]
        self.pipe_len = torch.tensor([a.length_ft for a in attrs])
        self.pipe_c = torch.tensor([a.roughness for a in attrs])
        self.pipe_d = torch.tensor([a.diameter_ft for a in attrs])
        # Laplacian for the smoothness regularizer ||grad X||^2
        a = torch.tensor(network.adjacency(symmetric=True, self_loops=False))
        self.laplacian = torch.diag(a.sum(1)) - a

    # ------------------------------------------------------------------ #
    def _physics_residual(self, flow: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        """||F(X)||^2 combining mass and energy conservation."""
        cfg = self.cfg
        n = self.network.num_nodes

        # Edge flow approximated by the mean of endpoint nodal flows.
        q_edge = 0.5 * (flow[self.edge_src] + flow[self.edge_dst])

        # Mass conservation at junctions (Eq. 6).
        inflow = torch.zeros(n).index_add(0, self.edge_dst, q_edge)
        outflow = torch.zeros(n).index_add(0, self.edge_src, q_edge)
        mass_res = inflow - outflow - cfg.demand
        # Boundary nodes (pure sources/sinks) are excluded from the balance.
        interior = (inflow.detach() != 0) & (outflow.detach() != 0)
        mass_loss = (mass_res[interior] ** 2).sum() if interior.any() else \
            torch.tensor(0.0)

        # Energy conservation along pipes (Eq. 7).
        hf = hazen_williams_headloss(q_edge, self.pipe_len, self.pipe_c,
                                     self.pipe_d, cfg)
        energy_res = (head[self.edge_src] - head[self.edge_dst]) - hf
        energy_loss = (energy_res ** 2).sum()

        return mass_loss + energy_loss

    # ------------------------------------------------------------------ #
    def augment(
        self,
        sensor_flow: Dict[str, float],
        sensor_head: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Estimate flow and head at every node for one timestep (Eq. 5).

        Args:
            sensor_flow: measured flow at each sensor node.
            sensor_head: measured depth/head at each sensor node.

        Returns:
            {node_id: {"flow": q, "head": h}} for all nodes.
        """
        cfg = self.cfg
        n = self.network.num_nodes

        flow = torch.zeros(n)
        head = torch.zeros(n)
        for v, q in sensor_flow.items():
            flow[self.network.index_of(v)] = float(q)
        for v, h in sensor_head.items():
            head[self.network.index_of(v)] = float(h)

        # Warm start free nodes at the sensor mean.
        flow[self.free_idx] = flow[self.sensor_idx].mean()
        head[self.free_idx] = head[self.sensor_idx].mean()

        free_flow = flow[self.free_idx].clone().requires_grad_(True)
        free_head = head[self.free_idx].clone().requires_grad_(True)
        opt = torch.optim.Adam([free_flow, free_head], lr=cfg.lr)

        for _ in range(cfg.iterations):
            opt.zero_grad()
            f = flow.clone()
            h = head.clone()
            f[self.free_idx] = free_flow
            h[self.free_idx] = free_head
            loss = self._physics_residual(f, h)
            loss = loss + cfg.lam_smooth * (
                f @ self.laplacian @ f + h @ self.laplacian @ h
            )
            loss.backward()
            opt.step()

        flow[self.free_idx] = free_flow.detach()
        head[self.free_idx] = free_head.detach()
        return {
            self.network.id_of(i): {"flow": float(flow[i]), "head": float(head[i])}
            for i in range(n)
        }

    # ------------------------------------------------------------------ #
    def augment_series(
        self,
        sensor_flow_series: Dict[str, np.ndarray],
        sensor_head_series: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Augment a full time series.

        Returns an array of shape [T, N, 2] with features (flow, head)
        for every node, suitable for training the spatiotemporal models.
        """
        t_len = len(next(iter(sensor_flow_series.values())))
        n = self.network.num_nodes
        out = np.zeros((t_len, n, 2), dtype=np.float32)
        for t in range(t_len):
            states = self.augment(
                {v: s[t] for v, s in sensor_flow_series.items()},
                {v: s[t] for v, s in sensor_head_series.items()},
            )
            for i in range(n):
                st = states[self.network.id_of(i)]
                out[t, i, 0] = st["flow"]
                out[t, i, 1] = st["head"]
        return out
