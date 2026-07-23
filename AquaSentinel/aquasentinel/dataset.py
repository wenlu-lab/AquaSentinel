"""Dataset utilities for spatiotemporal forecasting.

Builds 12-step sliding windows over network-wide state tensors of shape
[T, N, F] (time, nodes, features), supporting both single-step and
multi-step (sequence-to-sequence) forecasting targets, matching the
training configuration used in the deployed system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Scaler:
    """Per-feature z-score normalization fitted on training data."""

    mean: np.ndarray  # shape [F]
    std: np.ndarray   # shape [F]

    @classmethod
    def fit(cls, data: np.ndarray) -> "Scaler":
        mean = data.reshape(-1, data.shape[-1]).mean(axis=0)
        std = data.reshape(-1, data.shape[-1]).std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


class SlidingWindowDataset(Dataset):
    """Sliding windows over [T, N, F] data.

    Each item is (x, y) with
        x: [window, N, F]   the past `window` timesteps
        y: [horizon, N, F]  the next `horizon` timesteps
    For single-step forecasting set horizon=1.
    """

    def __init__(
        self,
        data: np.ndarray,
        window: int = 12,
        horizon: int = 1,
    ) -> None:
        assert data.ndim == 3, "expected data of shape [T, N, F]"
        self.data = data.astype(np.float32)
        self.window = window
        self.horizon = horizon

    def __len__(self) -> int:
        return max(0, self.data.shape[0] - self.window - self.horizon + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.window]
        y = self.data[idx + self.window: idx + self.window + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def train_val_test_split(
    data: np.ndarray, ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split preserving temporal order."""
    t = data.shape[0]
    t1 = int(t * ratios[0])
    t2 = int(t * (ratios[0] + ratios[1]))
    return data[:t1], data[t1:t2], data[t2:]


def load_state_csv(path: str, num_nodes: int, num_features: int = 2) -> np.ndarray:
    """Load a flat CSV of network states into a [T, N, F] tensor.

    Expected columns: timestep, node_index, feature_0..feature_{F-1}
    (see data/DATA_FORMAT.md for the exact schema).
    """
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    t_len = int(raw[:, 0].max()) + 1
    out = np.zeros((t_len, num_nodes, num_features), dtype=np.float32)
    for row in raw:
        t, n = int(row[0]), int(row[1])
        out[t, n] = row[2: 2 + num_features]
    return out
