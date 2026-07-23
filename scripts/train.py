"""Train the six spatiotemporal experts and initialize the MoE ensemble.

Usage:
    python scripts/train.py --config config/default.yaml

Expects (see data/DATA_FORMAT.md):
    * a network topology JSON
    * a normal-operation state CSV covering all nodes, produced either
      by the physics augmentation module from raw sensor exports or by
      a calibrated hydraulic model.
Outputs per-expert checkpoints, the fitted scaler, and the MoE state
under the configured checkpoint directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aquasentinel import (  # noqa: E402
    MixtureOfExperts, MoEConfig, PipelineNetwork, Scaler,
    SlidingWindowDataset, build_experts, train_val_test_split,
)
from aquasentinel.dataset import load_state_csv  # noqa: E402


def train_one_expert(name, model, train_loader, val_loader, a_hat, cfg, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    best_val, best_state = float("inf"), None

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x, a_hat)
            loss = torch.nn.functional.l1_loss(pred, y[:, 0])
            # ST-SSL contributes its auxiliary self-supervised loss.
            if hasattr(model, "ssl_loss"):
                loss = loss + cfg["training"]["ssl_weight"] * model.ssl_loss()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            errs = [
                torch.nn.functional.l1_loss(model(x.to(device), a_hat),
                                            y[:, 0].to(device)).item()
                for x, y in val_loader
            ]
        val = float(np.mean(errs)) if errs else float("inf")
        if val < best_val:
            best_val, best_state = val, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(f"[{name}] epoch {epoch + 1}: val MAE = {val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = PipelineNetwork.from_json(cfg["data"]["network_json"])
    a_hat = torch.tensor(network.normalized_adjacency(), device=device)

    data = load_state_csv(
        cfg["data"]["normal_states_csv"],
        num_nodes=network.num_nodes,
        num_features=cfg["data"]["num_features"],
    )
    train_raw, val_raw, _ = train_val_test_split(data)
    scaler = Scaler.fit(train_raw)
    window = cfg["training"]["window"]

    make_loader = lambda arr, shuffle: DataLoader(  # noqa: E731
        SlidingWindowDataset(scaler.transform(arr), window=window, horizon=1),
        batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
    )
    train_loader = make_loader(train_raw, True)
    val_loader = make_loader(val_raw, False)

    experts = build_experts(
        num_features=cfg["data"]["num_features"],
        num_nodes=network.num_nodes,
        window=window,
    )

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    val_scores = {}
    for name, model in experts.items():
        model, val = train_one_expert(
            name, model, train_loader, val_loader, a_hat, cfg, device
        )
        torch.save(model.state_dict(), ckpt_dir / f"{name}.pt")
        val_scores[name] = val

    # Initialize MoE with smoothed losses seeded from validation MAE so
    # stronger experts start with larger gating weights.
    moe = MixtureOfExperts(experts, MoEConfig(**cfg["moe"]))
    with torch.no_grad():
        moe.smoothed_loss.copy_(
            torch.tensor([val_scores[n] for n in moe.names])
        )
    torch.save(moe.state_dict(), ckpt_dir / "moe.pt")
    np.savez(ckpt_dir / "scaler.npz", mean=scaler.mean, std=scaler.std)

    print("Initial MoE weights:", moe.state_summary())
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
