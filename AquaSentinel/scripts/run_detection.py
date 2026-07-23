"""Real-time streaming anomaly detection.

Implements the deployed runtime loop:
  1. Load the trained MoE from checkpoints.
  2. For each incoming timestep, form a 12-step window from the previous
     11 timesteps plus the current reading and predict the next step.
  3. When the actual next reading arrives, compare it with the
     prediction through RTCA (instantaneous + cumulative errors).
  4. No alert while both errors stay under their adaptive thresholds.
  5. On persistent exceedance, confirm the anomaly, run causal
     flow-based localization, assess severity, and emit an LLM report.

Usage:
    python scripts/run_detection.py --config config/default.yaml \
        --stream data/stream_states.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aquasentinel import (  # noqa: E402
    LLMConfig, MixtureOfExperts, MoEConfig, PipelineNetwork, RTCAConfig,
    RTCADetector, ReportAgent, Scaler, assess, build_experts, localize,
)
from aquasentinel.dataset import load_state_csv  # noqa: E402


def load_moe(cfg, network, device):
    experts = build_experts(
        num_features=cfg["data"]["num_features"],
        num_nodes=network.num_nodes,
        window=cfg["training"]["window"],
    )
    moe = MixtureOfExperts(experts, MoEConfig(**cfg["moe"]))
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    moe.load_state_dict(torch.load(ckpt_dir / "moe.pt", map_location=device))
    moe.to(device).eval()
    sc = np.load(ckpt_dir / "scaler.npz")
    return moe, Scaler(mean=sc["mean"], std=sc["std"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--stream", required=True,
                        help="CSV of streaming network states (see DATA_FORMAT)")
    parser.add_argument("--feature", type=int, default=1,
                        help="feature index monitored by RTCA (default: depth)")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = PipelineNetwork.from_json(cfg["data"]["network_json"])
    a_hat = torch.tensor(network.normalized_adjacency(), device=device)
    moe, scaler = load_moe(cfg, network, device)

    detector = RTCADetector(network.node_ids, RTCAConfig(**cfg["rtca"]))
    agent = ReportAgent(
        llm_cfg=LLMConfig(**cfg["llm"]), prompt_dir=cfg["llm_prompt_dir"]
    )

    stream = load_state_csv(
        args.stream, num_nodes=network.num_nodes,
        num_features=cfg["data"]["num_features"],
    )
    window = cfg["training"]["window"]
    norm = scaler.transform(stream)

    active_alert = False
    for t in range(window, stream.shape[0]):
        x = torch.from_numpy(norm[t - window: t]).unsqueeze(0).to(device)
        y_hat = moe(x, a_hat)                                # [1, N, F]
        y_true = torch.from_numpy(norm[t]).unsqueeze(0).to(device)
        moe.update_weights(y_true)

        pred = scaler.inverse(y_hat.squeeze(0).cpu().numpy())
        actual = stream[t]
        reports = detector.step(
            {network.id_of(i): float(actual[i, args.feature])
             for i in range(network.num_nodes)},
            {network.id_of(i): float(pred[i, args.feature])
             for i in range(network.num_nodes)},
        )

        confirmed = set(detector.anomalous_nodes(reports))
        if confirmed and not active_alert:
            active_alert = True
            hypotheses = localize(network, confirmed)
            severity = assess(network, reports)
            detection = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timestep": t,
                "status": "ANOMALY",
                "realtime_anomalies": sorted(
                    v for v, r in reports.items() if r.exceeded
                ),
                "cumulative_anomalies": sorted(confirmed),
                "leak_hypotheses": [h.describe() for h in hypotheses],
                "severity": [vars(s) for s in severity],
            }
            context = {
                "num_nodes": network.num_nodes,
                "moe_weights": moe.state_summary(),
                "monitored_feature_index": args.feature,
            }
            print(agent.generate(detection, context))
            print("-" * 60)
        elif not confirmed:
            active_alert = False

    print("Stream processing finished.")


if __name__ == "__main__":
    main()
