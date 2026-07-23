"""Evaluation over the leakage scenario suite.

Runs the full detection pipeline on every scenario file (by default the
110-case suite: 22 conduits x 5 leakage types) and reports, per
scenario category and overall:
    * detection rate
    * average detection delay in timesteps (10 minutes per step)
    * proportion of cases detected within 10 timesteps
    * localization hit rate (source node inside the ground-truth
      conduit's endpoints), when ground truth is provided

Scenario files follow the naming convention
    leak_conduit{CID}_scenario{1..5}.csv
placed under the configured scenario directory, each accompanied by an
optional metadata JSON declaring the leak onset timestep and the
ground-truth conduit (see data/DATA_FORMAT.md).

Usage:
    python scripts/evaluate.py --config config/default.yaml \
        --scenario-dir data/scenarios
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aquasentinel import (  # noqa: E402
    MixtureOfExperts, MoEConfig, PipelineNetwork, RTCAConfig, RTCADetector,
    Scaler, build_experts, localize,
)
from aquasentinel.dataset import load_state_csv  # noqa: E402

SCENARIO_LABELS = {
    1: "Constant <5%",
    2: "Constant 5-15%",
    3: "Constant 15-25%",
    4: "Constant >25%",
    5: "Dynamic (0% to 35%)",
}


def run_scenario(path, meta, cfg, network, moe, scaler, a_hat, device, feature):
    """Returns (detected, delay, localization_hit)."""
    stream = load_state_csv(
        str(path), num_nodes=network.num_nodes,
        num_features=cfg["data"]["num_features"],
    )
    norm = scaler.transform(stream)
    window = cfg["training"]["window"]
    onset = int(meta.get("leak_onset_timestep", window))
    truth_nodes = set(meta.get("conduit_endpoints", []))

    detector = RTCADetector(network.node_ids, RTCAConfig(**cfg["rtca"]))
    for t in range(window, stream.shape[0]):
        x = torch.from_numpy(norm[t - window: t]).unsqueeze(0).to(device)
        y_hat = moe(x, a_hat)
        moe.update_weights(torch.from_numpy(norm[t]).unsqueeze(0).to(device))
        pred = scaler.inverse(y_hat.squeeze(0).cpu().numpy())

        reports = detector.step(
            {network.id_of(i): float(stream[t, i, feature])
             for i in range(network.num_nodes)},
            {network.id_of(i): float(pred[i, feature])
             for i in range(network.num_nodes)},
        )
        confirmed = set(detector.anomalous_nodes(reports))
        if confirmed and t >= onset:
            delay = t - onset
            hyps = localize(network, confirmed)
            hit = any(
                h.source_node in truth_nodes
                or (h.suspected_segment and set(h.suspected_segment) & truth_nodes)
                for h in hyps
            ) if truth_nodes else None
            return True, delay, hit
    return False, None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--scenario-dir", default="data/scenarios")
    parser.add_argument("--feature", type=int, default=1)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = PipelineNetwork.from_json(cfg["data"]["network_json"])
    a_hat = torch.tensor(network.normalized_adjacency(), device=device)

    experts = build_experts(cfg["data"]["num_features"], network.num_nodes,
                            cfg["training"]["window"])
    moe = MixtureOfExperts(experts, MoEConfig(**cfg["moe"]))
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    moe.load_state_dict(torch.load(ckpt_dir / "moe.pt", map_location=device))
    moe.to(device).eval()
    sc = np.load(ckpt_dir / "scaler.npz")
    scaler = Scaler(mean=sc["mean"], std=sc["std"])

    pattern = re.compile(r"leak_conduit(\w+)_scenario(\d)\.csv")
    results = defaultdict(list)

    for path in sorted(Path(args.scenario_dir).glob("leak_conduit*_scenario*.csv")):
        m = pattern.match(path.name)
        if not m:
            continue
        sid = int(m.group(2))
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        detected, delay, hit = run_scenario(
            path, meta, cfg, network, moe, scaler, a_hat, device, args.feature
        )
        results[sid].append({"detected": detected, "delay": delay, "loc": hit})
        print(f"{path.name}: detected={detected} delay={delay} loc_hit={hit}")

    # ------------------------------------------------------------------ #
    # Summary table in the same layout as the paper's results table.
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    header = (f"{'Leakage Scenario':<24}{'Detected':>10}{'Avg Delay':>12}"
              f"{'<=10 steps':>12}{'Det. Rate':>12}")
    print(header)
    print("-" * 78)
    all_cases = []
    for sid in sorted(results):
        cases = results[sid]
        all_cases.extend(cases)
        det = [c for c in cases if c["detected"]]
        delays = [c["delay"] for c in det]
        within10 = [c for c in det if c["delay"] is not None and c["delay"] <= 10]
        print(
            f"{SCENARIO_LABELS.get(sid, str(sid)):<24}"
            f"{'Yes' if len(det) == len(cases) else 'Partial':>10}"
            f"{(np.mean(delays) if delays else float('nan')):>12.1f}"
            f"{(100 * len(within10) / max(len(cases), 1)):>11.1f}%"
            f"{(100 * len(det) / max(len(cases), 1)):>11.1f}%"
        )
    det_all = [c for c in all_cases if c["detected"]]
    w10_all = [c for c in det_all if c["delay"] is not None and c["delay"] <= 10]
    print("-" * 78)
    print(
        f"{'Overall (' + str(len(all_cases)) + ' cases)':<24}{'':>10}{'':>12}"
        f"{(100 * len(w10_all) / max(len(all_cases), 1)):>11.1f}%"
        f"{(100 * len(det_all) / max(len(all_cases), 1)):>11.1f}%"
    )


if __name__ == "__main__":
    main()
