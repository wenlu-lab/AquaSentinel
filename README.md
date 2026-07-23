# AquaSentinel

Code for **"AquaSentinel: Next-Generation AI System Integrating Sensor
Networks for Urban Underground Water Pipeline Anomaly Detection via
Collaborative MoE-LLM Agent Architecture"** (AAAI 2026, IAAI Technical
Track on Emerging Applications of AI).

AquaSentinel is a physics-informed AI system for real-time anomaly
detection in urban underground water pipeline networks. It achieves
network-wide monitoring from sparse sensor deployments (20-30% node
coverage) by combining strategic sensor placement, physics-based
virtual sensors, a Mixture-of-Experts ensemble of spatiotemporal graph
neural networks, the RTCA dual-threshold detector, causal flow-based
leak localization, and an LLM agent for actionable reporting.

## System Overview

The pipeline mirrors the architecture described in the paper:

1. **Strategic sensor deployment** (`aquasentinel/sensor_placement.py`)
   scores nodes by betweenness centrality, hydraulic significance, and
   risk, then greedily selects a sensor set under a minimum-distance
   constraint.
2. **Physics-based state augmentation** (`aquasentinel/physics.py`)
   propagates sparse measurements to unmonitored nodes by solving a
   physics-constrained optimization enforcing mass conservation at
   junctions and Hazen-Williams energy conservation along pipes,
   creating virtual sensors across the network.
3. **Spatiotemporal prediction via MoE** (`aquasentinel/moe.py`)
   ensembles six expert forecasters: CaST, GMAN, ST-SSL, STG-MAMBA,
   STGCN, and HydroNet (one file per expert). A gating network weights
   experts by exponentially smoothed recent loss.
4. **RTCA detection** (`aquasentinel/rtca.py`) monitors instantaneous
   and cumulative relative errors against adaptive EMA thresholds
   (k1 = 2.5, k2 = 3.0) and confirms anomalies only after persistent
   exceedance, suppressing transient false positives.
5. **Causal localization** (`aquasentinel/localization.py`) traces
   confirmed anomalies upstream; source nodes are anomalous nodes with
   no anomalous ancestor, and the leaking segment is bounded by the
   nearest normal upstream neighbor.
6. **Severity and prioritization** (`aquasentinel/severity.py`)
   classifies Critical / Major / Minor and ranks maintenance priority
   by confidence, centrality, and downstream impact.
7. **Intelligent report generation** (`aquasentinel/llm_agent.py`)
   assembles detection results, network context, and history into a
   prompt and calls an LLM to produce field-ready reports. Prompts are
   loaded from `prompts/` at runtime and can be edited without code
   changes. Without an API key the agent falls back to a deterministic
   offline formatter.

## Repository Layout

```
AquaSentinel/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml          # all hyperparameters
├── prompts/
│   ├── report_system.txt     # LLM system prompt (loaded at runtime)
│   └── report_template.txt   # report template T
├── data/
│   └── DATA_FORMAT.md        # input data specification (no data shipped)
├── aquasentinel/
│   ├── __init__.py
│   ├── graph.py              # network topology, centrality, flow causality
│   ├── sensor_placement.py   # strategic sparse deployment
│   ├── physics.py            # virtual sensors via conservation laws
│   ├── dataset.py            # sliding windows, normalization, CSV loading
│   ├── cast.py               # expert: CaST
│   ├── gman.py               # expert: GMAN
│   ├── st_ssl.py             # expert: ST-SSL
│   ├── stg_mamba.py          # expert: STG-MAMBA
│   ├── stgcn.py              # expert: STGCN
│   ├── hydronet.py           # expert: HydroNet
│   ├── moe.py                # MoE ensemble with adaptive gating
│   ├── rtca.py               # dual-threshold anomaly detection
│   ├── localization.py       # causal flow-based leak localization
│   ├── severity.py           # severity classification and priority
│   └── llm_agent.py          # LLM report agent (prompts loaded from files)
└── scripts/
    ├── train.py              # train experts, initialize MoE
    ├── run_detection.py      # real-time streaming detection loop
    └── evaluate.py           # leakage scenario suite evaluation
```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.x are required. All models use dense
adjacency matrices, so no graph-learning extensions are needed.

## Data Preparation

This repository does not ship data. `data/DATA_FORMAT.md` specifies
the expected input formats: a network topology JSON, network-wide
state CSVs for normal operation and streaming, and the leakage
scenario suite (one CSV per conduit-scenario pair, five scenario types
per conduit). Network-wide states can be produced either by the
physics augmentation module from raw sparse sensor exports or by a
calibrated hydraulic model such as PCSWMM.

## Usage

Train the six experts and initialize the MoE ensemble:

```bash
python scripts/train.py --config config/default.yaml
```

Run real-time detection on a state stream (prediction, RTCA
monitoring, localization, severity assessment, and LLM reporting):

```bash
export LLM_API_KEY=...   # optional; offline formatter used if absent
python scripts/run_detection.py --config config/default.yaml \
    --stream data/stream_states.csv
```

Evaluate on the leakage scenario suite and print the per-scenario
summary table (detection rate, average delay, proportion detected
within 10 timesteps):

```bash
python scripts/evaluate.py --config config/default.yaml \
    --scenario-dir data/scenarios
```

## Configuration

All hyperparameters live in `config/default.yaml`, including the RTCA
thresholds (k1 = 2.5, k2 = 3.0), the cumulative window W, persistence
T, the MoE gating temperature, sensor placement weights and budget,
and the LLM backend (Anthropic or any OpenAI-compatible endpoint).

## Citation

```bibtex
@article{guo2026aquasentinel,
  title   = {AquaSentinel: Next-Generation AI System Integrating Sensor
             Networks for Urban Underground Water Pipeline Anomaly
             Detection via Collaborative MoE-LLM Agent Architecture},
  author  = {Guo, Qiming and Khatri, Bishal and Sun, Wenbo and
             Tang, Jinwen and Zhang, Hua and Wang, Wenlu},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {47},
  pages   = {40265--40271},
  year    = {2026},
  doi     = {10.1609/aaai.v40i47.41464}
}
```

## Acknowledgments

This work is supported by NSF award No. 2318641.
