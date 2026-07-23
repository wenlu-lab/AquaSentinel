# Data Format Specification

This repository ships without data. Prepare the following inputs to run
the pipeline.

## 1. Network topology (`data/network_topology.json`)

```json
{
  "nodes": ["J1", "J2", "J3"],
  "edges": [
    {"from": "J1", "to": "J2", "length_ft": 120.0, "diameter_ft": 1.0, "roughness": 130.0},
    {"from": "J2", "to": "J3", "length_ft": 95.0,  "diameter_ft": 1.0, "roughness": 130.0}
  ]
}
```

Edges are directed along the nominal flow direction. Pipe attributes
feed the Hazen-Williams term of the physics augmentation module.

## 2. Network state CSV (`data/normal_states.csv`, stream files, scenarios)

Flat CSV with header, one row per (timestep, node):

```
timestep,node_index,flow,depth
0,0,0.42,0.15
0,1,0.38,0.14
...
```

- `timestep`: integer starting at 0; native resolution is 10 minutes.
- `node_index`: integer index consistent with the order of `nodes` in
  the topology JSON.
- Feature columns in order: flow velocity (cfs), water depth (ft).

Network-wide states can be produced in two ways:
1. From raw sparse sensor exports via `PhysicsAugmenter.augment_series`,
   which fills unmonitored nodes with physics-based virtual sensors.
2. Exported directly from a calibrated hydraulic model (e.g. PCSWMM);
   in that case the simulator plays the role of the physics engine and
   the CSV is consumed as-is.

## 3. Leakage scenario suite (`data/scenarios/`)

One CSV per case, named:

```
leak_conduit{CID}_scenario{SID}.csv     SID in 1..5
```

Scenario IDs: 1 = constant <5%, 2 = constant 5-15%, 3 = constant
15-25%, 4 = constant >25%, 5 = dynamic leakage increasing from ~0.5%
to a maximum. The full suite covers every conduit under all five types.

Each CSV may be accompanied by a metadata JSON of the same stem:

```json
{
  "leak_onset_timestep": 144,
  "conduit_endpoints": ["J4", "J5"]
}
```

`conduit_endpoints` enables localization scoring in `scripts/evaluate.py`.

## 4. Raw sensor exports (optional)

Per-sensor time series used by the physics augmentation path. Any
tabular format is acceptable as long as it can be loaded into
`Dict[str, np.ndarray]` keyed by node ID (see `PhysicsAugmenter`).
