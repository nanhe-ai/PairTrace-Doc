# PairTrace-Doc

PairTrace-Doc is the configuration-driven research code accompanying
“Reference-Only Round-Trip Training for Registered-Reference Document Tamper
Localization.” It implements registered-reference document change
localization, reference-only geometric round-trip training, controlled
baselines, source-grouped evaluation, and reproducibility checks.

## Scope of this release

This release contains source code, paper-scale YAML configurations,
self-contained tests, toy inputs, permitted aggregate manifests, and
cryptographic checksums. It deliberately excludes third-party datasets,
restricted document pixels, credentials, prediction caches, and checkpoint
bytes whose redistribution has not been authorized.

The absence of restricted data and model weights does not change the paper's
reported evidence. Dataset access remains subject to the upstream providers'
terms, and checkpoint identities are represented only by approved metadata and
SHA-256 records where available.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

Optional dependencies are declared in `pyproject.toml` for selected baselines
and external-data readers.

## Self-contained check

The included release tests do not require restricted datasets or model
checkpoints:

```bash
pytest -q
python -m pairtrace_doc.pipelines.run_sanity --config configs/debug.yaml
```

The toy sanity outputs are schema and metric checks only and are never paper
evidence.

## Paper-scale reproduction boundary

The `configs/ijdar_*` files and frozen manifest metadata preserve the study's
seeds, dataset roles, thresholds, stopping rules, and artifact identities.
Experiments that depend on upstream datasets or non-redistributable checkpoint
bytes require users to obtain those artifacts independently and place them at
paths selected through the YAML configuration. Failures must remain explicit;
the pipelines do not silently replace or skip unavailable evidence.

See `DATA_AND_WEIGHTS.md` for the redistribution boundary and
`THIRD_PARTY.md` for dependency and upstream-license responsibilities.

## Authors

- Han Xiao
- Maorui Xue
- Yafeng Yang
- Hongcan Yan

Hongcan Yan is the corresponding author: `yanhongcan@ncst.edu.cn`.

## License

The eligible project code in this release is licensed under the Apache License
2.0. See `LICENSE`. This license does not override the separate terms that
apply to third-party datasets, checkpoints, or other excluded artifacts.
