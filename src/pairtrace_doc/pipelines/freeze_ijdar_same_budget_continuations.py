from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(config_path: Path, project_root: Path) -> dict[str, Any]:
    override = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_value = override.pop("base_config", None)
    expected_base = override.pop("expected_base_config_sha256", None)
    if base_value is None:
        return override
    base_path = _resolve(project_root, str(base_value))
    if expected_base and _sha256(base_path) != str(expected_base):
        raise ValueError(f"continuation base config changed: {config_path}")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    return _deep_merge(base, override)


def _validate_warm_start_manifest(
    identity: dict[str, Any], project_root: Path, manifest: dict[str, Any]
) -> None:
    experiment = identity["experiment"]
    model = identity["model"]
    arm = str(experiment["arm"])
    seed = int(experiment["seed"])
    matches = [
        record
        for record in manifest["records"]
        if str(record["arm"]) == arm and int(record["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError("warm-start manifest arm/seed binding changed")
    record = matches[0]
    if str(record["checkpoint"]) != str(model["warm_start_checkpoint"]) or str(
        record["sha256"]
    ) != str(model["warm_start_checkpoint_sha256"]):
        raise ValueError("warm-start manifest checkpoint binding changed")
    if _sha256(_resolve(project_root, str(record["checkpoint"]))) != str(
        record["sha256"]
    ):
        raise ValueError("warm-start checkpoint changed")


def _without(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["runtime"]["holdout_read_allowed"] or config["runtime"]["training_allowed"]:
        raise ValueError("freeze step cannot read a holdout or train a model")
    protocol_path = _resolve(project_root, str(config["experiment"]["protocol"]))
    if _sha256(protocol_path) != str(config["experiment"]["expected_protocol_sha256"]):
        raise ValueError("same-budget protocol changed")
    manifest_path = _resolve(
        project_root, str(config["experiment"]["warm_start_manifest"])
    )
    if _sha256(manifest_path) != str(
        config["experiment"]["expected_warm_start_manifest_sha256"]
    ):
        raise ValueError("same-budget warm-start manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("holdout_read") is not False or manifest.get("paper_evidence") is not False:
        raise ValueError("warm-start manifest crosses the evidence boundary")

    seeds = [int(value) for value in config["experiment"]["seeds"]]
    if seeds != [20260747, 20260763, 20260764]:
        raise ValueError("same-budget seed family changed")
    records: list[dict[str, Any]] = []
    outputs: set[str] = set()
    for specification in config["arms"]:
        slug = str(specification["slug"])
        arm = str(specification["arm"])
        for seed in seeds:
            roundtrip_path = project_root / str(
                config["templates"]["roundtrip_config"]
            ).format(slug=slug, seed=seed)
            identity_path = project_root / str(
                config["templates"]["identity_config"]
            ).format(slug=slug, seed=seed)
            roundtrip = _load_config(roundtrip_path, project_root)
            identity = _load_config(identity_path, project_root)
            if str(roundtrip["experiment"]["arm"]) != arm or int(
                roundtrip["experiment"]["seed"]
            ) != seed:
                raise ValueError(f"round-trip arm/seed binding changed: {roundtrip_path}")
            if str(identity["experiment"]["arm"]) != arm or int(
                identity["experiment"]["seed"]
            ) != seed:
                raise ValueError(f"identity arm/seed binding changed: {identity_path}")
            _validate_warm_start_manifest(identity, project_root, manifest)
            for section in (
                "data",
                "preprocessing",
                "sampling",
                "training",
                "operating_point",
                "runtime",
            ):
                if roundtrip[section] != identity[section]:
                    raise ValueError(
                        f"same-budget recipes differ outside augmentation: {slug}/{seed}/{section}"
                    )
            if _without(roundtrip["model"], "architecture") != _without(
                identity["model"], "architecture"
            ):
                raise ValueError(f"model binding differs beyond label: {slug}/{seed}")
            differing_augmentation = {
                key
                for key in roundtrip["augmentation"]
                if roundtrip["augmentation"][key] != identity["augmentation"][key]
            }
            if differing_augmentation != {
                "geometry_probabilities",
                "transform_application",
            }:
                raise ValueError(f"augmentation-only contrast changed: {slug}/{seed}")
            if roundtrip["augmentation"]["transform_application"] != (
                "reference_roundtrip_only"
            ) or identity["augmentation"]["transform_application"] != (
                "reference_clean_identity_only"
            ):
                raise ValueError(f"continuation labels changed: {slug}/{seed}")
            for output_name in (
                "checkpoint",
                "epoch_log",
                "summary",
                "log",
                "prediction_records",
                "metrics",
            ):
                path_string = str(identity["paths"][output_name])
                if path_string in outputs:
                    raise ValueError(f"duplicate identity output path: {path_string}")
                outputs.add(path_string)
            records.append(
                {
                    "slug": slug,
                    "arm": arm,
                    "seed": seed,
                    "optimizer_steps_each": int(identity["training"]["epochs"])
                    * int(identity["training"]["steps_per_epoch"]),
                    "warm_start_checkpoint": identity["model"][
                        "warm_start_checkpoint"
                    ],
                    "warm_start_checkpoint_sha256": identity["model"][
                        "warm_start_checkpoint_sha256"
                    ],
                    "roundtrip_config": str(roundtrip_path.relative_to(project_root)),
                    "roundtrip_config_sha256": _sha256(roundtrip_path),
                    "identity_config": str(identity_path.relative_to(project_root)),
                    "identity_config_sha256": _sha256(identity_path),
                    "only_semantic_difference": sorted(differing_augmentation),
                    "holdout_read": False,
                    "paper_evidence": False,
                }
            )
    if len(records) != 15:
        raise ValueError("same-budget continuation inventory changed")
    registry_path = _resolve(project_root, str(config["paths"]["registry"]))
    summary_path = _resolve(project_root, str(config["paths"]["summary"]))
    _write_jsonl(registry_path, records)
    summary = {
        "status": "same_budget_identity_continuations_frozen",
        "paper_evidence": False,
        "holdout_read": False,
        "training_started": False,
        "record_count": len(records),
        "arm_count": len({record["arm"] for record in records}),
        "seed_count": len({record["seed"] for record in records}),
        "optimizer_steps_per_run": 3000,
        "protocol_sha256": _sha256(protocol_path),
        "warm_start_manifest_sha256": _sha256(manifest_path),
        "config_sha256": _sha256(config_path),
        "registry": str(registry_path.relative_to(project_root)),
        "registry_sha256": _sha256(registry_path),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
