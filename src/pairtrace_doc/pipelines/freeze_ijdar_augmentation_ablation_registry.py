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
        raise ValueError(f"ablation base config changed: {config_path}")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    return _deep_merge(base, override)


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


def _validate_probability_family(
    probabilities: dict[str, float], family: str
) -> None:
    expected = (
        {"clean": 0.25, "translation": 0.25, "affine": 0.25, "perspective": 0.25}
        if family == "projective"
        else {"clean": 0.25, "perturb": 0.75}
    )
    if probabilities != expected:
        raise ValueError(f"ablation probability family changed: {family}")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["runtime"]["holdout_read_allowed"] or config["runtime"]["training_allowed"]:
        raise ValueError("ablation freeze cannot read a holdout or train")
    protocol_path = _resolve(project_root, str(config["experiment"]["protocol"]))
    if _sha256(protocol_path) != str(config["experiment"]["expected_protocol_sha256"]):
        raise ValueError("augmentation-ablation protocol changed")
    manifest_path = _resolve(
        project_root, str(config["experiment"]["warm_start_manifest"])
    )
    if _sha256(manifest_path) != str(
        config["experiment"]["expected_warm_start_manifest_sha256"]
    ):
        raise ValueError("augmentation-ablation warm-start manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("holdout_read") is not False or manifest.get("paper_evidence") is not False:
        raise ValueError("augmentation-ablation manifest crosses evidence boundary")
    for record in manifest["records"] + manifest["existing_controls"]:
        artifact = _resolve(project_root, str(record["checkpoint"]))
        if _sha256(artifact) != str(record["sha256"]):
            raise ValueError(f"augmentation checkpoint changed: {artifact}")

    seeds = [int(value) for value in config["experiment"]["seeds"]]
    if seeds != [20260747, 20260763, 20260764]:
        raise ValueError("augmentation seed family changed")
    warm_by_seed = {int(record["seed"]): record for record in manifest["records"]}
    records: list[dict[str, Any]] = []
    outputs: set[str] = set()
    for specification in config["new_arms"]:
        slug = str(specification["slug"])
        application = str(specification["application"])
        probability_family = str(specification["probability_family"])
        preflight_path = project_root / str(
            config["templates"]["preflight_config"]
        ).format(slug=slug)
        preflight = _load_config(preflight_path, project_root)
        if int(preflight["training"]["epochs"]) * int(
            preflight["training"]["steps_per_epoch"]
        ) != 100:
            raise ValueError(f"ablation preflight budget changed: {slug}")
        if preflight["augmentation"]["transform_application"] != application:
            raise ValueError(f"ablation preflight application changed: {slug}")
        for seed in seeds:
            full_path = project_root / str(config["templates"]["full_config"]).format(
                slug=slug, seed=seed
            )
            full = _load_config(full_path, project_root)
            if str(full["experiment"]["ablation"]) != slug or int(
                full["experiment"]["seed"]
            ) != seed:
                raise ValueError(f"ablation slug/seed binding changed: {full_path}")
            if str(full["experiment"]["arm"]) != "explicit_9ch":
                raise ValueError("augmentation ablation architecture changed")
            if full["augmentation"]["transform_application"] != application:
                raise ValueError(f"ablation application changed: {slug}/{seed}")
            _validate_probability_family(
                full["augmentation"]["geometry_probabilities"], probability_family
            )
            if int(full["training"]["epochs"]) * int(
                full["training"]["steps_per_epoch"]
            ) != 3000:
                raise ValueError(f"ablation optimizer budget changed: {slug}/{seed}")
            warm = warm_by_seed[seed]
            if str(full["model"]["warm_start_checkpoint"]) != str(
                warm["checkpoint"]
            ) or str(full["model"]["warm_start_checkpoint_sha256"]) != str(
                warm["sha256"]
            ):
                raise ValueError(f"ablation warm start changed: {slug}/{seed}")
            checkpoint = str(full["paths"]["checkpoint"])
            if not checkpoint.startswith(
                "../autodl-tmp/pairtrace-doc/ijdar_revision/checkpoints/"
            ):
                raise ValueError("ablation checkpoint is not routed to scratch storage")
            if checkpoint in outputs:
                raise ValueError(f"duplicate ablation checkpoint path: {checkpoint}")
            outputs.add(checkpoint)
            records.append(
                {
                    "ablation": slug,
                    "application": application,
                    "seed": seed,
                    "optimizer_steps": 3000,
                    "warm_start_checkpoint": warm["checkpoint"],
                    "warm_start_checkpoint_sha256": warm["sha256"],
                    "config": str(full_path.relative_to(project_root)),
                    "config_sha256": _sha256(full_path),
                    "preflight_config": str(preflight_path.relative_to(project_root)),
                    "preflight_config_sha256": _sha256(preflight_path),
                    "checkpoint": checkpoint,
                    "holdout_read": False,
                    "paper_evidence": False,
                }
            )
    if len(records) != 18:
        raise ValueError("augmentation-ablation inventory changed")
    registry_path = _resolve(project_root, str(config["paths"]["registry"]))
    summary_path = _resolve(project_root, str(config["paths"]["summary"]))
    _write_jsonl(registry_path, records)
    summary = {
        "status": "augmentation_ablation_registry_frozen",
        "paper_evidence": False,
        "holdout_read": False,
        "training_started": False,
        "new_run_count": len(records),
        "new_arm_count": len({record["ablation"] for record in records}),
        "existing_control_count": len(manifest["existing_controls"]),
        "total_learned_method_count": len(records) + len(manifest["existing_controls"]),
        "seed_count": len(seeds),
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
