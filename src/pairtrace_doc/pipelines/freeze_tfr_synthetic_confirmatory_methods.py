from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from pairtrace_doc.pipelines.train_resampling_robust_teacher import (
    _load_config as _load_continuation_config,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _resolve,
    _sha256,
    _write_json,
    _write_jsonl,
)
from pairtrace_doc.pipelines.train_tfr_equal_budget_arm import (
    _load_config as _load_clean_config,
)


def _method_name(regime: str, slug: str, seed: int) -> str:
    return f"{regime}_{slug}_seed{seed}"


def _validate_checkpoint_binding(
    checkpoint_path: Path,
    *,
    checkpoint_sha256: str,
    config_sha256: str,
    protocol_sha256: str,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError(f"checkpoint digest changed: {checkpoint_path}")
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    bound_arm = str(metadata.get("arm", metadata.get("representation_arm", "")))
    bound_seed = int(metadata.get("seed", metadata.get("training_seed", -1)))
    if bound_arm != arm or bound_seed != seed:
        raise ValueError(f"checkpoint arm/seed binding changed: {checkpoint_path}")
    if str(metadata.get("config_sha256")) != config_sha256:
        raise ValueError(f"checkpoint config binding changed: {checkpoint_path}")
    if str(metadata.get("protocol_sha256")) != protocol_sha256:
        raise ValueError(f"checkpoint protocol binding changed: {checkpoint_path}")
    if metadata.get("selection_rule") != "fixed_final_epoch":
        raise ValueError(f"checkpoint selection rule changed: {checkpoint_path}")
    return metadata


def _training_record(
    *,
    project_root: Path,
    regime: str,
    slug: str,
    family: str,
    arm: str,
    seed: int,
    config_path: Path,
    summary_path: Path,
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not config_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"training artifact is missing: {config_path} / {summary_path}")
    config_sha256 = _sha256(config_path)
    merged = (
        _load_clean_config(config_path)
        if regime == "clean"
        else _load_continuation_config(config_path)
    )
    if str(merged["experiment"]["expected_protocol_sha256"]) != protocol_sha256:
        raise ValueError(f"training protocol binding changed: {config_path}")
    if str(merged["experiment"]["arm"]) != arm or int(
        merged["experiment"]["seed"]
    ) != seed:
        raise ValueError(f"training arm/seed config changed: {config_path}")
    expected_steps = int(merged["training"]["epochs"]) * int(
        merged["training"]["steps_per_epoch"]
    )
    if expected_steps != 3000:
        raise ValueError(f"training step budget changed: {config_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("optimizer_steps_completed", expected_steps)) != expected_steps:
        raise ValueError(f"training did not complete its step budget: {summary_path}")
    if int(summary.get("silent_failures", 0)) != 0:
        raise ValueError(f"training summary contains failures: {summary_path}")
    if bool(summary.get("holdout_read", False)):
        raise ValueError(f"training crossed the confirmation boundary: {summary_path}")
    if int(summary.get("validation_records_used", -1)) != 150 or int(
        summary.get("validation_unique_authentic_groups", -1)
    ) != 150:
        raise ValueError(f"training validation inventory changed: {summary_path}")
    validation_ap = float(
        summary.get("validation_source_group_macro_pixel_ap_model_resolution", float("nan"))
    )
    if not math.isfinite(validation_ap):
        raise ValueError(f"training validation metric is not finite: {summary_path}")
    operating_point = summary.get("operating_point")
    if not isinstance(operating_point, dict):
        raise ValueError(f"training operating point is missing: {summary_path}")
    validation_threshold = float(operating_point["threshold"])
    if not 0.0 <= validation_threshold <= 1.0 or float(
        operating_point["unique_authentic_group_macro_pixel_fpr"]
    ) > 0.01 + 1e-12:
        raise ValueError(f"training operating point changed: {summary_path}")
    if regime != "clean":
        if int(summary.get("validation_prediction_records", -1)) != 300:
            raise ValueError(f"continuation validation output is incomplete: {summary_path}")
        for key in ("prediction_records", "metrics"):
            output_path = _resolve(project_root, str(summary["outputs"][key]))
            if _sha256(output_path) != str(summary["outputs"][f"{key}_sha256"]):
                raise ValueError(f"continuation validation artifact changed: {output_path}")
    checkpoint_path = _resolve(project_root, str(summary["checkpoint"]))
    checkpoint_sha256 = str(summary["checkpoint_sha256"])
    metadata = _validate_checkpoint_binding(
        checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        arm=arm,
        seed=seed,
    )
    return (
        {
            "name": _method_name(regime, slug, seed),
            "kind": "learned",
            "family": family,
            "training_regime": regime,
            "arm": arm,
            "seed": seed,
            "checkpoint": str(checkpoint_path.relative_to(project_root)),
            "checkpoint_sha256": checkpoint_sha256,
            "training_config": str(config_path.relative_to(project_root)),
            "training_config_sha256": config_sha256,
            "training_summary": str(summary_path.relative_to(project_root)),
            "training_summary_sha256": _sha256(summary_path),
            "validation_threshold": validation_threshold,
            "validation_source_group_macro_pixel_ap": validation_ap,
        },
        merged,
        metadata,
    )


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["runtime"]["confirmation_read_allowed"]:
        raise ValueError("method-registry freeze must not read confirmation data")
    protocol_path = _resolve(project_root, str(config["experiment"]["protocol"]))
    protocol_sha256 = _sha256(protocol_path)
    if protocol_sha256 != str(config["experiment"]["expected_protocol_sha256"]):
        raise ValueError("method-registry protocol changed")
    seeds = [int(value) for value in config["experiment"]["seeds"]]
    if seeds != [20260747, 20260763, 20260764]:
        raise ValueError("method-registry seed family changed")
    arms = list(config["arms"])
    methods: list[dict[str, Any]] = []
    clean_bindings: dict[tuple[str, int], str] = {}
    merged_configs: dict[tuple[str, str, int], dict[str, Any]] = {}
    continuation_metadata: dict[tuple[str, str, int], dict[str, Any]] = {}
    for seed in seeds:
        for specification in arms:
            slug = str(specification["slug"])
            arm = str(specification["arm"])
            family = str(specification["family"])
            clean_config = project_root / str(
                config["templates"]["clean_config"]
            ).format(slug=slug, seed=seed)
            clean_summary = project_root / str(
                config["templates"]["clean_summary"]
            ).format(slug=slug, seed=seed)
            record, merged, metadata = _training_record(
                project_root=project_root,
                regime="clean",
                slug=slug,
                family=family,
                arm=arm,
                seed=seed,
                config_path=clean_config,
                summary_path=clean_summary,
                protocol_sha256=protocol_sha256,
            )
            methods.append(record)
            clean_bindings[(slug, seed)] = str(record["checkpoint_sha256"])
            merged_configs[("clean", slug, seed)] = merged
            continuation_metadata[("clean", slug, seed)] = metadata

            roundtrip_config = project_root / str(
                config["templates"]["roundtrip_config"]
            ).format(slug=slug, seed=seed)
            roundtrip_summary = project_root / str(
                config["templates"]["roundtrip_summary"]
            ).format(slug=slug, seed=seed)
            record, merged, metadata = _training_record(
                project_root=project_root,
                regime="roundtrip",
                slug=slug,
                family=family,
                arm=arm,
                seed=seed,
                config_path=roundtrip_config,
                summary_path=roundtrip_summary,
                protocol_sha256=protocol_sha256,
            )
            if str(metadata.get("warm_start_checkpoint_sha256")) != clean_bindings[
                (slug, seed)
            ]:
                raise ValueError("round-trip continuation does not bind its clean checkpoint")
            methods.append(record)
            merged_configs[("roundtrip", slug, seed)] = merged
            continuation_metadata[("roundtrip", slug, seed)] = metadata

        clean_continuation_config = project_root / str(
            config["templates"]["clean_continuation_config"]
        ).format(seed=seed)
        clean_continuation_summary = project_root / str(
            config["templates"]["clean_continuation_summary"]
        ).format(seed=seed)
        record, merged, metadata = _training_record(
            project_root=project_root,
            regime="clean_continuation",
            slug="9ch",
            family="pairtrace_9ch",
            arm="explicit_9ch",
            seed=seed,
            config_path=clean_continuation_config,
            summary_path=clean_continuation_summary,
            protocol_sha256=protocol_sha256,
        )
        if str(metadata.get("warm_start_checkpoint_sha256")) != clean_bindings[
            ("9ch", seed)
        ]:
            raise ValueError("clean continuation does not bind its clean checkpoint")
        methods.append(record)
        merged_configs[("clean_continuation", "9ch", seed)] = merged
        continuation_metadata[("clean_continuation", "9ch", seed)] = metadata

    for seed in seeds:
        roundtrip = merged_configs[("roundtrip", "9ch", seed)]
        clean_continuation = merged_configs[("clean_continuation", "9ch", seed)]
        for section in ("data", "preprocessing", "sampling", "training", "runtime"):
            if roundtrip[section] != clean_continuation[section]:
                raise ValueError(f"causal continuation differs outside augmentation: {section}")
        if roundtrip["model"] | {"architecture": clean_continuation["model"]["architecture"]} != clean_continuation["model"]:
            raise ValueError("causal continuation model binding differs beyond its label")
        differing_augmentation = {
            key
            for key in roundtrip["augmentation"]
            if roundtrip["augmentation"][key] != clean_continuation["augmentation"][key]
        }
        if differing_augmentation != {"geometry_probabilities", "transform_application"}:
            raise ValueError("causal continuation augmentation difference changed")

    methods.extend(
        [
            {
                "name": "raw_rgb_difference",
                "kind": "nonlearned",
                "family": "raw_rgb_difference",
                "training_regime": "nonlearned",
                "seed": None,
            },
            {
                "name": "ssim_distance",
                "kind": "nonlearned",
                "family": "ssim_distance",
                "training_regime": "nonlearned",
                "seed": None,
            },
        ]
    )
    comparisons: list[dict[str, Any]] = []
    competitor_slugs = [str(item["slug"]) for item in arms if item["slug"] != "9ch"]
    for seed in seeds:
        for regime in ("clean", "roundtrip"):
            left = _method_name(regime, "9ch", seed)
            for slug in competitor_slugs:
                right = _method_name(regime, slug, seed)
                comparisons.append(
                    {
                        "name": f"{left}_minus_{right}",
                        "left": left,
                        "right": right,
                    }
                )
        comparisons.append(
            {
                "name": f"roundtrip_minus_clean_continuation_seed{seed}",
                "left": _method_name("roundtrip", "9ch", seed),
                "right": _method_name("clean_continuation", "9ch", seed),
            }
        )
        for control in ("raw_rgb_difference", "ssim_distance"):
            comparisons.append(
                {
                    "name": f"clean_pairtrace_seed{seed}_minus_{control}",
                    "left": _method_name("clean", "9ch", seed),
                    "right": control,
                }
            )
    learned_method_count = sum(item["kind"] == "learned" for item in methods)
    if len(methods) != int(config["experiment"]["expected_method_count"]):
        raise ValueError("frozen method inventory count changed")
    if learned_method_count != int(
        config["experiment"]["expected_learned_method_count"]
    ):
        raise ValueError("frozen learned-method inventory count changed")
    if len(comparisons) != int(config["experiment"]["expected_comparison_count"]):
        raise ValueError("frozen comparison inventory count changed")
    paths = config["paths"]
    registry_path = _resolve(project_root, str(paths["method_registry"]))
    comparison_path = _resolve(project_root, str(paths["comparison_registry"]))
    summary_path = _resolve(project_root, str(paths["summary"]))
    _write_jsonl(registry_path, methods)
    _write_jsonl(comparison_path, comparisons)
    summary = {
        "status": "confirmatory_method_registry_frozen",
        "paper_evidence": False,
        "confirmation_read": False,
        "protocol_sha256": protocol_sha256,
        "config_sha256": _sha256(config_path),
        "registry_code_sha256": _sha256(Path(__file__).resolve()),
        "method_count": len(methods),
        "learned_method_count": learned_method_count,
        "comparison_count": len(comparisons),
        "seeds": seeds,
        "arms": [str(item["arm"]) for item in arms],
        "outputs": {
            "method_registry": str(registry_path.relative_to(project_root)),
            "method_registry_sha256": _sha256(registry_path),
            "comparison_registry": str(comparison_path.relative_to(project_root)),
            "comparison_registry_sha256": _sha256(comparison_path),
        },
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
