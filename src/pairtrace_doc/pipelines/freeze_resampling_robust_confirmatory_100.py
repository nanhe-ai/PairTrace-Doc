from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from pairtrace_doc.pipelines.freeze_generator_balanced_1000 import (
    _candidate_variants,
    _read_jsonl,
    _resolve,
    _select_stratified,
    _sha256,
    _write_json,
    _write_jsonl,
)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment = config["experiment"]
    protocol_path = _resolve(project_root, experiment["protocol"])
    if _sha256(protocol_path) != experiment["expected_protocol_sha256"]:
        raise ValueError("confirmatory protocol SHA-256 changed")
    runtime = config["runtime"]
    if any(
        bool(runtime.get(name))
        for name in (
            "gpu_launch_authorized",
            "model_training_authorized",
            "model_evaluation_authorized",
            "viewed_development_read_allowed",
            "confirmatory_image_read_allowed",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("confirmatory freeze crossed an evidence boundary")
    if experiment["paper_evidence"]:
        raise ValueError("confirmatory development freeze cannot be paper evidence")

    inputs = config["input"]
    input_paths: dict[str, Path] = {}
    for name in (
        "frozen_split_manifest",
        "prior_pilot_manifest",
        "generator_balanced_manifest",
        "first_pair_at_inference_manifest",
    ):
        path = _resolve(project_root, inputs[name])
        if _sha256(path) != inputs[f"expected_{name}_sha256"]:
            raise ValueError(f"frozen {name} SHA-256 changed")
        input_paths[name] = path
    master_rows = _read_jsonl(input_paths["frozen_split_manifest"])
    selection = config["selection"]
    if selection["selection_uses_model_output"]:
        raise ValueError("confirmatory selection must be metadata-only")
    variants = _candidate_variants(master_rows, selection)
    excluded_rows = []
    for name in (
        "prior_pilot_manifest",
        "generator_balanced_manifest",
        "first_pair_at_inference_manifest",
    ):
        excluded_rows.extend(_read_jsonl(input_paths[name]))
    excluded_groups = {
        str(row["source_group_id"])
        for row in excluded_rows
        if row.get("source_group_id")
    }
    remaining = {
        group: generator_rows
        for group, generator_rows in variants.items()
        if group not in excluded_groups
    }
    capacity = {
        generator: sum(generator in generator_rows for generator_rows in remaining.values())
        for generator in selection["capacity_generators"]
    }
    expected_capacity = {
        str(name): int(value)
        for name, value in selection["expected_remaining_capacity"].items()
    }
    if capacity != expected_capacity:
        raise ValueError(f"confirmatory remaining capacity changed: {capacity}")

    selected: list[tuple[str, dict[str, Any]]] = []
    selected_groups: set[str] = set()
    strata: list[dict[str, Any]] = []
    for index, specification in enumerate(selection["development_sequence"]):
        generator = str(specification["generator"])
        candidates = [
            generator_rows[generator]
            for group, generator_rows in remaining.items()
            if group not in selected_groups and generator in generator_rows
        ]
        chosen, stratum_counts = _select_stratified(
            candidates,
            int(specification["count"]),
            list(selection["stratify_by"]),
            int(experiment["seed"]),
            f"resampling_confirmatory|{index}|{generator}",
            True,
        )
        for row in chosen:
            selected.append((generator, row))
            selected_groups.add(str(row["source_group_id"]))
        strata.append(
            {"generator": generator, "count": len(chosen), "strata": stratum_counts}
        )
    if len(selected) != int(selection["expected_development_groups"]):
        raise ValueError("confirmatory selection count changed")
    if selected_groups & excluded_groups:
        raise ValueError("confirmatory selection overlaps an excluded group")
    payload = {
        "inputs": {name: _sha256(path) for name, path in input_paths.items()},
        "protocol_sha256": _sha256(protocol_path),
        "seed": int(experiment["seed"]),
        "selection": selection,
        "selected": sorted(
            (generator, str(row["sample_id"])) for generator, row in selected
        ),
    }
    freeze_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_rows = [
        {
            **row,
            "source_audit_role": row.get("role"),
            "role": "validation",
            "pilot_role": "validation",
            "resampling_confirmatory_role": "development",
            "selected_generator": generator,
            "selection_used_model_output": False,
            "paper_evidence": False,
            "resampling_confirmatory_freeze_id": freeze_id,
        }
        for generator, row in sorted(
            selected, key=lambda item: (item[0], str(item[1]["source_group_id"]))
        )
    ]
    output_counts = Counter(str(row["selected_generator"]) for row in output_rows)
    expected_counts = {
        str(item["generator"]): int(item["count"])
        for item in selection["development_sequence"]
    }
    if dict(output_counts) != expected_counts:
        raise ValueError("confirmatory generator balance changed")
    output_path = _resolve(project_root, config["paths"]["output_manifest"])
    summary_path = _resolve(project_root, config["paths"]["output_summary"])
    _write_jsonl(output_path, output_rows)
    summary = {
        "experiment": experiment,
        "status": "resampling_robust_confirmatory_100_frozen_unread",
        "paper_evidence": False,
        "selection_used_model_output": False,
        "image_or_model_output_read": False,
        "final_reserve_read": False,
        "freeze_id": freeze_id,
        "protocol_sha256": _sha256(protocol_path),
        "input": {
            name: {"path": str(path.relative_to(project_root)), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "excluded_source_groups": len(excluded_groups),
        "remaining_generator_capacity_before_selection": capacity,
        "selection": {
            "groups": len(output_rows),
            "generator_counts": dict(sorted(output_counts.items())),
            "source_dataset_counts": dict(
                sorted(Counter(str(row["source_dataset"]) for row in output_rows).items())
            ),
            "strata": strata,
        },
        "output": {
            "path": str(output_path.relative_to(project_root)),
            "sha256": _sha256(output_path),
        },
        "qwen_generalization_estimable": False,
        "unused_qwen_candidate_groups": capacity.get("qwen-inpaint", 0),
        "runtime": runtime,
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
