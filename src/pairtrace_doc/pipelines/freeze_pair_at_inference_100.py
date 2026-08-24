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
    protocol_sha256 = _sha256(protocol_path)
    if protocol_sha256 != experiment["expected_protocol_sha256"]:
        raise ValueError("pair-at-inference protocol SHA-256 changed")
    runtime = config["runtime"]
    forbidden_authorizations = (
        "gpu_launch_authorized",
        "method_training_authorized",
        "viewed_diagnostic_read_allowed",
        "final_reserve_read_allowed",
    )
    if any(bool(runtime.get(name)) for name in forbidden_authorizations):
        raise ValueError("pair-at-inference data freeze crossed an evidence boundary")
    selection = config["selection"]
    if selection["selection_uses_model_output"]:
        raise ValueError("pair-at-inference development freeze must be metadata-only")

    inputs = config["input"]
    input_paths: dict[str, Path] = {}
    for name in ("frozen_split_manifest", "prior_pilot_manifest", "current_manifest"):
        path = _resolve(project_root, inputs[name])
        if _sha256(path) != inputs[f"expected_{name}_sha256"]:
            raise ValueError(f"{name} SHA-256 changed")
        input_paths[name] = path

    master_rows = _read_jsonl(input_paths["frozen_split_manifest"])
    prior_rows = _read_jsonl(input_paths["prior_pilot_manifest"])
    current_rows = _read_jsonl(input_paths["current_manifest"])
    variants = _candidate_variants(master_rows, selection)
    excluded_groups = {
        str(row["source_group_id"])
        for row in prior_rows + current_rows
        if row.get("source_group_id")
    }
    remaining = {
        group: rows for group, rows in variants.items() if group not in excluded_groups
    }
    capacity = {
        generator: sum(generator in rows for rows in remaining.values())
        for generator in selection["capacity_generators"]
    }
    expected_capacity = {
        str(name): int(value)
        for name, value in selection["expected_remaining_capacity"].items()
    }
    if capacity != expected_capacity:
        raise ValueError(f"remaining generator capacity changed: {capacity}")

    seed = int(experiment["seed"])
    selected_groups: set[str] = set()
    selected: list[tuple[str, dict[str, Any]]] = []
    selection_counts: list[dict[str, Any]] = []
    fields = list(selection["stratify_by"])
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
            fields,
            seed,
            f"pair_at_inference|{index}|{generator}",
            bool(selection["minimum_one_per_nonempty_stratum"]),
        )
        for row in chosen:
            group = str(row["source_group_id"])
            selected_groups.add(group)
            selected.append((generator, row))
        selection_counts.append(
            {
                "generator": generator,
                "count": len(chosen),
                "strata": stratum_counts,
            }
        )

    protocol_payload = {
        "inputs": {name: _sha256(path) for name, path in input_paths.items()},
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "selection": selection,
        "selected": sorted(
            (generator, str(row["sample_id"])) for generator, row in selected
        ),
    }
    freeze_id = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    output_rows = []
    for generator, source in sorted(
        selected, key=lambda item: (item[0], str(item[1]["source_group_id"]))
    ):
        output_rows.append(
            {
                **source,
                "source_audit_role": source.get("role"),
                "role": "validation",
                "pilot_role": "validation",
                "pair_at_inference_role": "development",
                "selected_generator": generator,
                "selection_used_model_output": False,
                "paper_evidence": False,
                "pair_at_inference_freeze_id": freeze_id,
            }
        )

    expected_groups = int(selection["expected_development_groups"])
    if len(output_rows) != expected_groups or len(selected_groups) != expected_groups:
        raise ValueError("pair-at-inference development count changed")
    if selected_groups & excluded_groups:
        raise ValueError("pair-at-inference development overlaps a viewed group")
    generator_counts = Counter(str(row["selected_generator"]) for row in output_rows)
    expected_counts = {
        str(item["generator"]): int(item["count"])
        for item in selection["development_sequence"]
    }
    if dict(generator_counts) != expected_counts:
        raise ValueError("pair-at-inference generator counts changed")

    output_path = _resolve(project_root, config["paths"]["output_manifest"])
    summary_path = _resolve(project_root, config["paths"]["output_summary"])
    _write_jsonl(output_path, output_rows)
    summary = {
        "experiment": experiment,
        "status": "pair_at_inference_new_development_frozen",
        "paper_evidence": False,
        "gpu_used": False,
        "method_training_authorized": False,
        "selection_used_model_output": False,
        "viewed_diagnostic_read": False,
        "final_reserve_read": False,
        "freeze_id": freeze_id,
        "protocol": {
            "path": str(protocol_path.relative_to(project_root)),
            "sha256": protocol_sha256,
        },
        "input": {
            name: {
                "path": str(path.relative_to(project_root)),
                "sha256": _sha256(path),
            }
            for name, path in input_paths.items()
        },
        "candidate_source_groups_before_exclusion": len(variants),
        "excluded_prior_or_current_source_groups": len(excluded_groups),
        "remaining_source_groups": len(remaining),
        "remaining_generator_capacity": capacity,
        "selection_sequence": selection_counts,
        "output": {
            "path": str(output_path.relative_to(project_root)),
            "sha256": _sha256(output_path),
            "rows": len(output_rows),
            "generator_counts": dict(sorted(generator_counts.items())),
            "source_dataset_counts": dict(
                sorted(Counter(str(row.get("source_dataset")) for row in output_rows).items())
            ),
        },
        "leakage_checks": {
            "development_prior_or_current_source_group_overlap": 0,
            "duplicate_source_groups": 0,
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
