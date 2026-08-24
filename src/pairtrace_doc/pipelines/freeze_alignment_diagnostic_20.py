from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from pairtrace_doc.pipelines.freeze_generator_balanced_1000 import (
    _read_jsonl,
    _resolve,
    _select_stratified,
    _sha256,
    _write_json,
    _write_jsonl,
)


def _select_rows(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    fields: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for index, (generator, count) in enumerate(sorted(counts.items())):
        candidates = [
            row for row in rows if str(row["selected_generator"]) == generator
        ]
        chosen, stratum_counts = _select_stratified(
            candidates,
            count,
            fields,
            seed,
            f"alignment_diagnostic|{index}|{generator}",
            True,
        )
        selected.extend(chosen)
        strata.append(
            {"generator": generator, "count": len(chosen), "strata": stratum_counts}
        )
    return selected, strata


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    experiment = config["experiment"]
    protocol_path = _resolve(project_root, experiment["protocol"])
    if _sha256(protocol_path) != experiment["expected_protocol_sha256"]:
        raise ValueError("alignment diagnostic protocol SHA-256 changed")
    runtime = config["runtime"]
    forbidden = (
        "gpu_launch_authorized",
        "method_training_authorized",
        "multi_seed_authorized",
        "unseen_development_read_allowed",
        "final_reserve_read_allowed",
    )
    if any(bool(runtime.get(name)) for name in forbidden):
        raise ValueError("alignment diagnostic freeze crossed an evidence boundary")
    if not runtime["viewed_method_development_read_allowed"]:
        raise ValueError("viewed method-development read was not explicitly authorized")
    if experiment["paper_evidence"]:
        raise ValueError("viewed alignment diagnostic cannot be paper evidence")

    input_config = config["input"]
    parent_path = _resolve(project_root, input_config["manifest"])
    if _sha256(parent_path) != input_config["expected_manifest_sha256"]:
        raise ValueError("pair-at-inference parent manifest SHA-256 changed")
    rows = _read_jsonl(parent_path)
    if len(rows) != int(input_config["expected_parent_groups"]):
        raise ValueError("pair-at-inference parent group count changed")
    if len({str(row["source_group_id"]) for row in rows}) != len(rows):
        raise ValueError("pair-at-inference parent contains duplicate groups")
    if any(bool(row.get("paper_evidence")) for row in rows):
        raise ValueError("parent manifest unexpectedly marks paper evidence")

    selection = config["selection"]
    if selection["selection_uses_model_output"]:
        raise ValueError("alignment diagnostic subset must be metadata-only")
    expected_parent_counts = {
        str(name): int(value)
        for name, value in input_config["expected_parent_generator_counts"].items()
    }
    parent_counts = Counter(str(row["selected_generator"]) for row in rows)
    if dict(parent_counts) != expected_parent_counts:
        raise ValueError(f"parent generator counts changed: {dict(parent_counts)}")
    selected_counts = {
        str(name): int(value)
        for name, value in selection["generator_counts"].items()
    }
    selected, strata = _select_rows(
        rows,
        selected_counts,
        list(selection["stratify_by"]),
        int(experiment["seed"]),
    )
    selected_groups = {str(row["source_group_id"]) for row in selected}
    if len(selected) != int(selection["expected_groups"]) or len(selected_groups) != len(selected):
        raise ValueError("alignment diagnostic selection count changed")
    if Counter(str(row["selected_generator"]) for row in selected) != Counter(selected_counts):
        raise ValueError("alignment diagnostic generator balance changed")

    payload = {
        "parent_manifest_sha256": _sha256(parent_path),
        "protocol_sha256": _sha256(protocol_path),
        "seed": int(experiment["seed"]),
        "selection": selection,
        "selected_source_groups": sorted(selected_groups),
    }
    freeze_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_rows = [
        {
            **row,
            "alignment_diagnostic_role": "viewed_method_development",
            "alignment_diagnostic_freeze_id": freeze_id,
            "selection_used_model_output": False,
            "paper_evidence": False,
        }
        for row in sorted(selected, key=lambda item: str(item["source_group_id"]))
    ]
    output_path = _resolve(project_root, config["paths"]["output_manifest"])
    summary_path = _resolve(project_root, config["paths"]["output_summary"])
    _write_jsonl(output_path, output_rows)
    output_counts = Counter(str(row["selected_generator"]) for row in output_rows)
    summary = {
        "experiment": experiment,
        "status": "viewed_alignment_diagnostic_subset_frozen",
        "paper_evidence": False,
        "viewed_method_development_read": True,
        "unseen_development_read": False,
        "final_reserve_read": False,
        "selection_used_model_output": False,
        "freeze_id": freeze_id,
        "protocol": {
            "path": str(protocol_path.relative_to(project_root)),
            "sha256": _sha256(protocol_path),
        },
        "input": {
            "path": str(parent_path.relative_to(project_root)),
            "sha256": _sha256(parent_path),
            "groups": len(rows),
            "generator_counts": dict(sorted(parent_counts.items())),
        },
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
