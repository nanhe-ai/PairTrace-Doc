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


def _select_training_rows(
    rows: list[dict[str, Any]],
    generator_counts: dict[str, int],
    fields: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for index, (generator, count) in enumerate(generator_counts.items()):
        candidates = [
            row
            for row in rows
            if row["pilot_role"] == "train"
            and str(row["selected_generator"]) == generator
        ]
        chosen, stratum_counts = _select_stratified(
            candidates,
            count,
            fields,
            seed,
            f"resampling_robust_teacher_100|{index}|{generator}",
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
        raise ValueError("resampling teacher protocol SHA-256 changed")
    runtime = config["runtime"]
    if any(
        bool(runtime.get(name))
        for name in (
            "gpu_launch_authorized",
            "model_training_authorized",
            "viewed_development_read_allowed",
            "unseen_development_read_allowed",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("training subset freeze crossed an evidence boundary")
    if experiment["paper_evidence"]:
        raise ValueError("training subset freeze cannot be paper evidence")

    input_config = config["input"]
    parent_path = _resolve(project_root, input_config["manifest"])
    if _sha256(parent_path) != input_config["expected_manifest_sha256"]:
        raise ValueError("generator-balanced parent manifest SHA-256 changed")
    rows = _read_jsonl(parent_path)
    train_rows = [row for row in rows if row["pilot_role"] == "train"]
    validation_rows = [row for row in rows if row["pilot_role"] == "validation"]
    if len(train_rows) != int(input_config["expected_parent_train_records"]):
        raise ValueError("parent training count changed")
    if len(validation_rows) != int(input_config["expected_parent_validation_records"]):
        raise ValueError("parent validation count changed")
    if {str(row["source_group_id"]) for row in train_rows} & {
        str(row["source_group_id"]) for row in validation_rows
    }:
        raise ValueError("parent train and validation groups overlap")

    selection = config["selection"]
    if selection["selection_uses_model_output"]:
        raise ValueError("training subset selection must be metadata-only")
    counts = {
        str(name): int(value)
        for name, value in selection["generator_counts"].items()
    }
    selected, strata = _select_training_rows(
        rows,
        counts,
        list(selection["stratify_by"]),
        int(experiment["seed"]),
    )
    groups = {str(row["source_group_id"]) for row in selected}
    if len(selected) != int(selection["expected_groups"]) or len(groups) != len(selected):
        raise ValueError("resampling teacher selection count changed")
    selected_counts = Counter(str(row["selected_generator"]) for row in selected)
    if dict(selected_counts) != counts:
        raise ValueError(f"resampling teacher generator counts changed: {dict(selected_counts)}")
    payload = {
        "parent_manifest_sha256": _sha256(parent_path),
        "protocol_sha256": _sha256(protocol_path),
        "seed": int(experiment["seed"]),
        "selection": selection,
        "selected_source_groups": sorted(groups),
    }
    freeze_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_rows = [
        {
            **row,
            "resampling_teacher_role": "train",
            "resampling_teacher_freeze_id": freeze_id,
            "selection_used_model_output": False,
            "paper_evidence": False,
        }
        for row in sorted(selected, key=lambda item: str(item["source_group_id"]))
    ]
    output_path = _resolve(project_root, config["paths"]["output_manifest"])
    summary_path = _resolve(project_root, config["paths"]["output_summary"])
    _write_jsonl(output_path, output_rows)
    summary = {
        "experiment": experiment,
        "status": f"resampling_robust_teacher_{len(output_rows)}_training_subset_frozen",
        "paper_evidence": False,
        "selection_used_model_output": False,
        "gpu_used": False,
        "viewed_development_read": False,
        "unseen_development_read": False,
        "final_reserve_read": False,
        "freeze_id": freeze_id,
        "protocol_sha256": _sha256(protocol_path),
        "input": {
            "path": str(parent_path.relative_to(project_root)),
            "sha256": _sha256(parent_path),
            "train_records": len(train_rows),
            "validation_records_not_selected": len(validation_rows),
        },
        "selection": {
            "records": len(output_rows),
            "generator_counts": dict(sorted(selected_counts.items())),
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
