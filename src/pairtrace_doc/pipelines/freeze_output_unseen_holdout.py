from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _allocate(sizes: dict[tuple[str, ...], int], target: int, minimum_one: bool) -> dict[tuple[str, ...], int]:
    if target > sum(sizes.values()):
        raise ValueError("holdout target exceeds candidate capacity")
    allocation = {key: int(minimum_one) for key in sorted(sizes)}
    if sum(allocation.values()) > target:
        raise ValueError("holdout target is smaller than the number of strata")
    remaining = target - sum(allocation.values())
    capacities = {key: sizes[key] - allocation[key] for key in sizes}
    total_capacity = sum(capacities.values())
    fractional: dict[tuple[str, ...], float] = {}
    for key in sorted(sizes):
        ideal = remaining * capacities[key] / total_capacity if total_capacity else 0.0
        extra = min(capacities[key], math.floor(ideal))
        allocation[key] += extra
        fractional[key] = ideal - extra
    remaining = target - sum(allocation.values())
    while remaining:
        eligible = [key for key in sizes if allocation[key] < sizes[key]]
        eligible.sort(key=lambda key: (-fractional[key], allocation[key] / sizes[key], key))
        for key in eligible:
            if not remaining:
                break
            allocation[key] += 1
            remaining -= 1
    return allocation


def _stable_key(group: str, seed: int, namespace: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}|{namespace}|{group}".encode("utf-8")).hexdigest()
    return digest, group


def _output_record(
    source: dict[str, Any], role: str, sample_kind: str, freeze_id: str
) -> dict[str, Any]:
    authentic = sample_kind == "authentic"
    return {
        "record_id": f"output_unseen:{role}:{sample_kind}:{source['source_group_id']}",
        "source_group_id": source["source_group_id"],
        "source_sample_id": source["sample_id"],
        "evaluation_role": role,
        "sample_kind": sample_kind,
        "edition": "authentic" if authentic else source["edition"],
        "generator": "authentic" if authentic else source.get("assigned_tool"),
        "source_dataset": source.get("source_dataset"),
        "image": source["authentic"] if authentic else source["image"],
        "image_sha256": source["authentic_sha256"] if authentic else source["image_sha256"],
        "mask": None if authentic else source["mask"],
        "mask_sha256": None if authentic else source["mask_sha256"],
        "height": source["authentic_height"] if authentic else source["image_height"],
        "width": source["authentic_width"] if authentic else source["image_width"],
        "model_or_threshold_selection_allowed": False,
        "selection_used_model_output": False,
        "paper_evidence": False,
        "holdout_freeze_id": freeze_id,
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["runtime"]["gpu_launch_authorized"] or config["runtime"]["method_training_authorized"]:
        raise ValueError("holdout freeze cannot authorize GPU use or method training")
    if config["selection"]["selection_uses_model_output"]:
        raise ValueError("output-unseen holdout selection must be metadata-only")

    input_config = config["input"]
    frozen_path = _resolve(project_root, input_config["frozen_split_manifest"])
    pilot_path = _resolve(project_root, input_config["diagnostic_pilot_manifest"])
    if _sha256(frozen_path) != input_config["expected_frozen_split_manifest_sha256"]:
        raise ValueError("frozen split manifest SHA-256 changed")
    if _sha256(pilot_path) != input_config["expected_diagnostic_pilot_manifest_sha256"]:
        raise ValueError("diagnostic pilot manifest SHA-256 changed")
    frozen_rows = _read_jsonl(frozen_path)
    pilot_rows = _read_jsonl(pilot_path)
    inspected_groups = {str(row["source_group_id"]) for row in pilot_rows}
    inspected_inputs: list[dict[str, Any]] = []
    for item in input_config["inspected_prediction_manifests"]:
        path = _resolve(project_root, item["path"])
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"inspected prediction manifest SHA-256 changed: {path}")
        rows = _read_jsonl(path)
        groups = {str(row["source_group_id"]) for row in rows}
        inspected_groups.update(groups)
        inspected_inputs.append(
            {
                "path": str(path.relative_to(project_root)),
                "sha256": item["sha256"],
                "rows": len(rows),
                "source_groups": len(groups),
            }
        )

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frozen_rows:
        group = row.get("source_group_id")
        if group:
            by_group[str(group)].append(row)
    selection = config["selection"]
    allowed_joins = set(selection["allowed_v2_join_statuses"])
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group, rows in by_group.items():
        if group in inspected_groups:
            continue
        v1 = sorted(
            (
                row
                for row in rows
                if row.get("edition") == selection["candidate_v1_edition"]
                and row.get("role") == selection["candidate_v1_role"]
                and row.get("valid")
                and not row.get("errors")
            ),
            key=lambda row: str(row["sample_id"]),
        )
        v2 = sorted(
            (
                row
                for row in rows
                if row.get("edition") == selection["paired_v2_edition"]
                and row.get("valid")
                and not row.get("errors")
                and row.get("join_status") in allowed_joins
            ),
            key=lambda row: str(row["sample_id"]),
        )
        if not v1 or not v2:
            continue
        joined = [row for row in v2 if row.get("joined_v1_sample_id") == v1[0]["sample_id"]]
        if joined:
            candidates.append((v1[0], joined[0]))

    fields = list(selection["stratify_by"])
    strata: dict[tuple[str, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in candidates:
        key = tuple(str(pair[0].get(field, "<missing>")) for field in fields)
        strata[key].append(pair)
    target = int(selection["holdout_groups"])
    allocation = _allocate(
        {key: len(values) for key, values in strata.items()},
        target,
        bool(selection["minimum_one_per_nonempty_stratum"]),
    )
    seed = int(config["experiment"]["seed"])
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    selected_counts: dict[str, int] = {}
    for key in sorted(strata):
        ordered = sorted(
            strata[key],
            key=lambda pair: _stable_key(
                str(pair[0]["source_group_id"]), seed, "output_unseen|" + "|".join(key)
            ),
        )
        chosen = ordered[: allocation[key]]
        selected.extend(chosen)
        selected_counts["|".join(key)] = len(chosen)
    selected.sort(key=lambda pair: str(pair[0]["source_group_id"]))
    selected_groups = {str(pair[0]["source_group_id"]) for pair in selected}
    if len(selected) != target or len(selected_groups) != target:
        raise ValueError("output-unseen holdout selection count changed")
    if selected_groups & inspected_groups:
        raise ValueError("output-unseen holdout overlaps an inspected source group")

    protocol = {
        "frozen_split_manifest_sha256": _sha256(frozen_path),
        "diagnostic_pilot_manifest_sha256": _sha256(pilot_path),
        "inspected_prediction_manifests": inspected_inputs,
        "selection": selection,
        "seed": seed,
        "selected_source_groups": sorted(selected_groups),
    }
    freeze_id = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_rows: list[dict[str, Any]] = []
    for v1, v2 in selected:
        output_rows.extend(
            [
                _output_record(v1, "in_domain_test", "forged", freeze_id),
                _output_record(v2, "generator_holdout", "forged", freeze_id),
                _output_record(v1, "final_test", "authentic", freeze_id),
            ]
        )
    record_ids = [row["record_id"] for row in output_rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("new holdout contains duplicate record IDs")

    output_path = _resolve(project_root, config["paths"]["output_manifest"])
    summary_path = _resolve(project_root, config["paths"]["output_summary"])
    _write_jsonl(output_path, output_rows)
    summary = {
        "experiment": config["experiment"],
        "status": "frozen_output_unseen_no_training_authorized",
        "paper_evidence": False,
        "gpu_used": False,
        "method_training_authorized": False,
        "holdout_freeze_id": freeze_id,
        "candidate_pairs": len(candidates),
        "selected_source_groups": len(selected_groups),
        "selected_strata": selected_counts,
        "selection_used_model_output": False,
        "inspected_source_group_overlap": 0,
        "inspected_prediction_manifests": inspected_inputs,
        "input": {
            "frozen_split_manifest": str(frozen_path.relative_to(project_root)),
            "frozen_split_manifest_sha256": _sha256(frozen_path),
            "diagnostic_pilot_manifest": str(pilot_path.relative_to(project_root)),
            "diagnostic_pilot_manifest_sha256": _sha256(pilot_path),
        },
        "output": {
            "path": str(output_path.relative_to(project_root)),
            "sha256": _sha256(output_path),
            "rows": len(output_rows),
            "counts": dict(
                sorted(
                    Counter(
                        f"{row['evaluation_role']}:{row['sample_kind']}" for row in output_rows
                    ).items()
                )
            ),
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
