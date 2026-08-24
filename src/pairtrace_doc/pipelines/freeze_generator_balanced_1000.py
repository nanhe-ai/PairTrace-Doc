from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(identity: str, seed: int, namespace: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}|{namespace}|{identity}".encode("utf-8")
    ).hexdigest()
    return digest, identity


def _allocate(
    sizes: dict[tuple[str, ...], int], target: int, minimum_one: bool
) -> dict[tuple[str, ...], int]:
    nonempty = {key: size for key, size in sizes.items() if size > 0}
    if target < 0 or target > sum(nonempty.values()):
        raise ValueError("selection target is outside candidate capacity")
    allocation = {key: int(minimum_one) for key in sorted(nonempty)}
    if sum(allocation.values()) > target:
        raise ValueError("selection target is smaller than the number of strata")
    remaining = target - sum(allocation.values())
    capacity = {key: nonempty[key] - allocation[key] for key in nonempty}
    total_capacity = sum(capacity.values())
    fractional: dict[tuple[str, ...], float] = {}
    for key in sorted(nonempty):
        ideal = remaining * capacity[key] / total_capacity if total_capacity else 0.0
        extra = min(capacity[key], math.floor(ideal))
        allocation[key] += extra
        fractional[key] = ideal - extra
    remaining = target - sum(allocation.values())
    while remaining:
        eligible = [
            key for key in nonempty if allocation[key] < nonempty[key]
        ]
        if not eligible:
            raise ValueError("stratified selection exhausted candidate capacity")
        eligible.sort(
            key=lambda key: (
                -fractional[key],
                allocation[key] / nonempty[key],
                key,
            )
        )
        for key in eligible:
            if not remaining:
                break
            allocation[key] += 1
            remaining -= 1
    return allocation


def _select_stratified(
    candidates: list[dict[str, Any]],
    target: int,
    fields: list[str],
    seed: int,
    namespace: str,
    minimum_one: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = tuple(str(row.get(field, "<missing>")) for field in fields)
        strata[key].append(row)
    allocation = _allocate(
        {key: len(values) for key, values in strata.items()}, target, minimum_one
    )
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for key in sorted(strata):
        ordered = sorted(
            strata[key],
            key=lambda row: _stable_key(
                str(row["source_group_id"]), seed, namespace + "|" + "|".join(key)
            ),
        )
        chosen = ordered[: allocation[key]]
        selected.extend(chosen)
        counts["|".join(key)] = len(chosen)
    return selected, counts


def _candidate_variants(
    rows: list[dict[str, Any]], selection: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    v1_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not (
            row.get("edition") == selection["candidate_v1_edition"]
            and row.get("role") == selection["candidate_v1_role"]
            and row.get("split") == selection["candidate_v1_split"]
            and row.get("valid")
            and not row.get("errors")
            and row.get("source_group_id")
        ):
            continue
        group = str(row["source_group_id"])
        current = v1_rows.get(group)
        if current is None or str(row["sample_id"]) < str(current["sample_id"]):
            v1_rows[group] = row

    variants: dict[str, dict[str, dict[str, Any]]] = {
        group: {str(row["assigned_tool"]): row} for group, row in v1_rows.items()
    }
    v1_by_id = {str(row["sample_id"]): row for row in v1_rows.values()}
    allowed_joins = set(selection["allowed_v2_join_statuses"])
    for row in rows:
        if not (
            row.get("edition") == selection["paired_v2_edition"]
            and row.get("valid")
            and not row.get("errors")
            and row.get("join_status") in allowed_joins
        ):
            continue
        parent = v1_by_id.get(str(row.get("joined_v1_sample_id")))
        if parent is None:
            continue
        group = str(parent["source_group_id"])
        if str(row.get("source_group_id")) != group:
            continue
        generator = str(row["assigned_tool"])
        current = variants[group].get(generator)
        if current is None or str(row["sample_id"]) < str(current["sample_id"]):
            variants[group][generator] = row
    return variants


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    protocol_sha256 = _sha256(protocol_path)
    if protocol_sha256 != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("generator-balanced protocol SHA-256 changed")
    runtime = config["runtime"]
    if any(
        bool(runtime.get(key))
        for key in (
            "gpu_launch_authorized",
            "method_training_authorized",
            "viewed_diagnostic_read_allowed",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("data freeze cannot authorize training or holdout access")
    if config["selection"]["selection_uses_model_output"]:
        raise ValueError("new development freeze must be metadata-only")

    input_config = config["input"]
    frozen_path = _resolve(project_root, input_config["frozen_split_manifest"])
    prior_path = _resolve(project_root, input_config["prior_pilot_manifest"])
    if _sha256(frozen_path) != input_config["expected_frozen_split_manifest_sha256"]:
        raise ValueError("frozen split manifest SHA-256 changed")
    if _sha256(prior_path) != input_config["expected_prior_pilot_manifest_sha256"]:
        raise ValueError("prior pilot manifest SHA-256 changed")
    rows = _read_jsonl(frozen_path)
    prior_rows = _read_jsonl(prior_path)
    variants = _candidate_variants(rows, config["selection"])

    prior_roles_by_group: dict[str, set[str]] = defaultdict(set)
    for row in prior_rows:
        if row.get("source_group_id") and row.get("pilot_role"):
            prior_roles_by_group[str(row["source_group_id"])].add(
                str(row["pilot_role"])
            )
    previously_viewed_groups = set(prior_roles_by_group)
    selected_groups: set[str] = set()
    selected: list[tuple[str, str, dict[str, Any]]] = []
    selection_counts: list[dict[str, Any]] = []
    seed = int(config["experiment"]["seed"])
    selection = config["selection"]
    fields = list(selection["stratify_by"])
    for index, item in enumerate(selection["selection_sequence"]):
        role = str(item["role"])
        generator = str(item["generator"])
        candidates = [
            generator_rows[generator]
            for group, generator_rows in variants.items()
            if group not in selected_groups
            and generator in generator_rows
            and (
                not item.get("require_never_viewed", False)
                or group not in previously_viewed_groups
            )
            and (
                role != "train"
                or not (
                    prior_roles_by_group.get(group, set())
                    - {selection["prior_training_role"]}
                )
            )
        ]
        chosen, stratum_counts = _select_stratified(
            candidates,
            int(item["count"]),
            fields,
            seed,
            f"{index}|{role}|{generator}",
            bool(selection["minimum_one_per_nonempty_stratum"]),
        )
        for row in chosen:
            group = str(row["source_group_id"])
            selected_groups.add(group)
            selected.append((role, generator, row))
        selection_counts.append(
            {
                "role": role,
                "generator": generator,
                "count": len(chosen),
                "require_never_viewed": bool(
                    item.get("require_never_viewed", False)
                ),
                "strata": stratum_counts,
            }
        )

    protocol_payload = {
        "frozen_split_manifest_sha256": _sha256(frozen_path),
        "prior_pilot_manifest_sha256": _sha256(prior_path),
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "selection": selection,
        "selected": sorted(
            (role, generator, str(row["sample_id"]))
            for role, generator, row in selected
        ),
    }
    freeze_id = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    output_rows: list[dict[str, Any]] = []
    for role, generator, source in sorted(
        selected,
        key=lambda item: (item[0], item[1], str(item[2]["source_group_id"])),
    ):
        source_role = source.get("role")
        output_rows.append(
            {
                **source,
                "source_audit_role": source_role,
                "role": "validation" if role == "development" else "train",
                "pilot_role": "validation" if role == "development" else "train",
                "generator_balanced_role": role,
                "selected_generator": generator,
                "selection_used_model_output": False,
                "paper_evidence": False,
                "generator_balanced_freeze_id": freeze_id,
            }
        )

    train_groups = {
        str(row["source_group_id"])
        for row in output_rows
        if row["generator_balanced_role"] == "train"
    }
    development_groups = {
        str(row["source_group_id"])
        for row in output_rows
        if row["generator_balanced_role"] == "development"
    }
    if train_groups & development_groups:
        raise ValueError("new train and development source groups overlap")
    development_prior_overlap = len(development_groups & previously_viewed_groups)
    if development_prior_overlap:
        raise ValueError("new development split overlaps a prior pilot group")
    prior_nontraining_groups = {
        group
        for group, roles in prior_roles_by_group.items()
        if roles - {selection["prior_training_role"]}
    }
    train_prior_nontraining_overlap = len(train_groups & prior_nontraining_groups)
    if train_prior_nontraining_overlap:
        raise ValueError("new training split overlaps prior validation/test groups")

    expected_counts = {
        "train": int(selection["expected_train_groups"]),
        "development": int(selection["expected_development_groups"]),
    }
    role_counts = Counter(row["generator_balanced_role"] for row in output_rows)
    if dict(role_counts) != expected_counts:
        raise ValueError(
            f"role counts changed: {dict(role_counts)} != {expected_counts}"
        )
    if len(output_rows) != len(selected_groups):
        raise ValueError("generator-balanced manifest must contain one record per group")

    output_path = _resolve(project_root, config["paths"]["output_manifest"])
    summary_path = _resolve(project_root, config["paths"]["output_summary"])
    _write_jsonl(output_path, output_rows)
    summary = {
        "experiment": config["experiment"],
        "status": "generator_balanced_1000_and_new_development_frozen",
        "paper_evidence": False,
        "gpu_used": False,
        "method_training_authorized": False,
        "viewed_diagnostic_read": False,
        "final_reserve_read": False,
        "selection_used_model_output": False,
        "freeze_id": freeze_id,
        "protocol": {
            "path": str(protocol_path.relative_to(project_root)),
            "sha256": protocol_sha256,
        },
        "input": {
            "frozen_split_manifest": str(frozen_path.relative_to(project_root)),
            "frozen_split_manifest_sha256": _sha256(frozen_path),
            "prior_pilot_manifest": str(prior_path.relative_to(project_root)),
            "prior_pilot_manifest_sha256": _sha256(prior_path),
            "candidate_source_groups": len(variants),
        },
        "selection_sequence": selection_counts,
        "output": {
            "path": str(output_path.relative_to(project_root)),
            "sha256": _sha256(output_path),
            "rows": len(output_rows),
            "role_counts": dict(sorted(role_counts.items())),
            "generator_counts": {
                role: dict(
                    sorted(
                        Counter(
                            str(row["selected_generator"])
                            for row in output_rows
                            if row["generator_balanced_role"] == role
                        ).items()
                    )
                )
                for role in ("train", "development")
            },
            "source_dataset_counts": {
                role: dict(
                    sorted(
                        Counter(
                            str(row.get("source_dataset"))
                            for row in output_rows
                            if row["generator_balanced_role"] == role
                        ).items()
                    )
                )
                for role in ("train", "development")
            },
        },
        "leakage_checks": {
            "train_development_source_group_overlap": 0,
            "development_prior_pilot_source_group_overlap": development_prior_overlap,
            "train_prior_validation_or_test_source_group_overlap": (
                train_prior_nontraining_overlap
            ),
            "prior_training_groups_reused_for_training": len(
                train_groups
                & {
                    group
                    for group, roles in prior_roles_by_group.items()
                    if roles == {selection["prior_training_role"]}
                }
            ),
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
