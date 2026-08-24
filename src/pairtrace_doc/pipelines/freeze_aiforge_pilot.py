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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_order(
    rows: Iterable[dict[str, Any]], seed: int, namespace: str
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[str, str]:
        identity = str(row.get("source_group_id") or row.get("sample_id"))
        digest = hashlib.sha256(
            f"{seed}|{namespace}|{identity}".encode("utf-8")
        ).hexdigest()
        return digest, str(row.get("sample_id"))

    return sorted(rows, key=key)


def _one_pair_per_group(
    pairs: Iterable[tuple[dict[str, Any], dict[str, Any]]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for v1_row, v2_row in pairs:
        group = str(v1_row["source_group_id"])
        current = selected.get(group)
        identity = (str(v1_row["sample_id"]), str(v2_row["sample_id"]))
        if current is None or identity < (
            str(current[0]["sample_id"]),
            str(current[1]["sample_id"]),
        ):
            selected[group] = (v1_row, v2_row)
    return list(selected.values())


def _one_per_group(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        group = str(row["source_group_id"])
        current = selected.get(group)
        if current is None or str(row["sample_id"]) < str(current["sample_id"]):
            selected[group] = row
    return list(selected.values())


def _allocate_stratified_counts(
    sizes: dict[tuple[str, ...], int], target: int, minimum_one: bool
) -> dict[tuple[str, ...], int]:
    if target < 0 or target > sum(sizes.values()):
        raise ValueError("stratified target is outside candidate capacity")
    nonempty = {key: size for key, size in sizes.items() if size > 0}
    if minimum_one and target < len(nonempty):
        raise ValueError("target is too small to select one row from every stratum")

    allocation = {
        key: (1 if minimum_one else 0)
        for key in sorted(nonempty)
    }
    remaining = target - sum(allocation.values())
    if remaining == 0:
        return allocation

    total_weight = sum(nonempty.values())
    fractional: dict[tuple[str, ...], float] = {}
    for key, size in nonempty.items():
        ideal = remaining * size / total_weight
        extra = min(size - allocation[key], math.floor(ideal))
        allocation[key] += extra
        fractional[key] = ideal - math.floor(ideal)
    remaining = target - sum(allocation.values())

    while remaining:
        eligible = [
            key for key, size in nonempty.items() if allocation[key] < size
        ]
        if not eligible:
            raise ValueError("stratified allocation exhausted all candidates")
        eligible.sort(
            key=lambda key: (
                -fractional.get(key, 0.0),
                allocation[key] / nonempty[key],
                key,
            )
        )
        for key in eligible:
            if remaining == 0:
                break
            allocation[key] += 1
            remaining -= 1
    return allocation


def _select_stratified_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    target: int,
    fields: list[str],
    seed: int,
    minimum_one: bool,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    strata: dict[tuple[str, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for pair in pairs:
        key = tuple(str(pair[0].get(field, "<missing>")) for field in fields)
        strata[key].append(pair)
    allocation = _allocate_stratified_counts(
        {key: len(values) for key, values in strata.items()},
        target,
        minimum_one,
    )

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    counts: dict[str, int] = {}
    for key in sorted(strata):
        ordered_v1 = _stable_order(
            (pair[0] for pair in strata[key]),
            seed,
            "extension|" + "|".join(key),
        )
        by_v1_id = {pair[0]["sample_id"]: pair for pair in strata[key]}
        chosen = [by_v1_id[row["sample_id"]] for row in ordered_v1[: allocation[key]]]
        selected.extend(chosen)
        counts["|".join(key)] = len(chosen)
    return selected, counts


def _group_role_overlaps(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    roles_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("role") == "excluded" or not row.get("source_group_id"):
            continue
        roles_by_group[str(row["source_group_id"])].add(str(row["role"]))
    return {
        "train_validation": sum(
            {"train", "validation"}.issubset(roles)
            for roles in roles_by_group.values()
        ),
        "fit_to_final_test": sum(
            bool(roles & {"train", "validation"})
            and bool(roles & {"in_domain_test", "generator_holdout"})
            for roles in roles_by_group.values()
        ),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_config = config["data"]
    selection_config = config["selection"]
    audited_manifest_path = _resolve(project_root, data_config["audited_manifest"])
    frozen_manifest_path = _resolve(
        project_root, data_config["frozen_split_manifest"]
    )
    pilot_manifest_path = _resolve(project_root, data_config["pilot_manifest"])
    summary_path = _resolve(project_root, data_config["freeze_summary"])

    audited_manifest_sha256 = _sha256_file(audited_manifest_path)
    expected_manifest_sha256 = data_config.get("expected_audited_manifest_sha256")
    if (
        expected_manifest_sha256
        and audited_manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError(
            "audited manifest SHA-256 does not match the frozen config: "
            f"{audited_manifest_sha256} != {expected_manifest_sha256}"
        )

    rows = _read_jsonl(audited_manifest_path)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("audited manifest contains duplicate sample_id values")

    v1_by_id = {
        str(row["sample_id"]): row for row in rows if row.get("edition") == "v1"
    }
    allowed_join_statuses = set(selection_config["allowed_join_statuses"])
    strict_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for v2_row in (row for row in rows if row.get("edition") == "v2"):
        v1_row = v1_by_id.get(str(v2_row.get("joined_v1_sample_id")))
        if v1_row is None:
            continue
        same_group = bool(v1_row.get("source_group_id")) and str(
            v1_row.get("source_group_id")
        ) == str(v2_row.get("source_group_id"))
        if not (
            v1_row.get("valid")
            and v2_row.get("valid")
            and not v1_row.get("errors")
            and not v2_row.get("errors")
            and v1_row.get("role") != "excluded"
            and same_group
            and v2_row.get("join_status") in allowed_join_statuses
        ):
            continue
        strict_pairs.append((v1_row, v2_row))
    strict_pair_rows_before_group_deduplication = len(strict_pairs)
    strict_pairs = _one_pair_per_group(strict_pairs)

    official_split = str(selection_config["official_safe_split"])
    official_pairs = [
        pair
        for pair in strict_pairs
        if pair[0].get("split") == official_split
        and pair[0].get("role") == "in_domain_test"
        and pair[1].get("role") == "generator_holdout"
    ]
    expected_official = int(selection_config["official_safe_count"])
    if len(official_pairs) != expected_official:
        raise ValueError(
            "official-safe strict pair count changed: "
            f"{len(official_pairs)} != {expected_official}"
        )

    extension_pairs = [
        pair
        for pair in strict_pairs
        if pair[0].get("split") == selection_config["extension_source_split"]
        and pair[0].get("role") in {"train", "validation"}
    ]
    extension_target = int(selection_config["training_extension_count"])
    selected_extension, selected_strata = _select_stratified_pairs(
        extension_pairs,
        extension_target,
        list(selection_config["stratify_by"]),
        int(config["experiment"]["seed"]),
        bool(selection_config.get("minimum_one_per_nonempty_stratum", True)),
    )
    selected_pairs = official_pairs + selected_extension
    expected_test_groups = int(selection_config["test_group_count"])
    selected_groups = {str(pair[0]["source_group_id"]) for pair in selected_pairs}
    if len(selected_pairs) != expected_test_groups or len(selected_groups) != expected_test_groups:
        raise ValueError("selected paired test cohort is not exactly the configured size")

    origin_by_group = {
        str(pair[0]["source_group_id"]): "official_safe"
        for pair in official_pairs
    }
    origin_by_group.update(
        {
            str(pair[0]["source_group_id"]): "training_extension"
            for pair in selected_extension
        }
    )
    protocol_payload = {
        "audited_manifest_sha256": audited_manifest_sha256,
        "seed": int(config["experiment"]["seed"]),
        "selection": selection_config,
        "selected_groups": sorted(selected_groups),
    }
    freeze_id = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    frozen_rows: list[dict[str, Any]] = []
    for input_row in rows:
        row = dict(input_row)
        audit_role = str(row.get("role"))
        audit_exclusion_reason = row.get("exclusion_reason")
        group = str(row.get("source_group_id"))
        selected = group in selected_groups
        row["audit_role"] = audit_role
        row["audit_exclusion_reason"] = audit_exclusion_reason
        row["freeze_id"] = freeze_id
        row["frozen_test_origin"] = origin_by_group.get(group)

        if selected and row.get("valid"):
            if row.get("edition") == "v1" and audit_role != "excluded":
                row["role"] = "in_domain_test"
                row["exclusion_reason"] = None
            elif (
                row.get("edition") == "v2"
                and row.get("join_status") in allowed_join_statuses
            ):
                row["role"] = "generator_holdout"
                row["exclusion_reason"] = None
            else:
                row["role"] = "excluded"
                row["exclusion_reason"] = audit_exclusion_reason
        elif row.get("edition") == "v1" and audit_role in {"train", "validation"}:
            row["role"] = audit_role
            row["exclusion_reason"] = None
        else:
            row["role"] = "excluded"
            if audit_role == "excluded":
                row["exclusion_reason"] = audit_exclusion_reason
            elif row.get("edition") == "v1":
                row["exclusion_reason"] = "not_in_audited_paired_test"
            else:
                row["exclusion_reason"] = "not_in_frozen_generator_holdout"
        frozen_rows.append(row)

    frozen_by_id = {str(row["sample_id"]): row for row in frozen_rows}
    seed = int(config["experiment"]["seed"])
    train_target = int(selection_config["pilot_train_count"])
    validation_target = int(selection_config["pilot_validation_count"])
    train_candidates = _stable_order(
        _one_per_group(
            row
            for row in frozen_rows
            if row.get("edition") == "v1" and row.get("role") == "train"
        ),
        seed,
        "pilot_train",
    )
    validation_candidates = _stable_order(
        _one_per_group(
            row
            for row in frozen_rows
            if row.get("edition") == "v1" and row.get("role") == "validation"
        ),
        seed,
        "pilot_validation",
    )
    if len(train_candidates) < train_target or len(validation_candidates) < validation_target:
        raise ValueError("insufficient source groups for the configured train/validation pilot")

    official_pairs = sorted(
        official_pairs, key=lambda pair: str(pair[0]["source_group_id"])
    )
    selected_extension = sorted(
        selected_extension, key=lambda pair: str(pair[0]["source_group_id"])
    )
    ordered_test_pairs = official_pairs + selected_extension
    pilot_rows: list[dict[str, Any]] = []
    for pilot_role, candidates in (
        ("train", train_candidates[:train_target]),
        ("validation", validation_candidates[:validation_target]),
    ):
        for rank, candidate in enumerate(candidates, start=1):
            pilot_rows.append(
                {
                    **candidate,
                    "pilot_role": pilot_role,
                    "pilot_selection_rank": rank,
                }
            )
    for rank, pair in enumerate(ordered_test_pairs, start=1):
        v1_row = frozen_by_id[str(pair[0]["sample_id"])]
        v2_row = frozen_by_id[str(pair[1]["sample_id"])]
        pilot_rows.append(
            {
                **v1_row,
                "pilot_role": "in_domain_test",
                "pilot_selection_rank": rank,
            }
        )
        pilot_rows.append(
            {
                **v2_row,
                "pilot_role": "generator_holdout",
                "pilot_selection_rank": rank,
            }
        )

    full_overlaps = _group_role_overlaps(frozen_rows)
    pilot_overlaps = _group_role_overlaps(pilot_rows)
    if any(full_overlaps.values()) or any(pilot_overlaps.values()):
        raise ValueError("source-group overlap detected after pilot freeze")

    pilot_role_counts = Counter(str(row["pilot_role"]) for row in pilot_rows)
    expected_pilot_role_counts = {
        "train": train_target,
        "validation": validation_target,
        "in_domain_test": expected_test_groups,
        "generator_holdout": expected_test_groups,
    }
    if dict(pilot_role_counts) != expected_pilot_role_counts:
        raise ValueError("pilot role counts do not match the configured targets")
    if any(row.get("role") != row.get("pilot_role") for row in pilot_rows):
        raise ValueError("pilot_role disagrees with the frozen full-manifest role")

    _write_jsonl(frozen_manifest_path, frozen_rows)
    _write_jsonl(pilot_manifest_path, pilot_rows)
    summary: dict[str, Any] = {
        "experiment": config["experiment"],
        "status": "split_frozen_stage_zero_blocked",
        "paper_evidence": False,
        "gpu_launch_authorized": bool(
            config.get("runtime", {}).get("gpu_launch_authorized", False)
        ),
        "stage_zero_blockers": config.get("runtime", {}).get(
            "stage_zero_blockers", []
        ),
        "paper_evidence_blockers": config.get("runtime", {}).get(
            "paper_evidence_blockers", []
        ),
        "freeze_id": freeze_id,
        "input": {
            "audited_manifest": str(audited_manifest_path.relative_to(project_root)),
            "rows": len(rows),
            "sha256": audited_manifest_sha256,
        },
        "eligibility": {
            "strict_pair_rows_before_group_deduplication": (
                strict_pair_rows_before_group_deduplication
            ),
            "strict_unique_source_groups": len(strict_pairs),
            "official_safe_groups": len(official_pairs),
            "training_extension_candidate_groups": len(extension_pairs),
        },
        "selection": {
            "policy": "all_official_safe_plus_seeded_stratified_training_extension",
            "official_safe_groups": len(official_pairs),
            "training_extension_groups": len(selected_extension),
            "test_groups": len(selected_groups),
            "stratify_by": list(selection_config["stratify_by"]),
            "training_extension_stratum_counts": selected_strata,
        },
        "frozen_manifest": {
            "path": str(frozen_manifest_path.relative_to(project_root)),
            "rows": len(frozen_rows),
            "sha256": _sha256_file(frozen_manifest_path),
            "role_row_counts": dict(
                sorted(Counter(str(row["role"]) for row in frozen_rows).items())
            ),
            "role_source_group_counts": {
                role: len(
                    {
                        str(row["source_group_id"])
                        for row in frozen_rows
                        if row.get("role") == role and row.get("source_group_id")
                    }
                )
                for role in (
                    "train",
                    "validation",
                    "in_domain_test",
                    "generator_holdout",
                    "excluded",
                )
            },
            "overlaps": full_overlaps,
        },
        "pilot_manifest": {
            "path": str(pilot_manifest_path.relative_to(project_root)),
            "rows": len(pilot_rows),
            "sha256": _sha256_file(pilot_manifest_path),
            "role_counts": dict(sorted(pilot_role_counts.items())),
            "overlaps": pilot_overlaps,
        },
    }
    if summary["gpu_launch_authorized"]:
        raise ValueError("pilot freeze config must not authorize a GPU launch")
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
