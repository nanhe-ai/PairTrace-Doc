from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _digest(seed: int, stage: str, *values: object) -> str:
    payload = "|".join([str(seed), stage, *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _editor_order(config: dict[str, Any]) -> dict[str, int]:
    editors = [str(row["id"]) for row in config["editors"]]
    if len(editors) != 4 or len(set(editors)) != 4:
        raise ValueError("exactly four unique editor IDs are required")
    return {editor: index for index, editor in enumerate(editors)}


def _canonical_pair(
    pair: Iterable[str], editor_order: dict[str, int]
) -> tuple[str, str]:
    values = tuple(str(value) for value in pair)
    if len(values) != 2 or values[0] == values[1]:
        raise ValueError(f"invalid editor pair: {values}")
    if any(value not in editor_order for value in values):
        raise ValueError(f"unknown editor in pair: {values}")
    return tuple(sorted(values, key=editor_order.__getitem__))  # type: ignore[return-value]


def _pair_key(pair: tuple[str, str]) -> str:
    return "+".join(pair)


def _validate_input_identity(path: Path, expected_sha256: str) -> None:
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected_sha256}")


def _balanced_final_assignment(
    rows: list[dict[str, Any]], config: dict[str, Any], freeze_id: str
) -> list[dict[str, Any]]:
    seed = int(config["experiment"]["seed"])
    editor_order = _editor_order(config)
    editors = list(editor_order)
    pairs = [
        _canonical_pair(pair, editor_order)
        for pair in itertools.combinations(editors, 2)
    ]
    assignment = config["assignment"]
    expected_datasets = ("DocLayNet", "NAF", "MIDV-500")
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["source_dataset"])].append(row)
    if set(by_dataset) != set(expected_datasets):
        raise ValueError(f"unexpected final datasets: {sorted(by_dataset)}")

    output: list[dict[str, Any]] = []
    for dataset in expected_datasets:
        dataset_rows = by_dataset[dataset]
        if len(dataset_rows) != 50:
            raise ValueError(f"expected 50 final rows for {dataset}, got {len(dataset_rows)}")
        dataset_rows.sort(
            key=lambda row: _digest(
                seed, "final-assignment", dataset, row["source_group_id"]
            )
        )
        extra_pairs = {
            _canonical_pair(pair, editor_order)
            for pair in assignment["dataset_pair_extra_matching"][dataset]
        }
        if len(extra_pairs) != 2:
            raise ValueError(f"{dataset} must declare two extra matching pairs")
        extra_editor_degrees = Counter(editor for pair in extra_pairs for editor in pair)
        if extra_editor_degrees != Counter({editor: 1 for editor in editors}):
            raise ValueError(f"{dataset} extra pairs are not a perfect matching")

        offset = 0
        for pair in pairs:
            count = 9 if pair in extra_pairs else 8
            for row in dataset_rows[offset : offset + count]:
                output.append(
                    {
                        "assignment_freeze_id": freeze_id,
                        "editor_ids": list(pair),
                        "editor_pair": _pair_key(pair),
                        "final_source_freeze_id": row["freeze_id"],
                        "source_dataset": dataset,
                        "source_group_id": row["source_group_id"],
                        "source_index": row["source_index"],
                        "source_stratum": row["source_stratum"],
                        "template_family_id": row["template_family_id"],
                    }
                )
            offset += count
        if offset != 50:
            raise AssertionError(f"internal allocation failure for {dataset}: {offset}")

    pair_counts = Counter(row["editor_pair"] for row in output)
    if set(pair_counts.values()) != {25} or len(pair_counts) != 6:
        raise ValueError(f"unbalanced final pair counts: {pair_counts}")
    editor_counts = Counter(editor for row in output for editor in row["editor_ids"])
    if editor_counts != Counter({editor: 75 for editor in editors}):
        raise ValueError(f"unbalanced final editor counts: {editor_counts}")
    per_dataset_editor = Counter(
        (row["source_dataset"], editor)
        for row in output
        for editor in row["editor_ids"]
    )
    expected = Counter(
        {(dataset, editor): 25 for dataset in expected_datasets for editor in editors}
    )
    if per_dataset_editor != expected:
        raise ValueError(f"unbalanced per-dataset editor counts: {per_dataset_editor}")
    if any("path" in row for row in output):
        raise AssertionError("final assignment must not contain source paths")
    return sorted(output, key=lambda row: int(row["source_index"]))


def _eligible_nonfinal_groups(
    audit_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    final_groups = {
        (str(row["source_dataset"]), str(row["template_family_id"]))
        for row in final_rows
    }
    final_encoded = {str(row["encoded_sha256"]) for row in final_rows}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        dataset = str(row["source_dataset"])
        if dataset not in {"DocLayNet", "NAF"}:
            continue
        group = str(row["source_group_key"])
        if row.get("selected"):
            continue
        if row.get("status") != "ok" or not row.get("hard_gate_eligible"):
            continue
        if (dataset, group) in final_groups:
            continue
        if str(row["encoded_sha256"]) in final_encoded:
            raise ValueError("non-final candidate has a final encoded SHA-256")
        grouped[(dataset, group)].append(row)

    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (dataset, group), records in grouped.items():
        selected = min(
            records,
            key=lambda row: _digest(
                seed,
                "one-per-nonfinal-group",
                dataset,
                group,
                row["record_id"],
            ),
        )
        output[dataset].append(selected)
    for dataset in output:
        output[dataset].sort(
            key=lambda row: _digest(
                seed,
                "nonfinal-group-order",
                dataset,
                row["source_group_key"],
                row["record_id"],
            )
        )
    return dict(output)


def _choose(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    stage: str,
    count: int,
    stratum: str | None = None,
    excluded: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded or set()
    candidates = [
        row
        for row in rows
        if (str(row["source_dataset"]), str(row["source_group_key"])) not in excluded
        and (stratum is None or str(row["source_stratum"]) == stratum)
    ]
    candidates.sort(
        key=lambda row: _digest(
            seed,
            stage,
            row["source_dataset"],
            row["source_group_key"],
            row["record_id"],
        )
    )
    if len(candidates) < count:
        raise ValueError(
            f"insufficient candidates for {stage}/{stratum}: {len(candidates)} < {count}"
        )
    return candidates[:count]


def _rehearsal_row(
    row: dict[str, Any],
    *,
    stage: str,
    pair: tuple[str, str],
    freeze_id: str,
) -> dict[str, Any]:
    dataset = str(row["source_dataset"])
    return {
        "editor_ids": list(pair),
        "editor_pair": _pair_key(pair),
        "encoded_sha256": row["encoded_sha256"],
        "decoded_pixel_sha256": row["decoded_pixel_sha256"],
        "height": row["height"],
        "path": row["path"],
        "rehearsal_freeze_id": freeze_id,
        "rehearsal_id": f"{stage}:{row['record_id']}",
        "rehearsal_stage": stage,
        "source_dataset": dataset,
        "source_group_key": row["source_group_key"],
        "source_record_id": row["record_id"],
        "source_stratum": row["source_stratum"],
        "source_was_selected_for_final": False,
        "handling": (
            "possible_historical_personal_data_local_only"
            if dataset == "NAF"
            else "public_source_local_editor_rehearsal"
        ),
        "width": row["width"],
    }


def _freeze_rehearsal_pools(
    audit_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    config: dict[str, Any],
    freeze_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(config["experiment"]["seed"])
    editor_order = _editor_order(config)
    pools = _eligible_nonfinal_groups(audit_rows, final_rows, seed)
    spec = config["rehearsal_selection"]
    toy_spec = spec["toy3"]
    toy_source_rows = [
        *_choose(
            pools["DocLayNet"],
            seed=seed,
            stage="toy3-financial",
            count=int(toy_spec["source_counts"]["DocLayNet"]["financial_reports"]),
            stratum="financial_reports",
        ),
        *_choose(
            pools["DocLayNet"],
            seed=seed,
            stage="toy3-tender",
            count=int(toy_spec["source_counts"]["DocLayNet"]["government_tenders"]),
            stratum="government_tenders",
        ),
        *_choose(
            pools["NAF"],
            seed=seed,
            stage="toy3-naf",
            count=int(toy_spec["source_counts"]["NAF"]["any"]),
        ),
    ]
    toy_pairs = [
        _canonical_pair(pair, editor_order) for pair in toy_spec["editor_pairs"]
    ]
    if len(toy_pairs) != len(toy_source_rows):
        raise ValueError("toy editor-pair count does not match toy source count")
    toy_rows = [
        _rehearsal_row(row, stage="toy3", pair=pair, freeze_id=freeze_id)
        for row, pair in zip(toy_source_rows, toy_pairs, strict=True)
    ]
    if set(editor_order) != {editor for row in toy_rows for editor in row["editor_ids"]}:
        raise ValueError("toy assignment does not cover every editor")

    excluded = {
        (str(row["source_dataset"]), str(row["source_group_key"]))
        for row in toy_source_rows
    }
    pilot_spec = spec["pilot100"]
    pilot_by_dataset = {
        "DocLayNet": [
            *_choose(
                pools["DocLayNet"],
                seed=seed,
                stage="pilot100-financial",
                count=int(
                    pilot_spec["source_counts"]["DocLayNet"]["financial_reports"]
                ),
                stratum="financial_reports",
                excluded=excluded,
            ),
            *_choose(
                pools["DocLayNet"],
                seed=seed,
                stage="pilot100-tender",
                count=int(
                    pilot_spec["source_counts"]["DocLayNet"][
                        "government_tenders"
                    ]
                ),
                stratum="government_tenders",
                excluded=excluded,
            ),
        ],
        "NAF": _choose(
            pools["NAF"],
            seed=seed,
            stage="pilot100-naf",
            count=int(pilot_spec["source_counts"]["NAF"]["distinct_base_families"]),
            excluded=excluded,
        ),
    }
    pilot_rows: list[dict[str, Any]] = []
    combined_pair_counts: Counter[str] = Counter()
    for dataset in ("DocLayNet", "NAF"):
        rows = sorted(
            pilot_by_dataset[dataset],
            key=lambda row: _digest(
                seed, "pilot100-row-order", dataset, row["source_group_key"]
            ),
        )
        offset = 0
        for key, count_value in pilot_spec["pair_counts_by_dataset"][dataset].items():
            pair = _canonical_pair(str(key).split("+"), editor_order)
            count = int(count_value)
            for row in rows[offset : offset + count]:
                pilot_rows.append(
                    _rehearsal_row(
                        row, stage="pilot100", pair=pair, freeze_id=freeze_id
                    )
                )
            offset += count
            combined_pair_counts[_pair_key(pair)] += count
        if offset != len(rows):
            raise ValueError(f"pilot pair allocation mismatch for {dataset}")

    expected_pair_counts = Counter(
        {str(key): int(value) for key, value in pilot_spec["combined_pair_counts"].items()}
    )
    if combined_pair_counts != expected_pair_counts:
        raise ValueError(
            f"pilot combined pair counts mismatch: {combined_pair_counts}"
        )
    editor_counts = Counter(
        editor for row in pilot_rows for editor in row["editor_ids"]
    )
    if editor_counts != Counter({editor: 25 for editor in editor_order}):
        raise ValueError(f"pilot editor counts are not balanced: {editor_counts}")
    toy_identity = {
        (row["source_dataset"], row["source_group_key"]) for row in toy_rows
    }
    pilot_identity = {
        (row["source_dataset"], row["source_group_key"]) for row in pilot_rows
    }
    if toy_identity & pilot_identity:
        raise ValueError("toy and pilot source groups overlap")
    if len(pilot_identity) != 50:
        raise ValueError(f"expected 50 distinct pilot groups, got {len(pilot_identity)}")
    return toy_rows, pilot_rows


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    protocol_path = _resolve(project_root, str(config["experiment"]["protocol"]))
    final_path = _resolve(project_root, str(config["inputs"]["final_source_manifest"]))
    audit_path = _resolve(project_root, str(config["inputs"]["quality_audit"]))
    _validate_input_identity(
        final_path, str(config["inputs"]["expected_final_source_manifest_sha256"])
    )
    _validate_input_identity(
        audit_path, str(config["inputs"]["expected_quality_audit_sha256"])
    )
    final_rows = _read_jsonl(final_path)
    audit_rows = _read_jsonl(audit_path)
    if len(final_rows) != int(config["inputs"]["expected_final_source_rows"]):
        raise ValueError("unexpected final source row count")
    freeze_ids = {str(row["freeze_id"]) for row in final_rows}
    if freeze_ids != {str(config["inputs"]["expected_final_source_freeze_id"])}:
        raise ValueError(f"unexpected final source freeze IDs: {freeze_ids}")

    config_sha256 = _sha256(config_path)
    protocol_sha256 = _sha256(protocol_path)
    freeze_payload = {
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "final_source_manifest_sha256": _sha256(final_path),
        "quality_audit_sha256": _sha256(audit_path),
    }
    freeze_id = hashlib.sha256(
        json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    assignment_rows = _balanced_final_assignment(final_rows, config, freeze_id)
    toy_rows, pilot_rows = _freeze_rehearsal_pools(
        audit_rows, final_rows, config, freeze_id
    )
    outputs = config["outputs"]
    assignment_path = _resolve(project_root, str(outputs["final_assignment_manifest"]))
    toy_path = _resolve(project_root, str(outputs["toy3_manifest"]))
    pilot_path = _resolve(project_root, str(outputs["pilot100_manifest"]))
    summary_path = _resolve(project_root, str(outputs["freeze_summary"]))
    _write_jsonl(assignment_path, assignment_rows)
    _write_jsonl(toy_path, toy_rows)
    _write_jsonl(pilot_path, pilot_rows)

    summary = {
        "authorization": {
            "final_source_images_resolved_or_decoded": False,
            "final_source_editor_execution": False,
            "final_source_detector_execution": False,
            "nonfinal_toy3_authorized_after_gpu_preflight": True,
            "nonfinal_pilot100_authorized_before_toy3_gate": False,
            "physical_capture_authorized": False,
        },
        "editor_ids": [str(row["id"]) for row in config["editors"]],
        "freeze_id": freeze_id,
        "freeze_payload": freeze_payload,
        "final_assignment": {
            "path": str(assignment_path.relative_to(project_root)),
            "rows": len(assignment_rows),
            "sha256": _sha256(assignment_path),
            "pair_counts": dict(Counter(row["editor_pair"] for row in assignment_rows)),
            "editor_counts": dict(
                Counter(
                    editor for row in assignment_rows for editor in row["editor_ids"]
                )
            ),
            "contains_source_paths": False,
        },
        "toy3": {
            "path": str(toy_path.relative_to(project_root)),
            "source_rows": len(toy_rows),
            "edit_calls": sum(len(row["editor_ids"]) for row in toy_rows),
            "sha256": _sha256(toy_path),
        },
        "pilot100": {
            "path": str(pilot_path.relative_to(project_root)),
            "source_rows": len(pilot_rows),
            "edit_calls": sum(len(row["editor_ids"]) for row in pilot_rows),
            "sha256": _sha256(pilot_path),
            "status": "frozen_closed_until_toy3_gate_passes",
        },
        "model_repository_bytes": {
            str(row["id"]): int(row["current_repository_bytes_at_freeze"])
            for row in config["editors"]
        },
        "storage_strategy": "one_exact_editor_revision_at_a_time",
    }
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path.relative_to(project_root))
    summary["summary_sha256"] = _sha256(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze prospective editor assignments and non-final rehearsal pools."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run(args.config, args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
