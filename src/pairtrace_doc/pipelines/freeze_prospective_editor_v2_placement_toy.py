from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
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


def _digest(seed: int, stage: str, *values: object) -> str:
    payload = "|".join([str(seed), stage, *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _write_json(path: Path, value: Any) -> None:
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


def _verify(path: Path, expected: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def _choose_group(
    rows: list[dict[str, Any]], seed: int, stage: str, stratum: str | None
) -> dict[str, Any]:
    filtered = [
        row
        for row in rows
        if stratum is None or str(row["source_stratum"]) == stratum
    ]
    filtered.sort(
        key=lambda row: _digest(
            seed,
            stage,
            row["source_dataset"],
            row["source_group_key"],
            row["record_id"],
        )
    )
    if not filtered:
        raise ValueError(f"no eligible source for {stage}/{stratum}")
    return filtered[0]


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")

    for entry in config.get("frozen_files", {}).values():
        path = _resolve(project_root, str(entry["path"]))
        _verify(path, str(entry["sha256"]))

    protocol_path = _resolve(project_root, str(config["experiment"]["protocol"]))
    expected_protocol_sha256 = config["experiment"].get("protocol_sha256")
    if expected_protocol_sha256 is not None:
        _verify(protocol_path, str(expected_protocol_sha256))

    inputs: dict[str, Path] = {}
    for name, entry in config["inputs"].items():
        path = _resolve(project_root, str(entry["path"]))
        _verify(path, str(entry["sha256"]))
        inputs[str(name)] = path

    final_rows = _read_jsonl(inputs["final_sources"])
    old_toy_rows = _read_jsonl(inputs["old_toy"])
    old_pilot_rows = _read_jsonl(inputs["old_pilot"])
    prior_placement_rows = (
        _read_jsonl(inputs["prior_placement"])
        if "prior_placement" in inputs
        else []
    )
    audit_rows = _read_jsonl(inputs["quality_audit"])

    final_groups = {
        (str(row["source_dataset"]), str(row["template_family_id"]))
        for row in final_rows
    }
    final_encoded = {str(row["encoded_sha256"]) for row in final_rows}
    historical_groups = {
        (str(row["source_dataset"]), str(row["source_group_key"]))
        for row in [*old_toy_rows, *old_pilot_rows, *prior_placement_rows]
    }
    historical_encoded = {
        str(row["encoded_sha256"])
        for row in [*old_toy_rows, *old_pilot_rows, *prior_placement_rows]
    }

    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        dataset = str(row["source_dataset"])
        group = str(row["source_group_key"])
        identity = (dataset, group)
        if dataset not in {"DocLayNet", "NAF"}:
            continue
        if row.get("selected"):
            continue
        if row.get("status") != "ok" or not row.get("hard_gate_eligible"):
            continue
        if identity in final_groups or identity in historical_groups:
            continue
        encoded = str(row["encoded_sha256"])
        if encoded in final_encoded or encoded in historical_encoded:
            raise ValueError("eligible V2 record overlaps a frozen source by hash")
        by_group[identity].append(row)

    seed = int(config["experiment"]["seed"])
    one_per_group: list[dict[str, Any]] = []
    for (dataset, group), records in by_group.items():
        one_per_group.append(
            min(
                records,
                key=lambda row: _digest(
                    seed, "v2-one-per-group", dataset, group, row["record_id"]
                ),
            )
        )
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in one_per_group:
        by_dataset[str(row["source_dataset"])].append(row)

    selected = [
        _choose_group(
            by_dataset["DocLayNet"],
            seed,
            "v2-placement-financial",
            "financial_reports",
        ),
        _choose_group(
            by_dataset["DocLayNet"],
            seed,
            "v2-placement-tender",
            "government_tenders",
        ),
        _choose_group(by_dataset["NAF"], seed, "v2-placement-naf", None),
    ]
    selected_identities = {
        (str(row["source_dataset"]), str(row["source_group_key"]))
        for row in selected
    }
    if len(selected_identities) != 3:
        raise ValueError("V2 placement sources are not three distinct groups")
    if selected_identities & final_groups or selected_identities & historical_groups:
        raise ValueError("V2 placement source group overlaps a frozen pool")

    freeze_payload = {
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(protocol_path),
        "input_sha256": {
            name: str(config["inputs"][name]["sha256"])
            for name in sorted(config["inputs"])
        },
    }
    freeze_id = hashlib.sha256(
        json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    output_rows = []
    for index, row in enumerate(selected):
        dataset = str(row["source_dataset"])
        output_rows.append(
            {
                "decoded_pixel_sha256": row["decoded_pixel_sha256"],
                "encoded_sha256": row["encoded_sha256"],
                "handling": (
                    "possible_historical_personal_data_local_only"
                    if dataset == "NAF"
                    else "public_source_local_editor_rehearsal"
                ),
                "height": int(row["height"]),
                "path": row["path"],
                "source_dataset": dataset,
                "source_group_key": row["source_group_key"],
                "source_record_id": row["record_id"],
                "source_stratum": row["source_stratum"],
                "source_was_selected_for_final": False,
                "v2_placement_freeze_id": freeze_id,
                "v2_placement_id": f"v2-placement:{index}:{row['record_id']}",
                "width": int(row["width"]),
            }
        )

    manifest_path = _resolve(project_root, str(config["outputs"]["manifest"]))
    summary_path = _resolve(project_root, str(config["outputs"]["summary"]))
    _write_jsonl(manifest_path, output_rows)
    summary = {
        "authorization": {
            "detector_inference_run": False,
            "editor_inference_run": False,
            "final_source_images_read": False,
            "pilot100_run": False,
            "v2_nonfinal_source_images_read": False,
        },
        "available_distinct_groups_after_exclusion": {
            dataset: len(rows) for dataset, rows in sorted(by_dataset.items())
        },
        "freeze_id": freeze_id,
        "freeze_payload": freeze_payload,
        "manifest": str(manifest_path.relative_to(project_root)),
        "manifest_sha256": _sha256(manifest_path),
        "rows": len(output_rows),
        "status": "v2_placement_toy_sources_frozen_images_not_read",
    }
    _write_json(summary_path, summary)
    summary["summary"] = str(summary_path.relative_to(project_root))
    summary["summary_sha256"] = _sha256(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze disjoint non-final sources for the V2 placement toy"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run(args.config, args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
