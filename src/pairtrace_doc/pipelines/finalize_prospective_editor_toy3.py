from __future__ import annotations

import argparse
import hashlib
import json
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _verify_bound_file(project_root: Path, entry: dict[str, Any]) -> Path:
    path = _resolve(project_root, str(entry["path"]))
    actual = _sha256(path)
    expected = str(entry["sha256"])
    if actual != expected:
        raise ValueError(f"frozen input changed: {path} ({actual} != {expected})")
    return path


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")

    verified_artifacts: list[dict[str, str]] = []
    verified_paths: dict[str, Path] = {}
    for entry in config["frozen_inputs"]:
        path = _verify_bound_file(project_root, entry)
        name = str(entry["name"])
        if name in verified_paths:
            raise ValueError(f"duplicate frozen input name: {name}")
        verified_paths[name] = path
        verified_artifacts.append(
            {
                "name": name,
                "path": str(entry["path"]),
                "sha256": str(entry["sha256"]),
            }
        )

    targets = _read_jsonl(verified_paths[str(config["inputs"]["target_manifest"] )])
    expected_calls: set[tuple[str, str]] = set()
    for target in targets:
        rehearsal_id = str(target["rehearsal_id"])
        for editor_id in target["editor_ids"]:
            expected_calls.add((rehearsal_id, str(editor_id)))
    if len(expected_calls) != int(config["gates"]["expected_calls"]):
        raise ValueError("target manifest does not contain the expected call count")

    automatic_reports = [
        _read_json(verified_paths[str(name)])
        for name in config["inputs"]["automatic_reports"]
    ]
    if not all(bool(report.get("automatic_gate_passed")) for report in automatic_reports):
        raise ValueError("one or more automatic audit stages did not pass")
    if not all(bool(report.get("no_cuda_oom")) for report in automatic_reports):
        raise ValueError("an automatic audit stage recorded CUDA OOM")

    review_rows: list[dict[str, Any]] = []
    expected_attempt_index: dict[str, int] = {}
    for review in config["inputs"]["visual_reviews"]:
        name = str(review["name"])
        attempt_index = int(review["attempt_index"])
        expected_attempt_index[name] = attempt_index
        rows = _read_jsonl(verified_paths[name])
        for row in rows:
            if int(row["attempt_index"]) != attempt_index:
                raise ValueError(f"review attempt index mismatch in {name}")
            if row.get("reviewer_type") != "agent_nonhuman":
                raise ValueError("Toy-3 visual verdict is not labeled agent_nonhuman")
            review_rows.append(row)

    rows_by_call: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_attempts: set[tuple[str, str, int]] = set()
    for row in review_rows:
        call_key = (str(row["rehearsal_id"]), str(row["editor_id"]))
        if call_key not in expected_calls:
            raise ValueError(f"visual review is not a frozen Toy-3 call: {call_key}")
        attempt_key = (*call_key, int(row["attempt_index"]))
        if attempt_key in seen_attempts:
            raise ValueError(f"duplicate visual review: {attempt_key}")
        seen_attempts.add(attempt_key)
        rows_by_call.setdefault(call_key, []).append(row)

    maximum_attempts = int(config["gates"]["maximum_attempts"])
    outcomes: list[dict[str, Any]] = []
    for rehearsal_id, editor_id in sorted(expected_calls):
        rows = sorted(
            rows_by_call.get((rehearsal_id, editor_id), []),
            key=lambda row: int(row["attempt_index"]),
        )
        if not rows or int(rows[0]["attempt_index"]) != 0:
            raise ValueError(f"missing attempt-0 visual review: {(rehearsal_id, editor_id)}")
        accepted_rows = [row for row in rows if bool(row["accepted_visual_gate"])]
        first_accepted = accepted_rows[0] if accepted_rows else None
        terminal_attempt = (
            int(first_accepted["attempt_index"])
            if first_accepted is not None
            else maximum_attempts - 1
        )
        observed_indices = [int(row["attempt_index"]) for row in rows]
        if observed_indices != list(range(terminal_attempt + 1)):
            raise ValueError(
                f"non-consecutive or post-acceptance reviews for {(rehearsal_id, editor_id)}: "
                f"{observed_indices}"
            )
        for row in rows:
            accepted = bool(row["accepted_visual_gate"])
            if accepted != (
                row.get("intended_replacement") == "passed"
                and row.get("legibility") == "passed"
                and row.get("outside_mask_change")
                == "passed_automatic_exact_zero"
            ):
                raise ValueError("visual verdict fields are internally inconsistent")
        outcomes.append(
            {
                "accepted_visual_gate": first_accepted is not None,
                "attempts_reviewed": len(rows),
                "editor_id": editor_id,
                "first_accepted_attempt": (
                    int(first_accepted["attempt_index"])
                    if first_accepted is not None
                    else None
                ),
                "rehearsal_id": rehearsal_id,
                "terminal_reason": str(rows[-1]["reason"]),
            }
        )

    accepted_outcomes = [row for row in outcomes if row["accepted_visual_gate"]]
    accepted_by_editor = {
        editor_id: sum(
            int(row["accepted_visual_gate"] and row["editor_id"] == editor_id)
            for row in outcomes
        )
        for editor_id in config["gates"]["required_editors"]
    }
    latest_storage = automatic_reports[-1]["storage"]
    gate_checks = {
        "all_automatic_audits_passed": True,
        "all_six_calls_have_attempt_records": len(rows_by_call) == len(expected_calls),
        "accepted_calls_at_least_minimum": len(accepted_outcomes)
        >= int(config["gates"]["minimum_accepted_calls"]),
        "each_editor_has_accepted_call": all(
            count >= 1 for count in accepted_by_editor.values()
        ),
        "no_cuda_oom": True,
        "accepted_candidates_zero_outside_mask": all(
            row.get("outside_mask_change") == "passed_automatic_exact_zero"
            for row in review_rows
            if bool(row["accepted_visual_gate"])
        ),
        "persistent_storage_above_floor": int(latest_storage["free_bytes"])
        >= int(latest_storage["minimum_free_bytes"]),
    }
    toy3_passed = all(gate_checks.values())

    outcomes_path = _resolve(project_root, str(config["outputs"]["call_outcomes"]))
    _write_jsonl(outcomes_path, outcomes)
    result = {
        "accepted_by_editor": accepted_by_editor,
        "accepted_calls": len(accepted_outcomes),
        "authorization": {
            "final150_authorized": False,
            "final_source_images_read": False,
            "pilot100_authorized": toy3_passed,
            "pilot100_run": False,
        },
        "automatic_attempt_records_reviewed": len(review_rows),
        "call_outcomes": str(config["outputs"]["call_outcomes"]),
        "call_outcomes_sha256": _sha256(outcomes_path),
        "config_path": str(config_path.relative_to(project_root)),
        "config_sha256": _sha256(config_path),
        "gate_checks": gate_checks,
        "maximum_attempts_per_call": maximum_attempts,
        "minimum_accepted_calls": int(config["gates"]["minimum_accepted_calls"]),
        "paper_evidence": False,
        "reviewer_type": "agent_nonhuman",
        "status": (
            "toy3_passed_pilot100_may_open_final150_closed"
            if toy3_passed
            else "toy3_failed_visual_gate_pilot100_closed_final150_closed"
        ),
        "toy3_passed": toy3_passed,
        "verified_artifacts": verified_artifacts,
    }
    result_path = _resolve(project_root, str(config["outputs"]["result"]))
    _write_json(result_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize the frozen Toy-3 editor gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args.config, args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
