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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


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


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    paths: dict[str, Path] = {}
    for name, entry in config["frozen_inputs"].items():
        path = _resolve(project_root, str(entry["path"]))
        observed = _sha256(path)
        if observed != str(entry["sha256"]):
            raise ValueError(f"frozen input changed: {path}")
        paths[str(name)] = path

    source_rows = _read_jsonl(paths["source_manifest"])
    run_rows = _read_jsonl(paths["run_records"])
    audit_rows = _read_jsonl(paths["audit_records"])
    visual_rows = _read_jsonl(paths["visual_reviews"])
    run_report = _read_json(paths["run_report"])
    audit_report = _read_json(paths["audit_report"])
    expected_ids = {str(row["v2_placement_id"]) for row in source_rows}
    if len(expected_ids) != int(config["gate"]["expected_sources"]):
        raise ValueError("unexpected frozen source count")

    def index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        output = {str(row["v2_placement_id"]): row for row in rows}
        if len(output) != len(rows) or set(output) != expected_ids:
            raise ValueError("V2 result rows do not match the frozen source IDs")
        return output

    run_by_id = index(run_rows)
    audit_by_id = index(audit_rows)
    visual_by_id = index(visual_rows)
    outcomes = []
    for placement_id in sorted(expected_ids):
        run_row = run_by_id[placement_id]
        audit_row = audit_by_id[placement_id]
        visual_row = visual_by_id[placement_id]
        if run_row.get("clear_source_text_persisted") is not False:
            raise ValueError("clear source text persistence invariant failed")
        if visual_row.get("reviewer_type") != "agent_nonhuman":
            raise ValueError("visual review is not labeled agent_nonhuman")
        outcomes.append(
            {
                "accepted_automatic_gate": bool(
                    audit_row["automatic_gate_recomputed"]
                ),
                "accepted_visual_gate": bool(visual_row["accepted_visual_gate"]),
                "method_id": str(run_row["method_id"]),
                "ocr_exact_replacement": bool(
                    audit_row["checks"]["ocr_exact_replacement"]
                ),
                "outside_changed_pixels": int(
                    audit_row["outside_changed_pixels"]
                ),
                "v2_placement_id": placement_id,
                "visual_reason": str(visual_row["reason"]),
            }
        )

    gate_checks = {
        "all_sources_have_records": len(outcomes)
        == int(config["gate"]["expected_sources"]),
        "all_sources_pass_independent_automatic_gate": all(
            row["accepted_automatic_gate"] for row in outcomes
        ),
        "all_sources_pass_agent_visual_gate": all(
            row["accepted_visual_gate"] for row in outcomes
        ),
        "all_sources_have_exact_ocr_replacement": all(
            row["ocr_exact_replacement"] for row in outcomes
        ),
        "all_sources_have_zero_outside_change": all(
            row["outside_changed_pixels"] == 0 for row in outcomes
        ),
        "run_and_audit_reports_agree": bool(run_report["automatic_gate_passed"])
        == bool(audit_report["automatic_gate_passed"]),
    }
    placement_passed = all(gate_checks.values())
    outcomes_path = _resolve(project_root, str(config["outputs"]["outcomes"]))
    result_path = _resolve(project_root, str(config["outputs"]["result"]))
    _write_jsonl(outcomes_path, outcomes)
    result = {
        "authorization": {
            "final150_authorized": False,
            "final_source_images_read": False,
            "hybrid_four_editor_toy_authorized": placement_passed,
            "neural_editor_inference_run": False,
            "pilot100_authorized": False,
            "pilot100_run": False,
        },
        "automatic_accepted": sum(
            int(row["accepted_automatic_gate"]) for row in outcomes
        ),
        "gate_checks": gate_checks,
        "method_id": str(config["method_id"]),
        "outcomes": str(outcomes_path.relative_to(project_root)),
        "outcomes_sha256": _sha256(outcomes_path),
        "paper_evidence": False,
        "placement_preflight_passed": placement_passed,
        "reviewer_type": "agent_nonhuman",
        "rows": len(outcomes),
        "status": (
            "v2_placement_preflight_passed_hybrid_toy_design_may_begin"
            if placement_passed
            else "v2_placement_preflight_failed_hybrid_toy_pilot100_final150_closed"
        ),
        "visual_accepted": sum(
            int(row["accepted_visual_gate"]) for row in outcomes
        ),
    }
    _write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the V2 placement toy")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run(args.config, args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
