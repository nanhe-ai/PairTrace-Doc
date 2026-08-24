from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    forbidden = (
        "license_gate_open",
        "archive_download_authorized",
        "archive_read_authorized",
        "dataset_image_decode_authorized",
        "model_scoring_authorized",
    )
    if any(bool(runtime[field]) for field in forbidden):
        raise ValueError("closed source-gate freeze cannot authorize data access or scoring")
    experiment = config["experiment"]
    bindings = (
        ("protocol", "expected_protocol_sha256", "real-scan protocol"),
        (
            "editor_specification",
            "expected_editor_specification_sha256",
            "edit-generator specification",
        ),
        ("editor_source", "expected_editor_source_sha256", "edit-generator source"),
    )
    bound_hashes: dict[str, str] = {}
    for path_field, hash_field, label in bindings:
        path = _resolve(project_root, str(experiment[path_field]))
        digest = _sha256(path)
        if digest != str(experiment[hash_field]):
            raise ValueError(f"{label} changed: {digest} != {experiment[hash_field]}")
        bound_hashes[path_field] = digest
    license_spec = config["license"]
    if str(license_spec["status"]) != "clarification_required":
        raise ValueError("closed source gate must remain clarification_required")
    if (
        license_spec.get("written_author_clarification") is not None
        or license_spec.get("institutional_approval") is not None
    ):
        raise ValueError("approval evidence needs a separate dated gate-opening config")
    request_draft = _resolve(
        project_root, str(license_spec["clarification_request_draft"])
    )
    if not request_draft.is_file():
        raise ValueError("license clarification request draft is missing")
    source = config["source"]
    record = {
        **source,
        "license_status": license_spec["status"],
        "license_gate_open": False,
        "archive_downloaded_by_pipeline": False,
        "archive_bytes_read": False,
        "dataset_image_decoded": False,
        "model_score_read": False,
        "editor_implementation": "pairtrace_descan_edit_generator_v1",
        "paper_evidence": False,
    }
    registry_path = _resolve(project_root, str(config["paths"]["registry"]))
    summary_path = _resolve(project_root, str(config["paths"]["summary"]))
    _write_jsonl(registry_path, [record])
    summary = {
        "status": "descan18k_source_frozen_license_gate_closed",
        "paper_evidence": False,
        "license_gate_open": False,
        "archive_downloaded_by_pipeline": False,
        "archive_bytes_read": False,
        "dataset_image_decoded": False,
        "model_scoring_started": False,
        "source_expected_pairs": int(source["expected_pairs"]),
        "source_archive_expected_bytes": int(source["expected_archive_bytes"]),
        "source_archive_expected_sha256": str(source["expected_archive_sha256"]),
        "bound_sha256": bound_hashes,
        "clarification_request_draft_sha256": _sha256(request_draft),
        "config_sha256": _sha256(config_path),
        "registry": str(registry_path.relative_to(project_root)),
        "registry_sha256": _sha256(registry_path),
        "next_authorized_action": "obtain_written_author_clarification_or_institutional_approval",
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
