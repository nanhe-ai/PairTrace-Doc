from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


SECRET_PATTERNS = {
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "openai_like_secret": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credentialed_url": re.compile(rb"https?://[^\s/:]+:[^\s/@]+@"),
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


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


def _checkpoint_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = list(payload.get("records", [])) + list(payload.get("existing_controls", []))
    records = []
    for value in values:
        checkpoint = value.get("checkpoint")
        expected = value.get("sha256") or value.get("checkpoint_sha256")
        if checkpoint is None:
            continue
        records.append(
            {
                "checkpoint": str(checkpoint),
                "expected_sha256": str(expected) if expected is not None else None,
                "source_manifest": path.name,
                "declared_status": value.get("status", "ready" if expected else "pending"),
            }
        )
    return records


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    if bool(experiment["package_copy_authorized"]):
        raise ValueError("inventory gate does not copy a release archive")
    if bool(experiment["checkpoint_redistribution_authorized"]):
        raise ValueError("checkpoint redistribution needs a separate license decision")
    prohibited = tuple(str(value).rstrip("/") + "/" for value in config["prohibited_prefixes"])
    generated_paths = {
        _resolve(project_root, str(config["paths"][field]))
        for field in ("inventory", "summary", "checksum_file")
    }
    paths: set[Path] = set()
    for pattern in config["include_patterns"]:
        for value in glob.glob(str(project_root / str(pattern)), recursive=True):
            path = Path(value).resolve()
            if path.is_file() and path not in generated_paths:
                paths.add(path)
    file_records: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    for path in sorted(paths):
        relative = path.relative_to(project_root).as_posix()
        if any((relative + "/").startswith(prefix) for prefix in prohibited):
            raise ValueError(f"prohibited release path selected: {relative}")
        payload = path.read_bytes()
        matches = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(payload)]
        if matches:
            findings.extend({"path": relative, "finding": match} for match in matches)
        file_records.append(
            {
                "record_type": "release_file",
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "redistribution_status": "candidate_after_author_anonymity_review",
                "contains_dataset_pixels": False,
                "contains_checkpoint_bytes": False,
            }
        )
    checkpoint_candidates: dict[str, dict[str, Any]] = {}
    for value in config["checkpoint_manifests"]:
        manifest = _resolve(project_root, str(value))
        for record in _checkpoint_records(manifest):
            checkpoint_candidates.setdefault(record["checkpoint"], record)
    checkpoint_records: list[dict[str, Any]] = []
    verified = 0
    pending = 0
    for relative, record in sorted(checkpoint_candidates.items()):
        path = _resolve(project_root, relative)
        exists = path.is_file()
        expected = record["expected_sha256"]
        actual = _sha256(path) if exists else None
        if exists and expected is not None and actual != expected:
            raise ValueError(f"checkpoint changed: {relative}")
        verified += int(exists and expected is not None)
        pending += int(not exists or expected is None)
        checkpoint_records.append(
            {
                "record_type": "checkpoint_inventory",
                "path": relative,
                "bytes": path.stat().st_size if exists else None,
                "sha256": actual or expected,
                "exists": exists,
                "declared_status": record["declared_status"],
                "source_manifest": record["source_manifest"],
                "redistribution_status": "metadata_only_pending_upstream_and_author_license_review",
                "contains_checkpoint_bytes": exists,
                "included_in_release_file_inventory": False,
            }
        )
    if findings:
        raise ValueError(f"potential credential material found: {findings}")
    inventory_path = _resolve(project_root, str(config["paths"]["inventory"]))
    checksum_path = _resolve(project_root, str(config["paths"]["checksum_file"]))
    summary_path = _resolve(project_root, str(config["paths"]["summary"]))
    records = file_records + checkpoint_records
    _write_jsonl(inventory_path, records)
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text(
        "".join(
            f"{record['sha256']}  {record['path']}\n"
            for record in file_records
        ),
        encoding="utf-8",
    )
    summary = {
        "status": "anonymous_release_inventory_built_not_packaged",
        "paper_evidence": False,
        "archive_created": False,
        "stable_archive_identifier_assigned": False,
        "candidate_release_files": len(file_records),
        "candidate_release_bytes": sum(record["bytes"] for record in file_records),
        "checkpoint_inventory_records": len(checkpoint_records),
        "checkpoint_hashes_verified": verified,
        "checkpoint_records_pending_training_or_hash": pending,
        "checkpoint_bytes_in_release": False,
        "restricted_dataset_pixels_in_release": False,
        "credential_findings": 0,
        "remaining_manual_gates": [
            "author anonymity review",
            "checkpoint and third-party license review",
            "stable anonymous archive upload and identifier",
        ],
        "inventory": str(inventory_path.relative_to(project_root)),
        "inventory_sha256": _sha256(inventory_path),
        "checksum_file": str(checksum_path.relative_to(project_root)),
        "checksum_file_sha256": _sha256(checksum_path),
        "config_sha256": _sha256(config_path),
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
