from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from pairtrace_doc.descan_edits import (
    IMPLEMENTATION_ID,
    copy_move_edit,
    local_erase_edit,
)


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _cache_payload(
    root: Path, category: str, payload: bytes, suffix: str
) -> tuple[Path, str, bool]:
    digest = _hash_bytes(payload)
    path = root / category / digest[:2] / f"{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    hit = path.is_file()
    if hit:
        if _sha256(path) != digest:
            raise ValueError(f"content-addressed cache collision: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return path, digest, hit


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe DESCAN archive member: {name}")
    return path


def _pair_members(
    names: list[str], scan_prefix: str, clean_prefix: str
) -> list[dict[str, str]]:
    scan_root = PurePosixPath(scan_prefix)
    clean_root = PurePosixPath(clean_prefix)
    scans: dict[str, str] = {}
    cleans: dict[str, str] = {}
    for name in names:
        path = _safe_member_name(name)
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        parent = path.parent
        target: dict[str, str] | None = None
        if parent == scan_root:
            target = scans
        elif parent == clean_root:
            target = cleans
        if target is None:
            continue
        basename = path.name
        if basename in target:
            raise ValueError(f"duplicate DESCAN basename in one side: {basename}")
        target[basename] = name
    if set(scans) != set(cleans):
        missing_scan = sorted(set(cleans) - set(scans))
        missing_clean = sorted(set(scans) - set(cleans))
        raise ValueError(
            f"DESCAN mate topology changed: missing_scan={missing_scan}, "
            f"missing_clean={missing_clean}"
        )
    return [
        {"basename": basename, "scan_member": scans[basename], "clean_member": cleans[basename]}
        for basename in sorted(scans)
    ]


def _select_pairs(
    pairs: list[dict[str, str]], *, salt: str, count: int
) -> list[dict[str, str]]:
    ordered = sorted(
        pairs,
        key=lambda row: hashlib.sha256(
            f"{salt}{row['basename']}".encode("utf-8")
        ).hexdigest(),
    )
    return ordered[:count]


def _decode_tiff(payload: bytes, label: str) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as handle:
        if handle.format != "TIFF":
            raise ValueError(f"{label} is not TIFF")
        image = np.asarray(handle.convert("RGB"))
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label} did not decode as uint8 RGB")
    return image


def _approval_path(project_root: Path, license_config: dict[str, Any]) -> Path:
    available = []
    for path_field, hash_field in (
        (
            "written_author_clarification",
            "expected_written_author_clarification_sha256",
        ),
        ("institutional_approval", "expected_institutional_approval_sha256"),
    ):
        value = license_config.get(path_field)
        expected = license_config.get(hash_field)
        if value is None and expected is None:
            continue
        if value is None or expected is None:
            raise ValueError(f"incomplete license approval binding: {path_field}")
        path = _resolve(project_root, str(value))
        if _sha256(path) != str(expected):
            raise ValueError(f"license approval artifact changed: {path}")
        available.append(path)
    if len(available) != 1:
        raise ValueError("exactly one license approval artifact is required")
    return available[0]


def _nested_value(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"missing prerequisite field: {dotted_path}")
        current = current[part]
    return current


def _validate_prerequisites(
    project_root: Path, prerequisites: list[dict[str, Any]]
) -> None:
    for prerequisite in prerequisites:
        path = _resolve(project_root, str(prerequisite["path"]))
        if _sha256(path) != str(prerequisite["sha256"]):
            raise ValueError(f"DESCAN prerequisite SHA-256 changed: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for dotted_path, expected in prerequisite.get("required_values", {}).items():
            actual = _nested_value(payload, str(dotted_path))
            if actual != expected:
                raise PermissionError(
                    f"DESCAN prerequisite gate failed: {dotted_path}={actual!r}, "
                    f"expected {expected!r}"
                )


def _materialize_pair(
    *,
    basename: str,
    scan_payload: bytes,
    clean_payload: bytes,
    cache_root: Path,
    scratch: Path,
) -> tuple[dict[str, Any], int, int]:
    scan = _decode_tiff(scan_payload, f"scan:{basename}")
    clean = _decode_tiff(clean_payload, f"clean:{basename}")
    if scan.shape != clean.shape:
        raise ValueError(f"DESCAN pair geometry mismatch: {basename}")
    group_id = "descan18k:" + hashlib.sha256(basename.encode("utf-8")).hexdigest()[:20]
    cached: dict[str, tuple[Path, str, bool]] = {
        "scan": _cache_payload(cache_root, "scan", _png_bytes(scan), ".png"),
        "clean": _cache_payload(cache_root, "clean", _png_bytes(clean), ".png"),
    }
    record: dict[str, Any] = {
        "source_group_id": group_id,
        "source_dataset": "DESCAN-18K",
        "source_basename": basename,
        "height": int(scan.shape[0]),
        "width": int(scan.shape[1]),
        "scan": str(cached["scan"][0].relative_to(scratch)),
        "scan_sha256": cached["scan"][1],
        "clean": str(cached["clean"][0].relative_to(scratch)),
        "clean_sha256": cached["clean"][1],
        "editor_implementation": IMPLEMENTATION_ID,
        "attacks": {},
        "dataset_image_decoded": True,
        "model_score_read": False,
        "paper_evidence": False,
    }
    generators = {
        "copy_move": copy_move_edit,
        "local_erase": local_erase_edit,
    }
    for attack, generator in generators.items():
        try:
            result = generator(scan, group_id)
            cached[f"{attack}_candidate"] = _cache_payload(
                cache_root, "candidate", _png_bytes(result.candidate), ".png"
            )
            cached[f"{attack}_mask"] = _cache_payload(
                cache_root,
                "mask",
                _png_bytes(result.mask.astype(np.uint8) * 255),
                ".png",
            )
            record["attacks"][attack] = {
                "status": "ok",
                "candidate": str(
                    cached[f"{attack}_candidate"][0].relative_to(scratch)
                ),
                "candidate_sha256": cached[f"{attack}_candidate"][1],
                "mask": str(cached[f"{attack}_mask"][0].relative_to(scratch)),
                "mask_sha256": cached[f"{attack}_mask"][1],
                "metadata": result.metadata,
            }
        except Exception as error:
            record["attacks"][attack] = {
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
                "metadata": {
                    "implementation": IMPLEMENTATION_ID,
                    "group_id": group_id,
                    "attack": attack,
                },
            }
    failed_attacks = [
        attack
        for attack, value in record["attacks"].items()
        if value["status"] != "ok"
    ]
    record["status"] = "ok" if not failed_attacks else "partial_attack_failure"
    record["failed_attacks"] = failed_attacks
    hits = sum(value[2] for value in cached.values())
    return record, hits, len(cached) - hits


def _group_failure_record(basename: str, error: Exception) -> dict[str, Any]:
    group_id = "descan18k:" + hashlib.sha256(basename.encode("utf-8")).hexdigest()[:20]
    return {
        "source_group_id": group_id,
        "source_dataset": "DESCAN-18K",
        "source_basename": basename,
        "status": "failed",
        "failure_type": type(error).__name__,
        "failure_reason": str(error),
        "attacks": {
            attack: {
                "status": "not_run_due_group_failure",
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
                "metadata": {
                    "implementation": IMPLEMENTATION_ID,
                    "group_id": group_id,
                    "attack": attack,
                },
            }
            for attack in ("copy_move", "local_erase")
        },
        "dataset_image_decoded": False,
        "model_score_read": False,
        "paper_evidence": False,
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    if bool(runtime["archive_download_authorized"]):
        raise ValueError("materializer never downloads the source archive")
    required = (
        "license_gate_open",
        "archive_read_authorized",
        "dataset_image_decode_authorized",
        "edit_materialization_authorized",
    )
    if not all(bool(runtime[field]) for field in required):
        raise PermissionError("DESCAN license/read/decode/materialization gate is closed")
    if bool(runtime["model_scoring_authorized"]):
        raise ValueError("materialization cannot authorize model scoring")
    approval = _approval_path(project_root, config["license"])
    experiment = config["experiment"]
    for path_field, hash_field in (
        ("protocol", "expected_protocol_sha256"),
        ("editor_specification", "expected_editor_specification_sha256"),
        ("editor_source", "expected_editor_source_sha256"),
    ):
        path = _resolve(project_root, str(experiment[path_field]))
        if _sha256(path) != str(experiment[hash_field]):
            raise ValueError(f"bound DESCAN artifact changed: {path}")
    _validate_prerequisites(project_root, list(config.get("prerequisites", [])))
    source = config["source"]
    archive = _resolve(project_root, str(source["archive"]))
    if archive.stat().st_size != int(source["expected_archive_bytes"]):
        raise ValueError("DESCAN archive size changed")
    if _sha256(archive) != str(source["expected_archive_sha256"]):
        raise ValueError("DESCAN archive SHA-256 changed")
    stage = str(experiment["stage"])
    count = int(config["stages"][stage])
    manifest_path = _resolve(project_root, str(config["paths"]["manifest"]))
    with zipfile.ZipFile(archive, mode="r") as handle:
        pairs = _pair_members(
            [item.filename for item in handle.infolist() if not item.is_dir()],
            str(source["scan_prefix"]),
            str(source["clean_prefix"]),
        )
        if len(pairs) != int(source["expected_full_pairs"]):
            raise ValueError("DESCAN full pair count changed")
        selected = _select_pairs(
            pairs, salt=str(source["selection_salt"]), count=count
        )
        scratch = _resolve(project_root, str(config["paths"]["scratch_default"]))
        cache_root = scratch / str(config["paths"]["cache_dir"])
        records = []
        cache_hits = 0
        cache_writes = 0
        for pair in selected:
            try:
                record, hits, writes = _materialize_pair(
                    basename=pair["basename"],
                    scan_payload=handle.read(pair["scan_member"]),
                    clean_payload=handle.read(pair["clean_member"]),
                    cache_root=cache_root,
                    scratch=scratch,
                )
            except Exception as error:
                record = _group_failure_record(pair["basename"], error)
                hits = 0
                writes = 0
            records.append(record)
            cache_hits += hits
            cache_writes += writes
            _write_jsonl(manifest_path, records)
    summary_path = _resolve(project_root, str(config["paths"]["summary"]))
    attack_successes = {
        attack: sum(
            record.get("attacks", {}).get(attack, {}).get("status") == "ok"
            for record in records
        )
        for attack in ("copy_move", "local_erase")
    }
    attack_failures = {
        attack: len(records) - successes
        for attack, successes in attack_successes.items()
    }
    summary = {
        "status": f"descan18k_{stage}_materialized",
        "paper_evidence": False,
        "stage": stage,
        "selected_groups": len(records),
        "attacks_per_group": 2,
        "license_gate_open": True,
        "license_approval_artifact": str(approval.relative_to(project_root)),
        "license_approval_sha256": _sha256(approval),
        "archive_sha256": _sha256(archive),
        "dataset_image_decoded": True,
        "model_scoring_started": False,
        "cache_hits": cache_hits,
        "cache_writes": cache_writes,
        "group_failures": sum(record.get("status") == "failed" for record in records),
        "partial_attack_failures": sum(
            record.get("status") == "partial_attack_failure" for record in records
        ),
        "attack_successes": attack_successes,
        "attack_failures": attack_failures,
        "silent_failures": 0,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "editor_implementation": IMPLEMENTATION_ID,
        "manifest": str(manifest_path.relative_to(project_root)),
        "manifest_sha256": _sha256(manifest_path),
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
