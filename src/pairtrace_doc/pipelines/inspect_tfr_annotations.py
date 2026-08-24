from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile, ZipInfo

import yaml


_CANDIDATE_KEY_TERMS = {
    "authentic",
    "file",
    "filename",
    "forged",
    "forgery",
    "groundtruth",
    "gt",
    "id",
    "image",
    "img",
    "label",
    "mask",
    "original",
    "pair",
    "path",
    "source",
    "tamper",
    "tampered",
}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _normalized_key_tokens(value: str) -> set[str]:
    normalized = "".join(character.lower() if character.isalnum() else " " for character in value)
    tokens = set(normalized.split())
    tokens.add("".join(normalized.split()))
    return tokens


def _candidate_key(value: str) -> bool:
    tokens = _normalized_key_tokens(value)
    return bool(tokens & _CANDIDATE_KEY_TERMS) or any(
        term in token
        for token in tokens
        for term in _CANDIDATE_KEY_TERMS
        if len(term) >= 4
    )


def _scalar_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    raise TypeError(f"unsupported JSON scalar type: {type(value).__name__}")


def _string_suffix(value: str) -> str:
    suffix = PurePosixPath(value.replace("\\", "/")).suffix.lower()
    return suffix or "<none>"


def _inventory_json(
    value: Any,
    *,
    member: str,
    sample_limit_per_field: int,
    unique_hash_limit_per_field: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    node_types: Counter[str] = Counter()
    list_lengths: Counter[int] = Counter()
    dict_key_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    field_types: dict[str, Counter[str]] = defaultdict(Counter)
    field_suffixes: dict[str, Counter[str]] = defaultdict(Counter)
    field_unique_hashes: dict[str, set[str]] = defaultdict(set)
    field_unique_capped: set[str] = set()
    candidate_counts: Counter[str] = Counter()
    candidate_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    doctamper_string_references = 0
    max_depth = 0

    def visit(item: Any, path: str, depth: int, leaf_key: str | None) -> None:
        nonlocal doctamper_string_references, max_depth
        max_depth = max(max_depth, depth)
        if isinstance(item, dict):
            node_types["object"] += 1
            for raw_key, child in item.items():
                key = str(raw_key)
                dict_key_counts[key] += 1
                visit(child, f"{path}.{key}", depth + 1, key)
            return
        if isinstance(item, list):
            node_types["array"] += 1
            list_lengths[len(item)] += 1
            for child in item:
                visit(child, f"{path}[]", depth + 1, leaf_key)
            return

        scalar_name = _scalar_type(item)
        node_types[scalar_name] += 1
        field_counts[path] += 1
        field_types[path][scalar_name] += 1
        serialized = json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
        value_hash = _sha256_bytes(serialized)
        if len(field_unique_hashes[path]) < unique_hash_limit_per_field:
            field_unique_hashes[path].add(value_hash)
        else:
            field_unique_capped.add(path)
        if isinstance(item, str):
            field_suffixes[path][_string_suffix(item)] += 1
            if "doctamper" in item.lower():
                doctamper_string_references += 1
        if leaf_key is not None and _candidate_key(leaf_key):
            candidate_counts[path] += 1
            samples = candidate_samples[path]
            if len(samples) < sample_limit_per_field:
                samples.append(
                    {
                        "member": member,
                        "field_path": path,
                        "value": item,
                        "value_sha256": value_hash,
                        "value_type": scalar_name,
                    }
                )

    visit(value, "$", 0, None)
    field_rows = []
    for path in sorted(field_counts):
        field_rows.append(
            {
                "field_path": path,
                "count": field_counts[path],
                "value_types": dict(sorted(field_types[path].items())),
                "string_suffixes": dict(field_suffixes[path].most_common(20)),
                "unique_value_hashes_observed": len(field_unique_hashes[path]),
                "unique_count_capped": path in field_unique_capped,
            }
        )
    private_rows = [
        row
        for path in sorted(candidate_samples)
        for row in candidate_samples[path]
    ]
    summary = {
        "root_type": type(value).__name__,
        "top_level_length": len(value) if isinstance(value, (dict, list)) else None,
        "node_type_counts": dict(sorted(node_types.items())),
        "max_depth": max_depth,
        "list_length_counts": {
            str(key): count for key, count in sorted(list_lengths.items())
        },
        "dict_key_counts": dict(dict_key_counts.most_common()),
        "field_statistics": field_rows,
        "candidate_field_counts": dict(sorted(candidate_counts.items())),
        "doctamper_string_reference_count": doctamper_string_references,
    }
    return summary, private_rows


def _read_json_member(
    archive: ZipFile,
    info: ZipInfo,
    password: str,
    max_member_bytes: int,
) -> tuple[Any, bytes, str]:
    if info.file_size > max_member_bytes:
        raise ValueError(
            f"annotation member exceeds configured byte limit: {info.filename}"
        )
    try:
        with archive.open(info, pwd=password.encode("utf-8")) as handle:
            payload = handle.read(max_member_bytes + 1)
    except (BadZipFile, RuntimeError, NotImplementedError) as error:
        raise ValueError(
            f"unable to decrypt/read TFR annotation member {info.filename}"
        ) from error
    if len(payload) > max_member_bytes:
        raise ValueError(
            f"annotation member exceeds configured byte limit: {info.filename}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 JSON in {info.filename}") from error
    try:
        return json.loads(text), payload, "json"
    except json.JSONDecodeError as document_error:
        records: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as line_error:
                raise ValueError(
                    f"invalid JSON/JSONL in {info.filename} at line {line_number}"
                ) from line_error
        if not records:
            raise ValueError(f"empty JSON/JSONL annotation member: {info.filename}") from document_error
        return records, payload, "jsonl"


def _private_record(env_name: str) -> dict[str, Any]:
    value = os.environ.get(env_name)
    return {
        "env": env_name,
        "present": bool(value),
        "private_record_id_sha256": _sha256_bytes(value.encode("utf-8"))
        if value
        else None,
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    runtime = config["runtime"]
    prohibited_flags = (
        "gpu_launch_authorized",
        "method_training_authorized",
        "final_reserve_access_authorized",
    )
    if any(runtime[name] for name in prohibited_flags):
        raise ValueError("TFR annotation inspection cannot authorize GPU, training, or final-reserve access")

    source = config["source"]
    password_env = source["tfr_password_env"]
    password = os.environ.get(password_env)
    if not password:
        raise ValueError(f"required TFR password environment variable is not set: {password_env}")
    permissions = {
        "TFR": _private_record(source["tfr_permission_record_env"]),
        "DocTamper": _private_record(source["doctamper_permission_record_env"]),
    }
    missing_permissions = [
        name for name, record in permissions.items() if not record["present"]
    ]
    if missing_permissions:
        raise ValueError(
            "required institutional permission records are not bound: "
            + ", ".join(missing_permissions)
        )

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    archive_path = _resolve(scratch, paths["tfr_inner_archive"])
    if archive_path.stat().st_size != int(source["tfr_inner_archive_bytes"]):
        raise ValueError("TFR inner archive size changed")
    expected_members = list(source["expected_annotation_members"])
    if len(expected_members) != len(set(expected_members)):
        raise ValueError("expected TFR annotation member list contains duplicates")
    if any(not _safe_member(name) for name in expected_members):
        raise ValueError("expected TFR annotation member list contains unsafe paths")

    summaries: dict[str, Any] = {}
    private_rows: list[dict[str, Any]] = []
    with ZipFile(archive_path) as archive:
        info_by_name = {info.filename: info for info in archive.infolist()}
        actual_members = sorted(
            info.filename
            for info in archive.infolist()
            if PurePosixPath(info.filename).parts[:2] == ("data", "annotation")
            and not info.is_dir()
        )
        if actual_members != sorted(expected_members):
            raise ValueError("TFR annotation member set changed")
        for member in expected_members:
            info = info_by_name[member]
            if source["require_encrypted_members"] and not (info.flag_bits & 0x1):
                raise ValueError(f"expected encrypted TFR annotation member: {member}")
            value, payload, serialization_format = _read_json_member(
                archive,
                info,
                password,
                int(runtime["max_annotation_member_bytes"]),
            )
            inventory, samples = _inventory_json(
                value,
                member=member,
                sample_limit_per_field=int(runtime["sample_limit_per_candidate_field"]),
                unique_hash_limit_per_field=int(runtime["unique_hash_limit_per_field"]),
            )
            summaries[member] = {
                "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "encrypted": bool(info.flag_bits & 0x1),
                "serialization_format": serialization_format,
                "payload_sha256": _sha256_bytes(payload),
                **inventory,
            }
            private_rows.extend(samples)

    summary_path = _resolve(project_root, paths["summary"])
    private_samples_path = _resolve(project_root, paths["private_samples"])
    _write_jsonl(private_samples_path, private_rows)
    explicit_doctamper_references = sum(
        item["doctamper_string_reference_count"] for item in summaries.values()
    )
    summary = {
        "experiment": config["experiment"],
        "status": "annotation_audit_complete_mapping_unverified",
        "paper_evidence": False,
        "gpu_used": False,
        "method_training_performed": False,
        "final_reserve_accessed": False,
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        },
        "license_and_access": permissions,
        "annotation_members": summaries,
        "mapping_assessment": {
            "sample_level_authentic_forgery_mapping_verified": False,
            "explicit_doctamper_string_reference_count": explicit_doctamper_references,
            "decision": "manual_schema_and_identity_review_required",
            "reason": (
                "Annotation structure alone cannot join a DocTamper LMDB index to "
                "a TFR authentic source; a released explicit identity rule must be verified."
            ),
        },
        "outputs": {
            "private_candidate_samples": str(private_samples_path.relative_to(project_root)),
            "private_candidate_samples_sha256": _sha256(private_samples_path),
            "private_candidate_sample_rows": len(private_rows),
        },
        "gpu_gate": {
            "authorized": False,
            "blockers": ["sample_level_authentic_forgery_mapping_unverified"],
        },
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect encrypted TFR annotation JSON without authorizing training"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
