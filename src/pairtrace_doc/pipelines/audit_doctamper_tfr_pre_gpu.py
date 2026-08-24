from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import os
import pickle
import pickletools
import subprocess
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile

import lmdb
import numpy as np
import yaml
from PIL import Image


_FORBIDDEN_PICKLE_OPCODES = {
    "BINPERSID",
    "BUILD",
    "EXT1",
    "EXT2",
    "EXT4",
    "GLOBAL",
    "INST",
    "NEWOBJ",
    "NEWOBJ_EX",
    "OBJ",
    "PERSID",
    "REDUCE",
    "STACK_GLOBAL",
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
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
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


def _safe_type_mapping(path: Path, expected_samples: int) -> dict[int, str]:
    payload = path.read_bytes()
    forbidden = [
        (opcode.name, position)
        for opcode, _argument, position in pickletools.genops(payload)
        if opcode.name in _FORBIDDEN_PICKLE_OPCODES
    ]
    if forbidden:
        raise ValueError(f"unsafe pickle opcodes in {path}: {forbidden[:5]}")
    value = pickle.loads(payload)  # noqa: S301 - opcode-gated plain dict only
    if not isinstance(value, dict):
        raise ValueError(f"tampering-type mapping is not a dict: {path}")
    mapping: dict[int, str] = {}
    for key, item in value.items():
        if not isinstance(key, int) or isinstance(key, bool):
            raise ValueError(f"non-integer tampering-type key in {path}")
        if item not in {"CM", "SP", "GE"}:
            raise ValueError(f"invalid tampering type in {path}: {item!r}")
        mapping[key] = str(item)
    if set(mapping) != set(range(expected_samples)):
        raise ValueError(f"tampering-type keys are not contiguous in {path}")
    return mapping


def _lmdb_inventory(path: Path, expected_samples: int) -> dict[str, Any]:
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=32,
    )
    prefixes: Counter[str] = Counter()
    image_indices: set[int] = set()
    label_indices: set[int] = set()
    unexpected: list[str] = []
    with env.begin(write=False) as txn:
        raw_samples = txn.get(b"num-samples")
        if raw_samples is None:
            raise ValueError(f"num-samples is missing from {path}")
        num_samples = int(raw_samples)
        cursor = txn.cursor()
        for key in cursor.iternext(keys=True, values=False):
            text = key.decode("utf-8", "replace")
            if text == "num-samples":
                prefixes["num-samples"] += 1
                continue
            prefix, separator, suffix = text.partition("-")
            prefixes[prefix] += 1
            if not separator or not suffix.isdigit():
                unexpected.append(text)
                continue
            index = int(suffix)
            if prefix == "image":
                image_indices.add(index)
            elif prefix == "label":
                label_indices.add(index)
            else:
                unexpected.append(text)
    stat = env.stat()
    env.close()
    expected_indices = set(range(expected_samples))
    errors: list[str] = []
    if num_samples != expected_samples:
        errors.append("num_samples_changed")
    if stat["entries"] != 2 * expected_samples + 1:
        errors.append("entry_count_changed")
    if image_indices != expected_indices:
        errors.append("image_indices_not_contiguous")
    if label_indices != expected_indices:
        errors.append("label_indices_not_contiguous")
    if unexpected:
        errors.append("unexpected_keys")
    return {
        "path": str(path),
        "num_samples": num_samples,
        "entries": int(stat["entries"]),
        "prefix_counts": dict(sorted(prefixes.items())),
        "image_indices_contiguous": image_indices == expected_indices,
        "label_indices_contiguous": label_indices == expected_indices,
        "source_identity_or_filename_keys": 0,
        "unexpected_key_count": len(unexpected),
        "unexpected_key_samples": unexpected[:20],
        "valid": not errors,
        "errors": errors,
    }


def _deterministic_indices(
    num_samples: int, limit: int, seed: int, namespace: str
) -> list[int]:
    if limit <= 0 or limit > num_samples:
        raise ValueError("decode limit must be between one and num_samples")
    ranked = heapq.nsmallest(
        limit,
        (
            (
                hashlib.sha256(
                    f"{seed}|{namespace}|{index}".encode("utf-8")
                ).digest(),
                index,
            )
            for index in range(num_samples)
        ),
    )
    return [index for _rank, index in ranked]


def _decode_record(
    txn: lmdb.Transaction,
    dataset_name: str,
    index: int,
    selection_rank: int,
    tampering_type: str,
    decode_limit: int,
) -> dict[str, Any]:
    image_key = f"image-{index:09d}".encode("utf-8")
    label_key = f"label-{index:09d}".encode("utf-8")
    image_payload = txn.get(image_key)
    mask_payload = txn.get(label_key)
    errors: list[str] = []
    record: dict[str, Any] = {
        "record_id": f"doctamper:{dataset_name}:{index:09d}",
        "source_dataset": "DocTamper",
        "official_partition": dataset_name,
        "official_index": index,
        "tampering_type": tampering_type,
        "selection_rank": selection_rank,
        "decode_limit_per_partition": decode_limit,
        "label": "forged",
        "exact_mask_claimed": True,
        "source_group_id": None,
        "authentic_pair_id": None,
        "paper_evidence": False,
        "gpu_used": False,
    }
    if image_payload is None:
        errors.append("missing_image_value")
    if mask_payload is None:
        errors.append("missing_label_value")
    if errors:
        return {**record, "valid": False, "errors": errors}
    assert image_payload is not None
    assert mask_payload is not None
    try:
        with Image.open(BytesIO(image_payload)) as image_handle:
            image_format = image_handle.format
            image_mode = image_handle.mode
            image_handle.load()
            image_width, image_height = image_handle.size
        with Image.open(BytesIO(mask_payload)) as mask_handle:
            mask_format = mask_handle.format
            mask_mode = mask_handle.mode
            mask_array = np.asarray(mask_handle.convert("L"))
            mask_width, mask_height = mask_handle.size
        mask_values = sorted(int(item) for item in np.unique(mask_array))
        if set(mask_values) not in ({0, 1}, {0, 255}):
            errors.append("mask_not_binary_0_1_or_0_255")
        positive_pixels = int(np.count_nonzero(mask_array))
        if positive_pixels == 0:
            errors.append("empty_forged_mask")
        if (image_width, image_height) != (mask_width, mask_height):
            errors.append("image_mask_dimension_mismatch")
        record.update(
            {
                "image_bytes": len(image_payload),
                "mask_bytes": len(mask_payload),
                "image_sha256": _sha256_bytes(image_payload),
                "mask_sha256": _sha256_bytes(mask_payload),
                "image_format": image_format,
                "image_mode": image_mode,
                "image_width": image_width,
                "image_height": image_height,
                "mask_format": mask_format,
                "mask_mode": mask_mode,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "mask_values": mask_values,
                "mask_positive_pixels": positive_pixels,
                "mask_positive_fraction": positive_pixels
                / float(mask_width * mask_height),
            }
        )
    except Exception as error:
        errors.append(f"decode_error:{type(error).__name__}:{error}")
    record["valid"] = not errors
    record["errors"] = errors
    return record


def _audit_samples(
    path: Path,
    dataset_name: str,
    num_samples: int,
    decode_limit: int,
    seed: int,
    tampering_types: dict[int, str],
) -> list[dict[str, Any]]:
    indices = _deterministic_indices(num_samples, decode_limit, seed, dataset_name)
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=32,
    )
    rows: list[dict[str, Any]] = []
    with env.begin(write=False) as txn:
        for rank, index in enumerate(indices):
            rows.append(
                _decode_record(
                    txn,
                    dataset_name,
                    index,
                    rank,
                    tampering_types[index],
                    decode_limit,
                )
            )
    env.close()
    return rows


def _git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _tfr_inventory(path: Path, password_env: str) -> dict[str, Any]:
    prefixes: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    unsafe: list[str] = []
    annotation_members: list[str] = []
    encrypted_members = 0
    regular_bytes = 0
    password = os.environ.get(password_env)
    password_valid: bool | None = None
    password_error: str | None = None
    with ZipFile(path) as archive:
        infos = archive.infolist()
        regular = [info for info in infos if not info.is_dir()]
        for info in regular:
            regular_bytes += info.file_size
            if info.flag_bits & 0x1:
                encrypted_members += 1
            if not _safe_member(info.filename):
                unsafe.append(info.filename)
            pure = PurePosixPath(info.filename)
            prefixes["/".join(pure.parts[:4])] += 1
            suffixes[pure.suffix.lower() or "<none>"] += 1
            if pure.parts[:2] == ("data", "annotation"):
                annotation_members.append(info.filename)
        if password:
            probe = next((item for item in regular if item.flag_bits & 0x1), None)
            if probe is None:
                password_valid = True
            else:
                try:
                    with archive.open(probe, pwd=password.encode("utf-8")) as handle:
                        handle.read(1)
                    password_valid = True
                except (BadZipFile, RuntimeError, NotImplementedError) as error:
                    password_valid = False
                    password_error = f"{type(error).__name__}:{error}"
    return {
        "path": str(path),
        "archive_bytes": path.stat().st_size,
        "members": len(infos),
        "regular_files": len(regular),
        "regular_uncompressed_bytes": regular_bytes,
        "encrypted_regular_files": encrypted_members,
        "unsafe_members": unsafe,
        "annotation_members": sorted(annotation_members),
        "suffix_counts": dict(sorted(suffixes.items())),
        "top_prefix_counts": dict(prefixes.most_common(80)),
        "password_env": password_env,
        "password_supplied": bool(password),
        "password_valid": password_valid,
        "password_error": password_error,
    }


def _permission_record(env_name: str) -> dict[str, Any]:
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
    if runtime["gpu_launch_authorized"]:
        raise ValueError("pre-GPU external-data audit cannot authorize GPU use")
    if runtime["method_training_authorized"]:
        raise ValueError("pre-GPU external-data audit cannot authorize training")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    summary_path = _resolve(project_root, paths["summary"])
    records_path = _resolve(project_root, paths["records"])
    log_path = _resolve(project_root, paths["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    source = config["source"]
    doctamper_root = _resolve(scratch, paths["doctamper_root"])
    type_root = _resolve(scratch, paths["doctamper_type_root"])
    tfr_archive = _resolve(scratch, paths["tfr_inner_archive"])
    doctamper_repo = _resolve(scratch, paths["doctamper_repository"])
    textshield_repo = _resolve(scratch, paths["textshield_repository"])
    if tfr_archive.stat().st_size != int(source["tfr_inner_archive_bytes"]):
        raise ValueError("TFR inner archive size changed")

    inventories: dict[str, Any] = {}
    type_counts: dict[str, dict[str, int]] = {}
    records: list[dict[str, Any]] = []
    for dataset_name, specification in source["doctamper_partitions"].items():
        expected_samples = int(specification["expected_samples"])
        lmdb_path = doctamper_root / dataset_name
        if (lmdb_path / "data.mdb").stat().st_size != int(
            specification["expected_data_mdb_bytes"]
        ):
            raise ValueError(f"data.mdb size changed for {dataset_name}")
        inventory = _lmdb_inventory(lmdb_path, expected_samples)
        inventories[dataset_name] = inventory
        type_mapping = _safe_type_mapping(
            type_root / f"{dataset_name}.pk", expected_samples
        )
        type_counts[dataset_name] = dict(
            sorted(Counter(type_mapping.values()).items())
        )
        records.extend(
            _audit_samples(
                lmdb_path,
                dataset_name,
                expected_samples,
                int(runtime["decode_limit_per_partition"]),
                int(runtime["seed"]),
                type_mapping,
            )
        )

    records.sort(key=lambda row: (row["official_partition"], row["selection_rank"]))
    _write_jsonl(records_path, records)
    invalid = [row for row in records if not row["valid"]]
    value_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    for row in records:
        if row.get("mask_values") is not None:
            value_counts[json.dumps(row["mask_values"])] += 1
        if row.get("image_format") is not None:
            format_counts[str(row["image_format"])] += 1

    tfr = _tfr_inventory(tfr_archive, source["tfr_password_env"])
    permissions = {
        "TFR": _permission_record(source["tfr_permission_record_env"]),
        "DocTamper": _permission_record(
            source["doctamper_permission_record_env"]
        ),
    }
    blockers: list[str] = []
    if tfr["unsafe_members"]:
        blockers.append("tfr_unsafe_archive_members")
    if not tfr["password_supplied"] or tfr["password_valid"] is not True:
        blockers.append("tfr_archive_password_missing_or_invalid")
    if not permissions["TFR"]["present"]:
        blockers.append("tfr_institutional_permission_record_missing")
    if not permissions["DocTamper"]["present"]:
        blockers.append("doctamper_institutional_permission_record_missing")
    if invalid:
        blockers.append("doctamper_decode_or_mask_quality_failures")
    if any(not item["valid"] for item in inventories.values()):
        blockers.append("doctamper_lmdb_key_integrity_failure")
    mapping_path_value = paths.get("pair_mapping")
    mapping_path = (
        _resolve(project_root, mapping_path_value) if mapping_path_value else None
    )
    if mapping_path is None or not mapping_path.is_file():
        blockers.append("sample_level_authentic_forgery_mapping_missing")

    source_revisions = {
        "DocTamper": _git_revision(doctamper_repo),
        "TextShield": _git_revision(textshield_repo),
    }
    if source_revisions["DocTamper"] != source["doctamper_revision"]:
        blockers.append("doctamper_repository_revision_changed")
    if source_revisions["TextShield"] != source["textshield_revision"]:
        blockers.append("textshield_repository_revision_changed")

    summary = {
        "experiment": config["experiment"],
        "status": "blocked_before_gpu" if blockers else "gpu_pilot_preconditions_met",
        "gpu_ready": not blockers,
        "gpu_used": False,
        "method_training_performed": False,
        "paper_evidence": False,
        "source_revisions": source_revisions,
        "license_and_access": permissions,
        "tfr": tfr,
        "doctamper": {
            "partitions": inventories,
            "tampering_type_counts": type_counts,
            "total_samples": sum(
                int(item["expected_samples"])
                for item in source["doctamper_partitions"].values()
            ),
            "lmdb_source_identity_or_filename_keys": sum(
                item["source_identity_or_filename_keys"]
                for item in inventories.values()
            ),
            "decoded_records": len(records),
            "valid_decoded_records": len(records) - len(invalid),
            "invalid_decoded_records": len(invalid),
            "mask_value_counts": dict(sorted(value_counts.items())),
            "image_format_counts": dict(sorted(format_counts.items())),
        },
        "pair_identity": {
            "mapping_path": str(mapping_path) if mapping_path else None,
            "mapping_present": bool(mapping_path and mapping_path.is_file()),
            "doctamper_lmdb_contains_source_identity": False,
            "one_to_one_or_one_to_many_mapping_verified": False,
        },
        "outputs": {
            "records": str(records_path.relative_to(project_root)),
            "records_sha256": _sha256(records_path),
            "log": str(log_path.relative_to(project_root)),
        },
        "gpu_gate": {
            "authorized": False,
            "blockers": sorted(set(blockers)),
            "final_reserve_accessed": False,
            "threshold_or_model_selected": False,
        },
    }
    _write_json(summary_path, summary)
    logging.info("DocTamper/TFR pre-GPU audit completed summary=%s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CPU-only DocTamper/TFR pre-GPU admission audit"
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
