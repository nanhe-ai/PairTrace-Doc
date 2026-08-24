from __future__ import annotations

import argparse
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile

import numpy as np
import yaml
from PIL import Image


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _private_record(env_name: str) -> dict[str, Any]:
    value = os.environ.get(env_name)
    return {
        "env": env_name,
        "present": bool(value),
        "private_record_id_sha256": _sha256_bytes(value.encode("utf-8"))
        if value
        else None,
    }


def _ranked_groups(groups: Iterable[str], seed: int) -> list[str]:
    return sorted(
        set(groups),
        key=lambda value: (
            hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest(),
            value,
        ),
    )


def _member_bytes(
    archive: ZipFile, name: str, password: str, max_member_bytes: int
) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > max_member_bytes:
        raise ValueError(f"TFR member exceeds byte limit: {name}")
    try:
        with archive.open(info, pwd=password.encode("utf-8")) as handle:
            payload = handle.read(max_member_bytes + 1)
    except (BadZipFile, RuntimeError, NotImplementedError) as error:
        raise ValueError(f"unable to decrypt/read TFR member: {name}") from error
    if len(payload) > max_member_bytes:
        raise ValueError(f"TFR member exceeds byte limit: {name}")
    return payload


def _binary_mask_png(payload: bytes, foreground_gt: int) -> tuple[bytes, np.ndarray]:
    with Image.open(BytesIO(payload)) as handle:
        mask = (np.asarray(handle.convert("L")) > foreground_gt).astype(np.uint8)
    if not np.any(mask):
        raise ValueError("thresholded TFR mask is empty")
    output = BytesIO()
    Image.fromarray(mask * 255).save(output, format="PNG")
    return output.getvalue(), mask


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if runtime["gpu_launch_authorized"] or runtime["method_training_authorized"]:
        raise ValueError("split freeze cannot authorize GPU or training")
    if runtime["final_reserve_access_authorized"]:
        raise ValueError("split freeze cannot access the consumed final reserve")

    source = config["source"]
    password = os.environ.get(source["tfr_password_env"])
    if not password:
        raise ValueError("required TFR password is not bound")
    permission = _private_record(source["tfr_permission_record_env"])
    if not permission["present"]:
        raise ValueError("required TFR permission record is not bound")

    paths = config["paths"]
    audit_records_path = _resolve(project_root, paths["audit_records"])
    if _sha256(audit_records_path) != source["expected_audit_records_sha256"]:
        raise ValueError("full TFR pair-audit records changed")
    rows = _read_jsonl(audit_records_path)
    if len(rows) != int(source["expected_pairs"]):
        raise ValueError("full TFR pair-audit record count changed")
    if any(not row["valid_pair"] for row in rows):
        raise ValueError("full TFR pair-audit contains failed pairs")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("full TFR pair-audit contains duplicate record IDs")

    split = config["split"]
    ranked_groups = _ranked_groups(
        (str(row["source_group_id"]) for row in rows), int(split["seed"])
    )
    expected_groups = int(source["expected_source_groups"])
    if len(ranked_groups) != expected_groups:
        raise ValueError("full TFR pair-audit source-group count changed")
    train_count = int(split["train_source_groups"])
    validation_count = int(split["validation_source_groups"])
    holdout_count = int(split["holdout_source_groups"])
    if train_count + validation_count + holdout_count != expected_groups:
        raise ValueError("configured split group counts do not cover the population")
    role_by_group = {
        group: (
            "train"
            if rank < train_count
            else "validation"
            if rank < train_count + validation_count
            else "holdout"
        )
        for rank, group in enumerate(ranked_groups)
    }
    assignment_payload = [
        {
            "record_id": row["record_id"],
            "source_group_id": row["source_group_id"],
            "role": role_by_group[str(row["source_group_id"])],
        }
        for row in sorted(rows, key=lambda item: item["record_id"])
    ]
    freeze_id = _sha256_bytes(
        json.dumps(
            {
                "audit_records_sha256": source["expected_audit_records_sha256"],
                "split": split,
                "assignments": assignment_payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    archive_path = _resolve(scratch, paths["tfr_inner_archive"])
    if archive_path.stat().st_size != int(source["tfr_inner_archive_bytes"]):
        raise ValueError("TFR inner archive size changed")
    derived_root = _resolve(scratch, paths["derived_root"])
    max_member_bytes = int(runtime["max_member_bytes"])
    foreground_gt = int(config["mask_processing"]["foreground_gt"])
    manifest_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    materialized = 0
    cache_hits = 0
    with ZipFile(archive_path) as archive:
        for row in sorted(rows, key=lambda item: item["record_id"]):
            group = str(row["source_group_id"])
            role = role_by_group[group]
            membership_rows.append(
                {
                    "record_id": row["record_id"],
                    "source_group_id": group,
                    "role": role,
                    "authentic_archive_path": row["authentic_path"],
                    "tampered_archive_path": row["tampered_path"],
                    "mask_archive_path": row["mask_path"],
                    "authentic_sha256": row["authentic_sha256"],
                    "tampered_sha256": row["tampered_sha256"],
                    "raw_mask_sha256": row["mask_sha256"],
                    "freeze_id": freeze_id,
                }
            )
            if role == "holdout":
                continue
            group_key = _sha256_bytes(group.encode("utf-8"))[:16]
            record_key = str(row["record_id"])[:16]
            directory = derived_root / role / group_key
            authentic_path = directory / "authentic.jpg"
            tampered_path = directory / f"{record_key}_tampered.jpg"
            mask_path = directory / f"{record_key}_mask.png"
            paths_ready = all(
                path.is_file() for path in (authentic_path, tampered_path, mask_path)
            )
            if paths_ready:
                cache_hits += 1
            else:
                authentic_payload = _member_bytes(
                    archive, row["authentic_path"], password, max_member_bytes
                )
                tampered_payload = _member_bytes(
                    archive, row["tampered_path"], password, max_member_bytes
                )
                mask_payload = _member_bytes(
                    archive, row["mask_path"], password, max_member_bytes
                )
                if _sha256_bytes(authentic_payload) != row["authentic_sha256"]:
                    raise ValueError("TFR authentic payload hash changed")
                if _sha256_bytes(tampered_payload) != row["tampered_sha256"]:
                    raise ValueError("TFR tampered payload hash changed")
                if _sha256_bytes(mask_payload) != row["mask_sha256"]:
                    raise ValueError("TFR mask payload hash changed")
                binary_mask_payload, _mask = _binary_mask_png(
                    mask_payload, foreground_gt
                )
                _atomic_bytes(authentic_path, authentic_payload)
                _atomic_bytes(tampered_path, tampered_payload)
                _atomic_bytes(mask_path, binary_mask_payload)
            materialized += 1
            manifest_rows.append(
                {
                    "sample_id": f"tfr-ettd:{row['record_id']}",
                    "source_group_id": group,
                    "source_dataset": "TFR-ETTD",
                    "selected_generator": "copy_move_splice",
                    "pilot_role": role,
                    "role": role,
                    "image": str(tampered_path.relative_to(scratch)),
                    "authentic": str(authentic_path.relative_to(scratch)),
                    "mask": str(mask_path.relative_to(scratch)),
                    "image_sha256": _sha256(tampered_path),
                    "authentic_sha256": _sha256(authentic_path),
                    "mask_sha256": _sha256(mask_path),
                    "image_height": int(row["image_height"]),
                    "image_width": int(row["image_width"]),
                    "mask_processing": "foreground_gt_160",
                    "mapping_rule": row["mapping_rule"],
                    "freeze_id": freeze_id,
                    "paper_evidence": False,
                    "valid": True,
                    "errors": [],
                }
            )

    manifest_path = _resolve(project_root, paths["train_validation_manifest"])
    membership_path = _resolve(project_root, paths["membership"])
    holdout_membership_path = _resolve(project_root, paths["holdout_membership"])
    summary_path = _resolve(project_root, paths["summary"])
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(membership_path, membership_rows)
    _write_jsonl(
        holdout_membership_path,
        (row for row in membership_rows if row["role"] == "holdout"),
    )
    pair_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for role in ("train", "validation", "holdout"):
        pair_counts[role] = sum(row["role"] == role for row in membership_rows)
        group_counts[role] = len(
            {
                row["source_group_id"]
                for row in membership_rows
                if row["role"] == role
            }
        )
    summary = {
        "experiment": config["experiment"],
        "status": "source_disjoint_split_frozen_gpu_preflight_ready",
        "paper_evidence": False,
        "gpu_used": False,
        "method_training_performed": False,
        "final_reserve_accessed": False,
        "license_and_access": {"TFR": permission},
        "freeze_id": freeze_id,
        "source_audit_records_sha256": source["expected_audit_records_sha256"],
        "split": {
            "seed": int(split["seed"]),
            "group_counts": group_counts,
            "pair_counts": pair_counts,
            "source_group_overlap": False,
        },
        "materialization": {
            "train_validation_pairs": materialized,
            "cache_hits": cache_hits,
            "holdout_pairs_materialized": 0,
            "raw_soft_masks_released_or_copied": False,
            "binary_mask_rule": "raw grayscale > 160",
        },
        "outputs": {
            "train_validation_manifest": str(manifest_path.relative_to(project_root)),
            "train_validation_manifest_sha256": _sha256(manifest_path),
            "membership": str(membership_path.relative_to(project_root)),
            "membership_sha256": _sha256(membership_path),
            "holdout_membership": str(
                holdout_membership_path.relative_to(project_root)
            ),
            "holdout_membership_sha256": _sha256(holdout_membership_path),
        },
        "gpu_gate": {"preflight_authorized": True, "full_training_authorized": False},
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze and materialize the TFR-internal paired split"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
