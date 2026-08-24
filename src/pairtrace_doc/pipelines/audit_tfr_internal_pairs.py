from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile, ZipInfo

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
                    default=_json_default,
                )
                + "\n"
            )
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


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _read_member(
    archive: ZipFile,
    info: ZipInfo,
    password: str,
    max_member_bytes: int,
) -> bytes:
    if not _safe_member(info.filename):
        raise ValueError(f"unsafe TFR member path: {info.filename}")
    if info.file_size > max_member_bytes:
        raise ValueError(f"TFR member exceeds byte limit: {info.filename}")
    try:
        with archive.open(info, pwd=password.encode("utf-8")) as handle:
            payload = handle.read(max_member_bytes + 1)
    except (BadZipFile, RuntimeError, NotImplementedError) as error:
        raise ValueError(f"unable to decrypt/read TFR member: {info.filename}") from error
    if len(payload) > max_member_bytes:
        raise ValueError(f"TFR member exceeds byte limit: {info.filename}")
    return payload


def _decode_rgb(payload: bytes) -> tuple[np.ndarray, str | None]:
    with Image.open(BytesIO(payload)) as handle:
        image_format = handle.format
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    return array, image_format


def _decode_mask(payload: bytes) -> tuple[np.ndarray, str | None, list[int]]:
    with Image.open(BytesIO(payload)) as handle:
        image_format = handle.format
        array = np.asarray(handle.convert("L"), dtype=np.uint8)
    values = sorted(int(item) for item in np.unique(array))
    return array, image_format, values


def _ranked_groups(groups: Iterable[str], seed: int, limit: int | None) -> list[str]:
    ranked = sorted(
        set(groups),
        key=lambda value: (
            hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest(),
            value,
        ),
    )
    if limit is None:
        return ranked
    if limit <= 0 or limit > len(ranked):
        raise ValueError("selected_group_limit must be positive and no larger than the pool")
    return ranked[:limit]


def _candidate_inventory(
    archive: ZipFile,
    *,
    split: str,
    authentic_directory: str,
    tampered_directory: str,
    authentic_pattern: str,
    tampered_pattern: str,
) -> tuple[dict[str, str], list[dict[str, str]], dict[str, Any]]:
    authentic_regex = re.compile(authentic_pattern)
    tampered_regex = re.compile(tampered_pattern)
    authentic_by_source: dict[str, str] = {}
    tampered_rows: list[dict[str, str]] = []
    mask_names: set[str] = set()
    ignored_authentic = 0
    ignored_tampered = 0
    duplicate_authentic_ids: list[str] = []
    for info in archive.infolist():
        parts = PurePosixPath(info.filename).parts
        if len(parts) != 5 or parts[:2] != ("data", split) or info.is_dir():
            continue
        directory, kind, filename = parts[2], parts[3], parts[4]
        if directory == authentic_directory and kind == "imgs":
            match = authentic_regex.fullmatch(filename)
            if match is None:
                ignored_authentic += 1
                continue
            source_id = match.group("source_id")
            if source_id in authentic_by_source:
                duplicate_authentic_ids.append(source_id)
            else:
                authentic_by_source[source_id] = info.filename
        elif directory == tampered_directory and kind == "imgs":
            match = tampered_regex.fullmatch(filename)
            if match is None:
                ignored_tampered += 1
                continue
            tampered_rows.append(
                {
                    "source_id": match.group("source_id"),
                    "tampered_path": info.filename,
                    "tampered_filename": filename,
                }
            )
        elif directory == tampered_directory and kind == "masks":
            mask_names.add(filename)
    if duplicate_authentic_ids:
        raise ValueError("candidate mapping contains duplicate authentic source IDs")

    candidates: list[dict[str, str]] = []
    missing_authentic = 0
    missing_mask = 0
    for row in tampered_rows:
        authentic_path = authentic_by_source.get(row["source_id"])
        if authentic_path is None:
            missing_authentic += 1
            continue
        tampered = PurePosixPath(row["tampered_path"])
        mask_filename = tampered.with_suffix(".png").name
        if mask_filename not in mask_names:
            missing_mask += 1
            continue
        mask_path = str(tampered.parent.parent / "masks" / mask_filename)
        candidates.append(
            {
                "source_id": row["source_id"],
                "authentic_path": authentic_path,
                "tampered_path": row["tampered_path"],
                "mask_path": mask_path,
            }
        )
    candidates.sort(
        key=lambda row: (row["source_id"], row["tampered_path"])
    )
    inventory = {
        "authentic_files_with_parseable_source_id": len(authentic_by_source),
        "tampered_files_with_parseable_source_id": len(tampered_rows),
        "candidate_pairs": len(candidates),
        "candidate_source_groups": len({row["source_id"] for row in candidates}),
        "ignored_authentic_filename_count": ignored_authentic,
        "ignored_tampered_filename_count": ignored_tampered,
        "tampered_missing_authentic_count": missing_authentic,
        "tampered_missing_mask_count": missing_mask,
        "candidate_multiplicity": dict(
            sorted(
                Counter(
                    Counter(row["source_id"] for row in candidates).values()
                ).items()
            )
        ),
    }
    return authentic_by_source, candidates, inventory


def _audit_pair(
    archive: ZipFile,
    info_by_name: dict[str, ZipInfo],
    row: dict[str, str],
    *,
    password: str,
    max_member_bytes: int,
    mask_mode: str,
    allowed_mask_values: list[list[int]],
    soft_background_lt: int,
    soft_foreground_gt: int,
    max_ignored_mask_fraction: float,
    max_mask_fraction: float,
    min_outside_psnr_db: float,
    require_inside_mae_gt_outside: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    record_id = _sha256_bytes(
        f"{row['authentic_path']}|{row['tampered_path']}|{row['mask_path']}".encode(
            "utf-8"
        )
    )
    try:
        payloads = {
            key: _read_member(
                archive,
                info_by_name[path],
                password,
                max_member_bytes,
            )
            for key, path in (
                ("authentic", row["authentic_path"]),
                ("tampered", row["tampered_path"]),
                ("mask", row["mask_path"]),
            )
        }
        authentic, authentic_format = _decode_rgb(payloads["authentic"])
        tampered, tampered_format = _decode_rgb(payloads["tampered"])
        mask, mask_format, mask_values = _decode_mask(payloads["mask"])
        if authentic.shape != tampered.shape:
            errors.append("authentic_tampered_dimension_mismatch")
        if authentic.shape[:2] != mask.shape:
            errors.append("image_mask_dimension_mismatch")
        if mask_mode == "binary":
            allowed = {tuple(values) for values in allowed_mask_values}
            if tuple(mask_values) not in allowed:
                errors.append("unexpected_mask_values")
            positive = mask > 0
            background = ~positive
            ignored = np.zeros_like(positive, dtype=bool)
        elif mask_mode == "soft_ignore_band":
            if not (0 <= soft_background_lt <= soft_foreground_gt <= 255):
                raise ValueError("invalid soft-mask thresholds")
            background = mask < soft_background_lt
            positive = mask > soft_foreground_gt
            ignored = ~(background | positive)
            if 0 not in mask_values or 255 not in mask_values:
                errors.append("soft_mask_missing_zero_or_255_endpoint")
        else:
            raise ValueError(f"unsupported mask_mode: {mask_mode}")
        mask_fraction = float(np.mean(positive))
        ignored_fraction = float(np.mean(ignored))
        if not np.any(positive):
            errors.append("empty_mask")
        if not np.any(background):
            errors.append("full_mask")
        if mask_fraction > max_mask_fraction:
            errors.append("mask_fraction_above_gate")
        if ignored_fraction > max_ignored_mask_fraction:
            errors.append("ignored_mask_fraction_above_gate")

        outside_mae = None
        inside_mae = None
        outside_psnr_db = None
        if not any(
            item in errors
            for item in (
                "authentic_tampered_dimension_mismatch",
                "image_mask_dimension_mismatch",
                "empty_mask",
                "full_mask",
            )
        ):
            difference = np.abs(
                authentic.astype(np.float32) - tampered.astype(np.float32)
            )
            outside_mae = float(np.mean(difference[background]))
            inside_mae = float(np.mean(difference[positive]))
            outside_mse = float(
                np.mean(
                    (
                        authentic.astype(np.float32)[background]
                        - tampered.astype(np.float32)[background]
                    )
                    ** 2
                )
            )
            outside_psnr_db = (
                math.inf
                if outside_mse == 0.0
                else 20.0 * math.log10(255.0 / math.sqrt(outside_mse))
            )
            if outside_psnr_db < min_outside_psnr_db:
                errors.append("outside_mask_psnr_below_gate")
            if require_inside_mae_gt_outside and inside_mae <= outside_mae:
                errors.append("inside_mask_change_not_greater_than_outside")
        metrics = {
            "image_height": int(authentic.shape[0]),
            "image_width": int(authentic.shape[1]),
            "authentic_format": authentic_format,
            "tampered_format": tampered_format,
            "mask_format": mask_format,
            "mask_values": mask_values,
            "mask_positive_fraction": mask_fraction,
            "mask_ignored_fraction": ignored_fraction,
            "outside_mask_mae_0_255": outside_mae,
            "inside_mask_mae_0_255": inside_mae,
            "outside_mask_psnr_db": outside_psnr_db,
        }
        payload_hashes = {
            f"{key}_sha256": _sha256_bytes(value) for key, value in payloads.items()
        }
    except Exception as error:
        errors.append(f"decode_error:{type(error).__name__}:{error}")
        metrics = {}
        payload_hashes = {}
    return {
        "record_id": record_id,
        "source_group_id": f"tfr:ettd:{row['source_id']}",
        "mapping_rule": "shared_embedded_ICDAR2017_MLT_img_source_id",
        "authentic_path": row["authentic_path"],
        "tampered_path": row["tampered_path"],
        "mask_path": row["mask_path"],
        **payload_hashes,
        **metrics,
        "valid_pair": not errors,
        "errors": errors,
        "paper_evidence": False,
        "gpu_used": False,
    }


def _metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) is not None],
        dtype=np.float64,
    )
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": float(values.size), "infinite_count": float(values.size)}
    return {
        "count": float(values.size),
        "finite_count": float(finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if any(
        runtime[name]
        for name in (
            "gpu_launch_authorized",
            "method_training_authorized",
            "final_reserve_access_authorized",
        )
    ):
        raise ValueError("pair admission audit cannot authorize GPU, training, or reserve access")

    source = config["source"]
    password = os.environ.get(source["tfr_password_env"])
    if not password:
        raise ValueError("required TFR archive password is not bound")
    permission = _private_record(source["tfr_permission_record_env"])
    if not permission["present"]:
        raise ValueError("required TFR permission record is not bound")

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
    if runtime["verify_archive_sha256"]:
        observed_hash = _sha256(archive_path)
        if observed_hash != source["tfr_inner_archive_sha256"]:
            raise ValueError("TFR inner archive SHA-256 changed")
    else:
        observed_hash = source["tfr_inner_archive_sha256"]

    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        info_by_name = {info.filename: info for info in infos}
        if len(info_by_name) != len(infos):
            raise ValueError("TFR archive contains duplicate member names")
        _authentic, candidates, inventory = _candidate_inventory(
            archive,
            split=source["split"],
            authentic_directory=source["authentic_directory"],
            tampered_directory=source["tampered_directory"],
            authentic_pattern=source["authentic_filename_pattern"],
            tampered_pattern=source["tampered_filename_pattern"],
        )
        selected_groups = _ranked_groups(
            (row["source_id"] for row in candidates),
            int(runtime["seed"]),
            runtime["selected_group_limit"],
        )
        selected = [row for row in candidates if row["source_id"] in selected_groups]
        records = [
            _audit_pair(
                archive,
                info_by_name,
                row,
                password=password,
                max_member_bytes=int(runtime["max_member_bytes"]),
                mask_mode=str(config["pair_gate"].get("mask_mode", "binary")),
                allowed_mask_values=config["pair_gate"]["allowed_mask_values"],
                soft_background_lt=int(
                    config["pair_gate"].get("soft_background_lt", 1)
                ),
                soft_foreground_gt=int(
                    config["pair_gate"].get("soft_foreground_gt", 0)
                ),
                max_ignored_mask_fraction=float(
                    config["pair_gate"].get("max_ignored_mask_fraction", 0.0)
                ),
                max_mask_fraction=float(config["pair_gate"]["max_mask_fraction"]),
                min_outside_psnr_db=float(
                    config["pair_gate"]["min_outside_mask_psnr_db"]
                ),
                require_inside_mae_gt_outside=bool(
                    config["pair_gate"]["require_inside_mae_gt_outside"]
                ),
            )
            for row in selected
        ]

    records_path = _resolve(project_root, paths["records"])
    summary_path = _resolve(project_root, paths["summary"])
    _write_jsonl(records_path, records)
    failed = [row for row in records if not row["valid_pair"]]
    failed_groups = sorted({row["source_group_id"] for row in failed})
    selected_group_count = len(selected_groups)
    all_pairs_valid = not failed and bool(records)
    stage = config["experiment"]["stage"]
    if stage == "toy":
        status = "toy_pair_gate_passed" if all_pairs_valid else "toy_pair_gate_failed"
    elif stage == "pilot100":
        status = (
            "gpu_pilot_preconditions_met"
            if all_pairs_valid and selected_group_count == 100
            else "pilot100_pair_gate_failed"
        )
    else:
        status = (
            "full_candidate_pair_gate_passed"
            if all_pairs_valid
            else "full_candidate_pair_gate_failed"
        )
    summary = {
        "experiment": config["experiment"],
        "status": status,
        "paper_evidence": False,
        "gpu_used": False,
        "method_training_performed": False,
        "final_reserve_accessed": False,
        "license_and_access": {"TFR": permission},
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": observed_hash,
        },
        "mapping": {
            "scope": "TFR-internal ETTD authentic-to-CMSP only",
            "rule": "shared embedded ICDAR2017-MLT image source ID",
            "array_or_archive_position_used": False,
            "doctamper_lmdb_used": False,
            **inventory,
        },
        "selection": {
            "seed": int(runtime["seed"]),
            "selected_group_limit": runtime["selected_group_limit"],
            "selected_source_groups": selected_group_count,
            "selected_pairs": len(records),
        },
        "pair_gate": config["pair_gate"],
        "results": {
            "valid_pairs": len(records) - len(failed),
            "failed_pairs": len(failed),
            "failed_source_groups": len(failed_groups),
            "failure_reasons": dict(
                sorted(Counter(error for row in failed for error in row["errors"]).items())
            ),
            "mask_positive_fraction": _metric_summary(
                records, "mask_positive_fraction"
            ),
            "mask_ignored_fraction": _metric_summary(
                records, "mask_ignored_fraction"
            ),
            "outside_mask_mae_0_255": _metric_summary(
                records, "outside_mask_mae_0_255"
            ),
            "inside_mask_mae_0_255": _metric_summary(
                records, "inside_mask_mae_0_255"
            ),
            "outside_mask_psnr_db": _metric_summary(
                records, "outside_mask_psnr_db"
            ),
        },
        "outputs": {
            "records": str(records_path.relative_to(project_root)),
            "records_sha256": _sha256(records_path),
        },
        "gpu_gate": {
            "authorized": status == "gpu_pilot_preconditions_met",
            "reason": status,
        },
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit TFR-internal authentic/tampered pair correspondence"
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
