from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml
from PIL import Image


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _checkpoint_key(namespace: str, stage: str, task: dict[str, Any]) -> str:
    payload = json.dumps(
        {"namespace": namespace, "stage": stage, "task": task},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(payload)


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    records: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            wrapper = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                print(
                    f"checkpoint_truncated path={path} line={index + 1}; recomputing",
                    flush=True,
                )
                continue
            raise ValueError(f"invalid checkpoint JSONL at {path}:{index + 1}")
        if not isinstance(wrapper, dict):
            raise ValueError(f"non-object checkpoint row at {path}:{index + 1}")
        cache_key = wrapper.get("cache_key")
        record = wrapper.get("record")
        if not isinstance(cache_key, str) or not isinstance(record, dict):
            raise ValueError(f"invalid checkpoint row at {path}:{index + 1}")
        records[cache_key] = record
    return records


def _run_checkpointed_tasks(
    *,
    namespace: str,
    stage: str,
    tasks: list[dict[str, Any]],
    task_function: Callable[[dict[str, Any]], dict[str, Any]],
    checkpoint_path: Path,
    workers: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _read_checkpoint(checkpoint_path)
    task_keys = [_checkpoint_key(namespace, stage, task) for task in tasks]
    remaining = [
        (cache_key, task)
        for cache_key, task in zip(task_keys, tasks, strict=True)
        if cache_key not in cache
    ]
    cache_hits = len(tasks) - len(remaining)
    print(
        f"stage={stage} total={len(tasks)} cache_hits={cache_hits} "
        f"remaining={len(remaining)}",
        flush=True,
    )

    with checkpoint_path.open("a", encoding="utf-8") as handle:
        if workers <= 1:
            iterator = map(task_function, (task for _, task in remaining))
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(
                task_function,
                (task for _, task in remaining),
                chunksize=8,
            )
        try:
            for completed, ((cache_key, _), record) in enumerate(
                zip(remaining, iterator, strict=True),
                1,
            ):
                cache[cache_key] = record
                handle.write(
                    json.dumps(
                        {"cache_key": cache_key, "record": record},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                handle.write("\n")
                handle.flush()
                total_completed = cache_hits + completed
                if (
                    total_completed % progress_every == 0
                    or total_completed == len(tasks)
                ):
                    print(
                        f"stage={stage} completed={total_completed}/{len(tasks)}",
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    missing = [cache_key for cache_key in task_keys if cache_key not in cache]
    if missing:
        raise RuntimeError(f"stage={stage} incomplete_records={len(missing)}")
    return [cache[cache_key] for cache_key in task_keys]


def _file_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pixel_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    contiguous = np.ascontiguousarray(array)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _difference_counts_by_mask(
    image_array: np.ndarray,
    authentic_array: np.ndarray,
    positive: np.ndarray,
    rows_per_chunk: int = 64,
) -> tuple[int, int, int]:
    inside_count = 0
    outside_count = 0
    difference_count = 0
    for top in range(0, image_array.shape[0], rows_per_chunk):
        bottom = min(top + rows_per_chunk, image_array.shape[0])
        difference = np.any(
            image_array[top:bottom] != authentic_array[top:bottom],
            axis=2,
        )
        chunk_difference_count = int(np.count_nonzero(difference))
        chunk_inside_count = int(
            np.count_nonzero(difference & positive[top:bottom])
        )
        difference_count += chunk_difference_count
        inside_count += chunk_inside_count
        outside_count += chunk_difference_count - chunk_inside_count
    return difference_count, inside_count, outside_count


def _corner_signatures(
    array: np.ndarray, positive_mask: np.ndarray | None = None
) -> list[str | None]:
    height, width = array.shape[:2]
    patch_size = min(64, height, width)
    locations = (
        (0, 0),
        (0, width - patch_size),
        (height - patch_size, 0),
        (height - patch_size, width - patch_size),
    )
    signatures: list[str | None] = []
    for top, left in locations:
        if (
            positive_mask is not None
            and positive_mask[top : top + patch_size, left : left + patch_size].any()
        ):
            signatures.append(None)
            continue
        patch = array[top : top + patch_size, left : left + patch_size]
        signatures.append(_pixel_sha256(patch))
    return signatures


def _grid_probe_signatures(
    array: np.ndarray,
    positive_mask: np.ndarray | None = None,
    grid_size: int = 5,
    patch_size: int = 64,
) -> list[str | None]:
    height, width = array.shape[:2]
    effective_patch_size = min(patch_size, height, width)
    max_top = height - effective_patch_size
    max_left = width - effective_patch_size
    signatures: list[str | None] = []
    for row_index in range(1, grid_size + 1):
        top = round(max_top * row_index / (grid_size + 1))
        for column_index in range(1, grid_size + 1):
            left = round(max_left * column_index / (grid_size + 1))
            if (
                positive_mask is not None
                and positive_mask[
                    top : top + effective_patch_size,
                    left : left + effective_patch_size,
                ].any()
            ):
                signatures.append(None)
                continue
            patch = array[
                top : top + effective_patch_size,
                left : left + effective_patch_size,
            ]
            signatures.append(_pixel_sha256(patch))
    return signatures


def _inventory_task(task: dict[str, Any]) -> dict[str, Any]:
    path = Path(task["path"])
    scratch = Path(task["scratch"])
    try:
        payload = _file_bytes(path)
        with Image.open(io.BytesIO(payload)) as handle:
            image_format = handle.format
            image_mode = handle.mode
            array = np.asarray(handle.convert("RGB"))
        return {
            "path": _relative_to_scratch(path, scratch),
            "absolute_path": str(path),
            "split_directory": path.parts[-3],
            "filename": path.name,
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
            "format": image_format,
            "mode": image_mode,
            "file_sha256": _sha256(payload),
            "pixel_sha256": _pixel_sha256(array),
            "corner_signatures": _corner_signatures(array),
            "grid_probe_signatures": _grid_probe_signatures(
                array,
                grid_size=int(task.get("probe_grid_size", 5)),
                patch_size=int(task.get("probe_patch_size", 64)),
            ),
            "error": None,
        }
    except Exception as error:  # pragma: no cover - decoder dependent
        return {
            "path": _relative_to_scratch(path, scratch),
            "absolute_path": str(path),
            "split_directory": path.parts[-3],
            "filename": path.name,
            "error": f"{type(error).__name__}:{error}",
        }


def _locator_task(task: dict[str, Any]) -> dict[str, Any]:
    row = task["metadata"]
    image_path = Path(task["image_path"])
    mask_path = Path(task["mask_path"])
    sample_id = str(task["sample_id"])
    try:
        with Image.open(image_path) as image_handle:
            image_array = np.asarray(image_handle.convert("RGB"))
        with Image.open(mask_path) as mask_handle:
            mask_array = np.asarray(mask_handle)
        if mask_array.ndim != 2 or mask_array.shape != image_array.shape[:2]:
            raise ValueError("invalid mask dimensions")
        positive = mask_array != 0
        return {
            "sample_id": sample_id,
            "width": int(image_array.shape[1]),
            "height": int(image_array.shape[0]),
            "corner_signatures": _corner_signatures(image_array, positive),
            "grid_probe_signatures": _grid_probe_signatures(
                image_array,
                positive,
                grid_size=int(task.get("probe_grid_size", 5)),
                patch_size=int(task.get("probe_patch_size", 64)),
            ),
            "error": None,
        }
    except Exception as error:  # pragma: no cover - decoder dependent
        return {
            "sample_id": sample_id,
            "error": f"{type(error).__name__}:{error}",
            "metadata": row,
        }


def _metadata_signature(row: dict[str, Any]) -> tuple[str, ...]:
    fields = (
        "source_dataset",
        "doc_type",
        "language",
        "field_name",
        "original_value",
        "forged_value",
        "bbox_xyxy",
    )
    return tuple(
        json.dumps(row.get(field), ensure_ascii=False, sort_keys=True)
        for field in fields
    )


def _sample_id(edition: str, row: dict[str, Any]) -> str:
    return f"{edition}:{row.get('split')}:{row.get('new_id')}"


def _relative_to_scratch(path: Path | None, scratch: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(scratch))
    except ValueError:
        return str(path)


def _audit_task(task: dict[str, Any]) -> dict[str, Any]:
    edition = str(task["edition"])
    metadata = dict(task["metadata"])
    image_path = Path(task["image_path"])
    mask_path = Path(task["mask_path"])
    authentic_paths = [Path(value) for value in task.get("authentic_paths", [])]
    require_authentic = bool(task.get("require_authentic", False))
    scratch = Path(task["scratch"])
    errors: list[str] = []

    result: dict[str, Any] = {
        "edition": edition,
        "sample_id": _sample_id(edition, metadata),
        "split": metadata.get("split"),
        "new_id": metadata.get("new_id"),
        "spec_id": metadata.get("spec_id"),
        "image_id": metadata.get("image_id"),
        "source_dataset": metadata.get("source_dataset"),
        "assigned_tool": metadata.get("assigned_tool"),
        "image": _relative_to_scratch(image_path, scratch),
        "mask": _relative_to_scratch(mask_path, scratch),
        "authentic": None,
        "authentic_candidate_count": len(authentic_paths),
        "authentic_mapping_method": task.get("authentic_mapping_method"),
        "join_status": task.get("join_status"),
        "joined_v1_sample_id": task.get("joined_v1_sample_id"),
        "joined_v1_split": task.get("joined_v1_split"),
        "paper_evidence": False,
    }

    if not image_path.is_file():
        errors.append("missing_image")
    if not mask_path.is_file():
        errors.append("missing_mask")
    if any(not path.is_file() for path in authentic_paths):
        errors.append("missing_authentic_candidate")
    if errors:
        result["errors"] = errors
        result["valid"] = False
        return result

    try:
        image_payload = _file_bytes(image_path)
        mask_payload = _file_bytes(mask_path)
        with Image.open(io.BytesIO(image_payload)) as image_handle:
            image_format = image_handle.format
            image_mode = image_handle.mode
            image_array = np.asarray(image_handle.convert("RGB"))
        with Image.open(io.BytesIO(mask_payload)) as mask_handle:
            mask_format = mask_handle.format
            mask_mode = mask_handle.mode
            mask_array = np.asarray(mask_handle)
    except Exception as error:  # pragma: no cover - exact decoder errors vary
        errors.append(f"decode_error:{type(error).__name__}:{error}")
        result["errors"] = errors
        result["valid"] = False
        return result

    result.update(
        {
            "image_sha256": _sha256(image_payload),
            "mask_sha256": _sha256(mask_payload),
            "image_format": image_format,
            "image_mode": image_mode,
            "mask_format": mask_format,
            "mask_mode": mask_mode,
            "image_width": int(image_array.shape[1]),
            "image_height": int(image_array.shape[0]),
        }
    )

    if mask_array.ndim != 2:
        errors.append("mask_not_single_channel")
        positive = np.zeros(image_array.shape[:2], dtype=bool)
        mask_values: list[int] = []
    else:
        mask_values = [int(value) for value in np.unique(mask_array)]
        positive = mask_array != 0
    result["mask_values"] = mask_values
    result["mask_binary_0_255"] = bool(set(mask_values).issubset({0, 255}))
    result["mask_positive_pixels"] = int(positive.sum())
    result["mask_nonempty"] = bool(positive.any())
    if not result["mask_binary_0_255"]:
        errors.append("mask_values_not_0_255")
    if not result["mask_nonempty"]:
        errors.append("empty_forgery_mask")
    if tuple(mask_array.shape[:2]) != tuple(image_array.shape[:2]):
        errors.append("image_mask_dimension_mismatch")

    mask_bbox: list[int] | None = None
    mask_is_rectangle = False
    if positive.any():
        y_indices, x_indices = np.where(positive)
        mask_bbox = [
            int(x_indices.min()),
            int(y_indices.min()),
            int(x_indices.max()) + 1,
            int(y_indices.max()) + 1,
        ]
        rectangle_area = (mask_bbox[2] - mask_bbox[0]) * (
            mask_bbox[3] - mask_bbox[1]
        )
        mask_is_rectangle = rectangle_area == int(positive.sum())
        del y_indices, x_indices
    result["mask_bbox_xyxy_exclusive"] = mask_bbox
    result["mask_is_filled_rectangle"] = mask_is_rectangle

    metadata_bbox = metadata.get("bbox_xyxy")
    bbox_valid = (
        isinstance(metadata_bbox, list)
        and len(metadata_bbox) == 4
        and all(isinstance(value, int) for value in metadata_bbox)
        and 0 <= metadata_bbox[0] < metadata_bbox[2] <= image_array.shape[1]
        and 0 <= metadata_bbox[1] < metadata_bbox[3] <= image_array.shape[0]
    )
    result["metadata_bbox_xyxy"] = metadata_bbox
    result["metadata_bbox_in_bounds"] = bbox_valid
    inclusive_bbox_match = bool(
        bbox_valid
        and mask_bbox
        == [
            metadata_bbox[0],
            metadata_bbox[1],
            metadata_bbox[2] + 1,
            metadata_bbox[3] + 1,
        ]
    )
    exclusive_bbox_match = mask_bbox == metadata_bbox
    result["mask_bbox_matches_metadata"] = bool(
        exclusive_bbox_match or inclusive_bbox_match
    )
    result["metadata_bbox_convention"] = (
        "exclusive"
        if exclusive_bbox_match
        else "inclusive"
        if inclusive_bbox_match
        else "mismatch"
    )
    if not bbox_valid:
        errors.append("metadata_bbox_out_of_bounds_or_invalid")
    elif not result["mask_bbox_matches_metadata"]:
        errors.append("mask_bbox_metadata_mismatch")

    verified_by_pixel_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    best_outside_difference: int | None = None
    dimension_match_count = 0
    comparison_image: np.ndarray = image_array
    comparison_positive: np.ndarray = positive
    comparison_shape = tuple(image_array.shape)
    use_memmap = bool(
        authentic_paths
        and image_array.shape[0] * image_array.shape[1]
        >= int(task.get("memmap_pixel_threshold", 8_000_000))
    )
    if use_memmap:
        work_dir = Path(
            task.get(
                "memmap_work_dir",
                scratch / "data/cache/aiforge_audit_work",
            )
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="pair_audit_",
            dir=work_dir,
        )
        temporary_path = Path(temporary_directory.name)
        image_memmap_path = temporary_path / "image.npy"
        positive_memmap_path = temporary_path / "positive.npy"
        np.save(image_memmap_path, image_array, allow_pickle=False)
        np.save(positive_memmap_path, positive, allow_pickle=False)
        del image_array, mask_array, positive
        gc.collect()
        comparison_image = np.load(
            image_memmap_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        comparison_positive = np.load(
            positive_memmap_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        image_memmap_path.unlink()
        positive_memmap_path.unlink()
        temporary_directory.cleanup()

    try:
        if comparison_positive.shape == comparison_image.shape[:2]:
            for authentic_path in authentic_paths:
                authentic_array: np.ndarray | None = None
                try:
                    authentic_payload = _file_bytes(authentic_path)
                    with Image.open(io.BytesIO(authentic_payload)) as authentic_handle:
                        authentic_format = authentic_handle.format
                        authentic_mode = authentic_handle.mode
                        authentic_array = np.asarray(
                            authentic_handle.convert("RGB")
                        )
                    if authentic_array.shape != comparison_shape:
                        continue
                    dimension_match_count += 1
                    difference_count, inside_count, outside_count = (
                        _difference_counts_by_mask(
                            comparison_image,
                            authentic_array,
                            comparison_positive,
                        )
                    )
                    if best_outside_difference is None:
                        best_outside_difference = outside_count
                    else:
                        best_outside_difference = min(
                            best_outside_difference, outside_count
                        )
                    if outside_count == 0 and difference_count > 0:
                        pixel_hash = _pixel_sha256(authentic_array)
                        verified_by_pixel_hash[pixel_hash].append(
                            {
                                "path": authentic_path,
                                "payload": authentic_payload,
                                "format": authentic_format,
                                "mode": authentic_mode,
                                "pixel_hash": pixel_hash,
                                "width": int(authentic_array.shape[1]),
                                "height": int(authentic_array.shape[0]),
                                "difference_pixels": difference_count,
                                "inside_difference_pixels": inside_count,
                            }
                        )
                except Exception as error:  # pragma: no cover - decoder dependent
                    errors.append(
                        f"authentic_decode_error:{type(error).__name__}:{error}"
                    )
                finally:
                    authentic_array = None
                    if use_memmap:
                        gc.collect()
    finally:
        if use_memmap:
            del comparison_image, comparison_positive
            gc.collect()

    result["authentic_dimension_match_count"] = dimension_match_count
    result["verified_authentic_content_count"] = len(verified_by_pixel_hash)
    result["best_pair_difference_outside_mask_pixels"] = best_outside_difference
    if len(verified_by_pixel_hash) == 1:
        matching = next(iter(verified_by_pixel_hash.values()))
        selected = sorted(matching, key=lambda item: str(item["path"]))[0]
        authentic_path = selected["path"]
        result.update(
            {
                "authentic": _relative_to_scratch(authentic_path, scratch),
                "authentic_duplicate_paths": [
                    _relative_to_scratch(item["path"], scratch)
                    for item in sorted(matching, key=lambda item: str(item["path"]))
                ],
                "authentic_sha256": _sha256(selected["payload"]),
                "authentic_pixel_sha256": selected["pixel_hash"],
                "authentic_format": selected["format"],
                "authentic_mode": selected["mode"],
                "authentic_width": selected["width"],
                "authentic_height": selected["height"],
                "image_authentic_dimensions_match": True,
                "pair_difference_pixels": selected["difference_pixels"],
                "pair_difference_inside_mask_pixels": selected[
                    "inside_difference_pixels"
                ],
                "pair_difference_outside_mask_pixels": 0,
                "pair_difference_outside_mask_fraction": 0.0,
            }
        )
    elif require_authentic:
        errors.append(
            "ambiguous_verified_authentic_pair"
            if len(verified_by_pixel_hash) > 1
            else "no_verified_authentic_pair"
        )

    result["errors"] = errors
    result["valid"] = not errors
    return result


def _join_v2_to_v1(
    v1_rows: list[dict[str, Any]], v2_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_spec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_signature: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in v1_rows:
        by_spec[str(row.get("spec_id"))].append(row)
        by_signature[_metadata_signature(row)].append(row)

    joins: list[dict[str, Any]] = []
    for row in v2_rows:
        spec_candidates = by_spec.get(str(row.get("spec_id")), [])
        matching_spec_candidates = [
            candidate
            for candidate in spec_candidates
            if _metadata_signature(candidate) == _metadata_signature(row)
        ]
        signature_candidates = by_signature.get(_metadata_signature(row), [])
        match: dict[str, Any] | None = None
        if len(spec_candidates) == 1 and len(matching_spec_candidates) == 1:
            status = "unique_spec_and_signature"
            match = spec_candidates[0]
        elif len(matching_spec_candidates) == 1:
            status = "duplicate_spec_resolved_by_signature"
            match = matching_spec_candidates[0]
        elif not spec_candidates and len(signature_candidates) == 1:
            status = "missing_spec_resolved_by_signature"
            match = signature_candidates[0]
        elif not spec_candidates and not signature_candidates:
            status = "missing"
        else:
            status = "ambiguous"
        joins.append({"v2": row, "v1": match, "status": status})
    return joins


def _directory_coverage(
    root: Path,
    rows: list[dict[str, Any]],
    require_authentic: bool,
) -> dict[str, Any]:
    expected: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        expected["images"].add(str(row["image"]))
        expected["masks"].add(str(row["mask"]))
    report: dict[str, Any] = {}
    for kind, expected_paths in expected.items():
        actual_paths = {
            str(path.relative_to(root))
            for path in root.glob(f"*Set/{kind}/*.png")
            if path.is_file()
        }
        report[kind] = {
            "expected": len(expected_paths),
            "actual": len(actual_paths),
            "missing": len(expected_paths - actual_paths),
            "unexpected": len(actual_paths - expected_paths),
            "missing_examples": sorted(expected_paths - actual_paths)[:10],
            "unexpected_examples": sorted(actual_paths - expected_paths)[:10],
        }
    if require_authentic:
        authentic_paths = {
            str(path.relative_to(root))
            for path in root.glob("*Set/authentic/*.png")
            if path.is_file()
        }
        report["authentic"] = {
            "expected": len(rows),
            "actual": len(authentic_paths),
            "missing": max(0, len(rows) - len(authentic_paths)),
            "unexpected": max(0, len(authentic_paths) - len(rows)),
            "missing_examples": [],
            "unexpected_examples": [],
            "note": "v1 metadata omits image_id; path coverage is count-only",
        }
    return report


def _release_integrity(
    root: Path,
    scratch: Path,
    audit_records: list[dict[str, Any]],
    inventory_records: list[dict[str, Any]],
    expected_bytes: int | None,
) -> dict[str, Any]:
    known_hashes: dict[Path, str] = {}
    for record in audit_records:
        for path_field, hash_field in (
            ("image", "image_sha256"),
            ("mask", "mask_sha256"),
        ):
            if record.get(path_field) and record.get(hash_field):
                path = scratch / str(record[path_field])
                if path.is_relative_to(root):
                    known_hashes[path] = str(record[hash_field])
    for record in inventory_records:
        if record.get("absolute_path") and record.get("file_sha256"):
            path = Path(str(record["absolute_path"]))
            if path.is_relative_to(root):
                known_hashes[path] = str(record["file_sha256"])

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    )
    manifest_lines: list[str] = []
    for path in files:
        digest = known_hashes.get(path)
        if digest is None:
            digest = _sha256(_file_bytes(path))
        manifest_lines.append(f"{digest}  {path.relative_to(root)}\n")
    actual_bytes = sum(path.stat().st_size for path in files)
    return {
        "files": len(files),
        "bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "expected_bytes_match": (
            actual_bytes == expected_bytes if expected_bytes is not None else None
        ),
        "tree_manifest_sha256": _sha256("".join(manifest_lines).encode("utf-8")),
        "tree_manifest_format": "sha256_two_spaces_relative_path_newline",
    }


def _stable_order(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> str:
        identity = str(row.get("source_group_id") or row.get("sample_id"))
        return hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()

    return sorted(rows, key=key)


def _one_per_group(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        group = str(row["source_group_id"])
        current = selected.get(group)
        if current is None or str(row["sample_id"]) < str(current["sample_id"]):
            selected[group] = row
    return list(selected.values())


def _count_errors(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.get("errors", []))
    return dict(sorted(counts.items()))


def _quality_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "decoded_images_and_masks": sum(
            bool(record.get("image_sha256")) and bool(record.get("mask_sha256"))
            for record in records
        ),
        "strict_binary_masks": sum(
            bool(record.get("mask_binary_0_255")) for record in records
        ),
        "nonempty_masks": sum(bool(record.get("mask_nonempty")) for record in records),
        "metadata_bbox_matches_mask": sum(
            bool(record.get("mask_bbox_matches_metadata")) for record in records
        ),
        "filled_rectangle_masks": sum(
            bool(record.get("mask_is_filled_rectangle")) for record in records
        ),
        "exact_pairs_outside_mask": sum(
            record.get("pair_difference_outside_mask_pixels") == 0
            and bool(record.get("authentic_pixel_sha256"))
            for record in records
        ),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    v1_root = _resolve(scratch, paths["v1_root"])
    v2_root = _resolve(scratch, paths["v2_root"])
    summary_path = _resolve(project_root, paths["summary"])
    audit_records_path = _resolve(project_root, paths["audit_records"])
    manifest_path = _resolve(project_root, paths["manifest"])
    pilot_manifest_path = _resolve(project_root, paths["pilot_manifest"])
    checkpoint_dir = _resolve(
        project_root,
        paths.get("checkpoint_dir", "data/cache/aiforge_audit"),
    )
    authentic_inventory_path = _resolve(
        project_root,
        paths.get(
            "authentic_inventory",
            "outputs/predictions/aiforge_authentic_inventory.jsonl",
        ),
    )

    v1_metadata_path = v1_root / "metadata.jsonl"
    v2_metadata_path = v2_root / "metadata.jsonl"
    if not v1_metadata_path.is_file():
        raise FileNotFoundError(v1_metadata_path)
    if not v2_metadata_path.is_file():
        raise FileNotFoundError(v2_metadata_path)

    v1_rows = _read_jsonl(v1_metadata_path)
    v2_rows = _read_jsonl(v2_metadata_path)
    joins = _join_v2_to_v1(v1_rows, v2_rows)
    workers = int(config["audit"].get("workers", 1))
    progress_every = int(config["audit"].get("progress_every", 100))
    if progress_every <= 0:
        raise ValueError("audit.progress_every must be positive")
    checkpoint_namespace = _sha256(
        json.dumps(
            {
                "audit_name": config["audit"]["name"],
                "cache_schema_version": config["audit"].get(
                    "cache_schema_version", 1
                ),
                "releases": config.get("releases", {}),
                "v1_root": str(v1_root),
                "v2_root": str(v2_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    )

    authentic_files = sorted(v1_root.glob("*Set/authentic/*.png"))
    probe_grid_size = int(config["audit"].get("probe_grid_size", 5))
    probe_patch_size = int(config["audit"].get("probe_patch_size", 64))
    if probe_grid_size <= 0 or probe_patch_size <= 0:
        raise ValueError("audit probe dimensions must be positive")
    memmap_pixel_threshold = int(
        config["audit"].get("memmap_pixel_threshold", 8_000_000)
    )
    if memmap_pixel_threshold <= 0:
        raise ValueError("audit.memmap_pixel_threshold must be positive")
    memmap_work_dir = _resolve(
        scratch,
        paths.get("audit_work_dir", "data/cache/aiforge_audit_work"),
    )
    inventory_tasks = [
        {
            "path": str(path),
            "scratch": str(scratch),
            "probe_grid_size": probe_grid_size,
            "probe_patch_size": probe_patch_size,
        }
        for path in authentic_files
    ]
    inventory_records = _run_checkpointed_tasks(
        namespace=checkpoint_namespace,
        stage="authentic_inventory",
        tasks=inventory_tasks,
        task_function=_inventory_task,
        checkpoint_path=checkpoint_dir / "authentic_inventory.jsonl",
        workers=workers,
        progress_every=progress_every,
    )
    _write_jsonl(authentic_inventory_path, inventory_records)

    authentic_by_dimensions: dict[tuple[int, int], set[str]] = defaultdict(set)
    authentic_by_corner: dict[tuple[int, int, int, str], set[str]] = defaultdict(
        set
    )
    authentic_by_probe: dict[tuple[int, int, int, str], set[str]] = defaultdict(
        set
    )
    for record in inventory_records:
        if record.get("error"):
            continue
        dimensions = (int(record["width"]), int(record["height"]))
        absolute_path = str(record["absolute_path"])
        authentic_by_dimensions[dimensions].add(absolute_path)
        for corner_index, signature in enumerate(record["corner_signatures"]):
            if signature is not None:
                authentic_by_corner[
                    (dimensions[0], dimensions[1], corner_index, signature)
                ].add(absolute_path)
        for probe_index, signature in enumerate(
            record.get("grid_probe_signatures", [])
        ):
            if signature is not None:
                authentic_by_probe[
                    (dimensions[0], dimensions[1], probe_index, signature)
                ].add(absolute_path)

    locator_tasks = [
        {
            "metadata": row,
            "sample_id": _sample_id("v1", row),
            "image_path": str(v1_root / row["image"]),
            "mask_path": str(v1_root / row["mask"]),
            "probe_grid_size": probe_grid_size,
            "probe_patch_size": probe_patch_size,
        }
        for row in v1_rows
    ]
    locator_records = _run_checkpointed_tasks(
        namespace=checkpoint_namespace,
        stage="v1_locator",
        tasks=locator_tasks,
        task_function=_locator_task,
        checkpoint_path=checkpoint_dir / "v1_locator.jsonl",
        workers=workers,
        progress_every=progress_every,
    )

    authentic_candidates_by_v1: dict[str, list[str]] = {}
    mapping_method_by_v1: dict[str, str] = {}
    for locator in locator_records:
        sample_id = str(locator["sample_id"])
        if locator.get("error"):
            authentic_candidates_by_v1[sample_id] = []
            mapping_method_by_v1[sample_id] = "locator_failed"
            continue
        dimensions = (int(locator["width"]), int(locator["height"]))
        corner_signature_sets = [
            authentic_by_corner.get(
                (dimensions[0], dimensions[1], corner_index, signature), set()
            )
            for corner_index, signature in enumerate(locator["corner_signatures"])
            if signature is not None
        ]
        probe_signature_sets = [
            authentic_by_probe.get(
                (dimensions[0], dimensions[1], probe_index, signature), set()
            )
            for probe_index, signature in enumerate(
                locator.get("grid_probe_signatures", [])
            )
            if signature is not None
        ]
        signature_sets = corner_signature_sets + probe_signature_sets
        if signature_sets:
            candidates = set.intersection(*signature_sets)
            method = "unaltered_corner_and_grid_probe_intersection"
        else:
            candidates = authentic_by_dimensions.get(dimensions, set())
            method = "dimension_exhaustive_fallback"
        authentic_candidates_by_v1[sample_id] = sorted(candidates)
        mapping_method_by_v1[sample_id] = method

    tasks: list[dict[str, Any]] = []
    for row in v1_rows:
        v1_sample_id = _sample_id("v1", row)
        tasks.append(
            {
                "edition": "v1",
                "metadata": row,
                "image_path": str(v1_root / row["image"]),
                "mask_path": str(v1_root / row["mask"]),
                "authentic_paths": authentic_candidates_by_v1[v1_sample_id],
                "require_authentic": True,
                "authentic_mapping_method": mapping_method_by_v1[v1_sample_id],
                "scratch": str(scratch),
                "memmap_pixel_threshold": memmap_pixel_threshold,
                "memmap_work_dir": str(memmap_work_dir),
                "join_status": "content_recovery_from_public_authentic_inventory",
            }
        )
    for join in joins:
        row = join["v2"]
        v1_row = join["v1"]
        tasks.append(
            {
                "edition": "v2",
                "metadata": row,
                "image_path": str(v2_root / row["image"]),
                "mask_path": str(v2_root / row["mask"]),
                "authentic_paths": (
                    authentic_candidates_by_v1.get(_sample_id("v1", v1_row), [])
                    if v1_row is not None
                    else []
                ),
                "require_authentic": True,
                "authentic_mapping_method": (
                    "inherit_joined_v1_content_candidates"
                    if v1_row is not None
                    else "no_unique_v1_join"
                ),
                "scratch": str(scratch),
                "memmap_pixel_threshold": memmap_pixel_threshold,
                "memmap_work_dir": str(memmap_work_dir),
                "join_status": join["status"],
                "joined_v1_sample_id": (
                    _sample_id("v1", v1_row) if v1_row is not None else None
                ),
                "joined_v1_split": v1_row.get("split") if v1_row else None,
            }
        )

    audit_records = _run_checkpointed_tasks(
        namespace=checkpoint_namespace,
        stage="pair_audit",
        tasks=tasks,
        task_function=_audit_task,
        checkpoint_path=checkpoint_dir / "pair_audit.jsonl",
        workers=workers,
        progress_every=progress_every,
    )
    _write_jsonl(audit_records_path, audit_records)

    by_sample_id = {record["sample_id"]: record for record in audit_records}
    v1_records = [record for record in audit_records if record["edition"] == "v1"]
    v2_records = [record for record in audit_records if record["edition"] == "v2"]
    release_config = config.get("releases", {})
    v1_integrity = _release_integrity(
        v1_root,
        scratch,
        v1_records,
        inventory_records,
        release_config.get("v1", {}).get("repository_bytes"),
    )
    v2_integrity = _release_integrity(
        v2_root,
        scratch,
        v2_records,
        [],
        release_config.get("v2", {}).get("repository_bytes"),
    )

    source_splits: dict[str, set[str]] = defaultdict(set)
    for record in v1_records:
        source_hash = record.get("authentic_pixel_sha256")
        if source_hash:
            source_splits[str(source_hash)].add(str(record["split"]))
    leaking_groups = {
        source_hash
        for source_hash, splits in source_splits.items()
        if len(splits) > 1
    }

    validation_fraction = float(config["split"]["validation_fraction"])
    split_seed = int(config["split"]["seed"])
    manifest_rows: list[dict[str, Any]] = []
    v1_roles: dict[str, str] = {}
    for record in v1_records:
        source_hash = record.get("authentic_pixel_sha256")
        if not source_hash or source_hash in leaking_groups or not record["valid"]:
            role = "excluded"
            reason = (
                "missing_source_hash"
                if not source_hash
                else "cross_split_source_group"
                if source_hash in leaking_groups
                else "content_audit_failed"
            )
        elif record["split"] == "testing":
            role = "in_domain_test"
            reason = None
        else:
            bucket = int(
                hashlib.sha256(
                    f"{split_seed}|validation|{source_hash}".encode()
                ).hexdigest()[:8],
                16,
            ) / 0xFFFFFFFF
            role = "validation" if bucket < validation_fraction else "train"
            reason = None
        v1_roles[record["sample_id"]] = role
        manifest_rows.append(
            {
                **record,
                "source_group_id": source_hash,
                "role": role,
                "exclusion_reason": reason,
            }
        )

    manifest_by_sample = {row["sample_id"]: row for row in manifest_rows}
    join_status_counts: Counter[str] = Counter()
    split_contingency: Counter[str] = Counter()
    for join in joins:
        v2_row = join["v2"]
        v1_row = join["v1"]
        v2_sample_id = _sample_id("v2", v2_row)
        record = by_sample_id[v2_sample_id]
        join_status_counts[join["status"]] += 1
        if v1_row is None:
            role = "excluded"
            reason = "no_unique_v1_authentic_join"
            source_hash = None
        else:
            v1_sample_id = _sample_id("v1", v1_row)
            v1_manifest_row = manifest_by_sample[v1_sample_id]
            source_hash = v1_manifest_row.get("source_group_id")
            split_contingency[
                f"v1_{v1_row['split']}__v2_{v2_row['split']}"
            ] += 1
            if not record["valid"]:
                role = "excluded"
                reason = "content_audit_failed"
            elif v1_manifest_row["role"] == "in_domain_test":
                role = "generator_holdout"
                reason = None
            else:
                role = "excluded"
                reason = "v1_source_not_in_master_test"
        manifest_rows.append(
            {
                **record,
                "source_group_id": source_hash,
                "role": role,
                "exclusion_reason": reason,
            }
        )

    pilot_count = int(config["pilot"]["per_role_count"])
    pilot_seed = int(config["pilot"]["seed"])
    train_candidates = _one_per_group(
        row for row in manifest_rows if row["role"] == "train"
    )
    validation_candidates = _one_per_group(
        row for row in manifest_rows if row["role"] == "validation"
    )
    v2_holdout_by_v1 = {
        str(row["joined_v1_sample_id"]): row
        for row in manifest_rows
        if row["role"] == "generator_holdout"
    }
    paired_test_candidates = [
        row
        for row in manifest_rows
        if row["role"] == "in_domain_test"
        and row["sample_id"] in v2_holdout_by_v1
    ]
    paired_test_candidates = _one_per_group(paired_test_candidates)

    selected_train = _stable_order(train_candidates, pilot_seed)[:pilot_count]
    selected_validation = _stable_order(validation_candidates, pilot_seed)[
        :pilot_count
    ]
    selected_test = _stable_order(paired_test_candidates, pilot_seed)[:pilot_count]
    pilot_rows: list[dict[str, Any]] = []
    for role, rows in (
        ("train", selected_train),
        ("validation", selected_validation),
        ("in_domain_test", selected_test),
    ):
        for row in rows:
            pilot_rows.append({**row, "pilot_role": role})
    for row in selected_test:
        v2_row = v2_holdout_by_v1[str(row["sample_id"])]
        pilot_rows.append({**v2_row, "pilot_role": "generator_holdout"})

    pilot_role_counts = Counter(row["pilot_role"] for row in pilot_rows)
    pilot_group_roles: dict[str, set[str]] = defaultdict(set)
    for row in pilot_rows:
        pilot_group_roles[str(row["source_group_id"])].add(str(row["pilot_role"]))
    forbidden_pilot_overlap = sum(
        bool(roles & {"train", "validation"})
        and bool(roles & {"in_domain_test", "generator_holdout"})
        for roles in pilot_group_roles.values()
    )
    train_validation_overlap = sum(
        {"train", "validation"}.issubset(roles)
        for roles in pilot_group_roles.values()
    )

    v1_metadata_keys = Counter(tuple(sorted(row)) for row in v1_rows)
    v2_metadata_keys = Counter(tuple(sorted(row)) for row in v2_rows)
    v1_spec_splits: dict[str, set[str]] = defaultdict(set)
    for row in v1_rows:
        v1_spec_splits[str(row.get("spec_id"))].add(str(row.get("split")))

    inventory_filename_splits: dict[str, set[str]] = defaultdict(set)
    inventory_hash_paths: dict[str, list[str]] = defaultdict(list)
    inventory_hash_splits: dict[str, set[str]] = defaultdict(set)
    for record in inventory_records:
        if record.get("error"):
            continue
        inventory_filename_splits[str(record["filename"])].add(
            str(record["split_directory"])
        )
        inventory_hash_paths[str(record["pixel_sha256"])].append(
            str(record["path"])
        )
        inventory_hash_splits[str(record["pixel_sha256"])].add(
            str(record["split_directory"])
        )

    coverage_v1 = _directory_coverage(v1_root, v1_rows, require_authentic=True)
    coverage_v2 = _directory_coverage(v2_root, v2_rows, require_authentic=False)
    coverage_clean = all(
        details["missing"] == 0 and details["unexpected"] == 0
        for coverage in (coverage_v1, coverage_v2)
        for details in coverage.values()
    )
    complete_pilot = all(
        pilot_role_counts[role] == pilot_count
        for role in ("train", "validation", "in_domain_test", "generator_holdout")
    )
    content_clean = all(record["valid"] for record in v1_records) and all(
        record["valid"]
        for record in v2_records
        if record.get("join_status") not in {"missing", "ambiguous"}
    )
    inventory_clean = all(not record.get("error") for record in inventory_records)
    expected_rows_clean = True
    for edition, rows in (("v1", v1_rows), ("v2", v2_rows)):
        expected = config.get("releases", {}).get(edition, {}).get(
            "expected_metadata_rows"
        )
        if expected is not None:
            expected_rows_clean = expected_rows_clean and len(rows) == int(expected)
    release_storage_clean = all(
        integrity["expected_bytes_match"] is not False
        for integrity in (v1_integrity, v2_integrity)
    )
    terms_frozen = bool(config["license_policy"]["accepted_by_user"])
    gpu_preflight_ready = (
        coverage_clean
        and content_clean
        and inventory_clean
        and expected_rows_clean
        and release_storage_clean
        and complete_pilot
        and forbidden_pilot_overlap == 0
        and train_validation_overlap == 0
        and terms_frozen
    )

    summary: dict[str, Any] = {
        "audit": config["audit"],
        "paper_evidence": False,
        "status": "gpu_preflight_ready" if gpu_preflight_ready else "blocked",
        "gpu_preflight_ready": gpu_preflight_ready,
        "gpu_used": False,
        "license_policy": config["license_policy"],
        "v1": {
            "repository": release_config.get("v1", {}),
            "integrity": v1_integrity,
            "metadata_rows": len(v1_rows),
            "metadata_schema_variants": len(v1_metadata_keys),
            "metadata_missing_image_id": sum(
                "image_id" not in row for row in v1_rows
            ),
            "duplicate_spec_id_rows": len(v1_rows)
            - len({row.get("spec_id") for row in v1_rows}),
            "spec_ids_spanning_official_splits": sum(
                len(splits) > 1 for splits in v1_spec_splits.values()
            ),
            "coverage": coverage_v1,
            "valid_records": sum(record["valid"] for record in v1_records),
            "quality_counts": _quality_counts(v1_records),
            "error_counts": _count_errors(v1_records),
            "source_groups": len(source_splits),
            "source_groups_spanning_official_splits": len(leaking_groups),
            "publisher_provided_row_to_authentic_mapping": False,
            "authentic_mapping_recovery": (
                "exact_match_outside_mask_after_unaltered_corner_and_grid_probe_search"
            ),
            "authentic_pairs_recovered_by_content": sum(
                bool(record.get("authentic_pixel_sha256")) for record in v1_records
            ),
            "unrecovered_authentic_pairs": sum(
                not bool(record.get("authentic_pixel_sha256"))
                for record in v1_records
            ),
            "locator_errors": sum(
                bool(record.get("error")) for record in locator_records
            ),
            "authentic_inventory": {
                "path": str(authentic_inventory_path.relative_to(project_root)),
                "files": len(inventory_records),
                "decode_errors": sum(
                    bool(record.get("error")) for record in inventory_records
                ),
                "unique_pixel_contents": len(inventory_hash_paths),
                "duplicate_pixel_content_groups": sum(
                    len(paths) > 1 for paths in inventory_hash_paths.values()
                ),
                "filenames_present_in_both_official_split_directories": sum(
                    len(splits) > 1
                    for splits in inventory_filename_splits.values()
                ),
                "pixel_contents_present_in_both_official_split_directories": sum(
                    len(splits) > 1 for splits in inventory_hash_splits.values()
                ),
            },
        },
        "v2": {
            "repository": release_config.get("v2", {}),
            "integrity": v2_integrity,
            "metadata_rows": len(v2_rows),
            "metadata_schema_variants": len(v2_metadata_keys),
            "coverage": coverage_v2,
            "valid_records": sum(record["valid"] for record in v2_records),
            "quality_counts": _quality_counts(v2_records),
            "error_counts": _count_errors(v2_records),
            "join_status_counts": dict(sorted(join_status_counts.items())),
            "official_split_contingency_against_v1": dict(
                sorted(split_contingency.items())
            ),
            "unsafe_official_v1_train_to_v2_test_matches": split_contingency[
                "v1_training__v2_testing"
            ],
        },
        "manifest": {
            "path": str(manifest_path.relative_to(project_root)),
            "role_counts": dict(
                sorted(Counter(row["role"] for row in manifest_rows).items())
            ),
        },
        "pilot": {
            "path": (
                str(pilot_manifest_path.relative_to(project_root))
                if gpu_preflight_ready
                else None
            ),
            "requested_path": str(
                pilot_manifest_path.relative_to(project_root)
            ),
            "generated": gpu_preflight_ready,
            "blocking_reason": None if gpu_preflight_ready else "gate_blocked",
            "per_role_target": pilot_count,
            "role_counts": dict(sorted(pilot_role_counts.items())),
            "forbidden_train_or_validation_to_test_group_overlap": forbidden_pilot_overlap,
            "train_validation_group_overlap": train_validation_overlap,
        },
    }

    _write_jsonl(audit_records_path, audit_records)
    _write_jsonl(authentic_inventory_path, inventory_records)
    _write_jsonl(manifest_path, manifest_rows)
    if gpu_preflight_ready:
        _write_jsonl(pilot_manifest_path, pilot_rows)
    else:
        pilot_manifest_path.unlink(missing_ok=True)
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
