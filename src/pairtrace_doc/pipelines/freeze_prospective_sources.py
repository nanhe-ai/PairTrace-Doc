from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _invalid_thread_value = int(os.environ.get(_thread_variable, "1")) < 1
    except ValueError:
        _invalid_thread_value = True
    if _invalid_thread_value:
        os.environ[_thread_variable] = "1"

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fingerprints(image: np.ndarray, hash_size: int = 8) -> tuple[str, str]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("fingerprint input must be uint8 HWC RGB")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    phash_input = cv2.resize(
        gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    coefficients = cv2.dct(phash_input)[:hash_size, :hash_size].reshape(-1)
    threshold = float(np.median(coefficients[1:]))
    phash_bits = coefficients >= threshold
    dhash_input = cv2.resize(
        gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA
    )
    dhash_bits = (dhash_input[:, 1:] >= dhash_input[:, :-1]).reshape(-1)

    def encode(bits: np.ndarray) -> str:
        value = 0
        for bit in bits:
            value = (value << 1) | int(bool(bit))
        return f"{value:0{len(bits) // 4}x}"

    return encode(phash_bits), encode(dhash_bits)


def _layout_vector(image: np.ndarray, raster_size: int = 256, grid_size: int = 16) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (raster_size, raster_size), interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = ink.astype(np.float32) / 255.0
    block = raster_size // grid_size
    grid = binary.reshape(grid_size, block, grid_size, block).mean(axis=(1, 3))
    rows = binary.mean(axis=1).reshape(grid_size, block).mean(axis=1)
    columns = binary.mean(axis=0).reshape(grid_size, block).mean(axis=1)
    vector = np.concatenate([grid.reshape(-1), rows, columns])
    return np.clip(np.rint(vector * 255.0), 0, 255).astype(np.uint8)


def _layout_b64(vector: np.ndarray) -> str:
    return base64.b64encode(vector.tobytes()).decode("ascii")


def _layout_from_b64(value: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value.encode("ascii")), dtype=np.uint8)


@lru_cache(maxsize=8192)
def _cached_layout_from_b64(value: str) -> np.ndarray:
    return _layout_from_b64(value)


def _layout_cosine(left: np.ndarray, right: np.ndarray) -> float:
    if np.array_equal(left, right):
        return 1.0
    left_float = left.astype(np.float64)
    right_float = right.astype(np.float64)
    denominator = float(np.linalg.norm(left_float) * np.linalg.norm(right_float))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_float, right_float) / denominator)


def _relative_aspect_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_ratio = float(left["width"]) / float(left["height"])
    right_ratio = float(right["width"]) / float(right["height"])
    return abs(left_ratio - right_ratio) / max(left_ratio, right_ratio)


def _hamming(left_hex: str, right_hex: str) -> int:
    return (int(left_hex, 16) ^ int(right_hex, 16)).bit_count()


def _quality_metrics(image: np.ndarray, long_side: int) -> dict[str, Any]:
    height, width = image.shape[:2]
    scale = min(1.0, float(long_side) / max(height, width))
    if scale < 1.0:
        sample = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        sample = image
    gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probability = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probability * np.log2(probability)).sum())
    foreground = gray < 245
    border_width = max(1, round(min(gray.shape) * 0.02))
    border = np.zeros_like(foreground)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    return {
        "grayscale_mean": round(float(gray.mean()), 6),
        "grayscale_std": round(float(gray.std()), 6),
        "entropy_bits": round(entropy, 6),
        "foreground_fraction_lt245": round(float(foreground.mean()), 8),
        "white_fraction_ge250": round(float((gray >= 250).mean()), 8),
        "border_foreground_fraction_lt245": round(float(foreground[border].mean()), 8),
        "laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
    }


def _decode_feature(task: dict[str, Any], quality_long_side: int) -> dict[str, Any]:
    path = Path(str(task["absolute_path"]))
    result = dict(task)
    result.pop("absolute_path", None)
    result["bytes"] = path.stat().st_size if path.is_file() else None
    try:
        encoded = _sha256(path)
        declared = task.get("declared_encoded_sha256")
        if declared and encoded != declared:
            raise ValueError(f"declared encoded SHA-256 mismatch: {encoded} != {declared}")
        with Image.open(path) as handle:
            source_format = handle.format
            exif_orientation = int(handle.getexif().get(274, 1))
            canonical = ImageOps.exif_transpose(handle).convert("RGB")
            image = np.asarray(canonical)
        phash, dhash = _fingerprints(image)
        layout = _layout_vector(image)
        result.update(
            {
                "status": "ok",
                "error": None,
                "format": source_format,
                "exif_orientation": exif_orientation,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "encoded_sha256": encoded,
                "decoded_pixel_sha256": _pixel_sha256(image),
                "phash64": phash,
                "dhash64": dhash,
                "layout_vector_b64": _layout_b64(layout),
                "layout_vector_sha256": hashlib.sha256(layout.tobytes()).hexdigest(),
                "quality": _quality_metrics(image, quality_long_side),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "format": None,
                "exif_orientation": None,
                "width": None,
                "height": None,
                "encoded_sha256": task.get("declared_encoded_sha256"),
                "decoded_pixel_sha256": None,
                "phash64": None,
                "dhash64": None,
                "layout_vector_b64": None,
                "layout_vector_sha256": None,
                "quality": None,
            }
        )
    return result


def _hard_gate(record: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    if record["status"] != "ok":
        return ["decode_or_identity_failure"]
    quality = record["quality"]
    reasons: list[str] = []
    if min(int(record["width"]), int(record["height"])) < int(thresholds["minimum_short_side"]):
        reasons.append("short_side_below_minimum")
    if float(quality["grayscale_std"]) < float(thresholds["minimum_grayscale_std"]):
        reasons.append("contrast_below_minimum")
    foreground = float(quality["foreground_fraction_lt245"])
    if foreground < float(thresholds["minimum_foreground_fraction"]):
        reasons.append("foreground_fraction_below_minimum")
    if foreground > float(thresholds["maximum_foreground_fraction"]):
        reasons.append("foreground_fraction_above_maximum")
    if float(quality["white_fraction_ge250"]) > float(thresholds["maximum_white_fraction"]):
        reasons.append("white_fraction_above_maximum")
    if float(quality["laplacian_variance"]) < float(thresholds["minimum_laplacian_variance"]):
        reasons.append("laplacian_variance_below_minimum")
    return reasons


def _polygon_area(points: Any) -> float:
    try:
        array = np.asarray(points, dtype=np.float32)
        if array.shape != (4, 2):
            return 0.0
        return float(abs(cv2.contourArea(array)))
    except Exception:
        return 0.0


def _naf_annotation(path: Path, group: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    fields = value.get("fieldBBs", [])
    counts = Counter(int(field.get("isBlank", -1)) for field in fields)
    areas = Counter()
    for field in fields:
        areas[int(field.get("isBlank", -1))] += _polygon_area(field.get("poly_points"))
    width = int(value.get("width", 0))
    height = int(value.get("height", 0))
    image_area = max(1, width * height)
    page_area = _polygon_area(value.get("actualPage_corners"))
    return {
        "annotation_relative_path": path.as_posix(),
        "annotation_image_filename": str(value.get("imageFilename", path.with_suffix(".jpg").name)),
        "annotation_width": width,
        "annotation_height": height,
        "official_group": group,
        "official_base_family": group.split("_", 1)[0],
        "field_count": len(fields),
        "preprinted_text_count": len(value.get("textBBs", [])),
        "handwriting_count": counts[1],
        "printed_or_stamp_count": counts[2],
        "blank_count": counts[3],
        "signature_count": counts[4],
        "handwriting_area_fraction": round(float(areas[1]) / image_area, 8),
        "printed_or_stamp_area_fraction": round(float(areas[2]) / image_area, 8),
        "signature_area_fraction": round(float(areas[4]) / image_area, 8),
        "actual_page_polygon_area_fraction": round(page_area / image_area, 8),
    }


def _load_candidates(
    project_root: Path, scratch: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    immutable = config["immutable_inputs"]
    doc_path = _resolve(project_root, immutable["doclaynet_manifest"]["path"])
    midv_path = _resolve(project_root, immutable["midv500_manifest"]["path"])
    if _sha256(doc_path) != immutable["doclaynet_manifest"]["sha256"]:
        raise ValueError("DocLayNet candidate manifest changed")
    if _sha256(midv_path) != immutable["midv500_manifest"]["sha256"]:
        raise ValueError("MIDV-500 source manifest changed")
    rows: list[dict[str, Any]] = []
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    for item in doc["candidates"]:
        absolute = Path(str(item["path"])).resolve()
        relative = absolute.relative_to(scratch).as_posix()
        rows.append(
            {
                "record_id": f"doclaynet::{item['doc_category']}::{item['sha256'][:16]}",
                "source_dataset": "DocLayNet",
                "source_stratum": str(item["doc_category"]),
                "source_group_key": str(item["doc_name"]),
                "absolute_path": str(absolute),
                "path": relative,
                "declared_encoded_sha256": str(item["sha256"]),
                "selection_metadata": {
                    "doc_name": str(item["doc_name"]),
                    "page_no": int(item["page_no"]),
                    "precedence": int(item["precedence"]),
                    "collection": str(item["collection"]),
                    "member": str(item["member"]),
                },
            }
        )

    metadata_root = _resolve(scratch, config["storage"]["naf_metadata_root"])
    image_root = _resolve(scratch, config["storage"]["naf_image_root"])
    seen_images: set[str] = set()
    for group_dir in sorted((metadata_root / "groups").iterdir(), key=lambda p: p.name):
        if not group_dir.is_dir():
            continue
        for annotation_path in sorted(group_dir.glob("*.json")):
            if annotation_path.name.startswith("template"):
                continue
            image_name = annotation_path.with_suffix(".jpg").name
            if image_name in seen_images:
                raise ValueError(f"NAF image has multiple annotation directories: {image_name}")
            seen_images.add(image_name)
            image_path = image_root / image_name
            annotation = _naf_annotation(annotation_path, group_dir.name)
            annotation["annotation_relative_path"] = annotation_path.relative_to(scratch).as_posix()
            rows.append(
                {
                    "record_id": f"naf::{group_dir.name}::{image_path.stem}",
                    "source_dataset": "NAF",
                    "source_stratum": "historical_scanned_form",
                    "source_group_key": annotation["official_base_family"],
                    "absolute_path": str(image_path),
                    "path": image_path.relative_to(scratch).as_posix(),
                    "declared_encoded_sha256": None,
                    "selection_metadata": annotation,
                }
            )
    if len(seen_images) != 865:
        raise ValueError(f"expected 865 NAF annotated images, found {len(seen_images)}")
    archive_images = {p.name for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    if archive_images != seen_images:
        raise ValueError("NAF image archive and annotation directories disagree")

    midv = json.loads(midv_path.read_text(encoding="utf-8"))
    for item in midv["sources"]:
        image_members = [
            member for member in item["selected_members"] if Path(str(member["path"])).suffix.lower() in {".tif", ".tiff"}
        ]
        if len(image_members) != 1 or int(item.get("video_frames_selected", -1)) != 0:
            raise ValueError(f"MIDV-500 source boundary changed: {item['document_type']}")
        image_member = image_members[0]
        absolute = Path(str(image_member["path"])).resolve()
        rows.append(
            {
                "record_id": f"midv500::{item['document_type']}",
                "source_dataset": "MIDV-500",
                "source_stratum": "identity_document_layout",
                "source_group_key": str(item["document_type"]),
                "absolute_path": str(absolute),
                "path": absolute.relative_to(scratch).as_posix(),
                "declared_encoded_sha256": str(image_member["sha256"]),
                "selection_metadata": {
                    "document_type": str(item["document_type"]),
                    "document_type_index": int(item["document_type_index"]),
                    "remote_archive": str(item["remote_archive"]),
                    "remote_archive_bytes": int(item["remote_archive_bytes"]),
                    "member": str(image_member["member"]),
                    "video_frames_selected": 0,
                },
            }
        )
    counts = Counter(row["source_dataset"] for row in rows)
    if counts != Counter({"DocLayNet": 200, "NAF": 865, "MIDV-500": 50}):
        raise ValueError(f"prospective candidate counts changed: {counts}")
    return rows


def _load_prior_tasks(
    project_root: Path, scratch: Path, specifications: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tasks: dict[tuple[str, str], dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for specification in specifications:
        name = str(specification["name"])
        if "manifest" in specification:
            manifest = _resolve(project_root, str(specification["manifest"]))
            digest = _sha256(manifest)
            if digest != str(specification["manifest_sha256"]):
                raise ValueError(f"prior manifest changed: {manifest}")
            input_hashes[manifest.relative_to(project_root).as_posix()] = digest
            rows = _read_jsonl(manifest)
            for row in rows:
                if specification.get("valid_only") and row.get("valid") is not True:
                    continue
                value = row.get(str(specification["image_key"]))
                if not value:
                    continue
                path = _resolve(scratch, str(value))
                declared = row.get(str(specification["declared_sha256_key"]))
                identity = str(row.get(str(specification["identity_key"]), path.name))
                key = (name, path.as_posix())
                candidate = {
                    "record_id": f"prior::{name}::{hashlib.sha256(path.as_posix().encode()).hexdigest()[:16]}",
                    "source_dataset": name,
                    "source_stratum": "prior_project_source",
                    "source_group_key": identity,
                    "absolute_path": str(path),
                    "path": path.relative_to(scratch).as_posix(),
                    "declared_encoded_sha256": str(declared) if declared else None,
                    "selection_metadata": {"prior_identity": identity},
                }
                if key in tasks:
                    existing = tasks[key]
                    if existing["declared_encoded_sha256"] != candidate["declared_encoded_sha256"]:
                        raise ValueError(f"prior path declared with conflicting hashes: {path}")
                    identities = set(existing["selection_metadata"].get("prior_identities", []))
                    identities.update([existing["source_group_key"], identity])
                    existing["selection_metadata"]["prior_identities"] = sorted(identities)
                else:
                    tasks[key] = candidate
        else:
            root = _resolve(scratch, str(specification["image_glob_root"]))
            paths = sorted(path for path in root.glob(str(specification["glob"])) if path.is_file())
            inventory_rows = [
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in paths
            ]
            payload = json.dumps(inventory_rows, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(payload).hexdigest()
            if len(paths) != int(specification["expected_records"]) or digest != str(
                specification["inventory_sha256"]
            ):
                raise ValueError(f"prior glob inventory changed: {name}")
            input_hashes[f"glob:{name}"] = digest
            for item, path in zip(inventory_rows, paths, strict=True):
                tasks[(name, path.as_posix())] = {
                    "record_id": f"prior::{name}::{item['sha256'][:16]}",
                    "source_dataset": name,
                    "source_stratum": "prior_project_source",
                    "source_group_key": path.stem,
                    "absolute_path": str(path),
                    "path": path.relative_to(scratch).as_posix(),
                    "declared_encoded_sha256": item["sha256"],
                    "selection_metadata": {"prior_identity": path.stem},
                }
    result = sorted(tasks.values(), key=lambda row: row["record_id"])
    if len({row["record_id"] for row in result}) != len(result):
        raise ValueError("prior inventory record IDs are not unique")
    return result, input_hashes


def _feature_all(tasks: list[dict[str, Any]], long_side: int, workers: int = 4) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(
            executor.map(lambda task: _decode_feature(task, long_side), tasks), 1
        ):
            results.append(result)
            if index % 250 == 0 or index == len(tasks):
                print(json.dumps({"features_completed": index, "features_total": len(tasks)}), flush=True)
    return results


def _seeded_digest(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}|{record_id}".encode("utf-8")).hexdigest()


def _naf_preference(record: dict[str, Any], seed: int) -> tuple[Any, ...]:
    metadata = record["selection_metadata"]
    return (
        int(int(metadata["signature_count"]) > 0),
        float(metadata["handwriting_area_fraction"]),
        int(metadata["handwriting_count"]),
        -int(metadata["printed_or_stamp_count"]),
        -int(metadata["preprinted_text_count"]),
        _seeded_digest(seed, record["record_id"]),
    )


def _public_feature(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "layout_vector_b64"}


def _select(
    candidates: list[dict[str, Any]], config: dict[str, Any], prior: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(config["experiment"]["seed"])
    thresholds = config["quality"]
    prior_encoded = {row["encoded_sha256"] for row in prior}
    prior_pixels = {row["decoded_pixel_sha256"] for row in prior}
    for record in candidates:
        if record["source_dataset"] in ("DocLayNet", "NAF"):
            reasons = _hard_gate(record, thresholds[record["source_dataset"].lower()])
        else:
            reasons = [] if record["status"] == "ok" else ["fixed_midv_decode_or_identity_failure"]
        if record.get("encoded_sha256") in prior_encoded:
            reasons.append("exact_encoded_duplicate_of_prior_source")
        if record.get("decoded_pixel_sha256") in prior_pixels:
            reasons.append("exact_pixel_duplicate_of_prior_source")
        record["hard_gate_reasons"] = sorted(set(reasons))
        record["hard_gate_eligible"] = not record["hard_gate_reasons"]
        record["selected"] = False
        record["selection_reason"] = None

    selected: list[dict[str, Any]] = []
    targets = config["targets"]["doclaynet"]
    for stratum in ("financial_reports", "government_tenders"):
        eligible = [
            row
            for row in candidates
            if row["source_dataset"] == "DocLayNet"
            and row["source_stratum"] == stratum
            and row["hard_gate_eligible"]
            and int(row["selection_metadata"]["precedence"]) == int(targets["required_precedence"])
        ]
        eligible.sort(key=lambda row: _seeded_digest(seed, row["record_id"]))
        chosen = eligible[: int(targets[stratum])]
        if len(chosen) != int(targets[stratum]):
            raise ValueError(f"insufficient eligible DocLayNet candidates for {stratum}")
        chosen_ids = {row["record_id"] for row in chosen}
        for row in candidates:
            if row["source_dataset"] == "DocLayNet" and row["source_stratum"] == stratum:
                if row["record_id"] in chosen_ids:
                    row["selected"] = True
                    row["selection_reason"] = "selected_by_frozen_seeded_permutation"
                elif row["hard_gate_eligible"]:
                    row["selection_reason"] = "eligible_not_selected_by_frozen_seeded_permutation"
                else:
                    row["selection_reason"] = "hard_gate_ineligible"
        selected.extend(chosen)

    naf_eligible = [
        row for row in candidates if row["source_dataset"] == "NAF" and row["hard_gate_eligible"]
    ]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in naf_eligible:
        by_family[str(row["selection_metadata"]["official_base_family"])].append(row)
    family_representatives = [
        min(rows, key=lambda row: _naf_preference(row, seed)) for rows in by_family.values()
    ]
    family_representatives.sort(key=lambda row: _naf_preference(row, seed))
    naf_chosen = family_representatives[: int(config["targets"]["naf"]["count"])]
    if len(naf_chosen) != int(config["targets"]["naf"]["count"]):
        raise ValueError("insufficient eligible NAF base families")
    naf_chosen_ids = {row["record_id"] for row in naf_chosen}
    representative_ids = {row["record_id"] for row in family_representatives}
    for row in candidates:
        if row["source_dataset"] != "NAF":
            continue
        if row["record_id"] in naf_chosen_ids:
            row["selected"] = True
            row["selection_reason"] = "selected_machine_print_preference_distinct_base_family"
        elif not row["hard_gate_eligible"]:
            row["selection_reason"] = "hard_gate_ineligible"
        elif row["record_id"] in representative_ids:
            row["selection_reason"] = "eligible_family_representative_not_in_top_50"
        else:
            row["selection_reason"] = "eligible_not_preferred_within_base_family"
    selected.extend(naf_chosen)

    midv = [row for row in candidates if row["source_dataset"] == "MIDV-500"]
    if len(midv) != int(config["targets"]["midv500"]["count"]) or any(
        not row["hard_gate_eligible"] for row in midv
    ):
        raise ValueError("MIDV-500 fixed source set failed identity/decode gate")
    for row in midv:
        row["selected"] = True
        row["selection_reason"] = "fixed_all_50_top_level_source_tiffs"
    selected.extend(midv)

    selected.sort(
        key=lambda row: (
            {"DocLayNet": 0, "NAF": 1, "MIDV-500": 2}[row["source_dataset"]],
            row["source_stratum"],
            row["source_group_key"],
            row["record_id"],
        )
    )
    counts = Counter(row["source_dataset"] for row in selected)
    if counts != Counter({"DocLayNet": 50, "NAF": 50, "MIDV-500": 50}):
        raise ValueError(f"selected source counts changed: {counts}")
    if len({row["source_group_key"] for row in selected if row["source_dataset"] == "NAF"}) != 50:
        raise ValueError("NAF base families are not distinct")
    return selected, candidates


def _pair_metrics(left: dict[str, Any], right: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    aspect = _relative_aspect_difference(left, right)
    phash = _hamming(str(left["phash64"]), str(right["phash64"]))
    dhash = _hamming(str(left["dhash64"]), str(right["dhash64"]))
    layout = _layout_cosine(
        _cached_layout_from_b64(str(left["layout_vector_b64"])),
        _cached_layout_from_b64(str(right["layout_vector_b64"])),
    )
    exact_encoded = left["encoded_sha256"] == right["encoded_sha256"]
    exact_pixels = left["decoded_pixel_sha256"] == right["decoded_pixel_sha256"]
    high = config["deduplication"]["high_priority_visual"]
    high_priority = (
        phash <= int(high["phash_max_distance"])
        and dhash <= int(high["dhash_max_distance"])
        and aspect <= float(high["aspect_relative_difference_max"])
    ) or (
        layout >= float(high["layout_cosine_min"])
        and phash <= int(high["layout_phash_max_distance"])
        and aspect <= float(high["layout_aspect_relative_difference_max"])
    )
    broad = config["deduplication"]["ocr_candidate_screen"]
    ocr_candidate = high_priority or (
        phash <= int(broad["phash_max_distance"])
        and dhash <= int(broad["dhash_max_distance"])
        and aspect <= float(broad["aspect_relative_difference_max"])
    ) or (
        layout >= float(broad["layout_cosine_min"])
        and aspect <= float(broad["layout_aspect_relative_difference_max"])
    )
    return {
        "exact_encoded_sha256": exact_encoded,
        "exact_decoded_pixel_sha256": exact_pixels,
        "phash_distance": phash,
        "dhash_distance": dhash,
        "aspect_ratio_relative_difference": round(aspect, 8),
        "layout_cosine_similarity": round(layout, 8),
        "high_priority_visual_flag": high_priority,
        "ocr_candidate": ocr_candidate,
    }


def _screen_pairs(
    selected: list[dict[str, Any]], prior: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left in selected:
        nearest: tuple[tuple[Any, ...], dict[str, Any]] | None = None
        for right in prior:
            metrics = _pair_metrics(left, right, config)
            key = (
                metrics["phash_distance"],
                metrics["dhash_distance"],
                -metrics["layout_cosine_similarity"],
                metrics["aspect_ratio_relative_difference"],
                right["record_id"],
            )
            candidate = {
                "pair_id": hashlib.sha256(
                    f"prior|{left['record_id']}|{right['record_id']}".encode()
                ).hexdigest(),
                "pair_scope": "prospective_vs_prior",
                "prospective_record_id": left["record_id"],
                "other_record_id": right["record_id"],
                "other_source_dataset": right["source_dataset"],
                "other_path": right["path"],
                **metrics,
            }
            if nearest is None or key < nearest[0]:
                nearest = (key, candidate)
            if metrics["ocr_candidate"] or metrics["exact_encoded_sha256"] or metrics["exact_decoded_pixel_sha256"]:
                rows.append(candidate)
        if nearest is not None and not any(
            row["prospective_record_id"] == left["record_id"]
            and row["other_record_id"] == nearest[1]["other_record_id"]
            for row in rows
        ):
            nearest[1]["nearest_diagnostic_only"] = True
            rows.append(nearest[1])
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            metrics = _pair_metrics(left, right, config)
            if not (
                metrics["ocr_candidate"]
                or metrics["exact_encoded_sha256"]
                or metrics["exact_decoded_pixel_sha256"]
            ):
                continue
            rows.append(
                {
                    "pair_id": hashlib.sha256(
                        f"within|{left['record_id']}|{right['record_id']}".encode()
                    ).hexdigest(),
                    "pair_scope": "within_prospective_150",
                    "prospective_record_id": left["record_id"],
                    "other_record_id": right["record_id"],
                    "other_source_dataset": right["source_dataset"],
                    "other_path": right["path"],
                    **metrics,
                }
            )
    rows.sort(key=lambda row: (row["pair_scope"], row["prospective_record_id"], row["other_record_id"]))
    return rows


def _render_contact_sheets(
    selected: list[dict[str, Any]], scratch: Path, output_dir: Path
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    per_sheet = 25
    columns = 5
    cell_width, cell_height = 320, 300
    for start in range(0, len(selected), per_sheet):
        subset = selected[start : start + per_sheet]
        rows = math.ceil(len(subset) / columns)
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(subset):
            row, column = divmod(offset, columns)
            x, y = column * cell_width, row * cell_height
            path = _resolve(scratch, str(record["path"]))
            with Image.open(path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
                image.thumbnail((cell_width - 16, cell_height - 54), Image.Resampling.LANCZOS)
            image_x = x + (cell_width - image.width) // 2
            sheet.paste(image, (image_x, y + 4))
            label = f"{start + offset + 1:03d} {record['source_dataset']}\n{record['record_id'][-34:]}"
            draw.rectangle((x, y + cell_height - 48, x + cell_width, y + cell_height), fill="white")
            draw.multiline_text((x + 5, y + cell_height - 45), label, fill="black", spacing=2)
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="gray")
        output = output_dir / f"selected_{start + 1:03d}_{start + len(subset):03d}.png"
        sheet.save(output, format="PNG", optimize=False)
        manifests.append(
            {
                "path": output.as_posix(),
                "first_index": start + 1,
                "last_index": start + len(subset),
                "sha256": _sha256(output),
            }
        )
    return manifests


def _draft_row(record: dict[str, Any], index: int, config_sha256: str, protocol_sha256: str) -> dict[str, Any]:
    metadata = record["selection_metadata"]
    if record["source_dataset"] == "DocLayNet":
        source_group_id = f"prospective:doclaynet:{record['encoded_sha256'][:20]}"
        license_name = "CDLA-Permissive-1.0"
        handling = "public_source_preserve_attribution_and_change_notice"
    elif record["source_dataset"] == "NAF":
        source_group_id = f"prospective:naf-family:{metadata['official_base_family']}"
        license_name = "CDLA-Permissive-1.0"
        handling = "local_only_possible_historical_personal_data"
    else:
        source_group_id = f"prospective:midv500:{metadata['document_type']}"
        license_name = "CC-BY-SA-2.5"
        handling = "preserve_per_item_origin_attribution_and_sharealike"
    return {
        "draft_index": index,
        "record_id": record["record_id"],
        "source_group_id": source_group_id,
        "source_dataset": record["source_dataset"],
        "source_stratum": record["source_stratum"],
        "template_family_id": str(record["source_group_key"]),
        "path": record["path"],
        "bytes": record["bytes"],
        "format": record["format"],
        "width": record["width"],
        "height": record["height"],
        "exif_orientation": record["exif_orientation"],
        "encoded_sha256": record["encoded_sha256"],
        "decoded_pixel_sha256": record["decoded_pixel_sha256"],
        "phash64": record["phash64"],
        "dhash64": record["dhash64"],
        "layout_representation": "layout-grid-v1",
        "layout_vector_sha256": record["layout_vector_sha256"],
        "quality": record["quality"],
        "selection_metadata": metadata,
        "selection_reason": record["selection_reason"],
        "license": license_name,
        "handling": handling,
        "selection_config_sha256": config_sha256,
        "selection_protocol_sha256": protocol_sha256,
        "selection_uses_detector_or_model_output": False,
        "ai_editor_exposure_before_freeze": False,
    }


def _ocr_input_rows(
    draft: list[dict[str, Any]], pair_rows: list[dict[str, Any]], prior: list[dict[str, Any]], scratch: Path
) -> list[dict[str, Any]]:
    draft_by_id = {str(row["record_id"]): row for row in draft}
    prior_by_id = {str(row["record_id"]): row for row in prior}
    records: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, Any]) -> None:
        records[str(row["record_id"])] = {
            "record_id": row["record_id"],
            "image": str(_resolve(scratch, row["path"])),
            "image_sha256": row["encoded_sha256"],
        }

    for pair in pair_rows:
        if not pair["ocr_candidate"]:
            continue
        add(draft_by_id[str(pair["prospective_record_id"])])
        other_id = str(pair["other_record_id"])
        add(draft_by_id[other_id] if other_id in draft_by_id else prior_by_id[other_id])
    return [records[key] for key in sorted(records)]


def prepare_ocr(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _verify_config(project_root, config_path, config)
    outputs = config["outputs"]
    scratch = Path(
        os.environ.get(
            config["storage"]["scratch_env"],
            str(_resolve(project_root, config["storage"]["scratch_default"])),
        )
    ).resolve()
    draft = _read_jsonl(_resolve(project_root, outputs["draft_selection_jsonl"]))
    pairs = _read_jsonl(_resolve(project_root, outputs["pair_screen_jsonl"]))
    prior = _read_jsonl(_resolve(project_root, outputs["old_inventory_jsonl"]))
    rows = _ocr_input_rows(draft, pairs, prior, scratch)
    output = _resolve(project_root, outputs["ocr_input_jsonl"])
    _write_jsonl(output, rows)
    summary_path = _resolve(project_root, outputs["draft_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ocr_input_records"] = len(rows)
    summary["ocr_input_sha256"] = _sha256(output)
    summary["ocr_scope"] = "only_records_in_frozen_broad_pair_screen"
    summary["next"] = "run frozen local CPU OCR for screened pairs, bind visual review, then finalize"
    _write_json(summary_path, summary)
    result = {
        "status": "screened_pair_ocr_input_ready",
        "records": len(rows),
        "record_ids": [row["record_id"] for row in rows],
        "sha256": _sha256(output),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def record_agent_review(config_path: Path, reviewed_at_utc: str) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _verify_config(project_root, config_path, config)
    outputs = config["outputs"]
    draft_path = _resolve(project_root, outputs["draft_selection_jsonl"])
    draft = _read_jsonl(draft_path)
    draft_sha256 = _sha256(draft_path)
    summary = json.loads(
        _resolve(project_root, outputs["draft_summary"]).read_text(encoding="utf-8")
    )
    sheets = summary.get("contact_sheets", [])
    if len(sheets) != 6:
        raise ValueError("expected six contact sheets for the agent review")
    for sheet in sheets:
        path = Path(str(sheet["path"])).resolve()
        if _sha256(path) != str(sheet["sha256"]):
            raise ValueError(f"contact sheet changed after review: {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(draft, 1):
        sheet = next(
            value
            for value in sheets
            if int(value["first_index"]) <= index <= int(value["last_index"])
        )
        issues: list[str] = []
        metadata = row["selection_metadata"]
        if row["source_dataset"] == "NAF":
            issues.append("possible_historical_personal_data_local_only")
            if int(metadata.get("handwriting_count", 0)) > 0:
                issues.append("historical_handwriting_present")
            if int(metadata.get("signature_count", 0)) > 0:
                issues.append("historical_signature_present")
        elif row["source_dataset"] == "MIDV-500":
            issues.append("released_identity_document_sample_preserve_attribution")
        if row["record_id"] == "doclaynet::financial_reports::dda7dbf98b2d87d6":
            issues.append("sparse_but_legible_company_contact_page")
        rows.append(
            {
                "record_id": row["record_id"],
                "draft_index": index,
                "draft_selection_sha256": draft_sha256,
                "review_mode": "agent_contact_sheet_visual_review",
                "reviewer_id": "codex-root-20260723",
                "reviewed_at_utc": reviewed_at_utc,
                "decision": "approve",
                "issues": issues,
                "review_checks": {
                    "decode_and_orientation": "pass",
                    "gross_legibility": "pass",
                    "severe_crop_or_missing_primary_document": "not_observed",
                    "blank_or_content_free": "not_observed",
                    "archival_carrier_or_published_border_allowed": True,
                    "detector_or_model_output_viewed": False,
                    "human_review_claimed": False,
                },
                "contact_sheet_sha256": sheet["sha256"],
                "notes": "Contact-sheet review supported by deterministic full-resolution quality metrics; six visually ambiguous items were also opened at original resolution.",
            }
        )
    output = _resolve(project_root, outputs["review_jsonl"])
    _write_jsonl(output, rows)
    result = {
        "status": "agent_visual_review_bound_to_draft",
        "records": len(rows),
        "approved": sum(row["decision"] == "approve" for row in rows),
        "human_review_claimed": False,
        "draft_selection_sha256": draft_sha256,
        "review_sha256": _sha256(output),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def _verify_config(project_root: Path, config_path: Path, config: dict[str, Any]) -> tuple[str, str]:
    config_sha256 = _sha256(config_path)
    protocol = _resolve(project_root, config["experiment"]["protocol"])
    protocol_sha256 = _sha256(protocol)
    if protocol_sha256 != str(config["experiment"]["expected_protocol_sha256"]):
        raise ValueError("source-freeze protocol SHA-256 changed")
    acquisition = config["immutable_inputs"]["acquisition_config"]
    if _sha256(_resolve(project_root, acquisition["path"])) != str(acquisition["sha256"]):
        raise ValueError("frozen acquisition config changed")
    auth = config["authorization"]
    forbidden = ("invoke_editors", "generate_edits", "run_detector_inference", "train_models", "tune_on_final_sources")
    if any(bool(auth[key]) for key in forbidden) or bool(auth["gpu_required"]):
        raise ValueError("source-freeze authorization boundary was widened")
    return config_sha256, protocol_sha256


def audit(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha256, protocol_sha256 = _verify_config(project_root, config_path, config)
    scratch = Path(
        os.environ.get(config["storage"]["scratch_env"], str(_resolve(project_root, config["storage"]["scratch_default"])))
    ).resolve()
    for archive_key in ("naf_image_archive", "naf_metadata_archive"):
        specification = config["immutable_inputs"][archive_key]
        path = _resolve(scratch, specification["relative_path"])
        if path.stat().st_size != int(specification["bytes"]) or _sha256(path) != str(specification["sha256"]):
            raise ValueError(f"immutable NAF archive changed: {path}")
    candidates = _load_candidates(project_root, scratch, config)
    prior_tasks, prior_input_hashes = _load_prior_tasks(project_root, scratch, config["prior_sources"])
    print(json.dumps({"candidate_features_start": len(candidates)}), flush=True)
    candidate_features = _feature_all(candidates, int(config["quality"]["metric_resize_long_side"]))
    print(json.dumps({"prior_features_start": len(prior_tasks)}), flush=True)
    prior_features = _feature_all(prior_tasks, int(config["quality"]["metric_resize_long_side"]))
    prior_failures = [row for row in prior_features if row["status"] != "ok"]
    if prior_failures:
        raise ValueError(f"prior source inventory has {len(prior_failures)} decode/hash failures")
    selected, candidate_features = _select(candidate_features, config, prior_features)
    pair_rows = _screen_pairs(selected, prior_features, config)
    outputs = config["outputs"]
    audit_path = _resolve(project_root, outputs["audit_jsonl"])
    prior_path = _resolve(project_root, outputs["old_inventory_jsonl"])
    pair_path = _resolve(project_root, outputs["pair_screen_jsonl"])
    _write_jsonl(audit_path, (_public_feature(row) for row in candidate_features))
    _write_jsonl(prior_path, prior_features)
    _write_jsonl(pair_path, pair_rows)
    draft_rows = [
        _draft_row(row, index, config_sha256, protocol_sha256)
        for index, row in enumerate(selected, 1)
    ]
    draft_path = _resolve(project_root, outputs["draft_selection_jsonl"])
    _write_jsonl(draft_path, draft_rows)
    draft_sha256 = _sha256(draft_path)
    contact_sheets = _render_contact_sheets(
        selected, scratch, _resolve(project_root, outputs["contact_sheet_dir"])
    )
    ocr_rows = _ocr_input_rows(draft_rows, pair_rows, prior_features, scratch)
    ocr_input_path = _resolve(project_root, outputs["ocr_input_jsonl"])
    _write_jsonl(ocr_input_path, ocr_rows)
    summary = {
        "status": "draft_150_ready_for_visual_review_and_ocr",
        "selection_config_sha256": config_sha256,
        "selection_protocol_sha256": protocol_sha256,
        "candidate_counts": dict(sorted(Counter(row["source_dataset"] for row in candidate_features).items())),
        "candidate_failures": sum(row["status"] != "ok" for row in candidate_features),
        "hard_gate_ineligible": sum(not row["hard_gate_eligible"] for row in candidate_features),
        "selected_counts": dict(sorted(Counter(row["source_dataset"] for row in selected).items())),
        "naf_official_groups": len(
            {row["selection_metadata"]["official_group"] for row in candidate_features if row["source_dataset"] == "NAF"}
        ),
        "naf_official_base_families": len(
            {row["selection_metadata"]["official_base_family"] for row in candidate_features if row["source_dataset"] == "NAF"}
        ),
        "prior_source_records": len(prior_features),
        "prior_input_hashes": prior_input_hashes,
        "draft_manifest": outputs["draft_selection_jsonl"],
        "draft_manifest_sha256": draft_sha256,
        "pair_screen_records": len(pair_rows),
        "high_priority_visual_pairs": sum(row["high_priority_visual_flag"] for row in pair_rows),
        "exact_pairs": sum(
            row["exact_encoded_sha256"] or row["exact_decoded_pixel_sha256"] for row in pair_rows
        ),
        "ocr_input_records": len(ocr_rows),
        "ocr_input_sha256": _sha256(ocr_input_path),
        "contact_sheets": contact_sheets,
        "freeze_complete": False,
        "next": "run frozen local CPU OCR, review six contact sheets, bind review decisions, then finalize",
    }
    _write_json(_resolve(project_root, outputs["draft_summary"]), summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _tokens(normalized_text: str) -> set[str]:
    return {value.lower() for value in TOKEN_PATTERN.findall(normalized_text)}


def _minhash(tokens: set[str], permutations: int = 128) -> list[int]:
    if not tokens:
        return []
    signature: list[int] = []
    for permutation in range(permutations):
        prefix = permutation.to_bytes(4, "big")
        signature.append(
            min(
                int.from_bytes(hashlib.sha256(prefix + token.encode("utf-8")).digest()[:8], "big")
                for token in tokens
            )
        )
    return signature


def _minhash_similarity(left: list[int], right: list[int]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def _review_map(path: Path, draft_sha256: str, draft_ids: set[str], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if record_id in result:
            raise ValueError(f"duplicate source review decision: {record_id}")
        if str(row.get("draft_selection_sha256")) != draft_sha256:
            raise ValueError(f"source review is not bound to current draft: {record_id}")
        if row.get("review_mode") not in config["review"]["allowed_review_modes"]:
            raise ValueError(f"unapproved source review mode: {record_id}")
        if row.get("decision") not in config["review"]["allowed_decisions"]:
            raise ValueError(f"unapproved source review decision: {record_id}")
        result[record_id] = row
    if set(result) != draft_ids:
        raise ValueError("source visual review does not cover exactly the draft 150")
    return result


def _pair_review_map(path: Path, pair_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not pair_ids:
        return {}
    rows = _read_jsonl(path)
    result = {str(row["pair_id"]): row for row in rows}
    if len(result) != len(rows) or set(result) != pair_ids:
        raise ValueError("near-duplicate pair review does not cover exactly the flagged pairs")
    return result


def _report(summary: dict[str, Any]) -> str:
    return f"""# Prospective 150-source freeze report

Status: `{summary['status']}`.

- Final sources: {summary['final_sources']} (DocLayNet 50, NAF 50, MIDV-500 50).
- NAF distinct official base families: {summary['naf_distinct_base_families']}.
- Prior materialized source records compared: {summary['prior_source_records']}.
- Exact encoded/pixel duplicate flags: {summary['exact_duplicate_flags']}.
- High-priority visual or OCR near-duplicate flags reviewed: {summary['near_duplicate_flags_reviewed']}.
- OCR records: {summary['ocr_records']} ({summary['ocr_errors']} item errors).
- Source visual reviews: {summary['source_visual_reviews']}.
- Final manifest SHA-256: `{summary['final_manifest_sha256']}`.
- Freeze ID: `{summary['freeze_id']}`.

All source selection used only frozen metadata, quality rules, content hashes,
classical image fingerprints, local OCR for de-duplication, and bound visual
review. No AI editor, detector output, training run, GPU, threshold selection,
or model selection was used. NAF items retain the local-only possible
historical-personal-data handling restriction. The final set remains blocked
from editing and inference until a separate editor/model/mask/device/budget/
cache/retry/reviewer protocol is written and hash-frozen.
"""


def finalize(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha256, protocol_sha256 = _verify_config(project_root, config_path, config)
    outputs = config["outputs"]
    draft_path = _resolve(project_root, outputs["draft_selection_jsonl"])
    draft = _read_jsonl(draft_path)
    draft_sha256 = _sha256(draft_path)
    if len(draft) != 150 or any(row["selection_config_sha256"] != config_sha256 for row in draft):
        raise ValueError("draft 150 is incomplete or bound to another config")
    draft_ids = {str(row["record_id"]) for row in draft}
    if len(draft_ids) != 150:
        raise ValueError("draft record IDs are not unique")
    reviews = _review_map(
        _resolve(project_root, outputs["review_jsonl"]), draft_sha256, draft_ids, config
    )
    rejected = [record_id for record_id, review in reviews.items() if review["decision"] != "approve"]
    if rejected:
        raise ValueError(f"source review rejected {len(rejected)} draft sources")

    ocr_path = _resolve(project_root, outputs["ocr_output_jsonl"])
    ocr_rows = _read_jsonl(ocr_path)
    ocr = {str(row["record_id"]): row for row in ocr_rows}
    if len(ocr) != len(ocr_rows):
        raise ValueError("OCR output contains duplicate records")
    permutations = int(config["representations"]["ocr_minhash"]["permutations"])
    minimum_tokens = int(config["representations"]["ocr_minhash"]["minimum_distinct_tokens"])
    signatures: dict[str, list[int]] = {}
    token_counts: dict[str, int] = {}
    for record_id, row in ocr.items():
        token_set = _tokens(str(row.get("normalized_text", ""))) if row.get("status") == "ok" else set()
        token_counts[record_id] = len(token_set)
        signatures[record_id] = _minhash(token_set, permutations)

    pair_rows = _read_jsonl(_resolve(project_root, outputs["pair_screen_jsonl"]))
    required_ocr_ids: set[str] = set()
    for row in pair_rows:
        if row.get("ocr_candidate"):
            required_ocr_ids.add(str(row["prospective_record_id"]))
            required_ocr_ids.add(str(row["other_record_id"]))
    if not required_ocr_ids.issubset(ocr):
        raise ValueError("OCR output is missing records from the frozen broad pair screen")
    flagged: list[dict[str, Any]] = []
    exact_flags = 0
    for row in pair_rows:
        left_id = str(row["prospective_record_id"])
        right_id = str(row["other_record_id"])
        similarity = None
        if row.get("ocr_candidate") and left_id in signatures and right_id in signatures:
            if token_counts[left_id] >= minimum_tokens and token_counts[right_id] >= minimum_tokens:
                similarity = _minhash_similarity(signatures[left_id], signatures[right_id])
        row["ocr_left_distinct_tokens"] = token_counts.get(left_id, 0)
        row["ocr_right_distinct_tokens"] = token_counts.get(right_id, 0)
        row["ocr_minhash_similarity"] = round(similarity, 8) if similarity is not None else None
        row["ocr_evidence_sufficient"] = similarity is not None
        row["ocr_near_duplicate_flag"] = similarity is not None and similarity >= float(
            config["deduplication"]["ocr_minhash_similarity_min"]
        )
        exact = bool(row["exact_encoded_sha256"] or row["exact_decoded_pixel_sha256"])
        if exact:
            exact_flags += 1
        if exact or row["high_priority_visual_flag"] or row["ocr_near_duplicate_flag"]:
            flagged.append(row)
    if exact_flags:
        raise ValueError(f"exact duplicate flags remain in the draft: {exact_flags}")
    pair_reviews = _pair_review_map(
        _resolve(project_root, outputs["pair_review_jsonl"]), {str(row["pair_id"]) for row in flagged}
    )
    rejected_pairs = [
        pair_id
        for pair_id, review in pair_reviews.items()
        if review.get("decision") not in {"distinct", "allow_with_disclosure"}
    ]
    if rejected_pairs:
        raise ValueError(f"near-duplicate review rejected {len(rejected_pairs)} pairs")

    identity_payload = [
        {
            "source_group_id": row["source_group_id"],
            "encoded_sha256": row["encoded_sha256"],
            "decoded_pixel_sha256": row["decoded_pixel_sha256"],
        }
        for row in draft
    ]
    freeze_id = hashlib.sha256(
        json.dumps(
            {
                "config_sha256": config_sha256,
                "protocol_sha256": protocol_sha256,
                "sources": identity_payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    final_rows: list[dict[str, Any]] = []
    for index, row in enumerate(draft, 1):
        record_id = str(row["record_id"])
        signature = signatures.get(record_id, [])
        signature_bytes = b"".join(value.to_bytes(8, "big") for value in signature)
        ocr_row = ocr.get(record_id)
        final_rows.append(
            {
                **{key: value for key, value in row.items() if key != "draft_index"},
                "source_index": index,
                "freeze_id": freeze_id,
                "freeze_status": "frozen_before_ai_editing_or_detector_inference",
                "ocr_status": ocr_row["status"] if ocr_row else "not_required_no_screened_pair",
                "ocr_normalized_text_sha256": ocr_row["normalized_text_sha256"] if ocr_row else None,
                "ocr_recognized_characters": ocr_row["recognized_characters"] if ocr_row else 0,
                "ocr_distinct_token_count": token_counts.get(record_id, 0),
                "ocr_minhash128_sha256": hashlib.sha256(signature_bytes).hexdigest() if signature else None,
                "visual_review": {
                    "review_mode": reviews[record_id]["review_mode"],
                    "reviewer_id": reviews[record_id]["reviewer_id"],
                    "decision": reviews[record_id]["decision"],
                    "issues": reviews[record_id].get("issues", []),
                    "reviewed_at_utc": reviews[record_id]["reviewed_at_utc"],
                    "draft_selection_sha256": draft_sha256,
                },
            }
        )
    if len({row["source_group_id"] for row in final_rows}) != 150:
        raise ValueError("final source-group IDs are not unique")
    final_path = _resolve(project_root, outputs["final_manifest_jsonl"])
    _write_jsonl(final_path, final_rows)
    final_sha256 = _sha256(final_path)
    sha_path = _resolve(project_root, outputs["final_manifest_sha256"])
    sha_path.parent.mkdir(parents=True, exist_ok=True)
    sha_path.write_text(f"{final_sha256}  {final_path.name}\n", encoding="utf-8")
    summary = {
        "status": "prospective_150_source_manifest_frozen",
        "final_sources": len(final_rows),
        "source_counts": dict(sorted(Counter(row["source_dataset"] for row in final_rows).items())),
        "naf_distinct_base_families": len(
            {row["template_family_id"] for row in final_rows if row["source_dataset"] == "NAF"}
        ),
        "prior_source_records": len(_read_jsonl(_resolve(project_root, outputs["old_inventory_jsonl"]))),
        "exact_duplicate_flags": exact_flags,
        "near_duplicate_flags_reviewed": len(flagged),
        "source_visual_reviews": len(reviews),
        "ocr_records": len(ocr_rows),
        "ocr_errors": sum(row.get("status") != "ok" for row in ocr_rows),
        "draft_manifest_sha256": draft_sha256,
        "selection_config_sha256": config_sha256,
        "selection_protocol_sha256": protocol_sha256,
        "ocr_output_sha256": _sha256(ocr_path),
        "final_manifest": outputs["final_manifest_jsonl"],
        "final_manifest_sha256": final_sha256,
        "freeze_id": freeze_id,
        "editor_authorized": False,
        "detector_inference_authorized": False,
        "gpu_required": False,
    }
    _write_json(_resolve(project_root, outputs["decision_json"]), summary)
    report_path = _resolve(project_root, outputs["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit, de-duplicate, and freeze the prospective 150 sources."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--phase", choices=("audit", "prepare-ocr", "record-agent-review", "finalize"), required=True
    )
    parser.add_argument("--reviewed-at-utc")
    args = parser.parse_args()
    if args.phase == "audit":
        audit(args.config)
    elif args.phase == "prepare-ocr":
        prepare_ocr(args.config)
    elif args.phase == "record-agent-review":
        if not args.reviewed_at_utc:
            parser.error("--reviewed-at-utc is required for record-agent-review")
        record_agent_review(args.config, args.reviewed_at_utc)
    else:
        finalize(args.config)


if __name__ == "__main__":
    main()
