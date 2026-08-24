from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import cv2
import numpy as np
import yaml
from PIL import Image

from pairtrace_doc.pipelines.audit_green_boundary_leakage import (
    _control_boxes,
    _edge_stats,
    _mask_bbox,
)
from pairtrace_doc.pipelines.audit_template_near_duplicate_leakage import (
    _fingerprints,
    _hamming,
    _relative_aspect_difference,
)
from pairtrace_doc.pipelines.freeze_tfr_internal_pair_split import _member_bytes
from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_json,
    _write_jsonl,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _pixel_sha256(image: np.ndarray) -> str:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("pixel hash input must be uint8 RGB")
    return _sha256_bytes(image.tobytes(order="C"))


def _decode_rgb(payload: bytes) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("OpenCV could not decode image")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _png_bytes(array: np.ndarray) -> bytes:
    if array.ndim == 2:
        encoded_input = array
    elif array.ndim == 3 and array.shape[2] == 3:
        encoded_input = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("PNG input has invalid geometry")
    ok, encoded = cv2.imencode(".png", encoded_input, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError("PNG encoding failed")
    return encoded.tobytes()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _green_mask(image: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    red = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    blue = image[:, :, 2].astype(np.int16)
    return (
        (green >= int(rule["min_green_channel"]))
        & ((green - red) >= int(rule["min_green_minus_red"]))
        & ((green - blue) >= int(rule["min_green_minus_blue"]))
    )


def _outer_band_fraction(mask: np.ndarray, band: int) -> float:
    if band <= 0 or min(mask.shape) <= 2 * band:
        raise ValueError("invalid outer-band width")
    selected = np.zeros(mask.shape, dtype=bool)
    selected[:band, :] = True
    selected[-band:, :] = True
    selected[:, :band] = True
    selected[:, -band:] = True
    return float(mask[selected].mean())


def _introduced_green_boundary_flag(
    authentic: np.ndarray,
    forged: np.ndarray,
    changed: np.ndarray,
    rule: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    original_green = _green_mask(authentic, rule)
    forged_green = _green_mask(forged, rule)
    introduced = forged_green & ~original_green
    bbox = _mask_bbox(changed)
    hits, pixels, fraction = _edge_stats(forged_green, bbox, int(rule["edge_band_px"]))
    controls = _control_boxes(bbox, changed.shape, int(rule["control_gap_px"]))
    control_fractions: list[float] = []
    for control in controls:
        _, _, value = _edge_stats(forged_green, control, int(rule["edge_band_px"]))
        control_fractions.append(value)
    control = float(np.mean(control_fractions)) if control_fractions else 0.0
    flag = (
        hits >= int(rule["min_edge_green_pixels"])
        and fraction >= float(rule["min_edge_green_fraction"])
        and fraction / max(control, 1e-12) >= float(rule["min_target_control_ratio"])
        and fraction - control >= float(rule["min_target_control_margin"])
    )
    # A pre-existing natural-green edge is not an introduced signal. The
    # zero-introduction invariant is stronger and is enforced separately.
    introduced_in_edge = bool(np.any(introduced & changed))
    return bool(flag and introduced_in_edge), {
        "introduced_green_pixels": int(introduced.sum()),
        "target_edge_green_pixels": hits,
        "target_edge_pixels": pixels,
        "target_edge_green_fraction": fraction,
        "control_edge_green_fraction": control,
    }


def _bits_hex(bits: np.ndarray) -> str:
    return np.packbits(bits.astype(np.uint8)).tobytes().hex()


def _bits_from_hex(value: str, bit_count: int) -> np.ndarray:
    unpacked = np.unpackbits(np.frombuffer(bytes.fromhex(value), dtype=np.uint8))
    return unpacked[:bit_count].astype(bool)


def _fingerprint_record(image: np.ndarray, hash_size: int) -> dict[str, Any]:
    phash, dhash = _fingerprints(image, hash_size)
    return {
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "pixel_sha256": _pixel_sha256(image),
        "phash": _bits_hex(phash),
        "dhash": _bits_hex(dhash),
    }


def _near_duplicate(
    left: dict[str, Any], right: dict[str, Any], screening: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    size = int(screening["phash_size"])
    bit_count = size * size
    phash = _hamming(
        _bits_from_hex(str(left["phash"]), bit_count),
        _bits_from_hex(str(right["phash"]), bit_count),
    )
    dhash = _hamming(
        _bits_from_hex(str(left["dhash"]), bit_count),
        _bits_from_hex(str(right["dhash"]), bit_count),
    )
    aspect = _relative_aspect_difference(
        (int(left["height"]), int(left["width"])),
        (int(right["height"]), int(right["width"])),
    )
    duplicate = phash <= int(screening["phash_only_max_distance"]) or (
        phash <= int(screening["combined_phash_max_distance"])
        and dhash <= int(screening["combined_dhash_max_distance"])
        and aspect <= float(screening["combined_max_aspect_difference"])
    )
    return duplicate, {
        "phash_distance": phash,
        "dhash_distance": dhash,
        "aspect_ratio_relative_difference": aspect,
    }


def _trigrams(value: str) -> set[str]:
    compact = "".join(value.split())
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[index : index + 3] for index in range(len(compact) - 2)}


def _ocr_duplicate(
    left: str, right: str, screening: dict[str, Any]
) -> tuple[bool, dict[str, float | bool]]:
    first = "".join(left.split())
    second = "".join(right.split())
    minimum = int(screening["ocr_min_characters_for_duplicate"])
    if min(len(first), len(second)) < minimum:
        return False, {"exact": False, "jaccard": 0.0, "length_ratio": 0.0}
    exact = first == second
    union = _trigrams(first) | _trigrams(second)
    jaccard = len(_trigrams(first) & _trigrams(second)) / max(1, len(union))
    length_ratio = min(len(first), len(second)) / max(len(first), len(second))
    duplicate = exact or (
        jaccard >= float(screening["ocr_trigram_jaccard_min"])
        and length_ratio >= float(screening["ocr_length_ratio_min"])
    )
    return duplicate, {"exact": exact, "jaccard": jaccard, "length_ratio": length_ratio}


def _usable_boxes(
    ocr: dict[str, Any], shape: tuple[int, int], screening: dict[str, Any]
) -> list[tuple[int, int, int, int]]:
    height, width = shape
    result: list[tuple[int, int, int, int]] = []
    for box, score in zip(ocr["boxes"], ocr["scores"], strict=True):
        if float(score) < float(screening["ocr_min_score"]):
            continue
        x1, y1, x2, y2 = (int(value) for value in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        margin = int(screening["ocr_box_margin_px"])
        if (
            x1 < margin
            or y1 < margin
            or x2 > width - margin
            or y2 > height - margin
            or x2 - x1 < int(screening["ocr_min_box_width"])
            or y2 - y1 < int(screening["ocr_min_box_height"])
        ):
            continue
        result.append((x1, y1, x2, y2))
    return result


def _hash_index(source_id: str, label: str, length: int) -> int:
    if length <= 0:
        raise ValueError("hash index requires a non-empty sequence")
    digest = hashlib.sha256(f"{source_id}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % length


def _copy_move(
    authentic: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    source_id: str,
) -> np.ndarray:
    target_index = _hash_index(source_id, "copy-target", len(boxes))
    donor_candidates = [index for index in range(len(boxes)) if index != target_index]
    donor_index = donor_candidates[_hash_index(source_id, "copy-donor", len(donor_candidates))]
    tx1, ty1, tx2, ty2 = boxes[target_index]
    sx1, sy1, sx2, sy2 = boxes[donor_index]
    donor = authentic[sy1:sy2, sx1:sx2]
    resized = cv2.resize(donor, (tx2 - tx1, ty2 - ty1), interpolation=cv2.INTER_AREA)
    result = authentic.copy()
    result[ty1:ty2, tx1:tx2] = resized
    return result


def _local_erase(
    authentic: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    source_id: str,
) -> np.ndarray:
    index = _hash_index(source_id, "erase-target", len(boxes))
    x1, y1, x2, y2 = boxes[index]
    gray = cv2.cvtColor(authentic, cv2.COLOR_RGB2GRAY)
    region = gray[y1:y2, x1:x2]
    threshold, _ = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = region < min(220.0, float(threshold) + 16.0)
    if int(foreground.sum()) < 8:
        raise ValueError("erase target lacks foreground pixels")
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = foreground.astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    bgr = cv2.cvtColor(authentic, cv2.COLOR_RGB2BGR)
    result = cv2.inpaint(bgr, mask, 3.0, cv2.INPAINT_TELEA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def _make_variant(
    authentic: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    source_id: str,
    attack: str,
    generation: dict[str, Any],
    green_rule: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if attack == "ocr_copy_move":
        forged = _copy_move(authentic, boxes, source_id)
    elif attack == "ocr_local_erase":
        forged = _local_erase(authentic, boxes, source_id)
    else:
        raise ValueError(f"unsupported attack: {attack}")
    introduced_green = _green_mask(forged, green_rule) & ~_green_mask(authentic, green_rule)
    forged[introduced_green] = authentic[introduced_green]
    # Re-decode the lossless representation before defining exact ground truth.
    forged = _decode_rgb(_png_bytes(forged))
    changed = np.any(forged != authentic, axis=2)
    fraction = float(changed.mean())
    if not float(generation["min_changed_fraction"]) <= fraction <= float(
        generation["max_changed_fraction"]
    ):
        raise ValueError(f"changed fraction outside frozen bounds: {fraction:.8f}")
    flag, green = _introduced_green_boundary_flag(
        authentic, forged, changed, green_rule
    )
    if green["introduced_green_pixels"] != 0 or flag:
        raise ValueError("generated variant introduced a green-boundary signal")
    bbox = _mask_bbox(changed)
    return forged, changed, {"changed_fraction": fraction, "bbox": bbox, **green}


def _reference_inventory(
    project_root: Path,
    scratch: Path,
    specifications: list[dict[str, Any]],
    screening: dict[str, Any],
    cache_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    manifest_hashes: dict[str, str] = {}
    sources: dict[tuple[str, str], dict[str, str]] = {}
    for specification in specifications:
        manifest = _resolve(project_root, str(specification["path"]))
        digest = _sha256(manifest)
        if digest != str(specification["expected_sha256"]):
            raise ValueError(f"reference manifest changed: {manifest}")
        manifest_hashes[str(manifest.relative_to(project_root))] = digest
        filters = {str(key): value for key, value in specification.get("filters", {}).items()}
        for row in _read_jsonl(manifest):
            if any(row.get(key) != value for key, value in filters.items()):
                continue
            path_value = row.get(str(specification["path_field"]))
            hash_value = row.get(str(specification["hash_field"]))
            if not path_value or not hash_value:
                raise ValueError(f"reference row lacks authentic path/hash: {manifest}")
            key = (str(path_value), str(hash_value))
            sources[key] = {"path": key[0], "encoded_sha256": key[1]}
    cache_key = _canonical_digest(
        {"manifest_hashes": manifest_hashes, "screening": screening, "sources": sorted(sources)}
    )
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key:
            return list(cached["records"]), manifest_hashes
    records: list[dict[str, Any]] = []
    for (path_value, expected_hash), source in sorted(sources.items()):
        path = _resolve(scratch, path_value)
        if _sha256(path) != expected_hash:
            raise ValueError(f"reference authentic hash changed: {path}")
        with Image.open(path) as handle:
            image = np.asarray(handle.convert("RGB"))
        records.append(
            {
                **source,
                **_fingerprint_record(image, int(screening["phash_size"])),
            }
        )
    _write_json(cache_path, {"cache_key": cache_key, "records": records})
    return records, manifest_hashes


def _run_ocr(
    project_root: Path,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    paths: dict[str, Any],
    scratch: Path,
) -> dict[str, dict[str, Any]]:
    cache_path = _resolve(scratch, str(paths["private_ocr_cache"]))
    cache_rows = _read_jsonl(cache_path) if cache_path.is_file() else []
    cache = {str(row["image_sha256"]): row for row in cache_rows}
    missing = [row for row in candidates if str(row["encoded_sha256"]) not in cache]
    if missing:
        input_path = _resolve(scratch, str(paths["private_ocr_input"]))
        output_path = _resolve(scratch, str(paths["private_ocr_output"]))
        _write_jsonl(
            input_path,
            [
                {
                    "record_id": row["private_member_id"],
                    "image": row["private_image_path"],
                    "image_sha256": row["encoded_sha256"],
                }
                for row in missing
            ],
        )
        ocr = config["ocr"]
        python_default = Path(str(ocr["python_default"])).expanduser()
        if not python_default.is_absolute():
            python_default = scratch / python_default
        python_path = Path(
            os.environ.get(str(ocr["python_env"]), str(python_default))
        ).expanduser().absolute()
        detector = Path(
            os.environ.get(
                str(ocr["detector_dir_env"]),
                str(_resolve(project_root, str(ocr["detector_dir_default"]))),
            )
        ).resolve()
        recognizer = Path(
            os.environ.get(
                str(ocr["recognizer_dir_env"]),
                str(_resolve(project_root, str(ocr["recognizer_dir_default"]))),
            )
        ).resolve()
        command = [
            str(python_path),
            "-m",
            "pairtrace_doc.pipelines.run_paddle_ocr_screen",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--detector-dir",
            str(detector),
            "--recognizer-dir",
            str(recognizer),
            "--detector-sha256",
            str(ocr["detector_weights_sha256"]),
            "--recognizer-sha256",
            str(ocr["recognizer_weights_sha256"]),
            "--recognition-threshold",
            str(config["screening"]["ocr_min_score"]),
            "--progress-every",
            str(config["runtime"]["progress_every"]),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "DISABLE_MODEL_SOURCE_CHECK": "True",
                "PYTHONPATH": str(project_root / "src"),
            }
        )
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "OCR worker failed; stderr tail: " + completed.stderr[-2000:]
            )
        for row in _read_jsonl(output_path):
            cache[str(row["image_sha256"])] = row
        _write_jsonl(cache_path, [cache[key] for key in sorted(cache)])
    result: dict[str, dict[str, Any]] = {}
    for row in candidates:
        digest = str(row["encoded_sha256"])
        if digest not in cache:
            raise ValueError(f"OCR cache lacks candidate {digest}")
        result[str(row["private_member_id"])] = cache[digest]
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    forbidden = (
        "model_inference_authorized",
        "model_training_authorized",
        "threshold_selection_authorized",
        "prediction_read_authorized",
    )
    if any(bool(runtime.get(key)) for key in forbidden) or not bool(
        runtime["data_materialization_authorized"]
    ):
        raise ValueError("confirmation materializer crossed its evidence boundary")
    experiment = config["experiment"]
    protocol = _resolve(project_root, str(experiment["protocol"]))
    if _sha256(protocol) != str(experiment["expected_protocol_sha256"]):
        raise ValueError("confirmation protocol SHA-256 changed")
    source = config["source"]
    password = os.environ.get(str(source["tfr_password_env"]))
    permission = os.environ.get(str(source["tfr_permission_record_env"]))
    if not password or not permission:
        raise ValueError("TFR password and permission record must be process-only environment values")
    permission_digest = _sha256_bytes(permission.encode("utf-8"))
    if permission_digest != str(source["expected_permission_record_id_sha256"]):
        raise ValueError("TFR permission record identity changed")
    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            str(paths["scratch_env"]),
            str(_resolve(project_root, str(paths["scratch_default"]))),
        )
    ).resolve()
    archive_path = _resolve(scratch, str(paths["tfr_inner_archive"]))
    if archive_path.stat().st_size != int(source["tfr_inner_archive_bytes"]):
        raise ValueError("TFR inner archive size changed")
    if runtime.get("verify_archive_sha256", False):
        if _sha256(archive_path) != str(source["tfr_inner_archive_sha256"]):
            raise ValueError("TFR inner archive SHA-256 changed")
    else:
        integrity_path = _resolve(project_root, str(source["archive_integrity_record"]))
        if _sha256(integrity_path) != str(source["archive_integrity_record_sha256"]):
            raise ValueError("frozen archive-integrity record changed")
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))["archive"]
        if (
            int(integrity["bytes"]) != int(source["tfr_inner_archive_bytes"])
            or str(integrity["sha256"]) != str(source["tfr_inner_archive_sha256"])
        ):
            raise ValueError("frozen archive-integrity record disagrees with config")

    screening = config["screening"]
    references, reference_hashes = _reference_inventory(
        project_root,
        scratch,
        config["reference_manifests"],
        screening,
        _resolve(scratch, str(paths["reference_fingerprint_cache"])),
    )
    target_groups = int(experiment["target_source_groups"])
    scan_budget = int(experiment["ocr_scan_candidates"])
    private_candidate_root = _resolve(scratch, str(paths["private_candidate_root"]))
    audit_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    with ZipFile(archive_path) as archive:
        prefix = str(source["eligible_member_prefix"])
        members = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and info.filename.startswith(prefix)
            and Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if len(members) != int(source["expected_eligible_members"]):
            raise ValueError("eligible TFR authentic-source universe changed")
        seed = int(experiment["selection_seed"])
        members.sort(
            key=lambda info: hashlib.sha256(
                f"pairtrace-confirmatory-v1|{seed}|{info.filename}".encode("utf-8")
            ).hexdigest()
        )
        for rank, info in enumerate(members, 1):
            if len(candidates) >= scan_budget:
                break
            private_id = _sha256_bytes(
                f"pairtrace-private|{seed}|{info.filename}".encode("utf-8")
            )
            public_id = _sha256_bytes(
                f"pairtrace-confirmatory-v1|{seed}|{info.filename}".encode("utf-8")
            )[:20]
            audit = {
                "rank": rank,
                "private_member_id": private_id,
                "public_member_id": public_id,
                "status": "rejected",
                "reason": None,
            }
            try:
                payload = _member_bytes(
                    archive,
                    info.filename,
                    password,
                    int(runtime["max_member_bytes"]),
                )
                image = _decode_rgb(payload)
                if min(image.shape[:2]) < int(screening["min_short_side"]):
                    raise ValueError("short_side_below_minimum")
                green_fraction = _outer_band_fraction(
                    _green_mask(image, screening["green"]),
                    int(screening["green"]["outer_band_px"]),
                )
                if green_fraction >= float(screening["green"]["max_outer_band_fraction"]):
                    raise ValueError("green_outer_band_screen")
                encoded_sha = _sha256_bytes(payload)
                candidate_path = private_candidate_root / f"{private_id[:24]}.jpg"
                if not candidate_path.is_file() or _sha256(candidate_path) != encoded_sha:
                    _atomic_bytes(candidate_path, payload)
                fingerprint = _fingerprint_record(image, int(screening["phash_size"]))
                candidates.append(
                    {
                        "rank": rank,
                        "private_member_id": private_id,
                        "public_member_id": public_id,
                        "private_archive_member": info.filename,
                        "private_image_path": str(candidate_path),
                        "encoded_sha256": encoded_sha,
                        "encoded_bytes": len(payload),
                        "green_outer_band_fraction": green_fraction,
                        **fingerprint,
                    }
                )
                audit["status"] = "ocr_pending"
                audit["reason"] = None
            except Exception as exc:
                audit["reason"] = str(exc)
            audit_rows.append(audit)
    if len(candidates) < target_groups:
        raise ValueError("too few source candidates reached OCR screening")

    ocr_by_id = _run_ocr(project_root, candidates, config, paths, scratch)
    accepted: list[dict[str, Any]] = []
    pair_payloads: list[dict[str, Any]] = []
    baseline_payloads: list[dict[str, Any]] = []
    generation = config["generation"]
    attacks = [str(value) for value in generation["attacks"]]
    if attacks != ["ocr_copy_move", "ocr_local_erase"]:
        raise ValueError("frozen attack set changed")
    derived_root = _resolve(scratch, str(paths["derived_root"]))
    audit_by_id = {str(row["private_member_id"]): row for row in audit_rows}
    for candidate in candidates:
        private_id = str(candidate["private_member_id"])
        audit = audit_by_id[private_id]
        ocr = ocr_by_id[private_id]
        if ocr["status"] != "ok":
            audit.update({"status": "rejected", "reason": "ocr_worker_error"})
            continue
        duplicate_reason: str | None = None
        duplicate_detail: dict[str, Any] | None = None
        for prior in [*references, *accepted]:
            if (
                candidate["encoded_sha256"] == prior["encoded_sha256"]
                or candidate["pixel_sha256"] == prior["pixel_sha256"]
            ):
                duplicate_reason = "exact_duplicate"
                break
            duplicate, detail = _near_duplicate(candidate, prior, screening)
            if duplicate:
                duplicate_reason = "perceptual_near_duplicate"
                duplicate_detail = detail
                break
        if duplicate_reason is None:
            for prior in accepted:
                duplicate, detail = _ocr_duplicate(
                    str(ocr["normalized_text"]),
                    str(prior["private_normalized_text"]),
                    screening,
                )
                if duplicate:
                    duplicate_reason = "ocr_text_near_duplicate"
                    duplicate_detail = detail
                    break
        if duplicate_reason:
            audit.update(
                {
                    "status": "rejected",
                    "reason": duplicate_reason,
                    "duplicate_detail": duplicate_detail,
                }
            )
            continue
        image = _decode_rgb(Path(str(candidate["private_image_path"])).read_bytes())
        boxes = _usable_boxes(ocr, image.shape[:2], screening)
        if len(boxes) < int(screening["ocr_min_usable_boxes"]):
            audit.update(
                {"status": "rejected", "reason": "insufficient_usable_ocr_boxes"}
            )
            continue
        source_group_id = f"tfr-doc-auth-confirm:{candidate['public_member_id']}"
        source_dir = derived_root / str(candidate["public_member_id"])
        authentic_path = source_dir / "authentic.jpg"
        authentic_payload = Path(str(candidate["private_image_path"])).read_bytes()
        _atomic_bytes(authentic_path, authentic_payload)
        variants: list[dict[str, Any]] = []
        try:
            for attack in attacks:
                forged, mask, details = _make_variant(
                    image,
                    boxes,
                    source_group_id,
                    attack,
                    generation,
                    screening["green"],
                )
                forged_path = source_dir / f"{attack}.png"
                mask_path = source_dir / f"{attack}_mask.png"
                _atomic_bytes(forged_path, _png_bytes(forged))
                _atomic_bytes(mask_path, _png_bytes(mask.astype(np.uint8) * 255))
                variants.append(
                    {
                        "attack": attack,
                        "forged_path": forged_path,
                        "mask_path": mask_path,
                        "forged_sha256": _sha256(forged_path),
                        "mask_sha256": _sha256(mask_path),
                        **details,
                    }
                )
        except Exception as exc:
            audit.update(
                {"status": "rejected", "reason": f"generation_failure:{type(exc).__name__}:{exc}"}
            )
            continue
        accepted_record = {
            **candidate,
            "source_group_id": source_group_id,
            "authentic_path": str(authentic_path.relative_to(scratch)),
            "authentic_sha256": _sha256(authentic_path),
            "private_normalized_text": str(ocr["normalized_text"]),
            "ocr_text_sha256": str(ocr["normalized_text_sha256"]),
            "ocr_characters": int(ocr["recognized_characters"]),
            "usable_ocr_boxes": len(boxes),
            "variants": variants,
        }
        accepted.append(accepted_record)
        audit.update(
            {
                "status": "accepted",
                "reason": None,
                "source_group_id": source_group_id,
                "ocr_text_sha256": accepted_record["ocr_text_sha256"],
                "ocr_characters": accepted_record["ocr_characters"],
                "usable_ocr_boxes": len(boxes),
            }
        )
        for variant in variants:
            pair_payloads.append(
                {
                    "sample_id": f"{source_group_id}:{variant['attack']}",
                    "source_group_id": source_group_id,
                    "source_dataset": "TFR-DOC_AUTH-self-generated-v1",
                    "selected_generator": variant["attack"],
                    "pilot_role": "confirmation",
                    "role": "confirmation",
                    "image": str(variant["forged_path"].relative_to(scratch)),
                    "authentic": str(authentic_path.relative_to(scratch)),
                    "mask": str(variant["mask_path"].relative_to(scratch)),
                    "image_sha256": variant["forged_sha256"],
                    "authentic_sha256": accepted_record["authentic_sha256"],
                    "mask_sha256": variant["mask_sha256"],
                    "image_height": int(image.shape[0]),
                    "image_width": int(image.shape[1]),
                    "mask_processing": "decoded_exact_any_channel_difference",
                    "mapping_rule": "deterministic_generation_from_stored_authentic",
                    "changed_fraction": variant["changed_fraction"],
                    "paper_evidence_candidate": "controlled_confirmation_only",
                    "valid": True,
                    "errors": [],
                }
            )
        baseline_payloads.append(
            {
                "record_id": f"{source_group_id}:authentic",
                "source_group_id": source_group_id,
                "evaluation_role": "controlled_confirmation",
                "sample_kind": "authentic",
                "image": str(authentic_path.relative_to(scratch)),
                "image_sha256": accepted_record["authentic_sha256"],
                "mask": None,
                "mask_sha256": None,
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "paper_evidence_candidate": "controlled_confirmation_only",
            }
        )
        for variant in variants:
            baseline_payloads.append(
                {
                    "record_id": f"{source_group_id}:{variant['attack']}",
                    "source_group_id": source_group_id,
                    "evaluation_role": "controlled_confirmation",
                    "sample_kind": "forged",
                    "image": str(variant["forged_path"].relative_to(scratch)),
                    "image_sha256": variant["forged_sha256"],
                    "mask": str(variant["mask_path"].relative_to(scratch)),
                    "mask_sha256": variant["mask_sha256"],
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                    "attack": variant["attack"],
                    "paper_evidence_candidate": "controlled_confirmation_only",
                }
            )
        if len(accepted) >= target_groups:
            break
    if len(accepted) != target_groups:
        reason_counts = Counter(str(row.get("reason")) for row in audit_rows)
        raise ValueError(
            f"could not materialize target groups: {len(accepted)}/{target_groups}; {dict(reason_counts)}"
        )
    for row in audit_rows:
        if row["status"] == "ocr_pending":
            row["status"] = "not_considered_after_target"
            row["reason"] = "not_considered_after_target"

    membership_payloads = [
        {
            "source_group_id": row["source_group_id"],
            "public_member_id": row["public_member_id"],
            "rank": row["rank"],
            "authentic": row["authentic_path"],
            "authentic_sha256": row["authentic_sha256"],
            "authentic_pixel_sha256": row["pixel_sha256"],
            "height": row["height"],
            "width": row["width"],
            "ocr_text_sha256": row["ocr_text_sha256"],
            "ocr_characters": row["ocr_characters"],
            "usable_ocr_boxes": row["usable_ocr_boxes"],
        }
        for row in accepted
    ]
    generator_path = Path(__file__).resolve()
    freeze_inputs = {
        "protocol_sha256": _sha256(protocol),
        "membership_payload_sha256": _canonical_digest(membership_payloads),
        "pair_payload_sha256": _canonical_digest(pair_payloads),
        "generator_sha256": _sha256(generator_path),
        "ocr_detector_weights_sha256": config["ocr"]["detector_weights_sha256"],
        "ocr_recognizer_weights_sha256": config["ocr"]["recognizer_weights_sha256"],
        "archive_sha256": source["tfr_inner_archive_sha256"],
        "archive_bytes": int(source["tfr_inner_archive_bytes"]),
        "selection_seed": int(experiment["selection_seed"]),
    }
    freeze_id = _canonical_digest(freeze_inputs)
    for rows in (membership_payloads, pair_payloads, baseline_payloads):
        for row in rows:
            row["freeze_id"] = freeze_id

    membership_path = _resolve(project_root, str(paths["membership_manifest"]))
    pair_path = _resolve(project_root, str(paths["pair_manifest"]))
    baseline_path = _resolve(project_root, str(paths["baseline_manifest"]))
    audit_path = _resolve(project_root, str(paths["candidate_audit"]))
    summary_csv = _resolve(project_root, str(paths["summary_csv"]))
    summary_path = _resolve(project_root, str(paths["summary_json"]))
    _write_jsonl(membership_path, membership_payloads)
    _write_jsonl(pair_path, pair_payloads)
    _write_jsonl(baseline_path, baseline_payloads)
    _write_jsonl(audit_path, audit_rows)
    reason_counts = Counter(
        "accepted" if row["status"] == "accepted" else str(row.get("reason"))
        for row in audit_rows
    )
    _write_csv(
        summary_csv,
        [
            {"outcome": reason, "count": count}
            for reason, count in sorted(reason_counts.items())
        ],
    )
    summary = {
        "status": "tfr_synthetic_clean_confirmatory_frozen",
        "experiment": experiment,
        "claim_boundary": "controlled_confirmation_only_not_official_tfr",
        "source_groups": len(membership_payloads),
        "forged_pairs": len(pair_payloads),
        "baseline_records": len(baseline_payloads),
        "candidate_audit_records": len(audit_rows),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "model_inference_performed": False,
        "model_training_performed": False,
        "threshold_selection_performed": False,
        "permission_record_id_sha256": permission_digest,
        "reference_manifest_sha256": reference_hashes,
        "freeze_id": freeze_id,
        "freeze_inputs": freeze_inputs,
        "outputs": {
            "membership_manifest": str(membership_path.relative_to(project_root)),
            "membership_manifest_sha256": _sha256(membership_path),
            "pair_manifest": str(pair_path.relative_to(project_root)),
            "pair_manifest_sha256": _sha256(pair_path),
            "baseline_manifest": str(baseline_path.relative_to(project_root)),
            "baseline_manifest_sha256": _sha256(baseline_path),
            "candidate_audit": str(audit_path.relative_to(project_root)),
            "candidate_audit_sha256": _sha256(audit_path),
            "summary_csv": str(summary_csv.relative_to(project_root)),
            "summary_csv_sha256": _sha256(summary_csv),
        },
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
