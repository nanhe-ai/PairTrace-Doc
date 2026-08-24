from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from PIL import Image


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            records.append(value)
    return records


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _scratch_root(project_root: Path, paths: dict[str, Any]) -> Path:
    environment = paths.get("scratch_env")
    override = os.environ.get(str(environment)) if environment else None
    if override:
        return Path(override).expanduser().resolve()
    return _resolve(project_root, str(paths["scratch_default"]))


def _artifact_path(scratch: Path, relative: str) -> Path:
    candidate = (scratch / relative).resolve()
    try:
        candidate.relative_to(scratch.resolve())
    except ValueError as error:
        raise ValueError(f"artifact escapes scratch root: {relative}") from error
    return candidate


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image is not uint8 RGB: {path}")
    return array


def _load_mask(path: Path) -> tuple[np.ndarray, list[int]]:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("L"), dtype=np.uint8)
    values = sorted(int(value) for value in np.unique(array))
    return array > 0, values


def _verify_artifact(
    scratch: Path,
    relative: str,
    expected_sha256: str,
    *,
    rgb: bool,
) -> tuple[np.ndarray, Path, list[int] | None]:
    path = _artifact_path(scratch, relative)
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"artifact SHA-256 changed: {relative}: {actual_sha256} != {expected_sha256}"
        )
    if rgb:
        return _load_rgb(path), path, None
    mask, values = _load_mask(path)
    return mask, path, values


def _resize_for_registration(
    image: np.ndarray, max_side: int
) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image, 1.0, 1.0
    scale = max_side / max(height, width)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    resized = cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
    return resized, target_width / width, target_height / height


def _mean_ssim(first: np.ndarray, second: np.ndarray, valid: np.ndarray) -> float:
    first_float = first.astype(np.float32) / 255.0
    second_float = second.astype(np.float32) / 255.0
    constant_1 = 0.01**2
    constant_2 = 0.03**2
    scores = []
    for channel in range(3):
        left = first_float[:, :, channel]
        right = second_float[:, :, channel]
        mean_left = cv2.GaussianBlur(left, (11, 11), 1.5)
        mean_right = cv2.GaussianBlur(right, (11, 11), 1.5)
        variance_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mean_left**2
        variance_right = (
            cv2.GaussianBlur(right * right, (11, 11), 1.5) - mean_right**2
        )
        covariance = (
            cv2.GaussianBlur(left * right, (11, 11), 1.5)
            - mean_left * mean_right
        )
        numerator = (2.0 * mean_left * mean_right + constant_1) * (
            2.0 * covariance + constant_2
        )
        denominator = (mean_left**2 + mean_right**2 + constant_1) * (
            variance_left + variance_right + constant_2
        )
        scores.append(numerator / np.maximum(denominator, 1e-12))
    score = np.mean(np.stack(scores, axis=2), axis=2)
    return float(score[valid].mean())


def _edge_overlap(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
    low_threshold: int,
    high_threshold: int,
) -> dict[str, float]:
    first_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    first_edge = cv2.Canny(first_gray, low_threshold, high_threshold) > 0
    second_edge = cv2.Canny(second_gray, low_threshold, high_threshold) > 0
    first_edge &= valid
    second_edge &= valid
    intersection = int(np.logical_and(first_edge, second_edge).sum())
    union = int(np.logical_or(first_edge, second_edge).sum())
    total = int(first_edge.sum()) + int(second_edge.sum())
    return {
        "edge_dice": float(2 * intersection / total) if total else 1.0,
        "edge_iou": float(intersection / union) if union else 1.0,
    }


def _registration_diagnostics(
    scan: np.ndarray, clean: np.ndarray, settings: dict[str, Any]
) -> dict[str, Any]:
    if scan.shape != clean.shape:
        raise ValueError("scan/clean registration requires equal image geometry")
    small_scan, scale_x, scale_y = _resize_for_registration(
        scan, int(settings["max_side"])
    )
    small_clean = cv2.resize(
        clean,
        (small_scan.shape[1], small_scan.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    template = cv2.cvtColor(small_scan, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    input_image = (
        cv2.cvtColor(small_clean, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    )
    phase_shift, phase_response = cv2.phaseCorrelate(template, input_image)
    if not np.isfinite(phase_shift).all() or not np.isfinite(phase_response):
        raise ValueError("phase correlation returned non-finite output")
    small_warp = np.eye(3, dtype=np.float64)
    small_warp[0, 2] = float(phase_shift[0])
    small_warp[1, 2] = float(phase_shift[1])
    alignment_status = "ecc_converged"
    failure_type = None
    failure_reason = None
    ecc_correlation = math.nan
    try:
        ecc_correlation, fitted = cv2.findTransformECC(
            template,
            input_image,
            small_warp.astype(np.float32),
            cv2.MOTION_HOMOGRAPHY,
            (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(settings["iterations"]),
                float(settings["epsilon"]),
            ),
            None,
            int(settings["gauss_filter_size"]),
        )
        small_warp = fitted.astype(np.float64)
        if not np.isfinite(small_warp).all() or not np.isfinite(ecc_correlation):
            raise ValueError("ECC returned non-finite output")
        if abs(float(small_warp[2, 2])) < 1e-12:
            raise ValueError("ECC returned singular homography normalization")
        small_warp /= small_warp[2, 2]
    except Exception as error:
        alignment_status = "phase_correlation_fallback_recorded"
        failure_type = type(error).__name__
        failure_reason = str(error)

    scale = np.asarray(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    warp = np.linalg.inv(scale) @ small_warp @ scale
    if abs(float(warp[2, 2])) < 1e-12:
        raise ValueError("full-resolution homography is singular")
    warp /= warp[2, 2]
    determinant = float(np.linalg.det(warp))
    condition = float(np.linalg.cond(warp))
    if not np.isfinite(determinant) or not np.isfinite(condition):
        raise ValueError("homography determinant/condition is non-finite")

    height, width = scan.shape[:2]
    output_size = (width, height)
    aligned_clean = cv2.warpPerspective(
        clean,
        warp.astype(np.float32),
        output_size,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    ones = np.ones((height, width), dtype=np.uint8)
    valid_forward = cv2.warpPerspective(
        ones,
        warp.astype(np.float32),
        output_size,
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    inverse_warp = np.linalg.inv(warp)
    valid_reverse = cv2.warpPerspective(
        ones,
        inverse_warp.astype(np.float32),
        output_size,
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    mutual_support = min(float(valid_forward.mean()), float(valid_reverse.mean()))
    normalized_rgb_difference = float(
        np.abs(scan.astype(np.float32) - aligned_clean.astype(np.float32))[
            valid_forward
        ].mean()
        / 255.0
    )
    result = {
        "status": alignment_status,
        "phase_shift_working_xy": [float(phase_shift[0]), float(phase_shift[1])],
        "phase_shift_full_xy": [
            float(phase_shift[0] / scale_x),
            float(phase_shift[1] / scale_y),
        ],
        "phase_correlation_response": float(phase_response),
        "ecc_converged": alignment_status == "ecc_converged",
        "ecc_correlation": float(ecc_correlation),
        "homography": warp.tolist(),
        "homography_determinant": determinant,
        "homography_condition": condition,
        "forward_valid_support": float(valid_forward.mean()),
        "reverse_valid_support": float(valid_reverse.mean()),
        "mutual_valid_support": mutual_support,
        "registered_ssim": _mean_ssim(scan, aligned_clean, valid_forward),
        "registered_normalized_rgb_difference": normalized_rgb_difference,
        "failure_type": failure_type,
        "failure_reason": failure_reason,
    }
    result.update(
        _edge_overlap(
            scan,
            aligned_clean,
            valid_forward,
            int(settings["canny_low_threshold"]),
            int(settings["canny_high_threshold"]),
        )
    )
    return result


def _marker_green(image: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    value = image.astype(np.int16)
    red, green, blue = value[:, :, 0], value[:, :, 1], value[:, :, 2]
    return (
        (green >= int(settings["min_green_channel"]))
        & (green - red >= int(settings["min_green_minus_red"]))
        & (green - blue >= int(settings["min_green_minus_blue"]))
    )


def _encoded_colors(image: np.ndarray) -> np.ndarray:
    value = image.astype(np.uint32)
    return (value[:, :, 0] << 16) | (value[:, :, 1] << 8) | value[:, :, 2]


def _rectangles_separated_xywh(
    first: list[int], second: list[int], gap: int
) -> bool:
    ax, ay, width_a, height_a = first
    bx, by, width_b, height_b = second
    return (
        ax + width_a + gap <= bx
        or bx + width_b + gap <= ax
        or ay + height_a + gap <= by
        or by + height_b + gap <= ay
    )


def _audit_attack(
    *,
    scratch: Path,
    scan: np.ndarray,
    attack_name: str,
    attack: dict[str, Any],
    marker_settings: dict[str, Any],
    required_gap: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if attack.get("status") == "failed":
        return {
            "attack": attack_name,
            "status": "failed",
            "errors": [str(attack.get("failure_reason", "materialization_failed"))],
            "materialization_failure_type": attack.get("failure_type"),
        }
    candidate, candidate_path, _ = _verify_artifact(
        scratch,
        str(attack["candidate"]),
        str(attack["candidate_sha256"]),
        rgb=True,
    )
    mask, mask_path, mask_values = _verify_artifact(
        scratch,
        str(attack["mask"]),
        str(attack["mask_sha256"]),
        rgb=False,
    )
    if candidate.shape != scan.shape:
        errors.append("candidate_scan_dimension_mismatch")
    if mask.shape != scan.shape[:2]:
        errors.append("mask_scan_dimension_mismatch")
    if mask_values != [0, 255]:
        errors.append("mask_values_not_exactly_0_255")
    if errors:
        exact_change = np.zeros(scan.shape[:2], dtype=bool)
    else:
        exact_change = np.any(candidate != scan, axis=2)
        if not np.array_equal(mask, exact_change):
            errors.append("mask_not_equal_to_exact_changed_pixels")
        if not exact_change.any():
            errors.append("empty_exact_changed_pixel_mask")

    metadata = attack.get("metadata", {})
    destination = [int(value) for value in metadata.get("destination_xywh", [])]
    if len(destination) != 4:
        errors.append("invalid_destination_xywh")
        allowed = np.zeros(scan.shape[:2], dtype=bool)
    else:
        x, y, width, height = destination
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            errors.append("invalid_destination_geometry")
        if x + width > scan.shape[1] or y + height > scan.shape[0]:
            errors.append("destination_out_of_bounds")
        allowed = np.zeros(scan.shape[:2], dtype=bool)
        if not errors or (
            x >= 0
            and y >= 0
            and width > 0
            and height > 0
            and x + width <= scan.shape[1]
            and y + height <= scan.shape[0]
        ):
            allowed[y : y + height, x : x + width] = True
    outside_change_count = int(np.logical_and(exact_change, ~allowed).sum())
    if outside_change_count:
        errors.append("changed_pixels_outside_destination")
    changed_pixels = int(exact_change.sum())
    if int(metadata.get("changed_pixels", -1)) != changed_pixels:
        errors.append("metadata_changed_pixels_mismatch")
    expected_fraction = changed_pixels / max(1, exact_change.size)
    if not math.isclose(
        float(metadata.get("changed_fraction", -1.0)),
        expected_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append("metadata_changed_fraction_mismatch")

    if attack_name == "copy_move":
        source = [int(value) for value in metadata.get("source_xywh", [])]
        if len(source) != 4:
            errors.append("invalid_source_xywh")
        elif not _rectangles_separated_xywh(source, destination, required_gap):
            errors.append("copy_source_destination_gap_violation")
        if len(destination) == 4:
            patch_area = destination[2] * destination[3]
            if changed_pixels / max(1, patch_area) < 0.25:
                errors.append("copy_changed_patch_fraction_below_0_25")
    elif attack_name == "local_erase":
        if changed_pixels < int(metadata.get("minimum_changed_pixels", 32)):
            errors.append("erase_changed_pixels_below_minimum")
    else:
        errors.append("unexpected_attack_name")

    scan_marker = _marker_green(scan, marker_settings)
    candidate_marker = _marker_green(candidate, marker_settings)
    introduced_marker = candidate_marker & ~scan_marker & exact_change
    scan_palette = np.unique(_encoded_colors(scan))
    candidate_colors = _encoded_colors(candidate)
    novel_marker = introduced_marker & ~np.isin(candidate_colors, scan_palette)
    exact_color_introductions: dict[str, int] = {}
    for color in marker_settings.get("exact_marker_rgb", []):
        color_array = np.asarray(color, dtype=np.uint8)
        scan_exact = np.all(scan == color_array, axis=2)
        candidate_exact = np.all(candidate == color_array, axis=2)
        exact_color_introductions[",".join(str(int(item)) for item in color)] = int(
            np.logical_and(candidate_exact, ~scan_exact).sum()
        )
    novel_marker_count = int(novel_marker.sum())
    if novel_marker_count:
        errors.append("novel_marker_green_color_introduced")

    return {
        "attack": attack_name,
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "candidate": str(attack["candidate"]),
        "candidate_sha256": _sha256(candidate_path),
        "mask": str(attack["mask"]),
        "mask_sha256": _sha256(mask_path),
        "mask_values": mask_values,
        "changed_pixels": changed_pixels,
        "changed_fraction": expected_fraction,
        "outside_destination_changed_pixels": outside_change_count,
        "introduced_marker_green_positions": int(introduced_marker.sum()),
        "novel_marker_green_pixels": novel_marker_count,
        "exact_marker_color_introductions": exact_color_introductions,
    }


def _audit_group(
    row: dict[str, Any], scratch: Path, audit: dict[str, Any]
) -> dict[str, Any]:
    group_id = str(row.get("source_group_id", "unknown"))
    basename = str(row.get("source_basename", "unknown"))
    errors: list[str] = []
    try:
        if row.get("status") == "failed":
            raise ValueError(str(row.get("failure_reason", "materialization_failed")))
        scan, _, _ = _verify_artifact(
            scratch, str(row["scan"]), str(row["scan_sha256"]), rgb=True
        )
        clean, _, _ = _verify_artifact(
            scratch, str(row["clean"]), str(row["clean_sha256"]), rgb=True
        )
        expected_height, expected_width = [
            int(value) for value in audit["expected_height_width"]
        ]
        if scan.shape != (expected_height, expected_width, 3):
            errors.append("unexpected_scan_geometry")
        if clean.shape != scan.shape:
            errors.append("scan_clean_dimension_mismatch")
        if int(row.get("height", -1)) != scan.shape[0] or int(
            row.get("width", -1)
        ) != scan.shape[1]:
            errors.append("manifest_geometry_mismatch")
        if row.get("model_score_read") is not False:
            errors.append("materialization_model_score_boundary_open")
        if row.get("paper_evidence") is not False:
            errors.append("materialization_paper_evidence_boundary_open")

        registration = _registration_diagnostics(
            scan, clean, audit["registration"]
        )
        attacks: dict[str, Any] = {}
        source_attacks = row.get("attacks", {})
        for attack_name in audit["expected_attacks"]:
            if attack_name not in source_attacks:
                attacks[attack_name] = {
                    "attack": attack_name,
                    "status": "failed",
                    "errors": ["missing_attack_record"],
                }
                continue
            try:
                attacks[attack_name] = _audit_attack(
                    scratch=scratch,
                    scan=scan,
                    attack_name=str(attack_name),
                    attack=source_attacks[attack_name],
                    marker_settings=audit["marker_green"],
                    required_gap=int(audit["copy_source_destination_gap_pixels"]),
                )
            except Exception as error:
                attacks[attack_name] = {
                    "attack": attack_name,
                    "status": "failed",
                    "errors": [f"{type(error).__name__}: {error}"],
                }
        unexpected_attacks = sorted(set(source_attacks) - set(audit["expected_attacks"]))
        if unexpected_attacks:
            errors.append(f"unexpected_attacks:{','.join(unexpected_attacks)}")
        for attack_name, attack_result in attacks.items():
            if attack_result["status"] != "ok":
                errors.append(f"attack_failed:{attack_name}")
        return {
            "source_group_id": group_id,
            "source_basename": basename,
            "status": "ok" if not errors else "failed",
            "errors": errors,
            "registration": registration,
            "attacks": attacks,
            "model_scoring_started": False,
            "paper_evidence": False,
        }
    except Exception as error:
        return {
            "source_group_id": group_id,
            "source_basename": basename,
            "status": "failed",
            "errors": [f"{type(error).__name__}: {error}"],
            "registration": None,
            "attacks": {},
            "model_scoring_started": False,
            "paper_evidence": False,
        }


def _referenced_storage_bytes(rows: list[dict[str, Any]], scratch: Path) -> int:
    paths: set[Path] = set()
    for row in rows:
        for field in ("scan", "clean"):
            if row.get(field):
                paths.add(_artifact_path(scratch, str(row[field])))
        for attack in row.get("attacks", {}).values():
            for field in ("candidate", "mask"):
                if attack.get(field):
                    paths.add(_artifact_path(scratch, str(attack[field])))
    return sum(path.stat().st_size for path in paths if path.is_file())


def _visual_review(
    project_root: Path,
    config: dict[str, Any],
    expected_group_ids: set[str],
) -> dict[str, Any]:
    review = config.get("visual_review")
    if not review or not bool(review.get("required")):
        return {
            "required": False,
            "status": "not_required",
            "reviewed_groups": 0,
            "passed_groups": 0,
        }
    path_value = review.get("records")
    expected_hash = review.get("expected_records_sha256")
    if not path_value or not expected_hash:
        return {
            "required": True,
            "status": "pending",
            "reviewed_groups": 0,
            "passed_groups": 0,
        }
    path = _resolve(project_root, str(path_value))
    if _sha256(path) != str(expected_hash):
        raise ValueError("visual-review record SHA-256 changed")
    records = _read_jsonl(path)
    by_group: dict[str, dict[str, Any]] = {}
    for record in records:
        group_id = str(record.get("source_group_id"))
        if group_id in by_group:
            raise ValueError(f"duplicate visual-review group: {group_id}")
        by_group[group_id] = record
    missing = sorted(expected_group_ids - set(by_group))
    extra = sorted(set(by_group) - expected_group_ids)
    passed = [
        group_id
        for group_id, record in by_group.items()
        if bool(record.get("visual_gate_passed"))
    ]
    status = "passed" if not missing and not extra and len(passed) == len(by_group) else "failed"
    return {
        "required": True,
        "status": status,
        "records": str(path.relative_to(project_root)),
        "records_sha256": _sha256(path),
        "reviewed_groups": len(by_group),
        "passed_groups": len(passed),
        "missing_groups": missing,
        "extra_groups": extra,
        "reviewer_types": sorted(
            {str(record.get("reviewer_type", "unspecified")) for record in records}
        ),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    if bool(runtime["model_scoring_authorized"]):
        raise ValueError("materialization audit cannot authorize model scoring")
    if not bool(runtime["verify_artifact_hashes"]):
        raise ValueError("DESCAN materialization audit requires hash verification")
    if bool(config["experiment"].get("paper_evidence")):
        raise ValueError("pre-scoring materialization audit is not paper evidence")

    for binding in config["bindings"]:
        path = _resolve(project_root, str(binding["path"]))
        if _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"bound audit artifact changed: {path}")

    paths = config["paths"]
    manifest_path = _resolve(project_root, str(paths["input_manifest"]))
    if _sha256(manifest_path) != str(paths["expected_input_manifest_sha256"]):
        raise ValueError("DESCAN materialization manifest SHA-256 changed")
    source_rows = _read_jsonl(manifest_path)
    scratch = _scratch_root(project_root, paths)
    records_path = _resolve(project_root, str(paths["audit_records"]))
    audited: list[dict[str, Any]] = []
    progress_every = max(1, int(runtime.get("progress_every", 1)))
    for index, row in enumerate(source_rows, 1):
        audited.append(_audit_group(row, scratch, config["audit"]))
        if index % progress_every == 0 or index == len(source_rows):
            _write_jsonl(records_path, audited)

    expected_groups = int(config["gates"]["expected_groups"])
    attack_names = [str(value) for value in config["audit"]["expected_attacks"]]
    attack_successes = {
        attack_name: sum(
            1
            for record in audited
            if record.get("attacks", {}).get(attack_name, {}).get("status") == "ok"
        )
        for attack_name in attack_names
    }
    registrations = [
        record["registration"]
        for record in audited
        if isinstance(record.get("registration"), dict)
    ]
    support_floor = float(config["gates"]["mutual_valid_support_min"])
    support_passes = sum(
        float(record["mutual_valid_support"]) >= support_floor
        for record in registrations
    )
    support_rate = support_passes / max(1, expected_groups)
    novel_marker_pixels = sum(
        int(attack.get("novel_marker_green_pixels", 0))
        for record in audited
        for attack in record.get("attacks", {}).values()
    )
    referenced_bytes = _referenced_storage_bytes(source_rows, scratch)
    estimated_full_bytes = math.ceil(
        referenced_bytes / max(1, len(source_rows))
        * int(config["gates"]["full_population_groups"])
    )
    visual = _visual_review(
        project_root,
        config,
        {str(row.get("source_group_id")) for row in source_rows},
    )
    checks = {
        "expected_group_count": len(source_rows) == expected_groups,
        "all_group_records_audited": len(audited) == expected_groups,
        "all_group_records_ok": all(record["status"] == "ok" for record in audited),
        "attack_success_floor": all(
            count >= int(config["gates"]["minimum_successes_per_attack"])
            for count in attack_successes.values()
        ),
        "registration_support_rate": support_rate
        >= float(config["gates"]["registration_support_rate_min"]),
        "no_novel_marker_green": novel_marker_pixels == 0,
        "full_storage_estimate": estimated_full_bytes
        < int(config["gates"]["full_storage_bytes_max_exclusive"]),
        "visual_review": visual["status"] in {"passed", "not_required"},
    }
    automatic_checks = {key: value for key, value in checks.items() if key != "visual_review"}
    summary = {
        "status": (
            f"descan18k_{config['experiment']['stage']}_audit_passed"
            if all(checks.values())
            else f"descan18k_{config['experiment']['stage']}_audit_failed"
        ),
        "paper_evidence": False,
        "stage": str(config["experiment"]["stage"]),
        "source_manifest": str(manifest_path.relative_to(project_root)),
        "source_manifest_sha256": _sha256(manifest_path),
        "records": {
            "groups": len(audited),
            "groups_ok": sum(record["status"] == "ok" for record in audited),
            "groups_failed": sum(record["status"] != "ok" for record in audited),
            "attack_successes": attack_successes,
            "novel_marker_green_pixels": novel_marker_pixels,
        },
        "registration": {
            "diagnostics_completed": len(registrations),
            "ecc_converged": sum(record["ecc_converged"] for record in registrations),
            "mutual_valid_support_minimum": (
                min(float(record["mutual_valid_support"]) for record in registrations)
                if registrations
                else None
            ),
            "support_floor": support_floor,
            "support_passes": support_passes,
            "support_rate_denominator_expected_groups": expected_groups,
            "support_rate": support_rate,
        },
        "storage": {
            "selected_referenced_bytes": referenced_bytes,
            "estimated_full_materialized_bytes": estimated_full_bytes,
            "full_storage_bytes_max_exclusive": int(
                config["gates"]["full_storage_bytes_max_exclusive"]
            ),
            "archive_excluded_from_estimate": True,
        },
        "visual_review": visual,
        "decision": {
            "automatic_gate_passed": all(automatic_checks.values()),
            "stage_gate_passed": all(checks.values()),
            "checks": checks,
            "model_scoring_authorized": False,
            "full_expansion_authorized_by_this_gate": all(checks.values())
            and str(config["experiment"]["stage"]) == "pilot20",
        },
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "config_sha256": _sha256(config_path),
    }
    summary_path = _resolve(project_root, str(paths["summary"]))
    summary["audit_records"] = str(records_path.relative_to(project_root))
    summary["audit_records_sha256"] = _sha256(records_path)
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
