from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _invalid_thread_value = int(os.environ.get(_thread_variable, "1")) < 1
    except ValueError:
        _invalid_thread_value = True
    if _invalid_thread_value:
        os.environ[_thread_variable] = "1"

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
    _aggregate_condition,
    _infer_pair_tiled,
    _raw_difference,
    _resize_image,
    _resize_reference,
)
from pairtrace_doc.pipelines.train_pairtrace_100 import _load_teacher
from pairtrace_doc.pipelines.train_student_100 import (
    ResNet18UNet,
    _infer_tiled,
    _ranking_metrics,
    _read_jsonl,
    _resolve,
    _sha256,
    _threshold_vectors,
    _write_csv,
    _write_json,
    _write_jsonl,
)


STRESSES = ("translation", "affine", "perspective")


def _required_conditions() -> set[str]:
    conditions = {
        "student_clean",
        "raw_difference_clean_unaligned",
        "raw_difference_clean_ecc",
        "pair_teacher_clean_unaligned",
        "pair_teacher_clean_ecc",
    }
    for stress in STRESSES:
        for scorer in ("raw_difference", "pair_teacher"):
            for alignment in ("unaligned", "oracle", "ecc"):
                conditions.add(f"{scorer}_{stress}_{alignment}")
    return conditions


def _stress_homography(
    shape: tuple[int, int],
    stress_name: str,
    stresses: dict[str, dict[str, Any]],
) -> np.ndarray:
    height, width = shape
    if stress_name == "clean":
        return np.eye(3, dtype=np.float64)
    specification = stresses[stress_name]
    if stress_name == "translation":
        return np.asarray(
            [
                [1.0, 0.0, float(specification["dx_pixels"])],
                [0.0, 1.0, float(specification["dy_pixels"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if stress_name == "affine":
        matrix = cv2.getRotationMatrix2D(
            ((width - 1) / 2.0, (height - 1) / 2.0),
            float(specification["rotation_degrees"]),
            float(specification["scale"]),
        ).astype(np.float64)
        matrix[0, 2] += float(specification["dx_pixels"])
        matrix[1, 2] += float(specification["dy_pixels"])
        return np.vstack([matrix, [0.0, 0.0, 1.0]])
    if stress_name == "perspective":
        source = np.asarray(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        destination = np.asarray(
            [
                [x * (width - 1.0), y * (height - 1.0)]
                for x, y in specification["destination_corners_normalized"]
            ],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(source, destination).astype(np.float64)
    raise ValueError(f"unsupported alignment stress: {stress_name}")


def _warp_reference(
    reference: np.ndarray,
    homography: np.ndarray,
    inverse: bool,
) -> np.ndarray:
    height, width = reference.shape[:2]
    flags = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if inverse else 0)
    return cv2.warpPerspective(
        reference,
        homography,
        (width, height),
        flags=flags,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _transform_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (homography.astype(np.float64) @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def _corner_errors(
    estimated: np.ndarray,
    oracle: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    corners = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )
    return np.linalg.norm(
        _transform_points(estimated, corners) - _transform_points(oracle, corners),
        axis=1,
    )


def _resize_for_registration(
    image: np.ndarray, max_side: int
) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image, 1.0, 1.0
    scale = max_side / max(height, width)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, target_width / width, target_height / height


def _estimate_ecc_alignment(
    candidate: np.ndarray,
    stressed_reference: np.ndarray,
    oracle_homography: np.ndarray,
    registration: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate.shape != stressed_reference.shape:
        raise ValueError("ECC alignment requires matched model-space geometry")
    small_candidate, scale_x, scale_y = _resize_for_registration(
        candidate, int(registration["max_side"])
    )
    small_reference = cv2.resize(
        stressed_reference,
        (small_candidate.shape[1], small_candidate.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    template = cv2.cvtColor(small_candidate, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    input_image = cv2.cvtColor(small_reference, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    status = "ecc_converged"
    failure_type = None
    failure_reason = None
    correlation = math.nan
    phase_response = math.nan
    small_warp = np.eye(3, dtype=np.float64)
    try:
        phase_shift, phase_response = cv2.phaseCorrelate(template, input_image)
        if not np.isfinite(phase_shift).all() or not np.isfinite(phase_response):
            raise ValueError("phase correlation returned non-finite initialization")
        small_warp[0, 2] = float(phase_shift[0])
        small_warp[1, 2] = float(phase_shift[1])
        correlation, small_warp = cv2.findTransformECC(
            template,
            input_image,
            small_warp.astype(np.float32),
            cv2.MOTION_HOMOGRAPHY,
            (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(registration["iterations"]),
                float(registration["epsilon"]),
            ),
            None,
            int(registration["gauss_filter_size"]),
        )
        small_warp = small_warp.astype(np.float64)
        if not np.isfinite(small_warp).all() or not np.isfinite(correlation):
            raise ValueError("ECC returned non-finite alignment output")
        if abs(float(small_warp[2, 2])) < 1e-12:
            raise ValueError("ECC returned a singular homography normalization")
        small_warp /= small_warp[2, 2]
        scale = np.asarray(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        warp = np.linalg.inv(scale) @ small_warp @ scale
        warp /= warp[2, 2]
    except Exception as error:
        status = (
            "phase_correlation_fallback_recorded"
            if np.isfinite(small_warp).all() and not np.allclose(small_warp, np.eye(3))
            else "identity_fallback_recorded"
        )
        failure_type = type(error).__name__
        failure_reason = str(error)
        scale = np.asarray(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        warp = np.linalg.inv(scale) @ small_warp @ scale
    aligned = _warp_reference(stressed_reference, warp, inverse=True)
    errors = _corner_errors(warp, oracle_homography, candidate.shape[:2])
    metadata = {
        "alignment_status": status,
        "ecc_correlation": float(correlation),
        "phase_correlation_response": float(phase_response),
        "estimated_homography": warp.tolist(),
        "corner_errors_pixels": errors.tolist(),
        "corner_error_mean_pixels": float(errors.mean()),
        "corner_error_max_pixels": float(errors.max()),
        "alignment_failure_type": failure_type,
        "alignment_failure_reason": failure_reason,
    }
    return aligned, metadata


def _alignment_decision(
    metrics: dict[str, dict[str, Any]],
    alignment_summary: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    effects: dict[str, float] = {}
    checks = {
        "registration_convergence_floor": float(alignment_summary["convergence_rate"])
        >= float(gate["convergence_rate_min"]),
        "corner_error_median_ceiling": float(
            alignment_summary["controlled_stress_corner_error_median_pixels"]
        )
        <= float(gate["corner_error_median_pixels_max"]),
        "corner_error_p95_ceiling": float(
            alignment_summary["controlled_stress_corner_error_p95_pixels"]
        )
        <= float(gate["corner_error_p95_pixels_max"]),
    }
    clean_unaligned = metrics["pair_teacher_clean_unaligned"]
    clean_ecc = metrics["pair_teacher_clean_ecc"]
    clean_drop = float(
        clean_unaligned["generator_macro_pixel_ap"]
        - clean_ecc["generator_macro_pixel_ap"]
    )
    effects["pair_teacher_clean_unaligned_minus_ecc"] = clean_drop
    checks["clean_ecc_ap_drop_ceiling"] = clean_drop <= float(
        gate["clean_ecc_ap_drop_max"]
    )
    checks["clean_ecc_authentic_fpr_ceiling"] = float(
        clean_ecc["authentic_pixel_fpr"]
    ) <= float(gate["authentic_pixel_fpr_max"])
    for stress in STRESSES:
        unaligned = metrics[f"pair_teacher_{stress}_unaligned"]
        oracle = metrics[f"pair_teacher_{stress}_oracle"]
        ecc = metrics[f"pair_teacher_{stress}_ecc"]
        effects[f"pair_teacher_{stress}_ecc_minus_unaligned"] = float(
            ecc["generator_macro_pixel_ap"] - unaligned["generator_macro_pixel_ap"]
        )
        oracle_gap = float(
            oracle["generator_macro_pixel_ap"] - ecc["generator_macro_pixel_ap"]
        )
        effects[f"pair_teacher_{stress}_oracle_minus_ecc"] = oracle_gap
        checks[f"{stress}_ecc_ap_floor"] = float(
            ecc["generator_macro_pixel_ap"]
        ) >= float(gate["ecc_pair_generator_macro_ap_min"])
        checks[f"{stress}_oracle_minus_ecc_ceiling"] = oracle_gap <= float(
            gate["oracle_minus_ecc_ap_gap_max"]
        )
        checks[f"{stress}_ecc_authentic_fpr_ceiling"] = float(
            ecc["authentic_pixel_fpr"]
        ) <= float(gate["authentic_pixel_fpr_max"])
    return {"effects": effects, "checks": checks, "overall_pass": all(checks.values())}


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not runtime["gpu_launch_authorized"] or not runtime["alignment_diagnostic_authorized"]:
        raise ValueError("alignment diagnostic was not explicitly authorized")
    if not runtime["viewed_method_development_read_allowed"]:
        raise ValueError("viewed method-development read was not authorized")
    if any(
        bool(runtime.get(name))
        for name in (
            "method_training_authorized",
            "multi_seed_authorized",
            "unseen_development_read_allowed",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("alignment diagnostic crossed an evidence boundary")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("viewed alignment diagnostic cannot be paper evidence")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("alignment diagnostic requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("alignment diagnostic protocol SHA-256 changed")
    input_config = config["input"]
    manifest_path = _resolve(project_root, input_config["manifest"])
    if _sha256(manifest_path) != input_config["expected_manifest_sha256"]:
        raise ValueError("alignment diagnostic manifest SHA-256 changed")
    rows = sorted(_read_jsonl(manifest_path), key=lambda row: str(row["source_group_id"]))
    if len(rows) != int(input_config["expected_groups"]):
        raise ValueError("alignment diagnostic group count changed")
    if len({str(row["source_group_id"]) for row in rows}) != len(rows):
        raise ValueError("alignment diagnostic contains duplicate groups")
    if {str(row["alignment_diagnostic_freeze_id"]) for row in rows} != {
        str(input_config["expected_freeze_id"])
    }:
        raise ValueError("alignment diagnostic freeze ID changed")
    counts = Counter(str(row["selected_generator"]) for row in rows)
    expected_counts = {
        str(name): int(value)
        for name, value in input_config["expected_generator_counts"].items()
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"alignment diagnostic generator counts changed: {dict(counts)}")
    max_groups = runtime.get("max_groups")
    if max_groups is not None:
        rows = rows[: int(max_groups)]

    condition_specs = {str(item["name"]): item for item in config["conditions"]}
    if set(condition_specs) != _required_conditions():
        raise ValueError("alignment diagnostic condition whitelist changed")
    stress_specs = {str(item["name"]): item for item in config["stresses"]}
    if set(stress_specs) != set(STRESSES):
        raise ValueError("alignment diagnostic stress whitelist changed")

    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    models = config["models"]
    student_path = _resolve(project_root, models["student_checkpoint"])
    teacher_path = _resolve(project_root, models["teacher_checkpoint"])
    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    encoder_path = _resolve(scratch, models["encoder_weights"])
    for path, expected, label in (
        (student_path, models["student_checkpoint_sha256"], "student checkpoint"),
        (teacher_path, models["teacher_checkpoint_sha256"], "teacher checkpoint"),
        (encoder_path, models["encoder_weights_sha256"], "encoder weights"),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen {label} changed")
    student_saved = torch.load(student_path, map_location="cpu", weights_only=True)
    student = ResNet18UNet()
    student.load_state_dict(student_saved["model_state"], strict=True)
    student = student.to(device).eval().requires_grad_(False)
    teacher = _load_teacher(encoder_path, models["teacher_conv1_coefficients"])
    teacher_saved = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher.load_state_dict(teacher_saved["model_state"], strict=True)
    teacher = teacher.to(device).eval().requires_grad_(False)

    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    alignment_cache_dir = _resolve(scratch, paths["alignment_cache_dir"])
    predictions_path = _resolve(project_root, paths["predictions"])
    metrics_path = _resolve(project_root, paths["metrics"])
    comparisons_path = _resolve(project_root, paths["comparisons"])
    alignments_path = _resolve(project_root, paths["alignments"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (
        score_cache_dir,
        alignment_cache_dir,
        predictions_path.parent,
        metrics_path.parent,
        comparisons_path.parent,
        alignments_path.parent,
        summary_path.parent,
        log_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    inference = config["inference"]
    preprocessing = config["preprocessing"]
    registration = config["registration"]
    thresholds = np.arange(
        float(config["operating_point"]["candidate_min"]),
        float(config["operating_point"]["candidate_max"]) + 1e-12,
        float(config["operating_point"]["candidate_step"]),
    )
    image_cache: dict[str, np.ndarray] = {}

    def load_image(row: dict[str, Any], path_field: str, sha_field: str) -> np.ndarray:
        path = _resolve(scratch, row[path_field])
        key = str(path)
        if key not in image_cache:
            if _sha256(path) != row[sha_field]:
                raise ValueError(f"{path_field} SHA-256 changed")
            with Image.open(path) as handle:
                image_cache[key] = np.asarray(handle.convert("RGB"))
        return image_cache[key]

    payloads: dict[str, dict[str, Any]] = {
        name: {
            "forged": [],
            "authentic_vectors": [],
            "authentic_fpr_max": config["operating_point"]["authentic_pixel_fpr_max"],
        }
        for name in condition_specs
    }
    prediction_rows: list[dict[str, Any]] = []
    alignment_records: dict[str, dict[str, Any]] = {}
    failures = 0
    score_cache_hits = 0
    alignment_cache_hits = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    for row_index, row in enumerate(rows, start=1):
        group = str(row["source_group_id"])
        generator = str(row["selected_generator"])
        forged_native = load_image(row, "image", "image_sha256")
        authentic_native = load_image(row, "authentic", "authentic_sha256")
        mask_path = _resolve(scratch, row["mask"])
        if _sha256(mask_path) != row["mask_sha256"]:
            raise ValueError("alignment diagnostic mask SHA-256 changed")
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0
        if forged_native.shape[:2] != native_mask.shape or authentic_native.shape[:2] != native_mask.shape:
            raise ValueError("alignment diagnostic native geometry changed")

        for condition_name, condition in condition_specs.items():
            for sample_kind, candidate_native in (("forged", forged_native), ("authentic", authentic_native)):
                prediction: dict[str, Any] = {
                    "record_id": f"{condition_name}:{sample_kind}:{group}",
                    "source_group_id": group,
                    "generator": generator,
                    "condition": condition_name,
                    "sample_kind": sample_kind,
                    "status": "failed",
                    "paper_evidence": False,
                    "viewed_method_development": True,
                    "unseen_development_read": False,
                    "final_reserve_read": False,
                    "threshold_selection_used": True,
                }
                try:
                    candidate = _resize_image(candidate_native, int(preprocessing["max_side"]))
                    clean_reference = _resize_reference(authentic_native, candidate.shape[:2])
                    stress_name = str(condition["stress"])
                    alignment = str(condition["alignment"])
                    scorer = str(condition["scorer"])
                    oracle_homography = _stress_homography(
                        candidate.shape[:2], stress_name, stress_specs
                    )
                    stressed_reference = _warp_reference(
                        clean_reference, oracle_homography, inverse=False
                    )
                    alignment_metadata: dict[str, Any] = {
                        "alignment_status": "not_requested",
                        "ecc_correlation": None,
                        "phase_correlation_response": None,
                        "estimated_homography": None,
                        "corner_errors_pixels": None,
                        "corner_error_mean_pixels": None,
                        "corner_error_max_pixels": None,
                        "alignment_failure_type": None,
                        "alignment_failure_reason": None,
                    }
                    alignment_cache_relative = None
                    if alignment == "unaligned":
                        reference = stressed_reference
                    elif alignment == "oracle":
                        reference = _warp_reference(
                            stressed_reference, oracle_homography, inverse=True
                        )
                        errors = _corner_errors(
                            oracle_homography,
                            oracle_homography,
                            candidate.shape[:2],
                        )
                        alignment_metadata.update(
                            {
                                "alignment_status": "oracle",
                                "estimated_homography": oracle_homography.tolist(),
                                "corner_errors_pixels": errors.tolist(),
                                "corner_error_mean_pixels": float(errors.mean()),
                                "corner_error_max_pixels": float(errors.max()),
                            }
                        )
                    elif alignment == "ecc":
                        candidate_sha256 = str(
                            row["image_sha256"]
                            if sample_kind == "forged"
                            else row["authentic_sha256"]
                        )
                        alignment_key = hashlib.sha256(
                            json.dumps(
                                {
                                    "candidate_sha256": candidate_sha256,
                                    "reference_sha256": row["authentic_sha256"],
                                    "candidate_shape": list(candidate.shape),
                                    "stress": stress_specs.get(stress_name, {"name": "clean"}),
                                    "registration": registration,
                                    "schema_version": preprocessing["alignment_cache_schema_version"],
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        alignment_path = alignment_cache_dir / stress_name / f"{alignment_key}.npz"
                        alignment_path.parent.mkdir(parents=True, exist_ok=True)
                        if alignment_path.is_file():
                            with np.load(alignment_path, allow_pickle=False) as archive:
                                reference = archive["aligned_reference"].astype(np.uint8)
                                estimated = archive["estimated_homography"].astype(np.float64)
                                corner_errors = archive["corner_errors_pixels"].astype(np.float64)
                                correlation_value = float(archive["ecc_correlation"].item())
                                phase_response_value = float(
                                    archive["phase_correlation_response"].item()
                                )
                                status_value = str(archive["alignment_status"].item())
                                failure_type_value = str(archive["alignment_failure_type"].item()) or None
                                failure_reason_value = str(archive["alignment_failure_reason"].item()) or None
                            alignment_metadata = {
                                "alignment_status": status_value,
                                "ecc_correlation": correlation_value if np.isfinite(correlation_value) else None,
                                "phase_correlation_response": (
                                    phase_response_value
                                    if np.isfinite(phase_response_value)
                                    else None
                                ),
                                "estimated_homography": estimated.tolist(),
                                "corner_errors_pixels": corner_errors.tolist(),
                                "corner_error_mean_pixels": float(corner_errors.mean()),
                                "corner_error_max_pixels": float(corner_errors.max()),
                                "alignment_failure_type": failure_type_value,
                                "alignment_failure_reason": failure_reason_value,
                            }
                            alignment_cache_hits += 1
                        else:
                            reference, alignment_metadata = _estimate_ecc_alignment(
                                candidate,
                                stressed_reference,
                                oracle_homography,
                                registration,
                            )
                            temporary = alignment_path.with_suffix(".npz.tmp")
                            with temporary.open("wb") as handle:
                                np.savez_compressed(
                                    handle,
                                    aligned_reference=reference.astype(np.uint8),
                                    estimated_homography=np.asarray(
                                        alignment_metadata["estimated_homography"], dtype=np.float64
                                    ),
                                    corner_errors_pixels=np.asarray(
                                        alignment_metadata["corner_errors_pixels"], dtype=np.float64
                                    ),
                                    ecc_correlation=np.asarray(
                                        math.nan
                                        if alignment_metadata["ecc_correlation"] is None
                                        else alignment_metadata["ecc_correlation"],
                                        dtype=np.float64,
                                    ),
                                    phase_correlation_response=np.asarray(
                                        math.nan
                                        if alignment_metadata["phase_correlation_response"] is None
                                        else alignment_metadata["phase_correlation_response"],
                                        dtype=np.float64,
                                    ),
                                    alignment_status=np.asarray(
                                        alignment_metadata["alignment_status"]
                                    ),
                                    alignment_failure_type=np.asarray(
                                        alignment_metadata["alignment_failure_type"] or ""
                                    ),
                                    alignment_failure_reason=np.asarray(
                                        alignment_metadata["alignment_failure_reason"] or ""
                                    ),
                                )
                            temporary.replace(alignment_path)
                        if reference.shape != candidate.shape or not np.isfinite(reference).all():
                            raise ValueError("alignment cache is invalid")
                        alignment_cache_relative = str(alignment_path.relative_to(scratch))
                        alignment_records.setdefault(
                            alignment_key,
                            {
                                "alignment_key": alignment_key,
                                "source_group_id": group,
                                "generator": generator,
                                "sample_kind": sample_kind,
                                "stress": stress_name,
                                "alignment_cache": alignment_cache_relative,
                                **alignment_metadata,
                            },
                        )
                    else:
                        raise ValueError(f"unsupported alignment mode: {alignment}")

                    model_identity = (
                        models["student_checkpoint_sha256"]
                        if scorer == "student"
                        else models["teacher_checkpoint_sha256"]
                        if scorer == "pair_teacher"
                        else "raw_difference_v1"
                    )
                    candidate_sha256 = str(
                        row["image_sha256"]
                        if sample_kind == "forged"
                        else row["authentic_sha256"]
                    )
                    score_key = hashlib.sha256(
                        json.dumps(
                            {
                                "candidate_sha256": candidate_sha256,
                                "reference_sha256": row["authentic_sha256"],
                                "condition": condition,
                                "stress": stress_specs.get(stress_name, {"name": "clean"}),
                                "alignment_metadata": alignment_metadata,
                                "model_identity": model_identity,
                                "preprocessing": preprocessing,
                                "inference": inference,
                                "sample_kind": sample_kind,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    score_path = score_cache_dir / condition_name / f"{score_key}.npz"
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    if score_path.is_file():
                        with np.load(score_path, allow_pickle=False) as archive:
                            probability = archive["scores"].astype(np.float32)
                        score_cache_hits += 1
                    else:
                        if scorer == "student":
                            probability = _infer_tiled(
                                student, candidate, device, inference, preprocessing
                            )
                        elif scorer == "pair_teacher":
                            probability = _infer_pair_tiled(
                                teacher,
                                candidate,
                                reference,
                                device,
                                inference,
                                preprocessing,
                            )
                        elif scorer == "raw_difference":
                            probability = _raw_difference(candidate, reference)
                        else:
                            raise ValueError(f"unsupported scorer: {scorer}")
                        temporary = score_path.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(handle, scores=probability.astype(np.float16))
                        temporary.replace(score_path)
                    if probability.shape != candidate.shape[:2] or not np.isfinite(probability).all():
                        raise ValueError("alignment diagnostic score cache is invalid")
                    native_probability = cv2.resize(
                        probability,
                        (native_mask.shape[1], native_mask.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    if sample_kind == "forged":
                        average_precision, auroc = _ranking_metrics(
                            native_probability, native_mask
                        )
                        vectors = _threshold_vectors(
                            native_probability, native_mask, thresholds
                        )
                        payloads[condition_name]["forged"].append(
                            {
                                "source_group_id": group,
                                "generator": generator,
                                "macro_pixel_ap": average_precision,
                                "pixel_auroc": auroc,
                                "threshold_vectors": vectors,
                            }
                        )
                        prediction.update(
                            {"macro_pixel_ap": average_precision, "pixel_auroc": auroc}
                        )
                    else:
                        histogram, _ = np.histogram(
                            native_probability, bins=np.r_[thresholds, np.inf]
                        )
                        payloads[condition_name]["authentic_vectors"].append(
                            np.cumsum(histogram[::-1], dtype=np.int64)[::-1]
                            / native_probability.size
                        )
                    prediction.update(
                        {
                            "status": "ok",
                            "score_cache": str(score_path.relative_to(scratch)),
                            "alignment_cache": alignment_cache_relative,
                            "score_shape": list(probability.shape),
                            "native_shape": list(native_probability.shape),
                            "scorer": scorer,
                            "stress": stress_name,
                            "alignment": alignment,
                            "model_identity": model_identity,
                            **alignment_metadata,
                        }
                    )
                except Exception as error:
                    failures += 1
                    prediction["failure_type"] = type(error).__name__
                    prediction["failure_reason"] = str(error)
                    logging.exception("record_id=%s failed", prediction["record_id"])
                prediction_rows.append(prediction)
        _write_jsonl(predictions_path, prediction_rows)
        _write_jsonl(alignments_path, list(alignment_records.values()))
        logging.info("completed_groups=%d total_groups=%d", row_index, len(rows))

    complete = failures == 0 and all(
        len(payload["forged"]) == len(rows)
        and len(payload["authentic_vectors"]) == len(rows)
        for payload in payloads.values()
    )
    if not complete:
        summary = {
            "experiment": config["experiment"],
            "status": "failed_incomplete",
            "paper_evidence": False,
            "viewed_method_development_read": True,
            "unseen_development_read": False,
            "final_reserve_read": False,
            "successful_prediction_records": len(prediction_rows) - failures,
            "failed_prediction_records": failures,
            "outputs": {
                "predictions": str(predictions_path.relative_to(project_root)),
                "predictions_sha256": _sha256(predictions_path),
                "alignments": str(alignments_path.relative_to(project_root)),
                "alignments_sha256": _sha256(alignments_path),
            },
        }
        _write_json(summary_path, summary)
        if runtime["require_all_records"]:
            raise RuntimeError(f"alignment diagnostic failed for {failures} records")
        return summary

    metrics = {
        name: _aggregate_condition(payload, thresholds)
        for name, payload in payloads.items()
    }
    alignment_rows = list(alignment_records.values())
    successful_alignments = [
        row for row in alignment_rows if row["alignment_status"] == "ecc_converged"
    ]
    controlled_errors = [
        float(error)
        for row in alignment_rows
        if row["stress"] in STRESSES
        for error in row["corner_errors_pixels"]
    ]
    correlations = [
        float(row["ecc_correlation"])
        for row in successful_alignments
        if row["ecc_correlation"] is not None
    ]
    if not alignment_rows or not controlled_errors or not correlations:
        raise RuntimeError("alignment diagnostic registration summary is empty")
    alignment_summary: dict[str, Any] = {
        "attempts": len(alignment_rows),
        "converged": len(successful_alignments),
        "identity_fallbacks": len(alignment_rows) - len(successful_alignments),
        "convergence_rate": len(successful_alignments) / len(alignment_rows),
        "controlled_stress_corner_error_median_pixels": float(
            np.median(controlled_errors)
        ),
        "controlled_stress_corner_error_p95_pixels": float(
            np.quantile(controlled_errors, 0.95)
        ),
        "ecc_correlation_median": float(np.median(correlations)),
        "ecc_correlation_min": float(np.min(correlations)),
        "by_stress": {},
    }
    for stress_name in ("clean", *STRESSES):
        subset = [row for row in alignment_rows if row["stress"] == stress_name]
        subset_success = [
            row for row in subset if row["alignment_status"] == "ecc_converged"
        ]
        errors = [
            float(error) for row in subset for error in row["corner_errors_pixels"]
        ]
        values = [
            float(row["ecc_correlation"])
            for row in subset_success
            if row["ecc_correlation"] is not None
        ]
        alignment_summary["by_stress"][stress_name] = {
            "attempts": len(subset),
            "converged": len(subset_success),
            "convergence_rate": len(subset_success) / len(subset),
            "corner_error_median_pixels": float(np.median(errors)),
            "corner_error_p95_pixels": float(np.quantile(errors, 0.95)),
            "ecc_correlation_median": float(np.median(values)) if values else None,
            "ecc_correlation_min": float(np.min(values)) if values else None,
        }
    decision = _alignment_decision(
        metrics, alignment_summary, config["alignment_gate"]
    )
    condition_rows = [{"condition": name, **values} for name, values in metrics.items()]
    comparison_rows = [
        {
            "comparison": name,
            "generator_macro_pixel_ap_difference": value,
            "viewed_method_development": True,
            "paper_evidence": False,
        }
        for name, value in decision["effects"].items()
    ]
    _write_csv(metrics_path, condition_rows)
    _write_csv(comparisons_path, comparison_rows)
    output = {
        "experiment": config["experiment"],
        "status": (
            "alignment_diagnostic_ecc_adequate"
            if decision["overall_pass"]
            else "alignment_diagnostic_ecc_escalation_required"
        ),
        "paper_evidence": False,
        "viewed_method_development_read": True,
        "unseen_development_read": False,
        "final_reserve_read": False,
        "method_training_performed": False,
        "ecc_front_end_freeze_authorized": bool(decision["overall_pass"]),
        "second_unseen_development_freeze_authorized": bool(
            decision["overall_pass"]
        ),
        "dense_matcher_protocol_amendment_required": not bool(
            decision["overall_pass"]
        ),
        "multi_seed_authorized": False,
        "input_manifest_sha256": _sha256(manifest_path),
        "protocol_sha256": _sha256(protocol_path),
        "selected_groups": len(rows),
        "successful_prediction_records": len(prediction_rows),
        "failed_prediction_records": 0,
        "score_cache_hits": score_cache_hits,
        "alignment_cache_hits": alignment_cache_hits,
        "conditions": metrics,
        "registration": alignment_summary,
        "alignment_gate": config["alignment_gate"],
        "decision": decision,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
            "alignments": str(alignments_path.relative_to(project_root)),
            "alignments_sha256": _sha256(alignments_path),
            "score_cache_dir": str(score_cache_dir.relative_to(scratch)),
            "alignment_cache_dir": str(alignment_cache_dir.relative_to(scratch)),
            "log": str(log_path.relative_to(project_root)),
        },
    }
    _write_json(summary_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
