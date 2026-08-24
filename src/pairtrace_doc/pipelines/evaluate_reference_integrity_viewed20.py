from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
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
    _infer_pair_tiled,
    _resize_image,
)
from pairtrace_doc.pipelines.train_pairtrace_100 import _load_teacher
from pairtrace_doc.pipelines.train_student_100 import (
    _ranking_metrics,
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)


def _deterministic_global_map(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError("global wrong-reference control requires two groups")
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"{seed}|{row['source_group_id']}".encode("utf-8")
            ).hexdigest(),
            str(row["source_group_id"]),
        ),
    )
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(ordered):
        target = ordered[(index + 1) % len(ordered)]
        if str(target["source_group_id"]) == str(row["source_group_id"]):
            raise ValueError("global wrong-reference mapping retained a group")
        result[str(row["source_group_id"])] = target
    return result


def _reference_size_distance(
    source: dict[str, Any], target: dict[str, Any]
) -> float:
    source_height = float(source["authentic_height"])
    source_width = float(source["authentic_width"])
    target_height = float(target["authentic_height"])
    target_width = float(target["authentic_width"])
    area_distance = abs(
        math.log((source_height * source_width) / (target_height * target_width))
    )
    aspect_distance = abs(
        math.log((source_width / source_height) / (target_width / target_height))
    )
    return area_distance + aspect_distance


def _same_dataset_size_map(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        source_group = str(source["source_group_id"])
        candidates = [
            target
            for target in rows
            if str(target["source_group_id"]) != source_group
            and str(target["source_dataset"]) == str(source["source_dataset"])
        ]
        if not candidates:
            raise ValueError(
                f"same-dataset wrong-reference pool is empty for {source_group}"
            )
        result[source_group] = min(
            candidates,
            key=lambda target: (
                _reference_size_distance(source, target),
                hashlib.sha256(
                    f"{seed}|{source_group}|{target['source_group_id']}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                str(target["source_group_id"]),
            ),
        )
    return result


def _center_crop_resized(image: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("retained linear fraction must be in (0, 1]")
    if math.isclose(fraction, 1.0, abs_tol=1e-12):
        return image.copy()
    height, width = image.shape[:2]
    crop_height = max(2, min(height, round(height * fraction)))
    crop_width = max(2, min(width, round(width * fraction)))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    crop = image[top : top + crop_height, left : left + crop_width]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


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


def _align_reference(
    candidate: np.ndarray,
    reference: np.ndarray,
    registration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if candidate.shape != reference.shape:
        raise ValueError("reference integrity alignment requires equal canvases")
    small_candidate, scale_x, scale_y = _resize_for_registration(
        candidate, int(registration["max_side"])
    )
    small_reference = cv2.resize(
        reference,
        (small_candidate.shape[1], small_candidate.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    template = (
        cv2.cvtColor(small_candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
        / 255.0
    )
    input_image = (
        cv2.cvtColor(small_reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
        / 255.0
    )
    status = "ecc_converged"
    failure_type = None
    failure_reason = None
    correlation = math.nan
    phase_response = math.nan
    small_warp = np.eye(3, dtype=np.float64)
    try:
        phase_shift, phase_response = cv2.phaseCorrelate(template, input_image)
        if not np.isfinite(phase_shift).all() or not np.isfinite(phase_response):
            raise ValueError("phase correlation returned a non-finite initializer")
        small_warp[0, 2] = float(phase_shift[0])
        small_warp[1, 2] = float(phase_shift[1])
        correlation, fitted = cv2.findTransformECC(
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
        small_warp = fitted.astype(np.float64)
        if not np.isfinite(small_warp).all() or not np.isfinite(correlation):
            raise ValueError("ECC returned a non-finite result")
        if abs(float(small_warp[2, 2])) < 1e-12:
            raise ValueError("ECC returned a singular normalization")
        small_warp /= small_warp[2, 2]
    except Exception as error:
        status = "phase_initializer_fallback_recorded"
        failure_type = type(error).__name__
        failure_reason = str(error)

    scale = np.asarray(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    warp = np.linalg.inv(scale) @ small_warp @ scale
    if abs(float(warp[2, 2])) > 1e-12:
        warp /= warp[2, 2]
    height, width = candidate.shape[:2]
    aligned = cv2.warpPerspective(
        reference,
        warp.astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    validity = cv2.warpPerspective(
        np.ones((height, width), dtype=np.uint8),
        warp.astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    metadata = {
        "alignment_status": status,
        "ecc_correlation": float(correlation),
        "phase_correlation_response": float(phase_response),
        "estimated_homography": warp.tolist(),
        "valid_area_fraction": float(validity.mean()),
        "alignment_failure_type": failure_type,
        "alignment_failure_reason": failure_reason,
    }
    return aligned, validity, metadata


def _threshold_metrics(
    probability: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = probability >= threshold
    target = target.astype(bool, copy=False)
    true_positive = int(np.logical_and(predicted, target).sum())
    false_positive = int(np.logical_and(predicted, ~target).sum())
    false_negative = int(np.logical_and(~predicted, target).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    iou = true_positive / max(1, true_positive + false_positive + false_negative)
    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pixel_iou": float(iou),
    }


def _binary_auroc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return math.nan
    positives = np.nan_to_num(np.asarray(positive), nan=-1.0)
    negatives = np.nan_to_num(np.asarray(negative), nan=-1.0)
    wins = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((wins + 0.5 * ties) / (positives.size * negatives.size))


def _aggregate_metrics(
    prediction_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    model_names: list[str],
    condition_names: list[str],
) -> list[dict[str, Any]]:
    alignments_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in alignment_rows:
        alignments_by_condition[str(row["condition"])].append(row)
    metrics: list[dict[str, Any]] = []
    for model_name in model_names:
        for condition_name in condition_names:
            selected = [
                row
                for row in prediction_rows
                if row["model"] == model_name
                and row["condition"] == condition_name
                and row["status"] == "ok"
            ]
            forged = [row for row in selected if row["sample_kind"] == "forged"]
            authentic = [
                row for row in selected if row["sample_kind"] == "authentic"
            ]
            generator_values: dict[str, list[float]] = defaultdict(list)
            for row in forged:
                generator_values[str(row["generator"])].append(
                    float(row["full_document_pixel_ap"])
                )
            generator_macro = float(
                np.mean([np.mean(values) for values in generator_values.values()])
            )
            overlap_values = [
                float(row["overlap_valid_pixel_ap"])
                for row in forged
                if row.get("overlap_valid_pixel_ap") is not None
            ]
            alignments = alignments_by_condition[condition_name]
            correlations = [
                float(row["ecc_correlation"])
                for row in alignments
                if np.isfinite(float(row["ecc_correlation"]))
            ]
            metrics.append(
                {
                    "model": model_name,
                    "training_seed": int(model_name.rsplit("_", 1)[-1]),
                    "condition": condition_name,
                    "forged_documents": len(forged),
                    "authentic_documents": len(authentic),
                    "generator_macro_pixel_ap": generator_macro,
                    "document_macro_pixel_ap": float(
                        np.mean([row["full_document_pixel_ap"] for row in forged])
                    ),
                    "document_macro_pixel_auroc": float(
                        np.mean([row["full_document_pixel_auroc"] for row in forged])
                    ),
                    "document_macro_pixel_f1": float(
                        np.mean([row["pixel_f1"] for row in forged])
                    ),
                    "document_macro_pixel_iou": float(
                        np.mean([row["pixel_iou"] for row in forged])
                    ),
                    "authentic_document_macro_pixel_fpr": float(
                        np.mean([row["authentic_pixel_fpr"] for row in authentic])
                    ),
                    "overlap_valid_document_macro_pixel_ap": (
                        float(np.mean(overlap_values)) if overlap_values else None
                    ),
                    "edited_documents_with_valid_overlap": len(overlap_values),
                    "mean_edited_pixel_reference_coverage": float(
                        np.mean([row["edited_pixel_reference_coverage"] for row in forged])
                    ),
                    "registration_convergence_rate": float(
                        np.mean(
                            [
                                row["alignment_status"] == "ecc_converged"
                                for row in alignments
                            ]
                        )
                    ),
                    "median_ecc_correlation": (
                        float(np.median(correlations)) if correlations else None
                    ),
                    "paper_evidence": False,
                    "threshold_selection_used": False,
                }
            )
    return metrics


def _aggregate_across_seeds(
    metrics: list[dict[str, Any]], condition_names: list[str]
) -> list[dict[str, Any]]:
    metric_names = (
        "generator_macro_pixel_ap",
        "document_macro_pixel_ap",
        "document_macro_pixel_f1",
        "document_macro_pixel_iou",
        "authentic_document_macro_pixel_fpr",
        "overlap_valid_document_macro_pixel_ap",
        "mean_edited_pixel_reference_coverage",
    )
    output: list[dict[str, Any]] = []
    for condition in condition_names:
        selected = [row for row in metrics if row["condition"] == condition]
        for metric_name in metric_names:
            values = [
                float(row[metric_name])
                for row in selected
                if row.get(metric_name) is not None
            ]
            if not values:
                continue
            output.append(
                {
                    "condition": condition,
                    "metric": metric_name,
                    "mean": float(np.mean(values)),
                    "sample_standard_deviation": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    ),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "seed_count": len(values),
                    "paper_evidence": False,
                }
            )
    return output


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not runtime["gpu_launch_authorized"] or not runtime[
        "reference_integrity_evaluation_authorized"
    ]:
        raise ValueError("reference-integrity evaluation was not authorized")
    if not runtime["viewed_development_read_allowed"]:
        raise ValueError("viewed-development input was not authorized")
    if any(
        bool(runtime.get(name))
        for name in (
            "model_training_authorized",
            "unseen_development_read_allowed",
            "final_reserve_read_allowed",
            "threshold_selection_allowed",
            "method_change_authorized",
        )
    ):
        raise ValueError("reference-integrity diagnostic crossed an evidence boundary")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("reference-integrity diagnostic cannot be final paper evidence")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("reference-integrity diagnostic requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("reference-integrity protocol SHA-256 changed")
    inputs = config["input"]
    manifest_path = _resolve(project_root, inputs["manifest"])
    if _sha256(manifest_path) != inputs["expected_manifest_sha256"]:
        raise ValueError("reference-integrity manifest SHA-256 changed")
    rows = sorted(
        _read_jsonl(manifest_path), key=lambda row: str(row["source_group_id"])
    )
    if len(rows) != int(inputs["expected_groups"]):
        raise ValueError("reference-integrity group count changed")
    generator_counts = Counter(str(row["selected_generator"]) for row in rows)
    if dict(generator_counts) != {
        str(name): int(value)
        for name, value in inputs["expected_generator_counts"].items()
    }:
        raise ValueError("reference-integrity generator counts changed")

    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    alignment_cache_dir = _resolve(scratch, paths["alignment_cache_dir"])
    predictions_path = _resolve(project_root, paths["predictions"])
    alignments_path = _resolve(project_root, paths["alignments"])
    metrics_path = _resolve(project_root, paths["metrics"])
    aggregate_path = _resolve(project_root, paths["aggregate"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (
        score_cache_dir,
        alignment_cache_dir,
        predictions_path.parent,
        alignments_path.parent,
        metrics_path.parent,
        aggregate_path.parent,
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

    preprocessing = config["preprocessing"]
    if (
        int(preprocessing["score_cache_schema_version"]) != 1
        or preprocessing["score_cache_dtype"] != "float32"
        or int(preprocessing["alignment_cache_schema_version"]) != 1
    ):
        raise ValueError("reference-integrity cache schema changed")
    registration = config["registration"]
    inference = config["inference"]
    conditions = config["conditions"]
    condition_names = [str(condition["name"]) for condition in conditions]
    if condition_names != [
        "correct_full",
        "correct_overlap_090",
        "correct_overlap_075",
        "correct_overlap_050",
        "wrong_same_dataset_size",
        "wrong_global",
    ]:
        raise ValueError("reference-integrity condition set changed")

    model_config = config["models"]
    family_seeds = [int(value) for value in model_config["family_seeds"]]
    if family_seeds != [20260747, 20260763, 20260764]:
        raise ValueError("reference-integrity model family changed")
    encoder_path = _resolve(scratch, model_config["encoder_weights"])
    if _sha256(encoder_path) != model_config["encoder_weights_sha256"]:
        raise ValueError("reference-integrity encoder weights changed")
    models: dict[str, torch.nn.Module] = {}
    model_hashes: dict[str, str] = {}
    model_thresholds: dict[str, float] = {}
    for training_seed in family_seeds:
        name = f"robust_{training_seed}"
        item = model_config[name]
        checkpoint_path = _resolve(project_root, item["checkpoint"])
        if _sha256(checkpoint_path) != item["checkpoint_sha256"]:
            raise ValueError(f"reference-integrity checkpoint changed: {name}")
        model = _load_teacher(
            encoder_path, model_config["teacher_conv1_coefficients"]
        )
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state"], strict=True)
        models[name] = model.to(device).eval().requires_grad_(False)
        model_hashes[name] = str(item["checkpoint_sha256"])
        model_thresholds[name] = float(item["fixed_pixel_threshold"])

    wrong_seed = int(config["controls"]["wrong_reference_seed"])
    global_map = _deterministic_global_map(rows, wrong_seed)
    same_dataset_map = _same_dataset_size_map(rows, wrong_seed)
    image_cache: dict[str, np.ndarray] = {}

    def load_image(row: dict[str, Any], field: str, sha_field: str) -> np.ndarray:
        path = _resolve(scratch, row[field])
        key = str(path)
        if key not in image_cache:
            if _sha256(path) != str(row[sha_field]):
                raise ValueError(f"reference-integrity {field} SHA-256 changed")
            with Image.open(path) as handle:
                image_cache[key] = np.asarray(handle.convert("RGB"))
        return image_cache[key]

    prediction_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    failed_predictions = 0
    failed_alignments = 0
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
        if _sha256(mask_path) != str(row["mask_sha256"]):
            raise ValueError("reference-integrity mask SHA-256 changed")
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0

        for sample_kind, candidate_native in (
            ("forged", forged_native),
            ("authentic", authentic_native),
        ):
            candidate = _resize_image(
                candidate_native, int(preprocessing["max_side"])
            )
            candidate_sha256 = str(
                row["image_sha256"]
                if sample_kind == "forged"
                else row["authentic_sha256"]
            )
            for condition in conditions:
                condition_name = str(condition["name"])
                reference_mode = str(condition["reference_mode"])
                retained_fraction = float(condition["retained_linear_fraction"])
                if reference_mode == "correct":
                    reference_row = row
                elif reference_mode == "wrong_same_dataset_size":
                    reference_row = same_dataset_map[group]
                elif reference_mode == "wrong_global":
                    reference_row = global_map[group]
                else:
                    raise ValueError(f"unknown reference mode: {reference_mode}")
                reference_group = str(reference_row["source_group_id"])
                reference_native = load_image(
                    reference_row, "authentic", "authentic_sha256"
                )
                reference = _resize_image(
                    reference_native, int(preprocessing["max_side"])
                )
                if reference.shape != candidate.shape:
                    reference = cv2.resize(
                        reference,
                        (candidate.shape[1], candidate.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                reference = _center_crop_resized(reference, retained_fraction)
                alignment_key = hashlib.sha256(
                    json.dumps(
                        {
                            "candidate_sha256": candidate_sha256,
                            "reference_sha256": str(reference_row["authentic_sha256"]),
                            "condition": condition,
                            "registration": registration,
                            "preprocessing_max_side": preprocessing["max_side"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                alignment_path = (
                    alignment_cache_dir / condition_name / f"{alignment_key}.npz"
                )
                alignment_path.parent.mkdir(parents=True, exist_ok=True)
                alignment_record: dict[str, Any] = {
                    "alignment_key": alignment_key,
                    "source_group_id": group,
                    "sample_kind": sample_kind,
                    "condition": condition_name,
                    "reference_mode": reference_mode,
                    "reference_source_group_id": reference_group,
                    "retained_linear_fraction": retained_fraction,
                    "paper_evidence": False,
                    "viewed_development": True,
                    "final_reserve_read": False,
                }
                try:
                    if not alignment_path.is_file():
                        aligned_reference, validity, metadata = _align_reference(
                            candidate, reference, registration
                        )
                        temporary = alignment_path.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(
                                handle,
                                aligned_reference=aligned_reference.astype(np.uint8),
                                validity=validity.astype(np.uint8),
                                estimated_homography=np.asarray(
                                    metadata["estimated_homography"], dtype=np.float64
                                ),
                                ecc_correlation=np.asarray(
                                    metadata["ecc_correlation"], dtype=np.float64
                                ),
                                phase_correlation_response=np.asarray(
                                    metadata["phase_correlation_response"],
                                    dtype=np.float64,
                                ),
                                alignment_status=np.asarray(
                                    metadata["alignment_status"]
                                ),
                                failure_type=np.asarray(
                                    metadata["alignment_failure_type"] or ""
                                ),
                                failure_reason=np.asarray(
                                    metadata["alignment_failure_reason"] or ""
                                ),
                            )
                        temporary.replace(alignment_path)
                    else:
                        alignment_cache_hits += 1
                    with np.load(alignment_path, allow_pickle=False) as archive:
                        aligned_reference = archive["aligned_reference"]
                        validity = archive["validity"]
                        warp = archive["estimated_homography"]
                        correlation = float(archive["ecc_correlation"])
                        phase_response = float(
                            archive["phase_correlation_response"]
                        )
                        status = str(archive["alignment_status"])
                        failure_type = str(archive["failure_type"])
                        failure_reason = str(archive["failure_reason"])
                    if (
                        aligned_reference.shape != candidate.shape
                        or validity.shape != candidate.shape[:2]
                        or not np.isfinite(warp).all()
                    ):
                        raise ValueError("reference-integrity alignment cache invalid")
                    alignment_record.update(
                        {
                            "status": "ok",
                            "alignment_status": status,
                            "ecc_correlation": correlation,
                            "phase_correlation_response": phase_response,
                            "estimated_homography": warp.tolist(),
                            "valid_area_fraction": float(validity.mean()),
                            "alignment_failure_type": failure_type or None,
                            "alignment_failure_reason": failure_reason or None,
                            "alignment_cache": str(
                                alignment_path.relative_to(scratch)
                            ),
                        }
                    )
                except Exception as error:
                    failed_alignments += 1
                    alignment_record.update(
                        {
                            "status": "failed",
                            "alignment_status": "failed",
                            "ecc_correlation": math.nan,
                            "phase_correlation_response": math.nan,
                            "valid_area_fraction": 0.0,
                            "alignment_failure_type": type(error).__name__,
                            "alignment_failure_reason": str(error),
                        }
                    )
                    logging.exception(
                        "alignment group=%s condition=%s failed", group, condition_name
                    )
                alignment_rows.append(alignment_record)
                _write_jsonl(alignments_path, alignment_rows)
                if alignment_record["status"] != "ok":
                    for model_name in models:
                        prediction_rows.append(
                            {
                                "record_id": f"{model_name}:{condition_name}:{sample_kind}:{group}",
                                "source_group_id": group,
                                "generator": generator,
                                "sample_kind": sample_kind,
                                "condition": condition_name,
                                "model": model_name,
                                "status": "failed",
                                "failure_type": "AlignmentFailure",
                                "failure_reason": alignment_record[
                                    "alignment_failure_reason"
                                ],
                                "paper_evidence": False,
                                "viewed_development": True,
                                "final_reserve_read": False,
                            }
                        )
                        failed_predictions += 1
                    continue

                for model_name, model in models.items():
                    threshold = model_thresholds[model_name]
                    prediction: dict[str, Any] = {
                        "record_id": f"{model_name}:{condition_name}:{sample_kind}:{group}",
                        "source_group_id": group,
                        "generator": generator,
                        "source_dataset": str(row["source_dataset"]),
                        "sample_kind": sample_kind,
                        "condition": condition_name,
                        "model": model_name,
                        "training_seed": int(model_name.rsplit("_", 1)[-1]),
                        "reference_mode": reference_mode,
                        "reference_source_group_id": reference_group,
                        "retained_linear_fraction": retained_fraction,
                        "fixed_pixel_threshold": threshold,
                        "threshold_selection_used": False,
                        "alignment_key": alignment_key,
                        "alignment_status": alignment_record["alignment_status"],
                        "ecc_correlation": alignment_record["ecc_correlation"],
                        "valid_area_fraction": alignment_record[
                            "valid_area_fraction"
                        ],
                        "status": "failed",
                        "paper_evidence": False,
                        "viewed_development": True,
                        "unseen_development_read": False,
                        "final_reserve_read": False,
                    }
                    try:
                        score_key = hashlib.sha256(
                            json.dumps(
                                {
                                    "alignment_key": alignment_key,
                                    "checkpoint_sha256": model_hashes[model_name],
                                    "candidate_sha256": candidate_sha256,
                                    "inference": inference,
                                    "preprocessing": preprocessing,
                                    "sample_kind": sample_kind,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        score_path = (
                            score_cache_dir
                            / model_name
                            / condition_name
                            / f"{score_key}.npz"
                        )
                        score_path.parent.mkdir(parents=True, exist_ok=True)
                        if not score_path.is_file():
                            probability = _infer_pair_tiled(
                                model,
                                candidate,
                                aligned_reference,
                                device,
                                inference,
                                preprocessing,
                            )
                            temporary = score_path.with_suffix(".npz.tmp")
                            with temporary.open("wb") as handle:
                                np.savez_compressed(
                                    handle, scores=probability.astype(np.float32)
                                )
                            temporary.replace(score_path)
                        else:
                            score_cache_hits += 1
                        with np.load(score_path, allow_pickle=False) as archive:
                            probability = archive["scores"]
                        if (
                            probability.dtype != np.float32
                            or probability.shape != candidate.shape[:2]
                            or not np.isfinite(probability).all()
                        ):
                            raise ValueError("reference-integrity score cache invalid")
                        native_probability = cv2.resize(
                            probability,
                            (native_mask.shape[1], native_mask.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        native_validity = cv2.resize(
                            validity,
                            (native_mask.shape[1], native_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                        if sample_kind == "forged":
                            pixel_ap, pixel_auroc = _ranking_metrics(
                                native_probability, native_mask
                            )
                            threshold_metrics = _threshold_metrics(
                                native_probability, native_mask, threshold
                            )
                            positive_pixels = int(native_mask.sum())
                            covered_positive = int(
                                np.logical_and(native_mask, native_validity).sum()
                            )
                            overlap_ap = None
                            overlap_target = native_mask[native_validity]
                            if (
                                covered_positive > 0
                                and overlap_target.size > covered_positive
                            ):
                                overlap_ap, _ = _ranking_metrics(
                                    native_probability[native_validity],
                                    overlap_target,
                                )
                            prediction.update(
                                {
                                    "full_document_pixel_ap": pixel_ap,
                                    "full_document_pixel_auroc": pixel_auroc,
                                    "overlap_valid_pixel_ap": overlap_ap,
                                    "edited_pixel_reference_coverage": covered_positive
                                    / max(1, positive_pixels),
                                    **threshold_metrics,
                                }
                            )
                        else:
                            prediction.update(
                                {
                                    "authentic_pixel_fpr": float(
                                        (native_probability >= threshold).mean()
                                    ),
                                    "edited_pixel_reference_coverage": 0.0,
                                }
                            )
                        prediction.update(
                            {
                                "status": "ok",
                                "score_cache": str(score_path.relative_to(scratch)),
                                "score_cache_schema_version": 1,
                                "score_cache_dtype": str(probability.dtype),
                                "score_shape": list(probability.shape),
                                "native_shape": list(native_probability.shape),
                                "checkpoint_sha256": model_hashes[model_name],
                            }
                        )
                    except Exception as error:
                        failed_predictions += 1
                        prediction["failure_type"] = type(error).__name__
                        prediction["failure_reason"] = str(error)
                        logging.exception("record_id=%s failed", prediction["record_id"])
                    prediction_rows.append(prediction)
                _write_jsonl(predictions_path, prediction_rows)
        logging.info("completed_groups=%d total_groups=%d", row_index, len(rows))

    expected_alignments = len(rows) * 2 * len(conditions)
    expected_predictions = expected_alignments * len(models)
    complete = (
        failed_alignments == 0
        and failed_predictions == 0
        and len(alignment_rows) == expected_alignments
        and len(prediction_rows) == expected_predictions
    )
    if not complete:
        output = {
            "experiment": config["experiment"],
            "status": "reference_integrity_viewed20_incomplete",
            "paper_evidence": False,
            "viewed_development_read": True,
            "final_reserve_read": False,
            "expected_alignment_records": expected_alignments,
            "alignment_records": len(alignment_rows),
            "failed_alignment_records": failed_alignments,
            "expected_prediction_records": expected_predictions,
            "prediction_records": len(prediction_rows),
            "failed_prediction_records": failed_predictions,
        }
        _write_json(summary_path, output)
        if runtime["require_all_records"]:
            raise RuntimeError("reference-integrity diagnostic was incomplete")
        return output

    model_names = list(models)
    metrics = _aggregate_metrics(
        prediction_rows, alignment_rows, model_names, condition_names
    )
    aggregate = _aggregate_across_seeds(metrics, condition_names)
    _write_csv(metrics_path, metrics)
    _write_csv(aggregate_path, aggregate)

    correlations = {
        condition_name: [
            float(row["ecc_correlation"])
            for row in alignment_rows
            if row["condition"] == condition_name
        ]
        for condition_name in condition_names
    }
    screening_auroc = {
        wrong_name: _binary_auroc(
            correlations["correct_full"], correlations[wrong_name]
        )
        for wrong_name in ("wrong_same_dataset_size", "wrong_global")
    }
    aggregate_lookup = {
        (row["condition"], row["metric"]): row for row in aggregate
    }
    condition_summary = {
        condition: {
            "generator_macro_pixel_ap_mean": aggregate_lookup[
                (condition, "generator_macro_pixel_ap")
            ]["mean"],
            "generator_macro_pixel_ap_sample_sd": aggregate_lookup[
                (condition, "generator_macro_pixel_ap")
            ]["sample_standard_deviation"],
            "authentic_pixel_fpr_mean": aggregate_lookup[
                (condition, "authentic_document_macro_pixel_fpr")
            ]["mean"],
            "edited_pixel_reference_coverage_mean": aggregate_lookup[
                (condition, "mean_edited_pixel_reference_coverage")
            ]["mean"],
        }
        for condition in condition_names
    }
    output = {
        "experiment": config["experiment"],
        "status": "reference_integrity_viewed20_complete",
        "paper_evidence": False,
        "post_final_limitation_diagnostic": True,
        "viewed_development_read": True,
        "unseen_development_read": False,
        "final_reserve_read": False,
        "threshold_selection_used": False,
        "model_or_method_change_authorized": False,
        "selected_groups": len(rows),
        "expected_alignment_records": expected_alignments,
        "successful_alignment_records": len(alignment_rows),
        "failed_alignment_records": 0,
        "expected_prediction_records": expected_predictions,
        "successful_prediction_records": len(prediction_rows),
        "failed_prediction_records": 0,
        "alignment_cache_hits": alignment_cache_hits,
        "score_cache_hits": score_cache_hits,
        "condition_summary": condition_summary,
        "ecc_correlation_correct_vs_wrong_auroc": screening_auroc,
        "protocol_sha256": _sha256(protocol_path),
        "input_manifest_sha256": _sha256(manifest_path),
        "model_checkpoint_sha256": model_hashes,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "alignments": str(alignments_path.relative_to(project_root)),
            "alignments_sha256": _sha256(alignments_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "aggregate": str(aggregate_path.relative_to(project_root)),
            "aggregate_sha256": _sha256(aggregate_path),
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

