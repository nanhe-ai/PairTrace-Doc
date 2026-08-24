from __future__ import annotations

import argparse
import hashlib
import io
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

from pairtrace_doc.pipelines.compare_generator_balanced_1000 import (
    _stratified_paired_bootstrap,
)
from pairtrace_doc.pipelines.train_pairtrace_100 import TraceUNet, _load_teacher
from pairtrace_doc.pipelines.train_student_100 import (
    ResNet18UNet,
    _infer_tiled,
    _positions,
    _ranking_metrics,
    _read_jsonl,
    _resolve,
    _sha256,
    _threshold_vectors,
    _write_csv,
    _write_json,
    _write_jsonl,
)


REQUIRED_CONDITIONS = {
    "student_clean",
    "raw_difference_clean",
    "pair_teacher_correct_clean",
    "pair_teacher_shuffled_clean",
    "pair_teacher_identical_clean",
    "raw_difference_matched_jpeg_q85",
    "pair_teacher_correct_matched_jpeg_q85",
}


def _resize_image(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image
    scale = max_side / max(height, width)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def _resize_reference(reference: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if reference.shape[:2] == shape:
        return reference
    interpolation = (
        cv2.INTER_AREA
        if reference.shape[0] > shape[0] or reference.shape[1] > shape[1]
        else cv2.INTER_LINEAR
    )
    return cv2.resize(reference, (shape[1], shape[0]), interpolation=interpolation)


def _jpeg_roundtrip(image: np.ndarray, specification: dict[str, Any]) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image).save(
        buffer,
        format="JPEG",
        quality=int(specification["quality"]),
        subsampling=int(specification["subsampling"]),
        optimize=bool(specification["optimize"]),
        progressive=bool(specification["progressive"]),
    )
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return np.asarray(handle.convert("RGB"))


def _apply_transform(
    candidate: np.ndarray,
    reference: np.ndarray | None,
    specification: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    transform = str(specification["transform"])
    if transform == "none":
        return candidate, reference
    if transform == "matched_jpeg":
        transformed_candidate = _jpeg_roundtrip(candidate, specification)
        transformed_reference = (
            None if reference is None else _jpeg_roundtrip(reference, specification)
        )
        return transformed_candidate, transformed_reference
    raise ValueError(f"unsupported pair-at-inference transform: {transform}")


def _pair_model_input(
    candidate_patches: np.ndarray,
    reference_patches: np.ndarray,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    if candidate_patches.shape != reference_patches.shape:
        raise ValueError("candidate and reference patch batches differ")
    mean = np.asarray(preprocessing["imagenet_mean"], dtype=np.float32)
    std = np.asarray(preprocessing["imagenet_std"], dtype=np.float32)
    candidate = (candidate_patches.astype(np.float32) / 255.0 - mean) / std
    reference = (reference_patches.astype(np.float32) / 255.0 - mean) / std
    return np.concatenate(
        [candidate, reference, candidate - reference], axis=3
    ).transpose(0, 3, 1, 2)


def _infer_pair_tiled(
    model: torch.nn.Module,
    candidate: np.ndarray,
    reference: np.ndarray,
    device: torch.device,
    inference: dict[str, Any],
    preprocessing: dict[str, Any],
) -> np.ndarray:
    if candidate.shape != reference.shape:
        raise ValueError("pair inference requires matched model-space geometry")
    tile = int(inference["validation_tile_size"])
    stride = int(inference["validation_tile_stride"])
    batch_size = int(inference["validation_tile_batch_size"])
    height, width = candidate.shape[:2]
    pad_height = max(0, tile - height)
    pad_width = max(0, tile - width)
    padding = ((0, pad_height), (0, pad_width), (0, 0))
    candidate_padded = np.pad(candidate, padding, mode="reflect")
    reference_padded = np.pad(reference, padding, mode="reflect")
    padded_height, padded_width = candidate_padded.shape[:2]
    coordinates = [
        (top, left)
        for top in _positions(padded_height, tile, stride)
        for left in _positions(padded_width, tile, stride)
    ]
    accumulator = np.zeros((padded_height, padded_width), dtype=np.float32)
    counts = np.zeros((padded_height, padded_width), dtype=np.float32)
    model.eval()
    for start in range(0, len(coordinates), batch_size):
        selected = coordinates[start : start + batch_size]
        candidate_patches = np.stack(
            [candidate_padded[top : top + tile, left : left + tile] for top, left in selected]
        )
        reference_patches = np.stack(
            [reference_padded[top : top + tile, left : left + tile] for top, left in selected]
        )
        tensor = torch.from_numpy(
            _pair_model_input(candidate_patches, reference_patches, preprocessing).copy()
        ).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=bool(inference["amp"])
        ):
            probabilities = torch.sigmoid(model(tensor)).squeeze(1).float().cpu().numpy()
        for probability, (top, left) in zip(probabilities, selected):
            accumulator[top : top + tile, left : left + tile] += probability
            counts[top : top + tile, left : left + tile] += 1.0
    return (accumulator / np.maximum(counts, 1.0))[:height, :width]


def _raw_difference(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if candidate.shape != reference.shape:
        raise ValueError("raw difference requires aligned model-space geometry")
    difference = np.abs(candidate.astype(np.int16) - reference.astype(np.int16))
    return difference.max(axis=2).astype(np.float32) / 255.0


def _shuffled_group_map(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    if len(rows) < 2:
        raise ValueError("shuffled reference control needs at least two groups")
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"{seed}|{row['source_group_id']}".encode("utf-8")
            ).hexdigest(),
            str(row["source_group_id"]),
        ),
    )
    result = {
        str(row["source_group_id"]): str(ordered[(index + 1) % len(ordered)]["source_group_id"])
        for index, row in enumerate(ordered)
    }
    if any(group == target for group, target in result.items()):
        raise ValueError("shuffled reference control retained a source group")
    return result


def _aggregate_condition(
    payload: dict[str, list[Any]],
    thresholds: np.ndarray,
) -> dict[str, Any]:
    forged = payload["forged"]
    authentic_vectors = payload["authentic_vectors"]
    if not forged or not authentic_vectors:
        raise ValueError("condition aggregation is incomplete")
    macro_precision = np.zeros_like(thresholds, dtype=float)
    macro_recall = np.zeros_like(thresholds, dtype=float)
    macro_f1 = np.zeros_like(thresholds, dtype=float)
    macro_iou = np.zeros_like(thresholds, dtype=float)
    for item in forged:
        tp, fp, positives = item["threshold_vectors"]
        fn = positives - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = tp / positives
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
        iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fp + fn) > 0)
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        macro_iou += iou
    count = len(forged)
    macro_precision /= count
    macro_recall /= count
    macro_f1 /= count
    macro_iou /= count
    authentic_fpr = np.mean(np.stack(authentic_vectors), axis=0)
    cap = float(payload["authentic_fpr_max"])
    feasible = np.flatnonzero(authentic_fpr <= cap + 1e-12)
    if not feasible.size:
        feasible = np.flatnonzero(authentic_fpr == authentic_fpr.min())
    best_f1 = macro_f1[feasible].max()
    candidates = feasible[np.isclose(macro_f1[feasible], best_f1, atol=1e-12, rtol=0)]
    best_authentic = authentic_fpr[candidates].min()
    candidates = candidates[np.isclose(authentic_fpr[candidates], best_authentic, atol=1e-12, rtol=0)]
    selected = int(candidates[-1])
    by_generator: dict[str, list[float]] = defaultdict(list)
    for item in forged:
        by_generator[str(item["generator"])].append(float(item["macro_pixel_ap"]))
    per_generator = {
        generator: float(np.mean(values)) for generator, values in sorted(by_generator.items())
    }
    metrics: dict[str, Any] = {
        "development_groups": count,
        "generator_macro_pixel_ap": float(np.mean(list(per_generator.values()))),
        "macro_pixel_ap": float(np.mean([item["macro_pixel_ap"] for item in forged])),
        "pixel_auroc": float(np.mean([item["pixel_auroc"] for item in forged])),
        "pixel_threshold": float(thresholds[selected]),
        "pixel_precision": float(macro_precision[selected]),
        "pixel_recall": float(macro_recall[selected]),
        "pixel_f1": float(macro_f1[selected]),
        "pixel_iou": float(macro_iou[selected]),
        "authentic_pixel_fpr": float(authentic_fpr[selected]),
        "paper_evidence": False,
    }
    for generator, value in per_generator.items():
        safe = "".join(character if character.isalnum() else "_" for character in generator).strip("_")
        metrics[f"macro_pixel_ap__{safe}"] = value
    return metrics


def _decision(
    metrics: dict[str, dict[str, Any]],
    forged_scores: dict[str, dict[str, tuple[str, float]]],
    gate: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    pairs = {
        "correct_minus_student": ("pair_teacher_correct_clean", "student_clean"),
        "correct_minus_shuffled": ("pair_teacher_correct_clean", "pair_teacher_shuffled_clean"),
        "correct_jpeg_minus_raw_jpeg": (
            "pair_teacher_correct_matched_jpeg_q85",
            "raw_difference_matched_jpeg_q85",
        ),
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for offset, (name, (left, right)) in enumerate(pairs.items()):
        comparisons[name] = _stratified_paired_bootstrap(
            forged_scores[left],
            forged_scores[right],
            int(bootstrap["seed"]) + offset,
            int(bootstrap["resamples"]),
            float(bootstrap["confidence_level"]),
        )
    correct = metrics["pair_teacher_correct_clean"]
    correct_jpeg = metrics["pair_teacher_correct_matched_jpeg_q85"]
    clean_drop = float(correct["generator_macro_pixel_ap"] - correct_jpeg["generator_macro_pixel_ap"])
    checks = {
        "correct_clean_ap_floor": float(correct["generator_macro_pixel_ap"])
        >= float(gate["correct_clean_generator_macro_ap_min"]),
        "correct_minus_student_effect_floor": comparisons["correct_minus_student"]["effect"]
        >= float(gate["correct_minus_student_min"]),
        "correct_minus_student_interval_positive": comparisons["correct_minus_student"]["ci_low"] > 0.0,
        "correct_minus_shuffled_effect_floor": comparisons["correct_minus_shuffled"]["effect"]
        >= float(gate["correct_minus_shuffled_min"]),
        "correct_minus_shuffled_interval_positive": comparisons["correct_minus_shuffled"]["ci_low"] > 0.0,
        "correct_authentic_fpr_ceiling": float(correct["authentic_pixel_fpr"])
        <= float(gate["authentic_pixel_fpr_max"]) + 1e-12,
        "jpeg_clean_drop_ceiling": clean_drop <= float(gate["jpeg_clean_drop_max"]) + 1e-12,
        "correct_jpeg_minus_raw_jpeg_effect_floor": comparisons[
            "correct_jpeg_minus_raw_jpeg"
        ]["effect"]
        >= float(gate["correct_jpeg_minus_raw_jpeg_min"]),
        "correct_jpeg_minus_raw_jpeg_interval_positive": comparisons[
            "correct_jpeg_minus_raw_jpeg"
        ]["ci_low"]
        > 0.0,
        "correct_jpeg_authentic_fpr_ceiling": float(correct_jpeg["authentic_pixel_fpr"])
        <= float(gate["authentic_pixel_fpr_max"]) + 1e-12,
    }
    return {
        "comparisons": comparisons,
        "correct_clean_minus_jpeg_generator_macro_ap": clean_drop,
        "checks": checks,
        "overall_pass": all(checks.values()),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not runtime["gpu_launch_authorized"] or not runtime["feasibility_evaluation_authorized"]:
        raise ValueError("pair-at-inference evaluation was not explicitly authorized")
    if runtime["method_training_authorized"] or runtime["multi_seed_authorized"]:
        raise ValueError("feasibility evaluation cannot authorize training or multi-seed compute")
    if runtime["viewed_diagnostic_read_allowed"] or runtime["final_reserve_read_allowed"]:
        raise ValueError("pair-at-inference evaluation crossed an evidence boundary")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("pair-at-inference feasibility output cannot be paper evidence")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("pair-at-inference evaluation requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("pair-at-inference protocol SHA-256 changed")
    input_config = config["input"]
    manifest_path = _resolve(project_root, input_config["manifest"])
    if _sha256(manifest_path) != input_config["expected_manifest_sha256"]:
        raise ValueError("pair-at-inference manifest SHA-256 changed")
    rows = sorted(_read_jsonl(manifest_path), key=lambda row: str(row["source_group_id"]))
    if len(rows) != int(input_config["expected_development_groups"]):
        raise ValueError("pair-at-inference development count changed")
    if len({str(row["source_group_id"]) for row in rows}) != len(rows):
        raise ValueError("pair-at-inference development contains duplicate groups")
    if {str(row["pair_at_inference_freeze_id"]) for row in rows} != {
        str(input_config["expected_freeze_id"])
    }:
        raise ValueError("pair-at-inference freeze ID changed")
    expected_generators = {
        str(name): int(value) for name, value in input_config["expected_generator_counts"].items()
    }
    counts = Counter(str(row["selected_generator"]) for row in rows)
    if dict(counts) != expected_generators:
        raise ValueError(f"pair-at-inference generator counts changed: {dict(counts)}")
    max_groups = runtime.get("max_groups")
    if max_groups is not None:
        rows = rows[: int(max_groups)]

    condition_specs = {str(item["name"]): item for item in config["conditions"]}
    if set(condition_specs) != REQUIRED_CONDITIONS:
        raise ValueError("pair-at-inference condition whitelist changed")
    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    models = config["models"]
    student_path = _resolve(project_root, models["student_checkpoint"])
    teacher_path = _resolve(project_root, models["teacher_checkpoint"])
    encoder_path = _resolve(Path(os.environ.get(config["paths"]["scratch_env"], str(_resolve(project_root, config["paths"]["scratch_default"])))), models["encoder_weights"])
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

    paths = config["paths"]
    scratch = Path(
        os.environ.get(paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"])))
    ).resolve()
    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    predictions_path = _resolve(project_root, paths["predictions"])
    metrics_path = _resolve(project_root, paths["metrics"])
    comparisons_path = _resolve(project_root, paths["comparisons"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (score_cache_dir, predictions_path.parent, metrics_path.parent, comparisons_path.parent, summary_path.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    inference = config["inference"]
    preprocessing = config["preprocessing"]
    thresholds = np.arange(
        float(config["operating_point"]["candidate_min"]),
        float(config["operating_point"]["candidate_max"]) + 1e-12,
        float(config["operating_point"]["candidate_step"]),
    )
    shuffled = _shuffled_group_map(rows, seed + int(config["controls"]["shuffle_seed_offset"]))
    rows_by_group = {str(row["source_group_id"]): row for row in rows}
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

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    prediction_rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {
        name: {"forged": [], "authentic_vectors": [], "authentic_fpr_max": config["operating_point"]["authentic_pixel_fpr_max"]}
        for name in condition_specs
    }
    forged_scores: dict[str, dict[str, tuple[str, float]]] = {name: {} for name in condition_specs}
    failures = 0
    cache_hits = 0

    for row_index, row in enumerate(rows, start=1):
        group = str(row["source_group_id"])
        generator = str(row["selected_generator"])
        forged_native = load_image(row, "image", "image_sha256")
        authentic_native = load_image(row, "authentic", "authentic_sha256")
        mask_path = _resolve(scratch, row["mask"])
        if _sha256(mask_path) != row["mask_sha256"]:
            raise ValueError("pair-at-inference mask SHA-256 changed")
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0
        if forged_native.shape[:2] != native_mask.shape or authentic_native.shape[:2] != native_mask.shape:
            raise ValueError("pair-at-inference aligned geometry changed")

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
                    "checkpoint_selection_used": False,
                    "threshold_selection_used": True,
                }
                try:
                    reference_mode = str(condition["reference_mode"])
                    if reference_mode == "none":
                        reference_native = None
                        reference_sha256 = None
                    elif reference_mode == "correct":
                        reference_native = authentic_native
                        reference_sha256 = str(row["authentic_sha256"])
                    elif reference_mode == "shuffled":
                        target = rows_by_group[shuffled[group]]
                        reference_native = load_image(target, "authentic", "authentic_sha256")
                        reference_sha256 = str(target["authentic_sha256"])
                    elif reference_mode == "identical":
                        reference_native = candidate_native
                        reference_sha256 = str(row["image_sha256"] if sample_kind == "forged" else row["authentic_sha256"])
                    else:
                        raise ValueError(f"unsupported reference mode: {reference_mode}")
                    candidate, reference = _apply_transform(candidate_native, reference_native, condition)
                    candidate = _resize_image(candidate, int(preprocessing["max_side"]))
                    if reference is not None:
                        reference = _resize_reference(reference, candidate.shape[:2])
                    candidate_sha256 = str(row["image_sha256"] if sample_kind == "forged" else row["authentic_sha256"])
                    scorer = str(condition["scorer"])
                    model_identity = (
                        models["student_checkpoint_sha256"]
                        if scorer == "student"
                        else models["teacher_checkpoint_sha256"]
                        if scorer == "pair_teacher"
                        else "raw_difference_v1"
                    )
                    cache_key = hashlib.sha256(
                        json.dumps(
                            {
                                "candidate_sha256": candidate_sha256,
                                "reference_sha256": reference_sha256,
                                "condition": condition,
                                "model_identity": model_identity,
                                "preprocessing": preprocessing,
                                "inference": inference,
                                "sample_kind": sample_kind,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    score_path = score_cache_dir / condition_name / f"{cache_key}.npz"
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    if score_path.is_file():
                        with np.load(score_path, allow_pickle=False) as archive:
                            probability = archive["scores"].astype(np.float32)
                        cache_hits += 1
                    else:
                        if scorer == "student":
                            probability = _infer_tiled(student, candidate, device, inference, preprocessing)
                        elif scorer == "pair_teacher":
                            if reference is None:
                                raise ValueError("pair teacher condition has no reference")
                            probability = _infer_pair_tiled(teacher, candidate, reference, device, inference, preprocessing)
                        elif scorer == "raw_difference":
                            if reference is None:
                                raise ValueError("raw difference condition has no reference")
                            probability = _raw_difference(candidate, reference)
                        else:
                            raise ValueError(f"unsupported scorer: {scorer}")
                        temporary = score_path.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(handle, scores=probability.astype(np.float16))
                        temporary.replace(score_path)
                    if probability.shape != candidate.shape[:2] or not np.isfinite(probability).all():
                        raise ValueError("pair-at-inference score cache is invalid")
                    native_probability = cv2.resize(
                        probability,
                        (native_mask.shape[1], native_mask.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    if sample_kind == "forged":
                        average_precision, auroc = _ranking_metrics(native_probability, native_mask)
                        vectors = _threshold_vectors(native_probability, native_mask, thresholds)
                        payloads[condition_name]["forged"].append(
                            {
                                "source_group_id": group,
                                "generator": generator,
                                "macro_pixel_ap": average_precision,
                                "pixel_auroc": auroc,
                                "threshold_vectors": vectors,
                            }
                        )
                        forged_scores[condition_name][group] = (generator, average_precision)
                        prediction.update({"macro_pixel_ap": average_precision, "pixel_auroc": auroc})
                    else:
                        histogram, _ = np.histogram(native_probability, bins=np.r_[thresholds, np.inf])
                        payloads[condition_name]["authentic_vectors"].append(
                            np.cumsum(histogram[::-1], dtype=np.int64)[::-1] / native_probability.size
                        )
                    prediction.update(
                        {
                            "status": "ok",
                            "score_cache": str(score_path.relative_to(scratch)),
                            "score_shape": list(probability.shape),
                            "native_shape": list(native_probability.shape),
                            "scorer": scorer,
                            "reference_mode": reference_mode,
                            "model_identity": model_identity,
                        }
                    )
                except Exception as error:
                    failures += 1
                    prediction["failure_type"] = type(error).__name__
                    prediction["failure_reason"] = str(error)
                    logging.exception("record_id=%s failed", prediction["record_id"])
                prediction_rows.append(prediction)
        _write_jsonl(predictions_path, prediction_rows)
        logging.info("completed_groups=%d total_groups=%d", row_index, len(rows))

    complete = failures == 0 and all(
        len(payload["forged"]) == len(rows) and len(payload["authentic_vectors"]) == len(rows)
        for payload in payloads.values()
    )
    if not complete:
        summary = {
            "experiment": config["experiment"],
            "status": "failed_incomplete",
            "paper_evidence": False,
            "successful_prediction_records": len(prediction_rows) - failures,
            "failed_prediction_records": failures,
            "final_reserve_read": False,
            "viewed_diagnostic_read": False,
            "outputs": {
                "predictions": str(predictions_path.relative_to(project_root)),
                "predictions_sha256": _sha256(predictions_path),
            },
        }
        _write_json(summary_path, summary)
        if runtime["require_all_records"]:
            raise RuntimeError(f"pair-at-inference evaluation failed for {failures} records")
        return summary

    metrics = {name: _aggregate_condition(payload, thresholds) for name, payload in payloads.items()}
    decision = _decision(metrics, forged_scores, config["feasibility_gate"], config["bootstrap"])
    condition_rows = [{"condition": name, **values} for name, values in metrics.items()]
    comparison_rows = []
    for name, values in decision["comparisons"].items():
        row: dict[str, Any] = {
            "comparison": name,
            "generator_macro_pixel_ap_difference": values["effect"],
            "ci_low": values["ci_low"],
            "ci_high": values["ci_high"],
            "bootstrap_resamples": int(config["bootstrap"]["resamples"]),
            "confidence_level": float(config["bootstrap"]["confidence_level"]),
            "paper_evidence": False,
        }
        for generator, value in values["per_generator_effect"].items():
            safe = "".join(character if character.isalnum() else "_" for character in generator).strip("_")
            row[f"difference__{safe}"] = value
        comparison_rows.append(row)
    _write_csv(metrics_path, condition_rows)
    _write_csv(comparisons_path, comparison_rows)
    output = {
        "experiment": config["experiment"],
        "status": "passed_feasibility_gate" if decision["overall_pass"] else "completed_success_criteria_not_met",
        "paper_evidence": False,
        "method_training_performed": False,
        "alignment_aware_method_development_authorized": bool(decision["overall_pass"]),
        "multi_seed_authorized": False,
        "final_reserve_read": False,
        "viewed_diagnostic_read": False,
        "qwen_generalization_estimable": False,
        "input_manifest_sha256": _sha256(manifest_path),
        "protocol_sha256": _sha256(protocol_path),
        "selected_development_groups": len(rows),
        "successful_prediction_records": len(prediction_rows),
        "failed_prediction_records": 0,
        "cache_hits": cache_hits,
        "conditions": metrics,
        "bootstrap": config["bootstrap"],
        "feasibility_gate": config["feasibility_gate"],
        "decision": decision,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
            "score_cache_dir": str(score_cache_dir.relative_to(scratch)),
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
