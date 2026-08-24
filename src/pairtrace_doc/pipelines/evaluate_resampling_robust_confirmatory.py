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

from pairtrace_doc.pipelines.compare_generator_balanced_1000 import (
    _stratified_paired_bootstrap,
)
from pairtrace_doc.pipelines.diagnose_pair_at_inference_alignment import (
    STRESSES,
    _estimate_ecc_alignment,
    _stress_homography,
    _warp_reference,
)
from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
    _aggregate_condition,
    _infer_pair_tiled,
    _raw_difference,
    _resize_image,
    _resize_reference,
    _shuffled_group_map,
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


GEOMETRIES = ("clean", *STRESSES)


def _required_conditions() -> set[str]:
    result = {"student_clean", "robust_shuffled_clean"}
    for geometry in GEOMETRIES:
        result.update(
            {
                f"raw_difference_{geometry}_ecc",
                f"baseline_{geometry}_ecc",
                f"robust_{geometry}_ecc",
            }
        )
    for stress in STRESSES:
        result.add(f"robust_{stress}_unaligned")
    return result


def _confirmatory_decision(
    metrics: dict[str, dict[str, Any]],
    forged_scores: dict[str, dict[str, tuple[str, float]]],
    registration: dict[str, Any],
    gate: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    pairs: dict[str, tuple[str, str]] = {}
    for geometry in GEOMETRIES:
        pairs[f"robust_minus_baseline__{geometry}"] = (
            f"robust_{geometry}_ecc",
            f"baseline_{geometry}_ecc",
        )
        pairs[f"robust_minus_raw__{geometry}"] = (
            f"robust_{geometry}_ecc",
            f"raw_difference_{geometry}_ecc",
        )
    pairs["robust_clean_minus_student"] = (
        "robust_clean_ecc",
        "student_clean",
    )
    pairs["robust_clean_minus_shuffled"] = (
        "robust_clean_ecc",
        "robust_shuffled_clean",
    )
    for stress in STRESSES:
        pairs[f"robust_ecc_minus_unaligned__{stress}"] = (
            f"robust_{stress}_ecc",
            f"robust_{stress}_unaligned",
        )
    comparisons = {
        name: _stratified_paired_bootstrap(
            forged_scores[left],
            forged_scores[right],
            int(bootstrap["seed"]) + index,
            int(bootstrap["resamples"]),
            float(bootstrap["confidence_level"]),
        )
        for index, (name, (left, right)) in enumerate(pairs.items())
    }
    robust_aps = {
        geometry: float(
            metrics[f"robust_{geometry}_ecc"]["generator_macro_pixel_ap"]
        )
        for geometry in GEOMETRIES
    }
    checks = {
        "robust_clean_ap_floor": robust_aps["clean"]
        >= float(gate["robust_clean_generator_macro_ap_min"]),
        "robust_minimum_stressed_ap_floor": min(
            robust_aps[stress] for stress in STRESSES
        )
        >= float(gate["robust_minimum_stressed_generator_macro_ap_min"]),
        "registration_convergence_floor": float(registration["convergence_rate"])
        >= float(gate["registration_convergence_rate_min"]),
        "registration_fallback_zero": int(registration["fallbacks"]) == 0,
        "registration_corner_p95_ceiling": float(
            registration["controlled_stress_corner_error_p95_pixels"]
        )
        <= float(gate["registration_corner_error_p95_pixels_max"]),
    }
    baseline_floors = gate["robust_minus_baseline_effect_min"]
    raw_floor = float(gate["robust_minus_raw_effect_min"])
    alignment_floor = float(gate["robust_ecc_minus_unaligned_effect_min"])
    for stress in STRESSES:
        baseline_comparison = comparisons[f"robust_minus_baseline__{stress}"]
        checks[f"robust_minus_baseline_effect__{stress}"] = float(
            baseline_comparison["effect"]
        ) >= float(baseline_floors[stress])
        checks[f"robust_minus_baseline_interval__{stress}"] = float(
            baseline_comparison["ci_low"]
        ) > 0.0
        raw_comparison = comparisons[f"robust_minus_raw__{stress}"]
        checks[f"robust_minus_raw_effect__{stress}"] = float(
            raw_comparison["effect"]
        ) >= raw_floor
        checks[f"robust_minus_raw_interval__{stress}"] = float(
            raw_comparison["ci_low"]
        ) > 0.0
        alignment_comparison = comparisons[f"robust_ecc_minus_unaligned__{stress}"]
        checks[f"robust_alignment_effect__{stress}"] = float(
            alignment_comparison["effect"]
        ) >= alignment_floor
        checks[f"robust_alignment_interval__{stress}"] = float(
            alignment_comparison["ci_low"]
        ) > 0.0
    for name, floor in (
        ("robust_clean_minus_student", gate["robust_clean_minus_student_effect_min"]),
        ("robust_clean_minus_shuffled", gate["robust_clean_minus_shuffled_effect_min"]),
    ):
        checks[f"{name}_effect"] = float(comparisons[name]["effect"]) >= float(floor)
        checks[f"{name}_interval"] = float(comparisons[name]["ci_low"]) > 0.0
    for geometry in GEOMETRIES:
        checks[f"robust_authentic_fpr__{geometry}"] = float(
            metrics[f"robust_{geometry}_ecc"]["authentic_pixel_fpr"]
        ) <= float(gate["authentic_pixel_fpr_max"])
    return {
        "robust_generator_macro_pixel_ap": robust_aps,
        "robust_minimum_stressed_generator_macro_pixel_ap": min(
            robust_aps[stress] for stress in STRESSES
        ),
        "comparisons": comparisons,
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
    if not runtime["gpu_launch_authorized"] or not runtime["confirmatory_evaluation_authorized"]:
        raise ValueError("confirmatory evaluation was not explicitly authorized")
    if not runtime["confirmatory_image_read_allowed"]:
        raise ValueError("confirmatory image read was not explicitly authorized")
    if any(
        bool(runtime.get(name))
        for name in (
            "model_training_authorized",
            "method_change_authorized",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("confirmatory evaluation crossed an evidence boundary")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("confirmatory validation cannot be paper evidence")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("confirmatory evaluation requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("confirmatory protocol SHA-256 changed")
    inputs = config["input"]
    manifest_path = _resolve(project_root, inputs["manifest"])
    if _sha256(manifest_path) != inputs["expected_manifest_sha256"]:
        raise ValueError("confirmatory manifest SHA-256 changed")
    rows = sorted(_read_jsonl(manifest_path), key=lambda row: str(row["source_group_id"]))
    if len(rows) != int(inputs["expected_groups"]):
        raise ValueError("confirmatory group count changed")
    if len({str(row["source_group_id"]) for row in rows}) != len(rows):
        raise ValueError("confirmatory manifest contains duplicate groups")
    if {str(row["resampling_confirmatory_freeze_id"]) for row in rows} != {
        str(inputs["expected_freeze_id"])
    }:
        raise ValueError("confirmatory freeze ID changed")
    counts = Counter(str(row["selected_generator"]) for row in rows)
    expected_counts = {
        str(name): int(value)
        for name, value in inputs["expected_generator_counts"].items()
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"confirmatory generator counts changed: {dict(counts)}")
    max_groups = runtime.get("max_groups")
    if max_groups is not None:
        rows = rows[: int(max_groups)]
    conditions = {str(item["name"]): item for item in config["conditions"]}
    if set(conditions) != _required_conditions():
        raise ValueError("confirmatory condition whitelist changed")
    stresses = {str(item["name"]): item for item in config["stresses"]}
    if set(stresses) != set(STRESSES):
        raise ValueError("confirmatory stress whitelist changed")

    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    model_config = config["models"]
    encoder_path = _resolve(scratch, model_config["encoder_weights"])
    if _sha256(encoder_path) != model_config["encoder_weights_sha256"]:
        raise ValueError("confirmatory encoder weights changed")
    student_path = _resolve(project_root, model_config["student"]["checkpoint"])
    if _sha256(student_path) != model_config["student"]["checkpoint_sha256"]:
        raise ValueError("confirmatory student checkpoint changed")
    saved_student = torch.load(student_path, map_location="cpu", weights_only=True)
    student = ResNet18UNet()
    student.load_state_dict(saved_student["model_state"], strict=True)
    student = student.to(device).eval().requires_grad_(False)
    pair_models: dict[str, torch.nn.Module] = {}
    pair_hashes: dict[str, str] = {}
    for name in ("baseline", "robust"):
        checkpoint_path = _resolve(project_root, model_config[name]["checkpoint"])
        expected = str(model_config[name]["checkpoint_sha256"])
        if _sha256(checkpoint_path) != expected:
            raise ValueError(f"confirmatory {name} checkpoint changed")
        model = _load_teacher(
            encoder_path, model_config["teacher_conv1_coefficients"]
        )
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state"], strict=True)
        pair_models[name] = model.to(device).eval().requires_grad_(False)
        pair_hashes[name] = expected

    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    alignment_cache_dir = _resolve(scratch, paths["alignment_cache_dir"])
    predictions_path = _resolve(project_root, paths["predictions"])
    alignments_path = _resolve(project_root, paths["alignments"])
    metrics_path = _resolve(project_root, paths["metrics"])
    comparisons_path = _resolve(project_root, paths["comparisons"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (
        score_cache_dir,
        alignment_cache_dir,
        predictions_path.parent,
        alignments_path.parent,
        metrics_path.parent,
        comparisons_path.parent,
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
    if int(preprocessing["score_cache_schema_version"]) != 2 or preprocessing["score_cache_dtype"] != "float32":
        raise ValueError("confirmatory evaluation requires float32 score caches")
    inference = config["inference"]
    registration_config = config["registration"]
    thresholds = np.arange(
        float(config["operating_point"]["candidate_min"]),
        float(config["operating_point"]["candidate_max"]) + 1e-12,
        float(config["operating_point"]["candidate_step"]),
    )
    shuffled = _shuffled_group_map(
        rows, seed + int(config["controls"]["shuffle_seed_offset"])
    )
    rows_by_group = {str(row["source_group_id"]): row for row in rows}
    payloads = {
        name: {
            "forged": [],
            "authentic_vectors": [],
            "authentic_fpr_max": config["operating_point"]["authentic_pixel_fpr_max"],
        }
        for name in conditions
    }
    forged_scores: dict[str, dict[str, tuple[str, float]]] = {
        name: {} for name in conditions
    }
    image_cache: dict[str, np.ndarray] = {}

    def load_image(row: dict[str, Any], field: str, sha_field: str) -> np.ndarray:
        path = _resolve(scratch, row[field])
        key = str(path)
        if key not in image_cache:
            if _sha256(path) != row[sha_field]:
                raise ValueError(f"{field} SHA-256 changed")
            with Image.open(path) as handle:
                image_cache[key] = np.asarray(handle.convert("RGB"))
        return image_cache[key]

    predictions: list[dict[str, Any]] = []
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
            raise ValueError("confirmatory mask SHA-256 changed")
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0
        for condition_name, condition in conditions.items():
            for sample_kind, candidate_native in (("forged", forged_native), ("authentic", authentic_native)):
                prediction: dict[str, Any] = {
                    "record_id": f"{condition_name}:{sample_kind}:{group}",
                    "source_group_id": group,
                    "generator": generator,
                    "condition": condition_name,
                    "sample_kind": sample_kind,
                    "status": "failed",
                    "paper_evidence": False,
                    "confirmatory_development": True,
                    "final_reserve_read": False,
                }
                try:
                    candidate = _resize_image(candidate_native, int(preprocessing["max_side"]))
                    scorer = str(condition["scorer"])
                    geometry = str(condition["geometry"])
                    alignment_mode = str(condition["alignment"])
                    alignment_metadata: dict[str, Any] = {
                        "alignment_status": "not_requested",
                        "ecc_correlation": None,
                        "corner_errors_pixels": None,
                    }
                    alignment_key = None
                    reference = None
                    if scorer != "student":
                        if alignment_mode == "shuffled":
                            target = rows_by_group[shuffled[group]]
                            reference_native = load_image(
                                target, "authentic", "authentic_sha256"
                            )
                            reference = _resize_reference(
                                reference_native, candidate.shape[:2]
                            )
                        else:
                            clean_reference = _resize_reference(
                                authentic_native, candidate.shape[:2]
                            )
                            oracle = _stress_homography(
                                candidate.shape[:2], geometry, stresses
                            )
                            stressed_reference = _warp_reference(
                                clean_reference, oracle, inverse=False
                            )
                            if alignment_mode == "unaligned":
                                reference = stressed_reference
                            elif alignment_mode == "ecc":
                                candidate_sha = str(
                                    row["image_sha256"]
                                    if sample_kind == "forged"
                                    else row["authentic_sha256"]
                                )
                                alignment_key = hashlib.sha256(
                                    json.dumps(
                                        {
                                            "candidate_sha256": candidate_sha,
                                            "reference_sha256": row["authentic_sha256"],
                                            "candidate_shape": list(candidate.shape),
                                            "geometry": stresses.get(geometry, {"name": "clean"}),
                                            "registration": registration_config,
                                            "schema_version": preprocessing["alignment_cache_schema_version"],
                                        },
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    ).encode("utf-8")
                                ).hexdigest()
                                cached_alignment = alignment_cache_dir / geometry / f"{alignment_key}.npz"
                                cached_alignment.parent.mkdir(parents=True, exist_ok=True)
                                if cached_alignment.is_file():
                                    with np.load(cached_alignment, allow_pickle=False) as archive:
                                        reference = archive["aligned_reference"].astype(np.uint8)
                                        estimated = archive["estimated_homography"].astype(float)
                                        errors = archive["corner_errors_pixels"].astype(float)
                                        correlation = float(archive["ecc_correlation"].item())
                                        status = str(archive["alignment_status"].item())
                                        failure_type = str(archive["alignment_failure_type"].item()) or None
                                        failure_reason = str(archive["alignment_failure_reason"].item()) or None
                                        phase_response = float(archive["phase_correlation_response"].item())
                                    alignment_metadata = {
                                        "alignment_status": status,
                                        "ecc_correlation": correlation if np.isfinite(correlation) else None,
                                        "phase_correlation_response": phase_response if np.isfinite(phase_response) else None,
                                        "estimated_homography": estimated.tolist(),
                                        "corner_errors_pixels": errors.tolist(),
                                        "corner_error_mean_pixels": float(errors.mean()),
                                        "corner_error_max_pixels": float(errors.max()),
                                        "alignment_failure_type": failure_type,
                                        "alignment_failure_reason": failure_reason,
                                    }
                                    alignment_cache_hits += 1
                                else:
                                    reference, alignment_metadata = _estimate_ecc_alignment(
                                        candidate,
                                        stressed_reference,
                                        oracle,
                                        registration_config,
                                    )
                                    temporary = cached_alignment.with_suffix(".npz.tmp")
                                    with temporary.open("wb") as handle:
                                        np.savez_compressed(
                                            handle,
                                            aligned_reference=reference.astype(np.uint8),
                                            estimated_homography=np.asarray(alignment_metadata["estimated_homography"], dtype=float),
                                            corner_errors_pixels=np.asarray(alignment_metadata["corner_errors_pixels"], dtype=float),
                                            ecc_correlation=np.asarray(math.nan if alignment_metadata["ecc_correlation"] is None else alignment_metadata["ecc_correlation"]),
                                            phase_correlation_response=np.asarray(math.nan if alignment_metadata["phase_correlation_response"] is None else alignment_metadata["phase_correlation_response"]),
                                            alignment_status=np.asarray(alignment_metadata["alignment_status"]),
                                            alignment_failure_type=np.asarray(alignment_metadata["alignment_failure_type"] or ""),
                                            alignment_failure_reason=np.asarray(alignment_metadata["alignment_failure_reason"] or ""),
                                        )
                                    temporary.replace(cached_alignment)
                                alignment_records.setdefault(
                                    alignment_key,
                                    {
                                        "alignment_key": alignment_key,
                                        "source_group_id": group,
                                        "generator": generator,
                                        "sample_kind": sample_kind,
                                        "geometry": geometry,
                                        "alignment_cache": str(cached_alignment.relative_to(scratch)),
                                        **alignment_metadata,
                                    },
                                )
                            else:
                                raise ValueError(f"unsupported alignment mode: {alignment_mode}")
                    model_identity = (
                        model_config["student"]["checkpoint_sha256"]
                        if scorer == "student"
                        else pair_hashes["baseline"]
                        if scorer == "baseline"
                        else pair_hashes["robust"]
                        if scorer == "robust"
                        else "raw_difference_v1"
                    )
                    candidate_sha = str(
                        row["image_sha256"]
                        if sample_kind == "forged"
                        else row["authentic_sha256"]
                    )
                    score_key = hashlib.sha256(
                        json.dumps(
                            {
                                "candidate_sha256": candidate_sha,
                                "condition": condition,
                                "alignment_key": alignment_key,
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
                    if not score_path.is_file():
                        if scorer == "student":
                            probability = _infer_tiled(
                                student, candidate, device, inference, preprocessing
                            )
                        elif scorer in pair_models:
                            probability = _infer_pair_tiled(
                                pair_models[scorer],
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
                            np.savez_compressed(handle, scores=probability.astype(np.float32))
                        temporary.replace(score_path)
                    else:
                        score_cache_hits += 1
                    with np.load(score_path, allow_pickle=False) as archive:
                        probability = archive["scores"]
                    if probability.dtype != np.float32 or probability.shape != candidate.shape[:2] or not np.isfinite(probability).all():
                        raise ValueError("confirmatory score cache is invalid")
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
                        forged_scores[condition_name][group] = (generator, average_precision)
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
                            "scorer": scorer,
                            "geometry": geometry,
                            "alignment": alignment_mode,
                            "alignment_key": alignment_key,
                            "score_cache": str(score_path.relative_to(scratch)),
                            "score_cache_dtype": str(probability.dtype),
                            "score_shape": list(probability.shape),
                            "native_shape": list(native_probability.shape),
                            **alignment_metadata,
                        }
                    )
                except Exception as error:
                    failures += 1
                    prediction["failure_type"] = type(error).__name__
                    prediction["failure_reason"] = str(error)
                    logging.exception("record_id=%s failed", prediction["record_id"])
                predictions.append(prediction)
        _write_jsonl(predictions_path, predictions)
        _write_jsonl(alignments_path, list(alignment_records.values()))
        logging.info("completed_groups=%d total_groups=%d", row_index, len(rows))

    complete = failures == 0 and all(
        len(payload["forged"]) == len(rows)
        and len(payload["authentic_vectors"]) == len(rows)
        for payload in payloads.values()
    )
    if not complete:
        output = {
            "experiment": config["experiment"],
            "status": "failed_incomplete",
            "paper_evidence": False,
            "successful_prediction_records": len(predictions) - failures,
            "failed_prediction_records": failures,
            "final_reserve_read": False,
        }
        _write_json(summary_path, output)
        if runtime["require_all_records"]:
            raise RuntimeError(f"confirmatory evaluation failed for {failures} records")
        return output

    alignment_rows = list(alignment_records.values())
    errors = [
        float(error)
        for item in alignment_rows
        if item["geometry"] in STRESSES
        for error in item["corner_errors_pixels"]
    ]
    converged = sum(
        item["alignment_status"] == "ecc_converged" for item in alignment_rows
    )
    registration = {
        "attempts": len(alignment_rows),
        "converged": converged,
        "fallbacks": len(alignment_rows) - converged,
        "convergence_rate": converged / len(alignment_rows),
        "controlled_stress_corner_error_median_pixels": float(np.median(errors)),
        "controlled_stress_corner_error_p95_pixels": float(np.quantile(errors, 0.95)),
    }
    metrics = {
        name: _aggregate_condition(payload, thresholds)
        for name, payload in payloads.items()
    }
    decision = _confirmatory_decision(
        metrics,
        forged_scores,
        registration,
        config["confirmatory_gate"],
        config["bootstrap"],
    )
    _write_csv(metrics_path, [{"condition": name, **value} for name, value in metrics.items()])
    comparison_rows = []
    for name, value in decision["comparisons"].items():
        comparison_rows.append(
            {
                "comparison": name,
                "generator_macro_pixel_ap_difference": value["effect"],
                "ci_low": value["ci_low"],
                "ci_high": value["ci_high"],
                "bootstrap_resamples": int(config["bootstrap"]["resamples"]),
                "paper_evidence": False,
            }
        )
    _write_csv(comparisons_path, comparison_rows)
    output = {
        "experiment": config["experiment"],
        "status": (
            "resampling_robust_confirmatory_gate_passed"
            if decision["overall_pass"]
            else "resampling_robust_confirmatory_gate_failed"
        ),
        "paper_evidence": False,
        "confirmatory_development_read": True,
        "final_reserve_read": False,
        "method_change_performed": False,
        "candidate_method_frozen": bool(decision["overall_pass"]),
        "multi_seed_training_authorized": bool(decision["overall_pass"]),
        "final_reserve_evaluation_authorized": False,
        "qwen_generalization_estimable": False,
        "selected_groups": len(rows),
        "successful_prediction_records": len(predictions),
        "failed_prediction_records": 0,
        "score_cache_hits": score_cache_hits,
        "alignment_cache_hits": alignment_cache_hits,
        "score_cache_schema_version": 2,
        "score_cache_dtype": "float32",
        "conditions": metrics,
        "registration": registration,
        "bootstrap": config["bootstrap"],
        "confirmatory_gate": config["confirmatory_gate"],
        "decision": decision,
        "input_manifest_sha256": _sha256(manifest_path),
        "protocol_sha256": _sha256(protocol_path),
        "model_checkpoint_sha256": {
            "student": model_config["student"]["checkpoint_sha256"],
            **pair_hashes,
        },
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
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
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
