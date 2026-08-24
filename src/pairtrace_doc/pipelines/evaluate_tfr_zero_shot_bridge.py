from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from collections import defaultdict
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
from PIL import Image

from pairtrace_doc.pipelines.diagnose_pair_at_inference_alignment import (
    _estimate_ecc_alignment,
    _stress_homography,
    _warp_reference,
)
from pairtrace_doc.pipelines.evaluate_baselines_100 import _roc_auc
from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
    _infer_pair_tiled,
    _raw_difference,
    _resize_image,
    _resize_reference,
    _shuffled_group_map,
)
from pairtrace_doc.pipelines.freeze_resampling_multiseed_image_thresholds import (
    _top_fraction_mean,
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
from pairtrace_doc.pipelines.train_tfr_equal_budget_arm import _load_config


GEOMETRIES = ("clean", "translation", "affine", "perspective")
ROBUST_MODELS = ("robust_20260747", "robust_20260763", "robust_20260764")
LEARNED_MODELS = ("clean_teacher", *ROBUST_MODELS)


def _required_conditions() -> set[str]:
    result = {
        f"{scorer}_{geometry}_ecc"
        for scorer in ("clean_teacher", "raw_difference", *ROBUST_MODELS)
        for geometry in GEOMETRIES
    }
    result.update(f"{scorer}_shuffled_clean" for scorer in ROBUST_MODELS)
    return result


def _binary_metrics(scores: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = np.asarray(scores) >= threshold
    target = np.asarray(mask, dtype=bool)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "pixel_iou": float(tp / max(1, tp + fp + fn)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _source_group_macro(rows: list[dict[str, Any]], field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group_id"])].append(float(row[field]))
    if not grouped:
        raise ValueError("source-group macro metric requires records")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _group_values(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group_id"])].append(float(row[field]))
    return {group: float(np.mean(values)) for group, values in grouped.items()}


def _aggregate_condition(
    payload: dict[str, list[dict[str, Any]]],
    pixel_threshold: float,
    image_threshold: float,
    paper_evidence: bool = False,
) -> dict[str, Any]:
    forged = payload["forged"]
    authentic = payload["authentic"]
    if not forged or not authentic:
        raise ValueError("bridge condition is incomplete")
    forged_image = np.asarray([row["image_score"] for row in forged], dtype=float)
    authentic_image = np.asarray([row["image_score"] for row in authentic], dtype=float)
    return {
        "forged_pairs": len(forged),
        "forged_source_groups": len({str(row["source_group_id"]) for row in forged}),
        "authentic_source_groups": len(authentic),
        "source_group_macro_pixel_ap": _source_group_macro(forged, "pixel_ap"),
        "source_group_macro_pixel_auroc": _source_group_macro(forged, "pixel_auroc"),
        "source_group_macro_pixel_precision": _source_group_macro(forged, "pixel_precision"),
        "source_group_macro_pixel_recall": _source_group_macro(forged, "pixel_recall"),
        "source_group_macro_pixel_f1": _source_group_macro(forged, "pixel_f1"),
        "source_group_macro_pixel_iou": _source_group_macro(forged, "pixel_iou"),
        "unique_authentic_group_macro_pixel_fpr": float(
            np.mean([row["pixel_fpr"] for row in authentic])
        ),
        "image_auroc": _roc_auc(
            np.r_[forged_image, authentic_image],
            np.r_[
                np.ones(forged_image.size, dtype=bool),
                np.zeros(authentic_image.size, dtype=bool),
            ],
        ),
        "image_tpr_at_aiforge_frozen_threshold": float(
            np.mean(forged_image >= image_threshold)
        ),
        "image_fpr_at_aiforge_frozen_threshold": float(
            np.mean(authentic_image >= image_threshold)
        ),
        "pixel_threshold_frozen_on_aiforge": pixel_threshold,
        "image_threshold_frozen_on_aiforge": image_threshold,
        "threshold_selected_on_tfr": False,
        "paper_evidence": paper_evidence,
    }


def _paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("bridge paired comparison source groups differ")
    groups = sorted(left)
    differences = np.asarray([left[group] - right[group] for group in groups], dtype=float)
    rng = np.random.default_rng(seed)
    replicas = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected = rng.integers(0, len(groups), size=len(groups))
        replicas[index] = float(differences[selected].mean())
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(replicas, [alpha, 1.0 - alpha])
    return {
        "effect": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence_level": confidence_level,
        "resamples": resamples,
        "source_groups": len(groups),
    }


def _mean_bootstrap(
    values: dict[str, float],
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not values:
        raise ValueError("mean bootstrap requires source groups")
    groups = sorted(values)
    observed = np.asarray([values[group] for group in groups], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(resamples, len(groups)))
    replicas = observed[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(replicas, [alpha, 1.0 - alpha])
    return {
        "estimate": float(observed.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence_level": confidence_level,
        "resamples": resamples,
        "source_groups": len(groups),
    }


def _bridge_decision(
    metrics: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    registration_convergence_rate: float,
) -> dict[str, Any]:
    clean_values = [
        float(metrics[f"{model}_clean_ecc"]["source_group_macro_pixel_ap"])
        for model in ROBUST_MODELS
    ]
    minimum_stressed = [
        min(
            float(metrics[f"{model}_{geometry}_ecc"]["source_group_macro_pixel_ap"])
            for geometry in GEOMETRIES
            if geometry != "clean"
        )
        for model in ROBUST_MODELS
    ]
    clean_fpr = [
        float(metrics[f"{model}_clean_ecc"]["unique_authentic_group_macro_pixel_fpr"])
        for model in ROBUST_MODELS
    ]
    correct_minus_shuffled = [
        float(metrics[f"{model}_clean_ecc"]["source_group_macro_pixel_ap"])
        - float(metrics[f"{model}_shuffled_clean"]["source_group_macro_pixel_ap"])
        for model in ROBUST_MODELS
    ]
    values = {
        "mean_clean_source_group_macro_ap": float(np.mean(clean_values)),
        "minimum_seed_clean_source_group_macro_ap": float(min(clean_values)),
        "mean_seed_minimum_stressed_source_group_macro_ap": float(
            np.mean(minimum_stressed)
        ),
        "mean_clean_authentic_pixel_fpr": float(np.mean(clean_fpr)),
        "mean_correct_minus_shuffled_clean_ap": float(np.mean(correct_minus_shuffled)),
        "registration_convergence_rate": float(registration_convergence_rate),
    }
    checks = {
        "mean_clean_ap_floor": values["mean_clean_source_group_macro_ap"]
        >= float(gate["mean_clean_source_group_macro_ap_min"]),
        "minimum_seed_clean_ap_floor": values[
            "minimum_seed_clean_source_group_macro_ap"
        ]
        >= float(gate["minimum_seed_clean_source_group_macro_ap_min"]),
        "mean_minimum_stressed_ap_floor": values[
            "mean_seed_minimum_stressed_source_group_macro_ap"
        ]
        >= float(gate["mean_seed_minimum_stressed_source_group_macro_ap_min"]),
        "mean_clean_authentic_fpr_ceiling": values[
            "mean_clean_authentic_pixel_fpr"
        ]
        <= float(gate["mean_clean_authentic_pixel_fpr_max"]),
        "mean_correct_minus_shuffled_floor": values[
            "mean_correct_minus_shuffled_clean_ap"
        ]
        >= float(gate["mean_correct_minus_shuffled_clean_ap_min"]),
        "registration_convergence_floor": registration_convergence_rate
        >= float(gate["registration_convergence_rate_min"]),
    }
    return {"values": values, "checks": checks, "overall_pass": all(checks.values())}


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = _load_config(config_path)
    runtime = config["runtime"]
    stage = str(config["experiment"]["stage"])
    one_shot_holdout = stage == "one_shot_holdout"
    paper_evidence = bool(config["experiment"]["paper_evidence"])
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not runtime["gpu_launch_authorized"]:
        raise ValueError("TFR bridge GPU evaluation was not authorized")
    if runtime["model_training_authorized"]:
        raise ValueError("TFR bridge cannot authorize model training")
    if one_shot_holdout:
        if not paper_evidence or not runtime["tfr_holdout_read_allowed"]:
            raise ValueError("TFR one-shot holdout evidence boundary is not authorized")
    elif runtime["tfr_holdout_read_allowed"] or paper_evidence:
        raise ValueError("TFR viewed-development bridge crossed its evidence boundary")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TFR bridge requires CUDA")
    torch.cuda.set_device(device)

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("TFR bridge protocol SHA-256 changed")
    input_config = config["input"]
    manifest_path = _resolve(project_root, input_config["manifest"])
    if _sha256(manifest_path) != input_config["expected_manifest_sha256"]:
        raise ValueError("TFR bridge manifest SHA-256 changed")
    materialization_summary: dict[str, Any] | None = None
    if one_shot_holdout:
        materialization_path = _resolve(
            project_root, input_config["materialization_summary"]
        )
        if (
            _sha256(materialization_path)
            != input_config["expected_materialization_summary_sha256"]
        ):
            raise ValueError("TFR holdout materialization summary changed")
        materialization_summary = json.loads(
            materialization_path.read_text(encoding="utf-8")
        )
        if (
            materialization_summary["status"] != "tfr_one_shot_holdout_materialized"
            or materialization_summary["outputs"]["pair_manifest_sha256"]
            != input_config["expected_manifest_sha256"]
        ):
            raise ValueError("TFR holdout materialization is incomplete")
    all_rows = _read_jsonl(manifest_path)
    if {str(row["freeze_id"]) for row in all_rows} != {
        str(input_config["expected_freeze_id"])
    }:
        raise ValueError("TFR bridge split freeze ID changed")
    rows = sorted(
        [row for row in all_rows if row["pilot_role"] == input_config["role"]],
        key=lambda row: (str(row["source_group_id"]), str(row["sample_id"])),
    )
    if len(rows) != int(input_config["expected_pairs"]):
        raise ValueError("TFR bridge validation pair count changed")
    if len({str(row["source_group_id"]) for row in rows}) != int(
        input_config["expected_source_groups"]
    ):
        raise ValueError("TFR bridge validation source-group count changed")
    if input_config.get("max_pairs") is not None:
        rows = rows[: int(input_config["max_pairs"])]
    if len({str(row["source_group_id"]) for row in rows}) < 2:
        raise ValueError("TFR bridge requires at least two source groups")

    condition_specs = {str(item["name"]): item for item in config["conditions"]}
    if set(condition_specs) != _required_conditions():
        raise ValueError("TFR bridge condition whitelist changed")
    if set(config["stresses"]) != set(GEOMETRIES):
        raise ValueError("TFR bridge stress whitelist changed")

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
    encoder_path = _resolve(scratch, config["models"]["encoder_weights"])
    if _sha256(encoder_path) != config["models"]["encoder_weights_sha256"]:
        raise ValueError("TFR bridge encoder initialization changed")
    models: dict[str, torch.nn.Module] = {}
    model_hashes: dict[str, str] = {}
    for model_name in LEARNED_MODELS:
        model_spec = config["models"][model_name]
        checkpoint_path = _resolve(project_root, model_spec["checkpoint"])
        expected = str(model_spec["checkpoint_sha256"])
        if _sha256(checkpoint_path) != expected:
            raise ValueError(f"TFR bridge checkpoint changed: {model_name}")
        model = _load_teacher(
            encoder_path, config["models"]["teacher_conv1_coefficients"]
        )
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state"], strict=True)
        models[model_name] = model.to(device).eval().requires_grad_(False)
        model_hashes[model_name] = expected

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

    representatives: dict[str, dict[str, Any]] = {}
    for row in rows:
        representatives.setdefault(str(row["source_group_id"]), row)
    representative_rows = list(representatives.values())
    shuffled = _shuffled_group_map(
        representative_rows, seed + int(config["controls"]["shuffle_seed_offset"])
    )
    image_cache: dict[str, np.ndarray] = {}

    def load_image(row: dict[str, Any], field: str, sha_field: str) -> np.ndarray:
        path = _resolve(scratch, row[field])
        key = str(path)
        if key not in image_cache:
            if _sha256(path) != row[sha_field]:
                raise ValueError(f"TFR bridge {field} SHA-256 changed")
            with Image.open(path) as handle:
                image_cache[key] = np.asarray(handle.convert("RGB"))
        return image_cache[key]

    payloads = {name: {"forged": [], "authentic": []} for name in condition_specs}
    prediction_rows: list[dict[str, Any]] = []
    alignment_records: dict[str, dict[str, Any]] = {}
    score_cache_hits = 0
    alignment_cache_hits = 0
    failures = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    tasks: list[tuple[str, dict[str, Any], str]] = []
    for row in rows:
        tasks.append(("forged", row, str(row["sample_id"])))
    for group, row in sorted(representatives.items()):
        tasks.append(("authentic", row, f"{group}:authentic"))

    for task_index, (sample_kind, row, sample_id) in enumerate(tasks, start=1):
        group = str(row["source_group_id"])
        if sample_kind == "forged":
            candidate_native = load_image(row, "image", "image_sha256")
            candidate_sha = str(row["image_sha256"])
            mask_path = _resolve(scratch, row["mask"])
            if _sha256(mask_path) != row["mask_sha256"]:
                raise ValueError("TFR bridge mask SHA-256 changed")
            with Image.open(mask_path) as handle:
                native_mask = np.asarray(handle.convert("L")) > 0
        else:
            candidate_native = load_image(row, "authentic", "authentic_sha256")
            candidate_sha = str(row["authentic_sha256"])
            native_mask = None
        reference_native = load_image(row, "authentic", "authentic_sha256")
        candidate = _resize_image(candidate_native, int(config["preprocessing"]["max_side"]))
        clean_reference = _resize_reference(reference_native, candidate.shape[:2])
        aligned_by_geometry: dict[str, tuple[np.ndarray, str]] = {}

        def aligned_reference(geometry: str) -> tuple[np.ndarray, str]:
            nonlocal alignment_cache_hits
            if geometry in aligned_by_geometry:
                return aligned_by_geometry[geometry]
            oracle = _stress_homography(
                candidate.shape[:2], geometry, config["stresses"]
            )
            stressed = _warp_reference(clean_reference, oracle, inverse=False)
            key = hashlib.sha256(
                json.dumps(
                    {
                        "schema": config["preprocessing"]["alignment_cache_schema_version"],
                        "candidate_sha256": candidate_sha,
                        "reference_sha256": row["authentic_sha256"],
                        "candidate_shape": list(candidate.shape),
                        "geometry": config["stresses"][geometry],
                        "registration": config["registration"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            cache_path = alignment_cache_dir / geometry / f"{key}.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as archive:
                    aligned = archive["aligned_reference"].astype(np.uint8)
                    metadata = json.loads(str(archive["metadata_json"].item()))
                alignment_cache_hits += 1
            else:
                aligned, metadata = _estimate_ecc_alignment(
                    candidate, stressed, oracle, config["registration"]
                )
                _save_npz(
                    cache_path,
                    aligned_reference=aligned.astype(np.uint8),
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
            alignment_records.setdefault(
                key,
                {
                    "alignment_key": key,
                    "sample_id": sample_id,
                    "source_group_id": group,
                    "sample_kind": sample_kind,
                    "geometry": geometry,
                    "alignment_cache": str(cache_path.relative_to(scratch)),
                    "paper_evidence": paper_evidence,
                    "tfr_holdout_read": one_shot_holdout,
                    **metadata,
                },
            )
            aligned_by_geometry[geometry] = (aligned, key)
            return aligned, key

        for condition_name, condition in condition_specs.items():
            record: dict[str, Any] = {
                "record_id": f"{condition_name}:{sample_id}",
                "sample_id": sample_id,
                "source_group_id": group,
                "sample_kind": sample_kind,
                "condition": condition_name,
                "scorer": condition["scorer"],
                "geometry": condition["geometry"],
                "alignment": condition["alignment"],
                "status": "failed",
                "paper_evidence": paper_evidence,
                "viewed_development": not one_shot_holdout,
                "tfr_holdout_read": one_shot_holdout,
                "threshold_selected_on_tfr": False,
            }
            try:
                scorer = str(condition["scorer"])
                if condition["alignment"] == "ecc":
                    reference, alignment_key = aligned_reference(str(condition["geometry"]))
                    reference_token = alignment_key
                elif condition["alignment"] == "shuffled":
                    target = representatives[shuffled[group]]
                    target_reference = load_image(target, "authentic", "authentic_sha256")
                    reference = _resize_reference(target_reference, candidate.shape[:2])
                    alignment_key = None
                    reference_token = f"shuffled:{target['authentic_sha256']}"
                else:
                    raise ValueError("unsupported TFR bridge alignment mode")
                model_identity = (
                    config["models"]["raw_difference"]["implementation"]
                    if scorer == "raw_difference"
                    else model_hashes[scorer]
                )
                score_key = hashlib.sha256(
                    json.dumps(
                        {
                            "schema": config["preprocessing"]["score_cache_schema_version"],
                            "candidate_sha256": candidate_sha,
                            "reference_token": reference_token,
                            "condition": condition,
                            "model_identity": model_identity,
                            "preprocessing": config["preprocessing"],
                            "inference": config["inference"],
                            "sample_kind": sample_kind,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                score_path = score_cache_dir / condition_name / f"{score_key}.npz"
                if score_path.is_file():
                    with np.load(score_path, allow_pickle=False) as archive:
                        probability = archive["scores"].astype(np.float32)
                    score_cache_hits += 1
                else:
                    probability = (
                        _raw_difference(candidate, reference)
                        if scorer == "raw_difference"
                        else _infer_pair_tiled(
                            models[scorer],
                            candidate,
                            reference,
                            device,
                            config["inference"],
                            config["preprocessing"],
                        )
                    ).astype(np.float32)
                    _save_npz(score_path, scores=probability)
                if probability.shape != candidate.shape[:2] or not np.isfinite(probability).all():
                    raise ValueError("TFR bridge score cache is invalid")
                if sample_kind == "forged":
                    assert native_mask is not None
                    native_scores = cv2.resize(
                        probability,
                        (native_mask.shape[1], native_mask.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    native_scores = cv2.resize(
                        probability,
                        (candidate_native.shape[1], candidate_native.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                pixel_threshold = float(config["models"][scorer]["pixel_threshold"])
                image_threshold = float(config["models"][scorer]["image_threshold"])
                image_score = _top_fraction_mean(
                    native_scores, float(config["image_score"]["top_fraction"])
                )
                if sample_kind == "forged":
                    average_precision, auroc = _ranking_metrics(native_scores, native_mask)
                    binary = _binary_metrics(native_scores, native_mask, pixel_threshold)
                    item = {
                        "source_group_id": group,
                        "pixel_ap": average_precision,
                        "pixel_auroc": auroc,
                        "image_score": image_score,
                        **binary,
                    }
                    payloads[condition_name]["forged"].append(item)
                    record.update(item)
                else:
                    pixel_fpr = float(np.mean(native_scores >= pixel_threshold))
                    item = {
                        "source_group_id": group,
                        "pixel_fpr": pixel_fpr,
                        "image_score": image_score,
                    }
                    payloads[condition_name]["authentic"].append(item)
                    record.update(item)
                record.update(
                    {
                        "status": "ok",
                        "score_cache": str(score_path.relative_to(scratch)),
                        "score_shape": list(probability.shape),
                        "checkpoint_sha256": None
                        if scorer == "raw_difference"
                        else model_hashes[scorer],
                        "alignment_key": alignment_key,
                        "pixel_threshold_frozen_on_aiforge": pixel_threshold,
                        "image_threshold_frozen_on_aiforge": image_threshold,
                    }
                )
            except Exception as error:
                failures += 1
                record["failure_type"] = type(error).__name__
                record["failure_reason"] = str(error)
                logging.exception("record_id=%s failed", record["record_id"])
            prediction_rows.append(record)
        logging.info("completed_tasks=%d total_tasks=%d", task_index, len(tasks))

    _write_jsonl(predictions_path, prediction_rows)
    _write_jsonl(alignments_path, alignment_records.values())
    expected_forged = len(rows)
    expected_authentic = len(representatives)
    complete = failures == 0 and all(
        len(payload["forged"]) == expected_forged
        and len(payload["authentic"]) == expected_authentic
        for payload in payloads.values()
    )
    if not complete and runtime["require_all_records"]:
        raise RuntimeError("TFR bridge did not complete every requested record")

    metrics: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    for condition_name, payload in sorted(payloads.items()):
        scorer = str(condition_specs[condition_name]["scorer"])
        result = _aggregate_condition(
            payload,
            float(config["models"][scorer]["pixel_threshold"]),
            float(config["models"][scorer]["image_threshold"]),
            paper_evidence,
        )
        metrics[condition_name] = result
        metric_rows.append({"condition": condition_name, **result})
    _write_csv(metrics_path, metric_rows)

    bootstrap = config["bootstrap"]
    comparisons: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    comparison_index = 0
    for model_name in ROBUST_MODELS:
        names = {
            "correct_minus_shuffled_clean": (
                f"{model_name}_clean_ecc",
                f"{model_name}_shuffled_clean",
            )
        }
        names.update(
            {
                f"robust_minus_clean_teacher_{geometry}": (
                    f"{model_name}_{geometry}_ecc",
                    f"clean_teacher_{geometry}_ecc",
                )
                for geometry in GEOMETRIES
            }
        )
        for label, (left_name, right_name) in names.items():
            key = f"{model_name}__{label}"
            result = _paired_bootstrap(
                _group_values(payloads[left_name]["forged"], "pixel_ap"),
                _group_values(payloads[right_name]["forged"], "pixel_ap"),
                int(bootstrap["seed"]) + comparison_index,
                int(bootstrap["resamples"]),
                float(bootstrap["confidence_level"]),
            )
            comparisons[key] = result
            comparison_rows.append(
                {"comparison": key, "left": left_name, "right": right_name, **result}
            )
            comparison_index += 1
    _write_csv(comparisons_path, comparison_rows)

    alignment_values = list(alignment_records.values())
    convergence_rate = float(
        np.mean([row["alignment_status"] == "ecc_converged" for row in alignment_values])
    )
    decision = None if stage == "preflight" else _bridge_decision(
        metrics, config["full_gate"], convergence_rate
    )
    if stage == "preflight":
        status = "preflight_complete"
    elif one_shot_holdout:
        status = (
            "tfr_holdout_gate_passed"
            if decision is not None and decision["overall_pass"]
            else "tfr_holdout_gate_failed"
        )
    else:
        status = (
            "bridge_gate_passed"
            if decision is not None and decision["overall_pass"]
            else "bridge_gate_failed"
        )
    primary_endpoint = None
    if one_shot_holdout:
        seed_group_values = {
            model: _group_values(
                payloads[f"{model}_clean_ecc"]["forged"], "pixel_ap"
            )
            for model in ROBUST_MODELS
        }
        if any(set(values) != set(seed_group_values[ROBUST_MODELS[0]]) for values in seed_group_values.values()):
            raise ValueError("TFR holdout primary endpoint seed groups differ")
        three_seed_group_values = {
            group: float(
                np.mean([seed_group_values[model][group] for model in ROBUST_MODELS])
            )
            for group in sorted(seed_group_values[ROBUST_MODELS[0]])
        }
        primary_endpoint = {
            "name": "three_robust_seed_mean_clean_source_group_macro_pixel_ap",
            **_mean_bootstrap(
                three_seed_group_values,
                int(bootstrap["seed"]),
                int(bootstrap["resamples"]),
                float(bootstrap["confidence_level"]),
            ),
            "paper_evidence": True,
            "selected_on_holdout": False,
        }
    summary = {
        "status": status,
        "experiment": config["experiment"],
        "paper_evidence": paper_evidence,
        "model_training_performed": False,
        "tfr_holdout_read": one_shot_holdout,
        "tfr_threshold_selection_performed": False,
        "validation_pairs": expected_forged,
        "validation_source_groups": expected_authentic,
        "evaluation_pairs": expected_forged,
        "evaluation_source_groups": expected_authentic,
        "conditions": len(condition_specs),
        "requested_predictions": len(condition_specs)
        * (expected_forged + expected_authentic),
        "completed_predictions": sum(row["status"] == "ok" for row in prediction_rows),
        "failed_predictions": failures,
        "alignment_records": len(alignment_values),
        "registration_convergence_rate": convergence_rate,
        "registration_corner_error_p95_pixels": float(
            np.quantile(
                [
                    error
                    for row in alignment_values
                    if row["geometry"] != "clean"
                    for error in row["corner_errors_pixels"]
                ],
                0.95,
            )
        ),
        "score_cache_hits": score_cache_hits,
        "alignment_cache_hits": alignment_cache_hits,
        "decision": decision,
        "primary_endpoint": primary_endpoint,
        "metrics": metrics,
        "comparisons": comparisons,
        "protocol_sha256": _sha256(protocol_path),
        "manifest_sha256": _sha256(manifest_path),
        "holdout_membership_sha256": input_config[
            "expected_holdout_membership_sha256"
        ],
        "holdout_membership_read_by_materializer": one_shot_holdout,
        "materialization_summary_sha256": (
            None
            if materialization_summary is None
            else input_config["expected_materialization_summary_sha256"]
        ),
        "predictions_sha256": _sha256(predictions_path),
        "alignments_sha256": _sha256(alignments_path),
        "metrics_sha256": _sha256(metrics_path),
        "comparisons_sha256": _sha256(comparisons_path),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "wall_time_seconds": float(time.monotonic() - started),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen AIForge PairTrace models on TFR")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
