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
from pairtrace_doc.pipelines.evaluate_reference_integrity_viewed20 import (
    _align_reference,
    _threshold_metrics,
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


def _cell_macro(rows: list[dict[str, Any]], value_field: str) -> float:
    by_cell: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cell[str(row["attack_device_cell"])].append(float(row[value_field]))
    if not by_cell:
        raise ValueError("attack-device macro requires at least one represented cell")
    return float(np.mean([np.mean(values) for values in by_cell.values()]))


def _grouped_macro(
    rows: list[dict[str, Any]], group_field: str, value_field: str
) -> dict[str, float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_field])].append(float(row[value_field]))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _aggregate_seed(
    prediction_rows: list[dict[str, Any]],
    same_device_rows: list[dict[str, Any]],
    model_name: str,
    training_seed: int,
    groups: set[str],
) -> dict[str, Any]:
    selected = [
        row
        for row in prediction_rows
        if row["model"] == model_name and row["status"] == "ok"
    ]
    forged = [row for row in selected if row["sample_kind"] == "forged"]
    authentic = [row for row in selected if row["sample_kind"] == "authentic"]
    same_forged = [
        row
        for row in same_device_rows
        if row.get("condition") == "robust_teacher_correct"
        and row.get("sample_kind") == "forged"
        and row.get("status") == "ok"
        and str(row["source_group_id"]) in groups
    ]
    same_authentic = [
        row
        for row in same_device_rows
        if row.get("condition") == "robust_teacher_correct"
        and row.get("sample_kind") == "authentic"
        and row.get("status") == "ok"
        and str(row["source_group_id"]) in groups
    ]
    if not (
        len(forged)
        == len(authentic)
        == len(same_forged)
        == len(same_authentic)
        == len(groups)
    ):
        raise ValueError(f"cross/same-device seed aggregation incomplete: {model_name}")
    cross_macro = _cell_macro(forged, "weak_box_mask_ap")
    same_macro = _cell_macro(same_forged, "macro_box_mask_ap")
    return {
        "model": model_name,
        "training_seed": training_seed,
        "groups": len(groups),
        "cross_device_attack_device_macro_weak_box_ap": cross_macro,
        "same_device_attack_device_macro_weak_box_ap": same_macro,
        "cross_device_ap_retention": cross_macro / same_macro,
        "cross_device_document_macro_weak_box_ap": float(
            np.mean([row["weak_box_mask_ap"] for row in forged])
        ),
        "cross_device_document_macro_weak_box_auroc": float(
            np.mean([row["weak_box_mask_auroc"] for row in forged])
        ),
        "cross_device_document_macro_pixel_f1": float(
            np.mean([row["pixel_f1"] for row in forged])
        ),
        "cross_device_document_macro_pixel_iou": float(
            np.mean([row["pixel_iou"] for row in forged])
        ),
        "cross_device_authentic_document_macro_pixel_fpr": float(
            np.mean([row["authentic_pixel_fpr"] for row in authentic])
        ),
        "same_device_authentic_document_macro_pixel_fpr": float(
            np.mean(
                [row["authentic_pixel_fpr_at_fixed_threshold"] for row in same_authentic]
            )
        ),
        "per_attack_cross_device_macro_weak_box_ap": _grouped_macro(
            forged, "attack_method", "weak_box_mask_ap"
        ),
        "per_candidate_device_cross_device_macro_weak_box_ap": _grouped_macro(
            forged, "candidate_device", "weak_box_mask_ap"
        ),
        "paper_evidence": False,
        "threshold_selection_used": False,
    }


def _aggregate_across_seeds(seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "cross_device_attack_device_macro_weak_box_ap",
        "same_device_attack_device_macro_weak_box_ap",
        "cross_device_ap_retention",
        "cross_device_document_macro_weak_box_ap",
        "cross_device_document_macro_weak_box_auroc",
        "cross_device_document_macro_pixel_f1",
        "cross_device_document_macro_pixel_iou",
        "cross_device_authentic_document_macro_pixel_fpr",
        "same_device_authentic_document_macro_pixel_fpr",
    )
    result: dict[str, Any] = {"seed_count": len(seed_metrics)}
    for field in fields:
        values = [float(row[field]) for row in seed_metrics]
        result[field] = {
            "mean": float(np.mean(values)),
            "sample_standard_deviation": _sample_sd(values),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return result


def _gate_decision(
    stage: str,
    expected_alignments: int,
    expected_predictions: int,
    alignment_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    successful_alignments = sum(row["status"] == "ok" for row in alignment_rows)
    successful_predictions = sum(row["status"] == "ok" for row in prediction_rows)
    engineering = {
        "all_alignments_complete": successful_alignments == expected_alignments,
        "all_predictions_complete": successful_predictions == expected_predictions,
        "zero_failed_alignments": successful_alignments == len(alignment_rows),
        "zero_failed_predictions": successful_predictions == len(prediction_rows),
        "zero_final_reserve_reads": True,
    }
    if stage == "toy3":
        return {
            "stage": stage,
            "engineering_checks": engineering,
            "scientific_gate_evaluated": False,
            "overall_pass": all(engineering.values()),
        }
    ap = aggregate["cross_device_attack_device_macro_weak_box_ap"]
    retention = aggregate["cross_device_ap_retention"]
    fpr = aggregate["cross_device_authentic_document_macro_pixel_fpr"]
    converged = [
        row["alignment_status"] == "ecc_converged"
        for row in alignment_rows
        if row["status"] == "ok"
    ]
    checks = {
        **engineering,
        "mean_cross_device_ap_floor": float(ap["mean"])
        >= float(gate["mean_attack_device_macro_ap_min"]),
        "minimum_seed_cross_device_ap_floor": float(ap["minimum"])
        >= float(gate["minimum_seed_attack_device_macro_ap_min"]),
        "mean_ap_retention_floor": float(retention["mean"])
        >= float(gate["mean_ap_retention_min"]),
        "mean_authentic_fpr_ceiling": float(fpr["mean"])
        <= float(gate["mean_authentic_pixel_fpr_max"]),
        "registration_convergence_floor": float(np.mean(converged))
        >= float(gate["registration_convergence_min"]),
    }
    return {
        "stage": stage,
        "checks": checks,
        "scientific_gate_evaluated": True,
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
    if not bool(runtime["gpu_launch_authorized"]) or not bool(
        runtime["cross_device_evaluation_authorized"]
    ):
        raise ValueError("cross-device evaluation was not authorized")
    if not bool(runtime["viewed_development_read_allowed"]):
        raise ValueError("cross-device viewed development was not authorized")
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
        raise ValueError("cross-device evaluation crossed an evidence boundary")
    if bool(config["experiment"]["paper_evidence"]):
        raise ValueError("cross-device diagnostic cannot be final paper evidence")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("cross-device evaluation requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != str(
        config["experiment"]["expected_protocol_sha256"]
    ):
        raise ValueError("cross-device protocol changed")
    inputs = config["input"]
    manifest_path = _resolve(project_root, inputs["manifest"])
    if _sha256(manifest_path) != str(inputs["expected_manifest_sha256"]):
        raise ValueError("cross-device manifest changed")
    rows = sorted(
        _read_jsonl(manifest_path), key=lambda row: int(row["selection_index"])
    )
    expected_groups = int(inputs["expected_groups"])
    if len(rows) != expected_groups or len(
        {str(row["source_group_id"]) for row in rows}
    ) != expected_groups:
        raise ValueError("cross-device group topology changed")
    if any(
        row.get("paper_evidence") is not False
        or row.get("cross_device_materialization_status") != "ok"
        or row.get("cross_device_reference_read") is not True
        or row.get("device") == row.get("cross_device_reference_device")
        for row in rows
    ):
        raise ValueError("cross-device input boundary changed")
    cell_counts = Counter(f"{row['attack_method']}|{row['device']}" for row in rows)
    if dict(sorted(cell_counts.items())) != {
        str(key): int(value)
        for key, value in inputs["expected_attack_device_counts"].items()
    }:
        raise ValueError("cross-device attack-device balance changed")

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
    registration = config["registration"]
    inference = config["inference"]
    model_config = config["models"]
    family_seeds = [int(value) for value in model_config["family_seeds"]]
    if family_seeds != [20260747, 20260763, 20260764]:
        raise ValueError("cross-device model family changed")
    encoder_path = _resolve(scratch, model_config["encoder_weights"])
    if _sha256(encoder_path) != str(model_config["encoder_weights_sha256"]):
        raise ValueError("cross-device encoder weights changed")
    models: dict[str, torch.nn.Module] = {}
    model_hashes: dict[str, str] = {}
    thresholds: dict[str, float] = {}
    same_device_sources: dict[str, list[dict[str, Any]]] = {}
    for training_seed in family_seeds:
        name = f"robust_{training_seed}"
        item = model_config[name]
        checkpoint = _resolve(project_root, item["checkpoint"])
        if _sha256(checkpoint) != str(item["checkpoint_sha256"]):
            raise ValueError(f"cross-device checkpoint changed: {name}")
        model = _load_teacher(
            encoder_path, model_config["teacher_conv1_coefficients"]
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state"], strict=True)
        models[name] = model.to(device).eval().requires_grad_(False)
        model_hashes[name] = str(item["checkpoint_sha256"])
        thresholds[name] = float(item["fixed_pixel_threshold"])
        same_path = _resolve(project_root, item["same_device_predictions"])
        if _sha256(same_path) != str(item["same_device_predictions_sha256"]):
            raise ValueError(f"same-device prediction source changed: {name}")
        same_device_sources[name] = _read_jsonl(same_path)

    image_cache: dict[str, np.ndarray] = {}

    def load_image(row: dict[str, Any], field: str, sha_field: str) -> np.ndarray:
        path = _resolve(scratch, row[field])
        key = str(path)
        if key not in image_cache:
            if _sha256(path) != str(row[sha_field]):
                raise ValueError(f"cross-device {field} SHA-256 changed")
            with Image.open(path) as handle:
                image_cache[key] = np.asarray(handle.convert("RGB"))
        return image_cache[key]

    prediction_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    score_cache_hits = 0
    alignment_cache_hits = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    for row_index, row in enumerate(rows, start=1):
        group = str(row["source_group_id"])
        forged_native = load_image(row, "image", "image_sha256")
        authentic_native = load_image(row, "authentic", "authentic_sha256")
        reference_native = load_image(
            row, "cross_device_reference", "cross_device_reference_sha256"
        )
        mask_path = _resolve(scratch, row["mask"])
        if _sha256(mask_path) != str(row["mask_sha256"]):
            raise ValueError("cross-device weak mask changed")
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0
        if forged_native.shape[:2] != native_mask.shape:
            raise ValueError("cross-device candidate/mask geometry changed")

        for sample_kind, candidate_native, candidate_sha256 in (
            ("forged", forged_native, str(row["image_sha256"])),
            ("authentic", authentic_native, str(row["authentic_sha256"])),
        ):
            candidate = _resize_image(candidate_native, int(preprocessing["max_side"]))
            reference = _resize_image(reference_native, int(preprocessing["max_side"]))
            if reference.shape != candidate.shape:
                reference = cv2.resize(
                    reference,
                    (candidate.shape[1], candidate.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            alignment_key = hashlib.sha256(
                json.dumps(
                    {
                        "candidate_sha256": candidate_sha256,
                        "reference_sha256": row["cross_device_reference_sha256"],
                        "registration": registration,
                        "preprocessing_max_side": preprocessing["max_side"],
                        "schema": 1,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            alignment_path = alignment_cache_dir / f"{alignment_key}.npz"
            alignment_record: dict[str, Any] = {
                "alignment_key": alignment_key,
                "source_group_id": group,
                "sample_kind": sample_kind,
                "candidate_device": str(row["device"]),
                "reference_device": str(row["cross_device_reference_device"]),
                "device_transition": f"{row['device']}->{row['cross_device_reference_device']}",
                "status": "failed",
                "paper_evidence": False,
                "development_only": True,
                "post_final_diagnostic": True,
                "final_reserve_read": False,
            }
            try:
                if alignment_path.is_file():
                    alignment_cache_hits += 1
                else:
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
                                metadata["phase_correlation_response"], dtype=np.float64
                            ),
                            alignment_status=np.asarray(metadata["alignment_status"]),
                            failure_type=np.asarray(
                                metadata["alignment_failure_type"] or ""
                            ),
                            failure_reason=np.asarray(
                                metadata["alignment_failure_reason"] or ""
                            ),
                        )
                    temporary.replace(alignment_path)
                with np.load(alignment_path, allow_pickle=False) as archive:
                    aligned_reference = archive["aligned_reference"]
                    validity = archive["validity"]
                    warp = archive["estimated_homography"]
                    correlation = float(archive["ecc_correlation"])
                    phase_response = float(archive["phase_correlation_response"])
                    alignment_status = str(archive["alignment_status"])
                    failure_type = str(archive["failure_type"])
                    failure_reason = str(archive["failure_reason"])
                if (
                    aligned_reference.shape != candidate.shape
                    or validity.shape != candidate.shape[:2]
                    or not np.isfinite(warp).all()
                ):
                    raise ValueError("cross-device alignment cache is invalid")
                alignment_record.update(
                    {
                        "status": "ok",
                        "alignment_status": alignment_status,
                        "ecc_correlation": correlation,
                        "phase_correlation_response": phase_response,
                        "valid_area_fraction": float(validity.mean()),
                        "estimated_homography": warp.tolist(),
                        "alignment_failure_type": failure_type or None,
                        "alignment_failure_reason": failure_reason or None,
                        "alignment_cache": str(alignment_path.relative_to(scratch)),
                    }
                )
            except Exception as error:
                alignment_record.update(
                    {
                        "alignment_status": "failed",
                        "alignment_failure_type": type(error).__name__,
                        "alignment_failure_reason": str(error),
                    }
                )
                logging.exception("cross-device alignment failed: %s", alignment_key)
            alignment_rows.append(alignment_record)
            _write_jsonl(alignments_path, alignment_rows)
            if alignment_record["status"] != "ok":
                for model_name in models:
                    prediction_rows.append(
                        {
                            "record_id": f"{model_name}:{sample_kind}:{group}",
                            "source_group_id": group,
                            "sample_kind": sample_kind,
                            "model": model_name,
                            "status": "failed",
                            "failure_type": "AlignmentFailure",
                            "failure_reason": alignment_record[
                                "alignment_failure_reason"
                            ],
                            "paper_evidence": False,
                            "development_only": True,
                            "post_final_diagnostic": True,
                            "final_reserve_read": False,
                        }
                    )
                continue

            for model_name, model in models.items():
                threshold = thresholds[model_name]
                prediction: dict[str, Any] = {
                    "record_id": f"{model_name}:{sample_kind}:{group}",
                    "source_group_id": group,
                    "source_template": str(row["source_template"]),
                    "attack_method": str(row["attack_method"]),
                    "candidate_device": str(row["device"]),
                    "reference_device": str(row["cross_device_reference_device"]),
                    "device_transition": f"{row['device']}->{row['cross_device_reference_device']}",
                    "attack_device_cell": f"{row['attack_method']}|{row['device']}",
                    "sample_kind": sample_kind,
                    "condition": "cross_device_registered",
                    "model": model_name,
                    "training_seed": int(model_name.rsplit("_", 1)[-1]),
                    "fixed_pixel_threshold": threshold,
                    "threshold_selection_used": False,
                    "alignment_key": alignment_key,
                    "alignment_status": alignment_record["alignment_status"],
                    "ecc_correlation": alignment_record["ecc_correlation"],
                    "status": "failed",
                    "mask_semantics": "box_mask_not_pixel_accurate",
                    "paper_evidence": False,
                    "development_only": True,
                    "post_final_diagnostic": True,
                    "final_reserve_read": False,
                }
                try:
                    score_key = hashlib.sha256(
                        json.dumps(
                            {
                                "candidate_sha256": candidate_sha256,
                                "reference_sha256": row[
                                    "cross_device_reference_sha256"
                                ],
                                "alignment_key": alignment_key,
                                "model_sha256": model_hashes[model_name],
                                "preprocessing": preprocessing,
                                "inference": inference,
                                "schema": 1,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    score_path = score_cache_dir / model_name / f"{score_key}.npz"
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    if score_path.is_file():
                        score_cache_hits += 1
                        with np.load(score_path, allow_pickle=False) as archive:
                            probability = archive["scores"].astype(np.float32)
                    else:
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
                    if probability.shape != candidate.shape[:2] or not np.isfinite(
                        probability
                    ).all():
                        raise ValueError("cross-device score cache is invalid")
                    native_probability = cv2.resize(
                        probability,
                        (native_mask.shape[1], native_mask.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    if sample_kind == "forged":
                        average_precision, auroc = _ranking_metrics(
                            native_probability, native_mask
                        )
                        prediction.update(
                            {
                                "weak_box_mask_ap": average_precision,
                                "weak_box_mask_auroc": auroc,
                                **_threshold_metrics(
                                    native_probability, native_mask, threshold
                                ),
                            }
                        )
                    else:
                        prediction["authentic_pixel_fpr"] = float(
                            np.mean(native_probability >= threshold)
                        )
                    prediction.update(
                        {
                            "status": "ok",
                            "score_cache": str(score_path.relative_to(scratch)),
                            "score_shape": list(probability.shape),
                            "native_shape": list(native_probability.shape),
                        }
                    )
                except Exception as error:
                    prediction.update(
                        {
                            "failure_type": type(error).__name__,
                            "failure_reason": str(error),
                        }
                    )
                    logging.exception("cross-device prediction failed: %s", model_name)
                prediction_rows.append(prediction)
            _write_jsonl(predictions_path, prediction_rows)
        logging.info("completed_groups=%d total_groups=%d", row_index, len(rows))

    expected_alignments = expected_groups * 2
    expected_predictions = expected_groups * 2 * len(models)
    groups = {str(row["source_group_id"]) for row in rows}
    if any(row["status"] != "ok" for row in alignment_rows) or any(
        row["status"] != "ok" for row in prediction_rows
    ):
        seed_metrics: list[dict[str, Any]] = []
        aggregate: dict[str, Any] = {}
    else:
        seed_metrics = [
            _aggregate_seed(
                prediction_rows,
                same_device_sources[model_name],
                model_name,
                training_seed,
                groups,
            )
            for training_seed, model_name in (
                (value, f"robust_{value}") for value in family_seeds
            )
        ]
        aggregate = _aggregate_across_seeds(seed_metrics)
    metric_rows = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            for key, value in row.items()
        }
        for row in seed_metrics
    ]
    _write_csv(metrics_path, metric_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for metric, values in aggregate.items():
        if isinstance(values, dict):
            aggregate_rows.append({"metric": metric, **values, "paper_evidence": False})
    _write_csv(aggregate_path, aggregate_rows)

    stage = str(config["experiment"]["stage"])
    decision = _gate_decision(
        stage,
        expected_alignments,
        expected_predictions,
        alignment_rows,
        prediction_rows,
        aggregate,
        config["gate"],
    )
    if stage == "toy3":
        status = (
            "cross_device_toy_engineering_passed"
            if decision["overall_pass"]
            else "cross_device_toy_engineering_failed"
        )
    else:
        status = (
            "cross_device_pilot_gate_passed"
            if decision["overall_pass"]
            else "cross_device_pilot_gate_not_met"
        )
    correlations = [
        float(row["ecc_correlation"])
        for row in alignment_rows
        if row["status"] == "ok" and np.isfinite(float(row["ecc_correlation"]))
    ]
    convergence_rate = float(
        np.mean(
            [
                row["alignment_status"] == "ecc_converged"
                for row in alignment_rows
                if row["status"] == "ok"
            ]
        )
    )
    elapsed = time.monotonic() - started
    summary = {
        "experiment": config["experiment"],
        "status": status,
        "paper_evidence": False,
        "development_only": True,
        "post_final_diagnostic": True,
        "final_reserve_read": False,
        "threshold_selection_used": False,
        "method_change_authorized": False,
        "input_manifest": str(manifest_path.relative_to(project_root)),
        "input_manifest_sha256": _sha256(manifest_path),
        "expected_alignment_records": expected_alignments,
        "successful_alignment_records": sum(
            row["status"] == "ok" for row in alignment_rows
        ),
        "failed_alignment_records": sum(
            row["status"] != "ok" for row in alignment_rows
        ),
        "expected_prediction_records": expected_predictions,
        "successful_prediction_records": sum(
            row["status"] == "ok" for row in prediction_rows
        ),
        "failed_prediction_records": sum(
            row["status"] != "ok" for row in prediction_rows
        ),
        "attack_device_counts": dict(sorted(cell_counts.items())),
        "device_transition_counts": dict(
            sorted(
                Counter(
                    f"{row['device']}->{row['cross_device_reference_device']}"
                    for row in rows
                ).items()
            )
        ),
        "registration_convergence_rate": convergence_rate,
        "registration_fallback_count": sum(
            row.get("alignment_status") == "phase_initializer_fallback_recorded"
            for row in alignment_rows
        ),
        "median_ecc_correlation": (
            float(np.median(correlations)) if correlations else None
        ),
        "median_valid_area_fraction": float(
            np.median(
                [
                    float(row["valid_area_fraction"])
                    for row in alignment_rows
                    if row["status"] == "ok"
                ]
            )
        ),
        "seed_metrics": seed_metrics,
        "aggregate": aggregate,
        "decision": decision,
        "pilot20_authorized": stage == "toy3" and bool(decision["overall_pass"]),
        "full88_authorized": stage != "toy3" and bool(decision["overall_pass"]),
        "score_cache_hits": score_cache_hits,
        "alignment_cache_hits": alignment_cache_hits,
        "wall_time_seconds": elapsed,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "alignments": str(alignments_path.relative_to(project_root)),
            "alignments_sha256": _sha256(alignments_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "aggregate": str(aggregate_path.relative_to(project_root)),
            "aggregate_sha256": _sha256(aggregate_path),
        },
    }
    _write_json(summary_path, summary)
    if bool(runtime["require_all_records"]) and (
        summary["failed_alignment_records"] or summary["failed_prediction_records"]
    ):
        raise RuntimeError("cross-device diagnostic incomplete")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
