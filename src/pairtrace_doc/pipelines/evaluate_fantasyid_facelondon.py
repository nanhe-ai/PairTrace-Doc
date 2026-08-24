from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
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
    _write_csv,
    _write_json,
    _write_jsonl,
)


REQUIRED_CONDITIONS = {
    "student",
    "raw_difference",
    "clean_teacher_correct",
    "robust_teacher_correct",
    "robust_teacher_shuffled",
}


def _validate_multiseed_external_authorization(config: dict[str, Any]) -> bool:
    authorized = bool(config["runtime"].get("multi_seed_authorized", False))
    policy = config.get("multi_seed")
    if not authorized:
        if policy is not None:
            raise ValueError("single-seed FantasyID evaluation cannot carry multi-seed policy")
        return False
    if (
        config["experiment"].get("stage")
        != "multiseed_fantasyid_full88_stability_evaluation"
    ):
        raise ValueError("FantasyID multi-seed evaluation stage is not frozen")
    if not isinstance(policy, dict):
        raise ValueError("FantasyID multi-seed evaluation policy is missing")
    if [int(value) for value in policy["family_seeds"]] != [
        20260747,
        20260763,
        20260764,
    ]:
        raise ValueError("FantasyID multi-seed family changed")
    if int(policy["training_seed"]) not in (20260747, 20260763, 20260764):
        raise ValueError("FantasyID multi-seed evaluation seed is not in the family")
    return True


def _cell_macro(values: list[dict[str, Any]], field: str) -> float:
    by_cell: defaultdict[str, list[float]] = defaultdict(list)
    for item in values:
        by_cell[str(item["attack_device_cell"])].append(float(item[field]))
    if not by_cell:
        raise ValueError("attack-device macro requires at least one cell")
    return float(np.mean([np.mean(cell_values) for cell_values in by_cell.values()]))


def _grouped_macro(
    values: list[dict[str, Any]], group_field: str, value_field: str
) -> dict[str, float]:
    first_level: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    secondary = "device" if group_field == "attack_method" else "attack_method"
    for item in values:
        first_level[str(item[group_field])][str(item[secondary])].append(
            float(item[value_field])
        )
    return {
        group: float(np.mean([np.mean(cell) for cell in cells.values()]))
        for group, cells in sorted(first_level.items())
    }


def _paired_cell_bootstrap(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired external-development group sets differ")
    by_cell: defaultdict[str, list[float]] = defaultdict(list)
    for group in sorted(left):
        left_item = left[group]
        right_item = right[group]
        if left_item["attack_device_cell"] != right_item["attack_device_cell"]:
            raise ValueError("paired external-development cell labels differ")
        by_cell[str(left_item["attack_device_cell"])].append(
            float(left_item["macro_box_mask_ap"])
            - float(right_item["macro_box_mask_ap"])
        )
    rng = np.random.default_rng(seed)
    replicates: list[np.ndarray] = []
    per_cell_effect: dict[str, float] = {}
    for cell, values in sorted(by_cell.items()):
        differences = np.asarray(values, dtype=float)
        indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
        replicates.append(differences[indices].mean(axis=1))
        per_cell_effect[cell] = float(differences.mean())
    combined = np.stack(replicates).mean(axis=0)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "effect": float(np.mean(list(per_cell_effect.values()))),
        "ci_low": float(np.quantile(combined, alpha)),
        "ci_high": float(np.quantile(combined, 1.0 - alpha)),
        "per_attack_device_cell_effect": per_cell_effect,
        "bootstrap_unit": "source_group_id_stratified_by_attack_device_cell",
    }


def _verify_frozen_thresholds(
    project_root: Path,
    sources: dict[str, Any],
    conditions: dict[str, dict[str, Any]],
) -> dict[str, str]:
    verified: dict[str, str] = {}
    cached_rows: dict[str, list[dict[str, str]]] = {}
    for condition_name, condition in conditions.items():
        source_name = str(condition["threshold_source"])
        specification = sources[source_name]
        path = _resolve(project_root, specification["path"])
        if _sha256(path) != specification["sha256"]:
            raise ValueError(f"frozen threshold source changed: {source_name}")
        if source_name not in cached_rows:
            with path.open("r", encoding="utf-8", newline="") as handle:
                cached_rows[source_name] = list(csv.DictReader(handle))
        source_condition = str(condition["threshold_source_condition"])
        matches = [
            row
            for row in cached_rows[source_name]
            if row.get("condition") == source_condition
        ]
        if len(matches) != 1:
            raise ValueError(f"frozen threshold row is not unique: {condition_name}")
        source_value = float(matches[0]["pixel_threshold"])
        if not np.isclose(
            source_value, float(condition["fixed_threshold"]), atol=1e-12, rtol=0.0
        ):
            raise ValueError(f"frozen threshold value changed: {condition_name}")
        verified[condition_name] = f"{source_name}:{source_condition}"
    return verified


def _aggregate_condition(
    forged: list[dict[str, Any]], authentic_fprs: list[float], threshold: float
) -> dict[str, Any]:
    if not forged or len(forged) != len(authentic_fprs):
        raise ValueError("external-development aggregation is incomplete")
    per_attack = _grouped_macro(forged, "attack_method", "macro_box_mask_ap")
    per_device = _grouped_macro(forged, "device", "macro_box_mask_ap")
    return {
        "development_groups": len(forged),
        "attack_device_macro_box_mask_ap": _cell_macro(
            forged, "macro_box_mask_ap"
        ),
        "document_macro_box_mask_ap": float(
            np.mean([item["macro_box_mask_ap"] for item in forged])
        ),
        "document_macro_box_mask_auroc": float(
            np.mean([item["box_mask_auroc"] for item in forged])
        ),
        "authentic_document_macro_pixel_fpr": float(np.mean(authentic_fprs)),
        "fixed_pixel_threshold": float(threshold),
        "per_attack_macro_box_mask_ap": per_attack,
        "per_device_macro_box_mask_ap": per_device,
        "mask_semantics": "box_mask_not_pixel_accurate",
        "threshold_selected_on_fantasyid": False,
        "paper_evidence": False,
    }


def _pilot_decision(
    metrics: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, dict[str, Any]]],
    gate: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    student = _paired_cell_bootstrap(
        scores["robust_teacher_correct"],
        scores["student"],
        int(bootstrap["seed"]),
        int(bootstrap["resamples"]),
        float(bootstrap["confidence_level"]),
    )
    shuffled = _paired_cell_bootstrap(
        scores["robust_teacher_correct"],
        scores["robust_teacher_shuffled"],
        int(bootstrap["seed"]) + 1,
        int(bootstrap["resamples"]),
        float(bootstrap["confidence_level"]),
    )
    robust = metrics["robust_teacher_correct"]
    per_attack = robust["per_attack_macro_box_mask_ap"]
    required_attacks = [str(value) for value in gate["required_attack_methods"]]
    checks: dict[str, bool] = {
        "all_predictions_complete": True,
        "robust_attack_device_macro_ap_floor": float(
            robust["attack_device_macro_box_mask_ap"]
        )
        >= float(gate["robust_attack_device_macro_box_mask_ap_min"]),
        "robust_minus_student_effect_floor": float(student["effect"])
        >= float(gate["robust_minus_student_min"]),
        "robust_minus_student_interval_positive": float(student["ci_low"]) > 0.0,
        "robust_minus_shuffled_effect_floor": float(shuffled["effect"])
        >= float(gate["robust_minus_shuffled_min"]),
        "robust_minus_shuffled_interval_positive": float(shuffled["ci_low"]) > 0.0,
        "robust_authentic_fpr_ceiling": float(
            robust["authentic_document_macro_pixel_fpr"]
        )
        <= float(gate["authentic_pixel_fpr_max"]),
    }
    for attack in required_attacks:
        checks[f"robust_per_attack_ap_floor__{attack}"] = (
            attack in per_attack
            and float(per_attack[attack]) >= float(gate["per_attack_macro_box_mask_ap_min"])
        )
    return {
        "comparisons": {
            "robust_minus_student": student,
            "robust_minus_shuffled": shuffled,
        },
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
    if not bool(runtime["gpu_launch_authorized"]) or not bool(
        runtime["external_development_evaluation_authorized"]
    ):
        raise ValueError("FantasyID external-development evaluation was not authorized")
    if not bool(runtime["selected_image_read_allowed"]):
        raise ValueError("FantasyID selected image read was not authorized")
    multi_seed_run = _validate_multiseed_external_authorization(config)
    if any(
        bool(runtime.get(name))
        for name in (
            "model_training_authorized",
            "method_change_authorized",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("FantasyID evaluation crossed an evidence boundary")
    if bool(config["experiment"]["paper_evidence"]):
        raise ValueError("FantasyID external development cannot be paper evidence")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FantasyID external-development evaluation requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("FantasyID external-development protocol changed")
    if multi_seed_run:
        identity_path = _resolve(
            project_root, config["multi_seed"]["paper_identity_amendment"]
        )
        if _sha256(identity_path) != config["multi_seed"][
            "expected_paper_identity_sha256"
        ]:
            raise ValueError("multi-seed paper identity amendment changed")
    inputs = config["input"]
    manifest_path = _resolve(project_root, inputs["manifest"])
    if _sha256(manifest_path) != inputs["expected_manifest_sha256"]:
        raise ValueError("FantasyID materialized manifest changed")
    rows = sorted(
        _read_jsonl(manifest_path), key=lambda row: int(row["selection_index"])
    )
    if len(rows) != int(inputs["expected_groups"]):
        raise ValueError("FantasyID evaluation group count changed")
    if len({str(row["source_group_id"]) for row in rows}) != len(rows):
        raise ValueError("FantasyID evaluation contains duplicate source groups")
    if {str(row["fantasyid_facelondon_freeze_id"]) for row in rows} != {
        str(inputs["expected_freeze_id"])
    }:
        raise ValueError("FantasyID evaluation freeze ID changed")
    if any(
        row.get("materialization_status") != "ok"
        or row.get("mask_semantics") != "box_mask_not_pixel_accurate"
        or row.get("paper_evidence") is not False
        for row in rows
    ):
        raise ValueError("FantasyID materialization evidence boundary changed")
    cell_counts = Counter(
        f"{row['attack_method']}|{row['device']}" for row in rows
    )
    expected_cell_counts = {
        str(key): int(value)
        for key, value in inputs["expected_attack_device_counts"].items()
    }
    if dict(sorted(cell_counts.items())) != expected_cell_counts:
        raise ValueError(f"FantasyID stage balance changed: {cell_counts}")

    condition_specs = {str(item["name"]): item for item in config["conditions"]}
    if set(condition_specs) != REQUIRED_CONDITIONS:
        raise ValueError("FantasyID condition whitelist changed")
    verified_thresholds = _verify_frozen_thresholds(
        project_root, config["threshold_sources"], condition_specs
    )
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
    models = config["models"]
    encoder_path = _resolve(scratch, models["encoder_weights"])
    if _sha256(encoder_path) != models["encoder_weights_sha256"]:
        raise ValueError("FantasyID encoder weights changed")
    student_path = _resolve(project_root, models["student"]["checkpoint"])
    if _sha256(student_path) != models["student"]["checkpoint_sha256"]:
        raise ValueError("FantasyID student checkpoint changed")
    saved_student = torch.load(student_path, map_location="cpu", weights_only=True)
    student = ResNet18UNet()
    student.load_state_dict(saved_student["model_state"], strict=True)
    student = student.to(device).eval().requires_grad_(False)
    pair_models: dict[str, torch.nn.Module] = {}
    for name in ("baseline", "robust"):
        checkpoint_path = _resolve(project_root, models[name]["checkpoint"])
        if _sha256(checkpoint_path) != models[name]["checkpoint_sha256"]:
            raise ValueError(f"FantasyID {name} checkpoint changed")
        model = _load_teacher(encoder_path, models["teacher_conv1_coefficients"])
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state"], strict=True)
        pair_models[name] = model.to(device).eval().requires_grad_(False)

    score_cache_dir = _resolve(project_root, paths["score_cache_dir"])
    predictions_path = _resolve(project_root, paths["predictions"])
    metrics_path = _resolve(project_root, paths["metrics"])
    comparisons_path = _resolve(project_root, paths["comparisons"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (
        score_cache_dir,
        predictions_path.parent,
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
    inference = config["inference"]
    shuffled = _shuffled_group_map(
        rows, seed + int(config["controls"]["shuffle_seed_offset"])
    )
    rows_by_group = {str(row["source_group_id"]): row for row in rows}
    image_cache: dict[str, np.ndarray] = {}

    def load_image(row: dict[str, Any], field: str, sha_field: str) -> np.ndarray:
        path = _resolve(scratch, row[field])
        key = str(path)
        if key not in image_cache:
            if _sha256(path) != row[sha_field]:
                raise ValueError(f"FantasyID {field} SHA-256 changed")
            with Image.open(path) as handle:
                image_cache[key] = np.asarray(handle.convert("RGB"))
        return image_cache[key]

    payloads: dict[str, dict[str, Any]] = {
        name: {"forged": [], "authentic_fprs": []} for name in condition_specs
    }
    paired_scores: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in condition_specs
    }
    prediction_rows: list[dict[str, Any]] = []
    failures = 0
    cache_hits = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for row_index, row in enumerate(rows, start=1):
        group = str(row["source_group_id"])
        forged_native = load_image(row, "image", "image_sha256")
        authentic_native = load_image(row, "authentic", "authentic_sha256")
        mask_path = _resolve(scratch, row["mask"])
        if _sha256(mask_path) != row["mask_sha256"]:
            raise ValueError("FantasyID box mask SHA-256 changed")
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0
        if (
            forged_native.shape[:2] != native_mask.shape
            or authentic_native.shape[:2] != native_mask.shape
        ):
            raise ValueError("FantasyID aligned pair geometry changed")

        for condition_name, condition in condition_specs.items():
            for sample_kind, candidate_native in (
                ("forged", forged_native),
                ("authentic", authentic_native),
            ):
                record: dict[str, Any] = {
                    "record_id": f"{condition_name}:{sample_kind}:{group}",
                    "source_group_id": group,
                    "source_template": row["source_template"],
                    "attack_method": row["attack_method"],
                    "device": row["device"],
                    "attack_device_cell": f"{row['attack_method']}|{row['device']}",
                    "condition": condition_name,
                    "sample_kind": sample_kind,
                    "status": "failed",
                    "mask_semantics": "box_mask_not_pixel_accurate",
                    "development_only": True,
                    "threshold_selected_on_fantasyid": False,
                    "final_reserve_read": False,
                    "paper_evidence": False,
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
                        reference_native = load_image(
                            target, "authentic", "authentic_sha256"
                        )
                        reference_sha256 = str(target["authentic_sha256"])
                        record["reference_source_group_id"] = target["source_group_id"]
                    else:
                        raise ValueError(f"unsupported FantasyID reference: {reference_mode}")
                    candidate = _resize_image(
                        candidate_native, int(preprocessing["max_side"])
                    )
                    reference = (
                        None
                        if reference_native is None
                        else _resize_reference(reference_native, candidate.shape[:2])
                    )
                    scorer = str(condition["scorer"])
                    model_identity = (
                        models["student"]["checkpoint_sha256"]
                        if scorer == "student"
                        else models["baseline"]["checkpoint_sha256"]
                        if scorer == "baseline"
                        else models["robust"]["checkpoint_sha256"]
                        if scorer == "robust"
                        else "raw_aligned_absolute_difference_v1"
                    )
                    candidate_sha256 = str(
                        row["image_sha256"]
                        if sample_kind == "forged"
                        else row["authentic_sha256"]
                    )
                    cache_key = hashlib.sha256(
                        json.dumps(
                            {
                                "candidate_sha256": candidate_sha256,
                                "reference_sha256": reference_sha256,
                                "condition": condition_name,
                                "scorer": scorer,
                                "model_identity": model_identity,
                                "preprocessing": preprocessing,
                                "inference": inference,
                                "score_cache_schema": 1,
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
                            probability = _infer_tiled(
                                student, candidate, device, inference, preprocessing
                            )
                        elif scorer == "raw_difference":
                            if reference is None:
                                raise ValueError("raw difference has no reference")
                            probability = _raw_difference(candidate, reference)
                        elif scorer in pair_models:
                            if reference is None:
                                raise ValueError("pair teacher has no reference")
                            probability = _infer_pair_tiled(
                                pair_models[scorer],
                                candidate,
                                reference,
                                device,
                                inference,
                                preprocessing,
                            )
                        else:
                            raise ValueError(f"unsupported FantasyID scorer: {scorer}")
                        temporary = score_path.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(
                                handle, scores=probability.astype(np.float32)
                            )
                        temporary.replace(score_path)
                    if probability.shape != candidate.shape[:2] or not np.isfinite(
                        probability
                    ).all():
                        raise ValueError("FantasyID score cache is invalid")
                    native_probability = cv2.resize(
                        probability,
                        (native_mask.shape[1], native_mask.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    threshold = float(condition["fixed_threshold"])
                    if sample_kind == "forged":
                        average_precision, auroc = _ranking_metrics(
                            native_probability, native_mask
                        )
                        item = {
                            "source_group_id": group,
                            "attack_method": str(row["attack_method"]),
                            "device": str(row["device"]),
                            "attack_device_cell": f"{row['attack_method']}|{row['device']}",
                            "macro_box_mask_ap": average_precision,
                            "box_mask_auroc": auroc,
                        }
                        payloads[condition_name]["forged"].append(item)
                        paired_scores[condition_name][group] = item
                        record.update(
                            {
                                "macro_box_mask_ap": average_precision,
                                "box_mask_auroc": auroc,
                            }
                        )
                    else:
                        fpr = float(np.mean(native_probability >= threshold))
                        payloads[condition_name]["authentic_fprs"].append(fpr)
                        record["authentic_pixel_fpr_at_fixed_threshold"] = fpr
                    record.update(
                        {
                            "status": "ok",
                            "scorer": scorer,
                            "reference_mode": reference_mode,
                            "model_identity": model_identity,
                            "fixed_threshold": threshold,
                            "score_cache": str(score_path.relative_to(project_root)),
                            "score_shape": list(probability.shape),
                            "native_shape": list(native_probability.shape),
                        }
                    )
                except Exception as error:
                    failures += 1
                    record["failure_type"] = type(error).__name__
                    record["failure_reason"] = str(error)
                    logging.exception("record_id=%s failed", record["record_id"])
                prediction_rows.append(record)
        _write_jsonl(predictions_path, prediction_rows)
        logging.info("completed_groups=%d total_groups=%d", row_index, len(rows))

    complete = failures == 0 and all(
        len(payload["forged"]) == len(rows)
        and len(payload["authentic_fprs"]) == len(rows)
        for payload in payloads.values()
    )
    if not complete:
        summary = {
            "experiment": config["experiment"],
            "status": "failed_incomplete",
            "paper_evidence": False,
            "failed_prediction_records": failures,
            "successful_prediction_records": len(prediction_rows) - failures,
            "final_reserve_read": False,
            "outputs": {
                "predictions": str(predictions_path.relative_to(project_root)),
                "predictions_sha256": _sha256(predictions_path),
            },
        }
        _write_json(summary_path, summary)
        if bool(runtime["require_all_records"]):
            raise RuntimeError("FantasyID external-development evaluation incomplete")
        return summary

    metrics = {
        name: _aggregate_condition(
            payload["forged"],
            payload["authentic_fprs"],
            float(condition_specs[name]["fixed_threshold"]),
        )
        for name, payload in payloads.items()
    }
    metric_rows: list[dict[str, Any]] = []
    for name, values in metrics.items():
        metric_rows.append(
            {
                "condition": name,
                "development_groups": values["development_groups"],
                "attack_device_macro_box_mask_ap": values[
                    "attack_device_macro_box_mask_ap"
                ],
                "document_macro_box_mask_ap": values["document_macro_box_mask_ap"],
                "document_macro_box_mask_auroc": values[
                    "document_macro_box_mask_auroc"
                ],
                "authentic_document_macro_pixel_fpr": values[
                    "authentic_document_macro_pixel_fpr"
                ],
                "fixed_pixel_threshold": values["fixed_pixel_threshold"],
                "mask_semantics": values["mask_semantics"],
                "threshold_selected_on_fantasyid": False,
                "paper_evidence": False,
            }
        )
    _write_csv(metrics_path, metric_rows)

    comparisons = {
        "robust_minus_student": _paired_cell_bootstrap(
            paired_scores["robust_teacher_correct"],
            paired_scores["student"],
            int(config["bootstrap"]["seed"]),
            int(config["bootstrap"]["resamples"]),
            float(config["bootstrap"]["confidence_level"]),
        ),
        "robust_minus_shuffled": _paired_cell_bootstrap(
            paired_scores["robust_teacher_correct"],
            paired_scores["robust_teacher_shuffled"],
            int(config["bootstrap"]["seed"]) + 1,
            int(config["bootstrap"]["resamples"]),
            float(config["bootstrap"]["confidence_level"]),
        ),
    }
    comparison_rows = [
        {
            "comparison": name,
            "attack_device_macro_box_mask_ap_difference": result["effect"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "confidence_level": float(config["bootstrap"]["confidence_level"]),
            "bootstrap_resamples": int(config["bootstrap"]["resamples"]),
            "bootstrap_seed": int(config["bootstrap"]["seed"])
            + int(name == "robust_minus_shuffled"),
            "bootstrap_unit": result["bootstrap_unit"],
            "paper_evidence": False,
        }
        for name, result in comparisons.items()
    ]
    _write_csv(comparisons_path, comparison_rows)

    decision_allowed = bool(config["gate"]["decision_allowed"])
    if decision_allowed:
        decision = _pilot_decision(
            metrics,
            paired_scores,
            config["gate"],
            config["bootstrap"],
        )
        status = (
            "passed_external_development_gate"
            if decision["overall_pass"]
            else "completed_external_development_gate_not_met"
        )
    else:
        decision = {
            "engineering_checks": {
                "all_predictions_complete": True,
                "all_cache_records_valid": True,
                "all_box_masks_nonempty": all(
                    int(row["mask_positive_pixels"]) > 0 for row in rows
                ),
                "zero_final_reserve_reads": True,
            },
            "scientific_gate_evaluated": False,
        }
        decision["overall_pass"] = all(decision["engineering_checks"].values())
        status = (
            "passed_toy_engineering_gate"
            if decision["overall_pass"]
            else "failed_toy_engineering_gate"
        )

    elapsed = time.monotonic() - started
    summary = {
        "experiment": config["experiment"],
        "status": status,
        "paper_evidence": False,
        "development_only": True,
        "mask_semantics": "box_mask_not_pixel_accurate",
        "original_confirmatory_gate_reopened": False,
        "final_reserve_read": False,
        "threshold_selected_on_fantasyid": False,
        "verified_threshold_sources": verified_thresholds,
        "input_manifest": str(manifest_path.relative_to(project_root)),
        "input_manifest_sha256": _sha256(manifest_path),
        "freeze_id": inputs["expected_freeze_id"],
        "development_groups": len(rows),
        "attack_device_counts": dict(sorted(cell_counts.items())),
        "successful_prediction_records": len(prediction_rows),
        "failed_prediction_records": 0,
        "score_cache_hits": cache_hits,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "metrics": metrics,
        "comparisons": comparisons,
        "decision": decision,
        "full88_authorized": decision_allowed and bool(decision["overall_pass"]),
        "multi_seed_authorized": multi_seed_run,
        "multi_seed_stability_evaluation": multi_seed_run,
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
        },
        "runtime": runtime,
    }
    _write_json(summary_path, summary)
    logging.info("summary=%s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
