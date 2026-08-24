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
from statistics import mean, stdev
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

from pairtrace_doc.pipelines.diagnose_pair_at_inference_alignment import (
    STRESSES,
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


GEOMETRIES = ("clean", *STRESSES)
FINAL_ROLES = ("in_domain_test", "generator_holdout")
FAMILY_SEEDS = (20260747, 20260763, 20260764)


def _required_conditions(seeds: tuple[int, ...] = FAMILY_SEEDS) -> set[str]:
    result = {"student_clean"}
    result.update(f"baseline_{geometry}_ecc" for geometry in GEOMETRIES)
    result.update(f"raw_difference_{stress}_ecc" for stress in STRESSES)
    for seed in seeds:
        result.update(f"robust_{seed}_{geometry}_ecc" for geometry in GEOMETRIES)
        result.add(f"robust_{seed}_shuffled_clean")
        result.add(f"robust_{seed}_affine_unaligned")
    return result


def _role_generator_macro(items: list[dict[str, Any]], value: str) -> dict[str, Any]:
    by_role_generator: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in items:
        by_role_generator[(str(item["evaluation_role"]), str(item["generator"]))].append(
            float(item[value])
        )
    per_role_generator = {
        f"{role}|{generator}": float(np.mean(values))
        for (role, generator), values in sorted(by_role_generator.items())
    }
    per_role: dict[str, float] = {}
    for role in FINAL_ROLES:
        selected = [
            metric
            for key, metric in per_role_generator.items()
            if key.startswith(f"{role}|")
        ]
        if not selected:
            raise ValueError(f"missing final forged role: {role}")
        per_role[role] = float(np.mean(selected))
    return {
        "role_macro_generator_macro": float(np.mean(list(per_role.values()))),
        "per_role": per_role,
        "per_role_generator": per_role_generator,
    }


def _aggregate_fixed_condition(
    payload: dict[str, list[dict[str, Any]]],
    pixel_threshold: float,
    image_threshold: float,
) -> dict[str, Any]:
    forged = payload["forged"]
    authentic = payload["authentic"]
    if not forged or not authentic:
        raise ValueError("final condition aggregation is incomplete")
    ranking = _role_generator_macro(forged, "macro_pixel_ap")
    forged_image_scores = np.asarray([item["image_score"] for item in forged])
    authentic_image_scores = np.asarray([item["image_score"] for item in authentic])
    metrics: dict[str, Any] = {
        "forged_documents": len(forged),
        "authentic_documents": len(authentic),
        "pixel_threshold": float(pixel_threshold),
        "image_threshold": float(image_threshold),
        "role_macro_generator_macro_pixel_ap": ranking[
            "role_macro_generator_macro"
        ],
        "document_macro_pixel_ap": float(
            np.mean([item["macro_pixel_ap"] for item in forged])
        ),
        "document_macro_pixel_auroc": float(
            np.mean([item["pixel_auroc"] for item in forged])
        ),
        "document_macro_pixel_precision": float(
            np.mean([item["pixel_precision"] for item in forged])
        ),
        "document_macro_pixel_recall": float(
            np.mean([item["pixel_recall"] for item in forged])
        ),
        "document_macro_pixel_f1": float(
            np.mean([item["pixel_f1"] for item in forged])
        ),
        "document_macro_pixel_iou": float(
            np.mean([item["pixel_iou"] for item in forged])
        ),
        "authentic_document_macro_pixel_fpr": float(
            np.mean([item["pixel_fpr"] for item in authentic])
        ),
        "image_auroc": _roc_auc(
            np.r_[forged_image_scores, authentic_image_scores],
            np.r_[
                np.ones(forged_image_scores.size, dtype=bool),
                np.zeros(authentic_image_scores.size, dtype=bool),
            ],
        ),
        "image_tpr_at_development_frozen_threshold": float(
            np.mean(forged_image_scores >= image_threshold)
        ),
        "image_fpr_at_development_frozen_threshold": float(
            np.mean(authentic_image_scores >= image_threshold)
        ),
        "threshold_selected_on_final_reserve": False,
        "paper_evidence": True,
    }
    for role, metric in ranking["per_role"].items():
        metrics[f"pixel_ap__{role}"] = metric
    for key, metric in ranking["per_role_generator"].items():
        safe = "".join(character if character.isalnum() else "_" for character in key)
        metrics[f"pixel_ap__{safe}"] = metric
    return metrics


def _clustered_paired_bootstrap(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("final paired comparison record sets differ")
    differences: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(left):
        left_item = left[key]
        right_item = right[key]
        for field in (
            "source_group_id",
            "source_stratum",
            "evaluation_role",
            "generator",
        ):
            if left_item[field] != right_item[field]:
                raise ValueError(f"final paired metadata differ: {field}")
        item = {
            **{field: left_item[field] for field in (
                "source_group_id",
                "source_stratum",
                "evaluation_role",
                "generator",
            )},
            "difference": float(left_item["value"]) - float(right_item["value"]),
        }
        differences.append(item)
        by_group[str(item["source_group_id"])].append(item)
    if any(len(items) != len(FINAL_ROLES) for items in by_group.values()):
        raise ValueError("final bootstrap did not retain both forged roles per group")
    observed = _role_generator_macro(differences, "difference")

    strata: dict[str, list[str]] = defaultdict(list)
    for group, items in by_group.items():
        stratum = {str(item["source_stratum"]) for item in items}
        if len(stratum) != 1:
            raise ValueError("source group stratum changed across forged roles")
        strata[next(iter(stratum))].append(group)
    rng = np.random.default_rng(seed)
    category_sums: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(resamples, dtype=float)
    )
    category_counts: Counter[tuple[str, str]] = Counter()
    for groups in strata.values():
        ordered = sorted(groups)
        indices = rng.integers(0, len(ordered), size=(resamples, len(ordered)))
        for role in FINAL_ROLES:
            role_items = [
                next(item for item in by_group[group] if item["evaluation_role"] == role)
                for group in ordered
            ]
            categories = {(str(item["evaluation_role"]), str(item["generator"])) for item in role_items}
            if len(categories) != 1:
                raise ValueError("bootstrap stratum mixes role-generator categories")
            category = next(iter(categories))
            values = np.asarray([item["difference"] for item in role_items])
            category_sums[category] += values[indices].sum(axis=1)
            category_counts[category] += len(ordered)
    category_replicates = {
        category: values / category_counts[category]
        for category, values in category_sums.items()
    }
    role_replicates = {
        role: np.stack(
            [
                values
                for (candidate_role, _), values in category_replicates.items()
                if candidate_role == role
            ]
        ).mean(axis=0)
        for role in FINAL_ROLES
    }
    replicates = np.stack(list(role_replicates.values())).mean(axis=0)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "effect": observed["role_macro_generator_macro"],
        "ci_low": float(np.quantile(replicates, alpha)),
        "ci_high": float(np.quantile(replicates, 1.0 - alpha)),
        "per_role_effect": observed["per_role"],
        "per_role_generator_effect": observed["per_role_generator"],
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "confidence_level": confidence_level,
        "unit": "source_group_id_stratified_by_in_domain_generator_and_source_dataset",
    }


def _mean_score_maps(
    maps: list[dict[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    if not maps:
        raise ValueError("cannot average an empty score-map family")
    keys = set(maps[0])
    if any(set(mapping) != keys for mapping in maps[1:]):
        raise ValueError("seed score maps have different record sets")
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        reference = maps[0][key]
        for mapping in maps[1:]:
            for field in (
                "source_group_id",
                "source_stratum",
                "evaluation_role",
                "generator",
            ):
                if mapping[key][field] != reference[field]:
                    raise ValueError("seed score-map metadata differ")
        result[key] = {
            **{field: reference[field] for field in (
                "source_group_id",
                "source_stratum",
                "evaluation_role",
                "generator",
            )},
            "value": float(np.mean([mapping[key]["value"] for mapping in maps])),
        }
    return result


def _descriptive(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("final seed aggregation requires multiple seeds")
    return {
        "mean": mean(values),
        "sample_standard_deviation": stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def _final_decision(
    metrics: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    registration: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    individual: dict[str, dict[str, bool]] = {}
    minimum_stressed_values = []
    for seed in FAMILY_SEEDS:
        clean = metrics[f"robust_{seed}_clean_ecc"]
        stressed = [metrics[f"robust_{seed}_{stress}_ecc"] for stress in STRESSES]
        minimum_stressed = min(
            float(item["role_macro_generator_macro_pixel_ap"]) for item in stressed
        )
        minimum_stressed_values.append(minimum_stressed)
        individual[str(seed)] = {
            "clean_ap_floor": float(clean["role_macro_generator_macro_pixel_ap"])
            >= float(gate["robust_clean_ap_min"]),
            "minimum_stressed_ap_floor": minimum_stressed
            >= float(gate["robust_minimum_stressed_ap_min"]),
            "clean_in_domain_ap_floor": float(clean["pixel_ap__in_domain_test"])
            >= float(gate["robust_clean_per_role_ap_min"]),
            "clean_output_holdout_ap_floor": float(
                clean["pixel_ap__generator_holdout"]
            )
            >= float(gate["robust_clean_per_role_ap_min"]),
            "minimum_stressed_in_domain_ap_floor": min(
                float(item["pixel_ap__in_domain_test"]) for item in stressed
            )
            >= float(gate["robust_minimum_stressed_per_role_ap_min"]),
            "minimum_stressed_output_holdout_ap_floor": min(
                float(item["pixel_ap__generator_holdout"]) for item in stressed
            )
            >= float(gate["robust_minimum_stressed_per_role_ap_min"]),
            "authentic_fpr_ceiling": max(
                float(metrics[f"robust_{seed}_{geometry}_ecc"][
                    "authentic_document_macro_pixel_fpr"
                ])
                for geometry in GEOMETRIES
            )
            <= float(gate["authentic_pixel_fpr_max"]),
        }
    statistics = _descriptive(minimum_stressed_values)
    across_seed = {
        "minimum_stressed_ap_sample_std_ceiling": statistics[
            "sample_standard_deviation"
        ]
        <= float(gate["minimum_stressed_ap_sample_std_max"]),
        "minimum_stressed_ap_range_ceiling": statistics["range"]
        <= float(gate["minimum_stressed_ap_range_max"]),
    }
    comparison_checks: dict[str, bool] = {}
    for stress, floor in gate["robust_minus_baseline_effect_min"].items():
        result = comparisons[f"robust_mean_minus_baseline__{stress}"]
        comparison_checks[f"robust_minus_baseline_effect__{stress}"] = float(
            result["effect"]
        ) >= float(floor)
        comparison_checks[f"robust_minus_baseline_interval__{stress}"] = float(
            result["ci_low"]
        ) > 0.0
    for stress in STRESSES:
        result = comparisons[f"robust_mean_minus_raw__{stress}"]
        comparison_checks[f"robust_minus_raw_effect__{stress}"] = float(
            result["effect"]
        ) >= float(gate["robust_minus_raw_effect_min"])
        comparison_checks[f"robust_minus_raw_interval__{stress}"] = float(
            result["ci_low"]
        ) > 0.0
    for name, floor in (
        ("robust_mean_clean_minus_student", gate["robust_clean_minus_student_min"]),
        ("robust_mean_clean_minus_shuffled", gate["robust_clean_minus_shuffled_min"]),
        ("robust_mean_affine_ecc_minus_unaligned", gate["robust_affine_ecc_minus_unaligned_min"]),
    ):
        comparison_checks[f"{name}_effect"] = float(comparisons[name]["effect"]) >= float(floor)
        comparison_checks[f"{name}_interval"] = float(comparisons[name]["ci_low"]) > 0.0
    registration_checks = {
        "registration_convergence_floor": float(registration["convergence_rate"])
        >= float(gate["registration_convergence_rate_min"]),
        "registration_fallback_zero": int(registration["fallbacks"]) == 0,
        "registration_corner_p95_ceiling": float(
            registration["controlled_stress_corner_error_p95_pixels"]
        )
        <= float(gate["registration_corner_error_p95_pixels_max"]),
    }
    overall_pass = all(
        value for checks in individual.values() for value in checks.values()
    ) and all(across_seed.values()) and all(comparison_checks.values()) and all(
        registration_checks.values()
    )
    return {
        "individual_seed_checks": individual,
        "across_seed_minimum_stressed_ap": statistics,
        "across_seed_checks": across_seed,
        "comparison_checks": comparison_checks,
        "registration_checks": registration_checks,
        "overall_pass": overall_pass,
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
    if not all(
        bool(runtime.get(key))
        for key in (
            "gpu_launch_authorized",
            "final_evaluation_authorized",
            "final_reserve_image_read_allowed",
        )
    ):
        raise ValueError("final reserve evaluation was not explicitly authorized")
    if any(
        bool(runtime.get(key))
        for key in ("model_training_authorized", "method_change_authorized")
    ):
        raise ValueError("final reserve evaluation cannot change the method")
    if not config["experiment"]["paper_evidence"]:
        raise ValueError("one-shot final reserve must be labeled paper evidence")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("final reserve evaluation requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("final reserve protocol SHA-256 changed")
    stability_path = _resolve(project_root, config["authorization"]["stability_summary"])
    if _sha256(stability_path) != config["authorization"]["expected_stability_summary_sha256"]:
        raise ValueError("multi-seed stability authorization changed")
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    if not stability["final_evaluation_protocol_freeze_authorized"] or not stability["decision"]["overall_pass"]:
        raise ValueError("multi-seed stability did not authorize the final protocol")
    image_threshold_path = _resolve(project_root, config["authorization"]["image_thresholds"])
    if _sha256(image_threshold_path) != config["authorization"]["expected_image_thresholds_sha256"]:
        raise ValueError("final image operating points changed")

    scratch = Path(
        os.environ.get(
            config["paths"]["scratch_env"],
            str(_resolve(project_root, config["paths"]["scratch_default"])),
        )
    ).resolve()
    manifest_path = _resolve(project_root, config["input"]["manifest"])
    if _sha256(manifest_path) != config["input"]["expected_manifest_sha256"]:
        raise ValueError("final reserve manifest SHA-256 changed")
    manifest_rows = _read_jsonl(manifest_path)
    if len(manifest_rows) != int(config["input"]["expected_records"]):
        raise ValueError("final reserve record count changed")
    if {str(row["holdout_freeze_id"]) for row in manifest_rows} != {
        str(config["input"]["expected_freeze_id"])
    }:
        raise ValueError("final reserve freeze ID changed")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in manifest_rows:
        group = str(row["source_group_id"])
        role = str(row["evaluation_role"])
        if role in grouped[group]:
            raise ValueError("final reserve repeats a group role")
        grouped[group][role] = row
    expected_roles = {"in_domain_test", "generator_holdout", "final_test"}
    if len(grouped) != int(config["input"]["expected_groups"]) or any(
        set(rows) != expected_roles for rows in grouped.values()
    ):
        raise ValueError("final reserve group topology changed")
    counts = Counter(
        f"{row['evaluation_role']}|{row['sample_kind']}|{row['generator']}"
        for row in manifest_rows
    )
    expected_counts = {
        str(key): int(value) for key, value in config["input"]["expected_counts"].items()
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"final reserve role counts changed: {dict(counts)}")

    conditions = {str(item["name"]): item for item in config["conditions"]}
    if set(conditions) != _required_conditions():
        raise ValueError("final reserve condition whitelist changed")
    stresses = {str(item["name"]): item for item in config["stresses"]}
    if set(stresses) != set(STRESSES):
        raise ValueError("final reserve stress whitelist changed")
    model_config = config["models"]
    if tuple(int(seed) for seed in model_config["family_seeds"]) != FAMILY_SEEDS:
        raise ValueError("final model seed family changed")

    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    encoder_path = _resolve(scratch, model_config["encoder_weights"])
    if _sha256(encoder_path) != model_config["encoder_weights_sha256"]:
        raise ValueError("final encoder weights changed")
    student_path = _resolve(project_root, model_config["student"]["checkpoint"])
    if _sha256(student_path) != model_config["student"]["checkpoint_sha256"]:
        raise ValueError("final student checkpoint changed")
    student_saved = torch.load(student_path, map_location="cpu", weights_only=True)
    student = ResNet18UNet()
    student.load_state_dict(student_saved["model_state"], strict=True)
    student = student.to(device).eval().requires_grad_(False)
    pair_models: dict[str, torch.nn.Module] = {}
    pair_hashes: dict[str, str] = {}
    for name, specification in model_config["pair_models"].items():
        checkpoint = _resolve(project_root, specification["checkpoint"])
        expected = str(specification["checkpoint_sha256"])
        if _sha256(checkpoint) != expected:
            raise ValueError(f"final pair checkpoint changed: {name}")
        model = _load_teacher(encoder_path, model_config["teacher_conv1_coefficients"])
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state"], strict=True)
        pair_models[str(name)] = model.to(device).eval().requires_grad_(False)
        pair_hashes[str(name)] = expected

    paths = config["paths"]
    score_cache_dir = _resolve(project_root, paths["score_cache_dir"])
    alignment_cache_dir = _resolve(project_root, paths["alignment_cache_dir"])
    predictions_path = _resolve(project_root, paths["predictions"])
    alignments_path = _resolve(project_root, paths["alignments"])
    metrics_path = _resolve(project_root, paths["metrics"])
    comparisons_path = _resolve(project_root, paths["comparisons"])
    seed_table_path = _resolve(project_root, paths["seed_table"])
    aggregate_table_path = _resolve(project_root, paths["aggregate_table"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (
        score_cache_dir,
        alignment_cache_dir,
        predictions_path.parent,
        alignments_path.parent,
        metrics_path.parent,
        comparisons_path.parent,
        seed_table_path.parent,
        aggregate_table_path.parent,
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
    if int(preprocessing["score_cache_schema_version"]) != 3 or preprocessing["score_cache_dtype"] != "float32":
        raise ValueError("final evaluation requires schema-3 float32 score caches")
    inference = config["inference"]
    registration_config = config["registration"]
    ordered_groups = [
        {
            "source_group_id": group,
            **rows,
        }
        for group, rows in sorted(grouped.items())
    ]
    shuffled = _shuffled_group_map(
        ordered_groups, seed + int(config["controls"]["shuffle_seed_offset"])
    )
    bundles_by_group = {str(bundle["source_group_id"]): bundle for bundle in ordered_groups}
    image_cache: dict[str, np.ndarray] = {}

    def load_image(row: dict[str, Any]) -> np.ndarray:
        path = _resolve(scratch, row["image"])
        key = str(path)
        if key not in image_cache:
            if _sha256(path) != row["image_sha256"]:
                raise ValueError("final image SHA-256 changed")
            with Image.open(path) as handle:
                image_cache[key] = np.asarray(handle.convert("RGB"))
        return image_cache[key]

    def load_mask(row: dict[str, Any]) -> np.ndarray:
        path = _resolve(scratch, row["mask"])
        if _sha256(path) != row["mask_sha256"]:
            raise ValueError("final mask SHA-256 changed")
        with Image.open(path) as handle:
            return np.asarray(handle.convert("L")) > 0

    payloads = {name: {"forged": [], "authentic": []} for name in conditions}
    forged_scores: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in conditions
    }
    predictions: list[dict[str, Any]] = []
    alignment_records: dict[str, dict[str, Any]] = {}
    failures = 0
    score_cache_hits = 0
    alignment_cache_hits = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for group_index, bundle in enumerate(ordered_groups, start=1):
        group = str(bundle["source_group_id"])
        authentic_row = bundle["final_test"]
        authentic_native = load_image(authentic_row)
        in_domain = bundle["in_domain_test"]
        source_stratum = f"{in_domain['generator']}|{in_domain['source_dataset']}"
        candidates = []
        for role in FINAL_ROLES:
            row = bundle[role]
            image = load_image(row)
            mask = load_mask(row)
            if image.shape[:2] != mask.shape or authentic_native.shape[:2] != mask.shape:
                raise ValueError("final aligned pair geometry changed")
            candidates.append((role, "forged", row, image, mask))
        candidates.append(("final_test", "authentic", authentic_row, authentic_native, None))
        group_alignment_cache: dict[tuple[str, str], tuple[np.ndarray, str, dict[str, Any]]] = {}

        def aligned_reference(
            role: str,
            sample_kind: str,
            row: dict[str, Any],
            candidate: np.ndarray,
            geometry: str,
        ) -> tuple[np.ndarray, str, dict[str, Any]]:
            nonlocal alignment_cache_hits
            cache_key = (role, geometry)
            if cache_key in group_alignment_cache:
                return group_alignment_cache[cache_key]
            clean_reference = _resize_reference(authentic_native, candidate.shape[:2])
            oracle = _stress_homography(candidate.shape[:2], geometry, stresses)
            stressed_reference = _warp_reference(clean_reference, oracle, inverse=False)
            candidate_sha = str(row["image_sha256"])
            alignment_key = hashlib.sha256(
                json.dumps(
                    {
                        "candidate_sha256": candidate_sha,
                        "reference_sha256": authentic_row["image_sha256"],
                        "candidate_shape": list(candidate.shape),
                        "geometry": stresses.get(geometry, {"name": "clean"}),
                        "registration": registration_config,
                        "schema_version": preprocessing["alignment_cache_schema_version"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            cache_path = alignment_cache_dir / geometry / f"{alignment_key}.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as archive:
                    reference = archive["aligned_reference"].astype(np.uint8)
                    estimated = archive["estimated_homography"].astype(float)
                    errors = archive["corner_errors_pixels"].astype(float)
                    correlation = float(archive["ecc_correlation"].item())
                    phase_response = float(archive["phase_correlation_response"].item())
                    status = str(archive["alignment_status"].item())
                    failure_type = str(archive["alignment_failure_type"].item()) or None
                    failure_reason = str(archive["alignment_failure_reason"].item()) or None
                metadata = {
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
                reference, metadata = _estimate_ecc_alignment(
                    candidate, stressed_reference, oracle, registration_config
                )
                temporary = cache_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        aligned_reference=reference.astype(np.uint8),
                        estimated_homography=np.asarray(metadata["estimated_homography"], dtype=float),
                        corner_errors_pixels=np.asarray(metadata["corner_errors_pixels"], dtype=float),
                        ecc_correlation=np.asarray(math.nan if metadata["ecc_correlation"] is None else metadata["ecc_correlation"]),
                        phase_correlation_response=np.asarray(math.nan if metadata["phase_correlation_response"] is None else metadata["phase_correlation_response"]),
                        alignment_status=np.asarray(metadata["alignment_status"]),
                        alignment_failure_type=np.asarray(metadata["alignment_failure_type"] or ""),
                        alignment_failure_reason=np.asarray(metadata["alignment_failure_reason"] or ""),
                    )
                temporary.replace(cache_path)
            alignment_records.setdefault(
                alignment_key,
                {
                    "alignment_key": alignment_key,
                    "source_group_id": group,
                    "source_stratum": source_stratum,
                    "evaluation_role": role,
                    "sample_kind": sample_kind,
                    "geometry": geometry,
                    "alignment_cache": str(cache_path.relative_to(project_root)),
                    "paper_evidence": True,
                    "final_reserve_read": True,
                    **metadata,
                },
            )
            result = (reference, alignment_key, metadata)
            group_alignment_cache[cache_key] = result
            return result

        for condition_name, condition in conditions.items():
            for role, sample_kind, row, native_candidate, native_mask in candidates:
                record: dict[str, Any] = {
                    "record_id": f"{condition_name}:{role}:{group}",
                    "source_group_id": group,
                    "source_stratum": source_stratum,
                    "source_dataset": row["source_dataset"],
                    "evaluation_role": role,
                    "generator": row["generator"],
                    "condition": condition_name,
                    "sample_kind": sample_kind,
                    "status": "failed",
                    "paper_evidence": True,
                    "final_reserve_read": True,
                    "threshold_selected_on_final_reserve": False,
                }
                try:
                    candidate = _resize_image(native_candidate, int(preprocessing["max_side"]))
                    scorer = str(condition["scorer"])
                    geometry = str(condition["geometry"])
                    alignment_mode = str(condition["alignment"])
                    alignment_key = None
                    metadata: dict[str, Any] = {
                        "alignment_status": "not_requested",
                        "ecc_correlation": None,
                        "corner_errors_pixels": None,
                    }
                    reference = None
                    if scorer != "student":
                        if alignment_mode == "shuffled":
                            target = bundles_by_group[shuffled[group]]["final_test"]
                            reference = _resize_reference(load_image(target), candidate.shape[:2])
                        else:
                            clean_reference = _resize_reference(authentic_native, candidate.shape[:2])
                            oracle = _stress_homography(candidate.shape[:2], geometry, stresses)
                            stressed_reference = _warp_reference(clean_reference, oracle, inverse=False)
                            if alignment_mode == "unaligned":
                                reference = stressed_reference
                            elif alignment_mode == "ecc":
                                reference, alignment_key, metadata = aligned_reference(
                                    role, sample_kind, row, candidate, geometry
                                )
                            else:
                                raise ValueError(f"unsupported final alignment: {alignment_mode}")
                    model_identity = (
                        model_config["student"]["checkpoint_sha256"]
                        if scorer == "student"
                        else pair_hashes[scorer]
                        if scorer in pair_hashes
                        else "raw_difference_v1"
                    )
                    score_key = hashlib.sha256(
                        json.dumps(
                            {
                                "candidate_sha256": row["image_sha256"],
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
                    if score_path.is_file():
                        score_cache_hits += 1
                    else:
                        if scorer == "student":
                            probability = _infer_tiled(student, candidate, device, inference, preprocessing)
                        elif scorer in pair_models:
                            probability = _infer_pair_tiled(pair_models[scorer], candidate, reference, device, inference, preprocessing)
                        elif scorer == "raw_difference":
                            probability = _raw_difference(candidate, reference)
                        else:
                            raise ValueError(f"unsupported final scorer: {scorer}")
                        temporary = score_path.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(handle, scores=probability.astype(np.float32))
                        temporary.replace(score_path)
                    with np.load(score_path, allow_pickle=False) as archive:
                        probability = archive["scores"]
                    if probability.dtype != np.float32 or probability.shape != candidate.shape[:2] or not np.isfinite(probability).all():
                        raise ValueError("final score cache is invalid")
                    image_score = _top_fraction_mean(
                        probability, float(config["image_score"]["top_fraction"])
                    )
                    native_probability = cv2.resize(
                        probability,
                        (native_candidate.shape[1], native_candidate.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    pixel_threshold = float(condition["fixed_pixel_threshold"])
                    if sample_kind == "forged":
                        average_precision, pixel_auroc = _ranking_metrics(native_probability, native_mask)
                        predicted = native_probability >= pixel_threshold
                        tp = int(np.count_nonzero(predicted & native_mask))
                        fp = int(np.count_nonzero(predicted & ~native_mask))
                        fn = int(np.count_nonzero(~predicted & native_mask))
                        precision = tp / (tp + fp) if tp + fp else 0.0
                        recall = tp / (tp + fn) if tp + fn else 0.0
                        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
                        item = {
                            "source_group_id": group,
                            "source_stratum": source_stratum,
                            "evaluation_role": role,
                            "generator": row["generator"],
                            "macro_pixel_ap": average_precision,
                            "pixel_auroc": pixel_auroc,
                            "pixel_precision": precision,
                            "pixel_recall": recall,
                            "pixel_f1": f1,
                            "pixel_iou": iou,
                            "image_score": image_score,
                        }
                        payloads[condition_name]["forged"].append(item)
                        forged_scores[condition_name][f"{group}|{role}"] = {
                            **{key: item[key] for key in (
                                "source_group_id",
                                "source_stratum",
                                "evaluation_role",
                                "generator",
                            )},
                            "value": average_precision,
                        }
                        record.update(
                            {
                                "macro_pixel_ap": average_precision,
                                "pixel_auroc": pixel_auroc,
                                "pixel_precision": precision,
                                "pixel_recall": recall,
                                "pixel_f1": f1,
                                "pixel_iou": iou,
                            }
                        )
                    else:
                        pixel_fpr = float(np.mean(native_probability >= pixel_threshold))
                        payloads[condition_name]["authentic"].append(
                            {"source_group_id": group, "pixel_fpr": pixel_fpr, "image_score": image_score}
                        )
                        record["authentic_pixel_fpr"] = pixel_fpr
                    record.update(
                        {
                            "status": "ok",
                            "scorer": scorer,
                            "geometry": geometry,
                            "alignment": alignment_mode,
                            "alignment_key": alignment_key,
                            "fixed_pixel_threshold": pixel_threshold,
                            "fixed_image_threshold": float(condition["fixed_image_threshold"]),
                            "image_score_top_fraction": float(config["image_score"]["top_fraction"]),
                            "image_score": image_score,
                            "score_cache": str(score_path.relative_to(project_root)),
                            "score_cache_dtype": str(probability.dtype),
                            "score_shape": list(probability.shape),
                            "native_shape": list(native_probability.shape),
                            **metadata,
                        }
                    )
                except Exception as error:
                    failures += 1
                    record["failure_type"] = type(error).__name__
                    record["failure_reason"] = str(error)
                    logging.exception("record_id=%s failed", record["record_id"])
                predictions.append(record)
        _write_jsonl(predictions_path, predictions)
        _write_jsonl(alignments_path, list(alignment_records.values()))
        logging.info("completed_groups=%d total_groups=%d failures=%d", group_index, len(ordered_groups), failures)

    expected_predictions = len(ordered_groups) * len(conditions) * 3
    complete = failures == 0 and len(predictions) == expected_predictions and all(
        len(payload["forged"]) == len(ordered_groups) * 2
        and len(payload["authentic"]) == len(ordered_groups)
        for payload in payloads.values()
    )
    if not complete:
        output = {
            "experiment": config["experiment"],
            "status": "final_reserve_failed_incomplete",
            "paper_evidence": True,
            "final_reserve_read": True,
            "successful_prediction_records": len(predictions) - failures,
            "failed_prediction_records": failures,
            "expected_prediction_records": expected_predictions,
            "method_change_performed": False,
        }
        _write_json(summary_path, output)
        if runtime["require_all_records"]:
            raise RuntimeError(f"final reserve evaluation failed for {failures} records")
        return output

    metrics = {
        name: _aggregate_fixed_condition(
            payload,
            float(conditions[name]["fixed_pixel_threshold"]),
            float(conditions[name]["fixed_image_threshold"]),
        )
        for name, payload in payloads.items()
    }
    errors = [
        float(error)
        for item in alignment_records.values()
        if item["geometry"] in STRESSES
        for error in item["corner_errors_pixels"]
    ]
    converged = sum(item["alignment_status"] == "ecc_converged" for item in alignment_records.values())
    registration = {
        "attempts": len(alignment_records),
        "converged": converged,
        "fallbacks": len(alignment_records) - converged,
        "convergence_rate": converged / len(alignment_records),
        "controlled_stress_corner_error_median_pixels": float(np.median(errors)),
        "controlled_stress_corner_error_p95_pixels": float(np.quantile(errors, 0.95)),
    }
    robust_means = {
        geometry: _mean_score_maps(
            [forged_scores[f"robust_{seed}_{geometry}_ecc"] for seed in FAMILY_SEEDS]
        )
        for geometry in GEOMETRIES
    }
    shuffled_mean = _mean_score_maps(
        [forged_scores[f"robust_{seed}_shuffled_clean"] for seed in FAMILY_SEEDS]
    )
    unaligned_mean = _mean_score_maps(
        [forged_scores[f"robust_{seed}_affine_unaligned"] for seed in FAMILY_SEEDS]
    )
    comparison_pairs: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {
        "robust_mean_clean_minus_student": (robust_means["clean"], forged_scores["student_clean"]),
        "robust_mean_clean_minus_shuffled": (robust_means["clean"], shuffled_mean),
        "robust_mean_affine_ecc_minus_unaligned": (robust_means["affine"], unaligned_mean),
    }
    for stress in STRESSES:
        comparison_pairs[f"robust_mean_minus_baseline__{stress}"] = (
            robust_means[stress], forged_scores[f"baseline_{stress}_ecc"]
        )
        comparison_pairs[f"robust_mean_minus_raw__{stress}"] = (
            robust_means[stress], forged_scores[f"raw_difference_{stress}_ecc"]
        )
    comparisons = {
        name: _clustered_paired_bootstrap(
            left,
            right,
            int(config["bootstrap"]["seed"]) + index,
            int(config["bootstrap"]["resamples"]),
            float(config["bootstrap"]["confidence_level"]),
        )
        for index, (name, (left, right)) in enumerate(comparison_pairs.items())
    }
    seed_rows = []
    for seed_value in FAMILY_SEEDS:
        clean = metrics[f"robust_{seed_value}_clean_ecc"]
        stressed = [metrics[f"robust_{seed_value}_{stress}_ecc"] for stress in STRESSES]
        seed_rows.append(
            {
                "training_seed": seed_value,
                "checkpoint_sha256": pair_hashes[f"robust_{seed_value}"],
                "clean_role_macro_generator_macro_pixel_ap": clean["role_macro_generator_macro_pixel_ap"],
                "minimum_stressed_role_macro_generator_macro_pixel_ap": min(item["role_macro_generator_macro_pixel_ap"] for item in stressed),
                "clean_in_domain_pixel_ap": clean["pixel_ap__in_domain_test"],
                "clean_output_holdout_pixel_ap": clean["pixel_ap__generator_holdout"],
                "minimum_stressed_in_domain_pixel_ap": min(item["pixel_ap__in_domain_test"] for item in stressed),
                "minimum_stressed_output_holdout_pixel_ap": min(item["pixel_ap__generator_holdout"] for item in stressed),
                "maximum_authentic_pixel_fpr": max(metrics[f"robust_{seed_value}_{geometry}_ecc"]["authentic_document_macro_pixel_fpr"] for geometry in GEOMETRIES),
                "clean_image_auroc": clean["image_auroc"],
                "clean_image_tpr_at_development_frozen_threshold": clean["image_tpr_at_development_frozen_threshold"],
                "paper_evidence": True,
            }
        )
    aggregate_fields = [
        "clean_role_macro_generator_macro_pixel_ap",
        "minimum_stressed_role_macro_generator_macro_pixel_ap",
        "clean_in_domain_pixel_ap",
        "clean_output_holdout_pixel_ap",
        "minimum_stressed_in_domain_pixel_ap",
        "minimum_stressed_output_holdout_pixel_ap",
        "maximum_authentic_pixel_fpr",
        "clean_image_auroc",
        "clean_image_tpr_at_development_frozen_threshold",
    ]
    aggregate_rows = [
        {
            "metric": field,
            **_descriptive([float(row[field]) for row in seed_rows]),
            "seed_count": len(FAMILY_SEEDS),
            "sample_standard_deviation_ddof": 1,
            "paper_evidence": True,
        }
        for field in aggregate_fields
    ]
    decision = _final_decision(metrics, comparisons, registration, config["final_gate"])
    _write_csv(metrics_path, [{"condition": name, **value} for name, value in metrics.items()])
    _write_csv(
        comparisons_path,
        [{"comparison": name, **value, "paper_evidence": True} for name, value in comparisons.items()],
    )
    _write_csv(seed_table_path, seed_rows)
    _write_csv(aggregate_table_path, aggregate_rows)
    output = {
        "experiment": config["experiment"],
        "status": "final_reserve_gate_passed" if decision["overall_pass"] else "final_reserve_gate_failed",
        "paper_evidence": True,
        "final_reserve_read": True,
        "output_unseen_not_generator_unseen": True,
        "method_change_performed": False,
        "checkpoint_selection_used": False,
        "threshold_selection_used": False,
        "best_seed_selection_used": False,
        "score_map_ensemble_used": False,
        "strong_baseline_final_evaluation_required": True,
        "strong_baseline_final_evaluation_complete": False,
        "selected_groups": len(ordered_groups),
        "successful_prediction_records": len(predictions),
        "failed_prediction_records": 0,
        "score_cache_hits": score_cache_hits,
        "alignment_cache_hits": alignment_cache_hits,
        "conditions": metrics,
        "seed_results": seed_rows,
        "aggregate_statistics": {row["metric"]: {key: value for key, value in row.items() if key not in ("metric", "paper_evidence")} for row in aggregate_rows},
        "comparisons": comparisons,
        "registration": registration,
        "final_gate": config["final_gate"],
        "decision": decision,
        "input_manifest_sha256": _sha256(manifest_path),
        "holdout_freeze_id": config["input"]["expected_freeze_id"],
        "protocol_sha256": _sha256(protocol_path),
        "config_sha256": _sha256(config_path),
        "model_checkpoint_sha256": {"student": model_config["student"]["checkpoint_sha256"], **pair_hashes},
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
            "seed_table": str(seed_table_path.relative_to(project_root)),
            "seed_table_sha256": _sha256(seed_table_path),
            "aggregate_table": str(aggregate_table_path.relative_to(project_root)),
            "aggregate_table_sha256": _sha256(aggregate_table_path),
            "score_cache_dir": str(score_cache_dir.relative_to(project_root)),
            "alignment_cache_dir": str(alignment_cache_dir.relative_to(project_root)),
            "log": str(log_path.relative_to(project_root)),
        },
    }
    _write_json(summary_path, output)
    logging.info("status=%s final_reserve_read=true", output["status"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
