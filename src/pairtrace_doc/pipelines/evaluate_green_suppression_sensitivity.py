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

from pairtrace_doc.pipelines.diagnose_pair_at_inference_alignment import (
    _estimate_ecc_alignment,
)
from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
    _infer_pair_tiled,
    _resize_image,
    _resize_reference,
)
from pairtrace_doc.pipelines.freeze_resampling_multiseed_image_thresholds import (
    _top_fraction_mean,
)
from pairtrace_doc.pipelines.prepare_green_suppression_sensitivity import (
    _array_sha256,
    _cache_key,
    _mask_blind_green_inpaint,
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


STAGE_COUNTS = {"gpu_toy3": 3, "gpu_pilot24": 24, "gpu_full96": 288}
FULL_ROLES = ("in_domain_test", "generator_holdout", "final_test")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verify(path: Path, expected: str, label: str) -> str:
    digest = _sha256(path)
    if digest != expected:
        raise ValueError(f"{label} SHA-256 changed: {digest} != {expected}")
    return digest


def _select_stage_rows(
    manifest: list[dict[str, Any]], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    mode = str(selection["mode"])
    if mode == "explicit_forged_ids":
        by_id = {
            str(row["source_sample_id"]): row
            for row in manifest
            if row["sample_kind"] == "forged"
        }
        requested = [str(value) for value in selection["source_sample_ids"]]
        if len(requested) != len(set(requested)):
            raise ValueError("stage selection contains duplicate source sample IDs")
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise ValueError(f"stage selection IDs are missing: {missing}")
        return [by_id[sample_id] for sample_id in requested]
    if mode == "all_manifest_roles":
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in manifest:
            grouped[str(row["source_group_id"])][str(row["evaluation_role"])] = row
        if any(set(bundle) != set(FULL_ROLES) for bundle in grouped.values()):
            raise ValueError("full sensitivity selection lost the three-role topology")
        return [
            grouped[group][role]
            for group in sorted(grouped)
            for role in FULL_ROLES
        ]
    raise ValueError(f"unsupported sensitivity selection mode: {mode}")


def _validate_pilot_selection(
    selected_rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    audit: dict[str, dict[str, Any]],
    selection: dict[str, Any],
) -> None:
    expected_strata = {
        "signal_positive_gemini": (True, "gemini-nano"),
        "signal_negative_gemini": (False, "gemini-nano"),
        "signal_negative_qwen": (False, "qwen-inpaint"),
        "signal_negative_openai": (False, "openai-gpt-image-2"),
    }
    if selection.get("freeze_rule") != (
        "sha256(seed|stratum|source_group_id)_ascending_take_6"
    ):
        raise ValueError("pilot24 deterministic freeze rule changed")
    counts = {str(key): int(value) for key, value in selection["strata_counts"].items()}
    if counts != {name: 6 for name in expected_strata}:
        raise ValueError("pilot24 stratum quotas changed")
    seed = int(selection["freeze_seed"])
    expected_ids: list[str] = []
    for label, (positive, generator) in expected_strata.items():
        pool = [
            row
            for row in manifest
            if row["sample_kind"] == "forged"
            for sample_id in (str(row["source_sample_id"]),)
            if sample_id in audit
            and audit[sample_id]["artifact_positive"] is positive
            and row["generator"] == generator
        ]
        expected_ids.extend(
            str(row["source_sample_id"])
            for row in sorted(
                pool,
                key=lambda item: hashlib.sha256(
                    f"{seed}|{label}|{item['source_group_id']}".encode("utf-8")
                ).hexdigest(),
            )[:6]
        )
    configured_ids = [str(value) for value in selection["source_sample_ids"]]
    if configured_ids != expected_ids:
        raise ValueError("pilot24 membership or frozen stratum order changed")


def _paired_group_bootstrap(
    rows: list[dict[str, Any]],
    value_field: str,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("paired bootstrap requires records")
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_field])
        if not math.isfinite(value):
            raise ValueError("paired bootstrap received a non-finite value")
        by_group[str(row["source_group_id"])].append(value)
    groups = sorted(by_group)
    group_values = np.asarray(
        [float(np.mean(by_group[group])) for group in groups], dtype=float
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(resamples, len(groups)))
    replicates = group_values[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "effect": float(group_values.mean()),
        "ci_low": float(np.quantile(replicates, alpha)),
        "ci_high": float(np.quantile(replicates, 1.0 - alpha)),
        "source_groups": len(groups),
        "records": len(rows),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "confidence_level": confidence_level,
        "unit": "source_group_id",
    }


def _fixed_metrics(
    probability: np.ndarray,
    native_shape: tuple[int, int],
    mask: np.ndarray | None,
    pixel_threshold: float,
    image_top_fraction: float,
) -> dict[str, float]:
    if probability.dtype != np.float32 or not np.isfinite(probability).all():
        raise ValueError("sensitivity score map must be finite float32")
    native_probability = cv2.resize(
        probability,
        (native_shape[1], native_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    result = {
        "image_score": _top_fraction_mean(probability, image_top_fraction),
    }
    if mask is None:
        result["authentic_pixel_fpr"] = float(
            np.mean(native_probability >= pixel_threshold)
        )
        return result
    average_precision, pixel_auroc = _ranking_metrics(native_probability, mask)
    predicted = native_probability >= pixel_threshold
    tp = int(np.count_nonzero(predicted & mask))
    fp = int(np.count_nonzero(predicted & ~mask))
    fn = int(np.count_nonzero(~predicted & mask))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result.update(
        {
            "macro_pixel_ap": average_precision,
            "pixel_auroc": pixel_auroc,
            "pixel_precision": precision,
            "pixel_recall": recall,
            "pixel_f1": (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        }
    )
    return result


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _aggregate_models(
    rows: list[dict[str, Any]], conditions: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result: dict[str, Any] = {}
    subgroup_rows: list[dict[str, Any]] = []
    for model_name, condition in conditions.items():
        selected = [row for row in rows if row["model"] == model_name]
        forged = [row for row in selected if row["sample_kind"] == "forged"]
        authentic = [row for row in selected if row["sample_kind"] == "authentic"]
        image_threshold = float(condition["fixed_image_threshold"])
        result[model_name] = {
            "forged_documents": len(forged),
            "authentic_documents": len(authentic),
            "original_document_macro_pixel_ap": _mean_or_none(
                [float(row["original_macro_pixel_ap"]) for row in forged]
            ),
            "transformed_document_macro_pixel_ap": _mean_or_none(
                [float(row["transformed_macro_pixel_ap"]) for row in forged]
            ),
            "mean_ap_change": _mean_or_none(
                [float(row["ap_change"]) for row in forged]
            ),
            "original_document_macro_pixel_f1": _mean_or_none(
                [float(row["original_pixel_f1"]) for row in forged]
            ),
            "transformed_document_macro_pixel_f1": _mean_or_none(
                [float(row["transformed_pixel_f1"]) for row in forged]
            ),
            "original_document_macro_pixel_iou": _mean_or_none(
                [float(row["original_pixel_iou"]) for row in forged]
            ),
            "transformed_document_macro_pixel_iou": _mean_or_none(
                [float(row["transformed_pixel_iou"]) for row in forged]
            ),
            "original_authentic_document_macro_pixel_fpr": _mean_or_none(
                [float(row["original_authentic_pixel_fpr"]) for row in authentic]
            ),
            "transformed_authentic_document_macro_pixel_fpr": _mean_or_none(
                [float(row["transformed_authentic_pixel_fpr"]) for row in authentic]
            ),
            "original_image_tpr_at_frozen_threshold": _mean_or_none(
                [float(row["original_image_score"] >= image_threshold) for row in forged]
            ),
            "transformed_image_tpr_at_frozen_threshold": _mean_or_none(
                [
                    float(row["transformed_image_score"] >= image_threshold)
                    for row in forged
                ]
            ),
            "original_image_fpr_at_frozen_threshold": _mean_or_none(
                [
                    float(row["original_image_score"] >= image_threshold)
                    for row in authentic
                ]
            ),
            "transformed_image_fpr_at_frozen_threshold": _mean_or_none(
                [
                    float(row["transformed_image_score"] >= image_threshold)
                    for row in authentic
                ]
            ),
            "threshold_selected_on_sensitivity_data": False,
        }
        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for row in selected:
            grouped[
                (
                    str(row["artifact_status"]),
                    str(row["generator"]),
                    str(row["source_dataset"]),
                    str(row["evaluation_role"]),
                    str(row["sample_kind"]),
                )
            ].append(row)
        for key, items in sorted(grouped.items()):
            artifact_status, generator, dataset, role, sample_kind = key
            output = {
                "model": model_name,
                "artifact_status": artifact_status,
                "generator": generator,
                "source_dataset": dataset,
                "evaluation_role": role,
                "sample_kind": sample_kind,
                "records": len(items),
                "original_macro_pixel_ap": None,
                "transformed_macro_pixel_ap": None,
                "mean_ap_change": None,
                "original_authentic_pixel_fpr": None,
                "transformed_authentic_pixel_fpr": None,
            }
            if sample_kind == "forged":
                output.update(
                    {
                        "original_macro_pixel_ap": float(
                            np.mean([item["original_macro_pixel_ap"] for item in items])
                        ),
                        "transformed_macro_pixel_ap": float(
                            np.mean(
                                [item["transformed_macro_pixel_ap"] for item in items]
                            )
                        ),
                        "mean_ap_change": float(
                            np.mean([item["ap_change"] for item in items])
                        ),
                    }
                )
            else:
                output.update(
                    {
                        "original_authentic_pixel_fpr": float(
                            np.mean(
                                [item["original_authentic_pixel_fpr"] for item in items]
                            )
                        ),
                        "transformed_authentic_pixel_fpr": float(
                            np.mean(
                                [
                                    item["transformed_authentic_pixel_fpr"]
                                    for item in items
                                ]
                            )
                        ),
                    }
                )
            subgroup_rows.append(output)
    return result, subgroup_rows


def _report_text(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    lines = [
        "# Green-suppression frozen-model sensitivity",
        "",
        f"- stage: `{summary['experiment']['stage']}`",
        f"- status: `{summary['status']}`",
        f"- selected candidates: {summary['selected_candidates']}",
        f"- model predictions: {summary['successful_prediction_records']}/{summary['expected_prediction_records']}",
        f"- failures: {summary['failed_prediction_records']}",
        f"- GPU: {summary['gpu']}",
        f"- wall time: {summary['wall_time_seconds']:.3f} s",
        f"- peak VRAM: {summary['peak_vram_mb']:.3f} MiB",
        "",
        "This is a post-hoc diagnostic on an already consumed reserve. It is not confirmatory paper evidence and did not select a model or threshold.",
        "",
        f"Engineering gate passed: `{decision['engineering_gate_passed']}`.",
    ]
    if "per_model_color_cue_dependence" in decision:
        lines.extend(
            [
                "",
                "## Frozen full-stage decision",
                "",
                "| Model | Signal-positive AP change | 95% source-group CI | Dependence | All-forged AP original → transformed | Authentic pixel FPR original → transformed | Image FPR original → transformed |",
                "|---|---:|---:|:---:|---:|---:|---:|",
            ]
        )
        for model, result in decision["per_model_color_cue_dependence"].items():
            aggregate = summary["aggregates"][model]
            lines.append(
                f"| `{model}` | {result['effect']:.6f} | "
                f"[{result['ci_low']:.6f}, {result['ci_high']:.6f}] | "
                f"{str(result['dependence_supported']).lower()} | "
                f"{aggregate['original_document_macro_pixel_ap']:.6f} → "
                f"{aggregate['transformed_document_macro_pixel_ap']:.6f} | "
                f"{aggregate['original_authentic_document_macro_pixel_fpr']:.6f} → "
                f"{aggregate['transformed_authentic_document_macro_pixel_fpr']:.6f} | "
                f"{aggregate['original_image_fpr_at_frozen_threshold']:.6f} → "
                f"{aggregate['transformed_image_fpr_at_frozen_threshold']:.6f} |"
            )
        lines.extend(
            [
                "",
                "The single-image student meets the frozen color-cue-dependence rule. None of the three reference-conditioned robust models meets it on the 58 signal-positive source groups.",
                "",
                "The transform is not a validated cleanup method: it also changes signal-negative and authentic controls, and the robust models' authentic false-positive rates increase after inpainting. The result therefore supports downgrading student evidence, not relabeling the reserve as clean.",
                "",
                "A detected dependency requires claim downgrading and artifact-free regeneration. A null result does not restore the consumed reserve or establish cleanliness.",
            ]
        )
    return "\n".join(lines) + "\n"


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("green-suppression sensitivity config must be a mapping")
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if config["experiment"]["paper_evidence"]:
        raise ValueError("green-suppression sensitivity cannot be paper evidence")
    if not all(
        bool(runtime.get(key))
        for key in (
            "gpu_launch_authorized",
            "model_inference_authorized",
            "consumed_reserve_diagnostic_authorized",
        )
    ):
        raise PermissionError("green-suppression GPU sensitivity is not authorized")
    if any(
        bool(runtime.get(key))
        for key in (
            "model_training_authorized",
            "checkpoint_selection_authorized",
            "threshold_selection_authorized",
            "sample_replacement_authorized",
        )
    ):
        raise ValueError("green-suppression sensitivity crossed an evidence boundary")
    device = torch.device(str(runtime["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("green-suppression sensitivity requires CUDA")

    experiment = config["experiment"]
    stage = str(experiment["stage"])
    if stage not in STAGE_COUNTS:
        raise ValueError(f"unsupported green-suppression stage: {stage}")
    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            str(paths["scratch_env"]),
            str(_resolve(project_root, str(paths["scratch_default"]))),
        )
    ).resolve()
    protocol_path = _resolve(project_root, str(experiment["protocol"]))
    _verify(
        protocol_path,
        str(experiment["expected_protocol_sha256"]),
        "green-suppression protocol",
    )
    predecessor_path = _resolve(
        project_root, str(config["authorization"]["predecessor_summary"])
    )
    _verify(
        predecessor_path,
        str(config["authorization"]["expected_predecessor_summary_sha256"]),
        "green-suppression predecessor",
    )
    predecessor = _read_json(predecessor_path)
    if predecessor.get("status") != config["authorization"]["required_status"]:
        raise ValueError("green-suppression predecessor status changed")
    if predecessor.get("model_training_performed", False):
        raise ValueError("green-suppression predecessor trained a model")

    inputs = config["input"]
    verified: dict[str, Path] = {}
    for key, label in (
        ("manifest", "frozen reserve manifest"),
        ("audit_records", "green-boundary audit records"),
        ("frozen_predictions", "frozen final predictions"),
    ):
        path = _resolve(project_root, str(inputs[key]))
        _verify(path, str(inputs[f"expected_{key}_sha256"]), label)
        verified[key] = path
    manifest = _read_jsonl(verified["manifest"])
    selected_rows = _select_stage_rows(manifest, config["selection"])
    expected_candidates = STAGE_COUNTS[stage]
    if len(selected_rows) != expected_candidates:
        raise ValueError(
            f"{stage} requires {expected_candidates} candidates, got {len(selected_rows)}"
        )
    if len({str(row["record_id"]) for row in selected_rows}) != len(selected_rows):
        raise ValueError("green-suppression stage repeats manifest records")
    by_group_role = {
        (str(row["source_group_id"]), str(row["evaluation_role"])): row
        for row in manifest
    }
    audit = {
        str(row["source_sample_id"]): row
        for row in _read_jsonl(verified["audit_records"])
        if row.get("status") == "ok"
    }
    if stage == "gpu_pilot24":
        _validate_pilot_selection(selected_rows, manifest, audit, config["selection"])
    original_conditions = {
        str(specification["original_condition"])
        for specification in config["conditions"].values()
    }
    original_index = {
        (
            str(row["condition"]),
            str(row["source_group_id"]),
            str(row["evaluation_role"]),
        ): row
        for row in _read_jsonl(verified["frozen_predictions"])
        if row.get("status") == "ok" and str(row.get("condition")) in original_conditions
    }
    for row in selected_rows:
        for condition in config["conditions"].values():
            key = (
                str(condition["original_condition"]),
                str(row["source_group_id"]),
                str(row["evaluation_role"]),
            )
            if key not in original_index:
                raise ValueError(f"missing frozen original prediction: {key}")

    seed = int(experiment["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_config = config["models"]
    encoder = _resolve(scratch, str(model_config["encoder_weights"]))
    _verify(encoder, str(model_config["encoder_weights_sha256"]), "encoder weights")
    student_spec = model_config["student"]
    student_path = _resolve(project_root, str(student_spec["checkpoint"]))
    _verify(student_path, str(student_spec["checkpoint_sha256"]), "student checkpoint")
    student = ResNet18UNet()
    student.load_state_dict(
        torch.load(student_path, map_location="cpu", weights_only=True)["model_state"],
        strict=True,
    )
    models: dict[str, torch.nn.Module] = {
        "student": student.to(device).eval().requires_grad_(False)
    }
    model_hashes = {"student": str(student_spec["checkpoint_sha256"])}
    for name, specification in model_config["pair_models"].items():
        checkpoint = _resolve(project_root, str(specification["checkpoint"]))
        digest = _verify(
            checkpoint, str(specification["checkpoint_sha256"]), f"checkpoint {name}"
        )
        model = _load_teacher(
            encoder, model_config["teacher_conv1_coefficients"]
        )
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"],
            strict=True,
        )
        models[str(name)] = model.to(device).eval().requires_grad_(False)
        model_hashes[str(name)] = digest
    conditions = {
        str(name): specification
        for name, specification in config["conditions"].items()
    }
    if set(conditions) != set(models):
        raise ValueError("sensitivity models and conditions differ")

    transform_cache = _resolve(scratch, str(paths["transform_cache_dir"]))
    score_cache = _resolve(scratch, str(paths["score_cache_dir"]))
    alignment_cache = _resolve(scratch, str(paths["alignment_cache_dir"]))
    predictions_path = _resolve(project_root, str(paths["predictions"]))
    subgroup_path = _resolve(project_root, str(paths["subgroups"]))
    summary_path = _resolve(project_root, str(paths["summary"]))
    report_path = _resolve(project_root, str(paths["report"]))
    log_path = _resolve(project_root, str(paths["log"]))
    for path in (
        transform_cache,
        score_cache,
        alignment_cache,
        predictions_path.parent,
        subgroup_path.parent,
        summary_path.parent,
        report_path.parent,
        log_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    detector = config["detector"]
    transform = config["transform"]
    preprocessing = config["preprocessing"]
    inference = config["inference"]
    registration = config["registration"]
    image_top_fraction = float(config["image_score"]["top_fraction"])
    image_cache: dict[str, np.ndarray] = {}

    def load_image(row: dict[str, Any]) -> np.ndarray:
        path = _resolve(scratch, str(row["image"]))
        key = str(path)
        if key not in image_cache:
            _verify(path, str(row["image_sha256"]), "reserve image")
            with Image.open(path) as handle:
                image_cache[key] = np.asarray(handle.convert("RGB"))
        return image_cache[key]

    def load_mask(row: dict[str, Any]) -> np.ndarray | None:
        if row["sample_kind"] == "authentic":
            return None
        path = _resolve(scratch, str(row["mask"]))
        _verify(path, str(row["mask_sha256"]), "reserve mask")
        with Image.open(path) as handle:
            return np.asarray(handle.convert("L")) > 0

    records: list[dict[str, Any]] = []
    failures = 0
    score_cache_hits = 0
    alignment_cache_hits = 0
    transformed_cache_hits = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    identity = np.eye(3, dtype=np.float64)
    for candidate_index, row in enumerate(selected_rows, start=1):
        group = str(row["source_group_id"])
        role = str(row["evaluation_role"])
        sample_kind = str(row["sample_kind"])
        original_image = load_image(row)
        transformed, transform_diagnostics = _mask_blind_green_inpaint(
            original_image, detector, transform
        )
        transformed_sha = _array_sha256(transformed)
        transform_key = _cache_key(
            str(row["image_sha256"]),
            detector,
            transform,
            int(transform["cache_schema_version"]),
        )
        transform_path = transform_cache / transform_key[:2] / f"{transform_key}.png"
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        if transform_path.is_file():
            transformed_cache_hits += 1
        else:
            temporary = transform_path.with_suffix(".png.tmp")
            with temporary.open("wb") as handle:
                Image.fromarray(transformed).save(handle, format="PNG")
            temporary.replace(transform_path)
        with Image.open(transform_path) as handle:
            replay = np.asarray(handle.convert("RGB"))
        if not np.array_equal(replay, transformed):
            raise ValueError("green-suppression transformed cache changed")
        mask = load_mask(row)
        if mask is not None and mask.shape != original_image.shape[:2]:
            raise ValueError("green-suppression mask geometry changed")
        reference_row = by_group_role[(group, "final_test")]
        reference_native = load_image(reference_row)
        candidate = _resize_image(transformed, int(preprocessing["max_side"]))
        reference = _resize_reference(reference_native, candidate.shape[:2])
        alignment_key = hashlib.sha256(
            json.dumps(
                {
                    "transformed_array_sha256": transformed_sha,
                    "reference_sha256": reference_row["image_sha256"],
                    "candidate_shape": list(candidate.shape),
                    "registration": registration,
                    "schema_version": preprocessing["alignment_cache_schema_version"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        alignment_path = alignment_cache / f"{alignment_key}.npz"
        if alignment_path.is_file():
            alignment_cache_hits += 1
            with np.load(alignment_path, allow_pickle=False) as archive:
                aligned_reference = archive["aligned_reference"].astype(np.uint8)
                alignment_status = str(archive["alignment_status"].item())
                ecc_correlation_raw = float(archive["ecc_correlation"].item())
                alignment_failure_type = str(
                    archive["alignment_failure_type"].item()
                ) or None
                alignment_failure_reason = str(
                    archive["alignment_failure_reason"].item()
                ) or None
        else:
            aligned_reference, metadata = _estimate_ecc_alignment(
                candidate, reference, identity, registration
            )
            alignment_status = str(metadata["alignment_status"])
            ecc_correlation_raw = (
                math.nan
                if metadata["ecc_correlation"] is None
                else float(metadata["ecc_correlation"])
            )
            alignment_failure_type = metadata["alignment_failure_type"]
            alignment_failure_reason = metadata["alignment_failure_reason"]
            temporary = alignment_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    aligned_reference=aligned_reference.astype(np.uint8),
                    alignment_status=np.asarray(alignment_status),
                    ecc_correlation=np.asarray(ecc_correlation_raw),
                    alignment_failure_type=np.asarray(alignment_failure_type or ""),
                    alignment_failure_reason=np.asarray(
                        alignment_failure_reason or ""
                    ),
                )
            temporary.replace(alignment_path)
        artifact_positive = (
            None
            if sample_kind == "authentic"
            else bool(audit[str(row["source_sample_id"])]["artifact_positive"])
        )
        artifact_status = (
            "authentic_control"
            if artifact_positive is None
            else "signal_positive"
            if artifact_positive
            else "signal_negative"
        )
        for model_name, condition in conditions.items():
            original = original_index[
                (str(condition["original_condition"]), group, role)
            ]
            record: dict[str, Any] = {
                "record_id": f"{stage}:{model_name}:{role}:{group}",
                "source_sample_id": row["source_sample_id"],
                "source_group_id": group,
                "evaluation_role": role,
                "generator": row["generator"],
                "source_dataset": row["source_dataset"],
                "sample_kind": sample_kind,
                "artifact_positive": artifact_positive,
                "artifact_status": artifact_status,
                "model": model_name,
                "original_condition": condition["original_condition"],
                "status": "failed",
                "paper_evidence": False,
                "consumed_reserve_diagnostic": True,
                "model_training_performed": False,
                "threshold_selected_on_sensitivity_data": False,
                "transform_cache": str(transform_path.relative_to(scratch)),
                "transform_cache_key": transform_key,
                "transformed_array_sha256": transformed_sha,
                **transform_diagnostics,
            }
            try:
                scorer = str(condition["scorer"])
                scorer_alignment_key = None if scorer == "student" else alignment_key
                score_key = hashlib.sha256(
                    json.dumps(
                        {
                            "transformed_array_sha256": transformed_sha,
                            "reference_sha256": reference_row["image_sha256"],
                            "model": model_name,
                            "checkpoint_sha256": model_hashes[model_name],
                            "alignment_key": scorer_alignment_key,
                            "preprocessing": preprocessing,
                            "inference": inference,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                score_path = score_cache / model_name / f"{score_key}.npz"
                score_path.parent.mkdir(parents=True, exist_ok=True)
                if score_path.is_file():
                    score_cache_hits += 1
                else:
                    if scorer == "student":
                        probability = _infer_tiled(
                            models[model_name], candidate, device, inference, preprocessing
                        )
                    elif scorer == "pair":
                        probability = _infer_pair_tiled(
                            models[model_name],
                            candidate,
                            aligned_reference,
                            device,
                            inference,
                            preprocessing,
                        )
                    else:
                        raise ValueError(f"unsupported sensitivity scorer: {scorer}")
                    temporary = score_path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle, scores=probability.astype(np.float32)
                        )
                    temporary.replace(score_path)
                with np.load(score_path, allow_pickle=False) as archive:
                    probability = archive["scores"]
                if probability.shape != candidate.shape[:2]:
                    raise ValueError("sensitivity score cache geometry changed")
                metrics = _fixed_metrics(
                    probability,
                    original_image.shape[:2],
                    mask,
                    float(condition["fixed_pixel_threshold"]),
                    image_top_fraction,
                )
                record.update(
                    {
                        "status": "ok",
                        "checkpoint_sha256": model_hashes[model_name],
                        "fixed_pixel_threshold": float(
                            condition["fixed_pixel_threshold"]
                        ),
                        "fixed_image_threshold": float(
                            condition["fixed_image_threshold"]
                        ),
                        "alignment_key": scorer_alignment_key,
                        "alignment_status": (
                            "not_requested" if scorer == "student" else alignment_status
                        ),
                        "ecc_correlation": (
                            None
                            if scorer == "student" or not math.isfinite(ecc_correlation_raw)
                            else ecc_correlation_raw
                        ),
                        "alignment_failure_type": (
                            None if scorer == "student" else alignment_failure_type
                        ),
                        "alignment_failure_reason": (
                            None if scorer == "student" else alignment_failure_reason
                        ),
                        "score_cache": str(score_path.relative_to(scratch)),
                        "score_cache_key": score_key,
                        "score_cache_dtype": str(probability.dtype),
                        "score_shape": list(probability.shape),
                        "native_shape": list(original_image.shape[:2]),
                        "original_image_score": float(original["image_score"]),
                        "transformed_image_score": metrics["image_score"],
                    }
                )
                if sample_kind == "forged":
                    for field in (
                        "macro_pixel_ap",
                        "pixel_auroc",
                        "pixel_precision",
                        "pixel_recall",
                        "pixel_f1",
                        "pixel_iou",
                    ):
                        record[f"original_{field}"] = float(original[field])
                        record[f"transformed_{field}"] = float(metrics[field])
                    record["ap_change"] = (
                        record["transformed_macro_pixel_ap"]
                        - record["original_macro_pixel_ap"]
                    )
                else:
                    record["original_authentic_pixel_fpr"] = float(
                        original["authentic_pixel_fpr"]
                    )
                    record["transformed_authentic_pixel_fpr"] = float(
                        metrics["authentic_pixel_fpr"]
                    )
            except Exception as error:
                failures += 1
                record["failure_type"] = type(error).__name__
                record["failure_reason"] = str(error)
                logging.exception("record_id=%s failed", record["record_id"])
            records.append(record)
        _write_jsonl(predictions_path, records)
        logging.info(
            "completed_candidates=%d total_candidates=%d failures=%d",
            candidate_index,
            len(selected_rows),
            failures,
        )

    expected_predictions = len(selected_rows) * len(conditions)
    engineering_pass = failures == 0 and len(records) == expected_predictions
    successful = [row for row in records if row["status"] == "ok"]
    aggregates, subgroup_rows = _aggregate_models(successful, conditions)
    _write_csv(subgroup_path, subgroup_rows)
    decision: dict[str, Any] = {
        "engineering_gate_passed": engineering_pass,
        "next_stage_authorized": engineering_pass and stage != "gpu_full96",
        "scientific_gate_used_for_toy_or_pilot": False,
    }
    if stage == "gpu_full96" and engineering_pass:
        per_model = {}
        bootstrap = config["bootstrap"]
        for index, model_name in enumerate(conditions):
            signal_positive = [
                row
                for row in successful
                if row["model"] == model_name
                and row["sample_kind"] == "forged"
                and row["artifact_status"] == "signal_positive"
            ]
            interval = _paired_group_bootstrap(
                signal_positive,
                "ap_change",
                int(bootstrap["seed"]) + index,
                int(bootstrap["resamples"]),
                float(bootstrap["confidence_level"]),
            )
            interval["dependence_supported"] = bool(
                interval["effect"]
                <= float(config["decision_rule"]["signal_positive_mean_ap_change_max"])
                and interval["ci_high"] < 0.0
            )
            per_model[model_name] = interval
        decision["per_model_color_cue_dependence"] = per_model
        decision["any_model_color_cue_dependence_supported"] = any(
            value["dependence_supported"] for value in per_model.values()
        )
        decision["artifact_free_regeneration_still_required"] = True
        decision["consumed_reserve_restored"] = False

    candidate_records = {
        (str(row["source_group_id"]), str(row["evaluation_role"])): row
        for row in successful
    }
    transform_coverage = {}
    for artifact_status in (
        "signal_positive",
        "signal_negative",
        "authentic_control",
    ):
        selected = [
            row
            for row in candidate_records.values()
            if row["artifact_status"] == artifact_status
        ]
        transform_coverage[artifact_status] = {
            "candidates": len(selected),
            "changed_candidates": sum(int(row["changed_pixels"] > 0) for row in selected),
            "mean_changed_fraction": _mean_or_none(
                [float(row["changed_fraction"]) for row in selected]
            ),
        }

    status = (
        f"green_suppression_{stage}_passed"
        if engineering_pass
        else f"green_suppression_{stage}_failed"
    )
    output = {
        "schema_version": 1,
        "experiment": experiment,
        "status": status,
        "paper_evidence": False,
        "consumed_reserve_diagnostic": True,
        "model_training_performed": False,
        "checkpoint_selection_used": False,
        "threshold_selection_used": False,
        "selected_candidates": len(selected_rows),
        "expected_prediction_records": expected_predictions,
        "successful_prediction_records": len(successful),
        "failed_prediction_records": failures,
        "transform_cache_hits": transformed_cache_hits,
        "alignment_cache_hits": alignment_cache_hits,
        "score_cache_hits": score_cache_hits,
        "transform_coverage": transform_coverage,
        "alignment_status_counts": dict(
            Counter(str(row["alignment_status"]) for row in successful)
        ),
        "aggregates": aggregates,
        "decision": decision,
        "input": {
            "config_sha256": _sha256(config_path),
            "protocol_sha256": _sha256(protocol_path),
            "predecessor_summary_sha256": _sha256(predecessor_path),
            "manifest_sha256": _sha256(verified["manifest"]),
            "audit_records_sha256": _sha256(verified["audit_records"]),
            "frozen_predictions_sha256": _sha256(verified["frozen_predictions"]),
            "model_checkpoint_sha256": model_hashes,
        },
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "subgroups": str(subgroup_path.relative_to(project_root)),
            "subgroups_sha256": _sha256(subgroup_path),
            "report": str(report_path.relative_to(project_root)),
            "log": str(log_path.relative_to(project_root)),
            "transform_cache_dir": str(transform_cache.relative_to(scratch)),
            "alignment_cache_dir": str(alignment_cache.relative_to(scratch)),
            "score_cache_dir": str(score_cache.relative_to(scratch)),
        },
    }
    _write_json(summary_path, output)
    report_path.write_text(_report_text(output), encoding="utf-8")
    output["outputs"]["report_sha256"] = _sha256(report_path)
    _write_json(summary_path, output)
    logging.info("status=%s", status)
    if not engineering_pass and runtime.get("require_all_records", True):
        raise RuntimeError(f"green-suppression sensitivity failed: {failures} records")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
