from __future__ import annotations

import argparse
import hashlib
import json
import logging
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
    _resize_image,
)
from pairtrace_doc.pipelines.train_pairtrace_100 import _load_teacher
from pairtrace_doc.pipelines.train_student_100 import (
    _ranking_metrics,
    _read_jsonl,
    _resolve,
    _sha256,
    _threshold_vectors,
    _write_csv,
    _write_json,
    _write_jsonl,
)


GEOMETRIES = ("clean", "translation", "affine", "perspective")
DEFAULT_MODEL_NAMES = ("baseline", "robust_100")


def _validate_multiseed_evaluation_authorization(config: dict[str, Any]) -> bool:
    authorized = bool(config["runtime"].get("multi_seed_authorized", False))
    has_gate = "multiseed_gate" in config
    if not authorized:
        if has_gate or config.get("multi_seed") is not None:
            raise ValueError("single-seed evaluation cannot carry a multi-seed block")
        return False
    if config["experiment"].get("stage") != "multiseed_viewed_20_stability_evaluation":
        raise ValueError("multi-seed evaluation stage is not frozen")
    policy = config.get("multi_seed")
    if not isinstance(policy, dict) or not has_gate:
        raise ValueError("multi-seed evaluation policy or gate is missing")
    if [int(value) for value in policy["family_seeds"]] != [
        20260747,
        20260763,
        20260764,
    ]:
        raise ValueError("multi-seed evaluation family changed")
    if int(policy["training_seed"]) not in (20260763, 20260764):
        raise ValueError("multi-seed evaluation training seed is not a new seed")
    return True


def _robustness_decision(
    metrics: dict[str, dict[str, Any]], gate: dict[str, Any]
) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for model_name in DEFAULT_MODEL_NAMES:
        stress_values = [
            float(
                metrics[f"{model_name}__{geometry}"]["generator_macro_pixel_ap"]
            )
            for geometry in GEOMETRIES
            if geometry != "clean"
        ]
        summaries[model_name] = {
            "clean_generator_macro_pixel_ap": float(
                metrics[f"{model_name}__clean"]["generator_macro_pixel_ap"]
            ),
            "minimum_stressed_generator_macro_pixel_ap": min(stress_values),
            "mean_stressed_generator_macro_pixel_ap": float(np.mean(stress_values)),
            "maximum_authentic_pixel_fpr": max(
                float(metrics[f"{model_name}__{geometry}"]["authentic_pixel_fpr"])
                for geometry in GEOMETRIES
            ),
        }
    improvement = float(
        summaries["robust_100"]["minimum_stressed_generator_macro_pixel_ap"]
        - summaries["baseline"]["minimum_stressed_generator_macro_pixel_ap"]
    )
    checks = {
        "robust_clean_ap_floor": summaries["robust_100"][
            "clean_generator_macro_pixel_ap"
        ]
        >= float(gate["robust_clean_generator_macro_ap_min"]),
        "robust_minimum_stressed_ap_floor": summaries["robust_100"][
            "minimum_stressed_generator_macro_pixel_ap"
        ]
        >= float(gate["robust_minimum_stressed_generator_macro_ap_min"]),
        "robust_minimum_stressed_gain_floor": improvement
        >= float(gate["robust_minimum_stressed_gain_over_baseline_min"]),
        "robust_authentic_fpr_ceiling": summaries["robust_100"][
            "maximum_authentic_pixel_fpr"
        ]
        <= float(gate["authentic_pixel_fpr_max"]),
    }
    effects = {
        f"robust_minus_baseline__{geometry}": float(
            metrics[f"robust_100__{geometry}"]["generator_macro_pixel_ap"]
            - metrics[f"baseline__{geometry}"]["generator_macro_pixel_ap"]
        )
        for geometry in GEOMETRIES
    }
    return {
        "model_summaries": summaries,
        "minimum_stressed_gain_over_baseline": improvement,
        "effects": effects,
        "checks": checks,
        "overall_pass": all(checks.values()),
    }


def _scale_decision(
    metrics: dict[str, dict[str, Any]], gate: dict[str, Any]
) -> dict[str, Any]:
    model_names = ("baseline", "robust_100", "robust_1000")
    summaries: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        stressed = [
            float(metrics[f"{model_name}__{geometry}"]["generator_macro_pixel_ap"])
            for geometry in GEOMETRIES
            if geometry != "clean"
        ]
        summaries[model_name] = {
            "clean_generator_macro_pixel_ap": float(
                metrics[f"{model_name}__clean"]["generator_macro_pixel_ap"]
            ),
            "minimum_stressed_generator_macro_pixel_ap": min(stressed),
            "mean_stressed_generator_macro_pixel_ap": float(np.mean(stressed)),
            "maximum_authentic_pixel_fpr": max(
                float(metrics[f"{model_name}__{geometry}"]["authentic_pixel_fpr"])
                for geometry in GEOMETRIES
            ),
        }
    robust = summaries["robust_1000"]
    baseline_gain = float(
        robust["minimum_stressed_generator_macro_pixel_ap"]
        - summaries["baseline"]["minimum_stressed_generator_macro_pixel_ap"]
    )
    difference_100 = float(
        robust["minimum_stressed_generator_macro_pixel_ap"]
        - summaries["robust_100"]["minimum_stressed_generator_macro_pixel_ap"]
    )
    checks = {
        "robust_1000_clean_ap_floor": robust["clean_generator_macro_pixel_ap"]
        >= float(gate["robust_1000_clean_generator_macro_ap_min"]),
        "robust_1000_minimum_stressed_ap_floor": robust[
            "minimum_stressed_generator_macro_pixel_ap"
        ]
        >= float(gate["robust_1000_minimum_stressed_generator_macro_ap_min"]),
        "robust_1000_noninferior_to_100": difference_100
        >= -float(gate["robust_1000_noninferiority_margin_to_100"]),
        "robust_1000_baseline_gain_floor": baseline_gain
        >= float(gate["robust_1000_minimum_stressed_gain_over_baseline_min"]),
        "robust_1000_authentic_fpr_ceiling": robust["maximum_authentic_pixel_fpr"]
        <= float(gate["authentic_pixel_fpr_max"]),
    }
    effects = {
        f"robust_1000_minus_baseline__{geometry}": float(
            metrics[f"robust_1000__{geometry}"]["generator_macro_pixel_ap"]
            - metrics[f"baseline__{geometry}"]["generator_macro_pixel_ap"]
        )
        for geometry in GEOMETRIES
    }
    effects.update(
        {
            f"robust_1000_minus_robust_100__{geometry}": float(
                metrics[f"robust_1000__{geometry}"]["generator_macro_pixel_ap"]
                - metrics[f"robust_100__{geometry}"]["generator_macro_pixel_ap"]
            )
            for geometry in GEOMETRIES
        }
    )
    return {
        "model_summaries": summaries,
        "minimum_stressed_gain_over_baseline": baseline_gain,
        "minimum_stressed_difference_from_robust_100": difference_100,
        "effects": effects,
        "checks": checks,
        "overall_pass": all(checks.values()),
    }


def _multiseed_decision(
    metrics: dict[str, dict[str, Any]], gate: dict[str, Any]
) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for model_name in ("baseline", "robust_1000"):
        stressed = [
            float(metrics[f"{model_name}__{geometry}"]["generator_macro_pixel_ap"])
            for geometry in GEOMETRIES
            if geometry != "clean"
        ]
        summaries[model_name] = {
            "clean_generator_macro_pixel_ap": float(
                metrics[f"{model_name}__clean"]["generator_macro_pixel_ap"]
            ),
            "minimum_stressed_generator_macro_pixel_ap": min(stressed),
            "mean_stressed_generator_macro_pixel_ap": float(np.mean(stressed)),
            "maximum_authentic_pixel_fpr": max(
                float(metrics[f"{model_name}__{geometry}"]["authentic_pixel_fpr"])
                for geometry in GEOMETRIES
            ),
        }
    robust = summaries["robust_1000"]
    gain = float(
        robust["minimum_stressed_generator_macro_pixel_ap"]
        - summaries["baseline"]["minimum_stressed_generator_macro_pixel_ap"]
    )
    checks = {
        "robust_clean_ap_floor": robust["clean_generator_macro_pixel_ap"]
        >= float(gate["robust_clean_generator_macro_ap_min"]),
        "robust_minimum_stressed_ap_floor": robust[
            "minimum_stressed_generator_macro_pixel_ap"
        ]
        >= float(gate["robust_minimum_stressed_generator_macro_ap_min"]),
        "robust_minimum_stressed_gain_floor": gain
        >= float(gate["robust_minimum_stressed_gain_over_baseline_min"]),
        "robust_authentic_fpr_ceiling": robust["maximum_authentic_pixel_fpr"]
        <= float(gate["authentic_pixel_fpr_max"]),
    }
    return {
        "model_summaries": summaries,
        "minimum_stressed_gain_over_baseline": gain,
        "effects": {
            f"robust_1000_minus_baseline__{geometry}": float(
                metrics[f"robust_1000__{geometry}"]["generator_macro_pixel_ap"]
                - metrics[f"baseline__{geometry}"]["generator_macro_pixel_ap"]
            )
            for geometry in GEOMETRIES
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
    if not runtime["gpu_launch_authorized"] or not runtime["robust_teacher_evaluation_authorized"]:
        raise ValueError("robust teacher evaluation was not explicitly authorized")
    if not runtime["viewed_development_read_allowed"]:
        raise ValueError("viewed development read was not authorized")
    multi_seed_run = _validate_multiseed_evaluation_authorization(config)
    if any(
        bool(runtime.get(name))
        for name in (
            "model_training_authorized",
            "unseen_development_read_allowed",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("robust teacher evaluation crossed an evidence boundary")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("robust teacher pilot evaluation cannot be paper evidence")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("robust teacher evaluation requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("robust teacher protocol SHA-256 changed")
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
    alignment_path = _resolve(project_root, inputs["alignment_records"])
    for path, expected, label in (
        (manifest_path, inputs["expected_manifest_sha256"], "manifest"),
        (alignment_path, inputs["expected_alignment_records_sha256"], "alignment records"),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen {label} SHA-256 changed")
    rows = sorted(_read_jsonl(manifest_path), key=lambda row: str(row["source_group_id"]))
    if len(rows) != int(inputs["expected_groups"]):
        raise ValueError("robust teacher evaluation group count changed")
    counts = Counter(str(row["selected_generator"]) for row in rows)
    expected_counts = {
        str(name): int(value)
        for name, value in inputs["expected_generator_counts"].items()
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"robust teacher generator counts changed: {dict(counts)}")
    max_groups = runtime.get("max_groups")
    if max_groups is not None:
        rows = rows[: int(max_groups)]

    alignment_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in _read_jsonl(alignment_path):
        geometry = str(item["stress"])
        if geometry not in GEOMETRIES:
            continue
        key = (str(item["source_group_id"]), str(item["sample_kind"]), geometry)
        if key in alignment_records or item["alignment_status"] != "ecc_converged":
            raise ValueError("alignment inputs are duplicate or unconverged")
        alignment_records[key] = item
    if len(alignment_records) != int(inputs["expected_groups"]) * 2 * len(GEOMETRIES):
        raise ValueError("alignment inputs are incomplete")

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
    model_names = tuple(
        str(value)
        for value in config.get("evaluation_models", list(DEFAULT_MODEL_NAMES))
    )
    if set(model_names) not in (
        set(DEFAULT_MODEL_NAMES),
        {"baseline", "robust_100", "robust_1000"},
    ):
        raise ValueError("robust teacher evaluation model whitelist changed")
    models: dict[str, torch.nn.Module] = {}
    model_hashes: dict[str, str] = {}
    model_config = config["models"]
    encoder_path = _resolve(scratch, model_config["encoder_weights"])
    if _sha256(encoder_path) != model_config["encoder_weights_sha256"]:
        raise ValueError("frozen encoder weights changed")
    for model_name in model_names:
        checkpoint_path = _resolve(project_root, model_config[model_name]["checkpoint"])
        expected = str(model_config[model_name]["checkpoint_sha256"])
        if _sha256(checkpoint_path) != expected:
            raise ValueError(f"frozen {model_name} checkpoint changed")
        model = _load_teacher(
            encoder_path, model_config["teacher_conv1_coefficients"]
        )
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state"], strict=True)
        models[model_name] = model.to(device).eval().requires_grad_(False)
        model_hashes[model_name] = expected

    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
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
    if int(preprocessing["score_cache_schema_version"]) != 2 or preprocessing["score_cache_dtype"] != "float32":
        raise ValueError("robust teacher evaluation requires float32 score cache schema v2")
    inference = config["inference"]
    thresholds = np.arange(
        float(config["operating_point"]["candidate_min"]),
        float(config["operating_point"]["candidate_max"]) + 1e-12,
        float(config["operating_point"]["candidate_step"]),
    )
    condition_names = {
        f"{model_name}__{geometry}"
        for model_name in model_names
        for geometry in GEOMETRIES
    }
    payloads = {
        name: {
            "forged": [],
            "authentic_vectors": [],
            "authentic_fpr_max": config["operating_point"]["authentic_pixel_fpr_max"],
        }
        for name in condition_names
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
    failures = 0
    cache_hits = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for row_index, row in enumerate(rows, start=1):
        group = str(row["source_group_id"])
        generator = str(row["selected_generator"])
        forged_native = load_image(row, "image", "image_sha256")
        authentic_native = load_image(row, "authentic", "authentic_sha256")
        mask_path = _resolve(scratch, row["mask"])
        if _sha256(mask_path) != row["mask_sha256"]:
            raise ValueError("robust teacher mask SHA-256 changed")
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0
        for sample_kind, candidate_native in (("forged", forged_native), ("authentic", authentic_native)):
            candidate = _resize_image(candidate_native, int(preprocessing["max_side"]))
            candidate_sha256 = str(
                row["image_sha256"] if sample_kind == "forged" else row["authentic_sha256"]
            )
            for geometry in GEOMETRIES:
                alignment = alignment_records[(group, sample_kind, geometry)]
                cached_alignment_path = _resolve(scratch, alignment["alignment_cache"])
                with np.load(cached_alignment_path, allow_pickle=False) as archive:
                    reference = archive["aligned_reference"].astype(np.uint8)
                if reference.shape != candidate.shape:
                    raise ValueError("robust teacher aligned reference geometry changed")
                for model_name, model in models.items():
                    condition_name = f"{model_name}__{geometry}"
                    prediction: dict[str, Any] = {
                        "record_id": f"{condition_name}:{sample_kind}:{group}",
                        "source_group_id": group,
                        "generator": generator,
                        "sample_kind": sample_kind,
                        "condition": condition_name,
                        "model": model_name,
                        "geometry": geometry,
                        "status": "failed",
                        "paper_evidence": False,
                        "viewed_development": True,
                        "unseen_development_read": False,
                        "final_reserve_read": False,
                    }
                    try:
                        cache_key = hashlib.sha256(
                            json.dumps(
                                {
                                    "candidate_sha256": candidate_sha256,
                                    "alignment_key": alignment["alignment_key"],
                                    "checkpoint_sha256": model_hashes[model_name],
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
                        if not score_path.is_file():
                            probability = _infer_pair_tiled(
                                model,
                                candidate,
                                reference,
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
                            cache_hits += 1
                        with np.load(score_path, allow_pickle=False) as archive:
                            probability = archive["scores"]
                        if probability.dtype != np.float32:
                            raise ValueError("robust teacher score cache dtype is not float32")
                        if probability.shape != candidate.shape[:2] or not np.isfinite(probability).all():
                            raise ValueError("robust teacher score cache is invalid")
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
                                "score_cache_schema_version": 2,
                                "score_cache_dtype": str(probability.dtype),
                                "alignment_key": alignment["alignment_key"],
                                "score_shape": list(probability.shape),
                                "native_shape": list(native_probability.shape),
                                "checkpoint_sha256": model_hashes[model_name],
                            }
                        )
                    except Exception as error:
                        failures += 1
                        prediction["failure_type"] = type(error).__name__
                        prediction["failure_reason"] = str(error)
                        logging.exception("record_id=%s failed", prediction["record_id"])
                    predictions.append(prediction)
        _write_jsonl(predictions_path, predictions)
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
            "failed_prediction_records": failures,
            "successful_prediction_records": len(predictions) - failures,
            "unseen_development_read": False,
            "final_reserve_read": False,
        }
        _write_json(summary_path, output)
        if runtime["require_all_records"]:
            raise RuntimeError(f"robust teacher evaluation failed for {failures} records")
        return output

    metrics = {
        name: _aggregate_condition(payload, thresholds)
        for name, payload in payloads.items()
    }
    multiseed_evaluation = "multiseed_gate" in config
    scale_evaluation = "scale_gate" in config
    if multiseed_evaluation:
        decision = _multiseed_decision(metrics, config["multiseed_gate"])
    elif scale_evaluation:
        decision = _scale_decision(metrics, config["scale_gate"])
    else:
        decision = _robustness_decision(metrics, config["pilot_gate"])
    _write_csv(metrics_path, [{"condition": name, **value} for name, value in metrics.items()])
    _write_csv(
        comparisons_path,
        [
            {
                "comparison": name,
                "generator_macro_pixel_ap_difference": value,
                "paper_evidence": False,
            }
            for name, value in decision["effects"].items()
        ],
    )
    output = {
        "experiment": config["experiment"],
        "status": (
            (
                "resampling_robust_multiseed_viewed_gate_passed"
                if decision["overall_pass"]
                else "resampling_robust_multiseed_viewed_gate_failed"
            )
            if multiseed_evaluation
            else (
                "resampling_robust_teacher_1000_scale_gate_passed"
                if decision["overall_pass"]
                else "resampling_robust_teacher_1000_scale_gate_failed"
            )
            if scale_evaluation
            else (
                "resampling_robust_teacher_100_gate_passed"
                if decision["overall_pass"]
                else "resampling_robust_teacher_100_gate_failed"
            )
        ),
        "paper_evidence": False,
        "viewed_development_read": True,
        "unseen_development_read": False,
        "final_reserve_read": False,
        "multi_seed_authorized": multi_seed_run,
        "multi_seed_stability_evaluation": multi_seed_run,
        "resampling_robust_teacher_1000_training_authorized": (
            False
            if scale_evaluation or multiseed_evaluation
            else bool(decision["overall_pass"])
        ),
        "second_unseen_development_freeze_authorized": (
            bool(decision["overall_pass"]) if scale_evaluation else False
        ),
        "candidate_method_checkpoint": (
            "robust_1000" if scale_evaluation and decision["overall_pass"] else None
        ),
        "selected_groups": len(rows),
        "successful_prediction_records": len(predictions),
        "failed_prediction_records": 0,
        "score_cache_hits": cache_hits,
        "score_cache_schema_version": 2,
        "score_cache_dtype": "float32",
        "conditions": metrics,
        "pilot_gate": config.get("pilot_gate"),
        "scale_gate": config.get("scale_gate"),
        "multiseed_gate": config.get("multiseed_gate"),
        "decision": decision,
        "protocol_sha256": _sha256(protocol_path),
        "input_manifest_sha256": _sha256(manifest_path),
        "alignment_records_sha256": _sha256(alignment_path),
        "model_checkpoint_sha256": model_hashes,
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
