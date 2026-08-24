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
    _jpeg_roundtrip,
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


def _apply_matched_normalization(
    candidate: np.ndarray,
    reference: np.ndarray,
    specification: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if candidate.shape != reference.shape:
        raise ValueError("matched normalization requires equal geometry")
    method = str(specification["method"])
    if method == "none":
        return candidate, reference
    if method == "matched_jpeg":
        return (
            _jpeg_roundtrip(candidate, specification),
            _jpeg_roundtrip(reference, specification),
        )
    if method == "matched_gaussian":
        kernel = int(specification["kernel_size"])
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("Gaussian kernel must be positive and odd")
        sigma = float(specification["sigma"])
        return (
            cv2.GaussianBlur(candidate, (kernel, kernel), sigma),
            cv2.GaussianBlur(reference, (kernel, kernel), sigma),
        )
    if method == "matched_resize":
        scale = float(specification["scale"])
        if not 0.0 < scale < 1.0:
            raise ValueError("matched resize scale must lie between zero and one")
        height, width = candidate.shape[:2]
        target = (max(1, round(width * scale)), max(1, round(height * scale)))

        def roundtrip(image: np.ndarray) -> np.ndarray:
            smaller = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
            return cv2.resize(smaller, (width, height), interpolation=cv2.INTER_LINEAR)

        return roundtrip(candidate), roundtrip(reference)
    raise ValueError(f"unsupported matched normalization: {method}")


def _select_normalization(
    metrics: dict[str, dict[str, Any]],
    variants: list[str],
    gate: dict[str, Any],
) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for variant in variants:
        clean = metrics[f"{variant}__clean"]
        stressed = [
            float(metrics[f"{variant}__{geometry}"]["generator_macro_pixel_ap"])
            for geometry in GEOMETRIES
            if geometry != "clean"
        ]
        fprs = [
            float(metrics[f"{variant}__{geometry}"]["authentic_pixel_fpr"])
            for geometry in GEOMETRIES
        ]
        summaries[variant] = {
            "clean_generator_macro_pixel_ap": float(
                clean["generator_macro_pixel_ap"]
            ),
            "minimum_stressed_generator_macro_pixel_ap": min(stressed),
            "mean_stressed_generator_macro_pixel_ap": float(np.mean(stressed)),
            "maximum_authentic_pixel_fpr": max(fprs),
        }
        summaries[variant]["selectable"] = (
            summaries[variant]["clean_generator_macro_pixel_ap"]
            >= float(gate["clean_generator_macro_ap_min"])
            and summaries[variant]["maximum_authentic_pixel_fpr"]
            <= float(gate["authentic_pixel_fpr_max"])
        )
    selectable = [name for name in variants if summaries[name]["selectable"]]
    selected = (
        sorted(
            selectable,
            key=lambda name: (
                -summaries[name]["minimum_stressed_generator_macro_pixel_ap"],
                -summaries[name]["mean_stressed_generator_macro_pixel_ap"],
                -summaries[name]["clean_generator_macro_pixel_ap"],
                name,
            ),
        )[0]
        if selectable
        else None
    )
    checks = {"selectable_variant_exists": selected is not None}
    improvement = None
    if selected is not None:
        improvement = float(
            summaries[selected]["minimum_stressed_generator_macro_pixel_ap"]
            - summaries["none"]["minimum_stressed_generator_macro_pixel_ap"]
        )
        checks.update(
            {
                "selected_minimum_stressed_ap_floor": summaries[selected][
                    "minimum_stressed_generator_macro_pixel_ap"
                ]
                >= float(gate["minimum_stressed_generator_macro_ap_min"]),
                "selected_minimum_stressed_gain_floor": improvement
                >= float(gate["minimum_stressed_gain_over_none_min"]),
            }
        )
    else:
        checks.update(
            {
                "selected_minimum_stressed_ap_floor": False,
                "selected_minimum_stressed_gain_floor": False,
            }
        )
    return {
        "variant_summaries": summaries,
        "selected_variant": selected,
        "selected_minimum_stressed_gain_over_none": improvement,
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
    if not runtime["gpu_launch_authorized"] or not runtime["normalization_diagnostic_authorized"]:
        raise ValueError("normalization diagnostic was not explicitly authorized")
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
        raise ValueError("normalization diagnostic crossed an evidence boundary")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("normalization diagnostic cannot be paper evidence")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("normalization diagnostic requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("normalization protocol SHA-256 changed")
    input_config = config["input"]
    verified_paths: dict[str, Path] = {}
    for name in ("manifest", "parent_predictions", "parent_alignments", "parent_summary"):
        path = _resolve(project_root, input_config[name])
        if _sha256(path) != input_config[f"expected_{name}_sha256"]:
            raise ValueError(f"frozen {name} SHA-256 changed")
        verified_paths[name] = path
    rows = sorted(
        _read_jsonl(verified_paths["manifest"]),
        key=lambda row: str(row["source_group_id"]),
    )
    if len(rows) != int(input_config["expected_groups"]):
        raise ValueError("normalization diagnostic group count changed")
    counts = Counter(str(row["selected_generator"]) for row in rows)
    expected_counts = {
        str(name): int(value)
        for name, value in input_config["expected_generator_counts"].items()
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"normalization diagnostic generator counts changed: {dict(counts)}")
    max_groups = runtime.get("max_groups")
    if max_groups is not None:
        rows = rows[: int(max_groups)]

    variant_specs = {str(item["name"]): item for item in config["variants"]}
    expected_variants = {
        "none",
        "matched_jpeg_q85",
        "matched_gaussian_sigma_0_75",
        "matched_resize_scale_0_75",
        "matched_resize_scale_0_50",
    }
    if set(variant_specs) != expected_variants:
        raise ValueError("normalization variant whitelist changed")
    condition_names = {
        f"{variant}__{geometry}"
        for variant in variant_specs
        for geometry in GEOMETRIES
    }

    parent_predictions = _read_jsonl(verified_paths["parent_predictions"])
    parent_scores: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in parent_predictions:
        condition = str(item["condition"])
        if not condition.startswith("pair_teacher_") or not condition.endswith("_ecc"):
            continue
        geometry = condition.removeprefix("pair_teacher_").removesuffix("_ecc")
        if geometry in GEOMETRIES:
            key = (str(item["source_group_id"]), str(item["sample_kind"]), geometry)
            if key in parent_scores or item["status"] != "ok":
                raise ValueError("parent ECC score records are duplicate or failed")
            parent_scores[key] = item
    parent_alignments = _read_jsonl(verified_paths["parent_alignments"])
    alignment_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in parent_alignments:
        geometry = str(item["stress"])
        if geometry not in GEOMETRIES:
            continue
        key = (str(item["source_group_id"]), str(item["sample_kind"]), geometry)
        if key in alignment_records or item["alignment_status"] != "ecc_converged":
            raise ValueError("parent alignment records are duplicate or unconverged")
        alignment_records[key] = item
    expected_parent_records = int(input_config["expected_groups"]) * 2 * len(GEOMETRIES)
    if len(parent_scores) != expected_parent_records or len(alignment_records) != expected_parent_records:
        raise ValueError("parent normalization inputs are incomplete")

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
    models = config["models"]
    teacher_path = _resolve(project_root, models["teacher_checkpoint"])
    encoder_path = _resolve(scratch, models["encoder_weights"])
    for path, expected, label in (
        (teacher_path, models["teacher_checkpoint_sha256"], "teacher checkpoint"),
        (encoder_path, models["encoder_weights_sha256"], "encoder weights"),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen {label} changed")
    teacher = _load_teacher(encoder_path, models["teacher_conv1_coefficients"])
    teacher_saved = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher.load_state_dict(teacher_saved["model_state"], strict=True)
    teacher = teacher.to(device).eval().requires_grad_(False)

    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    predictions_path = _resolve(project_root, paths["predictions"])
    metrics_path = _resolve(project_root, paths["metrics"])
    variants_path = _resolve(project_root, paths["variant_summary"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (
        score_cache_dir,
        predictions_path.parent,
        metrics_path.parent,
        variants_path.parent,
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
    thresholds = np.arange(
        float(config["operating_point"]["candidate_min"]),
        float(config["operating_point"]["candidate_max"]) + 1e-12,
        float(config["operating_point"]["candidate_step"]),
    )
    payloads: dict[str, dict[str, Any]] = {
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
    parent_score_reuses = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for row_index, row in enumerate(rows, start=1):
        group = str(row["source_group_id"])
        generator = str(row["selected_generator"])
        forged_native = load_image(row, "image", "image_sha256")
        authentic_native = load_image(row, "authentic", "authentic_sha256")
        mask_path = _resolve(scratch, row["mask"])
        if _sha256(mask_path) != row["mask_sha256"]:
            raise ValueError("normalization diagnostic mask SHA-256 changed")
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0
        for sample_kind, candidate_native in (("forged", forged_native), ("authentic", authentic_native)):
            candidate = _resize_image(candidate_native, int(preprocessing["max_side"]))
            candidate_sha256 = str(
                row["image_sha256"] if sample_kind == "forged" else row["authentic_sha256"]
            )
            for geometry in GEOMETRIES:
                parent = parent_scores[(group, sample_kind, geometry)]
                alignment = alignment_records[(group, sample_kind, geometry)]
                alignment_path = _resolve(scratch, alignment["alignment_cache"])
                with np.load(alignment_path, allow_pickle=False) as archive:
                    aligned_reference = archive["aligned_reference"].astype(np.uint8)
                if aligned_reference.shape != candidate.shape:
                    raise ValueError("parent aligned reference geometry changed")
                for variant_name, specification in variant_specs.items():
                    condition_name = f"{variant_name}__{geometry}"
                    prediction: dict[str, Any] = {
                        "record_id": f"{condition_name}:{sample_kind}:{group}",
                        "source_group_id": group,
                        "generator": generator,
                        "sample_kind": sample_kind,
                        "condition": condition_name,
                        "variant": variant_name,
                        "geometry": geometry,
                        "status": "failed",
                        "paper_evidence": False,
                        "viewed_method_development": True,
                        "unseen_development_read": False,
                        "final_reserve_read": False,
                    }
                    try:
                        if variant_name == "none":
                            score_path = _resolve(scratch, parent["score_cache"])
                            score_source = "frozen_parent_score"
                            parent_score_reuses += 1
                        else:
                            normalized_candidate, normalized_reference = _apply_matched_normalization(
                                candidate, aligned_reference, specification
                            )
                            cache_key = hashlib.sha256(
                                json.dumps(
                                    {
                                        "candidate_sha256": candidate_sha256,
                                        "alignment_key": alignment["alignment_key"],
                                        "variant": specification,
                                        "model_identity": models["teacher_checkpoint_sha256"],
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
                            score_source = "normalization_diagnostic_cache"
                            if not score_path.is_file():
                                probability = _infer_pair_tiled(
                                    teacher,
                                    normalized_candidate,
                                    normalized_reference,
                                    device,
                                    inference,
                                    preprocessing,
                                )
                                temporary = score_path.with_suffix(".npz.tmp")
                                with temporary.open("wb") as handle:
                                    np.savez_compressed(
                                        handle, scores=probability.astype(np.float16)
                                    )
                                temporary.replace(score_path)
                            else:
                                cache_hits += 1
                        with np.load(score_path, allow_pickle=False) as archive:
                            probability = archive["scores"].astype(np.float32)
                        if probability.shape != candidate.shape[:2] or not np.isfinite(probability).all():
                            raise ValueError("normalization score cache is invalid")
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
                                "score_source": score_source,
                                "alignment_key": alignment["alignment_key"],
                                "score_shape": list(probability.shape),
                                "native_shape": list(native_probability.shape),
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
            raise RuntimeError(f"normalization diagnostic failed for {failures} records")
        return output

    metrics = {
        name: _aggregate_condition(payload, thresholds)
        for name, payload in payloads.items()
    }
    decision = _select_normalization(
        metrics, list(variant_specs), config["normalization_gate"]
    )
    _write_csv(metrics_path, [{"condition": name, **value} for name, value in metrics.items()])
    _write_csv(
        variants_path,
        [
            {"variant": name, **value, "paper_evidence": False}
            for name, value in decision["variant_summaries"].items()
        ],
    )
    output = {
        "experiment": config["experiment"],
        "status": (
            "training_free_normalization_sufficient"
            if decision["overall_pass"]
            else "training_free_normalization_insufficient"
        ),
        "paper_evidence": False,
        "viewed_method_development_read": True,
        "unseen_development_read": False,
        "final_reserve_read": False,
        "method_training_performed": False,
        "selected_normalization": decision["selected_variant"],
        "normalization_with_ecc_freeze_authorized": bool(decision["overall_pass"]),
        "second_unseen_development_freeze_authorized": bool(decision["overall_pass"]),
        "resampling_augmented_teacher_100_pilot_authorized": not bool(
            decision["overall_pass"]
        ),
        "multi_seed_authorized": False,
        "selected_groups": len(rows),
        "successful_prediction_records": len(predictions),
        "failed_prediction_records": 0,
        "parent_score_reuses": parent_score_reuses,
        "normalization_score_cache_hits": cache_hits,
        "conditions": metrics,
        "normalization_gate": config["normalization_gate"],
        "decision": decision,
        "protocol_sha256": _sha256(protocol_path),
        "input_sha256": {name: _sha256(path) for name, path in verified_paths.items()},
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
            "variant_summary": str(variants_path.relative_to(project_root)),
            "variant_summary_sha256": _sha256(variants_path),
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
