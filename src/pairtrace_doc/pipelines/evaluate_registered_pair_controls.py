from __future__ import annotations

import argparse
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
import yaml
from PIL import Image

from pairtrace_doc.pipelines.compare_generator_balanced_1000 import (
    _stratified_paired_bootstrap,
)
from pairtrace_doc.pipelines.diagnose_pair_at_inference_alignment import (
    _estimate_ecc_alignment,
)
from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
    _aggregate_condition,
    _raw_difference,
    _resize_image,
    _resize_reference,
)
from pairtrace_doc.pipelines.run_spatial_lpips import (
    _cache_key,
    _select_round_robin,
    _valid_overlap,
)
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


METHODS = ("registered_normalized_rgb_difference", "registered_ssim_distance")
_RUNTIME_ONLY_PREDICTION_FIELDS = frozenset({"cache_hit", "latency_ms"})


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify(path: Path, expected: str, label: str) -> str:
    digest = _sha256(path)
    if digest != expected:
        raise ValueError(f"{label} SHA-256 changed: {digest} != {expected}")
    return digest


def _assert_stable_prediction_record(record: dict[str, Any]) -> None:
    transient = sorted(_RUNTIME_ONLY_PREDICTION_FIELDS.intersection(record))
    if transient:
        raise ValueError(
            "prediction record contains runtime-only fields: " + ", ".join(transient)
        )


def _ssim_distance(
    candidate: np.ndarray,
    reference: np.ndarray,
    specification: dict[str, Any],
) -> np.ndarray:
    if candidate.shape != reference.shape or candidate.ndim != 3:
        raise ValueError("SSIM requires matched HWC RGB arrays")
    if candidate.shape[2] != 3:
        raise ValueError("SSIM requires exactly three RGB channels")
    window = int(specification["window_size"])
    sigma = float(specification["sigma"])
    if window < 3 or window % 2 != 1 or sigma <= 0:
        raise ValueError("invalid SSIM Gaussian window")
    left = candidate.astype(np.float64) / 255.0
    right = reference.astype(np.float64) / 255.0
    blur = lambda value: cv2.GaussianBlur(
        value,
        (window, window),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    left_mean = blur(left)
    right_mean = blur(right)
    left_variance = np.maximum(blur(left * left) - left_mean * left_mean, 0.0)
    right_variance = np.maximum(
        blur(right * right) - right_mean * right_mean, 0.0
    )
    covariance = blur(left * right) - left_mean * right_mean
    c1 = float(specification["k1"]) ** 2
    c2 = float(specification["k2"]) ** 2
    numerator = (2.0 * left_mean * right_mean + c1) * (2.0 * covariance + c2)
    denominator = (
        left_mean * left_mean + right_mean * right_mean + c1
    ) * (left_variance + right_variance + c2)
    similarity = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0,
    )
    similarity = np.clip(similarity.mean(axis=2), -1.0, 1.0)
    return ((1.0 - similarity) / 2.0).astype(np.float32)


def _score(
    method: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    ssim: dict[str, Any],
) -> np.ndarray:
    if method == "registered_normalized_rgb_difference":
        return _raw_difference(candidate, reference).astype(np.float32)
    if method == "registered_ssim_distance":
        return _ssim_distance(candidate, reference, ssim)
    raise ValueError(f"unsupported registered pair control: {method}")


def _forged_score_map(
    rows: list[dict[str, Any]],
    *,
    condition: str | None = None,
    baseline: str | None = None,
) -> dict[str, tuple[str, float]]:
    selected = []
    for row in rows:
        if row.get("status") != "ok" or row.get("sample_kind") != "forged":
            continue
        if condition is not None and row.get("condition") != condition:
            continue
        if baseline is not None and row.get("baseline") != baseline:
            continue
        selected.append(row)
    result = {
        str(row["source_group_id"]): (
            str(row["generator"]),
            float(row["macro_pixel_ap"]),
        )
        for row in selected
    }
    if len(result) != len(selected):
        raise ValueError("dependency predictions contain duplicate source groups")
    return result


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    comparisons = summary["comparisons"]
    alignment = summary["posthoc_alignment_diagnostic"]
    metric_rows = []
    for method in METHODS:
        value = metrics[method]
        metric_rows.append(
            f"| {method} | {value['generator_macro_pixel_ap']:.6f} | "
            f"{value['macro_pixel_ap']:.6f} | {value['pixel_f1']:.6f} | "
            f"{value['pixel_iou']:.6f} | {value['authentic_pixel_fpr']:.6f} | "
            f"{value['authentic_image_fpr']:.6f} |"
        )
    comparison_rows = []
    for name, value in comparisons.items():
        comparison_rows.append(
            f"| {name} | {value['effect']:+.6f} | "
            f"[{value['ci_low']:.6f}, {value['ci_high']:.6f}] |"
        )
    return f"""# Registered equal-information pair controls: development-100

Status: `{summary['status']}`. These are viewed AIForge development results,
not independent final evidence.

## Results

| Method | Generator-macro AP | Document-macro AP | Pixel F1 | Pixel IoU | Authentic pixel FPR | Authentic image FPR |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

Both methods use the same candidate/authentic-reference pair, frozen ECC
registration, 1,024-pixel cap, native-geometry mask, and valid-overlap support.
The normalized-difference formula is `max(abs(RGB difference))/255`; SSIM uses
the frozen 11x11, sigma-1.5 RGB Gaussian formulation and reports `(1-SSIM)/2`.

## Paired source-group effects

| Comparison | AP effect | 95% interval |
|---|---:|---:|
{chr(10).join(comparison_rows)}

## Post-hoc registration diagnostic

The parent experiment's direct resized-pair normalized difference reached
generator-macro AP {alignment['direct_resized_pair_generator_macro_ap']:.6f},
whereas the otherwise matched ECC-registered difference reached
{metrics['registered_normalized_rgb_difference']['generator_macro_pixel_ap']:.6f}.
The registered-minus-direct paired effect was {alignment['effect']:+.6f}
with a 95% interval of [{alignment['ci_low']:.6f},
{alignment['ci_high']:.6f}]. This comparison was added after observing the
registered-control result and is explicitly exploratory; it diagnoses
registration damage on already aligned source pairs rather than serving as a
preregistered method claim.

All {summary['selected_development_groups']} source groups and
{summary['successful_records']} method/sample records were retained with
{summary['failed_records']} failures. No training or final-reserve read
occurred. Spatial LPIPS and the clean pair teacher are frozen dependencies on
the same viewed development groups. This closes deterministic/perceptual
coverage only; FC-Siam-diff still requires prospective external exact-mask
data.
"""


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("registered pair-control config must be a mapping")
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    if config["experiment"]["paper_evidence"]:
        raise ValueError("registered pair controls cannot be paper evidence")
    if runtime["model_training_authorized"] or runtime["final_reserve_read_allowed"]:
        raise ValueError("registered pair controls crossed an evidence boundary")
    stage = str(config["experiment"]["stage"])
    toy = stage == "cpu_toy3_structure_gate"
    development = stage == "cpu_development100"
    if not (toy or development):
        raise ValueError(f"unsupported registered pair-control stage: {stage}")
    if str(runtime["device"]) != "cpu" or not runtime["cpu_inference_authorized"]:
        raise PermissionError("registered pair controls require explicit CPU authorization")
    expected_groups = 3 if toy else 100
    if int(runtime["max_groups"]) != expected_groups:
        raise ValueError("registered pair-control group limit changed")
    if bool(runtime["threshold_selection_authorized"]) != development:
        raise ValueError("threshold authorization does not match the stage")

    experiment = config["experiment"]
    protocol_path = _resolve(project_root, str(experiment["protocol"]))
    _verify(
        protocol_path,
        str(experiment["expected_protocol_sha256"]),
        "registered pair-control protocol",
    )
    input_config = config["input"]
    manifest_path = _resolve(project_root, str(input_config["manifest"]))
    _verify(
        manifest_path,
        str(input_config["expected_manifest_sha256"]),
        "registered pair-control manifest",
    )
    all_rows = sorted(
        _read_jsonl(manifest_path), key=lambda row: str(row["source_group_id"])
    )
    if len(all_rows) != 100 or len(
        {str(row["source_group_id"]) for row in all_rows}
    ) != 100:
        raise ValueError("registered pair-control group inventory changed")
    if {str(row[input_config["freeze_field"]]) for row in all_rows} != {
        str(input_config["expected_freeze_id"])
    }:
        raise ValueError("registered pair-control freeze ID changed")
    counts = Counter(str(row[input_config["generator_field"]]) for row in all_rows)
    expected_counts = {
        str(key): int(value)
        for key, value in input_config["expected_generator_counts"].items()
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"registered pair-control generator counts changed: {dict(counts)}")
    rows = (
        _select_round_robin(
            all_rows, expected_groups, str(input_config["generator_field"])
        )
        if toy
        else all_rows
    )

    lpips_path: Path | None = None
    parent_path: Path | None = None
    lpips_scores: dict[str, tuple[str, float]] = {}
    teacher_scores: dict[str, tuple[str, float]] = {}
    parent_raw_scores: dict[str, tuple[str, float]] = {}
    if development:
        authorization = config["authorization"]
        toy_summary_path = _resolve(project_root, str(authorization["toy_summary"]))
        _verify(
            toy_summary_path,
            str(authorization["expected_toy_summary_sha256"]),
            "registered pair-control toy summary",
        )
        toy_summary = _read_json(toy_summary_path)
        if toy_summary.get("status") != "registered_pair_controls_toy3_passed" or not all(
            toy_summary.get("structure_gate", {}).values()
        ):
            raise ValueError("registered pair-control toy structure gate did not pass")
        dependencies = config["dependencies"]
        lpips_path = _resolve(
            project_root, str(dependencies["lpips_predictions"])
        )
        _verify(
            lpips_path,
            str(dependencies["expected_lpips_predictions_sha256"]),
            "spatial-LPIPS predictions",
        )
        lpips_scores = _forged_score_map(
            _read_jsonl(lpips_path), baseline="spatial_lpips"
        )
        parent_path = _resolve(
            project_root, str(dependencies["parent_predictions"])
        )
        _verify(
            parent_path,
            str(dependencies["expected_parent_predictions_sha256"]),
            "pair-at-inference parent predictions",
        )
        parent_rows = _read_jsonl(parent_path)
        teacher_scores = _forged_score_map(
            parent_rows, condition="pair_teacher_correct_clean"
        )
        parent_raw_scores = _forged_score_map(
            parent_rows, condition="raw_difference_clean"
        )

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            str(paths["scratch_env"]),
            str(_resolve(project_root, str(paths["scratch_default"]))),
        )
    ).resolve()
    score_cache_dir = _resolve(scratch, str(paths["score_cache_dir"]))
    alignment_cache_dir = _resolve(scratch, str(paths["alignment_cache_dir"]))
    predictions_path = _resolve(project_root, str(paths["predictions"]))
    summary_path = _resolve(project_root, str(paths["summary"]))
    log_path = _resolve(project_root, str(paths["log"]))
    metrics_path = (
        _resolve(project_root, str(paths["metrics"])) if development else None
    )
    comparisons_path = (
        _resolve(project_root, str(paths["comparisons"])) if development else None
    )
    report_path = (
        _resolve(project_root, str(paths["report"])) if development else None
    )
    directories = [
        score_cache_dir,
        alignment_cache_dir,
        predictions_path.parent,
        summary_path.parent,
        log_path.parent,
    ]
    if development:
        assert metrics_path is not None and comparisons_path is not None
        assert report_path is not None
        directories.extend(
            [metrics_path.parent, comparisons_path.parent, report_path.parent]
        )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    preprocessing = config["preprocessing"]
    registration = config["registration"]
    ssim = config["ssim"]
    if cv2.__version__ != str(ssim["opencv_version"]):
        raise ValueError(
            f"OpenCV version changed: {cv2.__version__} != {ssim['opencv_version']}"
        )
    thresholds = (
        np.arange(
            float(config["operating_point"]["candidate_min"]),
            float(config["operating_point"]["candidate_max"]) + 1e-12,
            float(config["operating_point"]["candidate_step"]),
        )
        if development
        else np.asarray([], dtype=float)
    )
    payloads = {
        method: {
            "forged": [],
            "authentic_vectors": [],
            "authentic_max_scores": [],
            "authentic_fpr_max": float(
                config.get("operating_point", {}).get("authentic_pixel_fpr_max", 0.01)
            ),
        }
        for method in METHODS
    }
    forged_scores: dict[str, dict[str, tuple[str, float]]] = {
        method: {} for method in METHODS
    }
    records: list[dict[str, Any]] = []
    failures = 0
    score_cache_hits = 0
    alignment_cache_hits = 0
    structure_observations = {
        "all_maps_finite_and_bounded": True,
        "identical_input_zero": True,
        "deterministic_recomputation": True,
        "pair_order_symmetric": True,
    }
    started = time.monotonic()

    for group_index, row in enumerate(rows, start=1):
        group = str(row["source_group_id"])
        generator = str(row[input_config["generator_field"]])
        for sample_kind in ("forged", "authentic"):
            candidate_field = "image" if sample_kind == "forged" else "authentic"
            candidate_sha_field = (
                "image_sha256" if sample_kind == "forged" else "authentic_sha256"
            )
            candidate_path = _resolve(scratch, str(row[candidate_field]))
            reference_path = _resolve(scratch, str(row["authentic"]))
            mask_path = _resolve(scratch, str(row["mask"]))
            shared_error: Exception | None = None
            try:
                for path, expected, label in (
                    (candidate_path, row[candidate_sha_field], "candidate"),
                    (reference_path, row["authentic_sha256"], "reference"),
                    (mask_path, row["mask_sha256"], "mask"),
                ):
                    _verify(path, str(expected), label)
                with Image.open(candidate_path) as handle:
                    native_candidate = np.asarray(handle.convert("RGB"))
                with Image.open(reference_path) as handle:
                    native_reference = np.asarray(handle.convert("RGB"))
                with Image.open(mask_path) as handle:
                    native_mask = np.asarray(handle.convert("L")) > 0
                if native_candidate.shape[:2] != native_mask.shape:
                    raise ValueError("registered pair-control candidate/mask geometry changed")
                candidate = _resize_image(native_candidate, int(preprocessing["max_side"]))
                reference = _resize_reference(native_reference, candidate.shape[:2])
                alignment_key = _cache_key(
                    {
                        "schema_version": preprocessing[
                            "alignment_cache_schema_version"
                        ],
                        "candidate_sha256": row[candidate_sha_field],
                        "reference_sha256": row["authentic_sha256"],
                        "candidate_shape": list(candidate.shape),
                        "registration": registration,
                    }
                )
                alignment_path = alignment_cache_dir / f"{alignment_key}.npz"
                if alignment_path.is_file():
                    with np.load(alignment_path, allow_pickle=False) as archive:
                        aligned_reference = archive["aligned_reference"].astype(np.uint8)
                        estimated = archive["estimated_homography"].astype(np.float64)
                        alignment_status = str(archive["alignment_status"].item())
                        ecc_correlation = float(archive["ecc_correlation"].item())
                    alignment_cache_hits += 1
                else:
                    aligned_reference, metadata = _estimate_ecc_alignment(
                        candidate,
                        reference,
                        np.eye(3, dtype=np.float64),
                        registration,
                    )
                    estimated = np.asarray(
                        metadata["estimated_homography"], dtype=np.float64
                    )
                    alignment_status = str(metadata["alignment_status"])
                    ecc_correlation = (
                        np.nan
                        if metadata["ecc_correlation"] is None
                        else float(metadata["ecc_correlation"])
                    )
                    temporary = alignment_path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            aligned_reference=aligned_reference.astype(np.uint8),
                            estimated_homography=estimated,
                            alignment_status=np.asarray(alignment_status),
                            ecc_correlation=np.asarray(ecc_correlation),
                        )
                    temporary.replace(alignment_path)
                valid_overlap = _valid_overlap(candidate.shape[:2], estimated)
                if not valid_overlap.any():
                    raise ValueError("registered pair-control ECC produced no valid overlap")
            except Exception as error:
                shared_error = error

            for method in METHODS:
                record: dict[str, Any] = {
                    "record_id": f"registered_pair_controls:{method}:{sample_kind}:{group}",
                    "source_group_id": group,
                    "source_dataset": row["source_dataset"],
                    "generator": generator,
                    "sample_kind": sample_kind,
                    "baseline": method,
                    "status": "failed",
                    "paper_evidence": False,
                    "final_reserve_read": False,
                    "model_training_performed": False,
                    "checkpoint_selection_used": False,
                }
                try:
                    if shared_error is not None:
                        raise shared_error
                    score_key = _cache_key(
                        {
                            "schema_version": preprocessing["score_cache_schema_version"],
                            "candidate_sha256": row[candidate_sha_field],
                            "reference_sha256": row["authentic_sha256"],
                            "sample_kind": sample_kind,
                            "alignment_key": alignment_key,
                            "method": method,
                            "ssim": ssim if method == "registered_ssim_distance" else None,
                            "preprocessing": preprocessing,
                        }
                    )
                    score_path = score_cache_dir / method / f"{score_key}.npz"
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    if score_path.is_file():
                        with np.load(score_path, allow_pickle=False) as archive:
                            scores = archive["scores"]
                            cached_overlap = archive["valid_overlap"].astype(bool)
                        if not np.array_equal(cached_overlap, valid_overlap):
                            raise ValueError("registered pair-control cached overlap changed")
                        score_cache_hits += 1
                    else:
                        scores = _score(method, candidate, aligned_reference, ssim)
                        temporary = score_path.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(
                                handle,
                                scores=scores.astype(np.float32),
                                valid_overlap=valid_overlap.astype(np.uint8),
                            )
                        temporary.replace(score_path)
                    if (
                        scores.dtype != np.float32
                        or scores.shape != candidate.shape[:2]
                        or not np.isfinite(scores).all()
                        or float(scores.min()) < -1e-7
                        or float(scores.max()) > 1.0 + 1e-7
                    ):
                        raise ValueError("registered pair-control score cache is invalid")
                    if toy:
                        recomputed = _score(method, candidate, aligned_reference, ssim)
                        structure_observations["deterministic_recomputation"] &= bool(
                            np.array_equal(scores, recomputed)
                        )
                        reversed_scores = _score(
                            method, aligned_reference, candidate, ssim
                        )
                        structure_observations["pair_order_symmetric"] &= bool(
                            np.array_equal(scores, reversed_scores)
                        )
                        if sample_kind == "authentic":
                            structure_observations["identical_input_zero"] &= bool(
                                float(np.max(np.abs(scores)))
                                <= float(config["structure_gate"]["identical_input_max_abs"])
                            )
                    native_scores = cv2.resize(
                        scores,
                        (native_candidate.shape[1], native_candidate.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    native_valid = cv2.resize(
                        valid_overlap.astype(np.uint8),
                        (native_candidate.shape[1], native_candidate.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    if not native_valid.any():
                        raise ValueError("registered pair-control native overlap is empty")
                    valid_scores = native_scores[native_valid]
                    record.update(
                        {
                            "status": "ok",
                            "cache_key": score_key,
                            "score_cache": str(score_path.relative_to(scratch)),
                            "alignment_cache": str(alignment_path.relative_to(scratch)),
                            "alignment_status": alignment_status,
                            "ecc_correlation": (
                                float(ecc_correlation)
                                if np.isfinite(ecc_correlation)
                                else None
                            ),
                            "score_shape": list(scores.shape),
                            "native_shape": list(native_candidate.shape[:2]),
                            "valid_overlap_fraction": float(native_valid.mean()),
                            "score_min_valid": float(valid_scores.min()),
                            "score_max_valid": float(valid_scores.max()),
                            "score_mean_valid": float(valid_scores.mean()),
                        }
                    )
                    if development and sample_kind == "forged":
                        valid_labels = native_mask[native_valid]
                        average_precision, pixel_auroc = _ranking_metrics(
                            valid_scores, valid_labels
                        )
                        payloads[method]["forged"].append(
                            {
                                "source_group_id": group,
                                "generator": generator,
                                "macro_pixel_ap": average_precision,
                                "pixel_auroc": pixel_auroc,
                                "threshold_vectors": _threshold_vectors(
                                    valid_scores, valid_labels, thresholds
                                ),
                            }
                        )
                        forged_scores[method][group] = (generator, average_precision)
                        record["macro_pixel_ap"] = average_precision
                        record["pixel_auroc"] = pixel_auroc
                    elif development:
                        histogram, _ = np.histogram(
                            valid_scores, bins=np.r_[thresholds, np.inf]
                        )
                        predicted = np.cumsum(
                            histogram[::-1], dtype=np.int64
                        )[::-1]
                        payloads[method]["authentic_vectors"].append(
                            predicted / valid_scores.size
                        )
                        payloads[method]["authentic_max_scores"].append(
                            float(valid_scores.max())
                        )
                except Exception as error:
                    failures += 1
                    record["failure_type"] = type(error).__name__
                    record["failure_reason"] = str(error)
                    logging.exception("record_id=%s failed", record["record_id"])
                _assert_stable_prediction_record(record)
                records.append(record)
                _write_jsonl(predictions_path, records)
        logging.info(
            "completed_groups=%d total_groups=%d failures=%d",
            group_index,
            len(rows),
            failures,
        )

    expected_records = expected_groups * 2 * len(METHODS)
    complete = failures == 0 and len(records) == expected_records
    wall_time = time.monotonic() - started
    cache_bytes = _directory_bytes(score_cache_dir)
    if toy:
        structure_gate = {
            "all_12_records_complete": complete,
            "zero_failures": failures == 0,
            **structure_observations,
            "no_training": True,
            "no_final_reserve_read": True,
            "threshold_selection_not_used": True,
        }
        status = (
            "registered_pair_controls_toy3_passed"
            if all(structure_gate.values())
            else "registered_pair_controls_toy3_failed"
        )
        output = {
            "schema_version": 1,
            "experiment": experiment,
            "status": status,
            "paper_evidence": False,
            "final_reserve_read": False,
            "model_training_performed": False,
            "selected_groups": len(rows),
            "successful_records": len(records) - failures,
            "failed_records": failures,
            "structure_gate": structure_gate,
            "score_cache_hits": score_cache_hits,
            "alignment_cache_hits": alignment_cache_hits,
            "wall_time_seconds": wall_time,
            "cache_bytes": cache_bytes,
            "input": {
                "config_sha256": _sha256(config_path),
                "protocol_sha256": _sha256(protocol_path),
                "manifest_sha256": _sha256(manifest_path),
            },
            "outputs": {
                "predictions": str(predictions_path.relative_to(project_root)),
                "predictions_sha256": _sha256(predictions_path),
                "score_cache_dir": str(score_cache_dir.relative_to(scratch)),
                "alignment_cache_dir": str(alignment_cache_dir.relative_to(scratch)),
            },
        }
        _write_json(summary_path, output)
        if status != "registered_pair_controls_toy3_passed" and runtime[
            "require_all_records"
        ]:
            raise RuntimeError("registered pair-control toy structure gate failed")
        return output

    if not complete:
        output = {
            "schema_version": 1,
            "experiment": experiment,
            "status": "registered_pair_controls_development100_failed_incomplete",
            "paper_evidence": False,
            "successful_records": len(records) - failures,
            "failed_records": failures,
            "expected_records": expected_records,
            "outputs": {
                "predictions": str(predictions_path.relative_to(project_root))
            },
        }
        _write_json(summary_path, output)
        if runtime["require_all_records"]:
            raise RuntimeError(f"registered pair controls failed: {failures} records")
        return output

    assert metrics_path is not None and comparisons_path is not None
    assert report_path is not None
    metrics: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        metric = _aggregate_condition(payloads[method], thresholds)
        selected_threshold = float(metric["pixel_threshold"])
        metric["authentic_image_fpr"] = float(
            np.mean(
                np.asarray(payloads[method]["authentic_max_scores"])
                >= selected_threshold
            )
        )
        metrics[method] = metric

    assert lpips_path is not None and parent_path is not None
    all_score_maps = {
        **forged_scores,
        "spatial_lpips": lpips_scores,
        "pair_teacher_correct_clean": teacher_scores,
        "direct_resized_pair_difference": parent_raw_scores,
    }
    if any(set(values) != set(forged_scores[METHODS[0]]) for values in all_score_maps.values()):
        raise ValueError("registered pair-control dependency group sets differ")
    comparison_pairs = {
        "registered_difference_minus_spatial_lpips": (
            "registered_normalized_rgb_difference",
            "spatial_lpips",
        ),
        "registered_ssim_minus_spatial_lpips": (
            "registered_ssim_distance",
            "spatial_lpips",
        ),
        "registered_difference_minus_registered_ssim": (
            "registered_normalized_rgb_difference",
            "registered_ssim_distance",
        ),
        "registered_difference_minus_pair_teacher": (
            "registered_normalized_rgb_difference",
            "pair_teacher_correct_clean",
        ),
        "registered_ssim_minus_pair_teacher": (
            "registered_ssim_distance",
            "pair_teacher_correct_clean",
        ),
    }
    bootstrap = config["bootstrap"]
    comparisons = {
        name: _stratified_paired_bootstrap(
            all_score_maps[left],
            all_score_maps[right],
            int(bootstrap["seed"]) + offset,
            int(bootstrap["resamples"]),
            float(bootstrap["confidence_level"]),
        )
        for offset, (name, (left, right)) in enumerate(comparison_pairs.items())
    }
    posthoc_alignment = _stratified_paired_bootstrap(
        forged_scores["registered_normalized_rgb_difference"],
        parent_raw_scores,
        int(bootstrap["seed"]) + 1000,
        int(bootstrap["resamples"]),
        float(bootstrap["confidence_level"]),
    )
    by_generator: dict[str, list[float]] = {}
    for generator, value in parent_raw_scores.values():
        by_generator.setdefault(generator, []).append(value)
    posthoc_alignment.update(
        {
            "preregistered": False,
            "role": "exploratory_registration_diagnostic",
            "direct_resized_pair_generator_macro_ap": float(
                np.mean([np.mean(values) for values in by_generator.values()])
            ),
        }
    )
    _write_csv(
        metrics_path,
        [{"baseline": method, **metrics[method]} for method in METHODS],
    )
    comparison_rows = []
    for name, value in comparisons.items():
        row: dict[str, Any] = {
            "comparison": name,
            "effect": value["effect"],
            "ci_low": value["ci_low"],
            "ci_high": value["ci_high"],
            "paper_evidence": False,
        }
        for generator, effect in value["per_generator_effect"].items():
            safe = "".join(
                character if character.isalnum() else "_" for character in generator
            ).strip("_")
            row[f"effect__{safe}"] = effect
        comparison_rows.append(row)
    _write_csv(comparisons_path, comparison_rows)
    engineering_gate = {
        "all_400_records_complete": len(records) == expected_records,
        "zero_failures": failures == 0,
        "all_scores_finite": all(
            np.isfinite(float(row["score_min_valid"]))
            and np.isfinite(float(row["score_max_valid"]))
            for row in records
        ),
        "both_generators_complete": Counter(
            row["generator"]
            for row in records
            if row["sample_kind"] == "forged"
            and row["baseline"] == METHODS[0]
        )
        == expected_counts,
        "no_final_reserve_read": True,
        "no_training": True,
        "wall_time_below_one_cpu_hour": wall_time < 3600.0,
        "new_cache_storage_below_two_gib": cache_bytes < 2 * 1024**3,
    }
    status = (
        "registered_pair_controls_development100_complete"
        if all(engineering_gate.values())
        else "registered_pair_controls_development100_engineering_gate_failed"
    )
    output = {
        "schema_version": 1,
        "experiment": experiment,
        "status": status,
        "paper_evidence": False,
        "development_only": True,
        "final_reserve_read": False,
        "model_training_performed": False,
        "checkpoint_selection_used": False,
        "threshold_selection_used": True,
        "selected_development_groups": len(rows),
        "successful_records": len(records),
        "failed_records": failures,
        "metrics": metrics,
        "comparisons": comparisons,
        "posthoc_alignment_diagnostic": posthoc_alignment,
        "engineering_gate": engineering_gate,
        "score_cache_hits": score_cache_hits,
        "alignment_cache_hits": alignment_cache_hits,
        "wall_time_seconds": wall_time,
        "cache_bytes": cache_bytes,
        "opencv_version": cv2.__version__,
        "input": {
            "config_sha256": _sha256(config_path),
            "protocol_sha256": _sha256(protocol_path),
            "toy_summary_sha256": _sha256(toy_summary_path),
            "manifest_sha256": _sha256(manifest_path),
            "lpips_predictions_sha256": _sha256(lpips_path),
            "parent_predictions_sha256": _sha256(parent_path),
        },
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
            "report": str(report_path.relative_to(project_root)),
            "score_cache_dir": str(score_cache_dir.relative_to(scratch)),
            "alignment_cache_dir": str(alignment_cache_dir.relative_to(scratch)),
        },
    }
    _write_json(summary_path, output)
    report_path.write_text(_report(output), encoding="utf-8")
    output["outputs"]["report_sha256"] = _sha256(report_path)
    _write_json(summary_path, output)
    if status != "registered_pair_controls_development100_complete" and runtime[
        "require_all_records"
    ]:
        raise RuntimeError("registered pair-control engineering gate failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
