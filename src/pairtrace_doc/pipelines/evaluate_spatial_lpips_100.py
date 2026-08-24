from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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

from pairtrace_doc.pipelines.compare_generator_balanced_1000 import (
    _stratified_paired_bootstrap,
)
from pairtrace_doc.pipelines.diagnose_pair_at_inference_alignment import (
    _estimate_ecc_alignment,
)
from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
    _aggregate_condition,
    _resize_image,
    _resize_reference,
)
from pairtrace_doc.pipelines.run_spatial_lpips import (
    _cache_key,
    _spatial_lpips_tiled,
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


_RUNTIME_ONLY_PREDICTION_FIELDS = frozenset({"cache_hit", "latency_ms"})


def _assert_stable_prediction_record(record: dict[str, Any]) -> None:
    transient = sorted(_RUNTIME_ONLY_PREDICTION_FIELDS.intersection(record))
    if transient:
        raise ValueError(
            "prediction record contains runtime-only fields: " + ", ".join(transient)
        )


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


def _parent_score_map(
    rows: list[dict[str, Any]], condition: str
) -> dict[str, tuple[str, float]]:
    selected = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("sample_kind") == "forged"
        and row.get("condition") == condition
    ]
    result = {
        str(row["source_group_id"]): (
            str(row["generator"]),
            float(row["macro_pixel_ap"]),
        )
        for row in selected
    }
    if len(result) != len(selected):
        raise ValueError(f"parent comparison contains duplicate groups: {condition}")
    return result


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _report(summary: dict[str, Any]) -> str:
    metric = summary["metrics"]
    comparisons = summary["comparisons"]
    return f"""# Spatial-LPIPS equal-information development-100 result

Status: `{summary['status']}`. This is viewed AIForge development evidence,
not confirmatory paper evidence.

## Result

| Method | Generator-macro pixel AP | Document-macro pixel AP | Pixel F1 | Pixel IoU | Authentic pixel FPR |
|---|---:|---:|---:|---:|---:|
| Spatial LPIPS | {metric['generator_macro_pixel_ap']:.6f} | {metric['macro_pixel_ap']:.6f} | {metric['pixel_f1']:.6f} | {metric['pixel_iou']:.6f} | {metric['authentic_pixel_fpr']:.6f} |

The development-selected LPIPS pixel threshold is
`{metric['pixel_threshold']:.6f}`. It was selected only on these viewed groups
under the frozen authentic-FPR constraint and is not a final operating point.

## Paired source-group comparisons

| Comparison | AP effect | 95% interval |
|---|---:|---:|
| Spatial LPIPS minus raw registered RGB difference | {comparisons['spatial_lpips_minus_raw_difference']['effect']:.6f} | [{comparisons['spatial_lpips_minus_raw_difference']['ci_low']:.6f}, {comparisons['spatial_lpips_minus_raw_difference']['ci_high']:.6f}] |
| Spatial LPIPS minus clean correct-reference teacher | {comparisons['spatial_lpips_minus_pair_teacher']['effect']:.6f} | [{comparisons['spatial_lpips_minus_pair_teacher']['ci_low']:.6f}, {comparisons['spatial_lpips_minus_pair_teacher']['ci_high']:.6f}] |

All 100 groups and 200 candidate/authentic records are retained. The baseline
uses the official LPIPS AlexNet spatial map, the same authentic reference, the
frozen ECC registration path, a 1,024-pixel maximum side, and valid-overlap
masking. No task training or final-reserve read occurred.

This closes the generic perceptual-control subtask only. It does not replace
the still-missing prospective FC-Siam-diff/equal-budget learned comparator.
"""


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("spatial-LPIPS development config must be a mapping")
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if config["experiment"]["paper_evidence"]:
        raise ValueError("spatial-LPIPS development output cannot be paper evidence")
    if not all(
        bool(runtime.get(key))
        for key in (
            "gpu_launch_authorized",
            "development_inference_authorized",
            "threshold_selection_authorized",
        )
    ):
        raise PermissionError("spatial-LPIPS development-100 is not authorized")
    if any(
        bool(runtime.get(key))
        for key in (
            "model_training_authorized",
            "checkpoint_selection_authorized",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("spatial-LPIPS development crossed an evidence boundary")
    if int(runtime["max_groups"]) != 100:
        raise ValueError("spatial-LPIPS development stage requires exactly 100 groups")
    device = torch.device(str(runtime["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("spatial-LPIPS development-100 requires CUDA")

    experiment = config["experiment"]
    protocol_path = _resolve(project_root, str(experiment["protocol"]))
    _verify(
        protocol_path,
        str(experiment["expected_protocol_sha256"]),
        "spatial-LPIPS development protocol",
    )
    authorization = config["authorization"]
    toy_summary_path = _resolve(project_root, str(authorization["toy_summary"]))
    _verify(
        toy_summary_path,
        str(authorization["expected_toy_summary_sha256"]),
        "spatial-LPIPS toy summary",
    )
    toy = _read_json(toy_summary_path)
    if toy.get("status") != "passed_structure_gate" or not all(
        toy.get("checks", {}).values()
    ):
        raise ValueError("spatial-LPIPS toy structure gate did not pass")
    if toy.get("model_training_performed") or toy.get("final_reserve_read"):
        raise ValueError("spatial-LPIPS toy crossed an evidence boundary")

    input_config = config["input"]
    manifest_path = _resolve(project_root, str(input_config["manifest"]))
    _verify(
        manifest_path,
        str(input_config["expected_manifest_sha256"]),
        "spatial-LPIPS development manifest",
    )
    rows = sorted(
        _read_jsonl(manifest_path), key=lambda row: str(row["source_group_id"])
    )
    if len(rows) != 100 or len({str(row["source_group_id"]) for row in rows}) != 100:
        raise ValueError("spatial-LPIPS development group inventory changed")
    if {str(row[input_config["freeze_field"]]) for row in rows} != {
        str(input_config["expected_freeze_id"])
    }:
        raise ValueError("spatial-LPIPS development freeze ID changed")
    counts = Counter(str(row[input_config["generator_field"]]) for row in rows)
    expected_counts = {
        str(key): int(value)
        for key, value in input_config["expected_generator_counts"].items()
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"spatial-LPIPS generator counts changed: {dict(counts)}")

    parent_summary_path = _resolve(project_root, str(input_config["parent_summary"]))
    _verify(
        parent_summary_path,
        str(input_config["expected_parent_summary_sha256"]),
        "pair-at-inference parent summary",
    )
    parent = _read_json(parent_summary_path)
    if parent.get("status") != "passed_feasibility_gate" or parent.get(
        "final_reserve_read"
    ):
        raise ValueError("pair-at-inference parent is not eligible development data")
    parent_predictions_path = _resolve(
        project_root, str(input_config["parent_predictions"])
    )
    _verify(
        parent_predictions_path,
        str(input_config["expected_parent_predictions_sha256"]),
        "pair-at-inference parent predictions",
    )
    parent_rows = _read_jsonl(parent_predictions_path)

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            str(paths["scratch_env"]),
            str(_resolve(project_root, str(paths["scratch_default"])),),
        )
    ).resolve()
    os.environ["TORCH_HOME"] = str(_resolve(scratch, str(paths["torch_home"])))
    model_config = config["model"]
    trunk_path = _resolve(scratch, str(model_config["trunk_weights"]))
    _verify(
        trunk_path,
        str(model_config["trunk_weights_sha256"]),
        "spatial-LPIPS trunk weights",
    )
    import lpips

    package_version = importlib.metadata.version("lpips")
    if package_version != str(model_config["package_version"]):
        raise ValueError("spatial-LPIPS package version changed")
    calibration_path = (
        Path(lpips.__file__).resolve().parent
        / str(model_config["calibration_weights_package_relative"])
    )
    _verify(
        calibration_path,
        str(model_config["calibration_weights_sha256"]),
        "spatial-LPIPS calibration weights",
    )
    torch.manual_seed(int(experiment["seed"]))
    torch.cuda.manual_seed_all(int(experiment["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = lpips.LPIPS(
        pretrained=True,
        net=str(model_config["trunk"]),
        version=str(model_config["lpips_version"]),
        lpips=True,
        spatial=True,
        model_path=str(calibration_path),
        eval_mode=True,
        verbose=False,
    ).to(device).eval().requires_grad_(False)

    score_cache_dir = _resolve(scratch, str(paths["score_cache_dir"]))
    alignment_cache_dir = _resolve(scratch, str(paths["alignment_cache_dir"]))
    predictions_path = _resolve(project_root, str(paths["predictions"]))
    metrics_path = _resolve(project_root, str(paths["metrics"]))
    comparisons_path = _resolve(project_root, str(paths["comparisons"]))
    summary_path = _resolve(project_root, str(paths["summary"]))
    report_path = _resolve(project_root, str(paths["report"]))
    log_path = _resolve(project_root, str(paths["log"]))
    for path in (
        score_cache_dir,
        alignment_cache_dir,
        predictions_path.parent,
        metrics_path.parent,
        comparisons_path.parent,
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
    preprocessing = config["preprocessing"]
    inference = config["inference"]
    registration = config["registration"]
    thresholds = np.arange(
        float(config["operating_point"]["candidate_min"]),
        float(config["operating_point"]["candidate_max"]) + 1e-12,
        float(config["operating_point"]["candidate_step"]),
    )
    payload = {
        "forged": [],
        "authentic_vectors": [],
        "authentic_fpr_max": float(
            config["operating_point"]["authentic_pixel_fpr_max"]
        ),
    }
    spatial_scores: dict[str, tuple[str, float]] = {}
    records: list[dict[str, Any]] = []
    failures = 0
    score_cache_hits = 0
    alignment_cache_hits = 0
    latencies: list[float] = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for group_index, row in enumerate(rows, start=1):
        group = str(row["source_group_id"])
        generator = str(row[input_config["generator_field"]])
        for sample_kind in ("forged", "authentic"):
            record: dict[str, Any] = {
                "record_id": f"spatial_lpips_development100:{sample_kind}:{group}",
                "source_group_id": group,
                "source_dataset": row["source_dataset"],
                "generator": generator,
                "sample_kind": sample_kind,
                "baseline": "spatial_lpips",
                "status": "failed",
                "paper_evidence": False,
                "final_reserve_read": False,
                "model_training_performed": False,
                "checkpoint_selection_used": False,
            }
            try:
                candidate_field = "image" if sample_kind == "forged" else "authentic"
                candidate_sha_field = (
                    "image_sha256" if sample_kind == "forged" else "authentic_sha256"
                )
                candidate_path = _resolve(scratch, str(row[candidate_field]))
                reference_path = _resolve(scratch, str(row["authentic"]))
                mask_path = _resolve(scratch, str(row["mask"]))
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
                    raise ValueError("spatial-LPIPS candidate/mask geometry changed")
                candidate = _resize_image(
                    native_candidate, int(preprocessing["max_side"])
                )
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
                    raise ValueError("spatial-LPIPS ECC produced no valid overlap")
                model_identity = {
                    "name": "spatial_lpips",
                    "package_version": package_version,
                    "repository_revision": model_config["repository_revision"],
                    "lpips_version": model_config["lpips_version"],
                    "trunk_weights_sha256": model_config["trunk_weights_sha256"],
                    "calibration_weights_sha256": model_config[
                        "calibration_weights_sha256"
                    ],
                }
                score_key = _cache_key(
                    {
                        "schema_version": preprocessing["score_cache_schema_version"],
                        "candidate_sha256": row[candidate_sha_field],
                        "reference_sha256": row["authentic_sha256"],
                        "sample_kind": sample_kind,
                        "alignment_key": alignment_key,
                        "model": model_identity,
                        "preprocessing": preprocessing,
                        "inference": inference,
                    }
                )
                score_path = score_cache_dir / f"{score_key}.npz"
                latency_ms = 0.0
                if score_path.is_file():
                    with np.load(score_path, allow_pickle=False) as archive:
                        scores = archive["scores"]
                        cached_overlap = archive["valid_overlap"].astype(bool)
                    if not np.array_equal(cached_overlap, valid_overlap):
                        raise ValueError("spatial-LPIPS cached overlap changed")
                    score_cache_hits += 1
                else:
                    inference_started = time.perf_counter()
                    scores = _spatial_lpips_tiled(
                        model,
                        candidate,
                        aligned_reference,
                        device,
                        int(inference["tile_size"]),
                        int(inference["tile_stride"]),
                        int(inference["tile_batch_size"]),
                    ).astype(np.float32)
                    latency_ms = (time.perf_counter() - inference_started) * 1000.0
                    temporary = score_path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            scores=scores,
                            valid_overlap=valid_overlap.astype(np.uint8),
                        )
                    temporary.replace(score_path)
                    latencies.append(latency_ms)
                if (
                    scores.dtype != np.float32
                    or scores.shape != candidate.shape[:2]
                    or not np.isfinite(scores).all()
                ):
                    raise ValueError("spatial-LPIPS development score cache is invalid")
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
                    raise ValueError("spatial-LPIPS native valid overlap is empty")
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
                if sample_kind == "forged":
                    valid_labels = native_mask[native_valid]
                    average_precision, pixel_auroc = _ranking_metrics(
                        valid_scores, valid_labels
                    )
                    payload["forged"].append(
                        {
                            "generator": generator,
                            "macro_pixel_ap": average_precision,
                            "pixel_auroc": pixel_auroc,
                            "threshold_vectors": _threshold_vectors(
                                valid_scores, valid_labels, thresholds
                            ),
                        }
                    )
                    spatial_scores[group] = (generator, average_precision)
                    record["macro_pixel_ap"] = average_precision
                    record["pixel_auroc"] = pixel_auroc
                else:
                    histogram, _ = np.histogram(
                        valid_scores, bins=np.r_[thresholds, np.inf]
                    )
                    predicted = np.cumsum(histogram[::-1], dtype=np.int64)[::-1]
                    payload["authentic_vectors"].append(predicted / valid_scores.size)
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

    expected_records = 200
    complete = failures == 0 and len(records) == expected_records
    if not complete:
        failed_output = {
            "experiment": experiment,
            "status": "spatial_lpips_development100_failed_incomplete",
            "paper_evidence": False,
            "successful_records": len(records) - failures,
            "failed_records": failures,
            "expected_records": expected_records,
            "outputs": {"predictions": str(predictions_path.relative_to(project_root))},
        }
        _write_json(summary_path, failed_output)
        if runtime["require_all_records"]:
            raise RuntimeError(f"spatial-LPIPS development failed: {failures} records")
        return failed_output

    metrics = _aggregate_condition(payload, thresholds)
    raw_scores = _parent_score_map(parent_rows, "raw_difference_clean")
    teacher_scores = _parent_score_map(parent_rows, "pair_teacher_correct_clean")
    if set(spatial_scores) != set(raw_scores) or set(spatial_scores) != set(
        teacher_scores
    ):
        raise ValueError("spatial-LPIPS and parent comparison groups differ")
    bootstrap = config["bootstrap"]
    comparisons = {
        "spatial_lpips_minus_raw_difference": _stratified_paired_bootstrap(
            spatial_scores,
            raw_scores,
            int(bootstrap["seed"]),
            int(bootstrap["resamples"]),
            float(bootstrap["confidence_level"]),
        ),
        "spatial_lpips_minus_pair_teacher": _stratified_paired_bootstrap(
            spatial_scores,
            teacher_scores,
            int(bootstrap["seed"]) + 1,
            int(bootstrap["resamples"]),
            float(bootstrap["confidence_level"]),
        ),
    }
    _write_csv(metrics_path, [{"baseline": "spatial_lpips", **metrics}])
    _write_csv(
        comparisons_path,
        [
            {"comparison": name, **value, "paper_evidence": False}
            for name, value in comparisons.items()
        ],
    )
    cache_bytes = _directory_bytes(score_cache_dir) + _directory_bytes(
        alignment_cache_dir
    )
    wall_time = time.monotonic() - started
    gate = {
        "all_200_records_complete": len(records) == expected_records,
        "zero_failures": failures == 0,
        "all_scores_finite": all(
            np.isfinite(float(row["score_min_valid"]))
            and np.isfinite(float(row["score_max_valid"]))
            for row in records
        ),
        "both_generators_complete": Counter(
            row["generator"] for row in records if row["sample_kind"] == "forged"
        )
        == expected_counts,
        "no_final_reserve_read": True,
        "no_training": True,
        "wall_time_below_one_gpu_hour": wall_time < 3600.0,
        "cache_storage_below_two_gib": cache_bytes < 2 * 1024**3,
    }
    status = (
        "spatial_lpips_development100_complete"
        if all(gate.values())
        else "spatial_lpips_development100_engineering_gate_failed"
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
        "engineering_gate": gate,
        "score_cache_hits": score_cache_hits,
        "alignment_cache_hits": alignment_cache_hits,
        "new_inference_latency_ms_total": float(sum(latencies)),
        "wall_time_seconds": wall_time,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "cache_bytes": cache_bytes,
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "lpips_package_version": package_version,
        "input": {
            "config_sha256": _sha256(config_path),
            "protocol_sha256": _sha256(protocol_path),
            "toy_summary_sha256": _sha256(toy_summary_path),
            "manifest_sha256": _sha256(manifest_path),
            "parent_summary_sha256": _sha256(parent_summary_path),
            "parent_predictions_sha256": _sha256(parent_predictions_path),
        },
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
            "report": str(report_path.relative_to(project_root)),
            "log": str(log_path.relative_to(project_root)),
            "score_cache_dir": str(score_cache_dir.relative_to(scratch)),
            "alignment_cache_dir": str(alignment_cache_dir.relative_to(scratch)),
        },
    }
    _write_json(summary_path, output)
    report_path.write_text(_report(output), encoding="utf-8")
    output["outputs"]["report_sha256"] = _sha256(report_path)
    _write_json(summary_path, output)
    if status != "spatial_lpips_development100_complete" and runtime[
        "require_all_records"
    ]:
        raise RuntimeError("spatial-LPIPS development engineering gate failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
