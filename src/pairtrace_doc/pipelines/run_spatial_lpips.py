from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from pairtrace_doc.pipelines.diagnose_pair_at_inference_alignment import (
    _estimate_ecc_alignment,
)
from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
    _resize_image,
    _resize_reference,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _positions,
    _read_jsonl,
    _resolve,
    _sha256,
    _write_json,
    _write_jsonl,
)


def _select_round_robin(
    rows: list[dict[str, Any]], count: int, generator_field: str
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("toy selection count must be positive")
    grouped: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: str(item["source_group_id"])):
        grouped[str(row[generator_field])].append(row)
    if not grouped:
        raise ValueError("cannot select from an empty manifest")
    selected: list[dict[str, Any]] = []
    generators = sorted(grouped)
    while len(selected) < min(count, len(rows)):
        advanced = False
        for generator in generators:
            if grouped[generator] and len(selected) < count:
                selected.append(grouped[generator].popleft())
                advanced = True
        if not advanced:
            break
    return selected


def _cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _spatial_lpips_tiled(
    model: Any,
    candidate: np.ndarray,
    reference: np.ndarray,
    device: Any,
    tile: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    if candidate.shape != reference.shape or candidate.ndim != 3:
        raise ValueError("spatial LPIPS requires matched HWC RGB arrays")
    if candidate.shape[2] != 3:
        raise ValueError("spatial LPIPS requires exactly three RGB channels")
    if tile < 64 or not 0 < stride <= tile or batch_size < 1:
        raise ValueError("invalid spatial LPIPS tiling configuration")
    height, width = candidate.shape[:2]
    if min(height, width) < 2:
        raise ValueError("spatial LPIPS cannot reflect-pad a one-pixel dimension")
    pad_height = max(0, tile - height)
    pad_width = max(0, tile - width)
    padding = ((0, pad_height), (0, pad_width), (0, 0))
    candidate_padded = np.pad(candidate, padding, mode="reflect")
    reference_padded = np.pad(reference, padding, mode="reflect")
    padded_height, padded_width = candidate_padded.shape[:2]
    coordinates = [
        (top, left)
        for top in _positions(padded_height, tile, stride)
        for left in _positions(padded_width, tile, stride)
    ]
    accumulator = np.zeros((padded_height, padded_width), dtype=np.float32)
    counts = np.zeros((padded_height, padded_width), dtype=np.float32)
    model.eval()
    for start in range(0, len(coordinates), batch_size):
        selected = coordinates[start : start + batch_size]
        candidate_batch = np.stack(
            [
                candidate_padded[top : top + tile, left : left + tile]
                for top, left in selected
            ]
        )
        reference_batch = np.stack(
            [
                reference_padded[top : top + tile, left : left + tile]
                for top, left in selected
            ]
        )
        candidate_tensor = torch.from_numpy(
            (candidate_batch.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
        ).to(device)
        reference_tensor = torch.from_numpy(
            (reference_batch.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
        ).to(device)
        with torch.inference_mode():
            output = model(candidate_tensor, reference_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim != 4 or output.shape[:2] != (len(selected), 1):
                raise ValueError(f"unexpected spatial LPIPS output shape: {output.shape}")
            if output.shape[2:] != (tile, tile):
                output = torch.nn.functional.interpolate(
                    output,
                    size=(tile, tile),
                    mode="bilinear",
                    align_corners=False,
                )
            maps = output[:, 0].detach().float().cpu().numpy()
        for score, (top, left) in zip(maps, selected):
            accumulator[top : top + tile, left : left + tile] += score
            counts[top : top + tile, left : left + tile] += 1.0
    if np.any(counts == 0):
        raise ValueError("spatial LPIPS tiling left uncovered pixels")
    return (accumulator / counts)[:height, :width]


def _valid_overlap(
    shape: tuple[int, int], estimated_homography: np.ndarray
) -> np.ndarray:
    height, width = shape
    source = np.ones((height, width), dtype=np.uint8)
    return cv2.warpPerspective(
        source,
        estimated_homography.astype(np.float64),
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_runtime(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    if str(runtime["device"]) != "cpu" or not runtime["cpu_structure_gate_authorized"]:
        raise ValueError("spatial-LPIPS toy gate requires explicit CPU authorization")
    if runtime["gpu_launch_authorized"] or runtime["model_training_authorized"]:
        raise ValueError("spatial-LPIPS toy gate cannot authorize GPU use or training")
    if runtime["threshold_selection_authorized"]:
        raise ValueError("spatial-LPIPS toy gate cannot select an operating point")
    if runtime["final_reserve_read_allowed"]:
        raise ValueError("spatial-LPIPS toy gate cannot read the final reserve")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("spatial-LPIPS toy output cannot be paper evidence")
    if int(runtime["max_groups"]) != 3:
        raise ValueError("the frozen spatial-LPIPS structure gate is toy-3 only")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_runtime(config)
    for name, value in config["runtime"].get("environment", {}).items():
        os.environ[str(name)] = str(value)

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("spatial-LPIPS protocol SHA-256 changed")
    input_config = config["input"]
    manifest_path = _resolve(project_root, input_config["manifest"])
    if _sha256(manifest_path) != input_config["expected_manifest_sha256"]:
        raise ValueError("spatial-LPIPS input manifest SHA-256 changed")
    rows = _read_jsonl(manifest_path)
    if len(rows) != int(input_config["expected_development_groups"]):
        raise ValueError("spatial-LPIPS development manifest count changed")
    if len({str(row["source_group_id"]) for row in rows}) != len(rows):
        raise ValueError("spatial-LPIPS development manifest contains duplicate groups")
    if {str(row[input_config["freeze_field"]]) for row in rows} != {
        str(input_config["expected_freeze_id"])
    }:
        raise ValueError("spatial-LPIPS input freeze ID changed")
    counts = Counter(str(row[input_config["generator_field"]]) for row in rows)
    if dict(counts) != {
        str(name): int(value)
        for name, value in input_config["expected_generator_counts"].items()
    }:
        raise ValueError("spatial-LPIPS generator counts changed")
    selected = _select_round_robin(
        rows,
        int(config["runtime"]["max_groups"]),
        str(input_config["generator_field"]),
    )

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    torch_home = _resolve(scratch, paths["torch_home"])
    os.environ["TORCH_HOME"] = str(torch_home)
    trunk_path = _resolve(scratch, config["model"]["trunk_weights"])
    if _sha256(trunk_path) != config["model"]["trunk_weights_sha256"]:
        raise ValueError("spatial-LPIPS AlexNet weights changed")

    import lpips
    import torch

    package_version = importlib.metadata.version("lpips")
    if package_version != str(config["model"]["package_version"]):
        raise ValueError("spatial-LPIPS package version changed")
    calibration_path = (
        Path(lpips.__file__).resolve().parent
        / config["model"]["calibration_weights_package_relative"]
    )
    if _sha256(calibration_path) != config["model"]["calibration_weights_sha256"]:
        raise ValueError("spatial-LPIPS calibration weights changed")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    torch.manual_seed(int(config["experiment"]["seed"]))
    device = torch.device("cpu")
    model = lpips.LPIPS(
        pretrained=True,
        net=str(config["model"]["trunk"]),
        version=str(config["model"]["lpips_version"]),
        lpips=True,
        spatial=True,
        model_path=str(calibration_path),
        eval_mode=True,
        verbose=False,
    ).to(device).eval().requires_grad_(False)

    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    alignment_cache_dir = _resolve(scratch, paths["alignment_cache_dir"])
    predictions_path = _resolve(project_root, paths["predictions"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (
        score_cache_dir,
        alignment_cache_dir,
        predictions_path.parent,
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

    inference = config["inference"]
    preprocessing = config["preprocessing"]
    registration = config["registration"]
    model_identity = {
        "name": "spatial_lpips",
        "package_version": package_version,
        "repository_revision": config["model"]["repository_revision"],
        "lpips_version": config["model"]["lpips_version"],
        "trunk": config["model"]["trunk"],
        "trunk_weights_sha256": config["model"]["trunk_weights_sha256"],
        "calibration_weights_sha256": config["model"][
            "calibration_weights_sha256"
        ],
    }

    probe = np.arange(64 * 64 * 3, dtype=np.uint32).reshape(64, 64, 3)
    probe = (probe % 256).astype(np.uint8)
    shifted_probe = np.roll(probe, 1, axis=1)
    probe_kwargs = {
        "model": model,
        "device": device,
        "tile": 64,
        "stride": 64,
        "batch_size": 1,
    }
    identical_probe = _spatial_lpips_tiled(
        candidate=probe, reference=probe, **probe_kwargs
    )
    determinism_probe = _spatial_lpips_tiled(
        candidate=probe, reference=shifted_probe, **probe_kwargs
    )
    determinism_repeat = _spatial_lpips_tiled(
        candidate=probe, reference=shifted_probe, **probe_kwargs
    )
    reverse_probe = _spatial_lpips_tiled(
        candidate=shifted_probe, reference=probe, **probe_kwargs
    )
    probe_results = {
        "identical_input_max_abs": float(np.max(np.abs(identical_probe))),
        "determinism_max_abs_difference": float(
            np.max(np.abs(determinism_probe - determinism_repeat))
        ),
        "pair_order_max_abs_difference": float(
            np.max(np.abs(determinism_probe - reverse_probe))
        ),
    }

    records: list[dict[str, Any]] = []
    cache_hits = 0
    alignment_cache_hits = 0
    latencies: list[float] = []
    started = time.monotonic()
    for row in selected:
        group = str(row["source_group_id"])
        for sample_kind in ("forged", "authentic"):
            record: dict[str, Any] = {
                "record_id": f"spatial_lpips:{sample_kind}:{group}",
                "source_group_id": group,
                "generator": str(row[input_config["generator_field"]]),
                "sample_kind": sample_kind,
                "baseline": "spatial_lpips",
                "status": "failed",
                "paper_evidence": False,
                "threshold_selection_used": False,
                "final_reserve_read": False,
            }
            try:
                candidate_field = "image" if sample_kind == "forged" else "authentic"
                candidate_sha_field = (
                    "image_sha256" if sample_kind == "forged" else "authentic_sha256"
                )
                candidate_path = _resolve(scratch, row[candidate_field])
                reference_path = _resolve(scratch, row["authentic"])
                mask_path = _resolve(scratch, row["mask"])
                for path, expected, label in (
                    (candidate_path, row[candidate_sha_field], "candidate"),
                    (reference_path, row["authentic_sha256"], "reference"),
                    (mask_path, row["mask_sha256"], "mask"),
                ):
                    if _sha256(path) != expected:
                        raise ValueError(f"{label} SHA-256 changed")
                with Image.open(candidate_path) as handle:
                    native_candidate = np.asarray(handle.convert("RGB"))
                with Image.open(reference_path) as handle:
                    native_reference = np.asarray(handle.convert("RGB"))
                with Image.open(mask_path) as handle:
                    native_mask = np.asarray(handle.convert("L"))
                if native_candidate.shape[:2] != native_mask.shape:
                    raise ValueError("candidate/mask native geometry changed")
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
                        phase_response = float(
                            archive["phase_correlation_response"].item()
                        )
                    alignment_cache_hits += 1
                else:
                    identity = np.eye(3, dtype=np.float64)
                    aligned_reference, metadata = _estimate_ecc_alignment(
                        candidate, reference, identity, registration
                    )
                    estimated = np.asarray(
                        metadata["estimated_homography"], dtype=np.float64
                    )
                    alignment_status = str(metadata["alignment_status"])
                    ecc_correlation = float(metadata["ecc_correlation"])
                    phase_response = float(metadata["phase_correlation_response"])
                    temporary = alignment_path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            aligned_reference=aligned_reference.astype(np.uint8),
                            estimated_homography=estimated,
                            alignment_status=np.asarray(alignment_status),
                            ecc_correlation=np.asarray(ecc_correlation),
                            phase_correlation_response=np.asarray(phase_response),
                        )
                    temporary.replace(alignment_path)
                valid_overlap = _valid_overlap(candidate.shape[:2], estimated)
                if not valid_overlap.any():
                    raise ValueError("ECC alignment produced no valid overlap")

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
                        scores = archive["scores"].astype(np.float32)
                        cached_overlap = archive["valid_overlap"].astype(bool)
                    if not np.array_equal(cached_overlap, valid_overlap):
                        raise ValueError("spatial-LPIPS cached overlap changed")
                    cache_hits += 1
                    cache_hit = True
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
                    )
                    latency_ms = (time.perf_counter() - inference_started) * 1000.0
                    temporary = score_path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            scores=scores.astype(np.float16),
                            valid_overlap=valid_overlap.astype(np.uint8),
                        )
                    temporary.replace(score_path)
                    cache_hit = False
                if scores.shape != candidate.shape[:2] or not np.isfinite(scores).all():
                    raise ValueError("spatial-LPIPS score cache is invalid")
                valid_scores = scores[valid_overlap]
                record.update(
                    {
                        "status": "ok",
                        "cache_hit": cache_hit,
                        "cache_key": score_key,
                        "score_cache": str(score_path.relative_to(scratch)),
                        "alignment_cache": str(alignment_path.relative_to(scratch)),
                        "alignment_status": alignment_status,
                        "ecc_correlation": _finite_or_none(ecc_correlation),
                        "phase_correlation_response": _finite_or_none(phase_response),
                        "score_shape": list(scores.shape),
                        "native_shape": list(native_candidate.shape[:2]),
                        "valid_overlap_fraction": float(valid_overlap.mean()),
                        "score_min_valid": float(valid_scores.min()),
                        "score_max_valid": float(valid_scores.max()),
                        "score_mean_valid": float(valid_scores.mean()),
                        "latency_ms": latency_ms,
                    }
                )
                if not cache_hit:
                    latencies.append(latency_ms)
            except Exception as error:
                record["failure_type"] = type(error).__name__
                record["failure_reason"] = str(error)
                logging.exception("record_id=%s failed", record["record_id"])
            records.append(record)
            _write_jsonl(predictions_path, records)

    expected_records = len(selected) * 2
    ok_records = [record for record in records if record["status"] == "ok"]
    checks = {
        "selected_exactly_three_groups": len(selected) == 3,
        "two_records_per_group": len(records) == expected_records == 6,
        "all_records_successful": len(ok_records) == expected_records,
        "all_score_shapes_two_dimensional": all(
            len(record["score_shape"]) == 2 for record in ok_records
        ),
        "all_valid_overlap_positive": all(
            float(record["valid_overlap_fraction"]) > 0.0 for record in ok_records
        ),
        "identical_input_near_zero": probe_results["identical_input_max_abs"]
        <= float(config["structure_gate"]["identical_input_max_abs"]),
        "deterministic_repeat": probe_results["determinism_max_abs_difference"]
        <= float(config["structure_gate"]["determinism_max_abs_difference"]),
        "pair_order_symmetric": probe_results["pair_order_max_abs_difference"]
        <= float(config["structure_gate"]["pair_order_max_abs_difference"]),
        "no_threshold_selection": not config["runtime"][
            "threshold_selection_authorized"
        ],
        "no_final_reserve_read": not config["runtime"]["final_reserve_read_allowed"],
    }
    status = "passed_structure_gate" if all(checks.values()) else "failed_structure_gate"
    output = {
        "experiment": config["experiment"],
        "status": status,
        "paper_evidence": False,
        "scientific_metrics_computed": False,
        "threshold_selection_used": False,
        "model_training_performed": False,
        "gpu_used": False,
        "final_reserve_read": False,
        "input_manifest": str(manifest_path.relative_to(project_root)),
        "input_manifest_sha256": _sha256(manifest_path),
        "protocol_sha256": _sha256(protocol_path),
        "selection_rule": input_config["selection_rule"],
        "selected_groups": [str(row["source_group_id"]) for row in selected],
        "selected_generators": [
            str(row[input_config["generator_field"]]) for row in selected
        ],
        "model": model_identity,
        "preprocessing": preprocessing,
        "inference": inference,
        "registration": registration,
        "structure_probes": probe_results,
        "checks": checks,
        "selected_group_count": len(selected),
        "successful_records": len(ok_records),
        "failed_records": len(records) - len(ok_records),
        "score_cache_hits": cache_hits,
        "alignment_cache_hits": alignment_cache_hits,
        "new_inference_latency_ms_total": float(sum(latencies)),
        "wall_time_seconds": float(time.monotonic() - started),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "score_cache_dir": str(score_cache_dir.relative_to(scratch)),
            "alignment_cache_dir": str(alignment_cache_dir.relative_to(scratch)),
            "log": str(log_path.relative_to(project_root)),
        },
    }
    _write_json(summary_path, output)
    if status != "passed_structure_gate" and config["runtime"]["require_all_records"]:
        raise RuntimeError(f"spatial-LPIPS toy gate failed; see {summary_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
