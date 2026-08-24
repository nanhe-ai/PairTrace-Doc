from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _invalid_thread_value = int(os.environ.get(_thread_variable, "1")) < 1
    except ValueError:
        _invalid_thread_value = True
    if _invalid_thread_value:
        os.environ[_thread_variable] = "1"

import numpy as np
import torch
import yaml

from pairtrace_doc.pipelines.train_pairtrace_100 import (
    TraceUNet,
    _load_teacher,
    _shuffled_authentic_map,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _positions,
    _prepare_pair_cache,
    _ranking_metrics,
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)


def _mean_ci(
    values: np.ndarray, seed: int, resamples: int
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def _paired_teacher_input(
    forged: np.ndarray,
    authentic: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    forged_float = forged.astype(np.float32) / 255.0
    authentic_float = authentic.astype(np.float32) / 255.0
    forged_normalized = (forged_float - mean) / std
    authentic_normalized = (authentic_float - mean) / std
    return np.concatenate(
        [
            forged_normalized.transpose(0, 3, 1, 2),
            authentic_normalized.transpose(0, 3, 1, 2),
            (forged_normalized - authentic_normalized).transpose(0, 3, 1, 2),
        ],
        axis=1,
    )


def _infer_pair_tiled(
    model: TraceUNet,
    forged: np.ndarray,
    authentic: np.ndarray,
    device: torch.device,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    if forged.shape != authentic.shape:
        raise ValueError("teacher diagnostic pair inputs are not aligned")
    tile = int(preprocessing["tile_size"])
    stride = int(preprocessing["tile_stride"])
    batch_size = int(preprocessing["tile_batch_size"])
    height, width = forged.shape[:2]
    pad_height = max(0, tile - height)
    pad_width = max(0, tile - width)
    padding = ((0, pad_height), (0, pad_width), (0, 0))
    forged_padded = np.pad(forged, padding, mode="reflect")
    authentic_padded = np.pad(authentic, padding, mode="reflect")
    padded_height, padded_width = forged_padded.shape[:2]
    coordinates = [
        (top, left)
        for top in _positions(padded_height, tile, stride)
        for left in _positions(padded_width, tile, stride)
    ]
    accumulator = np.zeros((padded_height, padded_width), dtype=np.float32)
    counts = np.zeros_like(accumulator)
    mean = np.asarray(preprocessing["imagenet_mean"], dtype=np.float32)
    std = np.asarray(preprocessing["imagenet_std"], dtype=np.float32)
    model.eval()
    for start in range(0, len(coordinates), batch_size):
        selected = coordinates[start : start + batch_size]
        forged_batch = np.stack(
            [
                forged_padded[top : top + tile, left : left + tile]
                for top, left in selected
            ]
        )
        authentic_batch = np.stack(
            [
                authentic_padded[top : top + tile, left : left + tile]
                for top, left in selected
            ]
        )
        tensor = torch.from_numpy(
            _paired_teacher_input(forged_batch, authentic_batch, mean, std).copy()
        ).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(preprocessing["amp"]),
        ):
            probabilities = torch.sigmoid(model(tensor)).squeeze(1).float().cpu().numpy()
        for probability, (top, left) in zip(probabilities, selected):
            accumulator[top : top + tile, left : left + tile] += probability
            counts[top : top + tile, left : left + tile] += 1.0
    return (accumulator / np.maximum(counts, 1.0))[:height, :width]


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not runtime["gpu_launch_authorized"] or runtime["training_authorized"]:
        raise ValueError("teacher diagnostic authorization changed")
    if runtime["checkpoint_selection_allowed"] or runtime["threshold_selection_allowed"]:
        raise ValueError("teacher diagnostic cannot select checkpoints or thresholds")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("teacher diagnostic cannot be paper evidence")
    if config["data"]["final_reserve_read_allowed"]:
        raise ValueError("teacher diagnostic cannot read the final reserve")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("teacher diagnostic requires CUDA")
    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("overnight iteration protocol SHA-256 changed")
    manifest_path = _resolve(project_root, config["data"]["manifest"])
    if _sha256(manifest_path) != config["data"]["expected_manifest_sha256"]:
        raise ValueError("teacher diagnostic manifest SHA-256 changed")
    rows = _read_jsonl(manifest_path)
    validation_rows = sorted(
        (
            row
            for row in rows
            if row["pilot_role"] == config["data"]["validation_role"]
        ),
        key=lambda row: str(row["source_group_id"]),
    )
    if len(validation_rows) != int(config["data"]["expected_validation_pairs"]):
        raise ValueError("teacher diagnostic validation count changed")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    pair_cache_dir = _resolve(scratch, paths["pair_cache_dir"])
    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    predictions_path = _resolve(project_root, paths["predictions"])
    metrics_path = _resolve(project_root, paths["metrics"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (
        pair_cache_dir,
        score_cache_dir,
        predictions_path.parent,
        metrics_path.parent,
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
    validation_cache: list[dict[str, Any]] = []
    cache_hits = 0
    cache_preprocessing = {
        "cache_schema_version": 1,
        "max_side": int(config["preprocessing"]["max_side"]),
        "crop_size": int(config["preprocessing"]["tile_size"]),
        "imagenet_mean": config["preprocessing"]["imagenet_mean"],
        "imagenet_std": config["preprocessing"]["imagenet_std"],
    }
    for row in validation_rows:
        record, hit = _prepare_pair_cache(
            row, scratch, pair_cache_dir, cache_preprocessing
        )
        validation_cache.append(record)
        cache_hits += int(hit)
    shuffled = _shuffled_authentic_map(
        validation_cache, int(config["interventions"]["shuffle_seed"])
    )

    model_config = config["models"]
    weights_path = _resolve(scratch, model_config["encoder_weights"])
    if _sha256(weights_path) != model_config["encoder_weights_sha256"]:
        raise ValueError("teacher diagnostic encoder weights changed")
    models: dict[str, tuple[TraceUNet, str]] = {}
    for model_name in ("correct_teacher", "shuffled_teacher"):
        item = model_config[model_name]
        checkpoint_path = _resolve(project_root, item["checkpoint"])
        if _sha256(checkpoint_path) != item["checkpoint_sha256"]:
            raise ValueError(f"{model_name} checkpoint SHA-256 changed")
        model = _load_teacher(
            weights_path, model_config["teacher_conv1_coefficients"]
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        models[model_name] = (model.to(device).eval(), item["checkpoint_sha256"])

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    prediction_rows: list[dict[str, Any]] = []
    values: dict[tuple[str, str], list[float]] = {}
    failures = 0
    conditions = list(config["interventions"]["conditions"])
    for model_name, (model, checkpoint_sha256) in models.items():
        for condition in conditions:
            key = (model_name, condition)
            values[key] = []
            for record in validation_cache:
                prediction: dict[str, Any] = {
                    "record_id": f"{model_name}:{condition}:{record['source_group_id']}",
                    "source_group_id": record["source_group_id"],
                    "model": model_name,
                    "condition": condition,
                    "status": "failed",
                    "paper_evidence": False,
                    "checkpoint_sha256": checkpoint_sha256,
                }
                try:
                    forged = np.asarray(np.load(record["forged"], mmap_mode="r"))
                    if condition == "aligned_authentic":
                        authentic = np.asarray(
                            np.load(record["authentic"], mmap_mode="r")
                        )
                    elif condition == "shuffled_authentic":
                        authentic = np.asarray(
                            np.load(
                                shuffled[str(record["source_group_id"])],
                                mmap_mode="r",
                            )
                        )
                        if authentic.shape != forged.shape:
                            import cv2

                            authentic = cv2.resize(
                                authentic,
                                (forged.shape[1], forged.shape[0]),
                                interpolation=cv2.INTER_AREA,
                            )
                    elif condition == "identical_forged":
                        authentic = forged
                    else:
                        raise ValueError(f"unsupported diagnostic condition: {condition}")
                    score_key = hashlib.sha256(
                        json.dumps(
                            {
                                "checkpoint_sha256": checkpoint_sha256,
                                "source_group_id": record["source_group_id"],
                                "condition": condition,
                                "preprocessing": config["preprocessing"],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    score_path = score_cache_dir / f"{score_key}.npz"
                    probability = _infer_pair_tiled(
                        model,
                        forged,
                        authentic,
                        device,
                        config["preprocessing"],
                    )
                    temporary = score_path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle, scores=probability.astype(np.float16)
                        )
                    temporary.replace(score_path)
                    mask = np.asarray(np.load(record["mask"], mmap_mode="r")).astype(
                        bool
                    )
                    average_precision, auroc = _ranking_metrics(probability, mask)
                    values[key].append(average_precision)
                    prediction.update(
                        {
                            "status": "ok",
                            "macro_pixel_ap": average_precision,
                            "pixel_auroc": auroc,
                            "score_cache": str(score_path.relative_to(scratch)),
                            "score_shape": list(probability.shape),
                        }
                    )
                except Exception as error:  # persist every diagnostic failure
                    failures += 1
                    prediction["failure_type"] = type(error).__name__
                    prediction["failure_reason"] = str(error)
                    logging.exception("teacher diagnostic record failed")
                prediction_rows.append(prediction)
                _write_jsonl(predictions_path, prediction_rows)
    if failures and runtime["require_all_records"]:
        raise RuntimeError(f"teacher diagnostic failed for {failures} records")

    bootstrap = config["uncertainty"]
    metric_rows: list[dict[str, Any]] = []
    for index, (key, document_values) in enumerate(sorted(values.items())):
        mean, low, high = _mean_ci(
            np.asarray(document_values),
            int(bootstrap["bootstrap_seed"]) + index,
            int(bootstrap["bootstrap_resamples"]),
        )
        metric_rows.append(
            {
                "model": key[0],
                "condition": key[1],
                "validation_pairs": len(document_values),
                "macro_pixel_ap": mean,
                "macro_pixel_ap_ci_low": low,
                "macro_pixel_ap_ci_high": high,
                "paper_evidence": False,
            }
        )
    metric_lookup = {
        (row["model"], row["condition"]): row["macro_pixel_ap"]
        for row in metric_rows
    }
    aligned = metric_lookup[("correct_teacher", "aligned_authentic")]
    shuffled_input = metric_lookup[("correct_teacher", "shuffled_authentic")]
    identical_input = metric_lookup[("correct_teacher", "identical_forged")]
    shuffled_teacher_aligned = metric_lookup[
        ("shuffled_teacher", "aligned_authentic")
    ]
    decision_config = config["decision"]
    decision = {
        "correct_aligned_minus_shuffled_input_ap": aligned - shuffled_input,
        "correct_aligned_minus_identical_input_ap": aligned - identical_input,
        "correct_teacher_minus_shuffled_teacher_aligned_ap": aligned
        - shuffled_teacher_aligned,
    }
    checks = {
        "aligned_minus_shuffled_input_pass": bool(
            decision["correct_aligned_minus_shuffled_input_ap"]
            >= float(
                decision_config["correct_aligned_minus_shuffled_input_ap_min"]
            )
        ),
        "aligned_minus_identical_input_pass": bool(
            decision["correct_aligned_minus_identical_input_ap"]
            >= float(
                decision_config["correct_aligned_minus_identical_input_ap_min"]
            )
        ),
        "correct_minus_shuffled_teacher_pass": bool(
            decision["correct_teacher_minus_shuffled_teacher_aligned_ap"]
            >= float(
                decision_config[
                    "correct_teacher_minus_shuffled_teacher_aligned_ap_min"
                ]
            )
        ),
    }
    pair_dependent = all(checks.values())
    _write_csv(metrics_path, metric_rows)
    summary = {
        "experiment": config["experiment"],
        "status": "pair_dependent" if pair_dependent else "pair_dependence_not_established",
        "paper_evidence": False,
        "training_performed": False,
        "checkpoint_selection_used": False,
        "threshold_selection_used": False,
        "final_reserve_read": False,
        "selected_validation_pairs": len(validation_cache),
        "expected_prediction_records": len(validation_cache)
        * len(models)
        * len(conditions),
        "successful_prediction_records": len(prediction_rows) - failures,
        "failed_prediction_records": failures,
        "pair_cache_hits": cache_hits,
        "metrics": metric_rows,
        "decision_values": decision,
        "decision_checks": checks,
        "pair_dependent": pair_dependent,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "log": str(log_path.relative_to(project_root)),
        },
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
