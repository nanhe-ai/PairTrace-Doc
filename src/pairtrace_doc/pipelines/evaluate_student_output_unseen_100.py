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
from typing import Any, Iterable

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

from pairtrace_doc.pipelines.train_student_100 import (
    ResNet18UNet,
    _infer_tiled,
    _ranking_metrics,
)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty output-unseen metric table")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resize_image(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image
    scale = max_side / max(height, width)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def _threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = scores >= threshold
    truth = labels.astype(bool)
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
    }


def _mean_ci(values: np.ndarray, seed: int, resamples: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("bootstrap requires at least one value")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def _training_view(preprocessing: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    training = {
        "validation_tile_size": preprocessing["validation_tile_size"],
        "validation_tile_stride": preprocessing["validation_tile_stride"],
        "validation_tile_batch_size": preprocessing["validation_tile_batch_size"],
        "amp": preprocessing["amp"],
    }
    normalization = {
        "imagenet_mean": preprocessing["imagenet_mean"],
        "imagenet_std": preprocessing["imagenet_std"],
    }
    return training, normalization


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not runtime["one_shot_evaluation_authorized"] or not runtime["gpu_launch_authorized"]:
        raise ValueError("output-unseen student evaluation was not explicitly authorized")
    if runtime["pairtrace_training_authorized"]:
        raise ValueError("student evaluation must not authorize PairTrace training")
    operating = config["operating_point"]
    if operating["checkpoint_selection_allowed"] or operating["threshold_selection_allowed"]:
        raise ValueError("output-unseen evaluation cannot select a checkpoint or threshold")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("current output-unseen pilot must remain non-paper evidence")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("output-unseen student evaluation requires CUDA")

    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    input_config = config["input"]
    manifest_path = _resolve(project_root, input_config["manifest"])
    if _sha256(manifest_path) != input_config["expected_manifest_sha256"]:
        raise ValueError("output-unseen manifest SHA-256 changed")
    rows = _read_jsonl(manifest_path)
    counts = Counter(str(row["evaluation_role"]) for row in rows)
    if dict(counts) != {str(key): int(value) for key, value in input_config["expected_counts"].items()}:
        raise ValueError(f"output-unseen role counts changed: {dict(counts)}")
    freeze_ids = {str(row["holdout_freeze_id"]) for row in rows}
    if freeze_ids != {input_config["expected_freeze_id"]}:
        raise ValueError("output-unseen freeze ID changed")
    if any(row["model_or_threshold_selection_allowed"] for row in rows):
        raise ValueError("holdout row unexpectedly allows model or threshold selection")

    model_config = config["model"]
    checkpoint_path = _resolve(project_root, model_config["checkpoint"])
    training_config_path = _resolve(project_root, model_config["training_config"])
    validation_metrics_path = _resolve(project_root, model_config["validation_metrics"])
    for path, expected, label in (
        (checkpoint_path, model_config["checkpoint_sha256"], "checkpoint"),
        (training_config_path, model_config["training_config_sha256"], "training config"),
        (validation_metrics_path, model_config["validation_metrics_sha256"], "validation metrics"),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen student {label} SHA-256 changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != model_config["architecture"]:
        raise ValueError("student checkpoint architecture changed")
    model = ResNet18UNet()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()

    paths = config["paths"]
    scratch = Path(
        os.environ.get(paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"])))
    ).resolve()
    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    predictions_path = _resolve(project_root, paths["predictions"])
    metrics_path = _resolve(project_root, paths["metrics"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (score_cache_dir, predictions_path.parent, metrics_path.parent, summary_path.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    started = time.monotonic()
    threshold = float(operating["pixel_threshold"])
    inference_training, normalization = _training_view(config["preprocessing"])
    torch.cuda.reset_peak_memory_stats(device)
    prediction_rows: list[dict[str, Any]] = []
    failures = 0
    cache_hits = 0
    role_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    role_group_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        prediction: dict[str, Any] = {
            "record_id": row["record_id"],
            "source_group_id": row["source_group_id"],
            "evaluation_role": row["evaluation_role"],
            "sample_kind": row["sample_kind"],
            "status": "failed",
            "paper_evidence": False,
            "checkpoint_sha256": model_config["checkpoint_sha256"],
            "pixel_threshold": threshold,
            "model_or_threshold_selection_used": False,
        }
        try:
            image_path = _resolve(scratch, row["image"])
            if _sha256(image_path) != row["image_sha256"]:
                raise ValueError("holdout image SHA-256 changed")
            with Image.open(image_path) as handle:
                native_image = np.asarray(handle.convert("RGB"))
            if native_image.shape[:2] != (int(row["height"]), int(row["width"])):
                raise ValueError("holdout image geometry changed")
            model_image = _resize_image(native_image, int(config["preprocessing"]["max_side"]))
            score_key = hashlib.sha256(
                json.dumps(
                    {
                        "checkpoint_sha256": model_config["checkpoint_sha256"],
                        "input_sha256": row["image_sha256"],
                        "preprocessing": config["preprocessing"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            score_path = score_cache_dir / f"{score_key}.npz"
            if score_path.is_file():
                with np.load(score_path, allow_pickle=False) as cached:
                    probability = cached["scores"].astype(np.float32)
                cache_hits += 1
            else:
                probability = _infer_tiled(
                    model, model_image, device, inference_training, normalization
                )
                temporary = score_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, scores=probability.astype(np.float16))
                temporary.replace(score_path)
            if probability.shape != model_image.shape[:2] or not np.isfinite(probability).all():
                raise ValueError("student holdout score cache is invalid")
            native_probability = cv2.resize(
                probability,
                (native_image.shape[1], native_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            if row["sample_kind"] == "forged":
                mask_path = _resolve(scratch, row["mask"])
                if _sha256(mask_path) != row["mask_sha256"]:
                    raise ValueError("holdout mask SHA-256 changed")
                with Image.open(mask_path) as handle:
                    mask = np.asarray(handle.convert("L")) > 0
                if mask.shape != native_probability.shape:
                    raise ValueError("holdout mask geometry changed")
                average_precision, auroc = _ranking_metrics(native_probability, mask)
                document_metrics = {
                    "macro_pixel_ap": average_precision,
                    "pixel_auroc": auroc,
                    **_threshold_metrics(native_probability, mask, threshold),
                }
            else:
                document_metrics = {
                    "authentic_pixel_fpr": float(np.mean(native_probability >= threshold))
                }
            role = str(row["evaluation_role"])
            role_metrics[role].append(document_metrics)
            role_group_metrics[role][str(row["source_group_id"])] = document_metrics
            prediction.update(document_metrics)
            prediction.update(
                {
                    "status": "ok",
                    "score_cache": str(score_path.relative_to(scratch)),
                    "score_shape": list(probability.shape),
                    "native_shape": list(native_probability.shape),
                }
            )
        except Exception as error:  # every failed item is persisted before aborting
            failures += 1
            prediction["failure_type"] = type(error).__name__
            prediction["failure_reason"] = str(error)
            logging.exception("record_id=%s failed", row["record_id"])
        prediction_rows.append(prediction)
        _write_jsonl(predictions_path, prediction_rows)
    if failures and runtime["require_all_records"]:
        raise RuntimeError(f"output-unseen evaluation failed for {failures}/{len(rows)} records")

    evaluation = config["evaluation"]
    resamples = int(evaluation["bootstrap_resamples"])
    metric_rows: list[dict[str, Any]] = []
    for role_index, role in enumerate(evaluation["forged_roles"]):
        documents = role_metrics[role]
        row: dict[str, Any] = {
            "evaluation_role": role,
            "sample_kind": "forged",
            "documents": len(documents),
            "pixel_threshold": threshold,
            "paper_evidence": False,
        }
        for metric_index, metric in enumerate(
            ("macro_pixel_ap", "pixel_auroc", "pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou")
        ):
            mean, low, high = _mean_ci(
                np.asarray([item[metric] for item in documents]),
                seed + role_index * 100 + metric_index,
                resamples,
            )
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        metric_rows.append(row)
    authentic_role = str(evaluation["authentic_role"])
    authentic_values = np.asarray(
        [item["authentic_pixel_fpr"] for item in role_metrics[authentic_role]]
    )
    mean, low, high = _mean_ci(authentic_values, seed + 999, resamples)
    metric_rows.append(
        {
            "evaluation_role": authentic_role,
            "sample_kind": "authentic",
            "documents": len(authentic_values),
            "pixel_threshold": threshold,
            "authentic_pixel_fpr": mean,
            "authentic_pixel_fpr_ci_low": low,
            "authentic_pixel_fpr_ci_high": high,
            "paper_evidence": False,
        }
    )
    first_role, second_role = [str(value) for value in evaluation["forged_roles"]]
    groups = sorted(set(role_group_metrics[first_role]) & set(role_group_metrics[second_role]))
    if len(groups) != int(input_config["expected_counts"][first_role]):
        raise ValueError("paired output-unseen forged source groups changed")
    paired_row: dict[str, Any] = {
        "evaluation_role": f"paired_delta:{second_role}-minus-{first_role}",
        "sample_kind": "forged_paired_delta",
        "documents": len(groups),
        "pixel_threshold": threshold,
        "paper_evidence": False,
    }
    for metric_index, metric in enumerate(("macro_pixel_ap", "pixel_f1", "pixel_iou")):
        differences = np.asarray(
            [role_group_metrics[second_role][group][metric] - role_group_metrics[first_role][group][metric] for group in groups]
        )
        mean, low, high = _mean_ci(differences, seed + 2000 + metric_index, resamples)
        paired_row[metric] = mean
        paired_row[f"{metric}_ci_low"] = low
        paired_row[f"{metric}_ci_high"] = high
    metric_rows.append(paired_row)
    _write_csv(metrics_path, metric_rows)
    summary = {
        "experiment": config["experiment"],
        "status": "passed",
        "paper_evidence": False,
        "one_shot_evaluation": True,
        "checkpoint_selection_used": False,
        "threshold_selection_used": False,
        "pairtrace_training_authorized": False,
        "manifest_sha256": _sha256(manifest_path),
        "holdout_freeze_id": input_config["expected_freeze_id"],
        "checkpoint_sha256": model_config["checkpoint_sha256"],
        "training_config_sha256": model_config["training_config_sha256"],
        "validation_metrics_sha256": model_config["validation_metrics_sha256"],
        "pixel_threshold": threshold,
        "selected_records": len(rows),
        "successful_records": len(rows) - failures,
        "failed_records": failures,
        "score_cache_hits": cache_hits,
        "bootstrap_resamples": resamples,
        "metric_rows": metric_rows,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
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
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
