from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
from collections import defaultdict
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

from pairtrace_doc.pipelines.train_student_100 import (
    _prepare_pair_cache,
    _ranking_metrics,
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)
from pairtrace_doc.pipelines.train_tfr_equal_budget_arm import (
    ARMS,
    _build_model,
    _infer_pair_tiled,
    _load_config,
)


CONDITIONS = {
    "clean",
    "reference_jpeg_q85",
    "reference_translation_0_5px",
    "reference_translation_1px",
    "reference_translation_2px",
}


def _transform_reference(image: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return np.array(image, copy=True)
    if condition == "reference_jpeg_q85":
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        with Image.open(buffer) as handle:
            return np.asarray(handle.convert("RGB"))
    translations = {
        "reference_translation_0_5px": 0.5,
        "reference_translation_1px": 1.0,
        "reference_translation_2px": 2.0,
    }
    if condition not in translations:
        raise ValueError(f"unsupported pair-imperfection condition: {condition}")
    offset = translations[condition]
    matrix = np.asarray([[1.0, 0.0, offset], [0.0, 1.0, offset]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _group_macro(values: list[float], groups: list[str]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        grouped[group].append(float(value))
    return float(np.mean([np.mean(items) for items in grouped.values()]))


def _binary_metrics(scores: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = np.asarray(scores) >= threshold
    target = np.asarray(mask, dtype=bool)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    return {
        "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "iou": float(tp / max(1, tp + fp + fn)),
        "recall": float(tp / max(1, tp + fn)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _score_key(
    checkpoint_sha256: str,
    arm: str,
    condition: str,
    sample_kind: str,
    sample_id: str,
    candidate_sha256: str,
    reference_sha256: str,
    training: dict[str, Any],
    preprocessing: dict[str, Any],
) -> str:
    payload = {
        "schema": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "arm": arm,
        "condition": condition,
        "sample_kind": sample_kind,
        "sample_id": sample_id,
        "candidate_sha256": candidate_sha256,
        "reference_sha256": reference_sha256,
        "training": training,
        "preprocessing": preprocessing,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cached_infer(
    model: torch.nn.Module,
    candidate: np.ndarray,
    reference: np.ndarray,
    arm: str,
    condition: str,
    sample_kind: str,
    sample_id: str,
    candidate_sha256: str,
    reference_sha256: str,
    checkpoint_sha256: str,
    cache_dir: Path,
    device: torch.device,
    training: dict[str, Any],
    preprocessing: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    key = _score_key(
        checkpoint_sha256,
        arm,
        condition,
        sample_kind,
        sample_id,
        candidate_sha256,
        reference_sha256,
        training,
        preprocessing,
    )
    path = cache_dir / arm / condition / f"{key}.npy"
    if path.is_file():
        return np.asarray(np.load(path, mmap_mode="r")), True
    transformed = _transform_reference(reference, condition)
    scores = _infer_pair_tiled(
        model, candidate, transformed, arm, device, training, preprocessing
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, scores, allow_pickle=False)
    temporary.replace(path)
    return scores, False


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = _load_config(config_path)
    experiment = config["experiment"]
    if experiment["paper_evidence"]:
        raise ValueError("pair-imperfection diagnostic cannot be paper evidence")
    protocol_path = _resolve(project_root, experiment["protocol"])
    if _sha256(protocol_path) != experiment["expected_protocol_sha256"]:
        raise ValueError("pair-imperfection protocol SHA-256 changed")
    conditions = [str(value) for value in config["conditions"]]
    if set(conditions) != CONDITIONS or len(conditions) != len(CONDITIONS):
        raise ValueError("pair-imperfection condition set changed")
    runtime = config["runtime"]
    if not runtime["gpu_launch_authorized"] or not runtime["training_disabled"]:
        raise ValueError("diagnostic requires explicit GPU authorization with training disabled")
    if not runtime["must_not_read_holdout"]:
        raise ValueError("diagnostic must keep the holdout unread")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("pair-imperfection diagnostic requires CUDA")
    torch.cuda.set_device(device)

    data = config["data"]
    manifest_path = _resolve(project_root, data["manifest"])
    if _sha256(manifest_path) != data["expected_manifest_sha256"]:
        raise ValueError("diagnostic manifest SHA-256 changed")
    rows = _read_jsonl(manifest_path)
    validation_rows = sorted(
        [row for row in rows if row["pilot_role"] == "validation"],
        key=lambda row: (str(row["source_group_id"]), str(row["sample_id"])),
    )
    if len(validation_rows) != int(data["expected_validation_records"]):
        raise ValueError("diagnostic validation record count changed")
    if data.get("max_validation_pairs") is not None:
        validation_rows = validation_rows[: int(data["max_validation_pairs"])]

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    pair_cache_dir = _resolve(scratch, paths["pair_cache_dir"])
    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    predictions_path = _resolve(project_root, paths["prediction_records"])
    metrics_path = _resolve(project_root, paths["metrics"])
    summary_path = _resolve(project_root, paths["summary"])
    validation_cache: list[dict[str, Any]] = []
    pair_cache_hits = 0
    for row in validation_rows:
        record, hit = _prepare_pair_cache(
            row, scratch, pair_cache_dir, config["preprocessing"]
        )
        validation_cache.append(record)
        pair_cache_hits += int(hit)

    weights_path = _resolve(scratch, config["model"]["encoder_weights"])
    if _sha256(weights_path) != config["model"]["encoder_weights_sha256"]:
        raise ValueError("diagnostic encoder initialization SHA-256 changed")
    encoder_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    score_cache_hits = 0
    clean_replay_differences: list[float] = []

    for arm in sorted(ARMS):
        arm_config = config["arms"][arm]
        arm_summary_path = _resolve(project_root, arm_config["summary"])
        checkpoint_path = _resolve(project_root, arm_config["checkpoint"])
        original_predictions_path = _resolve(project_root, arm_config["predictions"])
        with arm_summary_path.open("r", encoding="utf-8") as handle:
            arm_summary = json.load(handle)
        if arm_summary["status"] != "development_arm_complete":
            raise ValueError(f"diagnostic arm {arm} was not completed")
        checkpoint_sha256 = _sha256(checkpoint_path)
        if checkpoint_sha256 != arm_summary["checkpoint_sha256"]:
            raise ValueError(f"diagnostic arm {arm} checkpoint SHA-256 changed")
        if _sha256(original_predictions_path) != arm_summary["prediction_records_sha256"]:
            raise ValueError(f"diagnostic arm {arm} original predictions changed")
        original_predictions = {
            str(row["sample_id"]): row
            for row in _read_jsonl(original_predictions_path)
            if row["sample_kind"] == "forged_pair"
        }
        model = _build_model(arm, encoder_state)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state"], strict=True)
        model = model.to(device).eval()
        threshold = float(arm_summary["operating_point"]["threshold"])

        for condition in conditions:
            forged_ap: list[float] = []
            forged_f1: list[float] = []
            forged_iou: list[float] = []
            forged_recall: list[float] = []
            forged_groups: list[str] = []
            for record in validation_cache:
                source = record["source_row"]
                candidate = np.asarray(np.load(record["forged"], mmap_mode="r"))
                reference = np.asarray(np.load(record["authentic"], mmap_mode="r"))
                mask = np.asarray(np.load(record["mask"], mmap_mode="r"), dtype=bool)
                scores, hit = _cached_infer(
                    model,
                    candidate,
                    reference,
                    arm,
                    condition,
                    "forged_pair",
                    str(record["sample_id"]),
                    str(source["image_sha256"]),
                    str(source["authentic_sha256"]),
                    checkpoint_sha256,
                    score_cache_dir,
                    device,
                    config["inference"],
                    config["preprocessing"],
                )
                score_cache_hits += int(hit)
                ap, auroc = _ranking_metrics(scores, mask)
                binary = _binary_metrics(scores, mask, threshold)
                group = str(record["source_group_id"])
                forged_ap.append(ap)
                forged_f1.append(binary["f1"])
                forged_iou.append(binary["iou"])
                forged_recall.append(binary["recall"])
                forged_groups.append(group)
                if condition == "clean":
                    clean_replay_differences.append(
                        abs(ap - float(original_predictions[str(record["sample_id"])]["average_precision"]))
                    )
                prediction_rows.append(
                    {
                        "arm": arm,
                        "condition": condition,
                        "sample_kind": "forged_pair",
                        "sample_id": record["sample_id"],
                        "source_group_id": group,
                        "average_precision": ap,
                        "auroc": auroc,
                        **binary,
                        "threshold_frozen_on_clean_validation": threshold,
                        "paper_evidence": False,
                    }
                )

            authentic_fprs: list[float] = []
            representatives: dict[str, dict[str, Any]] = {}
            for record in validation_cache:
                representatives.setdefault(str(record["source_group_id"]), record)
            for group, record in representatives.items():
                source = record["source_row"]
                authentic = np.asarray(np.load(record["authentic"], mmap_mode="r"))
                scores, hit = _cached_infer(
                    model,
                    authentic,
                    authentic,
                    arm,
                    condition,
                    "authentic_pair",
                    f"{group}:authentic",
                    str(source["authentic_sha256"]),
                    str(source["authentic_sha256"]),
                    checkpoint_sha256,
                    score_cache_dir,
                    device,
                    config["inference"],
                    config["preprocessing"],
                )
                score_cache_hits += int(hit)
                fpr = float(np.mean(scores >= threshold))
                authentic_fprs.append(fpr)
                prediction_rows.append(
                    {
                        "arm": arm,
                        "condition": condition,
                        "sample_kind": "authentic_pair",
                        "sample_id": f"{group}:authentic",
                        "source_group_id": group,
                        "fpr": fpr,
                        "threshold_frozen_on_clean_validation": threshold,
                        "paper_evidence": False,
                    }
                )
            metric_rows.append(
                {
                    "arm": arm,
                    "condition": condition,
                    "source_group_macro_pixel_ap": _group_macro(forged_ap, forged_groups),
                    "source_group_macro_pixel_f1": _group_macro(forged_f1, forged_groups),
                    "source_group_macro_pixel_iou": _group_macro(forged_iou, forged_groups),
                    "source_group_macro_pixel_recall": _group_macro(forged_recall, forged_groups),
                    "unique_authentic_group_macro_pixel_fpr": float(np.mean(authentic_fprs)),
                    "threshold_frozen_on_clean_validation": threshold,
                    "forged_pairs": len(forged_ap),
                    "authentic_groups": len(authentic_fprs),
                    "paper_evidence": False,
                }
            )
        del model
        torch.cuda.empty_cache()

    clean_by_arm = {
        row["arm"]: row for row in metric_rows if row["condition"] == "clean"
    }
    for row in metric_rows:
        clean = clean_by_arm[row["arm"]]
        for metric in (
            "source_group_macro_pixel_ap",
            "source_group_macro_pixel_f1",
            "source_group_macro_pixel_iou",
            "source_group_macro_pixel_recall",
            "unique_authentic_group_macro_pixel_fpr",
        ):
            row[f"delta_from_clean_{metric}"] = float(row[metric]) - float(clean[metric])
    max_replay_difference = max(clean_replay_differences, default=math.inf)
    if max_replay_difference > float(config["gate"]["max_clean_ap_replay_difference"]):
        raise RuntimeError("clean-condition AP did not replay the frozen predictions")
    _write_jsonl(predictions_path, prediction_rows)
    _write_csv(metrics_path, metric_rows)
    summary = {
        "status": "preflight_complete"
        if experiment["stage"] == "preflight"
        else "exploratory_development_diagnostic_complete",
        "stage": experiment["stage"],
        "paper_evidence": False,
        "holdout_read": False,
        "training_performed": False,
        "arms": sorted(ARMS),
        "conditions": conditions,
        "validation_pairs": len(validation_cache),
        "validation_source_groups": len(
            {str(record["source_group_id"]) for record in validation_cache}
        ),
        "pair_cache_hits": pair_cache_hits,
        "score_cache_hits": score_cache_hits,
        "clean_ap_max_absolute_replay_difference": max_replay_difference,
        "failed_items": 0,
        "protocol_sha256": _sha256(protocol_path),
        "prediction_records_sha256": _sha256(predictions_path),
        "metrics_sha256": _sha256(metrics_path),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "wall_time_seconds": float(time.monotonic() - started),
        "metrics": metric_rows,
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen TFR pair-imperfection stress")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
