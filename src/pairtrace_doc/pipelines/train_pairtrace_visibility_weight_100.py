from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from pairtrace_doc.pipelines.train_pairtrace_100 import (
    _load_student,
    _native_validation,
    _shuffled_authentic_map,
)
from pairtrace_doc.pipelines.train_pairtrace_trace_aux_100 import _TraceAuxDataset
from pairtrace_doc.pipelines.train_student_100 import (
    _dice_loss,
    _infer_tiled,
    _prepare_pair_cache,
    _ranking_metrics,
    _read_jsonl,
    _resolve,
    _save_checkpoint,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(config_path: Path) -> tuple[dict[str, Any], Path, str]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle)
    base_path = _resolve(project_root, override["base_config"])
    expected = str(override["expected_base_config_sha256"])
    if _sha256(base_path) != expected:
        raise ValueError("PairTrace visibility-weight common config SHA-256 changed")
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    effective_override = {
        key: value
        for key, value in override.items()
        if key not in {"base_config", "expected_base_config_sha256"}
    }
    return _deep_merge(base, effective_override), base_path, expected


def _visibility_weighted_bce(
    logits: torch.Tensor,
    masks: torch.Tensor,
    visibility: torch.Tensor,
    positive_weight: torch.Tensor,
    edited_pixel_weight: float,
) -> torch.Tensor:
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, masks, pos_weight=positive_weight, reduction="none"
    )
    pixel_weight = 1.0 + (edited_pixel_weight - 1.0) * masks * visibility
    return (per_pixel * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config, base_config_path, base_config_sha256 = _load_config(config_path)
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not runtime["gpu_launch_authorized"] or not runtime[
        "visibility_weight_training_authorized"
    ]:
        raise ValueError("PairTrace visibility-weight training was not authorized")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("PairTrace visibility-weight pilot cannot be paper evidence")
    data_config = config["data"]
    if not data_config["training_must_not_read_viewed_diagnostic"]:
        raise ValueError("viewed diagnostic must remain excluded")
    if not data_config["training_must_not_read_final_reserve"]:
        raise ValueError("final reserve must remain excluded")
    target_mode = str(config["experiment"]["target_mode"])
    if target_mode not in {"correct_trace", "shuffled_trace"}:
        raise ValueError(f"unsupported visibility target mode: {target_mode}")
    probability_sum = sum(
        float(config["sampling"][name])
        for name in (
            "forged_positive_probability",
            "forged_random_probability",
            "authentic_random_probability",
        )
    )
    if not math.isclose(probability_sum, 1.0, abs_tol=1e-12):
        raise ValueError("visibility-weight sampling probabilities must sum to one")
    visibility = config["visibility"]
    if not visibility["target_constructed_before_augmentation"]:
        raise ValueError("visibility target timing changed")
    if visibility["difference_reduction"] != "max_abs_rgb":
        raise ValueError("visibility difference reduction changed")
    if not visibility["restrict_weight_to_mask"] or not visibility[
        "normalize_by_weight_sum"
    ]:
        raise ValueError("visibility-weight normalization changed")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PairTrace visibility-weight pilot requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("PairTrace visibility-weight protocol SHA-256 changed")
    manifest_path = _resolve(project_root, data_config["manifest"])
    if _sha256(manifest_path) != data_config["expected_manifest_sha256"]:
        raise ValueError("PairTrace visibility-weight manifest SHA-256 changed")
    student_summary_path = _resolve(project_root, config["matched_student"]["summary"])
    if _sha256(student_summary_path) != config["matched_student"][
        "expected_summary_sha256"
    ]:
        raise ValueError("matched-student summary SHA-256 changed")

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = _read_jsonl(manifest_path)
    train_rows = sorted(
        (row for row in rows if row["pilot_role"] == data_config["train_role"]),
        key=lambda row: str(row["source_group_id"]),
    )
    validation_rows = sorted(
        (
            row
            for row in rows
            if row["pilot_role"] == data_config["validation_role"]
        ),
        key=lambda row: str(row["source_group_id"]),
    )
    expected = int(data_config["expected_per_role"])
    if len(train_rows) != expected or len(validation_rows) != expected:
        raise ValueError("visibility-weight train/validation role counts changed")
    if {row["source_group_id"] for row in train_rows} & {
        row["source_group_id"] for row in validation_rows
    }:
        raise ValueError("visibility-weight train/validation groups overlap")
    if data_config.get("max_train_pairs") is not None:
        train_rows = train_rows[: int(data_config["max_train_pairs"])]
    if data_config.get("max_validation_pairs") is not None:
        validation_rows = validation_rows[: int(data_config["max_validation_pairs"])]

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    pair_cache_dir = _resolve(scratch, paths["pair_cache_dir"])
    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    checkpoint_path = _resolve(project_root, paths["checkpoint"])
    epoch_log_path = _resolve(project_root, paths["epoch_log"])
    predictions_path = _resolve(project_root, paths["predictions"])
    metrics_path = _resolve(project_root, paths["metrics"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (
        pair_cache_dir,
        score_cache_dir,
        checkpoint_path.parent,
        epoch_log_path.parent,
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
    started = time.monotonic()
    cache_hits = 0
    train_cache: list[dict[str, Any]] = []
    validation_cache: list[dict[str, Any]] = []
    for target, selected in ((train_cache, train_rows), (validation_cache, validation_rows)):
        for row in selected:
            record, hit = _prepare_pair_cache(
                row, scratch, pair_cache_dir, config["preprocessing"]
            )
            target.append(record)
            cache_hits += int(hit)
    shuffled_authentic = _shuffled_authentic_map(
        train_cache, int(config["sampling"]["shuffle_seed"])
    )
    weights_path = _resolve(scratch, config["model"]["encoder_weights"])
    weights_sha256 = _sha256(weights_path)
    if weights_sha256 != config["model"]["encoder_weights_sha256"]:
        raise ValueError("visibility-weight encoder weights SHA-256 changed")
    model = _load_student(weights_path).to(device)
    training = config["training"]
    dataset = _TraceAuxDataset(
        train_cache,
        target_mode,
        shuffled_authentic,
        config["sampling"],
        config["preprocessing"],
        {
            "target_scale": visibility["target_scale"],
        },
        seed,
        int(training["steps_per_epoch"]) * int(training["batch_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        persistent_workers=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    positive_weight = torch.tensor(
        float(training["bce_positive_weight"]), device=device
    )
    torch.cuda.reset_peak_memory_stats(device)
    epoch_records: list[dict[str, Any]] = []
    best_ap = -math.inf
    best_epoch = -1
    for epoch in range(int(training["epochs"])):
        dataset.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        weighted_bce_losses: list[float] = []
        dice_losses: list[float] = []
        for images, masks, traces in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            traces = traces.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])
            ):
                logits = model(images)
                weighted_bce = _visibility_weighted_bce(
                    logits,
                    masks,
                    traces,
                    positive_weight,
                    float(visibility["edited_pixel_weight"]),
                )
                dice = _dice_loss(logits, masks)
                loss = float(training["bce_loss_weight"]) * weighted_bce + float(
                    training["dice_loss_weight"]
                ) * dice
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            weighted_bce_losses.append(float(weighted_bce.detach().cpu()))
            dice_losses.append(float(dice.detach().cpu()))
        scheduler.step()
        validation_aps: list[float] = []
        for record in validation_cache:
            image = np.asarray(np.load(record["forged"], mmap_mode="r"))
            mask = np.asarray(np.load(record["mask"], mmap_mode="r")).astype(bool)
            probability = _infer_tiled(
                model, image, device, training, config["preprocessing"]
            )
            average_precision, _ = _ranking_metrics(probability, mask)
            validation_aps.append(average_precision)
        validation_ap = float(np.mean(validation_aps))
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "visibility_weighted_bce_loss": float(np.mean(weighted_bce_losses)),
            "dice_loss": float(np.mean(dice_losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_forged_document_macro_pixel_ap_model_resolution": validation_ap,
            "target_mode": target_mode,
            "paper_evidence": False,
        }
        epoch_records.append(record)
        _write_jsonl(epoch_log_path, epoch_records)
        logging.info("visibility-weight epoch=%d metrics=%s", epoch + 1, record)
        if validation_ap > best_ap:
            best_ap = validation_ap
            best_epoch = epoch + 1
            _save_checkpoint(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "epoch": best_epoch,
                    "validation_macro_pixel_ap_model_resolution": best_ap,
                    "target_mode": target_mode,
                    "config_sha256": _sha256(config_path),
                    "base_config_sha256": base_config_sha256,
                    "protocol_sha256": _sha256(protocol_path),
                    "encoder_weights_sha256": weights_sha256,
                    "seed": seed,
                    "architecture": config["model"]["architecture"],
                },
            )

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state"], strict=True)
    model = model.to(device).eval()
    checkpoint_sha256 = _sha256(checkpoint_path)
    metric_row, prediction_records = _native_validation(
        model,
        validation_cache,
        scratch,
        score_cache_dir,
        checkpoint_sha256,
        device,
        training,
        config["preprocessing"],
        config["operating_point"],
    )
    metric_row["best_epoch"] = best_epoch
    metric_row["target_mode"] = target_mode
    _write_jsonl(predictions_path, prediction_records)
    _write_csv(metrics_path, [metric_row])
    success = config["success"]
    individual_success = bool(
        metric_row["macro_pixel_ap"] >= float(success["native_macro_pixel_ap_min"])
        and metric_row["pixel_iou"] >= float(success["native_pixel_iou_min"])
        and metric_row["authentic_pixel_fpr"]
        <= float(success["authentic_pixel_fpr_max"]) + 1e-12
    )
    summary = {
        "experiment": config["experiment"],
        "status": "passed_individual_thresholds"
        if individual_success
        else "completed_individual_thresholds_not_met",
        "paper_evidence": False,
        "gpu_used": True,
        "target_mode": target_mode,
        "viewed_diagnostic_read": False,
        "final_reserve_read": False,
        "protocol_sha256": _sha256(protocol_path),
        "config_sha256": _sha256(config_path),
        "base_config": str(base_config_path.relative_to(project_root)),
        "base_config_sha256": base_config_sha256,
        "input_manifest_sha256": _sha256(manifest_path),
        "train_pairs": len(train_cache),
        "validation_pairs": len(validation_cache),
        "pair_cache_hits": cache_hits,
        "best_epoch": best_epoch,
        "best_validation_macro_pixel_ap_model_resolution": best_ap,
        "checkpoint": str(checkpoint_path.relative_to(project_root)),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_metrics_native_geometry": metric_row,
        "individual_success": {
            "native_macro_pixel_ap_min": float(success["native_macro_pixel_ap_min"]),
            "native_pixel_iou_min": float(success["native_pixel_iou_min"]),
            "authentic_pixel_fpr_max": float(success["authentic_pixel_fpr_max"]),
            "passed": individual_success,
        },
        "epochs": epoch_records,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "outputs": {
            "epoch_log": str(epoch_log_path.relative_to(project_root)),
            "epoch_log_sha256": _sha256(epoch_log_path),
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
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
