from __future__ import annotations

import argparse
import io
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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pairtrace_doc.pipelines.train_pairtrace_100 import (
    TraceUNet,
    _native_validation,
    _shuffled_authentic_map,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _ConvBlock,
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
        raise ValueError("PairTrace trace-aux common config SHA-256 changed")
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    effective_override = {
        key: value
        for key, value in override.items()
        if key not in {"base_config", "expected_base_config_sha256"}
    }
    return _deep_merge(base, effective_override), base_path, expected


class TraceAuxUNet(TraceUNet):
    def __init__(self, auxiliary_channels: int = 32) -> None:
        super().__init__()
        self.trace_output = nn.Sequential(
            _ConvBlock(64, auxiliary_channels),
            nn.Conv2d(auxiliary_channels, 1, 1),
        )

    def forward_with_aux(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        segmentation, features = super().forward_with_features(value)
        trace_feature = F.interpolate(
            features["decoder1"],
            size=value.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return segmentation, self.trace_output(trace_feature)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        segmentation, _ = self.forward_with_aux(value)
        return segmentation


def _load_model(weights_path: Path, auxiliary_channels: int) -> TraceAuxUNet:
    model = TraceAuxUNet(auxiliary_channels=auxiliary_channels)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(state, strict=True)
    return model


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return np.asarray(handle.convert("RGB"))


def _pad(
    image: np.ndarray,
    mask: np.ndarray,
    trace: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    pad_height = max(0, crop_size - height)
    pad_width = max(0, crop_size - width)
    if pad_height or pad_width:
        image = np.pad(
            image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="reflect",
        )
        mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode="constant")
        trace = np.pad(trace, ((0, pad_height), (0, pad_width)), mode="constant")
    return image, mask, trace


class _TraceAuxDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        records: list[dict[str, Any]],
        target_mode: str,
        shuffled_authentic: dict[str, str],
        sampling: dict[str, Any],
        preprocessing: dict[str, Any],
        auxiliary: dict[str, Any],
        seed: int,
        length: int,
    ) -> None:
        if target_mode not in {"correct_trace", "shuffled_trace", "zero_trace"}:
            raise ValueError(f"unsupported auxiliary trace mode: {target_mode}")
        self.records = records
        self.target_mode = target_mode
        self.shuffled_authentic = shuffled_authentic
        self.sampling = sampling
        self.preprocessing = preprocessing
        self.auxiliary = auxiliary
        self.seed = seed
        self.length = length
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def _trace_target(
        self, record: dict[str, Any], forged: np.ndarray
    ) -> np.ndarray:
        if self.target_mode == "zero_trace":
            return np.zeros(forged.shape[:2], dtype=np.float32)
        if self.target_mode == "correct_trace":
            authentic_path = record["authentic"]
        else:
            authentic_path = self.shuffled_authentic[str(record["source_group_id"])]
        authentic = np.asarray(np.load(authentic_path, mmap_mode="r"))
        if authentic.shape != forged.shape:
            authentic = cv2.resize(
                authentic,
                (forged.shape[1], forged.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        difference = np.abs(
            forged.astype(np.int16, copy=False)
            - authentic.astype(np.int16, copy=False)
        ).max(axis=2)
        return difference.astype(np.float32) / float(self.auxiliary["target_scale"])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # This stream intentionally matches the frozen single-image student.
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        record = self.records[int(rng.integers(0, len(self.records)))]
        probability = float(rng.random())
        positive_limit = float(self.sampling["forged_positive_probability"])
        forged_limit = positive_limit + float(
            self.sampling["forged_random_probability"]
        )
        stored_mask = np.asarray(np.load(record["mask"], mmap_mode="r"))
        if probability < forged_limit:
            image = np.asarray(np.load(record["forged"], mmap_mode="r"))
            mask = stored_mask
            trace = self._trace_target(record, image)
            positive_crop = probability < positive_limit
        else:
            image = np.asarray(np.load(record["authentic"], mmap_mode="r"))
            mask = np.zeros(stored_mask.shape, dtype=np.uint8)
            trace = np.zeros(stored_mask.shape, dtype=np.float32)
            positive_crop = False

        crop_size = int(self.preprocessing["crop_size"])
        image, mask, trace = _pad(image, mask, trace, crop_size)
        height, width = mask.shape
        if positive_crop:
            x1, y1, x2, y2 = record["bbox_xyxy"]
            center_x = int(rng.integers(x1, max(x1 + 1, x2)))
            center_y = int(rng.integers(y1, max(y1 + 1, y2)))
            left = int(
                np.clip(
                    center_x - int(rng.integers(0, crop_size)),
                    0,
                    width - crop_size,
                )
            )
            top = int(
                np.clip(
                    center_y - int(rng.integers(0, crop_size)),
                    0,
                    height - crop_size,
                )
            )
        else:
            left = int(rng.integers(0, width - crop_size + 1))
            top = int(rng.integers(0, height - crop_size + 1))
        image_crop = np.array(
            image[top : top + crop_size, left : left + crop_size], copy=True
        )
        mask_crop = np.array(
            mask[top : top + crop_size, left : left + crop_size], copy=True
        )
        trace_crop = np.array(
            trace[top : top + crop_size, left : left + crop_size], copy=True
        )
        if rng.random() < float(self.sampling["brightness_contrast_probability"]):
            brightness = rng.uniform(
                -float(self.sampling["brightness_delta"]),
                float(self.sampling["brightness_delta"]),
            ) * 255.0
            contrast = 1.0 + rng.uniform(
                -float(self.sampling["contrast_delta"]),
                float(self.sampling["contrast_delta"]),
            )
            image_crop = np.clip(
                image_crop.astype(np.float32) * contrast + brightness, 0, 255
            ).astype(np.uint8)
        if rng.random() < float(self.sampling["jpeg_probability"]):
            quality = int(
                rng.integers(
                    int(self.sampling["jpeg_quality_min"]),
                    int(self.sampling["jpeg_quality_max"]) + 1,
                )
            )
            image_crop = _jpeg(image_crop, quality)
        mean = np.asarray(self.preprocessing["imagenet_mean"], dtype=np.float32)
        std = np.asarray(self.preprocessing["imagenet_std"], dtype=np.float32)
        image_float = (image_crop.astype(np.float32) / 255.0 - mean) / std
        return (
            torch.from_numpy(image_float.transpose(2, 0, 1).copy()),
            torch.from_numpy(mask_crop.astype(np.float32, copy=False)).unsqueeze(0),
            torch.from_numpy(trace_crop.astype(np.float32, copy=False)).unsqueeze(0),
        )


def _auxiliary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    edited_pixel_weight: float,
) -> torch.Tensor:
    per_pixel = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = 1.0 + (edited_pixel_weight - 1.0) * masks
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config, base_config_path, base_config_sha256 = _load_config(config_path)
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not runtime["gpu_launch_authorized"] or not runtime[
        "trace_aux_training_authorized"
    ]:
        raise ValueError("PairTrace auxiliary-trace training was not authorized")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("PairTrace auxiliary-trace pilot cannot be paper evidence")
    data_config = config["data"]
    if not data_config["training_must_not_read_viewed_diagnostic"]:
        raise ValueError("viewed diagnostic must remain excluded")
    if not data_config["training_must_not_read_final_reserve"]:
        raise ValueError("final reserve must remain excluded")
    target_mode = str(config["experiment"]["target_mode"])
    if target_mode not in {"correct_trace", "shuffled_trace", "zero_trace"}:
        raise ValueError(f"unsupported auxiliary trace mode: {target_mode}")
    probability_sum = sum(
        float(config["sampling"][name])
        for name in (
            "forged_positive_probability",
            "forged_random_probability",
            "authentic_random_probability",
        )
    )
    if not math.isclose(probability_sum, 1.0, abs_tol=1e-12):
        raise ValueError("auxiliary-trace sampling probabilities must sum to one")
    if not config["auxiliary"]["target_constructed_before_augmentation"]:
        raise ValueError("auxiliary trace target timing changed")
    if config["auxiliary"]["difference_reduction"] != "max_abs_rgb":
        raise ValueError("auxiliary trace reduction changed")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PairTrace auxiliary-trace pilot requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("PairTrace auxiliary-trace protocol SHA-256 changed")
    manifest_path = _resolve(project_root, data_config["manifest"])
    if _sha256(manifest_path) != data_config["expected_manifest_sha256"]:
        raise ValueError("PairTrace auxiliary-trace manifest SHA-256 changed")
    student_summary_path = _resolve(
        project_root, config["matched_student"]["summary"]
    )
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
        raise ValueError("auxiliary-trace role counts changed")
    if {row["source_group_id"] for row in train_rows} & {
        row["source_group_id"] for row in validation_rows
    }:
        raise ValueError("auxiliary-trace train/validation groups overlap")
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
        raise ValueError("auxiliary-trace encoder weights SHA-256 changed")
    model = _load_model(
        weights_path, int(config["model"]["auxiliary_head_channels"])
    ).to(device)
    training = config["training"]
    dataset = _TraceAuxDataset(
        train_cache,
        target_mode,
        shuffled_authentic,
        config["sampling"],
        config["preprocessing"],
        config["auxiliary"],
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
        direct_losses: list[float] = []
        auxiliary_losses: list[float] = []
        for images, masks, traces in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            traces = traces.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])
            ):
                segmentation_logits, trace_logits = model.forward_with_aux(images)
                bce = F.binary_cross_entropy_with_logits(
                    segmentation_logits, masks, pos_weight=positive_weight
                )
                dice = _dice_loss(segmentation_logits, masks)
                direct = float(training["bce_loss_weight"]) * bce + float(
                    training["dice_loss_weight"]
                ) * dice
                auxiliary_loss = _auxiliary_loss(
                    trace_logits,
                    traces,
                    masks,
                    float(config["auxiliary"]["edited_pixel_weight"]),
                )
                loss = direct + float(config["auxiliary"]["loss_weight"]) * auxiliary_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            direct_losses.append(float(direct.detach().cpu()))
            auxiliary_losses.append(float(auxiliary_loss.detach().cpu()))
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
            "direct_loss": float(np.mean(direct_losses)),
            "auxiliary_trace_loss": float(np.mean(auxiliary_losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_forged_document_macro_pixel_ap_model_resolution": validation_ap,
            "target_mode": target_mode,
            "paper_evidence": False,
        }
        epoch_records.append(record)
        _write_jsonl(epoch_log_path, epoch_records)
        logging.info("trace-aux epoch=%d metrics=%s", epoch + 1, record)
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
