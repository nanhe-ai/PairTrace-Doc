from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _invalid_thread_value = int(os.environ.get(_thread_variable, "1")) < 1
    except ValueError:
        _invalid_thread_value = True
    if _invalid_thread_value:
        os.environ[_thread_variable] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from pairtrace_doc.pipelines.train_pairtrace_100 import _PairTraceDataset
from pairtrace_doc.pipelines.train_student_100 import (
    ResNet18UNet,
    _ConvBlock,
    _dice_loss,
    _positions,
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


ARMS = {
    "signed_difference_3ch",
    "candidate_reference_6ch",
    "explicit_9ch",
    "fc_siam_diff",
}
EXTENDED_ARMS = ARMS | {"absolute_difference_3ch", "fc_siam_conc"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle)
    base_value = override.pop("base_config", None)
    expected_base_sha256 = override.pop("expected_base_config_sha256", None)
    if base_value is None:
        return override
    project_root = config_path.parent.parent
    base_path = _resolve(project_root, str(base_value))
    if expected_base_sha256 and _sha256(base_path) != expected_base_sha256:
        raise ValueError("equal-budget base config SHA-256 changed")
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    return _deep_merge(base, override)


class FCSiamDiffUNet(nn.Module):
    """Shared ResNet-18 streams decoded from absolute feature differences."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.decoder4 = _ConvBlock(512 + 256, 256)
        self.decoder3 = _ConvBlock(256 + 128, 128)
        self.decoder2 = _ConvBlock(128 + 64, 64)
        self.decoder1 = _ConvBlock(64 + 64, 64)
        self.output = nn.Sequential(_ConvBlock(64, 32), nn.Conv2d(32, 1, 1))

    def _encode(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(value)))
        layer1 = self.encoder.layer1(self.encoder.maxpool(stem))
        layer2 = self.encoder.layer2(layer1)
        layer3 = self.encoder.layer3(layer2)
        layer4 = self.encoder.layer4(layer3)
        return stem, layer1, layer2, layer3, layer4

    @staticmethod
    def _upsample(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] != 6:
            raise ValueError("FC-Siam-diff expects candidate/reference RGB concatenation")
        candidate = self._encode(value[:, :3])
        reference = self._encode(value[:, 3:6])
        differences = [torch.abs(left - right) for left, right in zip(candidate, reference)]
        stem, layer1, layer2, layer3, layer4 = differences
        value = self.decoder4(torch.cat([self._upsample(layer4, layer3), layer3], dim=1))
        value = self.decoder3(torch.cat([self._upsample(value, layer2), layer2], dim=1))
        value = self.decoder2(torch.cat([self._upsample(value, layer1), layer1], dim=1))
        value = self.decoder1(torch.cat([self._upsample(value, stem), stem], dim=1))
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.output(value)


class FCSiamConcUNet(nn.Module):
    """Shared ResNet-18 streams decoded from concatenated paired features."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.decoder4 = _ConvBlock(1024 + 512, 256)
        self.decoder3 = _ConvBlock(256 + 256, 128)
        self.decoder2 = _ConvBlock(128 + 128, 64)
        self.decoder1 = _ConvBlock(64 + 128, 64)
        self.output = nn.Sequential(_ConvBlock(64, 32), nn.Conv2d(32, 1, 1))

    def _encode(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(value)))
        layer1 = self.encoder.layer1(self.encoder.maxpool(stem))
        layer2 = self.encoder.layer2(layer1)
        layer3 = self.encoder.layer3(layer2)
        layer4 = self.encoder.layer4(layer3)
        return stem, layer1, layer2, layer3, layer4

    @staticmethod
    def _upsample(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] != 6:
            raise ValueError("FC-Siam-conc expects candidate/reference RGB concatenation")
        candidate = self._encode(value[:, :3])
        reference = self._encode(value[:, 3:6])
        paired = [torch.cat([left, right], dim=1) for left, right in zip(candidate, reference)]
        stem, layer1, layer2, layer3, layer4 = paired
        value = self.decoder4(torch.cat([self._upsample(layer4, layer3), layer3], dim=1))
        value = self.decoder3(torch.cat([self._upsample(value, layer2), layer2], dim=1))
        value = self.decoder2(torch.cat([self._upsample(value, layer1), layer1], dim=1))
        value = self.decoder1(torch.cat([self._upsample(value, stem), stem], dim=1))
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.output(value)


def _replace_conv1(model: ResNet18UNet, channel_coefficients: list[float]) -> None:
    original = model.encoder.conv1
    replacement = nn.Conv2d(
        3 * len(channel_coefficients),
        original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=False,
    )
    with torch.no_grad():
        for index, coefficient in enumerate(channel_coefficients):
            replacement.weight[:, index * 3 : (index + 1) * 3].copy_(
                original.weight * coefficient
            )
    model.encoder.conv1 = replacement


def _build_model(arm: str, encoder_state: dict[str, torch.Tensor]) -> nn.Module:
    if arm not in EXTENDED_ARMS:
        raise ValueError(f"unsupported representation arm: {arm}")
    if arm == "fc_siam_diff":
        model = FCSiamDiffUNet()
        model.encoder.load_state_dict(encoder_state, strict=True)
        return model
    if arm == "fc_siam_conc":
        model = FCSiamConcUNet()
        model.encoder.load_state_dict(encoder_state, strict=True)
        return model
    model = ResNet18UNet()
    model.encoder.load_state_dict(encoder_state, strict=True)
    if arm == "candidate_reference_6ch":
        _replace_conv1(model, [0.5, 0.5])
    elif arm == "explicit_9ch":
        _replace_conv1(model, [0.25, 0.25, 0.5])
    return model


def _select_representation(teacher_tensor: torch.Tensor, arm: str) -> torch.Tensor:
    if teacher_tensor.shape[-3] != 9:
        raise ValueError("paired representation source must contain nine channels")
    if arm == "signed_difference_3ch":
        return teacher_tensor[..., 6:9, :, :]
    if arm == "absolute_difference_3ch":
        return torch.abs(teacher_tensor[..., 6:9, :, :])
    if arm in {"candidate_reference_6ch", "fc_siam_diff", "fc_siam_conc"}:
        return teacher_tensor[..., :6, :, :]
    if arm == "explicit_9ch":
        return teacher_tensor
    raise ValueError(f"unsupported representation arm: {arm}")


class _RepresentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, paired: _PairTraceDataset, arm: str) -> None:
        self.paired = paired
        self.arm = arm

    def set_epoch(self, epoch: int) -> None:
        self.paired.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.paired)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, teacher, mask, _ = self.paired[index]
        return _select_representation(teacher, self.arm), mask


def _group_macro(values: Iterable[float], records: Iterable[dict[str, Any]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, record in zip(values, records):
        grouped[str(record["source_group_id"])].append(float(value))
    if not grouped:
        raise ValueError("source-group macro metric requires at least one group")
    return float(np.mean([np.mean(group_values) for group_values in grouped.values()]))


def _threshold_count(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, dtype=np.float32).reshape(-1))
    return ordered.size - np.searchsorted(ordered, thresholds, side="left")


def _select_operating_point(
    forged_scores: list[np.ndarray],
    forged_masks: list[np.ndarray],
    forged_groups: list[str],
    authentic_scores: dict[str, np.ndarray],
    thresholds: np.ndarray,
    max_authentic_fpr: float,
) -> dict[str, Any]:
    f1_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    for scores, mask, group in zip(forged_scores, forged_masks, forged_groups):
        labels = np.asarray(mask, dtype=bool).reshape(-1)
        flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        positives = int(labels.sum())
        if positives == 0 or positives == labels.size:
            raise ValueError("forged operating-point sample requires both pixel classes")
        tp = _threshold_count(flat_scores[labels], thresholds)
        fp = _threshold_count(flat_scores[~labels], thresholds)
        fn = positives - tp
        denominator = 2 * tp + fp + fn
        f1_by_group[group].append(
            np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
        )
    forged_group_macro_f1 = np.mean(
        [np.mean(values, axis=0) for values in f1_by_group.values()], axis=0
    )
    authentic_group_fprs = np.stack(
        [
            _threshold_count(scores, thresholds) / np.asarray(scores).size
            for scores in authentic_scores.values()
        ]
    )
    authentic_macro_fpr = authentic_group_fprs.mean(axis=0)
    feasible = np.flatnonzero(authentic_macro_fpr <= max_authentic_fpr + 1e-12)
    if feasible.size == 0:
        raise ValueError("no validation threshold satisfies authentic FPR constraint")
    feasible_f1 = forged_group_macro_f1[feasible]
    best_f1 = float(feasible_f1.max())
    tied = feasible[np.flatnonzero(np.isclose(feasible_f1, best_f1, atol=1e-12))]
    selected = int(tied[-1])
    return {
        "threshold": float(thresholds[selected]),
        "source_group_macro_forged_pixel_f1": float(forged_group_macro_f1[selected]),
        "unique_authentic_group_macro_pixel_fpr": float(authentic_macro_fpr[selected]),
        "feasible_threshold_count": int(feasible.size),
    }


def _pair_patches(
    candidate: np.ndarray,
    reference: np.ndarray,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    mean = np.asarray(preprocessing["imagenet_mean"], dtype=np.float32)
    std = np.asarray(preprocessing["imagenet_std"], dtype=np.float32)
    candidate_float = (candidate.astype(np.float32) / 255.0 - mean) / std
    reference_float = (reference.astype(np.float32) / 255.0 - mean) / std
    return np.concatenate(
        [candidate_float, reference_float, candidate_float - reference_float], axis=-1
    )


def _infer_pair_tiled(
    model: nn.Module,
    candidate: np.ndarray,
    reference: np.ndarray,
    arm: str,
    device: torch.device,
    training: dict[str, Any],
    preprocessing: dict[str, Any],
) -> np.ndarray:
    if candidate.shape != reference.shape:
        raise ValueError("paired validation images are not aligned")
    tile = int(training["validation_tile_size"])
    stride = int(training["validation_tile_stride"])
    batch_size = int(training["validation_tile_batch_size"])
    height, width = candidate.shape[:2]
    pad_height = max(0, tile - height)
    pad_width = max(0, tile - width)
    padding = ((0, pad_height), (0, pad_width), (0, 0))
    candidate_pad = np.pad(candidate, padding, mode="reflect")
    reference_pad = np.pad(reference, padding, mode="reflect")
    padded_height, padded_width = candidate_pad.shape[:2]
    coordinates = [
        (top, left)
        for top in _positions(padded_height, tile, stride)
        for left in _positions(padded_width, tile, stride)
    ]
    accumulator = np.zeros((padded_height, padded_width), dtype=np.float32)
    counts = np.zeros_like(accumulator)
    model.eval()
    for start in range(0, len(coordinates), batch_size):
        selected = coordinates[start : start + batch_size]
        nine_channel = np.stack(
            [
                _pair_patches(
                    candidate_pad[top : top + tile, left : left + tile],
                    reference_pad[top : top + tile, left : left + tile],
                    preprocessing,
                )
                for top, left in selected
            ]
        )
        tensor = torch.from_numpy(nine_channel.transpose(0, 3, 1, 2).copy())
        tensor = _select_representation(tensor, arm).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])
        ):
            probabilities = torch.sigmoid(model(tensor)).squeeze(1).float().cpu().numpy()
        for probability, (top, left) in zip(probabilities, selected):
            accumulator[top : top + tile, left : left + tile] += probability
            counts[top : top + tile, left : left + tile] += 1.0
    return (accumulator / np.maximum(counts, 1.0))[:height, :width]


def _evaluate_forged(
    model: nn.Module,
    records: list[dict[str, Any]],
    arm: str,
    device: torch.device,
    training: dict[str, Any],
    preprocessing: dict[str, Any],
    keep_scores: bool,
) -> tuple[float, list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    results: list[dict[str, Any]] = []
    score_arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for record in records:
        candidate = np.asarray(np.load(record["forged"], mmap_mode="r"))
        reference = np.asarray(np.load(record["authentic"], mmap_mode="r"))
        mask = np.asarray(np.load(record["mask"], mmap_mode="r"), dtype=bool)
        scores = _infer_pair_tiled(
            model, candidate, reference, arm, device, training, preprocessing
        )
        average_precision, auroc = _ranking_metrics(scores, mask)
        results.append(
            {
                "sample_id": record["sample_id"],
                "source_group_id": record["source_group_id"],
                "average_precision": average_precision,
                "auroc": auroc,
            }
        )
        if keep_scores:
            score_arrays.append(scores)
            masks.append(mask)
    group_macro = _group_macro(
        [result["average_precision"] for result in results], records
    )
    return group_macro, results, score_arrays, masks


def _evaluate_authentic(
    model: nn.Module,
    records: list[dict[str, Any]],
    arm: str,
    device: torch.device,
    training: dict[str, Any],
    preprocessing: dict[str, Any],
) -> dict[str, np.ndarray]:
    representatives: dict[str, dict[str, Any]] = {}
    for record in records:
        representatives.setdefault(str(record["source_group_id"]), record)
    result: dict[str, np.ndarray] = {}
    for group, record in representatives.items():
        authentic = np.asarray(np.load(record["authentic"], mmap_mode="r"))
        result[group] = _infer_pair_tiled(
            model, authentic, authentic, arm, device, training, preprocessing
        )
    return result


def _validate_config(config: dict[str, Any]) -> None:
    if config["experiment"]["arm"] not in EXTENDED_ARMS:
        raise ValueError("unknown equal-budget arm")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("development representation pilot cannot be paper evidence")
    runtime = config["runtime"]
    if not runtime["gpu_launch_authorized"]:
        raise ValueError("equal-budget GPU experiment was not explicitly authorized")
    if not runtime["training_must_not_read_holdout"]:
        raise ValueError("equal-budget training must keep the holdout unread")
    training = config["training"]
    expected_steps = int(training["epochs"]) * int(training["steps_per_epoch"])
    if expected_steps != int(training["expected_optimizer_steps"]):
        raise ValueError("configured optimizer-step budget is inconsistent")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = _load_config(config_path)
    _validate_config(config)
    for name, value in config["runtime"].get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("equal-budget representation experiment requires CUDA")

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("equal-budget protocol SHA-256 changed")
    manifest_path = _resolve(project_root, config["data"]["manifest"])
    if _sha256(manifest_path) != config["data"]["expected_manifest_sha256"]:
        raise ValueError("equal-budget train/validation manifest SHA-256 changed")
    rows = _read_jsonl(manifest_path)
    role_field = str(config["data"].get("role_field", "pilot_role"))
    train_role = str(config["data"].get("train_role", "train"))
    validation_role = str(config["data"].get("validation_role", "validation"))
    train_rows = sorted(
        [row for row in rows if str(row[role_field]) == train_role],
        key=lambda row: (str(row["source_group_id"]), str(row["sample_id"])),
    )
    validation_rows = sorted(
        [row for row in rows if str(row[role_field]) == validation_role],
        key=lambda row: (str(row["source_group_id"]), str(row["sample_id"])),
    )
    data = config["data"]
    if len(train_rows) != int(data["expected_train_records"]):
        raise ValueError("equal-budget training record count changed")
    if len(validation_rows) != int(data["expected_validation_records"]):
        raise ValueError("equal-budget validation record count changed")
    train_groups = {str(row["source_group_id"]) for row in train_rows}
    validation_groups = {str(row["source_group_id"]) for row in validation_rows}
    if train_groups & validation_groups:
        raise ValueError("equal-budget train/validation source groups overlap")
    if len(train_groups) != int(data["expected_train_groups"]):
        raise ValueError("equal-budget training source-group count changed")
    if len(validation_groups) != int(data["expected_validation_groups"]):
        raise ValueError("equal-budget validation source-group count changed")
    freeze_field = str(data.get("freeze_field", "freeze_id"))
    if {str(row[freeze_field]) for row in rows} != {str(data["expected_freeze_id"])}:
        raise ValueError("equal-budget split freeze ID changed")
    if data.get("max_train_pairs") is not None:
        train_rows = train_rows[: int(data["max_train_pairs"])]
    if data.get("max_validation_pairs") is not None:
        validation_rows = validation_rows[: int(data["max_validation_pairs"])]
    if not train_rows or not validation_rows:
        raise ValueError("equal-budget arm requires non-empty train and validation subsets")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    pair_cache_dir = _resolve(scratch, paths["pair_cache_dir"])
    checkpoint_path = _resolve(project_root, paths["checkpoint"])
    epoch_log_path = _resolve(project_root, paths["epoch_log"])
    predictions_path = _resolve(project_root, paths["prediction_records"])
    metrics_path = _resolve(project_root, paths["metrics"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (
        pair_cache_dir,
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
    for target, selected_rows in ((train_cache, train_rows), (validation_cache, validation_rows)):
        for row in selected_rows:
            record, hit = _prepare_pair_cache(
                row, scratch, pair_cache_dir, config["preprocessing"]
            )
            target.append(record)
            cache_hits += int(hit)

    weights_path = _resolve(scratch, config["model"]["encoder_weights"])
    weights_sha256 = _sha256(weights_path)
    if weights_sha256 != config["model"]["encoder_weights_sha256"]:
        raise ValueError("ResNet-18 ImageNet initialization SHA-256 changed")
    encoder_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    arm = str(config["experiment"]["arm"])
    model = _build_model(arm, encoder_state).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"])
    )
    positive_weight = torch.tensor(float(training["bce_positive_weight"]), device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    paired_dataset = _PairTraceDataset(
        train_cache,
        phase=str(training.get("dataset_phase", "student")),
        pair_mode="correct_pair",
        shuffled_authentic={},
        sampling=config["sampling"],
        preprocessing=config["preprocessing"],
        seed=seed,
        length=int(training["steps_per_epoch"]) * int(training["batch_size"]),
    )
    dataset = _RepresentationDataset(paired_dataset, arm)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        persistent_workers=False,
    )

    torch.cuda.reset_peak_memory_stats(device)
    epoch_records: list[dict[str, Any]] = []
    best_ap = -math.inf
    best_epoch = -1
    completed_steps = 0
    for epoch in range(int(training["epochs"])):
        dataset.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        bce_losses: list[float] = []
        dice_losses: list[float] = []
        for inputs, masks in loader:
            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])
            ):
                logits = model(inputs)
                bce = F.binary_cross_entropy_with_logits(
                    logits, masks, pos_weight=positive_weight
                )
                dice = _dice_loss(logits, masks)
                loss = (
                    float(training["bce_loss_weight"]) * bce
                    + float(training["dice_loss_weight"]) * dice
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {completed_steps + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            completed_steps += 1
            losses.append(float(loss.detach().cpu()))
            bce_losses.append(float(bce.detach().cpu()))
            dice_losses.append(float(dice.detach().cpu()))
        scheduler.step()
        run_epoch_validation = bool(training.get("validate_every_epoch", True)) or epoch == int(
            training["epochs"]
        ) - 1
        validation_ap = None
        if run_epoch_validation:
            validation_ap, _, _, _ = _evaluate_forged(
                model,
                validation_cache,
                arm,
                device,
                training,
                config["preprocessing"],
                keep_scores=False,
            )
        epoch_record = {
            "arm": arm,
            "epoch": epoch + 1,
            "optimizer_steps_completed": completed_steps,
            "loss": float(np.mean(losses)),
            "bce_loss": float(np.mean(bce_losses)),
            "dice_loss": float(np.mean(dice_losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_source_group_macro_pixel_ap_model_resolution": validation_ap,
            "paper_evidence": False,
        }
        epoch_records.append(epoch_record)
        _write_jsonl(epoch_log_path, epoch_records)
        logging.info("epoch=%d metrics=%s", epoch + 1, epoch_record)
        checkpoint_rule = str(training.get("checkpoint_rule", "best_validation_ap"))
        should_save = (
            checkpoint_rule == "fixed_final_epoch" and epoch == int(training["epochs"]) - 1
        ) or (
            checkpoint_rule == "best_validation_ap"
            and validation_ap is not None
            and validation_ap > best_ap
        )
        if should_save:
            if validation_ap is None:
                raise RuntimeError("checkpoint save requires a validation metric")
            best_ap = validation_ap
            best_epoch = epoch + 1
            _save_checkpoint(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "arm": arm,
                    "epoch": best_epoch,
                    "validation_source_group_macro_pixel_ap_model_resolution": best_ap,
                    "config_sha256": _sha256(config_path),
                    "protocol_sha256": _sha256(protocol_path),
                    "encoder_weights_sha256": weights_sha256,
                    "seed": seed,
                    "parameter_count": parameter_count,
                    "selection_rule": checkpoint_rule,
                },
            )

    if completed_steps != int(training["expected_optimizer_steps"]):
        raise RuntimeError("completed optimizer steps differ from frozen budget")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state"], strict=True)
    model = model.to(device).eval()
    checkpoint_sha256 = _sha256(checkpoint_path)
    group_macro_ap, forged_results, forged_scores, forged_masks = _evaluate_forged(
        model,
        validation_cache,
        arm,
        device,
        training,
        config["preprocessing"],
        keep_scores=True,
    )
    authentic_scores = _evaluate_authentic(
        model,
        validation_cache,
        arm,
        device,
        training,
        config["preprocessing"],
    )
    operating = config["operating_point"]
    thresholds = np.arange(
        float(operating["candidate_min"]),
        float(operating["candidate_max"]) + float(operating["candidate_step"]) / 2,
        float(operating["candidate_step"]),
    )
    selected = _select_operating_point(
        forged_scores,
        forged_masks,
        [str(record["source_group_id"]) for record in validation_cache],
        authentic_scores,
        thresholds,
        float(operating["max_authentic_fpr"]),
    )
    threshold = float(selected["threshold"])
    prediction_records: list[dict[str, Any]] = []
    for record, result, scores, mask in zip(
        validation_cache, forged_results, forged_scores, forged_masks
    ):
        binary = scores >= threshold
        tp = int(np.count_nonzero(binary & mask))
        fp = int(np.count_nonzero(binary & ~mask))
        fn = int(np.count_nonzero(~binary & mask))
        prediction_records.append(
            {
                "arm": arm,
                "role": "validation",
                "sample_kind": "forged_pair",
                "sample_id": record["sample_id"],
                "source_group_id": record["source_group_id"],
                "average_precision": result["average_precision"],
                "auroc": result["auroc"],
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pixel_count": int(mask.size),
                "paper_evidence": False,
            }
        )
    for group, scores in authentic_scores.items():
        prediction_records.append(
            {
                "arm": arm,
                "role": "validation",
                "sample_kind": "authentic_pair",
                "sample_id": f"{group}:authentic",
                "source_group_id": group,
                "threshold": threshold,
                "false_positive_pixels": int(np.count_nonzero(scores >= threshold)),
                "pixel_count": int(scores.size),
                "fpr": float(np.mean(scores >= threshold)),
                "paper_evidence": False,
            }
        )
    _write_jsonl(predictions_path, prediction_records)
    metric_row = {
        "arm": arm,
        "best_epoch": best_epoch,
        "optimizer_steps": completed_steps,
        "validation_source_group_macro_pixel_ap": group_macro_ap,
        "validation_threshold": threshold,
        "validation_source_group_macro_forged_pixel_f1": selected[
            "source_group_macro_forged_pixel_f1"
        ],
        "validation_unique_authentic_group_macro_pixel_fpr": selected[
            "unique_authentic_group_macro_pixel_fpr"
        ],
        "paper_evidence": False,
    }
    _write_csv(metrics_path, [metric_row])
    summary = {
        "status": "preflight_complete"
        if config["experiment"]["stage"] == "preflight"
        else "development_arm_complete",
        "experiment": config["experiment"]["name"],
        "stage": config["experiment"]["stage"],
        "arm": arm,
        "seed": seed,
        "paper_evidence": False,
        "protocol_sha256": _sha256(protocol_path),
        "manifest_sha256": _sha256(manifest_path),
        "holdout_membership_sha256_configured_but_not_read": data[
            "expected_holdout_membership_sha256"
        ],
        "holdout_read": False,
        "train_records_used": len(train_cache),
        "validation_records_used": len(validation_cache),
        "validation_unique_authentic_groups": len(authentic_scores),
        "cache_hits": cache_hits,
        "parameter_count": parameter_count,
        "optimizer_steps_completed": completed_steps,
        "best_epoch": best_epoch,
        "validation_source_group_macro_pixel_ap_model_resolution": group_macro_ap,
        "operating_point": selected,
        "checkpoint": str(checkpoint_path.relative_to(project_root)),
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_records_sha256": _sha256(predictions_path),
        "metrics_sha256": _sha256(metrics_path),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "wall_time_seconds": float(time.monotonic() - started),
        "silent_failures": 0,
    }
    _write_json(summary_path, summary)
    logging.info("summary=%s", json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one frozen TFR equal-budget arm")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
