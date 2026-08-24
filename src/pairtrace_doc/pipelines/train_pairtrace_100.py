from __future__ import annotations

import argparse
import hashlib
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

from pairtrace_doc.pipelines.train_student_100 import (
    ResNet18UNet,
    _dice_loss,
    _expected_role_counts,
    _generator_sampling_pools,
    _grouped_mean,
    _infer_tiled,
    _prepare_pair_cache,
    _ranking_metrics,
    _read_jsonl,
    _resolve,
    _save_checkpoint,
    _sha256,
    _threshold_vectors,
    _write_csv,
    _write_json,
    _write_jsonl,
)


class TraceUNet(ResNet18UNet):
    def forward_with_features(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stem = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(value)))
        layer1 = self.encoder.layer1(self.encoder.maxpool(stem))
        layer2 = self.encoder.layer2(layer1)
        layer3 = self.encoder.layer3(layer2)
        layer4 = self.encoder.layer4(layer3)
        decoder4 = self.decoder4(
            torch.cat([self._upsample(layer4, layer3), layer3], dim=1)
        )
        decoder3 = self.decoder3(
            torch.cat([self._upsample(decoder4, layer2), layer2], dim=1)
        )
        decoder2 = self.decoder2(
            torch.cat([self._upsample(decoder3, layer1), layer1], dim=1)
        )
        decoder1 = self.decoder1(
            torch.cat([self._upsample(decoder2, stem), stem], dim=1)
        )
        output_feature = F.interpolate(
            decoder1, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return self.output(output_feature), {
            "decoder3": decoder3,
            "decoder2": decoder2,
            "decoder1": decoder1,
        }

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(value)
        return logits


def _load_student(weights_path: Path) -> TraceUNet:
    model = TraceUNet()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(state, strict=True)
    return model


def _load_teacher(
    weights_path: Path, coefficients: dict[str, float]
) -> TraceUNet:
    model = _load_student(weights_path)
    original = model.encoder.conv1
    replacement = nn.Conv2d(
        9,
        original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=False,
    )
    with torch.no_grad():
        replacement.weight[:, 0:3].copy_(
            original.weight * float(coefficients["forged"])
        )
        replacement.weight[:, 3:6].copy_(
            original.weight * float(coefficients["authentic"])
        )
        replacement.weight[:, 6:9].copy_(
            original.weight * float(coefficients["signed_difference"])
        )
    model.encoder.conv1 = replacement
    return model


def _shuffled_authentic_map(
    records: list[dict[str, Any]], seed: int
) -> dict[str, str]:
    representatives: dict[str, dict[str, Any]] = {}
    authentic_fingerprints: dict[str, str] = {}
    for record in records:
        source = str(record["source_group_id"])
        source_row = record.get("source_row", record)
        fingerprint = str(
            source_row.get("authentic_sha256", record["authentic"])
        )
        if source in authentic_fingerprints:
            if authentic_fingerprints[source] != fingerprint:
                raise ValueError(
                    "one source group has multiple authentic identities"
                )
            continue
        authentic_fingerprints[source] = fingerprint
        representatives[source] = record
    if len(representatives) < 2:
        raise ValueError("shuffled-pair control requires at least two source groups")
    ordered = sorted(
        representatives.values(),
        key=lambda item: (
            hashlib.sha256(
                f"{seed}|{item['source_group_id']}".encode("utf-8")
            ).hexdigest(),
            str(item["source_group_id"]),
        ),
    )
    result: dict[str, str] = {}
    for index, record in enumerate(ordered):
        source = str(record["source_group_id"])
        target = ordered[(index + 1) % len(ordered)]
        if source == str(target["source_group_id"]):
            raise ValueError("shuffled-pair mapping retained a source group")
        result[source] = str(target["authentic"])
    return result


def _pad_triplet(
    forged: np.ndarray,
    authentic: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    pad_height = max(0, crop_size - height)
    pad_width = max(0, crop_size - width)
    if pad_height or pad_width:
        padding = ((0, pad_height), (0, pad_width), (0, 0))
        forged = np.pad(forged, padding, mode="reflect")
        authentic = np.pad(authentic, padding, mode="reflect")
        mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode="constant")
    return forged, authentic, mask


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return np.asarray(handle.convert("RGB"))


class _PairTraceDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        records: list[dict[str, Any]],
        phase: str,
        pair_mode: str,
        shuffled_authentic: dict[str, str],
        sampling: dict[str, Any],
        preprocessing: dict[str, Any],
        seed: int,
        length: int,
    ) -> None:
        if phase not in {"teacher", "student"}:
            raise ValueError(f"unsupported PairTrace phase: {phase}")
        if pair_mode not in {"correct_pair", "shuffled_pair"}:
            raise ValueError(f"unsupported PairTrace pair mode: {pair_mode}")
        self.records = records
        self.phase = phase
        self.pair_mode = pair_mode
        self.shuffled_authentic = shuffled_authentic
        self.sampling = sampling
        self.preprocessing = preprocessing
        self.seed = seed
        self.length = length
        self.epoch = 0
        self.generator_sampling = _generator_sampling_pools(records, sampling)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def _paired_authentic(
        self, record: dict[str, Any], forged_shape: tuple[int, int]
    ) -> np.ndarray:
        if self.pair_mode == "correct_pair":
            return np.asarray(np.load(record["authentic"], mmap_mode="r"))
        target_path = self.shuffled_authentic[str(record["source_group_id"])]
        authentic = np.asarray(np.load(target_path, mmap_mode="r"))
        if authentic.shape[:2] != forged_shape:
            authentic = cv2.resize(
                authentic,
                (forged_shape[1], forged_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return authentic

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Student sampling deliberately matches the frozen single-image control.
        # Teacher pretraining uses a disjoint deterministic stream.
        phase_offset = 10_000_019 if self.phase == "teacher" else 0
        rng = np.random.default_rng(
            self.seed + phase_offset + self.epoch * 1_000_003 + index
        )
        if self.generator_sampling is None:
            record = self.records[int(rng.integers(0, len(self.records)))]
        else:
            generators, probabilities, pools = self.generator_sampling
            generator = str(rng.choice(generators, p=probabilities))
            pool = pools[generator]
            record = pool[int(rng.integers(0, len(pool)))]
        if self.phase == "teacher":
            forged_sample = True
            positive_crop = rng.random() < float(
                self.sampling["teacher_forged_positive_probability"]
            )
        else:
            probability = float(rng.random())
            positive_limit = float(
                self.sampling["student_forged_positive_probability"]
            )
            forged_limit = positive_limit + float(
                self.sampling["student_forged_random_probability"]
            )
            forged_sample = probability < forged_limit
            positive_crop = probability < positive_limit

        stored_mask = np.asarray(np.load(record["mask"], mmap_mode="r"))
        if forged_sample:
            forged = np.asarray(np.load(record["forged"], mmap_mode="r"))
            authentic = self._paired_authentic(record, stored_mask.shape)
            mask = stored_mask
            distillation_active = 1.0
        else:
            forged = np.asarray(np.load(record["authentic"], mmap_mode="r"))
            authentic = forged
            mask = np.zeros(stored_mask.shape, dtype=np.uint8)
            distillation_active = 0.0

        crop_size = int(self.preprocessing["crop_size"])
        forged, authentic, mask = _pad_triplet(
            forged, authentic, mask, crop_size
        )
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
        forged_crop = np.array(
            forged[top : top + crop_size, left : left + crop_size], copy=True
        )
        authentic_crop = np.array(
            authentic[top : top + crop_size, left : left + crop_size], copy=True
        )
        mask_crop = np.array(
            mask[top : top + crop_size, left : left + crop_size], copy=True
        )

        if rng.random() < float(
            self.sampling["brightness_contrast_probability"]
        ):
            brightness = rng.uniform(
                -float(self.sampling["brightness_delta"]),
                float(self.sampling["brightness_delta"]),
            ) * 255.0
            contrast = 1.0 + rng.uniform(
                -float(self.sampling["contrast_delta"]),
                float(self.sampling["contrast_delta"]),
            )
            forged_crop = np.clip(
                forged_crop.astype(np.float32) * contrast + brightness, 0, 255
            ).astype(np.uint8)
            authentic_crop = np.clip(
                authentic_crop.astype(np.float32) * contrast + brightness, 0, 255
            ).astype(np.uint8)
        if rng.random() < float(self.sampling["jpeg_probability"]):
            quality = int(
                rng.integers(
                    int(self.sampling["jpeg_quality_min"]),
                    int(self.sampling["jpeg_quality_max"]) + 1,
                )
            )
            forged_crop = _jpeg(forged_crop, quality)
            authentic_crop = _jpeg(authentic_crop, quality)

        mean = np.asarray(self.preprocessing["imagenet_mean"], dtype=np.float32)
        std = np.asarray(self.preprocessing["imagenet_std"], dtype=np.float32)
        forged_float = (forged_crop.astype(np.float32) / 255.0 - mean) / std
        authentic_float = (
            authentic_crop.astype(np.float32) / 255.0 - mean
        ) / std
        student_tensor = torch.from_numpy(
            forged_float.transpose(2, 0, 1).copy()
        )
        teacher_tensor = torch.from_numpy(
            np.concatenate(
                [
                    forged_float.transpose(2, 0, 1),
                    authentic_float.transpose(2, 0, 1),
                    (forged_float - authentic_float).transpose(2, 0, 1),
                ],
                axis=0,
            ).copy()
        )
        return (
            student_tensor,
            teacher_tensor,
            torch.from_numpy(mask_crop.astype(np.float32, copy=False)).unsqueeze(0),
            torch.tensor(distillation_active, dtype=torch.float32),
        )


def _feature_distillation_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    masks: torch.Tensor,
    active: torch.Tensor,
    feature_names: list[str],
    edited_pixel_weight: float,
) -> torch.Tensor:
    active_sum = active.sum().clamp_min(1.0)
    losses: list[torch.Tensor] = []
    for name in feature_names:
        student = F.normalize(student_features[name], dim=1)
        teacher = F.normalize(teacher_features[name], dim=1)
        resized_mask = F.interpolate(masks, size=student.shape[-2:], mode="nearest")
        spatial_weight = 1.0 + (edited_pixel_weight - 1.0) * resized_mask
        per_pixel = (student - teacher).square().mean(dim=1, keepdim=True)
        per_sample = (per_pixel * spatial_weight).sum(dim=(1, 2, 3)) / spatial_weight.sum(
            dim=(1, 2, 3)
        ).clamp_min(1.0)
        losses.append((per_sample * active).sum() / active_sum)
    return torch.stack(losses).mean()


def _logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    active: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    teacher_probability = torch.sigmoid(teacher_logits / temperature)
    per_pixel = F.binary_cross_entropy_with_logits(
        student_logits / temperature, teacher_probability, reduction="none"
    )
    per_sample = per_pixel.mean(dim=(1, 2, 3)) * temperature**2
    return (per_sample * active).sum() / active.sum().clamp_min(1.0)


def _native_validation(
    model: nn.Module,
    records: list[dict[str, Any]],
    scratch: Path,
    score_cache_dir: Path,
    checkpoint_sha256: str,
    device: torch.device,
    training: dict[str, Any],
    preprocessing: dict[str, Any],
    operating: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = float(operating["candidate_min"])
    stop = float(operating["candidate_max"])
    step = float(operating["candidate_step"])
    thresholds = start + np.arange(int(round((stop - start) / step)) + 1) * step
    forged_vectors: list[tuple[np.ndarray, np.ndarray, int]] = []
    authentic_fpr_vectors: list[np.ndarray] = []
    native_aps: list[float] = []
    native_aurocs: list[float] = []
    predictions: list[dict[str, Any]] = []
    for record in records:
        source = record["source_row"]
        native_shape = (int(source["image_height"]), int(source["image_width"]))
        mask_path = _resolve(scratch, source["mask"])
        if _sha256(mask_path) != source["mask_sha256"]:
            raise ValueError("PairTrace validation mask SHA-256 changed")
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0
        for sample_kind in ("forged", "authentic"):
            image = np.asarray(np.load(record[sample_kind], mmap_mode="r"))
            probability = _infer_tiled(
                model, image, device, training, preprocessing
            )
            input_sha256 = source[
                "image_sha256" if sample_kind == "forged" else "authentic_sha256"
            ]
            score_key = hashlib.sha256(
                json.dumps(
                    {
                        "checkpoint_sha256": checkpoint_sha256,
                        "input_sha256": input_sha256,
                        "preprocessing": preprocessing,
                        "sample_kind": sample_kind,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            score_path = score_cache_dir / f"{score_key}.npz"
            temporary = score_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, scores=probability.astype(np.float16))
            temporary.replace(score_path)
            native_probability = cv2.resize(
                probability,
                (native_shape[1], native_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            prediction: dict[str, Any] = {
                "record_id": f"validation:{sample_kind}:{source['source_group_id']}",
                "source_group_id": source["source_group_id"],
                "sample_kind": sample_kind,
                "status": "ok",
                "paper_evidence": False,
                "score_cache": str(score_path.relative_to(scratch)),
                "score_shape": list(probability.shape),
                "native_shape": list(native_shape),
                "checkpoint_sha256": checkpoint_sha256,
                "generator": source.get(
                    str(training.get("checkpoint_group_field", "assigned_tool")),
                    source.get("assigned_tool"),
                ),
            }
            if sample_kind == "forged":
                average_precision, auroc = _ranking_metrics(
                    native_probability, native_mask
                )
                native_aps.append(average_precision)
                native_aurocs.append(auroc)
                forged_vectors.append(
                    _threshold_vectors(native_probability, native_mask, thresholds)
                )
                prediction["macro_pixel_ap"] = average_precision
                prediction["pixel_auroc"] = auroc
            else:
                histogram, _ = np.histogram(
                    native_probability, bins=np.r_[thresholds, np.inf]
                )
                authentic_fpr_vectors.append(
                    np.cumsum(histogram[::-1], dtype=np.int64)[::-1]
                    / native_probability.size
                )
            predictions.append(prediction)

    macro_precision = np.zeros_like(thresholds, dtype=float)
    macro_recall = np.zeros_like(thresholds, dtype=float)
    macro_f1 = np.zeros_like(thresholds, dtype=float)
    macro_iou = np.zeros_like(thresholds, dtype=float)
    for tp, fp, positives in forged_vectors:
        fn = positives - tp
        precision = np.divide(
            tp,
            tp + fp,
            out=np.zeros_like(tp, dtype=float),
            where=(tp + fp) > 0,
        )
        recall = tp / positives
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        iou = np.divide(
            tp,
            tp + fp + fn,
            out=np.zeros_like(tp, dtype=float),
            where=(tp + fp + fn) > 0,
        )
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        macro_iou += iou
    count = len(forged_vectors)
    macro_precision /= count
    macro_recall /= count
    macro_f1 /= count
    macro_iou /= count
    authentic_fpr = np.mean(np.stack(authentic_fpr_vectors), axis=0)
    cap = float(operating["authentic_document_macro_pixel_fpr_max"])
    feasible = np.flatnonzero(authentic_fpr <= cap + 1e-12)
    if not feasible.size:
        feasible = np.flatnonzero(authentic_fpr == authentic_fpr.min())
    best_f1 = macro_f1[feasible].max()
    candidates = feasible[
        np.isclose(macro_f1[feasible], best_f1, atol=1e-12, rtol=0)
    ]
    best_authentic = authentic_fpr[candidates].min()
    candidates = candidates[
        np.isclose(authentic_fpr[candidates], best_authentic, atol=1e-12, rtol=0)
    ]
    selected = int(candidates[-1])
    generator_macro_ap, ap_by_generator = _grouped_mean(
        native_aps, records, training.get("checkpoint_group_field")
    )
    metrics = {
        "validation_pairs": count,
        "pixel_threshold": float(thresholds[selected]),
        "macro_pixel_ap": float(np.mean(native_aps)),
        "generator_macro_pixel_ap": generator_macro_ap,
        "pixel_auroc": float(np.mean(native_aurocs)),
        "pixel_precision": float(macro_precision[selected]),
        "pixel_recall": float(macro_recall[selected]),
        "pixel_f1": float(macro_f1[selected]),
        "pixel_iou": float(macro_iou[selected]),
        "authentic_pixel_fpr": float(authentic_fpr[selected]),
        "paper_evidence": False,
    }
    for generator, value in ap_by_generator.items():
        safe_generator = "".join(
            character if character.isalnum() else "_" for character in generator
        ).strip("_")
        metrics[f"macro_pixel_ap__{safe_generator}"] = value
    return metrics, predictions


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for name, value in config["runtime"].get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    runtime = config["runtime"]
    if not runtime["gpu_launch_authorized"] or not runtime["pairtrace_training_authorized"]:
        raise ValueError("PairTrace pilot was not explicitly authorized")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("PairTrace 100-pair pilot cannot be paper evidence")
    data_config = config["data"]
    if not data_config["training_must_not_read_viewed_diagnostic"]:
        raise ValueError("viewed diagnostic must be excluded from PairTrace training")
    if not data_config["training_must_not_read_final_reserve"]:
        raise ValueError("final reserve must be excluded from PairTrace training")
    pair_mode = str(config["experiment"]["pair_mode"])
    if pair_mode not in {"correct_pair", "shuffled_pair"}:
        raise ValueError(f"unsupported PairTrace pair mode: {pair_mode}")
    teacher_only = bool(runtime.get("teacher_only", False))
    if teacher_only and pair_mode != "correct_pair":
        raise ValueError("teacher-only training requires correct pairs")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PairTrace pilot requires CUDA")
    sampling = config["sampling"]
    student_probability_sum = sum(
        float(sampling[name])
        for name in (
            "student_forged_positive_probability",
            "student_forged_random_probability",
            "student_authentic_random_probability",
        )
    )
    teacher_probability_sum = sum(
        float(sampling[name])
        for name in (
            "teacher_forged_positive_probability",
            "teacher_forged_random_probability",
        )
    )
    if not math.isclose(student_probability_sum, 1.0, abs_tol=1e-12):
        raise ValueError("PairTrace student sampling probabilities must sum to one")
    if not math.isclose(teacher_probability_sum, 1.0, abs_tol=1e-12):
        raise ValueError("PairTrace teacher sampling probabilities must sum to one")
    if config["distillation"]["authentic_distillation_allowed"]:
        raise ValueError("PairTrace pilot cannot distill authentic negative crops")
    if int(config["model"]["teacher_input_channels"]) != 9:
        raise ValueError("PairTrace frozen teacher requires nine input channels")
    coefficient_sum = sum(
        float(value)
        for value in config["model"]["teacher_conv1_coefficients"].values()
    )
    if not math.isclose(coefficient_sum, 1.0, abs_tol=1e-12):
        raise ValueError("PairTrace teacher conv1 coefficients must sum to one")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("PairTrace pilot protocol SHA-256 changed")
    manifest_path = _resolve(project_root, data_config["manifest"])
    if _sha256(manifest_path) != data_config["expected_manifest_sha256"]:
        raise ValueError("PairTrace pilot manifest SHA-256 changed")
    if not teacher_only:
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
    expected_train, expected_validation = _expected_role_counts(data_config)
    if len(train_rows) != expected_train or len(validation_rows) != expected_validation:
        raise ValueError("PairTrace train/validation role counts changed")
    if {row["source_group_id"] for row in train_rows} & {
        row["source_group_id"] for row in validation_rows
    }:
        raise ValueError("PairTrace train/validation source groups overlap")
    if data_config.get("max_train_pairs") is not None:
        train_rows = train_rows[: int(data_config["max_train_pairs"])]
    if data_config.get("max_validation_pairs") is not None:
        validation_rows = validation_rows[: int(data_config["max_validation_pairs"])]

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    pair_cache_dir = _resolve(scratch, paths["pair_cache_dir"])
    score_cache_dir = _resolve(scratch, paths["score_cache_dir"])
    checkpoint_path = _resolve(project_root, paths["checkpoint"])
    teacher_checkpoint_path = _resolve(project_root, paths["teacher_checkpoint"])
    teacher_log_path = _resolve(project_root, paths["teacher_epoch_log"])
    student_log_path = _resolve(project_root, paths["student_epoch_log"])
    predictions_path = _resolve(project_root, paths["prediction_records"])
    metrics_path = _resolve(project_root, paths["metrics"])
    summary_path = _resolve(project_root, paths["summary"])
    run_log_path = _resolve(project_root, paths["log"])
    for path in (
        pair_cache_dir,
        score_cache_dir,
        checkpoint_path.parent,
        teacher_checkpoint_path.parent,
        teacher_log_path.parent,
        student_log_path.parent,
        predictions_path.parent,
        metrics_path.parent,
        summary_path.parent,
        run_log_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=run_log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    started = time.monotonic()
    cache_hits = 0
    train_cache: list[dict[str, Any]] = []
    validation_cache: list[dict[str, Any]] = []
    cache_selections = [(train_cache, train_rows)]
    if not teacher_only:
        cache_selections.append((validation_cache, validation_rows))
    for target, selected_rows in cache_selections:
        for row in selected_rows:
            record, hit = _prepare_pair_cache(
                row, scratch, pair_cache_dir, config["preprocessing"]
            )
            target.append(record)
            cache_hits += int(hit)

    shuffle_seed = int(config["experiment"].get("shuffle_seed", 20260721))
    shuffled_authentic = _shuffled_authentic_map(train_cache, shuffle_seed)
    weights_path = _resolve(scratch, config["model"]["encoder_weights"])
    weights_sha256 = _sha256(weights_path)
    if weights_sha256 != config["model"]["encoder_weights_sha256"]:
        raise ValueError("PairTrace ResNet-18 initialization SHA-256 changed")
    # Construct the student first so its randomly initialized decoder exactly
    # matches the frozen single-image control under the shared seed.
    student = _load_student(weights_path).to(device)
    teacher = _load_teacher(
        weights_path, config["model"]["teacher_conv1_coefficients"]
    ).to(device)
    torch.cuda.reset_peak_memory_stats(device)

    teacher_training = config["teacher_training"]
    teacher_dataset = _PairTraceDataset(
        train_cache,
        "teacher",
        pair_mode,
        shuffled_authentic,
        config["sampling"],
        config["preprocessing"],
        seed,
        int(teacher_training["steps_per_epoch"])
        * int(teacher_training["batch_size"]),
    )
    teacher_loader = DataLoader(
        teacher_dataset,
        batch_size=int(teacher_training["batch_size"]),
        shuffle=False,
        num_workers=int(teacher_training["num_workers"]),
        pin_memory=True,
        persistent_workers=False,
    )
    teacher_optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=float(teacher_training["learning_rate"]),
        weight_decay=float(teacher_training["weight_decay"]),
    )
    teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        teacher_optimizer, T_max=int(teacher_training["epochs"])
    )
    teacher_scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(teacher_training["amp"])
    )
    teacher_positive_weight = torch.tensor(
        float(teacher_training["bce_positive_weight"]), device=device
    )
    teacher_epoch_records: list[dict[str, Any]] = []
    for epoch in range(int(teacher_training["epochs"])):
        teacher_dataset.set_epoch(epoch)
        teacher.train()
        losses: list[float] = []
        for _, teacher_inputs, masks, _ in teacher_loader:
            teacher_inputs = teacher_inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            teacher_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=bool(teacher_training["amp"]),
            ):
                logits = teacher(teacher_inputs)
                bce = F.binary_cross_entropy_with_logits(
                    logits, masks, pos_weight=teacher_positive_weight
                )
                dice = _dice_loss(logits, masks)
                loss = float(teacher_training["bce_loss_weight"]) * bce + float(
                    teacher_training["dice_loss_weight"]
                ) * dice
            teacher_scaler.scale(loss).backward()
            teacher_scaler.unscale_(teacher_optimizer)
            torch.nn.utils.clip_grad_norm_(
                teacher.parameters(),
                float(teacher_training["gradient_clip_norm"]),
            )
            teacher_scaler.step(teacher_optimizer)
            teacher_scaler.update()
            losses.append(float(loss.detach().cpu()))
        teacher_scheduler.step()
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "learning_rate": float(teacher_optimizer.param_groups[0]["lr"]),
            "pair_mode": pair_mode,
            "paper_evidence": False,
        }
        teacher_epoch_records.append(record)
        _write_jsonl(teacher_log_path, teacher_epoch_records)
        logging.info("teacher epoch=%d metrics=%s", epoch + 1, record)

    _save_checkpoint(
        teacher_checkpoint_path,
        {
            "model_state": teacher.state_dict(),
            "epochs": int(teacher_training["epochs"]),
            "pair_mode": pair_mode,
            "config_sha256": _sha256(config_path),
            "protocol_sha256": _sha256(protocol_path),
            "encoder_weights_sha256": weights_sha256,
            "architecture": config["model"]["architecture"],
            "teacher_input_channels": int(config["model"]["teacher_input_channels"]),
        },
    )
    teacher_checkpoint_sha256 = _sha256(teacher_checkpoint_path)
    teacher.eval().requires_grad_(False)

    if teacher_only:
        summary = {
            "experiment": config["experiment"],
            "status": "teacher_training_complete",
            "paper_evidence": False,
            "gpu_used": True,
            "teacher_only": True,
            "pair_mode": pair_mode,
            "viewed_diagnostic_read": False,
            "final_reserve_read": False,
            "protocol": str(protocol_path.relative_to(project_root)),
            "protocol_sha256": _sha256(protocol_path),
            "config_sha256": _sha256(config_path),
            "input_manifest_sha256": _sha256(manifest_path),
            "train_pairs": len(train_cache),
            "validation_pairs_read": 0,
            "pair_cache_hits": cache_hits,
            "teacher_epochs": teacher_epoch_records,
            "teacher_checkpoint": str(
                teacher_checkpoint_path.relative_to(project_root)
            ),
            "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
            "wall_time_seconds": time.monotonic() - started,
            "peak_vram_mb": float(
                torch.cuda.max_memory_allocated(device) / 1024**2
            ),
            "gpu": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "outputs": {
                "teacher_epoch_log": str(
                    teacher_log_path.relative_to(project_root)
                ),
                "teacher_epoch_log_sha256": _sha256(teacher_log_path),
                "log": str(run_log_path.relative_to(project_root)),
            },
        }
        _write_json(summary_path, summary)
        return summary

    student_training = config["student_training"]
    student_dataset = _PairTraceDataset(
        train_cache,
        "student",
        pair_mode,
        shuffled_authentic,
        config["sampling"],
        config["preprocessing"],
        seed,
        int(student_training["steps_per_epoch"])
        * int(student_training["batch_size"]),
    )
    student_loader = DataLoader(
        student_dataset,
        batch_size=int(student_training["batch_size"]),
        shuffle=False,
        num_workers=int(student_training["num_workers"]),
        pin_memory=True,
        persistent_workers=False,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(student_training["learning_rate"]),
        weight_decay=float(student_training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(student_training["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(student_training["amp"]))
    positive_weight = torch.tensor(
        float(student_training["bce_positive_weight"]), device=device
    )
    distillation = config["distillation"]
    student_epoch_records: list[dict[str, Any]] = []
    best_ap = -math.inf
    best_epoch = -1
    for epoch in range(int(student_training["epochs"])):
        student_dataset.set_epoch(epoch)
        student.train()
        losses: list[float] = []
        direct_losses: list[float] = []
        logit_losses: list[float] = []
        feature_losses: list[float] = []
        for student_inputs, teacher_inputs, masks, active in student_loader:
            student_inputs = student_inputs.to(device, non_blocking=True)
            teacher_inputs = teacher_inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            active = active.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=bool(student_training["amp"]),
            ):
                teacher_logits, teacher_features = teacher.forward_with_features(
                    teacher_inputs
                )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=bool(student_training["amp"]),
            ):
                student_logits, student_features = student.forward_with_features(
                    student_inputs
                )
                bce = F.binary_cross_entropy_with_logits(
                    student_logits, masks, pos_weight=positive_weight
                )
                dice = _dice_loss(student_logits, masks)
                direct = float(student_training["bce_loss_weight"]) * bce + float(
                    student_training["dice_loss_weight"]
                ) * dice
                logit_distillation = _logit_distillation_loss(
                    student_logits,
                    teacher_logits,
                    active,
                    float(distillation["temperature"]),
                )
                feature_distillation = _feature_distillation_loss(
                    student_features,
                    teacher_features,
                    masks,
                    active,
                    list(distillation["feature_names"]),
                    float(distillation["edited_pixel_weight"]),
                )
                loss = (
                    direct
                    + float(distillation["logit_weight"]) * logit_distillation
                    + float(distillation["feature_weight"]) * feature_distillation
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                float(student_training["gradient_clip_norm"]),
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            direct_losses.append(float(direct.detach().cpu()))
            logit_losses.append(float(logit_distillation.detach().cpu()))
            feature_losses.append(float(feature_distillation.detach().cpu()))
        scheduler.step()

        validation_aps: list[float] = []
        for record in validation_cache:
            image = np.asarray(np.load(record["forged"], mmap_mode="r"))
            mask = np.asarray(np.load(record["mask"], mmap_mode="r")).astype(bool)
            probability = _infer_tiled(
                student,
                image,
                device,
                student_training,
                config["preprocessing"],
            )
            average_precision, _ = _ranking_metrics(probability, mask)
            validation_aps.append(average_precision)
        checkpoint_group_field = student_training.get("checkpoint_group_field")
        checkpoint_ap, validation_ap_by_group = _grouped_mean(
            validation_aps, validation_cache, checkpoint_group_field
        )
        validation_ap = float(np.mean(validation_aps))
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "direct_loss": float(np.mean(direct_losses)),
            "logit_distillation_loss": float(np.mean(logit_losses)),
            "feature_distillation_loss": float(np.mean(feature_losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_forged_document_macro_pixel_ap_model_resolution": validation_ap,
            "checkpoint_selection_macro_pixel_ap_model_resolution": checkpoint_ap,
            "checkpoint_group_field": checkpoint_group_field,
            "checkpoint_group_macro_pixel_ap_model_resolution": validation_ap_by_group,
            "pair_mode": pair_mode,
            "paper_evidence": False,
        }
        student_epoch_records.append(record)
        _write_jsonl(student_log_path, student_epoch_records)
        logging.info("student epoch=%d metrics=%s", epoch + 1, record)
        if checkpoint_ap > best_ap:
            best_ap = checkpoint_ap
            best_epoch = epoch + 1
            _save_checkpoint(
                checkpoint_path,
                {
                    "model_state": student.state_dict(),
                    "epoch": best_epoch,
                    "validation_macro_pixel_ap_model_resolution": best_ap,
                    "checkpoint_group_field": checkpoint_group_field,
                    "pair_mode": pair_mode,
                    "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
                    "config_sha256": _sha256(config_path),
                    "protocol_sha256": _sha256(protocol_path),
                    "encoder_weights_sha256": weights_sha256,
                    "seed": seed,
                    "architecture": config["model"]["architecture"],
                },
            )

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    student.load_state_dict(saved["model_state"], strict=True)
    student = student.to(device).eval()
    checkpoint_sha256 = _sha256(checkpoint_path)
    metric_row, prediction_records = _native_validation(
        student,
        validation_cache,
        scratch,
        score_cache_dir,
        checkpoint_sha256,
        device,
        student_training,
        config["preprocessing"],
        config["operating_point"],
    )
    metric_row["best_epoch"] = best_epoch
    metric_row["pair_mode"] = pair_mode
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
        "pairtrace_training_authorized": True,
        "viewed_diagnostic_read": False,
        "final_reserve_read": False,
        "viewed_diagnostic_expected_sha256": data_config[
            "expected_viewed_diagnostic_sha256"
        ],
        "final_reserve_expected_sha256": data_config["expected_final_reserve_sha256"],
        "final_reserve_expected_freeze_id": data_config[
            "expected_final_reserve_freeze_id"
        ],
        "protocol": str(protocol_path.relative_to(project_root)),
        "protocol_sha256": _sha256(protocol_path),
        "config_sha256": _sha256(config_path),
        "input_manifest_sha256": _sha256(manifest_path),
        "pair_mode": pair_mode,
        "train_pairs": len(train_cache),
        "validation_pairs": len(validation_cache),
        "pair_cache_hits": cache_hits,
        "teacher_epochs": teacher_epoch_records,
        "student_epochs": student_epoch_records,
        "best_epoch": best_epoch,
        "best_validation_macro_pixel_ap_model_resolution": best_ap,
        "teacher_checkpoint": str(teacher_checkpoint_path.relative_to(project_root)),
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "checkpoint": str(checkpoint_path.relative_to(project_root)),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_metrics_native_geometry": metric_row,
        "individual_success": {
            "native_macro_pixel_ap_min": float(success["native_macro_pixel_ap_min"]),
            "native_pixel_iou_min": float(success["native_pixel_iou_min"]),
            "authentic_pixel_fpr_max": float(success["authentic_pixel_fpr_max"]),
            "passed": individual_success,
        },
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "outputs": {
            "teacher_epoch_log": str(teacher_log_path.relative_to(project_root)),
            "teacher_epoch_log_sha256": _sha256(teacher_log_path),
            "student_epoch_log": str(student_log_path.relative_to(project_root)),
            "student_epoch_log_sha256": _sha256(student_log_path),
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "log": str(run_log_path.relative_to(project_root)),
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
