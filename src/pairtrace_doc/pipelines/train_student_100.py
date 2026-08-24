from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import time
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


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
        raise ValueError("cannot write an empty student metric table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = 8 if output_channels >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ResNet18UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        encoder = resnet18(weights=None)
        self.encoder = encoder
        self.decoder4 = _ConvBlock(512 + 256, 256)
        self.decoder3 = _ConvBlock(256 + 128, 128)
        self.decoder2 = _ConvBlock(128 + 64, 64)
        self.decoder1 = _ConvBlock(64 + 64, 64)
        self.output = nn.Sequential(
            _ConvBlock(64, 32),
            nn.Conv2d(32, 1, 1),
        )

    @staticmethod
    def _upsample(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        stem = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(value)))
        layer1 = self.encoder.layer1(self.encoder.maxpool(stem))
        layer2 = self.encoder.layer2(layer1)
        layer3 = self.encoder.layer3(layer2)
        layer4 = self.encoder.layer4(layer3)
        value = self.decoder4(torch.cat([self._upsample(layer4, layer3), layer3], dim=1))
        value = self.decoder3(torch.cat([self._upsample(value, layer2), layer2], dim=1))
        value = self.decoder2(torch.cat([self._upsample(value, layer1), layer1], dim=1))
        value = self.decoder1(torch.cat([self._upsample(value, stem), stem], dim=1))
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.output(value)


def _pair_cache_key(row: dict[str, Any], preprocessing: dict[str, Any]) -> str:
    payload = {
        "authentic_sha256": row["authentic_sha256"],
        "image_sha256": row["image_sha256"],
        "mask_sha256": row["mask_sha256"],
        "preprocessing": preprocessing,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resize_triplet(
    forged: np.ndarray,
    authentic: np.ndarray,
    mask: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    if forged.shape != authentic.shape or forged.shape[:2] != mask.shape:
        raise ValueError("student pair cache inputs are not aligned")
    if max(height, width) <= max_side:
        return forged, authentic, mask
    scale = max_side / max(height, width)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return (
        cv2.resize(forged, target, interpolation=cv2.INTER_AREA),
        cv2.resize(authentic, target, interpolation=cv2.INTER_AREA),
        cv2.resize(mask, target, interpolation=cv2.INTER_NEAREST),
    )


def _prepare_pair_cache(
    row: dict[str, Any],
    scratch: Path,
    cache_dir: Path,
    preprocessing: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    cache_key = _pair_cache_key(row, preprocessing)
    directory = cache_dir / cache_key
    paths = {
        "forged": directory / "forged.npy",
        "authentic": directory / "authentic.npy",
        "mask": directory / "mask.npy",
        "meta": directory / "meta.json",
    }
    cache_hit = all(path.is_file() for path in paths.values())
    if not cache_hit:
        image_path = _resolve(scratch, row["image"])
        authentic_path = _resolve(scratch, row["authentic"])
        mask_path = _resolve(scratch, row["mask"])
        if _sha256(image_path) != row["image_sha256"]:
            raise ValueError("student forged input SHA-256 changed")
        if _sha256(authentic_path) != row["authentic_sha256"]:
            raise ValueError("student authentic input SHA-256 changed")
        if _sha256(mask_path) != row["mask_sha256"]:
            raise ValueError("student mask input SHA-256 changed")
        with Image.open(image_path) as handle:
            forged = np.asarray(handle.convert("RGB"))
        with Image.open(authentic_path) as handle:
            authentic = np.asarray(handle.convert("RGB"))
        with Image.open(mask_path) as handle:
            mask = (np.asarray(handle.convert("L")) > 0).astype(np.uint8)
        forged, authentic, mask = _resize_triplet(
            forged, authentic, mask, int(preprocessing["max_side"])
        )
        positive_y, positive_x = np.nonzero(mask)
        if not len(positive_y):
            raise ValueError("student training mask became empty after resize")
        bbox = [
            int(positive_x.min()),
            int(positive_y.min()),
            int(positive_x.max()) + 1,
            int(positive_y.max()) + 1,
        ]
        directory.mkdir(parents=True, exist_ok=True)
        temporary_paths = {
            name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()
        }
        for name, array in (("forged", forged), ("authentic", authentic), ("mask", mask)):
            with temporary_paths[name].open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            temporary_paths[name].replace(paths[name])
        meta = {
            "cache_key": cache_key,
            "source_group_id": row["source_group_id"],
            "shape": list(mask.shape),
            "bbox_xyxy": bbox,
            "native_shape": [int(row["image_height"]), int(row["image_width"])],
        }
        temporary_paths["meta"].write_text(
            json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_paths["meta"].replace(paths["meta"])
    with paths["meta"].open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return {
        "source_group_id": row["source_group_id"],
        "sample_id": row["sample_id"],
        "role": row["pilot_role"],
        "cache_key": cache_key,
        "directory": str(directory),
        "forged": str(paths["forged"]),
        "authentic": str(paths["authentic"]),
        "mask": str(paths["mask"]),
        "bbox_xyxy": meta["bbox_xyxy"],
        "shape": meta["shape"],
        "native_shape": meta["native_shape"],
        "source_row": row,
    }, cache_hit


def _expected_role_counts(data_config: dict[str, Any]) -> tuple[int, int]:
    if "expected_per_role" in data_config:
        expected = int(data_config["expected_per_role"])
        return expected, expected
    return (
        int(data_config["expected_train_records"]),
        int(data_config["expected_validation_records"]),
    )


def _generator_sampling_pools(
    records: list[dict[str, Any]], sampling: dict[str, Any]
) -> tuple[list[str], np.ndarray, dict[str, list[dict[str, Any]]]] | None:
    configured = sampling.get("generator_probabilities")
    if configured is None:
        return None
    probabilities = {str(key): float(value) for key, value in configured.items()}
    if not probabilities or not math.isclose(
        sum(probabilities.values()), 1.0, abs_tol=1e-12
    ):
        raise ValueError("generator sampling probabilities must sum to one")
    if any(value <= 0.0 for value in probabilities.values()):
        raise ValueError("generator sampling probabilities must be positive")
    field = str(sampling.get("generator_field", "assigned_tool"))
    pools: dict[str, list[dict[str, Any]]] = {
        generator: [] for generator in probabilities
    }
    for record in records:
        source = record.get("source_row", record)
        generator = str(source.get(field, "<missing>"))
        if generator in pools:
            pools[generator].append(record)
    missing = [generator for generator, values in pools.items() if not values]
    if missing:
        raise ValueError(f"generator sampling pools are empty: {missing}")
    generators = list(probabilities)
    return (
        generators,
        np.asarray([probabilities[name] for name in generators], dtype=float),
        pools,
    )


def _grouped_mean(
    values: list[float],
    records: list[dict[str, Any]],
    field: str | None,
) -> tuple[float, dict[str, float]]:
    overall = float(np.mean(values))
    if field is None:
        return overall, {}
    grouped: dict[str, list[float]] = {}
    for value, record in zip(values, records, strict=True):
        source = record.get("source_row", record)
        group = str(source.get(field, "<missing>"))
        grouped.setdefault(group, []).append(float(value))
    per_group = {
        group: float(np.mean(group_values))
        for group, group_values in sorted(grouped.items())
    }
    return float(np.mean(list(per_group.values()))), per_group


class _PatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        sampling: dict[str, Any],
        preprocessing: dict[str, Any],
        seed: int,
        length: int,
    ) -> None:
        self.records = records
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

    @staticmethod
    def _pad(
        image: np.ndarray, mask: np.ndarray, crop_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
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
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        if self.generator_sampling is None:
            record = self.records[int(rng.integers(0, len(self.records)))]
        else:
            generators, probabilities, pools = self.generator_sampling
            generator = str(rng.choice(generators, p=probabilities))
            pool = pools[generator]
            record = pool[int(rng.integers(0, len(pool)))]
        probability = float(rng.random())
        positive_limit = float(self.sampling["forged_positive_probability"])
        forged_limit = positive_limit + float(self.sampling["forged_random_probability"])
        if probability < forged_limit:
            image = np.load(record["forged"], mmap_mode="r")
            mask = np.load(record["mask"], mmap_mode="r")
            positive_crop = probability < positive_limit
        else:
            image = np.load(record["authentic"], mmap_mode="r")
            source_mask = np.load(record["mask"], mmap_mode="r")
            mask = np.zeros(source_mask.shape, dtype=np.uint8)
            positive_crop = False
        crop_size = int(self.preprocessing["crop_size"])
        image_array, mask_array = self._pad(np.asarray(image), np.asarray(mask), crop_size)
        height, width = mask_array.shape
        if positive_crop:
            x1, y1, x2, y2 = record["bbox_xyxy"]
            center_x = int(rng.integers(x1, max(x1 + 1, x2)))
            center_y = int(rng.integers(y1, max(y1 + 1, y2)))
            left = int(np.clip(center_x - int(rng.integers(0, crop_size)), 0, width - crop_size))
            top = int(np.clip(center_y - int(rng.integers(0, crop_size)), 0, height - crop_size))
        else:
            left = int(rng.integers(0, width - crop_size + 1))
            top = int(rng.integers(0, height - crop_size + 1))
        image_crop = np.array(
            image_array[top : top + crop_size, left : left + crop_size], copy=True
        )
        mask_crop = np.array(
            mask_array[top : top + crop_size, left : left + crop_size], copy=True
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
            image_crop = np.clip(image_crop.astype(np.float32) * contrast + brightness, 0, 255).astype(
                np.uint8
            )
        if rng.random() < float(self.sampling["jpeg_probability"]):
            quality = int(
                rng.integers(
                    int(self.sampling["jpeg_quality_min"]),
                    int(self.sampling["jpeg_quality_max"]) + 1,
                )
            )
            buffer = io.BytesIO()
            Image.fromarray(image_crop).save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as handle:
                image_crop = np.asarray(handle.convert("RGB"))
        image_float = image_crop.astype(np.float32) / 255.0
        mean = np.asarray(self.preprocessing["imagenet_mean"], dtype=np.float32)
        std = np.asarray(self.preprocessing["imagenet_std"], dtype=np.float32)
        image_float = (image_float - mean) / std
        return (
            torch.from_numpy(image_float.transpose(2, 0, 1).copy()),
            torch.from_numpy(mask_crop.astype(np.float32, copy=False)).unsqueeze(0),
        )


def _ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    flat_labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    positives = int(flat_labels.sum())
    negatives = int(flat_labels.size - positives)
    if not positives or not negatives:
        raise ValueError("student ranking metrics require both pixel classes")
    order = np.argsort(flat_scores, kind="mergesort")[::-1]
    ranked_scores = flat_scores[order]
    ranked_labels = flat_labels[order]
    tp = np.cumsum(ranked_labels, dtype=np.int64)
    fp = np.cumsum(1 - ranked_labels, dtype=np.int64)
    ends = np.r_[ranked_scores[1:] != ranked_scores[:-1], True]
    recall = tp[ends] / positives
    precision = tp[ends] / (tp[ends] + fp[ends])
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    auroc = float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fp[ends] / negatives]))
    return average_precision, auroc


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def _infer_tiled(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    training: dict[str, Any],
    preprocessing: dict[str, Any],
) -> np.ndarray:
    tile = int(training["validation_tile_size"])
    stride = int(training["validation_tile_stride"])
    batch_size = int(training["validation_tile_batch_size"])
    height, width = image.shape[:2]
    pad_height = max(0, tile - height)
    pad_width = max(0, tile - width)
    padded = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect")
    padded_height, padded_width = padded.shape[:2]
    coordinates = [
        (top, left)
        for top in _positions(padded_height, tile, stride)
        for left in _positions(padded_width, tile, stride)
    ]
    accumulator = np.zeros((padded_height, padded_width), dtype=np.float32)
    counts = np.zeros((padded_height, padded_width), dtype=np.float32)
    mean = np.asarray(preprocessing["imagenet_mean"], dtype=np.float32)
    std = np.asarray(preprocessing["imagenet_std"], dtype=np.float32)
    model.eval()
    for start in range(0, len(coordinates), batch_size):
        selected = coordinates[start : start + batch_size]
        patches = np.stack(
            [padded[top : top + tile, left : left + tile] for top, left in selected]
        ).astype(np.float32) / 255.0
        patches = (patches - mean) / std
        tensor = torch.from_numpy(patches.transpose(0, 3, 1, 2).copy()).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])
        ):
            probabilities = torch.sigmoid(model(tensor)).squeeze(1).float().cpu().numpy()
        for probability, (top, left) in zip(probabilities, selected):
            accumulator[top : top + tile, left : left + tile] += probability
            counts[top : top + tile, left : left + tile] += 1.0
    return (accumulator / np.maximum(counts, 1.0))[:height, :width]


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _threshold_vectors(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    bins = np.r_[thresholds, np.inf]
    positive, _ = np.histogram(scores[labels], bins=bins)
    negative, _ = np.histogram(scores[~labels], bins=bins)
    return (
        np.cumsum(positive[::-1], dtype=np.int64)[::-1],
        np.cumsum(negative[::-1], dtype=np.int64)[::-1],
        int(np.count_nonzero(labels)),
    )


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for name, value in config["runtime"].get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not config["runtime"]["gpu_launch_authorized"]:
        raise ValueError("student GPU pilot was not explicitly authorized")
    if config["runtime"]["pairtrace_training_authorized"]:
        raise ValueError("matched student config must not authorize PairTrace training")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("matched student pilot cannot be paper evidence")
    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("matched student pilot requires CUDA")

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    expected_protocol_sha256 = config["experiment"].get(
        "expected_protocol_sha256"
    )
    if expected_protocol_sha256 and _sha256(protocol_path) != expected_protocol_sha256:
        raise ValueError("matched student protocol SHA-256 changed")
    manifest_path = _resolve(project_root, config["data"]["manifest"])
    if _sha256(manifest_path) != config["data"]["expected_manifest_sha256"]:
        raise ValueError("student pilot manifest SHA-256 changed")
    # The output-unseen manifest is intentionally not opened by this process.
    if not config["data"]["training_must_not_read_output_unseen_holdout"]:
        raise ValueError("student training must keep the new holdout unread")
    rows = _read_jsonl(manifest_path)
    train_rows = sorted(
        (row for row in rows if row["pilot_role"] == config["data"]["train_role"]),
        key=lambda row: str(row["source_group_id"]),
    )
    validation_rows = sorted(
        (row for row in rows if row["pilot_role"] == config["data"]["validation_role"]),
        key=lambda row: str(row["source_group_id"]),
    )
    expected_train, expected_validation = _expected_role_counts(config["data"])
    if len(train_rows) != expected_train or len(validation_rows) != expected_validation:
        raise ValueError("student train/validation role counts changed")
    if {row["source_group_id"] for row in train_rows} & {
        row["source_group_id"] for row in validation_rows
    }:
        raise ValueError("student train/validation source groups overlap")
    if config["data"].get("max_train_pairs") is not None:
        train_rows = train_rows[: int(config["data"]["max_train_pairs"])]
    if config["data"].get("max_validation_pairs") is not None:
        validation_rows = validation_rows[: int(config["data"]["max_validation_pairs"])]

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
    predictions_path = _resolve(project_root, paths["prediction_records"])
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
        raise ValueError("ResNet-18 initialization SHA-256 changed")
    model = ResNet18UNet()
    encoder_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(encoder_state, strict=True)
    model = model.to(device)
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
    dataset = _PatchDataset(
        train_cache,
        config["sampling"],
        config["preprocessing"],
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

    torch.cuda.reset_peak_memory_stats(device)
    epoch_records: list[dict[str, Any]] = []
    best_ap = -math.inf
    best_epoch = -1
    for epoch in range(int(training["epochs"])):
        dataset.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        bce_losses: list[float] = []
        dice_losses: list[float] = []
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])
            ):
                logits = model(images)
                bce = F.binary_cross_entropy_with_logits(
                    logits, masks, pos_weight=positive_weight
                )
                dice = _dice_loss(logits, masks)
                loss = float(training["bce_loss_weight"]) * bce + float(
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
            bce_losses.append(float(bce.detach().cpu()))
            dice_losses.append(float(dice.detach().cpu()))
        scheduler.step()

        validation_aps: list[float] = []
        for record in validation_cache:
            image = np.load(record["forged"], mmap_mode="r")
            mask = np.load(record["mask"], mmap_mode="r").astype(bool)
            probability = _infer_tiled(
                model, np.asarray(image), device, training, config["preprocessing"]
            )
            average_precision, _ = _ranking_metrics(probability, mask)
            validation_aps.append(average_precision)
        checkpoint_group_field = training.get("checkpoint_group_field")
        checkpoint_ap, validation_ap_by_group = _grouped_mean(
            validation_aps, validation_cache, checkpoint_group_field
        )
        validation_ap = float(np.mean(validation_aps))
        epoch_record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "bce_loss": float(np.mean(bce_losses)),
            "dice_loss": float(np.mean(dice_losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_forged_document_macro_pixel_ap_model_resolution": validation_ap,
            "checkpoint_selection_macro_pixel_ap_model_resolution": checkpoint_ap,
            "checkpoint_group_field": checkpoint_group_field,
            "checkpoint_group_macro_pixel_ap_model_resolution": validation_ap_by_group,
            "paper_evidence": False,
        }
        epoch_records.append(epoch_record)
        _write_jsonl(epoch_log_path, epoch_records)
        logging.info("epoch=%d metrics=%s", epoch + 1, epoch_record)
        if checkpoint_ap > best_ap:
            best_ap = checkpoint_ap
            best_epoch = epoch + 1
            _save_checkpoint(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "epoch": best_epoch,
                    "validation_macro_pixel_ap_model_resolution": best_ap,
                    "checkpoint_group_field": checkpoint_group_field,
                    "config_sha256": _sha256(config_path),
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
    operating = config["operating_point"]
    start = float(operating["candidate_min"])
    stop = float(operating["candidate_max"])
    step = float(operating["candidate_step"])
    thresholds = start + np.arange(int(round((stop - start) / step)) + 1) * step
    forged_vectors: list[tuple[np.ndarray, np.ndarray, int]] = []
    authentic_fpr_vectors: list[np.ndarray] = []
    native_aps: list[float] = []
    native_aurocs: list[float] = []
    prediction_records: list[dict[str, Any]] = []
    for record in validation_cache:
        source = record["source_row"]
        native_shape = (int(source["image_height"]), int(source["image_width"]))
        mask_path = _resolve(scratch, source["mask"])
        if _sha256(mask_path) != source["mask_sha256"]:
            raise ValueError("validation mask SHA-256 changed")
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0
        for sample_kind in ("forged", "authentic"):
            image = np.load(record[sample_kind], mmap_mode="r")
            probability = _infer_tiled(
                model, np.asarray(image), device, training, config["preprocessing"]
            )
            score_key = hashlib.sha256(
                json.dumps(
                    {
                        "checkpoint_sha256": checkpoint_sha256,
                        "input_sha256": source[
                            "image_sha256" if sample_kind == "forged" else "authentic_sha256"
                        ],
                        "preprocessing": config["preprocessing"],
                        "sample_kind": sample_kind,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            score_path = score_cache_dir / f"{score_key}.npz"
            with score_path.with_suffix(".npz.tmp").open("wb") as handle:
                np.savez_compressed(handle, scores=probability.astype(np.float16))
            score_path.with_suffix(".npz.tmp").replace(score_path)
            native_probability = cv2.resize(
                probability,
                (native_shape[1], native_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            prediction_record: dict[str, Any] = {
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
                average_precision, auroc = _ranking_metrics(native_probability, native_mask)
                native_aps.append(average_precision)
                native_aurocs.append(auroc)
                forged_vectors.append(
                    _threshold_vectors(native_probability, native_mask, thresholds)
                )
                prediction_record["macro_pixel_ap"] = average_precision
                prediction_record["pixel_auroc"] = auroc
            else:
                histogram, _ = np.histogram(
                    native_probability, bins=np.r_[thresholds, np.inf]
                )
                predicted = np.cumsum(histogram[::-1], dtype=np.int64)[::-1]
                authentic_fpr_vectors.append(predicted / native_probability.size)
            prediction_records.append(prediction_record)
    _write_jsonl(predictions_path, prediction_records)

    macro_f1 = np.zeros_like(thresholds, dtype=float)
    macro_iou = np.zeros_like(thresholds, dtype=float)
    macro_precision = np.zeros_like(thresholds, dtype=float)
    macro_recall = np.zeros_like(thresholds, dtype=float)
    for tp, fp, positives in forged_vectors:
        fn = positives - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = tp / positives
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fp + fn) > 0)
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
    candidates = feasible[np.isclose(macro_f1[feasible], best_f1, atol=1e-12, rtol=0)]
    best_authentic_fpr = authentic_fpr[candidates].min()
    candidates = candidates[
        np.isclose(authentic_fpr[candidates], best_authentic_fpr, atol=1e-12, rtol=0)
    ]
    selected_index = int(candidates[-1])
    threshold = float(thresholds[selected_index])
    native_generator_macro_ap, native_ap_by_generator = _grouped_mean(
        native_aps,
        validation_cache,
        training.get("checkpoint_group_field"),
    )
    metric_row = {
        "validation_pairs": len(validation_cache),
        "best_epoch": best_epoch,
        "pixel_threshold": threshold,
        "macro_pixel_ap": float(np.mean(native_aps)),
        "generator_macro_pixel_ap": native_generator_macro_ap,
        "pixel_auroc": float(np.mean(native_aurocs)),
        "pixel_precision": float(macro_precision[selected_index]),
        "pixel_recall": float(macro_recall[selected_index]),
        "pixel_f1": float(macro_f1[selected_index]),
        "pixel_iou": float(macro_iou[selected_index]),
        "authentic_pixel_fpr": float(authentic_fpr[selected_index]),
        "paper_evidence": False,
    }
    for generator, value in native_ap_by_generator.items():
        safe_generator = "".join(
            character if character.isalnum() else "_" for character in generator
        ).strip("_")
        metric_row[f"macro_pixel_ap__{safe_generator}"] = value
    _write_csv(metrics_path, [metric_row])
    success = config["success"]
    if "validation_native_generator_macro_pixel_ap_min" in success:
        success_ap_key = "generator_macro_pixel_ap"
        success_ap_min = float(
            success["validation_native_generator_macro_pixel_ap_min"]
        )
    else:
        success_ap_key = "macro_pixel_ap"
        success_ap_min = float(success["validation_native_macro_pixel_ap_min"])
    success_pass = bool(
        metric_row[success_ap_key] >= success_ap_min
        and metric_row["authentic_pixel_fpr"]
        <= float(success["validation_authentic_macro_pixel_fpr_max"]) + 1e-12
    )
    summary = {
        "experiment": config["experiment"],
        "status": "passed" if success_pass else "completed_success_criteria_not_met",
        "paper_evidence": False,
        "gpu_used": True,
        "pairtrace_training_authorized": False,
        "output_unseen_holdout_read": False,
        "output_unseen_holdout_expected_sha256": config["data"][
            "expected_output_unseen_holdout_sha256"
        ],
        "protocol": str(protocol_path.relative_to(project_root)),
        "protocol_sha256": _sha256(protocol_path),
        "input_manifest_sha256": _sha256(manifest_path),
        "train_pairs": len(train_cache),
        "validation_pairs": len(validation_cache),
        "pair_cache_hits": cache_hits,
        "best_epoch": best_epoch,
        "best_validation_macro_pixel_ap_model_resolution": best_ap,
        "checkpoint_group_field": training.get("checkpoint_group_field"),
        "native_macro_pixel_ap_by_checkpoint_group": native_ap_by_generator,
        "checkpoint": str(checkpoint_path.relative_to(project_root)),
        "checkpoint_sha256": checkpoint_sha256,
        "encoder_weights_sha256": weights_sha256,
        "validation_metrics_native_geometry": metric_row,
        "success": {
            "primary_ap_metric": success_ap_key,
            "primary_ap_min": success_ap_min,
            "validation_authentic_macro_pixel_fpr_max": float(
                success["validation_authentic_macro_pixel_fpr_max"]
            ),
            "passed": success_pass,
        },
        "epochs": epoch_records,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
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
