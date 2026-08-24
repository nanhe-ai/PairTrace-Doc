from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
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
    _warp_reference,
)
from pairtrace_doc.pipelines.evaluate_baselines_100 import _roc_auc
from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
    _infer_pair_tiled,
    _jpeg_roundtrip,
    _resize_image,
    _resize_reference,
)
from pairtrace_doc.pipelines.freeze_resampling_multiseed_image_thresholds import (
    _top_fraction_mean,
)
from pairtrace_doc.pipelines.run_spatial_lpips import (
    _cache_key,
    _select_round_robin,
)
from pairtrace_doc.pipelines.train_pairtrace_100 import _load_teacher
from pairtrace_doc.pipelines.train_student_100 import (
    _ranking_metrics,
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)


FAMILIES = (
    "translation",
    "rotation",
    "scale",
    "perspective",
    "exposure",
    "white_balance",
    "blur",
    "jpeg",
    "reflection",
    "fold",
    "nonrigid",
)
_RUNTIME_ONLY_FIELDS = frozenset({"cache_hit", "latency_ms"})


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify(path: Path, expected: str, label: str) -> str:
    digest = _sha256(path)
    if digest != expected:
        raise ValueError(f"{label} SHA-256 changed: {digest} != {expected}")
    return digest


def _latent_seed(experiment_seed: int, group: str, family: str) -> int:
    digest = hashlib.sha256(
        f"{experiment_seed}|{group}|{family}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _geometric_homography(
    shape: tuple[int, int],
    family: str,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = shape
    if family == "translation":
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        return np.asarray(
            [
                [1.0, 0.0, severity * np.cos(angle)],
                [0.0, 1.0, severity * np.sin(angle)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if family in {"rotation", "scale"}:
        sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        angle = sign * severity if family == "rotation" else 0.0
        scale = 1.0 + sign * severity if family == "scale" else 1.0
        affine = cv2.getRotationMatrix2D(
            ((width - 1.0) / 2.0, (height - 1.0) / 2.0), angle, scale
        ).astype(np.float64)
        return np.vstack([affine, [0.0, 0.0, 1.0]])
    if family == "perspective":
        source = np.asarray(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            dtype=np.float32,
        )
        directions = rng.uniform(-1.0, 1.0, size=(4, 2))
        destination = source + directions.astype(np.float32) * np.asarray(
            [severity * (width - 1.0), severity * (height - 1.0)],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(source, destination).astype(np.float64)
    raise ValueError(f"unsupported geometric stress family: {family}")


def _reflection_proxy(
    reference: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    height, width = reference.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    center = (
        int(round(rng.uniform(0.25, 0.75) * (width - 1))),
        int(round(rng.uniform(0.25, 0.75) * (height - 1))),
    )
    axes = (
        max(2, int(round(rng.uniform(0.18, 0.32) * width))),
        max(2, int(round(rng.uniform(0.04, 0.10) * height))),
    )
    cv2.ellipse(
        mask,
        center,
        axes,
        float(rng.uniform(-70.0, 70.0)),
        0.0,
        360.0,
        1.0,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    sigma = max(1.0, 0.015 * min(height, width))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha = np.clip(mask[..., None] * severity, 0.0, 1.0)
    output = reference.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _fold_proxy(
    reference: np.ndarray, half_width: float, rng: np.random.Generator
) -> np.ndarray:
    height, width = reference.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = float(rng.uniform(-np.pi, np.pi))
    cosine, sine = np.cos(angle), np.sin(angle)
    u = cosine * (xx - (width - 1.0) / 2.0) + sine * (
        yy - (height - 1.0) / 2.0
    )
    v = -sine * (xx - (width - 1.0) / 2.0) + cosine * (
        yy - (height - 1.0) / 2.0
    )
    offset = float(rng.uniform(-0.15, 0.15) * min(height, width))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    curve = offset + 0.025 * min(height, width) * np.sin(
        2.0 * np.pi * u / max(width, height) + phase
    )
    distance = v - curve
    dark = np.exp(-0.5 * (distance / max(half_width, 1e-6)) ** 2)
    highlight_center = 1.8 * half_width
    highlight = np.exp(
        -0.5 * ((distance - highlight_center) / max(0.8 * half_width, 1e-6)) ** 2
    )
    multiplier = 1.0 - 0.45 * dark + 0.20 * highlight
    output = reference.astype(np.float32) * multiplier[..., None]
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _nonrigid_warp(
    reference: np.ndarray, amplitude: float, rng: np.random.Generator
) -> np.ndarray:
    height, width = reference.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    phase_x = float(rng.uniform(0.0, 2.0 * np.pi))
    phase_y = float(rng.uniform(0.0, 2.0 * np.pi))
    period_x = float(rng.uniform(0.30, 0.45) * max(width, 2))
    period_y = float(rng.uniform(0.30, 0.45) * max(height, 2))
    map_x = xx + amplitude * np.sin(2.0 * np.pi * yy / period_y + phase_x)
    map_y = yy + 0.5 * amplitude * np.sin(
        2.0 * np.pi * xx / period_x + phase_y
    )
    return cv2.remap(
        reference,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _apply_stress(
    reference: np.ndarray,
    condition: dict[str, Any],
    experiment_seed: int,
    group: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if reference.ndim != 3 or reference.shape[2] != 3 or reference.dtype != np.uint8:
        raise ValueError("extended stress requires uint8 HWC RGB input")
    family = str(condition["family"])
    severity = float(condition.get("severity", 0.0))
    latent = _latent_seed(experiment_seed, group, family)
    rng = np.random.default_rng(latent)
    identity = np.eye(3, dtype=np.float64)
    if family == "clean":
        transformed = reference.copy()
        oracle = identity
    elif family in {"translation", "rotation", "scale", "perspective"}:
        oracle = _geometric_homography(reference.shape[:2], family, severity, rng)
        transformed = _warp_reference(reference, oracle, inverse=False)
    elif family == "exposure":
        sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        transformed = np.clip(
            np.rint(reference.astype(np.float32) * (2.0 ** (sign * severity))),
            0,
            255,
        ).astype(np.uint8)
        oracle = identity
    elif family == "white_balance":
        axis = rng.normal(size=3)
        axis -= axis.mean()
        axis /= max(float(np.max(np.abs(axis))), 1e-12)
        gains = 1.0 + severity * axis
        transformed = np.clip(
            np.rint(reference.astype(np.float32) * gains[None, None, :]), 0, 255
        ).astype(np.uint8)
        oracle = identity
    elif family == "blur":
        radius = max(1, int(math.ceil(3.0 * severity)))
        kernel = 2 * radius + 1
        transformed = cv2.GaussianBlur(
            reference,
            (kernel, kernel),
            sigmaX=severity,
            sigmaY=severity,
            borderType=cv2.BORDER_REFLECT_101,
        )
        oracle = identity
    elif family == "jpeg":
        transformed = _jpeg_roundtrip(
            reference,
            {
                "quality": int(round(severity)),
                "subsampling": 2,
                "optimize": False,
                "progressive": False,
            },
        )
        oracle = identity
    elif family == "reflection":
        transformed = _reflection_proxy(reference, severity, rng)
        oracle = identity
    elif family == "fold":
        transformed = _fold_proxy(reference, severity, rng)
        oracle = identity
    elif family == "nonrigid":
        transformed = _nonrigid_warp(reference, severity, rng)
        oracle = identity
    else:
        raise ValueError(f"unsupported extended stress family: {family}")
    if transformed.shape != reference.shape or transformed.dtype != np.uint8:
        raise ValueError("extended stress changed image schema")
    return transformed, oracle, {
        "latent_seed": latent,
        "family": family,
        "level": int(condition["level"]),
        "severity": severity,
        "train_relation": str(condition["train_relation"]),
    }


def _validate_conditions(conditions: list[dict[str, Any]]) -> None:
    if len(conditions) != 23 or len({str(item["name"]) for item in conditions}) != 23:
        raise ValueError("extended stress condition inventory must contain 23 unique names")
    counts = Counter(str(item["family"]) for item in conditions)
    if counts != Counter({"clean": 1, **{family: 2 for family in FAMILIES}}):
        raise ValueError(f"extended stress family inventory changed: {dict(counts)}")
    if any(int(item["level"]) not in ({0} if item["family"] == "clean" else {1, 2}) for item in conditions):
        raise ValueError("extended stress level inventory changed")


def _pixel_operating_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & labels))
    fp = int(np.count_nonzero(predicted & ~labels))
    fn = int(np.count_nonzero(~predicted & labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
    }


def _aggregate(
    payload: dict[str, list[dict[str, Any]]],
    pixel_threshold: float,
    image_threshold: float,
) -> dict[str, Any]:
    forged = payload["forged"]
    authentic = payload["authentic"]
    if not forged or not authentic:
        raise ValueError("extended stress condition aggregation is incomplete")
    by_generator: dict[str, list[float]] = defaultdict(list)
    for item in forged:
        by_generator[str(item["generator"])].append(float(item["macro_pixel_ap"]))
    per_generator = {
        generator: float(np.mean(values))
        for generator, values in sorted(by_generator.items())
    }
    forged_images = np.asarray([item["image_score"] for item in forged])
    authentic_images = np.asarray([item["image_score"] for item in authentic])
    metrics: dict[str, Any] = {
        "development_groups": len(forged),
        "generator_macro_pixel_ap": float(np.mean(list(per_generator.values()))),
        "document_macro_pixel_ap": float(np.mean([item["macro_pixel_ap"] for item in forged])),
        "document_macro_pixel_auroc": float(np.mean([item["pixel_auroc"] for item in forged])),
        "document_macro_pixel_precision": float(np.mean([item["pixel_precision"] for item in forged])),
        "document_macro_pixel_recall": float(np.mean([item["pixel_recall"] for item in forged])),
        "document_macro_pixel_f1": float(np.mean([item["pixel_f1"] for item in forged])),
        "document_macro_pixel_iou": float(np.mean([item["pixel_iou"] for item in forged])),
        "authentic_document_macro_pixel_fpr": float(np.mean([item["pixel_fpr"] for item in authentic])),
        "image_auroc": _roc_auc(
            np.r_[forged_images, authentic_images],
            np.r_[
                np.ones(forged_images.size, dtype=bool),
                np.zeros(authentic_images.size, dtype=bool),
            ],
        ),
        "forged_image_tpr": float(np.mean(forged_images >= image_threshold)),
        "authentic_image_fpr": float(np.mean(authentic_images >= image_threshold)),
        "fixed_pixel_threshold": pixel_threshold,
        "fixed_image_threshold": image_threshold,
        "paper_evidence": False,
    }
    for generator, value in per_generator.items():
        safe = "".join(character if character.isalnum() else "_" for character in generator).strip("_")
        metrics[f"pixel_ap__{safe}"] = value
    return metrics


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _report(summary: dict[str, Any]) -> str:
    rows = []
    for condition in summary["condition_order"]:
        specification = summary["condition_specs"][condition]
        metric = summary["metrics"][condition]
        comparison = summary["clean_comparisons"].get(condition)
        effect = "--" if comparison is None else f"{comparison['effect']:+.4f}"
        interval = (
            "--"
            if comparison is None
            else f"[{comparison['ci_low']:.4f}, {comparison['ci_high']:.4f}]"
        )
        rows.append(
            f"| {condition} | {specification['train_relation']} | "
            f"{metric['generator_macro_pixel_ap']:.4f} | {effect} | {interval} | "
            f"{metric['authentic_document_macro_pixel_fpr']:.4f} | "
            f"{metric['authentic_image_fpr']:.4f} |"
        )
    return f"""# Extended stress curves: development-100

Status: `{summary['status']}`. This is viewed development evidence and is not
an independent or final robustness evaluation.

| Condition | Training relation | Generator-macro AP | AP minus clean | 95% interval | Authentic pixel FPR | Authentic image FPR |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

All {summary['successful_records']} planned records completed with
{summary['failed_records']} failures. The pixel/image operating points were
fixed before this run. Reflection, fold, and nonrigid conditions are synthetic
controlled proxies, not real camera/scan evidence. Negative and non-monotone
curves remain part of the result. No final-reserve read, model training,
checkpoint selection, threshold selection, or method selection occurred.
"""


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("extended stress config must be a mapping")
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if config["experiment"]["paper_evidence"]:
        raise ValueError("extended stress output cannot be paper evidence")
    if not runtime["gpu_launch_authorized"] or not runtime["development_inference_authorized"]:
        raise PermissionError("extended stress inference was not explicitly authorized")
    if any(
        bool(runtime.get(key))
        for key in (
            "model_training_authorized",
            "checkpoint_selection_authorized",
            "threshold_selection_authorized",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("extended stress experiment crossed an evidence boundary")
    stage = str(config["experiment"]["stage"])
    toy = stage == "gpu_toy3_structure_gate"
    development = stage == "gpu_development100"
    if not (toy or development):
        raise ValueError(f"unsupported extended stress stage: {stage}")
    expected_groups = 3 if toy else 100
    if int(runtime["max_groups"]) != expected_groups:
        raise ValueError("extended stress group limit changed")
    device = torch.device(str(runtime["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("extended stress inference requires CUDA")

    experiment = config["experiment"]
    protocol_path = _resolve(project_root, str(experiment["protocol"]))
    _verify(protocol_path, str(experiment["expected_protocol_sha256"]), "extended stress protocol")
    input_config = config["input"]
    manifest_path = _resolve(project_root, str(input_config["manifest"]))
    _verify(manifest_path, str(input_config["expected_manifest_sha256"]), "extended stress manifest")
    all_rows = sorted(_read_jsonl(manifest_path), key=lambda row: str(row["source_group_id"]))
    if len(all_rows) != 100 or len({str(row["source_group_id"]) for row in all_rows}) != 100:
        raise ValueError("extended stress group inventory changed")
    if {str(row[input_config["freeze_field"]]) for row in all_rows} != {str(input_config["expected_freeze_id"])}:
        raise ValueError("extended stress freeze ID changed")
    expected_counts = {str(key): int(value) for key, value in input_config["expected_generator_counts"].items()}
    counts = Counter(str(row[input_config["generator_field"]]) for row in all_rows)
    if dict(counts) != expected_counts:
        raise ValueError(f"extended stress generator inventory changed: {dict(counts)}")
    rows = (
        _select_round_robin(all_rows, 3, str(input_config["generator_field"]))
        if toy
        else all_rows
    )
    if development:
        authorization = config["authorization"]
        toy_summary_path = _resolve(project_root, str(authorization["toy_summary"]))
        _verify(
            toy_summary_path,
            str(authorization["expected_toy_summary_sha256"]),
            "extended stress toy summary",
        )
        toy_summary = _read_json(toy_summary_path)
        if toy_summary.get("status") != "extended_stress_toy3_passed" or not all(
            toy_summary.get("structure_gate", {}).values()
        ):
            raise ValueError("extended stress toy gate did not pass")

    conditions_list = config["conditions"]
    _validate_conditions(conditions_list)
    conditions = {str(item["name"]): item for item in conditions_list}
    model_config = config["model"]
    scratch = Path(
        os.environ.get(
            str(config["paths"]["scratch_env"]),
            str(_resolve(project_root, str(config["paths"]["scratch_default"]))),
        )
    ).resolve()
    encoder_path = _resolve(scratch, str(model_config["encoder_weights"]))
    _verify(encoder_path, str(model_config["encoder_weights_sha256"]), "extended stress encoder")
    checkpoint_path = _resolve(project_root, str(model_config["checkpoint"]))
    _verify(checkpoint_path, str(model_config["checkpoint_sha256"]), "extended stress checkpoint")
    model = _load_teacher(encoder_path, model_config["teacher_conv1_coefficients"])
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    torch.manual_seed(int(experiment["seed"]))
    torch.cuda.manual_seed_all(int(experiment["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    paths = config["paths"]
    score_cache_dir = _resolve(scratch, str(paths["score_cache_dir"]))
    alignment_cache_dir = _resolve(scratch, str(paths["alignment_cache_dir"]))
    predictions_path = _resolve(project_root, str(paths["predictions"]))
    alignments_path = _resolve(project_root, str(paths["alignments"]))
    summary_path = _resolve(project_root, str(paths["summary"]))
    log_path = _resolve(project_root, str(paths["log"]))
    metrics_path = _resolve(project_root, str(paths["metrics"])) if development else None
    comparisons_path = _resolve(project_root, str(paths["comparisons"])) if development else None
    report_path = _resolve(project_root, str(paths["report"])) if development else None
    directories = [score_cache_dir, alignment_cache_dir, predictions_path.parent, alignments_path.parent, summary_path.parent, log_path.parent]
    if development:
        assert metrics_path is not None and comparisons_path is not None and report_path is not None
        directories.extend([metrics_path.parent, comparisons_path.parent, report_path.parent])
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    preprocessing = config["preprocessing"]
    inference = config["inference"]
    registration = config["registration"]
    operating = config["operating_point"]
    pixel_threshold = float(operating["fixed_pixel_threshold"])
    image_threshold = float(operating["fixed_image_threshold"])
    payloads = {name: {"forged": [], "authentic": []} for name in conditions}
    forged_scores: dict[str, dict[str, tuple[str, float]]] = {name: {} for name in conditions}
    predictions: list[dict[str, Any]] = []
    alignment_records: dict[str, dict[str, Any]] = {}
    failures = 0
    score_cache_hits = 0
    alignment_cache_hits = 0
    transform_deterministic = True
    severe_transforms_nonidentical = True
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    for group_index, row in enumerate(rows, start=1):
        group = str(row["source_group_id"])
        generator = str(row[input_config["generator_field"]])
        forged_path = _resolve(scratch, str(row["image"]))
        authentic_path = _resolve(scratch, str(row["authentic"]))
        mask_path = _resolve(scratch, str(row["mask"]))
        _verify(forged_path, str(row["image_sha256"]), "extended stress forged image")
        _verify(authentic_path, str(row["authentic_sha256"]), "extended stress authentic image")
        _verify(mask_path, str(row["mask_sha256"]), "extended stress mask")
        with Image.open(forged_path) as handle:
            native_forged = np.asarray(handle.convert("RGB"))
        with Image.open(authentic_path) as handle:
            native_authentic = np.asarray(handle.convert("RGB"))
        with Image.open(mask_path) as handle:
            native_mask = np.asarray(handle.convert("L")) > 0
        if native_forged.shape[:2] != native_mask.shape or native_authentic.shape[:2] != native_mask.shape:
            raise ValueError("extended stress native geometry changed")
        forged = _resize_image(native_forged, int(preprocessing["max_side"]))
        authentic = _resize_image(native_authentic, int(preprocessing["max_side"]))
        reference = _resize_reference(native_authentic, forged.shape[:2])
        if authentic.shape != forged.shape:
            authentic = _resize_reference(native_authentic, forged.shape[:2])

        for condition_name, condition in conditions.items():
            transformed, oracle, transform_metadata = _apply_stress(
                reference, condition, int(experiment["seed"]), group
            )
            if toy:
                replay, replay_oracle, replay_metadata = _apply_stress(
                    reference, condition, int(experiment["seed"]), group
                )
                transform_deterministic &= bool(
                    np.array_equal(transformed, replay)
                    and np.array_equal(oracle, replay_oracle)
                    and transform_metadata == replay_metadata
                )
                if int(condition["level"]) == 2:
                    severe_transforms_nonidentical &= not np.array_equal(
                        transformed, reference
                    )
            for sample_kind, candidate, native_candidate, candidate_sha in (
                ("forged", forged, native_forged, str(row["image_sha256"])),
                ("authentic", authentic, native_authentic, str(row["authentic_sha256"])),
            ):
                record: dict[str, Any] = {
                    "record_id": f"extended_stress:{condition_name}:{sample_kind}:{group}",
                    "source_group_id": group,
                    "source_dataset": row["source_dataset"],
                    "generator": generator,
                    "condition": condition_name,
                    "stress_family": condition["family"],
                    "stress_level": int(condition["level"]),
                    "stress_severity": float(condition.get("severity", 0.0)),
                    "train_relation": condition["train_relation"],
                    "sample_kind": sample_kind,
                    "status": "failed",
                    "paper_evidence": False,
                    "final_reserve_read": False,
                    "model_training_performed": False,
                    "checkpoint_selection_used": False,
                    "threshold_selection_used": False,
                }
                try:
                    alignment_key = _cache_key(
                        {
                            "schema_version": preprocessing["alignment_cache_schema_version"],
                            "candidate_sha256": candidate_sha,
                            "reference_sha256": row["authentic_sha256"],
                            "candidate_shape": list(candidate.shape),
                            "condition": condition,
                            "transform_metadata": transform_metadata,
                            "registration": registration,
                        }
                    )
                    alignment_path = alignment_cache_dir / condition_name / f"{alignment_key}.json"
                    alignment_path.parent.mkdir(parents=True, exist_ok=True)
                    aligned_reference: np.ndarray | None = None
                    if alignment_path.is_file():
                        metadata = _read_json(alignment_path)
                        alignment_cache_hits += 1
                    else:
                        aligned_reference, raw_metadata = _estimate_ecc_alignment(
                            candidate, transformed, oracle, registration
                        )
                        metadata = {
                            key: (
                                None
                                if isinstance(value, float) and not np.isfinite(value)
                                else value
                            )
                            for key, value in raw_metadata.items()
                        }
                        _write_json(alignment_path, metadata)
                    alignment_records.setdefault(
                        alignment_key,
                        {
                            "alignment_key": alignment_key,
                            "source_group_id": group,
                            "generator": generator,
                            "condition": condition_name,
                            "sample_kind": sample_kind,
                            "alignment_cache": str(alignment_path.relative_to(scratch)),
                            "paper_evidence": False,
                            "final_reserve_read": False,
                            **transform_metadata,
                            **metadata,
                        },
                    )
                    score_key = _cache_key(
                        {
                            "schema_version": preprocessing["score_cache_schema_version"],
                            "candidate_sha256": candidate_sha,
                            "reference_sha256": row["authentic_sha256"],
                            "condition": condition,
                            "transform_metadata": transform_metadata,
                            "alignment_key": alignment_key,
                            "checkpoint_sha256": model_config["checkpoint_sha256"],
                            "preprocessing": preprocessing,
                            "inference": inference,
                        }
                    )
                    score_path = score_cache_dir / condition_name / f"{score_key}.npz"
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    if score_path.is_file():
                        score_cache_hits += 1
                    else:
                        if aligned_reference is None:
                            estimated = np.asarray(metadata["estimated_homography"], dtype=np.float64)
                            aligned_reference = _warp_reference(
                                transformed, estimated, inverse=True
                            )
                        probability = _infer_pair_tiled(
                            model,
                            candidate,
                            aligned_reference,
                            device,
                            inference,
                            preprocessing,
                        ).astype(np.float32)
                        temporary = score_path.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(handle, scores=probability)
                        temporary.replace(score_path)
                    with np.load(score_path, allow_pickle=False) as archive:
                        probability = archive["scores"]
                    if probability.dtype != np.float32 or probability.shape != candidate.shape[:2] or not np.isfinite(probability).all():
                        raise ValueError("extended stress score cache is invalid")
                    native_probability = cv2.resize(
                        probability,
                        (native_candidate.shape[1], native_candidate.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    image_score = _top_fraction_mean(
                        probability, float(config["image_score"]["top_fraction"])
                    )
                    record.update(
                        {
                            "status": "ok",
                            "cache_key": score_key,
                            "score_cache": str(score_path.relative_to(scratch)),
                            "alignment_key": alignment_key,
                            "alignment_cache": str(alignment_path.relative_to(scratch)),
                            "alignment_status": metadata["alignment_status"],
                            "ecc_correlation": metadata["ecc_correlation"],
                            "score_shape": list(probability.shape),
                            "native_shape": list(native_candidate.shape[:2]),
                            "score_min": float(probability.min()),
                            "score_max": float(probability.max()),
                            "score_mean": float(probability.mean()),
                            "image_score": image_score,
                        }
                    )
                    if sample_kind == "forged":
                        average_precision, pixel_auroc = _ranking_metrics(
                            native_probability, native_mask
                        )
                        operating_metrics = _pixel_operating_metrics(
                            native_probability, native_mask, pixel_threshold
                        )
                        item = {
                            "source_group_id": group,
                            "generator": generator,
                            "macro_pixel_ap": average_precision,
                            "pixel_auroc": pixel_auroc,
                            "image_score": image_score,
                            **operating_metrics,
                        }
                        payloads[condition_name]["forged"].append(item)
                        forged_scores[condition_name][group] = (generator, average_precision)
                        record.update({key: value for key, value in item.items() if key not in {"source_group_id", "generator"}})
                    else:
                        pixel_fpr = float(np.mean(native_probability >= pixel_threshold))
                        payloads[condition_name]["authentic"].append(
                            {
                                "source_group_id": group,
                                "generator": generator,
                                "pixel_fpr": pixel_fpr,
                                "image_score": image_score,
                            }
                        )
                        record["pixel_fpr"] = pixel_fpr
                    if _RUNTIME_ONLY_FIELDS.intersection(record):
                        raise ValueError("extended stress prediction contains runtime-only fields")
                except Exception as error:
                    failures += 1
                    record["failure_type"] = type(error).__name__
                    record["failure_reason"] = str(error)
                    logging.exception("record_id=%s failed", record["record_id"])
                predictions.append(record)
                _write_jsonl(predictions_path, predictions)
        _write_jsonl(alignments_path, list(alignment_records.values()))
        logging.info("completed_groups=%d total_groups=%d failures=%d", group_index, len(rows), failures)

    expected_records = len(rows) * len(conditions) * 2
    complete = failures == 0 and len(predictions) == expected_records
    wall_time = time.monotonic() - started
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    cache_bytes = _directory_bytes(score_cache_dir) + _directory_bytes(alignment_cache_dir)
    common = {
        "schema_version": 1,
        "experiment": experiment,
        "paper_evidence": False,
        "final_reserve_read": False,
        "model_training_performed": False,
        "checkpoint_selection_used": False,
        "threshold_selection_used": False,
        "selected_groups": len(rows),
        "successful_records": len(predictions) - failures,
        "failed_records": failures,
        "score_cache_hits": score_cache_hits,
        "alignment_cache_hits": alignment_cache_hits,
        "wall_time_seconds": wall_time,
        "peak_vram_mb": peak_vram_mb,
        "cache_bytes": cache_bytes,
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "input": {
            "config_sha256": _sha256(config_path),
            "protocol_sha256": _sha256(protocol_path),
            "manifest_sha256": _sha256(manifest_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "alignments": str(alignments_path.relative_to(project_root)),
            "alignments_sha256": _sha256(alignments_path),
            "score_cache_dir": str(score_cache_dir.relative_to(scratch)),
            "alignment_cache_dir": str(alignment_cache_dir.relative_to(scratch)),
        },
    }
    if toy:
        structure_gate = {
            "all_138_records_complete": complete,
            "zero_failures": failures == 0,
            "all_scores_finite": all(
                row.get("status") == "ok"
                and np.isfinite(float(row["score_min"]))
                and np.isfinite(float(row["score_max"]))
                for row in predictions
            ),
            "transforms_deterministic": transform_deterministic,
            "severe_transforms_nonidentical": severe_transforms_nonidentical,
            "no_training": True,
            "no_threshold_selection": True,
            "no_final_reserve_read": True,
        }
        status = "extended_stress_toy3_passed" if all(structure_gate.values()) else "extended_stress_toy3_failed"
        output = {**common, "status": status, "structure_gate": structure_gate}
        _write_json(summary_path, output)
        if status != "extended_stress_toy3_passed" and runtime["require_all_records"]:
            raise RuntimeError("extended stress toy structure gate failed")
        return output

    if not complete:
        output = {**common, "status": "extended_stress_development100_failed_incomplete"}
        _write_json(summary_path, output)
        if runtime["require_all_records"]:
            raise RuntimeError(f"extended stress development failed: {failures} records")
        return output

    assert metrics_path is not None and comparisons_path is not None and report_path is not None
    metrics = {
        name: _aggregate(payloads[name], pixel_threshold, image_threshold)
        for name in conditions
    }
    clean_name = next(name for name, item in conditions.items() if item["family"] == "clean")
    bootstrap = config["bootstrap"]
    comparisons = {
        name: _stratified_paired_bootstrap(
            forged_scores[name],
            forged_scores[clean_name],
            int(bootstrap["seed"]) + offset,
            int(bootstrap["resamples"]),
            float(bootstrap["confidence_level"]),
        )
        for offset, name in enumerate(name for name in conditions if name != clean_name)
    }
    metric_rows = []
    for name, condition in conditions.items():
        metric_rows.append(
            {
                "condition": name,
                "family": condition["family"],
                "level": condition["level"],
                "severity": condition.get("severity", 0.0),
                "train_relation": condition["train_relation"],
                **metrics[name],
            }
        )
    comparison_rows = []
    for name, value in comparisons.items():
        row: dict[str, Any] = {
            "condition": name,
            "comparison": f"{name}_minus_{clean_name}",
            "effect": value["effect"],
            "ci_low": value["ci_low"],
            "ci_high": value["ci_high"],
            "paper_evidence": False,
        }
        for generator, effect in value["per_generator_effect"].items():
            safe = "".join(character if character.isalnum() else "_" for character in generator).strip("_")
            row[f"effect__{safe}"] = effect
        comparison_rows.append(row)
    _write_csv(metrics_path, metric_rows)
    _write_csv(comparisons_path, comparison_rows)
    alignment_statuses = Counter(str(row["alignment_status"]) for row in alignment_records.values())
    engineering_gate = {
        "all_4600_records_complete": len(predictions) == expected_records,
        "zero_failures": failures == 0,
        "all_scores_finite": all(
            np.isfinite(float(row["score_min"])) and np.isfinite(float(row["score_max"]))
            for row in predictions
        ),
        "all_100_groups_retained": len({row["source_group_id"] for row in predictions}) == 100,
        "both_generators_retained": Counter(
            row["generator"]
            for row in predictions
            if row["condition"] == clean_name and row["sample_kind"] == "forged"
        ) == expected_counts,
        "clean_anchor_ap_sanity": metrics[clean_name]["generator_macro_pixel_ap"] >= float(config["engineering_gate"]["clean_generator_macro_ap_min"]),
        "clean_anchor_fpr_sanity": metrics[clean_name]["authentic_document_macro_pixel_fpr"] <= float(config["engineering_gate"]["clean_authentic_pixel_fpr_max"]),
        "wall_time_below_four_gpu_hours": wall_time < 4.0 * 3600.0,
        "peak_vram_below_eight_gib": peak_vram_mb < 8.0 * 1024.0,
        "cache_below_eight_gib": cache_bytes < 8 * 1024**3,
        "no_training": True,
        "no_threshold_selection": True,
        "no_final_reserve_read": True,
    }
    status = "extended_stress_development100_complete" if all(engineering_gate.values()) else "extended_stress_development100_engineering_gate_failed"
    output = {
        **common,
        "status": status,
        "development_only": True,
        "selected_development_groups": len(rows),
        "condition_order": list(conditions),
        "condition_specs": conditions,
        "metrics": metrics,
        "clean_comparisons": comparisons,
        "alignment_status_counts": dict(sorted(alignment_statuses.items())),
        "engineering_gate": engineering_gate,
    }
    output["outputs"].update(
        {
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
            "report": str(report_path.relative_to(project_root)),
        }
    )
    _write_json(summary_path, output)
    report_path.write_text(_report(output), encoding="utf-8")
    output["outputs"]["report_sha256"] = _sha256(report_path)
    _write_json(summary_path, output)
    if status != "extended_stress_development100_complete" and runtime["require_all_records"]:
        raise RuntimeError("extended stress development engineering gate failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
