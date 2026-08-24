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
    _PairTraceDataset,
    _load_student,
    _load_teacher,
    _native_validation,
    _shuffled_authentic_map,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _dice_loss,
    _expected_role_counts,
    _grouped_mean,
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
        raise ValueError("PairTrace spatial-relation common config SHA-256 changed")
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    effective_override = {
        key: value
        for key, value in override.items()
        if key not in {"base_config", "expected_base_config_sha256"}
    }
    return _deep_merge(base, effective_override), base_path, expected


def _normalized_spatial_gram(feature: torch.Tensor, grid_size: int) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(feature, (grid_size, grid_size))
    vectors = pooled.flatten(2).transpose(1, 2)
    vectors = F.normalize(vectors.float(), p=2, dim=2, eps=1e-6)
    gram = torch.bmm(vectors, vectors.transpose(1, 2))
    return gram / gram.flatten(1).norm(p=2, dim=1).clamp_min(1e-6).view(-1, 1, 1)


def _spatial_relation_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    masks: torch.Tensor,
    active: torch.Tensor,
    feature_names: list[str],
    grid_size: int,
    edited_pixel_weight: float,
) -> torch.Tensor:
    active_sum = active.sum()
    if float(active_sum.detach().cpu()) == 0.0:
        return sum(value.sum() * 0.0 for value in student_features.values())
    pooled_mask = F.adaptive_max_pool2d(masks.float(), (grid_size, grid_size))
    spatial_weight = 1.0 + (edited_pixel_weight - 1.0) * pooled_mask.flatten(1)
    relation_weight = spatial_weight.unsqueeze(2) * spatial_weight.unsqueeze(1)
    scale = float((grid_size * grid_size) ** 2)
    losses: list[torch.Tensor] = []
    for name in feature_names:
        student_gram = _normalized_spatial_gram(student_features[name], grid_size)
        teacher_gram = _normalized_spatial_gram(
            teacher_features[name].detach(), grid_size
        )
        squared = (student_gram - teacher_gram).square()
        per_sample = (squared * relation_weight).sum(dim=(1, 2))
        per_sample = per_sample / relation_weight.sum(dim=(1, 2)).clamp_min(1.0)
        per_sample = per_sample * scale
        losses.append((per_sample * active).sum() / active_sum.clamp_min(1.0))
    return torch.stack(losses).mean()


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config, base_config_path, base_config_sha256 = _load_config(config_path)
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not runtime["gpu_launch_authorized"] or not runtime[
        "spatial_relation_training_authorized"
    ]:
        raise ValueError("PairTrace spatial-relation training was not authorized")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("PairTrace spatial-relation pilot cannot be paper evidence")
    data_config = config["data"]
    if not data_config["training_must_not_read_viewed_diagnostic"]:
        raise ValueError("viewed diagnostic must remain excluded")
    if not data_config["training_must_not_read_final_reserve"]:
        raise ValueError("final reserve must remain excluded")
    pair_mode = str(config["experiment"]["pair_mode"])
    if pair_mode not in {"correct_pair", "shuffled_pair"}:
        raise ValueError(f"unsupported spatial-relation pair mode: {pair_mode}")
    relation = config["relation"]
    if relation["authentic_relation_allowed"]:
        raise ValueError("authentic negative crops cannot activate relation loss")
    if not relation["normalize_channel_vectors"] or not relation[
        "normalize_gram_frobenius"
    ]:
        raise ValueError("frozen spatial-relation normalization changed")
    student_probability_sum = sum(
        float(config["sampling"][name])
        for name in (
            "student_forged_positive_probability",
            "student_forged_random_probability",
            "student_authentic_random_probability",
        )
    )
    if not math.isclose(student_probability_sum, 1.0, abs_tol=1e-12):
        raise ValueError("spatial-relation sampling probabilities must sum to one")
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PairTrace spatial-relation pilot requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("PairTrace spatial-relation protocol SHA-256 changed")
    manifest_path = _resolve(project_root, data_config["manifest"])
    if _sha256(manifest_path) != data_config["expected_manifest_sha256"]:
        raise ValueError("PairTrace spatial-relation manifest SHA-256 changed")
    student_summary_path = _resolve(project_root, config["matched_student"]["summary"])
    if _sha256(student_summary_path) != config["matched_student"][
        "expected_summary_sha256"
    ]:
        raise ValueError("matched-student summary SHA-256 changed")
    teacher_checkpoint_path = _resolve(
        project_root, config["model"]["teacher_checkpoint"]
    )
    if _sha256(teacher_checkpoint_path) != config["model"][
        "teacher_checkpoint_sha256"
    ]:
        raise ValueError("frozen correct-pair teacher checkpoint SHA-256 changed")

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
        raise ValueError("spatial-relation train/validation role counts changed")
    if {row["source_group_id"] for row in train_rows} & {
        row["source_group_id"] for row in validation_rows
    }:
        raise ValueError("spatial-relation train/validation groups overlap")
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
        raise ValueError("spatial-relation encoder weights SHA-256 changed")
    student = _load_student(weights_path).to(device)
    teacher = _load_teacher(
        weights_path, config["model"]["teacher_conv1_coefficients"]
    )
    teacher_saved = torch.load(
        teacher_checkpoint_path, map_location="cpu", weights_only=True
    )
    teacher.load_state_dict(teacher_saved["model_state"], strict=True)
    teacher = teacher.to(device).eval().requires_grad_(False)

    training = config["student_training"]
    dataset = _PairTraceDataset(
        train_cache,
        "student",
        pair_mode,
        shuffled_authentic,
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
    optimizer = torch.optim.AdamW(
        student.parameters(),
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
        student.train()
        losses: list[float] = []
        direct_losses: list[float] = []
        relation_losses: list[float] = []
        for student_inputs, teacher_inputs, masks, active in loader:
            student_inputs = student_inputs.to(device, non_blocking=True)
            teacher_inputs = teacher_inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            active = active.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])
            ):
                _, teacher_features = teacher.forward_with_features(teacher_inputs)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])
            ):
                logits, student_features = student.forward_with_features(student_inputs)
                bce = F.binary_cross_entropy_with_logits(
                    logits, masks, pos_weight=positive_weight
                )
                dice = _dice_loss(logits, masks)
                direct = float(training["bce_loss_weight"]) * bce + float(
                    training["dice_loss_weight"]
                ) * dice
                relation_loss = _spatial_relation_loss(
                    student_features,
                    teacher_features,
                    masks,
                    active,
                    list(relation["feature_names"]),
                    int(relation["grid_size"]),
                    float(relation["edited_pixel_weight"]),
                )
                loss = direct + float(relation["loss_weight"]) * relation_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            direct_losses.append(float(direct.detach().cpu()))
            relation_losses.append(float(relation_loss.detach().cpu()))
        scheduler.step()
        validation_aps: list[float] = []
        for record in validation_cache:
            image = np.asarray(np.load(record["forged"], mmap_mode="r"))
            mask = np.asarray(np.load(record["mask"], mmap_mode="r")).astype(bool)
            probability = _infer_tiled(
                student, image, device, training, config["preprocessing"]
            )
            average_precision, _ = _ranking_metrics(probability, mask)
            validation_aps.append(average_precision)
        checkpoint_group_field = training.get("checkpoint_group_field")
        checkpoint_ap, validation_ap_by_group = _grouped_mean(
            validation_aps, validation_cache, checkpoint_group_field
        )
        validation_ap = float(np.mean(validation_aps))
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "direct_loss": float(np.mean(direct_losses)),
            "spatial_relation_loss": float(np.mean(relation_losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_forged_document_macro_pixel_ap_model_resolution": validation_ap,
            "checkpoint_selection_macro_pixel_ap_model_resolution": checkpoint_ap,
            "checkpoint_group_field": checkpoint_group_field,
            "checkpoint_group_macro_pixel_ap_model_resolution": validation_ap_by_group,
            "pair_mode": pair_mode,
            "paper_evidence": False,
        }
        epoch_records.append(record)
        _write_jsonl(epoch_log_path, epoch_records)
        logging.info("spatial-relation epoch=%d metrics=%s", epoch + 1, record)
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
                    "config_sha256": _sha256(config_path),
                    "base_config_sha256": base_config_sha256,
                    "protocol_sha256": _sha256(protocol_path),
                    "encoder_weights_sha256": weights_sha256,
                    "teacher_checkpoint_sha256": _sha256(teacher_checkpoint_path),
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
        training,
        config["preprocessing"],
        config["operating_point"],
    )
    metric_row["best_epoch"] = best_epoch
    metric_row["pair_mode"] = pair_mode
    _write_jsonl(predictions_path, prediction_records)
    _write_csv(metrics_path, [metric_row])
    success = config["success"]
    if "native_generator_macro_pixel_ap_min" in success:
        success_ap_key = "generator_macro_pixel_ap"
        success_ap_min = float(success["native_generator_macro_pixel_ap_min"])
    else:
        success_ap_key = "macro_pixel_ap"
        success_ap_min = float(success["native_macro_pixel_ap_min"])
    individual_success = bool(
        metric_row[success_ap_key] >= success_ap_min
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
        "pair_mode": pair_mode,
        "viewed_diagnostic_read": False,
        "final_reserve_read": False,
        "protocol_sha256": _sha256(protocol_path),
        "config_sha256": _sha256(config_path),
        "base_config": str(base_config_path.relative_to(project_root)),
        "base_config_sha256": base_config_sha256,
        "input_manifest_sha256": _sha256(manifest_path),
        "teacher_checkpoint": str(teacher_checkpoint_path.relative_to(project_root)),
        "teacher_checkpoint_sha256": _sha256(teacher_checkpoint_path),
        "train_pairs": len(train_cache),
        "validation_pairs": len(validation_cache),
        "pair_cache_hits": cache_hits,
        "best_epoch": best_epoch,
        "best_validation_macro_pixel_ap_model_resolution": best_ap,
        "checkpoint": str(checkpoint_path.relative_to(project_root)),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_metrics_native_geometry": metric_row,
        "individual_success": {
            "primary_ap_metric": success_ap_key,
            "primary_ap_min": success_ap_min,
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
