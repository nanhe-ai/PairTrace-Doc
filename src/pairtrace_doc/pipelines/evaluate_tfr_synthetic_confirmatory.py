from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        if int(os.environ.get(_name, "1")) < 1:
            os.environ[_name] = "1"
    except ValueError:
        os.environ[_name] = "1"

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from pairtrace_doc.pipelines.evaluate_registered_pair_controls import _ssim_distance
from pairtrace_doc.pipelines.train_resampling_robust_teacher import _reference_roundtrip
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
    EXTENDED_ARMS,
    _build_model,
    _infer_pair_tiled,
    _select_operating_point,
)


NONLEARNED = {"raw_rgb_difference", "ssim_distance"}


def _load_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.asarray(np.load(path, mmap_mode="r"))
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"invalid cached RGB array: {path}")
        return value
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.asarray(np.load(path, mmap_mode="r"), dtype=bool)
    else:
        with Image.open(path) as handle:
            value = np.asarray(handle.convert("L")) > 0
    if value.ndim != 2 or not value.any() or value.all():
        raise ValueError(f"invalid forged mask: {path}")
    return value


def _condition_reference(
    reference: np.ndarray,
    *,
    condition: str,
    source_group_id: str,
    freeze_id: str,
    selection_seed: int,
    augmentation: dict[str, Any],
) -> np.ndarray:
    mapping = {
        "clean": "clean",
        "translation_roundtrip": "translation",
        "affine_roundtrip": "affine",
        "perspective_roundtrip": "perspective",
    }
    if condition not in mapping:
        raise ValueError(f"unsupported confirmation condition: {condition}")
    digest = hashlib.sha256(
        f"{freeze_id}|{source_group_id}|{condition}|{selection_seed}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return _reference_roundtrip(reference, mapping[condition], rng, augmentation)


def _nonlearned_score(
    method: str, candidate: np.ndarray, reference: np.ndarray, ssim: dict[str, Any]
) -> np.ndarray:
    if candidate.shape != reference.shape:
        raise ValueError("nonlearned pair score requires aligned arrays")
    if method == "raw_rgb_difference":
        return np.max(
            np.abs(candidate.astype(np.float32) - reference.astype(np.float32)), axis=2
        ) / 255.0
    if method == "ssim_distance":
        return _ssim_distance(candidate, reference, ssim)
    raise ValueError(f"unsupported nonlearned method: {method}")


def _cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _score_with_cache(
    cache_dir: Path,
    key_payload: dict[str, Any],
    expected_shape: tuple[int, int],
    compute: Callable[[], np.ndarray],
) -> tuple[np.ndarray, bool]:
    key = _cache_key(key_payload)
    path = cache_dir / f"{key}.npy"
    if path.is_file():
        stored = np.asarray(np.load(path, mmap_mode="r"))
        if stored.dtype != np.uint16 or stored.shape != expected_shape:
            raise ValueError("confirmation score cache is invalid")
        return stored.astype(np.float32) / 65535.0, True
    value = np.asarray(compute(), dtype=np.float32)
    if (
        value.shape != expected_shape
        or not np.isfinite(value).all()
        or float(value.min()) < 0.0
        or float(value.max()) > 1.0
    ):
        raise ValueError("confirmation score computation is invalid")
    quantized = np.rint(np.clip(value, 0.0, 1.0) * 65535.0).astype(np.uint16)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, quantized)
    temporary.replace(path)
    return quantized.astype(np.float32) / 65535.0, False


def _bootstrap_group_mean(
    values: dict[str, list[float]], seed: int, replicates: int
) -> tuple[float, float]:
    groups = sorted(values)
    group_values = np.asarray([np.mean(values[group]) for group in groups], dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        samples[index] = float(rng.choice(group_values, len(group_values), replace=True).mean())
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _tpr_at_fixed_fpr(
    forged_scores: np.ndarray, authentic_scores: np.ndarray, target_fpr: float
) -> tuple[float, float, float]:
    forged = np.asarray(forged_scores, dtype=float)
    authentic = np.asarray(authentic_scores, dtype=float)
    if (
        forged.ndim != 1
        or authentic.ndim != 1
        or forged.size == 0
        or authentic.size == 0
        or not np.isfinite(forged).all()
        or not np.isfinite(authentic).all()
        or not 0.0 <= target_fpr <= 1.0
    ):
        raise ValueError("fixed-FPR image scores are invalid")
    candidates = np.concatenate(
        [
            np.unique(authentic),
            np.asarray([np.nextafter(float(authentic.max()), np.inf)]),
        ]
    )
    feasible = [
        float(threshold)
        for threshold in candidates
        if float(np.mean(authentic >= threshold)) <= target_fpr + 1e-12
    ]
    if not feasible:
        raise ValueError("no image threshold satisfies the fixed-FPR target")
    threshold = min(feasible)
    observed_fpr = float(np.mean(authentic >= threshold))
    tpr = float(np.mean(forged >= threshold))
    return tpr, observed_fpr, threshold


def _image_score_top_fraction(scores: np.ndarray, fraction: float) -> float:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or not 0.0 < fraction <= 1.0:
        raise ValueError("image-level top-fraction score is invalid")
    count = max(1, int(np.ceil(values.size * fraction)))
    return float(np.partition(values, values.size - count)[-count:].mean())


def _metric_summary(
    records: list[dict[str, Any]],
    seed: int,
    replicates: int,
    fixed_fpr_targets: list[float] | None = None,
) -> dict[str, Any]:
    forged = [row for row in records if row["sample_kind"] == "forged" and row["status"] == "ok"]
    authentic = [row for row in records if row["sample_kind"] == "authentic" and row["status"] == "ok"]
    grouped_ap: dict[str, list[float]] = defaultdict(list)
    grouped_f1: dict[str, list[float]] = defaultdict(list)
    grouped_iou: dict[str, list[float]] = defaultdict(list)
    attack_ap: dict[str, list[float]] = defaultdict(list)
    for row in forged:
        group = str(row["source_group_id"])
        grouped_ap[group].append(float(row["pixel_ap"]))
        grouped_f1[group].append(float(row["pixel_f1"]))
        grouped_iou[group].append(float(row["pixel_iou"]))
        attack_ap[str(row["attack"])].append(float(row["pixel_ap"]))
    low, high = _bootstrap_group_mean(grouped_ap, seed, replicates)
    forged_image_scores = np.asarray(
        [float(row["image_score"]) for row in forged], dtype=float
    )
    authentic_image_scores = np.asarray(
        [float(row["image_score"]) for row in authentic], dtype=float
    )
    _, image_auroc = _ranking_metrics(
        np.concatenate([forged_image_scores, authentic_image_scores]),
        np.concatenate(
            [
                np.ones(forged_image_scores.size, dtype=bool),
                np.zeros(authentic_image_scores.size, dtype=bool),
            ]
        ),
    )
    result = {
        "source_groups": len(grouped_ap),
        "forged_pairs": len(forged),
        "authentic_records": len(authentic),
        "document_macro_pixel_ap": float(
            np.mean([np.mean(values) for values in grouped_ap.values()])
        ),
        "document_macro_pixel_ap_ci95_low": low,
        "document_macro_pixel_ap_ci95_high": high,
        "document_macro_pixel_f1": float(
            np.mean([np.mean(values) for values in grouped_f1.values()])
        ),
        "document_macro_pixel_iou": float(
            np.mean([np.mean(values) for values in grouped_iou.values()])
        ),
        "attack_macro_pixel_ap": {
            attack: float(np.mean(values)) for attack, values in sorted(attack_ap.items())
        },
        "authentic_document_macro_pixel_fpr": float(
            np.mean([float(row["pixel_fpr"]) for row in authentic])
        ),
        "authentic_image_fpr": float(
            np.mean([float(row["positive_pixels"]) > 0 for row in authentic])
        ),
        "image_auroc_top_1pct": image_auroc,
    }
    for target in fixed_fpr_targets or []:
        tpr, observed_fpr, threshold = _tpr_at_fixed_fpr(
            forged_image_scores, authentic_image_scores, float(target)
        )
        suffix = f"{float(target):.4f}".rstrip("0").rstrip(".").replace(".", "p")
        result[f"image_tpr_at_fpr_{suffix}"] = tpr
        result[f"image_observed_fpr_at_target_{suffix}"] = observed_fpr
        result[f"image_threshold_at_fpr_{suffix}"] = threshold
    return result


def _paired_comparison_rows(
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    conditions: list[str],
    seed: int,
    replicates: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in records:
        if row["status"] == "ok" and row["sample_kind"] == "forged":
            grouped[
                (
                    str(row["method"]),
                    str(row["condition"]),
                    str(row["attack"]),
                    str(row["source_group_id"]),
                )
            ].append(float(row["pixel_ap"]))
    output: list[dict[str, Any]] = []
    attacks = sorted({attack for _, _, attack, _ in grouped})
    for comparison_index, specification in enumerate(comparisons):
        left = str(specification["left"])
        right = str(specification["right"])
        for condition_index, condition in enumerate(conditions):
            for attack_index, attack_scope in enumerate(["pooled", *attacks]):
                def group_values(method_name: str) -> dict[str, float]:
                    selected: dict[str, list[float]] = defaultdict(list)
                    for (
                        method,
                        current_condition,
                        attack,
                        group,
                    ), values in grouped.items():
                        if (
                            method == method_name
                            and current_condition == condition
                            and (attack_scope == "pooled" or attack == attack_scope)
                        ):
                            selected[group].extend(values)
                    return {
                        group: float(np.mean(values))
                        for group, values in selected.items()
                    }

                left_groups = group_values(left)
                right_groups = group_values(right)
                if set(left_groups) != set(right_groups) or not left_groups:
                    raise ValueError(
                        "paired comparison groups differ: "
                        f"{left} versus {right}, {condition}, {attack_scope}"
                    )
                deltas = {
                    group: [left_groups[group] - right_groups[group]]
                    for group in sorted(left_groups)
                }
                low, high = _bootstrap_group_mean(
                    deltas,
                    seed
                    + comparison_index * 10_007
                    + condition_index * 101
                    + attack_index,
                    replicates,
                )
                values = np.asarray(
                    [item[0] for item in deltas.values()], dtype=float
                )
                output.append(
                    {
                        "comparison": str(
                            specification.get("name", f"{left}_minus_{right}")
                        ),
                        "left": left,
                        "right": right,
                        "condition": condition,
                        "attack": attack_scope,
                        "source_groups": len(deltas),
                        "document_macro_pixel_ap_delta": float(values.mean()),
                        "delta_ci95_low": low,
                        "delta_ci95_high": high,
                        "groups_left_better": int(np.count_nonzero(values > 0)),
                        "groups_tied": int(np.count_nonzero(values == 0)),
                        "groups_right_better": int(np.count_nonzero(values < 0)),
                    }
                )
    return output


def _select_method_shard(
    methods: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    runtime: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select a predeclared seed shard without changing the frozen registry.

    The three seed shards are disjoint. Non-learned controls belong to exactly
    one shard, so a later merge has one and only one prediction stream per
    frozen method. Comparisons may be deferred until all shards are merged.
    """

    raw_seed = runtime.get("learned_seed_filter")
    defer = bool(runtime.get("defer_comparisons_to_merge", False))
    if raw_seed is None:
        if defer or "include_nonlearned" in runtime:
            raise ValueError("confirmation shard controls require a learned-seed filter")
        return methods, comparisons, {
            "enabled": False,
            "learned_seed": None,
            "include_nonlearned": True,
            "comparisons_deferred": False,
        }
    seed = int(raw_seed)
    include_nonlearned = bool(runtime.get("include_nonlearned", False))
    selected = [
        method
        for method in methods
        if (
            method["kind"] == "learned"
            and int(method["seed"]) == seed
        )
        or (method["kind"] == "nonlearned" and include_nonlearned)
    ]
    if not selected or not any(method["kind"] == "learned" for method in selected):
        raise ValueError(f"confirmation method shard is empty for seed {seed}")
    selected_names = {str(method["name"]) for method in selected}
    selected_comparisons = [
        comparison
        for comparison in comparisons
        if str(comparison["left"]) in selected_names
        and str(comparison["right"]) in selected_names
    ]
    if defer:
        selected_comparisons = []
    return selected, selected_comparisons, {
        "enabled": True,
        "learned_seed": seed,
        "include_nonlearned": include_nonlearned,
        "comparisons_deferred": defer,
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    code_sha = _sha256(Path(__file__).resolve())
    if code_sha != str(config["experiment"]["expected_evaluator_code_sha256"]):
        raise ValueError("confirmation evaluator code changed")
    runtime = config["runtime"]
    if (
        not runtime["confirmation_read_authorized"]
        or not runtime["validation_threshold_selection_authorized"]
        or runtime["model_training_authorized"]
        or runtime["confirmation_selection_authorized"]
    ):
        raise ValueError("confirmation evaluation authorization boundary changed")
    protocol = _resolve(project_root, str(config["experiment"]["protocol"]))
    if _sha256(protocol) != str(config["experiment"]["expected_protocol_sha256"]):
        raise ValueError("confirmation protocol changed")
    data = config["data"]
    for key in (
        "membership",
        "pair_manifest",
        "materialization_summary",
        "verification_summary",
        "validation_manifest",
    ):
        specification = data[key]
        path = _resolve(project_root, str(specification["path"]))
        if _sha256(path) != str(specification["expected_sha256"]):
            raise ValueError(f"confirmation evaluation input changed: {key}")
    membership = _read_jsonl(_resolve(project_root, str(data["membership"]["path"])))
    pair_rows = _read_jsonl(_resolve(project_root, str(data["pair_manifest"]["path"])))
    materialization = json.loads(
        _resolve(project_root, str(data["materialization_summary"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    freeze_id = str(materialization["freeze_id"])
    verification = json.loads(
        _resolve(project_root, str(data["verification_summary"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if (
        verification.get("status") != "confirmation_integrity_verified"
        or int(verification.get("error_count", -1)) != 0
        or bool(verification.get("model_inference_performed"))
        or str(verification.get("freeze_id")) != freeze_id
    ):
        raise ValueError("confirmation integrity verification gate changed")
    if len(membership) != int(data["expected_source_groups"]) or len(pair_rows) != int(
        data["expected_forged_pairs"]
    ):
        raise ValueError("confirmation inventory count changed")
    if {str(row["freeze_id"]) for row in [*membership, *pair_rows]} != {freeze_id}:
        raise ValueError("confirmation freeze ID changed")
    validation_all = _read_jsonl(
        _resolve(project_root, str(data["validation_manifest"]["path"]))
    )
    validation_rows = [
        row
        for row in validation_all
        if str(row[data["validation_role_field"]]) == str(data["validation_role"])
    ]
    if len(validation_rows) != int(data["expected_validation_records"]):
        raise ValueError("validation record count changed")
    if len({str(row["source_group_id"]) for row in validation_rows}) != len(
        validation_rows
    ):
        raise ValueError("validation source groups are not unique")
    if {str(row["source_group_id"]) for row in validation_rows} & {
        str(row["source_group_id"]) for row in membership
    }:
        raise ValueError("validation and confirmation source groups overlap")
    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            str(paths["scratch_env"]),
            str(_resolve(project_root, str(paths["scratch_default"]))),
        )
    ).resolve()
    validation_cache_dir = _resolve(scratch, str(paths["validation_pair_cache_dir"]))
    validation_cache = [
        _prepare_pair_cache(row, scratch, validation_cache_dir, config["preprocessing"])[0]
        for row in validation_rows
    ]
    score_cache = _resolve(scratch, str(paths["score_cache_dir"]))
    log_path = _resolve(project_root, str(paths["log"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    device = torch.device(str(runtime["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("confirmation learned evaluation requires CUDA")
    torch.set_num_threads(max(1, int(runtime.get("torch_threads", 1))))
    encoder_path = _resolve(scratch, str(config["model"]["encoder_weights"]))
    if _sha256(encoder_path) != str(config["model"]["encoder_weights_sha256"]):
        raise ValueError("encoder initialization changed")
    encoder_state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    registry_specification = config["method_registry"]
    registry_path = _resolve(project_root, str(registry_specification["path"]))
    if _sha256(registry_path) != str(registry_specification["expected_sha256"]):
        raise ValueError("confirmation method registry changed")
    method_specifications = _read_jsonl(registry_path)
    comparison_specification = config["comparison_registry"]
    comparison_path = _resolve(project_root, str(comparison_specification["path"]))
    if _sha256(comparison_path) != str(comparison_specification["expected_sha256"]):
        raise ValueError("confirmation comparison registry changed")
    comparison_specifications = _read_jsonl(comparison_path)
    methods: list[dict[str, Any]] = []
    method_names: set[str] = set()
    for specification in method_specifications:
        method = dict(specification)
        name = str(method["name"])
        if name in method_names:
            raise ValueError(f"duplicate confirmation method name: {name}")
        method_names.add(name)
        if method["kind"] == "learned":
            if str(method["arm"]) not in EXTENDED_ARMS:
                raise ValueError(f"unsupported learned method arm: {method['arm']}")
            checkpoint = _resolve(project_root, str(method["checkpoint"]))
            if _sha256(checkpoint) != str(method["checkpoint_sha256"]):
                raise ValueError(f"checkpoint changed: {name}")
            training_config = _resolve(project_root, str(method["training_config"]))
            if _sha256(training_config) != str(method["training_config_sha256"]):
                raise ValueError(f"training config changed: {name}")
            training_summary = _resolve(project_root, str(method["training_summary"]))
            if _sha256(training_summary) != str(method["training_summary_sha256"]):
                raise ValueError(f"training summary changed: {name}")
            validation_threshold = float(method["validation_threshold"])
            if not 0.0 <= validation_threshold <= 1.0 or not np.isclose(
                validation_threshold * 100.0,
                round(validation_threshold * 100.0),
                atol=1e-10,
            ):
                raise ValueError(f"validation threshold grid binding changed: {name}")
            metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
            checkpoint_arm = str(
                metadata.get("arm", metadata.get("representation_arm", ""))
            )
            checkpoint_seed = int(
                metadata.get("seed", metadata.get("training_seed", -1))
            )
            if checkpoint_arm != str(method["arm"]):
                raise ValueError(f"checkpoint arm binding changed: {name}")
            if checkpoint_seed != int(method["seed"]):
                raise ValueError(f"checkpoint seed binding changed: {name}")
            if str(metadata.get("config_sha256")) != str(
                method["training_config_sha256"]
            ):
                raise ValueError(f"checkpoint training-config binding changed: {name}")
            if str(metadata.get("protocol_sha256")) != str(
                config["experiment"]["expected_protocol_sha256"]
            ):
                raise ValueError(f"checkpoint protocol binding changed: {name}")
            if metadata.get("selection_rule") != "fixed_final_epoch":
                raise ValueError(f"checkpoint selection rule changed: {name}")
            method["checkpoint_path"] = checkpoint
        elif method["kind"] == "nonlearned":
            if name not in NONLEARNED:
                raise ValueError(f"unsupported nonlearned method: {name}")
        else:
            raise ValueError(f"unsupported method kind: {method['kind']}")
        methods.append(method)
    for comparison in comparison_specifications:
        if str(comparison["left"]) not in method_names or str(
            comparison["right"]
        ) not in method_names:
            raise ValueError("comparison references an unknown method")
    methods, selected_comparisons, shard = _select_method_shard(
        methods, comparison_specifications, runtime
    )
    conditions = [str(value) for value in config["conditions"]]
    if conditions != [
        "clean",
        "translation_roundtrip",
        "affine_roundtrip",
        "perspective_roundtrip",
    ]:
        raise ValueError("confirmation condition order changed")
    image_level = config["image_level"]
    if image_level["score_aggregation"] != "mean_top_fraction":
        raise ValueError("confirmation image-level score aggregation changed")
    if float(image_level["top_fraction"]) != 0.01:
        raise ValueError("confirmation image-level top fraction changed")
    fixed_fpr_targets = [float(value) for value in image_level["fixed_fpr_targets"]]
    if fixed_fpr_targets != [0.01, 0.05]:
        raise ValueError("confirmation fixed-FPR targets changed")
    operating_point = config["operating_point"]
    if {
        "candidate_min": float(operating_point["candidate_min"]),
        "candidate_max": float(operating_point["candidate_max"]),
        "candidate_step": float(operating_point["candidate_step"]),
        "max_authentic_fpr": float(operating_point["max_authentic_fpr"]),
    } != {
        "candidate_min": 0.0,
        "candidate_max": 1.0,
        "candidate_step": 0.01,
        "max_authentic_fpr": 0.01,
    }:
        raise ValueError("confirmation operating-point grid changed")
    training = config["inference"]
    thresholds = np.arange(
        float(config["operating_point"]["candidate_min"]),
        float(config["operating_point"]["candidate_max"])
        + float(config["operating_point"]["candidate_step"]) / 2,
        float(config["operating_point"]["candidate_step"]),
    )
    all_records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    cache_hits = 0
    # Some CUDA/PyTorch builds reject a ``torch.device`` argument here even
    # though the same object is accepted by tensor transfers. The configured
    # device has already been validated above and is the current single GPU.
    torch.cuda.reset_peak_memory_stats()

    for method in methods:
        name = str(method["name"])
        logging.info("method_start name=%s", name)
        arm = str(method.get("arm", ""))
        checkpoint_sha = str(method.get("checkpoint_sha256", "nonlearned"))
        model: torch.nn.Module | None = None
        if method["kind"] == "learned":
            model = _build_model(arm, encoder_state)
            checkpoint = torch.load(method["checkpoint_path"], map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model = model.to(device).eval()

        def score_pair(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
            if model is None:
                return _nonlearned_score(name, candidate, reference, config["ssim"])
            return _infer_pair_tiled(
                model, candidate, reference, arm, device, training, config["preprocessing"]
            )

        if method["kind"] == "learned":
            threshold = float(method["validation_threshold"])
        else:
            validation_scores: list[np.ndarray] = []
            validation_masks: list[np.ndarray] = []
            validation_groups: list[str] = []
            validation_authentic: dict[str, np.ndarray] = {}
            for row in validation_cache:
                candidate = _load_rgb(Path(row["forged"]))
                reference = _load_rgb(Path(row["authentic"]))
                mask = _load_mask(Path(row["mask"]))
                scores, hit = _score_with_cache(
                    score_cache,
                    {
                        "scope": "validation_forged",
                        "method": name,
                        "checkpoint": checkpoint_sha,
                        "sample": row["sample_id"],
                        "candidate": _sha256(Path(row["forged"])),
                        "reference": _sha256(Path(row["authentic"])),
                        "code": code_sha,
                    },
                    mask.shape,
                    lambda c=candidate, r=reference: score_pair(c, r),
                )
                cache_hits += int(hit)
                validation_scores.append(scores)
                validation_masks.append(mask)
                group = str(row["source_group_id"])
                validation_groups.append(group)
                if group not in validation_authentic:
                    scores_auth, hit = _score_with_cache(
                        score_cache,
                        {
                            "scope": "validation_authentic",
                            "method": name,
                            "checkpoint": checkpoint_sha,
                            "group": group,
                            "reference": _sha256(Path(row["authentic"])),
                            "code": code_sha,
                        },
                        reference.shape[:2],
                        lambda r=reference: score_pair(r, r),
                    )
                    cache_hits += int(hit)
                    validation_authentic[group] = scores_auth
            selected = _select_operating_point(
                validation_scores,
                validation_masks,
                validation_groups,
                validation_authentic,
                thresholds,
                float(config["operating_point"]["max_authentic_fpr"]),
            )
            threshold = float(selected["threshold"])
        logging.info("method_threshold name=%s threshold=%.6f", name, threshold)

        representative = {}
        for row in pair_rows:
            representative.setdefault(str(row["source_group_id"]), row)
        for condition in conditions:
            condition_records: list[dict[str, Any]] = []
            for row in pair_rows:
                sample_id = str(row["sample_id"])
                group = str(row["source_group_id"])
                try:
                    candidate_path = _resolve(scratch, str(row["image"]))
                    reference_path = _resolve(scratch, str(row["authentic"]))
                    mask_path = _resolve(scratch, str(row["mask"]))
                    candidate = _load_rgb(candidate_path)
                    reference = _condition_reference(
                        _load_rgb(reference_path),
                        condition=condition,
                        source_group_id=group,
                        freeze_id=freeze_id,
                        selection_seed=int(config["experiment"]["selection_seed"]),
                        augmentation=config["augmentation"],
                    )
                    mask = _load_mask(mask_path)
                    scores, hit = _score_with_cache(
                        score_cache,
                        {
                            "scope": "confirmation_forged",
                            "method": name,
                            "checkpoint": checkpoint_sha,
                            "sample": sample_id,
                            "condition": condition,
                            "image": row["image_sha256"],
                            "reference": row["authentic_sha256"],
                            "freeze": freeze_id,
                            "code": code_sha,
                        },
                        mask.shape,
                        lambda c=candidate, r=reference: score_pair(c, r),
                    )
                    cache_hits += int(hit)
                    pixel_ap, pixel_auroc = _ranking_metrics(scores, mask)
                    binary = scores >= threshold
                    tp = int(np.count_nonzero(binary & mask))
                    fp = int(np.count_nonzero(binary & ~mask))
                    fn = int(np.count_nonzero(~binary & mask))
                    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                    iou = tp / max(1, tp + fp + fn)
                    record = {
                        "status": "ok",
                        "method": name,
                        "kind": method["kind"],
                        "arm": arm or None,
                        "family": method.get("family", name),
                        "training_regime": method.get("training_regime"),
                        "seed": method.get("seed"),
                        "condition": condition,
                        "sample_kind": "forged",
                        "sample_id": sample_id,
                        "source_group_id": group,
                        "attack": row["selected_generator"],
                        "pixel_ap": pixel_ap,
                        "pixel_auroc": pixel_auroc,
                        "threshold": threshold,
                        "pixel_f1": f1,
                        "pixel_iou": iou,
                        "image_score": _image_score_top_fraction(
                            scores, float(image_level["top_fraction"])
                        ),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "error": None,
                    }
                except Exception as exc:
                    record = {
                        "status": "error",
                        "method": name,
                        "kind": method["kind"],
                        "arm": arm or None,
                        "family": method.get("family", name),
                        "training_regime": method.get("training_regime"),
                        "seed": method.get("seed"),
                        "condition": condition,
                        "sample_kind": "forged",
                        "sample_id": sample_id,
                        "source_group_id": group,
                        "attack": row["selected_generator"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                condition_records.append(record)
                all_records.append(record)
            for group, row in sorted(representative.items()):
                try:
                    authentic_path = _resolve(scratch, str(row["authentic"]))
                    authentic = _load_rgb(authentic_path)
                    stressed = _condition_reference(
                        authentic,
                        condition=condition,
                        source_group_id=group,
                        freeze_id=freeze_id,
                        selection_seed=int(config["experiment"]["selection_seed"]),
                        augmentation=config["augmentation"],
                    )
                    scores, hit = _score_with_cache(
                        score_cache,
                        {
                            "scope": "confirmation_authentic",
                            "method": name,
                            "checkpoint": checkpoint_sha,
                            "group": group,
                            "condition": condition,
                            "reference": row["authentic_sha256"],
                            "freeze": freeze_id,
                            "code": code_sha,
                        },
                        authentic.shape[:2],
                        lambda a=authentic, r=stressed: score_pair(a, r),
                    )
                    cache_hits += int(hit)
                    positive = int(np.count_nonzero(scores >= threshold))
                    record = {
                        "status": "ok",
                        "method": name,
                        "kind": method["kind"],
                        "arm": arm or None,
                        "family": method.get("family", name),
                        "training_regime": method.get("training_regime"),
                        "seed": method.get("seed"),
                        "condition": condition,
                        "sample_kind": "authentic",
                        "sample_id": f"{group}:authentic",
                        "source_group_id": group,
                        "attack": None,
                        "threshold": threshold,
                        "positive_pixels": positive,
                        "pixel_count": int(scores.size),
                        "pixel_fpr": positive / scores.size,
                        "image_score": _image_score_top_fraction(
                            scores, float(image_level["top_fraction"])
                        ),
                        "error": None,
                    }
                except Exception as exc:
                    record = {
                        "status": "error",
                        "method": name,
                        "kind": method["kind"],
                        "arm": arm or None,
                        "family": method.get("family", name),
                        "training_regime": method.get("training_regime"),
                        "seed": method.get("seed"),
                        "condition": condition,
                        "sample_kind": "authentic",
                        "sample_id": f"{group}:authentic",
                        "source_group_id": group,
                        "attack": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                condition_records.append(record)
                all_records.append(record)
            failures = [row for row in condition_records if row["status"] != "ok"]
            if failures:
                continue
            metrics = _metric_summary(
                condition_records,
                seed=int(config["statistics"]["bootstrap_seed"]),
                replicates=int(config["statistics"]["bootstrap_replicates"]),
                fixed_fpr_targets=fixed_fpr_targets,
            )
            metric_rows.append(
                {
                    "method": name,
                    "kind": method["kind"],
                    "arm": arm or None,
                    "family": method.get("family", name),
                    "training_regime": method.get("training_regime"),
                    "seed": method.get("seed"),
                    "condition": condition,
                    "validation_threshold": threshold,
                    **{key: value for key, value in metrics.items() if not isinstance(value, dict)},
                    "attack_macro_pixel_ap_json": json.dumps(
                        metrics["attack_macro_pixel_ap"], sort_keys=True
                    ),
                }
            )
            logging.info(
                "condition_complete method=%s condition=%s document_macro_pixel_ap=%.8f",
                name,
                condition,
                float(metrics["document_macro_pixel_ap"]),
            )
        del model
        torch.cuda.empty_cache()

    predictions_path = _resolve(project_root, str(paths["predictions"]))
    metrics_path = _resolve(project_root, str(paths["metrics"]))
    comparisons_path = _resolve(project_root, str(paths["comparisons"]))
    summary_path = _resolve(project_root, str(paths["summary"]))
    _write_jsonl(predictions_path, all_records)
    _write_csv(metrics_path, metric_rows)
    failures = [row for row in all_records if row["status"] != "ok"]
    comparison_rows = (
        _paired_comparison_rows(
            all_records,
            selected_comparisons,
            conditions,
            seed=int(config["statistics"]["bootstrap_seed"]),
            replicates=int(config["statistics"]["bootstrap_replicates"]),
        )
        if not failures and not shard["comparisons_deferred"]
        else [
            {
                "status": (
                    "deferred_to_frozen_shard_merge"
                    if shard["comparisons_deferred"] and not failures
                    else "not_computed_due_to_item_failures"
                ),
                "item_failures": len(failures),
            }
        ]
    )
    _write_csv(comparisons_path, comparison_rows)
    summary = {
        "status": "confirmation_evaluation_complete" if not failures else "confirmation_evaluation_failed",
        "claim_boundary": "controlled_confirmation_only_not_official_tfr",
        "freeze_id": freeze_id,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(protocol),
        "evaluator_code_sha256": code_sha,
        "input_sha256": {
            key: str(data[key]["expected_sha256"])
            for key in (
                "membership",
                "pair_manifest",
                "materialization_summary",
                "verification_summary",
                "validation_manifest",
            )
        },
        "method_registry_sha256": str(registry_specification["expected_sha256"]),
        "comparison_registry_sha256": str(
            comparison_specification["expected_sha256"]
        ),
        "methods": [str(method["name"]) for method in methods],
        "conditions": conditions,
        "method_shard": shard,
        "prediction_records": len(all_records),
        "metric_rows": len(metric_rows),
        "failures": len(failures),
        "cache_hits": cache_hits,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "confirmation_selection_performed": False,
        "threshold_selection_scope": "frozen_aiforge_validation_only",
        "learned_threshold_source": "pre_confirmation_method_registry",
        "nonlearned_threshold_source": "frozen_aiforge_validation_computed_in_run",
        "wall_time_seconds": time.monotonic() - started,
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
            "log": str(log_path.relative_to(project_root)),
        },
    }
    logging.info("evaluation_summary %s", json.dumps(summary, sort_keys=True))
    summary["outputs"]["log_sha256"] = _sha256(log_path)
    _write_json(summary_path, summary)
    if failures:
        raise RuntimeError(f"confirmation evaluation recorded {len(failures)} item failures")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
