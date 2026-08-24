from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from pairtrace_doc.baselines.visualdiff_style import visualdiff_style_score
from pairtrace_doc.pipelines.evaluate_registered_pair_controls import _ssim_distance
from pairtrace_doc.pipelines.train_student_100 import _ranking_metrics
from pairtrace_doc.pipelines.train_tfr_equal_budget_arm import (
    _build_model,
    _infer_pair_tiled,
)


REFERENCE_CONDITIONS = ["scan_reference", "digital_reference"]
NONLEARNED_METHODS = {
    "raw_rgb_difference",
    "ssim_distance",
    "visualdiff_style_dense_sift",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        value = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"invalid RGB artifact: {path}")
    return value


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        value = np.asarray(handle.convert("L"), dtype=np.uint8) > 0
    if value.ndim != 2 or not value.any() or value.all():
        raise ValueError(f"invalid nonempty binary mask: {path}")
    return value


def _scratch_root(project_root: Path, paths: dict[str, Any]) -> Path:
    environment = paths.get("scratch_env")
    override = os.environ.get(str(environment)) if environment else None
    if override:
        return Path(override).expanduser().resolve()
    return _resolve(project_root, str(paths["scratch_default"]))


def _scratch_path(scratch: Path, relative: str) -> Path:
    path = (scratch / relative).resolve()
    try:
        path.relative_to(scratch.resolve())
    except ValueError as error:
        raise ValueError(f"DESCAN artifact escapes scratch: {relative}") from error
    return path


def _cache_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _score_cache(
    cache_dir: Path,
    key_payload: dict[str, Any],
    expected_shape: tuple[int, int],
    compute: Callable[[], tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    key = _cache_key(key_payload)
    path = cache_dir / key[:2] / f"{key}.npz"
    hit = path.is_file()
    if hit:
        with np.load(path, allow_pickle=False) as stored:
            quantized = np.asarray(stored["score"], dtype=np.uint16)
            valid = np.asarray(stored["valid"], dtype=np.uint8).astype(bool)
            metadata = json.loads(str(stored["metadata_json"].item()))
    else:
        score, valid, metadata = compute()
        score = np.asarray(score, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        if score.shape != expected_shape or valid.shape != expected_shape:
            raise ValueError("DESCAN score-cache shape changed")
        if not valid.any() or not np.isfinite(score[valid]).all():
            raise ValueError("DESCAN score contains no finite valid support")
        if float(score[valid].min()) < 0.0 or float(score[valid].max()) > 1.0:
            raise ValueError("DESCAN score is outside [0,1]")
        quantized = np.zeros(expected_shape, dtype=np.uint16)
        quantized[valid] = np.rint(score[valid] * 65535.0).astype(np.uint16)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            score=quantized,
            valid=valid.astype(np.uint8),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        temporary.replace(path)
    if quantized.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError("cached DESCAN score artifact is invalid")
    score = quantized.astype(np.float32) / 65535.0
    return score, valid, metadata, {
        "cache_key": key,
        "cache_path": path,
        "cache_sha256": _sha256(path),
        "cache_hit": hit,
        "cache_bytes": path.stat().st_size,
    }


def _method_arm(method: dict[str, Any]) -> str:
    family = str(method.get("family"))
    if family == "pairtrace_9c_roundtrip":
        return "explicit_9ch"
    if family in {
        "fc_siam_diff_roundtrip",
        "fc_siam_diff_identity_continuation",
    }:
        return "fc_siam_diff"
    raise ValueError(f"unsupported learned DESCAN family: {family}")


def _validate_methods(
    project_root: Path,
    registry: list[dict[str, Any]],
    expected_names: list[str],
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for source in registry:
        method = dict(source)
        name = str(method["name"])
        if name in by_name:
            raise ValueError(f"duplicate DESCAN method name: {name}")
        if method["kind"] == "learned":
            if method.get("status") != "ready":
                raise ValueError(f"learned DESCAN method is not ready: {name}")
            checkpoint = _resolve(project_root, str(method["checkpoint"]))
            if _sha256(checkpoint) != str(method["checkpoint_sha256"]):
                raise ValueError(f"DESCAN checkpoint changed: {name}")
            summary = _resolve(project_root, str(method["validation_summary"]))
            if _sha256(summary) != str(method["validation_summary_sha256"]):
                raise ValueError(f"DESCAN validation summary changed: {name}")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            arm = _method_arm(method)
            if str(saved.get("representation_arm")) != arm:
                raise ValueError(f"DESCAN checkpoint representation changed: {name}")
            if int(saved.get("training_seed", -1)) != int(method["seed"]):
                raise ValueError(f"DESCAN checkpoint seed changed: {name}")
            if saved.get("selection_rule") != "fixed_final_epoch":
                raise ValueError(f"DESCAN checkpoint selection rule changed: {name}")
            method["arm"] = arm
            method["checkpoint_path"] = checkpoint
            method["threshold"] = float(method["validation_threshold"])
            method["implementation_sha256"] = str(method["checkpoint_sha256"])
        elif method["kind"] == "nonlearned":
            if name not in NONLEARNED_METHODS:
                raise ValueError(f"unsupported DESCAN nonlearned method: {name}")
            if method.get("status") != "ready_for_toy":
                raise ValueError(f"nonlearned DESCAN method is not frozen: {name}")
            threshold_path = _resolve(project_root, str(method["threshold_artifact"]))
            if _sha256(threshold_path) != str(method["threshold_artifact_sha256"]):
                raise ValueError(f"DESCAN threshold artifact changed: {name}")
            method["threshold"] = float(method["pixel_threshold"])
            method["implementation_sha256"] = str(
                method.get("source_sha256", method.get("implementation"))
            )
            if name == "visualdiff_style_dense_sift":
                source_path = _resolve(project_root, str(method["source"]))
                if _sha256(source_path) != str(method["source_sha256"]):
                    raise ValueError("VisualDiff-style source changed")
        else:
            raise ValueError(f"unsupported DESCAN method kind: {method['kind']}")
        if not 0.0 <= float(method["threshold"]) <= 1.0:
            raise ValueError(f"invalid frozen DESCAN threshold: {name}")
        by_name[name] = method
    if list(by_name) != expected_names:
        raise ValueError("DESCAN method order or membership changed")
    return [by_name[name] for name in expected_names]


def _registered_reference(
    reference: np.ndarray, homography: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = reference.shape[:2]
    registered = cv2.warpPerspective(
        reference,
        homography.astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    valid = cv2.warpPerspective(
        np.ones((height, width), dtype=np.uint8),
        homography.astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return registered, valid


def _compute_score(
    method: dict[str, Any],
    candidate: np.ndarray,
    reference: np.ndarray,
    condition: str,
    audit: dict[str, Any],
    model: torch.nn.Module | None,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    name = str(method["name"])
    if model is not None:
        score = _infer_pair_tiled(
            model,
            candidate,
            reference,
            str(method["arm"]),
            device,
            config["inference"],
            config["preprocessing"],
        )
        return score, np.ones(score.shape, dtype=bool), {
            "reference_registration": "dataset_supplied_alignment",
            "arm": str(method["arm"]),
        }
    if name in {"raw_rgb_difference", "ssim_distance"}:
        valid = np.ones(candidate.shape[:2], dtype=bool)
        scoring_reference = reference
        registration = "identity_scan_reference"
        if condition == "digital_reference":
            homography = np.asarray(audit["registration"]["homography"], dtype=np.float64)
            scoring_reference, valid = _registered_reference(reference, homography)
            registration = "frozen_pre_scoring_ecc_homography"
        if name == "raw_rgb_difference":
            score = np.max(
                np.abs(
                    candidate.astype(np.float32)
                    - scoring_reference.astype(np.float32)
                ),
                axis=2,
            ) / 255.0
        else:
            score = _ssim_distance(candidate, scoring_reference, config["ssim"])
        return score, valid, {
            "reference_registration": registration,
            "valid_support_fraction": float(valid.mean()),
        }
    if name == "visualdiff_style_dense_sift":
        result = visualdiff_style_score(candidate, reference)
        return result.score_map, result.valid_mask, result.metadata
    raise ValueError(f"unsupported DESCAN method: {name}")


def _top_fraction_mean(score: np.ndarray, valid: np.ndarray, fraction: float) -> float:
    values = np.asarray(score[valid], dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("DESCAN image score requires finite valid pixels")
    count = max(1, int(math.ceil(values.size * fraction)))
    return float(np.partition(values, values.size - count)[-count:].mean())


def _scanner(basename: str) -> str:
    value = basename.split("_", 1)[0]
    if value not in {"scanner01", "scanner02"}:
        raise ValueError(f"unexpected DESCAN scanner basename: {basename}")
    return value


def _bootstrap_group_mean(
    values: dict[str, list[float]], seed: int, replicates: int
) -> tuple[float, float]:
    groups = sorted(values)
    if not groups:
        raise ValueError("DESCAN bootstrap requires source groups")
    group_values = np.asarray(
        [np.mean(values[group]) for group in groups], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        samples[index] = float(
            rng.choice(group_values, len(group_values), replace=True).mean()
        )
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _tpr_at_fixed_fpr(
    forged: np.ndarray, authentic: np.ndarray, target: float
) -> dict[str, float]:
    thresholds = np.r_[
        np.unique(authentic), np.nextafter(float(authentic.max()), np.inf)
    ]
    eligible = [
        float(threshold)
        for threshold in thresholds
        if float(np.mean(authentic >= threshold)) <= target + 1e-12
    ]
    if not eligible:
        raise ValueError("DESCAN fixed-FPR endpoint has no eligible threshold")
    threshold = min(eligible)
    return {
        "tpr": float(np.mean(forged >= threshold)),
        "observed_fpr": float(np.mean(authentic >= threshold)),
        "evaluation_threshold": threshold,
    }


def _summarize_records(
    records: list[dict[str, Any]], statistics: dict[str, Any]
) -> dict[str, Any]:
    failures = [record for record in records if record["status"] != "ok"]
    forged = [
        record
        for record in records
        if record["status"] == "ok" and record["sample_kind"] == "forged"
    ]
    authentic = [
        record
        for record in records
        if record["status"] == "ok" and record["sample_kind"] == "authentic"
    ]
    if not forged or not authentic:
        return {
            "status": "incomplete",
            "failures": len(failures),
            "forged_ok": len(forged),
            "authentic_ok": len(authentic),
        }
    grouped_ap: dict[str, list[float]] = defaultdict(list)
    grouped_f1: dict[str, list[float]] = defaultdict(list)
    grouped_iou: dict[str, list[float]] = defaultdict(list)
    attack_ap: dict[str, list[float]] = defaultdict(list)
    scanner_ap: dict[str, list[float]] = defaultdict(list)
    for record in forged:
        group = str(record["source_group_id"])
        grouped_ap[group].append(float(record["pixel_ap"]))
        grouped_f1[group].append(float(record["pixel_f1"]))
        grouped_iou[group].append(float(record["pixel_iou"]))
        attack_ap[str(record["attack"])].append(float(record["pixel_ap"]))
        scanner_ap[str(record["scanner"])].append(float(record["pixel_ap"]))
    low, high = _bootstrap_group_mean(
        grouped_ap,
        int(statistics["bootstrap_seed"]),
        int(statistics["bootstrap_replicates"]),
    )
    forged_image = np.asarray([record["image_score"] for record in forged], dtype=float)
    authentic_image = np.asarray(
        [record["image_score"] for record in authentic], dtype=float
    )
    _, image_auroc = _ranking_metrics(
        np.r_[forged_image, authentic_image],
        np.r_[
            np.ones(forged_image.size, dtype=bool),
            np.zeros(authentic_image.size, dtype=bool),
        ],
    )
    result: dict[str, Any] = {
        "status": "complete" if not failures else "incomplete",
        "failures": len(failures),
        "source_groups": len(grouped_ap),
        "forged_ok": len(forged),
        "authentic_ok": len(authentic),
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
            key: float(np.mean(values)) for key, values in sorted(attack_ap.items())
        },
        "scanner_macro_pixel_ap": {
            key: float(np.mean(values)) for key, values in sorted(scanner_ap.items())
        },
        "authentic_document_macro_pixel_fpr": float(
            np.mean([record["pixel_fpr"] for record in authentic])
        ),
        "authentic_image_fpr": float(
            np.mean([record["positive_pixels"] > 0 for record in authentic])
        ),
        "image_auroc_top_fraction": float(image_auroc),
    }
    for target in statistics["fixed_fpr_targets"]:
        suffix = str(target).replace(".", "p")
        fixed = _tpr_at_fixed_fpr(forged_image, authentic_image, float(target))
        result[f"image_tpr_at_fpr_{suffix}"] = fixed["tpr"]
        result[f"image_observed_fpr_at_target_{suffix}"] = fixed["observed_fpr"]
        result[f"image_evaluation_threshold_at_fpr_{suffix}"] = fixed[
            "evaluation_threshold"
        ]
    return result


def _paired_comparisons(
    records: list[dict[str, Any]],
    comparisons: list[dict[str, str]],
    conditions: list[str],
    statistics: dict[str, Any],
) -> list[dict[str, Any]]:
    values: dict[tuple[str, str, str, str], float] = {}
    for record in records:
        if record["status"] != "ok" or record["sample_kind"] != "forged":
            continue
        key = (
            str(record["method"]),
            str(record["reference_condition"]),
            str(record["attack"]),
            str(record["source_group_id"]),
        )
        if key in values:
            raise ValueError(f"duplicate DESCAN forged comparison record: {key}")
        values[key] = float(record["pixel_ap"])
    attacks = ["pooled", "copy_move", "local_erase"]
    output: list[dict[str, Any]] = []
    for comparison_index, comparison in enumerate(comparisons):
        left = str(comparison["left"])
        right = str(comparison["right"])
        for condition_index, condition in enumerate(conditions):
            for attack_index, attack in enumerate(attacks):
                grouped_delta: dict[str, list[float]] = defaultdict(list)
                group_ids = sorted(
                    {
                        group
                        for method, current_condition, _, group in values
                        if method == left and current_condition == condition
                    }
                )
                for group in group_ids:
                    scopes = (
                        ["copy_move", "local_erase"] if attack == "pooled" else [attack]
                    )
                    left_values = [
                        values[(left, condition, scope, group)] for scope in scopes
                    ]
                    right_values = [
                        values[(right, condition, scope, group)] for scope in scopes
                    ]
                    grouped_delta[group].append(
                        float(np.mean(left_values) - np.mean(right_values))
                    )
                if not grouped_delta:
                    continue
                low, high = _bootstrap_group_mean(
                    grouped_delta,
                    int(statistics["bootstrap_seed"])
                    + comparison_index * 10_007
                    + condition_index * 101
                    + attack_index,
                    int(statistics["bootstrap_replicates"]),
                )
                delta = np.asarray(
                    [items[0] for items in grouped_delta.values()], dtype=float
                )
                output.append(
                    {
                        "comparison": str(comparison["name"]),
                        "left": left,
                        "right": right,
                        "reference_condition": condition,
                        "attack": attack,
                        "source_groups": len(grouped_delta),
                        "document_macro_pixel_ap_delta": float(delta.mean()),
                        "delta_ci95_low": low,
                        "delta_ci95_high": high,
                        "groups_left_better": int(np.count_nonzero(delta > 0)),
                        "groups_tied": int(np.count_nonzero(delta == 0)),
                        "groups_right_better": int(np.count_nonzero(delta < 0)),
                    }
                )
    return output


def _metric_rows(
    predictions: list[dict[str, Any]],
    methods: list[dict[str, Any]],
    conditions: list[str],
    statistics: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in methods:
        for condition in conditions:
            selected = [
                record
                for record in predictions
                if record["method"] == method["name"]
                and record["reference_condition"] == condition
            ]
            summary = _summarize_records(selected, statistics)
            output.append(
                {
                    "method": method["name"],
                    "family": method.get("family", method["name"]),
                    "kind": method["kind"],
                    "seed": method.get("seed"),
                    "reference_condition": condition,
                    "validation_frozen_threshold": method["threshold"],
                    **{
                        key: value
                        for key, value in summary.items()
                        if not isinstance(value, dict)
                    },
                    "attack_macro_pixel_ap_json": json.dumps(
                        summary.get("attack_macro_pixel_ap", {}), sort_keys=True
                    ),
                    "scanner_macro_pixel_ap_json": json.dumps(
                        summary.get("scanner_macro_pixel_ap", {}), sort_keys=True
                    ),
                }
            )
    return output


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluator_sha256 = _sha256(Path(__file__).resolve())
    if evaluator_sha256 != str(config["experiment"]["expected_evaluator_sha256"]):
        raise ValueError("DESCAN evaluator source changed")
    runtime = config["runtime"]
    if (
        not bool(runtime["model_scoring_authorized"])
        or bool(runtime["model_training_authorized"])
        or bool(runtime["threshold_selection_authorized"])
        or bool(runtime["sample_selection_authorized"])
    ):
        raise ValueError("DESCAN scoring authorization boundary changed")
    if [str(value) for value in config["reference_conditions"]] != REFERENCE_CONDITIONS:
        raise ValueError("DESCAN reference-condition order changed")
    for binding in config["bindings"]:
        path = _resolve(project_root, str(binding["path"]))
        if _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"bound DESCAN scoring artifact changed: {path}")

    paths = config["paths"]
    scratch = _scratch_root(project_root, paths)
    for name in ("manifest", "audit_records"):
        specification = config["data"][name]
        path = _resolve(project_root, str(specification["path"]))
        if _sha256(path) != str(specification["expected_sha256"]):
            raise ValueError(f"DESCAN scoring data artifact changed: {name}")
    manifest_path = _resolve(project_root, str(config["data"]["manifest"]["path"]))
    rows = _read_jsonl(manifest_path)
    if len(rows) != int(config["data"]["expected_groups"]):
        raise ValueError("DESCAN scoring group count changed")
    if len({str(row["source_group_id"]) for row in rows}) != len(rows):
        raise ValueError("DESCAN scoring source groups are not unique")
    audit_path = _resolve(project_root, str(config["data"]["audit_records"]["path"]))
    audit_rows = _read_jsonl(audit_path)
    audit_by_group = {str(row["source_group_id"]): row for row in audit_rows}
    if set(audit_by_group) != {str(row["source_group_id"]) for row in rows}:
        raise ValueError("DESCAN audit and scoring group sets differ")
    flagged_groups = set(str(value) for value in config["data"]["flagged_groups"])
    if not flagged_groups <= set(audit_by_group):
        raise ValueError("DESCAN flagged group is outside the scoring manifest")

    registry_path = _resolve(
        project_root, str(config["method_registry"]["path"])
    )
    if _sha256(registry_path) != str(config["method_registry"]["expected_sha256"]):
        raise ValueError("DESCAN scoring method registry changed")
    methods = _validate_methods(
        project_root,
        _read_jsonl(registry_path),
        [str(value) for value in config["method_registry"]["expected_names"]],
    )
    conditions = [str(value) for value in config["reference_conditions"]]
    device = torch.device(str(runtime["device"]))
    if any(method["kind"] == "learned" for method in methods):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("DESCAN learned scoring requires CUDA")
        encoder_path = _resolve(
            project_root, str(config["model"]["encoder_weights"])
        )
        if _sha256(encoder_path) != str(config["model"]["encoder_weights_sha256"]):
            raise ValueError("DESCAN encoder initialization changed")
        encoder_state = torch.load(encoder_path, map_location="cpu", weights_only=True)
        torch.cuda.reset_peak_memory_stats()
    else:
        encoder_state = None
    torch.set_num_threads(max(1, int(runtime.get("torch_threads", 1))))

    predictions_path = _resolve(project_root, str(paths["predictions"]))
    if predictions_path.exists():
        raise FileExistsError(f"DESCAN prediction output already exists: {predictions_path}")
    partial_path = predictions_path.with_suffix(predictions_path.suffix + ".partial")
    if partial_path.exists():
        raise FileExistsError(f"DESCAN partial output requires audit: {partial_path}")
    score_cache_dir = _resolve(scratch, str(paths["score_cache_dir"]))
    predictions: list[dict[str, Any]] = []
    cache_hits = 0
    cache_bytes = 0
    started = time.monotonic()

    for method_index, method in enumerate(methods):
        model: torch.nn.Module | None = None
        if method["kind"] == "learned":
            assert encoder_state is not None
            model = _build_model(str(method["arm"]), encoder_state)
            checkpoint = torch.load(
                method["checkpoint_path"], map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model = model.to(device).eval()
        for row in rows:
            group = str(row["source_group_id"])
            basename = str(row["source_basename"])
            scanner = _scanner(basename)
            audit = audit_by_group[group]
            scan_path = _scratch_path(scratch, str(row["scan"]))
            clean_path = _scratch_path(scratch, str(row["clean"]))
            scan = _load_rgb(scan_path)
            clean = _load_rgb(clean_path)
            for condition in conditions:
                reference = scan if condition == "scan_reference" else clean
                reference_sha256 = (
                    str(row["scan_sha256"])
                    if condition == "scan_reference"
                    else str(row["clean_sha256"])
                )
                items: list[tuple[str, str | None, np.ndarray, str, np.ndarray | None]] = [
                    (
                        "authentic",
                        None,
                        scan,
                        str(row["scan_sha256"]),
                        None,
                    )
                ]
                for attack_name in ("copy_move", "local_erase"):
                    attack = row["attacks"][attack_name]
                    candidate_path = _scratch_path(scratch, str(attack["candidate"]))
                    mask_path = _scratch_path(scratch, str(attack["mask"]))
                    items.append(
                        (
                            "forged",
                            attack_name,
                            _load_rgb(candidate_path),
                            str(attack["candidate_sha256"]),
                            _load_mask(mask_path),
                        )
                    )
                for sample_kind, attack_name, candidate, candidate_sha256, mask in items:
                    sample_id = (
                        f"{group}:{condition}:authentic"
                        if sample_kind == "authentic"
                        else f"{group}:{condition}:{attack_name}"
                    )
                    base_record = {
                        "method": str(method["name"]),
                        "family": method.get("family", method["name"]),
                        "kind": method["kind"],
                        "seed": method.get("seed"),
                        "reference_condition": condition,
                        "sample_kind": sample_kind,
                        "sample_id": sample_id,
                        "source_group_id": group,
                        "source_basename": basename,
                        "scanner": scanner,
                        "attack": attack_name,
                        "green_audit_flag": group in flagged_groups,
                        "candidate_sha256": candidate_sha256,
                        "reference_sha256": reference_sha256,
                        "validation_frozen_threshold": float(method["threshold"]),
                        "model_training_performed": False,
                        "threshold_selection_performed": False,
                        "sample_selection_performed": False,
                    }
                    try:
                        score, valid, score_metadata, cache = _score_cache(
                            score_cache_dir,
                            {
                                "schema": 1,
                                "stage": str(config["experiment"]["stage"]),
                                "method": str(method["name"]),
                                "implementation": method["implementation_sha256"],
                                "candidate": candidate_sha256,
                                "reference": reference_sha256,
                                "reference_condition": condition,
                                "audit_records": str(
                                    config["data"]["audit_records"]["expected_sha256"]
                                ),
                                "evaluator": evaluator_sha256,
                                "preprocessing": config["preprocessing"],
                                "inference": config["inference"],
                            },
                            candidate.shape[:2],
                            lambda m=method, c=candidate, r=reference, co=condition, a=audit, mo=model: _compute_score(
                                m, c, r, co, a, mo, device, config
                            ),
                        )
                        cache_hits += int(cache["cache_hit"])
                        cache_bytes += int(cache["cache_bytes"])
                        threshold = float(method["threshold"])
                        binary = score >= threshold
                        image_score = _top_fraction_mean(
                            score,
                            valid,
                            float(config["image_level"]["top_fraction"]),
                        )
                        record = {
                            **base_record,
                            "status": "ok",
                            "failure_type": None,
                            "failure_reason": None,
                            "valid_pixels": int(valid.sum()),
                            "valid_support_fraction": float(valid.mean()),
                            "image_score": image_score,
                            "positive_pixels": int(np.logical_and(binary, valid).sum()),
                            "score_cache_key": cache["cache_key"],
                            "score_cache_path": str(
                                cache["cache_path"].relative_to(scratch)
                            ),
                            "score_cache_sha256": cache["cache_sha256"],
                            "score_cache_hit": bool(cache["cache_hit"]),
                            "score_metadata": score_metadata,
                        }
                        if sample_kind == "forged":
                            assert mask is not None
                            if mask.shape != valid.shape or not np.logical_and(mask, valid).any():
                                raise ValueError("forged mask has no positive valid support")
                            pixel_ap, pixel_auroc = _ranking_metrics(
                                score[valid], mask[valid]
                            )
                            tp = int(np.logical_and(binary & mask, valid).sum())
                            fp = int(np.logical_and(binary & ~mask, valid).sum())
                            fn = int(np.logical_and(~binary & mask, valid).sum())
                            record.update(
                                {
                                    "pixel_ap": float(pixel_ap),
                                    "pixel_auroc": float(pixel_auroc),
                                    "pixel_f1": float(
                                        2 * tp / max(1, 2 * tp + fp + fn)
                                    ),
                                    "pixel_iou": float(tp / max(1, tp + fp + fn)),
                                    "tp": tp,
                                    "fp": fp,
                                    "fn": fn,
                                    "mask_positive_pixels": int(mask.sum()),
                                    "mask_valid_support_fraction": float(
                                        np.logical_and(mask, valid).sum() / mask.sum()
                                    ),
                                }
                            )
                        else:
                            record["pixel_fpr"] = float(
                                record["positive_pixels"] / record["valid_pixels"]
                            )
                    except Exception as error:
                        record = {
                            **base_record,
                            "status": "error",
                            "failure_type": type(error).__name__,
                            "failure_reason": str(error),
                        }
                    predictions.append(record)
        _write_jsonl(partial_path, predictions)
        progress_path = _resolve(project_root, str(paths["progress"]))
        _write_json(
            progress_path,
            {
                "status": "descan18k_scoring_in_progress",
                "stage": str(config["experiment"]["stage"]),
                "methods_completed": method_index + 1,
                "methods_total": len(methods),
                "prediction_records": len(predictions),
                "failures": sum(row["status"] != "ok" for row in predictions),
                "model_selection_performed": False,
            },
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.replace(predictions_path)
    metrics = _metric_rows(predictions, methods, conditions, config["statistics"])
    comparisons = _paired_comparisons(
        predictions,
        list(config["comparisons"]),
        conditions,
        config["statistics"],
    )
    metrics_path = _resolve(project_root, str(paths["metrics"]))
    comparisons_path = _resolve(project_root, str(paths["comparisons"]))
    _write_csv(metrics_path, metrics)
    _write_csv(comparisons_path, comparisons)
    failures = [record for record in predictions if record["status"] != "ok"]
    cache_paths = {
        str(record["score_cache_path"])
        for record in predictions
        if record["status"] == "ok"
    }
    summary = {
        "status": (
            f"descan18k_{config['experiment']['stage']}_scoring_complete"
            if not failures
            else f"descan18k_{config['experiment']['stage']}_scoring_incomplete"
        ),
        "paper_evidence": bool(config["experiment"]["paper_evidence"]),
        "stage": str(config["experiment"]["stage"]),
        "source_groups": len(rows),
        "methods": len(methods),
        "reference_conditions": conditions,
        "expected_prediction_records": len(methods) * len(rows) * len(conditions) * 3,
        "prediction_records": len(predictions),
        "successful_prediction_records": len(predictions) - len(failures),
        "failed_prediction_records": len(failures),
        "silent_failures": 0,
        "flagged_groups_included": sorted(flagged_groups),
        "model_training_performed": False,
        "threshold_selection_performed": False,
        "sample_selection_performed": False,
        "cache_hits": cache_hits,
        "unique_score_cache_artifacts": len(cache_paths),
        "score_cache_referenced_bytes": sum(
            _scratch_path(scratch, value).stat().st_size for value in cache_paths
        ),
        "score_cache_observation_bytes_with_repeats": cache_bytes,
        "wall_seconds": time.monotonic() - started,
        "peak_vram_mib": (
            float(torch.cuda.max_memory_allocated() / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        "predictions": str(predictions_path.relative_to(project_root)),
        "predictions_sha256": _sha256(predictions_path),
        "metrics": str(metrics_path.relative_to(project_root)),
        "metrics_sha256": _sha256(metrics_path),
        "comparisons": str(comparisons_path.relative_to(project_root)),
        "comparisons_sha256": _sha256(comparisons_path),
        "method_registry_sha256": _sha256(registry_path),
        "evaluator_sha256": evaluator_sha256,
        "config_sha256": _sha256(config_path),
    }
    summary_path = _resolve(project_root, str(paths["summary"]))
    _write_json(summary_path, summary)
    _write_json(
        _resolve(project_root, str(paths["progress"])),
        {
            "status": summary["status"],
            "methods_completed": len(methods),
            "methods_total": len(methods),
            "prediction_records": len(predictions),
            "failures": len(failures),
            "summary": str(summary_path.relative_to(project_root)),
            "summary_sha256": _sha256(summary_path),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
