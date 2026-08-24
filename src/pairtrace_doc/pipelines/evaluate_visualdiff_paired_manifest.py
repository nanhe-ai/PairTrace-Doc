from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from pairtrace_doc.baselines.visualdiff_style import (
    VisualDiffAlignmentError,
    visualdiff_style_score,
)
from pairtrace_doc.metrics import average_precision


METHOD = "visualdiff_style_dense_sift"


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
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cache_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _resize_pair(
    candidate: np.ndarray, reference: np.ndarray, max_side: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if candidate.shape != reference.shape:
        raise ValueError("VisualDiff paired inputs must have matched native RGB geometry")
    height, width = candidate.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    if target == (width, height):
        return candidate, reference, 1.0, 1.0
    return (
        cv2.resize(candidate, target, interpolation=cv2.INTER_AREA),
        cv2.resize(reference, target, interpolation=cv2.INTER_AREA),
        target[0] / width,
        target[1] / height,
    )


def _top_fraction_mean(scores: np.ndarray, fraction: float) -> float:
    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("image score requires finite valid-support pixels")
    count = max(1, int(math.ceil(len(values) * fraction)))
    selected = np.partition(values, len(values) - count)[-count:]
    return float(selected.mean())


def _threshold_counts(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive_prefix = np.r_[0, np.cumsum(sorted_labels, dtype=np.int64)]
    indices = np.searchsorted(sorted_scores, thresholds, side="left")
    total_positive = int(sorted_labels.sum())
    total_negative = len(sorted_labels) - total_positive
    tp = total_positive - positive_prefix[indices]
    predicted = len(sorted_labels) - indices
    fp = predicted - tp
    fn = total_positive - tp
    if total_negative < 0:
        raise AssertionError("negative count invariant failed")
    return tp.astype(np.float64), fp.astype(np.float64), fn.astype(np.float64)


def _f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denominator = 2.0 * tp + fp + fn
    return np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator > 0,
    )


def _select_thresholds(
    forged: list[dict[str, Any]],
    authentic: list[dict[str, Any]],
    specification: dict[str, Any],
) -> dict[str, Any]:
    thresholds = np.arange(
        float(specification["candidate_min"]),
        float(specification["candidate_max"]) + 1e-12,
        float(specification["candidate_step"]),
        dtype=np.float64,
    )
    document_f1 = []
    for row in forged:
        if row["status"] != "ok":
            document_f1.append(np.zeros_like(thresholds))
            continue
        tp, fp, fn = _threshold_counts(row["valid_scores"], row["valid_labels"], thresholds)
        document_f1.append(_f1(tp, fp, fn))
    authentic_fpr = []
    for row in authentic:
        if row["status"] != "ok":
            authentic_fpr.append(np.ones_like(thresholds))
            continue
        scores = np.sort(np.asarray(row["valid_scores"], dtype=np.float64))
        indices = np.searchsorted(scores, thresholds, side="left")
        authentic_fpr.append((len(scores) - indices) / len(scores))
    mean_f1 = np.mean(document_f1, axis=0)
    mean_fpr = np.mean(authentic_fpr, axis=0)
    eligible = mean_fpr <= float(specification["authentic_pixel_fpr_max"])
    if not eligible.any():
        raise ValueError("no VisualDiff pixel threshold satisfies the authentic-FPR gate")
    best_value = float(mean_f1[eligible].max())
    best_indices = np.flatnonzero(eligible & np.isclose(mean_f1, best_value))
    pixel_index = int(best_indices[-1])

    forged_image_scores = np.asarray(
        [1.0 if row["status"] != "ok" else row["image_score"] for row in forged]
    )
    authentic_image_scores = np.asarray(
        [1.0 if row["status"] != "ok" else row["image_score"] for row in authentic]
    )
    image_tp, image_fp, image_fn = _threshold_counts(
        np.r_[forged_image_scores, authentic_image_scores],
        np.r_[np.ones(len(forged)), np.zeros(len(authentic))],
        thresholds,
    )
    image_f1 = _f1(image_tp, image_fp, image_fn)
    image_fpr = np.asarray(
        [np.mean(authentic_image_scores >= threshold) for threshold in thresholds]
    )
    image_eligible = image_fpr <= float(specification["authentic_image_fpr_max"])
    if not image_eligible.any():
        raise ValueError("no VisualDiff image threshold satisfies the authentic-FPR gate")
    image_best = float(image_f1[image_eligible].max())
    image_indices = np.flatnonzero(image_eligible & np.isclose(image_f1, image_best))
    image_index = int(image_indices[-1])
    return {
        "method": METHOD,
        "pixel": {
            "threshold": float(thresholds[pixel_index]),
            "development_document_macro_pixel_f1": float(mean_f1[pixel_index]),
            "development_authentic_pixel_fpr": float(mean_fpr[pixel_index]),
        },
        "image": {
            "threshold": float(thresholds[image_index]),
            "development_image_f1": float(image_f1[image_index]),
            "development_authentic_image_fpr": float(image_fpr[image_index]),
        },
        "selection_partition": "AIForge development validation",
        "selection_used_test_or_evaluation": False,
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        raise ValueError("AUROC requires both classes")
    return float(
        np.mean(positive[:, None] > negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def _score_item(
    *,
    candidate_path: Path,
    reference_path: Path,
    candidate_sha256: str,
    reference_sha256: str,
    mask_path: Path | None,
    mask_sha256: str | None,
    max_side: int,
    top_fraction: float,
    cache_dir: Path,
    scratch: Path,
    source_sha256: str,
) -> dict[str, Any]:
    key = _cache_key(
        {
            "schema": 2,
            "method": METHOD,
            "source_sha256": source_sha256,
            "candidate_sha256": candidate_sha256,
            "reference_sha256": reference_sha256,
            "mask_sha256": mask_sha256,
            "max_side": max_side,
            "top_fraction": top_fraction,
        }
    )
    cache_path = cache_dir / key[:2] / f"{key}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            scores = archive["scores"].astype(np.float32)
            valid = archive["valid"].astype(bool)
            labels = archive["labels"].astype(bool)
            image_score = float(archive["image_score"].item())
            metadata = json.loads(str(archive["metadata_json"].item()))
        cache_hit = True
    else:
        if _sha256(candidate_path) != candidate_sha256:
            raise ValueError(f"candidate SHA-256 changed: {candidate_path}")
        if _sha256(reference_path) != reference_sha256:
            raise ValueError(f"reference SHA-256 changed: {reference_path}")
        with Image.open(candidate_path) as handle:
            native_candidate = np.asarray(handle.convert("RGB"))
        with Image.open(reference_path) as handle:
            native_reference = np.asarray(handle.convert("RGB"))
        candidate, reference, scale_x, scale_y = _resize_pair(
            native_candidate, native_reference, max_side
        )
        if candidate_sha256 == reference_sha256 and mask_path is None:
            scores = np.zeros(candidate.shape[:2], dtype=np.float32)
            support = np.ones(candidate.shape[:2], dtype=np.uint8)
            kernel = np.ones((17, 17), dtype=np.uint8)
            valid = cv2.erode(
                support, kernel, borderType=cv2.BORDER_CONSTANT
            ).astype(bool)
            scores[~valid] = np.nan
            result_metadata = {
                "implementation": "pairtrace_visualdiff_style_dense_sift_v1",
                "official_implementation": False,
                "identical_checksum_identity_fast_path": True,
                "dense_grid_step": 8,
                "dense_keypoint_size": 16,
                "valid_support_fraction": float(valid.mean()),
                "homography": np.eye(3, dtype=float).tolist(),
            }
        else:
            result = visualdiff_style_score(candidate, reference)
            scores = result.score_map.astype(np.float32)
            valid = result.valid_mask.astype(bool)
            result_metadata = result.metadata
        if mask_path is None:
            labels = np.zeros(scores.shape, dtype=bool)
        else:
            if mask_sha256 is None or _sha256(mask_path) != mask_sha256:
                raise ValueError(f"mask SHA-256 changed: {mask_path}")
            with Image.open(mask_path) as handle:
                native_mask = np.asarray(handle.convert("L")) > 0
            labels = cv2.resize(
                native_mask.astype(np.uint8),
                (scores.shape[1], scores.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if not valid.any() or (mask_path is not None and not np.any(labels & valid)):
            raise ValueError("VisualDiff valid support contains no evaluable positive pixels")
        image_score = _top_fraction_mean(scores, top_fraction)
        metadata = {
            **result_metadata,
            "native_height": int(native_candidate.shape[0]),
            "native_width": int(native_candidate.shape[1]),
            "model_height": int(scores.shape[0]),
            "model_width": int(scores.shape[1]),
            "resize_scale_x": scale_x,
            "resize_scale_y": scale_y,
        }
        temporary = cache_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                scores=scores,
                valid=valid.astype(np.uint8),
                labels=labels.astype(np.uint8),
                image_score=np.asarray(image_score),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        temporary.replace(cache_path)
        cache_hit = False
    valid_scores = scores[valid]
    valid_labels = labels[valid]
    record = {
        "status": "ok",
        "cache_key": key,
        "score_cache": str(cache_path.relative_to(scratch)),
        "cache_hit": cache_hit,
        "valid_support_fraction": float(valid.mean()),
        "image_score": image_score,
        "metadata": metadata,
        "valid_scores": valid_scores,
        "valid_labels": valid_labels,
    }
    if valid_labels.any():
        record["pixel_ap"] = average_precision(valid_scores, valid_labels)
    return record


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    if not bool(runtime["cpu_inference_authorized"]):
        raise PermissionError("VisualDiff CPU inference is not authorized")
    if bool(runtime["model_training_authorized"]):
        raise ValueError("VisualDiff evaluator cannot authorize model training")
    experiment = config["experiment"]
    protocol_path = _resolve(project_root, str(experiment["protocol"]))
    source_path = _resolve(project_root, str(experiment["source"]))
    if _sha256(protocol_path) != str(experiment["expected_protocol_sha256"]):
        raise ValueError("direct paired-baseline protocol changed")
    source_sha256 = _sha256(source_path)
    if source_sha256 != str(experiment["expected_source_sha256"]):
        raise ValueError("VisualDiff-style source changed")
    if cv2.__version__ != str(config["preprocessing"]["opencv_version"]):
        raise ValueError("OpenCV version changed")
    input_spec = config["input"]
    manifest_path = _resolve(project_root, str(input_spec["manifest"]))
    if _sha256(manifest_path) != str(input_spec["expected_manifest_sha256"]):
        raise ValueError("paired evaluation manifest changed")
    rows = _read_jsonl(manifest_path)
    if len(rows) != int(input_spec["expected_records"]):
        raise ValueError("paired evaluation manifest record count changed")
    scratch = Path(
        os.environ.get(
            str(config["paths"]["scratch_env"]),
            str(_resolve(project_root, str(config["paths"]["scratch_default"]))),
        )
    ).resolve()
    cache_dir = scratch / str(config["paths"]["score_cache_dir"])
    fields = input_spec["fields"]
    forged_internal: list[dict[str, Any]] = []
    authentic_internal: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    start = time.perf_counter()

    def evaluate(
        row: dict[str, Any], sample_kind: str, candidate_field: str, mask: bool
    ) -> dict[str, Any]:
        group = str(row[fields["group_id"]])
        base = {
            "record_id": f"{METHOD}:{sample_kind}:{row[fields['record_id']]}",
            "source_group_id": group,
            "sample_kind": sample_kind,
            "attack": row.get(fields["attack"]) if fields.get("attack") else None,
            "method": METHOD,
            "official_implementation": False,
            "status": "failed",
            "paper_evidence": bool(experiment["paper_evidence"]),
            "failure_metric_policy": "worst_case",
        }
        try:
            result = _score_item(
                candidate_path=_resolve(scratch, str(row[candidate_field])),
                reference_path=_resolve(scratch, str(row[fields["reference"]])),
                candidate_sha256=str(row[fields["candidate_sha256"]])
                if sample_kind == "forged"
                else str(row[fields["reference_sha256"]]),
                reference_sha256=str(row[fields["reference_sha256"]]),
                mask_path=_resolve(scratch, str(row[fields["mask"]])) if mask else None,
                mask_sha256=str(row[fields["mask_sha256"]]) if mask else None,
                max_side=int(config["preprocessing"]["max_side"]),
                top_fraction=float(config["image_score"]["top_fraction"]),
                cache_dir=cache_dir,
                scratch=scratch,
                source_sha256=source_sha256,
            )
            base.update(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"valid_scores", "valid_labels"}
                }
            )
            result.update(base)
            return result
        except Exception as error:
            base.update(
                {
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                    "alignment_failure": isinstance(error, VisualDiffAlignmentError),
                }
            )
            return base

    tasks: list[tuple[dict[str, Any], str, str, bool]] = []
    authentic_seen: set[str] = set()
    for row in rows:
        tasks.append((row, "forged", fields["candidate"], True))
        group = str(row[fields["group_id"]])
        if group not in authentic_seen:
            authentic_seen.add(group)
            tasks.append((row, "authentic", fields["reference"], False))

    def execute(
        task: tuple[dict[str, Any], str, str, bool]
    ) -> tuple[str, dict[str, Any]]:
        row, sample_kind, candidate_field, has_mask = task
        return sample_kind, evaluate(row, sample_kind, candidate_field, has_mask)

    workers = int(runtime.get("workers", 1))
    if workers < 1 or workers > 8:
        raise ValueError("VisualDiff worker count must be between 1 and 8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        completed = list(executor.map(execute, tasks))
    for sample_kind, item in completed:
        if sample_kind == "forged":
            forged_internal.append(item)
        else:
            authentic_internal.append(item)
        predictions.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"valid_scores", "valid_labels"}
            }
        )
    success_fraction = np.mean(
        [row["status"] == "ok" for row in forged_internal + authentic_internal]
    )
    if success_fraction < float(config["gate"]["minimum_success_fraction"]):
        status = "visualdiff_success_gate_failed"
    else:
        status = "visualdiff_paired_evaluation_complete"
    thresholds: dict[str, Any]
    if bool(runtime["threshold_selection_authorized"]):
        if bool(experiment["paper_evidence"]):
            raise ValueError("threshold selection cannot be paper evidence")
        thresholds = _select_thresholds(
            forged_internal, authentic_internal, config["operating_point"]
        )
        threshold_source_sha256 = None
    else:
        threshold_path = _resolve(project_root, str(input_spec["thresholds"]))
        if _sha256(threshold_path) != str(input_spec["expected_thresholds_sha256"]):
            raise ValueError("VisualDiff thresholds changed")
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold_source_sha256 = _sha256(threshold_path)
    pixel_threshold = float(thresholds["pixel"]["threshold"])
    image_threshold = float(thresholds["image"]["threshold"])
    forged_aps = [
        float(row["pixel_ap"]) if row["status"] == "ok" else 0.0
        for row in forged_internal
    ]
    authentic_fprs = [
        float(np.mean(row["valid_scores"] >= pixel_threshold))
        if row["status"] == "ok"
        else 1.0
        for row in authentic_internal
    ]
    forged_images = np.asarray(
        [float(row["image_score"]) if row["status"] == "ok" else 1.0 for row in forged_internal]
    )
    authentic_images = np.asarray(
        [float(row["image_score"]) if row["status"] == "ok" else 1.0 for row in authentic_internal]
    )
    groups: dict[str, list[float]] = {}
    attacks: dict[str, list[float]] = {}
    for row, ap in zip(forged_internal, forged_aps, strict=True):
        groups.setdefault(str(row["source_group_id"]), []).append(ap)
        attacks.setdefault(str(row.get("attack") or "unspecified"), []).append(ap)
    metric_row = {
        "method": METHOD,
        "stage": experiment["stage"],
        "forged_items": len(forged_internal),
        "authentic_items": len(authentic_internal),
        "source_groups": len(groups),
        "success_fraction": float(success_fraction),
        "failed_items": int(
            sum(row["status"] != "ok" for row in forged_internal + authentic_internal)
        ),
        "item_macro_pixel_ap_worst_case_failures": float(np.mean(forged_aps)),
        "source_group_macro_pixel_ap_worst_case_failures": float(
            np.mean([np.mean(values) for values in groups.values()])
        ),
        "authentic_pixel_fpr": float(np.mean(authentic_fprs)),
        "authentic_image_fpr": float(np.mean(authentic_images >= image_threshold)),
        "forged_image_tpr": float(np.mean(forged_images >= image_threshold)),
        "image_auroc": _auc(
            np.r_[forged_images, authentic_images],
            np.r_[np.ones(len(forged_images)), np.zeros(len(authentic_images))],
        ),
        "pixel_threshold": pixel_threshold,
        "image_threshold": image_threshold,
        "threshold_selected_on_this_evaluation": bool(
            runtime["threshold_selection_authorized"]
        ),
        "paper_evidence": bool(experiment["paper_evidence"]),
    }
    for attack, values in sorted(attacks.items()):
        metric_row[f"attack_macro_ap__{attack}"] = float(np.mean(values))
    prediction_path = _resolve(project_root, str(config["paths"]["predictions"]))
    metrics_path = _resolve(project_root, str(config["paths"]["metrics"]))
    summary_path = _resolve(project_root, str(config["paths"]["summary"]))
    threshold_output = _resolve(project_root, str(config["paths"]["thresholds"]))
    _write_jsonl(prediction_path, predictions)
    _write_csv(metrics_path, [metric_row])
    if bool(runtime["threshold_selection_authorized"]):
        _write_json(threshold_output, thresholds)
        threshold_source_sha256 = _sha256(threshold_output)
    summary = {
        "status": status,
        "paper_evidence": bool(experiment["paper_evidence"]),
        "post_hoc_existing_evaluation": bool(experiment["post_hoc_existing_evaluation"]),
        "method": METHOD,
        "official_implementation": False,
        "threshold_selection_used": bool(runtime["threshold_selection_authorized"]),
        "threshold_source_sha256": threshold_source_sha256,
        "success_fraction": float(success_fraction),
        "metrics": metric_row,
        "runtime_seconds": time.perf_counter() - start,
        "input_manifest_sha256": _sha256(manifest_path),
        "source_sha256": source_sha256,
        "outputs": {
            "predictions": str(prediction_path.relative_to(project_root)),
            "predictions_sha256": _sha256(prediction_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "thresholds": str(threshold_output.relative_to(project_root)),
            "thresholds_sha256": _sha256(threshold_output),
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
