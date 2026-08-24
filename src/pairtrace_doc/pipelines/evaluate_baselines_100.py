from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


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
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
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
    if not rows:
        raise ValueError("cannot write an empty metric table")
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_native_scores(
    scratch: Path,
    prediction: dict[str, Any],
    native_shape: tuple[int, int],
) -> np.ndarray:
    import cv2

    cache_path = _resolve(scratch, prediction["score_cache"])
    with np.load(cache_path, allow_pickle=False) as cached:
        scores = np.asarray(cached["scores"], dtype=np.float32)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError(f"invalid score cache {cache_path}")
    height, width = native_shape
    if scores.shape != native_shape:
        scores = cv2.resize(scores, (width, height), interpolation=cv2.INTER_LINEAR)
    if scores.shape != native_shape or not np.isfinite(scores).all():
        raise ValueError(f"could not restore {cache_path} to native shape")
    return np.clip(scores, 0.0, 1.0)


def _load_mask(scratch: Path, row: dict[str, Any]) -> np.ndarray:
    mask_path = _resolve(scratch, row["mask"])
    if _sha256(mask_path) != row["mask_sha256"]:
        raise ValueError("ground-truth mask SHA-256 changed")
    with Image.open(mask_path) as handle:
        mask = np.asarray(handle.convert("L")) > 0
    expected = (int(row["height"]), int(row["width"]))
    if mask.shape != expected or not mask.any():
        raise ValueError("forged mask is empty or has the wrong shape")
    return mask


def _ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    flat_labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    positives = int(flat_labels.sum())
    negatives = int(flat_labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("ranking metrics require positive and negative pixels")
    order = np.argsort(flat_scores, kind="mergesort")[::-1]
    ranked_scores = flat_scores[order]
    ranked_labels = flat_labels[order]
    cumulative_tp = np.cumsum(ranked_labels, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - ranked_labels, dtype=np.int64)
    threshold_ends = np.r_[ranked_scores[1:] != ranked_scores[:-1], True]
    tp = cumulative_tp[threshold_ends]
    fp = cumulative_fp[threshold_ends]
    recall = tp / positives
    precision = tp / (tp + fp)
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    tpr = np.r_[0.0, recall]
    fpr = np.r_[0.0, fp / negatives]
    auroc = float(np.trapezoid(tpr, fpr))
    return average_precision, auroc


def _threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    prediction = scores >= threshold
    truth = labels.astype(bool, copy=False)
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = tp + fp + fn
    iou = tp / union if union else 0.0
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
    }


def _candidate_thresholds(config: dict[str, Any]) -> np.ndarray:
    start = float(config["candidate_min"])
    stop = float(config["candidate_max"])
    step = float(config["candidate_step"])
    count = int(round((stop - start) / step))
    thresholds = start + np.arange(count + 1, dtype=np.float64) * step
    if not np.isclose(thresholds[-1], stop):
        raise ValueError("threshold grid does not end at candidate_max")
    return thresholds


def _threshold_count_vectors(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    bins = np.r_[thresholds, np.inf]
    positive_histogram, _ = np.histogram(scores[labels], bins=bins)
    negative_histogram, _ = np.histogram(scores[~labels], bins=bins)
    tp = np.cumsum(positive_histogram[::-1], dtype=np.int64)[::-1]
    fp = np.cumsum(negative_histogram[::-1], dtype=np.int64)[::-1]
    return tp, fp, int(np.count_nonzero(labels))


def _freeze_pixel_threshold(
    scratch: Path,
    manifest_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = _candidate_thresholds(config)
    forged_f1: list[np.ndarray] = []
    authentic_fpr: list[np.ndarray] = []
    validation = [row for row in manifest_rows if row["evaluation_role"] == "validation"]
    for row in validation:
        scores = _load_native_scores(
            scratch,
            predictions[row["record_id"]],
            (int(row["height"]), int(row["width"])),
        )
        if row["sample_kind"] == "forged":
            labels = _load_mask(scratch, row)
            tp, fp, positives = _threshold_count_vectors(scores, labels, thresholds)
            fn = positives - tp
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
            recall = tp / positives
            forged_f1.append(
                np.divide(
                    2 * precision * recall,
                    precision + recall,
                    out=np.zeros_like(precision),
                    where=(precision + recall) > 0,
                )
            )
        else:
            histogram, _ = np.histogram(scores, bins=np.r_[thresholds, np.inf])
            predicted = np.cumsum(histogram[::-1], dtype=np.int64)[::-1]
            authentic_fpr.append(predicted / scores.size)
    macro_f1 = np.mean(np.stack(forged_f1), axis=0)
    macro_authentic_fpr = np.mean(np.stack(authentic_fpr), axis=0)
    cap = float(config["authentic_document_macro_pixel_fpr_max"])
    feasible = np.flatnonzero(macro_authentic_fpr <= cap + 1e-12)
    fallback_used = False
    if feasible.size == 0:
        fallback_used = True
        feasible = np.flatnonzero(macro_authentic_fpr == macro_authentic_fpr.min())
    best_f1 = macro_f1[feasible].max()
    candidates = feasible[np.isclose(macro_f1[feasible], best_f1, rtol=0, atol=1e-12)]
    best_fpr = macro_authentic_fpr[candidates].min()
    candidates = candidates[
        np.isclose(macro_authentic_fpr[candidates], best_fpr, rtol=0, atol=1e-12)
    ]
    selected = int(candidates[-1])
    return {
        "threshold": float(thresholds[selected]),
        "validation_forged_document_macro_pixel_f1": float(macro_f1[selected]),
        "validation_authentic_document_macro_pixel_fpr": float(
            macro_authentic_fpr[selected]
        ),
        "constraint": cap,
        "fallback_used": fallback_used,
        "candidate_count": int(thresholds.size),
        "selected_using_final_test": False,
    }


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")[::-1]
    ranked_scores = scores[order]
    ranked_labels = labels[order].astype(np.uint8)
    positives = int(ranked_labels.sum())
    negatives = int(ranked_labels.size - positives)
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    tp = np.cumsum(ranked_labels)
    fp = np.cumsum(1 - ranked_labels)
    ends = np.r_[ranked_scores[1:] != ranked_scores[:-1], True]
    return float(np.trapezoid(np.r_[0, tp[ends] / positives], np.r_[0, fp[ends] / negatives]))


def _freeze_image_threshold(
    manifest_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    validation = [row for row in manifest_rows if row["evaluation_role"] == "validation"]
    scores = np.asarray([predictions[row["record_id"]]["image_score"] for row in validation])
    labels = np.asarray([row["sample_kind"] == "forged" for row in validation])
    candidates = np.r_[np.inf, np.unique(scores)]
    tpr = np.asarray([np.mean(scores[labels] >= value) for value in candidates])
    fpr = np.asarray([np.mean(scores[~labels] >= value) for value in candidates])
    cap = float(config["authentic_image_fpr_max"])
    feasible = np.flatnonzero(fpr <= cap + 1e-12)
    best_tpr = tpr[feasible].max()
    selected = feasible[np.isclose(tpr[feasible], best_tpr, rtol=0, atol=1e-12)]
    best_fpr = fpr[selected].min()
    selected = selected[np.isclose(fpr[selected], best_fpr, rtol=0, atol=1e-12)]
    index = int(selected[-1])
    return {
        "threshold": float(candidates[index]),
        "validation_forged_image_tpr": float(tpr[index]),
        "validation_authentic_image_fpr": float(fpr[index]),
        "constraint": cap,
        "selected_using_final_test": False,
    }


def _bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
    confidence: float,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _evaluate_role(
    *,
    baseline_name: str,
    role: str,
    groups: set[str],
    scratch: Path,
    manifest_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    pixel_threshold: float,
    image_threshold: float,
    rng: np.random.Generator,
    bootstrap_resamples: int,
    confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    forged_rows = {
        row["source_group_id"]: row
        for row in manifest_rows
        if row["evaluation_role"] == role
        and row["sample_kind"] == "forged"
        and row["source_group_id"] in groups
    }
    authentic_role = "validation" if role == "validation" else "final_test"
    authentic_rows = {
        row["source_group_id"]: row
        for row in manifest_rows
        if row["evaluation_role"] == authentic_role
        and row["sample_kind"] == "authentic"
        and row["source_group_id"] in groups
    }
    if set(forged_rows) != groups or set(authentic_rows) != groups:
        raise ValueError(f"role {role} does not have one forged/authentic pair per group")

    details: list[dict[str, Any]] = []
    for group in sorted(groups):
        forged = forged_rows[group]
        authentic = authentic_rows[group]
        forged_scores = _load_native_scores(
            scratch,
            predictions[forged["record_id"]],
            (int(forged["height"]), int(forged["width"])),
        )
        mask = _load_mask(scratch, forged)
        average_precision, pixel_auroc = _ranking_metrics(forged_scores, mask)
        operational = _threshold_metrics(forged_scores, mask, pixel_threshold)
        authentic_scores = _load_native_scores(
            scratch,
            predictions[authentic["record_id"]],
            (int(authentic["height"]), int(authentic["width"])),
        )
        authentic_fpr = float(np.mean(authentic_scores >= pixel_threshold))
        details.append(
            {
                "baseline": baseline_name,
                "evaluation_role": role,
                "source_group_id": group,
                "forged_record_id": forged["record_id"],
                "authentic_record_id": authentic["record_id"],
                "macro_pixel_ap": average_precision,
                "pixel_auroc": pixel_auroc,
                **operational,
                "authentic_pixel_fpr": authentic_fpr,
                "forged_image_score": float(predictions[forged["record_id"]]["image_score"]),
                "authentic_image_score": float(
                    predictions[authentic["record_id"]]["image_score"]
                ),
                "paper_evidence": False,
            }
        )

    aggregate: dict[str, Any] = {
        "baseline": baseline_name,
        "evaluation_role": role,
        "groups": len(details),
        "pixel_threshold": pixel_threshold,
        "image_threshold": image_threshold,
        "paper_evidence": False,
    }
    metric_names = (
        "macro_pixel_ap",
        "pixel_auroc",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pixel_iou",
        "authentic_pixel_fpr",
    )
    for metric in metric_names:
        values = np.asarray([row[metric] for row in details], dtype=np.float64)
        low, high = _bootstrap_interval(values, rng, bootstrap_resamples, confidence)
        aggregate[metric] = float(values.mean())
        aggregate[f"{metric}_ci_low"] = low
        aggregate[f"{metric}_ci_high"] = high
    forged_image_scores = np.asarray([row["forged_image_score"] for row in details])
    authentic_image_scores = np.asarray([row["authentic_image_score"] for row in details])
    aggregate["image_auroc"] = _roc_auc(
        np.r_[forged_image_scores, authentic_image_scores],
        np.r_[np.ones(len(details), dtype=bool), np.zeros(len(details), dtype=bool)],
    )
    aggregate["image_tpr_at_validation_frozen_fpr"] = float(
        np.mean(forged_image_scores >= image_threshold)
    )
    aggregate["image_fpr_at_validation_frozen_fpr"] = float(
        np.mean(authentic_image_scores >= image_threshold)
    )
    return aggregate, details


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["runtime"]["method_training_authorized"]:
        raise ValueError("baseline evaluation must not authorize method training")
    if config["runtime"]["final_test_model_selection_allowed"]:
        raise ValueError("final-test model selection is prohibited")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("stage-zero baseline metrics cannot be paper evidence")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            config["runtime"]["scratch_env"],
            str(_resolve(project_root, config["runtime"]["scratch_default"])),
        )
    ).resolve()
    manifest_path = _resolve(project_root, config["input"]["manifest"])
    if _sha256(manifest_path) != config["input"]["expected_manifest_sha256"]:
        raise ValueError("baseline manifest SHA-256 changed")
    manifest_rows = _read_jsonl(manifest_path)
    if len(manifest_rows) != int(config["input"]["expected_records"]):
        raise ValueError("baseline manifest record count changed")
    output_group_metrics = _resolve(project_root, paths["output_group_metrics"])
    output_metrics = _resolve(project_root, paths["output_metrics"])
    output_thresholds = _resolve(project_root, paths["output_thresholds"])
    output_summary = _resolve(project_root, paths["output_summary"])
    log_path = _resolve(project_root, paths["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    official_groups = {
        str(row["source_group_id"])
        for row in manifest_rows
        if row["evaluation_role"] == "in_domain_test"
        and ":testing:" in str(row["source_sample_id"])
    }
    if config["evaluation"]["report_safe_official_59_subset"] and len(official_groups) != 59:
        raise ValueError(f"expected 59 safe official groups, found {len(official_groups)}")

    started = time.monotonic()
    threshold_records: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    baseline_summaries: dict[str, Any] = {}
    for baseline_index, baseline_config in enumerate(config["baselines"].values()):
        prediction_path = _resolve(project_root, baseline_config["predictions"])
        summary_path = _resolve(project_root, baseline_config["run_summary"])
        with summary_path.open("r", encoding="utf-8") as handle:
            run_summary = json.load(handle)
        if (
            run_summary.get("status") != "passed"
            or run_summary.get("successful_records") != len(manifest_rows)
            or run_summary.get("failed_records") != 0
            or run_summary.get("output_predictions_sha256") != _sha256(prediction_path)
        ):
            raise RuntimeError(f"baseline run is incomplete: {summary_path}")
        prediction_rows = _read_jsonl(prediction_path)
        predictions = {row["record_id"]: row for row in prediction_rows}
        if len(predictions) != len(manifest_rows) or any(
            row.get("status") != "ok" for row in prediction_rows
        ):
            raise RuntimeError(f"baseline predictions are incomplete: {prediction_path}")

        pixel_operating_point = _freeze_pixel_threshold(
            scratch,
            manifest_rows,
            predictions,
            config["threshold_selection"],
        )
        image_operating_point = _freeze_image_threshold(
            manifest_rows,
            predictions,
            config["image_operating_point"],
        )
        baseline_name = str(baseline_config["name"])
        threshold_records[baseline_name] = {
            "pixel": pixel_operating_point,
            "image": image_operating_point,
        }
        rng = np.random.default_rng(int(config["experiment"]["seed"]) + baseline_index)
        for role in config["evaluation"]["roles"]:
            all_groups = {
                str(row["source_group_id"])
                for row in manifest_rows
                if row["evaluation_role"] == role and row["sample_kind"] == "forged"
            }
            aggregate, details = _evaluate_role(
                baseline_name=baseline_name,
                role=role,
                groups=all_groups,
                scratch=scratch,
                manifest_rows=manifest_rows,
                predictions=predictions,
                pixel_threshold=float(pixel_operating_point["threshold"]),
                image_threshold=float(image_operating_point["threshold"]),
                rng=rng,
                bootstrap_resamples=int(config["evaluation"]["bootstrap_resamples"]),
                confidence=float(config["evaluation"]["confidence_level"]),
            )
            table_rows.append(aggregate)
            group_rows.extend(details)
            if role in {"in_domain_test", "generator_holdout"} and config["evaluation"][
                "report_safe_official_59_subset"
            ]:
                subset_aggregate, subset_details = _evaluate_role(
                    baseline_name=baseline_name,
                    role=role,
                    groups=official_groups,
                    scratch=scratch,
                    manifest_rows=manifest_rows,
                    predictions=predictions,
                    pixel_threshold=float(pixel_operating_point["threshold"]),
                    image_threshold=float(image_operating_point["threshold"]),
                    rng=rng,
                    bootstrap_resamples=int(config["evaluation"]["bootstrap_resamples"]),
                    confidence=float(config["evaluation"]["confidence_level"]),
                )
                subset_aggregate["evaluation_role"] += "_safe_official_59"
                for detail in subset_details:
                    detail["evaluation_role"] += "_safe_official_59"
                table_rows.append(subset_aggregate)
                group_rows.extend(subset_details)
        baseline_summaries[baseline_name] = {
            "run_summary": str(summary_path.relative_to(project_root)),
            "run_summary_sha256": _sha256(summary_path),
            "predictions": str(prediction_path.relative_to(project_root)),
            "predictions_sha256": _sha256(prediction_path),
        }

    _write_jsonl(output_group_metrics, group_rows)
    _write_csv(output_metrics, table_rows)
    _write_json(
        output_thresholds,
        {
            "experiment": config["experiment"],
            "selection_policy": config["threshold_selection"],
            "image_operating_point_policy": config["image_operating_point"],
            "final_test_used_for_selection": False,
            "baselines": threshold_records,
        },
    )
    summary = {
        "experiment": config["experiment"],
        "status": "passed",
        "paper_evidence": False,
        "method_training_authorized": False,
        "final_test_used_for_model_or_threshold_selection": False,
        "input_manifest_sha256": _sha256(manifest_path),
        "safe_official_subset_groups": len(official_groups),
        "baseline_runs": baseline_summaries,
        "metric_rows": len(table_rows),
        "group_metric_rows": len(group_rows),
        "wall_time_seconds": time.monotonic() - started,
        "outputs": {
            "group_metrics": str(output_group_metrics.relative_to(project_root)),
            "group_metrics_sha256": _sha256(output_group_metrics),
            "metrics": str(output_metrics.relative_to(project_root)),
            "metrics_sha256": _sha256(output_metrics),
            "thresholds": str(output_thresholds.relative_to(project_root)),
            "thresholds_sha256": _sha256(output_thresholds),
            "log": str(log_path.relative_to(project_root)),
        },
    }
    _write_json(output_summary, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
