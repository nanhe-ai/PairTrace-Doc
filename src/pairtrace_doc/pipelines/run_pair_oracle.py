from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import platform
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
        raise ValueError("cannot write an empty oracle metrics table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _difference_score(authentic: np.ndarray, forged: np.ndarray) -> np.ndarray:
    if authentic.shape != forged.shape or authentic.ndim != 3 or authentic.shape[2] != 3:
        raise ValueError("oracle inputs must be aligned HxWx3 RGB arrays")
    difference = np.abs(forged.astype(np.int16) - authentic.astype(np.int16))
    return difference.max(axis=2).astype(np.uint8)


def _jpeg_roundtrip(image: np.ndarray, condition: dict[str, Any]) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image).save(
        buffer,
        format="JPEG",
        quality=int(condition["quality"]),
        subsampling=int(condition["subsampling"]),
        optimize=bool(condition["optimize"]),
        progressive=bool(condition["progressive"]),
    )
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return np.asarray(handle.convert("RGB"))


def _apply_condition(
    authentic: np.ndarray, forged: np.ndarray, condition: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    transform = str(condition["transform"])
    if transform == "none":
        return authentic, forged
    if transform == "matched_jpeg":
        return _jpeg_roundtrip(authentic, condition), _jpeg_roundtrip(forged, condition)
    raise ValueError(f"unsupported oracle transform {transform!r}")


def _ranking_metrics_uint8(scores: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if scores.dtype != np.uint8 or scores.shape != mask.shape:
        raise ValueError("uint8 scores and mask must have identical shapes")
    labels = mask.astype(bool, copy=False)
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)
    if not positives or not negatives:
        raise ValueError("oracle ranking metrics require both pixel classes")
    positive_histogram = np.bincount(scores[labels], minlength=256)
    negative_histogram = np.bincount(scores[~labels], minlength=256)
    tp = np.cumsum(positive_histogram[::-1], dtype=np.int64)
    fp = np.cumsum(negative_histogram[::-1], dtype=np.int64)
    recall = tp / positives
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    auroc = float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fp / negatives]))
    return average_precision, auroc


def _cache_key(row: dict[str, Any], condition: dict[str, Any], oracle: dict[str, Any]) -> str:
    payload = {
        "cache_schema_version": oracle["cache_schema_version"],
        "authentic_sha256": row["authentic_sha256"],
        "forged_sha256": row["image_sha256"],
        "condition": condition,
        "score": oracle["score"],
        "mask_used_for_score_construction": oracle["mask_used_for_score_construction"],
        "pillow_version": Image.__version__,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if runtime["device"] != "cpu" or runtime["gpu_launch_authorized"]:
        raise ValueError("pair-difference oracle is frozen to CPU")
    if runtime["method_training_authorized"]:
        raise ValueError("pair-difference oracle must not authorize method training")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("validation-only oracle cannot be paper evidence")

    amendment_path = _resolve(project_root, config["experiment"]["protocol_amendment"])
    if not amendment_path.is_file():
        raise FileNotFoundError(amendment_path)
    input_path = _resolve(project_root, config["input"]["manifest"])
    input_sha256 = _sha256(input_path)
    if input_sha256 != config["input"]["expected_manifest_sha256"]:
        raise ValueError("pair-oracle input manifest SHA-256 changed")
    all_rows = _read_jsonl(input_path)
    role = str(config["input"]["role"])
    if role in set(config["input"]["forbidden_roles"]):
        raise ValueError("pair oracle selected a forbidden final-test role")
    selected = sorted(
        (row for row in all_rows if row.get("pilot_role") == role),
        key=lambda row: str(row["source_group_id"]),
    )
    expected_pairs = int(config["input"]["expected_pairs"])
    if len(selected) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} validation pairs, found {len(selected)}")
    max_pairs = config["input"].get("max_pairs")
    if max_pairs is not None:
        selected = selected[: int(max_pairs)]
    if len({row["source_group_id"] for row in selected}) != len(selected):
        raise ValueError("pair-oracle input has duplicate source groups")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    cache_dir = _resolve(scratch, paths["cache_dir"])
    prediction_path = _resolve(project_root, paths["output_predictions"])
    metrics_path = _resolve(project_root, paths["output_metrics"])
    summary_path = _resolve(project_root, paths["output_summary"])
    log_path = _resolve(project_root, paths["log"])
    for path in (cache_dir, prediction_path.parent, metrics_path.parent, summary_path.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    started = time.monotonic()
    records: list[dict[str, Any]] = []
    cache_hits = 0
    for pair_index, row in enumerate(selected, start=1):
        authentic_path = _resolve(scratch, row["authentic"])
        forged_path = _resolve(scratch, row["image"])
        mask_path = _resolve(scratch, row["mask"])
        base_record = {
            "source_group_id": row["source_group_id"],
            "sample_id": row["sample_id"],
            "evaluation_role": role,
            "paper_evidence": False,
        }
        try:
            if _sha256(authentic_path) != row["authentic_sha256"]:
                raise ValueError("authentic image SHA-256 changed")
            if _sha256(forged_path) != row["image_sha256"]:
                raise ValueError("forged image SHA-256 changed")
            if _sha256(mask_path) != row["mask_sha256"]:
                raise ValueError("mask SHA-256 changed")
            with Image.open(authentic_path) as handle:
                authentic = np.asarray(handle.convert("RGB"))
            with Image.open(forged_path) as handle:
                forged = np.asarray(handle.convert("RGB"))
            with Image.open(mask_path) as handle:
                mask = np.asarray(handle.convert("L")) > 0
            if authentic.shape[:2] != mask.shape or forged.shape[:2] != mask.shape:
                raise ValueError("aligned pair or mask shape changed")
        except Exception as error:
            logging.exception("pair input failed source_group_id=%s", row["source_group_id"])
            for condition in config["oracle"]["conditions"]:
                records.append(
                    {
                        **base_record,
                        "condition": condition["name"],
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            continue

        for condition in config["oracle"]["conditions"]:
            cache_key = _cache_key(row, condition, config["oracle"])
            cache_path = cache_dir / f"{cache_key}.npz"
            record = {
                **base_record,
                "condition": condition["name"],
                "cache_key": cache_key,
                "score_cache": str(cache_path.relative_to(scratch)),
            }
            condition_started = time.perf_counter()
            try:
                cache_hit = cache_path.is_file()
                if cache_hit:
                    with np.load(cache_path, allow_pickle=False) as cached:
                        scores = np.asarray(cached["scores"], dtype=np.uint8)
                    cache_hits += 1
                else:
                    transformed_authentic, transformed_forged = _apply_condition(
                        authentic, forged, condition
                    )
                    scores = _difference_score(transformed_authentic, transformed_forged)
                    temporary = cache_path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, scores=scores)
                    temporary.replace(cache_path)
                if scores.shape != mask.shape:
                    raise ValueError("oracle score shape differs from mask")
                average_precision, auroc = _ranking_metrics_uint8(scores, mask)
                nonzero = scores > 0
                inside_coverage = float(np.mean(nonzero[mask]))
                outside_fraction = float(np.mean(nonzero[~mask]))
                record.update(
                    {
                        "status": "ok",
                        "cache_hit": cache_hit,
                        "shape": list(scores.shape),
                        "score_min_uint8": int(scores.min()),
                        "score_max_uint8": int(scores.max()),
                        "macro_pixel_ap": average_precision,
                        "pixel_auroc": auroc,
                        "inside_nonzero_coverage": inside_coverage,
                        "outside_nonzero_fraction": outside_fraction,
                        "elapsed_seconds": time.perf_counter() - condition_started,
                    }
                )
            except Exception as error:
                record.update(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                logging.exception(
                    "oracle failed source_group_id=%s condition=%s",
                    row["source_group_id"],
                    condition["name"],
                )
            records.append(record)
        if pair_index % 10 == 0 or pair_index == len(selected):
            logging.info("progress completed_pairs=%d total_pairs=%d", pair_index, len(selected))

    _write_jsonl(prediction_path, records)
    expected_records = len(selected) * len(config["oracle"]["conditions"])
    failures = sum(record["status"] != "ok" for record in records)
    complete = len(records) == expected_records and failures == 0
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    metric_rows: list[dict[str, Any]] = []
    for condition in config["oracle"]["conditions"]:
        condition_records = [
            record
            for record in records
            if record["condition"] == condition["name"] and record["status"] == "ok"
        ]
        metric_row: dict[str, Any] = {
            "condition": condition["name"],
            "pairs": len(condition_records),
            "paper_evidence": False,
        }
        for metric in (
            "macro_pixel_ap",
            "pixel_auroc",
            "inside_nonzero_coverage",
            "outside_nonzero_fraction",
        ):
            values = np.asarray([record[metric] for record in condition_records], dtype=float)
            if values.size:
                low, high = _bootstrap_interval(
                    values,
                    rng,
                    int(config["uncertainty"]["bootstrap_resamples"]),
                    float(config["uncertainty"]["confidence_level"]),
                )
                metric_row[metric] = float(values.mean())
                metric_row[f"{metric}_ci_low"] = low
                metric_row[f"{metric}_ci_high"] = high
            else:
                metric_row[metric] = None
                metric_row[f"{metric}_ci_low"] = None
                metric_row[f"{metric}_ci_high"] = None
        metric_rows.append(metric_row)
    _write_csv(metrics_path, metric_rows)

    metrics_by_condition = {row["condition"]: row for row in metric_rows}
    clean_ap = metrics_by_condition["clean"]["macro_pixel_ap"]
    degraded_ap = metrics_by_condition["matched_jpeg_q85"]["macro_pixel_ap"]
    decision = config["decision"]
    baseline_ap = float(decision["frozen_best_released_validation_macro_pixel_ap"])
    advantage = float(decision["oracle_advantage_min"])
    clean_advantage = clean_ap - baseline_ap if clean_ap is not None else None
    degraded_advantage = degraded_ap - baseline_ap if degraded_ap is not None else None
    degradation_drop = clean_ap - degraded_ap if clean_ap is not None and degraded_ap is not None else None
    clean_pass = bool(clean_advantage is not None and clean_advantage >= advantage)
    degraded_advantage_pass = bool(
        degraded_advantage is not None and degraded_advantage >= advantage
    )
    degradation_retention_pass = bool(
        degradation_drop is not None
        and degradation_drop <= float(decision["matched_degradation_clean_drop_max"])
    )
    criteria_pass = bool(
        complete
        and clean_pass
        and degradation_retention_pass
        and (
            degraded_advantage_pass
            or not bool(decision["require_advantage_under_degradation"])
        )
    )
    status = "passed" if complete and criteria_pass else (
        "completed_success_criteria_not_met" if complete else "failed_incomplete"
    )
    summary = {
        "experiment": config["experiment"],
        "status": status,
        "paper_evidence": False,
        "gpu_used": False,
        "method_training_authorized": False,
        "input_manifest_sha256": input_sha256,
        "protocol_amendment": str(amendment_path.relative_to(project_root)),
        "protocol_amendment_sha256": _sha256(amendment_path),
        "selected_role": role,
        "forbidden_final_roles_read": False,
        "selected_pairs": len(selected),
        "expected_item_records": expected_records,
        "successful_item_records": len(records) - failures,
        "failed_item_records": failures,
        "cache_hits": cache_hits,
        "metrics": metrics_by_condition,
        "decision": {
            "frozen_best_released_validation_baseline": decision[
                "frozen_best_released_validation_baseline"
            ],
            "frozen_best_released_validation_macro_pixel_ap": baseline_ap,
            "oracle_advantage_min": advantage,
            "clean_advantage": clean_advantage,
            "clean_advantage_pass": clean_pass,
            "degraded_advantage": degraded_advantage,
            "degraded_advantage_pass": degraded_advantage_pass,
            "matched_degradation_clean_drop": degradation_drop,
            "matched_degradation_clean_drop_max": float(
                decision["matched_degradation_clean_drop_max"]
            ),
            "degradation_retention_pass": degradation_retention_pass,
            "success_criteria_enforced": bool(runtime["enforce_success_criteria"]),
            "success_criteria_pass": criteria_pass,
        },
        "wall_time_seconds": time.monotonic() - started,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "device": "cpu",
        },
        "outputs": {
            "predictions": str(prediction_path.relative_to(project_root)),
            "predictions_sha256": _sha256(prediction_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "cache_dir": str(cache_dir.relative_to(scratch)),
            "log": str(log_path.relative_to(project_root)),
        },
    }
    _write_json(summary_path, summary)
    if not complete and runtime["require_all_pairs"]:
        raise RuntimeError(f"pair oracle incomplete; see {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
