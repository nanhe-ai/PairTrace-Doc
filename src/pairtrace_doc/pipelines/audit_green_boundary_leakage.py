from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from pairtrace_doc.pipelines.train_student_100 import _sha256, _write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stratum_type",
        "stratum_value",
        "total",
        "ok",
        "failures",
        "artifact_positive",
        "artifact_positive_rate",
        "target_green_fraction_mean",
        "target_green_fraction_median",
        "green_excess_ap_mean",
        "green_excess_ap_median",
        "model_ap_positive_mean",
        "model_ap_negative_mean",
        "model_ap_difference",
        "model_ap_difference_ci95_low",
        "model_ap_difference_ci95_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _scratch_root(project_root: Path, paths: dict[str, Any]) -> Path:
    environment_name = str(paths["scratch_env"])
    override = os.environ.get(environment_name)
    if override:
        return Path(override).expanduser().resolve()
    return _resolve(project_root, str(paths["scratch_default"])).resolve()


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    digest = _sha256(path)
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 changed: {digest} != {expected_sha256}")


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("mask has no positive pixels")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _edge_stats(
    green: np.ndarray, bbox: tuple[int, int, int, int], band_px: int
) -> tuple[int, int, float]:
    height, width = green.shape
    x1, y1, x2, y2 = bbox
    left = max(0, x1 - band_px)
    top = max(0, y1 - band_px)
    right = min(width, x2 + band_px)
    bottom = min(height, y2 + band_px)
    local = green[top:bottom, left:right]
    edge = np.zeros(local.shape, dtype=bool)
    lx1, ly1, lx2, ly2 = x1 - left, y1 - top, x2 - left, y2 - top
    edge[max(0, ly1 - band_px) : min(local.shape[0], ly1 + band_px + 1), lx1:lx2] = True
    edge[max(0, ly2 - 1 - band_px) : min(local.shape[0], ly2 + band_px), lx1:lx2] = True
    edge[ly1:ly2, max(0, lx1 - band_px) : min(local.shape[1], lx1 + band_px + 1)] = True
    edge[ly1:ly2, max(0, lx2 - 1 - band_px) : min(local.shape[1], lx2 + band_px)] = True
    pixels = int(edge.sum())
    hits = int((local & edge).sum())
    return hits, pixels, hits / max(1, pixels)


def _boxes_overlap(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


def _control_boxes(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    gap_px: int,
) -> list[tuple[int, int, int, int]]:
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    box_width, box_height = x2 - x1, y2 - y1
    dx, dy = box_width + gap_px, box_height + gap_px
    candidates = [
        (x1 + dx, y1, x2 + dx, y2),
        (x1 - dx, y1, x2 - dx, y2),
        (x1, y1 + dy, x2, y2 + dy),
        (x1, y1 - dy, x2, y2 - dy),
        (gap_px, gap_px, gap_px + box_width, gap_px + box_height),
        (
            width - gap_px - box_width,
            height - gap_px - box_height,
            width - gap_px,
            height - gap_px,
        ),
    ]
    controls: list[tuple[int, int, int, int]] = []
    for candidate in candidates:
        cx1, cy1, cx2, cy2 = candidate
        if cx1 < 0 or cy1 < 0 or cx2 > width or cy2 > height:
            continue
        if _boxes_overlap(candidate, bbox) or candidate in controls:
            continue
        controls.append(candidate)
    return controls


def _binned_average_precision(score: np.ndarray, mask: np.ndarray) -> float:
    positives = int(mask.sum())
    if positives == 0:
        raise ValueError("average precision requires a non-empty mask")
    positive_counts = np.bincount(score[mask].ravel(), minlength=256)
    negative_counts = np.bincount(score[~mask].ravel(), minlength=256)
    true_positive = 0
    false_positive = 0
    ap = 0.0
    for value in range(255, -1, -1):
        new_positive = int(positive_counts[value])
        true_positive += new_positive
        false_positive += int(negative_counts[value])
        if new_positive:
            precision = true_positive / max(1, true_positive + false_positive)
            ap += (new_positive / positives) * precision
    return float(ap)


def _prediction_map(
    path: Path, condition: str
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for row in _read_jsonl(path):
        if row.get("condition") != condition or row.get("status") != "ok":
            continue
        if "macro_pixel_ap" not in row:
            continue
        key = (str(row.get("evaluation_role")), str(row.get("source_group_id")))
        if key in values:
            raise ValueError(f"duplicate frozen prediction for {key}")
        values[key] = float(row["macro_pixel_ap"])
    return values


def _select_rows(
    rows: list[dict[str, Any]], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    forged = [row for row in rows if row.get("sample_kind") == "forged"]
    mode = str(selection["mode"])
    if mode == "all_forged":
        return forged
    if mode != "explicit_source_sample_ids":
        raise ValueError(f"unsupported selection mode: {mode}")
    requested = [str(value) for value in selection["source_sample_ids"]]
    by_id = {str(row.get("source_sample_id")): row for row in forged}
    missing = [value for value in requested if value not in by_id]
    if missing:
        raise ValueError(f"explicit audit sample IDs missing: {missing}")
    return [by_id[value] for value in requested]


def _bootstrap_difference(
    positive: list[float],
    negative: list[float],
    *,
    seed: int,
    replicates: int,
) -> tuple[float | None, float | None]:
    if not positive or not negative or replicates <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        differences[index] = float(
            rng.choice(pos, len(pos), replace=True).mean()
            - rng.choice(neg, len(neg), replace=True).mean()
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def _summary_row(
    stratum_type: str,
    stratum_value: str,
    records: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    ok = [row for row in records if row["status"] == "ok"]
    positive = [row for row in ok if row["artifact_positive"]]
    negative = [row for row in ok if not row["artifact_positive"]]
    fractions = [float(row["target_green_fraction"]) for row in ok]
    artifact_aps = [float(row["green_excess_pixel_ap"]) for row in ok]
    positive_model = [
        float(row["frozen_model_macro_pixel_ap"])
        for row in positive
        if row.get("frozen_model_macro_pixel_ap") is not None
    ]
    negative_model = [
        float(row["frozen_model_macro_pixel_ap"])
        for row in negative
        if row.get("frozen_model_macro_pixel_ap") is not None
    ]
    low, high = _bootstrap_difference(
        positive_model,
        negative_model,
        seed=seed,
        replicates=bootstrap_replicates,
    )
    return {
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "total": len(records),
        "ok": len(ok),
        "failures": len(records) - len(ok),
        "artifact_positive": len(positive),
        "artifact_positive_rate": len(positive) / max(1, len(ok)),
        "target_green_fraction_mean": mean(fractions) if fractions else None,
        "target_green_fraction_median": median(fractions) if fractions else None,
        "green_excess_ap_mean": mean(artifact_aps) if artifact_aps else None,
        "green_excess_ap_median": median(artifact_aps) if artifact_aps else None,
        "model_ap_positive_mean": mean(positive_model) if positive_model else None,
        "model_ap_negative_mean": mean(negative_model) if negative_model else None,
        "model_ap_difference": (
            mean(positive_model) - mean(negative_model)
            if positive_model and negative_model
            else None
        ),
        "model_ap_difference_ci95_low": low,
        "model_ap_difference_ci95_high": high,
    }


def _markdown_report(summary: dict[str, Any], csv_rows: list[dict[str, Any]]) -> str:
    overall = csv_rows[0]
    lines = [
        "# Green annotation-boundary shortcut audit",
        "",
        "Status: CPU diagnostic completed; post-hoc and not confirmatory paper evidence.",
        "",
        "The audit uses exact masks only to test whether annotation-like green pixels",
        "are enriched on the manipulated-region bounding-box edge. This is a dataset",
        "integrity diagnostic, not an inference-time detector.",
        "",
        "## Frozen-reserve decision",
        "",
        f"- Audited forged records: {overall['ok']}/{overall['total']}.",
        f"- High-confidence boundary-signal records: {overall['artifact_positive']} "
        f"({overall['artifact_positive_rate']:.2%}).",
        f"- Cleanliness gate passed: `{str(summary['decision']['final_reserve_cleanliness_gate_passed']).lower()}`.",
        "- Existing reserve remains consumed and must not be used to select a cleanup",
        "  rule, threshold, model, or new operating point.",
        "",
        "## Stratified counts",
        "",
        "| Stratum | OK | Signal | Rate | Artifact-only AP mean | Frozen-model AP difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in csv_rows[1:]:
        difference = row["model_ap_difference"]
        difference_text = "NA" if difference is None else f"{difference:.4f}"
        artifact_ap = row["green_excess_ap_mean"]
        artifact_text = "NA" if artifact_ap is None else f"{artifact_ap:.4f}"
        lines.append(
            f"| `{row['stratum_type']}={row['stratum_value']}` | {row['ok']} | "
            f"{row['artifact_positive']} | {row['artifact_positive_rate']:.2%} | "
            f"{artifact_text} | {difference_text} |"
        )
    lines.extend(
        [
            "",
            "## Required follow-up",
            "",
            "1. Run a same-image, frozen-model sensitivity experiment with a mask-blind",
            "   green-suppression transform and an identically transformed negative control.",
            "2. Regenerate or replace annotation-contaminated candidates before creating",
            "   any new confirmatory split; do not edit `data/raw/` in place.",
            "3. Freeze a new source-group-disjoint final set before model or threshold",
            "   selection and report results by generator.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("green-boundary audit config must be a mapping")
    experiment = config["experiment"]
    if bool(experiment["paper_evidence"]):
        raise ValueError("post-hoc shortcut audit cannot be confirmatory paper evidence")
    runtime = config["runtime"]
    if runtime["device"] != "cpu":
        raise ValueError("green-boundary audit must be CPU-only")
    if bool(runtime["model_inference_authorized"]) or bool(
        runtime["model_training_authorized"]
    ):
        raise ValueError("green-boundary audit cannot run or train a model")
    if bool(runtime["threshold_selection_authorized"]) or bool(
        runtime["sample_replacement_authorized"]
    ):
        raise ValueError("green-boundary audit crossed a frozen evidence boundary")

    paths = config["paths"]
    protocol_path = _resolve(project_root, str(paths["protocol"]))
    manifest_path = _resolve(project_root, str(paths["input_manifest"]))
    predictions_path = _resolve(project_root, str(paths["frozen_predictions"]))
    _verify_file(
        protocol_path,
        str(paths["expected_protocol_sha256"]),
        "green-boundary audit protocol",
    )
    _verify_file(
        manifest_path,
        str(paths["expected_input_manifest_sha256"]),
        "frozen reserve manifest",
    )
    _verify_file(
        predictions_path,
        str(paths["expected_frozen_predictions_sha256"]),
        "frozen predictions",
    )
    scratch_root = _scratch_root(project_root, paths)
    rows = _select_rows(_read_jsonl(manifest_path), config["selection"])
    prediction_values = _prediction_map(
        predictions_path, str(config["prediction_join"]["condition"])
    )
    detector = config["detector"]
    band_px = int(detector["edge_band_px"])
    control_gap_px = int(detector["control_gap_px"])
    min_green = int(detector["min_green_channel"])
    min_red_delta = int(detector["min_green_minus_red"])
    min_blue_delta = int(detector["min_green_minus_blue"])
    min_hits = int(detector["min_edge_green_pixels"])
    min_fraction = float(detector["min_edge_green_fraction"])
    control_ratio = float(detector["min_target_control_ratio"])
    control_margin = float(detector["min_target_control_margin"])
    verify_hashes = bool(runtime["verify_input_hashes"])

    records: list[dict[str, Any]] = []
    log_lines: list[str] = []
    for index, source in enumerate(rows, 1):
        base = {
            "record_id": source.get("record_id"),
            "source_sample_id": source.get("source_sample_id"),
            "source_group_id": source.get("source_group_id"),
            "evaluation_role": source.get("evaluation_role"),
            "generator": source.get("generator"),
            "source_dataset": source.get("source_dataset"),
            "image": source.get("image"),
            "mask": source.get("mask"),
            "paper_evidence": False,
            "final_reserve_read": True,
            "selection_or_threshold_change_authorized": False,
        }
        try:
            image_path = _resolve(scratch_root, str(source["image"]))
            mask_path = _resolve(scratch_root, str(source["mask"]))
            if verify_hashes:
                _verify_file(image_path, str(source["image_sha256"]), "candidate image")
                _verify_file(mask_path, str(source["mask_sha256"]), "exact mask")
            image = np.asarray(Image.open(image_path).convert("RGB"))
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            if image.shape[:2] != mask.shape:
                raise ValueError(
                    f"candidate/mask shape mismatch: {image.shape[:2]} != {mask.shape}"
                )
            bbox = _mask_bbox(mask)
            red = image[..., 0].astype(np.int16)
            green_channel = image[..., 1].astype(np.int16)
            blue = image[..., 2].astype(np.int16)
            strict_green = (
                (green_channel >= min_green)
                & (green_channel - red >= min_red_delta)
                & (green_channel - blue >= min_blue_delta)
            )
            target_hits, target_pixels, target_fraction = _edge_stats(
                strict_green, bbox, band_px
            )
            controls = _control_boxes(bbox, mask.shape, control_gap_px)
            control_values = [
                _edge_stats(strict_green, control, band_px)[2]
                for control in controls
            ]
            max_control = max(control_values, default=0.0)
            artifact_positive = bool(
                target_hits >= min_hits
                and target_fraction >= min_fraction
                and target_fraction >= max_control * control_ratio
                and target_fraction >= max_control + control_margin
            )
            green_excess = np.clip(
                green_channel - np.maximum(red, blue), 0, 255
            ).astype(np.uint8)
            artifact_ap = _binned_average_precision(green_excess, mask)
            prediction_key = (
                str(source.get("evaluation_role")),
                str(source.get("source_group_id")),
            )
            record = {
                **base,
                "status": "ok",
                "error": None,
                "mask_bbox_xyxy_exclusive": list(bbox),
                "edge_band_px": band_px,
                "target_green_hits": target_hits,
                "target_edge_pixels": target_pixels,
                "target_green_fraction": target_fraction,
                "control_box_count": len(controls),
                "max_control_green_fraction": max_control,
                "target_control_enrichment": target_fraction / max(max_control, 1e-12),
                "artifact_positive": artifact_positive,
                "green_excess_pixel_ap": artifact_ap,
                "frozen_model_macro_pixel_ap": prediction_values.get(prediction_key),
            }
        except Exception as exc:
            record = {
                **base,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "artifact_positive": None,
            }
            log_lines.append(f"record={base['record_id']} error={record['error']}")
        records.append(record)
        if index % int(runtime["progress_every"]) == 0 or index == len(rows):
            print(f"green_boundary_audit completed={index}/{len(rows)}", flush=True)

    strata: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("overall", "all_forged", records)
    ]
    for field in ("evaluation_role", "generator", "source_dataset"):
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get(field))].append(record)
        strata.extend((field, key, grouped[key]) for key in sorted(grouped))
    bootstrap = config["statistics"]
    csv_rows = [
        _summary_row(
            kind,
            value,
            group,
            seed=int(bootstrap["seed"]),
            bootstrap_replicates=int(bootstrap["bootstrap_replicates"]),
        )
        for kind, value, group in strata
    ]
    overall = csv_rows[0]
    failure_reasons = Counter(
        str(row.get("error")) for row in records if row["status"] != "ok"
    )
    decision = {
        "final_reserve_cleanliness_gate_passed": overall["artifact_positive"] == 0,
        "high_confidence_boundary_signal_detected": overall["artifact_positive"] > 0,
        "existing_final_reserve_remains_consumed": True,
        "existing_final_reserve_allowed_for_cleanup_selection": False,
        "requires_mask_blind_frozen_model_sensitivity": overall[
            "artifact_positive"
        ]
        > 0,
        "requires_clean_candidate_regeneration_before_new_confirmation": overall[
            "artifact_positive"
        ]
        > 0,
    }
    implementation_path = Path(__file__).resolve()
    implementation_label = (
        str(implementation_path.relative_to(project_root))
        if implementation_path.is_relative_to(project_root)
        else f"src/pairtrace_doc/pipelines/{implementation_path.name}"
    )
    summary = {
        "schema_version": 1,
        "experiment": experiment,
        "status": (
            "high_confidence_annotation_boundary_signal_detected"
            if overall["artifact_positive"] > 0
            else "no_high_confidence_annotation_boundary_signal_detected"
        ),
        "paper_evidence": False,
        "post_hoc_dataset_integrity_diagnostic": True,
        "model_inference_performed": False,
        "model_training_performed": False,
        "threshold_or_model_selection_performed": False,
        "sample_replacement_performed": False,
        "input": {
            "config": str(config_path.relative_to(project_root)),
            "config_sha256": _sha256(config_path),
            "protocol": str(protocol_path.relative_to(project_root)),
            "protocol_sha256": _sha256(protocol_path),
            "manifest": str(manifest_path.relative_to(project_root)),
            "manifest_sha256": _sha256(manifest_path),
            "frozen_predictions": str(predictions_path.relative_to(project_root)),
            "frozen_predictions_sha256": _sha256(predictions_path),
            "implementation": implementation_label,
            "implementation_sha256": _sha256(implementation_path),
        },
        "detector": detector,
        "selection": config["selection"],
        "records": {
            "total": len(records),
            "ok": int(overall["ok"]),
            "failures": int(overall["failures"]),
            "failure_reasons": dict(sorted(failure_reasons.items())),
        },
        "decision": decision,
        "strata": csv_rows,
    }
    predictions_output = _resolve(project_root, str(paths["audit_records"]))
    table_output = _resolve(project_root, str(paths["summary_csv"]))
    summary_output = _resolve(project_root, str(paths["summary_json"]))
    report_output = _resolve(project_root, str(paths["report"]))
    log_output = _resolve(project_root, str(paths["log"]))
    _write_jsonl(predictions_output, records)
    _write_csv(table_output, csv_rows)
    _write_json(summary_output, summary)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(_markdown_report(summary, csv_rows), encoding="utf-8")
    log_output.parent.mkdir(parents=True, exist_ok=True)
    log_output.write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
    summary["output"] = {
        "audit_records": str(predictions_output.relative_to(project_root)),
        "audit_records_sha256": _sha256(predictions_output),
        "summary_csv": str(table_output.relative_to(project_root)),
        "summary_csv_sha256": _sha256(table_output),
        "summary_json": str(summary_output.relative_to(project_root)),
        "report": str(report_output.relative_to(project_root)),
        "report_sha256": _sha256(report_output),
        "log": str(log_output.relative_to(project_root)),
        "log_sha256": _sha256(log_output),
    }
    _write_json(summary_output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
