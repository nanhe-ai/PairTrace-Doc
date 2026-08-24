from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pairtrace_doc.pipelines.evaluate_resampling_robust_final_reserve import (
    _aggregate_fixed_condition,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
)


_REPLAY_FIELDS = (
    "forged_documents",
    "authentic_documents",
    "pixel_threshold",
    "image_threshold",
    "role_macro_generator_macro_pixel_ap",
    "document_macro_pixel_ap",
    "document_macro_pixel_auroc",
    "document_macro_pixel_precision",
    "document_macro_pixel_recall",
    "document_macro_pixel_f1",
    "document_macro_pixel_iou",
    "authentic_document_macro_pixel_fpr",
    "image_auroc",
    "image_tpr_at_development_frozen_threshold",
    "image_fpr_at_development_frozen_threshold",
)


def _affected_groups_from_neighbors(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row["final_source_group_id"])
        for row in rows
        if int(row["high_priority_match_count"]) > 0
    }


def _aggregate_prediction_records(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") != "ok":
            raise ValueError(f"failed prediction record retained: {record.get('record_id')}")
        grouped[str(record["condition"])].append(record)
    metrics: dict[str, dict[str, Any]] = {}
    for condition, rows in sorted(grouped.items()):
        pixel_thresholds = {float(row["fixed_pixel_threshold"]) for row in rows}
        image_thresholds = {float(row["fixed_image_threshold"]) for row in rows}
        if len(pixel_thresholds) != 1 or len(image_thresholds) != 1:
            raise ValueError(f"condition thresholds disagree: {condition}")
        forged = [
            {
                "source_group_id": str(row["source_group_id"]),
                "source_stratum": str(row["source_stratum"]),
                "evaluation_role": str(row["evaluation_role"]),
                "generator": str(row["generator"]),
                "macro_pixel_ap": float(row["macro_pixel_ap"]),
                "pixel_auroc": float(row["pixel_auroc"]),
                "pixel_precision": float(row["pixel_precision"]),
                "pixel_recall": float(row["pixel_recall"]),
                "pixel_f1": float(row["pixel_f1"]),
                "pixel_iou": float(row["pixel_iou"]),
                "image_score": float(row["image_score"]),
            }
            for row in rows
            if row["sample_kind"] == "forged"
        ]
        authentic = [
            {
                "source_group_id": str(row["source_group_id"]),
                "pixel_fpr": float(row["authentic_pixel_fpr"]),
                "image_score": float(row["image_score"]),
            }
            for row in rows
            if row["sample_kind"] == "authentic"
        ]
        metrics[condition] = _aggregate_fixed_condition(
            {"forged": forged, "authentic": authentic},
            next(iter(pixel_thresholds)),
            next(iter(image_thresholds)),
        )
    return metrics


def _read_frozen_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["condition"]: row for row in csv.DictReader(handle)}


def _verify_metric_replay(
    replay: dict[str, dict[str, Any]],
    frozen: dict[str, dict[str, str]],
    tolerance: float,
) -> float:
    if set(replay) != set(frozen):
        raise ValueError("replayed and frozen condition inventories differ")
    maximum_error = 0.0
    for condition, values in replay.items():
        for field in _REPLAY_FIELDS:
            observed = float(values[field])
            expected = float(frozen[condition][field])
            error = abs(observed - expected)
            maximum_error = max(maximum_error, error)
            if not np.isclose(observed, expected, rtol=0.0, atol=tolerance):
                raise ValueError(
                    f"frozen metric replay differs: {condition}.{field} "
                    f"{observed} != {expected}"
                )
    return maximum_error


def _report(summary: dict[str, Any], robust_rows: list[dict[str, Any]]) -> str:
    table = [
        "| Geometry | Original AP | Retained-90 AP | Delta | Original image FPR | Retained-90 image FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in robust_rows:
        table.append(
            "| {geometry} | {original_role_macro_generator_macro_pixel_ap:.4f} | "
            "{retained_role_macro_generator_macro_pixel_ap:.4f} | "
            "{delta_role_macro_generator_macro_pixel_ap:+.4f} | "
            "{original_image_fpr:.4f} | {retained_image_fpr:.4f} |".format(**row)
        )
    return """# Template-contamination sensitivity: final 96

Status: `{status}`.

The authentic-image audit found zero source-ID, encoded-file, or decoded-pixel
hash overlap, but fixed visual review confirmed substantive template/capture
near-duplicates for {affected_groups}/96 final groups ({cord_groups} CORD and
{xfund_groups} XFUND). The original 96-group result remains primary. This
post-final sensitivity replays the frozen item records after excluding those
six preidentified groups; it does not create a new unseen final set.

{table}

All {total_prediction_records} frozen prediction records were present and the
96-group aggregate replayed with maximum absolute error
`{maximum_metric_replay_absolute_error:.3g}`. The retained analysis contains
{retained_groups} groups, {retained_forged_per_condition} forged records, and
{retained_authentic_per_condition} authentic records per condition. No model,
threshold, checkpoint, condition, mask, or operating point was changed.

Even if numerical changes are small, this analysis does not prove that the
other 90 groups are semantically template-independent. The correct conclusion
is that exact identity separation passed, template-level independence failed,
and an independently sourced exact-mask test set remains necessary.
""".format(table="\n".join(table), **summary)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if not runtime["prediction_read_allowed"] or any(
        bool(runtime[key])
        for key in (
            "mask_read_allowed",
            "image_read_allowed",
            "model_training_authorized",
            "checkpoint_selection_authorized",
            "threshold_selection_authorized",
            "primary_result_replacement_authorized",
        )
    ):
        raise ValueError("template sensitivity crossed its frozen read boundary")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("template sensitivity protocol SHA-256 changed")
    inputs = config["inputs"]
    verified_paths: dict[str, Path] = {}
    for name, hash_name in (
        ("template_audit_summary", "expected_template_audit_summary_sha256"),
        ("template_audit_neighbors", "expected_template_audit_neighbors_sha256"),
        ("final_predictions", "expected_final_predictions_sha256"),
        ("final_metrics", "expected_final_metrics_sha256"),
    ):
        path = _resolve(project_root, inputs[name])
        if _sha256(path) != inputs[hash_name]:
            raise ValueError(f"frozen input SHA-256 changed: {name}")
        verified_paths[name] = path

    audit_summary = json.loads(
        verified_paths["template_audit_summary"].read_text(encoding="utf-8")
    )
    if not audit_summary["exact_identity_gate_passed"]:
        raise ValueError("exact identity gate did not pass")
    neighbors = _read_jsonl(verified_paths["template_audit_neighbors"])
    affected = set(map(str, config["affected_source_groups"]))
    if affected != _affected_groups_from_neighbors(neighbors):
        raise ValueError("fixed affected groups differ from perceptual audit flags")

    records = _read_jsonl(verified_paths["final_predictions"])
    expected = config["expected"]
    if len(records) != int(expected["total_prediction_records"]):
        raise ValueError("final prediction topology changed")
    all_groups = {str(row["source_group_id"]) for row in records}
    if len(all_groups) != int(expected["original_groups"]):
        raise ValueError("original final group inventory changed")
    if not affected.issubset(all_groups):
        raise ValueError("affected group missing from final predictions")
    retained_records = [
        row for row in records if str(row["source_group_id"]) not in affected
    ]
    retained_groups = {str(row["source_group_id"]) for row in retained_records}
    if len(retained_groups) != int(expected["retained_groups"]):
        raise ValueError("retained group inventory changed")

    original_metrics = _aggregate_prediction_records(records)
    retained_metrics = _aggregate_prediction_records(retained_records)
    if len(original_metrics) != int(expected["conditions"]):
        raise ValueError("condition inventory changed")
    replay_error = _verify_metric_replay(
        original_metrics,
        _read_frozen_metrics(verified_paths["final_metrics"]),
        float(expected["metric_replay_absolute_tolerance"]),
    )

    condition_rows: list[dict[str, Any]] = []
    for condition in sorted(original_metrics):
        original = original_metrics[condition]
        retained = retained_metrics[condition]
        if (
            original["forged_documents"] != int(expected["original_forged_per_condition"])
            or original["authentic_documents"] != int(expected["original_authentic_per_condition"])
            or retained["forged_documents"] != int(expected["retained_forged_per_condition"])
            or retained["authentic_documents"] != int(expected["retained_authentic_per_condition"])
        ):
            raise ValueError(f"condition topology changed: {condition}")
        row: dict[str, Any] = {
            "condition": condition,
            "original_groups": int(expected["original_groups"]),
            "retained_groups": int(expected["retained_groups"]),
            "affected_groups": len(affected),
            "postfinal_sensitivity": True,
            "primary_result_replaced": False,
        }
        for field in (
            "role_macro_generator_macro_pixel_ap",
            "document_macro_pixel_f1",
            "document_macro_pixel_iou",
            "authentic_document_macro_pixel_fpr",
            "image_fpr_at_development_frozen_threshold",
        ):
            row[f"original_{field}"] = float(original[field])
            row[f"retained_{field}"] = float(retained[field])
            row[f"delta_{field}"] = float(retained[field]) - float(original[field])
        condition_rows.append(row)

    robust_rows: list[dict[str, Any]] = []
    seeds = [str(seed) for seed in expected["robust_seeds"]]
    for geometry in expected["geometries"]:
        names = [f"robust_{seed}_{geometry}_ecc" for seed in seeds]
        original_ap = float(
            np.mean(
                [original_metrics[name]["role_macro_generator_macro_pixel_ap"] for name in names]
            )
        )
        retained_ap = float(
            np.mean(
                [retained_metrics[name]["role_macro_generator_macro_pixel_ap"] for name in names]
            )
        )
        original_image_fpr = float(
            np.mean(
                [original_metrics[name]["image_fpr_at_development_frozen_threshold"] for name in names]
            )
        )
        retained_image_fpr = float(
            np.mean(
                [retained_metrics[name]["image_fpr_at_development_frozen_threshold"] for name in names]
            )
        )
        robust_rows.append(
            {
                "geometry": geometry,
                "training_seeds": len(seeds),
                "original_role_macro_generator_macro_pixel_ap": original_ap,
                "retained_role_macro_generator_macro_pixel_ap": retained_ap,
                "delta_role_macro_generator_macro_pixel_ap": retained_ap - original_ap,
                "original_image_fpr": original_image_fpr,
                "retained_image_fpr": retained_image_fpr,
                "delta_image_fpr": retained_image_fpr - original_image_fpr,
                "postfinal_sensitivity": True,
                "primary_result_replaced": False,
            }
        )

    outputs = config["outputs"]
    condition_path = _resolve(project_root, outputs["condition_metrics"])
    robust_path = _resolve(project_root, outputs["robust_family"])
    summary_path = _resolve(project_root, outputs["summary"])
    report_path = _resolve(project_root, outputs["report"])
    _write_csv(condition_path, condition_rows)
    _write_csv(robust_path, robust_rows)
    affected_datasets = {
        str(row["final_source_group_id"]): str(row["final_source_dataset"])
        for row in neighbors
        if str(row["final_source_group_id"]) in affected
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "template_contamination_sensitivity_complete",
        "paper_evidence_role": "postfinal_contamination_sensitivity",
        "postfinal_sensitivity": True,
        "primary_result_replaced": False,
        "selection_performed": False,
        "model_or_threshold_change_performed": False,
        "exact_identity_gate_passed": True,
        "template_level_independence_passed": False,
        "total_prediction_records": len(records),
        "original_groups": len(all_groups),
        "retained_groups": len(retained_groups),
        "affected_groups": len(affected),
        "affected_source_group_ids": sorted(affected),
        "affected_group_counts_by_dataset": {
            dataset: sum(value == dataset for value in affected_datasets.values())
            for dataset in sorted(set(affected_datasets.values()))
        },
        "cord_groups": sum(value == "cord" for value in affected_datasets.values()),
        "xfund_groups": sum(value == "xfund" for value in affected_datasets.values()),
        "retained_forged_per_condition": int(expected["retained_forged_per_condition"]),
        "retained_authentic_per_condition": int(expected["retained_authentic_per_condition"]),
        "maximum_metric_replay_absolute_error": replay_error,
        "robust_family": {row["geometry"]: row for row in robust_rows},
        "checks": {
            "all_7488_predictions_present": len(records) == 7488,
            "all_records_successful": all(row["status"] == "ok" for row in records),
            "affected_groups_equal_frozen_flags": affected
            == _affected_groups_from_neighbors(neighbors),
            "original_metric_replay_passed": replay_error
            <= float(expected["metric_replay_absolute_tolerance"]),
            "retained_90_groups": len(retained_groups) == 90,
            "no_masks_or_images_read": True,
            "no_model_threshold_or_primary_result_change": True,
        },
        "inputs": {
            "config_sha256": _sha256(config_path),
            "protocol_sha256": _sha256(protocol_path),
            **{
                name + "_sha256": _sha256(path)
                for name, path in verified_paths.items()
            },
        },
        "outputs": {
            "condition_metrics": str(condition_path.relative_to(project_root)),
            "condition_metrics_sha256": _sha256(condition_path),
            "robust_family": str(robust_path.relative_to(project_root)),
            "robust_family_sha256": _sha256(robust_path),
            "report": str(report_path.relative_to(project_root)),
        },
    }
    if not all(summary["checks"].values()) and runtime["require_all_records"]:
        _write_json(summary_path, summary)
        raise RuntimeError("template contamination sensitivity failed")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(summary, robust_rows), encoding="utf-8")
    summary["outputs"]["report_sha256"] = _sha256(report_path)
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
