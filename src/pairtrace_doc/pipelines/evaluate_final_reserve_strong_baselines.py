from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pairtrace_doc.pipelines.evaluate_baselines_100 import (
    _bootstrap_interval,
    _load_mask,
    _load_native_scores,
    _roc_auc,
    _threshold_metrics,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
)


FINAL_ROLES = ("in_domain_test", "generator_holdout")


def _aggregate_role(
    baseline: str,
    role: str,
    details: list[dict[str, Any]],
    pixel_threshold: float,
    image_threshold: float,
    bootstrap: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    if not details:
        raise ValueError(f"missing final baseline role: {baseline} {role}")
    rng = np.random.default_rng(int(bootstrap["seed"]) + seed_offset)
    result: dict[str, Any] = {
        "baseline": baseline,
        "evaluation_role": role,
        "groups": len(details),
        "pixel_threshold": pixel_threshold,
        "image_threshold": image_threshold,
        "threshold_selected_on_final_reserve": False,
        "paper_evidence": True,
    }
    for metric in (
        "macro_pixel_ap",
        "pixel_auroc",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pixel_iou",
        "authentic_pixel_fpr",
    ):
        values = np.asarray([row[metric] for row in details], dtype=float)
        low, high = _bootstrap_interval(
            values,
            rng,
            int(bootstrap["resamples"]),
            float(bootstrap["confidence_level"]),
        )
        result[metric] = float(values.mean())
        result[f"{metric}_ci_low"] = low
        result[f"{metric}_ci_high"] = high
    forged_image = np.asarray([row["forged_image_score"] for row in details])
    authentic_image = np.asarray([row["authentic_image_score"] for row in details])
    result["image_auroc"] = _roc_auc(
        np.r_[forged_image, authentic_image],
        np.r_[np.ones(len(details), dtype=bool), np.zeros(len(details), dtype=bool)],
    )
    result["image_tpr_at_development_frozen_threshold"] = float(
        np.mean(forged_image >= image_threshold)
    )
    result["image_fpr_at_development_frozen_threshold"] = float(
        np.mean(authentic_image >= image_threshold)
    )
    return result


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if runtime["gpu_launch_authorized"] or runtime["method_training_authorized"] or runtime["method_change_authorized"]:
        raise ValueError("final baseline evaluation must be artifact-only")
    if not runtime["final_reserve_score_read_allowed"]:
        raise ValueError("final baseline score read was not authorized")
    protocol = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("final protocol SHA-256 changed")
    manifest_path = _resolve(project_root, config["input"]["manifest"])
    if _sha256(manifest_path) != config["input"]["expected_manifest_sha256"]:
        raise ValueError("final manifest SHA-256 changed")
    manifest = _read_jsonl(manifest_path)
    if len(manifest) != int(config["input"]["expected_records"]):
        raise ValueError("final manifest record count changed")
    by_id = {str(row["record_id"]): row for row in manifest}
    authentic_by_group = {
        str(row["source_group_id"]): row
        for row in manifest
        if row["evaluation_role"] == "final_test" and row["sample_kind"] == "authentic"
    }
    scratch = Path(
        os.environ.get(
            config["paths"]["scratch_env"],
            str(_resolve(project_root, config["paths"]["scratch_default"])),
        )
    ).resolve()
    threshold_path = _resolve(project_root, config["input"]["thresholds"])
    if _sha256(threshold_path) != config["input"]["expected_thresholds_sha256"]:
        raise ValueError("strong-baseline thresholds changed")
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))["baselines"]

    metric_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    for baseline_index, (baseline, specification) in enumerate(config["baselines"].items()):
        summary_path = _resolve(project_root, specification["summary"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["status"] != "passed" or int(summary["successful_records"]) != len(manifest):
            raise ValueError(f"final baseline inference incomplete: {baseline}")
        if summary["input_manifest_sha256"] != config["input"]["expected_manifest_sha256"]:
            raise ValueError(f"final baseline manifest changed: {baseline}")
        if summary["checkpoint_sha256"] != specification["checkpoint_sha256"]:
            raise ValueError(f"final baseline checkpoint changed: {baseline}")
        prediction_path = _resolve(project_root, summary["output_predictions"])
        if _sha256(prediction_path) != summary["output_predictions_sha256"]:
            raise ValueError(f"final baseline predictions changed: {baseline}")
        predictions = {str(row["record_id"]): row for row in _read_jsonl(prediction_path)}
        if set(predictions) != set(by_id) or any(row["status"] != "ok" for row in predictions.values()):
            raise ValueError(f"final baseline prediction topology changed: {baseline}")
        pixel_threshold = float(thresholds[baseline]["pixel"]["threshold"])
        image_threshold = float(thresholds[baseline]["image"]["threshold"])
        for role_index, role in enumerate(FINAL_ROLES):
            forged_rows = [
                row for row in manifest
                if row["evaluation_role"] == role and row["sample_kind"] == "forged"
            ]
            details = []
            for forged in forged_rows:
                group = str(forged["source_group_id"])
                authentic = authentic_by_group[group]
                forged_scores = _load_native_scores(
                    scratch,
                    predictions[str(forged["record_id"])],
                    (int(forged["height"]), int(forged["width"])),
                )
                mask = _load_mask(scratch, forged)
                from pairtrace_doc.pipelines.train_student_100 import _ranking_metrics
                average_precision, pixel_auroc = _ranking_metrics(forged_scores, mask)
                operational = _threshold_metrics(forged_scores, mask, pixel_threshold)
                authentic_scores = _load_native_scores(
                    scratch,
                    predictions[str(authentic["record_id"])],
                    (int(authentic["height"]), int(authentic["width"])),
                )
                details.append(
                    {
                        "macro_pixel_ap": average_precision,
                        "pixel_auroc": pixel_auroc,
                        **operational,
                        "authentic_pixel_fpr": float(np.mean(authentic_scores >= pixel_threshold)),
                        "forged_image_score": float(predictions[str(forged["record_id"])]["image_score"]),
                        "authentic_image_score": float(predictions[str(authentic["record_id"])]["image_score"]),
                    }
                )
            metric_rows.append(
                _aggregate_role(
                    baseline,
                    role,
                    details,
                    pixel_threshold,
                    image_threshold,
                    config["bootstrap"],
                    baseline_index * 10 + role_index,
                )
            )
        input_hashes[baseline] = {
            "run_summary_sha256": _sha256(summary_path),
            "predictions_sha256": _sha256(prediction_path),
        }
    metrics_path = _resolve(project_root, config["paths"]["metrics"])
    summary_path = _resolve(project_root, config["paths"]["summary"])
    _write_csv(metrics_path, metric_rows)
    output = {
        "experiment": config["experiment"],
        "status": "final_strong_baseline_evaluation_complete",
        "paper_evidence": True,
        "final_reserve_read": True,
        "threshold_selection_used": False,
        "selected_groups": len(authentic_by_group),
        "metrics": metric_rows,
        "input_artifact_sha256": input_hashes,
        "outputs": {
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
        },
    }
    _write_json(summary_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
