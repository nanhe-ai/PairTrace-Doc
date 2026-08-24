from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pairtrace_doc.pipelines.evaluate_baselines_100 import (
    _load_mask,
    _load_native_scores,
    _roc_auc,
    _threshold_metrics,
)
from pairtrace_doc.pipelines.evaluate_tfr_zero_shot_bridge import (
    ROBUST_MODELS,
    _paired_bootstrap,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _ranking_metrics,
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)


FORGED_FIELDS = (
    "pixel_ap",
    "pixel_auroc",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pixel_iou",
    "forged_image_score",
)


def _group_macro_rows(
    forged: list[dict[str, Any]],
    authentic: dict[str, dict[str, Any]],
    paper_evidence: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in forged:
        grouped[str(row["source_group_id"])].append(row)
    if set(grouped) != set(authentic):
        raise ValueError("baseline forged/authentic source groups differ")
    result: list[dict[str, Any]] = []
    for group, variants in sorted(grouped.items()):
        item: dict[str, Any] = {
            "source_group_id": group,
            "forged_variants": len(variants),
            "authentic_pixel_fpr": float(authentic[group]["authentic_pixel_fpr"]),
            "authentic_image_score": float(authentic[group]["authentic_image_score"]),
            "paper_evidence": paper_evidence,
        }
        for field in FORGED_FIELDS:
            item[field] = float(np.mean([row[field] for row in variants]))
        result.append(item)
    return result


def _bootstrap_mean(
    values: np.ndarray,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    replicas = values[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(replicas, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _aggregate_groups(
    baseline: str,
    groups: list[dict[str, Any]],
    pixel_threshold: float,
    image_threshold: float,
    bootstrap: dict[str, Any],
    seed_offset: int,
    paper_evidence: bool = False,
) -> dict[str, Any]:
    if not groups:
        raise ValueError("baseline aggregation requires source groups")
    result: dict[str, Any] = {
        "baseline": baseline,
        "source_groups": len(groups),
        "forged_pairs": sum(int(row["forged_variants"]) for row in groups),
        "pixel_threshold_frozen_on_aiforge": pixel_threshold,
        "image_threshold_frozen_on_aiforge": image_threshold,
        "threshold_selected_on_tfr": False,
        "paper_evidence": paper_evidence,
    }
    metric_fields = (
        *FORGED_FIELDS[:-1],
        "authentic_pixel_fpr",
    )
    for metric_index, field in enumerate(metric_fields):
        values = np.asarray([row[field] for row in groups], dtype=float)
        low, high = _bootstrap_mean(
            values,
            int(bootstrap["seed"]) + seed_offset * 100 + metric_index,
            int(bootstrap["resamples"]),
            float(bootstrap["confidence_level"]),
        )
        name = "source_group_macro_" + field
        result[name] = float(values.mean())
        result[name + "_ci_low"] = low
        result[name + "_ci_high"] = high
    forged_image = np.asarray([row["forged_image_score"] for row in groups])
    authentic_image = np.asarray([row["authentic_image_score"] for row in groups])
    result["source_group_balanced_image_auroc"] = _roc_auc(
        np.r_[forged_image, authentic_image],
        np.r_[
            np.ones(len(groups), dtype=bool),
            np.zeros(len(groups), dtype=bool),
        ],
    )
    result["source_group_macro_image_tpr_at_aiforge_frozen_threshold"] = float(
        np.mean(forged_image >= image_threshold)
    )
    result["unique_authentic_image_fpr_at_aiforge_frozen_threshold"] = float(
        np.mean(authentic_image >= image_threshold)
    )
    return result


def _core_group_ap(predictions: list[dict[str, Any]]) -> dict[str, float]:
    by_model: dict[str, dict[str, list[float]]] = {
        model: defaultdict(list) for model in ROBUST_MODELS
    }
    for row in predictions:
        for model in ROBUST_MODELS:
            if (
                row["condition"] == f"{model}_clean_ecc"
                and row["sample_kind"] == "forged"
            ):
                by_model[model][str(row["source_group_id"])].append(
                    float(row["pixel_ap"])
                )
    group_sets = [set(values) for values in by_model.values()]
    if not group_sets or any(groups != group_sets[0] for groups in group_sets[1:]):
        raise ValueError("robust core seed group topology differs")
    return {
        group: float(
            np.mean(
                [np.mean(by_model[model][group]) for model in ROBUST_MODELS]
            )
        )
        for group in sorted(group_sets[0])
    }


def _evidence_mode(config: dict[str, Any]) -> bool:
    stage = str(config["experiment"]["stage"])
    runtime = config["runtime"]
    if runtime["gpu_launch_authorized"] or runtime["method_training_authorized"]:
        raise ValueError("TFR baseline aggregation must be artifact-only")
    if stage == "one_shot_holdout_artifact_aggregation":
        if (
            not config["experiment"]["paper_evidence"]
            or not runtime["tfr_holdout_read_allowed"]
        ):
            raise ValueError("TFR holdout aggregation evidence boundary is not authorized")
        return True
    if stage != "viewed_development_artifact_aggregation":
        raise ValueError("unsupported TFR strong-baseline aggregation stage")
    if runtime["tfr_holdout_read_allowed"] or config["experiment"]["paper_evidence"]:
        raise ValueError("TFR viewed-development baselines crossed their evidence boundary")
    return False


def _validate_manifest_boundary(
    manifest: list[dict[str, Any]], paper_evidence: bool
) -> None:
    if paper_evidence:
        if any(
            row.get("paper_evidence_candidate") is not True
            or row.get("tfr_holdout_read") is not True
            or row.get("evaluation_role") != "one_shot_holdout"
            for row in manifest
        ):
            raise ValueError("TFR holdout baseline manifest crossed its evidence boundary")
        return
    if any(
        bool(row.get("tfr_holdout_read")) or bool(row.get("paper_evidence"))
        for row in manifest
    ):
        raise ValueError("TFR viewed-development baseline manifest crossed its evidence boundary")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paper_evidence = _evidence_mode(config)

    protocol = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("TFR strong-baseline protocol SHA-256 changed")
    input_config = config["input"]
    manifest_path = _resolve(project_root, input_config["manifest"])
    if _sha256(manifest_path) != input_config["expected_manifest_sha256"]:
        raise ValueError("TFR baseline manifest SHA-256 changed")
    manifest = _read_jsonl(manifest_path)
    if len(manifest) != int(input_config["expected_records"]):
        raise ValueError("TFR baseline manifest record count changed")
    forged_manifest = [row for row in manifest if row["sample_kind"] == "forged"]
    authentic_manifest = {
        str(row["source_group_id"]): row
        for row in manifest
        if row["sample_kind"] == "authentic"
    }
    if len(forged_manifest) != int(input_config["expected_forged_pairs"]):
        raise ValueError("TFR baseline forged-pair count changed")
    if len(authentic_manifest) != int(input_config["expected_source_groups"]):
        raise ValueError("TFR baseline authentic source-group count changed")
    _validate_manifest_boundary(manifest, paper_evidence)

    materialization_summary: dict[str, Any] | None = None
    if paper_evidence:
        materialization_path = _resolve(
            project_root, input_config["materialization_summary"]
        )
        if (
            _sha256(materialization_path)
            != input_config["expected_materialization_summary_sha256"]
        ):
            raise ValueError("TFR holdout materialization summary changed")
        materialization_summary = json.loads(
            materialization_path.read_text(encoding="utf-8")
        )
        if (
            materialization_summary["status"]
            != "tfr_one_shot_holdout_materialized"
            or materialization_summary["outputs"]["baseline_manifest_sha256"]
            != input_config["expected_manifest_sha256"]
            or int(materialization_summary["baseline_records"]) != len(manifest)
            or int(materialization_summary["forged_pairs"]) != len(forged_manifest)
            or int(materialization_summary["source_groups"]) != len(authentic_manifest)
        ):
            raise ValueError("TFR holdout materialization is incomplete")

    thresholds_path = _resolve(project_root, input_config["thresholds"])
    if _sha256(thresholds_path) != input_config["expected_thresholds_sha256"]:
        raise ValueError("AIForge baseline thresholds changed")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))["baselines"]
    core_summary_path = _resolve(project_root, input_config["core_summary"])
    if _sha256(core_summary_path) != input_config["expected_core_summary_sha256"]:
        raise ValueError("TFR core bridge summary changed")
    core_summary = json.loads(core_summary_path.read_text(encoding="utf-8"))
    accepted_core_statuses = {
        str(value)
        for value in input_config.get(
            "accepted_core_statuses",
            [input_config.get("expected_core_status", "bridge_gate_passed")],
        )
    }
    if core_summary["status"] not in accepted_core_statuses:
        raise ValueError("TFR core bridge status is not an accepted final outcome")
    if (
        bool(core_summary.get("paper_evidence")) != paper_evidence
        or bool(core_summary.get("tfr_holdout_read")) != paper_evidence
    ):
        raise ValueError("TFR core summary crossed its evidence boundary")
    core_predictions_path = _resolve(project_root, input_config["core_predictions"])
    if _sha256(core_predictions_path) != input_config["expected_core_predictions_sha256"]:
        raise ValueError("TFR core bridge predictions changed")
    core_predictions = _read_jsonl(core_predictions_path)
    if any(row["status"] != "ok" for row in core_predictions):
        raise ValueError("TFR core bridge predictions are incomplete")
    if any(
        bool(row.get("paper_evidence")) != paper_evidence
        or bool(row.get("tfr_holdout_read")) != paper_evidence
        for row in core_predictions
    ):
        raise ValueError("TFR core predictions crossed their evidence boundary")
    robust_group_ap = _core_group_ap(core_predictions)
    if set(robust_group_ap) != set(authentic_manifest):
        raise ValueError("TFR core and baseline source groups differ")
    robust_mean_ap = float(np.mean(list(robust_group_ap.values())))
    frozen_core_mean = float(
        core_summary["decision"]["values"]["mean_clean_source_group_macro_ap"]
    )
    if not np.isclose(robust_mean_ap, frozen_core_mean, rtol=0, atol=1e-12):
        raise ValueError("TFR core AP reconstruction changed")
    if paper_evidence:
        primary_endpoint = core_summary.get("primary_endpoint")
        if (
            not isinstance(primary_endpoint, dict)
            or primary_endpoint.get("paper_evidence") is not True
            or not np.isclose(
                robust_mean_ap,
                float(primary_endpoint["estimate"]),
                rtol=0,
                atol=1e-12,
            )
        ):
            raise ValueError("TFR holdout primary endpoint reconstruction changed")

    scratch = Path(
        os.environ.get(
            config["paths"]["scratch_env"],
            str(_resolve(project_root, config["paths"]["scratch_default"])),
        )
    ).resolve()
    manifest_by_id = {str(row["record_id"]): row for row in manifest}
    all_group_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    artifact_hashes: dict[str, Any] = {}
    for baseline_index, (baseline, specification) in enumerate(
        config["baselines"].items()
    ):
        summary_path = _resolve(project_root, specification["run_summary"])
        if _sha256(summary_path) != specification["expected_run_summary_sha256"]:
            raise ValueError(f"{baseline} run summary changed")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary["status"] != "passed"
            or int(summary["successful_records"]) != len(manifest)
            or int(summary["failed_records"]) != 0
            or summary["checkpoint_sha256"] != specification["checkpoint_sha256"]
            or bool(summary.get("paper_evidence")) != paper_evidence
            or summary.get("input_manifest_sha256")
            != input_config["expected_manifest_sha256"]
        ):
            raise ValueError(f"{baseline} inference is incomplete")
        prediction_path = _resolve(project_root, specification["predictions"])
        if _sha256(prediction_path) != specification["expected_predictions_sha256"]:
            raise ValueError(f"{baseline} predictions changed")
        prediction_rows = _read_jsonl(prediction_path)
        predictions = {str(row["record_id"]): row for row in prediction_rows}
        if (
            set(predictions) != set(manifest_by_id)
            or any(row["status"] != "ok" for row in prediction_rows)
            or any(
                bool(row.get("paper_evidence")) != paper_evidence
                for row in prediction_rows
            )
        ):
            raise ValueError(f"{baseline} prediction topology is incomplete")
        pixel_threshold = float(thresholds[baseline]["pixel"]["threshold"])
        image_threshold = float(thresholds[baseline]["image"]["threshold"])

        authentic_details: dict[str, dict[str, Any]] = {}
        for group, row in authentic_manifest.items():
            scores = _load_native_scores(
                scratch,
                predictions[str(row["record_id"])],
                (int(row["height"]), int(row["width"])),
            )
            authentic_details[group] = {
                "authentic_pixel_fpr": float(np.mean(scores >= pixel_threshold)),
                "authentic_image_score": float(
                    predictions[str(row["record_id"])]["image_score"]
                ),
            }
        forged_details: list[dict[str, Any]] = []
        for row in forged_manifest:
            prediction = predictions[str(row["record_id"])]
            scores = _load_native_scores(
                scratch,
                prediction,
                (int(row["height"]), int(row["width"])),
            )
            mask = _load_mask(scratch, row)
            pixel_ap, pixel_auroc = _ranking_metrics(scores, mask)
            forged_details.append(
                {
                    "record_id": str(row["record_id"]),
                    "source_group_id": str(row["source_group_id"]),
                    "pixel_ap": pixel_ap,
                    "pixel_auroc": pixel_auroc,
                    **_threshold_metrics(scores, mask, pixel_threshold),
                    "forged_image_score": float(prediction["image_score"]),
                }
            )
        groups = _group_macro_rows(
            forged_details, authentic_details, paper_evidence=paper_evidence
        )
        for row in groups:
            row["baseline"] = baseline
        all_group_rows.extend(groups)
        aggregate = _aggregate_groups(
            baseline,
            groups,
            pixel_threshold,
            image_threshold,
            config["bootstrap"],
            baseline_index,
            paper_evidence=paper_evidence,
        )
        aggregate_rows.append(aggregate)
        baseline_group_ap = {
            str(row["source_group_id"]): float(row["pixel_ap"]) for row in groups
        }
        comparison = _paired_bootstrap(
            robust_group_ap,
            baseline_group_ap,
            int(config["bootstrap"]["seed"]) + baseline_index,
            int(config["bootstrap"]["resamples"]),
            float(config["bootstrap"]["confidence_level"]),
        )
        comparison_rows.append(
            {
                "comparison": f"robust_three_seed_clean_minus_{baseline}",
                "left_information": "candidate_plus_authentic_reference",
                "right_information": "candidate_only",
                "left_source_group_macro_pixel_ap": robust_mean_ap,
                "right_source_group_macro_pixel_ap": aggregate[
                    "source_group_macro_pixel_ap"
                ],
                **comparison,
                "paper_evidence": paper_evidence,
            }
        )
        artifact_hashes[baseline] = {
            "run_summary_sha256": _sha256(summary_path),
            "predictions_sha256": _sha256(prediction_path),
        }

    paths = config["paths"]
    group_metrics_path = _resolve(project_root, paths["group_metrics"])
    metrics_path = _resolve(project_root, paths["metrics"])
    comparisons_path = _resolve(project_root, paths["comparisons"])
    summary_path = _resolve(project_root, paths["summary"])
    _write_jsonl(group_metrics_path, all_group_rows)
    _write_csv(metrics_path, aggregate_rows)
    _write_csv(comparisons_path, comparison_rows)
    output = {
        "status": "tfr_strong_baseline_evaluation_complete",
        "experiment": config["experiment"],
        "paper_evidence": paper_evidence,
        "tfr_holdout_read": paper_evidence,
        "tfr_threshold_selection_performed": False,
        "source_groups": len(authentic_manifest),
        "forged_pairs": len(forged_manifest),
        "baseline_predictions_completed": len(config["baselines"]) * len(manifest),
        "baseline_prediction_failures": 0,
        "robust_three_seed_clean_source_group_macro_pixel_ap": robust_mean_ap,
        "metrics": aggregate_rows,
        "comparisons": comparison_rows,
        "input_artifact_sha256": artifact_hashes,
        "protocol_sha256": _sha256(protocol),
        "manifest_sha256": _sha256(manifest_path),
        "materialization_summary_sha256": (
            None
            if materialization_summary is None
            else input_config["expected_materialization_summary_sha256"]
        ),
        "outputs": {
            "group_metrics": str(group_metrics_path.relative_to(project_root)),
            "group_metrics_sha256": _sha256(group_metrics_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
        },
    }
    _write_json(summary_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
