from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_json,
)


def _select_group_positions(
    group_scores: dict[str, float], cohorts: dict[str, dict[str, list[int]]]
) -> list[tuple[str, str]]:
    ordered = sorted(group_scores, key=lambda group: (group_scores[group], group))
    selected: list[tuple[str, str]] = []
    for cohort, specification in cohorts.items():
        for position in specification["positions"]:
            selected.append((cohort, ordered[int(position)]))
    if len({group for _, group in selected}) != len(selected):
        raise ValueError("qualitative audit selection contains duplicate groups")
    return selected


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    prohibited = (
        "tfr_holdout_read_allowed",
        "selected_image_read_authorized",
        "model_inference_authorized",
        "model_training_authorized",
        "threshold_selection_authorized",
        "sample_replacement_authorized",
        "human_audit_completion_authorized",
    )
    if any(runtime[name] for name in prohibited):
        raise ValueError("TFR audit freeze crossed its evidence boundary")
    protocol = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("TFR qualitative protocol SHA-256 changed")

    input_config = config["input"]
    manifest_path = _resolve(project_root, input_config["manifest"])
    if _sha256(manifest_path) != input_config["expected_manifest_sha256"]:
        raise ValueError("TFR train/validation manifest changed")
    rows = sorted(
        [
            row
            for row in _read_jsonl(manifest_path)
            if row["pilot_role"] == input_config["role"]
        ],
        key=lambda row: (str(row["source_group_id"]), str(row["sample_id"])),
    )
    if len(rows) != int(input_config["expected_pairs"]):
        raise ValueError("TFR qualitative pair count changed")
    if len({str(row["source_group_id"]) for row in rows}) != int(
        input_config["expected_source_groups"]
    ):
        raise ValueError("TFR qualitative source-group count changed")
    predictions_path = _resolve(project_root, input_config["bridge_predictions"])
    if _sha256(predictions_path) != input_config["expected_bridge_predictions_sha256"]:
        raise ValueError("TFR bridge predictions changed")
    predictions = _read_jsonl(predictions_path)
    if any(row["status"] != "ok" for row in predictions):
        raise ValueError("TFR bridge predictions are incomplete")

    selection = config["selection"]
    models = [str(value) for value in selection["robust_models"]]
    clean_by_model: dict[str, dict[str, dict[str, float]]] = {
        model: defaultdict(dict) for model in models
    }
    prediction_index: dict[tuple[str, str], dict[str, Any]] = {}
    required_conditions = {
        f"{scorer}_{geometry}_ecc"
        for scorer in selection["render_scorers"]
        for geometry in selection["render_geometries"]
    }
    for record in predictions:
        condition = str(record["condition"])
        sample_id = str(record["sample_id"])
        if record["sample_kind"] == "forged" and condition in required_conditions:
            prediction_index[(sample_id, condition)] = record
        for model in models:
            if (
                record["sample_kind"] == "forged"
                and condition == f"{model}_clean_ecc"
            ):
                clean_by_model[model][str(record["source_group_id"])][sample_id] = float(
                    record["pixel_ap"]
                )
    manifest_by_sample = {str(row["sample_id"]): row for row in rows}
    group_samples: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group_samples[str(row["source_group_id"])].append(str(row["sample_id"]))
    if any(set(clean_by_model[model]) != set(group_samples) for model in models):
        raise ValueError("TFR qualitative clean-score topology changed")

    sample_scores: dict[str, float] = {}
    group_scores: dict[str, float] = {}
    for group, samples in group_samples.items():
        for sample in samples:
            sample_scores[sample] = float(
                np.mean([clean_by_model[model][group][sample] for model in models])
            )
        group_scores[group] = float(np.mean([sample_scores[sample] for sample in samples]))
    selected = _select_group_positions(group_scores, selection["cohorts"])

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    cases: list[dict[str, Any]] = []
    cohort_counts: dict[str, int] = defaultdict(int)
    for cohort, group in selected:
        cohort_counts[cohort] += 1
        samples = group_samples[group]
        sample_id = min(
            samples,
            key=lambda sample: (
                abs(sample_scores[sample] - group_scores[group]),
                sample,
            ),
        )
        row = manifest_by_sample[sample_id]
        maps: list[dict[str, Any]] = []
        for geometry in selection["render_geometries"]:
            for scorer in selection["render_scorers"]:
                condition = f"{scorer}_{geometry}_ecc"
                record = prediction_index[(sample_id, condition)]
                cache_path = _resolve(scratch, record["score_cache"])
                maps.append(
                    {
                        "condition": condition,
                        "scorer": scorer,
                        "geometry": geometry,
                        "score_cache": str(cache_path.relative_to(scratch)),
                        "score_cache_sha256": _sha256(cache_path),
                        "score_shape": record["score_shape"],
                        "pixel_threshold": float(selection["pixel_thresholds"][scorer]),
                        "pixel_ap": float(record["pixel_ap"]),
                    }
                )
        cases.append(
            {
                "case_id": f"tfr_{cohort}_{cohort_counts[cohort]:02d}",
                "cohort": cohort,
                "source_group_id": group,
                "sample_id": sample_id,
                "group_three_seed_mean_clean_ap": group_scores[group],
                "variant_three_seed_mean_clean_ap": sample_scores[sample_id],
                "forged_variants_in_group": len(samples),
                "candidate": str(row["image"]),
                "candidate_sha256": str(row["image_sha256"]),
                "authentic": str(row["authentic"]),
                "authentic_sha256": str(row["authentic_sha256"]),
                "mask": str(row["mask"]),
                "mask_sha256": str(row["mask_sha256"]),
                "maps": maps,
                "paper_evidence": False,
                "tfr_holdout_read": False,
            }
        )
    case_manifest = {
        "status": "tfr_qualitative_cases_frozen_before_image_render",
        "experiment": config["experiment"],
        "paper_evidence": False,
        "tfr_holdout_read": False,
        "sample_replacement_allowed": False,
        "selection": selection,
        "case_count": len(cases),
        "cases": cases,
        "source_artifacts": {
            "manifest_sha256": _sha256(manifest_path),
            "bridge_predictions_sha256": _sha256(predictions_path),
        },
    }
    case_manifest_path = _resolve(project_root, paths["case_manifest"])
    worksheet_path = _resolve(project_root, paths["blank_worksheet"])
    _write_json(case_manifest_path, case_manifest)
    worksheet = {
        "schema_version": 1,
        "status": "pending_independent_human_review",
        "paper_evidence": False,
        "human_review_complete": False,
        "case_manifest": str(case_manifest_path.relative_to(project_root)),
        "case_manifest_sha256": _sha256(case_manifest_path),
        "allowed_values": {
            "mapping_valid": ["yes", "no", "uncertain"],
            "mask_valid": ["yes", "no", "uncertain"],
            "registration_artifact": ["none", "minor", "major", "uncertain"],
            "localization_quality": ["good", "partial", "failed", "uncertain"],
        },
        "reviews": [
            {
                "case_id": case["case_id"],
                "mapping_valid": None,
                "mask_valid": None,
                "registration_artifact": None,
                "localization_quality": None,
                "failure_mode": None,
                "reviewer_note": None,
                "reviewer_identifier": None,
                "reviewed_at_utc": None,
            }
            for case in cases
        ],
    }
    _write_json(worksheet_path, worksheet)
    return {
        "status": case_manifest["status"],
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "case_manifest": str(case_manifest_path.relative_to(project_root)),
        "case_manifest_sha256": _sha256(case_manifest_path),
        "blank_worksheet": str(worksheet_path.relative_to(project_root)),
        "blank_worksheet_sha256": _sha256(worksheet_path),
        "human_review_complete": False,
        "paper_evidence": False,
        "tfr_holdout_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
