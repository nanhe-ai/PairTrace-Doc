from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from pairtrace_doc.pipelines.evaluate_baselines_100 import (
    _load_mask,
    _load_native_scores,
    _ranking_metrics,
)
from pairtrace_doc.pipelines.evaluate_resampling_robust_final_reserve import (
    FAMILY_SEEDS,
    FINAL_ROLES,
    _clustered_paired_bootstrap,
    _mean_score_maps,
    _role_generator_macro,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)


def _validate_artifact(project_root: Path, specification: dict[str, Any]) -> Path:
    path = _resolve(project_root, specification["path"])
    if _sha256(path) != specification["sha256"]:
        raise ValueError(f"final evidence artifact changed: {path}")
    return path


def _method_seed_maps(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in FAMILY_SEEDS}
    condition_to_seed = {f"robust_{seed}_clean_ecc": seed for seed in FAMILY_SEEDS}
    for row in rows:
        seed = condition_to_seed.get(str(row.get("condition")))
        if seed is None or row.get("sample_kind") != "forged":
            continue
        key = f'{row["source_group_id"]}|{row["evaluation_role"]}'
        if key in result[seed]:
            raise ValueError(f"duplicate clean robust final record: {seed} {key}")
        result[seed][key] = {
            "source_group_id": str(row["source_group_id"]),
            "source_stratum": str(row["source_stratum"]),
            "evaluation_role": str(row["evaluation_role"]),
            "generator": str(row["generator"]),
            "value": float(row["macro_pixel_ap"]),
        }
    expected = len(FINAL_ROLES) * 96
    if any(len(mapping) != expected for mapping in result.values()):
        raise ValueError("clean robust final prediction topology is incomplete")
    return result


def _baseline_map(
    baseline: str,
    scratch: Path,
    manifest: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    in_domain_by_group = {
        str(row["source_group_id"]): row
        for row in manifest
        if row["evaluation_role"] == "in_domain_test" and row["sample_kind"] == "forged"
    }
    mapping: dict[str, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []
    for row in manifest:
        if row["evaluation_role"] not in FINAL_ROLES or row["sample_kind"] != "forged":
            continue
        record_id = str(row["record_id"])
        prediction = predictions[record_id]
        scores = _load_native_scores(
            scratch,
            prediction,
            (int(row["height"]), int(row["width"])),
        )
        mask = _load_mask(scratch, row)
        pixel_ap, _ = _ranking_metrics(scores, mask)
        group = str(row["source_group_id"])
        in_domain = in_domain_by_group[group]
        item = {
            "baseline": baseline,
            "record_id": record_id,
            "source_group_id": group,
            "source_stratum": f'{in_domain["generator"]}|{in_domain["source_dataset"]}',
            "evaluation_role": str(row["evaluation_role"]),
            "generator": str(row["generator"]),
            "macro_pixel_ap": pixel_ap,
            "paper_evidence": True,
        }
        key = f'{group}|{row["evaluation_role"]}'
        if key in mapping:
            raise ValueError(f"duplicate strong-baseline final record: {baseline} {key}")
        mapping[key] = {
            **{field: item[field] for field in (
                "source_group_id",
                "source_stratum",
                "evaluation_role",
                "generator",
            )},
            "value": pixel_ap,
        }
        details.append(item)
    expected = len(FINAL_ROLES) * 96
    if len(mapping) != expected:
        raise ValueError(f"strong-baseline final topology is incomplete: {baseline}")
    return mapping, details


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if any(runtime.get(key) for key in (
        "gpu_launch_authorized",
        "method_training_authorized",
        "method_change_authorized",
        "threshold_selection_authorized",
    )):
        raise ValueError("final evidence summary must be artifact-only and selection-free")
    protocol = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("final evidence protocol SHA-256 changed")

    manifest_path = _validate_artifact(project_root, config["input"]["manifest"])
    method_summary_path = _validate_artifact(project_root, config["input"]["method_summary"])
    method_predictions_path = _validate_artifact(
        project_root, config["input"]["method_predictions"]
    )
    strong_summary_path = _validate_artifact(project_root, config["input"]["strong_summary"])
    method_summary = json.loads(method_summary_path.read_text(encoding="utf-8"))
    strong_summary = json.loads(strong_summary_path.read_text(encoding="utf-8"))
    if method_summary["status"] != "final_reserve_gate_passed":
        raise ValueError("method final gate did not pass")
    if strong_summary["status"] != "final_strong_baseline_evaluation_complete":
        raise ValueError("strong-baseline final evaluation is incomplete")

    manifest = _read_jsonl(manifest_path)
    if len(manifest) != int(config["input"]["expected_records"]):
        raise ValueError("final evidence manifest record count changed")
    method_seed_maps = _method_seed_maps(_read_jsonl(method_predictions_path))
    robust_mean = _mean_score_maps(list(method_seed_maps.values()))
    robust_ranking = _role_generator_macro(list(robust_mean.values()), "value")

    scratch = Path(
        os.environ.get(
            config["paths"]["scratch_env"],
            str(_resolve(project_root, config["paths"]["scratch_default"])),
        )
    ).resolve()
    comparison_rows: list[dict[str, Any]] = []
    comparison_details: list[dict[str, Any]] = []
    comparison_results: dict[str, Any] = {}
    for index, (baseline, specification) in enumerate(config["baselines"].items()):
        prediction_path = _validate_artifact(project_root, specification["predictions"])
        predictions = {str(row["record_id"]): row for row in _read_jsonl(prediction_path)}
        if len(predictions) != len(manifest) or any(row["status"] != "ok" for row in predictions.values()):
            raise ValueError(f"strong-baseline final predictions are incomplete: {baseline}")
        baseline_mapping, details = _baseline_map(
            baseline, scratch, manifest, predictions
        )
        comparison_details.extend(details)
        baseline_ranking = _role_generator_macro(list(baseline_mapping.values()), "value")
        comparison = _clustered_paired_bootstrap(
            robust_mean,
            baseline_mapping,
            int(config["bootstrap"]["seed"]) + index,
            int(config["bootstrap"]["resamples"]),
            float(config["bootstrap"]["confidence_level"]),
        )
        comparison_results[baseline] = {
            "robust_mean": robust_ranking,
            "baseline": baseline_ranking,
            "robust_mean_minus_baseline": comparison,
        }
        comparison_rows.append(
            {
                "baseline": baseline,
                "robust_role_macro_generator_macro_pixel_ap": robust_ranking[
                    "role_macro_generator_macro"
                ],
                "baseline_role_macro_generator_macro_pixel_ap": baseline_ranking[
                    "role_macro_generator_macro"
                ],
                "difference": comparison["effect"],
                "ci_low": comparison["ci_low"],
                "ci_high": comparison["ci_high"],
                "robust_in_domain_pixel_ap": robust_ranking["per_role"]["in_domain_test"],
                "baseline_in_domain_pixel_ap": baseline_ranking["per_role"]["in_domain_test"],
                "robust_generator_holdout_pixel_ap": robust_ranking["per_role"]["generator_holdout"],
                "baseline_generator_holdout_pixel_ap": baseline_ranking["per_role"]["generator_holdout"],
                "bootstrap_resamples": comparison["bootstrap_resamples"],
                "threshold_selection_used": False,
                "paper_evidence": True,
            }
        )

    details_path = _resolve(project_root, config["paths"]["details"])
    comparisons_path = _resolve(project_root, config["paths"]["comparisons"])
    summary_path = _resolve(project_root, config["paths"]["summary"])
    _write_jsonl(details_path, comparison_details)
    _write_csv(comparisons_path, comparison_rows)
    result = {
        "experiment": config["experiment"],
        "status": "final_evidence_summary_complete",
        "paper_evidence": True,
        "final_reserve_read": True,
        "method_or_threshold_selection_used": False,
        "selected_groups": 96,
        "method_final_gate_passed": True,
        "strong_baseline_final_evaluation_complete": True,
        "comparison_results": comparison_results,
        "input_artifact_sha256": {
            "manifest": _sha256(manifest_path),
            "method_summary": _sha256(method_summary_path),
            "method_predictions": _sha256(method_predictions_path),
            "strong_summary": _sha256(strong_summary_path),
        },
        "outputs": {
            "details": str(details_path.relative_to(project_root)),
            "details_sha256": _sha256(details_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
        },
    }
    _write_json(summary_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
