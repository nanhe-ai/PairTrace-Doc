from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
)
from pairtrace_doc.pipelines.train_tfr_equal_budget_arm import ARMS


def _group_ap(predictions: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in predictions:
        if row["sample_kind"] != "forged_pair":
            continue
        grouped.setdefault(str(row["source_group_id"]), []).append(
            float(row["average_precision"])
        )
    if not grouped:
        raise ValueError("comparison received no forged-pair predictions")
    return {group: float(np.mean(values)) for group, values in grouped.items()}


def _paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    seed: int,
    replicates: int,
) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("paired bootstrap source groups differ across arms")
    groups = sorted(left)
    observed = float(np.mean([left[group] - right[group] for group in groups]))
    rng = np.random.default_rng(seed)
    differences = np.asarray([left[group] - right[group] for group in groups])
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(groups), size=len(groups))
        samples[index] = float(differences[selected].mean())
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "delta": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "ci95_excludes_zero": bool(lower > 0 or upper < 0),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("equal-budget comparison protocol SHA-256 changed")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("development comparison cannot be paper evidence")

    arm_summaries: dict[str, dict[str, Any]] = {}
    arm_groups: dict[str, dict[str, float]] = {}
    for arm, paths in config["arms"].items():
        if arm not in ARMS:
            raise ValueError(f"unknown configured arm: {arm}")
        summary_path = _resolve(project_root, paths["summary"])
        prediction_path = _resolve(project_root, paths["predictions"])
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary["arm"] != arm or summary["status"] != "development_arm_complete":
            raise ValueError(f"arm {arm} did not complete the frozen development run")
        if summary["prediction_records_sha256"] != _sha256(prediction_path):
            raise ValueError(f"arm {arm} prediction hash changed")
        if summary["holdout_read"] or summary["silent_failures"] != 0:
            raise ValueError(f"arm {arm} violates the comparison data/failure boundary")
        predictions = _read_jsonl(prediction_path)
        groups = _group_ap(predictions)
        computed = float(np.mean(list(groups.values())))
        reported = float(summary["validation_source_group_macro_pixel_ap_model_resolution"])
        if not np.isclose(computed, reported, atol=1e-12):
            raise ValueError(f"arm {arm} summary AP is inconsistent with predictions")
        arm_summaries[arm] = summary
        arm_groups[arm] = groups
    if set(arm_summaries) != ARMS:
        raise ValueError("comparison requires all four frozen arms")
    group_sets = {tuple(sorted(groups)) for groups in arm_groups.values()}
    if len(group_sets) != 1:
        raise ValueError("comparison arms do not share identical validation groups")
    group_count = len(next(iter(arm_groups.values())))
    if group_count != int(config["data"]["expected_validation_groups"]):
        raise ValueError("comparison validation group count changed")

    bootstrap = config["bootstrap"]
    seed = int(bootstrap["seed"])
    replicates = int(bootstrap["replicates"])
    contrasts = {
        "explicit_9ch_minus_candidate_reference_6ch": _paired_bootstrap(
            arm_groups["explicit_9ch"],
            arm_groups["candidate_reference_6ch"],
            seed,
            replicates,
        ),
        "explicit_9ch_minus_fc_siam_diff": _paired_bootstrap(
            arm_groups["explicit_9ch"],
            arm_groups["fc_siam_diff"],
            seed,
            replicates,
        ),
        "explicit_9ch_minus_signed_difference_3ch": _paired_bootstrap(
            arm_groups["explicit_9ch"],
            arm_groups["signed_difference_3ch"],
            seed,
            replicates,
        ),
    }
    continuation = config["continuation_gate"]
    every_arm_complete = len(arm_summaries) == 4
    authentic_fpr_pass = all(
        float(summary["operating_point"]["unique_authentic_group_macro_pixel_fpr"])
        <= float(continuation["max_authentic_fpr"])
        for summary in arm_summaries.values()
    )
    delta_9_minus_6_pass = (
        contrasts["explicit_9ch_minus_candidate_reference_6ch"]["delta"]
        >= float(continuation["minimum_explicit_9ch_delta"])
    )
    delta_9_minus_fc_pass = (
        contrasts["explicit_9ch_minus_fc_siam_diff"]["delta"]
        >= float(continuation["minimum_explicit_9ch_delta"])
    )
    required_ci_excludes_zero = any(
        contrasts[name]["ci95_excludes_zero"]
        for name in (
            "explicit_9ch_minus_candidate_reference_6ch",
            "explicit_9ch_minus_fc_siam_diff",
        )
    )
    gate_checks = {
        "every_arm_complete": every_arm_complete,
        "every_arm_authentic_fpr_at_most_0_01": authentic_fpr_pass,
        "explicit_9ch_minus_6ch_at_least_0_02": delta_9_minus_6_pass,
        "explicit_9ch_minus_fc_siam_diff_at_least_0_02": delta_9_minus_fc_pass,
        "at_least_one_required_ci_excludes_zero": required_ci_excludes_zero,
    }
    continuation_pass = all(gate_checks.values())
    arm_rows = []
    for arm in sorted(ARMS):
        summary = arm_summaries[arm]
        arm_rows.append(
            {
                "arm": arm,
                "validation_source_group_macro_pixel_ap": summary[
                    "validation_source_group_macro_pixel_ap_model_resolution"
                ],
                "validation_source_group_macro_forged_pixel_f1": summary[
                    "operating_point"
                ]["source_group_macro_forged_pixel_f1"],
                "validation_unique_authentic_group_macro_pixel_fpr": summary[
                    "operating_point"
                ]["unique_authentic_group_macro_pixel_fpr"],
                "best_epoch": summary["best_epoch"],
                "optimizer_steps": summary["optimizer_steps_completed"],
                "parameter_count": summary["parameter_count"],
                "paper_evidence": False,
            }
        )
    arm_table_path = _resolve(project_root, config["paths"]["arm_table"])
    contrast_table_path = _resolve(project_root, config["paths"]["contrast_table"])
    summary_path = _resolve(project_root, config["paths"]["summary"])
    _write_csv(arm_table_path, arm_rows)
    contrast_rows = [
        {"contrast": name, **values, "bootstrap_replicates": replicates}
        for name, values in contrasts.items()
    ]
    _write_csv(contrast_table_path, contrast_rows)
    result = {
        "status": "continuation_gate_passed" if continuation_pass else "continuation_gate_failed",
        "paper_evidence": False,
        "holdout_read": False,
        "validation_source_groups": group_count,
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "arms": {row["arm"]: row for row in arm_rows},
        "contrasts": contrasts,
        "gate_checks": gate_checks,
        "continuation_pass": continuation_pass,
        "decision": (
            "eligible_for_separately_frozen_holdout_evaluation"
            if continuation_pass
            else "do_not_open_holdout_report_negative_development_result"
        ),
        "protocol_sha256": _sha256(protocol_path),
        "arm_table_sha256": _sha256(arm_table_path),
        "contrast_table_sha256": _sha256(contrast_table_path),
    }
    _write_json(summary_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen TFR equal-budget arms")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
