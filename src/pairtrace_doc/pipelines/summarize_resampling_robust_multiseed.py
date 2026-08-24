from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
)


AGGREGATE_FIELDS = (
    "aiforge_clean_generator_macro_pixel_ap",
    "aiforge_minimum_stressed_generator_macro_pixel_ap",
    "aiforge_minimum_stressed_gain_over_clean_teacher",
    "aiforge_maximum_authentic_pixel_fpr",
    "aiforge_clean_frozen_threshold",
    "fantasyid_correct_attack_device_macro_box_mask_ap",
    "fantasyid_correct_minus_student_effect",
    "fantasyid_correct_minus_shuffled_effect",
    "fantasyid_correct_authentic_pixel_fpr",
)


def _descriptive_statistics(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("multi-seed aggregation requires at least two values")
    minimum = min(values)
    maximum = max(values)
    return {
        "mean": mean(values),
        "sample_standard_deviation": stdev(values),
        "minimum": minimum,
        "maximum": maximum,
        "range": maximum - minimum,
    }


def _stability_decision(
    seed_rows: list[dict[str, Any]], gates: dict[str, Any]
) -> dict[str, Any]:
    individual: dict[str, dict[str, bool]] = {}
    for row in seed_rows:
        seed = str(row["training_seed"])
        individual[seed] = {
            "training_complete": bool(row["training_complete"]),
            "aiforge_evaluation_complete": bool(row["aiforge_evaluation_complete"]),
            "fantasyid_evaluation_complete": bool(row["fantasyid_evaluation_complete"]),
            "aiforge_clean_ap_floor": row[
                "aiforge_clean_generator_macro_pixel_ap"
            ]
            >= float(gates["aiforge_clean_generator_macro_pixel_ap_min"]),
            "aiforge_minimum_stressed_ap_floor": row[
                "aiforge_minimum_stressed_generator_macro_pixel_ap"
            ]
            >= float(
                gates["aiforge_minimum_stressed_generator_macro_pixel_ap_min"]
            ),
            "aiforge_minimum_stressed_gain_floor": row[
                "aiforge_minimum_stressed_gain_over_clean_teacher"
            ]
            >= float(
                gates[
                    "aiforge_minimum_stressed_gain_over_clean_teacher_min"
                ]
            ),
            "aiforge_authentic_fpr_ceiling": row[
                "aiforge_maximum_authentic_pixel_fpr"
            ]
            <= float(gates["authentic_pixel_fpr_max"]),
            "fantasyid_correct_ap_floor": row[
                "fantasyid_correct_attack_device_macro_box_mask_ap"
            ]
            >= float(
                gates["fantasyid_correct_attack_device_macro_box_mask_ap_min"]
            ),
            "fantasyid_minus_student_floor": row[
                "fantasyid_correct_minus_student_effect"
            ]
            >= float(gates["fantasyid_correct_minus_student_effect_min"]),
            "fantasyid_minus_shuffled_floor": row[
                "fantasyid_correct_minus_shuffled_effect"
            ]
            >= float(gates["fantasyid_correct_minus_shuffled_effect_min"]),
            "fantasyid_authentic_fpr_ceiling": row[
                "fantasyid_correct_authentic_pixel_fpr"
            ]
            <= float(gates["authentic_pixel_fpr_max"]),
        }

    aiforge_values = [
        float(row["aiforge_minimum_stressed_generator_macro_pixel_ap"])
        for row in seed_rows
    ]
    fantasyid_values = [
        float(row["fantasyid_correct_attack_device_macro_box_mask_ap"])
        for row in seed_rows
    ]
    aiforge_statistics = _descriptive_statistics(aiforge_values)
    fantasyid_statistics = _descriptive_statistics(fantasyid_values)
    across_seed = {
        "aiforge_minimum_stressed_ap_sample_std_ceiling": aiforge_statistics[
            "sample_standard_deviation"
        ]
        <= float(gates["aiforge_minimum_stressed_ap_sample_std_max"]),
        "aiforge_minimum_stressed_ap_range_ceiling": aiforge_statistics["range"]
        <= float(gates["aiforge_minimum_stressed_ap_range_max"]),
        "fantasyid_correct_ap_sample_std_ceiling": fantasyid_statistics[
            "sample_standard_deviation"
        ]
        <= float(gates["fantasyid_correct_ap_sample_std_max"]),
        "fantasyid_correct_ap_range_ceiling": fantasyid_statistics["range"]
        <= float(gates["fantasyid_correct_ap_range_max"]),
    }
    overall_pass = all(
        value for checks in individual.values() for value in checks.values()
    ) and all(across_seed.values())
    return {
        "individual_seed_checks": individual,
        "across_seed_checks": across_seed,
        "overall_pass": overall_pass,
    }


def _read_bound_summary(
    project_root: Path, specification: dict[str, Any], label: str
) -> tuple[dict[str, Any], str]:
    path = _resolve(project_root, specification["path"])
    digest = _sha256(path)
    if digest != specification["sha256"]:
        raise ValueError(f"{label} summary SHA-256 changed")
    return json.loads(path.read_text(encoding="utf-8")), digest


def _validate_prediction_artifact(
    project_root: Path,
    summary: dict[str, Any],
    expected_records: int,
    label: str,
) -> None:
    outputs = summary["outputs"]
    path = _resolve(project_root, outputs["predictions"])
    if _sha256(path) != outputs["predictions_sha256"]:
        raise ValueError(f"{label} prediction SHA-256 changed")
    rows = _read_jsonl(path)
    if len(rows) != expected_records:
        raise ValueError(f"{label} prediction count changed")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError(f"{label} contains failed predictions")
    if int(summary["successful_prediction_records"]) != expected_records:
        raise ValueError(f"{label} successful prediction count changed")
    if int(summary["failed_prediction_records"]) != 0:
        raise ValueError(f"{label} reports failed predictions")


def _validate_training_summary(
    project_root: Path,
    summary: dict[str, Any],
    specification: dict[str, Any],
    seed: int,
) -> None:
    if summary["status"] != "resampling_robust_teacher_training_complete":
        raise ValueError(f"seed {seed} training is incomplete")
    if int(summary["experiment"]["seed"]) != seed:
        raise ValueError(f"seed {seed} training identity changed")
    if summary.get("paper_evidence") or summary.get("final_reserve_read"):
        raise ValueError(f"seed {seed} training crossed an evidence boundary")
    if summary.get("checkpoint_selection_used"):
        raise ValueError(f"seed {seed} used checkpoint selection")
    if int(summary["fixed_final_epoch"]) != 6 or len(summary["epochs"]) != 6:
        raise ValueError(f"seed {seed} fixed training schedule changed")
    if int(summary["train_pairs"]) != 1000 or int(summary["pair_cache_hits"]) != 1000:
        raise ValueError(f"seed {seed} training data completeness changed")
    if summary["checkpoint_sha256"] != specification["checkpoint_sha256"]:
        raise ValueError(f"seed {seed} checkpoint identity changed")
    checkpoint_path = _resolve(project_root, summary["checkpoint"])
    if _sha256(checkpoint_path) != specification["checkpoint_sha256"]:
        raise ValueError(f"seed {seed} checkpoint SHA-256 changed")
    epoch_log = _resolve(project_root, summary["outputs"]["epoch_log"])
    if _sha256(epoch_log) != summary["outputs"]["epoch_log_sha256"]:
        raise ValueError(f"seed {seed} epoch log SHA-256 changed")


def _validate_development_summary(
    project_root: Path,
    summary: dict[str, Any],
    expected_records: int,
    expected_status: str,
    label: str,
) -> None:
    if summary["status"] != expected_status:
        raise ValueError(f"{label} did not pass its frozen gate")
    if summary.get("paper_evidence") or summary.get("final_reserve_read"):
        raise ValueError(f"{label} crossed an evidence boundary")
    if not summary["decision"]["overall_pass"]:
        raise ValueError(f"{label} frozen decision is negative")
    _validate_prediction_artifact(project_root, summary, expected_records, label)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if any(
        bool(runtime.get(key))
        for key in (
            "gpu_launch_authorized",
            "model_training_authorized",
            "method_change_authorized",
            "selected_image_read_allowed",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("multi-seed aggregation must be artifact-only")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("development stability aggregation cannot be paper evidence")
    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("multi-seed protocol SHA-256 changed")

    log_path = _resolve(project_root, config["paths"]["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    expected_seeds = [int(value) for value in config["family_seeds"]]
    if expected_seeds != [20260747, 20260763, 20260764]:
        raise ValueError("multi-seed family changed")
    seed_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    for seed in expected_seeds:
        specification = config["seeds"][str(seed)]
        training, training_hash = _read_bound_summary(
            project_root, specification["training_summary"], f"seed {seed} training"
        )
        aiforge, aiforge_hash = _read_bound_summary(
            project_root, specification["aiforge_summary"], f"seed {seed} AIForge"
        )
        fantasyid, fantasyid_hash = _read_bound_summary(
            project_root, specification["fantasyid_summary"], f"seed {seed} FantasyID"
        )
        _validate_training_summary(project_root, training, specification, seed)
        _validate_development_summary(
            project_root,
            aiforge,
            int(runtime["expected_aiforge_prediction_records"]),
            str(specification["aiforge_expected_status"]),
            f"seed {seed} AIForge",
        )
        _validate_development_summary(
            project_root,
            fantasyid,
            int(runtime["expected_fantasyid_prediction_records"]),
            "passed_external_development_gate",
            f"seed {seed} FantasyID",
        )
        if aiforge["model_checkpoint_sha256"]["robust_1000"] != specification[
            "checkpoint_sha256"
        ]:
            raise ValueError(f"seed {seed} AIForge checkpoint changed")
        if fantasyid["metrics"]["robust_teacher_correct"][
            "fixed_pixel_threshold"
        ] != aiforge["conditions"]["robust_1000__clean"]["pixel_threshold"]:
            raise ValueError(f"seed {seed} FantasyID threshold was not frozen on AIForge")
        aiforge_model = aiforge["decision"]["model_summaries"]["robust_1000"]
        row = {
            "training_seed": seed,
            "checkpoint_sha256": specification["checkpoint_sha256"],
            "training_complete": True,
            "aiforge_evaluation_complete": True,
            "fantasyid_evaluation_complete": True,
            "aiforge_clean_generator_macro_pixel_ap": float(
                aiforge_model["clean_generator_macro_pixel_ap"]
            ),
            "aiforge_minimum_stressed_generator_macro_pixel_ap": float(
                aiforge_model["minimum_stressed_generator_macro_pixel_ap"]
            ),
            "aiforge_minimum_stressed_gain_over_clean_teacher": float(
                aiforge["decision"]["minimum_stressed_gain_over_baseline"]
            ),
            "aiforge_maximum_authentic_pixel_fpr": float(
                aiforge_model["maximum_authentic_pixel_fpr"]
            ),
            "aiforge_clean_frozen_threshold": float(
                aiforge["conditions"]["robust_1000__clean"]["pixel_threshold"]
            ),
            "fantasyid_correct_attack_device_macro_box_mask_ap": float(
                fantasyid["metrics"]["robust_teacher_correct"][
                    "attack_device_macro_box_mask_ap"
                ]
            ),
            "fantasyid_correct_minus_student_effect": float(
                fantasyid["comparisons"]["robust_minus_student"]["effect"]
            ),
            "fantasyid_correct_minus_student_ci_low": float(
                fantasyid["comparisons"]["robust_minus_student"]["ci_low"]
            ),
            "fantasyid_correct_minus_student_ci_high": float(
                fantasyid["comparisons"]["robust_minus_student"]["ci_high"]
            ),
            "fantasyid_correct_minus_shuffled_effect": float(
                fantasyid["comparisons"]["robust_minus_shuffled"]["effect"]
            ),
            "fantasyid_correct_minus_shuffled_ci_low": float(
                fantasyid["comparisons"]["robust_minus_shuffled"]["ci_low"]
            ),
            "fantasyid_correct_minus_shuffled_ci_high": float(
                fantasyid["comparisons"]["robust_minus_shuffled"]["ci_high"]
            ),
            "fantasyid_correct_authentic_pixel_fpr": float(
                fantasyid["metrics"]["robust_teacher_correct"][
                    "authentic_document_macro_pixel_fpr"
                ]
            ),
            "paper_evidence": False,
            "final_reserve_read": False,
        }
        seed_rows.append(row)
        input_hashes[str(seed)] = {
            "training_summary_sha256": training_hash,
            "aiforge_summary_sha256": aiforge_hash,
            "fantasyid_summary_sha256": fantasyid_hash,
        }

    aggregate_rows = []
    aggregate_statistics: dict[str, dict[str, float]] = {}
    for field in AGGREGATE_FIELDS:
        statistics = _descriptive_statistics(
            [float(row[field]) for row in seed_rows]
        )
        aggregate_statistics[field] = statistics
        aggregate_rows.append(
            {
                "metric": field,
                **statistics,
                "seed_count": len(seed_rows),
                "sample_standard_deviation_ddof": 1,
                "paper_evidence": False,
            }
        )

    decision = _stability_decision(seed_rows, config["stability_gate"])
    seed_table = _resolve(project_root, config["paths"]["seed_table"])
    aggregate_table = _resolve(project_root, config["paths"]["aggregate_table"])
    summary_path = _resolve(project_root, config["paths"]["summary"])
    _write_csv(seed_table, seed_rows)
    _write_csv(aggregate_table, aggregate_rows)
    output = {
        "experiment": config["experiment"],
        "status": (
            "multiseed_stability_gate_passed"
            if decision["overall_pass"]
            else "multiseed_stability_gate_failed"
        ),
        "paper_evidence": False,
        "development_only": True,
        "mask_semantics": "fantasyid_box_mask_not_pixel_accurate",
        "family_seeds": expected_seeds,
        "seed_count": len(seed_rows),
        "seed_results": seed_rows,
        "aggregate_statistics": aggregate_statistics,
        "stability_gate": config["stability_gate"],
        "decision": decision,
        "input_summary_sha256": input_hashes,
        "original_confirmatory_gate_reopened": False,
        "final_reserve_read": False,
        "final_reserve_read_authorized": False,
        "final_evaluation_protocol_freeze_authorized": bool(
            decision["overall_pass"]
        ),
        "outputs": {
            "seed_table": str(seed_table.relative_to(project_root)),
            "seed_table_sha256": _sha256(seed_table),
            "aggregate_table": str(aggregate_table.relative_to(project_root)),
            "aggregate_table_sha256": _sha256(aggregate_table),
            "log": str(log_path.relative_to(project_root)),
        },
    }
    _write_json(summary_path, output)
    logging.info(
        "status=%s seeds=%s final_reserve_read=false",
        output["status"],
        expected_seeds,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
