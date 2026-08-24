from __future__ import annotations

import argparse
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


def _stratified_paired_bootstrap(
    left: dict[str, tuple[str, float]],
    right: dict[str, tuple[str, float]],
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired bootstrap group sets differ")
    by_generator: dict[str, list[float]] = {}
    for group in sorted(left):
        left_generator, left_value = left[group]
        right_generator, right_value = right[group]
        if left_generator != right_generator:
            raise ValueError("paired bootstrap generator labels differ")
        by_generator.setdefault(left_generator, []).append(left_value - right_value)
    rng = np.random.default_rng(seed)
    generator_replicates: list[np.ndarray] = []
    per_generator: dict[str, float] = {}
    for generator, values in sorted(by_generator.items()):
        differences = np.asarray(values, dtype=float)
        indices = rng.integers(
            0, len(differences), size=(resamples, len(differences))
        )
        generator_replicates.append(differences[indices].mean(axis=1))
        per_generator[generator] = float(differences.mean())
    replicates = np.stack(generator_replicates).mean(axis=0)
    effect = float(np.mean(list(per_generator.values())))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "effect": effect,
        "ci_low": float(np.quantile(replicates, alpha)),
        "ci_high": float(np.quantile(replicates, 1.0 - alpha)),
        "per_generator_effect": per_generator,
    }


def _generator_macro(
    predictions: dict[str, tuple[str, float]]
) -> tuple[float, dict[str, float]]:
    by_generator: dict[str, list[float]] = {}
    for generator, value in predictions.values():
        by_generator.setdefault(generator, []).append(value)
    per_generator = {
        generator: float(np.mean(values))
        for generator, values in sorted(by_generator.items())
    }
    return float(np.mean(list(per_generator.values()))), per_generator


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
            "multi_seed_authorized",
            "viewed_diagnostic_read_allowed",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("comparison cannot authorize compute or holdout access")
    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("generator-balanced protocol SHA-256 changed")

    expected_groups = int(runtime["expected_development_groups"])
    summaries: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, tuple[str, float]]] = {}
    method_rows: list[dict[str, Any]] = []
    reference_groups: set[str] | None = None
    for name, specification in config["methods"].items():
        summary_path = _resolve(project_root, specification["summary"])
        digest = _sha256(summary_path)
        if digest != specification["expected_summary_sha256"]:
            raise ValueError(f"{name} summary SHA-256 changed")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("paper_evidence"):
            raise ValueError("development result cannot already be paper evidence")
        if summary.get("final_reserve_read") or summary.get(
            "output_unseen_holdout_read"
        ):
            raise ValueError(f"{name} crossed a frozen evidence boundary")
        if summary["input_manifest_sha256"] != config["input"][
            "expected_manifest_sha256"
        ]:
            raise ValueError(f"{name} input manifest changed")
        prediction_path = _resolve(project_root, summary["outputs"]["predictions"])
        if _sha256(prediction_path) != summary["outputs"]["predictions_sha256"]:
            raise ValueError(f"{name} prediction SHA-256 changed")
        all_rows = _read_jsonl(prediction_path)
        if any(row.get("status") != "ok" for row in all_rows):
            raise ValueError(f"{name} contains a failed prediction")
        forged = {
            str(row["source_group_id"]): (
                str(row["generator"]),
                float(row["macro_pixel_ap"]),
            )
            for row in all_rows
            if row["sample_kind"] == "forged"
        }
        if len(forged) != expected_groups or len(all_rows) != expected_groups * 2:
            raise ValueError(f"{name} prediction count changed")
        if reference_groups is None:
            reference_groups = set(forged)
        elif set(forged) != reference_groups:
            raise ValueError(f"{name} development group set changed")
        generator_macro, per_generator = _generator_macro(forged)
        metric = summary["validation_metrics_native_geometry"]
        if not np.isclose(
            generator_macro,
            float(metric["generator_macro_pixel_ap"]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"{name} generator-macro AP cannot be reproduced")
        method_rows.append(
            {
                "method": name,
                "generator_macro_pixel_ap": generator_macro,
                "macro_pixel_ap": float(metric["macro_pixel_ap"]),
                "pixel_iou": float(metric["pixel_iou"]),
                "authentic_pixel_fpr": float(metric["authentic_pixel_fpr"]),
                "qwen_inpaint_ap": per_generator["qwen-inpaint"],
                "gemini_nano_ap": per_generator["gemini-nano"],
                "openai_gpt_image_2_ap": per_generator["openai-gpt-image-2"],
                "development_groups": len(forged),
                "paper_evidence": False,
            }
        )
        summaries[name] = summary
        predictions[name] = forged

    bootstrap = config["bootstrap"]
    comparison_rows: list[dict[str, Any]] = []
    comparison_results: dict[str, dict[str, Any]] = {}
    for offset, (name, pair) in enumerate(config["comparisons"].items()):
        left, right = pair
        result = _stratified_paired_bootstrap(
            predictions[left],
            predictions[right],
            int(bootstrap["seed"]) + offset,
            int(bootstrap["resamples"]),
            float(bootstrap["confidence_level"]),
        )
        comparison_results[name] = result
        row: dict[str, Any] = {
            "comparison": name,
            "left_method": left,
            "right_method": right,
            "generator_macro_pixel_ap_difference": result["effect"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "confidence_level": float(bootstrap["confidence_level"]),
            "bootstrap_resamples": int(bootstrap["resamples"]),
            "bootstrap_seed": int(bootstrap["seed"]) + offset,
            "unit": "source_group_id_stratified_by_generator",
            "paper_evidence": False,
        }
        for generator, value in result["per_generator_effect"].items():
            safe = "".join(
                character if character.isalnum() else "_" for character in generator
            ).strip("_")
            row[f"difference__{safe}"] = value
        comparison_rows.append(row)

    methods_by_name = {row["method"]: row for row in method_rows}
    gates = config["promotion_gate"]
    correct_student = comparison_results["correct_minus_student"]
    correct_shuffled = comparison_results["correct_minus_shuffled"]
    correct = methods_by_name["correct_relation"]
    student = methods_by_name["student"]
    checks = {
        "correct_minus_student_effect_floor": correct_student["effect"]
        >= float(gates["correct_minus_student_generator_macro_ap_min"]),
        "correct_minus_shuffled_effect_floor": correct_shuffled["effect"]
        >= float(gates["correct_minus_shuffled_generator_macro_ap_min"]),
        "correct_minus_student_direction": correct_student["effect"] > 0.0
        and correct_student["ci_high"] > 0.0,
        "correct_iou_not_below_student": correct["pixel_iou"]
        >= student["pixel_iou"] - 1e-12,
        "correct_authentic_fpr_ceiling": correct["authentic_pixel_fpr"]
        <= float(gates["authentic_pixel_fpr_max"]) + 1e-12,
        "all_predictions_complete": True,
    }
    overall_pass = all(checks.values())

    method_path = _resolve(project_root, config["paths"]["method_table"])
    comparison_path = _resolve(project_root, config["paths"]["comparison_table"])
    summary_path = _resolve(project_root, config["paths"]["summary"])
    _write_csv(method_path, method_rows)
    _write_csv(comparison_path, comparison_rows)
    output = {
        "experiment": config["experiment"],
        "status": "passed_promotion_gate"
        if overall_pass
        else "completed_success_criteria_not_met",
        "paper_evidence": False,
        "multi_seed_authorized": overall_pass,
        "final_reserve_read": False,
        "viewed_diagnostic_read": False,
        "development_groups": len(reference_groups or set()),
        "bootstrap": bootstrap,
        "methods": method_rows,
        "comparisons": comparison_rows,
        "promotion_gate": gates,
        "checks": checks,
        "overall_pass": overall_pass,
        "outputs": {
            "method_table": str(method_path.relative_to(project_root)),
            "method_table_sha256": _sha256(method_path),
            "comparison_table": str(comparison_path.relative_to(project_root)),
            "comparison_table_sha256": _sha256(comparison_path),
        },
    }
    _write_json(summary_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
