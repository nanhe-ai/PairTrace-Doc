from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from pairtrace_doc.pipelines.train_student_100 import (
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if runtime["gpu_launch_authorized"] or runtime["final_reserve_read_allowed"]:
        raise ValueError("PairTrace decision cannot use GPU or final reserve")
    if runtime["paper_evidence"] or config["experiment"]["paper_evidence"]:
        raise ValueError("PairTrace 100-pair decision cannot be paper evidence")
    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("PairTrace decision protocol SHA-256 changed")
    inputs = config["inputs"]
    student_path = _resolve(project_root, inputs["matched_student_summary"])
    if _sha256(student_path) != inputs["matched_student_summary_sha256"]:
        raise ValueError("matched-student decision input SHA-256 changed")
    student = _read_json(student_path)
    correct = _read_json(_resolve(project_root, inputs["correct_pair_summary"]))
    shuffled = _read_json(_resolve(project_root, inputs["shuffled_pair_summary"]))
    if correct["pair_mode"] != "correct_pair" or shuffled["pair_mode"] != "shuffled_pair":
        raise ValueError("PairTrace decision pair modes changed")
    expected_protocol = config["experiment"]["expected_protocol_sha256"]
    if correct["protocol_sha256"] != expected_protocol or shuffled["protocol_sha256"] != expected_protocol:
        raise ValueError("PairTrace run protocol hashes differ")
    if correct["final_reserve_read"] or shuffled["final_reserve_read"]:
        raise ValueError("a PairTrace run opened the final reserve")
    correct_metric = correct["validation_metrics_native_geometry"]
    shuffled_metric = shuffled["validation_metrics_native_geometry"]
    student_metric = student["validation_metrics_native_geometry"]
    success = config["success"]
    checks = {
        "all_runs_complete": bool(
            correct["train_pairs"] == correct["validation_pairs"] == 100
            and shuffled["train_pairs"] == shuffled["validation_pairs"] == 100
        ),
        "correct_ap_pass": bool(
            correct_metric["macro_pixel_ap"]
            >= float(success["correct_pair_native_macro_pixel_ap_min"])
        ),
        "correct_iou_pass": bool(
            correct_metric["pixel_iou"]
            >= float(success["correct_pair_native_pixel_iou_min"])
        ),
        "correct_authentic_fpr_pass": bool(
            correct_metric["authentic_pixel_fpr"]
            <= float(success["correct_pair_authentic_pixel_fpr_max"]) + 1e-12
        ),
        "correct_minus_shuffled_ap_pass": bool(
            correct_metric["macro_pixel_ap"] - shuffled_metric["macro_pixel_ap"]
            >= float(success["correct_minus_shuffled_native_macro_pixel_ap_min"])
        ),
    }
    passed = all(checks.values())
    metric_row = {
        "matched_student_macro_pixel_ap": student_metric["macro_pixel_ap"],
        "correct_pair_macro_pixel_ap": correct_metric["macro_pixel_ap"],
        "shuffled_pair_macro_pixel_ap": shuffled_metric["macro_pixel_ap"],
        "correct_minus_student_macro_pixel_ap": correct_metric["macro_pixel_ap"]
        - student_metric["macro_pixel_ap"],
        "correct_minus_shuffled_macro_pixel_ap": correct_metric["macro_pixel_ap"]
        - shuffled_metric["macro_pixel_ap"],
        "matched_student_pixel_iou": student_metric["pixel_iou"],
        "correct_pair_pixel_iou": correct_metric["pixel_iou"],
        "correct_minus_student_pixel_iou": correct_metric["pixel_iou"]
        - student_metric["pixel_iou"],
        "correct_pair_authentic_pixel_fpr": correct_metric["authentic_pixel_fpr"],
        "decision_passed": passed,
        "paper_evidence": False,
    }
    paths = config["paths"]
    metrics_path = _resolve(project_root, paths["metrics"])
    summary_path = _resolve(project_root, paths["summary"])
    _write_csv(metrics_path, [metric_row])
    summary = {
        "experiment": config["experiment"],
        "status": "passed" if passed else "completed_success_criteria_not_met",
        "paper_evidence": False,
        "final_reserve_read": False,
        "checks": checks,
        "metrics": metric_row,
        "inputs": {
            "matched_student_summary_sha256": _sha256(student_path),
            "correct_pair_summary_sha256": _sha256(
                _resolve(project_root, inputs["correct_pair_summary"])
            ),
            "shuffled_pair_summary_sha256": _sha256(
                _resolve(project_root, inputs["shuffled_pair_summary"])
            ),
        },
        "outputs": {
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
