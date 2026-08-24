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


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if runtime["gpu_launch_authorized"] or runtime["final_reserve_read_allowed"]:
        raise ValueError("trace-aux decision cannot use GPU or final reserve")
    if runtime["paper_evidence"] or config["experiment"]["paper_evidence"]:
        raise ValueError("trace-aux decision cannot be paper evidence")
    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("trace-aux decision protocol SHA-256 changed")
    inputs = config["inputs"]
    student_path = _resolve(project_root, inputs["matched_student_summary"])
    if _sha256(student_path) != inputs["matched_student_summary_sha256"]:
        raise ValueError("matched student summary changed")
    summaries = {
        "student": _read(student_path),
        "correct": _read(_resolve(project_root, inputs["correct_trace_summary"])),
        "shuffled": _read(_resolve(project_root, inputs["shuffled_trace_summary"])),
        "zero": _read(_resolve(project_root, inputs["zero_trace_summary"])),
    }
    for name, mode in (
        ("correct", "correct_trace"),
        ("shuffled", "shuffled_trace"),
        ("zero", "zero_trace"),
    ):
        if summaries[name]["target_mode"] != mode:
            raise ValueError(f"trace-aux {name} target mode changed")
        if summaries[name]["final_reserve_read"]:
            raise ValueError(f"trace-aux {name} opened the final reserve")
        if summaries[name]["protocol_sha256"] != config["experiment"][
            "expected_protocol_sha256"
        ]:
            raise ValueError(f"trace-aux {name} protocol hash changed")
    metrics = {
        name: value["validation_metrics_native_geometry"]
        for name, value in summaries.items()
    }
    success = config["success"]
    checks = {
        "all_runs_complete": bool(
            all(
                summaries[name]["train_pairs"]
                == summaries[name]["validation_pairs"]
                == 100
                for name in ("correct", "shuffled", "zero")
            )
        ),
        "correct_ap_pass": bool(
            metrics["correct"]["macro_pixel_ap"]
            >= float(success["correct_native_macro_pixel_ap_min"])
        ),
        "correct_iou_pass": bool(
            metrics["correct"]["pixel_iou"]
            >= float(success["correct_native_pixel_iou_min"])
        ),
        "correct_fpr_pass": bool(
            metrics["correct"]["authentic_pixel_fpr"]
            <= float(success["correct_authentic_pixel_fpr_max"]) + 1e-12
        ),
        "correct_minus_shuffled_pass": bool(
            metrics["correct"]["macro_pixel_ap"]
            - metrics["shuffled"]["macro_pixel_ap"]
            >= float(success["correct_minus_shuffled_macro_pixel_ap_min"])
        ),
        "correct_minus_zero_pass": bool(
            metrics["correct"]["macro_pixel_ap"] - metrics["zero"]["macro_pixel_ap"]
            >= float(success["correct_minus_zero_macro_pixel_ap_min"])
        ),
    }
    passed = all(checks.values())
    row = {
        "student_macro_pixel_ap": metrics["student"]["macro_pixel_ap"],
        "correct_macro_pixel_ap": metrics["correct"]["macro_pixel_ap"],
        "shuffled_macro_pixel_ap": metrics["shuffled"]["macro_pixel_ap"],
        "zero_macro_pixel_ap": metrics["zero"]["macro_pixel_ap"],
        "correct_minus_student_macro_pixel_ap": metrics["correct"]["macro_pixel_ap"]
        - metrics["student"]["macro_pixel_ap"],
        "correct_minus_shuffled_macro_pixel_ap": metrics["correct"]["macro_pixel_ap"]
        - metrics["shuffled"]["macro_pixel_ap"],
        "correct_minus_zero_macro_pixel_ap": metrics["correct"]["macro_pixel_ap"]
        - metrics["zero"]["macro_pixel_ap"],
        "student_pixel_iou": metrics["student"]["pixel_iou"],
        "correct_pixel_iou": metrics["correct"]["pixel_iou"],
        "correct_minus_student_pixel_iou": metrics["correct"]["pixel_iou"]
        - metrics["student"]["pixel_iou"],
        "correct_authentic_pixel_fpr": metrics["correct"]["authentic_pixel_fpr"],
        "decision_passed": passed,
        "paper_evidence": False,
    }
    metrics_path = _resolve(project_root, config["paths"]["metrics"])
    summary_path = _resolve(project_root, config["paths"]["summary"])
    _write_csv(metrics_path, [row])
    summary = {
        "experiment": config["experiment"],
        "status": "passed" if passed else "completed_success_criteria_not_met",
        "paper_evidence": False,
        "final_reserve_read": False,
        "checks": checks,
        "metrics": row,
        "inputs": {
            f"{name}_summary_sha256": _sha256(
                student_path
                if name == "student"
                else _resolve(project_root, inputs[f"{name}_trace_summary"])
            )
            for name in ("student", "correct", "shuffled", "zero")
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
