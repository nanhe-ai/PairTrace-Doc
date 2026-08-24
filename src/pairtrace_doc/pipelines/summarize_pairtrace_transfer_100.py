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


def _paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("paired bootstrap group sets differ")
    groups = sorted(left)
    differences = np.asarray([left[group] - right[group] for group in groups])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(resamples, len(groups)))
    replicates = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "effect": float(differences.mean()),
        "ci_low": float(np.quantile(replicates, alpha)),
        "ci_high": float(np.quantile(replicates, 1.0 - alpha)),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if runtime["gpu_launch_authorized"]:
        raise ValueError("transfer summary cannot launch GPU work")
    if runtime["viewed_diagnostic_read_allowed"] or runtime[
        "final_reserve_read_allowed"
    ]:
        raise ValueError("transfer summary cannot read diagnostic or reserve data")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("development transfer summary cannot be paper evidence")

    summaries: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, float]] = {}
    summary_hashes: dict[str, str] = {}
    method_rows: list[dict[str, Any]] = []
    expected_groups = int(runtime["expected_groups"])
    reference_groups: set[str] | None = None
    for name, specification in config["methods"].items():
        summary_path = _resolve(project_root, specification["summary"])
        digest = _sha256(summary_path)
        if digest != specification["expected_summary_sha256"]:
            raise ValueError(f"{name} summary SHA-256 changed")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        prediction_path = _resolve(project_root, summary["outputs"]["predictions"])
        if _sha256(prediction_path) != summary["outputs"]["predictions_sha256"]:
            raise ValueError(f"{name} prediction SHA-256 changed")
        forged = {
            str(row["source_group_id"]): float(row["macro_pixel_ap"])
            for row in _read_jsonl(prediction_path)
            if row["sample_kind"] == "forged" and row["status"] == "ok"
        }
        if runtime["require_all_groups"] and len(forged) != expected_groups:
            raise ValueError(f"{name} forged prediction group count changed")
        if reference_groups is None:
            reference_groups = set(forged)
        elif set(forged) != reference_groups:
            raise ValueError(f"{name} validation group set changed")
        metric = summary["validation_metrics_native_geometry"]
        method_rows.append(
            {
                "method": name,
                "macro_pixel_ap": float(metric["macro_pixel_ap"]),
                "pixel_iou": float(metric["pixel_iou"]),
                "authentic_pixel_fpr": float(metric["authentic_pixel_fpr"]),
                "validation_groups": len(forged),
                "paper_evidence": False,
            }
        )
        summaries[name] = summary
        predictions[name] = forged
        summary_hashes[name] = digest

    bootstrap = config["bootstrap"]
    comparison_rows: list[dict[str, Any]] = []
    for offset, (name, pair) in enumerate(config["comparisons"].items()):
        left, right = pair
        result = _paired_bootstrap(
            predictions[left],
            predictions[right],
            int(bootstrap["seed"]) + offset,
            int(bootstrap["resamples"]),
            float(bootstrap["confidence_level"]),
        )
        comparison_rows.append(
            {
                "comparison": name,
                "left_method": left,
                "right_method": right,
                "macro_pixel_ap_difference": result["effect"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "confidence_level": float(bootstrap["confidence_level"]),
                "bootstrap_resamples": int(bootstrap["resamples"]),
                "bootstrap_seed": int(bootstrap["seed"]) + offset,
                "unit": bootstrap["unit"],
                "paper_evidence": False,
            }
        )

    method_path = _resolve(project_root, config["paths"]["method_table"])
    comparison_path = _resolve(project_root, config["paths"]["comparison_table"])
    summary_path = _resolve(project_root, config["paths"]["summary"])
    _write_csv(method_path, method_rows)
    _write_csv(comparison_path, comparison_rows)
    output = {
        "experiment": config["experiment"],
        "status": "completed",
        "paper_evidence": False,
        "viewed_diagnostic_read": False,
        "final_reserve_read": False,
        "validation_groups": len(reference_groups or set()),
        "bootstrap": bootstrap,
        "method_summary_sha256": summary_hashes,
        "methods": method_rows,
        "comparisons": comparison_rows,
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
