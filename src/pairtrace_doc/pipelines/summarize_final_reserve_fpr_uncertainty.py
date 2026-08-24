from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist, stdev
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


def _bootstrap_mean_interval(
    values: np.ndarray,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    replicates = values[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(values.mean()),
        float(np.quantile(replicates, alpha)),
        float(np.quantile(replicates, 1.0 - alpha)),
    )


def _wilson_interval(
    successes: int, total: int, confidence_level: float
) -> tuple[float, float, float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("invalid Wilson interval counts")
    probability = successes / total
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return probability, float(max(0.0, center - radius)), float(min(1.0, center + radius))


def _report(summary: dict[str, Any]) -> str:
    rows = []
    for item in summary["robust_family"]:
        rows.append(
            f"| {item['suffix']} | {item['pixel_fpr_mean']:.6f} | "
            f"[{item['pixel_fpr_ci_low']:.6f}, {item['pixel_fpr_ci_high']:.6f}] | "
            f"{item['pixel_fpr_seed_sample_sd']:.6f} | "
            f"{item['image_fpr_mean']:.6f} | "
            f"[{item['image_fpr_ci_low']:.6f}, {item['image_fpr_ci_high']:.6f}] |"
        )
    return f"""# Final-reserve false-positive uncertainty addendum

Status: `{summary['status']}`. This cache-only analysis retains all 2,496
authentic prediction records from the already consumed 96-group reserve. It
does not restore an unseen evidence boundary and performs no model, threshold,
condition, or subgroup selection.

## Three-seed robust-family false positives

| Condition suffix | Pixel FPR mean | 95% group-bootstrap interval | Seed sample SD | Image FPR mean | 95% group-bootstrap interval |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Pixel intervals use 5,000 source-group bootstrap resamples after averaging the
three fixed seeds within group. They describe group sampling for this frozen
seed family, not uncertainty over arbitrary training seeds. The complete
condition table additionally uses Wilson intervals for binary image FPR. The
subgroup table retains CORD, SROIE, WildReceipt, and XFUND separately; SROIE
has only three groups, so its interval must be interpreted as low precision.
"""


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
            "image_read_allowed",
            "mask_read_allowed",
            "score_cache_read_allowed",
            "model_training_authorized",
            "checkpoint_selection_authorized",
            "threshold_selection_authorized",
        )
    ):
        raise ValueError("FPR addendum crossed its cache-only evidence boundary")
    protocol_path = _resolve(project_root, str(config["experiment"]["protocol"]))
    if _sha256(protocol_path) != str(config["experiment"]["expected_protocol_sha256"]):
        raise ValueError("FPR uncertainty protocol SHA-256 changed")
    input_config = config["input"]
    predictions_path = _resolve(project_root, str(input_config["predictions"]))
    if _sha256(predictions_path) != str(input_config["expected_predictions_sha256"]):
        raise ValueError("final-reserve prediction SHA-256 changed")
    all_rows = _read_jsonl(predictions_path)
    rows = [
        row
        for row in all_rows
        if row.get("sample_kind") == input_config["required_sample_kind"]
    ]
    if len(rows) != int(input_config["expected_authentic_records"]):
        raise ValueError("final authentic prediction count changed")
    if any(row.get("status") != input_config["required_status"] for row in rows):
        raise ValueError("final authentic prediction contains failure")
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise ValueError("final authentic predictions contain duplicate record IDs")
    conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        conditions[str(row["condition"])].append(row)
    if len(conditions) != int(input_config["expected_conditions"]):
        raise ValueError("final authentic condition count changed")
    expected_groups = int(input_config["expected_groups"])
    if any(
        len(items) != expected_groups
        or len({str(item["source_group_id"]) for item in items}) != expected_groups
        for items in conditions.values()
    ):
        raise ValueError("final authentic condition group coverage changed")

    statistics = config["statistics"]
    bootstrap_seed = int(statistics["bootstrap_seed"])
    resamples = int(statistics["bootstrap_resamples"])
    confidence = float(statistics["confidence_level"])
    condition_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    condition_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for condition_index, (condition, items) in enumerate(sorted(conditions.items())):
        ordered = sorted(items, key=lambda item: str(item["source_group_id"]))
        pixel_thresholds = {float(item["fixed_pixel_threshold"]) for item in ordered}
        image_thresholds = {float(item["fixed_image_threshold"]) for item in ordered}
        if len(pixel_thresholds) != 1 or len(image_thresholds) != 1:
            raise ValueError(f"embedded thresholds vary within condition: {condition}")
        pixel_values = np.asarray(
            [float(item["authentic_pixel_fpr"]) for item in ordered], dtype=float
        )
        image_values = np.asarray(
            [
                float(item["image_score"]) >= float(item["fixed_image_threshold"])
                for item in ordered
            ],
            dtype=bool,
        )
        if (
            not np.isfinite(pixel_values).all()
            or np.any(pixel_values < 0.0)
            or np.any(pixel_values > 1.0)
        ):
            raise ValueError(f"invalid authentic pixel FPR: {condition}")
        pixel_mean, pixel_low, pixel_high = _bootstrap_mean_interval(
            pixel_values,
            bootstrap_seed + condition_index,
            resamples,
            confidence,
        )
        image_mean, image_low, image_high = _wilson_interval(
            int(image_values.sum()), len(image_values), confidence
        )
        condition_rows.append(
            {
                "condition": condition,
                "groups": len(ordered),
                "fixed_pixel_threshold": next(iter(pixel_thresholds)),
                "fixed_image_threshold": next(iter(image_thresholds)),
                "pixel_fpr_mean": pixel_mean,
                "pixel_fpr_ci_low": pixel_low,
                "pixel_fpr_ci_high": pixel_high,
                "image_false_positives": int(image_values.sum()),
                "image_fpr_mean": image_mean,
                "image_fpr_ci_low": image_low,
                "image_fpr_ci_high": image_high,
                "confidence_level": confidence,
                "pixel_interval": "source_group_percentile_bootstrap",
                "image_interval": "wilson_binomial",
                "paper_evidence": True,
            }
        )
        condition_maps[condition] = {
            str(item["source_group_id"]): item for item in ordered
        }
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in ordered:
            by_dataset[str(item["source_dataset"])].append(item)
        for subgroup_index, (dataset, selected) in enumerate(sorted(by_dataset.items())):
            values = np.asarray(
                [float(item["authentic_pixel_fpr"]) for item in selected], dtype=float
            )
            mean, low, high = _bootstrap_mean_interval(
                values,
                bootstrap_seed + 10000 + condition_index * 10 + subgroup_index,
                resamples,
                confidence,
            )
            binary = np.asarray(
                [
                    float(item["image_score"]) >= float(item["fixed_image_threshold"])
                    for item in selected
                ],
                dtype=bool,
            )
            image_mean, image_low, image_high = _wilson_interval(
                int(binary.sum()), len(binary), confidence
            )
            subgroup_rows.append(
                {
                    "condition": condition,
                    "source_dataset": dataset,
                    "groups": len(selected),
                    "pixel_fpr_mean": mean,
                    "pixel_fpr_ci_low": low,
                    "pixel_fpr_ci_high": high,
                    "image_false_positives": int(binary.sum()),
                    "image_fpr_mean": image_mean,
                    "image_fpr_ci_low": image_low,
                    "image_fpr_ci_high": image_high,
                    "confidence_level": confidence,
                    "paper_evidence": True,
                }
            )

    robust_seeds = [int(seed) for seed in statistics["robust_seeds"]]
    robust_rows: list[dict[str, Any]] = []
    for suffix_index, suffix in enumerate(statistics["robust_suffixes"]):
        names = [f"robust_{seed}_{suffix}" for seed in robust_seeds]
        if any(name not in condition_maps for name in names):
            raise ValueError(f"missing robust FPR condition family: {suffix}")
        groups = sorted(condition_maps[names[0]])
        if any(set(condition_maps[name]) != set(groups) for name in names):
            raise ValueError(f"robust seed group sets differ: {suffix}")
        group_pixel = []
        group_image = []
        seed_pixel_means = []
        for name in names:
            seed_pixel_means.append(
                float(
                    np.mean(
                        [
                            float(condition_maps[name][group]["authentic_pixel_fpr"])
                            for group in groups
                        ]
                    )
                )
            )
        for group in groups:
            items = [condition_maps[name][group] for name in names]
            group_pixel.append(
                float(np.mean([float(item["authentic_pixel_fpr"]) for item in items]))
            )
            group_image.append(
                float(
                    np.mean(
                        [
                            float(item["image_score"])
                            >= float(item["fixed_image_threshold"])
                            for item in items
                        ]
                    )
                )
            )
        pixel_mean, pixel_low, pixel_high = _bootstrap_mean_interval(
            np.asarray(group_pixel),
            bootstrap_seed + 20000 + suffix_index,
            resamples,
            confidence,
        )
        image_mean, image_low, image_high = _bootstrap_mean_interval(
            np.asarray(group_image),
            bootstrap_seed + 21000 + suffix_index,
            resamples,
            confidence,
        )
        robust_rows.append(
            {
                "suffix": suffix,
                "groups": len(groups),
                "seeds": len(robust_seeds),
                "pixel_fpr_mean": pixel_mean,
                "pixel_fpr_ci_low": pixel_low,
                "pixel_fpr_ci_high": pixel_high,
                "pixel_fpr_seed_sample_sd": stdev(seed_pixel_means),
                "image_fpr_mean": image_mean,
                "image_fpr_ci_low": image_low,
                "image_fpr_ci_high": image_high,
                "confidence_level": confidence,
                "interval_unit": "source_group_after_within_group_seed_mean",
                "paper_evidence": True,
            }
        )

    paths = config["paths"]
    condition_path = _resolve(project_root, str(paths["condition_table"]))
    robust_path = _resolve(project_root, str(paths["robust_family_table"]))
    subgroup_path = _resolve(project_root, str(paths["subgroup_table"]))
    summary_path = _resolve(project_root, str(paths["summary"]))
    report_path = _resolve(project_root, str(paths["report"]))
    _write_csv(condition_path, condition_rows)
    _write_csv(robust_path, robust_rows)
    _write_csv(subgroup_path, subgroup_rows)
    output = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "final_reserve_fpr_uncertainty_complete",
        "cache_only": True,
        "final_reserve_already_consumed": True,
        "selection_performed": False,
        "successful_authentic_records": len(rows),
        "conditions": len(condition_rows),
        "source_groups": len({str(row["source_group_id"]) for row in rows}),
        "subgroup_rows": len(subgroup_rows),
        "robust_family": robust_rows,
        "checks": {
            "input_hash_matches": True,
            "all_2496_authentic_records_retained": len(rows) == 2496,
            "all_records_successful": True,
            "all_26_conditions_complete": len(condition_rows) == 26,
            "all_conditions_have_96_groups": all(row["groups"] == 96 for row in condition_rows),
            "thresholds_constant_within_condition": True,
            "values_finite_and_bounded": True,
            "no_selection": True,
        },
        "input": {
            "config_sha256": _sha256(config_path),
            "protocol_sha256": _sha256(protocol_path),
            "predictions_sha256": _sha256(predictions_path),
        },
        "outputs": {
            "condition_table": str(condition_path.relative_to(project_root)),
            "condition_table_sha256": _sha256(condition_path),
            "robust_family_table": str(robust_path.relative_to(project_root)),
            "robust_family_table_sha256": _sha256(robust_path),
            "subgroup_table": str(subgroup_path.relative_to(project_root)),
            "subgroup_table_sha256": _sha256(subgroup_path),
            "report": str(report_path.relative_to(project_root)),
        },
    }
    if not all(output["checks"].values()) and runtime["require_complete"]:
        raise RuntimeError("FPR uncertainty completeness gate failed")
    _write_json(summary_path, output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(output), encoding="utf-8")
    output["outputs"]["report_sha256"] = _sha256(report_path)
    _write_json(summary_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
