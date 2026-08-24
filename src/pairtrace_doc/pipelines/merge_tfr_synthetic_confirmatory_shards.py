from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from pairtrace_doc.pipelines.evaluate_tfr_synthetic_confirmatory import (
    _paired_comparison_rows,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _validate_method_coverage(
    predictions: list[dict[str, Any]],
    method_names: list[str],
    conditions: list[str],
    expected_records_per_method_condition: int,
) -> None:
    names = set(method_names)
    keys: set[tuple[str, str, str, str]] = set()
    counts: Counter[tuple[str, str]] = Counter()
    for row in predictions:
        if row.get("status") != "ok":
            raise ValueError("confirmation shard merge found an item failure")
        method = str(row["method"])
        condition = str(row["condition"])
        if method not in names or condition not in conditions:
            raise ValueError("confirmation shard merge found an unknown method or condition")
        key = (
            method,
            condition,
            str(row["sample_kind"]),
            str(row["sample_id"]),
        )
        if key in keys:
            raise ValueError("confirmation shard merge found a duplicate item prediction")
        keys.add(key)
        counts[(method, condition)] += 1
    expected = {
        (method, condition): expected_records_per_method_condition
        for method in method_names
        for condition in conditions
    }
    if counts != Counter(expected):
        raise ValueError("confirmation shard method-condition coverage changed")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    code_sha = _sha256(Path(__file__).resolve())
    experiment = config["experiment"]
    if code_sha != str(experiment["expected_merger_code_sha256"]):
        raise ValueError("confirmation shard merger code changed")
    protocol = _resolve(project_root, str(experiment["protocol"]))
    if _sha256(protocol) != str(experiment["expected_protocol_sha256"]):
        raise ValueError("confirmation protocol changed before shard merge")

    method_specification = config["method_registry"]
    method_path = _resolve(project_root, str(method_specification["path"]))
    if _sha256(method_path) != str(method_specification["expected_sha256"]):
        raise ValueError("confirmation method registry changed before shard merge")
    methods = _read_jsonl(method_path)
    method_names = [str(row["name"]) for row in methods]
    if len(method_names) != len(set(method_names)):
        raise ValueError("confirmation method registry contains duplicate names")

    comparison_specification = config["comparison_registry"]
    comparison_path = _resolve(project_root, str(comparison_specification["path"]))
    if _sha256(comparison_path) != str(comparison_specification["expected_sha256"]):
        raise ValueError("confirmation comparison registry changed before shard merge")
    comparisons = _read_jsonl(comparison_path)

    expected_seeds = [int(value) for value in experiment["expected_shard_seeds"]]
    all_predictions: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    seen_methods: set[str] = set()
    seen_seeds: list[int] = []
    for shard_specification in config["shards"]:
        summary_path = _resolve(project_root, str(shard_specification["summary"]))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard = summary.get("method_shard", {})
        seed = int(shard.get("learned_seed", -1))
        if (
            summary.get("status") != "confirmation_evaluation_complete"
            or int(summary.get("failures", -1)) != 0
            or summary.get("claim_boundary")
            != "controlled_confirmation_only_not_official_tfr"
            or not shard.get("enabled")
            or not shard.get("comparisons_deferred")
        ):
            raise ValueError("confirmation scoring shard did not pass its completion gate")
        if str(summary.get("protocol_sha256")) != str(
            experiment["expected_protocol_sha256"]
        ) or str(summary.get("evaluator_code_sha256")) != str(
            experiment["expected_evaluator_code_sha256"]
        ):
            raise ValueError("confirmation scoring shard code/protocol binding changed")
        if str(summary.get("method_registry_sha256")) != str(
            method_specification["expected_sha256"]
        ) or str(summary.get("comparison_registry_sha256")) != str(
            comparison_specification["expected_sha256"]
        ):
            raise ValueError("confirmation scoring shard registry binding changed")
        shard_methods = [str(value) for value in summary["methods"]]
        if seen_methods & set(shard_methods):
            raise ValueError("confirmation scoring shards overlap in method membership")
        seen_methods.update(shard_methods)
        seen_seeds.append(seed)

        outputs = summary["outputs"]
        prediction_path = _resolve(project_root, str(outputs["predictions"]))
        metrics_path = _resolve(project_root, str(outputs["metrics"]))
        if _sha256(prediction_path) != str(outputs["predictions_sha256"]) or _sha256(
            metrics_path
        ) != str(outputs["metrics_sha256"]):
            raise ValueError("confirmation scoring shard output changed before merge")
        predictions = _read_jsonl(prediction_path)
        metrics = _read_csv(metrics_path)
        if len(predictions) != int(summary["prediction_records"]) or len(metrics) != int(
            summary["metric_rows"]
        ):
            raise ValueError("confirmation scoring shard output count changed")
        all_predictions.extend(predictions)
        all_metrics.extend(metrics)
        shard_records.append(
            {
                "seed": seed,
                "summary": str(summary_path.relative_to(project_root)),
                "summary_sha256": _sha256(summary_path),
                "methods": shard_methods,
                "wall_time_seconds": float(summary["wall_time_seconds"]),
                "peak_gpu_memory_mb": float(summary["peak_gpu_memory_mb"]),
            }
        )

    if seen_seeds != expected_seeds or seen_methods != set(method_names):
        raise ValueError("confirmation scoring shard partition changed")
    conditions = [str(value) for value in config["conditions"]]
    expected_per_condition = int(config["expected_records_per_method_condition"])
    _validate_method_coverage(
        all_predictions, method_names, conditions, expected_per_condition
    )
    if len(all_predictions) != int(config["expected_prediction_records"]):
        raise ValueError("confirmation merged prediction count changed")
    metric_keys = [
        (str(row["method"]), str(row["condition"])) for row in all_metrics
    ]
    expected_metric_keys = {
        (method, condition) for method in method_names for condition in conditions
    }
    if len(metric_keys) != len(set(metric_keys)) or set(metric_keys) != expected_metric_keys:
        raise ValueError("confirmation merged metric coverage changed")
    if len(all_metrics) != int(config["expected_metric_rows"]):
        raise ValueError("confirmation merged metric count changed")

    method_order = {name: index for index, name in enumerate(method_names)}
    condition_order = {name: index for index, name in enumerate(conditions)}
    all_predictions.sort(
        key=lambda row: (
            method_order[str(row["method"])],
            condition_order[str(row["condition"])],
            str(row["sample_kind"]),
            str(row["sample_id"]),
        )
    )
    all_metrics.sort(
        key=lambda row: (
            method_order[str(row["method"])],
            condition_order[str(row["condition"])],
        )
    )
    comparison_rows = _paired_comparison_rows(
        all_predictions,
        comparisons,
        conditions,
        seed=int(config["statistics"]["bootstrap_seed"]),
        replicates=int(config["statistics"]["bootstrap_replicates"]),
    )
    if len(comparison_rows) != int(config["expected_comparison_rows"]):
        raise ValueError("confirmation merged comparison count changed")

    paths = config["paths"]
    predictions_path = _resolve(project_root, str(paths["predictions"]))
    metrics_path = _resolve(project_root, str(paths["metrics"]))
    comparisons_path = _resolve(project_root, str(paths["comparisons"]))
    summary_path = _resolve(project_root, str(paths["summary"]))
    _write_jsonl(predictions_path, all_predictions)
    _write_csv(metrics_path, all_metrics)
    _write_csv(comparisons_path, comparison_rows)
    summary = {
        "status": "confirmation_shards_merged",
        "claim_boundary": "controlled_confirmation_only_not_official_tfr",
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(protocol),
        "evaluator_code_sha256": str(experiment["expected_evaluator_code_sha256"]),
        "merger_code_sha256": code_sha,
        "method_registry_sha256": str(method_specification["expected_sha256"]),
        "comparison_registry_sha256": str(comparison_specification["expected_sha256"]),
        "methods": method_names,
        "conditions": conditions,
        "prediction_records": len(all_predictions),
        "metric_rows": len(all_metrics),
        "comparison_rows": len(comparison_rows),
        "failures": 0,
        "confirmation_selection_performed": False,
        "shards": shard_records,
        "outputs": {
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": _sha256(predictions_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
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
