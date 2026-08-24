from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


METHOD_MAP = {
    "registered_normalized_rgb_difference": "raw_rgb_difference",
    "registered_ssim_distance": "ssim_distance",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(root: Path, path_value: str, expected: str, label: str) -> Path:
    path = _resolve(root, path_value)
    digest = _sha256(path)
    if digest != expected:
        raise ValueError(f"{label} changed: {digest} != {expected}")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    if bool(experiment["paper_evidence"]) or bool(experiment["new_evaluation_read"]):
        raise ValueError("threshold adoption is development-only governance")
    protocol = _verify(
        project_root,
        str(experiment["protocol"]),
        str(experiment["expected_protocol_sha256"]),
        "direct paired-baseline protocol",
    )
    source = config["source"]
    source_config_path = _verify(
        project_root,
        str(source["config"]),
        str(source["expected_config_sha256"]),
        "source calibration config",
    )
    summary_path = _verify(
        project_root,
        str(source["summary"]),
        str(source["expected_summary_sha256"]),
        "source calibration summary",
    )
    metrics_path = _verify(
        project_root,
        str(source["metrics"]),
        str(source["expected_metrics_sha256"]),
        "source calibration metrics",
    )
    predictions_path = _verify(
        project_root,
        str(source["predictions"]),
        str(source["expected_predictions_sha256"]),
        "source calibration predictions",
    )
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    if bool(source_config["operating_point"]["threshold_transfer_authorized"]):
        raise ValueError("expected original experiment to prohibit unscoped transfer")
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        source_summary.get("status") != "registered_pair_controls_development100_complete"
        or not source_summary.get("development_only")
        or source_summary.get("final_reserve_read")
        or not source_summary.get("threshold_selection_used")
        or int(source_summary.get("failed_records", -1)) != 0
        or int(source_summary.get("successful_records", -1)) != 400
    ):
        raise ValueError("source calibration is not a complete development-only run")
    authorization = config["authorization"]
    if not bool(
        authorization["new_protocol_authorizes_transfer_to_future_untouched_evaluation"]
    ):
        raise PermissionError("new direct protocol did not authorize threshold adoption")
    if bool(authorization["source_test_or_final_reserve_read"]) or bool(
        authorization["model_or_checkpoint_selection"]
    ):
        raise ValueError("threshold adoption crossed the development boundary")
    methods = {}
    for source_name, target_name in METHOD_MAP.items():
        metric = source_summary["metrics"][source_name]
        methods[target_name] = {
            "pixel": {
                "threshold": float(metric["pixel_threshold"]),
                "development_document_macro_pixel_f1": float(metric["pixel_f1"]),
                "development_authentic_pixel_fpr": float(
                    metric["authentic_pixel_fpr"]
                ),
            },
            "image": {
                "threshold": float(metric["pixel_threshold"]),
                "development_authentic_image_fpr": float(
                    metric["authentic_image_fpr"]
                ),
                "score_rule": "maximum_valid_pixel_score",
            },
            "selection_partition": str(authorization["selection_partition"]),
            "selection_used_test_or_evaluation": False,
        }
    thresholds = {
        "status": "ijdar_nonlearned_thresholds_adopted_before_new_evaluation",
        "paper_evidence": False,
        "methods": methods,
        "source_experiment_threshold_transfer_authorized": False,
        "new_direct_protocol_transfer_authorized": True,
        "new_evaluation_read": False,
        "source_summary_sha256": _sha256(summary_path),
        "source_metrics_sha256": _sha256(metrics_path),
        "source_predictions_sha256": _sha256(predictions_path),
        "protocol_sha256": _sha256(protocol),
    }
    threshold_path = _resolve(project_root, str(config["paths"]["thresholds"]))
    summary_output = _resolve(project_root, str(config["paths"]["summary"]))
    _write_json(threshold_path, thresholds)
    output = {
        **{key: value for key, value in thresholds.items() if key != "methods"},
        "method_count": len(methods),
        "thresholds": methods,
        "threshold_file": str(threshold_path.relative_to(project_root)),
        "threshold_file_sha256": _sha256(threshold_path),
        "config_sha256": _sha256(config_path),
    }
    _write_json(summary_output, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
