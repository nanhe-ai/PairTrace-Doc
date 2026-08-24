from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["runtime"]["training_allowed"] or config["runtime"][
        "new_evaluation_read_allowed"
    ]:
        raise ValueError("baseline freeze cannot train or read the new evaluation")
    protocol_path = _resolve(project_root, str(config["experiment"]["protocol"]))
    if _sha256(protocol_path) != str(config["experiment"]["expected_protocol_sha256"]):
        raise ValueError("direct paired-baseline protocol changed")
    amendment_value = config["experiment"].get("amendment")
    expected_amendment_sha256 = config["experiment"].get(
        "expected_amendment_sha256"
    )
    amendment_path: Path | None = None
    if (amendment_value is None) != (expected_amendment_sha256 is None):
        raise ValueError("direct paired-baseline amendment binding is incomplete")
    if amendment_value is not None:
        amendment_path = _resolve(project_root, str(amendment_value))
        if _sha256(amendment_path) != str(expected_amendment_sha256):
            raise ValueError("direct paired-baseline amendment changed")
    records: list[dict[str, Any]] = []
    for specification in config["ready_learned_methods"]:
        checkpoint = _resolve(project_root, str(specification["checkpoint"]))
        summary_path = _resolve(project_root, str(specification["validation_summary"]))
        if _sha256(checkpoint) != str(specification["checkpoint_sha256"]):
            raise ValueError(f"ready baseline checkpoint changed: {checkpoint}")
        if _sha256(summary_path) != str(specification["validation_summary_sha256"]):
            raise ValueError(f"ready baseline validation summary changed: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        threshold = float(summary["operating_point"]["threshold"])
        if threshold != float(specification["validation_threshold"]):
            raise ValueError(f"ready baseline threshold changed: {summary_path}")
        records.append(
            {
                **specification,
                "kind": "learned",
                "status": "ready",
                "paper_evidence": False,
                "new_evaluation_read": False,
            }
        )
    for specification in config["pending_learned_methods"]:
        training_config = _resolve(project_root, str(specification["training_config"]))
        if not training_config.is_file():
            raise ValueError(f"pending training config is missing: {training_config}")
        checkpoint = _resolve(project_root, str(specification["checkpoint"]))
        if checkpoint.exists():
            raise ValueError(
                "pending baseline checkpoint now exists; finalize it with a dated amendment"
            )
        records.append(
            {
                **specification,
                "training_config_sha256": _sha256(training_config),
                "kind": "learned",
                "status": "pending_training",
                "paper_evidence": False,
                "new_evaluation_read": False,
            }
        )
    for specification in config["nonlearned_methods"]:
        record = {
            **specification,
            "kind": "nonlearned",
            "status": "ready_for_toy",
            "paper_evidence": False,
            "new_evaluation_read": False,
        }
        source_value = specification.get("source")
        if source_value is not None:
            source = _resolve(project_root, str(source_value))
            if _sha256(source) != str(specification["source_sha256"]):
                raise ValueError(f"nonlearned baseline source changed: {source}")
        calibration_summary_value = specification.get("calibration_summary")
        threshold_artifact_value = specification.get("threshold_artifact")
        if calibration_summary_value is not None or threshold_artifact_value is not None:
            if calibration_summary_value is None or threshold_artifact_value is None:
                raise ValueError("nonlearned calibration binding is incomplete")
            calibration_summary = _resolve(project_root, str(calibration_summary_value))
            threshold_artifact = _resolve(project_root, str(threshold_artifact_value))
            if _sha256(calibration_summary) != str(
                specification["calibration_summary_sha256"]
            ):
                raise ValueError(f"nonlearned calibration summary changed: {calibration_summary}")
            if _sha256(threshold_artifact) != str(
                specification["threshold_artifact_sha256"]
            ):
                raise ValueError(f"nonlearned threshold artifact changed: {threshold_artifact}")
            if str(specification.get("calibration_status")) != "ready_development_only":
                raise ValueError("calibrated nonlearned method status changed")
        records.append(record)
    names = [str(record["name"]) for record in records]
    if len(records) != 12 or len(names) != len(set(names)):
        raise ValueError("direct paired-baseline method inventory changed")
    registry_path = _resolve(project_root, str(config["paths"]["registry"]))
    summary_path = _resolve(project_root, str(config["paths"]["summary"]))
    _write_jsonl(registry_path, records)
    summary = {
        "status": "direct_paired_baseline_registry_frozen",
        "paper_evidence": False,
        "new_evaluation_read": False,
        "training_started": False,
        "method_count": len(records),
        "ready_learned_count": sum(
            record["status"] == "ready" for record in records
        ),
        "pending_learned_count": sum(
            record["status"] == "pending_training" for record in records
        ),
        "nonlearned_count": sum(record["kind"] == "nonlearned" for record in records),
        "calibrated_nonlearned_count": sum(
            record.get("calibration_status") == "ready_development_only"
            for record in records
        ),
        "protocol_sha256": _sha256(protocol_path),
        "amendment": (
            str(amendment_path.relative_to(project_root))
            if amendment_path is not None
            else None
        ),
        "amendment_sha256": (
            _sha256(amendment_path) if amendment_path is not None else None
        ),
        "config_sha256": _sha256(config_path),
        "registry": str(registry_path.relative_to(project_root)),
        "registry_sha256": _sha256(registry_path),
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
