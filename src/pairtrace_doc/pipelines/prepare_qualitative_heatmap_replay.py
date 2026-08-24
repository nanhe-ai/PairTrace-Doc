from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_json,
)


def _validate_runtime(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    if runtime["device"] != "cpu" or not runtime["preflight_authorized"]:
        raise ValueError("qualitative heatmap preflight must be CPU-authorized")
    prohibited = (
        "model_inference_authorized",
        "gpu_launch_authorized",
        "final_reserve_selection_authorized",
        "threshold_selection_authorized",
        "metric_computation_authorized",
        "sample_replacement_authorized",
        "human_audit_completion_authorized",
    )
    if any(bool(runtime[name]) for name in prohibited):
        raise ValueError("qualitative heatmap preflight crossed an evidence boundary")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("qualitative heatmap preflight cannot create paper evidence")


def _select_by_prefix(record_ids: list[str], prefix: str) -> str:
    matches = [
        record_id
        for record_id in record_ids
        if record_id == prefix or record_id.startswith(prefix + ":")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"replay record prefix {prefix!r} selected {len(matches)} records"
        )
    return matches[0]


def _record_index(rows: list[dict[str, Any]], source_name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row["record_id"])
        if record_id in result:
            raise ValueError(f"duplicate record in replay source {source_name}: {record_id}")
        result[record_id] = row
    return result


def _fixed_threshold(row: dict[str, Any]) -> float:
    value = row.get("fixed_pixel_threshold", row.get("fixed_threshold"))
    if value is None:
        raise ValueError(f"source record has no frozen pixel threshold: {row['record_id']}")
    return float(value)


def _reference_sha256(case: dict[str, Any], display_group: str) -> str:
    comparison_groups = {
        "shuffled_clean",
        "wrong_same_dataset",
        "robust_cross_device",
    }
    if display_group in comparison_groups:
        reference = case.get("wrong_reference") or case.get("selected_reference")
    else:
        reference = case.get("correct_reference") or case.get(
            "correct_same_device_reference"
        )
    if not isinstance(reference, dict) or not reference.get("sha256"):
        raise ValueError(
            f"qualitative replay reference is missing for {case['case_id']} "
            f"display group {display_group}"
        )
    return str(reference["sha256"])


def _replay_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_runtime(config)

    experiment = config["experiment"]
    protocol_path = _resolve(project_root, experiment["protocol"])
    protocol_hash = _sha256(protocol_path)
    if protocol_hash != str(experiment["expected_protocol_sha256"]):
        raise ValueError("qualitative heatmap replay protocol SHA-256 changed")

    inputs = config["input"]
    case_manifest_path = _resolve(project_root, inputs["case_manifest"])
    render_manifest_path = _resolve(project_root, inputs["render_manifest"])
    case_manifest_hash = _sha256(case_manifest_path)
    render_manifest_hash = _sha256(render_manifest_path)
    if case_manifest_hash != str(inputs["expected_case_manifest_sha256"]):
        raise ValueError("qualitative case manifest SHA-256 changed")
    if render_manifest_hash != str(inputs["expected_render_manifest_sha256"]):
        raise ValueError("qualitative render manifest SHA-256 changed")
    with case_manifest_path.open("r", encoding="utf-8") as handle:
        case_manifest = json.load(handle)
    with render_manifest_path.open("r", encoding="utf-8") as handle:
        render_manifest = json.load(handle)
    cases = case_manifest["cases"]
    if len(cases) != int(inputs["expected_case_count"]):
        raise ValueError("qualitative case count changed")
    if case_manifest["rendering"]["sample_replacement_allowed"]:
        raise ValueError("frozen qualitative manifest allows sample replacement")
    if render_manifest["human_review_complete"]:
        raise ValueError("heatmap preflight cannot follow a claimed completed audit")

    source_rows: dict[str, dict[str, dict[str, Any]]] = {}
    source_artifacts: dict[str, dict[str, Any]] = {}
    for name, specification in config["sources"].items():
        predictions_path = _resolve(project_root, specification["predictions"])
        evaluation_path = _resolve(project_root, specification["evaluation_config"])
        predictions_hash = _sha256(predictions_path)
        evaluation_hash = _sha256(evaluation_path)
        if predictions_hash != str(specification["predictions_sha256"]):
            raise ValueError(f"qualitative replay prediction source changed: {name}")
        if evaluation_hash != str(specification["evaluation_config_sha256"]):
            raise ValueError(f"qualitative replay evaluation config changed: {name}")
        source_rows[name] = _record_index(_read_jsonl(predictions_path), name)
        source_artifacts[name] = {
            "cohort": specification["cohort"],
            "predictions": str(predictions_path.relative_to(project_root)),
            "predictions_sha256": predictions_hash,
            "evaluation_config": str(evaluation_path.relative_to(project_root)),
            "evaluation_config_sha256": evaluation_hash,
        }

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            str(paths["scratch_env"]),
            str(_resolve(project_root, str(paths["scratch_default"]))),
        )
    ).resolve()
    encoder_path = _resolve(scratch, config["models"]["encoder_weights"])
    encoder_hash = _sha256(encoder_path)
    if encoder_hash != str(config["models"]["encoder_weights_sha256"]):
        raise ValueError("qualitative replay encoder weights changed")
    checkpoints: dict[str, dict[str, str]] = {}
    for name, specification in config["models"].items():
        if name in {"encoder_weights", "encoder_weights_sha256"}:
            continue
        checkpoint_path = _resolve(project_root, specification["checkpoint"])
        checkpoint_hash = _sha256(checkpoint_path)
        if checkpoint_hash != str(specification["checkpoint_sha256"]):
            raise ValueError(f"qualitative replay checkpoint changed: {name}")
        checkpoints[name] = {
            "path": str(checkpoint_path.relative_to(project_root)),
            "sha256": checkpoint_hash,
        }

    cases_by_id = {str(case["case_id"]): case for case in cases}
    case_specs = config["replay"]["case_records"]
    if list(case_specs) != [str(case["case_id"]) for case in cases]:
        raise ValueError("qualitative replay case order or membership changed")
    replay_cache_dir = _resolve(scratch, paths["replay_cache_dir"])
    records: list[dict[str, Any]] = []
    used_source_records: set[tuple[str, str]] = set()
    per_case_counts: dict[str, int] = {}
    for case_id, case_spec in case_specs.items():
        case = cases_by_id[str(case_id)]
        source_name = str(case_spec["source"])
        source = source_artifacts[source_name]
        if str(case["cohort"]) != str(source["cohort"]):
            raise ValueError(f"qualitative replay cohort mismatch: {case_id}")
        manifest_record_ids = [
            str(value) for value in case["source_predictions"]["record_ids"]
        ]
        manifest_cache_paths = {
            str(value) for value in case["source_predictions"]["score_cache_paths"]
        }
        case_count = 0
        for record_spec in case_spec["records"]:
            record_id = _select_by_prefix(
                manifest_record_ids, str(record_spec["prefix"])
            )
            source_key = (source_name, record_id)
            if source_key in used_source_records:
                raise ValueError(f"qualitative replay record was reused: {record_id}")
            used_source_records.add(source_key)
            row = source_rows[source_name].get(record_id)
            if row is None:
                raise ValueError(f"qualitative replay record is absent: {record_id}")
            if row.get("status") != "ok" or row.get("sample_kind") != "forged":
                raise ValueError(f"qualitative replay record is not a successful forgery: {record_id}")
            if str(row["source_group_id"]) != str(case["source_group_id"]):
                raise ValueError(f"qualitative replay source group changed: {record_id}")
            original_cache = str(row["score_cache"])
            if original_cache not in manifest_cache_paths:
                raise ValueError(f"qualitative replay cache path changed: {record_id}")
            checkpoint_name = str(record_spec["checkpoint"])
            checkpoint = checkpoints[checkpoint_name]
            row_checkpoint = row.get("checkpoint_sha256", row.get("model_identity"))
            if row_checkpoint is not None and str(row_checkpoint) != checkpoint["sha256"]:
                raise ValueError(f"qualitative replay model identity changed: {record_id}")
            input_sha256 = str(case["candidate"]["sha256"])
            reference_sha256 = _reference_sha256(
                case, str(record_spec["display_group"])
            )
            mask_sha256 = str(case["mask"]["sha256"])
            key_payload = {
                "schema_version": 1,
                "case_manifest_sha256": case_manifest_hash,
                "case_id": case_id,
                "source_record_id": record_id,
                "source_predictions_sha256": source["predictions_sha256"],
                "source_evaluation_config_sha256": source[
                    "evaluation_config_sha256"
                ],
                "checkpoint_sha256": checkpoint["sha256"],
                "encoder_weights_sha256": encoder_hash,
                "input_sha256": input_sha256,
                "reference_sha256": reference_sha256,
                "mask_sha256": mask_sha256,
                "original_score_cache": original_cache,
                "score_dtype": config["replay"]["score_dtype"],
                "global_probability_scale": config["replay"][
                    "global_probability_scale"
                ],
            }
            key = _replay_key(key_payload)
            replay_path = replay_cache_dir / key[:2] / f"{key}.npz"
            records.append(
                {
                    "case_id": case_id,
                    "cohort": case["cohort"],
                    "mask_semantics": case["mask_semantics"],
                    "display_group": record_spec["display_group"],
                    "source_record_id": record_id,
                    "source_predictions": source["predictions"],
                    "source_predictions_sha256": source["predictions_sha256"],
                    "source_evaluation_config": source["evaluation_config"],
                    "source_evaluation_config_sha256": source[
                        "evaluation_config_sha256"
                    ],
                    "checkpoint_name": checkpoint_name,
                    "checkpoint": checkpoint["path"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "encoder_weights_sha256": encoder_hash,
                    "input_sha256": input_sha256,
                    "reference_sha256": reference_sha256,
                    "mask_sha256": mask_sha256,
                    "original_score_cache": original_cache,
                    "original_score_cache_currently_available": _resolve(
                        scratch, original_cache
                    ).is_file(),
                    "replay_key": key,
                    "replay_score_cache": str(replay_path.relative_to(scratch)),
                    "native_shape": row["native_shape"],
                    "score_shape": row["score_shape"],
                    "score_dtype": config["replay"]["score_dtype"],
                    "fixed_pixel_threshold": _fixed_threshold(row),
                    "alignment_key": row.get("alignment_key"),
                    "alignment_status": row.get("alignment_status"),
                    "model_inference_authorized": False,
                }
            )
            case_count += 1
        per_case_counts[str(case_id)] = case_count

    expected_count = int(config["replay"]["expected_record_count"])
    if len(records) != expected_count:
        raise ValueError(
            f"qualitative replay record count changed: {len(records)} != {expected_count}"
        )
    required_fields = [str(value) for value in config["output_schema"]["required_fields"]]
    if len(required_fields) != len(set(required_fields)):
        raise ValueError("qualitative replay output schema has duplicate fields")

    output = {
        "experiment": experiment,
        "status": "qualitative_heatmap_replay_preflight_passed_execution_not_authorized",
        "paper_evidence": False,
        "new_scientific_metrics_computed": False,
        "model_inference_performed": False,
        "execution_authorized": False,
        "gpu_launch_authorized": False,
        "human_audit_completion_authorized": False,
        "case_manifest": {
            "path": str(case_manifest_path.relative_to(project_root)),
            "sha256": case_manifest_hash,
        },
        "render_manifest": {
            "path": str(render_manifest_path.relative_to(project_root)),
            "sha256": render_manifest_hash,
        },
        "protocol_sha256": protocol_hash,
        "encoder_weights": {
            "path": str(encoder_path.relative_to(scratch)),
            "sha256": encoder_hash,
        },
        "checkpoints": checkpoints,
        "sources": source_artifacts,
        "rendering": {
            "global_probability_scale": config["replay"][
                "global_probability_scale"
            ],
            "per_image_normalization_allowed": config["replay"][
                "per_image_normalization_allowed"
            ],
            "aggregate_rule": config["replay"]["aggregate_rule"],
        },
        "output_schema_required_fields": required_fields,
        "case_count": len(cases),
        "record_count": len(records),
        "per_case_record_counts": per_case_counts,
        "original_score_caches_currently_available": sum(
            bool(record["original_score_cache_currently_available"])
            for record in records
        ),
        "records": records,
    }
    output_path = _resolve(project_root, paths["output_plan"])
    _write_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
