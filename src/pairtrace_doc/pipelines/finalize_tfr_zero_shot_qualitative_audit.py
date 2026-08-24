from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import yaml

from pairtrace_doc.pipelines.train_student_100 import _resolve, _sha256, _write_json


IMMUTABLE_MEMBERS = {
    "README_FOR_REVIEWER.md",
    "case_manifest.json",
    "human_review_packet.pdf",
    "tfr_zero_shot_qualitative_audit_protocol.md",
}
EXPECTED_MEMBERS = IMMUTABLE_MEMBERS | {"human_review_worksheet.json", "integrity.json"}
REVIEW_FIELDS = {
    "case_id",
    "failure_mode",
    "localization_quality",
    "mapping_valid",
    "mask_valid",
    "registration_artifact",
    "reviewed_at_utc",
    "reviewer_identifier",
    "reviewer_note",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _validate_reviews(
    worksheet: dict[str, Any],
    case_ids: list[str],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[datetime]]:
    if worksheet.get("status") != "human_review_complete":
        raise ValueError("TFR human review status is not complete")
    if worksheet.get("human_review_complete") is not True:
        raise ValueError("TFR human review completion flag is false")
    reviews = worksheet.get("reviews")
    if not isinstance(reviews, list) or [row.get("case_id") for row in reviews] != case_ids:
        raise ValueError("TFR human review case topology changed")
    allowed = worksheet.get("allowed_values")
    if not isinstance(allowed, dict):
        raise ValueError("TFR human review allowed values are missing")
    rules = config["review_rules"]
    prohibited = {str(value).casefold() for value in rules["prohibited_reviewer_identifiers"]}
    reviewers: set[str] = set()
    timestamps: list[datetime] = []
    for row in reviews:
        case_id = str(row.get("case_id"))
        if set(row) != REVIEW_FIELDS:
            raise ValueError(f"{case_id}: review field topology changed")
        for field in ("mapping_valid", "mask_valid", "registration_artifact", "localization_quality"):
            if row[field] not in allowed[field]:
                raise ValueError(f"{case_id}: invalid {field}")
        failure_mode = row["failure_mode"]
        if (
            not isinstance(failure_mode, str)
            or not failure_mode.strip()
            or failure_mode != failure_mode.strip()
            or len(failure_mode) > int(rules["max_failure_mode_characters"])
        ):
            raise ValueError(f"{case_id}: malformed failure_mode")
        note = row["reviewer_note"]
        if (
            not isinstance(note, str)
            or not note.strip()
            or note != note.strip()
            or "\n" in note
            or len(note) > int(rules["max_note_characters"])
        ):
            raise ValueError(f"{case_id}: malformed reviewer_note")
        reviewer = row["reviewer_identifier"]
        if (
            not isinstance(reviewer, str)
            or not reviewer.strip()
            or reviewer != reviewer.strip()
            or reviewer.casefold() in prohibited
        ):
            raise ValueError(f"{case_id}: malformed reviewer identifier")
        reviewers.add(reviewer)
        try:
            timestamp = datetime.fromisoformat(str(row["reviewed_at_utc"]).replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{case_id}: invalid review timestamp") from error
        if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
            raise ValueError(f"{case_id}: review timestamp is not explicit UTC")
        if timestamp > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError(f"{case_id}: review timestamp is in the future")
        timestamps.append(timestamp.astimezone(timezone.utc))
    if rules["single_reviewer_required"] and len(reviewers) != 1:
        raise ValueError("TFR audit requires one stable reviewer identifier")
    if reviewers != {str(config["input"]["expected_reviewer_identifier"])}:
        raise ValueError("TFR reviewer identity differs from user confirmation")
    return reviews, timestamps


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if runtime["device"] != "cpu":
        raise ValueError("TFR audit finalization must be CPU-only")
    prohibited = (
        "model_inference_authorized",
        "model_training_authorized",
        "metric_computation_authorized",
        "threshold_selection_authorized",
        "sample_replacement_authorized",
        "tfr_holdout_read_allowed",
    )
    if any(runtime[name] for name in prohibited):
        raise ValueError("TFR audit finalization crossed its evidence boundary")
    independence = config["reviewer_independence"]
    if not (
        independence["confirmed_by_user"]
        and independence["reviewer_did_not_participate_in_training_tuning_or_case_selection"]
        and not independence["machine_verifiable"]
    ):
        raise ValueError("TFR reviewer independence was not properly confirmed")

    input_config = config["input"]
    original_path = _resolve(project_root, input_config["original_packet"])
    labeled_path = _resolve(project_root, input_config["labeled_packet"])
    if _sha256(original_path) != input_config["expected_original_packet_sha256"]:
        raise ValueError("original TFR review packet changed")
    if _sha256(labeled_path) != input_config["expected_labeled_packet_sha256"]:
        raise ValueError("labeled TFR review packet changed")
    with ZipFile(original_path) as original_archive, ZipFile(labeled_path) as labeled_archive:
        if set(original_archive.namelist()) != EXPECTED_MEMBERS:
            raise ValueError("original TFR packet member set changed")
        if set(labeled_archive.namelist()) != EXPECTED_MEMBERS:
            raise ValueError("labeled TFR packet member set changed")
        original = {name: original_archive.read(name) for name in EXPECTED_MEMBERS}
        labeled = {name: labeled_archive.read(name) for name in EXPECTED_MEMBERS}
    for name in IMMUTABLE_MEMBERS:
        if original[name] != labeled[name]:
            raise ValueError(f"immutable TFR packet member changed: {name}")
    if _sha256_bytes(labeled["case_manifest.json"]) != input_config["expected_case_manifest_sha256"]:
        raise ValueError("TFR case manifest hash changed")
    if _sha256_bytes(labeled["tfr_zero_shot_qualitative_audit_protocol.md"]) != input_config["expected_protocol_sha256"]:
        raise ValueError("TFR audit protocol hash changed")
    if _sha256_bytes(labeled["human_review_packet.pdf"]) != input_config["expected_pdf_sha256"]:
        raise ValueError("TFR audit PDF hash changed")
    worksheet_sha256 = _sha256_bytes(labeled["human_review_worksheet.json"])
    if worksheet_sha256 != input_config["expected_completed_worksheet_sha256"]:
        raise ValueError("completed TFR worksheet hash changed")
    integrity = _json_bytes(labeled["integrity.json"], "integrity")
    if integrity.get("worksheet_sha256") != worksheet_sha256:
        raise ValueError("TFR worksheet integrity link changed")
    manifest = _json_bytes(labeled["case_manifest.json"], "case manifest")
    case_ids = [str(case["case_id"]) for case in manifest["cases"]]
    if len(case_ids) != int(input_config["expected_case_count"]):
        raise ValueError("TFR audit case count changed")
    worksheet = _json_bytes(labeled["human_review_worksheet.json"], "worksheet")
    reviews, timestamps = _validate_reviews(worksheet, case_ids, config)

    counts = {
        field: dict(sorted(Counter(str(row[field]) for row in reviews).items()))
        for field in (
            "mapping_valid",
            "mask_valid",
            "registration_artifact",
            "localization_quality",
            "failure_mode",
        )
    }
    gate_config = config["gate"]
    checks = {
        "all_mapping_valid": (
            not gate_config["require_all_mapping_valid"]
            or all(row["mapping_valid"] == "yes" for row in reviews)
        ),
        "all_masks_valid": (
            not gate_config["require_all_masks_valid"]
            or all(row["mask_valid"] == "yes" for row in reviews)
        ),
        "reviewer_independence_confirmed": True,
        "immutable_packet_members_unchanged": True,
    }
    status = "tfr_independent_qualitative_audit_gate_passed" if all(checks.values()) else "tfr_independent_qualitative_audit_gate_failed"
    paths = config["paths"]
    frozen_worksheet_path = _resolve(project_root, paths["frozen_worksheet"])
    frozen_worksheet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = frozen_worksheet_path.with_suffix(frozen_worksheet_path.suffix + ".tmp")
    temporary.write_bytes(labeled["human_review_worksheet.json"])
    temporary.replace(frozen_worksheet_path)
    summary = {
        "status": status,
        "paper_evidence": False,
        "human_review_complete": True,
        "reviewer_independence_confirmed_by_user": True,
        "reviewer_independence_machine_verifiable": False,
        "reviewer_identifier": str(input_config["expected_reviewer_identifier"]),
        "case_count": len(reviews),
        "case_ids": case_ids,
        "reviewed_at_utc_min": min(timestamps).isoformat(),
        "reviewed_at_utc_max": max(timestamps).isoformat(),
        "category_counts": counts,
        "checks": checks,
        "quantitative_results_changed": False,
        "threshold_selection_performed": False,
        "sample_replacement_used": False,
        "tfr_holdout_read": False,
        "labeled_packet_sha256": _sha256(labeled_path),
        "frozen_worksheet": str(frozen_worksheet_path.relative_to(project_root)),
        "frozen_worksheet_sha256": _sha256(frozen_worksheet_path),
        "reviews": reviews,
    }
    summary_path = _resolve(project_root, paths["summary"])
    _write_json(summary_path, summary)
    report_lines = [
        "# TFR zero-shot independent qualitative audit result",
        "",
        f"Status: `{status}`.",
        "",
        f"Reviewer: `{input_config['expected_reviewer_identifier']}`; independence confirmed by the user and not machine-verifiable.",
        f"Completed cases: {len(reviews)}/{len(reviews)}. Mapping valid: {counts['mapping_valid']}. Mask valid: {counts['mask_valid']}.",
        f"Registration artifacts: {counts['registration_artifact']}. Localization quality: {counts['localization_quality']}.",
        f"Failure modes: {counts['failure_mode']}.",
        "",
        "The frozen cases found no authentic-pair mapping or exact-mask defect. Tail limitations were attributed to diffuse false positives, registration residuals, and one model miss. These stratified cases are descriptive and do not estimate population failure rates.",
        "",
        f"Frozen completed worksheet SHA-256: `{worksheet_sha256}`.",
        "",
    ]
    report_path = _resolve(project_root, paths["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    output = {
        "status": status,
        "experiment": config["experiment"],
        "human_review_complete": True,
        "tfr_holdout_read": False,
        "input": {
            "labeled_packet": str(labeled_path.relative_to(project_root)),
            "labeled_packet_sha256": _sha256(labeled_path),
        },
        "outputs": {
            "frozen_worksheet": str(frozen_worksheet_path.relative_to(project_root)),
            "frozen_worksheet_sha256": _sha256(frozen_worksheet_path),
            "summary": str(summary_path.relative_to(project_root)),
            "summary_sha256": _sha256(summary_path),
            "report": str(report_path.relative_to(project_root)),
            "report_sha256": _sha256(report_path),
        },
    }
    manifest_path = _resolve(project_root, paths["manifest"])
    _write_json(manifest_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
