from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from pairtrace_doc.pipelines.train_student_100 import _sha256, _write_json


ADMINISTRATIVE_REVIEW_FIELDS = {
    "case_id",
    "reviewer_note",
    "reviewer_identifier",
    "reviewed_at_utc",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _verify_file(
    project_root: Path, relative: str, expected_sha256: str, label: str
) -> Path:
    path = _resolve(project_root, relative)
    digest = _sha256(path)
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 changed: {digest} != {expected_sha256}")
    return path


def _parse_utc_timestamp(value: Any, case_id: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{case_id}: reviewed_at_utc is empty")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{case_id}: reviewed_at_utc is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{case_id}: reviewed_at_utc must be explicitly UTC")
    if parsed > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError(f"{case_id}: reviewed_at_utc is in the future")
    return parsed.astimezone(timezone.utc)


def _validate_completed_worksheet(
    completed: dict[str, Any],
    blank: dict[str, Any],
    expected_case_ids: list[str],
    *,
    max_note_characters: int,
    single_reviewer_required: bool,
    prohibited_reviewer_identifiers: set[str],
) -> tuple[list[dict[str, Any]], list[datetime]]:
    if set(completed) != set(blank):
        raise ValueError("completed worksheet top-level fields changed")
    for field in set(blank) - {"status", "human_review_complete", "reviews"}:
        if completed[field] != blank[field]:
            raise ValueError(f"completed worksheet changed frozen field: {field}")
    if completed.get("status") != "human_review_complete":
        raise ValueError("completed worksheet status is not human_review_complete")
    if completed.get("human_review_complete") is not True:
        raise ValueError("completed worksheet is not marked complete")
    blank_reviews = blank.get("reviews")
    reviews = completed.get("reviews")
    if not isinstance(blank_reviews, list) or not isinstance(reviews, list):
        raise ValueError("worksheet reviews must be lists")
    if len(reviews) != len(expected_case_ids):
        raise ValueError("completed worksheet review count changed")
    case_ids = [str(review.get("case_id")) for review in reviews]
    if case_ids != expected_case_ids:
        raise ValueError("completed worksheet case order or membership changed")
    allowed_values = blank.get("allowed_values")
    if not isinstance(allowed_values, dict) or not allowed_values:
        raise ValueError("blank worksheet allowed_values are missing")
    categorical_fields = set(allowed_values)
    expected_review_fields = categorical_fields | ADMINISTRATIVE_REVIEW_FIELDS
    timestamps: list[datetime] = []
    reviewer_identifiers: set[str] = set()
    for blank_review, review in zip(blank_reviews, reviews, strict=True):
        case_id = str(review.get("case_id"))
        if set(review) != set(blank_review) or set(review) != expected_review_fields:
            raise ValueError(f"{case_id}: review fields changed")
        if review["case_id"] != blank_review["case_id"]:
            raise ValueError(f"{case_id}: frozen case identifier changed")
        for field in sorted(categorical_fields):
            choices = allowed_values[field]
            if not isinstance(choices, list) or review[field] not in choices:
                raise ValueError(f"{case_id}: invalid {field}: {review[field]!r}")
        note = review["reviewer_note"]
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"{case_id}: reviewer_note is empty")
        if note != note.strip() or "\n" in note or "\r" in note:
            raise ValueError(f"{case_id}: reviewer_note must be one trimmed line")
        if len(note) > max_note_characters:
            raise ValueError(f"{case_id}: reviewer_note is too long")
        reviewer_identifier = review["reviewer_identifier"]
        if not isinstance(reviewer_identifier, str) or not reviewer_identifier.strip():
            raise ValueError(f"{case_id}: reviewer_identifier is empty")
        if reviewer_identifier != reviewer_identifier.strip() or len(reviewer_identifier) > 128:
            raise ValueError(f"{case_id}: reviewer_identifier is malformed")
        if reviewer_identifier.casefold() in prohibited_reviewer_identifiers:
            raise ValueError(f"{case_id}: reviewer_identifier cannot identify an AI reviewer")
        reviewer_identifiers.add(reviewer_identifier)
        timestamps.append(_parse_utc_timestamp(review["reviewed_at_utc"], case_id))
    if single_reviewer_required and len(reviewer_identifiers) != 1:
        raise ValueError("all cases must use one stable reviewer_identifier")
    return reviews, timestamps


def _category_counts(
    reviews: list[dict[str, Any]], categorical_fields: list[str]
) -> dict[str, dict[str, int]]:
    return {
        field: dict(sorted(Counter(str(review[field]) for review in reviews).items()))
        for field in categorical_fields
    }


def _markdown_report(
    worksheet_sha256: str,
    reviews: list[dict[str, Any]],
    category_counts: dict[str, dict[str, int]],
) -> str:
    reviewers = sorted({str(review["reviewer_identifier"]) for review in reviews})
    lines = [
        "# PairTrace-Doc independent qualitative human-review result",
        "",
        "Status: complete and schema-validated; descriptive qualitative evidence only.",
        "",
        f"Frozen completed worksheet SHA-256: `{worksheet_sha256}`.",
        f"Reviewed cases: {len(reviews)}/{len(reviews)}. Reviewer identifier(s): "
        + ", ".join(f"`{value}`" for value in reviewers)
        + ".",
        "",
        "The validator confirmed fixed case membership and order, fixed categorical",
        "enumerations, non-empty one-line notes, stable reviewer identity, and explicit",
        "UTC timestamps. Reviewer independence is a procedural requirement stated in",
        "the frozen packet and is not machine-verifiable.",
        "",
        "## Category counts",
        "",
    ]
    for field, counts in category_counts.items():
        lines.append(
            f"- `{field}`: "
            + ", ".join(f"`{key}`={value}" for key, value in counts.items())
        )
    lines.extend(
        [
            "",
            "## Case-level review",
            "",
            "| Case | Coverage | Dominant failure | Note |",
            "|---|---|---|---|",
        ]
    )
    for review in reviews:
        note = str(review["reviewer_note"]).replace("|", "\\|")
        lines.append(
            f"| `{review['case_id']}` | `{review['prediction_coverage']}` | "
            f"`{review['dominant_failure']}` | {note} |"
        )
    lines.extend(
        [
            "",
            "These judgments cannot change quantitative metrics, thresholds, selected",
            "cases, mask semantics, or evidence status. Weak-box and viewed-development",
            "cases remain limitation evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("human-review finalization config must be a mapping")
    experiment = config["experiment"]
    runtime = config["runtime"]
    if runtime["device"] != "cpu":
        raise ValueError("human-review validation must be CPU-only")
    if runtime["external_human_review_required"] is not True:
        raise ValueError("external human-review requirement was disabled")
    if runtime["ai_review_substitution_authorized"] is not False:
        raise ValueError("AI review cannot substitute for the human audit")
    prohibited = (
        "model_inference_authorized",
        "model_training_authorized",
        "metric_computation_authorized",
        "threshold_selection_authorized",
        "sample_replacement_authorized",
    )
    if any(bool(runtime[name]) for name in prohibited):
        raise ValueError("human-review validation crossed an evidence boundary")

    specification = config["input"]
    verified: dict[str, Path] = {}
    for key, label in (
        ("protocol", "qualitative audit protocol"),
        ("case_manifest", "qualitative case manifest"),
        ("blank_worksheet", "blank heatmap-ready worksheet"),
        ("review_packet_manifest", "human-review packet manifest"),
        ("review_packet_zip", "human-review packet ZIP"),
        ("input_packet", "qualitative input packet"),
        ("heatmap_packet", "qualitative heatmap packet"),
    ):
        verified[key] = _verify_file(
            project_root,
            str(specification[key]),
            str(specification[f"expected_{key}_sha256"]),
            label,
        )
    expected_case_ids = [str(value) for value in specification["expected_case_ids"]]
    case_manifest = _read_json(verified["case_manifest"])
    cases = case_manifest.get("cases")
    if not isinstance(cases, list) or [str(case.get("case_id")) for case in cases] != expected_case_ids:
        raise ValueError("qualitative case order or membership changed")
    packet_manifest = _read_json(verified["review_packet_manifest"])
    if packet_manifest.get("status") != "independent_human_review_packet_ready":
        raise ValueError("human-review packet manifest is not ready")
    if packet_manifest.get("case_ids") != expected_case_ids:
        raise ValueError("human-review packet case order or membership changed")
    if packet_manifest.get("output", {}).get("review_packet_zip_sha256") != specification["expected_review_packet_zip_sha256"]:
        raise ValueError("human-review packet manifest ZIP link changed")

    completed_path = _resolve(project_root, str(specification["completed_worksheet"]))
    if not completed_path.is_file():
        raise FileNotFoundError(
            "completed human-review worksheet is missing; do not substitute AI review"
        )
    blank = _read_json(verified["blank_worksheet"])
    completed = _read_json(completed_path)
    review_rules = config["review_rules"]
    reviews, timestamps = _validate_completed_worksheet(
        completed,
        blank,
        expected_case_ids,
        max_note_characters=int(review_rules["max_note_characters"]),
        single_reviewer_required=bool(review_rules["single_reviewer_required"]),
        prohibited_reviewer_identifiers={
            str(value).casefold()
            for value in review_rules["prohibited_reviewer_identifiers"]
        },
    )
    categorical_fields = sorted(str(value) for value in blank["allowed_values"])
    category_counts = _category_counts(reviews, categorical_fields)
    completed_sha256 = _sha256(completed_path)

    paths = config["paths"]
    frozen_path = _resolve(project_root, str(paths["frozen_worksheet"]))
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = frozen_path.with_suffix(frozen_path.suffix + ".tmp")
    temporary.write_bytes(completed_path.read_bytes())
    temporary.replace(frozen_path)
    if _sha256(frozen_path) != completed_sha256:
        raise ValueError("frozen worksheet copy changed bytes")

    summary = {
        "schema_version": 1,
        "status": "independent_human_review_complete",
        "paper_evidence_role": "descriptive_qualitative_and_limitation_evidence",
        "quantitative_claims_changed": False,
        "case_membership_changed": False,
        "case_count": len(reviews),
        "case_ids": expected_case_ids,
        "reviewer_identifiers": sorted(
            {str(review["reviewer_identifier"]) for review in reviews}
        ),
        "reviewed_at_utc_min": min(timestamps).isoformat(),
        "reviewed_at_utc_max": max(timestamps).isoformat(),
        "category_counts": category_counts,
        "frozen_worksheet": str(frozen_path.relative_to(project_root)),
        "frozen_worksheet_sha256": completed_sha256,
    }
    summary_path = _resolve(project_root, str(paths["summary"]))
    _write_json(summary_path, summary)
    report_path = _resolve(project_root, str(paths["report"]))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _markdown_report(completed_sha256, reviews, category_counts),
        encoding="utf-8",
    )

    implementation_path = Path(__file__).resolve()
    implementation_label = (
        str(implementation_path.relative_to(project_root))
        if implementation_path.is_relative_to(project_root)
        else f"src/pairtrace_doc/pipelines/{implementation_path.name}"
    )
    result = {
        "schema_version": 1,
        "experiment": experiment,
        "status": "independent_human_review_complete",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "human_review_complete": True,
        "human_review_was_machine_generated": False,
        "reviewer_independence_machine_verifiable": False,
        "model_inference_performed": False,
        "new_scientific_metrics_computed": False,
        "quantitative_claims_changed": False,
        "threshold_selection_used": False,
        "sample_replacement_used": False,
        "case_count": len(reviews),
        "input": {
            "config": str(config_path.relative_to(project_root)),
            "config_sha256": _sha256(config_path),
            "implementation": implementation_label,
            "implementation_sha256": _sha256(implementation_path),
            "completed_worksheet": str(completed_path.relative_to(project_root)),
            "completed_worksheet_sha256": completed_sha256,
            **{
                key: {
                    "path": str(path.relative_to(project_root)),
                    "sha256": _sha256(path),
                }
                for key, path in verified.items()
            },
        },
        "output": {
            "frozen_worksheet": str(frozen_path.relative_to(project_root)),
            "frozen_worksheet_sha256": _sha256(frozen_path),
            "summary": str(summary_path.relative_to(project_root)),
            "summary_sha256": _sha256(summary_path),
            "report": str(report_path.relative_to(project_root)),
            "report_sha256": _sha256(report_path),
        },
    }
    manifest_path = _resolve(project_root, str(paths["manifest"]))
    _write_json(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
