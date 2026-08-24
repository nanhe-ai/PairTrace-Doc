from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from jsonschema import Draft202012Validator, FormatChecker
from PIL import Image


PAIR_SCHEMA_VERSION = "ijdar_human_edit_pair_manifest_20260726_v1"
REVIEW_SCHEMA_VERSION = "ijdar_human_edit_visual_review_20260726_v1"
STUDY_ID = "pairtrace-human-edit-20260726"
LOSSLESS_FORMATS = {"PNG", "TIFF"}
LAYERED_SUFFIXES = {".psd", ".xcf", ".kra"}


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _output_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"output path must be project-relative: {value}")
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"output path escapes project root: {value}") from error
    return resolved


def _intake_root(project_root: Path, paths: dict[str, Any]) -> Path:
    environment = paths.get("intake_env")
    override = os.environ.get(str(environment)) if environment else None
    if override:
        return Path(override).expanduser().resolve()
    return _resolve(project_root, str(paths["intake_default"]))


def _intake_path(intake_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"intake artifact path must be relative: {value}")
    resolved = (intake_root / path).resolve()
    try:
        resolved.relative_to(intake_root)
    except ValueError as error:
        raise ValueError(f"intake artifact escapes intake root: {value}") from error
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_source_group_id(source_sha256: str) -> str:
    digest = hashlib.sha256(
        f"pairtrace-human-edit-source-20260726:{source_sha256}".encode("utf-8")
    ).hexdigest()
    return f"humanedit-source:{digest[:20]}"


def source_order_key(source_sha256: str) -> str:
    return hashlib.sha256(
        f"pairtrace-human-edit-20260726:{source_sha256}".encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_accounted(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not a JSON object")
                rows.append({"line_number": line_number, "record": value, "error": None})
            except (json.JSONDecodeError, ValueError) as error:
                rows.append(
                    {
                        "line_number": line_number,
                        "record": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _schema_errors(
    validator: Draft202012Validator, record: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    for error in sorted(validator.iter_errors(record), key=lambda item: list(item.path)):
        location = ".".join(str(value) for value in error.path) or "$"
        errors.append(f"schema:{location}:{error.message}")
    return errors


def _verify_file(intake_root: Path, relative: str, expected_hash: str) -> Path:
    path = _intake_path(intake_root, relative)
    if not path.is_file():
        raise ValueError(f"missing intake artifact: {relative}")
    actual = _sha256(path)
    if actual != expected_hash:
        raise ValueError(f"artifact SHA-256 changed: {relative}: {actual} != {expected_hash}")
    return path


def _profile_sha256(value: Any) -> str | None:
    if value is None:
        return None
    payload = value if isinstance(value, bytes) else bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _load_rgb(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as handle:
        image_format = str(handle.format or "").upper()
        mode = str(handle.mode)
        size = [int(handle.width), int(handle.height)]
        orientation = handle.getexif().get(274)
        profile_hash = _profile_sha256(handle.info.get("icc_profile"))
        if image_format not in LOSSLESS_FORMATS:
            raise ValueError(f"image is not lossless PNG/TIFF: {path.name}")
        if mode != "RGB":
            raise ValueError(f"image is not encoded RGB: {path.name}: {mode}")
        if orientation not in (None, 1):
            raise ValueError(f"image has non-identity EXIF orientation: {path.name}")
        array = np.asarray(handle, dtype=np.uint8).copy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"decoded image is not HxWx3 uint8 RGB: {path.name}")
    return array, {
        "format": image_format,
        "mode": mode,
        "size_width_height": size,
        "icc_profile_sha256": profile_hash,
        "exif_orientation": orientation,
    }


def _load_binary_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as handle:
        image_format = str(handle.format or "").upper()
        mode = str(handle.mode)
        if image_format not in LOSSLESS_FORMATS:
            raise ValueError(f"intent mask is not lossless PNG/TIFF: {path.name}")
        if mode not in {"1", "L"}:
            raise ValueError(f"intent mask mode is not binary-compatible: {path.name}: {mode}")
        array = np.asarray(handle.convert("L"), dtype=np.uint8)
    values = sorted(int(value) for value in np.unique(array))
    if not values or not set(values).issubset({0, 255}) or 255 not in values:
        raise ValueError(f"intent mask values are not binary 0/255: {values}")
    return array == 255, {"format": image_format, "mode": mode, "values": values}


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp lacks a UTC offset")
    return parsed


def _verify_assignment(path: Path, record: dict[str, Any]) -> datetime:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError("assignment record is not a JSON object")
    for field in ("attempt_id", "source_group_id", "editor_id", "edit_type"):
        if value.get(field) != record.get(field):
            raise ValueError(f"assignment field mismatch: {field}")
    frozen = value.get("assignment_frozen_utc")
    if not isinstance(frozen, str):
        raise ValueError("assignment record lacks assignment_frozen_utc")
    return _parse_utc(frozen)


def _review_rows(
    path: Path,
    validator: Draft202012Validator,
) -> tuple[dict[str, dict[str, Any]], list[str], int]:
    indexed: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    entries = _read_jsonl_accounted(path)
    for entry in entries:
        line = int(entry["line_number"])
        record = entry["record"]
        if record is None:
            errors.append(f"review_line_{line}:{entry['error']}")
            continue
        validation = _schema_errors(validator, record)
        if validation:
            errors.extend(f"review_line_{line}:{value}" for value in validation)
            continue
        attempt_id = str(record["attempt_id"])
        if attempt_id in indexed:
            errors.append(f"duplicate_review_attempt_id:{attempt_id}")
            continue
        indexed[attempt_id] = record
    return indexed, errors, len(entries)


def _review_errors(record: dict[str, Any], review: dict[str, Any] | None) -> list[str]:
    if review is None:
        return ["missing_independent_visual_review"]
    errors: list[str] = []
    if str(review["reviewer_id"]).removeprefix("reviewer:") == str(
        record["editor_id"]
    ).removeprefix("editor:"):
        errors.append("visual_reviewer_matches_editor")
    checks = (
        "semantic_edit_meaningful",
        "no_annotation_marker",
        "intent_mask_covers_visible_edit",
        "no_unintended_global_redraw",
        "no_prohibited_content_visible",
        "review_passed",
    )
    errors.extend(f"visual_review_failed:{field}" for field in checks if not review[field])
    return errors


def _safe_mask_name(attempt_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", attempt_id) + ".png"


def _write_exact_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(mask.astype(np.uint8) * 255).save(
        temporary, format="PNG", optimize=False
    )
    temporary.replace(path)


def _audit_record(
    *,
    record: dict[str, Any],
    line_number: int,
    project_root: Path,
    intake_root: Path,
    expected_origin: str,
    review: dict[str, Any] | None,
    exact_mask_root: Path,
    duplicate_attempt: bool,
    duplicate_sequence: bool,
    duplicate_variant: bool,
    expected_editor_id: str,
    expected_edit_type: str,
) -> dict[str, Any]:
    errors: list[str] = []
    attempt_id = str(record["attempt_id"])
    if record["record_origin"] != expected_origin:
        errors.append(
            f"record_origin_mismatch:{record['record_origin']}!={expected_origin}"
        )
    if record["source_group_id"] != expected_source_group_id(record["source_sha256"]):
        errors.append("source_group_id_not_derived_from_source_sha256")
    if duplicate_attempt:
        errors.append("duplicate_attempt_id")
    if duplicate_sequence:
        errors.append("duplicate_source_group_attempt_sequence")
    if duplicate_variant:
        errors.append("duplicate_edited_variant_within_source_group")
    if record["editor_id"] != expected_editor_id:
        errors.append(
            f"assignment_policy_editor_mismatch:{record['editor_id']}!={expected_editor_id}"
        )
    if record["edit_type"] != expected_edit_type:
        errors.append(
            f"assignment_policy_edit_type_mismatch:{record['edit_type']}!={expected_edit_type}"
        )

    verified: dict[str, str] = {}
    assignment_frozen: datetime | None = None
    try:
        authorization = _verify_file(
            intake_root,
            str(record["authorization_record"]),
            str(record["authorization_record_sha256"]),
        )
        verified["authorization_record_sha256"] = _sha256(authorization)
    except Exception as error:
        errors.append(f"authorization:{type(error).__name__}:{error}")
    try:
        consent = _verify_file(
            intake_root,
            str(record["editor_consent_record"]),
            str(record["editor_consent_record_sha256"]),
        )
        verified["editor_consent_record_sha256"] = _sha256(consent)
    except Exception as error:
        errors.append(f"consent:{type(error).__name__}:{error}")
    try:
        assignment = _verify_file(
            intake_root,
            str(record["assignment_record"]),
            str(record["assignment_record_sha256"]),
        )
        assignment_frozen = _verify_assignment(assignment, record)
        verified["assignment_record_sha256"] = _sha256(assignment)
    except Exception as error:
        errors.append(f"assignment:{type(error).__name__}:{error}")

    source: np.ndarray | None = None
    source_metadata: dict[str, Any] | None = None
    try:
        source_path = _verify_file(
            intake_root, str(record["source_path"]), str(record["source_sha256"])
        )
        source, source_metadata = _load_rgb(source_path)
        verified["source_sha256"] = _sha256(source_path)
    except Exception as error:
        errors.append(f"source:{type(error).__name__}:{error}")

    result: dict[str, Any] = {
        "line_number": line_number,
        "attempt_id": attempt_id,
        "attempt_status": str(record["status"]),
        "record_origin": str(record["record_origin"]),
        "source_group_id": str(record["source_group_id"]),
        "editor_id": str(record["editor_id"]),
        "edit_type": str(record["edit_type"]),
        "source_order_key": source_order_key(str(record["source_sha256"])),
        "verified_hashes": verified,
        "source_image": source_metadata,
        "edited_image": None,
        "intent_mask": None,
        "exact_changed_pixels": None,
        "exact_changed_fraction": None,
        "outside_intent_changed_pixels": None,
        "exact_mask_path": None,
        "exact_mask_sha256": None,
        "visual_review_status": "not_applicable",
        "errors": errors,
        "model_score_read": False,
        "training_started": False,
        "paper_evidence": False,
    }
    if record["status"] != "complete":
        result["registration_status"] = (
            "retained_noncomplete" if not errors else "rejected"
        )
        return result

    exact_change: np.ndarray | None = None
    completed: datetime | None = None
    try:
        started = _parse_utc(str(record["edit_started_utc"]))
        completed = _parse_utc(str(record["edit_completed_utc"]))
        if completed < started:
            raise ValueError("edit_completed_utc precedes edit_started_utc")
        if assignment_frozen is not None and assignment_frozen > started:
            raise ValueError("assignment was frozen after editing started")
    except Exception as error:
        errors.append(f"timestamps:{type(error).__name__}:{error}")

    try:
        layered = _verify_file(
            intake_root,
            str(record["layered_project_path"]),
            str(record["layered_project_sha256"]),
        )
        if layered.suffix.lower() not in LAYERED_SUFFIXES:
            raise ValueError(f"unsupported layered-project suffix: {layered.suffix}")
        verified["layered_project_sha256"] = _sha256(layered)
    except Exception as error:
        errors.append(f"layered_project:{type(error).__name__}:{error}")

    try:
        edited_path = _verify_file(
            intake_root, str(record["edited_path"]), str(record["edited_sha256"])
        )
        edited, edited_metadata = _load_rgb(edited_path)
        result["edited_image"] = edited_metadata
        verified["edited_sha256"] = _sha256(edited_path)
        if source is None or source_metadata is None:
            raise ValueError("source image unavailable")
        if edited.shape != source.shape:
            raise ValueError("edited/source canvas dimensions differ")
        if edited_metadata["icc_profile_sha256"] != source_metadata["icc_profile_sha256"]:
            raise ValueError("edited/source ICC profiles differ")
        exact_change = np.any(edited != source, axis=2)
    except Exception as error:
        errors.append(f"edited:{type(error).__name__}:{error}")

    try:
        mask_path = _verify_file(
            intake_root,
            str(record["intent_mask_path"]),
            str(record["intent_mask_sha256"]),
        )
        intent_mask, mask_metadata = _load_binary_mask(mask_path)
        result["intent_mask"] = mask_metadata
        verified["intent_mask_sha256"] = _sha256(mask_path)
        if source is None or intent_mask.shape != source.shape[:2]:
            raise ValueError("intent-mask/source canvas dimensions differ")
        if exact_change is None:
            raise ValueError("exact changed-pixel mask unavailable")
        changed_pixels = int(exact_change.sum())
        changed_fraction = float(changed_pixels / exact_change.size)
        outside = int(np.logical_and(exact_change, ~intent_mask).sum())
        result["exact_changed_pixels"] = changed_pixels
        result["exact_changed_fraction"] = changed_fraction
        result["outside_intent_changed_pixels"] = outside
        if changed_pixels < 32:
            errors.append("exact_changed_pixels_below_32")
        if changed_fraction > 0.25:
            errors.append("exact_changed_fraction_above_0_25")
        if outside:
            errors.append("changed_pixels_outside_intent_mask")
    except Exception as error:
        errors.append(f"intent_mask:{type(error).__name__}:{error}")

    visual_errors = _review_errors(record, review)
    if review is not None and completed is not None:
        try:
            if _parse_utc(str(review["review_completed_utc"])) < completed:
                visual_errors.append("visual_review_precedes_edit_completion")
        except Exception as error:
            visual_errors.append(f"visual_review_timestamp:{type(error).__name__}:{error}")
    errors.extend(visual_errors)
    result["visual_review_status"] = "passed" if not visual_errors else "failed"
    if not errors and exact_change is not None:
        exact_path = exact_mask_root / _safe_mask_name(attempt_id)
        _write_exact_mask(exact_path, exact_change)
        result["exact_mask_path"] = str(exact_path.relative_to(project_root))
        result["exact_mask_sha256"] = _sha256(exact_path)
        result["registration_status"] = "admitted"
    else:
        result["registration_status"] = "rejected"
    return result


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    experiment = config["experiment"]
    runtime = config["runtime"]
    if bool(experiment.get("paper_evidence")):
        raise ValueError("human-edit intake audit cannot be paper evidence")
    if bool(runtime["training_authorized"]):
        raise ValueError("human-edit intake audit cannot authorize training")
    if bool(runtime["model_scoring_authorized"]):
        raise ValueError("human-edit intake audit cannot authorize model scoring")
    if bool(runtime["copy_intake_artifacts_authorized"]):
        raise ValueError("human-edit intake artifacts must remain read-only")
    if not bool(runtime["verify_artifact_hashes"]):
        raise ValueError("human-edit intake requires artifact hash verification")
    if not bool(runtime["collection_authorized"]):
        raise ValueError(
            "human-edit collection is closed; create a dated checksum-bound amendment"
        )
    expected_origin = str(runtime["expected_record_origin"])
    if expected_origin not in {"human_submission", "synthetic_fixture"}:
        raise ValueError(f"unsupported expected_record_origin: {expected_origin}")
    assignment = config["assignment"]
    if assignment["policy_version"] != "human_edit_round_robin_20260726_v1":
        raise ValueError("unsupported human-edit assignment policy")
    editor_roster = sorted(str(value) for value in assignment["editor_roster"])
    if len(editor_roster) < 2 or len(editor_roster) != len(set(editor_roster)):
        raise ValueError("assignment editor roster requires at least two unique IDs")
    if any(not re.fullmatch(r"editor:[a-z0-9][a-z0-9_-]{2,31}", value) for value in editor_roster):
        raise ValueError("assignment editor roster contains an invalid editor ID")
    variants_per_source = int(assignment["variants_per_source"])
    if variants_per_source != 2:
        raise ValueError("Toy-3 assignment policy requires exactly two variants per source")
    required_edit_types = [str(value) for value in config["gates"]["edit_types"]]
    if required_edit_types != ["text_replacement", "copy_move", "local_removal"]:
        raise ValueError("human-edit assignment edit-type order changed")

    for binding in config["bindings"]:
        bound_path = _resolve(project_root, str(binding["path"]))
        if _sha256(bound_path) != str(binding["sha256"]):
            raise ValueError(f"bound artifact changed: {bound_path}")

    pair_schema_path = _resolve(project_root, str(config["schemas"]["pair_manifest"]))
    review_schema_path = _resolve(project_root, str(config["schemas"]["visual_review"]))
    pair_schema = _read_json(pair_schema_path)
    review_schema = _read_json(review_schema_path)
    Draft202012Validator.check_schema(pair_schema)
    Draft202012Validator.check_schema(review_schema)
    pair_validator = Draft202012Validator(pair_schema, format_checker=FormatChecker())
    review_validator = Draft202012Validator(
        review_schema, format_checker=FormatChecker()
    )

    paths = config["paths"]
    intake_root = _intake_root(project_root, paths)
    manifest_path = _intake_path(intake_root, str(paths["input_manifest"]))
    review_path = _intake_path(intake_root, str(paths["visual_reviews"]))
    manifest_hash = paths.get("expected_input_manifest_sha256")
    review_hash = paths.get("expected_visual_reviews_sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise ValueError("input manifest is not checksum-bound")
    if not isinstance(review_hash, str) or len(review_hash) != 64:
        raise ValueError("visual-review manifest is not checksum-bound")
    if _sha256(manifest_path) != manifest_hash:
        raise ValueError("input manifest SHA-256 changed")
    if _sha256(review_path) != review_hash:
        raise ValueError("visual-review manifest SHA-256 changed")

    entries = _read_jsonl_accounted(manifest_path)
    valid_records: list[tuple[int, dict[str, Any]]] = []
    audit_rows: list[dict[str, Any]] = []
    for entry in entries:
        line = int(entry["line_number"])
        record = entry["record"]
        if record is None:
            audit_rows.append(
                {
                    "line_number": line,
                    "attempt_id": f"unparseable-line:{line}",
                    "registration_status": "rejected",
                    "errors": [str(entry["error"])],
                    "model_score_read": False,
                    "training_started": False,
                    "paper_evidence": False,
                }
            )
            continue
        validation = _schema_errors(pair_validator, record)
        if validation:
            audit_rows.append(
                {
                    "line_number": line,
                    "attempt_id": str(record.get("attempt_id", f"invalid-line:{line}")),
                    "registration_status": "rejected",
                    "errors": validation,
                    "model_score_read": False,
                    "training_started": False,
                    "paper_evidence": False,
                }
            )
            continue
        valid_records.append((line, record))

    attempt_counts = Counter(str(row["attempt_id"]) for _, row in valid_records)
    sequence_counts = Counter(
        (str(row["source_group_id"]), int(row["attempt_sequence"]))
        for _, row in valid_records
    )
    variant_counts = Counter(
        (str(row["source_group_id"]), str(row["edited_sha256"]))
        for _, row in valid_records
        if row["status"] == "complete"
    )
    reviews, review_errors, review_input_rows = _review_rows(
        review_path, review_validator
    )
    ordered_source_hashes = sorted(
        {str(row["source_sha256"]) for _, row in valid_records}, key=source_order_key
    )
    source_ranks = {value: index for index, value in enumerate(ordered_source_hashes)}
    exact_mask_base = _output_path(project_root, str(paths["exact_mask_root"]))
    try:
        exact_mask_base.relative_to(intake_root)
    except ValueError:
        pass
    else:
        raise ValueError("exact-mask output root must not be inside read-only intake")
    exact_mask_root = exact_mask_base / f"{manifest_hash[:16]}_{review_hash[:16]}"

    for line, record in valid_records:
        assignment_index = (
            source_ranks[str(record["source_sha256"])] * variants_per_source
            + int(record["attempt_sequence"])
        )
        audit_rows.append(
            _audit_record(
                record=record,
                line_number=line,
                project_root=project_root,
                intake_root=intake_root,
                expected_origin=expected_origin,
                review=reviews.get(str(record["attempt_id"])),
                exact_mask_root=exact_mask_root,
                duplicate_attempt=attempt_counts[str(record["attempt_id"])] > 1,
                duplicate_sequence=sequence_counts[
                    (str(record["source_group_id"]), int(record["attempt_sequence"]))
                ]
                > 1,
                duplicate_variant=(
                    record["status"] == "complete"
                    and variant_counts[
                        (str(record["source_group_id"]), str(record["edited_sha256"]))
                    ]
                    > 1
                ),
                expected_editor_id=editor_roster[assignment_index % len(editor_roster)],
                expected_edit_type=required_edit_types[
                    assignment_index % len(required_edit_types)
                ],
            )
        )

    audit_rows.sort(key=lambda row: int(row["line_number"]))
    canonical_records = [row for _, row in valid_records]
    canonical_records.sort(
        key=lambda row: (
            source_order_key(str(row["source_sha256"])),
            int(row["attempt_sequence"]),
            str(row["attempt_id"]),
        )
    )
    registered_path = _output_path(project_root, str(paths["registered_manifest"]))
    audit_path = _output_path(project_root, str(paths["audit_records"]))
    summary_path = _output_path(project_root, str(paths["summary"]))
    _write_jsonl(registered_path, canonical_records)
    _write_jsonl(audit_path, audit_rows)

    admitted_ids = {
        str(row["attempt_id"])
        for row in audit_rows
        if row["registration_status"] == "admitted"
    }
    admitted_records = [
        row for _, row in valid_records if str(row["attempt_id"]) in admitted_ids
    ]
    admitted_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in admitted_records:
        admitted_groups[str(row["source_group_id"])].append(row)
    edit_counts = Counter(str(row["edit_type"]) for row in admitted_records)
    counts_in_order = [edit_counts[value] for value in required_edit_types]
    complete_input_count = sum(
        row["status"] == "complete" for _, row in valid_records
    )
    noncomplete_input_count = sum(
        row["status"] != "complete" for _, row in valid_records
    )
    metadata_rejected = sum(
        row["registration_status"] == "rejected"
        and row.get("attempt_status") in {"failed", "abandoned"}
        for row in audit_rows
    )
    extra_reviews = sorted(set(reviews) - {str(row["attempt_id"]) for _, row in valid_records})
    checks = {
        "all_manifest_rows_accounted": len(audit_rows) == len(entries),
        "all_manifest_rows_schema_valid": len(valid_records) == len(entries),
        "all_complete_records_admitted": len(admitted_records) == complete_input_count,
        "noncomplete_records_retained_without_metadata_rejection": metadata_rejected == 0,
        "no_failed_or_abandoned_toy_assignments": noncomplete_input_count == 0,
        "no_review_manifest_errors": not review_errors,
        "no_extra_review_records": not extra_reviews,
        "expected_source_groups": len(admitted_groups)
        == int(config["gates"]["expected_source_groups"]),
        "minimum_complete_pairs": len(admitted_records)
        >= int(config["gates"]["minimum_complete_pairs"]),
        "minimum_pairs_per_source_group": bool(admitted_groups)
        and all(
            len(rows) >= int(config["gates"]["minimum_pairs_per_source_group"])
            for rows in admitted_groups.values()
        ),
        "minimum_distinct_editors": len(
            {str(row["editor_id"]) for row in admitted_records}
        )
        >= int(config["gates"]["minimum_distinct_editors"]),
        "minimum_distinct_editors_per_group": bool(admitted_groups)
        and all(
            len({str(row["editor_id"]) for row in rows})
            >= int(config["gates"]["minimum_distinct_editors_per_group"])
            for rows in admitted_groups.values()
        ),
        "all_edit_types_present": all(edit_counts[value] > 0 for value in required_edit_types),
        "edit_type_balance": bool(counts_in_order)
        and max(counts_in_order) - min(counts_in_order)
        <= int(config["gates"]["maximum_edit_type_count_imbalance"]),
        "record_origin_matches_config": all(
            row["record_origin"] == expected_origin for _, row in valid_records
        ),
    }
    engineering_passed = all(checks.values())
    actual_human_records = bool(admitted_records) and all(
        row["record_origin"] == "human_submission" for row in admitted_records
    )
    human_evidence_toy3_passed = (
        engineering_passed
        and actual_human_records
        and str(experiment["stage"]) == "toy3"
    )
    collection_expansion = (
        human_evidence_toy3_passed
        and bool(runtime["allow_collection_expansion_after_pass"])
    )
    if engineering_passed and expected_origin == "synthetic_fixture":
        status = "synthetic_toy3_engineering_gate_passed_human_evidence_closed"
    elif human_evidence_toy3_passed:
        status = (
            "human_toy3_audit_passed_expansion_authorized"
            if collection_expansion
            else "human_toy3_audit_passed_expansion_closed"
        )
    else:
        status = f"{expected_origin}_toy3_engineering_gate_failed"

    summary = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": status,
        "stage": str(experiment["stage"]),
        "paper_evidence": False,
        "input": {
            "manifest": str(paths["input_manifest"]),
            "manifest_sha256": manifest_hash,
            "manifest_rows": len(entries),
            "visual_reviews": str(paths["visual_reviews"]),
            "visual_reviews_sha256": review_hash,
            "visual_review_rows": review_input_rows,
        },
        "registration": {
            "schema_valid_rows": len(valid_records),
            "complete_input_records": complete_input_count,
            "noncomplete_input_records_retained": noncomplete_input_count,
            "admitted_complete_records": len(admitted_records),
            "rejected_records": sum(
                row["registration_status"] == "rejected" for row in audit_rows
            ),
            "source_groups_admitted": len(admitted_groups),
            "edit_type_counts": dict(sorted(edit_counts.items())),
            "editor_count": len({str(row["editor_id"]) for row in admitted_records}),
            "exact_masks_written": sum(
                row.get("exact_mask_sha256") is not None for row in audit_rows
            ),
            "registered_manifest": str(registered_path.relative_to(project_root)),
            "registered_manifest_sha256": _sha256(registered_path),
            "audit_records": str(audit_path.relative_to(project_root)),
            "audit_records_sha256": _sha256(audit_path),
        },
        "visual_review": {
            "valid_indexed_records": len(reviews),
            "errors": review_errors,
            "extra_attempt_ids": extra_reviews,
        },
        "decision": {
            "checks": checks,
            "engineering_gate_passed": engineering_passed,
            "human_evidence_toy3_passed": human_evidence_toy3_passed,
            "synthetic_fixture_counts_as_human_evidence": False,
            "collection_expansion_authorized": collection_expansion,
            "model_scoring_authorized": False,
            "training_authorized": False,
        },
    }
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path.relative_to(project_root))
    summary["summary_sha256"] = _sha256(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Register and audit checksum-bound prospective human-edited pairs "
            "without training or model scoring."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
