from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import yaml

from pairtrace_doc.pipelines.train_student_100 import _sha256, _write_json


FIXED_ZIP_TIMESTAMP = (2026, 7, 19, 0, 0, 0)


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


def _validate_blank_worksheet(
    worksheet: dict[str, Any], expected_case_ids: list[str]
) -> None:
    if worksheet.get("schema_version") != 2:
        raise ValueError("human-review worksheet schema changed")
    if worksheet.get("status") != "pending_independent_human_review":
        raise ValueError("human-review worksheet is not pending")
    if worksheet.get("model_heatmaps_available") is not True:
        raise ValueError("human-review worksheet does not expose frozen heatmaps")
    if worksheet.get("human_review_complete") is not False:
        raise ValueError("human-review worksheet is already marked complete")
    reviews = worksheet.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("human-review worksheet reviews must be a list")
    case_ids = [str(review.get("case_id")) for review in reviews]
    if case_ids != expected_case_ids:
        raise ValueError("human-review worksheet case order or membership changed")
    for review in reviews:
        if any(value is not None for key, value in review.items() if key != "case_id"):
            raise ValueError("human-review worksheet contains prefilled review fields")


def _reviewer_readme(expected_case_count: int) -> str:
    return f"""# PairTrace-Doc independent qualitative review

This packet contains {expected_case_count} cases selected by frozen deterministic
rules. Review all cases; do not replace, reorder, omit, or add a case.

## Independence gate

Proceed only if you did not select these cases and did not tune the models,
checkpoints, losses, thresholds, or operating points used to produce them. If
that is not true, return the packet without filling the worksheet.

## Review procedure

1. Open `qualitative_audit_inputs.pdf` and
   `qualitative_audit_heatmaps.pdf` side by side.
2. For each case, fill every null field in `human_review_worksheet.json`.
3. Use only the categorical values listed in the worksheet's
   `allowed_values` object.
4. Write `reviewer_note` as one short, factual, single-line sentence.
5. Use a stable, non-empty `reviewer_identifier` and an ISO-8601 UTC timestamp
   such as `2026-07-19T15:30:00Z` for `reviewed_at_utc`.
6. After all {expected_case_count} rows are complete, change only the top-level
   `status` to `human_review_complete` and `human_review_complete` to `true`.
7. Return the completed JSON file. Do not edit either PDF, the protocol,
   `allowed_values`, case identifiers, instructions, or integrity metadata.

The review is descriptive. It cannot change quantitative results, remove an
unfavorable example, select a new threshold, or promote viewed-development or
weak-box cases to independent confirmation evidence.
"""


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_zip_member(
    archive: zipfile.ZipFile, name: str, payload: bytes
) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("human-review packet config must be a mapping")
    experiment = config["experiment"]
    if bool(experiment["paper_evidence"]):
        raise ValueError("preparing a reviewer packet cannot create paper evidence")
    runtime = config["runtime"]
    if runtime["device"] != "cpu":
        raise ValueError("human-review packet preparation must be CPU-only")
    prohibited = (
        "model_inference_authorized",
        "model_training_authorized",
        "metric_computation_authorized",
        "threshold_selection_authorized",
        "sample_replacement_authorized",
        "human_review_completion_authorized",
    )
    if any(bool(runtime[name]) for name in prohibited):
        raise ValueError("human-review packet preparation crossed an evidence boundary")

    specification = config["input"]
    expected_case_ids = [str(value) for value in specification["expected_case_ids"]]
    expected_case_count = int(specification["expected_case_count"])
    if len(expected_case_ids) != expected_case_count:
        raise ValueError("expected human-review case count is inconsistent")
    sources: dict[str, Path] = {}
    for key, label in (
        ("protocol", "qualitative audit protocol"),
        ("case_manifest", "qualitative case manifest"),
        ("input_render_manifest", "qualitative input render manifest"),
        ("heatmap_render_manifest", "qualitative heatmap render manifest"),
        ("input_packet", "qualitative input packet"),
        ("heatmap_packet", "qualitative heatmap packet"),
        ("blank_worksheet", "blank heatmap-ready worksheet"),
    ):
        sources[key] = _verify_file(
            project_root,
            str(specification[key]),
            str(specification[f"expected_{key}_sha256"]),
            label,
        )
    case_manifest = _read_json(sources["case_manifest"])
    cases = case_manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("qualitative case manifest cases must be a list")
    if [str(case.get("case_id")) for case in cases] != expected_case_ids:
        raise ValueError("qualitative case order or membership changed")
    worksheet = _read_json(sources["blank_worksheet"])
    _validate_blank_worksheet(worksheet, expected_case_ids)

    member_payloads = {
        "README_FOR_REVIEWER.md": _reviewer_readme(expected_case_count).encode(
            "utf-8"
        ),
        "qualitative_audit_protocol.md": sources["protocol"].read_bytes(),
        "qualitative_audit_inputs.pdf": sources["input_packet"].read_bytes(),
        "qualitative_audit_heatmaps.pdf": sources["heatmap_packet"].read_bytes(),
        "human_review_worksheet.json": sources["blank_worksheet"].read_bytes(),
    }
    integrity = {
        "schema_version": 1,
        "packet_role": "independent_human_review_input_only",
        "paper_evidence": False,
        "human_review_complete": False,
        "case_count": expected_case_count,
        "case_ids": expected_case_ids,
        "source_files": {
            key: {
                "path": str(path.relative_to(project_root)),
                "sha256": _sha256(path),
            }
            for key, path in sources.items()
        },
    }
    member_payloads["integrity.json"] = _json_bytes(integrity)

    paths = config["paths"]
    archive_path = _resolve(project_root, str(paths["review_packet_zip"]))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        for name in sorted(member_payloads):
            _write_zip_member(archive, name, member_payloads[name])
    temporary.replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("human-review packet ZIP integrity check failed")
        archived_names = archive.namelist()
        if archived_names != sorted(member_payloads):
            raise ValueError("human-review packet ZIP member order changed")

    implementation_path = Path(__file__).resolve()
    implementation_label = (
        str(implementation_path.relative_to(project_root))
        if implementation_path.is_relative_to(project_root)
        else f"src/pairtrace_doc/pipelines/{implementation_path.name}"
    )
    result = {
        "schema_version": 1,
        "experiment": experiment,
        "status": "independent_human_review_packet_ready",
        "paper_evidence": False,
        "human_review_complete": False,
        "model_inference_performed": False,
        "new_scientific_metrics_computed": False,
        "sample_replacement_used": False,
        "case_count": expected_case_count,
        "case_ids": expected_case_ids,
        "input": {
            "config": str(config_path.relative_to(project_root)),
            "config_sha256": _sha256(config_path),
            "implementation": implementation_label,
            "implementation_sha256": _sha256(implementation_path),
            **integrity["source_files"],
        },
        "output": {
            "review_packet_zip": str(archive_path.relative_to(project_root)),
            "review_packet_zip_sha256": _sha256(archive_path),
            "members": {
                name: {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
                for name, payload in sorted(member_payloads.items())
            },
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
