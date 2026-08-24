from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import yaml

from pairtrace_doc.pipelines.train_student_100 import _sha256, _write_json


REQUIRED_INPUT_PACKET_MEMBERS = {
    "integrity.json",
    "qualitative_audit_heatmaps.pdf",
    "qualitative_audit_inputs.pdf",
    "qualitative_audit_protocol.md",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


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


def _write_zip_member(
    archive: zipfile.ZipFile,
    name: str,
    payload: bytes,
    fixed_timestamp: tuple[int, int, int, int, int, int],
) -> None:
    info = zipfile.ZipInfo(name, fixed_timestamp)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def _completion_readme(case_count: int, source_zip_sha256: str) -> str:
    return f"""# PairTrace-Doc completed qualitative human review

This archive is the immutable completion companion to the original reviewer
input packet. The original input packet remains unchanged and correctly records
`human_review_complete=false`; this archive records the completed and validated
review for all {case_count} frozen cases.

Original reviewer input packet SHA-256: `{source_zip_sha256}`.

The review is descriptive qualitative and limitation evidence. It did not
change case membership, quantitative metrics, model weights, checkpoints,
thresholds, or operating points. The archive verifies procedural separation
from case selection and model/threshold changes, but contains no signed
attestation establishing that the reviewer had no prior project involvement.
It must therefore not be described as an independent review.

Use `completion_integrity.json` to verify the source chain and every completion
member. `input_packet_integrity.json` is the unchanged integrity record embedded
in the original reviewer packet.
"""


def _review_status_amendment() -> str:
    return """# Reviewer-status amendment

The frozen upstream worksheet, summary, report, protocol, and input-packet
integrity use the word `independent` because that was the original procedural
requirement. They contain no signed no-prior-involvement attestation from
reviewer `alirio`, so this v2 completion archive does not reassert that label.

The defensible description is: **second human reviewer using frozen cases,
models, thresholds, schema, and display scales; reviewer independence not
established**. The upstream files are retained byte-for-byte to preserve the
audit chain, and this amendment governs their interpretation.
"""


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("human-review archive config must be a mapping")
    experiment = config["experiment"]
    if bool(experiment["paper_evidence"]):
        raise ValueError("archiving a review cannot create quantitative paper evidence")
    runtime = config["runtime"]
    if runtime["device"] != "cpu":
        raise ValueError("human-review archiving must be CPU-only")
    prohibited = (
        "model_inference_authorized",
        "model_training_authorized",
        "metric_computation_authorized",
        "threshold_selection_authorized",
        "sample_replacement_authorized",
        "human_review_editing_authorized",
    )
    if any(bool(runtime[name]) for name in prohibited):
        raise ValueError("human-review archiving crossed an evidence boundary")

    specification = config["input"]
    sources: dict[str, Path] = {}
    for key, label in (
        ("review_packet_zip", "original reviewer input packet"),
        ("review_packet_manifest", "original reviewer packet manifest"),
        ("freeze_manifest", "completed review freeze manifest"),
        ("frozen_worksheet", "frozen completed worksheet"),
        ("summary", "completed review summary"),
        ("report", "completed review report"),
    ):
        sources[key] = _verify_file(
            project_root,
            str(specification[key]),
            str(specification[f"expected_{key}_sha256"]),
            label,
        )

    packet_manifest = _read_json(sources["review_packet_manifest"])
    freeze_manifest = _read_json(sources["freeze_manifest"])
    worksheet = _read_json(sources["frozen_worksheet"])
    summary = _read_json(sources["summary"])
    if packet_manifest.get("status") != "independent_human_review_packet_ready":
        raise ValueError("original reviewer packet manifest is not ready")
    if packet_manifest.get("human_review_complete") is not False:
        raise ValueError("original reviewer packet must remain an incomplete input packet")
    if (
        packet_manifest.get("output", {}).get("review_packet_zip_sha256")
        != specification["expected_review_packet_zip_sha256"]
    ):
        raise ValueError("original reviewer packet manifest ZIP link changed")
    if freeze_manifest.get("status") != "independent_human_review_complete":
        raise ValueError("completed review freeze manifest is not complete")
    if freeze_manifest.get("human_review_complete") is not True:
        raise ValueError("completed review freeze manifest is not marked complete")
    if worksheet.get("status") != "human_review_complete":
        raise ValueError("frozen worksheet status is not complete")
    if worksheet.get("human_review_complete") is not True:
        raise ValueError("frozen worksheet is not marked complete")
    if summary.get("status") != "independent_human_review_complete":
        raise ValueError("completed review summary is not complete")

    frozen_sha256 = _sha256(sources["frozen_worksheet"])
    if freeze_manifest.get("output", {}).get("frozen_worksheet_sha256") != frozen_sha256:
        raise ValueError("freeze manifest worksheet link changed")
    if summary.get("frozen_worksheet_sha256") != frozen_sha256:
        raise ValueError("summary worksheet link changed")
    case_ids = summary.get("case_ids")
    if not isinstance(case_ids, list) or not case_ids:
        raise ValueError("completed review summary has no frozen case IDs")
    if int(summary.get("case_count", -1)) != len(case_ids):
        raise ValueError("completed review summary case count is inconsistent")

    with zipfile.ZipFile(sources["review_packet_zip"]) as input_archive:
        if input_archive.testzip() is not None:
            raise ValueError("original reviewer input packet ZIP failed integrity check")
        names = set(input_archive.namelist())
        missing = REQUIRED_INPUT_PACKET_MEMBERS - names
        if missing:
            raise ValueError(f"original reviewer packet is missing members: {sorted(missing)}")
        input_integrity_bytes = input_archive.read("integrity.json")
        input_integrity = json.loads(input_integrity_bytes)
        if input_integrity.get("packet_role") != "independent_human_review_input_only":
            raise ValueError("embedded input-packet integrity role changed")
        if input_integrity.get("human_review_complete") is not False:
            raise ValueError("embedded input-packet integrity must remain incomplete")
        carried_members = {
            "qualitative_audit_protocol.md": input_archive.read(
                "qualitative_audit_protocol.md"
            ),
            "qualitative_audit_inputs.pdf": input_archive.read(
                "qualitative_audit_inputs.pdf"
            ),
            "qualitative_audit_heatmaps.pdf": input_archive.read(
                "qualitative_audit_heatmaps.pdf"
            ),
        }

    source_zip_sha256 = _sha256(sources["review_packet_zip"])
    member_payloads = {
        "README_COMPLETED_REVIEW.md": _completion_readme(
            len(case_ids), source_zip_sha256
        ).encode("utf-8"),
        "REVIEW_STATUS_AMENDMENT.md": _review_status_amendment().encode("utf-8"),
        **carried_members,
        "human_review_worksheet_completed.json": sources[
            "frozen_worksheet"
        ].read_bytes(),
        "human_review_summary.json": sources["summary"].read_bytes(),
        "human_review_result.md": sources["report"].read_bytes(),
        "input_packet_integrity.json": input_integrity_bytes,
    }
    implementation_path = Path(__file__).resolve()
    implementation_label = (
        str(implementation_path.relative_to(project_root))
        if implementation_path.is_relative_to(project_root)
        else f"src/pairtrace_doc/pipelines/{implementation_path.name}"
    )
    completion_integrity = {
        "schema_version": 1,
        "packet_role": "completed_human_review_archive",
        "paper_evidence_role": "descriptive_qualitative_and_limitation_evidence",
        "human_review_complete": True,
        "human_review_was_machine_generated": False,
        "reviewer_independence_machine_verifiable": False,
        "reviewer_independence_claim": "not_established",
        "reviewer_no_prior_involvement_attestation_present": False,
        "procedural_freeze_verified": True,
        "legacy_upstream_independence_labels_reasserted": False,
        "quantitative_claims_changed": False,
        "case_membership_changed": False,
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "source_artifacts": {
            key: {
                "path": str(path.relative_to(project_root)),
                "sha256": _sha256(path),
            }
            for key, path in sources.items()
        },
        "implementation": {
            "path": implementation_label,
            "sha256": _sha256(implementation_path),
        },
        "member_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(member_payloads.items())
        },
    }
    member_payloads["completion_integrity.json"] = _json_bytes(completion_integrity)

    paths = config["paths"]
    archive_path = _resolve(project_root, str(paths["completed_review_zip"]))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp_values = tuple(int(value) for value in config["archive"]["fixed_zip_timestamp"])
    if len(timestamp_values) != 6:
        raise ValueError("fixed_zip_timestamp must contain six integers")
    fixed_timestamp = timestamp_values
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        for name in sorted(member_payloads):
            _write_zip_member(archive, name, member_payloads[name], fixed_timestamp)
    temporary.replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("completed human-review archive failed integrity check")
        if archive.namelist() != sorted(member_payloads):
            raise ValueError("completed human-review archive member order changed")

    result = {
        "schema_version": 1,
        "experiment": experiment,
        "status": "completed_human_review_archived",
        "paper_evidence": False,
        "human_review_complete": True,
        "original_input_packet_mutated": False,
        "model_inference_performed": False,
        "new_scientific_metrics_computed": False,
        "threshold_selection_used": False,
        "sample_replacement_used": False,
        "case_count": len(case_ids),
        "input": {
            "config": str(config_path.relative_to(project_root)),
            "config_sha256": _sha256(config_path),
            **completion_integrity["source_artifacts"],
        },
        "output": {
            "completed_review_zip": str(archive_path.relative_to(project_root)),
            "completed_review_zip_sha256": _sha256(archive_path),
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
