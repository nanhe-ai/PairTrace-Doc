from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from PIL import Image

from pairtrace_doc.pipelines.freeze_fantasyid_facelondon_88 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_json,
    _write_jsonl,
)
from pairtrace_doc.pipelines.materialize_fantasyid_facelondon import (
    _cache_payload,
    _stage_rows,
)


DEVICES = ("huawei", "iphone15pro", "scan")


def _cross_device_reference(
    source_group_id: str, candidate_device: str, seed: int
) -> str:
    if candidate_device not in DEVICES:
        raise ValueError(f"unsupported FantasyID device: {candidate_device}")
    candidates = [device for device in DEVICES if device != candidate_device]
    return min(
        candidates,
        key=lambda device: (
            hashlib.sha256(
                f"{seed}|{source_group_id}|{candidate_device}|{device}".encode(
                    "utf-8"
                )
            ).hexdigest(),
            device,
        ),
    )


def _reference_members(row: dict[str, Any], reference_device: str) -> tuple[str, str]:
    stem = str(row["source_card_stem"])
    prefix = f"FantasyID/train/bonafide/{reference_device}/{stem}"
    return f"{prefix}.jpg", f"{prefix}.json"


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment = config["experiment"]
    runtime = config["runtime"]
    if bool(experiment["paper_evidence"]):
        raise ValueError("cross-device materialization cannot be paper evidence")
    if not bool(runtime["selected_cross_device_reference_read_authorized"]):
        raise ValueError("cross-device reference read was not authorized")
    if any(
        bool(runtime.get(name))
        for name in (
            "model_training_authorized",
            "model_evaluation_authorized",
            "final_reserve_read_authorized",
            "full_archive_extraction_authorized",
        )
    ):
        raise ValueError("cross-device materialization crossed an evidence boundary")

    protocol_path = _resolve(project_root, experiment["protocol"])
    if _sha256(protocol_path) != str(experiment["expected_protocol_sha256"]):
        raise ValueError("cross-device protocol changed")
    inputs = config["input"]
    manifest_path = _resolve(project_root, inputs["manifest"])
    if _sha256(manifest_path) != str(inputs["expected_manifest_sha256"]):
        raise ValueError("FantasyID materialized manifest changed")
    rows = _read_jsonl(manifest_path)
    if len(rows) != int(inputs["expected_full_groups"]):
        raise ValueError("FantasyID full materialized capacity changed")
    if {str(row["fantasyid_facelondon_freeze_id"]) for row in rows} != {
        str(inputs["expected_freeze_id"])
    }:
        raise ValueError("FantasyID freeze ID changed")
    stage = str(experiment["stage"])
    selected = _stage_rows(rows, stage)
    if len(selected) != int(inputs["expected_stage_groups"]):
        raise ValueError("cross-device stage group count changed")
    if any(
        row.get("paper_evidence") is not False
        or row.get("materialization_status") != "ok"
        or row.get("mask_semantics") != "box_mask_not_pixel_accurate"
        for row in selected
    ):
        raise ValueError("FantasyID input evidence boundary changed")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    archive_path = _resolve(scratch, paths["archive"])
    if archive_path.stat().st_size != int(inputs["expected_archive_bytes"]):
        raise ValueError("FantasyID archive size changed")
    if _sha256(archive_path) != str(inputs["expected_archive_sha256"]):
        raise ValueError("FantasyID archive SHA-256 changed")

    assignment_seed = int(config["assignment"]["seed"])
    assignments: dict[str, dict[str, str]] = {}
    required_members: set[str] = set()
    for row in selected:
        group = str(row["source_group_id"])
        reference_device = _cross_device_reference(
            group, str(row["device"]), assignment_seed
        )
        image_member, metadata_member = _reference_members(row, reference_device)
        assignments[group] = {
            "reference_device": reference_device,
            "image_member": image_member,
            "metadata_member": metadata_member,
        }
        required_members.update((image_member, metadata_member))
    if any(
        PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        for name in required_members
    ):
        raise ValueError("unsafe cross-device archive member requested")

    member_payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if member.name not in required_members:
                continue
            if not member.isfile() or member.name in member_payloads:
                raise ValueError(f"invalid cross-device member: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"unreadable cross-device member: {member.name}")
            member_payloads[member.name] = extracted.read()
    missing = sorted(required_members - set(member_payloads))
    if missing:
        raise ValueError(f"cross-device members are missing: {missing}")

    cache_root = _resolve(scratch, paths["cache_dir"])
    cache_hits = 0
    cache_writes = 0
    output_rows: list[dict[str, Any]] = []
    for row in selected:
        assignment = assignments[str(row["source_group_id"])]
        image_bytes = member_payloads[assignment["image_member"]]
        metadata_bytes = member_payloads[assignment["metadata_member"]]
        metadata = json.loads(metadata_bytes)
        with Image.open(io.BytesIO(image_bytes)) as handle:
            if handle.format != "JPEG":
                raise ValueError("cross-device reference member is not JPEG")
            width, height = handle.size
            handle.verify()
        person = metadata.get("person_info", {})
        if (
            str(person.get("face_db")),
            str(person.get("face_id")),
        ) != (str(row["face_db"]), str(row["face_id"])):
            raise ValueError(
                f"cross-device identity changed: {row['source_group_id']}"
            )
        cached_image = _cache_payload(cache_root, "images", image_bytes, ".jpg")
        cached_metadata = _cache_payload(
            cache_root, "metadata", metadata_bytes, ".json"
        )
        cache_hits += int(cached_image[2]) + int(cached_metadata[2])
        cache_writes += int(not cached_image[2]) + int(not cached_metadata[2])
        output_rows.append(
            {
                **row,
                "cross_device_assignment_seed": assignment_seed,
                "cross_device_reference_device": assignment["reference_device"],
                "cross_device_reference_member": assignment["image_member"],
                "cross_device_reference_metadata_member": assignment[
                    "metadata_member"
                ],
                "cross_device_reference": str(
                    cached_image[0].relative_to(scratch)
                ),
                "cross_device_reference_sha256": cached_image[1],
                "cross_device_reference_metadata": str(
                    cached_metadata[0].relative_to(scratch)
                ),
                "cross_device_reference_metadata_sha256": cached_metadata[1],
                "cross_device_reference_width": width,
                "cross_device_reference_height": height,
                "cross_device_reference_read": True,
                "cross_device_materialization_status": "ok",
                "post_final_diagnostic": True,
                "development_only": True,
                "final_reserve_read": False,
                "paper_evidence": False,
            }
        )

    unique_paths = {
        _resolve(scratch, row[field])
        for row in output_rows
        for field in (
            "cross_device_reference",
            "cross_device_reference_metadata",
        )
    }
    cache_bytes = sum(path.stat().st_size for path in unique_paths)
    if cache_bytes > int(config["storage"]["maximum_cross_reference_cache_bytes"]):
        raise ValueError("cross-device reference cache exceeds frozen budget")
    output_manifest = _resolve(project_root, paths["output_manifest"])
    output_summary = _resolve(project_root, paths["output_summary"])
    _write_jsonl(output_manifest, output_rows)
    summary = {
        "experiment": experiment,
        "status": f"fantasyid_cross_device_{stage}_materialized",
        "paper_evidence": False,
        "development_only": True,
        "post_final_diagnostic": True,
        "final_reserve_read": False,
        "input_manifest": str(manifest_path.relative_to(project_root)),
        "input_manifest_sha256": _sha256(manifest_path),
        "archive_sha256": str(inputs["expected_archive_sha256"]),
        "groups": len(output_rows),
        "assignment_seed": assignment_seed,
        "directed_device_transition_counts": {},
        "cache": {
            "unique_files": len(unique_paths),
            "bytes": cache_bytes,
            "budget_bytes": int(
                config["storage"]["maximum_cross_reference_cache_bytes"]
            ),
            "hits": cache_hits,
            "writes": cache_writes,
            "content_addressed": True,
        },
        "output_manifest": str(output_manifest.relative_to(project_root)),
        "output_manifest_sha256": _sha256(output_manifest),
    }
    transition_counts: dict[str, int] = {}
    for row in output_rows:
        transition = f"{row['device']}->{row['cross_device_reference_device']}"
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
    summary["directed_device_transition_counts"] = dict(
        sorted(transition_counts.items())
    )
    _write_json(output_summary, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
