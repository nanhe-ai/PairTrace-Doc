from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from PIL import Image

from pairtrace_doc.pipelines.audit_fantasyid import _valid_altered_rectangle
from pairtrace_doc.pipelines.freeze_fantasyid_facelondon_88 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_json,
    _write_jsonl,
)


RASTERIZER_VERSION = "fantasyid_box_mask_v1_floor_ceil"


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stage_rows(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    field_by_stage = {
        "toy3": "toy3_member",
        "pilot20": "pilot20_member",
        "full88": "full88_member",
    }
    if stage not in field_by_stage:
        raise ValueError(f"unsupported FantasyID materialization stage: {stage}")
    field = field_by_stage[stage]
    return sorted(
        (row for row in rows if row.get(field) is True),
        key=lambda row: int(row["selection_index"]),
    )


def _altered_rectangles(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        region
        for region in metadata.get("regions", [])
        if region.get("region_attributes", {}).get("region_provenance") == "altered"
    ]


def _rasterize_box_mask(
    metadata: dict[str, Any], width: int, height: int
) -> np.ndarray:
    rectangles = _altered_rectangles(metadata)
    if not rectangles:
        raise ValueError("attack metadata has no altered rectangles")
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in rectangles:
        if not _valid_altered_rectangle(region, width, height):
            raise ValueError("attack metadata contains an invalid altered rectangle")
        shape = region["shape_attributes"]
        x1 = int(np.floor(float(shape["x"])))
        y1 = int(np.floor(float(shape["y"])))
        x2 = int(np.ceil(float(shape["x"]) + float(shape["width"])))
        y2 = int(np.ceil(float(shape["y"]) + float(shape["height"])))
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("rasterized rectangle exceeds image bounds")
        mask[y1:y2, x1:x2] = 255
    if not np.any(mask):
        raise ValueError("rasterized box mask is empty")
    return mask


def _cache_payload(
    cache_root: Path, kind: str, payload: bytes, suffix: str
) -> tuple[Path, str, bool]:
    digest = _hash_bytes(payload)
    path = cache_root / kind / digest[:2] / f"{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    hit = path.is_file()
    if hit:
        if _sha256(path) != digest:
            raise ValueError(f"content-addressed cache collision: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return path, digest, hit


def _png_bytes(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment = config["experiment"]
    if bool(experiment["paper_evidence"]):
        raise ValueError("FantasyID materialization cannot be paper evidence")
    protocol_path = _resolve(project_root, experiment["protocol"])
    if _sha256(protocol_path) != experiment["expected_protocol_sha256"]:
        raise ValueError("FantasyID external-development protocol changed")
    runtime = config["runtime"]
    if not bool(runtime["selected_image_read_authorized"]):
        raise ValueError("selected FantasyID image read was not authorized")
    if any(
        bool(runtime.get(name))
        for name in (
            "model_training_authorized",
            "model_evaluation_authorized",
            "final_reserve_read_authorized",
            "full_archive_extraction_authorized",
        )
    ):
        raise ValueError("FantasyID materialization crossed an evidence boundary")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    archive_path = _resolve(scratch, paths["archive"])
    manifest_path = _resolve(project_root, config["input"]["manifest"])
    if _sha256(manifest_path) != config["input"]["expected_manifest_sha256"]:
        raise ValueError("FantasyID FaceLondon manifest changed")
    rows = _read_jsonl(manifest_path)
    freeze_ids = {str(row["fantasyid_facelondon_freeze_id"]) for row in rows}
    if freeze_ids != {str(config["input"]["expected_freeze_id"])}:
        raise ValueError("FantasyID FaceLondon freeze ID changed")
    if len(rows) != int(config["input"]["expected_full_groups"]):
        raise ValueError("FantasyID FaceLondon full capacity changed")
    stage = str(experiment["stage"])
    selected = _stage_rows(rows, stage)
    if len(selected) != int(config["input"]["expected_stage_groups"]):
        raise ValueError("FantasyID materialization stage size changed")
    if archive_path.stat().st_size != int(config["input"]["expected_archive_bytes"]):
        raise ValueError("FantasyID archive size changed")
    if _sha256(archive_path) != config["input"]["expected_archive_sha256"]:
        raise ValueError("FantasyID archive SHA-256 changed")

    required_members = {
        str(row[field])
        for row in selected
        for field in (
            "forged_image_member",
            "forged_metadata_member",
            "authentic_image_member",
            "authentic_metadata_member",
        )
    }
    if any(
        PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        for name in required_members
    ):
        raise ValueError("unsafe member requested")
    member_payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if member.name not in required_members:
                continue
            if not member.isfile() or member.name in member_payloads:
                raise ValueError(f"invalid or duplicate selected member: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"selected member is unreadable: {member.name}")
            member_payloads[member.name] = extracted.read()
    missing_members = sorted(required_members - set(member_payloads))
    if missing_members:
        raise ValueError(f"selected archive members are missing: {missing_members}")

    cache_root = _resolve(scratch, paths["cache_dir"])
    cache_hits = 0
    cache_writes = 0
    output_rows: list[dict[str, Any]] = []
    for row in selected:
        forged_bytes = member_payloads[str(row["forged_image_member"])]
        authentic_bytes = member_payloads[str(row["authentic_image_member"])]
        forged_json_bytes = member_payloads[str(row["forged_metadata_member"])]
        authentic_json_bytes = member_payloads[str(row["authentic_metadata_member"])]
        forged_metadata = json.loads(forged_json_bytes)
        authentic_metadata = json.loads(authentic_json_bytes)
        with Image.open(io.BytesIO(forged_bytes)) as handle:
            if handle.format != "JPEG":
                raise ValueError("selected forged member is not JPEG")
            forged_size = handle.size
            handle.verify()
        with Image.open(io.BytesIO(authentic_bytes)) as handle:
            if handle.format != "JPEG":
                raise ValueError("selected authentic member is not JPEG")
            authentic_size = handle.size
            handle.verify()
        expected_size = (int(row["annotation_width"]), int(row["annotation_height"]))
        if forged_size != expected_size or authentic_size != expected_size:
            raise ValueError(f"selected pair geometry changed: {row['sample_id']}")
        forged_person = forged_metadata.get("person_info", {})
        authentic_person = authentic_metadata.get("person_info", {})
        expected_identity = (str(row["face_db"]), str(row["face_id"]))
        if (
            str(forged_person.get("face_db")),
            str(forged_person.get("face_id")),
        ) != expected_identity or (
            str(authentic_person.get("face_db")),
            str(authentic_person.get("face_id")),
        ) != expected_identity:
            raise ValueError(f"selected pair identity changed: {row['sample_id']}")
        mask = _rasterize_box_mask(forged_metadata, *expected_size)

        cached: dict[str, tuple[Path, str, bool]] = {
            "forged": _cache_payload(cache_root, "images", forged_bytes, ".jpg"),
            "authentic": _cache_payload(
                cache_root, "images", authentic_bytes, ".jpg"
            ),
            "forged_metadata": _cache_payload(
                cache_root, "metadata", forged_json_bytes, ".json"
            ),
            "authentic_metadata": _cache_payload(
                cache_root, "metadata", authentic_json_bytes, ".json"
            ),
            "mask": _cache_payload(cache_root, "box_masks", _png_bytes(mask), ".png"),
        }
        cache_hits += sum(item[2] for item in cached.values())
        cache_writes += sum(not item[2] for item in cached.values())
        output_rows.append(
            {
                **row,
                "materialization_stage": stage,
                "materialization_status": "ok",
                "image": str(cached["forged"][0].relative_to(scratch)),
                "image_sha256": cached["forged"][1],
                "authentic": str(cached["authentic"][0].relative_to(scratch)),
                "authentic_sha256": cached["authentic"][1],
                "forged_metadata": str(
                    cached["forged_metadata"][0].relative_to(scratch)
                ),
                "forged_metadata_sha256": cached["forged_metadata"][1],
                "authentic_metadata": str(
                    cached["authentic_metadata"][0].relative_to(scratch)
                ),
                "authentic_metadata_sha256": cached["authentic_metadata"][1],
                "mask": str(cached["mask"][0].relative_to(scratch)),
                "mask_sha256": cached["mask"][1],
                "mask_positive_pixels": int(np.count_nonzero(mask)),
                "mask_rasterizer_version": RASTERIZER_VERSION,
                "selected_image_read": True,
                "final_reserve_read": False,
                "paper_evidence": False,
            }
        )

    unique_cache_paths = {
        _resolve(scratch, row[field])
        for row in output_rows
        for field in (
            "image",
            "authentic",
            "forged_metadata",
            "authentic_metadata",
            "mask",
        )
    }
    cache_bytes = sum(path.stat().st_size for path in unique_cache_paths)
    if cache_bytes > int(config["storage"]["maximum_selected_cache_bytes"]):
        raise ValueError("selected FantasyID cache exceeds its frozen budget")
    output_manifest = _resolve(project_root, paths["output_manifest"])
    output_summary = _resolve(project_root, paths["output_summary"])
    _write_jsonl(output_manifest, output_rows)
    summary = {
        "experiment": experiment,
        "status": f"fantasyid_facelondon_{stage}_materialized",
        "paper_evidence": False,
        "development_only": True,
        "selected_image_read": True,
        "full_archive_extracted": False,
        "final_reserve_read": False,
        "input": {
            "manifest": str(manifest_path.relative_to(project_root)),
            "manifest_sha256": _sha256(manifest_path),
            "archive_sha256": config["input"]["expected_archive_sha256"],
            "freeze_id": next(iter(freeze_ids)),
        },
        "selection": {
            "groups": len(output_rows),
            "archive_members_read": len(member_payloads),
            "selection_indices": [int(row["selection_index"]) for row in output_rows],
        },
        "cache": {
            "root": str(cache_root.relative_to(scratch)),
            "unique_files": len(unique_cache_paths),
            "bytes": cache_bytes,
            "budget_bytes": int(config["storage"]["maximum_selected_cache_bytes"]),
            "hits": cache_hits,
            "writes": cache_writes,
            "content_addressed": True,
        },
        "output": {
            "path": str(output_manifest.relative_to(project_root)),
            "sha256": _sha256(output_manifest),
        },
        "runtime": runtime,
    }
    _write_json(output_summary, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
