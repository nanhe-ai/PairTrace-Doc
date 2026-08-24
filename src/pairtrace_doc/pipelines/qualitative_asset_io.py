from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from pairtrace_doc.pipelines.materialize_fantasyid_facelondon import (
    RASTERIZER_VERSION,
    _png_bytes,
    _rasterize_box_mask,
)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _archive_members(cases: list[dict[str, Any]]) -> set[str]:
    members: set[str] = set()
    for case in cases:
        for field in (
            "candidate",
            "correct_reference",
            "correct_same_device_reference",
            "selected_reference",
            "wrong_reference",
            "mask",
        ):
            reference = case.get(field)
            if isinstance(reference, dict) and reference.get("archive_member"):
                members.add(str(reference["archive_member"]))
        generation = case.get("mask_generation")
        if isinstance(generation, dict):
            members.add(str(generation["source_metadata_archive_member"]))
    return members


def _read_selected_tar_members(
    archive_path: Path, requested: set[str]
) -> dict[str, bytes]:
    if any(
        PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        for name in requested
    ):
        raise ValueError("unsafe FantasyID archive member requested")
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if member.name not in requested:
                continue
            if not member.isfile() or member.name in payloads:
                raise ValueError(
                    f"invalid or duplicate selected archive member: {member.name}"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"selected archive member is unreadable: {member.name}")
            payloads[member.name] = extracted.read()
            if len(payloads) == len(requested):
                break
    missing = sorted(requested - set(payloads))
    if missing:
        raise ValueError(f"selected FantasyID archive members are missing: {missing}")
    return payloads


def _reference_bytes(
    reference: dict[str, Any], scratch: Path, payloads: dict[str, bytes]
) -> bytes:
    path = _resolve(scratch, str(reference["path"]))
    if path.is_file():
        payload = path.read_bytes()
    else:
        member = reference.get("archive_member")
        if not member:
            raise FileNotFoundError(path)
        payload = payloads[str(member)]
    if _hash_bytes(payload) != str(reference["sha256"]):
        raise ValueError(f"selected input SHA-256 changed: {reference['path']}")
    return payload


def _mask_for_case(
    case: dict[str, Any],
    scratch: Path,
    payloads: dict[str, bytes],
    candidate_size: tuple[int, int],
) -> Image.Image:
    reference = case["mask"]
    path = _resolve(scratch, str(reference["path"]))
    if path.is_file():
        payload = path.read_bytes()
    else:
        generation = case.get("mask_generation")
        if not isinstance(generation, dict):
            raise FileNotFoundError(path)
        if str(generation["rasterizer_version"]) != RASTERIZER_VERSION:
            raise ValueError("FantasyID mask rasterizer version changed")
        metadata_payload = payloads[str(generation["source_metadata_archive_member"])]
        if _hash_bytes(metadata_payload) != str(generation["source_metadata_sha256"]):
            raise ValueError("selected FantasyID metadata SHA-256 changed")
        metadata = json.loads(metadata_payload)
        mask_array = _rasterize_box_mask(
            metadata,
            int(generation["annotation_width"]),
            int(generation["annotation_height"]),
        )
        payload = _png_bytes(mask_array)
    if _hash_bytes(payload) != str(reference["sha256"]):
        raise ValueError(f"selected mask SHA-256 changed: {reference['path']}")
    with Image.open(io.BytesIO(payload)) as handle:
        handle.load()
        mask = handle.convert("L")
    if mask.size != candidate_size:
        raise ValueError(
            f"candidate/mask geometry mismatch for {case['case_id']}: "
            f"{candidate_size} versus {mask.size}"
        )
    return mask.point(lambda value: 255 if value > 0 else 0)
