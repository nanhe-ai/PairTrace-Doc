from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import yaml


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _hashes(handle: BinaryIO) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - required to verify the publisher checksum
    sha256 = hashlib.sha256()
    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
        md5.update(chunk)
        sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _is_safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _member_type(member: tarfile.TarInfo) -> str:
    if member.isfile():
        return "file"
    if member.isdir():
        return "directory"
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    return "other"


def _dataset_identity(path: PurePosixPath) -> dict[str, str] | None:
    parts = path.parts
    if len(parts) == 5 and parts[0] == "FantasyID" and parts[2] == "bonafide":
        split, device, filename = parts[1], parts[3], parts[4]
        method = "none"
        category = "bonafide"
    elif len(parts) == 6 and parts[0] == "FantasyID" and parts[2] == "attack":
        split, method, device, filename = parts[1], parts[3], parts[4], parts[5]
        category = "attack"
    else:
        return None
    if split not in {"train", "val", "test"}:
        return None
    return {
        "split": split,
        "category": category,
        "method": method,
        "device": device,
        "filename": filename,
        "stem": PurePosixPath(filename).stem,
        "cohort": f"{split}/{category}/{method}",
        "relative_path": str(PurePosixPath(*parts[1:])),
    }


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"could not read archive member: {member.name}")
    return extracted.read()


def _jpeg_dimensions(handle: BinaryIO) -> tuple[int, int]:
    if handle.read(2) != b"\xff\xd8":
        raise ValueError("missing JPEG start-of-image marker")
    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while True:
        prefix = handle.read(1)
        while prefix and prefix != b"\xff":
            prefix = handle.read(1)
        if not prefix:
            raise ValueError("JPEG ended before a start-of-frame marker")
        marker_byte = handle.read(1)
        while marker_byte == b"\xff":
            marker_byte = handle.read(1)
        if not marker_byte:
            raise ValueError("truncated JPEG marker")
        marker = marker_byte[0]
        if marker in {0x01, *range(0xD0, 0xDA)}:
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            raise ValueError("truncated JPEG segment length")
        segment_length = int.from_bytes(length_bytes, "big")
        if segment_length < 2:
            raise ValueError("invalid JPEG segment length")
        if marker in start_of_frame_markers:
            header = handle.read(5)
            if len(header) != 5:
                raise ValueError("truncated JPEG start-of-frame header")
            height = int.from_bytes(header[1:3], "big")
            width = int.from_bytes(header[3:5], "big")
            if width <= 0 or height <= 0:
                raise ValueError("invalid JPEG dimensions")
            return width, height
        handle.seek(segment_length - 2, 1)


def _valid_altered_rectangle(region: dict[str, Any], width: int, height: int) -> bool:
    shape = region.get("shape_attributes", {})
    if shape.get("name") != "rect":
        return False
    values = [shape.get(key) for key in ("x", "y", "width", "height")]
    if not all(isinstance(value, (int, float)) for value in values):
        return False
    x, y, box_width, box_height = values
    return (
        x >= 0
        and y >= 0
        and box_width > 0
        and box_height > 0
        and x + box_width <= width
        and y + box_height <= height
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    archive_path = _resolve(scratch, paths["archive"])
    member_records_path = _resolve(project_root, paths["member_records"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    for output in (member_records_path, summary_path, log_path):
        output.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    release = config["release"]
    expected_bytes = int(release["expected_bytes"])
    expected_md5 = str(release["expected_md5"]).lower()
    report: dict[str, Any] = {
        "audit": config["audit"],
        "release": release,
        "archive": str(archive_path.relative_to(scratch)),
        "status": "missing",
        "paper_evidence": False,
    }

    if not archive_path.is_file():
        _write_json(summary_path, report)
        raise FileNotFoundError(archive_path)

    actual_bytes = archive_path.stat().st_size
    report["actual_bytes"] = actual_bytes
    report["size_matches"] = actual_bytes == expected_bytes
    if actual_bytes != expected_bytes:
        report["status"] = "incomplete_or_wrong_size"
        _write_json(summary_path, report)
        logging.error("archive size mismatch report=%s", report)
        raise RuntimeError(
            f"FantasyID archive size mismatch: expected {expected_bytes}, got {actual_bytes}"
        )

    with archive_path.open("rb") as handle:
        actual_md5, actual_sha256 = _hashes(handle)
    report["actual_md5"] = actual_md5
    report["actual_sha256"] = actual_sha256
    report["md5_matches"] = actual_md5 == expected_md5
    if actual_md5 != expected_md5:
        report["status"] = "checksum_failed"
        _write_json(summary_path, report)
        logging.error("archive checksum mismatch report=%s", report)
        raise RuntimeError(
            f"FantasyID MD5 mismatch: expected {expected_md5}, got {actual_md5}"
        )

    member_types: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    roots: Counter[str] = Counter()
    regular_bytes = 0
    unsafe_members: list[str] = []
    documentation_candidates: list[str] = []
    split_components: Counter[str] = Counter()
    member_count = 0
    documentation_tokens = ("license", "readme", "citation", "metadata", "manifest")
    split_tokens = {"train", "training", "val", "validation", "test", "testing"}
    image_records: dict[str, dict[str, str]] = {}
    image_dimensions: dict[str, tuple[int, int]] = {}
    jpeg_errors: list[dict[str, str]] = []
    metadata_records: dict[str, dict[str, Any]] = {}
    altered_regions_by_path: dict[str, list[dict[str, Any]]] = {}
    csv_manifests: dict[str, list[dict[str, str]]] = {}
    json_errors: list[dict[str, str]] = []
    readme_text = ""

    with member_records_path.open("w", encoding="utf-8") as records_handle:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                member_count += 1
                safe = _is_safe_member(member.name)
                if not safe:
                    unsafe_members.append(member.name)
                path = PurePosixPath(member.name)
                kind = _member_type(member)
                member_types[kind] += 1
                if path.parts:
                    roots[path.parts[0]] += 1
                if member.isfile():
                    regular_bytes += member.size
                    suffixes[path.suffix.lower() or "<none>"] += 1
                lowered_parts = tuple(part.lower() for part in path.parts)
                for part in lowered_parts:
                    if part in split_tokens:
                        split_components[part] += 1
                lowered_name = path.name.lower()
                if any(token in lowered_name for token in documentation_tokens):
                    documentation_candidates.append(member.name)
                identity = _dataset_identity(path)
                record = {
                    "name": member.name,
                    "type": kind,
                    "size": member.size,
                    "suffix": path.suffix.lower(),
                    "safe_path": safe,
                }
                if identity is not None:
                    record["dataset_identity"] = identity
                    relative_path = identity["relative_path"]
                    if member.isfile() and path.suffix.lower() == ".jpg":
                        image_records[relative_path] = identity
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            jpeg_errors.append(
                                {"path": relative_path, "error": "member unreadable"}
                            )
                        else:
                            try:
                                image_dimensions[relative_path] = _jpeg_dimensions(
                                    extracted
                                )
                            except (OSError, ValueError) as error:
                                jpeg_errors.append(
                                    {"path": relative_path, "error": str(error)}
                                )
                    elif member.isfile() and path.suffix.lower() == ".json":
                        try:
                            metadata = json.loads(_read_member(archive, member))
                            regions = metadata.get("regions", [])
                            altered_regions = [
                                region
                                for region in regions
                                if region.get("region_attributes", {}).get(
                                    "region_provenance"
                                )
                                == "altered"
                            ]
                            cropping = metadata.get("cropping_info", {}) or metadata.get(
                                "cropping_info-altered-recaptured", {}
                            )
                            width = int(cropping.get("resulted_cropped_image_width", 0))
                            height = int(cropping.get("resulted_cropped_image_height", 0))
                            valid_altered = sum(
                                _valid_altered_rectangle(region, width, height)
                                for region in altered_regions
                            )
                            person_info = metadata.get("person_info", {})
                            summary = {
                                **identity,
                                "region_count": len(regions),
                                "altered_region_count": len(altered_regions),
                                "valid_altered_rectangle_count": valid_altered,
                                "annotation_width": width,
                                "annotation_height": height,
                                "face_db": person_info.get("face_db"),
                                "face_id": person_info.get("face_id"),
                            }
                            metadata_records[relative_path] = summary
                            altered_regions_by_path[relative_path] = altered_regions
                            record["metadata_summary"] = summary
                        except (json.JSONDecodeError, TypeError, ValueError) as error:
                            json_errors.append(
                                {"path": relative_path, "error": str(error)}
                            )
                if member.isfile() and member.name in {
                    "FantasyID/train.csv",
                    "FantasyID/val.csv",
                    "FantasyID/test.csv",
                }:
                    split = path.stem
                    text = _read_member(archive, member).decode("utf-8-sig")
                    csv_manifests[split] = list(csv.DictReader(io.StringIO(text)))
                elif member.isfile() and member.name == "FantasyID/README.md":
                    readme_text = _read_member(archive, member).decode("utf-8")
                records_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    image_paths = set(image_records)
    metadata_paths = set(metadata_records)
    missing_metadata = sorted(
        path
        for path in image_paths
        if str(PurePosixPath(path).with_suffix(".json")) not in metadata_paths
    )
    metadata_without_image = sorted(
        path
        for path in metadata_paths
        if str(PurePosixPath(path).with_suffix(".jpg")) not in image_paths
    )
    annotation_dimension_mismatches: list[dict[str, Any]] = []
    for metadata_path, metadata in metadata_records.items():
        image_path = str(PurePosixPath(metadata_path).with_suffix(".jpg"))
        image_size = image_dimensions.get(image_path)
        annotation_size = (
            int(metadata["annotation_width"]),
            int(metadata["annotation_height"]),
        )
        if image_size is not None and image_size != annotation_size:
            annotation_dimension_mismatches.append(
                {
                    "path": metadata_path,
                    "image_width": image_size[0],
                    "image_height": image_size[1],
                    "annotation_width": annotation_size[0],
                    "annotation_height": annotation_size[1],
                }
            )

    image_counts: Counter[str] = Counter()
    image_pixels: Counter[str] = Counter()
    source_cards: defaultdict[str, set[str]] = defaultdict(set)
    devices: defaultdict[str, set[str]] = defaultdict(set)
    for identity in image_records.values():
        cohort = identity["cohort"]
        image_counts[cohort] += 1
        width, height = image_dimensions.get(identity["relative_path"], (0, 0))
        image_pixels[cohort] += width * height
        source_cards[cohort].add(identity["stem"])
        devices[cohort].add(identity["device"])

    attack_pairs: Counter[str] = Counter()
    missing_attack_pairs: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    missing_attack_pair_face_databases: defaultdict[str, Counter[str]] = defaultdict(
        Counter
    )
    pair_dimension_mismatches: defaultdict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    pair_identity_mismatches: defaultdict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    cross_split_attack_pairs = 0
    for attack_path, identity in image_records.items():
        if identity["category"] != "attack":
            continue
        source_split = (
            "train"
            if identity["split"] == "test" and identity["method"] == "digital_3"
            else identity["split"]
        )
        source_path = str(
            PurePosixPath(
                source_split,
                "bonafide",
                identity["device"],
                identity["filename"],
            )
        )
        cohort = identity["cohort"]
        if source_path in image_paths:
            attack_pairs[cohort] += 1
            cross_split_attack_pairs += int(source_split != identity["split"])
            if image_dimensions.get(attack_path) != image_dimensions.get(source_path):
                pair_dimension_mismatches[cohort].append(
                    {
                        "attack": attack_path,
                        "authentic": source_path,
                        "attack_size": image_dimensions.get(attack_path),
                        "authentic_size": image_dimensions.get(source_path),
                    }
                )
            attack_metadata = metadata_records.get(
                str(PurePosixPath(attack_path).with_suffix(".json")), {}
            )
            source_metadata = metadata_records.get(
                str(PurePosixPath(source_path).with_suffix(".json")), {}
            )
            attack_identity = (
                attack_metadata.get("face_db"),
                attack_metadata.get("face_id"),
            )
            source_identity = (
                source_metadata.get("face_db"),
                source_metadata.get("face_id"),
            )
            if attack_identity != source_identity:
                pair_identity_mismatches[cohort].append(
                    {
                        "attack": attack_path,
                        "authentic": source_path,
                        "attack_identity": attack_identity,
                        "authentic_identity": source_identity,
                    }
                )
        else:
            missing_attack_pairs[cohort].append(
                {"attack": attack_path, "expected_authentic": source_path}
            )
            attack_metadata = metadata_records.get(
                str(PurePosixPath(attack_path).with_suffix(".json")), {}
            )
            face_db = attack_metadata.get("face_db")
            if face_db:
                missing_attack_pair_face_databases[cohort][str(face_db)] += 1

    attack_annotation_counts: Counter[str] = Counter()
    altered_region_counts: Counter[str] = Counter()
    invalid_altered_rectangles: Counter[str] = Counter()
    invalid_altered_rectangle_paths: defaultdict[str, list[str]] = defaultdict(list)
    invalid_image_space_rectangles: Counter[str] = Counter()
    invalid_image_space_rectangle_paths: defaultdict[str, list[str]] = defaultdict(
        list
    )
    attacks_without_altered_regions: defaultdict[str, list[str]] = defaultdict(list)
    face_databases: Counter[str] = Counter()
    bonafide_face_ids: defaultdict[str, set[str]] = defaultdict(set)
    for metadata_path, metadata in metadata_records.items():
        face_db = metadata.get("face_db")
        if face_db:
            face_databases[str(face_db)] += 1
        if metadata["category"] == "bonafide" and metadata.get("face_id"):
            bonafide_face_ids[metadata["split"]].add(str(metadata["face_id"]))
        if metadata["category"] != "attack":
            continue
        cohort = metadata["cohort"]
        attack_annotation_counts[cohort] += 1
        altered = int(metadata["altered_region_count"])
        valid = int(metadata["valid_altered_rectangle_count"])
        altered_region_counts[cohort] += altered
        invalid_altered_rectangles[cohort] += altered - valid
        if altered != valid:
            invalid_altered_rectangle_paths[cohort].append(metadata_path)
        image_path = str(PurePosixPath(metadata_path).with_suffix(".jpg"))
        image_width, image_height = image_dimensions.get(image_path, (0, 0))
        valid_in_image = sum(
            _valid_altered_rectangle(region, image_width, image_height)
            for region in altered_regions_by_path.get(metadata_path, [])
        )
        invalid_image_space_rectangles[cohort] += altered - valid_in_image
        if altered != valid_in_image:
            invalid_image_space_rectangle_paths[cohort].append(metadata_path)
        if altered == 0:
            attacks_without_altered_regions[cohort].append(metadata_path)

    expected_attack_types = {
        "digital_1": "face_text",
        "digital_2": "face_text",
        "digital_3": "text",
        "facedancer": "face",
        "textdiffuserft_bfei": "text",
    }
    csv_audit: dict[str, Any] = {}
    for split, rows in sorted(csv_manifests.items()):
        manifest_paths = {row.get("path", "") for row in rows}
        archive_split_paths = {
            path for path, identity in image_records.items() if identity["split"] == split
        }
        label_mismatches: list[str] = []
        for row in rows:
            identity = image_records.get(row.get("path", ""))
            if identity is None:
                continue
            expected_is_attack = str(identity["category"] == "attack")
            expected_type = expected_attack_types.get(identity["method"], "none")
            if (
                row.get("is_attack") != expected_is_attack
                or row.get("attack_type") != expected_type
            ):
                label_mismatches.append(row.get("path", ""))
        csv_audit[split] = {
            "row_count": len(rows),
            "archive_image_count": len(archive_split_paths),
            "missing_from_csv": sorted(archive_split_paths - manifest_paths)[:20],
            "missing_from_csv_count": len(archive_split_paths - manifest_paths),
            "extra_in_csv": sorted(manifest_paths - archive_split_paths)[:20],
            "extra_in_csv_count": len(manifest_paths - archive_split_paths),
            "label_mismatch_count": len(label_mismatches),
            "label_mismatch_sample": label_mismatches[:20],
        }

    train_bonafide_cards = source_cards.get("train/bonafide/none", set())
    test_bonafide_cards = source_cards.get("test/bonafide/none", set())
    test_attack_overlap: dict[str, int] = {}
    for cohort, cards in source_cards.items():
        if cohort.startswith("test/attack/"):
            test_attack_overlap[cohort.rsplit("/", 1)[-1]] = len(
                cards & train_bonafide_cards
            )

    missing_pair_counts = {
        cohort: len(items) for cohort, items in sorted(missing_attack_pairs.items())
    }
    content_audit = {
        "status": "completed_with_gate_blockers",
        "gate_a_eligible": False,
        "paper_evidence": False,
        "gate_blockers": [
            "annotations are weak altered-region rectangles, not pixel-accurate masks",
            "some altered rectangles exceed image boundaries",
            "some deterministic pairs differ in image dimensions",
            "some digital_3 attacks have no open authentic counterpart",
            "digital_3 reuses open-train source cards",
            "HQ-WMCA, AMFD, and Flickr provenance needs policy resolution",
        ],
        "annotation_type": "altered-region rectangles; rasterizable weak masks, not pixel-accurate masks",
        "dataset_image_count": len(image_records),
        "dataset_metadata_count": len(metadata_records),
        "image_counts_by_cohort": dict(sorted(image_counts.items())),
        "image_pixels_by_cohort": dict(sorted(image_pixels.items())),
        "total_image_pixels": sum(image_pixels.values()),
        "source_card_counts_by_cohort": {
            cohort: len(cards) for cohort, cards in sorted(source_cards.items())
        },
        "source_template_counts_by_cohort": {
            cohort: dict(
                sorted(Counter(card.split("-", 1)[0] for card in cards).items())
            )
            for cohort, cards in sorted(source_cards.items())
        },
        "devices_by_cohort": {
            cohort: sorted(values) for cohort, values in sorted(devices.items())
        },
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata_sample": missing_metadata[:20],
        "metadata_without_image_count": len(metadata_without_image),
        "metadata_without_image_sample": metadata_without_image[:20],
        "json_error_count": len(json_errors),
        "json_error_sample": json_errors[:20],
        "jpeg_dimension_count": len(image_dimensions),
        "jpeg_error_count": len(jpeg_errors),
        "jpeg_error_sample": jpeg_errors[:20],
        "annotation_dimension_mismatch_count": len(
            annotation_dimension_mismatches
        ),
        "annotation_dimension_mismatch_sample": annotation_dimension_mismatches[:20],
        "attack_annotation_counts_by_cohort": dict(
            sorted(attack_annotation_counts.items())
        ),
        "altered_region_counts_by_cohort": dict(sorted(altered_region_counts.items())),
        "invalid_altered_rectangle_counts_by_cohort": dict(
            sorted(invalid_altered_rectangles.items())
        ),
        "images_with_invalid_altered_rectangles_by_cohort": {
            cohort: len(paths)
            for cohort, paths in sorted(invalid_altered_rectangle_paths.items())
        },
        "invalid_altered_rectangle_path_samples": {
            cohort: paths[:20]
            for cohort, paths in sorted(invalid_altered_rectangle_paths.items())
        },
        "invalid_image_space_rectangle_counts_by_cohort": dict(
            sorted(invalid_image_space_rectangles.items())
        ),
        "images_with_invalid_image_space_rectangles_by_cohort": {
            cohort: len(paths)
            for cohort, paths in sorted(invalid_image_space_rectangle_paths.items())
        },
        "invalid_image_space_rectangle_path_samples": {
            cohort: paths[:20]
            for cohort, paths in sorted(invalid_image_space_rectangle_paths.items())
        },
        "attacks_without_altered_regions": {
            cohort: paths[:20]
            for cohort, paths in sorted(attacks_without_altered_regions.items())
        },
        "paired_attack_counts_by_cohort": dict(sorted(attack_pairs.items())),
        "pair_dimension_mismatch_counts_by_cohort": {
            cohort: len(items)
            for cohort, items in sorted(pair_dimension_mismatches.items())
        },
        "pair_dimension_mismatch_samples": {
            cohort: items[:20]
            for cohort, items in sorted(pair_dimension_mismatches.items())
        },
        "pair_identity_mismatch_counts_by_cohort": {
            cohort: len(items)
            for cohort, items in sorted(pair_identity_mismatches.items())
        },
        "pair_identity_mismatch_samples": {
            cohort: items[:20]
            for cohort, items in sorted(pair_identity_mismatches.items())
        },
        "missing_attack_pair_counts_by_cohort": missing_pair_counts,
        "missing_attack_pair_samples": {
            cohort: items[:20] for cohort, items in sorted(missing_attack_pairs.items())
        },
        "missing_attack_pair_face_databases_by_cohort": {
            cohort: dict(sorted(counts.items()))
            for cohort, counts in sorted(missing_attack_pair_face_databases.items())
        },
        "cross_split_attack_pair_count": cross_split_attack_pairs,
        "train_test_bonafide_source_card_overlap_count": len(
            train_bonafide_cards & test_bonafide_cards
        ),
        "test_attack_source_card_overlap_with_train_bonafide": dict(
            sorted(test_attack_overlap.items())
        ),
        "train_test_bonafide_face_id_overlap_count": len(
            bonafide_face_ids.get("train", set())
            & bonafide_face_ids.get("test", set())
        ),
        "face_database_record_counts": dict(sorted(face_databases.items())),
        "csv_manifests": csv_audit,
        "readme_present": bool(readme_text),
        "readme_train_test_cc4_statement": (
            "All images in `train` and `test` folder are released under Creative Commons 4.0 license"
            in readme_text
        ),
        "readme_val_noncommercial_statement": (
            "All images in `val` are released under non-commercial academic license"
            in readme_text
        ),
        "val_archive_member_count": sum(
            count
            for component, count in split_components.items()
            if component in {"val", "validation"}
        ),
    }

    report.update(
        {
            "archive_member_count": member_count,
            "archive_regular_bytes": regular_bytes,
            "member_types": dict(sorted(member_types.items())),
            "suffix_counts": dict(sorted(suffixes.items())),
            "root_counts": dict(sorted(roots.items())),
            "split_component_counts": dict(sorted(split_components.items())),
            "documentation_candidates": sorted(documentation_candidates),
            "unsafe_members": unsafe_members,
            "member_records": str(member_records_path.relative_to(project_root)),
            "content_audit": content_audit,
        }
    )
    if unsafe_members:
        report["status"] = "unsafe_archive_paths"
        _write_json(summary_path, report)
        logging.error("unsafe archive members report=%s", report)
        raise RuntimeError(f"unsafe archive paths found: {unsafe_members[:3]}")

    report["status"] = "integrity_passed"
    _write_json(summary_path, report)
    logging.info("FantasyID archive audit completed report=%s", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the official open FantasyID archive")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
