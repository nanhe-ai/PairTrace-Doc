from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from pairtrace_doc.pipelines.audit_aiforge import _pixel_sha256
from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)


def _fingerprints(image: np.ndarray, hash_size: int) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("fingerprint input must be uint8 HWC RGB")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    phash_input = cv2.resize(
        gray,
        (hash_size * 4, hash_size * 4),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    coefficients = cv2.dct(phash_input)[:hash_size, :hash_size].reshape(-1)
    threshold = float(np.median(coefficients[1:]))
    phash = coefficients >= threshold
    dhash_input = cv2.resize(
        gray,
        (hash_size + 1, hash_size),
        interpolation=cv2.INTER_AREA,
    )
    dhash = (dhash_input[:, 1:] >= dhash_input[:, :-1]).reshape(-1)
    return phash.astype(bool), dhash.astype(bool)


def _hamming(left: np.ndarray, right: np.ndarray) -> int:
    if left.shape != right.shape or left.dtype != bool or right.dtype != bool:
        raise ValueError("Hamming inputs must be matched boolean vectors")
    return int(np.count_nonzero(left ^ right))


def _relative_aspect_difference(
    left_shape: tuple[int, int], right_shape: tuple[int, int]
) -> float:
    left_ratio = left_shape[1] / left_shape[0]
    right_ratio = right_shape[1] / right_shape[0]
    return float(abs(left_ratio - right_ratio) / max(left_ratio, right_ratio))


def _development_inventory(
    project_root: Path, scratch: Path, specifications: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    groups: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for specification in specifications:
        manifest_path = _resolve(project_root, str(specification["path"]))
        digest = _sha256(manifest_path)
        if digest != str(specification["expected_sha256"]):
            raise ValueError(f"development manifest SHA-256 changed: {manifest_path}")
        hashes[str(manifest_path.relative_to(project_root))] = digest
        for row in _read_jsonl(manifest_path):
            group = str(row["source_group_id"])
            candidate = {
                "source_group_id": group,
                "source_dataset": str(row["source_dataset"]),
                "path": str(row["authentic"]),
                "encoded_sha256": str(row["authentic_sha256"]),
                "declared_pixel_sha256": str(row["authentic_pixel_sha256"]),
            }
            if group in groups and groups[group] != candidate:
                raise ValueError(f"development source group identity disagrees: {group}")
            groups[group] = candidate
    return groups, hashes


def _final_inventory(
    project_root: Path, specification: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], str]:
    manifest_path = _resolve(project_root, str(specification["path"]))
    digest = _sha256(manifest_path)
    if digest != str(specification["expected_sha256"]):
        raise ValueError("final manifest SHA-256 changed")
    rows = [
        row
        for row in _read_jsonl(manifest_path)
        if str(row["evaluation_role"]) == str(specification["authentic_role"])
    ]
    groups = {
        str(row["source_group_id"]): {
            "source_group_id": str(row["source_group_id"]),
            "source_dataset": str(row["source_dataset"]),
            "path": str(row["image"]),
            "encoded_sha256": str(row["image_sha256"]),
            "declared_pixel_sha256": None,
        }
        for row in rows
    }
    if len(rows) != len(groups) or len(groups) != int(specification["expected_groups"]):
        raise ValueError("final authentic group inventory changed")
    return groups, digest


def _decode_inventory(
    inventory: dict[str, dict[str, Any]], scratch: Path, hash_size: int
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for group, record in sorted(inventory.items()):
        path = _resolve(scratch, str(record["path"]))
        if _sha256(path) != record["encoded_sha256"]:
            raise ValueError(f"authentic encoded SHA-256 changed: {path}")
        with Image.open(path) as handle:
            image = np.asarray(handle.convert("RGB"))
        pixel_sha256 = _pixel_sha256(image)
        declared = record["declared_pixel_sha256"]
        if declared is not None and pixel_sha256 != declared:
            raise ValueError(f"authentic decoded pixel SHA-256 changed: {path}")
        phash, dhash = _fingerprints(image, hash_size)
        result[group] = {
            **record,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "decoded_pixel_sha256": pixel_sha256,
            "phash": phash,
            "dhash": dhash,
        }
    return result


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"phash", "dhash", "declared_pixel_sha256"}
    }


def _report(summary: dict[str, Any]) -> str:
    return f"""# Template and near-duplicate leakage audit: final 96

Status: `{summary['status']}`.

- Development/fitting source groups: {summary['development_groups']}.
- Final source groups: {summary['final_groups']}.
- Cross-role source-group overlaps: {summary['exact_overlap']['source_group_ids']}.
- Cross-role encoded-file hash overlaps: {summary['exact_overlap']['encoded_sha256']}.
- Cross-role decoded-pixel hash overlaps: {summary['exact_overlap']['decoded_pixel_sha256']}.
- Combined high-priority perceptual flags: {summary['perceptual_screen']['final_groups_with_high_priority_match']}.
- pHash-only high-similarity flags: {summary['perceptual_screen']['final_groups_with_phash_only_match']}.

Exact identity checks {'passed' if summary['exact_identity_gate_passed'] else 'failed'}.
Perceptual flags are screening evidence, not proof of common template identity.
{'A fixed manual/contact-sheet review is still required before claiming template-level separation.' if summary['manual_template_review_required'] else 'No combined perceptual flag fired, but this does not prove semantic template separation.'}
The audit is post-final and does not restore the consumed reserve to unseen
status. It read authentic images only, modified no data, and used no masks,
predictions, model outputs, or thresholds.
"""


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if any(
        bool(runtime.get(key))
        for key in (
            "gpu_launch_authorized",
            "prediction_read_allowed",
            "mask_read_allowed",
            "model_training_authorized",
            "selection_authorized",
            "modify_images_authorized",
        )
    ):
        raise ValueError("template audit crossed its read-only evidence boundary")
    protocol_path = _resolve(project_root, str(config["experiment"]["protocol"]))
    if _sha256(protocol_path) != str(config["experiment"]["expected_protocol_sha256"]):
        raise ValueError("template audit protocol SHA-256 changed")
    scratch = Path(
        os.environ.get(
            str(config["paths"]["scratch_env"]),
            str(_resolve(project_root, str(config["paths"]["scratch_default"]))),
        )
    ).resolve()
    development, development_hashes = _development_inventory(
        project_root, scratch, config["development_inputs"]
    )
    final, final_hash = _final_inventory(project_root, config["final_input"])
    screening = config["screening"]
    hash_size = int(screening["phash_size"])
    decoded_development = _decode_inventory(development, scratch, hash_size)
    decoded_final = _decode_inventory(final, scratch, hash_size)

    group_overlap = sorted(set(decoded_development) & set(decoded_final))
    development_encoded = {record["encoded_sha256"] for record in decoded_development.values()}
    final_encoded = {record["encoded_sha256"] for record in decoded_final.values()}
    development_pixels = {record["decoded_pixel_sha256"] for record in decoded_development.values()}
    final_pixels = {record["decoded_pixel_sha256"] for record in decoded_final.values()}
    encoded_overlap = sorted(development_encoded & final_encoded)
    pixel_overlap = sorted(development_pixels & final_pixels)

    nearest_rows: list[dict[str, Any]] = []
    by_dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for final_group, final_record in sorted(decoded_final.items()):
        comparisons: list[dict[str, Any]] = []
        for development_group, development_record in decoded_development.items():
            phash_distance = _hamming(final_record["phash"], development_record["phash"])
            dhash_distance = _hamming(final_record["dhash"], development_record["dhash"])
            aspect_difference = _relative_aspect_difference(
                (final_record["height"], final_record["width"]),
                (development_record["height"], development_record["width"]),
            )
            same_dataset = final_record["source_dataset"] == development_record["source_dataset"]
            comparisons.append(
                {
                    "development_source_group_id": development_group,
                    "development_source_dataset": development_record["source_dataset"],
                    "development_path": development_record["path"],
                    "development_height": development_record["height"],
                    "development_width": development_record["width"],
                    "same_source_dataset": same_dataset,
                    "phash_distance": phash_distance,
                    "dhash_distance": dhash_distance,
                    "aspect_ratio_relative_difference": aspect_difference,
                }
            )
        global_nearest = min(
            comparisons,
            key=lambda item: (
                item["phash_distance"],
                item["dhash_distance"],
                item["aspect_ratio_relative_difference"],
                item["development_source_group_id"],
            ),
        )
        same_dataset_comparisons = [item for item in comparisons if item["same_source_dataset"]]
        within_nearest = min(
            same_dataset_comparisons,
            key=lambda item: (
                item["phash_distance"],
                item["dhash_distance"],
                item["aspect_ratio_relative_difference"],
                item["development_source_group_id"],
            ),
        )
        high_priority = [
            item
            for item in same_dataset_comparisons
            if item["phash_distance"] <= int(screening["phash_high_priority_max_distance"])
            and item["dhash_distance"] <= int(screening["dhash_high_priority_max_distance"])
            and item["aspect_ratio_relative_difference"]
            <= float(screening["aspect_ratio_relative_difference_max"])
        ]
        phash_only = [
            item
            for item in same_dataset_comparisons
            if item["phash_distance"] <= int(screening["phash_only_high_similarity_max_distance"])
        ]
        high_priority.sort(
            key=lambda item: (
                item["phash_distance"],
                item["dhash_distance"],
                item["aspect_ratio_relative_difference"],
                item["development_source_group_id"],
            )
        )
        phash_only.sort(
            key=lambda item: (
                item["phash_distance"],
                item["dhash_distance"],
                item["development_source_group_id"],
            )
        )
        dataset = str(final_record["source_dataset"])
        by_dataset_counts[dataset]["final_groups"] += 1
        by_dataset_counts[dataset]["high_priority"] += bool(high_priority)
        by_dataset_counts[dataset]["phash_only"] += bool(phash_only)
        nearest_rows.append(
            {
                "final_source_group_id": final_group,
                "final_source_dataset": dataset,
                "final_path": final_record["path"],
                "final_height": final_record["height"],
                "final_width": final_record["width"],
                "global_nearest": global_nearest,
                "within_dataset_nearest": within_nearest,
                "high_priority_match_count": len(high_priority),
                "high_priority_matches": high_priority,
                "phash_only_match_count": len(phash_only),
                "phash_only_matches": phash_only,
                "status": "ok",
                "paper_evidence": True,
                "postfinal_audit": True,
            }
        )

    dataset_rows = [
        {
            "source_dataset": dataset,
            "final_groups": counts["final_groups"],
            "groups_with_high_priority_match": counts["high_priority"],
            "groups_with_phash_only_match": counts["phash_only"],
            "paper_evidence": True,
        }
        for dataset, counts in sorted(by_dataset_counts.items())
    ]
    paths = config["paths"]
    nearest_path = _resolve(project_root, str(paths["nearest_neighbors"]))
    dataset_path = _resolve(project_root, str(paths["dataset_summary"]))
    summary_path = _resolve(project_root, str(paths["summary"]))
    report_path = _resolve(project_root, str(paths["report"]))
    _write_jsonl(nearest_path, nearest_rows)
    _write_csv(dataset_path, dataset_rows)
    high_priority_count = sum(row["high_priority_match_count"] > 0 for row in nearest_rows)
    phash_only_count = sum(row["phash_only_match_count"] > 0 for row in nearest_rows)
    exact_gate = not group_overlap and not encoded_overlap and not pixel_overlap
    output = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": (
            "template_near_duplicate_audit_exact_gate_passed_flags_require_review"
            if exact_gate and high_priority_count
            else "template_near_duplicate_audit_passed_no_combined_flags"
            if exact_gate
            else "template_near_duplicate_audit_exact_leakage_detected"
        ),
        "paper_evidence": True,
        "postfinal_audit": True,
        "selection_performed": False,
        "development_groups": len(decoded_development),
        "final_groups": len(decoded_final),
        "exact_overlap": {
            "source_group_ids": len(group_overlap),
            "source_group_id_values": group_overlap,
            "encoded_sha256": len(encoded_overlap),
            "encoded_sha256_values": encoded_overlap,
            "decoded_pixel_sha256": len(pixel_overlap),
            "decoded_pixel_sha256_values": pixel_overlap,
        },
        "exact_identity_gate_passed": exact_gate,
        "perceptual_screen": {
            "final_groups_with_high_priority_match": high_priority_count,
            "final_groups_with_phash_only_match": phash_only_count,
            "high_priority_rule": screening,
        },
        "manual_template_review_required": high_priority_count > 0,
        "checks": {
            "all_96_final_groups_retained": len(nearest_rows) == 96,
            "all_records_successful": all(row["status"] == "ok" for row in nearest_rows),
            "source_group_disjoint": not group_overlap,
            "encoded_hash_disjoint": not encoded_overlap,
            "decoded_pixel_hash_disjoint": not pixel_overlap,
            "no_predictions_or_masks_read": True,
            "no_data_modified": True,
        },
        "input": {
            "config_sha256": _sha256(config_path),
            "protocol_sha256": _sha256(protocol_path),
            "development_manifest_sha256": development_hashes,
            "final_manifest_sha256": final_hash,
        },
        "outputs": {
            "nearest_neighbors": str(nearest_path.relative_to(project_root)),
            "nearest_neighbors_sha256": _sha256(nearest_path),
            "dataset_summary": str(dataset_path.relative_to(project_root)),
            "dataset_summary_sha256": _sha256(dataset_path),
            "report": str(report_path.relative_to(project_root)),
        },
    }
    if (not all(output["checks"].values())) and runtime["require_all_records"]:
        _write_json(summary_path, output)
        raise RuntimeError("template near-duplicate leakage audit failed")
    _write_json(summary_path, output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(output), encoding="utf-8")
    output["outputs"]["report_sha256"] = _sha256(report_path)
    _write_json(summary_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
