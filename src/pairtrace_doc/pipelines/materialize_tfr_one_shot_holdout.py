from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import yaml
import numpy as np
from PIL import Image

from pairtrace_doc.pipelines.freeze_tfr_internal_pair_split import (
    _atomic_bytes,
    _binary_mask_png,
    _member_bytes,
    _sha256_bytes,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_json,
    _write_jsonl,
)


def _private_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _image_shape(path: Path) -> tuple[int, int]:
    with Image.open(path) as handle:
        width, height = handle.size
    return height, width


def _validate_cached_binary_mask(path: Path) -> None:
    with Image.open(path) as handle:
        mask = np.asarray(handle.convert("L"))
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise ValueError("cached TFR holdout mask is not binary")
    if 255 not in values:
        raise ValueError("cached TFR holdout mask is empty")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if (
        runtime["gpu_launch_authorized"]
        or runtime["method_training_authorized"]
        or runtime["threshold_selection_authorized"]
        or not runtime["holdout_materialization_authorized"]
    ):
        raise ValueError("TFR holdout materialization boundary changed")
    experiment = config["experiment"]
    protocol = _resolve(project_root, experiment["protocol"])
    if _sha256(protocol) != experiment["expected_protocol_sha256"]:
        raise ValueError("TFR one-shot protocol SHA-256 changed")

    input_config = config["input"]
    split_summary_path = _resolve(project_root, input_config["split_summary"])
    if _sha256(split_summary_path) != input_config["expected_split_summary_sha256"]:
        raise ValueError("TFR split summary changed")
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    if (
        split_summary["materialization"]["holdout_pairs_materialized"] != 0
        or split_summary["split"]["group_counts"]["holdout"] != int(input_config["expected_source_groups"])
        or split_summary["split"]["pair_counts"]["holdout"] != int(input_config["expected_pairs"])
    ):
        raise ValueError("TFR holdout split boundary changed")
    audit_path = _resolve(project_root, input_config["qualitative_audit_summary"])
    if _sha256(audit_path) != input_config["expected_qualitative_audit_summary_sha256"]:
        raise ValueError("TFR qualitative audit summary changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != input_config["accepted_qualitative_audit_status"]:
        raise ValueError("TFR qualitative audit gate did not pass")

    source = config["source"]
    password = os.environ.get(source["tfr_password_env"])
    permission_record = os.environ.get(source["tfr_permission_record_env"])
    if not password:
        raise ValueError("required TFR archive password environment variable is missing")
    if not permission_record:
        raise ValueError("required TFR permission record environment variable is missing")
    if _private_hash(permission_record) != source["expected_permission_record_id_sha256"]:
        raise ValueError("TFR permission record identity changed")

    membership_path = _resolve(project_root, input_config["holdout_membership"])
    if _sha256(membership_path) != input_config["expected_holdout_membership_sha256"]:
        raise ValueError("TFR holdout membership changed")
    membership = sorted(_read_jsonl(membership_path), key=lambda row: str(row["record_id"]))
    if len(membership) != int(input_config["expected_pairs"]):
        raise ValueError("TFR holdout pair count changed")
    if len({str(row["source_group_id"]) for row in membership}) != int(
        input_config["expected_source_groups"]
    ):
        raise ValueError("TFR holdout source-group count changed")
    if {str(row["role"]) for row in membership} != {"holdout"}:
        raise ValueError("TFR holdout membership role changed")
    if {str(row["freeze_id"]) for row in membership} != {
        str(input_config["expected_freeze_id"])
    }:
        raise ValueError("TFR holdout freeze ID changed")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    archive_path = _resolve(scratch, paths["tfr_inner_archive"])
    if archive_path.stat().st_size != int(source["tfr_inner_archive_bytes"]):
        raise ValueError("TFR inner archive size changed")
    derived_root = _resolve(scratch, paths["derived_root"])
    foreground_gt = int(config["mask_processing"]["foreground_gt"])
    max_member_bytes = int(runtime["max_member_bytes"])
    pair_rows: list[dict[str, Any]] = []
    authentic_by_group: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    with ZipFile(archive_path) as archive:
        for row in membership:
            group = str(row["source_group_id"])
            group_key = _sha256_bytes(group.encode("utf-8"))[:16]
            record_key = str(row["record_id"])[:16]
            directory = derived_root / group_key
            authentic_path = directory / "authentic.jpg"
            tampered_path = directory / f"{record_key}_tampered.jpg"
            mask_path = directory / f"{record_key}_mask.png"
            paths_ready = all(path.is_file() for path in (authentic_path, tampered_path, mask_path))
            if paths_ready:
                if (
                    _sha256(authentic_path) != row["authentic_sha256"]
                    or _sha256(tampered_path) != row["tampered_sha256"]
                ):
                    raise ValueError("cached TFR holdout image hash changed")
                _validate_cached_binary_mask(mask_path)
                cache_hits += 1
            else:
                authentic_payload = _member_bytes(
                    archive, row["authentic_archive_path"], password, max_member_bytes
                )
                tampered_payload = _member_bytes(
                    archive, row["tampered_archive_path"], password, max_member_bytes
                )
                mask_payload = _member_bytes(
                    archive, row["mask_archive_path"], password, max_member_bytes
                )
                if _sha256_bytes(authentic_payload) != row["authentic_sha256"]:
                    raise ValueError("TFR holdout authentic payload hash changed")
                if _sha256_bytes(tampered_payload) != row["tampered_sha256"]:
                    raise ValueError("TFR holdout tampered payload hash changed")
                if _sha256_bytes(mask_payload) != row["raw_mask_sha256"]:
                    raise ValueError("TFR holdout raw-mask payload hash changed")
                binary_mask_payload, _ = _binary_mask_png(mask_payload, foreground_gt)
                _atomic_bytes(authentic_path, authentic_payload)
                _atomic_bytes(tampered_path, tampered_payload)
                _atomic_bytes(mask_path, binary_mask_payload)
            height, width = _image_shape(tampered_path)
            if (height, width) != _image_shape(mask_path):
                raise ValueError("TFR holdout image/mask geometry changed")
            sample_id = f"tfr-ettd:{row['record_id']}"
            pair = {
                "sample_id": sample_id,
                "source_group_id": group,
                "source_dataset": "TFR-ETTD",
                "selected_generator": "copy_move_splice",
                "pilot_role": "holdout",
                "role": "holdout",
                "image": str(tampered_path.relative_to(scratch)),
                "authentic": str(authentic_path.relative_to(scratch)),
                "mask": str(mask_path.relative_to(scratch)),
                "image_sha256": _sha256(tampered_path),
                "authentic_sha256": _sha256(authentic_path),
                "mask_sha256": _sha256(mask_path),
                "image_height": height,
                "image_width": width,
                "mask_processing": "foreground_gt_160",
                "mapping_rule": "frozen_membership_shared_embedded_source_id",
                "freeze_id": str(row["freeze_id"]),
                "paper_evidence_candidate": True,
                "holdout": True,
                "valid": True,
                "errors": [],
            }
            pair_rows.append(pair)
            authentic_by_group.setdefault(group, pair)

    baseline_rows: list[dict[str, Any]] = []
    for row in pair_rows:
        baseline_rows.append(
            {
                "record_id": str(row["sample_id"]),
                "source_group_id": str(row["source_group_id"]),
                "source_sample_id": str(row["sample_id"]),
                "evaluation_role": "one_shot_holdout",
                "sample_kind": "forged",
                "image": str(row["image"]),
                "image_sha256": str(row["image_sha256"]),
                "mask": str(row["mask"]),
                "mask_sha256": str(row["mask_sha256"]),
                "height": int(row["image_height"]),
                "width": int(row["image_width"]),
                "tfr_split_freeze_id": str(row["freeze_id"]),
                "paper_evidence_candidate": True,
                "tfr_holdout_read": True,
            }
        )
    for group, row in sorted(authentic_by_group.items()):
        authentic_path = _resolve(scratch, row["authentic"])
        height, width = _image_shape(authentic_path)
        baseline_rows.append(
            {
                "record_id": f"{group}:authentic",
                "source_group_id": group,
                "source_sample_id": str(row["sample_id"]),
                "evaluation_role": "one_shot_holdout",
                "sample_kind": "authentic",
                "image": str(row["authentic"]),
                "image_sha256": str(row["authentic_sha256"]),
                "mask": None,
                "mask_sha256": None,
                "height": height,
                "width": width,
                "tfr_split_freeze_id": str(row["freeze_id"]),
                "paper_evidence_candidate": True,
                "tfr_holdout_read": True,
            }
        )
    if len({row["record_id"] for row in baseline_rows}) != len(baseline_rows):
        raise ValueError("TFR holdout baseline manifest has duplicate IDs")

    pair_manifest_path = _resolve(project_root, paths["pair_manifest"])
    baseline_manifest_path = _resolve(project_root, paths["baseline_manifest"])
    summary_path = _resolve(project_root, paths["summary"])
    _write_jsonl(pair_manifest_path, pair_rows)
    _write_jsonl(baseline_manifest_path, baseline_rows)
    summary = {
        "status": "tfr_one_shot_holdout_materialized",
        "experiment": experiment,
        "paper_evidence_candidate": True,
        "holdout_read": True,
        "model_inference_performed": False,
        "threshold_selection_performed": False,
        "source_groups": len(authentic_by_group),
        "forged_pairs": len(pair_rows),
        "baseline_records": len(baseline_rows),
        "cache_hits": cache_hits,
        "permission_record_id_sha256": _private_hash(permission_record),
        "protocol_sha256": _sha256(protocol),
        "holdout_membership_sha256": _sha256(membership_path),
        "outputs": {
            "pair_manifest": str(pair_manifest_path.relative_to(project_root)),
            "pair_manifest_sha256": _sha256(pair_manifest_path),
            "baseline_manifest": str(baseline_manifest_path.relative_to(project_root)),
            "baseline_manifest_sha256": _sha256(baseline_manifest_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
