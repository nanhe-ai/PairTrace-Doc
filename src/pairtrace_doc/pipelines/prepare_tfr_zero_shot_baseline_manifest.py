from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_json,
    _write_jsonl,
)


def _image_shape(path: Path) -> tuple[int, int]:
    with Image.open(path) as handle:
        width, height = handle.size
    return height, width


def _forged_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": str(row["sample_id"]),
        "source_group_id": str(row["source_group_id"]),
        "source_sample_id": str(row["sample_id"]),
        "evaluation_role": "viewed_development",
        "sample_kind": "forged",
        "image": str(row["image"]),
        "image_sha256": str(row["image_sha256"]),
        "mask": str(row["mask"]),
        "mask_sha256": str(row["mask_sha256"]),
        "height": int(row["image_height"]),
        "width": int(row["image_width"]),
        "tfr_split_freeze_id": str(row["freeze_id"]),
        "viewed_development": True,
        "paper_evidence": False,
        "tfr_holdout_read": False,
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["runtime"]["tfr_holdout_read_allowed"]:
        raise ValueError("TFR baseline adapter cannot read the holdout")
    if config["runtime"]["method_training_authorized"]:
        raise ValueError("TFR baseline adapter cannot authorize training")

    protocol = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("TFR baseline protocol SHA-256 changed")
    input_path = _resolve(project_root, config["input"]["manifest"])
    if _sha256(input_path) != config["input"]["expected_manifest_sha256"]:
        raise ValueError("TFR train/validation manifest SHA-256 changed")
    all_rows = _read_jsonl(input_path)
    rows = sorted(
        [row for row in all_rows if row["pilot_role"] == config["input"]["role"]],
        key=lambda row: (str(row["source_group_id"]), str(row["sample_id"])),
    )
    expected_freeze_id = str(config["input"]["expected_freeze_id"])
    if {str(row["freeze_id"]) for row in rows} != {expected_freeze_id}:
        raise ValueError("TFR split freeze ID changed")
    if len(rows) != int(config["input"]["expected_forged_pairs"]):
        raise ValueError("TFR validation forged-pair count changed")
    groups = {str(row["source_group_id"]) for row in rows}
    if len(groups) != int(config["input"]["expected_source_groups"]):
        raise ValueError("TFR validation source-group count changed")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    records = [_forged_record(row) for row in rows]
    representatives: dict[str, dict[str, Any]] = {}
    for row in rows:
        representatives.setdefault(str(row["source_group_id"]), row)
    for group, row in sorted(representatives.items()):
        image_path = _resolve(scratch, row["authentic"])
        if _sha256(image_path) != row["authentic_sha256"]:
            raise ValueError("TFR authentic image SHA-256 changed")
        height, width = _image_shape(image_path)
        records.append(
            {
                "record_id": f"{group}:authentic",
                "source_group_id": group,
                "source_sample_id": str(row["sample_id"]),
                "evaluation_role": "viewed_development",
                "sample_kind": "authentic",
                "image": str(row["authentic"]),
                "image_sha256": str(row["authentic_sha256"]),
                "mask": None,
                "mask_sha256": None,
                "height": height,
                "width": width,
                "tfr_split_freeze_id": expected_freeze_id,
                "viewed_development": True,
                "paper_evidence": False,
                "tfr_holdout_read": False,
            }
        )
    if len({record["record_id"] for record in records}) != len(records):
        raise ValueError("TFR baseline adapter produced duplicate record IDs")

    output_manifest = _resolve(project_root, paths["output_manifest"])
    output_summary = _resolve(project_root, paths["output_summary"])
    _write_jsonl(output_manifest, records)
    summary = {
        "status": "passed",
        "experiment": config["experiment"],
        "paper_evidence": False,
        "tfr_holdout_read": False,
        "input_manifest_sha256": _sha256(input_path),
        "tfr_split_freeze_id": expected_freeze_id,
        "forged_records": len(rows),
        "authentic_records": len(representatives),
        "source_groups": len(representatives),
        "output_manifest": str(output_manifest.relative_to(project_root)),
        "output_manifest_sha256": _sha256(output_manifest),
    }
    _write_json(output_summary, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
