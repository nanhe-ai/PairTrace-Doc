from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from pairtrace_doc.pipelines.materialize_tfr_synthetic_clean_confirmatory import _green_mask
from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if any(
        bool(config["runtime"].get(key))
        for key in (
            "model_inference_authorized",
            "model_training_authorized",
            "threshold_selection_authorized",
            "membership_change_authorized",
        )
    ):
        raise ValueError("confirmation verifier crossed its read-only boundary")
    inputs = config["inputs"]
    loaded: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for name in ("membership", "pairs", "baseline"):
        specification = inputs[name]
        path = _resolve(project_root, str(specification["path"]))
        if _sha256(path) != str(specification["expected_sha256"]):
            raise ValueError(f"{name} manifest changed")
        loaded[name] = (path, _read_jsonl(path))
    summary_path = _resolve(project_root, str(inputs["summary"]["path"]))
    if _sha256(summary_path) != str(inputs["summary"]["expected_sha256"]):
        raise ValueError("materialization summary changed")
    materialization = json.loads(summary_path.read_text(encoding="utf-8"))
    freeze_id = str(materialization["freeze_id"])
    membership = loaded["membership"][1]
    pairs = loaded["pairs"][1]
    baseline = loaded["baseline"][1]
    expected = config["expected"]
    if len(membership) != int(expected["source_groups"]):
        raise ValueError("membership source-group count changed")
    if len(pairs) != int(expected["forged_pairs"]):
        raise ValueError("forged-pair count changed")
    if len(baseline) != int(expected["baseline_records"]):
        raise ValueError("baseline-record count changed")
    if any(str(row["freeze_id"]) != freeze_id for row in [*membership, *pairs, *baseline]):
        raise ValueError("freeze ID is inconsistent across manifests")
    if any(any(key.startswith("private_") for key in row) for row in [*membership, *pairs, *baseline]):
        raise ValueError("public manifest leaks private source information")
    scratch = Path(
        os.environ.get(
            str(config["paths"]["scratch_env"]),
            str(_resolve(project_root, str(config["paths"]["scratch_default"]))),
        )
    ).resolve()
    green_rule = config["green"]
    verification_rows: list[dict[str, Any]] = []
    attacks_by_group: dict[str, list[str]] = {}
    errors: list[str] = []
    for pair in pairs:
        sample_id = str(pair["sample_id"])
        group = str(pair["source_group_id"])
        attack = str(pair["selected_generator"])
        attacks_by_group.setdefault(group, []).append(attack)
        authentic_path = _resolve(scratch, str(pair["authentic"]))
        forged_path = _resolve(scratch, str(pair["image"]))
        mask_path = _resolve(scratch, str(pair["mask"]))
        status = "ok"
        item_errors: list[str] = []
        for path, field in (
            (authentic_path, "authentic_sha256"),
            (forged_path, "image_sha256"),
            (mask_path, "mask_sha256"),
        ):
            if _sha256(path) != str(pair[field]):
                item_errors.append(f"{field}_mismatch")
        authentic = cv2.imread(str(authentic_path), cv2.IMREAD_COLOR)
        forged = cv2.imread(str(forged_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if authentic is None or forged is None or mask is None:
            item_errors.append("decode_failure")
            changed_fraction = None
            introduced_green = None
        elif authentic.shape != forged.shape or authentic.shape[:2] != mask.shape:
            item_errors.append("geometry_mismatch")
            changed_fraction = None
            introduced_green = None
        else:
            values = set(int(value) for value in np.unique(mask))
            if not values.issubset({0, 255}) or 255 not in values:
                item_errors.append("mask_not_nonempty_binary")
            exact = np.any(authentic != forged, axis=2)
            if not np.array_equal(exact, mask > 0):
                item_errors.append("mask_not_decoded_exact_difference")
            # cv2 arrays are BGR; green logic expects RGB.
            authentic_rgb = cv2.cvtColor(authentic, cv2.COLOR_BGR2RGB)
            forged_rgb = cv2.cvtColor(forged, cv2.COLOR_BGR2RGB)
            introduced_green = int(
                np.count_nonzero(
                    _green_mask(forged_rgb, green_rule)
                    & ~_green_mask(authentic_rgb, green_rule)
                )
            )
            if introduced_green:
                item_errors.append("introduced_green_pixels")
            changed_fraction = float(exact.mean())
            if not float(expected["min_changed_fraction"]) <= changed_fraction <= float(
                expected["max_changed_fraction"]
            ):
                item_errors.append("changed_fraction_outside_bounds")
        if item_errors:
            status = "error"
            errors.extend(f"{sample_id}:{value}" for value in item_errors)
        verification_rows.append(
            {
                "sample_id": sample_id,
                "source_group_id": group,
                "attack": attack,
                "status": status,
                "changed_fraction": changed_fraction,
                "introduced_green_pixels": introduced_green,
                "errors": item_errors,
            }
        )
    expected_attacks = set(str(value) for value in expected["attacks"])
    incomplete = {
        group: attacks
        for group, attacks in attacks_by_group.items()
        if set(attacks) != expected_attacks or len(attacks) != len(expected_attacks)
    }
    if set(attacks_by_group) != {str(row["source_group_id"]) for row in membership}:
        errors.append("membership_pair_group_disagreement")
    if incomplete:
        errors.append("incomplete_attack_groups")
    output_rows = config["paths"]
    records_path = _resolve(project_root, str(output_rows["verification_records"]))
    summary_path_out = _resolve(project_root, str(output_rows["verification_summary"]))
    metrics_path = _resolve(project_root, str(output_rows["verification_metrics"]))
    _write_jsonl(records_path, verification_rows)
    _write_csv(
        metrics_path,
        [
            {
                "source_groups": len(attacks_by_group),
                "forged_pairs": len(pairs),
                "ok_pairs": sum(row["status"] == "ok" for row in verification_rows),
                "error_pairs": sum(row["status"] != "ok" for row in verification_rows),
                "minimum_changed_fraction": min(
                    float(row["changed_fraction"])
                    for row in verification_rows
                    if row["changed_fraction"] is not None
                ),
                "maximum_changed_fraction": max(
                    float(row["changed_fraction"])
                    for row in verification_rows
                    if row["changed_fraction"] is not None
                ),
            }
        ],
    )
    summary = {
        "status": "confirmation_integrity_verified" if not errors else "confirmation_integrity_failed",
        "freeze_id": freeze_id,
        "source_groups": len(attacks_by_group),
        "forged_pairs": len(pairs),
        "error_count": len(errors),
        "error_categories": dict(Counter(value.split(":")[-1] for value in errors)),
        "model_inference_performed": False,
        "threshold_selection_performed": False,
        "outputs": {
            "records": str(records_path.relative_to(project_root)),
            "records_sha256": _sha256(records_path),
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
        },
    }
    _write_json(summary_path_out, summary)
    if errors:
        raise RuntimeError(f"confirmation integrity verification failed: {errors[:10]}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
