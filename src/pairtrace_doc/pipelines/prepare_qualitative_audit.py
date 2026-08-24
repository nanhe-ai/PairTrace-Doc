from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import yaml

from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
    _shuffled_group_map,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_json,
)


ROBUST_SEEDS = (20260747, 20260763, 20260764)


def _tie_key(item: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item.get(field, ""))
        for field in (
            "evaluation_role",
            "source_group_id",
            "attack_method",
            "candidate_device",
            "device",
        )
    )


def _lower_median(
    items: list[dict[str, Any]], value: Callable[[dict[str, Any]], float]
) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot select a median from no cases")
    ordered = sorted(items, key=lambda item: (float(value(item)), _tie_key(item)))
    return ordered[(len(ordered) - 1) // 2]


def _bottom_quartile_highest_ecc(
    items: list[dict[str, Any]], ap_field: str, ecc_field: str
) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot select a quartile case from no cases")
    ordered_ap = sorted(float(item[ap_field]) for item in items)
    boundary_index = max(0, math.ceil(0.25 * len(ordered_ap)) - 1)
    boundary = ordered_ap[boundary_index]
    eligible = [item for item in items if float(item[ap_field]) <= boundary]
    return sorted(
        eligible,
        key=lambda item: (-float(item[ecc_field]), _tie_key(item)),
    )[0]


def _group_mean(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    value_field: str,
) -> dict[tuple[str, ...], float]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in key_fields)].append(
            float(row[value_field])
        )
    return {key: float(mean(values)) for key, values in grouped.items()}


def _prediction_refs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths = sorted(
        {str(row["score_cache"]) for row in rows if row.get("score_cache")}
    )
    return {
        "record_ids": sorted(str(row["record_id"]) for row in rows),
        "score_cache_paths": paths,
        "score_caches_currently_available": all(Path(path).is_file() for path in paths),
    }


def _file_ref(
    row: dict[str, Any],
    path_field: str,
    sha_field: str,
    scratch: Path,
    member_field: str | None = None,
) -> dict[str, Any] | None:
    value = row.get(path_field)
    if value is None:
        return None
    path = _resolve(scratch, str(value))
    result: dict[str, Any] = {
        "path": str(value),
        "sha256": row.get(sha_field),
        "currently_available": path.is_file(),
    }
    if member_field is not None:
        result["archive_member"] = row.get(member_field)
    return result


def _final_cases(
    predictions: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    scratch: Path,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    forged = [
        row
        for row in predictions
        if row["sample_kind"] == "forged" and row["status"] == "ok"
    ]
    key_fields = ("source_group_id", "evaluation_role")
    clean_names = {f"robust_{seed}_clean_ecc" for seed in ROBUST_SEEDS}
    affine_names = {f"robust_{seed}_affine_ecc" for seed in ROBUST_SEEDS}
    shuffled_names = {f"robust_{seed}_shuffled_clean" for seed in ROBUST_SEEDS}
    clean_rows = [row for row in forged if row["condition"] in clean_names]
    affine_rows = [row for row in forged if row["condition"] in affine_names]
    shuffled_rows = [row for row in forged if row["condition"] in shuffled_names]
    baseline_affine_rows = [
        row for row in forged if row["condition"] == "baseline_affine_ecc"
    ]
    clean = _group_mean(clean_rows, key_fields, "macro_pixel_ap")
    affine = _group_mean(affine_rows, key_fields, "macro_pixel_ap")
    shuffled = _group_mean(shuffled_rows, key_fields, "macro_pixel_ap")
    baseline_affine = _group_mean(
        baseline_affine_rows, key_fields, "macro_pixel_ap"
    )
    if not (set(clean) == set(affine) == set(shuffled) == set(baseline_affine)):
        raise ValueError("final qualitative conditions have different record sets")
    candidates = [
        {
            "source_group_id": group,
            "evaluation_role": role,
            "clean_ap_mean": clean[(group, role)],
            "affine_ap_mean": affine[(group, role)],
            "baseline_affine_ap": baseline_affine[(group, role)],
            "affine_gain": affine[(group, role)] - baseline_affine[(group, role)],
            "shuffled_ap_mean": shuffled[(group, role)],
            "wrong_reference_collapse": clean[(group, role)]
            - shuffled[(group, role)],
        }
        for group, role in sorted(clean)
    ]
    selected = [
        (
            "final_in_domain_median_clean",
            _lower_median(
                [item for item in candidates if item["evaluation_role"] == "in_domain_test"],
                lambda item: item["clean_ap_mean"],
            ),
            "clean_ap_mean",
        ),
        (
            "final_generator_holdout_median_clean",
            _lower_median(
                [item for item in candidates if item["evaluation_role"] == "generator_holdout"],
                lambda item: item["clean_ap_mean"],
            ),
            "clean_ap_mean",
        ),
        (
            "final_global_worst_clean",
            sorted(candidates, key=lambda item: (item["clean_ap_mean"], _tie_key(item)))[0],
            "clean_ap_mean",
        ),
        (
            "final_median_affine_gain",
            _lower_median(candidates, lambda item: item["affine_gain"]),
            "affine_gain",
        ),
        (
            "final_median_wrong_reference_collapse",
            _lower_median(candidates, lambda item: item["wrong_reference_collapse"]),
            "wrong_reference_collapse",
        ),
    ]

    bundles: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in manifest:
        bundles[str(row["source_group_id"])][str(row["evaluation_role"])] = row
    ordered_groups = [
        {"source_group_id": group, **roles}
        for group, roles in sorted(bundles.items())
    ]
    shuffled_map = _shuffled_group_map(ordered_groups, shuffle_seed)
    prediction_index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in forged:
        prediction_index[
            (str(row["source_group_id"]), str(row["evaluation_role"]), str(row["condition"]))
        ].append(row)

    cases: list[dict[str, Any]] = []
    for case_id, item, scalar_field in selected:
        group = str(item["source_group_id"])
        role = str(item["evaluation_role"])
        candidate = bundles[group][role]
        authentic = bundles[group]["final_test"]
        condition_names = sorted(clean_names | affine_names | shuffled_names | {"baseline_affine_ecc"})
        source_rows = [
            row
            for condition in condition_names
            for row in prediction_index.get((group, role, condition), [])
        ]
        case: dict[str, Any] = {
            "case_id": case_id,
            "cohort": "aiforge_final_reserve_96",
            "paper_evidence": True,
            "mask_semantics": "pixel_exact_binary_mask",
            "source_group_id": group,
            "evaluation_role": role,
            "generator": candidate["generator"],
            "source_dataset": candidate["source_dataset"],
            "selection_scalar": scalar_field,
            "selection_value": float(item[scalar_field]),
            "component_metrics": {
                key: float(value)
                for key, value in item.items()
                if isinstance(value, float)
            },
            "candidate": _file_ref(candidate, "image", "image_sha256", scratch),
            "correct_reference": _file_ref(
                authentic, "image", "image_sha256", scratch
            ),
            "mask": _file_ref(candidate, "mask", "mask_sha256", scratch),
            "source_predictions": _prediction_refs(source_rows),
        }
        if case_id == "final_median_wrong_reference_collapse":
            wrong_group = shuffled_map[group]
            case["wrong_reference_source_group_id"] = wrong_group
            case["wrong_reference"] = _file_ref(
                bundles[wrong_group]["final_test"],
                "image",
                "image_sha256",
                scratch,
            )
        cases.append(case)
    return cases


def _reference_integrity_cases(
    predictions: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    scratch: Path,
) -> list[dict[str, Any]]:
    forged = [
        row
        for row in predictions
        if row["sample_kind"] == "forged" and row["status"] == "ok"
    ]
    full_rows = [row for row in forged if row["condition"] == "correct_full"]
    half_rows = [row for row in forged if row["condition"] == "correct_overlap_050"]
    wrong_rows = [row for row in forged if row["condition"] == "wrong_same_dataset_size"]
    full = _group_mean(full_rows, ("source_group_id",), "full_document_pixel_ap")
    half = _group_mean(half_rows, ("source_group_id",), "full_document_pixel_ap")
    wrong_ap = _group_mean(wrong_rows, ("source_group_id",), "full_document_pixel_ap")
    wrong_ecc = _group_mean(wrong_rows, ("source_group_id",), "ecc_correlation")
    integrity = [
        {
            "source_group_id": key[0],
            "full_ap_mean": full[key],
            "half_view_ap_mean": half[key],
            "half_view_drop": full[key] - half[key],
        }
        for key in sorted(full)
    ]
    wrong = [
        {
            "source_group_id": key[0],
            "wrong_reference_ap_mean": wrong_ap[key],
            "wrong_reference_ecc_mean": wrong_ecc[key],
        }
        for key in sorted(wrong_ap)
    ]
    selected_integrity = _lower_median(integrity, lambda item: item["half_view_drop"])
    selected_wrong = _bottom_quartile_highest_ecc(
        wrong, "wrong_reference_ap_mean", "wrong_reference_ecc_mean"
    )
    manifest_by_group = {str(row["source_group_id"]): row for row in manifest}
    cases: list[dict[str, Any]] = []
    for case_id, item, conditions, scalar in (
        (
            "reference_integrity_median_half_view_drop",
            selected_integrity,
            {"correct_full", "correct_overlap_050"},
            "half_view_drop",
        ),
        (
            "wrong_reference_high_ecc_low_ap",
            selected_wrong,
            {"wrong_same_dataset_size"},
            "wrong_reference_ecc_mean",
        ),
    ):
        group = str(item["source_group_id"])
        source = manifest_by_group[group]
        source_rows = [
            row
            for row in forged
            if row["source_group_id"] == group and row["condition"] in conditions
        ]
        reference_groups = sorted(
            {
                str(row["reference_source_group_id"])
                for row in source_rows
                if row.get("reference_source_group_id")
            }
        )
        if len(reference_groups) != 1:
            raise ValueError("reference-integrity selected case has ambiguous reference")
        reference = manifest_by_group[reference_groups[0]]
        cases.append(
            {
                "case_id": case_id,
                "cohort": "reference_integrity_viewed20",
                "paper_evidence": False,
                "evidence_role": "post_final_viewed_development_limitation",
                "mask_semantics": "pixel_exact_binary_mask",
                "source_group_id": group,
                "generator": source["selected_generator"],
                "source_dataset": source["source_dataset"],
                "selection_scalar": scalar,
                "selection_value": float(item[scalar]),
                "component_metrics": {
                    key: float(value)
                    for key, value in item.items()
                    if isinstance(value, float)
                },
                "candidate": _file_ref(source, "image", "image_sha256", scratch),
                "correct_reference": _file_ref(
                    source, "authentic", "authentic_sha256", scratch
                ),
                "selected_reference_source_group_id": reference_groups[0],
                "selected_reference": _file_ref(
                    reference, "authentic", "authentic_sha256", scratch
                ),
                "mask": _file_ref(source, "mask", "mask_sha256", scratch),
                "source_predictions": _prediction_refs(source_rows),
            }
        )
    return cases


def _fantasy_cases(
    same_predictions: list[dict[str, Any]],
    cross_predictions: list[dict[str, Any]],
    same_manifest: list[dict[str, Any]],
    cross_manifest: list[dict[str, Any]],
    scratch: Path,
) -> list[dict[str, Any]]:
    same_rows = [
        row
        for row in same_predictions
        if row["sample_kind"] == "forged"
        and row["condition"] == "robust_teacher_correct"
        and row["status"] == "ok"
    ]
    selected_same = _lower_median(same_rows, lambda row: row["macro_box_mask_ap"])
    cross_rows = [
        row
        for row in cross_predictions
        if row["sample_kind"] == "forged" and row["status"] == "ok"
    ]
    cross_ap = _group_mean(cross_rows, ("source_group_id",), "weak_box_mask_ap")
    cross_ecc = _group_mean(cross_rows, ("source_group_id",), "ecc_correlation")
    cross_items = [
        {
            "source_group_id": key[0],
            "cross_device_ap_mean": cross_ap[key],
            "cross_device_ecc_mean": cross_ecc[key],
        }
        for key in sorted(cross_ap)
    ]
    selected_cross = _bottom_quartile_highest_ecc(
        cross_items, "cross_device_ap_mean", "cross_device_ecc_mean"
    )
    same_by_group = {str(row["source_group_id"]): row for row in same_manifest}
    cross_by_group = {str(row["source_group_id"]): row for row in cross_manifest}
    cases: list[dict[str, Any]] = []
    for case_id, item, materialized, scalar, source_rows in (
        (
            "fantasyid_same_device_median",
            selected_same,
            same_by_group[str(selected_same["source_group_id"])],
            "macro_box_mask_ap",
            [selected_same],
        ),
        (
            "fantasyid_cross_device_high_ecc_low_ap",
            selected_cross,
            cross_by_group[str(selected_cross["source_group_id"])],
            "cross_device_ecc_mean",
            [
                row
                for row in cross_rows
                if row["source_group_id"] == selected_cross["source_group_id"]
            ],
        ),
    ):
        component_metrics = {
            key: float(value)
            for key, value in item.items()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        }
        cases.append(
            {
                "case_id": case_id,
                "cohort": (
                    "fantasyid_same_device_viewed88"
                    if "same_device" in case_id
                    else "fantasyid_cross_device_pilot20"
                ),
                "paper_evidence": False,
                "evidence_role": "viewed_development_weak_box_limitation",
                "mask_semantics": "box_mask_not_pixel_accurate",
                "source_group_id": str(materialized["source_group_id"]),
                "attack_method": materialized["attack_method"],
                "candidate_device": materialized["device"],
                "reference_device": materialized.get(
                    "cross_device_reference_device", materialized["device"]
                ),
                "selection_scalar": scalar,
                "selection_value": float(item[scalar]),
                "component_metrics": component_metrics,
                "candidate": _file_ref(
                    materialized,
                    "image",
                    "image_sha256",
                    scratch,
                    "forged_image_member",
                ),
                "correct_same_device_reference": _file_ref(
                    materialized,
                    "authentic",
                    "authentic_sha256",
                    scratch,
                    "authentic_image_member",
                ),
                "selected_reference": _file_ref(
                    materialized,
                    (
                        "cross_device_reference"
                        if "cross_device" in case_id
                        else "authentic"
                    ),
                    (
                        "cross_device_reference_sha256"
                        if "cross_device" in case_id
                        else "authentic_sha256"
                    ),
                    scratch,
                    (
                        "cross_device_reference_member"
                        if "cross_device" in case_id
                        else "authentic_image_member"
                    ),
                ),
                "mask": _file_ref(
                    materialized, "mask", "mask_sha256", scratch
                ),
                "mask_generation": {
                    "source_metadata_archive_member": materialized[
                        "forged_metadata_member"
                    ],
                    "source_metadata_sha256": materialized[
                        "forged_metadata_sha256"
                    ],
                    "rasterizer_version": materialized[
                        "mask_rasterizer_version"
                    ],
                    "annotation_width": int(materialized["annotation_width"]),
                    "annotation_height": int(materialized["annotation_height"]),
                    "archive_sha256": materialized["archive_sha256"],
                },
                "source_predictions": _prediction_refs(source_rows),
            }
        )
    return cases


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if not runtime["cpu_selection_authorized"] or runtime["gpu_launch_authorized"]:
        raise ValueError("qualitative selection must be explicitly CPU-only")
    if runtime["model_inference_authorized"] or runtime["threshold_selection_authorized"]:
        raise ValueError("qualitative selection cannot run inference or select thresholds")
    if runtime["final_reserve_image_read_allowed"]:
        raise ValueError("qualitative selection cannot reopen final images")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("qualitative selection output is not new paper evidence")

    protocol = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("qualitative audit protocol SHA-256 changed")
    loaded: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    for name, specification in config["sources"].items():
        path = _resolve(project_root, specification["path"])
        actual = _sha256(path)
        if actual != specification["sha256"]:
            raise ValueError(f"qualitative source changed: {name}")
        loaded[name] = _read_jsonl(path)
        source_hashes[str(path.relative_to(project_root))] = actual

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    cases = []
    cases.extend(
        _final_cases(
            loaded["final_predictions"],
            loaded["final_manifest"],
            scratch,
            int(config["selection"]["final_shuffle_seed"]),
        )
    )
    cases.extend(
        _reference_integrity_cases(
            loaded["reference_integrity_predictions"],
            loaded["pair_development_manifest"],
            scratch,
        )
    )
    cases.extend(
        _fantasy_cases(
            loaded["fantasy_same_predictions"],
            loaded["fantasy_cross_predictions"],
            loaded["fantasy_same_manifest"],
            loaded["fantasy_cross_manifest"],
            scratch,
        )
    )
    expected_ids = [str(value) for value in config["selection"]["expected_case_ids"]]
    if [case["case_id"] for case in cases] != expected_ids:
        raise ValueError("qualitative case set changed")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("qualitative case identifiers are not unique")

    output = {
        "experiment": config["experiment"],
        "status": "qualitative_sample_manifest_frozen",
        "paper_evidence": False,
        "new_scientific_metrics_computed": False,
        "model_inference_performed": False,
        "threshold_selection_used": False,
        "final_reserve_images_read": False,
        "selection": config["selection"],
        "source_artifact_sha256": source_hashes,
        "case_count": len(cases),
        "cases": cases,
        "rendering": {
            "candidate_reference_mask_panels_authorized": True,
            "model_heatmaps_available": all(
                case["source_predictions"]["score_caches_currently_available"]
                for case in cases
            ),
            "artifact_only_model_replay_authorized": False,
            "sample_replacement_allowed": False,
        },
    }
    output_path = _resolve(project_root, paths["manifest"])
    _write_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
