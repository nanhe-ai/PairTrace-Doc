from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not item for item in ids):
        raise ValueError(f"blank identifier in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate identifier in {path}")
    return ids


def _audit_record(task: tuple[str, Path, Path, Path]) -> dict[str, Any]:
    identifier, image_path, mask_path, scratch = task
    errors: list[str] = []
    record: dict[str, Any] = {
        "record_id": f"rtm:test:{identifier}",
        "source_dataset": "RTM",
        "official_identifier": identifier,
        "official_split": "test",
        "role": "external_test",
        "sample_id": identifier,
        "image": str(image_path.relative_to(scratch)),
        "mask": str(mask_path.relative_to(scratch)),
        "model_or_threshold_selection_allowed": False,
        "paper_evidence": False,
    }
    if not image_path.is_file():
        errors.append("missing_image")
    if not mask_path.is_file():
        errors.append("missing_mask")
    if errors:
        return {**record, "valid": False, "errors": errors}
    try:
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            image_width, image_height = image.size
        with Image.open(mask_path) as mask_handle:
            mask_mode = mask_handle.mode
            mask = np.asarray(mask_handle.convert("L"))
        values = sorted(int(value) for value in np.unique(mask))
        if values not in ([0], [0, 255]):
            errors.append("mask_not_binary_0_255")
        if list(mask.shape) != [image_height, image_width]:
            errors.append("image_mask_dimension_mismatch")
        positive_pixels = int(np.count_nonzero(mask))
        image_sha256 = _sha256(image_path)
        record.update(
            {
                "image_sha256": image_sha256,
                "mask_sha256": _sha256(mask_path),
                "image_height": image_height,
                "image_width": image_width,
                "mask_mode": mask_mode,
                "mask_values": values,
                "mask_positive_pixels": positive_pixels,
                "label": "forged" if positive_pixels else "authentic",
                "uncertainty_group_id": f"rtm:exact-image:{image_sha256}",
            }
        )
    except Exception as error:
        errors.append(f"decode_error:{type(error).__name__}:{error}")
    record["valid"] = not errors
    record["errors"] = errors
    return record


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=lambda item: item.isoformat()
            if hasattr(item, "isoformat")
            else str(item),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["runtime"]["gpu_launch_authorized"]:
        raise ValueError("RTM manifest freeze must not authorize GPU use")
    if config["runtime"]["method_training_authorized"]:
        raise ValueError("RTM external source must never authorize method training")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    source = config["source"]
    archive_path = _resolve(scratch, source["archive"]).resolve()
    if _sha256(archive_path) != source["archive_sha256"]:
        raise ValueError("RTM archive SHA-256 changed")
    extracted_root = _resolve(scratch, source["extracted_root"]).resolve()
    train_ids = _read_ids(extracted_root / source["train_ids"])
    test_ids = _read_ids(extracted_root / source["test_ids"])
    if set(train_ids) & set(test_ids):
        raise ValueError("RTM official train and test identifiers overlap")
    if len(test_ids) != int(source["expected_test_records"]):
        raise ValueError("RTM official test count changed")

    manifest_path = _resolve(project_root, paths["manifest"]).resolve()
    manifest_cache_hit = False
    if config["runtime"].get("reuse_complete_manifest") and manifest_path.is_file():
        records = _read_jsonl(manifest_path)
        cached_ids = [str(row.get("official_identifier")) for row in records]
        if (
            len(records) == len(test_ids)
            and cached_ids == sorted(test_ids)
            and all(row.get("valid") for row in records)
        ):
            manifest_cache_hit = True
        else:
            records = []
    else:
        records = []
    if not manifest_cache_hit:
        image_root = extracted_root / source["image_directory"]
        mask_root = extracted_root / source["mask_directory"]
        tasks = [
            (
                identifier,
                image_root / f"{identifier}.jpg",
                mask_root / f"{identifier}.png",
                scratch,
            )
            for identifier in test_ids
        ]
        with ThreadPoolExecutor(
            max_workers=int(config["runtime"]["workers"])
        ) as executor:
            records = list(executor.map(_audit_record, tasks))
        records.sort(key=lambda row: str(row["official_identifier"]))

    invalid = [row for row in records if not row["valid"]]
    labels = Counter(row.get("label") for row in records if row["valid"])
    if labels["forged"] != int(source["expected_test_forged"]):
        raise ValueError("RTM forged test count changed")
    if labels["authentic"] != int(source["expected_test_authentic"]):
        raise ValueError("RTM authentic test count changed")
    group_counts = Counter(
        str(row["uncertainty_group_id"]) for row in records if row["valid"]
    )
    duplicate_groups = {group: count for group, count in group_counts.items() if count > 1}

    summary_path = _resolve(project_root, paths["summary"]).resolve()
    if not manifest_cache_hit:
        _write_jsonl(manifest_path, records)
    summary = {
        "experiment": config["experiment"],
        "status": "provisionally_eligible_external_test" if not invalid else "blocked",
        "paper_evidence": False,
        "gpu_used": False,
        "source": {
            "repository": source["repository"],
            "repository_revision": source["repository_revision"],
            "archive_sha256": source["archive_sha256"],
        },
        "license_policy": config["license_policy"],
        "evaluation_policy": config["evaluation_policy"],
        "manifest": {
            "path": str(manifest_path.relative_to(project_root)),
            "cache_hit": manifest_cache_hit,
            "sha256": _sha256(manifest_path),
            "records": len(records),
            "valid_records": len(records) - len(invalid),
            "invalid_records": len(invalid),
            "label_counts": dict(sorted(labels.items())),
            "uncertainty_groups": len(group_counts),
            "exact_duplicate_groups": len(duplicate_groups),
            "exact_duplicate_records": sum(duplicate_groups.values()),
        },
        "leakage": {
            "rtm_records_used_for_fitting_or_model_selection": 0,
            "official_train_test_identifier_overlap": 0,
            "source_document_ids_available": False,
            "exact_duplicates_grouped_for_uncertainty": True,
        },
        "paper_evidence_blockers": [
            "redacted_or_private_audit_reference_for_author_permission_pending"
        ],
    }
    _write_json(summary_path, summary)
    if invalid:
        raise RuntimeError(f"RTM external manifest contains {len(invalid)} invalid rows")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=lambda item: item.isoformat()
            if hasattr(item, "isoformat")
            else str(item),
        )
    )


if __name__ == "__main__":
    main()
