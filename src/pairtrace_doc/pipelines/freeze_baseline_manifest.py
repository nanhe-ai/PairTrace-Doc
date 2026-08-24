from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - publisher-provided integrity checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_baseline_assets(
    scratch: Path, baselines: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for name, baseline in baselines.items():
        repository = _resolve(scratch, baseline["repository_path"]).resolve()
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != baseline["revision"]:
            raise ValueError(f"{name} repository revision changed")
        assets: dict[str, Any] = {}
        for key, value in baseline.items():
            if not key.endswith("_path") or key == "repository_path":
                continue
            path = _resolve(scratch, str(value)).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            prefix = key.removesuffix("_path")
            actual: dict[str, str] = {}
            expected_sha256 = baseline.get(f"{prefix}_sha256")
            if expected_sha256 is not None:
                actual["sha256"] = _sha256_file(path)
                if actual["sha256"] != expected_sha256:
                    raise ValueError(f"{name} {key} SHA-256 changed")
            expected_md5 = baseline.get(f"{prefix}_md5")
            if expected_md5 is not None:
                actual["md5"] = _md5_file(path)
                if actual["md5"] != expected_md5:
                    raise ValueError(f"{name} {key} MD5 changed")
            if not actual:
                raise ValueError(f"{name} {key} has no frozen checksum")
            assets[key] = {"path": str(value), **actual}
        validated[name] = {
            "repository_revision": revision,
            "assets": assets,
        }
    return validated


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record(
    source: dict[str, Any],
    evaluation_role: str,
    sample_kind: str,
) -> dict[str, Any]:
    authentic = sample_kind == "authentic"
    edition = "authentic" if authentic else str(source["edition"])
    record_id = f"{evaluation_role}:{sample_kind}:{source['source_group_id']}"
    if not authentic:
        record_id += f":{source['sample_id']}"
    return {
        "record_id": record_id,
        "source_sample_id": source["sample_id"],
        "source_group_id": source["source_group_id"],
        "evaluation_role": evaluation_role,
        "sample_kind": sample_kind,
        "edition": edition,
        "generator": "authentic" if authentic else source.get("assigned_tool"),
        "source_dataset": source.get("source_dataset"),
        "image": source["authentic"] if authentic else source["image"],
        "image_sha256": (
            source.get("authentic_sha256") if authentic else source.get("image_sha256")
        ),
        "image_pixel_sha256": (
            source.get("authentic_pixel_sha256") if authentic else None
        ),
        "mask": None if authentic else source["mask"],
        "mask_sha256": None if authentic else source.get("mask_sha256"),
        "mask_state": "implicit_all_zero" if authentic else "nonempty_ground_truth",
        "height": (
            source.get("authentic_height") if authentic else source.get("image_height")
        ),
        "width": (
            source.get("authentic_width") if authentic else source.get("image_width")
        ),
        "model_or_threshold_selection_allowed": evaluation_role == "validation",
        "paper_evidence": False,
        "freeze_id": source.get("freeze_id"),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    asset_validation = None
    if config["runtime"].get("validate_assets", False):
        scratch = Path(
            os.environ.get(
                config["runtime"]["scratch_env"],
                str(_resolve(project_root, config["runtime"]["scratch_default"])),
            )
        ).resolve()
        asset_validation = _validate_baseline_assets(scratch, config["baselines"])

    data_config = config["data"]
    pilot_path = _resolve(project_root, data_config["pilot_manifest"])
    output_path = _resolve(project_root, data_config["baseline_manifest"])
    summary_path = _resolve(project_root, data_config["freeze_summary"])
    pilot_sha256 = _sha256_file(pilot_path)
    expected_sha256 = data_config.get("expected_pilot_manifest_sha256")
    if expected_sha256 and pilot_sha256 != expected_sha256:
        raise ValueError(
            f"pilot manifest SHA-256 changed: {pilot_sha256} != {expected_sha256}"
        )

    pilot_rows = _read_jsonl(pilot_path)
    by_role: dict[str, list[dict[str, Any]]] = {}
    for role in ("validation", "in_domain_test", "generator_holdout"):
        by_role[role] = [
            row for row in pilot_rows if row.get("pilot_role") == role
        ]
    expected = int(data_config["expected_per_role"])
    if any(len(rows) != expected for rows in by_role.values()):
        raise ValueError("pilot roles do not contain the expected number of records")

    validation_groups = {
        str(row["source_group_id"]) for row in by_role["validation"]
    }
    in_domain_by_group = {
        str(row["source_group_id"]): row for row in by_role["in_domain_test"]
    }
    holdout_by_group = {
        str(row["source_group_id"]): row for row in by_role["generator_holdout"]
    }
    test_groups = set(in_domain_by_group)
    if len(validation_groups) != expected or len(test_groups) != expected:
        raise ValueError("validation or test role contains duplicate source groups")
    if set(holdout_by_group) != test_groups:
        raise ValueError("v1 in-domain and v2 holdout source groups are not paired")
    if validation_groups & test_groups:
        raise ValueError("validation and final-test source groups overlap")
    for group in sorted(test_groups):
        v1_row = in_domain_by_group[group]
        v2_row = holdout_by_group[group]
        if v2_row.get("joined_v1_sample_id") != v1_row.get("sample_id"):
            raise ValueError(f"v2-to-v1 pair identity mismatch for source group {group}")

    output_rows: list[dict[str, Any]] = []
    for row in sorted(by_role["validation"], key=lambda item: item["source_group_id"]):
        output_rows.append(_record(row, "validation", "forged"))
        output_rows.append(_record(row, "validation", "authentic"))
    for group in sorted(test_groups):
        v1_row = in_domain_by_group[group]
        v2_row = holdout_by_group[group]
        output_rows.append(_record(v1_row, "in_domain_test", "forged"))
        output_rows.append(_record(v2_row, "generator_holdout", "forged"))
        output_rows.append(_record(v1_row, "final_test", "authentic"))

    record_ids = [str(row["record_id"]) for row in output_rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("baseline manifest contains duplicate record IDs")
    if any(
        row["evaluation_role"] != "validation"
        and row["model_or_threshold_selection_allowed"]
        for row in output_rows
    ):
        raise ValueError("a final-test row permits model or threshold selection")

    protocol = {
        "pilot_manifest_sha256": pilot_sha256,
        "seed": int(config["experiment"]["seed"]),
        "record_ids": record_ids,
        "baselines": config["baselines"],
        "evaluation": config["evaluation"],
    }
    baseline_freeze_id = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for row in output_rows:
        row["baseline_freeze_id"] = baseline_freeze_id

    _write_jsonl(output_path, output_rows)
    summary = {
        "experiment": config["experiment"],
        "status": "baseline_inputs_frozen_gpu_not_authorized",
        "paper_evidence": False,
        "gpu_launch_authorized": bool(config["runtime"]["gpu_launch_authorized"]),
        "method_training_authorized": bool(
            config["runtime"]["method_training_authorized"]
        ),
        "baseline_freeze_id": baseline_freeze_id,
        "input": {
            "path": str(pilot_path.relative_to(project_root)),
            "sha256": pilot_sha256,
            "rows": len(pilot_rows),
        },
        "output": {
            "path": str(output_path.relative_to(project_root)),
            "sha256": _sha256_file(output_path),
            "rows": len(output_rows),
            "counts": dict(
                sorted(
                    Counter(
                        f"{row['evaluation_role']}:{row['sample_kind']}"
                        for row in output_rows
                    ).items()
                )
            ),
        },
        "leakage": {
            "validation_source_groups": len(validation_groups),
            "final_test_source_groups": len(test_groups),
            "validation_to_final_test_overlap": len(validation_groups & test_groups),
            "in_domain_to_generator_holdout_pair_mismatches": 0,
        },
        "baselines": config["baselines"],
        "asset_validation": asset_validation,
    }
    if summary["gpu_launch_authorized"] or summary["method_training_authorized"]:
        raise ValueError("input-freeze config must not authorize GPU use or training")
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
