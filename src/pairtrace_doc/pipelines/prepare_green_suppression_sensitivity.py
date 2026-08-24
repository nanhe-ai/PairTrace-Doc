from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_json,
    _write_jsonl,
)


def _green_selection(image: np.ndarray, detector: dict[str, Any]) -> np.ndarray:
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    return (
        (green >= int(detector["min_green_channel"]))
        & (green - red >= int(detector["min_green_minus_red"]))
        & (green - blue >= int(detector["min_green_minus_blue"]))
    )


def _mask_blind_green_inpaint(
    image: np.ndarray,
    detector: dict[str, Any],
    transform: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("green inpaint expects an uint8 RGB image")
    selected = _green_selection(image, detector)
    iterations = int(transform["dilation_iterations"])
    kernel_size = int(transform["dilation_kernel_size"])
    if kernel_size < 1 or kernel_size % 2 == 0 or iterations < 0:
        raise ValueError("green inpaint dilation configuration is invalid")
    mask = selected.astype(np.uint8) * 255
    if iterations:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=iterations)
    if np.any(mask):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        method = str(transform["method"])
        if method != "telea":
            raise ValueError(f"unsupported inpaint method: {method}")
        transformed_bgr = cv2.inpaint(
            bgr,
            mask,
            float(transform["radius_px"]),
            cv2.INPAINT_TELEA,
        )
        transformed = cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2RGB)
    else:
        transformed = image.copy()
    changed = np.any(transformed != image, axis=2)
    total = image.shape[0] * image.shape[1]
    return transformed, {
        "selected_green_pixels": int(selected.sum()),
        "selected_green_fraction": float(selected.mean()),
        "inpaint_mask_pixels": int(np.count_nonzero(mask)),
        "inpaint_mask_fraction": float(np.count_nonzero(mask) / total),
        "changed_pixels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
    }


def _array_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(image.tobytes(order="C")).hexdigest()


def _cache_key(
    source_sha256: str,
    detector: dict[str, Any],
    transform: dict[str, Any],
    schema_version: int,
) -> str:
    payload = {
        "source_sha256": source_sha256,
        "detector": detector,
        "transform": transform,
        "schema_version": schema_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _verify(path: Path, expected: str, label: str) -> None:
    digest = _sha256(path)
    if digest != expected:
        raise ValueError(f"{label} SHA-256 changed: {digest} != {expected}")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("green-suppression preflight config must be a mapping")
    experiment = config["experiment"]
    runtime = config["runtime"]
    if bool(experiment["paper_evidence"]):
        raise ValueError("post-hoc green sensitivity cannot be confirmatory evidence")
    if runtime["device"] != "cpu" or not bool(runtime["preflight_only"]):
        raise ValueError("green-suppression preparation must be CPU preflight only")
    if any(
        bool(runtime[key])
        for key in (
            "gpu_launch_authorized",
            "model_inference_authorized",
            "model_training_authorized",
            "threshold_selection_authorized",
            "sample_replacement_authorized",
        )
    ):
        raise ValueError("green-suppression preflight crossed an evidence boundary")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            str(paths["scratch_env"]),
            str(_resolve(project_root, str(paths["scratch_default"]))),
        )
    ).resolve()
    verified: dict[str, Path] = {}
    for key, label in (
        ("protocol", "green-suppression sensitivity protocol"),
        ("manifest", "frozen reserve manifest"),
        ("audit_records", "green-boundary audit records"),
        ("audit_summary", "green-boundary audit summary"),
        ("frozen_predictions", "frozen final predictions"),
    ):
        path = _resolve(project_root, str(paths[key]))
        _verify(path, str(paths[f"expected_{key}_sha256"]), label)
        verified[key] = path
    encoder = _resolve(scratch, str(config["models"]["encoder_weights"]))
    _verify(
        encoder,
        str(config["models"]["encoder_weights_sha256"]),
        "frozen encoder weights",
    )
    checkpoints: dict[str, dict[str, str]] = {}
    for name, specification in config["models"]["checkpoints"].items():
        path = _resolve(project_root, str(specification["path"]))
        _verify(path, str(specification["sha256"]), f"frozen checkpoint {name}")
        checkpoints[str(name)] = {
            "path": str(path.relative_to(project_root)),
            "sha256": _sha256(path),
        }

    manifest = _read_jsonl(verified["manifest"])
    forged = {
        str(row["source_sample_id"]): row
        for row in manifest
        if row.get("sample_kind") == "forged"
    }
    requested = [str(value) for value in config["selection"]["source_sample_ids"]]
    missing = [value for value in requested if value not in forged]
    if missing:
        raise ValueError(f"preflight source sample IDs are missing: {missing}")
    audit = {
        str(row["source_sample_id"]): row
        for row in _read_jsonl(verified["audit_records"])
        if row.get("status") == "ok"
    }
    known_positive = set(
        str(value) for value in config["expectations"]["artifact_positive_ids"]
    )
    known_negative = set(
        str(value) for value in config["expectations"]["artifact_negative_ids"]
    )
    if known_positive | known_negative != set(requested):
        raise ValueError("preflight artifact expectations do not cover toy selection")
    for sample_id in known_positive:
        if audit.get(sample_id, {}).get("artifact_positive") is not True:
            raise ValueError(f"expected artifact-positive toy case changed: {sample_id}")
    for sample_id in known_negative:
        if audit.get(sample_id, {}).get("artifact_positive") is not False:
            raise ValueError(f"expected artifact-negative toy case changed: {sample_id}")

    detector = config["detector"]
    transform = config["transform"]
    cache_schema = int(transform["cache_schema_version"])
    cache_dir = _resolve(scratch, str(paths["transform_cache_dir"]))
    records: list[dict[str, Any]] = []
    failures = 0
    for sample_id in requested:
        row = forged[sample_id]
        record = {
            "source_sample_id": sample_id,
            "source_group_id": row["source_group_id"],
            "evaluation_role": row["evaluation_role"],
            "generator": row["generator"],
            "source_dataset": row["source_dataset"],
            "artifact_positive": bool(audit[sample_id]["artifact_positive"]),
            "paper_evidence": False,
            "model_inference_performed": False,
        }
        try:
            image_path = _resolve(scratch, str(row["image"]))
            _verify(image_path, str(row["image_sha256"]), "toy candidate")
            with Image.open(image_path) as handle:
                image = np.asarray(handle.convert("RGB"))
            transformed, diagnostics = _mask_blind_green_inpaint(
                image, detector, transform
            )
            key = _cache_key(str(row["image_sha256"]), detector, transform, cache_schema)
            cache_path = cache_dir / key[:2] / f"{key}.png"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not cache_path.is_file():
                temporary = cache_path.with_suffix(".png.tmp")
                with temporary.open("wb") as handle:
                    Image.fromarray(transformed).save(handle, format="PNG")
                temporary.replace(cache_path)
            with Image.open(cache_path) as handle:
                replay = np.asarray(handle.convert("RGB"))
            if not np.array_equal(replay, transformed):
                raise ValueError("transformed cache changed pixels")
            if _sha256(image_path) != row["image_sha256"]:
                raise ValueError("raw candidate changed during preflight")
            if bool(record["artifact_positive"]) and diagnostics["changed_pixels"] == 0:
                raise ValueError("artifact-positive toy case had no transformed pixels")
            record.update(
                {
                    "status": "ok",
                    "error": None,
                    "source_image_sha256": row["image_sha256"],
                    "transformed_array_sha256": _array_sha256(transformed),
                    "transform_cache_key": key,
                    "transform_cache": str(cache_path.relative_to(scratch)),
                    **diagnostics,
                }
            )
        except Exception as exc:
            failures += 1
            record.update(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            )
        records.append(record)

    records_path = _resolve(project_root, str(paths["preflight_records"]))
    summary_path = _resolve(project_root, str(paths["preflight_summary"]))
    _write_jsonl(records_path, records)
    implementation = Path(__file__).resolve()
    summary = {
        "schema_version": 1,
        "experiment": experiment,
        "status": "green_suppression_cpu_preflight_passed" if failures == 0 else "failed",
        "paper_evidence": False,
        "model_inference_performed": False,
        "model_training_performed": False,
        "gpu_required_for_next_stage": failures == 0,
        "records": {"total": len(records), "ok": len(records) - failures, "failures": failures},
        "transform": transform,
        "detector": detector,
        "input": {
            "config": str(config_path.relative_to(project_root)),
            "config_sha256": _sha256(config_path),
            "implementation": (
                str(implementation.relative_to(project_root))
                if implementation.is_relative_to(project_root)
                else f"src/pairtrace_doc/pipelines/{implementation.name}"
            ),
            "implementation_sha256": _sha256(implementation),
            **{
                key: {
                    "path": str(path.relative_to(project_root)),
                    "sha256": _sha256(path),
                }
                for key, path in verified.items()
            },
            "encoder_weights_sha256": _sha256(encoder),
            "checkpoints": checkpoints,
        },
        "output": {
            "preflight_records": str(records_path.relative_to(project_root)),
            "preflight_records_sha256": _sha256(records_path),
            "transform_cache_dir": str(cache_dir.relative_to(scratch)),
        },
    }
    _write_json(summary_path, summary)
    if failures:
        raise RuntimeError(f"green-suppression CPU preflight failures: {failures}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
