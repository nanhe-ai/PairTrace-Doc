from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _invalid_thread_value = int(os.environ.get(_thread_variable, "1")) < 1
    except ValueError:
        _invalid_thread_value = True
    if _invalid_thread_value:
        os.environ[_thread_variable] = "1"
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

import numpy as np
import yaml
from PIL import Image, ImageOps

from pairtrace_doc.pipelines.run_prospective_editor_v2_placement_toy import (
    _ocr_exact,
    _pixel_sha256,
)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _pixel_metrics(
    source: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> tuple[int, int, float]:
    if source.shape != candidate.shape:
        raise ValueError("candidate shape differs from source")
    if source.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError("source and candidate must be uint8")
    changed = np.any(source != candidate, axis=2)
    outside = int(np.count_nonzero(changed & ~mask))
    inside = int(np.count_nonzero(changed & mask))
    fraction = float(inside / np.count_nonzero(mask)) if np.any(mask) else 0.0
    return outside, inside, fraction


def run(
    config_path: Path,
    project_root: Path,
    storage_root: Path,
    detection_model_dir: Path,
    recognition_model_dir: Path,
) -> dict[str, Any]:
    from paddleocr import PaddleOCR

    config_path = config_path.resolve()
    project_root = project_root.resolve()
    storage_root = storage_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    for entry in config["frozen_inputs"].values():
        path = _resolve(project_root, str(entry["path"]))
        observed = _sha256(path)
        if observed != str(entry["sha256"]):
            raise ValueError(f"frozen input changed: {path}")
    if _sha256(detection_model_dir / "inference.pdiparams") != str(
        config["ocr"]["detection_weight_sha256"]
    ):
        raise ValueError("OCR detection weight changed")
    if _sha256(recognition_model_dir / "inference.pdiparams") != str(
        config["ocr"]["recognition_weight_sha256"]
    ):
        raise ValueError("OCR recognition weight changed")

    source_manifest = _read_jsonl(
        _resolve(
            project_root,
            str(config["frozen_inputs"]["source_manifest"]["path"]),
        )
    )
    sources = {str(row["v2_placement_id"]): row for row in source_manifest}
    records = _read_jsonl(
        _resolve(project_root, str(config["frozen_inputs"]["records"]["path"]))
    )
    if len(records) != 3 or set(sources) != {
        str(row["v2_placement_id"]) for row in records
    }:
        raise ValueError("records do not match the frozen three-source manifest")

    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir=str(detection_model_dir.resolve()),
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        text_recognition_model_dir=str(recognition_model_dir.resolve()),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=int(config["ocr"]["text_det_limit_side_len"]),
        text_det_limit_type="max",
        device="cpu",
    )
    audit_rows: list[dict[str, Any]] = []
    for record in records:
        source_row = sources[str(record["v2_placement_id"])]
        source_path = _resolve(storage_root, str(source_row["path"]))
        candidate_path = _resolve(storage_root, str(record["candidate_path"]))
        mask_path = _resolve(storage_root, str(record["erase_mask_path"]))
        alpha_path = _resolve(storage_root, str(record["render_alpha_path"]))
        for path, expected in (
            (source_path, source_row["encoded_sha256"]),
            (candidate_path, record["candidate_sha256"]),
            (mask_path, record["erase_mask_sha256"]),
            (alpha_path, record["render_alpha_sha256"]),
        ):
            if _sha256(path) != str(expected):
                raise ValueError(f"artifact hash changed: {path}")
        with Image.open(source_path) as handle:
            source = np.asarray(ImageOps.exif_transpose(handle).convert("RGB"))
        with Image.open(candidate_path) as handle:
            candidate = np.asarray(handle.convert("RGB"))
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("L")) > 0
        with Image.open(alpha_path) as handle:
            alpha = np.asarray(handle.convert("L"))
        if _pixel_sha256(source) != str(source_row["decoded_pixel_sha256"]):
            raise ValueError("source decoded-pixel hash changed")
        outside, inside, fraction = _pixel_metrics(source, candidate, mask)
        box = [int(value) for value in record["target_crop_box_xyxy"]]
        crop = candidate[box[1] : box[3], box[0] : box[2]]
        verification = _ocr_exact(
            ocr,
            crop,
            str(record["replacement_text"]),
            int(config["verification"]["ocr_upsample_scale"]),
        )
        checks = {
            "alpha_inside_declared_mask": int(
                np.count_nonzero((alpha > 0) & ~mask)
            )
            == 0,
            "candidate_dimensions_match_source": candidate.shape == source.shape,
            "candidate_finite": bool(
                np.isfinite(candidate.astype(np.float32)).all()
            ),
            "inside_changed_fraction_at_least_minimum": fraction
            >= float(config["verification"]["minimum_inside_changed_fraction"]),
            "ocr_exact_replacement": bool(verification["exact_match"]),
            "outside_mask_changed_pixels_equal_zero": outside == 0,
        }
        audit_rows.append(
            {
                "automatic_gate_recomputed": all(checks.values()),
                "checks": checks,
                "inside_changed_fraction": round(fraction, 8),
                "inside_changed_pixels": inside,
                "ocr_verification": verification,
                "outside_changed_pixels": outside,
                "v2_placement_id": record["v2_placement_id"],
            }
        )

    audit_path = _resolve(project_root, str(config["outputs"]["records"]))
    report_path = _resolve(project_root, str(config["outputs"]["report"]))
    _write_jsonl(audit_path, audit_rows)
    result = {
        "authorization": {
            "detector_inference_run": False,
            "final_source_images_read": False,
            "neural_editor_inference_run": False,
            "pilot100_run": False,
            "v2_nonfinal_candidates_revalidated": len(audit_rows),
        },
        "automatic_gate_passed": len(audit_rows) == 3
        and all(row["automatic_gate_recomputed"] for row in audit_rows),
        "automatic_passed_records": sum(
            int(row["automatic_gate_recomputed"]) for row in audit_rows
        ),
        "records": str(audit_path.relative_to(project_root)),
        "records_sha256": _sha256(audit_path),
        "rows": len(audit_rows),
        "status": "independent_automatic_audit_complete",
    }
    _write_json(report_path, result)
    result["report"] = str(report_path.relative_to(project_root))
    result["report_sha256"] = _sha256(report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently audit the V2 CPU placement toy"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--detection-model-dir", type=Path, required=True)
    parser.add_argument("--recognition-model-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.config,
        args.project_root,
        args.storage_root,
        args.detection_model_dir,
        args.recognition_model_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
