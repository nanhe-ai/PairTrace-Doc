from __future__ import annotations

import argparse
import json
import os
import unicodedata
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _invalid_thread_value = int(os.environ.get(_thread_variable, "1")) < 1
    except ValueError:
        _invalid_thread_value = True
    if _invalid_thread_value:
        os.environ[_thread_variable] = "1"
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from pairtrace_doc.pipelines.prepare_prospective_editor_toy3 import (
    _replacement,
    _select_target,
)
from pairtrace_doc.pipelines.run_prospective_editor_v2_placement_toy import (
    _hash_text,
    _order_quad,
    _pixel_sha256,
    _quad_size,
    _read_jsonl,
    _render_blend as _render_blend_v1,
    _resolve,
    _save_png,
    _sha256,
    _target_crop,
    _verify_package_versions,
    _write_json,
    _write_jsonl,
)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32)
    return (
        values[..., 0] * 0.2126
        + values[..., 1] * 0.7152
        + values[..., 2] * 0.0722
    )


def _estimate_ink_v1_1(
    source: np.ndarray,
    polygon_mask: np.ndarray,
    darkest_fraction: float,
    minimum_contrast: float,
) -> tuple[np.ndarray, dict[str, float]]:
    pixels = source[polygon_mask]
    if len(pixels) == 0:
        raise ValueError("empty undilated target polygon")
    luminance = _luminance(pixels)
    foreground_cutoff = float(np.quantile(luminance, darkest_fraction))
    background_cutoff = float(np.quantile(luminance, 0.5))
    foreground_pixels = pixels[luminance <= foreground_cutoff]
    background_pixels = pixels[luminance >= background_cutoff]
    if len(foreground_pixels) == 0 or len(background_pixels) == 0:
        raise ValueError("cannot estimate target foreground/background")
    foreground = np.median(foreground_pixels, axis=0).astype(np.float32)
    background = np.median(background_pixels, axis=0).astype(np.float32)
    background_luminance = float(_luminance(background))
    foreground_luminance = float(_luminance(foreground))
    observed_contrast = background_luminance - foreground_luminance
    required_luminance = max(0.0, background_luminance - minimum_contrast)
    if foreground_luminance > required_luminance:
        if foreground_luminance > 0:
            foreground *= required_luminance / foreground_luminance
        else:
            foreground[:] = 0
    ink = np.clip(np.rint(foreground), 0, 255).astype(np.uint8)
    final_contrast = background_luminance - float(_luminance(ink))
    return ink, {
        "background_luminance": round(background_luminance, 8),
        "darkest_fraction": float(darkest_fraction),
        "final_contrast": round(final_contrast, 8),
        "minimum_contrast": float(minimum_contrast),
        "observed_contrast": round(observed_contrast, 8),
    }


def _render_blend_v1_1(
    source: np.ndarray,
    polygon: list[list[float]],
    cleartext: str,
    replacement: str,
    fonts: list[dict[str, Any]],
    dilation_fraction: float,
    minimum_dilation: int,
    inpaint_radius: float,
    supersampling: int,
    darkest_fraction: float,
    minimum_contrast: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    _, erase_mask, alpha, inpainted, metadata = _render_blend_v1(
        source,
        polygon,
        cleartext,
        replacement,
        fonts,
        dilation_fraction,
        minimum_dilation,
        inpaint_radius,
        supersampling,
    )
    quad = _order_quad(polygon)
    polygon_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(polygon_mask, np.rint(quad).astype(np.int32), 255)
    ink, ink_metadata = _estimate_ink_v1_1(
        source,
        polygon_mask > 0,
        darkest_fraction,
        minimum_contrast,
    )
    base = source.copy()
    base[erase_mask] = inpainted[erase_mask]
    alpha_float = alpha.astype(np.float32)[..., None] / 255.0
    candidate = np.rint(
        base.astype(np.float32) * (1.0 - alpha_float)
        + ink.astype(np.float32)[None, None, :] * alpha_float
    ).astype(np.uint8)
    candidate[~erase_mask] = source[~erase_mask]
    changed = np.any(candidate != source, axis=2)
    metadata.update(
        {
            "ink_estimator": "darkest_10_percent_contrast_floor_v1_1",
            "ink_estimator_metadata": ink_metadata,
            "ink_rgb": [int(value) for value in ink],
            "inside_changed_fraction": round(
                float(
                    np.count_nonzero(changed & erase_mask)
                    / np.count_nonzero(erase_mask)
                ),
                8,
            ),
            "inside_changed_pixels": int(np.count_nonzero(changed & erase_mask)),
            "outside_changed_pixels": int(np.count_nonzero(changed & ~erase_mask)),
        }
    )
    return candidate, erase_mask, alpha, inpainted, metadata


def _normalize_ocr_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(character for character in normalized if not character.isspace())


def _boundary_normalized_match(token: str, replacement: str) -> str | None:
    normalized_token = _normalize_ocr_token(token)
    normalized_replacement = _normalize_ocr_token(replacement)
    if normalized_token == normalized_replacement:
        return "exact"
    if normalized_token.count(normalized_replacement) != 1:
        return None
    prefix, suffix = normalized_token.split(normalized_replacement, 1)
    boundary = prefix + suffix
    if boundary and all(not character.isalnum() for character in boundary):
        return "nonalphanumeric_boundary"
    return None


def _rectify_target(array: np.ndarray, polygon: list[list[float]]) -> np.ndarray:
    quad = _order_quad(polygon)
    width, height = _quad_size(quad)
    rectangle = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(quad, rectangle)
    return cv2.warpPerspective(
        array,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _ocr_rectified(
    ocr: Any,
    candidate: np.ndarray,
    polygon: list[list[float]],
    replacement: str,
    scale: int,
    padding_fraction: float,
) -> dict[str, Any]:
    rectified = _rectify_target(candidate, polygon)
    upsampled = cv2.resize(
        rectified,
        (rectified.shape[1] * scale, rectified.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    pad = max(16, int(round(min(upsampled.shape[:2]) * padding_fraction)))
    padded = cv2.copyMakeBorder(
        upsampled, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    result_objects = list(ocr.predict(padded))
    texts: list[str] = []
    scores: list[float] = []
    for result in result_objects:
        payload = result.json["res"]
        for text, score in zip(
            payload.get("rec_texts", []), payload.get("rec_scores", [])
        ):
            normalized = _normalize_ocr_token(str(text))
            if normalized:
                texts.append(normalized)
                scores.append(float(score))
    reason = (
        _boundary_normalized_match(texts[0], replacement)
        if len(texts) == 1
        else None
    )
    return {
        "accepted": reason is not None,
        "acceptance_reason": reason,
        "matching_score": round(scores[0], 8) if reason is not None else None,
        "recognized_text_hashes": [_hash_text(text) for text in texts],
        "recognized_text_lengths": [len(text) for text in texts],
        "recognized_token_count": len(texts),
        "rectified_target_size": [int(rectified.shape[1]), int(rectified.shape[0])],
    }


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _review_sheet(
    source_crop: np.ndarray,
    inpaint_crop: np.ndarray,
    candidate_crop: np.ndarray,
    replacement: str,
    record_id: str,
) -> Image.Image:
    panel_size = (512, 384)
    sheet = Image.new("RGB", (1536, 460), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 12),
        f"cpu_render_blend_v1_1 | {record_id} | replacement={replacement} | non-human review",
        fill="black",
        font=ImageFont.load_default(),
    )
    for index, (array, label) in enumerate(
        [
            (source_crop, "frozen source target"),
            (inpaint_crop, "TELEA background"),
            (candidate_crop, "rendered candidate"),
        ]
    ):
        image = _fit(Image.fromarray(array), panel_size)
        sheet.paste(image, (index * 512, 64))
        draw.text(
            (index * 512 + 8, 438),
            label,
            fill="black",
            font=ImageFont.load_default(),
        )
    return sheet


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
    _verify_package_versions(config["environment"]["packages"])
    for entry in config["frozen_inputs"].values():
        path = _resolve(project_root, str(entry["path"]))
        if _sha256(path) != str(entry["sha256"]):
            raise ValueError(f"frozen input changed: {path}")
    for entry in config["fonts"]:
        path = Path(str(entry["absolute_path"]))
        if _sha256(path) != str(entry["sha256"]):
            raise ValueError(f"font changed: {path}")
    if _sha256(detection_model_dir / "inference.pdiparams") != str(
        config["ocr"]["detection_weight_sha256"]
    ):
        raise ValueError("OCR detection weight changed")
    if _sha256(recognition_model_dir / "inference.pdiparams") != str(
        config["ocr"]["recognition_weight_sha256"]
    ):
        raise ValueError("OCR recognition weight changed")

    manifest_path = _resolve(
        project_root, str(config["frozen_inputs"]["source_manifest"]["path"])
    )
    audit_path = _resolve(
        project_root, str(config["frozen_inputs"]["quality_audit"]["path"])
    )
    sources = _read_jsonl(manifest_path)
    if len(sources) != 3:
        raise ValueError("expected exactly three frozen V2.1 placement sources")
    audit_by_id = {
        str(row["record_id"]): row for row in _read_jsonl(audit_path)
    }
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

    artifact_root = _resolve(storage_root, str(config["outputs"]["artifact_root"]))
    records: list[dict[str, Any]] = []
    for source_row in sources:
        source_path = _resolve(storage_root, str(source_row["path"]))
        if _sha256(source_path) != str(source_row["encoded_sha256"]):
            raise ValueError(f"V2.1 source hash changed: {source_row['v2_placement_id']}")
        with Image.open(source_path) as handle:
            source_image = ImageOps.exif_transpose(handle).convert("RGB")
        source = np.asarray(source_image)
        if _pixel_sha256(source) != str(source_row["decoded_pixel_sha256"]):
            raise ValueError("V2.1 decoded-pixel hash changed")
        if source_image.size != (int(source_row["width"]), int(source_row["height"])):
            raise ValueError("V2.1 source dimensions changed")
        source_ocr_objects = list(ocr.predict(source))
        if len(source_ocr_objects) != 1:
            raise ValueError("unexpected OCR page count")
        source_ocr = source_ocr_objects[0].json["res"]
        naf_fields: list[dict[str, Any]] = []
        if source_row["source_dataset"] == "NAF":
            audit_row = audit_by_id[str(source_row["source_record_id"])]
            annotation_path = _resolve(
                storage_root,
                str(audit_row["selection_metadata"]["annotation_relative_path"]),
            )
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            naf_fields = list(annotation.get("fieldBBs", []))
        target_selection_row = dict(source_row)
        target_selection_row["rehearsal_id"] = source_row["v2_placement_id"]
        selected = _select_target(source_ocr, target_selection_row, config, naf_fields)
        cleartext = str(selected.pop("_cleartext"))
        replacement = _replacement(
            cleartext,
            f"{config['experiment']['seed']}|replacement|{source_row['v2_placement_id']}",
        )
        candidate, erase_mask, alpha, inpainted, metadata = _render_blend_v1_1(
            source,
            selected["polygon"],
            cleartext,
            replacement,
            config["fonts"],
            float(config["render"]["dilation_fraction_of_target_height"]),
            int(config["render"]["minimum_dilation_pixels"]),
            float(config["render"]["inpaint_radius"]),
            int(config["render"]["supersampling"]),
            float(config["render"]["ink_darkest_fraction"]),
            float(config["render"]["minimum_luminance_contrast"]),
        )
        margin = max(12, int(round(metadata["target_rectangle_size"][1] * 1.5)))
        candidate_crop, crop_box = _target_crop(candidate, erase_mask, margin)
        source_crop = source[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
        inpaint_crop = inpainted[
            crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]
        ]
        verification = _ocr_rectified(
            ocr,
            candidate,
            selected["polygon"],
            replacement,
            int(config["verification"]["ocr_upsample_scale"]),
            float(config["verification"]["ocr_padding_fraction"]),
        )

        artifact_id = _hash_text(
            f"{source_row['v2_placement_freeze_id']}|{source_row['encoded_sha256']}|"
            f"{selected['rank_digest']}|{config['method']['id']}"
        )
        artifact_dir = artifact_root / artifact_id
        candidate_path = artifact_dir / "candidate_full.png"
        mask_path = artifact_dir / "erase_mask.png"
        alpha_path = artifact_dir / "render_alpha.png"
        inpaint_path = artifact_dir / "inpainted_target_crop.png"
        review_path = artifact_dir / "review_sheet.png"
        _save_png(candidate_path, Image.fromarray(candidate))
        _save_png(mask_path, Image.fromarray(erase_mask.astype(np.uint8) * 255))
        _save_png(alpha_path, Image.fromarray(alpha))
        _save_png(inpaint_path, Image.fromarray(inpaint_crop))
        _save_png(
            review_path,
            _review_sheet(
                source_crop,
                inpaint_crop,
                candidate_crop,
                replacement,
                str(source_row["v2_placement_id"]),
            ),
        )
        automatic_checks = {
            "alpha_inside_declared_mask": metadata["alpha_outside_mask_pixels"] == 0,
            "glyph_not_clipped": not metadata["glyph_clipped"],
            "inside_changed_fraction_at_least_minimum": metadata[
                "inside_changed_fraction"
            ]
            >= float(config["verification"]["minimum_inside_changed_fraction"]),
            "ocr_rectified_boundary_exact_replacement": verification["accepted"],
            "outside_mask_changed_pixels_equal_zero": metadata[
                "outside_changed_pixels"
            ]
            == 0,
        }
        records.append(
            {
                "artifact_id": artifact_id,
                "automatic_checks": automatic_checks,
                "automatic_gate_passed": all(automatic_checks.values()),
                "candidate_path": candidate_path.relative_to(storage_root).as_posix(),
                "candidate_sha256": _sha256(candidate_path),
                "character_pattern": selected["character_pattern"],
                "clear_source_text_persisted": False,
                "eligible_ocr_candidate_count": selected["eligible_candidate_count"],
                "erase_mask_path": mask_path.relative_to(storage_root).as_posix(),
                "erase_mask_sha256": _sha256(mask_path),
                "font_id": metadata["font_id"],
                "font_sha256": metadata["font_sha256"],
                "font_size": metadata["font_size"],
                "inpainted_target_crop_path": inpaint_path.relative_to(
                    storage_root
                ).as_posix(),
                "inpainted_target_crop_sha256": _sha256(inpaint_path),
                "method_id": str(config["method"]["id"]),
                "ocr_score": round(float(selected["ocr_score"]), 8),
                "render_alpha_path": alpha_path.relative_to(storage_root).as_posix(),
                "render_alpha_sha256": _sha256(alpha_path),
                "render_metadata": metadata,
                "replacement_text": replacement,
                "review_sheet_path": review_path.relative_to(storage_root).as_posix(),
                "review_sheet_sha256": _sha256(review_path),
                "source_decoded_pixel_sha256": source_row["decoded_pixel_sha256"],
                "source_encoded_sha256": source_row["encoded_sha256"],
                "source_text_length": len(cleartext),
                "source_text_sha256": _hash_text(cleartext),
                "target_crop_box_xyxy": crop_box,
                "target_polygon_source_xy": selected["polygon"],
                "v2_placement_id": source_row["v2_placement_id"],
                "verification": verification,
                "visual_review": "pending_agent_nonhuman_review",
            }
        )
        del cleartext

    records_path = _resolve(project_root, str(config["outputs"]["records"]))
    report_path = _resolve(project_root, str(config["outputs"]["report"]))
    _write_jsonl(records_path, records)
    result = {
        "authorization": {
            "detector_inference_run": False,
            "final_source_images_read": False,
            "neural_editor_inference_run": False,
            "pilot100_run": False,
            "v2_1_nonfinal_source_images_read": len(records),
        },
        "automatic_gate_passed": len(records) == 3
        and all(row["automatic_gate_passed"] for row in records),
        "automatic_passed_records": sum(
            int(row["automatic_gate_passed"]) for row in records
        ),
        "clear_source_text_persisted": False,
        "method_id": str(config["method"]["id"]),
        "records": str(records_path.relative_to(project_root)),
        "records_sha256": _sha256(records_path),
        "review_status": "pending_agent_nonhuman_review",
        "rows": len(records),
        "status": "automatic_complete_agent_visual_review_pending",
    }
    _write_json(report_path, result)
    result["report"] = str(report_path.relative_to(project_root))
    result["report_sha256"] = _sha256(report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run final deterministic CPU text placement on frozen V2.1 toy"
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
