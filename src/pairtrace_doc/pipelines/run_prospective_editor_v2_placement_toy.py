from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import unicodedata
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

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from pairtrace_doc.pipelines.prepare_prospective_editor_toy3 import (
    _replacement,
    _select_target,
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


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pixel_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
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


def _save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", optimize=False)
    temporary.replace(path)


def _order_quad(points: list[list[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.shape != (4, 2):
        raise ValueError("target polygon must contain exactly four points")
    center = array.mean(axis=0)
    angles = np.arctan2(array[:, 1] - center[1], array[:, 0] - center[0])
    ordered = array[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    first_edge = ordered[1] - ordered[0]
    second_edge = ordered[2] - ordered[1]
    cross = float(
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
    )
    if cross < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def _quad_size(quad: np.ndarray) -> tuple[int, int]:
    width = max(
        float(np.linalg.norm(quad[1] - quad[0])),
        float(np.linalg.norm(quad[2] - quad[3])),
    )
    height = max(
        float(np.linalg.norm(quad[3] - quad[0])),
        float(np.linalg.norm(quad[2] - quad[1])),
    )
    return max(2, int(round(width))), max(2, int(round(height)))


def _font_measure(font_path: Path, size: int, text: str) -> tuple[int, int]:
    font = ImageFont.truetype(str(font_path), size=size)
    left, top, right, bottom = font.getbbox(text)
    return max(1, right - left), max(1, bottom - top)


def _choose_font(
    cleartext: str,
    replacement: str,
    target_size: tuple[int, int],
    fonts: list[dict[str, Any]],
) -> tuple[Path, int, str]:
    target_width, target_height = target_size
    maximum_size = max(8, int(math.ceil(target_height * 2.5)))
    best: tuple[float, Path, int, str] | None = None
    for entry in fonts:
        path = Path(str(entry["absolute_path"]))
        for size in range(4, maximum_size + 1):
            width, height = _font_measure(path, size, cleartext)
            score = abs(math.log(width / target_width)) + abs(
                math.log(height / target_height)
            )
            candidate = (score, path, size, str(entry["id"]))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("no frozen font candidate")
    _, path, size, font_id = best
    while size > 4:
        width, height = _font_measure(path, size, replacement)
        if width <= target_width * 0.96 and height <= target_height * 0.96:
            break
        size -= 1
    width, height = _font_measure(path, size, replacement)
    if width > target_width or height > target_height:
        raise ValueError("replacement cannot fit the target rectangle")
    return path, size, font_id


def _render_alpha(
    replacement: str,
    font_path: Path,
    font_size: int,
    target_size: tuple[int, int],
    supersampling: int,
) -> tuple[np.ndarray, bool]:
    width, height = target_size
    canvas = Image.new("L", (width * supersampling, height * supersampling), 0)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path), size=font_size * supersampling)
    box = draw.textbbox((0, 0), replacement, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    x = (canvas.width - text_width) / 2.0 - box[0]
    y = (canvas.height - text_height) / 2.0 - box[1]
    draw.text((x, y), replacement, font=font, fill=255)
    alpha_high = np.asarray(canvas)
    clipped = bool(
        np.any(alpha_high[0, :])
        or np.any(alpha_high[-1, :])
        or np.any(alpha_high[:, 0])
        or np.any(alpha_high[:, -1])
    )
    alpha = np.asarray(
        canvas.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    return alpha, clipped


def _estimate_ink(source: np.ndarray, polygon_mask: np.ndarray) -> np.ndarray:
    pixels = source[polygon_mask]
    if len(pixels) == 0:
        raise ValueError("empty undilated target polygon")
    luminance = pixels.astype(np.float32).mean(axis=1)
    cutoff = float(np.quantile(luminance, 0.25))
    ink_pixels = pixels[luminance <= cutoff]
    ink = np.median(ink_pixels, axis=0) if len(ink_pixels) else np.array([0, 0, 0])
    background = np.median(pixels, axis=0)
    if float(ink.mean()) > float(background.mean()) - 20.0:
        ink = np.clip(background.astype(np.float32) - 80.0, 0, 255)
    return np.rint(ink).astype(np.uint8)


def _render_blend(
    source: np.ndarray,
    polygon: list[list[float]],
    cleartext: str,
    replacement: str,
    fonts: list[dict[str, Any]],
    dilation_fraction: float,
    minimum_dilation: int,
    inpaint_radius: float,
    supersampling: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("source must be uint8 RGB")
    quad = _order_quad(polygon)
    target_size = _quad_size(quad)
    target_height = target_size[1]
    polygon_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(polygon_mask, np.rint(quad).astype(np.int32), 255)
    dilation = max(minimum_dilation, int(math.ceil(dilation_fraction * target_height)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
    )
    erase_mask = cv2.dilate(polygon_mask, kernel) > 0
    inpainted = cv2.inpaint(
        source, erase_mask.astype(np.uint8) * 255, inpaint_radius, cv2.INPAINT_TELEA
    )
    font_path, font_size, font_id = _choose_font(
        cleartext, replacement, target_size, fonts
    )
    rectangle_alpha, clipped = _render_alpha(
        replacement, font_path, font_size, target_size, supersampling
    )
    rectangle = np.array(
        [
            [0, 0],
            [target_size[0] - 1, 0],
            [target_size[0] - 1, target_size[1] - 1],
            [0, target_size[1] - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(rectangle, quad)
    alpha = cv2.warpPerspective(
        rectangle_alpha,
        transform,
        (source.shape[1], source.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha_outside_mask = int(np.count_nonzero((alpha > 0) & ~erase_mask))
    ink = _estimate_ink(source, polygon_mask > 0)
    base = source.copy()
    base[erase_mask] = inpainted[erase_mask]
    alpha_float = alpha.astype(np.float32)[..., None] / 255.0
    candidate = np.rint(
        base.astype(np.float32) * (1.0 - alpha_float)
        + ink.astype(np.float32)[None, None, :] * alpha_float
    ).astype(np.uint8)
    candidate[~erase_mask] = source[~erase_mask]
    changed = np.any(candidate != source, axis=2)
    metadata = {
        "alpha_outside_mask_pixels": alpha_outside_mask,
        "erase_mask_pixels": int(np.count_nonzero(erase_mask)),
        "font_id": font_id,
        "font_path": str(font_path),
        "font_sha256": _sha256(font_path),
        "font_size": font_size,
        "glyph_clipped": clipped,
        "ink_rgb": [int(value) for value in ink],
        "inside_changed_fraction": round(
            float(np.count_nonzero(changed & erase_mask) / np.count_nonzero(erase_mask)),
            8,
        ),
        "inside_changed_pixels": int(np.count_nonzero(changed & erase_mask)),
        "outside_changed_pixels": int(np.count_nonzero(changed & ~erase_mask)),
        "target_rectangle_size": list(target_size),
    }
    return candidate, erase_mask, alpha, inpainted, metadata


def _target_crop(array: np.ndarray, mask: np.ndarray, margin: int) -> tuple[np.ndarray, list[int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(array.shape[1], int(xs.max()) + margin + 1)
    y1 = min(array.shape[0], int(ys.max()) + margin + 1)
    return array[y0:y1, x0:x1], [x0, y0, x1, y1]


def _ocr_exact(ocr: Any, crop: np.ndarray, replacement: str, scale: int) -> dict[str, Any]:
    upsampled = cv2.resize(
        crop,
        (crop.shape[1] * scale, crop.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    pad = max(16, int(round(min(upsampled.shape[:2]) * 0.15)))
    padded = cv2.copyMakeBorder(
        upsampled, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    result_objects = list(ocr.predict(padded))
    texts: list[str] = []
    scores: list[float] = []
    for result in result_objects:
        payload = result.json["res"]
        texts.extend(
            unicodedata.normalize("NFKC", str(value)).strip()
            for value in payload.get("rec_texts", [])
        )
        scores.extend(float(value) for value in payload.get("rec_scores", []))
    normalized = unicodedata.normalize("NFKC", replacement).strip()
    matching_scores = [score for text, score in zip(texts, scores) if text == normalized]
    return {
        "exact_match": bool(matching_scores),
        "matching_score": round(max(matching_scores), 8) if matching_scores else None,
        "recognized_text_hashes": [_hash_text(text) for text in texts],
        "recognized_text_lengths": [len(text) for text in texts],
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
        f"cpu_render_blend_v1 | {record_id} | replacement={replacement} | non-human review",
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
            (index * 512 + 8, 438), label, fill="black", font=ImageFont.load_default()
        )
    return sheet


def _verify_package_versions(expected: dict[str, str]) -> None:
    for distribution, version in expected.items():
        observed = importlib.metadata.version(distribution)
        if observed != str(version):
            raise ValueError(
                f"package version changed: {distribution} {observed} != {version}"
            )


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
        raise ValueError("expected exactly three frozen V2 placement sources")
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
            raise ValueError(f"V2 source hash changed: {source_row['v2_placement_id']}")
        with Image.open(source_path) as handle:
            source_image = ImageOps.exif_transpose(handle).convert("RGB")
        source = np.asarray(source_image)
        if _pixel_sha256(source) != str(source_row["decoded_pixel_sha256"]):
            raise ValueError("V2 decoded-pixel hash changed")
        if source_image.size != (int(source_row["width"]), int(source_row["height"])):
            raise ValueError("V2 source dimensions changed")
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
        selected = _select_target(
            source_ocr, target_selection_row, config, naf_fields
        )
        cleartext = str(selected.pop("_cleartext"))
        replacement = _replacement(
            cleartext,
            f"{config['experiment']['seed']}|replacement|{source_row['v2_placement_id']}",
        )
        candidate, erase_mask, alpha, inpainted, metadata = _render_blend(
            source,
            selected["polygon"],
            cleartext,
            replacement,
            config["fonts"],
            float(config["render"]["dilation_fraction_of_target_height"]),
            int(config["render"]["minimum_dilation_pixels"]),
            float(config["render"]["inpaint_radius"]),
            int(config["render"]["supersampling"]),
        )
        margin = max(12, int(round(metadata["target_rectangle_size"][1] * 1.5)))
        candidate_crop, crop_box = _target_crop(candidate, erase_mask, margin)
        source_crop = source[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
        inpaint_crop = inpainted[
            crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]
        ]
        verification = _ocr_exact(
            ocr,
            candidate_crop,
            replacement,
            int(config["verification"]["ocr_upsample_scale"]),
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
            "ocr_exact_replacement": verification["exact_match"],
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
            "v2_nonfinal_source_images_read": len(records),
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
        description="Run deterministic CPU text placement on the frozen V2 toy"
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
