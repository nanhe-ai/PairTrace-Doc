from __future__ import annotations

import argparse
import hashlib
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

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageOps


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", optimize=False)
    temporary.replace(path)


def _character_pattern(value: str) -> str:
    pattern = []
    for character in value:
        if character.isdigit():
            pattern.append("D")
        elif character.isalpha() and character.isupper():
            pattern.append("U")
        elif character.isalpha() and character.islower():
            pattern.append("L")
        elif character.isalpha():
            pattern.append("A")
        elif character.isalnum():
            pattern.append("N")
        else:
            pattern.append(character)
    return "".join(pattern)


def _replacement(value: str, seed_material: str) -> str:
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    output: list[str] = []
    for index, character in enumerate(value):
        offset = digest[index % len(digest)]
        if character.isdigit():
            output.append(str((int(character) + 1 + offset % 9) % 10))
        elif "A" <= character <= "Z":
            output.append(chr((ord(character) - 65 + 1 + offset % 25) % 26 + 65))
        elif "a" <= character <= "z":
            output.append(chr((ord(character) - 97 + 1 + offset % 25) % 26 + 97))
        elif character.isalpha():
            replacement = chr(65 + offset % 26) if character.isupper() else chr(97 + offset % 26)
            output.append(replacement if replacement != character else ("Z" if character.isupper() else "z"))
        elif character.isalnum():
            output.append(chr(65 + offset % 26))
        else:
            output.append(character)
    result = "".join(output)
    if len(result) != len(value):
        raise ValueError("replacement changed string length")
    for left, right in zip(value, result, strict=True):
        if left.isalnum() and left == right:
            raise ValueError("replacement retained an alphanumeric character")
    return result


def _polygon_area(points: list[list[float]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) / 2.0


def _bounds(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _intersection_fraction(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    return intersection / area


def _naf_print_class(
    polygon: list[list[float]], fields: list[dict[str, Any]], overlap_threshold: float
) -> str:
    candidate_bounds = _bounds(polygon)
    labels: list[tuple[float, int]] = []
    for field in fields:
        field_points = field.get("poly_points")
        if not isinstance(field_points, list) or len(field_points) < 3:
            continue
        overlap = _intersection_fraction(candidate_bounds, _bounds(field_points))
        if overlap >= overlap_threshold:
            labels.append((overlap, int(field.get("isBlank", -1))))
    if not labels:
        return "unclassified"
    label = max(labels)[1]
    if label == 2:
        return "printed_or_stamp"
    if label == 1:
        return "handwritten"
    return "unclassified"


def _context_box(
    polygon: list[list[float]],
    width: int,
    height: int,
    expansion_fraction: float,
    minimum_side: int,
    maximum_pixels: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _bounds(polygon)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    box_width = max(float(minimum_side), (x1 - x0) * (1.0 + expansion_fraction))
    box_height = max(float(minimum_side), (y1 - y0) * (1.0 + expansion_fraction))
    box_width = min(box_width, float(width))
    box_height = min(box_height, float(height))
    if box_width * box_height > maximum_pixels:
        scale = math.sqrt(maximum_pixels / (box_width * box_height))
        box_width *= scale
        box_height *= scale
    left = min(max(0.0, center_x - box_width / 2.0), width - box_width)
    top = min(max(0.0, center_y - box_height / 2.0), height - box_height)
    right = left + box_width
    bottom = top + box_height
    return (
        int(math.floor(left)),
        int(math.floor(top)),
        int(math.ceil(right)),
        int(math.ceil(bottom)),
    )


def _select_target(
    ocr_result: dict[str, Any],
    row: dict[str, Any],
    config: dict[str, Any],
    naf_fields: list[dict[str, Any]],
) -> dict[str, Any]:
    target = config["target_and_mask"]
    scores = list(ocr_result["rec_scores"])
    texts = list(ocr_result["rec_texts"])
    polygons = list(ocr_result["rec_polys"])
    if not (len(scores) == len(texts) == len(polygons)):
        raise ValueError("OCR result arrays have inconsistent lengths")
    width = int(row["width"])
    height = int(row["height"])
    candidates: list[dict[str, Any]] = []
    for score_value, text_value, polygon_value in zip(scores, texts, polygons, strict=True):
        score = float(score_value)
        text = unicodedata.normalize("NFKC", str(text_value)).strip()
        polygon = [[float(point[0]), float(point[1])] for point in polygon_value]
        if len(polygon) < 3 or score < float(target["minimum_ocr_score"]):
            continue
        alphanumeric = sum(character.isalnum() for character in text)
        if not (
            int(target["minimum_alphanumeric_characters"])
            <= alphanumeric
            <= int(target["maximum_alphanumeric_characters"])
        ):
            continue
        x0, y0, x1, y1 = _bounds(polygon)
        box_height = y1 - y0
        area_fraction = _polygon_area(polygon) / float(width * height)
        if box_height < int(target["minimum_box_height_pixels"]):
            continue
        if not (
            float(target["minimum_box_area_fraction"])
            <= area_fraction
            <= float(target["maximum_box_area_fraction"])
        ):
            continue
        border_fraction = float(target["minimum_border_distance_fraction"])
        if x0 < border_fraction * width or y0 < border_fraction * height:
            continue
        if x1 > (1.0 - border_fraction) * width or y1 > (1.0 - border_fraction) * height:
            continue
        print_class = (
            _naf_print_class(
                polygon,
                naf_fields,
                float(config["target_preparation"]["naf_field_overlap_threshold"]),
            )
            if row["source_dataset"] == "NAF"
            else "machine_printed_assumed_from_doclaynet"
        )
        class_rank = {
            "printed_or_stamp": 0,
            "machine_printed_assumed_from_doclaynet": 0,
            "unclassified": 1,
            "handwritten": 2,
        }[print_class]
        candidates.append(
            {
                "_cleartext": text,
                "area_fraction": area_fraction,
                "box_height": box_height,
                "character_pattern": _character_pattern(text),
                "class_rank": class_rank,
                "ocr_score": score,
                "polygon": polygon,
                "print_class": print_class,
                "rank_digest": _hash_text(
                    f"{config['experiment']['seed']}|target|{row['rehearsal_id']}|"
                    f"{_hash_text(text)}|{polygon}"
                ),
            }
        )
    if not candidates:
        raise ValueError(f"no eligible OCR target for {row['rehearsal_id']}")
    candidates.sort(
        key=lambda item: (
            item["class_rank"],
            -item["ocr_score"],
            -item["box_height"],
            item["rank_digest"],
        )
    )
    selected = candidates[0]
    selected["eligible_candidate_count"] = len(candidates)
    return selected


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
    for field, expected in config["frozen_inputs"].items():
        if not field.endswith("_sha256"):
            continue
        path_field = field.removesuffix("_sha256")
        path = _resolve(project_root, str(config["frozen_inputs"][path_field]))
        if _sha256(path) != str(expected):
            raise ValueError(f"frozen input changed: {path}")
    if _sha256(detection_model_dir / "inference.pdiparams") != str(
        config["ocr"]["detection_weight_sha256"]
    ):
        raise ValueError("OCR detection weight changed")
    if _sha256(recognition_model_dir / "inference.pdiparams") != str(
        config["ocr"]["recognition_weight_sha256"]
    ):
        raise ValueError("OCR recognition weight changed")

    toy_path = _resolve(project_root, str(config["frozen_inputs"]["toy3_manifest"]))
    audit_path = _resolve(project_root, str(config["frozen_inputs"]["quality_audit"]))
    toy_rows = _read_jsonl(toy_path)
    audit = {row["record_id"]: row for row in _read_jsonl(audit_path)}
    if len(toy_rows) != 3 or any(row["rehearsal_stage"] != "toy3" for row in toy_rows):
        raise ValueError("expected exactly three toy3 rows")
    if any(row["source_was_selected_for_final"] for row in toy_rows):
        raise ValueError("toy manifest contains a final source")

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
    target = config["target_and_mask"]
    output_rows: list[dict[str, Any]] = []
    artifact_root = _resolve(storage_root, str(config["outputs"]["artifact_root"]))
    for row in toy_rows:
        source_path = _resolve(storage_root, str(row["path"]))
        if _sha256(source_path) != str(row["encoded_sha256"]):
            raise ValueError(f"toy source hash changed: {row['rehearsal_id']}")
        with Image.open(source_path) as handle:
            source = ImageOps.exif_transpose(handle).convert("RGB")
        if source.size != (int(row["width"]), int(row["height"])):
            raise ValueError(f"toy source dimensions changed: {row['rehearsal_id']}")
        result_objects = list(ocr.predict(np.asarray(source)))
        if len(result_objects) != 1:
            raise ValueError(f"unexpected OCR page count: {row['rehearsal_id']}")
        ocr_result = result_objects[0].json["res"]
        naf_fields: list[dict[str, Any]] = []
        if row["source_dataset"] == "NAF":
            audit_row = audit[row["source_record_id"]]
            annotation_path = _resolve(
                storage_root,
                str(audit_row["selection_metadata"]["annotation_relative_path"]),
            )
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            naf_fields = list(annotation.get("fieldBBs", []))
        selected = _select_target(ocr_result, row, config, naf_fields)
        cleartext = str(selected.pop("_cleartext"))
        replacement = _replacement(
            cleartext,
            f"{config['experiment']['seed']}|replacement|{row['rehearsal_id']}",
        )
        polygon = selected["polygon"]
        context_box = _context_box(
            polygon,
            source.width,
            source.height,
            float(target["context_expansion_fraction"]),
            int(target["minimum_context_side_pixels"]),
            int(target["maximum_context_pixels"]),
        )
        context = source.crop(context_box)
        relative_polygon = [
            [point[0] - context_box[0], point[1] - context_box[1]]
            for point in polygon
        ]
        mask = Image.new("L", context.size, color=0)
        draw = ImageDraw.Draw(mask)
        draw.polygon([tuple(point) for point in relative_polygon], fill=255)
        dilation = int(target["mask_dilation_pixels"])
        if dilation:
            mask = mask.filter(ImageFilter.MaxFilter(dilation * 2 + 1))
        if not np.asarray(mask).any():
            raise ValueError(f"empty target mask: {row['rehearsal_id']}")

        artifact_id = _hash_text(
            f"{row['rehearsal_id']}|{row['encoded_sha256']}|{selected['rank_digest']}"
        )
        artifact_dir = artifact_root / artifact_id
        context_path = artifact_dir / "context.png"
        mask_path = artifact_dir / "mask.png"
        _save_png(context_path, context)
        _save_png(mask_path, mask)
        prompt = str(config["prompt"]["template"]).format(
            replacement_text=replacement
        )
        output_rows.append(
            {
                **row,
                "artifact_id": artifact_id,
                "character_pattern": selected["character_pattern"],
                "context_box_xyxy": list(context_box),
                "context_path": context_path.relative_to(storage_root).as_posix(),
                "context_sha256": _sha256(context_path),
                "eligible_ocr_candidate_count": selected["eligible_candidate_count"],
                "mask_path": mask_path.relative_to(storage_root).as_posix(),
                "mask_sha256": _sha256(mask_path),
                "ocr_score": round(float(selected["ocr_score"]), 8),
                "print_class": selected["print_class"],
                "prompt": prompt,
                "prompt_sha256": _hash_text(prompt),
                "replacement_text": replacement,
                "source_text_length": len(cleartext),
                "source_text_sha256": _hash_text(cleartext),
                "target_polygon_source_xy": polygon,
                "target_preparation_status": "ok_cleartext_not_persisted",
            }
        )
        del cleartext

    output_path = _resolve(project_root, str(config["outputs"]["target_manifest"]))
    _write_jsonl(output_path, output_rows)
    summary = {
        "authorization": {
            "final_source_manifest_read": False,
            "final_source_image_read": False,
            "nonfinal_toy_source_images_read": len(output_rows),
            "editor_inference_run": False,
            "detector_inference_run": False,
        },
        "config_sha256": _sha256(config_path),
        "rows": len(output_rows),
        "status_counts": {"ok_cleartext_not_persisted": len(output_rows)},
        "source_text_cleartext_persisted": False,
        "target_manifest": str(output_path.relative_to(project_root)),
        "target_manifest_sha256": _sha256(output_path),
        "artifact_root": str(artifact_root),
        "context_and_mask_artifacts": len(output_rows) * 2,
    }
    summary_path = _resolve(project_root, str(config["outputs"]["summary"]))
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path.relative_to(project_root))
    summary["summary_sha256"] = _sha256(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local OCR targets and masks for non-final editor Toy-3."
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
