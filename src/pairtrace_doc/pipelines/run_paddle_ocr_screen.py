from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_text(values: list[str]) -> str:
    text = " ".join(values)
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    current: list[str] = []
    for character in text:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return " ".join(tokens)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or "record_id" not in value or "image" not in value:
                raise ValueError(f"invalid OCR input at line {line_number}")
            rows.append(value)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Imports stay inside the worker so the main reproducible environment does
    # not need PaddleOCR and can unit-test materialization without model setup.
    from paddleocr import PaddleOCR

    detector = Path(args.detector_dir).resolve()
    recognizer = Path(args.recognizer_dir).resolve()
    if _sha256(detector / "inference.pdiparams") != args.detector_sha256:
        raise ValueError("OCR detector weights changed")
    if _sha256(recognizer / "inference.pdiparams") != args.recognizer_sha256:
        raise ValueError("OCR recognizer weights changed")
    rows = _load_rows(Path(args.input).resolve())
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate OCR input record IDs")
    os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
    pipeline = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir=str(detector),
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        text_recognition_model_dir=str(recognizer),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=float(args.recognition_threshold),
    )
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        record_id = str(row["record_id"])
        image = Path(str(row["image"])).resolve()
        try:
            results = list(pipeline.predict(str(image)))
            if len(results) != 1:
                raise ValueError(f"expected one OCR result, got {len(results)}")
            result = results[0]
            texts = [str(value) for value in result.get("rec_texts", [])]
            scores = [float(value) for value in result.get("rec_scores", [])]
            boxes_value = result.get("rec_boxes", [])
            boxes = [[int(coordinate) for coordinate in box] for box in boxes_value]
            if not (len(texts) == len(scores) == len(boxes)):
                raise ValueError("OCR text/score/box lengths disagree")
            normalized = _normalise_text(texts)
            output_rows.append(
                {
                    "record_id": record_id,
                    "image_sha256": str(row["image_sha256"]),
                    "status": "ok",
                    "normalized_text": normalized,
                    "normalized_text_sha256": hashlib.sha256(
                        normalized.encode("utf-8")
                    ).hexdigest(),
                    "recognized_characters": len(normalized.replace(" ", "")),
                    "boxes": boxes,
                    "scores": scores,
                    "error": None,
                }
            )
        except Exception as exc:  # item-level failure is deliberately recorded
            output_rows.append(
                {
                    "record_id": record_id,
                    "image_sha256": str(row["image_sha256"]),
                    "status": "error",
                    "normalized_text": "",
                    "normalized_text_sha256": hashlib.sha256(b"").hexdigest(),
                    "recognized_characters": 0,
                    "boxes": [],
                    "scores": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if args.progress_every and index % int(args.progress_every) == 0:
            print(json.dumps({"ocr_completed": index, "ocr_total": len(rows)}), flush=True)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)
    summary = {
        "status": "ocr_screen_complete",
        "records": len(output_rows),
        "ok": sum(row["status"] == "ok" for row in output_rows),
        "errors": sum(row["status"] != "ok" for row in output_rows),
        "output_sha256": _sha256(output),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector-dir", required=True)
    parser.add_argument("--recognizer-dir", required=True)
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--recognizer-sha256", required=True)
    parser.add_argument("--recognition-threshold", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
