from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from pairtrace_doc.pipelines.render_qualitative_audit import (
    _font,
    _save_png_atomic,
)
from pairtrace_doc.pipelines.render_qualitative_heatmaps import (
    MASK_CONTOUR_RGB,
    PREDICTION_CONTOUR_RGB,
    _colorize_score,
    _draw_colorbar,
    _fit_tile,
    _mask_at_score_shape,
    _mask_rgb,
    _preview,
    _save_pdf_atomic,
)
from pairtrace_doc.pipelines.train_student_100 import _resolve, _sha256, _write_json


FIXED_ZIP_TIMESTAMP = (2026, 7, 22, 0, 0, 0)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify(path: Path, expected: str, label: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 changed")


def _open_verified_image(path: Path, expected: str, mode: str) -> Image.Image:
    _verify(path, expected, str(path))
    with Image.open(path) as handle:
        return handle.convert(mode)


def _load_score(path: Path, record: dict[str, Any]) -> np.ndarray:
    _verify(path, str(record["score_cache_sha256"]), str(record["condition"]))
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["scores"]:
            raise ValueError("TFR audit score-cache schema changed")
        score = np.asarray(archive["scores"])
    if (
        score.dtype != np.float32
        or list(score.shape) != list(record["score_shape"])
        or not np.isfinite(score).all()
        or float(score.min()) < 0.0
        or float(score.max()) > 1.0
    ):
        raise ValueError("TFR audit score cache is invalid")
    return score


def _render_page(
    case: dict[str, Any],
    geometry: str,
    candidate: Image.Image,
    authentic: Image.Image,
    mask: Image.Image,
    heatmaps: list[tuple[dict[str, Any], Image.Image]],
    layout: dict[str, Any],
) -> Image.Image:
    width = int(layout["page_width"])
    height = int(layout["page_height"])
    margin = int(layout["margin"])
    gap = int(layout["gap"])
    title_height = int(layout["title_height"])
    footer_height = int(layout["footer_height"])
    columns = int(layout["grid_columns"])
    label_height = int(layout["tile_label_height"])
    items: list[tuple[str, Image.Image, Image.Resampling]] = [
        ("Candidate", candidate, Image.Resampling.LANCZOS),
        ("Authentic reference", authentic, Image.Resampling.LANCZOS),
        ("Exact mask", _mask_rgb(mask), Image.Resampling.NEAREST),
    ]
    for record, rendered in heatmaps:
        scorer = str(record["scorer"]).replace("robust_", "seed ").replace("_", " ")
        items.append(
            (
                f"{scorer}\nAP {float(record['pixel_ap']):.3f} · t={float(record['pixel_threshold']):.2f}",
                rendered,
                Image.Resampling.NEAREST,
            )
        )
    rows = (len(items) + columns - 1) // columns
    tile_width = (width - 2 * margin - (columns - 1) * gap) // columns
    available_height = height - 2 * margin - title_height - footer_height
    tile_total_height = (available_height - (rows - 1) * gap) // rows
    image_height = tile_total_height - label_height
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    draw.text(
        (margin, margin),
        f"{case['case_id']} · {case['cohort']} · {geometry} reference after ECC",
        font=_font(40, bold=True),
        fill="#111827",
    )
    draw.text(
        (margin, margin + 58),
        (
            f"group {case['source_group_id']} · group clean AP "
            f"{float(case['group_three_seed_mean_clean_ap']):.3f} · "
            "frozen before rendering · pending independent review"
        ),
        font=_font(23),
        fill="#475569",
    )
    draw.text(
        (margin, margin + 98),
        "fixed [0,1] score scale · no per-image normalization · no smoothing · no sample replacement",
        font=_font(21),
        fill="#475569",
    )
    grid_top = margin + title_height
    for index, (label, image, interpolation) in enumerate(items):
        row_index, column_index = divmod(index, columns)
        left = margin + column_index * (tile_width + gap)
        top = grid_top + row_index * (tile_total_height + gap)
        fitted = _fit_tile(image, (tile_width, image_height), interpolation)
        page.paste(fitted, (left, top))
        draw.rectangle(
            (left, top, left + tile_width - 1, top + image_height - 1),
            outline="#cbd5e1",
            width=3,
        )
        draw.multiline_text(
            (left + tile_width // 2, top + image_height + 8),
            label,
            anchor="ma",
            align="center",
            spacing=3,
            font=_font(18, bold=True),
            fill="#1f2937",
        )
    footer_top = height - margin - footer_height + 8
    _draw_colorbar(page, margin, footer_top, 500, 28)
    draw.text(
        (margin + 540, footer_top + 2),
        "cyan: frozen threshold",
        font=_font(20, bold=True),
        fill=tuple(int(value) for value in PREDICTION_CONTOUR_RGB),
    )
    draw.text(
        (margin + 950, footer_top + 2),
        "yellow: exact mask",
        font=_font(20, bold=True),
        fill=tuple(int(value) for value in MASK_CONTOUR_RGB),
    )
    return page


def _reviewer_readme() -> str:
    return """# TFR zero-shot independent qualitative review

Review all seven frozen cases in the PDF. Do not replace, omit, or add cases.
For each case, inspect both the clean and affine-ECC page, then fill every null
field in the JSON worksheet using only the listed categorical values. Record a
short factual failure mode and note. Change `human_review_complete` and
`status` only after every case is reviewed by an independent person.

This packet is viewed-development evidence. It cannot tune a threshold, change
a quantitative result, authorize the TFR holdout, or create paper evidence.
"""


def _zip_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if runtime["device"] != "cpu" or not runtime["selected_frozen_image_read_authorized"]:
        raise ValueError("TFR qualitative render requires authorized CPU image reads")
    if not runtime["verified_score_cache_read_authorized"]:
        raise ValueError("TFR qualitative render requires verified cache reads")
    prohibited = (
        "model_inference_authorized",
        "model_training_authorized",
        "metric_computation_authorized",
        "threshold_selection_authorized",
        "sample_replacement_authorized",
        "human_audit_completion_authorized",
        "tfr_holdout_read_allowed",
    )
    if any(runtime[name] for name in prohibited):
        raise ValueError("TFR qualitative rendering crossed its evidence boundary")
    rendering = config["rendering"]
    if (
        rendering["global_probability_scale"] != [0.0, 1.0]
        or rendering["per_image_normalization_allowed"]
        or rendering["smoothing_allowed"]
        or rendering["morphology_allowed"]
        or rendering["score_resize_interpolation"] != "nearest"
        or rendering["mask_resize_interpolation"] != "nearest"
        or rendering["colormap"] != "pairtrace_magma_v1"
    ):
        raise ValueError("TFR qualitative rendering rules changed")

    experiment = config["experiment"]
    protocol = _resolve(project_root, experiment["protocol"])
    _verify(protocol, experiment["expected_protocol_sha256"], "protocol")
    input_config = config["input"]
    case_manifest_path = _resolve(project_root, input_config["case_manifest"])
    blank_worksheet_path = _resolve(project_root, input_config["blank_worksheet"])
    _verify(
        case_manifest_path,
        input_config["expected_case_manifest_sha256"],
        "case manifest",
    )
    _verify(
        blank_worksheet_path,
        input_config["expected_blank_worksheet_sha256"],
        "blank worksheet",
    )
    case_manifest = _read_json(case_manifest_path)
    cases = case_manifest["cases"]
    if len(cases) != int(input_config["expected_case_count"]):
        raise ValueError("TFR qualitative case count changed")
    worksheet = _read_json(blank_worksheet_path)
    if worksheet["human_review_complete"] or any(
        any(value is not None for key, value in row.items() if key != "case_id")
        for row in worksheet["reviews"]
    ):
        raise ValueError("TFR qualitative worksheet is not blank")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    pages_dir = _resolve(project_root, paths["pages_dir"])
    pages_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Image.Image] = []
    page_records: list[dict[str, Any]] = []
    verified_maps = 0
    for case_index, case in enumerate(cases, start=1):
        candidate = _open_verified_image(
            _resolve(scratch, case["candidate"]), case["candidate_sha256"], "RGB"
        )
        authentic = _open_verified_image(
            _resolve(scratch, case["authentic"]), case["authentic_sha256"], "RGB"
        )
        mask = _open_verified_image(
            _resolve(scratch, case["mask"]), case["mask_sha256"], "L"
        )
        for geometry in ("clean", "affine"):
            rendered: list[tuple[dict[str, Any], Image.Image]] = []
            for record in case["maps"]:
                if record["geometry"] != geometry:
                    continue
                score = _load_score(_resolve(scratch, record["score_cache"]), record)
                mask_array = _mask_at_score_shape(mask, score.shape)
                heatmap = _colorize_score(
                    score, float(record["pixel_threshold"]), mask_array
                )
                rendered.append((record, heatmap))
                verified_maps += 1
            if len(rendered) != 4:
                raise ValueError("TFR audit page map count changed")
            page = _render_page(
                case,
                geometry,
                candidate,
                authentic,
                mask,
                rendered,
                config["layout"],
            )
            page_path = pages_dir / f"{case_index:02d}_{case['case_id']}_{geometry}.png"
            _save_png_atomic(page, page_path)
            pages.append(page)
            page_records.append(
                {
                    "case_id": case["case_id"],
                    "geometry": geometry,
                    "page": str(page_path.relative_to(project_root)),
                    "page_sha256": _sha256(page_path),
                }
            )
    if verified_maps != len(cases) * 8:
        raise ValueError("TFR audit did not verify every frozen map")

    pdf_path = _resolve(project_root, paths["pdf"])
    preview_path = _resolve(project_root, paths["preview_png"])
    _save_pdf_atomic(pages, pdf_path, int(config["layout"]["pdf_resolution_dpi"]))
    _save_png_atomic(
        _preview(pages, int(config["layout"]["preview_columns"])), preview_path
    )
    ready_worksheet = json.loads(json.dumps(worksheet))
    ready_worksheet.update(
        {
            "status": "pending_independent_human_review",
            "human_review_complete": False,
            "model_heatmaps_available": True,
            "review_packet_pdf": str(pdf_path.relative_to(project_root)),
            "review_packet_pdf_sha256": _sha256(pdf_path),
        }
    )
    ready_worksheet_path = _resolve(project_root, paths["ready_worksheet"])
    _write_json(ready_worksheet_path, ready_worksheet)

    integrity = {
        "paper_evidence": False,
        "human_review_complete": False,
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "protocol_sha256": _sha256(protocol),
        "case_manifest_sha256": _sha256(case_manifest_path),
        "pdf_sha256": _sha256(pdf_path),
        "worksheet_sha256": _sha256(ready_worksheet_path),
    }
    members = {
        "README_FOR_REVIEWER.md": _reviewer_readme().encode("utf-8"),
        "tfr_zero_shot_qualitative_audit_protocol.md": protocol.read_bytes(),
        "case_manifest.json": case_manifest_path.read_bytes(),
        "human_review_packet.pdf": pdf_path.read_bytes(),
        "human_review_worksheet.json": ready_worksheet_path.read_bytes(),
        "integrity.json": (
            json.dumps(integrity, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    zip_path = _resolve(project_root, paths["review_packet_zip"])
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        for name, payload in sorted(members.items()):
            _zip_member(archive, name, payload)
    temporary.replace(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("TFR audit ZIP integrity check failed")

    result = {
        "status": "tfr_qualitative_packet_ready_pending_independent_human_review",
        "experiment": experiment,
        "paper_evidence": False,
        "human_review_complete": False,
        "tfr_holdout_read": False,
        "model_inference_performed": False,
        "threshold_selection_performed": False,
        "sample_replacement_used": False,
        "case_count": len(cases),
        "rendered_pages": len(pages),
        "verified_score_maps": verified_maps,
        "outputs": {
            "pdf": str(pdf_path.relative_to(project_root)),
            "pdf_sha256": _sha256(pdf_path),
            "preview_png": str(preview_path.relative_to(project_root)),
            "preview_png_sha256": _sha256(preview_path),
            "worksheet": str(ready_worksheet_path.relative_to(project_root)),
            "worksheet_sha256": _sha256(ready_worksheet_path),
            "review_packet_zip": str(zip_path.relative_to(project_root)),
            "review_packet_zip_sha256": _sha256(zip_path),
            "pages": page_records,
        },
        "member_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(members.items())
        },
    }
    render_manifest_path = _resolve(project_root, paths["render_manifest"])
    _write_json(render_manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
