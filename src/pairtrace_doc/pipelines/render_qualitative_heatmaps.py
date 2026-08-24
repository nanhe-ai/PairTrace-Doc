from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from pairtrace_doc.pipelines.qualitative_asset_io import (
    _archive_members,
    _mask_for_case,
    _read_selected_tar_members,
    _reference_bytes,
    _resolve,
)
from pairtrace_doc.pipelines.render_qualitative_audit import (
    _font,
    _open_rgb,
    _save_png_atomic,
    _short_case_title,
)
from pairtrace_doc.pipelines.train_student_100 import _sha256, _write_json


COLORMAP_ANCHORS = np.asarray(
    [
        [0, 0, 4],
        [31, 12, 72],
        [85, 15, 109],
        [187, 55, 84],
        [249, 142, 8],
        [252, 255, 164],
    ],
    dtype=np.float64,
)
COLORMAP_POSITIONS = np.asarray([0.0, 0.13, 0.35, 0.55, 0.75, 1.0])
PREDICTION_CONTOUR_RGB = np.asarray([34, 211, 238], dtype=np.uint8)
MASK_CONTOUR_RGB = np.asarray([250, 204, 21], dtype=np.uint8)


def _colormap_lut() -> np.ndarray:
    values = np.linspace(0.0, 1.0, 256)
    channels = [
        np.interp(values, COLORMAP_POSITIONS, COLORMAP_ANCHORS[:, channel])
        for channel in range(3)
    ]
    return np.rint(np.stack(channels, axis=1)).astype(np.uint8)


COLORMAP_LUT = _colormap_lut()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _validate_runtime(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    runtime = config["runtime"]
    rendering = config["rendering"]
    if bool(experiment["paper_evidence"]):
        raise ValueError("heatmap rendering cannot create paper evidence")
    if runtime["device"] != "cpu" or bool(runtime["gpu_launch_authorized"]):
        raise ValueError("heatmap rendering must remain CPU-only")
    if not bool(runtime["selected_frozen_image_read_authorized"]):
        raise PermissionError("selected frozen image reads are not authorized")
    if not bool(runtime["verified_score_cache_read_authorized"]):
        raise PermissionError("verified score-cache reads are not authorized")
    prohibited_runtime = (
        "model_inference_authorized",
        "model_training_authorized",
        "threshold_selection_authorized",
        "metric_computation_authorized",
        "sample_replacement_authorized",
        "human_audit_completion_authorized",
    )
    if any(bool(runtime[name]) for name in prohibited_runtime):
        raise ValueError("heatmap rendering crossed a frozen evidence boundary")
    if list(rendering["global_probability_scale"]) != [0.0, 1.0]:
        raise ValueError("heatmap probability scale changed")
    if rendering["score_dtype"] != "float32":
        raise ValueError("heatmap score dtype changed")
    prohibited_rendering = (
        "per_image_normalization_allowed",
        "smoothing_allowed",
        "morphology_allowed",
        "aggregate_seed_map_allowed",
    )
    if any(bool(rendering[name]) for name in prohibited_rendering):
        raise ValueError("heatmap rendering rule crossed the frozen protocol")
    if rendering["threshold_contours"] != "frozen_per_record_threshold_only":
        raise ValueError("heatmap threshold-contour rule changed")
    if rendering["ground_truth_contours"] != "frozen_mask_only":
        raise ValueError("heatmap mask-contour rule changed")
    if rendering["colormap"] != "pairtrace_magma_v1":
        raise ValueError("heatmap colormap changed")
    if rendering["score_resize_interpolation"] != "nearest":
        raise ValueError("heatmap score interpolation changed")
    if rendering["mask_resize_interpolation"] != "nearest":
        raise ValueError("heatmap mask interpolation changed")


def _verify_file(project_root: Path, relative: str, expected: str, label: str) -> Path:
    path = _resolve(project_root, relative)
    digest = _sha256(path)
    if digest != expected:
        raise ValueError(f"{label} SHA-256 changed: {digest} != {expected}")
    return path


def _binary_boundary(value: np.ndarray) -> np.ndarray:
    binary = np.asarray(value, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("contour input must be two-dimensional")
    boundary = np.zeros(binary.shape, dtype=bool)
    vertical = binary[1:, :] != binary[:-1, :]
    horizontal = binary[:, 1:] != binary[:, :-1]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[0, :] |= binary[0, :]
    boundary[-1, :] |= binary[-1, :]
    boundary[:, 0] |= binary[:, 0]
    boundary[:, -1] |= binary[:, -1]
    return boundary


def _colorize_score(
    score: np.ndarray, threshold: float, mask: np.ndarray
) -> Image.Image:
    value = np.asarray(score)
    if value.dtype != np.float32:
        raise ValueError(f"score dtype is {value.dtype}, not float32")
    if value.ndim != 2 or mask.shape != value.shape:
        raise ValueError("score/mask geometry changed during rendering")
    if not np.isfinite(value).all():
        raise ValueError("score contains non-finite values")
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError("score is outside the frozen [0, 1] scale")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("frozen threshold is outside [0, 1]")
    indices = np.rint(value * 255.0).astype(np.uint8)
    rendered = COLORMAP_LUT[indices].copy()
    rendered[_binary_boundary(mask > 0)] = MASK_CONTOUR_RGB
    prediction_boundary = _binary_boundary(value >= threshold)
    rendered[prediction_boundary] = PREDICTION_CONTOUR_RGB
    return Image.fromarray(rendered)


def _load_score(path: Path, record: dict[str, Any]) -> np.ndarray:
    digest = _sha256(path)
    expected_digest = str(record["replay_score_sha256"])
    if digest != expected_digest:
        raise ValueError(
            f"replay score SHA-256 changed for {record['source_record_id']}"
        )
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["scores"]:
            raise ValueError(f"unexpected score-cache fields: {archive.files}")
        score = archive["scores"]
    expected_shape = tuple(int(value) for value in record["score_shape"])
    if score.dtype != np.float32:
        raise ValueError(f"replay score dtype changed: {score.dtype}")
    if score.shape != expected_shape:
        raise ValueError(
            f"replay score shape changed: {score.shape} != {expected_shape}"
        )
    if not np.isfinite(score).all():
        raise ValueError("replay score contains non-finite values")
    if float(score.min()) < 0.0 or float(score.max()) > 1.0:
        raise ValueError("replay score is outside [0, 1]")
    return score


def _model_label(record: dict[str, Any]) -> str:
    prefix = str(record["source_record_id"]).split(":", 1)[0]
    if prefix.startswith("robust_202607"):
        return f"seed {prefix.removeprefix('robust_')[:8]}"
    if prefix == "baseline_affine_ecc":
        return "clean teacher"
    if prefix == "robust_teacher_correct":
        return "robust fixed model"
    return prefix.replace("_", " ")


def _fit_tile(
    image: Image.Image, size: tuple[int, int], interpolation: Image.Resampling
) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, interpolation)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _mask_rgb(mask: Image.Image) -> Image.Image:
    mask_array = np.asarray(mask, dtype=np.uint8)
    rendered = np.full((*mask_array.shape, 3), [17, 24, 39], dtype=np.uint8)
    rendered[mask_array > 0] = MASK_CONTOUR_RGB
    return Image.fromarray(rendered)


def _mask_at_score_shape(mask: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    expected_size = (int(shape[1]), int(shape[0]))
    resized = (
        mask
        if mask.size == expected_size
        else mask.resize(expected_size, Image.Resampling.NEAREST)
    )
    value = np.asarray(resized, dtype=np.uint8)
    if value.shape != shape:
        raise ValueError("mask resize did not produce the frozen score geometry")
    return np.where(value > 0, 255, 0).astype(np.uint8)


def _draw_colorbar(
    page: Image.Image, left: int, top: int, width: int, height: int
) -> None:
    indices = np.linspace(0, 255, width).round().astype(np.uint8)
    bar = np.repeat(COLORMAP_LUT[indices][None, :, :], height, axis=0)
    page.paste(Image.fromarray(bar), (left, top))
    draw = ImageDraw.Draw(page)
    draw.rectangle((left, top, left + width - 1, top + height - 1), outline="#64748b", width=2)
    draw.text((left, top + height + 4), "0", font=_font(24), fill="#334155")
    draw.text(
        (left + width, top + height + 4),
        "1",
        anchor="ra",
        font=_font(24),
        fill="#334155",
    )


def _render_case_page(
    case: dict[str, Any],
    candidate: Image.Image,
    mask: Image.Image,
    rendered_records: list[tuple[dict[str, Any], Image.Image]],
    layout: dict[str, Any],
) -> Image.Image:
    page_width = int(layout["page_width"])
    page_height = int(layout["page_height"])
    margin = int(layout["margin"])
    gap = int(layout["gap"])
    title_height = int(layout["title_height"])
    footer_height = int(layout["footer_height"])
    columns = int(layout["grid_columns"])
    label_height = int(layout["tile_label_height"])
    items: list[tuple[str, Image.Image, Image.Resampling]] = [
        ("Candidate", candidate, Image.Resampling.LANCZOS),
        (
            "Weak box mask" if case["mask_semantics"] == "box_mask_not_pixel_accurate" else "Exact binary mask",
            _mask_rgb(mask),
            Image.Resampling.NEAREST,
        ),
    ]
    for record, rendered in rendered_records:
        group = str(record["display_group"]).replace("_", " ")
        label = (
            f"{group}\n{_model_label(record)} · "
            f"frozen t={float(record['fixed_pixel_threshold']):.2f}"
        )
        items.append((label, rendered, Image.Resampling.NEAREST))
    rows = (len(items) + columns - 1) // columns
    usable_width = page_width - 2 * margin - (columns - 1) * gap
    usable_height = page_height - 2 * margin - title_height - footer_height
    tile_width = usable_width // columns
    tile_total_height = (usable_height - (rows - 1) * gap) // rows
    image_height = tile_total_height - label_height
    if min(tile_width, image_height) < 120:
        raise ValueError("qualitative heatmap layout is too small")

    page = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(page)
    draw.text(
        (margin, margin),
        _short_case_title(case),
        font=_font(52, bold=True),
        fill="#111827",
    )
    evidence = "paper-evidence case" if case.get("paper_evidence") else "limitation/development case"
    draw.text(
        (margin, margin + 68),
        f"{case['case_id']}  |  {case['cohort']}  |  {evidence}",
        font=_font(26),
        fill="#475569",
    )
    draw.text(
        (margin, margin + 112),
        (
            f"mask: {case['mask_semantics']}  |  all frozen maps shown  |  "
            "no per-image normalization"
        ),
        font=_font(26),
        fill="#475569",
    )

    grid_top = margin + title_height
    for index, (label, image, interpolation) in enumerate(items):
        row = index // columns
        column = index % columns
        left = margin + column * (tile_width + gap)
        top = grid_top + row * (tile_total_height + gap)
        fitted = _fit_tile(image, (tile_width, image_height), interpolation)
        page.paste(fitted, (left, top))
        draw.rectangle(
            (left, top, left + tile_width - 1, top + image_height - 1),
            outline="#cbd5e1",
            width=3,
        )
        draw.multiline_text(
            (left + tile_width // 2, top + image_height + 12),
            label,
            anchor="ma",
            align="center",
            spacing=5,
            font=_font(24, bold=True),
            fill="#1f2937",
        )

    footer_top = page_height - margin - footer_height + 8
    _draw_colorbar(page, margin, footer_top, 620, 34)
    draw.text(
        (margin + 660, footer_top + 4),
        "fixed probability scale [0,1]",
        font=_font(26, bold=True),
        fill="#334155",
    )
    draw.text(
        (margin + 1250, footer_top + 4),
        "cyan: frozen-threshold contour",
        font=_font(26, bold=True),
        fill=tuple(int(value) for value in PREDICTION_CONTOUR_RGB),
    )
    draw.text(
        (margin + 2050, footer_top + 4),
        "yellow: frozen mask contour",
        font=_font(26, bold=True),
        fill="#ca8a04",
    )
    return page


def _save_pdf_atomic(pages: list[Image.Image], path: Path, resolution: int) -> None:
    if not pages:
        raise ValueError("cannot save an empty heatmap packet")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pages[0].save(
        temporary,
        format="PDF",
        resolution=resolution,
        save_all=True,
        append_images=pages[1:],
        title="PairTrace-Doc frozen qualitative heatmaps",
        author="PairTrace-Doc",
        subject="Fixed [0,1] score maps; pending independent human review",
    )
    temporary.replace(path)


def _preview(pages: list[Image.Image], columns: int) -> Image.Image:
    if columns < 1:
        raise ValueError("preview column count must be positive")
    preview_width = 1800
    tile_width = preview_width // columns
    tile_height = round(tile_width * pages[0].height / pages[0].width)
    rows = (len(pages) + columns - 1) // columns
    canvas = Image.new("RGB", (preview_width, rows * tile_height), "#e5e7eb")
    for index, page in enumerate(pages):
        tile = page.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        canvas.paste(
            tile,
            ((index % columns) * tile_width, (index // columns) * tile_height),
        )
    return canvas


def _heatmap_ready_worksheet(
    blank: dict[str, Any], blank_sha256: str, pdf_path: str
) -> dict[str, Any]:
    if bool(blank.get("human_review_complete")):
        raise ValueError("blank human-review worksheet is already marked complete")
    reviews = blank.get("reviews")
    if not isinstance(reviews, list) or any(
        any(value is not None for key, value in review.items() if key != "case_id")
        for review in reviews
    ):
        raise ValueError("human-review worksheet is not blank")
    output = json.loads(json.dumps(blank))
    output.update(
        {
            "schema_version": 2,
            "status": "pending_independent_human_review",
            "model_heatmaps_available": True,
            "human_review_complete": False,
            "instruction": (
                "An independent human reviewer must complete every fixed field after "
                "viewing the frozen input/reference/mask packet and frozen heatmap packet; "
                "do not replace a case."
            ),
            "source_blank_worksheet_sha256": blank_sha256,
            "heatmap_packet": pdf_path,
        }
    )
    return output


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("heatmap render config must be a mapping")
    _validate_runtime(config)
    specification = config["input"]
    experiment = config["experiment"]
    protocol_path = _verify_file(
        project_root,
        str(experiment["protocol"]),
        str(experiment["expected_protocol_sha256"]),
        "qualitative heatmap protocol",
    )
    case_manifest_path = _verify_file(
        project_root,
        str(specification["case_manifest"]),
        str(specification["expected_case_manifest_sha256"]),
        "qualitative case manifest",
    )
    input_render_manifest_path = _verify_file(
        project_root,
        str(specification["input_render_manifest"]),
        str(specification["expected_input_render_manifest_sha256"]),
        "qualitative input render manifest",
    )
    preflight_path = _verify_file(
        project_root,
        str(specification["replay_preflight_manifest"]),
        str(specification["expected_replay_preflight_manifest_sha256"]),
        "qualitative replay preflight",
    )
    execution_path = _verify_file(
        project_root,
        str(specification["replay_execution_manifest"]),
        str(specification["expected_replay_execution_manifest_sha256"]),
        "qualitative replay execution manifest",
    )
    predictions_path = _verify_file(
        project_root,
        str(specification["replay_predictions"]),
        str(specification["expected_replay_predictions_sha256"]),
        "qualitative replay predictions",
    )
    blank_worksheet_path = _verify_file(
        project_root,
        str(specification["blank_human_review_worksheet"]),
        str(specification["expected_blank_human_review_worksheet_sha256"]),
        "blank human-review worksheet",
    )
    cases = _read_json(case_manifest_path)["cases"]
    expected_case_ids = [str(value) for value in specification["expected_case_ids"]]
    if len(cases) != int(specification["expected_case_count"]):
        raise ValueError("qualitative heatmap case count changed")
    if [str(case["case_id"]) for case in cases] != expected_case_ids:
        raise ValueError("qualitative heatmap case order or membership changed")
    preflight = _read_json(preflight_path)
    execution = _read_json(execution_path)
    records = _read_jsonl(predictions_path)
    expected_record_count = int(specification["expected_record_count"])
    if preflight.get("record_count") != expected_record_count:
        raise ValueError("qualitative replay preflight record count changed")
    if execution.get("status") != "qualitative_heatmap_replay_complete":
        raise ValueError("qualitative replay execution is incomplete")
    if execution.get("config_sha256") != specification["expected_replay_execution_config_sha256"]:
        raise ValueError("qualitative replay execution config changed")
    if execution.get("predictions_sha256") != specification["expected_replay_predictions_sha256"]:
        raise ValueError("qualitative replay prediction link changed")
    if execution.get("failed_records") != 0 or len(records) != expected_record_count:
        raise ValueError("qualitative replay output is incomplete")
    plan_records = preflight["records"]
    for plan_record, record in zip(plan_records, records, strict=True):
        if record.get("status") != "ok":
            raise ValueError("qualitative replay contains a failed record")
        frozen_identity = ("case_id", "source_record_id", "replay_key", "replay_score_cache")
        if any(plan_record[field] != record[field] for field in frozen_identity):
            raise ValueError("qualitative replay record order or identity changed")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            str(paths["scratch_env"]),
            str(_resolve(project_root, str(paths["scratch_default"]))),
        )
    ).resolve()
    archive_path = _resolve(scratch, str(specification["fantasyid_archive"]))
    if archive_path.stat().st_size != int(specification["expected_fantasyid_archive_bytes"]):
        raise ValueError("FantasyID archive size changed")
    archive_sha256 = _sha256(archive_path)
    if archive_sha256 != str(specification["expected_fantasyid_archive_sha256"]):
        raise ValueError("FantasyID archive SHA-256 changed")
    requested_members = _archive_members(cases)
    archive_payloads = _read_selected_tar_members(archive_path, requested_members)

    records_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_case[str(record["case_id"])].append(record)
    record_dir = _resolve(project_root, str(paths["record_heatmaps_dir"]))
    case_dir = _resolve(project_root, str(paths["case_panels_dir"]))
    record_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Image.Image] = []
    record_outputs: list[dict[str, Any]] = []
    case_outputs: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        candidate = _open_rgb(
            _reference_bytes(case["candidate"], scratch, archive_payloads),
            f"{case['case_id']}:candidate",
        )
        mask = _mask_for_case(case, scratch, archive_payloads, candidate.size)
        rendered_records: list[tuple[dict[str, Any], Image.Image]] = []
        for record_index, record in enumerate(records_by_case[str(case["case_id"])], start=1):
            if record["input_sha256"] != case["candidate"]["sha256"]:
                raise ValueError("heatmap candidate identity changed")
            if record["mask_sha256"] != case["mask"]["sha256"]:
                raise ValueError("heatmap mask identity changed")
            score_path = _resolve(scratch, str(record["replay_score_cache"]))
            score = _load_score(score_path, record)
            mask_array = _mask_at_score_shape(mask, score.shape)
            rendered = _colorize_score(
                score, float(record["fixed_pixel_threshold"]), mask_array
            )
            heatmap_path = record_dir / (
                f"{case_index:02d}_{record_index:02d}_{record['replay_key'][:12]}.png"
            )
            _save_png_atomic(rendered, heatmap_path)
            rendered_records.append((record, rendered))
            record_outputs.append(
                {
                    "case_id": case["case_id"],
                    "source_record_id": record["source_record_id"],
                    "display_group": record["display_group"],
                    "fixed_pixel_threshold": record["fixed_pixel_threshold"],
                    "replay_key": record["replay_key"],
                    "replay_score_cache": record["replay_score_cache"],
                    "replay_score_sha256": record["replay_score_sha256"],
                    "score_shape": record["score_shape"],
                    "score_dtype": record["score_dtype"],
                    "heatmap": str(heatmap_path.relative_to(project_root)),
                    "heatmap_sha256": _sha256(heatmap_path),
                }
            )
        page = _render_case_page(
            case, candidate, mask, rendered_records, config["layout"]
        )
        case_path = case_dir / f"{case_index:02d}_{case['case_id']}.png"
        _save_png_atomic(page, case_path)
        pages.append(page)
        case_outputs.append(
            {
                "case_id": case["case_id"],
                "record_count": len(rendered_records),
                "panel": str(case_path.relative_to(project_root)),
                "panel_sha256": _sha256(case_path),
                "width": page.width,
                "height": page.height,
                "mask_semantics": case["mask_semantics"],
            }
        )

    if len(record_outputs) != expected_record_count:
        raise ValueError("not all frozen heatmaps were rendered")
    pdf_path = _resolve(project_root, str(paths["pdf"]))
    preview_path = _resolve(project_root, str(paths["preview_png"]))
    _save_pdf_atomic(pages, pdf_path, int(config["layout"]["pdf_resolution_dpi"]))
    _save_png_atomic(
        _preview(pages, int(config["layout"]["preview_columns"])), preview_path
    )
    worksheet_path = _resolve(project_root, str(paths["human_review_worksheet"]))
    worksheet = _heatmap_ready_worksheet(
        _read_json(blank_worksheet_path),
        str(specification["expected_blank_human_review_worksheet_sha256"]),
        str(pdf_path.relative_to(project_root)),
    )
    _write_json(worksheet_path, worksheet)

    result = {
        "schema_version": 1,
        "experiment": experiment,
        "status": "qualitative_heatmap_packet_rendered_pending_human_review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_evidence": False,
        "model_inference_performed": False,
        "score_cache_records_verified": len(record_outputs),
        "new_scientific_metrics_computed": False,
        "threshold_selection_used": False,
        "sample_replacement_used": False,
        "per_image_normalization_used": False,
        "smoothing_used": False,
        "morphology_used": False,
        "seed_aggregation_used": False,
        "all_frozen_score_maps_visible": len(record_outputs) == expected_record_count,
        "global_probability_scale": [0.0, 1.0],
        "human_review_complete": False,
        "human_review_status": "pending_independent_human_review",
        "input": {
            "config": str(config_path.relative_to(project_root)),
            "config_sha256": _sha256(config_path),
            "implementation": str(Path(__file__).resolve().relative_to(project_root)),
            "implementation_sha256": _sha256(Path(__file__).resolve()),
            "protocol": str(protocol_path.relative_to(project_root)),
            "protocol_sha256": _sha256(protocol_path),
            "case_manifest": str(case_manifest_path.relative_to(project_root)),
            "case_manifest_sha256": _sha256(case_manifest_path),
            "input_render_manifest": str(input_render_manifest_path.relative_to(project_root)),
            "input_render_manifest_sha256": _sha256(input_render_manifest_path),
            "replay_preflight_manifest": str(preflight_path.relative_to(project_root)),
            "replay_preflight_manifest_sha256": _sha256(preflight_path),
            "replay_execution_manifest": str(execution_path.relative_to(project_root)),
            "replay_execution_manifest_sha256": _sha256(execution_path),
            "replay_predictions": str(predictions_path.relative_to(project_root)),
            "replay_predictions_sha256": _sha256(predictions_path),
            "fantasyid_archive_sha256": archive_sha256,
            "fantasyid_archive_members_read": sorted(requested_members),
            "full_archive_extracted": False,
        },
        "rendering": {
            "colormap": "pairtrace_magma_v1",
            "prediction_contour_rgb": PREDICTION_CONTOUR_RGB.tolist(),
            "mask_contour_rgb": MASK_CONTOUR_RGB.tolist(),
            "score_resize_interpolation": "nearest",
            "mask_resize_interpolation": "nearest",
        },
        "output": {
            "pdf": str(pdf_path.relative_to(project_root)),
            "pdf_sha256": _sha256(pdf_path),
            "preview_png": str(preview_path.relative_to(project_root)),
            "preview_png_sha256": _sha256(preview_path),
            "human_review_worksheet": str(worksheet_path.relative_to(project_root)),
            "human_review_worksheet_sha256": _sha256(worksheet_path),
            "case_panels": case_outputs,
            "record_heatmaps": record_outputs,
        },
        "case_count": len(cases),
        "record_count": len(record_outputs),
    }
    manifest_path = _resolve(project_root, str(paths["render_manifest"]))
    _write_json(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
