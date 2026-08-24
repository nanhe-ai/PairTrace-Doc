from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from pairtrace_doc.pipelines.materialize_fantasyid_facelondon import (
    RASTERIZER_VERSION,
    _png_bytes,
    _rasterize_box_mask,
)
from pairtrace_doc.pipelines.train_student_100 import _resolve, _sha256, _write_json


YES_NO_UNCERTAIN = ["yes", "no", "uncertain"]
REGISTRATION_ARTIFACT = ["none", "local", "dense"]
PREDICTION_COVERAGE = [
    "miss",
    "partial",
    "matched",
    "over-segmented",
    "dense-false-positive",
]
DOMINANT_FAILURE = [
    "none",
    "registration",
    "acquisition_shift",
    "reference_identity",
    "field_of_view",
    "weak_annotation",
    "other",
]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_runtime(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    if runtime["device"] != "cpu" or runtime["gpu_launch_authorized"]:
        raise ValueError("qualitative input rendering must remain CPU-only")
    if not runtime["selected_frozen_image_read_authorized"]:
        raise ValueError("selected frozen image reads were not authorized")
    prohibited = (
        "model_inference_authorized",
        "score_cache_read_authorized",
        "threshold_selection_authorized",
        "sample_replacement_authorized",
        "full_archive_extraction_authorized",
    )
    if any(bool(runtime[name]) for name in prohibited):
        raise ValueError("qualitative rendering crossed a frozen evidence boundary")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("rendering cannot create new paper evidence")


def _read_selected_tar_members(
    archive_path: Path, requested: set[str]
) -> dict[str, bytes]:
    if any(
        PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        for name in requested
    ):
        raise ValueError("unsafe FantasyID archive member requested")
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if member.name not in requested:
                continue
            if not member.isfile() or member.name in payloads:
                raise ValueError(
                    f"invalid or duplicate selected archive member: {member.name}"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"selected archive member is unreadable: {member.name}")
            payloads[member.name] = extracted.read()
            if len(payloads) == len(requested):
                break
    missing = sorted(requested - set(payloads))
    if missing:
        raise ValueError(f"selected FantasyID archive members are missing: {missing}")
    return payloads


def _archive_members(cases: list[dict[str, Any]]) -> set[str]:
    members: set[str] = set()
    for case in cases:
        for field in (
            "candidate",
            "correct_reference",
            "correct_same_device_reference",
            "selected_reference",
            "wrong_reference",
            "mask",
        ):
            reference = case.get(field)
            if isinstance(reference, dict) and reference.get("archive_member"):
                members.add(str(reference["archive_member"]))
        generation = case.get("mask_generation")
        if isinstance(generation, dict):
            members.add(str(generation["source_metadata_archive_member"]))
    return members


def _reference_bytes(
    reference: dict[str, Any], scratch: Path, payloads: dict[str, bytes]
) -> bytes:
    path = _resolve(scratch, str(reference["path"]))
    if path.is_file():
        payload = path.read_bytes()
    else:
        member = reference.get("archive_member")
        if not member:
            raise FileNotFoundError(path)
        payload = payloads[str(member)]
    if _hash_bytes(payload) != str(reference["sha256"]):
        raise ValueError(f"selected input SHA-256 changed: {reference['path']}")
    return payload


def _open_rgb(payload: bytes, label: str) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as handle:
        handle.load()
        image = handle.convert("RGB")
    if image.width < 1 or image.height < 1:
        raise ValueError(f"selected image has invalid geometry: {label}")
    return image


def _mask_for_case(
    case: dict[str, Any],
    scratch: Path,
    payloads: dict[str, bytes],
    candidate_size: tuple[int, int],
) -> Image.Image:
    reference = case["mask"]
    path = _resolve(scratch, str(reference["path"]))
    if path.is_file():
        payload = path.read_bytes()
    else:
        generation = case.get("mask_generation")
        if not isinstance(generation, dict):
            raise FileNotFoundError(path)
        if str(generation["rasterizer_version"]) != RASTERIZER_VERSION:
            raise ValueError("FantasyID mask rasterizer version changed")
        metadata_payload = payloads[str(generation["source_metadata_archive_member"])]
        if _hash_bytes(metadata_payload) != str(generation["source_metadata_sha256"]):
            raise ValueError("selected FantasyID metadata SHA-256 changed")
        metadata = json.loads(metadata_payload)
        mask_array = _rasterize_box_mask(
            metadata,
            int(generation["annotation_width"]),
            int(generation["annotation_height"]),
        )
        payload = _png_bytes(mask_array)
    if _hash_bytes(payload) != str(reference["sha256"]):
        raise ValueError(f"selected mask SHA-256 changed: {reference['path']}")
    with Image.open(io.BytesIO(payload)) as handle:
        handle.load()
        mask = handle.convert("L")
    if mask.size != candidate_size:
        raise ValueError(
            f"candidate/mask geometry mismatch for {case['case_id']}: "
            f"{candidate_size} versus {mask.size}"
        )
    return mask.point(lambda value: 255 if value > 0 else 0)


def _center_crop_resized_visual(image: Image.Image, fraction: float) -> Image.Image:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("retained fraction must be in (0, 1]")
    width, height = image.size
    crop_width = max(2, min(width, round(width * fraction)))
    crop_height = max(2, min(height, round(height * fraction)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    crop = image.crop((left, top, left + crop_width, top + crop_height))
    return crop.resize(image.size, Image.Resampling.BILINEAR)


def _mask_rgb(mask: Image.Image) -> Image.Image:
    black = Image.new("RGB", mask.size, "#111827")
    yellow = Image.new("RGB", mask.size, "#facc15")
    return Image.composite(yellow, black, mask)


def _mask_overlay(candidate: Image.Image, mask: Image.Image) -> Image.Image:
    base = candidate.convert("RGBA")
    fill = Image.new("RGBA", candidate.size, (236, 72, 153, 112))
    base = Image.composite(fill, base, mask)
    expanded = mask.filter(ImageFilter.MaxFilter(15))
    boundary = ImageChops.subtract(expanded, mask)
    outline = Image.new("RGBA", candidate.size, (250, 204, 21, 255))
    return Image.composite(outline, base, boundary).convert("RGB")


def _comparison_reference(
    case: dict[str, Any],
    correct_reference: Image.Image,
    selected_reference: Image.Image | None,
) -> tuple[Image.Image | None, str]:
    case_id = str(case["case_id"])
    if case_id == "reference_integrity_median_half_view_drop":
        return _center_crop_resized_visual(correct_reference, 0.5), "Audit ref.: center 50% crop"
    if case.get("wrong_reference") is not None:
        return selected_reference, "Audit ref.: shuffled wrong"
    if case_id == "wrong_reference_high_ecc_low_ap":
        return selected_reference, "Audit ref.: same-dataset wrong"
    if case_id == "fantasyid_cross_device_high_ecc_low_ap":
        device = str(case.get("reference_device", "cross-device"))
        return selected_reference, f"Audit ref.: cross-device ({device})"
    return None, "Mask only"


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    left = (size[0] - fitted.width) // 2
    top = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def _short_case_title(case: dict[str, Any]) -> str:
    mapping = {
        "final_in_domain_median_clean": "In-domain median clean",
        "final_generator_holdout_median_clean": "Generator-holdout median clean",
        "final_global_worst_clean": "Global worst clean",
        "final_median_affine_gain": "Median affine gain",
        "final_median_wrong_reference_collapse": "Median wrong-reference collapse",
        "reference_integrity_median_half_view_drop": "Median half-view drop",
        "wrong_reference_high_ecc_low_ap": "High-ECC wrong-reference failure",
        "fantasyid_same_device_median": "FantasyID same-device median",
        "fantasyid_cross_device_high_ecc_low_ap": "High-ECC cross-device failure",
    }
    return mapping[str(case["case_id"])]


def _render_case_panel(
    case: dict[str, Any],
    candidate: Image.Image,
    correct_reference: Image.Image,
    selected_reference: Image.Image | None,
    mask: Image.Image,
    layout: dict[str, Any],
) -> Image.Image:
    page_width = int(layout["page_width"])
    page_height = int(layout["page_height"])
    margin = int(layout["margin"])
    gap = int(layout["gap"])
    title_height = int(layout["title_height"])
    label_height = int(layout["label_height"])
    tile_width = (page_width - 2 * margin - 3 * gap) // 4
    tile_height = page_height - 2 * margin - title_height - label_height
    if min(tile_width, tile_height) < 100:
        raise ValueError("qualitative rendering layout is too small")

    comparison, comparison_label = _comparison_reference(
        case, correct_reference, selected_reference
    )
    third = comparison if comparison is not None else _mask_rgb(mask)
    semantics = str(case["mask_semantics"])
    mask_label = "Weak box overlay" if semantics == "box_mask_not_pixel_accurate" else "Exact-mask overlay"
    images = [candidate, correct_reference, third, _mask_overlay(candidate, mask)]
    labels = ["Candidate", "Correct reference", comparison_label, mask_label]

    page = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(page)
    draw.text(
        (margin, margin),
        _short_case_title(case),
        font=_font(50, bold=True),
        fill="#111827",
    )
    evidence = "paper-evidence case" if case.get("paper_evidence") else "limitation/development case"
    scalar = f"{case['selection_scalar']}={float(case['selection_value']):.6f}"
    detail = f"{case['case_id']}  |  {case['cohort']}  |  {evidence}"
    draw.text((margin, margin + 64), detail, font=_font(25), fill="#475569")
    draw.text(
        (margin, margin + 104),
        f"{scalar}  |  mask: {semantics}  |  heatmap: unavailable",
        font=_font(25),
        fill="#475569",
    )

    top = margin + title_height
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        left = margin + index * (tile_width + gap)
        page.paste(_fit_image(image, (tile_width, tile_height)), (left, top))
        draw.rectangle(
            (left, top, left + tile_width - 1, top + tile_height - 1),
            outline="#cbd5e1",
            width=3,
        )
        draw.text(
            (left + tile_width // 2, top + tile_height + 12),
            label,
            anchor="ma",
            font=_font(26, bold=True),
            fill="#1f2937",
        )
    return page


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", optimize=False)
    temporary.replace(path)


def _save_pdf_atomic(pages: list[Image.Image], path: Path, resolution: int) -> None:
    if not pages:
        raise ValueError("cannot save an empty qualitative packet")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pages[0].save(
        temporary,
        format="PDF",
        resolution=resolution,
        save_all=True,
        append_images=pages[1:],
        title="PairTrace-Doc fixed-rule qualitative audit inputs",
        author="PairTrace-Doc",
        subject="Frozen candidate, reference, and mask panels; no model heatmaps",
    )
    temporary.replace(path)


def _preview(pages: list[Image.Image], columns: int) -> Image.Image:
    if columns < 1:
        raise ValueError("preview column count must be positive")
    preview_width = 1600
    tile_width = preview_width // columns
    tile_height = round(tile_width * pages[0].height / pages[0].width)
    rows = (len(pages) + columns - 1) // columns
    canvas = Image.new("RGB", (preview_width, rows * tile_height), "#e5e7eb")
    for index, page in enumerate(pages):
        tile = page.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        canvas.paste(tile, (left, top))
    return canvas


def _human_review_worksheet(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pending_human_review_and_model_heatmaps",
        "model_heatmaps_available": False,
        "human_review_complete": False,
        "instruction": (
            "Complete every field only after viewing the frozen panel and the "
            "corresponding frozen model heatmap; do not replace a case."
        ),
        "allowed_values": {
            "tampered_region_legible_without_zoom": YES_NO_UNCERTAIN,
            "reference_identity_visually_plausible": YES_NO_UNCERTAIN,
            "registration_artifact_outside_manipulated_region": REGISTRATION_ARTIFACT,
            "prediction_coverage": PREDICTION_COVERAGE,
            "dominant_failure": DOMINANT_FAILURE,
        },
        "reviews": [
            {
                "case_id": case["case_id"],
                "tampered_region_legible_without_zoom": None,
                "reference_identity_visually_plausible": None,
                "registration_artifact_outside_manipulated_region": None,
                "prediction_coverage": None,
                "dominant_failure": None,
                "reviewer_note": None,
                "reviewer_identifier": None,
                "reviewed_at_utc": None,
            }
            for case in cases
        ],
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_runtime(config)

    experiment = config["experiment"]
    protocol_path = _resolve(project_root, experiment["protocol"])
    if _sha256(protocol_path) != str(experiment["expected_protocol_sha256"]):
        raise ValueError("qualitative audit protocol SHA-256 changed")
    specification = config["input"]
    case_manifest_path = _resolve(project_root, specification["case_manifest"])
    case_manifest_hash = _sha256(case_manifest_path)
    if case_manifest_hash != str(specification["expected_case_manifest_sha256"]):
        raise ValueError("qualitative case manifest SHA-256 changed")
    case_manifest = _read_json(case_manifest_path)
    cases = case_manifest["cases"]
    if len(cases) != int(specification["expected_case_count"]):
        raise ValueError("qualitative case count changed")
    expected_ids = [str(value) for value in specification["expected_case_ids"]]
    if [str(case["case_id"]) for case in cases] != expected_ids:
        raise ValueError("qualitative case order or membership changed")
    if case_manifest["rendering"]["sample_replacement_allowed"]:
        raise ValueError("qualitative case manifest unexpectedly allows replacement")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            str(paths["scratch_env"]),
            str(_resolve(project_root, str(paths["scratch_default"]))),
        )
    ).resolve()
    archive_path = _resolve(scratch, specification["fantasyid_archive"])
    if archive_path.stat().st_size != int(specification["expected_fantasyid_archive_bytes"]):
        raise ValueError("FantasyID archive size changed")
    archive_hash = _sha256(archive_path)
    if archive_hash != str(specification["expected_fantasyid_archive_sha256"]):
        raise ValueError("FantasyID archive SHA-256 changed")
    requested_members = _archive_members(cases)
    archive_payloads = _read_selected_tar_members(archive_path, requested_members)

    panel_dir = _resolve(project_root, paths["case_panels_dir"])
    panel_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Image.Image] = []
    case_outputs: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        candidate = _open_rgb(
            _reference_bytes(case["candidate"], scratch, archive_payloads),
            f"{case['case_id']}:candidate",
        )
        correct_spec = case.get("correct_reference") or case.get(
            "correct_same_device_reference"
        )
        if not isinstance(correct_spec, dict):
            raise ValueError(f"selected case lacks a correct reference: {case['case_id']}")
        correct = _open_rgb(
            _reference_bytes(correct_spec, scratch, archive_payloads),
            f"{case['case_id']}:correct_reference",
        )
        selected_spec = case.get("wrong_reference") or case.get("selected_reference")
        selected = (
            _open_rgb(
                _reference_bytes(selected_spec, scratch, archive_payloads),
                f"{case['case_id']}:selected_reference",
            )
            if isinstance(selected_spec, dict)
            else None
        )
        mask = _mask_for_case(
            case, scratch, archive_payloads, candidate.size
        )
        page = _render_case_panel(
            case, candidate, correct, selected, mask, config["layout"]
        )
        panel_path = panel_dir / f"{index:02d}_{case['case_id']}.png"
        _save_png_atomic(page, panel_path)
        pages.append(page)
        case_outputs.append(
            {
                "case_id": case["case_id"],
                "panel": str(panel_path.relative_to(project_root)),
                "panel_sha256": _sha256(panel_path),
                "candidate_sha256": case["candidate"]["sha256"],
                "mask_sha256": case["mask"]["sha256"],
                "mask_semantics": case["mask_semantics"],
                "width": page.width,
                "height": page.height,
            }
        )

    pdf_path = _resolve(project_root, paths["pdf"])
    preview_path = _resolve(project_root, paths["preview_png"])
    _save_pdf_atomic(
        pages, pdf_path, int(config["layout"]["pdf_resolution_dpi"])
    )
    _save_png_atomic(
        _preview(pages, int(config["layout"]["preview_columns"])), preview_path
    )
    worksheet_path = _resolve(project_root, paths["human_review_worksheet"])
    _write_json(worksheet_path, _human_review_worksheet(cases))

    result = {
        "experiment": experiment,
        "status": "qualitative_input_reference_mask_packet_rendered",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_evidence": False,
        "new_scientific_metrics_computed": False,
        "model_inference_performed": False,
        "score_caches_read": False,
        "threshold_selection_used": False,
        "sample_replacement_used": False,
        "full_archive_extracted": False,
        "human_review_complete": False,
        "human_review_blocker": (
            "Frozen model heatmaps are unavailable and a human reviewer has not "
            "completed the fixed rubric."
        ),
        "input": {
            "config": str(config_path.relative_to(project_root)),
            "config_sha256": _sha256(config_path),
            "implementation": str(
                Path(__file__).resolve().relative_to(project_root)
            ),
            "implementation_sha256": _sha256(Path(__file__).resolve()),
            "case_manifest": str(case_manifest_path.relative_to(project_root)),
            "case_manifest_sha256": case_manifest_hash,
            "protocol_sha256": _sha256(protocol_path),
            "fantasyid_archive_sha256": archive_hash,
            "fantasyid_archive_members_read": sorted(requested_members),
        },
        "output": {
            "pdf": str(pdf_path.relative_to(project_root)),
            "pdf_sha256": _sha256(pdf_path),
            "preview_png": str(preview_path.relative_to(project_root)),
            "preview_png_sha256": _sha256(preview_path),
            "human_review_worksheet": str(worksheet_path.relative_to(project_root)),
            "human_review_worksheet_sha256": _sha256(worksheet_path),
            "case_panels": case_outputs,
        },
        "case_count": len(cases),
    }
    render_manifest_path = _resolve(project_root, paths["render_manifest"])
    _write_json(render_manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
