from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def _pixel_metrics(
    source: np.ndarray, candidate: np.ndarray, full_mask: np.ndarray
) -> tuple[int, int, float]:
    if source.shape != candidate.shape:
        raise ValueError("candidate shape differs from source")
    changed = np.any(source != candidate, axis=2)
    outside = int(np.count_nonzero(changed & ~full_mask))
    inside = int(np.count_nonzero(changed & full_mask))
    mask_pixels = int(np.count_nonzero(full_mask))
    fraction = float(inside / mask_pixels) if mask_pixels else 0.0
    return outside, inside, fraction


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _labeled_panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    header = 36
    panel = Image.new("RGB", (size[0], size[1] + header), "white")
    panel.paste(_fit(image, size), (0, header))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 10), label, fill="black", font=ImageFont.load_default())
    return panel


def _review_sheet(
    editor_id: str,
    target: dict[str, Any],
    source: Image.Image,
    context: Image.Image,
    mask: Image.Image,
    raw: Image.Image,
    candidate: Image.Image,
) -> Image.Image:
    box = tuple(int(value) for value in target["context_box_xyxy"])
    candidate_context = candidate.crop(box)
    source_context = source.crop(box)
    if candidate_context.size != context.size:
        raise ValueError("candidate context geometry differs from frozen context")

    local_mask = np.asarray(mask.convert("L")) > 0
    ys, xs = np.nonzero(local_mask)
    if len(xs) == 0:
        raise ValueError("empty review mask")
    margin = 32
    zoom_box = (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(context.width, int(xs.max()) + margin + 1),
        min(context.height, int(ys.max()) + margin + 1),
    )
    original_zoom = source_context.crop(zoom_box)
    candidate_zoom = candidate_context.crop(zoom_box)
    raw_resized = raw.convert("RGB").resize(context.size, Image.Resampling.BICUBIC)
    raw_zoom = raw_resized.crop(zoom_box)

    diff = np.abs(
        np.asarray(candidate_context, dtype=np.int16)
        - np.asarray(source_context, dtype=np.int16)
    ).max(axis=2)
    diff_visual = np.zeros((*diff.shape, 3), dtype=np.uint8)
    diff_visual[..., 0] = np.clip(diff * 6, 0, 255).astype(np.uint8)
    diff_visual[..., 1] = np.where(local_mask, 40, 0).astype(np.uint8)
    diff_image = Image.fromarray(diff_visual, mode="RGB")

    panels = [
        _labeled_panel(source_context, "frozen context", (512, 512)),
        _labeled_panel(raw_resized, "raw editor output", (512, 512)),
        _labeled_panel(candidate_context, "exact-mask candidate", (512, 512)),
        _labeled_panel(original_zoom, "target zoom: source", (512, 256)),
        _labeled_panel(raw_zoom, "target zoom: raw", (512, 256)),
        _labeled_panel(candidate_zoom, "target zoom: candidate", (512, 256)),
        _labeled_panel(diff_image, "difference map (red) + mask (green)", (512, 512)),
    ]
    title_height = 58
    sheet = Image.new("RGB", (3 * 512, title_height + 548 + 292 + 548), "#eeeeee")
    title = (
        f"{editor_id} | {target['rehearsal_id']} | replacement="
        f"{target['replacement_text']} | agent-only non-human review"
    )
    ImageDraw.Draw(sheet).text(
        (12, 18), title, fill="black", font=ImageFont.load_default()
    )
    for column, panel in enumerate(panels[:3]):
        sheet.paste(panel, (column * 512, title_height))
    for column, panel in enumerate(panels[3:6]):
        sheet.paste(panel, (column * 512, title_height + 548))
    sheet.paste(panels[6], (0, title_height + 548 + 292))
    return sheet


def run(config_path: Path, project_root: Path, storage_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    storage_root = storage_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    for field, expected in config["frozen_inputs"].items():
        if not field.endswith("_sha256"):
            continue
        path = _resolve(
            project_root, str(config["frozen_inputs"][field.removesuffix("_sha256")])
        )
        if _sha256(path) != str(expected):
            raise ValueError(f"frozen input changed: {path}")

    target_path = _resolve(project_root, str(config["inputs"]["target_manifest"]))
    targets = _read_jsonl(target_path)
    target_by_id = {str(row["rehearsal_id"]): row for row in targets}
    attempts: list[dict[str, Any]] = []
    reports = []
    for editor in config["inputs"]["editors"]:
        attempt_path = _resolve(project_root, str(editor["attempts"]))
        report_path = _resolve(project_root, str(editor["report"]))
        if _sha256(attempt_path) != str(editor["attempts_sha256"]):
            raise ValueError(f"attempt file changed: {attempt_path}")
        if _sha256(report_path) != str(editor["report_sha256"]):
            raise ValueError(f"editor report changed: {report_path}")
        reports.append(_read_json(report_path))
        attempts.extend(_read_jsonl(attempt_path))

    accepted_attempts = [
        row for row in attempts if bool(row.get("accepted_automated_gate"))
    ]
    expected_calls = int(config["gates"]["expected_calls"])
    if len(accepted_attempts) != expected_calls:
        raise ValueError(
            f"expected {expected_calls} accepted attempts, found {len(accepted_attempts)}"
        )
    call_keys = {
        (str(row["rehearsal_id"]), str(row["editor_id"])) for row in accepted_attempts
    }
    if len(call_keys) != expected_calls:
        raise ValueError("accepted attempts do not represent six distinct calls")

    review_root = _resolve(storage_root, str(config["outputs"]["review_root"]))
    audit_rows: list[dict[str, Any]] = []
    for attempt in sorted(
        accepted_attempts, key=lambda row: (row["rehearsal_id"], row["editor_id"])
    ):
        target = target_by_id[str(attempt["rehearsal_id"])]
        if str(attempt["editor_id"]) not in target["editor_ids"]:
            raise ValueError("accepted editor is absent from frozen target assignment")
        source_path = _resolve(storage_root, str(target["path"]))
        context_path = _resolve(storage_root, str(target["context_path"]))
        mask_path = _resolve(storage_root, str(target["mask_path"]))
        candidate_path = _resolve(storage_root, str(attempt["candidate_path"]))
        raw_path = _resolve(storage_root, str(attempt["raw_context_path"]))
        for path, expected in (
            (source_path, target["encoded_sha256"]),
            (context_path, target["context_sha256"]),
            (mask_path, target["mask_sha256"]),
            (candidate_path, attempt["candidate_sha256"]),
            (raw_path, attempt["raw_context_sha256"]),
        ):
            if _sha256(path) != str(expected):
                raise ValueError(f"artifact hash changed: {path}")
        with Image.open(source_path) as handle:
            source = ImageOps.exif_transpose(handle).convert("RGB")
        with Image.open(context_path) as handle:
            context = handle.convert("RGB")
        with Image.open(mask_path) as handle:
            mask = handle.convert("L")
        with Image.open(candidate_path) as handle:
            candidate = handle.convert("RGB")
        with Image.open(raw_path) as handle:
            raw = handle.convert("RGB")
        if candidate.size != source.size:
            raise ValueError("candidate dimensions differ from source")
        candidate_array = np.asarray(candidate)
        if not np.isfinite(candidate_array.astype(np.float32)).all():
            raise ValueError("candidate contains nonfinite values")
        full_mask = np.zeros((source.height, source.width), dtype=bool)
        box = tuple(int(value) for value in target["context_box_xyxy"])
        full_mask[box[1] : box[3], box[0] : box[2]] = np.asarray(mask) > 0
        outside, inside, fraction = _pixel_metrics(
            np.asarray(source), candidate_array, full_mask
        )
        if outside != 0:
            raise ValueError("accepted candidate has outside-mask differences")
        if fraction < float(config["gates"]["minimum_changed_fraction_inside_mask"]):
            raise ValueError("accepted candidate fails inside-mask change threshold")
        if outside != int(attempt["outside_mask_changed_pixels"]):
            raise ValueError("outside-mask count differs from attempt record")
        if inside != int(attempt["changed_pixels_inside_mask"]):
            raise ValueError("inside-mask count differs from attempt record")

        sheet_path = review_root / (
            f"{target['artifact_id']}__{attempt['editor_id']}__attempt_"
            f"{attempt['attempt_index']}.png"
        )
        _save_png(
            sheet_path,
            _review_sheet(
                str(attempt["editor_id"]),
                target,
                source,
                context,
                mask,
                raw,
                candidate,
            ),
        )
        audit_rows.append(
            {
                "accepted_automated_gate_recomputed": True,
                "attempt_index": int(attempt["attempt_index"]),
                "candidate_path": str(candidate_path),
                "candidate_sha256": _sha256(candidate_path),
                "changed_fraction_inside_mask": round(fraction, 8),
                "changed_pixels_inside_mask": inside,
                "editor_id": str(attempt["editor_id"]),
                "outside_mask_changed_pixels": outside,
                "rehearsal_id": str(attempt["rehearsal_id"]),
                "replacement_text": str(target["replacement_text"]),
                "review_sheet": str(sheet_path),
                "review_sheet_sha256": _sha256(sheet_path),
                "source_dimensions": list(source.size),
                "visual_review": "pending_agent_nonhuman_review",
            }
        )

    accepted_by_editor: dict[str, int] = {}
    for row in audit_rows:
        accepted_by_editor[row["editor_id"]] = (
            accepted_by_editor.get(row["editor_id"], 0) + 1
        )
    no_oom = all(
        "out of memory" not in str(row.get("failure_reason", "")).lower()
        for row in attempts
    ) and all(report.get("status") == "passed" for report in reports)
    free_bytes = shutil.disk_usage(storage_root).free
    minimum_free = int(float(config["gates"]["minimum_free_space_gib"]) * 1024**3)
    automatic_pass = all(
        (
            len(call_keys) == expected_calls,
            len(audit_rows) >= int(config["gates"]["minimum_accepted_calls"]),
            set(accepted_by_editor) == set(config["gates"]["required_editors"]),
            all(value >= 1 for value in accepted_by_editor.values()),
            no_oom,
            all(row["outside_mask_changed_pixels"] == 0 for row in audit_rows),
            free_bytes >= minimum_free,
        )
    )
    review_manifest_path = _resolve(
        project_root, str(config["outputs"]["review_manifest"])
    )
    _write_jsonl(review_manifest_path, audit_rows)
    result = {
        "accepted_by_editor": accepted_by_editor,
        "accepted_calls": len(audit_rows),
        "all_six_calls_have_records": len(call_keys) == expected_calls,
        "authorization": {
            "final_source_images_read": False,
            "nonfinal_toy_images_revalidated": len(audit_rows),
            "pilot100_run": False,
        },
        "automatic_gate_passed": automatic_pass,
        "failed_execution_evidence": config["inputs"]["failed_execution_evidence"],
        "first_valid_attempt_acceptance": sum(
            int(row["attempt_index"] == 0) for row in audit_rows
        ),
        "minimum_accepted_calls": int(config["gates"]["minimum_accepted_calls"]),
        "no_cuda_oom": no_oom,
        "reports": [str(row["editor_id"]) for row in reports],
        "review_manifest": str(review_manifest_path.relative_to(project_root)),
        "review_manifest_sha256": _sha256(review_manifest_path),
        "storage": {
            "free_bytes": free_bytes,
            "minimum_free_bytes": minimum_free,
        },
        "status": (
            "automatic_gate_passed_agent_visual_review_pending"
            if automatic_pass
            else "automatic_gate_failed"
        ),
    }
    report_path = _resolve(project_root, str(config["outputs"]["report"]))
    _write_json(report_path, result)
    result["report_path"] = str(report_path.relative_to(project_root))
    result["report_sha256"] = _sha256(report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute frozen Toy-3 editor gates and render agent-review sheets."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.project_root, args.storage_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
