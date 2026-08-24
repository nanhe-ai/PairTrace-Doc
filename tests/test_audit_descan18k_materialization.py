from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from pairtrace_doc.pipelines.audit_descan18k_materialization import run
from pairtrace_doc.pipelines.materialize_descan18k_edits import _materialize_pair


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiff_bytes(shift: int) -> bytes:
    image = np.full((192, 240, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (230, 182), (25, 35, 45), 2)
    cv2.circle(image, (175, 52), 28, (80 + shift, 120, 170), -1)
    for y in range(45, 170, 18):
        cv2.putText(
            image,
            "DESCAN 12345",
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (35 + shift, 45, 55),
            1,
            cv2.LINE_AA,
        )
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="TIFF")
    return buffer.getvalue()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config(tmp_path: Path, *, visual: bool = True) -> Path:
    project = tmp_path / "project"
    scratch = tmp_path / "scratch"
    record, _, _ = _materialize_pair(
        basename="fixture.tif",
        scan_payload=_tiff_bytes(0),
        clean_payload=_tiff_bytes(2),
        cache_root=scratch / "cache",
        scratch=scratch,
    )
    manifest = project / "outputs/manifest.jsonl"
    _write_jsonl(manifest, [record])
    protocol = project / "docs/protocol.md"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text("frozen test protocol\n", encoding="utf-8")
    review = project / "outputs/review.jsonl"
    if visual:
        _write_jsonl(
            review,
            [
                {
                    "source_group_id": record["source_group_id"],
                    "reviewer_type": "synthetic_fixture",
                    "visual_gate_passed": True,
                }
            ],
        )
    config = {
        "experiment": {
            "name": "test_descan_audit",
            "stage": "pilot20",
            "paper_evidence": False,
        },
        "bindings": [{"path": "docs/protocol.md", "sha256": _sha256(protocol)}],
        "runtime": {
            "model_scoring_authorized": False,
            "verify_artifact_hashes": True,
        },
        "audit": {
            "expected_height_width": [192, 240],
            "expected_attacks": ["copy_move", "local_erase"],
            "copy_source_destination_gap_pixels": 8,
            "marker_green": {
                "min_green_channel": 160,
                "min_green_minus_red": 40,
                "min_green_minus_blue": 20,
                "exact_marker_rgb": [[0, 255, 0]],
            },
            "registration": {
                "max_side": 256,
                "iterations": 100,
                "epsilon": 1.0e-6,
                "gauss_filter_size": 5,
                "canny_low_threshold": 50,
                "canny_high_threshold": 150,
            },
        },
        "gates": {
            "expected_groups": 1,
            "minimum_successes_per_attack": 1,
            "mutual_valid_support_min": 0.9,
            "registration_support_rate_min": 1.0,
            "full_population_groups": 3,
            "full_storage_bytes_max_exclusive": 100_000_000,
        },
        "visual_review": {
            "required": True,
            "records": "outputs/review.jsonl" if visual else None,
            "expected_records_sha256": _sha256(review) if visual else None,
        },
        "paths": {
            "scratch_default": str(scratch),
            "input_manifest": "outputs/manifest.jsonl",
            "expected_input_manifest_sha256": _sha256(manifest),
            "audit_records": "outputs/audit.jsonl",
            "summary": "outputs/summary.json",
        },
    }
    path = project / "configs/audit.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_audit_passes_exact_masks_registration_and_visual_gate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = run(config)
    assert summary["status"] == "descan18k_pilot20_audit_passed"
    assert summary["records"]["groups_ok"] == 1
    assert summary["records"]["attack_successes"] == {
        "copy_move": 1,
        "local_erase": 1,
    }
    assert summary["records"]["novel_marker_green_pixels"] == 0
    assert summary["registration"]["support_rate"] == 1.0
    assert summary["decision"]["full_expansion_authorized_by_this_gate"] is True
    record = json.loads(
        (config.parent.parent / "outputs/audit.jsonl").read_text().splitlines()[0]
    )
    assert record["attacks"]["copy_move"]["mask_values"] == [0, 255]
    assert record["attacks"]["local_erase"][
        "outside_destination_changed_pixels"
    ] == 0


def test_visual_review_is_an_explicit_non_silent_gate(tmp_path: Path) -> None:
    config = _config(tmp_path, visual=False)
    summary = run(config)
    assert summary["decision"]["automatic_gate_passed"] is True
    assert summary["decision"]["stage_gate_passed"] is False
    assert summary["visual_review"]["status"] == "pending"


def test_corrupt_candidate_is_recorded_as_a_group_failure(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = config_path.parent.parent
    manifest = project / config["paths"]["input_manifest"]
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    row["attacks"]["copy_move"]["candidate_sha256"] = "0" * 64
    _write_jsonl(manifest, [row])
    config["paths"]["expected_input_manifest_sha256"] = _sha256(manifest)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = run(config_path)
    assert summary["records"]["groups_failed"] == 1
    assert summary["records"]["attack_successes"]["copy_move"] == 0
    record = json.loads((project / "outputs/audit.jsonl").read_text().splitlines()[0])
    assert record["attacks"]["copy_move"]["status"] == "failed"
    assert "SHA-256 changed" in record["attacks"]["copy_move"]["errors"][0]
