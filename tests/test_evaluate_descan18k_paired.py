from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from pairtrace_doc.pipelines.evaluate_descan18k_paired import (
    _registered_reference,
    _score_cache,
    run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_score_cache_retains_validity_metadata_and_hash(tmp_path: Path) -> None:
    score = np.linspace(0.0, 1.0, 30, dtype=np.float32).reshape(5, 6)
    valid = np.ones((5, 6), dtype=bool)
    valid[0, 0] = False
    first = _score_cache(
        tmp_path,
        {"fixed": "key"},
        (5, 6),
        lambda: (score, valid, {"registration": "fixture"}),
    )
    second = _score_cache(
        tmp_path,
        {"fixed": "key"},
        (5, 6),
        lambda: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert first[3]["cache_hit"] is False
    assert second[3]["cache_hit"] is True
    assert np.array_equal(first[1], valid)
    assert second[2] == {"registration": "fixture"}
    assert second[3]["cache_sha256"] == _sha256(second[3]["cache_path"])


def test_registered_reference_reports_expected_translation_support() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[10:20, 15:25] = 255
    homography = np.eye(3, dtype=np.float64)
    homography[0, 2] = 3.0
    registered, valid = _registered_reference(image, homography)
    assert registered.shape == image.shape
    assert valid.shape == image.shape[:2]
    assert 0.90 < float(valid.mean()) < 1.0


def _make_config(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    scratch = tmp_path / "scratch"
    cache = scratch / "materialized_cache"
    cache.mkdir(parents=True)

    scan = np.full((128, 160, 3), 225, dtype=np.uint8)
    cv2.rectangle(scan, (12, 12), (148, 116), (45, 55, 65), 2)
    cv2.putText(
        scan,
        "DESCAN",
        (24, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (25, 35, 45),
        2,
        cv2.LINE_AA,
    )
    clean = np.clip(scan.astype(np.int16) + 2, 0, 255).astype(np.uint8)
    attacks = {}
    for index, attack in enumerate(("copy_move", "local_erase")):
        candidate = scan.copy()
        x1 = 30 + index * 50
        candidate[80:96, x1 : x1 + 20] = (120 + index * 30, 80, 40)
        mask = np.any(candidate != scan, axis=2).astype(np.uint8) * 255
        candidate_path = cache / f"{attack}.png"
        mask_path = cache / f"{attack}_mask.png"
        Image.fromarray(candidate).save(candidate_path)
        Image.fromarray(mask).save(mask_path)
        attacks[attack] = {
            "status": "ok",
            "candidate": str(candidate_path.relative_to(scratch)),
            "candidate_sha256": _sha256(candidate_path),
            "mask": str(mask_path.relative_to(scratch)),
            "mask_sha256": _sha256(mask_path),
        }
    scan_path = cache / "scan.png"
    clean_path = cache / "clean.png"
    Image.fromarray(scan).save(scan_path)
    Image.fromarray(clean).save(clean_path)
    group_id = "descan18k:fixture"
    manifest_row = {
        "source_group_id": group_id,
        "source_basename": "scanner01_fixture.tif",
        "scan": str(scan_path.relative_to(scratch)),
        "scan_sha256": _sha256(scan_path),
        "clean": str(clean_path.relative_to(scratch)),
        "clean_sha256": _sha256(clean_path),
        "attacks": attacks,
    }
    manifest = project / "outputs/manifest.jsonl"
    _write_jsonl(manifest, [manifest_row])
    audit = project / "outputs/audit.jsonl"
    _write_jsonl(
        audit,
        [
            {
                "source_group_id": group_id,
                "status": "ok",
                "registration": {
                    "homography": np.eye(3, dtype=float).tolist(),
                },
            }
        ],
    )
    threshold = project / "outputs/threshold.json"
    threshold.write_text("{}\n", encoding="utf-8")
    registry_rows = [
        {
            "name": "raw_rgb_difference",
            "kind": "nonlearned",
            "status": "ready_for_toy",
            "implementation": "fixture_raw_v1",
            "pixel_threshold": 0.01,
            "threshold_artifact": "outputs/threshold.json",
            "threshold_artifact_sha256": _sha256(threshold),
        },
        {
            "name": "ssim_distance",
            "kind": "nonlearned",
            "status": "ready_for_toy",
            "implementation": "fixture_ssim_v1",
            "pixel_threshold": 0.01,
            "threshold_artifact": "outputs/threshold.json",
            "threshold_artifact_sha256": _sha256(threshold),
        },
    ]
    registry = project / "outputs/registry.jsonl"
    _write_jsonl(registry, registry_rows)
    protocol = project / "docs/protocol.md"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text("frozen test protocol\n", encoding="utf-8")
    evaluator = Path(__file__).resolve().parents[1] / (
        "src/pairtrace_doc/pipelines/evaluate_descan18k_paired.py"
    )
    config = {
        "experiment": {
            "name": "test_descan_scoring",
            "stage": "toy3",
            "paper_evidence": False,
            "expected_evaluator_sha256": _sha256(evaluator),
        },
        "bindings": [{"path": "docs/protocol.md", "sha256": _sha256(protocol)}],
        "data": {
            "manifest": {
                "path": "outputs/manifest.jsonl",
                "expected_sha256": _sha256(manifest),
            },
            "audit_records": {
                "path": "outputs/audit.jsonl",
                "expected_sha256": _sha256(audit),
            },
            "expected_groups": 1,
            "flagged_groups": [],
        },
        "method_registry": {
            "path": "outputs/registry.jsonl",
            "expected_sha256": _sha256(registry),
            "expected_names": ["raw_rgb_difference", "ssim_distance"],
        },
        "model": {},
        "reference_conditions": ["scan_reference", "digital_reference"],
        "preprocessing": {
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        },
        "inference": {
            "validation_tile_size": 128,
            "validation_tile_stride": 96,
            "validation_tile_batch_size": 2,
            "amp": False,
        },
        "ssim": {
            "implementation": "pairtrace_opencv_gaussian_ssim_v1",
            "color_space": "RGB",
            "data_range": 1.0,
            "window_size": 11,
            "sigma": 1.5,
            "k1": 0.01,
            "k2": 0.03,
            "covariance": "population",
            "channel_reduction": "arithmetic_mean",
            "border_mode": "reflect_101",
            "distance": "one_minus_ssim_divided_by_two",
        },
        "image_level": {"top_fraction": 0.01},
        "statistics": {
            "bootstrap_seed": 7,
            "bootstrap_replicates": 100,
            "fixed_fpr_targets": [0.01, 0.05],
        },
        "comparisons": [
            {
                "name": "raw_minus_ssim",
                "left": "raw_rgb_difference",
                "right": "ssim_distance",
            }
        ],
        "runtime": {
            "model_scoring_authorized": True,
            "model_training_authorized": False,
            "threshold_selection_authorized": False,
            "sample_selection_authorized": False,
            "device": "cpu",
            "torch_threads": 1,
        },
        "paths": {
            "scratch_default": str(scratch),
            "score_cache_dir": "score_cache",
            "predictions": "outputs/predictions.jsonl",
            "metrics": "outputs/metrics.csv",
            "comparisons": "outputs/comparisons.csv",
            "summary": "outputs/summary.json",
            "progress": "outputs/progress.json",
        },
    }
    config_path = project / "configs/evaluate.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_cpu_nonlearned_toy_run_writes_complete_item_records(tmp_path: Path) -> None:
    config_path = _make_config(tmp_path)
    summary = run(config_path)
    assert summary["status"] == "descan18k_toy3_scoring_complete"
    assert summary["prediction_records"] == 12
    assert summary["failed_prediction_records"] == 0
    assert summary["threshold_selection_performed"] is False
    predictions = [
        json.loads(line)
        for line in (config_path.parent.parent / "outputs/predictions.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(record["score_cache_key"] for record in predictions)
    assert all(record["score_cache_sha256"] for record in predictions)
    assert {record["reference_condition"] for record in predictions} == {
        "scan_reference",
        "digital_reference",
    }
    with pytest.raises(FileExistsError, match="prediction output already exists"):
        run(config_path)
