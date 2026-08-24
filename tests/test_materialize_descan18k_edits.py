from __future__ import annotations

import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from pairtrace_doc.pipelines.materialize_descan18k_edits import (
    _materialize_pair,
    _pair_members,
    _select_pairs,
    _validate_prerequisites,
    run,
)
import pairtrace_doc.pipelines.materialize_descan18k_edits as materializer


def _tiff_bytes(offset: int) -> bytes:
    image = np.full((192, 240, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (228, 180), (25, 35, 45), 2)
    for y in range(35, 165, 18):
        cv2.putText(
            image,
            f"SCAN {offset:02d} 123456",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (35 + offset, 45, 55),
            1,
            cv2.LINE_AA,
        )
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="TIFF")
    return buffer.getvalue()


def test_pending_config_refuses_before_archive_access() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with pytest.raises(PermissionError, match="gate is closed"):
        run(
            project_root
            / "configs"
            / "materialize_descan18k_toy3_pending_license_20260725.yaml"
        )


def test_pairing_and_hash_order_are_deterministic() -> None:
    names = [
        "Test/scan/b.tif",
        "Test/clean/a.tif",
        "Test/scan/a.tif",
        "Test/clean/b.tif",
        "README.md",
    ]
    pairs = _pair_members(names, "Test/scan", "Test/clean")
    assert {row["basename"] for row in pairs} == {"a.tif", "b.tif"}
    assert _select_pairs(pairs, salt="fixed:", count=1) == _select_pairs(
        list(reversed(pairs)), salt="fixed:", count=1
    )


def test_synthetic_pair_materialization_writes_exact_masks(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    record, hits, writes = _materialize_pair(
        basename="fixture.tif",
        scan_payload=_tiff_bytes(1),
        clean_payload=_tiff_bytes(2),
        cache_root=scratch / "cache",
        scratch=scratch,
    )
    assert hits == 0
    assert writes == 6
    assert record["status"] == "ok"
    assert set(record["attacks"]) == {"copy_move", "local_erase"}
    for attack in record["attacks"].values():
        assert attack["status"] == "ok"
        candidate = np.asarray(Image.open(scratch / attack["candidate"]).convert("RGB"))
        scan = np.asarray(Image.open(scratch / record["scan"]).convert("RGB"))
        mask = np.asarray(Image.open(scratch / attack["mask"]).convert("L")) > 0
        assert np.array_equal(mask, np.any(candidate != scan, axis=2))


def test_attack_failure_is_retained_in_pair_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_local_erase(scan: np.ndarray, group_id: str) -> None:
        raise RuntimeError(f"frozen_fixture_failure:{group_id}:{scan.shape[0]}")

    monkeypatch.setattr(materializer, "local_erase_edit", fail_local_erase)
    scratch = tmp_path / "scratch"
    record, hits, writes = _materialize_pair(
        basename="fixture.tif",
        scan_payload=_tiff_bytes(1),
        clean_payload=_tiff_bytes(2),
        cache_root=scratch / "cache",
        scratch=scratch,
    )
    assert hits == 0
    assert writes == 4
    assert record["status"] == "partial_attack_failure"
    assert record["failed_attacks"] == ["local_erase"]
    assert record["attacks"]["copy_move"]["status"] == "ok"
    assert record["attacks"]["local_erase"]["status"] == "failed"
    assert "frozen_fixture_failure" in record["attacks"]["local_erase"][
        "failure_reason"
    ]


def test_full_prerequisite_requires_hash_bound_pass_decision(tmp_path: Path) -> None:
    gate = tmp_path / "pilot_gate.json"
    gate.write_text(
        '{"decision":{"stage_gate_passed":true,'
        '"full_expansion_authorized_by_this_gate":true}}\n',
        encoding="utf-8",
    )
    prerequisite = {
        "path": str(gate),
        "sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
        "required_values": {
            "decision.stage_gate_passed": True,
            "decision.full_expansion_authorized_by_this_gate": True,
        },
    }
    _validate_prerequisites(tmp_path, [prerequisite])
    prerequisite["required_values"]["decision.stage_gate_passed"] = False
    with pytest.raises(PermissionError, match="prerequisite gate failed"):
        _validate_prerequisites(tmp_path, [prerequisite])
