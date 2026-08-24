from pathlib import Path

from pairtrace_doc.pipelines.freeze_descan18k_source_gate import run


def test_descan_source_freeze_keeps_license_and_pixel_gates_closed() -> None:
    project_root = Path(__file__).resolve().parents[1]
    summary = run(
        project_root / "configs" / "freeze_descan18k_source_gate_20260725.yaml"
    )
    assert summary["status"] == "descan18k_source_frozen_license_gate_closed"
    assert summary["license_gate_open"] is False
    assert summary["archive_bytes_read"] is False
    assert summary["dataset_image_decoded"] is False
    assert summary["model_scoring_started"] is False
    assert summary["source_expected_pairs"] == 360
