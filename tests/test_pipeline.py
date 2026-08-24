import csv
import json
from pathlib import Path

from pairtrace_doc.pipelines.run_sanity import run


def test_debug_pipeline_outputs_required_artifacts(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_root)
    metrics = run(Path("configs/debug.yaml"))

    prediction_path = Path("outputs/predictions/debug_predictions.jsonl")
    metric_path = Path("outputs/tables/debug_metrics.csv")
    log_path = Path("outputs/logs/debug.log")
    assert prediction_path.exists()
    assert metric_path.exists()
    assert log_path.exists()
    assert metrics["paper_evidence"] is False
    assert metrics["successful_records"] == 3

    records = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    assert all("predicted_mask" in record for record in records)
    with metric_path.open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    for field in ("macro_pixel_ap", "pixel_f1", "pixel_iou", "authentic_pixel_fpr"):
        assert field in row

