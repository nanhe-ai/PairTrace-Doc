from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from pairtrace_doc.metrics import BinaryCounts, average_precision, binary_counts, precision_recall_f1_iou


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return records


def _combine_counts(items: list[BinaryCounts]) -> BinaryCounts:
    return BinaryCounts(
        tp=sum(item.tp for item in items),
        fp=sum(item.fp for item in items),
        fn=sum(item.fn for item in items),
        tn=sum(item.tn for item in items),
    )


def run(config_path: Path) -> dict[str, float | int | bool | str]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    paths = {name: Path(value) for name, value in config["paths"].items()}
    for name in ("output_predictions", "metrics", "log"):
        paths[name].parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=paths["log"],
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    threshold = float(config["evaluation"]["pixel_threshold"])
    records = _load_jsonl(paths["input_predictions"])
    valid_records: list[dict] = []
    failures = 0
    counts: list[BinaryCounts] = []
    forged_aps: list[float] = []
    authentic_fp = 0
    authentic_pixels = 0

    for record in records:
        if record.get("status") != "ok":
            failures += 1
            logging.error("failed input record example_id=%s", record.get("example_id"))
            continue
        scores = np.asarray(record["scores"], dtype=float)
        mask = np.asarray(record["mask"], dtype=int)
        record_counts = binary_counts(scores, mask, threshold)
        counts.append(record_counts)
        if bool(record["is_forged"]):
            forged_aps.append(average_precision(scores, mask))
        else:
            authentic_fp += record_counts.fp
            authentic_pixels += int(mask.size)

        output_record = dict(record)
        output_record["predicted_mask"] = (scores >= threshold).astype(int).tolist()
        output_record["pixel_threshold"] = threshold
        valid_records.append(output_record)

    if not valid_records or not forged_aps:
        raise RuntimeError("sanity evaluation requires valid records and at least one forged record")

    pooled = _combine_counts(counts)
    metrics: dict[str, float | int | bool | str] = {
        "experiment": config["experiment"]["name"],
        "stage": config["experiment"]["stage"],
        "paper_evidence": bool(config["experiment"]["paper_evidence"]),
        "records": len(records),
        "successful_records": len(valid_records),
        "failed_records": failures,
        "pixel_threshold": threshold,
        "macro_pixel_ap": float(np.mean(forged_aps)),
        **precision_recall_f1_iou(pooled),
        "authentic_pixel_fpr": authentic_fp / authentic_pixels if authentic_pixels else 0.0,
    }

    with paths["output_predictions"].open("w", encoding="utf-8") as handle:
        for record in valid_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with paths["metrics"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    logging.info("completed records=%d failures=%d metrics=%s", len(valid_records), failures, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PairTrace-Doc toy sanity evaluation")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    metrics = run(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

