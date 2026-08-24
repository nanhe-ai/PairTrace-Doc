from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pairtrace_doc.pipelines.train_student_100 import (
    _read_jsonl,
    _resolve,
    _sha256,
    _write_csv,
    _write_json,
)


def _top_fraction_mean(scores: np.ndarray, fraction: float) -> float:
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("image score map must be a finite matrix")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("image score top fraction is invalid")
    flattened = scores.astype(np.float64, copy=False).reshape(-1)
    count = max(1, int(math.ceil(flattened.size * fraction)))
    return float(np.partition(flattened, flattened.size - count)[-count:].mean())


def _select_image_threshold(
    forged_scores: np.ndarray,
    authentic_scores: np.ndarray,
    fpr_max: float,
) -> dict[str, Any]:
    if not forged_scores.size or not authentic_scores.size:
        raise ValueError("image threshold selection requires both classes")
    candidates = np.r_[np.inf, np.unique(np.r_[forged_scores, authentic_scores])]
    tpr = np.asarray(
        [np.mean(forged_scores >= threshold) for threshold in candidates]
    )
    fpr = np.asarray(
        [np.mean(authentic_scores >= threshold) for threshold in candidates]
    )
    feasible = np.flatnonzero(fpr <= fpr_max + 1e-12)
    best_tpr = tpr[feasible].max()
    selected = feasible[np.isclose(tpr[feasible], best_tpr, atol=1e-12, rtol=0)]
    best_fpr = fpr[selected].min()
    selected = selected[np.isclose(fpr[selected], best_fpr, atol=1e-12, rtol=0)]
    index = int(selected[np.argmax(candidates[selected])])
    return {
        "threshold": float(candidates[index]),
        "development_forged_image_tpr": float(tpr[index]),
        "development_authentic_image_fpr": float(fpr[index]),
        "authentic_image_fpr_max": float(fpr_max),
        "candidate_count": int(candidates.size),
        "selected_using_final_reserve": False,
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = config["runtime"]
    if any(
        bool(runtime.get(key))
        for key in (
            "gpu_launch_authorized",
            "model_training_authorized",
            "method_change_authorized",
            "selected_image_read_allowed",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("image threshold freeze must be cache-only")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("development threshold freeze cannot be paper evidence")
    scratch = Path(
        os.environ.get(
            config["paths"]["scratch_env"],
            str(_resolve(project_root, config["paths"]["scratch_default"])),
        )
    ).resolve()

    source_cache: dict[str, tuple[dict[str, Any], list[dict[str, Any]], str]] = {}
    result_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for name, specification in config["scorers"].items():
        source_name = str(specification["source"])
        if source_name not in source_cache:
            source = config["sources"][source_name]
            summary_path = _resolve(project_root, source["summary"])
            digest = _sha256(summary_path)
            if digest != source["expected_summary_sha256"]:
                raise ValueError(f"{source_name} summary SHA-256 changed")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("final_reserve_read") or summary.get("paper_evidence"):
                raise ValueError(f"{source_name} crossed an evidence boundary")
            predictions_path = _resolve(
                project_root, summary["outputs"]["predictions"]
            )
            if _sha256(predictions_path) != summary["outputs"][
                "predictions_sha256"
            ]:
                raise ValueError(f"{source_name} prediction SHA-256 changed")
            rows = _read_jsonl(predictions_path)
            if any(row.get("status") != "ok" for row in rows):
                raise ValueError(f"{source_name} contains failed predictions")
            source_cache[source_name] = (summary, rows, digest)
            source_hashes[source_name] = digest
        _, predictions, _ = source_cache[source_name]
        condition = str(specification["condition"])
        selected = [row for row in predictions if row["condition"] == condition]
        forged_rows = [row for row in selected if row["sample_kind"] == "forged"]
        authentic_rows = [
            row for row in selected if row["sample_kind"] == "authentic"
        ]
        expected = int(specification["expected_documents_per_class"])
        if len(forged_rows) != expected or len(authentic_rows) != expected:
            raise ValueError(f"{name} development image count changed")

        def image_scores(rows: list[dict[str, Any]]) -> np.ndarray:
            values = []
            for row in rows:
                cache_path = _resolve(scratch, row["score_cache"])
                with np.load(cache_path, allow_pickle=False) as archive:
                    scores = archive["scores"]
                values.append(
                    _top_fraction_mean(
                        scores, float(config["image_score"]["top_fraction"])
                    )
                )
            return np.asarray(values, dtype=float)

        forged_scores = image_scores(forged_rows)
        authentic_scores = image_scores(authentic_rows)
        selected_threshold = _select_image_threshold(
            forged_scores,
            authentic_scores,
            float(config["image_score"]["authentic_image_fpr_max"]),
        )
        result_rows.append(
            {
                "scorer": name,
                "source": source_name,
                "condition": condition,
                "development_documents_per_class": expected,
                "image_score": "top_fraction_mean",
                "image_score_top_fraction": float(
                    config["image_score"]["top_fraction"]
                ),
                **selected_threshold,
                "paper_evidence": False,
                "final_reserve_read": False,
            }
        )

    table_path = _resolve(project_root, config["paths"]["table"])
    summary_path = _resolve(project_root, config["paths"]["summary"])
    _write_csv(table_path, result_rows)
    output = {
        "experiment": config["experiment"],
        "status": "final_image_operating_points_frozen_on_viewed_development",
        "paper_evidence": False,
        "viewed_development_cache_read": True,
        "selected_image_read": False,
        "final_reserve_read": False,
        "image_score": config["image_score"],
        "source_summary_sha256": source_hashes,
        "operating_points": {row["scorer"]: row for row in result_rows},
        "outputs": {
            "table": str(table_path.relative_to(project_root)),
            "table_sha256": _sha256(table_path),
        },
    }
    _write_json(summary_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
