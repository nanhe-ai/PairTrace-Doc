from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(root: Path, value: str, expected: str, label: str) -> Path:
    path = _resolve(root, value)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} changed: {actual} != {expected}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap(
    left: np.ndarray,
    right: np.ndarray | None,
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> tuple[float, float, float]:
    if right is not None and left.shape != right.shape:
        raise ValueError("paired bootstrap arrays differ")
    values = left if right is None else left - right
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    distribution = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(values.mean()),
        float(np.quantile(distribution, alpha)),
        float(np.quantile(distribution, 1.0 - alpha)),
    )


def _latex_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        label = str(row["label"])
        provenance = str(row["provenance"])
        body.append(
            f"{label} & {row['training_regime']} & "
            f"{row['source_group_macro_ap']:.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] & "
            f"{row['failed_forged_items']} & {provenance} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Clean-pair direct baselines on the 120-group controlled TFR construction. The first four rows are from the prospectively frozen experiment; the VisualDiff-style dense-SIFT reimplementation is a separately preregistered post-hoc addition with AIForge-development thresholds. AP uses 5,000 source-group bootstrap resamples; failed VisualDiff items count as AP zero. This is not an official VisualDiff reproduction or an official TFR benchmark.}",
            r"\label{tab:tfr_direct_paired_posthoc}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Method & Training & Source-group AP [95\% CI] & Failed forged & Provenance \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    protocol = _verify(
        project_root,
        str(experiment["protocol"]),
        str(experiment["expected_protocol_sha256"]),
        "direct baseline protocol",
    )
    inputs = config["inputs"]
    original_path = _verify(
        project_root,
        str(inputs["original_predictions"]),
        str(inputs["expected_original_predictions_sha256"]),
        "original controlled predictions",
    )
    visual_path = _verify(
        project_root,
        str(inputs["visualdiff_predictions"]),
        str(inputs["expected_visualdiff_predictions_sha256"]),
        "VisualDiff predictions",
    )
    visual_summary_path = _verify(
        project_root,
        str(inputs["visualdiff_summary"]),
        str(inputs["expected_visualdiff_summary_sha256"]),
        "VisualDiff summary",
    )
    visual_summary = json.loads(visual_summary_path.read_text(encoding="utf-8"))
    if (
        visual_summary.get("status") != "visualdiff_paired_evaluation_complete"
        or not visual_summary.get("post_hoc_existing_evaluation")
        or visual_summary.get("threshold_selection_used")
    ):
        raise ValueError("VisualDiff post-hoc evidence boundary changed")
    original = _read_jsonl(original_path)
    visual = _read_jsonl(visual_path)
    expected_groups = int(inputs["expected_groups"])
    method_values: dict[str, dict[str, list[float]]] = {}
    failed: dict[str, int] = {}
    specs = {str(item["key"]): item for item in config["methods"]}
    for key, spec in specs.items():
        grouped: dict[str, list[float]] = {}
        failures = 0
        if key == "visualdiff_style":
            selected = [row for row in visual if row["sample_kind"] == "forged"]
            for row in selected:
                group = str(row["source_group_id"])
                value = float(row["pixel_ap"]) if row["status"] == "ok" else 0.0
                failures += int(row["status"] != "ok")
                grouped.setdefault(group, []).append(value)
        else:
            selected = [
                row
                for row in original
                if row["sample_kind"] == "forged"
                and row["condition"] == "clean"
                and row["family"] == spec["family"]
                and row["training_regime"] == spec["training_regime"]
            ]
            for row in selected:
                if row["status"] != "ok":
                    failures += 1
                value = float(row["pixel_ap"]) if row["status"] == "ok" else 0.0
                grouped.setdefault(str(row["source_group_id"]), []).append(value)
        if len(grouped) != expected_groups or any(len(values) == 0 for values in grouped.values()):
            raise ValueError(f"direct method group topology changed: {key}")
        method_values[key] = grouped
        failed[key] = failures
    group_order = sorted(next(iter(method_values.values())))
    if any(set(values) != set(group_order) for values in method_values.values()):
        raise ValueError("direct method group identities differ")
    bootstrap = config["bootstrap"]
    arrays = {
        key: np.asarray([np.mean(values[group]) for group in group_order], dtype=float)
        for key, values in method_values.items()
    }
    metric_rows = []
    for offset, (key, spec) in enumerate(specs.items()):
        mean, low, high = _bootstrap(
            arrays[key],
            None,
            seed=int(bootstrap["seed"]) + offset,
            resamples=int(bootstrap["resamples"]),
            confidence=float(bootstrap["confidence_level"]),
        )
        metric_rows.append(
            {
                "key": key,
                "label": spec["label"],
                "training_regime": spec["training_regime"],
                "provenance": spec["provenance"],
                "source_groups": expected_groups,
                "source_group_macro_ap": mean,
                "ci_low": low,
                "ci_high": high,
                "failed_forged_items": failed[key],
                "paper_evidence": True,
                "post_hoc_addition": key == "visualdiff_style",
            }
        )
    comparison_rows = []
    for offset, item in enumerate(config["comparisons"]):
        effect, low, high = _bootstrap(
            arrays[str(item["left"])],
            arrays[str(item["right"])],
            seed=int(bootstrap["seed"]) + 100 + offset,
            resamples=int(bootstrap["resamples"]),
            confidence=float(bootstrap["confidence_level"]),
        )
        comparison_rows.append(
            {
                "comparison": item["name"],
                "left": item["left"],
                "right": item["right"],
                "effect": effect,
                "ci_low": low,
                "ci_high": high,
                "source_groups": expected_groups,
                "bootstrap_resamples": int(bootstrap["resamples"]),
                "paper_evidence": True,
                "post_hoc": True,
            }
        )
    paths = config["paths"]
    metrics_path = _resolve(project_root, str(paths["metrics"]))
    comparisons_path = _resolve(project_root, str(paths["comparisons"]))
    table_path = _resolve(project_root, str(paths["paper_table"]))
    summary_path = _resolve(project_root, str(paths["summary"]))
    _write_csv(metrics_path, metric_rows)
    _write_csv(comparisons_path, comparison_rows)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_latex_table(metric_rows), encoding="utf-8")
    summary = {
        "status": "ijdar_tfr_direct_paired_posthoc_summary_complete",
        "paper_evidence": True,
        "post_hoc_existing_evaluation": True,
        "official_visualdiff_reproduction": False,
        "groups": expected_groups,
        "bootstrap": bootstrap,
        "metrics": {row["key"]: row for row in metric_rows},
        "comparisons": {row["comparison"]: row for row in comparison_rows},
        "input_sha256": {
            "protocol": _sha256(protocol),
            "original_predictions": _sha256(original_path),
            "visualdiff_predictions": _sha256(visual_path),
            "visualdiff_summary": _sha256(visual_summary_path),
        },
        "outputs": {
            "metrics": str(metrics_path.relative_to(project_root)),
            "metrics_sha256": _sha256(metrics_path),
            "comparisons": str(comparisons_path.relative_to(project_root)),
            "comparisons_sha256": _sha256(comparisons_path),
            "paper_table": str(table_path.relative_to(project_root)),
            "paper_table_sha256": _sha256(table_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
