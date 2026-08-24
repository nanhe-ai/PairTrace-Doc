from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from pairtrace_doc.pipelines.train_student_100 import _resolve, _sha256, _write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_sd(values: list[float]) -> tuple[float, float | None]:
    if not values:
        raise ValueError("paper summary cannot aggregate an empty metric list")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def _format_metric(mean: float, sd: float | None) -> str:
    return f"{mean:.4f}" if sd is None else f"{mean:.4f} $\\pm$ {sd:.4f}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs: dict[str, Path] = {}
    for name, specification in config["inputs"].items():
        path = _resolve(project_root, str(specification["path"]))
        if _sha256(path) != str(specification["expected_sha256"]):
            raise ValueError(f"paper summary input changed: {name}")
        inputs[name] = path
    merged = json.loads(inputs["summary"].read_text(encoding="utf-8"))
    if (
        merged.get("status") != "confirmation_shards_merged"
        or int(merged.get("failures", -1)) != 0
        or int(merged.get("prediction_records", -1)) != 59040
        or int(merged.get("metric_rows", -1)) != 164
        or int(merged.get("comparison_rows", -1)) != 468
    ):
        raise ValueError("paper summary requires the complete merged confirmation run")
    metrics = _read_csv(inputs["metrics"])
    comparisons = _read_csv(inputs["comparisons"])
    conditions = [str(value) for value in config["conditions"]]

    architecture_rows: list[dict[str, Any]] = []
    for specification in config["architecture_rows"]:
        family = str(specification["family"])
        regime = str(specification["training_regime"])
        row: dict[str, Any] = {
            "label": str(specification["label"]),
            "family": family,
            "training_regime": regime,
            "conditions": {},
        }
        for condition in conditions:
            values = [
                float(record["document_macro_pixel_ap"])
                for record in metrics
                if record["family"] == family
                and record["training_regime"] == regime
                and record["condition"] == condition
            ]
            mean, sd = _mean_sd(values)
            expected_count = 1 if regime == "nonlearned" else 3
            if len(values) != expected_count:
                raise ValueError(f"paper architecture aggregation count changed: {family}/{regime}")
            row["conditions"][condition] = {
                "mean": mean,
                "sample_sd": sd,
                "seeds": len(values),
            }
        architecture_rows.append(row)

    def architecture(family: str, regime: str) -> dict[str, Any]:
        return next(
            row
            for row in architecture_rows
            if row["family"] == family and row["training_regime"] == regime
        )

    roundtrip = architecture("pairtrace_9ch", "roundtrip")
    continuation = architecture("pairtrace_9ch", "clean_continuation")
    causal_rows: list[dict[str, Any]] = []
    causal_seed_rows: list[dict[str, Any]] = []
    for condition in conditions:
        left = roundtrip["conditions"][condition]
        right = continuation["conditions"][condition]
        causal_rows.append(
            {
                "condition": condition,
                "roundtrip": left,
                "clean_continuation": right,
                "mean_delta": float(left["mean"]) - float(right["mean"]),
            }
        )
        selected = [
            row
            for row in comparisons
            if row["comparison"].startswith("roundtrip_minus_clean_continuation_seed")
            and row["condition"] == condition
            and row["attack"] == "pooled"
        ]
        if len(selected) != 3:
            raise ValueError("paper causal comparison seed coverage changed")
        for row in selected:
            causal_seed_rows.append(
                {
                    "comparison": row["comparison"],
                    "condition": condition,
                    "delta": float(row["document_macro_pixel_ap_delta"]),
                    "ci95_low": float(row["delta_ci95_low"]),
                    "ci95_high": float(row["delta_ci95_high"]),
                }
            )

    pairtrace_fc = [
        row
        for row in comparisons
        if row["comparison"].startswith("roundtrip_9ch_seed")
        and "minus_roundtrip_fc_siam_diff_seed" in row["comparison"]
        and row["attack"] == "pooled"
    ]
    if len(pairtrace_fc) != 12:
        raise ValueError("paper FC-Siam comparison coverage changed")
    if not all(
        float(row["document_macro_pixel_ap_delta"]) < 0
        and float(row["delta_ci95_high"]) < 0
        for row in pairtrace_fc
    ):
        raise ValueError("paper negative architecture conclusion changed")

    metric_lookup = {
        (row["family"], row["training_regime"], row["condition"], row["seed"]): row
        for row in metrics
    }
    operational: dict[str, Any] = {}
    for family, regime in (
        ("pairtrace_9ch", "roundtrip"),
        ("pairtrace_9ch", "clean_continuation"),
        ("fc_siam_diff", "roundtrip"),
    ):
        key = f"{family}:{regime}"
        operational[key] = {}
        for condition in conditions:
            selected = [
                row
                for (current_family, current_regime, current_condition, _), row in metric_lookup.items()
                if current_family == family
                and current_regime == regime
                and current_condition == condition
            ]
            if len(selected) != 3:
                raise ValueError("paper operational metric seed coverage changed")
            operational[key][condition] = {
                metric: statistics.mean(float(row[metric]) for row in selected)
                for metric in (
                    "document_macro_pixel_ap",
                    "document_macro_pixel_f1",
                    "document_macro_pixel_iou",
                    "authentic_document_macro_pixel_fpr",
                    "image_auroc_top_1pct",
                    "image_tpr_at_fpr_0p01",
                )
            }
            attacks = [json.loads(row["attack_macro_pixel_ap_json"]) for row in selected]
            operational[key][condition]["attack_macro_pixel_ap"] = {
                attack: statistics.mean(value[attack] for value in attacks)
                for attack in sorted(attacks[0])
            }

    verified = {
        "status": "paper_summary_verified",
        "claim_boundary": "controlled_confirmation_only_not_official_tfr",
        "config_sha256": _sha256(config_path),
        "input_sha256": {
            name: str(specification["expected_sha256"])
            for name, specification in config["inputs"].items()
        },
        "conditions": conditions,
        "architecture_rows": architecture_rows,
        "causal_rows": causal_rows,
        "causal_seed_rows": causal_seed_rows,
        "pairtrace_minus_fc_siam_diff": [
            {
                "comparison": row["comparison"],
                "condition": row["condition"],
                "delta": float(row["document_macro_pixel_ap_delta"]),
                "ci95_low": float(row["delta_ci95_low"]),
                "ci95_high": float(row["delta_ci95_high"]),
            }
            for row in pairtrace_fc
        ],
        "operational": operational,
        "selection_performed": False,
    }

    table_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Prospective controlled confirmation on 120 previously unused authentic-document source groups. Learned rows report the mean $\\pm$ sample SD across three predeclared seeds. Every method receives the same candidate/reference pair and evaluation conditions. Raw difference at clean is a construction sanity upper bound, not architecture evidence. This controlled set is not an official TFR benchmark.}",
        "\\label{tab:controlled_confirmation}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Family & Training & Clean & Translation & Affine & Perspective \\\\",
        "\\midrule",
    ]
    regime_labels = {
        "nonlearned": "non-learned",
        "clean": "clean",
        "roundtrip": "round-trip",
        "clean_continuation": "clean continuation",
    }
    for row in architecture_rows:
        values = [
            _format_metric(
                float(row["conditions"][condition]["mean"]),
                row["conditions"][condition]["sample_sd"],
            )
            for condition in conditions
        ]
        label = str(row["label"]).replace("_", "\\_")
        table_lines.append(
            f"{label} & {regime_labels[row['training_regime']]} & "
            + " & ".join(values)
            + " \\\\" 
        )
    table_lines.extend(
        ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""]
    )

    causal_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Same-start, same-budget PairTrace-9C continuation. Values are mean AP $\\pm$ sample SD across three seeds; $\\Delta$ is round-trip minus clean continuation.}",
        "\\label{tab:controlled_causal}",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Condition & Round-trip & Clean cont. & $\\Delta$ \\\\",
        "\\midrule",
    ]
    condition_labels = {
        "clean": "Clean",
        "translation_roundtrip": "Translation",
        "affine_roundtrip": "Affine",
        "perspective_roundtrip": "Perspective",
    }
    for row in causal_rows:
        causal_lines.append(
            f"{condition_labels[row['condition']]} & "
            f"{_format_metric(row['roundtrip']['mean'], row['roundtrip']['sample_sd'])} & "
            f"{_format_metric(row['clean_continuation']['mean'], row['clean_continuation']['sample_sd'])} & "
            f"{row['mean_delta']:+.4f} \\\\" 
        )
    causal_lines.extend(
        ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""]
    )

    paths = config["paths"]
    verified_path = _resolve(project_root, str(paths["verified_summary"]))
    architecture_path = _resolve(project_root, str(paths["architecture_table"]))
    causal_path = _resolve(project_root, str(paths["causal_table"]))
    _write_json(verified_path, verified)
    _write_text(architecture_path, "\n".join(table_lines))
    _write_text(causal_path, "\n".join(causal_lines))
    result = {
        "status": "paper_summary_complete",
        "verified_summary": str(verified_path.relative_to(project_root)),
        "verified_summary_sha256": _sha256(verified_path),
        "architecture_table": str(architecture_path.relative_to(project_root)),
        "architecture_table_sha256": _sha256(architecture_path),
        "causal_table": str(causal_path.relative_to(project_root)),
        "causal_table_sha256": _sha256(causal_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
