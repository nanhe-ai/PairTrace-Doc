from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _key(method: str, device: str) -> str:
    return f"{method}|{device}"


def _build_eligible_groups(
    member_rows: list[dict[str, Any]],
    face_db: str,
    methods: list[str],
    devices: list[str],
    excluded_templates: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, int]]:
    image_paths = {
        str(row["dataset_identity"]["relative_path"])
        for row in member_rows
        if row.get("type") == "file"
        and row.get("suffix") == ".jpg"
        and row.get("safe_path") is True
        and row.get("dataset_identity")
    }
    metadata = [
        row["metadata_summary"]
        for row in member_rows
        if row.get("type") == "file"
        and row.get("suffix") == ".json"
        and row.get("safe_path") is True
        and row.get("metadata_summary", {}).get("face_db") == face_db
    ]
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        grouped[str(row["stem"])].append(row)

    eligible: dict[str, dict[str, Any]] = {}
    excluded: dict[str, list[str]] = {}
    expected_keys = {_key("none", device) for device in devices} | {
        _key(method, device) for method in methods for device in devices
    }
    for stem, rows in sorted(grouped.items()):
        reasons: list[str] = []
        template = stem.split("-", 1)[0]
        if template in excluded_templates:
            reasons.append("excluded_template")
        variants: dict[str, dict[str, Any]] = {}
        for row in rows:
            variant_key = _key(str(row["method"]), str(row["device"]))
            if variant_key in variants:
                reasons.append(f"duplicate_variant:{variant_key}")
            variants[variant_key] = row
        missing = sorted(expected_keys - set(variants))
        extra = sorted(set(variants) - expected_keys)
        if missing:
            reasons.append(f"missing_variants:{','.join(missing)}")
        if extra:
            reasons.append(f"unexpected_variants:{','.join(extra)}")
        identities = {
            (str(row.get("face_db")), str(row.get("face_id"))) for row in rows
        }
        if len(identities) != 1:
            reasons.append("identity_mismatch")
        for variant_key in sorted(expected_keys & set(variants)):
            row = variants[variant_key]
            json_path = str(row["relative_path"])
            image_path = str(PurePosixPath(json_path).with_suffix(".jpg"))
            if image_path not in image_paths:
                reasons.append(f"missing_image:{variant_key}")
            if int(row.get("annotation_width", 0)) <= 0 or int(
                row.get("annotation_height", 0)
            ) <= 0:
                reasons.append(f"invalid_dimensions:{variant_key}")
            if str(row["method"]) != "none":
                altered = int(row.get("altered_region_count", 0))
                valid = int(row.get("valid_altered_rectangle_count", 0))
                if altered <= 0:
                    reasons.append(f"no_altered_rectangle:{variant_key}")
                if valid != altered:
                    reasons.append(f"invalid_altered_rectangle:{variant_key}")
        for method in methods:
            for device in devices:
                attack = variants.get(_key(method, device))
                authentic = variants.get(_key("none", device))
                if attack is None or authentic is None:
                    continue
                attack_size = (
                    int(attack.get("annotation_width", 0)),
                    int(attack.get("annotation_height", 0)),
                )
                authentic_size = (
                    int(authentic.get("annotation_width", 0)),
                    int(authentic.get("annotation_height", 0)),
                )
                if attack_size != authentic_size:
                    reasons.append(f"pair_dimension_mismatch:{method}|{device}")
        if reasons:
            excluded[stem] = sorted(set(reasons))
            continue
        face_id = next(iter(identities))[1]
        eligible[stem] = {
            "stem": stem,
            "template": template,
            "face_db": face_db,
            "face_id": face_id,
            "variants": variants,
        }
    audit_counts = {
        "facelab_metadata_records": len(metadata),
        "facelab_source_groups": len(grouped),
        "eligible_source_groups": len(eligible),
        "excluded_source_groups": len(excluded),
    }
    return eligible, excluded, audit_counts


def _hash_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _assign_balanced_cells(
    groups: dict[str, dict[str, Any]],
    methods: list[str],
    devices: list[str],
    seed: int,
) -> list[tuple[dict[str, Any], str, str]]:
    ordered_groups = sorted(
        groups.values(), key=lambda row: (_hash_key(seed, str(row["stem"])), row["stem"])
    )
    cells = [(method, device) for method in methods for device in devices]
    ordered_cells = sorted(
        cells,
        key=lambda cell: (
            _hash_key(seed, f"{cell[0]}|{cell[1]}"),
            cell[0],
            cell[1],
        ),
    )
    return [
        (group, *ordered_cells[index % len(ordered_cells)])
        for index, group in enumerate(ordered_groups)
    ]


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    experiment = config["experiment"]
    protocol_path = _resolve(project_root, experiment["protocol"])
    protocol_sha256 = _sha256(protocol_path)
    if protocol_sha256 != experiment["expected_protocol_sha256"]:
        raise ValueError("external-development protocol SHA-256 changed")
    if bool(experiment["paper_evidence"]):
        raise ValueError("FantasyID external-development freeze cannot be paper evidence")
    runtime = config["runtime"]
    if any(bool(value) for value in runtime.values()):
        raise ValueError("metadata freeze crossed a read, training, or evaluation boundary")

    inputs = config["input"]
    member_records_path = _resolve(project_root, inputs["member_records"])
    audit_summary_path = _resolve(project_root, inputs["audit_summary"])
    for name, path in (
        ("member_records", member_records_path),
        ("audit_summary", audit_summary_path),
    ):
        expected = str(inputs[f"expected_{name}_sha256"])
        if _sha256(path) != expected:
            raise ValueError(f"FantasyID {name} SHA-256 changed")
    with audit_summary_path.open("r", encoding="utf-8") as handle:
        archive_audit = json.load(handle)
    if archive_audit.get("actual_sha256") != inputs["expected_archive_sha256"]:
        raise ValueError("FantasyID archive identity changed")
    if archive_audit.get("status") != "integrity_passed":
        raise ValueError("FantasyID archive integrity is not passed")

    selection = config["selection"]
    if bool(selection["selection_uses_image_or_model_output"]):
        raise ValueError("selection must remain metadata-only")
    methods = [str(value) for value in selection["attack_methods"]]
    devices = [str(value) for value in selection["devices"]]
    member_rows = _read_jsonl(member_records_path)
    eligible, excluded, audit_counts = _build_eligible_groups(
        member_rows,
        str(selection["face_db"]),
        methods,
        devices,
        {str(value) for value in selection["excluded_templates"]},
    )
    expected_counts = {
        str(key): int(value) for key, value in selection["expected_counts"].items()
    }
    if audit_counts != expected_counts:
        raise ValueError(f"FantasyID eligible capacity changed: {audit_counts}")
    expected_templates = {
        str(key): int(value)
        for key, value in selection["expected_template_counts"].items()
    }
    template_counts = Counter(str(row["template"]) for row in eligible.values())
    if dict(sorted(template_counts.items())) != expected_templates:
        raise ValueError(f"FantasyID template counts changed: {template_counts}")

    seed = int(experiment["seed"])
    assigned = _assign_balanced_cells(eligible, methods, devices, seed)
    toy_count = int(selection["toy_groups"])
    pilot_count = int(selection["pilot_groups"])
    if not 0 < toy_count <= pilot_count <= len(assigned):
        raise ValueError("invalid nested stage sizes")
    selected_identity = [
        (str(group["stem"]), method, device)
        for group, method, device in assigned
    ]
    freeze_payload = {
        "protocol_sha256": protocol_sha256,
        "member_records_sha256": _sha256(member_records_path),
        "audit_summary_sha256": _sha256(audit_summary_path),
        "archive_sha256": inputs["expected_archive_sha256"],
        "seed": seed,
        "selection": selection,
        "selected": selected_identity,
    }
    freeze_id = hashlib.sha256(
        json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    output_rows: list[dict[str, Any]] = []
    for index, (group, method, device) in enumerate(assigned, start=1):
        attack = group["variants"][_key(method, device)]
        authentic = group["variants"][_key("none", device)]
        attack_json = str(attack["relative_path"])
        authentic_json = str(authentic["relative_path"])
        output_rows.append(
            {
                "sample_id": f"fantasyid::{group['stem']}::{method}::{device}",
                "source_group_id": f"fantasyid::{group['stem']}",
                "source_dataset": "FantasyID-open-v1",
                "source_card_stem": group["stem"],
                "source_template": group["template"],
                "face_db": group["face_db"],
                "face_id": group["face_id"],
                "attack_method": method,
                "device": device,
                "attack_split": attack["split"],
                "authentic_split": authentic["split"],
                "forged_image_member": f"FantasyID/{PurePosixPath(attack_json).with_suffix('.jpg')}",
                "forged_metadata_member": f"FantasyID/{attack_json}",
                "authentic_image_member": f"FantasyID/{PurePosixPath(authentic_json).with_suffix('.jpg')}",
                "authentic_metadata_member": f"FantasyID/{authentic_json}",
                "annotation_width": int(attack["annotation_width"]),
                "annotation_height": int(attack["annotation_height"]),
                "altered_rectangle_count": int(attack["altered_region_count"]),
                "mask_semantics": "box_mask_not_pixel_accurate",
                "license_boundary": "FantasyID open record CC-BY-4.0; facelab_london only",
                "archive_sha256": inputs["expected_archive_sha256"],
                "selection_seed": seed,
                "selection_index": index,
                "toy3_member": index <= toy_count,
                "pilot20_member": index <= pilot_count,
                "full88_member": True,
                "development_only": True,
                "selection_used_image_or_model_output": False,
                "paper_evidence": False,
                "fantasyid_facelondon_freeze_id": freeze_id,
            }
        )

    cell_counts = Counter(
        f"{row['attack_method']}|{row['device']}" for row in output_rows
    )
    pilot_cell_counts = Counter(
        f"{row['attack_method']}|{row['device']}"
        for row in output_rows
        if row["pilot20_member"]
    )
    if max(cell_counts.values()) - min(cell_counts.values()) > 1:
        raise ValueError("full-88 attack-device assignment is not balanced")
    if max(pilot_cell_counts.values()) - min(pilot_cell_counts.values()) > 1:
        raise ValueError("pilot-20 attack-device assignment is not balanced")

    output_path = _resolve(project_root, config["paths"]["output_manifest"])
    summary_path = _resolve(project_root, config["paths"]["output_summary"])
    _write_jsonl(output_path, output_rows)
    excluded_reason_counts = Counter(
        reason.split(":", 1)[0]
        for reasons in excluded.values()
        for reason in reasons
    )
    summary = {
        "experiment": experiment,
        "status": "fantasyid_facelondon_88_frozen_metadata_only_unread",
        "paper_evidence": False,
        "development_only": True,
        "image_or_model_output_read": False,
        "final_reserve_read": False,
        "original_confirmatory_gate_reopened": False,
        "freeze_id": freeze_id,
        "protocol_sha256": protocol_sha256,
        "input": {
            "member_records": {
                "path": str(member_records_path.relative_to(project_root)),
                "sha256": _sha256(member_records_path),
            },
            "audit_summary": {
                "path": str(audit_summary_path.relative_to(project_root)),
                "sha256": _sha256(audit_summary_path),
            },
            "archive_sha256": inputs["expected_archive_sha256"],
        },
        "eligibility": {
            **audit_counts,
            "eligible_template_counts": dict(sorted(template_counts.items())),
            "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
            "excluded_groups": excluded,
        },
        "selection": {
            "seed": seed,
            "groups": len(output_rows),
            "toy_groups": sum(row["toy3_member"] for row in output_rows),
            "pilot_groups": sum(row["pilot20_member"] for row in output_rows),
            "attack_counts": dict(
                sorted(Counter(row["attack_method"] for row in output_rows).items())
            ),
            "device_counts": dict(
                sorted(Counter(row["device"] for row in output_rows).items())
            ),
            "attack_device_counts": dict(sorted(cell_counts.items())),
            "pilot_attack_device_counts": dict(sorted(pilot_cell_counts.items())),
            "selection_used_image_or_model_output": False,
        },
        "output": {
            "path": str(output_path.relative_to(project_root)),
            "sha256": _sha256(output_path),
        },
        "runtime": runtime,
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

