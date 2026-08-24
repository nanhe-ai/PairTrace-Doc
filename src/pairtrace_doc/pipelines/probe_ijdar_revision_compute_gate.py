from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_experiment_config(config_path: Path, project_root: Path) -> dict[str, Any]:
    override = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_value = override.pop("base_config", None)
    override.pop("expected_base_config_sha256", None)
    if base_value is None:
        return override
    base_path = _resolve(project_root, str(base_value))
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    return _deep_merge(base, override)


def _audit_training_artifacts(
    config_path: Path, project_root: Path
) -> tuple[str, list[str]]:
    config = _load_experiment_config(config_path, project_root)
    paths = {
        key: _resolve(project_root, str(config["paths"][key]))
        for key in (
            "checkpoint",
            "epoch_log",
            "summary",
            "log",
            "prediction_records",
            "metrics",
        )
    }
    existing = [key for key, path in paths.items() if path.is_file()]
    if not existing:
        return "pending", []
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        return "invalid", [f"missing_{key}" for key in missing]
    reasons: list[str] = []
    try:
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", [f"summary_decode_failed:{type(exc).__name__}"]
    expected_steps = int(
        config["training"].get(
            "expected_optimizer_steps",
            int(config["training"]["epochs"])
            * int(config["training"]["steps_per_epoch"]),
        )
    )
    checks = {
        "status_not_complete": summary.get("status")
        != "resampling_robust_teacher_training_complete",
        "config_hash_mismatch": summary.get("config_sha256") != _sha256(config_path),
        "checkpoint_hash_mismatch": summary.get("checkpoint_sha256")
        != _sha256(paths["checkpoint"]),
        "optimizer_steps_mismatch": summary.get("optimizer_steps_completed")
        != expected_steps,
        "silent_failures_nonzero": summary.get("silent_failures") != 0,
        "paper_evidence_true": summary.get("paper_evidence") is not False,
        "evaluation_boundary_open": any(
            summary.get(key) is not False
            for key in (
                "holdout_read",
                "viewed_development_read",
                "unseen_development_read",
                "final_reserve_read",
            )
        ),
        "epoch_log_hash_mismatch": summary.get("outputs", {}).get(
            "epoch_log_sha256"
        )
        != _sha256(paths["epoch_log"]),
        "prediction_hash_mismatch": summary.get("outputs", {}).get(
            "prediction_records_sha256"
        )
        != _sha256(paths["prediction_records"]),
        "metrics_hash_mismatch": summary.get("outputs", {}).get("metrics_sha256")
        != _sha256(paths["metrics"]),
    }
    reasons.extend(name for name, failed in checks.items() if failed)
    return ("invalid", reasons) if reasons else ("complete", [])


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if bool(config["experiment"]["training_authorized"]):
        raise ValueError("compute probe cannot authorize training")
    families = []
    completed_training_configs = 0
    pending_training_configs = 0
    invalid_training_configs = 0
    for family in config["training_families"]:
        matches = sorted(
            Path(path).resolve()
            for path in glob.glob(str(project_root / str(family["config_pattern"])))
            if str(family["exclude_substring"]) not in Path(path).name
        )
        if len(matches) != int(family["expected_configs"]):
            raise ValueError(
                f"training config count changed for {family['name']}: {len(matches)}"
            )
        run_records = []
        for path in matches:
            artifact_status, reasons = _audit_training_artifacts(path, project_root)
            run_records.append(
                {
                    "config": str(path.relative_to(project_root)),
                    "artifact_status": artifact_status,
                    "reasons": reasons,
                }
            )
            completed_training_configs += int(artifact_status == "complete")
            pending_training_configs += int(artifact_status == "pending")
            invalid_training_configs += int(artifact_status == "invalid")
        families.append(
            {
                "name": family["name"],
                "config_count": len(matches),
                "configs": [str(path.relative_to(project_root)) for path in matches],
                "completed_config_count": sum(
                    record["artifact_status"] == "complete" for record in run_records
                ),
                "pending_config_count": sum(
                    record["artifact_status"] == "pending" for record in run_records
                ),
                "invalid_config_count": sum(
                    record["artifact_status"] == "invalid" for record in run_records
                ),
                "runs": run_records,
            }
        )
    device_nodes = sorted(glob.glob("/dev/nvidia*"))
    nvidia_smi = shutil.which("nvidia-smi")
    smi_result: dict[str, Any] = {
        "path": nvidia_smi,
        "executable": bool(nvidia_smi and os.access(nvidia_smi, os.X_OK)),
        "returncode": None,
        "first_output_line": None,
    }
    if smi_result["executable"]:
        completed = subprocess.run(
            [str(nvidia_smi), "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
            timeout=15,
        )
        smi_result["returncode"] = completed.returncode
        lines = (completed.stdout or completed.stderr).splitlines()
        smi_result["first_output_line"] = lines[0] if lines else None
    import torch

    torch_state = {
        "version": torch.__version__,
        "compiled_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    gpu_gate_open = bool(
        device_nodes
        and smi_result["executable"]
        and smi_result["returncode"] == 0
        and torch_state["cuda_available"]
        and torch_state["cuda_device_count"] > 0
    )
    checkpoint_dir = _resolve(project_root, str(config["checkpoint_directory"]))
    existing_revision_checkpoints = (
        sorted(path.name for path in checkpoint_dir.glob("*.pt"))
        if checkpoint_dir.is_dir()
        else []
    )
    planned_training_configs = sum(item["config_count"] for item in families)
    if invalid_training_configs:
        status = "training_artifact_audit_failed"
        next_action = "inspect invalid or partial artifacts before any rerun or evaluation"
    elif pending_training_configs == 0:
        status = "training_artifacts_complete"
        next_action = (
            "preserve validation-only artifacts; wait for the separately frozen "
            "new-evaluation source and license gate before confirmatory scoring"
        )
    elif gpu_gate_open:
        status = "gpu_ready"
        next_action = "run preflight configs before any remaining full arm"
    else:
        status = "blocked_gpu_unavailable"
        next_action = (
            "restore a visible CUDA device and working nvidia-smi, then rerun this probe"
        )
    summary = {
        "status": status,
        "paper_evidence": False,
        "training_started_by_probe": False,
        "gpu_gate_open": gpu_gate_open,
        "device_nodes": device_nodes,
        "nvidia_smi": smi_result,
        "torch": torch_state,
        "training_families": families,
        "planned_training_configs": planned_training_configs,
        "completed_training_configs": completed_training_configs,
        "pending_training_configs": pending_training_configs,
        "invalid_training_configs": invalid_training_configs,
        "revision_checkpoint_directory": str(checkpoint_dir),
        "existing_revision_checkpoints": existing_revision_checkpoints,
        "next_action": next_action,
    }
    output = _resolve(project_root, str(config["paths"]["summary"]))
    _write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
