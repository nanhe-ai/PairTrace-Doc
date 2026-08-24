from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _cgroup_memory_limit_bytes() -> int | None:
    path = Path("/sys/fs/cgroup/memory.max")
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return None if value == "max" else int(value)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config["runtime"]["device"] != "cpu":
        raise ValueError("this diagnostic preflight is frozen to CPU")
    if config["runtime"]["gpu_launch_authorized"]:
        raise ValueError("CPU preflight config must not authorize GPU use")
    if config["runtime"]["method_training_authorized"]:
        raise ValueError("baseline preflight must not authorize method training")

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    repository = _resolve(scratch, paths["repository"]).resolve()
    checkpoint = _resolve(scratch, paths["checkpoint"]).resolve()
    cache_dir = _resolve(project_root, paths["cache_dir"]).resolve()
    prediction_path = _resolve(project_root, paths["output_prediction"]).resolve()
    manifest_path = _resolve(project_root, paths["output_manifest"]).resolve()
    log_path = _resolve(project_root, paths["log"]).resolve()
    input_manifest_path = _resolve(project_root, config["input"]["manifest"]).resolve()
    for path in (cache_dir, prediction_path.parent, manifest_path.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    input_manifest_sha256 = _sha256(input_manifest_path)
    if input_manifest_sha256 != config["input"]["expected_manifest_sha256"]:
        raise ValueError("baseline input manifest SHA-256 changed")
    repository_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if repository_revision != config["baseline"]["revision"]:
        raise ValueError("TruFor repository revision changed")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    candidates = [
        row
        for row in _read_jsonl(input_manifest_path)
        if row.get("evaluation_role") == config["input"]["select_evaluation_role"]
        and row.get("sample_kind") == config["input"]["select_sample_kind"]
    ]
    candidates.sort(key=lambda row: str(row["record_id"]))
    selected = candidates[int(config["input"]["selection_index"])]
    image_path = _resolve(scratch, selected["image"]).resolve()
    if _sha256(image_path) != selected["image_sha256"]:
        raise ValueError("selected preflight image SHA-256 changed")

    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != config["baseline"]["checkpoint_sha256"]:
        raise ValueError("TruFor checkpoint SHA-256 changed")
    if not config["baseline"]["trusted_official_checkpoint_allow_legacy_pickle"]:
        raise ValueError("legacy checkpoint loading was not explicitly authorized")
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "baseline_revision": repository_revision,
                "checkpoint_sha256": checkpoint_sha256,
                "image_sha256": selected["image_sha256"],
                "experiment_config": config["baseline"]["experiment_config"],
                "official_image_divisor": 256.0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    score_path = cache_dir / f"{cache_key}.npz"
    command = [
        sys.executable,
        "test.py",
        "--gpu",
        "-1",
        "--input",
        str(image_path),
        "--output",
        str(score_path),
        "--experiment",
        str(config["baseline"]["experiment_config"]),
        "TEST.MODEL_FILE",
        str(checkpoint),
    ]
    started = time.monotonic()
    cache_hit = score_path.is_file()
    if not cache_hit:
        completed = subprocess.run(
            command,
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=int(config["runtime"]["timeout_seconds"]),
            env={
                **os.environ,
                **{
                    str(name): str(value)
                    for name, value in config["runtime"].get(
                        "environment", {}
                    ).items()
                },
            },
        )
        log_path.write_text(
            "COMMAND\n"
            + json.dumps(command, ensure_ascii=False)
            + "\nSTDOUT\n"
            + completed.stdout
            + "\nSTDERR\n"
            + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            failure_reason = (
                "subprocess_signal_9_possible_cgroup_memory_limit"
                if completed.returncode == -9
                else f"subprocess_returncode_{completed.returncode}"
            )
            prediction = {
                "record_id": selected["record_id"],
                "status": "failed",
                "paper_evidence": False,
                "baseline": config["baseline"]["name"],
                "cache_key": cache_key,
                "failure_reason": failure_reason,
                "subprocess_returncode": completed.returncode,
            }
            prediction_path.write_text(
                json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            failure_manifest = {
                "experiment": config["experiment"],
                "baseline": config["baseline"],
                "status": "failed",
                "paper_evidence": False,
                "device": "cpu",
                "gpu_used": False,
                "failure_reason": failure_reason,
                "subprocess_returncode": completed.returncode,
                "cgroup_memory_limit_bytes": _cgroup_memory_limit_bytes(),
                "repository_revision": repository_revision,
                "checkpoint_sha256": checkpoint_sha256,
                "input_manifest_sha256": input_manifest_sha256,
                "selected_record_id": selected["record_id"],
                "prediction": str(prediction_path.relative_to(project_root)),
                "log": str(log_path.relative_to(project_root)),
            }
            _write_json(manifest_path, failure_manifest)
            raise RuntimeError(
                f"TruFor official inference returned {completed.returncode}; see {log_path}"
            )
    elapsed_seconds = time.monotonic() - started
    if not score_path.is_file():
        raise RuntimeError("TruFor returned without producing the expected score cache")

    with np.load(score_path, allow_pickle=False) as output:
        localization = np.asarray(output["map"])
        confidence = np.asarray(output["conf"])
        image_score = float(np.asarray(output["score"]).reshape(-1)[0])
        reported_size = [int(value) for value in np.asarray(output["imgsize"])]
    finite_output = bool(
        np.isfinite(localization).all()
        and np.isfinite(confidence).all()
        and np.isfinite(image_score)
    )
    expected_shape = [int(selected["height"]), int(selected["width"])]
    shape_matches = list(localization.shape) == expected_shape
    status = "passed" if finite_output and shape_matches else "failed"
    prediction = {
        "record_id": selected["record_id"],
        "status": "ok" if status == "passed" else "failed",
        "paper_evidence": False,
        "baseline": config["baseline"]["name"],
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "score_cache": str(score_path.relative_to(project_root)),
        "localization_shape": list(localization.shape),
        "image_score": image_score,
        "finite_output": finite_output,
    }
    prediction_path.write_text(
        json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "experiment": config["experiment"],
        "baseline": config["baseline"],
        "status": status,
        "paper_evidence": False,
        "device": "cpu",
        "gpu_used": False,
        "repository_revision": repository_revision,
        "checkpoint": str(checkpoint.relative_to(scratch)),
        "checkpoint_sha256": checkpoint_sha256,
        "trusted_official_checkpoint_legacy_pickle": True,
        "subprocess_environment": config["runtime"].get("environment", {}),
        "input_manifest_sha256": input_manifest_sha256,
        "selected_record_id": selected["record_id"],
        "selected_image_sha256": selected["image_sha256"],
        "finite_output": finite_output,
        "localization_shape_matches_native_image": shape_matches,
        "reported_image_size": reported_size,
        "elapsed_seconds": elapsed_seconds,
        "prediction": str(prediction_path.relative_to(project_root)),
        "log": str(log_path.relative_to(project_root)),
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
