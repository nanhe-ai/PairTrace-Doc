from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _worker_path(path: Path, worker_index: int, worker_count: int) -> Path:
    if worker_count == 1:
        return path
    return path.with_name(
        f"{path.stem}.worker-{worker_index:02d}-of-{worker_count:02d}{path.suffix}"
    )


def _worker_identity() -> tuple[int, int]:
    worker_index = int(os.environ.get("PAIRTRACE_OCR_WORKER_INDEX", "0"))
    worker_count = int(os.environ.get("PAIRTRACE_OCR_WORKER_COUNT", "1"))
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("invalid OCR worker index/count")
    return worker_index, worker_count


def _cgroup_memory_limit_bytes() -> int | None:
    path = Path("/sys/fs/cgroup/memory.max")
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return None if value == "max" else int(value)


def _rasterize_text_mask(
    height: int,
    width: int,
    polygons: Iterable[np.ndarray],
    scores: Iterable[float],
    threshold: float,
) -> tuple[np.ndarray, int]:
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    accepted = 0
    for polygon, score in zip(polygons, scores):
        if float(score) <= threshold:
            continue
        points = np.asarray(polygon, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError("PaddleOCR returned a malformed text polygon")
        cv2.fillPoly(mask, [points], 1)
        accepted += 1
    return mask, accepted


def _select_rows(
    rows: list[dict[str, Any]], input_config: dict[str, Any]
) -> list[dict[str, Any]]:
    sample_kinds = set(input_config["sample_kinds"])
    limit = input_config.get("max_records_per_role")
    selected: list[dict[str, Any]] = []
    for role in input_config["evaluation_roles"]:
        candidates = sorted(
            (
                row
                for row in rows
                if row.get("evaluation_role") == role
                and row.get("sample_kind") in sample_kinds
            ),
            key=lambda row: str(row["record_id"]),
        )
        if limit is not None:
            candidates = candidates[: int(limit)]
        selected.extend(candidates)
    return selected


def _cache_key(row: dict[str, Any], config: dict[str, Any], version: str) -> str:
    payload = {
        "image_sha256": row["image_sha256"],
        "model_name": str(config["ocr"]["model_name"]),
        "paddleocr_version": version,
        "score_threshold": float(config["ocr"]["score_threshold"]),
        "adcdnet_revision": config["ocr"]["adcdnet_revision"],
        "device": config["ocr"]["device"],
        "enable_mkldnn": config["ocr"]["enable_mkldnn"],
        "cpu_threads": config["ocr"]["cpu_threads"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_config(config_path: Path) -> tuple[Path, dict[str, Any]]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return project_root, config


def _validate_runtime(config: dict[str, Any]) -> None:
    if config["runtime"]["device"] != "cpu":
        raise ValueError("OCR preflight is frozen to CPU")
    if config["runtime"]["gpu_launch_authorized"]:
        raise ValueError("OCR preflight must not authorize GPU use")
    if config["runtime"]["method_training_authorized"]:
        raise ValueError("OCR preprocessing must not authorize method training")


def _run_worker(config_path: Path) -> dict[str, Any]:
    project_root, config = _load_config(config_path)
    _validate_runtime(config)
    for name, value in config["runtime"].get("environment", {}).items():
        os.environ[str(name)] = str(value)

    try:
        from paddleocr import TextDetection
    except ImportError as error:
        raise RuntimeError(
            "PaddleOCR is required; run this pipeline in the frozen OCR environment"
        ) from error

    paddleocr_version = importlib.metadata.version("paddleocr")
    expected_version = str(config["ocr"]["expected_paddleocr_version"])
    if paddleocr_version != expected_version:
        raise ValueError(
            f"PaddleOCR version changed: {paddleocr_version} != {expected_version}"
        )

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    input_path = _resolve(project_root, config["input"]["manifest"]).resolve()
    cache_dir = _resolve(scratch, paths["cache_dir"]).resolve()
    worker_index, worker_count = _worker_identity()
    output_records = _worker_path(
        _resolve(project_root, paths["output_records"]).resolve(),
        worker_index,
        worker_count,
    )
    output_summary = _worker_path(
        _resolve(project_root, paths["output_summary"]).resolve(),
        worker_index,
        worker_count,
    )
    log_path = _worker_path(
        _resolve(project_root, paths["log"]).resolve(),
        worker_index,
        worker_count,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    input_sha256 = _sha256(input_path)
    if input_sha256 != config["input"]["expected_manifest_sha256"]:
        raise ValueError("baseline input manifest SHA-256 changed")
    selected_all = _select_rows(_read_jsonl(input_path), config["input"])
    selected = selected_all[worker_index::worker_count]
    model_name = str(config["ocr"]["model_name"])
    threshold = float(config["ocr"]["score_threshold"])
    detector = TextDetection(
        model_name=model_name,
        device=str(config["ocr"]["device"]),
        enable_mkldnn=bool(config["ocr"]["enable_mkldnn"]),
        cpu_threads=int(config["ocr"]["cpu_threads"]),
    )

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for row in selected:
        cache_key = _cache_key(row, config, paddleocr_version)
        cache_path = cache_dir / f"{cache_key}.npz"
        record = {
            "record_id": row["record_id"],
            "cache_key": cache_key,
            "cache_path": str(cache_path.relative_to(scratch)),
            "paper_evidence": False,
        }
        try:
            cache_hit = cache_path.is_file()
            if cache_hit:
                with np.load(cache_path, allow_pickle=False) as cached:
                    mask = np.asarray(cached["mask"])
                    polygons = np.asarray(cached["polygons"], dtype=np.float32)
                    scores = np.asarray(cached["scores"])
                    accepted = int(np.asarray(cached["accepted"]).item())
            else:
                image_path = _resolve(scratch, row["image"]).resolve()
                if _sha256(image_path) != row["image_sha256"]:
                    raise ValueError("input image SHA-256 changed")
                with Image.open(image_path) as image_handle:
                    image_width, image_height = image_handle.size
                output = detector.predict(input=str(image_path), batch_size=1)
                result = output[0]
                polygons = np.asarray(result["dt_polys"], dtype=np.float32)
                scores = np.asarray(result["dt_scores"], dtype=np.float32)
                mask, accepted = _rasterize_text_mask(
                    image_height, image_width, polygons, scores, threshold
                )
                np.savez_compressed(
                    cache_path,
                    mask=mask,
                    polygons=polygons,
                    scores=scores,
                    accepted=np.asarray(accepted, dtype=np.int32),
                )
            if list(mask.shape) != [int(row["height"]), int(row["width"])]:
                raise ValueError("OCR mask shape differs from the frozen image shape")
            record.update(
                {
                    "status": "ok",
                    "cache_hit": cache_hit,
                    "detected_polygons": len(polygons),
                    "accepted_polygons": accepted,
                    "positive_pixels": int(mask.sum()),
                    "shape": list(mask.shape),
                }
            )
        except Exception as error:  # record every per-sample failure
            record.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            logging.exception("OCR preprocessing failed record_id=%s", row["record_id"])
            if config["runtime"]["fail_on_sample_error"]:
                records.append(record)
                _write_jsonl(output_records, records)
                raise
        records.append(record)

    _write_jsonl(output_records, records)
    failures = sum(row["status"] != "ok" for row in records)
    summary = {
        "experiment": config["experiment"],
        "status": "passed" if failures == 0 else "failed",
        "paper_evidence": False,
        "gpu_used": False,
        "input_manifest_sha256": input_sha256,
        "paddleocr_version": paddleocr_version,
        "model_name": model_name,
        "score_threshold": threshold,
        "inference_runtime": {
            "device": config["ocr"]["device"],
            "enable_mkldnn": config["ocr"]["enable_mkldnn"],
            "cpu_threads": config["ocr"]["cpu_threads"],
        },
        "selected_records": len(selected),
        "successful_records": len(records) - failures,
        "failed_records": failures,
        "cache_hits": sum(bool(row.get("cache_hit")) for row in records),
        "worker_index": worker_index,
        "worker_count": worker_count,
        "wall_time_seconds": time.monotonic() - started,
        "records": str(output_records.relative_to(project_root)),
        "log": str(log_path.relative_to(project_root)),
    }
    _write_json(output_summary, summary)
    if failures and config["runtime"]["require_all_selected_records"]:
        raise RuntimeError(f"ADCD-Net OCR preflight recorded {failures} failures")
    return summary


def run(config_path: Path) -> dict[str, Any]:
    """Run OCR in an isolated worker so fatal resource exits remain auditable."""

    config_path = config_path.resolve()
    project_root, config = _load_config(config_path)
    _validate_runtime(config)
    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    input_path = _resolve(project_root, config["input"]["manifest"]).resolve()
    output_records = _resolve(project_root, paths["output_records"]).resolve()
    output_summary = _resolve(project_root, paths["output_summary"]).resolve()
    log_path = _resolve(project_root, paths["log"]).resolve()
    for path in (output_records.parent, output_summary.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    input_sha256 = _sha256(input_path)
    if input_sha256 != config["input"]["expected_manifest_sha256"]:
        raise ValueError("baseline input manifest SHA-256 changed")
    selected = _select_rows(_read_jsonl(input_path), config["input"])
    expected_version = str(config["ocr"]["expected_paddleocr_version"])
    # Preserve a virtual environment's symlink path; resolving it would bypass
    # pyvenv.cfg and silently execute against the base interpreter packages.
    worker_python = _resolve(scratch, config["runtime"]["worker_python"])
    if not worker_python.is_file():
        raise FileNotFoundError(worker_python)

    environment = {
        **os.environ,
        **{
            str(name): str(value)
            for name, value in config["runtime"].get("environment", {}).items()
        },
    }
    source_path = str((project_root / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_path, environment.get("PYTHONPATH", ""))
        if value
    )
    command = [
        str(worker_python),
        "-m",
        "pairtrace_doc.pipelines.prepare_adcdnet_ocr",
        "--config",
        str(config_path),
        "--worker",
    ]
    worker_count = int(config["runtime"].get("worker_count", 1))
    if worker_count > 1:
        return _run_parallel_workers(
            command=command,
            worker_count=worker_count,
            environment=environment,
            timeout_seconds=int(config["runtime"].get("timeout_seconds", 1800)),
            project_root=project_root,
            scratch=scratch,
            config=config,
            selected=selected,
            input_sha256=input_sha256,
            expected_version=expected_version,
            output_records=output_records,
            output_summary=output_summary,
            log_path=log_path,
        )
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=int(config["runtime"].get("timeout_seconds", 1800)),
            env=environment,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = None
        stdout = error.stdout or ""
        stderr = error.stderr or ""
    wall_time_seconds = time.monotonic() - started
    log_path.write_text(
        "COMMAND\n"
        + json.dumps(command, ensure_ascii=False)
        + "\nSTDOUT\n"
        + stdout
        + "\nSTDERR\n"
        + stderr,
        encoding="utf-8",
    )

    if returncode == 0:
        if not output_summary.is_file():
            raise RuntimeError("OCR worker succeeded without producing its summary")
        with output_summary.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    killed_for_resources = returncode in {-9, 137}
    if timed_out:
        failure_reason = "subprocess_timeout"
    elif killed_for_resources:
        failure_reason = "worker_signal_9_possible_cgroup_memory_limit"
    else:
        failure_reason = f"worker_returncode_{returncode}"
    error_type = (
        "environment_resource_limit"
        if killed_for_resources
        else "worker_process_failure"
    )
    records = []
    cache_dir = _resolve(scratch, paths["cache_dir"]).resolve()
    for row in selected:
        cache_key = _cache_key(row, config, expected_version)
        records.append(
            {
                "record_id": row["record_id"],
                "status": "failed",
                "paper_evidence": False,
                "cache_key": cache_key,
                "cache_path": str((cache_dir / f"{cache_key}.npz").relative_to(scratch)),
                "error_type": error_type,
                "error": failure_reason,
                "worker_returncode": returncode,
            }
        )
    _write_jsonl(output_records, records)
    summary = {
        "experiment": config["experiment"],
        "status": "failed_environment_resource_limit"
        if killed_for_resources
        else "failed_worker_process",
        "paper_evidence": False,
        "gpu_used": False,
        "input_manifest_sha256": input_sha256,
        "expected_paddleocr_version": expected_version,
        "worker_python": (
            str(worker_python.relative_to(scratch))
            if worker_python.is_relative_to(scratch)
            else str(worker_python)
        ),
        "model_name": str(config["ocr"]["model_name"]),
        "cgroup_memory_limit_bytes": _cgroup_memory_limit_bytes(),
        "selected_records": len(selected),
        "successful_records": 0,
        "failed_records": len(selected),
        "failure_reason": failure_reason,
        "worker_returncode": returncode,
        "alternate_model_substituted": False,
        "wall_time_seconds": wall_time_seconds,
        "records": str(output_records.relative_to(project_root)),
        "log": str(log_path.relative_to(project_root)),
    }
    _write_json(output_summary, summary)
    raise RuntimeError(
        f"ADCD-Net OCR worker failed ({failure_reason}); see {log_path}"
    )


def _run_parallel_workers(
    *,
    command: list[str],
    worker_count: int,
    environment: dict[str, str],
    timeout_seconds: int,
    project_root: Path,
    scratch: Path,
    config: dict[str, Any],
    selected: list[dict[str, Any]],
    input_sha256: str,
    expected_version: str,
    output_records: Path,
    output_summary: Path,
    log_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    processes: list[subprocess.Popen[str]] = []
    for worker_index in range(worker_count):
        worker_environment = {
            **environment,
            "PAIRTRACE_OCR_WORKER_INDEX": str(worker_index),
            "PAIRTRACE_OCR_WORKER_COUNT": str(worker_count),
        }
        processes.append(
            subprocess.Popen(
                command,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=worker_environment,
            )
        )

    outputs: list[tuple[str, str]] = []
    timed_out = False
    deadline = started + timeout_seconds
    try:
        for process in processes:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                stdout, stderr = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                for candidate in processes:
                    if candidate.poll() is None:
                        candidate.kill()
                stdout, stderr = process.communicate()
            outputs.append((stdout, stderr))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()

    log_parts = ["COMMAND", json.dumps(command, ensure_ascii=False)]
    for worker_index, (process, (stdout, stderr)) in enumerate(zip(processes, outputs)):
        log_parts.extend(
            [
                f"WORKER {worker_index} RETURN_CODE {process.returncode}",
                "STDOUT",
                stdout,
                "STDERR",
                stderr,
            ]
        )
    log_path.write_text("\n".join(log_parts), encoding="utf-8")

    record_by_id: dict[str, dict[str, Any]] = {}
    worker_summaries: list[dict[str, Any]] = []
    for worker_index in range(worker_count):
        records_path = _worker_path(output_records, worker_index, worker_count)
        summary_path = _worker_path(output_summary, worker_index, worker_count)
        if records_path.is_file():
            for record in _read_jsonl(records_path):
                record_by_id[str(record["record_id"])] = record
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as handle:
                worker_summaries.append(json.load(handle))

    return_codes = [process.returncode for process in processes]
    for row in selected:
        record_id = str(row["record_id"])
        if record_id in record_by_id:
            continue
        cache_key = _cache_key(row, config, expected_version)
        cache_dir = _resolve(scratch, config["paths"]["cache_dir"]).resolve()
        record_by_id[record_id] = {
            "record_id": record_id,
            "status": "failed",
            "paper_evidence": False,
            "cache_key": cache_key,
            "cache_path": str((cache_dir / f"{cache_key}.npz").relative_to(scratch)),
            "error_type": "worker_process_failure",
            "error": "parallel_worker_timeout" if timed_out else "parallel_worker_failure",
        }
    ordered_records = [record_by_id[str(row["record_id"])] for row in selected]
    _write_jsonl(output_records, ordered_records)
    failures = sum(record.get("status") != "ok" for record in ordered_records)
    wall_time_seconds = time.monotonic() - started
    all_workers_passed = (
        not timed_out
        and all(code == 0 for code in return_codes)
        and len(worker_summaries) == worker_count
        and failures == 0
    )
    summary = {
        "experiment": config["experiment"],
        "status": "passed" if all_workers_passed else "failed_worker_process",
        "paper_evidence": False,
        "gpu_used": False,
        "input_manifest_sha256": input_sha256,
        "paddleocr_version": expected_version,
        "model_name": str(config["ocr"]["model_name"]),
        "score_threshold": float(config["ocr"]["score_threshold"]),
        "inference_runtime": {
            "device": config["ocr"]["device"],
            "enable_mkldnn": config["ocr"]["enable_mkldnn"],
            "cpu_threads_per_worker": config["ocr"]["cpu_threads"],
            "worker_count": worker_count,
        },
        "selected_records": len(selected),
        "successful_records": len(ordered_records) - failures,
        "failed_records": failures,
        "cache_hits": sum(int(item.get("cache_hits", 0)) for item in worker_summaries),
        "worker_return_codes": return_codes,
        "worker_summaries": worker_summaries,
        "wall_time_seconds": wall_time_seconds,
        "records": str(output_records.relative_to(project_root)),
        "log": str(log_path.relative_to(project_root)),
    }
    _write_json(output_summary, summary)
    if not all_workers_passed and config["runtime"]["require_all_selected_records"]:
        raise RuntimeError(f"parallel ADCD-Net OCR workers failed; see {log_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    function = _run_worker if args.worker else run
    print(json.dumps(function(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
