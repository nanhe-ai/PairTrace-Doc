from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _invalid_thread_value = int(os.environ.get(_thread_variable, "1")) < 1
    except ValueError:
        _invalid_thread_value = True
    if _invalid_thread_value:
        os.environ[_thread_variable] = "1"

import torch
import yaml
from huggingface_hub import HfApi


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one visible GPU, got {rows}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise ValueError(f"unexpected nvidia-smi output: {rows[0]}")
    return {
        "index": int(fields[0]),
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "driver_version": fields[3],
    }


def _synthetic_cuda_probe(device_index: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("torch reports CUDA unavailable")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    generator = torch.Generator(device=device).manual_seed(20260723)
    left = torch.randn((2048, 2048), generator=generator, device=device, dtype=torch.bfloat16)
    right = torch.randn((2048, 2048), generator=generator, device=device, dtype=torch.bfloat16)
    result = left @ right
    checksum = float(result[:16, :16].float().sum().item())
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    del left, right, result
    torch.cuda.empty_cache()
    return {
        "operation": "bfloat16_2048_square_matrix_multiply",
        "checksum": checksum,
        "elapsed_seconds": round(elapsed, 6),
        "peak_allocated_bytes": peak,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _verify_model_metadata(editor: dict[str, Any], api: HfApi) -> dict[str, Any]:
    repository = str(editor["model_repository"])
    revision = str(editor["model_revision"])
    info = api.model_info(repository, revision=revision, files_metadata=True)
    if str(info.sha) != revision:
        raise ValueError(f"model revision mismatch for {repository}: {info.sha}")
    siblings = list(info.siblings or [])
    current_bytes = sum(int(sibling.size or 0) for sibling in siblings)
    declared_bytes = int(editor["current_repository_bytes_at_freeze"])
    if current_bytes != declared_bytes:
        raise ValueError(
            f"model repository bytes changed for {repository}: "
            f"{current_bytes} != {declared_bytes}"
        )
    return {
        "editor_id": str(editor["id"]),
        "repository": repository,
        "revision": revision,
        "gated": bool(info.gated),
        "files": len(siblings),
        "current_bytes": current_bytes,
        "remote_metadata_verified": True,
        "local_model_download_status": "pending_sequential_editor_preflight",
    }


def _verify_code_commit(url: str, revision: str) -> dict[str, Any]:
    base = url.removesuffix(".git").rstrip("/")
    commit_url = f"{base}/commit/{revision}"
    request = urllib.request.Request(commit_url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        status = int(response.status)
    if status != 200:
        raise ValueError(f"code commit unavailable: {commit_url} ({status})")
    return {"repository": base, "revision": revision, "url": commit_url, "status": status}


def run(config_path: Path, project_root: Path, storage_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    storage_root = storage_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    protocol_path = _resolve(project_root, str(config["experiment"]["protocol"]))
    freeze_summary_path = _resolve(
        project_root, str(config["outputs"]["freeze_summary"])
    )
    freeze_summary = json.loads(freeze_summary_path.read_text(encoding="utf-8"))
    if freeze_summary["freeze_payload"]["config_sha256"] != _sha256(config_path):
        raise ValueError("configuration changed after rehearsal freeze")
    if freeze_summary["freeze_payload"]["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("protocol changed after rehearsal freeze")

    gpu = _nvidia_smi()
    device = config["device_and_budget"]
    if gpu["index"] != int(device["gpu_index"]):
        raise ValueError(f"unexpected GPU index: {gpu}")
    if gpu["name"] != str(device["expected_gpu_name"]):
        raise ValueError(f"unexpected GPU name: {gpu}")
    if gpu["memory_total_mib"] < int(device["minimum_memory_mib"]):
        raise ValueError(f"insufficient GPU memory: {gpu}")
    if gpu["driver_version"] != str(device["driver_version"]):
        raise ValueError(f"unexpected GPU driver: {gpu}")

    usage = shutil.disk_usage(storage_root)
    floor_bytes = int(
        float(config["cache_and_storage"]["minimum_free_space_before_model_download_gib"])
        * 1024**3
    )
    largest_model = max(
        int(editor["current_repository_bytes_at_freeze"])
        for editor in config["editors"]
    )
    if usage.free - largest_model < floor_bytes:
        raise ValueError(
            f"largest editor would violate disk floor: free={usage.free}, "
            f"model={largest_model}, floor={floor_bytes}"
        )

    api = HfApi()
    models = [_verify_model_metadata(editor, api) for editor in config["editors"]]
    code_commits = []
    seen: set[tuple[str, str]] = set()
    for editor in config["editors"]:
        for repo_field, revision_field in (
            ("code_repository", "code_revision"),
            ("diffusers_repository", "diffusers_revision"),
        ):
            if repo_field not in editor:
                continue
            identity = (str(editor[repo_field]), str(editor[revision_field]))
            if identity in seen:
                continue
            seen.add(identity)
            code_commits.append(_verify_code_commit(*identity))

    result = {
        "authorization": {
            "source_manifests_read": False,
            "source_images_read": False,
            "editor_models_loaded": False,
            "editor_inference_run": False,
            "detector_inference_run": False,
        },
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(protocol_path),
        "rehearsal_freeze_id": freeze_summary["freeze_id"],
        "gpu": gpu,
        "synthetic_cuda_probe": _synthetic_cuda_probe(gpu["index"]),
        "storage": {
            "root": str(storage_root),
            "free_bytes": usage.free,
            "largest_model_declared_bytes": largest_model,
            "required_floor_bytes": floor_bytes,
            "largest_model_download_preserves_floor": True,
        },
        "model_remote_metadata": models,
        "code_commits": code_commits,
        "status": "hardware_and_remote_metadata_passed_model_download_preflights_pending",
    }
    output_path = _resolve(project_root, str(config["outputs"]["preflight_report"]))
    _write_json(output_path, result)
    result["output_path"] = str(output_path.relative_to(project_root))
    result["output_sha256"] = _sha256(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight prospective local editors without reading source manifests."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.project_root, args.storage_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
