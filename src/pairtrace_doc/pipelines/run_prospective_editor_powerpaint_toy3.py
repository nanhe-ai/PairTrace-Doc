from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _invalid_thread_value = int(os.environ.get(_thread_variable, "1")) < 1
    except ValueError:
        _invalid_thread_value = True
    if _invalid_thread_value:
        os.environ[_thread_variable] = "1"

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps


EDITOR_ID = "powerpaint_v2_1"


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pixel_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _configure_determinism() -> dict[str, Any]:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def _initialize_cuda_memory_tracking(device_index: int = 0) -> None:
    """Create the CUDA context before resetting allocator peak statistics."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen PowerPaint Toy-3 run")
    torch.cuda.init()
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", optimize=False)
    temporary.replace(path)


def _seed(global_seed: int, source_group: str, attempt: int) -> int:
    payload = f"{global_seed}|{source_group}|{EDITOR_ID}|{attempt}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def _inventory(root: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        size = path.stat().st_size
        total += size
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": _sha256(path),
            }
        )
    return rows, total


def _add_task(prompt: str) -> tuple[str, str, str, str]:
    return " P_obj", " P_obj", "P_obj", "P_obj"


def _load_pipeline(model_dir: Path, code_dir: Path, dtype: torch.dtype):
    sys.path.insert(0, str(code_dir))
    from diffusers import UniPCMultistepScheduler
    from powerpaint.models.BrushNet_CA import BrushNetModel
    from powerpaint.models.unet_2d_condition import UNet2DConditionModel
    from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import (
        StableDiffusionPowerPaintBrushNetPipeline,
    )
    from powerpaint.utils.utils import TokenizerWrapper, add_tokens
    from safetensors.torch import load_model
    from transformers import CLIPTextModel

    base_model = model_dir / "realisticVisionV60B1_v51VAE"
    unet = UNet2DConditionModel.from_pretrained(
        base_model, subfolder="unet", torch_dtype=dtype, local_files_only=True
    )
    text_encoder_brushnet = CLIPTextModel.from_pretrained(
        base_model,
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=True,
    )
    brushnet = BrushNetModel.from_unet(unet)
    pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
        base_model,
        brushnet=brushnet,
        text_encoder_brushnet=text_encoder_brushnet,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        safety_checker=None,
        local_files_only=True,
    )
    pipe.unet = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder="unet",
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.tokenizer = TokenizerWrapper(
        from_pretrained=base_model,
        subfolder="tokenizer",
        local_files_only=True,
    )
    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder_brushnet,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"],
        num_vectors_per_token=10,
    )
    load_model(
        pipe.brushnet,
        model_dir / "PowerPaint_Brushnet" / "diffusion_pytorch_model.safetensors",
    )
    state = torch.load(
        model_dir / "PowerPaint_Brushnet" / "pytorch_model.bin",
        map_location="cpu",
    )
    pipe.text_encoder_brushnet.load_state_dict(state, strict=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _raw_edit(
    pipe,
    context: Image.Image,
    mask: Image.Image,
    prompt: str,
    seed: int,
    parameters: dict[str, Any],
) -> Image.Image:
    original_width, original_height = context.size
    if original_width < original_height:
        resized_width = 640
        resized_height = int(original_height / original_width * 640)
    else:
        resized_width = int(original_width / original_height * 640)
        resized_height = 640
    context = context.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    mask = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    height = context.height - context.height % 8
    width = context.width - context.width % 8
    context = context.resize((width, height), Image.Resampling.LANCZOS)
    mask = mask.resize((width, height), Image.Resampling.NEAREST)
    prompt_a, prompt_b, negative_a, negative_b = _add_task(prompt)
    masked = np.asarray(context, dtype=np.uint8) * (
        1 - (np.asarray(mask.convert("L"), dtype=np.uint8) > 0)[..., None]
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    result = pipe(
        promptA=prompt_a,
        promptB=prompt_b,
        promptU=prompt,
        tradoff=float(parameters["fitting_degree"]),
        tradoff_nag=float(parameters["fitting_degree"]),
        image=Image.fromarray(masked.astype(np.uint8), mode="RGB"),
        mask=mask.convert("RGB"),
        num_inference_steps=int(parameters["num_inference_steps"]),
        generator=torch.Generator("cuda").manual_seed(seed),
        brushnet_conditioning_scale=1.0,
        negative_promptA=negative_a,
        negative_promptB=negative_b,
        negative_promptU="",
        guidance_scale=float(parameters["guidance_scale"]),
        width=width,
        height=height,
    ).images[0]
    return result.resize((original_width, original_height), Image.Resampling.BICUBIC)


def _patchback(
    row: dict[str, Any], storage_root: Path, raw: Image.Image
) -> tuple[Image.Image, int, int, float]:
    source_path = _resolve(storage_root, str(row["path"]))
    context_path = _resolve(storage_root, str(row["context_path"]))
    mask_path = _resolve(storage_root, str(row["mask_path"]))
    with Image.open(source_path) as handle:
        source = ImageOps.exif_transpose(handle).convert("RGB")
    with Image.open(context_path) as handle:
        context = handle.convert("RGB")
    with Image.open(mask_path) as handle:
        mask = handle.convert("L")
    if raw.size != context.size:
        raw = raw.resize(context.size, Image.Resampling.BICUBIC)
    edited_context = Image.composite(raw, context, mask)
    candidate = source.copy()
    box = tuple(int(value) for value in row["context_box_xyxy"])
    candidate.paste(edited_context, (box[0], box[1]))

    source_array = np.asarray(source)
    candidate_array = np.asarray(candidate)
    full_mask = np.zeros(source_array.shape[:2], dtype=bool)
    local_mask = np.asarray(mask) > 0
    full_mask[box[1] : box[3], box[0] : box[2]] = local_mask
    changed = np.any(source_array != candidate_array, axis=2)
    outside = int(np.count_nonzero(changed & ~full_mask))
    inside = int(np.count_nonzero(changed & full_mask))
    mask_pixels = int(np.count_nonzero(full_mask))
    fraction = float(inside / mask_pixels) if mask_pixels else 0.0
    return candidate, outside, inside, fraction


def run(
    config_path: Path,
    project_root: Path,
    storage_root: Path,
    model_dir: Path,
    code_dir: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    storage_root = storage_root.resolve()
    model_dir = model_dir.resolve()
    code_dir = code_dir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    expected_cublas_workspace = config.get("environment", {}).get(
        "cublas_workspace_config"
    )
    if expected_cublas_workspace is not None and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) != str(expected_cublas_workspace):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG differs from the run configuration")
    for field, expected in config["frozen_inputs"].items():
        if not field.endswith("_sha256"):
            continue
        path = _resolve(
            project_root, str(config["frozen_inputs"][field.removesuffix("_sha256")])
        )
        if _sha256(path) != str(expected):
            raise ValueError(f"frozen input changed: {path}")
    observed_code_revision = subprocess.run(
        ["git", "-C", str(code_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_code_revision != str(config["editor"]["code_revision"]):
        raise ValueError("PowerPaint code revision changed")

    inventory, model_bytes = _inventory(model_dir)
    if len(inventory) != int(config["editor"]["expected_model_files"]):
        raise ValueError(f"unexpected PowerPaint model file count: {len(inventory)}")
    if model_bytes != int(config["editor"]["expected_model_bytes"]):
        raise ValueError(f"unexpected PowerPaint model bytes: {model_bytes}")
    inventory_path = _resolve(project_root, str(config["outputs"]["model_inventory"]))
    _write_json(inventory_path, inventory)
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    resolved_lock_path = _resolve(
        project_root, str(config["environment"]["resolved_lock"])
    )
    if _sha256(resolved_lock_path) != str(
        config["environment"]["resolved_lock_sha256"]
    ):
        raise ValueError("PowerPaint resolved environment lock changed")
    if packages != resolved_lock_path.read_text(encoding="utf-8").splitlines():
        raise ValueError("PowerPaint runtime differs from resolved environment lock")
    environment_path = _resolve(project_root, str(config["outputs"]["environment_lock"]))
    determinism = _configure_determinism()
    _write_json(
        environment_path,
        {
            "executable": sys.executable,
            "packages": sorted(packages),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "determinism": determinism,
        },
    )
    environment_sha256 = _sha256(environment_path)
    inventory_sha256 = _sha256(inventory_path)

    target_path = _resolve(project_root, str(config["frozen_inputs"]["target_manifest"]))
    tasks = [row for row in _read_jsonl(target_path) if EDITOR_ID in row["editor_ids"]]
    selected_rehearsal_ids = config["retry"].get("selected_rehearsal_ids")
    if selected_rehearsal_ids is not None:
        selected = {str(value) for value in selected_rehearsal_ids}
        tasks = [row for row in tasks if str(row["rehearsal_id"]) in selected]
    if len(tasks) != int(config["editor"]["expected_toy_calls"]):
        raise ValueError(f"unexpected PowerPaint toy task count: {len(tasks)}")
    parameters = config["editor"]["parameters"]
    inference_parameters_sha256 = _canonical_sha256(parameters)
    attempt_indices = [
        int(value)
        for value in config["retry"].get(
            "attempt_indices", range(int(config["retry"]["maximum_attempts"]))
        )
    ]
    if (
        not attempt_indices
        or attempt_indices != sorted(set(attempt_indices))
        or min(attempt_indices) < 0
        or max(attempt_indices) >= int(config["retry"]["maximum_attempts"])
    ):
        raise ValueError("invalid PowerPaint attempt_indices")
    _initialize_cuda_memory_tracking(0)
    load_started = time.monotonic()
    pipe = _load_pipeline(model_dir, code_dir, torch.float16)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - load_started

    attempts: list[dict[str, Any]] = []
    accepted = 0
    output_root = _resolve(storage_root, str(config["outputs"]["artifact_root"]))
    for row in tasks:
        source_path = _resolve(storage_root, str(row["path"]))
        context_path = _resolve(storage_root, str(row["context_path"]))
        mask_path = _resolve(storage_root, str(row["mask_path"]))
        if _sha256(source_path) != str(row["encoded_sha256"]):
            raise ValueError("source encoded hash changed")
        with Image.open(source_path) as handle:
            source_pixels = np.asarray(ImageOps.exif_transpose(handle).convert("RGB"))
        if _pixel_sha256(source_pixels) != str(row["decoded_pixel_sha256"]):
            raise ValueError("source decoded-pixel hash changed")
        if (int(source_pixels.shape[1]), int(source_pixels.shape[0])) != (
            int(row["width"]),
            int(row["height"]),
        ):
            raise ValueError("source dimensions changed")
        if _sha256(context_path) != str(row["context_sha256"]):
            raise ValueError("context hash changed")
        if _sha256(mask_path) != str(row["mask_sha256"]):
            raise ValueError("mask hash changed")
        with Image.open(context_path) as handle:
            context = handle.convert("RGB")
        with Image.open(mask_path) as handle:
            mask = handle.convert("L")
        task_accepted = False
        for attempt_index in attempt_indices:
            seed = _seed(int(config["experiment"]["seed"]), row["source_group_key"], attempt_index)
            cache_binding = {
                "code_revision": config["editor"]["code_revision"],
                "environment_lock_sha256": environment_sha256,
                "inference_parameters_sha256": inference_parameters_sha256,
                "mask_sha256": row["mask_sha256"],
                "model_inventory_sha256": inventory_sha256,
                "model_revision": config["editor"]["model_revision"],
                "prompt_sha256": row["prompt_sha256"],
                "seed": seed,
                "source_decoded_pixel_sha256": row["decoded_pixel_sha256"],
                "source_encoded_sha256": row["encoded_sha256"],
            }
            attempt_started = time.monotonic()
            record: dict[str, Any] = {
                "attempt_index": attempt_index,
                "cache_key": _canonical_sha256(cache_binding),
                "code_revision": config["editor"]["code_revision"],
                "editor_id": EDITOR_ID,
                "environment_lock_sha256": environment_sha256,
                "inference_parameters_sha256": inference_parameters_sha256,
                "mask_sha256": row["mask_sha256"],
                "model_inventory_sha256": inventory_sha256,
                "model_revision": config["editor"]["model_revision"],
                "prompt_sha256": row["prompt_sha256"],
                "rehearsal_id": row["rehearsal_id"],
                "seed": seed,
                "source_decoded_pixel_sha256": row["decoded_pixel_sha256"],
                "source_encoded_sha256": row["encoded_sha256"],
            }
            try:
                raw = _raw_edit(pipe, context, mask, row["prompt"], seed, parameters)
                candidate, outside, inside, fraction = _patchback(row, storage_root, raw)
                attempt_dir = output_root / row["artifact_id"] / f"attempt_{attempt_index}"
                raw_path = attempt_dir / "raw_context.png"
                candidate_path = attempt_dir / "candidate_full.png"
                _save_png(raw_path, raw)
                _save_png(candidate_path, candidate)
                accepted_attempt = (
                    outside == 0
                    and fraction >= float(config["retry"]["minimum_changed_fraction_inside_mask"])
                )
                record.update(
                    {
                        "accepted_automated_gate": accepted_attempt,
                        "candidate_path": candidate_path.relative_to(storage_root).as_posix(),
                        "candidate_sha256": _sha256(candidate_path),
                        "changed_fraction_inside_mask": round(fraction, 8),
                        "changed_pixels_inside_mask": inside,
                        "candidate_dimensions": list(candidate.size),
                        "candidate_finite": bool(
                            np.isfinite(np.asarray(candidate, dtype=np.float32)).all()
                        ),
                        "candidate_decoded": True,
                        "failure_reason": None if accepted_attempt else "inside_change_below_frozen_minimum",
                        "outside_mask_changed_pixels": outside,
                        "raw_context_path": raw_path.relative_to(storage_root).as_posix(),
                        "raw_context_sha256": _sha256(raw_path),
                        "source_dimensions": [int(row["width"]), int(row["height"])],
                        "status": "ok" if accepted_attempt else "quality_gate_failed",
                    }
                )
                if accepted_attempt:
                    task_accepted = True
            except Exception as exc:
                record.update(
                    {
                        "accepted_automated_gate": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "status": "technical_failure",
                    }
                )
            record["wall_seconds"] = round(time.monotonic() - attempt_started, 6)
            record["peak_vram_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
            attempts.append(record)
            if task_accepted:
                accepted += 1
                break

    attempts_path = _resolve(project_root, str(config["outputs"]["attempts"]))
    _write_jsonl(attempts_path, attempts)
    result = {
        "authorization": {
            "final_source_images_read": False,
            "nonfinal_toy_editor_calls": len(tasks),
            "detector_inference_run": False,
        },
        "editor_id": EDITOR_ID,
        "model_revision": config["editor"]["model_revision"],
        "model_files": len(inventory),
        "model_bytes": model_bytes,
        "model_inventory": str(inventory_path.relative_to(project_root)),
        "model_inventory_sha256": inventory_sha256,
        "environment_lock": str(environment_path.relative_to(project_root)),
        "environment_lock_sha256": environment_sha256,
        "determinism": determinism,
        "load_seconds": round(load_seconds, 6),
        "attempt_records": len(attempts),
        "accepted_tasks": accepted,
        "expected_tasks": len(tasks),
        "executed_attempt_indices": attempt_indices,
        "attempts": str(attempts_path.relative_to(project_root)),
        "attempts_sha256": _sha256(attempts_path),
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "status": "passed" if accepted == len(tasks) else "failed",
    }
    report_path = _resolve(project_root, str(config["outputs"]["report"]))
    _write_json(report_path, result)
    result["report_path"] = str(report_path.relative_to(project_root))
    result["report_sha256"] = _sha256(report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen non-final PowerPaint Toy-3 calls.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.config, args.project_root, args.storage_root, args.model_dir, args.code_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
