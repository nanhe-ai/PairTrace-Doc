from __future__ import annotations

import argparse
import hashlib
import json
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps


SUPPORTED_EDITORS = {
    "omnigen2",
    "qwen_image_edit_2511",
    "step1x_edit_v1p2",
}
DOWNLOAD_CACHE_PARTS = frozenset({".cache", ".huggingface"})


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


def _seed(global_seed: int, source_group: str, editor_id: str, attempt: int) -> int:
    payload = f"{global_seed}|{source_group}|{editor_id}|{attempt}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


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


def _inventory(root: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or DOWNLOAD_CACHE_PARTS.intersection(
            path.relative_to(root).parts
        ):
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen instruction-editor Toy-3 run")
    torch.cuda.init()
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)


def _validate_code_revisions(
    code_dirs: list[Path], expected: list[dict[str, str]]
) -> None:
    observed_by_name = {path.name: path for path in code_dirs}
    if set(observed_by_name) != {str(row["directory_name"]) for row in expected}:
        raise ValueError("runtime source directories differ from the run configuration")
    for row in expected:
        directory = observed_by_name[str(row["directory_name"])]
        revision = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != str(row["revision"]):
            raise ValueError(f"runtime source revision changed: {directory}")


def _load_pipeline(
    editor_id: str,
    model_dir: Path,
    code_dirs: list[Path],
    parameters: dict[str, Any],
) -> tuple[Any, Any | None]:
    if editor_id == "omnigen2":
        omni_source = next(path for path in code_dirs if path.name == "omnigen2")
        sys.path.insert(0, str(omni_source))
        from omnigen2.models.transformers.transformer_omnigen2 import (
            OmniGen2Transformer2DModel,
        )
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

        pipe = OmniGen2Pipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )
        pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            str(model_dir),
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        return pipe, None
    if editor_id == "qwen_image_edit_2511":
        from diffusers import QwenImageEditPlusPipeline

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        return pipe, None
    if editor_id == "step1x_edit_v1p2":
        from diffusers import Step1XEditPipelineV1P2
        from RegionE import RegionEHelper

        pipe = Step1XEditPipelineV1P2.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        if str(parameters["region_e"]) == "enabled_default_parameters":
            region_e = RegionEHelper(pipe)
            region_e.set_params()
            region_e.enable()
            return pipe, region_e
        if str(parameters["region_e"]).startswith("disabled_"):
            return pipe, None
        raise ValueError("unknown frozen Step1X RegionE mode")
    raise ValueError(f"unsupported editor: {editor_id}")


def _metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    return str(value)


def _raw_edit(
    editor_id: str,
    pipe: Any,
    context: Image.Image,
    prompt: str,
    seed: int,
    parameters: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    with torch.inference_mode():
        if editor_id == "omnigen2":
            output = pipe(
                prompt=prompt,
                input_images=[context],
                num_inference_steps=int(parameters["num_inference_steps"]),
                text_guidance_scale=float(parameters["text_guidance_scale"]),
                image_guidance_scale=float(parameters["image_guidance_scale"]),
                max_pixels=int(parameters["max_pixels"]),
                max_input_image_side_length=int(
                    parameters["max_input_image_side_length"]
                ),
                num_images_per_prompt=int(parameters["num_images_per_prompt"]),
                generator=torch.Generator("cuda").manual_seed(seed),
                output_type="pil",
            )
            return output.images[0], {}
        if editor_id == "qwen_image_edit_2511":
            output = pipe(
                image=[context],
                prompt=prompt,
                negative_prompt=str(parameters["negative_prompt"]),
                num_inference_steps=int(parameters["num_inference_steps"]),
                true_cfg_scale=float(parameters["true_cfg_scale"]),
                guidance_scale=float(parameters["guidance_scale"]),
                num_images_per_prompt=int(parameters["num_images_per_prompt"]),
                generator=torch.Generator().manual_seed(seed),
            )
            return output.images[0], {}
        if editor_id == "step1x_edit_v1p2":
            output = pipe(
                image=context,
                prompt=prompt,
                num_inference_steps=int(parameters["num_inference_steps"]),
                true_cfg_scale=float(parameters["true_cfg_scale"]),
                num_images_per_prompt=int(parameters["num_images_per_prompt"]),
                generator=torch.Generator().manual_seed(seed),
                enable_thinking_mode=bool(parameters["enable_thinking_mode"]),
                enable_reflection_mode=bool(parameters["enable_reflection_mode"]),
            )
            metadata = {
                key: _metadata_value(getattr(output, key, None))
                for key in ("reformat_prompt", "think_info", "best_info")
            }
            return output.final_images[0], metadata
    raise ValueError(f"unsupported editor: {editor_id}")


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
    edited_context = Image.composite(raw.convert("RGB"), context, mask)
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
    code_dirs: list[Path],
) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    storage_root = storage_root.resolve()
    model_dir = model_dir.resolve()
    code_dirs = [path.resolve() for path in code_dirs]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    editor_id = str(config["editor"]["id"])
    if editor_id not in SUPPORTED_EDITORS:
        raise ValueError(f"unsupported editor: {editor_id}")
    expected_cublas = str(config["environment"]["cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected_cublas:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG differs from the run configuration")
    for field, expected in config["frozen_inputs"].items():
        if not field.endswith("_sha256"):
            continue
        path = _resolve(
            project_root, str(config["frozen_inputs"][field.removesuffix("_sha256")])
        )
        if _sha256(path) != str(expected):
            raise ValueError(f"frozen input changed: {path}")
    _validate_code_revisions(code_dirs, config["editor"]["code_revisions"])

    inventory, model_bytes = _inventory(model_dir)
    if len(inventory) != int(config["editor"]["expected_model_files"]):
        raise ValueError(f"unexpected model file count: {len(inventory)}")
    if model_bytes != int(config["editor"]["expected_model_bytes"]):
        raise ValueError(f"unexpected model bytes: {model_bytes}")
    inventory_path = _resolve(project_root, str(config["outputs"]["model_inventory"]))
    _write_json(inventory_path, inventory)
    inventory_sha256 = _sha256(inventory_path)

    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    ).stdout.splitlines()
    normalized_packages = sorted(packages, key=str.casefold)
    resolved_lock_path = _resolve(
        project_root, str(config["environment"]["resolved_lock"])
    )
    if _sha256(resolved_lock_path) != str(
        config["environment"]["resolved_lock_sha256"]
    ):
        raise ValueError("resolved environment lock changed")
    if normalized_packages != resolved_lock_path.read_text(encoding="utf-8").splitlines():
        raise ValueError("runtime differs from the resolved environment lock")
    determinism = _configure_determinism()
    environment_path = _resolve(project_root, str(config["outputs"]["environment_lock"]))
    _write_json(
        environment_path,
        {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda": torch.version.cuda,
            "determinism": determinism,
            "executable": sys.executable,
            "packages": normalized_packages,
            "torch": torch.__version__,
        },
    )
    environment_sha256 = _sha256(environment_path)

    target_path = _resolve(project_root, str(config["frozen_inputs"]["target_manifest"]))
    tasks = [row for row in _read_jsonl(target_path) if editor_id in row["editor_ids"]]
    selected_rehearsal_ids = config["retry"].get("selected_rehearsal_ids")
    if selected_rehearsal_ids is not None:
        selected = {str(value) for value in selected_rehearsal_ids}
        tasks = [row for row in tasks if str(row["rehearsal_id"]) in selected]
    if len(tasks) != int(config["editor"]["expected_toy_calls"]):
        raise ValueError(f"unexpected Toy-3 task count: {len(tasks)}")
    if any(bool(row.get("source_was_selected_for_final")) for row in tasks):
        raise ValueError("final-source member reached non-final Toy-3 runner")
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
        raise ValueError("invalid attempt_indices")

    _initialize_cuda_memory_tracking(0)
    load_started = time.monotonic()
    pipe, accelerator_helper = _load_pipeline(
        editor_id, model_dir, code_dirs, parameters
    )
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - load_started

    attempts: list[dict[str, Any]] = []
    accepted = 0
    output_root = _resolve(storage_root, str(config["outputs"]["artifact_root"]))
    attempts_path = _resolve(project_root, str(config["outputs"]["attempts"]))
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

        task_accepted = False
        for attempt_index in attempt_indices:
            seed = _seed(
                int(config["experiment"]["seed"]),
                str(row["source_group_key"]),
                editor_id,
                attempt_index,
            )
            cache_binding = {
                "code_revisions": config["editor"]["code_revisions"],
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
                "code_revisions": config["editor"]["code_revisions"],
                "editor_id": editor_id,
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
                raw, response_metadata = _raw_edit(
                    editor_id, pipe, context, str(row["prompt"]), seed, parameters
                )
                candidate, outside, inside, fraction = _patchback(
                    row, storage_root, raw
                )
                attempt_dir = output_root / str(row["artifact_id"]) / f"attempt_{attempt_index}"
                raw_path = attempt_dir / "raw_context.png"
                candidate_path = attempt_dir / "candidate_full.png"
                metadata_path = attempt_dir / "response_metadata.json"
                _save_png(raw_path, raw.convert("RGB"))
                _save_png(candidate_path, candidate)
                _write_json(metadata_path, response_metadata)
                accepted_attempt = outside == 0 and fraction >= float(
                    config["retry"]["minimum_changed_fraction_inside_mask"]
                )
                failure_reason = None
                if outside != 0:
                    failure_reason = "outside_mask_change_nonzero"
                elif not accepted_attempt:
                    failure_reason = "inside_change_below_frozen_minimum"
                record.update(
                    {
                        "accepted_automated_gate": accepted_attempt,
                        "candidate_decoded": True,
                        "candidate_dimensions": list(candidate.size),
                        "candidate_finite": bool(
                            np.isfinite(np.asarray(candidate, dtype=np.float32)).all()
                        ),
                        "candidate_path": candidate_path.relative_to(storage_root).as_posix(),
                        "candidate_sha256": _sha256(candidate_path),
                        "changed_fraction_inside_mask": round(fraction, 8),
                        "changed_pixels_inside_mask": inside,
                        "failure_reason": failure_reason,
                        "outside_mask_changed_pixels": outside,
                        "raw_context_path": raw_path.relative_to(storage_root).as_posix(),
                        "raw_context_sha256": _sha256(raw_path),
                        "response_metadata_path": metadata_path.relative_to(storage_root).as_posix(),
                        "response_metadata_sha256": _sha256(metadata_path),
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
            record["peak_vram_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(0)
            )
            record["wall_seconds"] = round(time.monotonic() - attempt_started, 6)
            attempts.append(record)
            _write_jsonl(attempts_path, attempts)
            if task_accepted:
                accepted += 1
                break

    if accelerator_helper is not None:
        accelerator_helper.disable()
    result = {
        "accepted_tasks": accepted,
        "attempt_records": len(attempts),
        "attempts": str(attempts_path.relative_to(project_root)),
        "attempts_sha256": _sha256(attempts_path),
        "authorization": {
            "detector_inference_run": False,
            "final_source_images_read": False,
            "nonfinal_toy_editor_calls": len(tasks),
        },
        "determinism": determinism,
        "editor_id": editor_id,
        "environment_lock": str(environment_path.relative_to(project_root)),
        "environment_lock_sha256": environment_sha256,
        "expected_tasks": len(tasks),
        "executed_attempt_indices": attempt_indices,
        "load_seconds": round(load_seconds, 6),
        "model_bytes": model_bytes,
        "model_files": len(inventory),
        "model_inventory": str(inventory_path.relative_to(project_root)),
        "model_inventory_sha256": inventory_sha256,
        "model_revision": config["editor"]["model_revision"],
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "status": "passed" if accepted == len(tasks) else "failed",
    }
    report_path = _resolve(project_root, str(config["outputs"]["report"]))
    _write_json(report_path, result)
    result["report_path"] = str(report_path.relative_to(project_root))
    result["report_sha256"] = _sha256(report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one frozen non-final Toy-3 instruction editor."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, action="append", required=True)
    args = parser.parse_args()
    result = run(
        args.config,
        args.project_root,
        args.storage_root,
        args.model_dir,
        args.code_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
