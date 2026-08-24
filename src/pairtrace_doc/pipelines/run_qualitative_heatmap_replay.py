from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import yaml
from PIL import Image

from pairtrace_doc.pipelines.qualitative_asset_io import (
    _archive_members,
    _mask_for_case,
    _read_selected_tar_members,
    _reference_bytes,
)


AUTHORIZATION_PHRASE = "QUALITATIVE_HEATMAP_REPLAY_20260719"


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


class InferenceBackend(Protocol):
    def load_model(
        self,
        checkpoint: Path,
        encoder_weights: Path,
        coefficients: dict[str, float],
        device_name: str,
    ) -> Any: ...

    def infer(
        self,
        model: Any,
        candidate: np.ndarray,
        reference: np.ndarray,
        device_name: str,
        inference: dict[str, Any],
        preprocessing: dict[str, Any],
    ) -> np.ndarray: ...

    def reset_peak_memory(self, device_name: str) -> None: ...

    def peak_memory_bytes(self, device_name: str) -> int: ...

    def release_model(self, model: Any, device_name: str) -> None: ...


class TorchPairBackend:
    """GPU backend imported only after every execution gate has passed."""

    def __init__(self) -> None:
        import torch

        from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
            _infer_pair_tiled,
        )
        from pairtrace_doc.pipelines.train_pairtrace_100 import _load_teacher

        self.torch = torch
        self._infer_pair_tiled = _infer_pair_tiled
        self._load_teacher = _load_teacher

    def validate_device(self, device_name: str) -> None:
        if not device_name.startswith("cuda:"):
            raise ValueError("authorized qualitative replay requires an explicit CUDA device")
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")
        index = int(device_name.split(":", 1)[1])
        if index >= self.torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA device is absent: {device_name}")

    def load_model(
        self,
        checkpoint: Path,
        encoder_weights: Path,
        coefficients: dict[str, float],
        device_name: str,
    ) -> Any:
        model = self._load_teacher(encoder_weights, coefficients)
        saved = self.torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state"], strict=True)
        return model.to(self.torch.device(device_name)).eval().requires_grad_(False)

    def infer(
        self,
        model: Any,
        candidate: np.ndarray,
        reference: np.ndarray,
        device_name: str,
        inference: dict[str, Any],
        preprocessing: dict[str, Any],
    ) -> np.ndarray:
        return self._infer_pair_tiled(
            model,
            candidate,
            reference,
            self.torch.device(device_name),
            inference,
            preprocessing,
        )

    def reset_peak_memory(self, device_name: str) -> None:
        self.torch.cuda.reset_peak_memory_stats(self.torch.device(device_name))

    def peak_memory_bytes(self, device_name: str) -> int:
        return int(
            self.torch.cuda.max_memory_allocated(self.torch.device(device_name))
        )

    def release_model(self, model: Any, device_name: str) -> None:
        model.to("cpu")
        del model
        self.torch.cuda.empty_cache()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_config(config_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    resolved = config_path.resolve()
    project_root = resolved.parent.parent
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("qualitative replay config must be a mapping")
    return resolved, project_root, config


def _validate_evidence_boundary(config: dict[str, Any]) -> None:
    if config["experiment"]["paper_evidence"]:
        raise ValueError("qualitative replay cannot create new paper evidence")
    replay = config["replay"]
    if list(replay["global_probability_scale"]) != [0.0, 1.0]:
        raise ValueError("qualitative replay probability scale changed")
    if replay["score_dtype"] != "float32" or replay["per_image_normalization_allowed"]:
        raise ValueError("qualitative replay score representation changed")
    prohibited_replay = (
        "threshold_search_allowed",
        "metric_computation_allowed",
        "sample_replacement_allowed",
    )
    if any(bool(replay[name]) for name in prohibited_replay):
        raise ValueError("qualitative replay crossed a frozen evidence boundary")
    runtime = config["runtime"]
    prohibited_runtime = (
        "model_training_authorized",
        "checkpoint_selection_authorized",
        "final_reserve_selection_authorized",
        "threshold_selection_authorized",
        "metric_computation_authorized",
        "sample_replacement_authorized",
        "human_audit_completion_authorized",
    )
    if any(bool(runtime[name]) for name in prohibited_runtime):
        raise ValueError("qualitative replay crossed a runtime evidence boundary")
    if runtime["device"] != "cuda:0":
        raise ValueError("frozen qualitative replay device changed")
    expected_phrase_hash = hashlib.sha256(AUTHORIZATION_PHRASE.encode("utf-8")).hexdigest()
    if str(config["authorization"]["required_cli_phrase_sha256"]) != expected_phrase_hash:
        raise ValueError("qualitative replay CLI authorization phrase changed")


def _require_execution_authorization(
    config: dict[str, Any], authorization_phrase: str | None
) -> None:
    authorization = config["authorization"]
    flags = (
        bool(authorization["execution_authorized"]),
        bool(authorization["model_inference_authorized"]),
        bool(authorization["gpu_launch_authorized"]),
    )
    if not all(flags):
        raise PermissionError(
            "GPU replay is not authorized: all three execution flags remain false"
        )
    if authorization_phrase is None:
        raise PermissionError("GPU replay requires the explicit CLI authorization phrase")
    phrase_hash = hashlib.sha256(authorization_phrase.encode("utf-8")).hexdigest()
    if phrase_hash != str(authorization["required_cli_phrase_sha256"]):
        raise PermissionError("GPU replay authorization phrase did not match")


def _scratch_path(project_root: Path, config: dict[str, Any]) -> Path:
    paths = config["paths"]
    return Path(
        os.environ.get(
            str(paths["scratch_env"]),
            str(_resolve(project_root, str(paths["scratch_default"]))),
        )
    ).resolve()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi_not_found", "devices": []}
    command = [
        executable,
        "--query-gpu=index,name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    if result.returncode != 0:
        return {
            "available": False,
            "reason": result.stderr.strip() or f"nvidia-smi_exit_{result.returncode}",
            "devices": [],
        }
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        devices.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "driver_version": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
            }
        )
    return {"available": bool(devices), "reason": None, "devices": devices}


def _verify_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 changed: {path}")
    return actual


def _verify_plan(
    project_root: Path, scratch: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = config["input"]
    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    _verify_hash(
        protocol_path,
        str(config["experiment"]["expected_protocol_sha256"]),
        "qualitative replay protocol",
    )
    preflight_path = _resolve(project_root, inputs["preflight_manifest"])
    _verify_hash(
        preflight_path,
        str(inputs["expected_preflight_manifest_sha256"]),
        "qualitative replay preflight",
    )
    plan = _read_json(preflight_path)
    if plan["status"] != str(inputs["expected_preflight_status"]):
        raise ValueError("qualitative replay preflight status changed")
    if plan["model_inference_performed"] or plan["execution_authorized"]:
        raise ValueError("qualitative replay preflight crossed its execution boundary")
    case_manifest_path = _resolve(project_root, inputs["case_manifest"])
    case_hash = _verify_hash(
        case_manifest_path,
        str(inputs["expected_case_manifest_sha256"]),
        "qualitative case manifest",
    )
    if plan["case_manifest"]["sha256"] != case_hash:
        raise ValueError("preflight and execution case manifests differ")
    cases = _read_json(case_manifest_path)
    expected_cases = int(config["replay"]["expected_case_count"])
    expected_records = int(config["replay"]["expected_record_count"])
    if len(cases["cases"]) != expected_cases or plan["case_count"] != expected_cases:
        raise ValueError("qualitative replay case count changed")
    if len(plan["records"]) != expected_records or plan["record_count"] != expected_records:
        raise ValueError("qualitative replay record count changed")
    if len({record["replay_key"] for record in plan["records"]}) != expected_records:
        raise ValueError("qualitative replay keys are not unique")

    encoder = plan["encoder_weights"]
    encoder_path = _resolve(scratch, encoder["path"])
    _verify_hash(encoder_path, str(encoder["sha256"]), "encoder weights")
    for name, specification in plan["checkpoints"].items():
        checkpoint = _resolve(project_root, specification["path"])
        _verify_hash(checkpoint, str(specification["sha256"]), f"checkpoint {name}")
    for name, specification in plan["sources"].items():
        predictions = _resolve(project_root, specification["predictions"])
        evaluation = _resolve(project_root, specification["evaluation_config"])
        _verify_hash(
            predictions,
            str(specification["predictions_sha256"]),
            f"prediction source {name}",
        )
        _verify_hash(
            evaluation,
            str(specification["evaluation_config_sha256"]),
            f"evaluation config {name}",
        )
    return plan, cases


def _open_rgb_bytes(payload: bytes, label: str) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as handle:
        handle.load()
        image = np.asarray(handle.convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"invalid RGB asset: {label}")
    return image


def _verify_case_assets(
    scratch: Path,
    project_root: Path,
    config: dict[str, Any],
    plan: dict[str, Any],
    cases_document: dict[str, Any],
    retain_arrays: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    inputs = config["input"]
    archive_path = _resolve(scratch, inputs["fantasyid_archive"])
    expected_bytes = int(inputs["expected_fantasyid_archive_bytes"])
    if archive_path.stat().st_size != expected_bytes:
        raise ValueError("FantasyID archive byte size changed")
    archive_hash = _verify_hash(
        archive_path,
        str(inputs["expected_fantasyid_archive_sha256"]),
        "FantasyID archive",
    )
    cases = cases_document["cases"]
    requested = _archive_members(cases)
    payloads = _read_selected_tar_members(archive_path, requested) if requested else {}
    native_shapes: dict[str, tuple[int, int]] = {}
    for record in plan["records"]:
        case_id = str(record["case_id"])
        shape = tuple(int(value) for value in record["native_shape"])
        if case_id in native_shapes and native_shapes[case_id] != shape:
            raise ValueError(f"qualitative native shape differs within case: {case_id}")
        native_shapes[case_id] = shape
    assets: dict[str, dict[str, np.ndarray]] = {}
    verified_fields = 0
    for case in cases:
        case_assets: dict[str, np.ndarray] = {}
        candidate_bytes = _reference_bytes(case["candidate"], scratch, payloads)
        candidate = _open_rgb_bytes(candidate_bytes, f"{case['case_id']}:candidate")
        native_shape = (candidate.shape[0], candidate.shape[1])
        for field in (
            "correct_reference",
            "correct_same_device_reference",
            "selected_reference",
            "wrong_reference",
        ):
            reference = case.get(field)
            if not isinstance(reference, dict):
                continue
            reference_bytes = _reference_bytes(reference, scratch, payloads)
            case_assets[field] = _open_rgb_bytes(
                reference_bytes, f"{case['case_id']}:{field}"
            )
            verified_fields += 1
        _mask_for_case(
            case,
            scratch,
            payloads,
            candidate_size=(candidate.shape[1], candidate.shape[0]),
        )
        if native_shape != native_shapes[str(case["case_id"])]:
            raise ValueError(f"qualitative candidate native shape changed: {case['case_id']}")
        verified_fields += 2
        if retain_arrays:
            case_assets["candidate"] = candidate
            assets[str(case["case_id"])] = case_assets
    return assets, {
        "fantasyid_archive": str(archive_path.relative_to(scratch)),
        "fantasyid_archive_sha256": archive_hash,
        "fantasyid_archive_bytes": expected_bytes,
        "selected_archive_members_read": len(payloads),
        "case_asset_fields_verified": verified_fields,
        "full_archive_extracted": False,
    }


def _resource_estimate(plan: dict[str, Any]) -> dict[str, int]:
    score_bytes = [
        int(record["score_shape"][0]) * int(record["score_shape"][1]) * 4
        for record in plan["records"]
    ]
    return {
        "raw_float32_score_bytes_total": int(sum(score_bytes)),
        "raw_float32_score_bytes_largest_record": int(max(score_bytes)),
        "record_count": len(score_bytes),
        "npz_compressed_bytes_estimate": -1,
    }


def dry_run(config_path: Path, retain_arrays: bool = False) -> dict[str, Any]:
    torch_imported_before = "torch" in sys.modules
    resolved, project_root, config = _load_config(config_path)
    _validate_evidence_boundary(config)
    scratch = _scratch_path(project_root, config)
    plan, cases = _verify_plan(project_root, scratch, config)
    _, asset_summary = _verify_case_assets(
        scratch, project_root, config, plan, cases, retain_arrays=retain_arrays
    )
    free_scratch = shutil.disk_usage(scratch).free
    minimum_free = int(config["replay"]["minimum_free_scratch_bytes"])
    if free_scratch < minimum_free:
        raise RuntimeError(
            f"insufficient scratch space: {free_scratch} < {minimum_free} bytes"
        )
    authorization = config["authorization"]
    nvidia_probe = _probe_nvidia_smi()
    torch_imported_by_runner = not torch_imported_before and "torch" in sys.modules
    if torch_imported_by_runner:
        raise RuntimeError("dry-run unexpectedly imported torch")
    authorization_pending = not all(
        bool(authorization[name])
        for name in (
            "execution_authorized",
            "model_inference_authorized",
            "gpu_launch_authorized",
        )
    )
    if not nvidia_probe["available"]:
        status = "cpu_preflight_passed_gpu_not_visible_execution_not_authorized"
    elif authorization_pending:
        status = "gpu_execution_ready_but_not_authorized"
    else:
        status = "gpu_execution_authorized_pending_explicit_cli_phrase"
    output = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": status,
        "config": str(resolved.relative_to(project_root)),
        "config_sha256": _sha256(resolved),
        "paper_evidence": False,
        "model_inference_performed": False,
        "checkpoint_deserialization_performed": False,
        "torch_imported_by_runner": False,
        "cuda_runtime_initialized_by_runner": False,
        "new_scientific_metrics_computed": False,
        "case_count": plan["case_count"],
        "record_count": plan["record_count"],
        "preflight_manifest_sha256": config["input"][
            "expected_preflight_manifest_sha256"
        ],
        "asset_verification": asset_summary,
        "resource_estimate": _resource_estimate(plan),
        "scratch_free_bytes": int(free_scratch),
        "minimum_free_scratch_bytes": minimum_free,
        "runtime_packages_without_importing_torch": {
            "python": sys.version.split()[0],
            "numpy": _package_version("numpy"),
            "pillow": _package_version("Pillow"),
            "pyyaml": _package_version("PyYAML"),
            "opencv_python": _package_version("opencv-python"),
            "opencv_python_headless": _package_version("opencv-python-headless"),
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
        },
        "nvidia_smi_probe_only": nvidia_probe,
        "authorization": {
            "execution_authorized": bool(authorization["execution_authorized"]),
            "model_inference_authorized": bool(
                authorization["model_inference_authorized"]
            ),
            "gpu_launch_authorized": bool(authorization["gpu_launch_authorized"]),
            "explicit_cli_phrase_required": True,
        },
    }
    output_path = _resolve(project_root, config["paths"]["dry_run_manifest"])
    _write_json(output_path, output)
    return output


def _preparation_mode(record: dict[str, Any]) -> str:
    display = str(record["display_group"])
    source = str(record["source_evaluation_config"])
    if source.endswith("evaluate_resampling_robust_final_reserve.yaml"):
        if display == "shuffled_clean":
            return "final_shuffled_clean"
        if "affine" in display:
            return "final_affine_ecc"
        return "final_clean_ecc"
    if source.endswith("evaluate_reference_integrity_viewed20.yaml"):
        if display == "center_050":
            return "reference_center_050_ecc"
        if display == "wrong_same_dataset":
            return "reference_wrong_ecc"
        return "reference_correct_full_ecc"
    if source.endswith("evaluate_fantasyid_facelondon_full88.yaml"):
        return "fantasy_same_device_correct"
    if source.endswith("evaluate_fantasyid_cross_device_pilot20.yaml"):
        return "fantasy_cross_device_ecc"
    raise ValueError(f"unsupported qualitative replay source: {source}")


def _preparation_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["case_id"]),
        _preparation_mode(record),
        str(record["reference_sha256"]),
    )


def _correct_reference(case_assets: dict[str, np.ndarray]) -> np.ndarray:
    if "correct_reference" in case_assets:
        return case_assets["correct_reference"]
    return case_assets["correct_same_device_reference"]


def _selected_reference(case_assets: dict[str, np.ndarray]) -> np.ndarray:
    if "wrong_reference" in case_assets:
        return case_assets["wrong_reference"]
    return case_assets["selected_reference"]


def _prepare_pair(
    record: dict[str, Any],
    case_assets: dict[str, np.ndarray],
    source_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str | None]:
    # These imports intentionally occur only on the authorized execution path.
    import cv2

    from pairtrace_doc.pipelines.diagnose_pair_at_inference_alignment import (
        _estimate_ecc_alignment,
        _stress_homography,
        _warp_reference,
    )
    from pairtrace_doc.pipelines.evaluate_pair_at_inference_100 import (
        _resize_image,
        _resize_reference,
    )
    from pairtrace_doc.pipelines.evaluate_reference_integrity_viewed20 import (
        _align_reference,
        _center_crop_resized,
    )

    preprocessing = source_config["preprocessing"]
    max_side = int(preprocessing["max_side"])
    candidate = _resize_image(case_assets["candidate"], max_side)
    mode = _preparation_mode(record)
    status: str | None = record.get("alignment_status")
    if mode == "final_shuffled_clean":
        reference = _resize_reference(_selected_reference(case_assets), candidate.shape[:2])
        computed_status = "not_requested"
    elif mode in {"final_clean_ecc", "final_affine_ecc"}:
        reference = _resize_reference(_correct_reference(case_assets), candidate.shape[:2])
        geometry = "affine" if mode == "final_affine_ecc" else "clean"
        stresses = {
            str(specification["name"]): specification
            for specification in source_config["stresses"]
        }
        oracle = _stress_homography(candidate.shape[:2], geometry, stresses)
        stressed = _warp_reference(reference, oracle, inverse=False)
        reference, metadata = _estimate_ecc_alignment(
            candidate, stressed, oracle, source_config["registration"]
        )
        computed_status = str(metadata["alignment_status"])
    elif mode.startswith("reference_"):
        reference = (
            _selected_reference(case_assets)
            if mode == "reference_wrong_ecc"
            else _correct_reference(case_assets)
        )
        reference = _resize_image(reference, max_side)
        if reference.shape != candidate.shape:
            reference = cv2.resize(
                reference,
                (candidate.shape[1], candidate.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        fraction = 0.5 if mode == "reference_center_050_ecc" else 1.0
        reference = _center_crop_resized(reference, fraction)
        reference, _, metadata = _align_reference(
            candidate, reference, source_config["registration"]
        )
        computed_status = str(metadata["alignment_status"])
    elif mode == "fantasy_same_device_correct":
        reference = _resize_reference(_correct_reference(case_assets), candidate.shape[:2])
        computed_status = None
    elif mode == "fantasy_cross_device_ecc":
        reference = _resize_image(_selected_reference(case_assets), max_side)
        if reference.shape != candidate.shape:
            reference = cv2.resize(
                reference,
                (candidate.shape[1], candidate.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        reference, _, metadata = _align_reference(
            candidate, reference, source_config["registration"]
        )
        computed_status = str(metadata["alignment_status"])
    else:
        raise AssertionError(mode)
    if candidate.shape != reference.shape:
        raise ValueError("prepared candidate/reference geometry differs")
    expected_shape = tuple(int(value) for value in record["score_shape"])
    if candidate.shape[:2] != expected_shape:
        raise ValueError(
            f"prepared score shape changed for {record['source_record_id']}: "
            f"{candidate.shape[:2]} != {expected_shape}"
        )
    if computed_status != status:
        raise ValueError(
            f"alignment status changed for {record['source_record_id']}: "
            f"{computed_status!r} != {status!r}"
        )
    return candidate, reference, computed_status


def _validate_probability(probability: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(probability)
    if value.dtype != np.float32:
        raise ValueError(f"qualitative replay output dtype is {value.dtype}, not float32")
    if value.shape != shape:
        raise ValueError(f"qualitative replay output shape changed: {value.shape} != {shape}")
    if not np.isfinite(value).all():
        raise ValueError("qualitative replay output contains non-finite values")
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError("qualitative replay output is outside [0, 1]")
    return value


def _write_score_cache(path: Path, probability: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, scores=probability)
    temporary.replace(path)
    return _sha256(path)


def _read_score_cache(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        probability = _validate_probability(archive["scores"], shape)
    return probability, _sha256(path)


def _result_skeleton(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "case_id",
        "display_group",
        "source_record_id",
        "source_predictions",
        "source_predictions_sha256",
        "source_evaluation_config",
        "source_evaluation_config_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "input_sha256",
        "reference_sha256",
        "mask_sha256",
        "original_score_cache",
        "replay_key",
        "replay_score_cache",
        "native_shape",
        "score_shape",
        "score_dtype",
        "fixed_pixel_threshold",
        "alignment_key",
        "alignment_status",
    )
    return {field: record.get(field) for field in fields}


def preprocess_rehearsal(config_path: Path) -> dict[str, Any]:
    resolved, project_root, config = _load_config(config_path)
    _validate_evidence_boundary(config)
    if not bool(config["runtime"]["cpu_preprocessing_rehearsal_authorized"]):
        raise PermissionError("CPU preprocessing rehearsal is not authorized")
    scratch = _scratch_path(project_root, config)
    plan, cases = _verify_plan(project_root, scratch, config)
    case_assets, asset_summary = _verify_case_assets(
        scratch, project_root, config, plan, cases, retain_arrays=True
    )
    source_configs: dict[str, dict[str, Any]] = {}
    for record in plan["records"]:
        path = str(record["source_evaluation_config"])
        if path not in source_configs:
            with _resolve(project_root, path).open("r", encoding="utf-8") as handle:
                source_configs[path] = yaml.safe_load(handle)
    representatives: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_record_ids: dict[tuple[str, str, str], list[str]] = {}
    for record in plan["records"]:
        key = _preparation_key(record)
        representatives.setdefault(key, record)
        source_record_ids.setdefault(key, []).append(str(record["source_record_id"]))

    rows: list[dict[str, Any]] = []
    for key, record in representatives.items():
        started = time.monotonic()
        row: dict[str, Any] = {
            "case_id": key[0],
            "preparation_mode": key[1],
            "reference_sha256": key[2],
            "frozen_record_count": len(source_record_ids[key]),
            "source_record_ids": source_record_ids[key],
            "expected_score_shape": record["score_shape"],
            "expected_alignment_status": record.get("alignment_status"),
            "status": "failed",
            "failure_reason": None,
        }
        try:
            candidate, reference, alignment_status = _prepare_pair(
                record,
                case_assets[str(record["case_id"])],
                source_configs[str(record["source_evaluation_config"])],
            )
            row.update(
                {
                    "candidate_shape": list(candidate.shape),
                    "reference_shape": list(reference.shape),
                    "alignment_status": alignment_status,
                    "status": "ok",
                }
            )
        except Exception as error:
            row["failure_reason"] = f"{type(error).__name__}: {error}"
        row["wall_time_seconds"] = time.monotonic() - started
        rows.append(row)
    failures = sum(row["status"] != "ok" for row in rows)
    output = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "cpu_preprocessing_rehearsal_passed"
        if failures == 0
        else "cpu_preprocessing_rehearsal_failed",
        "config": str(resolved.relative_to(project_root)),
        "config_sha256": _sha256(resolved),
        "paper_evidence": False,
        "model_inference_performed": False,
        "checkpoint_deserialization_performed": False,
        "cuda_api_called_by_runner": False,
        "new_scientific_metrics_computed": False,
        "case_count": plan["case_count"],
        "record_count": plan["record_count"],
        "unique_preparations": len(rows),
        "successful_preparations": len(rows) - failures,
        "failed_preparations": failures,
        "asset_verification": asset_summary,
        "preparations": rows,
    }
    output_path = _resolve(
        project_root, config["paths"]["preprocess_rehearsal_manifest"]
    )
    _write_json(output_path, output)
    if failures:
        raise RuntimeError(f"CPU preprocessing rehearsal recorded {failures} failures")
    return output


def _run_prepared_records(
    plan: dict[str, Any],
    project_root: Path,
    scratch: Path,
    config: dict[str, Any],
    case_assets: dict[str, dict[str, np.ndarray]],
    backend: InferenceBackend,
    progress_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    source_configs: dict[str, dict[str, Any]] = {}
    for record in plan["records"]:
        path = str(record["source_evaluation_config"])
        if path not in source_configs:
            with _resolve(project_root, path).open("r", encoding="utf-8") as handle:
                source_configs[path] = yaml.safe_load(handle)

    prepared: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, str | None]] = {}
    preparation_errors: dict[tuple[str, str, str], Exception] = {}
    for record in plan["records"]:
        key = _preparation_key(record)
        if key in prepared or key in preparation_errors:
            continue
        try:
            prepared[key] = _prepare_pair(
                record,
                case_assets[str(record["case_id"])],
                source_configs[str(record["source_evaluation_config"])],
            )
        except Exception as error:  # Recorded for every affected frozen item.
            preparation_errors[key] = error

    records_by_checkpoint: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(plan["records"]):
        records_by_checkpoint.setdefault(str(record["checkpoint_name"]), []).append(
            (index, record)
        )
    results_by_index: dict[int, dict[str, Any]] = {}
    device_name = str(config["runtime"]["device"])
    encoder_path = _resolve(scratch, plan["encoder_weights"]["path"])
    for checkpoint_name, indexed_records in records_by_checkpoint.items():
        model: Any | None = None
        checkpoint = _resolve(
            project_root, plan["checkpoints"][checkpoint_name]["path"]
        )
        representative_config = source_configs[
            str(indexed_records[0][1]["source_evaluation_config"])
        ]
        coefficients = representative_config["models"][
            "teacher_conv1_coefficients"
        ]
        model_error: Exception | None = None
        try:
            model = backend.load_model(
                checkpoint, encoder_path, coefficients, device_name
            )
        except Exception as error:
            model_error = error
        for index, record in indexed_records:
            result = _result_skeleton(record)
            result.update(
                {
                    "replay_score_sha256": None,
                    "wall_time_seconds": 0.0,
                    "peak_vram_bytes": 0,
                    "status": "failed",
                    "failure_reason": None,
                    "cache_hit": False,
                    "inference_attempted": False,
                    "paper_evidence": False,
                    "new_scientific_metrics_computed": False,
                }
            )
            prep_key = _preparation_key(record)
            if prep_key in preparation_errors:
                error = preparation_errors[prep_key]
                result["failure_reason"] = f"{type(error).__name__}: {error}"
                results_by_index[index] = result
                if progress_callback is not None:
                    progress_callback(
                        [results_by_index[key] for key in sorted(results_by_index)]
                    )
                continue
            if model_error is not None or model is None:
                error = model_error or RuntimeError("model was not loaded")
                result["failure_reason"] = f"{type(error).__name__}: {error}"
                results_by_index[index] = result
                if progress_callback is not None:
                    progress_callback(
                        [results_by_index[key] for key in sorted(results_by_index)]
                    )
                continue
            candidate, reference, _ = prepared[prep_key]
            score_path = _resolve(scratch, str(record["replay_score_cache"]))
            shape = tuple(int(value) for value in record["score_shape"])
            started = time.monotonic()
            try:
                if score_path.is_file():
                    _, score_hash = _read_score_cache(score_path, shape)
                    result["cache_hit"] = True
                else:
                    backend.reset_peak_memory(device_name)
                    source_config = source_configs[
                        str(record["source_evaluation_config"])
                    ]
                    result["inference_attempted"] = True
                    probability = backend.infer(
                        model,
                        candidate,
                        reference,
                        device_name,
                        source_config["inference"],
                        source_config["preprocessing"],
                    )
                    probability = _validate_probability(probability, shape)
                    score_hash = _write_score_cache(score_path, probability)
                    result["peak_vram_bytes"] = backend.peak_memory_bytes(device_name)
                result.update(
                    {
                        "replay_score_sha256": score_hash,
                        "wall_time_seconds": time.monotonic() - started,
                        "status": "ok",
                        "failure_reason": None,
                    }
                )
            except Exception as error:
                result.update(
                    {
                        "wall_time_seconds": time.monotonic() - started,
                        "failure_reason": f"{type(error).__name__}: {error}",
                    }
                )
            results_by_index[index] = result
            if progress_callback is not None:
                progress_callback(
                    [results_by_index[key] for key in sorted(results_by_index)]
                )
        if model is not None:
            backend.release_model(model, device_name)
            model = None
    return [results_by_index[index] for index in range(len(plan["records"]))]


def execute(
    config_path: Path,
    authorization_phrase: str | None,
    backend: InferenceBackend | None = None,
) -> dict[str, Any]:
    resolved, project_root, config = _load_config(config_path)
    _validate_evidence_boundary(config)
    _require_execution_authorization(config, authorization_phrase)
    for name, value in config["runtime"]["environment"].items():
        os.environ[str(name)] = str(value)
    scratch = _scratch_path(project_root, config)
    plan, cases = _verify_plan(project_root, scratch, config)
    case_assets, asset_summary = _verify_case_assets(
        scratch, project_root, config, plan, cases, retain_arrays=True
    )
    free_scratch = shutil.disk_usage(scratch).free
    minimum_free = int(config["replay"]["minimum_free_scratch_bytes"])
    if free_scratch < minimum_free:
        raise RuntimeError("insufficient scratch space before GPU execution")

    paths = config["paths"]
    predictions_path = _resolve(project_root, paths["predictions"])
    manifest_path = _resolve(project_root, paths["execution_manifest"])
    log_path = _resolve(project_root, paths["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    active_backend = backend or TorchPairBackend()
    if isinstance(active_backend, TorchPairBackend):
        active_backend.validate_device(str(config["runtime"]["device"]))
    started = time.monotonic()
    results = _run_prepared_records(
        plan,
        project_root,
        scratch,
        config,
        case_assets,
        active_backend,
        progress_callback=lambda rows: _write_jsonl(predictions_path, rows),
    )
    _write_jsonl(predictions_path, results)
    failures = sum(result["status"] != "ok" for result in results)
    output = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "qualitative_heatmap_replay_complete"
        if failures == 0
        else "qualitative_heatmap_replay_failed",
        "config": str(resolved.relative_to(project_root)),
        "config_sha256": _sha256(resolved),
        "preflight_manifest_sha256": config["input"][
            "expected_preflight_manifest_sha256"
        ],
        "paper_evidence": False,
        "model_inference_performed": any(
            bool(result["inference_attempted"]) for result in results
        ),
        "new_scientific_metrics_computed": False,
        "human_audit_completion_authorized": False,
        "case_count": plan["case_count"],
        "record_count": len(results),
        "successful_records": len(results) - failures,
        "failed_records": failures,
        "all_records_attempted": len(results) == int(config["replay"]["expected_record_count"]),
        "wall_time_seconds": time.monotonic() - started,
        "asset_verification": asset_summary,
        "resource_estimate": _resource_estimate(plan),
        "predictions": str(predictions_path.relative_to(project_root)),
        "predictions_sha256": _sha256(predictions_path),
        "log": str(log_path.relative_to(project_root)),
    }
    _write_json(manifest_path, output)
    if failures and config["replay"]["require_all_records"]:
        raise RuntimeError(
            f"qualitative heatmap replay recorded {failures} failed frozen records"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--preprocess-rehearsal", action="store_true")
    actions.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    if args.dry_run:
        result = dry_run(args.config)
    elif args.preprocess_rehearsal:
        result = preprocess_rehearsal(args.config)
    else:
        result = execute(args.config, args.authorization_phrase)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
