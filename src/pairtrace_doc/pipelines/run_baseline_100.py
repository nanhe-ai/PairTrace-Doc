from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator

import yaml


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_thread_environment(config: dict[str, Any]) -> None:
    configured = config["runtime"].get("environment", {})
    for name, value in configured.items():
        os.environ[str(name)] = str(value)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        try:
            invalid = int(os.environ.get(name, "1")) < 1
        except ValueError:
            invalid = True
        if invalid:
            os.environ[name] = "1"


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _repository_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_prerequisites(project_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for prerequisite in config.get("prerequisites", []):
        path = _resolve(project_root, prerequisite["path"])
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        accepted = str(prerequisite["accepted_status"])
        if payload.get("status") != accepted:
            raise RuntimeError(f"prerequisite {path} has status {payload.get('status')!r}")
        validated.append(
            {
                "path": str(path.relative_to(project_root)),
                "sha256": _sha256(path),
                "status": accepted,
            }
        )
    return validated


def _cache_key(
    row: dict[str, Any], config: dict[str, Any], checkpoint_sha256: str
) -> str:
    payload = {
        "backend": config["baseline"]["backend"],
        "revision": config["baseline"]["revision"],
        "checkpoint_sha256": checkpoint_sha256,
        "image_sha256": row["image_sha256"],
        "preprocessing": config["preprocessing"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resize_longest_pil(image: Any, max_side: int) -> Any:
    from PIL import Image

    width, height = image.size
    if max(width, height) <= max_side:
        return image
    scale = max_side / max(width, height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(resized, Image.Resampling.BILINEAR)


def _top_fraction_mean(values: Any, fraction: float) -> float:
    import numpy as np

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top fraction must be in (0, 1]")
    count = max(1, int(round(flat.size * fraction)))
    start = flat.size - count
    return float(np.partition(flat, start)[start:].mean())


def _load_trufor(
    repository: Path,
    checkpoint: Path,
    config: dict[str, Any],
    device: Any,
) -> tuple[Any, Callable[[Path], tuple[Any, float, list[int], float]]]:
    import numpy as np
    import torch
    from PIL import Image

    if not config["baseline"].get("trusted_official_checkpoint_allow_legacy_pickle"):
        raise ValueError("TruFor legacy checkpoint loading was not explicitly authorized")
    sys.path.insert(0, str(repository))
    try:
        with _working_directory(repository):
            from lib.config import config as official_config
            from lib.config import update_config
            from lib.utils import get_model

            args = SimpleNamespace(
                experiment=str(config["baseline"]["experiment_config"]),
                gpu=[int(device.index or 0)],
                opts=["TEST.MODEL_FILE", str(checkpoint)],
            )
            update_config(official_config, args)
            model = get_model(official_config)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"], strict=True)
        model = model.eval().to(device)
    finally:
        sys.path.pop(0)

    divisor = float(config["preprocessing"]["image_divisor"])
    max_side = int(config["preprocessing"]["max_side"])

    def infer(image_path: Path) -> tuple[Any, float, list[int], float]:
        with Image.open(image_path) as handle:
            native_size = [handle.height, handle.width]
            image = _resize_longest_pil(handle.convert("RGB"), max_side)
            array = np.asarray(image)
        tensor = torch.tensor(
            array.transpose(2, 0, 1), dtype=torch.float32, device=device
        ).unsqueeze(0) / divisor
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            prediction, _confidence, detection, _noiseprint = model(tensor, save_np=False)
            scores = torch.softmax(prediction.squeeze(0), dim=0)[1]
            if detection is None:
                image_score = _top_fraction_mean(scores.detach().cpu().numpy(), 0.01)
            else:
                image_score = float(torch.sigmoid(detection).item())
        torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        output = scores.detach().float().cpu().numpy()
        return output, image_score, native_size, latency_ms

    return model, infer


def _load_adcdnet(
    repository: Path,
    checkpoint: Path,
    qt_table: Path,
    ocr_by_record: dict[str, Path],
    config: dict[str, Any],
    device: Any,
) -> tuple[Any, Callable[[Path, str], tuple[Any, float, list[int], float]]]:
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(repository))
    try:
        import cfg as adcd_cfg
        from ds import load_qt, multi_jpeg
        from model.model import ADCDNet

        adcd_cfg.docres_ckpt_path = "not_used_for_full_checkpoint_inference"
        ADCDNet.load_docres = lambda self: None
        model = ADCDNet()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {key.replace("module.", ""): value for key, value in saved["model"].items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"ADCD-Net checkpoint mismatch missing={missing} unexpected={unexpected}"
            )
        model = model.eval().to(device)
        quantization_tables = load_qt(qt_table)
    finally:
        sys.path.pop(0)

    max_side = int(config["preprocessing"]["max_side"])
    quality = int(config["preprocessing"]["jpeg_quality"])
    pad_divisor = int(config["preprocessing"]["pad_divisor"])
    top_fraction = float(config["preprocessing"]["image_score_top_fraction"])

    def infer(image_path: Path, record_id: str) -> tuple[Any, float, list[int], float]:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"OpenCV could not decode {image_path}")
        native_height, native_width = image_bgr.shape[:2]
        with np.load(ocr_by_record[record_id], allow_pickle=False) as cached:
            ocr_mask = np.asarray(cached["mask"], dtype=np.uint8)
        if list(ocr_mask.shape) != [native_height, native_width]:
            raise ValueError("OCR mask shape differs from input image shape")

        if max(native_height, native_width) > max_side:
            scale = max_side / max(native_height, native_width)
            width = max(1, round(native_width * scale))
            height = max(1, round(native_height * scale))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            image_bgr = cv2.resize(image_bgr, (width, height), interpolation=interpolation)
            ocr_mask = cv2.resize(
                ocr_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )

        dct, jpeg_image, qualities = multi_jpeg(
            Image.fromarray(image_bgr),
            num_jpeg=-1,
            min_qf=-1,
            upper_bound=-1,
            jpeg_record=[quality],
        )
        qt = quantization_tables[qualities[-1]].clamp(0, 63).unsqueeze(0)
        rgb = np.asarray(jpeg_image, dtype=np.float32) / 255.0
        rgb = (rgb - np.asarray((0.485, 0.455, 0.406), dtype=np.float32)) / np.asarray(
            (0.229, 0.224, 0.225), dtype=np.float32
        )
        image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
        dct_tensor = torch.from_numpy(np.clip(np.abs(dct), 0, 20))
        ocr_tensor = torch.from_numpy(ocr_mask).unsqueeze(0).long()
        height, width = image_tensor.shape[-2:]
        dct_height, dct_width = dct_tensor.shape[-2:]
        square = max(height, width, dct_height, dct_width)
        square = ((square + pad_divisor - 1) // pad_divisor) * pad_divisor
        image_tensor = torch.nn.functional.pad(
            image_tensor, (0, square - width, 0, square - height), value=0.0
        ).unsqueeze(0)
        dct_tensor = torch.nn.functional.pad(
            dct_tensor,
            (0, square - dct_width, 0, square - dct_height),
            value=0,
        ).unsqueeze(0)
        ocr_tensor = torch.nn.functional.pad(
            ocr_tensor, (0, square - width, 0, square - height), value=-1
        ).unsqueeze(0)
        dummy_mask = torch.zeros_like(ocr_tensor)
        tensors = [image_tensor, dct_tensor, qt, dummy_mask, ocr_tensor]
        image_tensor, dct_tensor, qt, dummy_mask, ocr_tensor = (
            value.to(device) for value in tensors
        )

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(config["runtime"]["amp"]),
        ):
            logits = model(
                image_tensor,
                dct_tensor,
                qt,
                dummy_mask,
                ocr_tensor,
                is_train=False,
            )[0]
            scores = torch.softmax(logits.float(), dim=1)[0, 1, :height, :width]
        torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        output = scores.detach().cpu().numpy()
        image_score = _top_fraction_mean(output, top_fraction)
        return output, image_score, [native_height, native_width], latency_ms

    return model, infer


def _load_ocr_records(
    project_root: Path, scratch: Path, config: dict[str, Any]
) -> dict[str, Path]:
    summary_path = _resolve(project_root, config["ocr"]["summary"])
    expected_summary_sha256 = config["ocr"].get("expected_summary_sha256")
    if (
        expected_summary_sha256 is not None
        and _sha256(summary_path) != expected_summary_sha256
    ):
        raise ValueError("ADCD-Net OCR summary SHA-256 changed")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    expected = int(config["ocr"]["expected_records"])
    if summary.get("status") != "passed" or summary.get("successful_records") != expected:
        raise RuntimeError("ADCD-Net full OCR cache is incomplete")
    records_path = _resolve(project_root, config["ocr"]["records"])
    expected_records_sha256 = config["ocr"].get("expected_records_sha256")
    if (
        expected_records_sha256 is not None
        and _sha256(records_path) != expected_records_sha256
    ):
        raise ValueError("ADCD-Net OCR records SHA-256 changed")
    records = _read_jsonl(records_path)
    if len(records) != expected or any(row.get("status") != "ok" for row in records):
        raise RuntimeError("ADCD-Net OCR item records are incomplete")
    result: dict[str, Path] = {}
    for row in records:
        path = _resolve(scratch, row["cache_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(row["record_id"])] = path
    return result


def _paper_evidence_authorized(config: dict[str, Any]) -> bool:
    paper_evidence = bool(config["experiment"].get("paper_evidence", False))
    if not paper_evidence:
        return False
    if config["experiment"].get("stage") != "one_shot_final_strong_baseline":
        raise ValueError(
            "paper-evidence baseline inference requires the frozen final-baseline stage"
        )
    if config["input"].get("max_records") is not None:
        raise ValueError("paper-evidence baseline inference must use the full manifest")
    if not config["runtime"].get("require_all_records", False):
        raise ValueError("paper-evidence baseline inference must require every record")
    if "protocol" not in config["experiment"] or "expected_protocol_sha256" not in config["experiment"]:
        raise ValueError("paper-evidence baseline inference must bind a frozen protocol")
    return True


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _safe_thread_environment(config)

    import numpy as np
    import torch

    if not config["runtime"]["gpu_launch_authorized"]:
        raise ValueError("baseline GPU launch has not been explicitly authorized")
    if config["runtime"]["method_training_authorized"]:
        raise ValueError("baseline execution must not authorize method training")
    paper_evidence = _paper_evidence_authorized(config)
    if "protocol" in config["experiment"]:
        if "expected_protocol_sha256" not in config["experiment"]:
            raise ValueError("baseline protocol is missing its frozen SHA-256")
        protocol = _resolve(project_root, config["experiment"]["protocol"])
        if _sha256(protocol) != config["experiment"]["expected_protocol_sha256"]:
            raise ValueError("baseline protocol SHA-256 changed")
    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("baseline execution requires an available CUDA device")

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, paths["scratch_default"]))
        )
    ).resolve()
    repository = _resolve(scratch, paths["repository"])
    checkpoint = _resolve(scratch, paths["checkpoint"])
    input_manifest = _resolve(project_root, config["input"]["manifest"])
    cache_dir = _resolve(scratch, paths["cache_dir"])
    output_predictions = _resolve(project_root, paths["output_predictions"])
    output_summary = _resolve(project_root, paths["output_summary"])
    log_path = _resolve(project_root, paths["log"])
    partial_path = output_predictions.with_suffix(output_predictions.suffix + ".partial")
    for path in (cache_dir, output_predictions.parent, output_summary.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    prerequisite_records = _validate_prerequisites(project_root, config)
    manifest_sha256 = _sha256(input_manifest)
    if manifest_sha256 != config["input"]["expected_manifest_sha256"]:
        raise ValueError("baseline input manifest SHA-256 changed")
    rows = _read_jsonl(input_manifest)
    expected_records = int(config["input"]["expected_records"])
    if len(rows) != expected_records:
        raise ValueError(f"expected {expected_records} input records, found {len(rows)}")
    max_records = config["input"].get("max_records")
    if max_records is not None:
        rows = rows[: int(max_records)]

    revision = _repository_revision(repository)
    if revision != config["baseline"]["revision"]:
        raise ValueError("baseline repository revision changed")
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != paths["checkpoint_sha256"]:
        raise ValueError("baseline checkpoint SHA-256 changed")

    backend = str(config["baseline"]["backend"])
    ocr_by_record: dict[str, Path] | None = None
    if backend == "trufor":
        model, infer = _load_trufor(repository, checkpoint, config, device)
    elif backend == "adcdnet":
        docres = _resolve(scratch, paths["docres"])
        qt_table = _resolve(scratch, paths["qt_table"])
        if _sha256(docres) != paths["docres_sha256"]:
            raise ValueError("ADCD-Net DocRes SHA-256 changed")
        if _sha256(qt_table) != paths["qt_table_sha256"]:
            raise ValueError("ADCD-Net quantization-table SHA-256 changed")
        ocr_by_record = _load_ocr_records(project_root, scratch, config)
        model, infer = _load_adcdnet(
            repository, checkpoint, qt_table, ocr_by_record, config, device
        )
    else:
        raise ValueError(f"unknown baseline backend {backend!r}")

    del model  # inference closure retains the model; avoid a second local reference
    if partial_path.exists():
        partial_path.unlink()
    torch.cuda.reset_peak_memory_stats(device)
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    cache_hits = 0
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        cache_key = _cache_key(row, config, checkpoint_sha256)
        cache_path = cache_dir / f"{cache_key}.npz"
        record: dict[str, Any] = {
            "record_id": row["record_id"],
            "source_group_id": row["source_group_id"],
            "evaluation_role": row["evaluation_role"],
            "sample_kind": row["sample_kind"],
            "baseline": config["baseline"]["name"],
            "cache_key": cache_key,
            "score_cache": str(cache_path.relative_to(scratch)),
            "paper_evidence": paper_evidence,
        }
        try:
            image_path = _resolve(scratch, row["image"])
            if _sha256(image_path) != row["image_sha256"]:
                raise ValueError("input image SHA-256 changed")
            cache_hit = cache_path.is_file()
            if cache_hit:
                with np.load(cache_path, allow_pickle=False) as cached:
                    scores = np.asarray(cached["scores"], dtype=np.float32)
                    image_score = float(np.asarray(cached["image_score"]).item())
                    native_size = [int(value) for value in cached["native_size"]]
                    latency_ms = float(np.asarray(cached["latency_ms"]).item())
                cache_hits += 1
            else:
                if backend == "trufor":
                    scores, image_score, native_size, latency_ms = infer(image_path)
                else:
                    scores, image_score, native_size, latency_ms = infer(
                        image_path, str(row["record_id"])
                    )
                temporary_cache = cache_path.with_suffix(".npz.tmp")
                with temporary_cache.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        scores=np.asarray(scores, dtype=np.float32),
                        image_score=np.asarray(image_score, dtype=np.float32),
                        native_size=np.asarray(native_size, dtype=np.int32),
                        latency_ms=np.asarray(latency_ms, dtype=np.float32),
                    )
                temporary_cache.replace(cache_path)
            if not np.isfinite(scores).all() or not np.isfinite(image_score):
                raise ValueError("baseline produced non-finite output")
            expected_native = [int(row["height"]), int(row["width"])]
            if native_size != expected_native:
                raise ValueError(
                    f"native image shape changed: {native_size} != {expected_native}"
                )
            record.update(
                {
                    "status": "ok",
                    "cache_hit": cache_hit,
                    "score_shape": list(scores.shape),
                    "native_shape": native_size,
                    "score_min": float(scores.min()),
                    "score_max": float(scores.max()),
                    "image_score": image_score,
                    "latency_ms": latency_ms,
                }
            )
            latencies.append(latency_ms)
        except Exception as error:  # every item-level failure remains visible
            record.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            logging.exception("baseline failed record_id=%s", row["record_id"])
            records.append(record)
            _append_jsonl(partial_path, record)
            if (
                config["runtime"].get("fail_fast_on_cuda_oom", True)
                and "out of memory" in str(error).lower()
            ):
                break
            continue
        records.append(record)
        _append_jsonl(partial_path, record)
        if index % 10 == 0 or index == len(rows):
            logging.info(
                "progress completed=%d total=%d cache_hits=%d",
                index,
                len(rows),
                cache_hits,
            )

    _write_jsonl(output_predictions, records)
    partial_path.unlink(missing_ok=True)
    failed = sum(row["status"] != "ok" for row in records)
    incomplete = len(records) != len(rows)
    status = "passed" if not failed and not incomplete else "failed"
    wall_time = time.monotonic() - started
    summary = {
        "experiment": config["experiment"],
        "baseline": config["baseline"],
        "status": status,
        "paper_evidence": paper_evidence,
        "gpu_launch_authorized": True,
        "method_training_authorized": False,
        "input_manifest": str(input_manifest.relative_to(project_root)),
        "input_manifest_sha256": manifest_sha256,
        "baseline_freeze_id": (
            rows[0].get("baseline_freeze_id") if rows else None
        ),
        "input_freeze_field": str(
            config["input"].get("freeze_field", "baseline_freeze_id")
        ),
        "input_freeze_id": (
            rows[0].get(
                str(config["input"].get("freeze_field", "baseline_freeze_id"))
            )
            if rows
            else None
        ),
        "repository_revision": revision,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing": config["preprocessing"],
        "prerequisites": prerequisite_records,
        "selected_records": len(rows),
        "written_records": len(records),
        "successful_records": len(records) - failed,
        "failed_records": failed + int(incomplete),
        "cache_hits": cache_hits,
        "wall_time_seconds": wall_time,
        "latency_ms_median": median(latencies) if latencies else None,
        "latency_ms_total": sum(latencies),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
        "output_predictions": str(output_predictions.relative_to(project_root)),
        "output_predictions_sha256": _sha256(output_predictions),
        "cache_dir": str(cache_dir.relative_to(scratch)),
        "log": str(log_path.relative_to(project_root)),
    }
    _write_json(output_summary, summary)
    if status != "passed" and config["runtime"]["require_all_records"]:
        raise RuntimeError(
            f"{config['baseline']['name']} baseline incomplete; see {output_summary}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
