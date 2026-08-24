from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


def _resolve(project_root: Path, base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _find_unique(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {filename} below {root}, found {len(matches)}")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ocr_mask(path: Path, image_key: str, height: int, width: int) -> np.ndarray:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    boxes = records[image_key]
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, box_width, box_height in boxes:
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2 = min(width, x1 + int(box_width))
        y2 = min(height, y1 + int(box_height))
        mask[y1:y2, x1:x2] = 1
    return mask


def _prepare_input(
    image_path: Path,
    mask_path: Path,
    ocr_path: Path,
    image_key: str,
    image_size: int,
    qt_path: Path,
    external_repo: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sys.path.insert(0, str(external_repo))
    try:
        from ds import load_qt, multi_jpeg
    finally:
        sys.path.pop(0)

    source = Image.open(image_path).convert("RGB")
    source_mask = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
    ocr = _ocr_mask(ocr_path, image_key, source.height, source.width)

    source = source.resize((image_size, image_size), Image.Resampling.BILINEAR)
    source_mask = cv2.resize(source_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    ocr = cv2.resize(ocr, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    dct, jpeg_image, qualities = multi_jpeg(
        source, num_jpeg=-1, min_qf=-1, upper_bound=-1, jpeg_record=[100]
    )
    qts = load_qt(qt_path)
    qt = qts[qualities[-1]].clamp(0, 63)

    rgb = np.asarray(jpeg_image, dtype=np.float32) / 255.0
    rgb = (rgb - np.asarray((0.485, 0.455, 0.406), dtype=np.float32)) / np.asarray(
        (0.229, 0.224, 0.225), dtype=np.float32
    )
    image = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    return (
        image,
        torch.from_numpy(np.clip(np.abs(dct), 0, 20)).unsqueeze(0),
        qt.unsqueeze(0),
        torch.from_numpy(source_mask).unsqueeze(0).unsqueeze(0).long(),
        torch.from_numpy(ocr).unsqueeze(0).unsqueeze(0).long(),
    )


def run(config_path: Path) -> dict[str, Any]:
    project_root = config_path.resolve().parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"], str(_resolve(project_root, project_root, paths["scratch_default"]))
        )
    ).resolve()
    adcd_repo = _resolve(project_root, scratch, paths["adcd_repository"])
    doctamper_repo = _resolve(project_root, scratch, paths["doctamper_repository"])
    checkpoint_dir = _resolve(project_root, scratch, paths["checkpoint_directory"])
    output_prediction = _resolve(project_root, project_root, paths["output_prediction"])
    output_scores = _resolve(project_root, project_root, paths["output_scores"])
    output_manifest = _resolve(project_root, project_root, paths["output_manifest"])
    log_path = _resolve(project_root, project_root, paths["log"])
    for output in (output_prediction, output_scores, output_manifest, log_path):
        output.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this preflight requires an available CUDA device")
    checkpoint = _find_unique(checkpoint_dir, "ADCDNet.pth")
    qt_path = doctamper_repo / "qt_table.pk"

    sys.path.insert(0, str(adcd_repo))
    try:
        import cfg as adcd_cfg

        from model.model import ADCDNet

        # The released inference checkpoint contains the complete model state.
        # DocRes is only used by the upstream constructor before that state is
        # loaded, so skip the redundant initialization download here and demand
        # an exact load of the final checkpoint below.
        adcd_cfg.docres_ckpt_path = "not_used_for_full_checkpoint_inference"
        ADCDNet.load_docres = lambda self: None
        model = ADCDNet()
    finally:
        sys.path.pop(0)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.replace("module.", ""): value for key, value in saved["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.eval().to(device)

    input_config = config["input"]
    inputs = _prepare_input(
        doctamper_repo / input_config["image"],
        doctamper_repo / input_config["mask"],
        doctamper_repo / input_config["ocr_boxes"],
        input_config["image_key"],
        int(input_config["image_size"]),
        qt_path,
        adcd_repo,
    )
    image, dct, qt, mask, ocr_mask = (value.to(device) for value in inputs)

    amp = bool(config["runtime"]["amp"])
    warmup_runs = int(config["runtime"]["warmup_runs"])
    measured_runs = int(config["runtime"]["measured_runs"])
    torch.cuda.reset_peak_memory_stats(device)
    latencies_ms: list[float] = []
    probability: torch.Tensor | None = None
    with torch.inference_mode():
        for run_index in range(warmup_runs + measured_runs):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(image, dct, qt, mask, ocr_mask, is_train=False)[0]
            probability = torch.softmax(logits.float(), dim=1)[:, 1]
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if run_index >= warmup_runs:
                latencies_ms.append(elapsed_ms)

    assert probability is not None
    scores = probability.squeeze(0).cpu().numpy()
    finite = bool(np.isfinite(scores).all())
    compatibility_ok = finite and not missing and not unexpected
    np.savez_compressed(output_scores, scores=scores, mask=mask.squeeze().cpu().numpy())
    record = {
        "example_id": "doctamper_stg_0",
        "status": "ok" if compatibility_ok else "failed",
        "paper_evidence": False,
        "baseline": config["baseline"]["name"],
        "input": input_config["name"],
        "score_shape": list(scores.shape),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "scores_file": str(output_scores.relative_to(project_root)),
    }
    with output_prediction.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest: dict[str, Any] = {
        "experiment": config["experiment"],
        "baseline": config["baseline"],
        "input": input_config,
        "status": "passed" if compatibility_ok else "failed",
        "finite_output": finite,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "latency_ms_median": float(np.median(latencies_ms)),
        "latency_ms_runs": latencies_ms,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "checkpoint": str(checkpoint.relative_to(scratch)),
        "checkpoint_sha256": _sha256(checkpoint),
        "constructor_docres_used": False,
        "input_image_sha256": _sha256(doctamper_repo / input_config["image"]),
        "prediction": str(output_prediction.relative_to(project_root)),
    }
    with output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    logging.info("preflight completed manifest=%s", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official ADCD-Net CUDA compatibility preflight")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
