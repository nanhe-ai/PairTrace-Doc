from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        _invalid_thread_value = int(os.environ.get(_thread_variable, "1")) < 1
    except ValueError:
        _invalid_thread_value = True
    if _invalid_thread_value:
        os.environ[_thread_variable] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from pairtrace_doc.pipelines.train_pairtrace_100 import (
    _jpeg,
    _load_teacher,
    _pad_triplet,
)
from pairtrace_doc.pipelines.train_student_100 import (
    _dice_loss,
    _generator_sampling_pools,
    _prepare_pair_cache,
    _read_jsonl,
    _resolve,
    _save_checkpoint,
    _sha256,
    _write_csv,
    _write_json,
    _write_jsonl,
)
from pairtrace_doc.pipelines.train_tfr_equal_budget_arm import (
    EXTENDED_ARMS,
    _build_model,
    _evaluate_authentic,
    _evaluate_forged,
    _select_operating_point,
    _select_representation,
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _portable_path(path: Path, project_root: Path) -> str:
    return Path(os.path.relpath(path, project_root)).as_posix()


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle)
    base_value = override.pop("base_config", None)
    expected_base = override.pop("expected_base_config_sha256", None)
    if base_value is None:
        return override
    project_root = config_path.parent.parent
    base_path = _resolve(project_root, str(base_value))
    if expected_base and _sha256(base_path) != str(expected_base):
        raise ValueError("resampling continuation base config changed")
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    return _deep_merge(base, override)


def _validate_warm_start_manifest(
    config: dict[str, Any], project_root: Path
) -> dict[str, Any] | None:
    experiment = config["experiment"]
    manifest_value = experiment.get("warm_start_manifest")
    if manifest_value is None:
        return None
    manifest_path = _resolve(project_root, str(manifest_value))
    expected_manifest_sha256 = str(
        experiment.get("expected_warm_start_manifest_sha256", "")
    )
    if not expected_manifest_sha256 or _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("warm-start manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("holdout_read") is not False or manifest.get("paper_evidence") is not False:
        raise ValueError("warm-start manifest crosses the evidence boundary")
    arm = str(experiment.get("arm", "explicit_9ch"))
    seed = int(experiment["seed"])
    matches = [
        record
        for record in manifest.get("records", [])
        if str(record.get("arm")) == arm and int(record.get("seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise ValueError("warm-start manifest does not bind exactly one arm/seed record")
    record = matches[0]
    model = config["model"]
    if str(record.get("checkpoint")) != str(model["warm_start_checkpoint"]):
        raise ValueError("warm-start manifest checkpoint path changed")
    if str(record.get("sha256")) != str(model["warm_start_checkpoint_sha256"]):
        raise ValueError("warm-start manifest checkpoint digest changed")
    checkpoint_path = _resolve(project_root, str(record["checkpoint"]))
    if _sha256(checkpoint_path) != str(record["sha256"]):
        raise ValueError("warm-start checkpoint changed after manifest freeze")
    return record


def _validate_multiseed_training_authorization(config: dict[str, Any]) -> bool:
    runtime = config["runtime"]
    authorized = bool(runtime.get("multi_seed_authorized", False))
    if not authorized:
        if config.get("multi_seed") is not None:
            raise ValueError("single-seed run cannot carry a multi-seed authorization block")
        return False
    experiment = config["experiment"]
    if experiment.get("stage") != "frozen_multiseed_stability_training":
        raise ValueError("multi-seed training stage is not frozen")
    policy = config.get("multi_seed")
    if not isinstance(policy, dict):
        raise ValueError("multi-seed training policy is missing")
    seed = int(experiment["seed"])
    new_seeds = [int(value) for value in policy["new_training_seeds"]]
    family_seeds = [int(value) for value in policy["family_seeds"]]
    if family_seeds != [20260747, 20260763, 20260764]:
        raise ValueError("multi-seed family changed")
    if new_seeds != [20260763, 20260764] or seed not in new_seeds:
        raise ValueError("training seed is not authorized by the frozen family")
    if policy.get("seed_is_only_intended_recipe_difference") is not True:
        raise ValueError("multi-seed recipe difference policy changed")
    return True


def _validate_multiseed_recipe(config: dict[str, Any], project_root: Path) -> None:
    policy = config["multi_seed"]
    identity_path = _resolve(project_root, policy["paper_identity_amendment"])
    if _sha256(identity_path) != policy["expected_paper_identity_sha256"]:
        raise ValueError("multi-seed paper identity amendment changed")
    recipe_path = _resolve(project_root, policy["recipe_reference_config"])
    if _sha256(recipe_path) != policy["expected_recipe_reference_sha256"]:
        raise ValueError("multi-seed recipe reference changed")
    with recipe_path.open("r", encoding="utf-8") as handle:
        reference = yaml.safe_load(handle)
    for section in (
        "data",
        "model",
        "preprocessing",
        "sampling",
        "augmentation",
        "training",
    ):
        if config[section] != reference[section]:
            raise ValueError(f"multi-seed recipe section changed: {section}")


def _reference_roundtrip(
    reference: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    augmentation: dict[str, Any],
) -> np.ndarray:
    if mode == "clean":
        return reference
    height, width = reference.shape[:2]
    homography = _sample_homography(height, width, mode, rng, augmentation)
    return _warp_roundtrip(reference, homography)


def _sample_homography(
    height: int,
    width: int,
    mode: str,
    rng: np.random.Generator,
    augmentation: dict[str, Any],
) -> np.ndarray:
    if mode == "translation":
        limit = float(augmentation["translation_max_pixels"])
        dx, dy = rng.uniform(-limit, limit, size=2)
        homography = np.asarray(
            [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    elif mode == "affine":
        angle = rng.uniform(
            -float(augmentation["affine_rotation_max_degrees"]),
            float(augmentation["affine_rotation_max_degrees"]),
        )
        scale_delta = float(augmentation["affine_scale_delta"])
        scale = rng.uniform(1.0 - scale_delta, 1.0 + scale_delta)
        limit = float(augmentation["affine_translation_max_pixels"])
        dx, dy = rng.uniform(-limit, limit, size=2)
        affine = cv2.getRotationMatrix2D(
            ((width - 1) / 2.0, (height - 1) / 2.0), angle, scale
        ).astype(np.float64)
        affine[0, 2] += dx
        affine[1, 2] += dy
        homography = np.vstack([affine, [0.0, 0.0, 1.0]])
    elif mode == "perspective":
        source = np.asarray(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        fraction = float(augmentation["perspective_corner_jitter_fraction"])
        jitter = rng.uniform(-fraction, fraction, size=(4, 2)).astype(np.float32)
        jitter[:, 0] *= width - 1.0
        jitter[:, 1] *= height - 1.0
        homography = cv2.getPerspectiveTransform(source, source + jitter).astype(
            np.float64
        )
    else:
        raise ValueError(f"unsupported projective augmentation mode: {mode}")
    return homography


def _warp_once(image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _warp_roundtrip(image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    stressed = cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return cv2.warpPerspective(
        stressed,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _resize_restore(
    image: np.ndarray, rng: np.random.Generator, augmentation: dict[str, Any]
) -> np.ndarray:
    height, width = image.shape[:2]
    factor = float(
        rng.uniform(
            float(augmentation["resize_factor_min"]),
            float(augmentation["resize_factor_max"]),
        )
    )
    resized_width = max(2, int(round(width * factor)))
    resized_height = max(2, int(round(height * factor)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    return cv2.resize(resized, (width, height), interpolation=cv2.INTER_LINEAR)


def _gaussian_blur(
    image: np.ndarray, rng: np.random.Generator, augmentation: dict[str, Any]
) -> np.ndarray:
    sigma = float(
        rng.uniform(
            float(augmentation["blur_sigma_min"]),
            float(augmentation["blur_sigma_max"]),
        )
    )
    kernel = 2 * int(math.ceil(3.0 * sigma)) + 1
    return cv2.GaussianBlur(
        image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101
    )


def _asymmetric_jpeg(
    image: np.ndarray, rng: np.random.Generator, augmentation: dict[str, Any]
) -> np.ndarray:
    quality = int(
        rng.integers(
            int(augmentation["asymmetric_jpeg_quality_min"]),
            int(augmentation["asymmetric_jpeg_quality_max"]) + 1,
        )
    )
    return _jpeg(image, quality)


def _apply_pair_augmentation(
    candidate: np.ndarray,
    reference: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    augmentation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    application = str(
        augmentation.get("transform_application", "reference_roundtrip_only")
    )
    if application == "reference_clean_identity_only":
        if mode != "clean":
            raise ValueError("identity continuation sampled a non-clean mode")
        return candidate, reference
    if application == "reference_roundtrip_only":
        return candidate, _reference_roundtrip(reference, mode, rng, augmentation)
    if mode == "clean":
        return candidate, reference
    if application in {
        "candidate_roundtrip_only",
        "joint_roundtrip",
        "reference_single_warp_only",
    }:
        height, width = reference.shape[:2]
        homography = _sample_homography(height, width, mode, rng, augmentation)
        if application == "candidate_roundtrip_only":
            return _warp_roundtrip(candidate, homography), reference
        if application == "joint_roundtrip":
            return (
                _warp_roundtrip(candidate, homography),
                _warp_roundtrip(reference, homography),
            )
        return candidate, _warp_once(reference, homography)
    if mode != "perturb":
        raise ValueError(f"nonprojective augmentation sampled invalid mode: {mode}")
    if application == "reference_resize_restore_only":
        return candidate, _resize_restore(reference, rng, augmentation)
    if application == "reference_blur_only":
        return candidate, _gaussian_blur(reference, rng, augmentation)
    if application == "reference_jpeg_only":
        return candidate, _asymmetric_jpeg(reference, rng, augmentation)
    raise ValueError(f"unsupported pair augmentation application: {application}")


class ResamplingRobustTeacherDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]
):
    def __init__(
        self,
        records: list[dict[str, Any]],
        sampling: dict[str, Any],
        augmentation: dict[str, Any],
        preprocessing: dict[str, Any],
        seed: int,
        length: int,
    ) -> None:
        self.records = records
        self.sampling = sampling
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.seed = seed
        self.length = length
        self.epoch = 0
        self.generator_sampling = _generator_sampling_pools(records, sampling)
        self.geometry_names = list(augmentation["geometry_probabilities"])
        self.geometry_probabilities = np.asarray(
            [augmentation["geometry_probabilities"][name] for name in self.geometry_names],
            dtype=float,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        rng = np.random.default_rng(
            self.seed + 20_000_033 + self.epoch * 1_000_003 + index
        )
        if self.generator_sampling is None:
            record = self.records[int(rng.integers(0, len(self.records)))]
        else:
            generators, probabilities, pools = self.generator_sampling
            generator = str(rng.choice(generators, p=probabilities))
            pool = pools[generator]
            record = pool[int(rng.integers(0, len(pool)))]

        draw = float(rng.random())
        positive_limit = float(self.sampling["teacher_forged_positive_probability"])
        forged_limit = positive_limit + float(
            self.sampling["teacher_forged_random_probability"]
        )
        stored_mask = np.asarray(np.load(record["mask"], mmap_mode="r"))
        if draw < forged_limit:
            candidate = np.asarray(np.load(record["forged"], mmap_mode="r"))
            reference = np.asarray(np.load(record["authentic"], mmap_mode="r"))
            mask = stored_mask
            positive_crop = draw < positive_limit
            sample_kind = "forged_positive" if positive_crop else "forged_random"
        else:
            candidate = np.asarray(np.load(record["authentic"], mmap_mode="r"))
            reference = candidate
            mask = np.zeros_like(stored_mask, dtype=np.uint8)
            positive_crop = False
            sample_kind = "authentic_random"

        crop_size = int(self.preprocessing["crop_size"])
        candidate, reference, mask = _pad_triplet(
            candidate, reference, mask, crop_size
        )
        height, width = mask.shape
        if positive_crop:
            x1, y1, x2, y2 = record["bbox_xyxy"]
            center_x = int(rng.integers(x1, max(x1 + 1, x2)))
            center_y = int(rng.integers(y1, max(y1 + 1, y2)))
            left = int(
                np.clip(
                    center_x - int(rng.integers(0, crop_size)),
                    0,
                    width - crop_size,
                )
            )
            top = int(
                np.clip(
                    center_y - int(rng.integers(0, crop_size)),
                    0,
                    height - crop_size,
                )
            )
        else:
            left = int(rng.integers(0, width - crop_size + 1))
            top = int(rng.integers(0, height - crop_size + 1))
        candidate_crop = np.array(
            candidate[top : top + crop_size, left : left + crop_size], copy=True
        )
        reference_crop = np.array(
            reference[top : top + crop_size, left : left + crop_size], copy=True
        )
        mask_crop = np.array(
            mask[top : top + crop_size, left : left + crop_size], copy=True
        )

        if rng.random() < float(self.sampling["brightness_contrast_probability"]):
            brightness = rng.uniform(
                -float(self.sampling["brightness_delta"]),
                float(self.sampling["brightness_delta"]),
            ) * 255.0
            contrast = 1.0 + rng.uniform(
                -float(self.sampling["contrast_delta"]),
                float(self.sampling["contrast_delta"]),
            )
            candidate_crop = np.clip(
                candidate_crop.astype(np.float32) * contrast + brightness, 0, 255
            ).astype(np.uint8)
            reference_crop = np.clip(
                reference_crop.astype(np.float32) * contrast + brightness, 0, 255
            ).astype(np.uint8)
        if rng.random() < float(self.sampling["jpeg_probability"]):
            quality = int(
                rng.integers(
                    int(self.sampling["jpeg_quality_min"]),
                    int(self.sampling["jpeg_quality_max"]) + 1,
                )
            )
            candidate_crop = _jpeg(candidate_crop, quality)
            reference_crop = _jpeg(reference_crop, quality)

        geometry = str(
            rng.choice(self.geometry_names, p=self.geometry_probabilities)
        )
        candidate_crop, reference_crop = _apply_pair_augmentation(
            candidate_crop,
            reference_crop,
            geometry,
            rng,
            self.augmentation,
        )
        mean = np.asarray(self.preprocessing["imagenet_mean"], dtype=np.float32)
        std = np.asarray(self.preprocessing["imagenet_std"], dtype=np.float32)
        candidate_float = (candidate_crop.astype(np.float32) / 255.0 - mean) / std
        reference_float = (reference_crop.astype(np.float32) / 255.0 - mean) / std
        teacher_input = np.concatenate(
            [candidate_float, reference_float, candidate_float - reference_float],
            axis=2,
        ).transpose(2, 0, 1)
        return (
            torch.from_numpy(teacher_input.copy()),
            torch.from_numpy(mask_crop.astype(np.float32)).unsqueeze(0),
            geometry,
            sample_kind,
        )


class ResamplingRepresentationDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]
):
    def __init__(self, paired: ResamplingRobustTeacherDataset, arm: str) -> None:
        if arm not in EXTENDED_ARMS:
            raise ValueError(f"unsupported robust representation arm: {arm}")
        self.paired = paired
        self.arm = arm

    def set_epoch(self, epoch: int) -> None:
        self.paired.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.paired)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        teacher, mask, geometry, sample_kind = self.paired[index]
        return _select_representation(teacher, self.arm), mask, geometry, sample_kind


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = _load_config(config_path)
    runtime = config["runtime"]
    for name, value in runtime.get("environment", {}).items():
        os.environ[str(name)] = str(value)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if not runtime["gpu_launch_authorized"] or not runtime["resampling_teacher_training_authorized"]:
        raise ValueError("resampling teacher training was not explicitly authorized")
    multi_seed_run = _validate_multiseed_training_authorization(config)
    if any(
        bool(runtime.get(name))
        for name in (
            "viewed_development_read_allowed",
            "unseen_development_read_allowed",
            "final_reserve_read_allowed",
        )
    ):
        raise ValueError("resampling teacher training crossed an evidence boundary")
    if config["experiment"]["paper_evidence"]:
        raise ValueError("resampling teacher pilot cannot be paper evidence")
    _validate_warm_start_manifest(config, project_root)
    device = torch.device(runtime["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("resampling teacher training requires CUDA")

    protocol_path = _resolve(project_root, config["experiment"]["protocol"])
    if _sha256(protocol_path) != config["experiment"]["expected_protocol_sha256"]:
        raise ValueError("resampling teacher protocol SHA-256 changed")
    if multi_seed_run:
        _validate_multiseed_recipe(config, project_root)
    data = config["data"]
    manifest_path = _resolve(project_root, data["manifest"])
    if _sha256(manifest_path) != data["expected_manifest_sha256"]:
        raise ValueError("resampling teacher manifest SHA-256 changed")
    rows = sorted(
        _read_jsonl(manifest_path), key=lambda row: str(row["source_group_id"])
    )
    if len(rows) != int(data["expected_train_records"]):
        raise ValueError("resampling teacher train count changed")
    if len({str(row["source_group_id"]) for row in rows}) != len(rows):
        raise ValueError("resampling teacher contains duplicate source groups")
    if any(row["resampling_teacher_role"] != "train" for row in rows):
        raise ValueError("resampling teacher manifest contains a non-training role")
    if {str(row["resampling_teacher_freeze_id"]) for row in rows} != {
        str(data["expected_freeze_id"])
    }:
        raise ValueError("resampling teacher freeze ID changed")
    max_train_pairs = data.get("max_train_pairs")
    if max_train_pairs is not None:
        rows = rows[: int(max_train_pairs)]
    validation_rows: list[dict[str, Any]] = []
    if data.get("validation_manifest") is not None:
        validation_manifest_path = _resolve(project_root, data["validation_manifest"])
        if _sha256(validation_manifest_path) != data["expected_validation_manifest_sha256"]:
            raise ValueError("continuation validation manifest changed")
        validation_rows = [
            row
            for row in _read_jsonl(validation_manifest_path)
            if str(row[data["validation_role_field"]]) == str(data["validation_role"])
        ]
        if len(validation_rows) != int(data["expected_validation_records"]):
            raise ValueError("continuation validation count changed")
        max_validation_pairs = data.get("max_validation_pairs")
        if max_validation_pairs is not None:
            validation_rows = validation_rows[: int(max_validation_pairs)]

    sampling = config["sampling"]
    if not math.isclose(
        sum(
            float(sampling[name])
            for name in (
                "teacher_forged_positive_probability",
                "teacher_forged_random_probability",
                "teacher_authentic_random_probability",
            )
        ),
        1.0,
        abs_tol=1e-12,
    ):
        raise ValueError("teacher sample-kind probabilities must sum to one")
    augmentation = config["augmentation"]
    transform_application = str(augmentation.get("transform_application"))
    projective_applications = {
        "reference_roundtrip_only",
        "candidate_roundtrip_only",
        "joint_roundtrip",
        "reference_single_warp_only",
    }
    nonprojective_applications = {
        "reference_resize_restore_only",
        "reference_blur_only",
        "reference_jpeg_only",
    }
    probability_keys = set(augmentation["geometry_probabilities"])
    expected_probability_keys = (
        {"clean", "translation", "affine", "perspective"}
        if transform_application in projective_applications
        else {"clean", "perturb"}
        if transform_application in nonprojective_applications
        else {"clean", "translation", "affine", "perspective"}
        if probability_keys == {"clean", "translation", "affine", "perspective"}
        else {"clean"}
    )
    if probability_keys != expected_probability_keys or not math.isclose(
        sum(float(value) for value in augmentation["geometry_probabilities"].values()),
        1.0,
        abs_tol=1e-12,
    ):
        raise ValueError("reference geometry probabilities changed")
    if augmentation.get("interpolation") != "bilinear" or augmentation.get(
        "border_mode"
    ) != "reflect_101":
        raise ValueError("reference round-trip resampling policy changed")
    supported_applications = projective_applications | nonprojective_applications | {
        "reference_clean_identity_only"
    }
    if transform_application not in supported_applications:
        raise ValueError("reference transform application changed")
    if transform_application == "reference_clean_identity_only":
        identity_probabilities = augmentation["geometry_probabilities"]
        if float(identity_probabilities.get("clean", 0.0)) != 1.0 or any(
            float(value) != 0.0
            for key, value in identity_probabilities.items()
            if key != "clean"
        ):
            raise ValueError("reference transform label does not match its distribution")
    if transform_application == "reference_resize_restore_only" and not (
        0.0 < float(augmentation["resize_factor_min"])
        < float(augmentation["resize_factor_max"])
    ):
        raise ValueError("resize-restore range changed")
    if transform_application == "reference_blur_only" and not (
        0.0 < float(augmentation["blur_sigma_min"])
        <= float(augmentation["blur_sigma_max"])
    ):
        raise ValueError("asymmetric blur range changed")
    if transform_application == "reference_jpeg_only" and not (
        1
        <= int(augmentation["asymmetric_jpeg_quality_min"])
        <= int(augmentation["asymmetric_jpeg_quality_max"])
        <= 100
    ):
        raise ValueError("asymmetric JPEG range changed")
    training = config["training"]
    expected_steps = int(training["epochs"]) * int(training["steps_per_epoch"])
    if int(training.get("expected_optimizer_steps", expected_steps)) != expected_steps:
        raise ValueError("resampling continuation optimizer-step budget changed")

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    paths = config["paths"]
    scratch = Path(
        os.environ.get(
            paths["scratch_env"],
            str(_resolve(project_root, paths["scratch_default"])),
        )
    ).resolve()
    pair_cache_dir = _resolve(scratch, paths["pair_cache_dir"])
    checkpoint_path = _resolve(project_root, paths["checkpoint"])
    epoch_log_path = _resolve(project_root, paths["epoch_log"])
    summary_path = _resolve(project_root, paths["summary"])
    log_path = _resolve(project_root, paths["log"])
    predictions_path = (
        _resolve(project_root, paths["prediction_records"]) if validation_rows else None
    )
    metrics_path = _resolve(project_root, paths["metrics"]) if validation_rows else None
    for path in (
        pair_cache_dir,
        checkpoint_path.parent,
        epoch_log_path.parent,
        summary_path.parent,
        log_path.parent,
        *(path.parent for path in (predictions_path, metrics_path) if path is not None),
    ):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    started = time.monotonic()
    pair_cache: list[dict[str, Any]] = []
    validation_cache: list[dict[str, Any]] = []
    cache_hits = 0
    for row in rows:
        record, hit = _prepare_pair_cache(
            row, scratch, pair_cache_dir, config["preprocessing"]
        )
        pair_cache.append(record)
        cache_hits += int(hit)
    for row in validation_rows:
        record, hit = _prepare_pair_cache(
            row, scratch, pair_cache_dir, config["preprocessing"]
        )
        validation_cache.append(record)
        cache_hits += int(hit)

    model_config = config["model"]
    encoder_path = _resolve(scratch, model_config["encoder_weights"])
    warm_path = _resolve(project_root, model_config["warm_start_checkpoint"])
    for path, expected, label in (
        (encoder_path, model_config["encoder_weights_sha256"], "encoder weights"),
        (warm_path, model_config["warm_start_checkpoint_sha256"], "warm checkpoint"),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen {label} changed")
    arm = str(config["experiment"].get("arm", "explicit_9ch"))
    if arm not in EXTENDED_ARMS:
        raise ValueError("unsupported resampling representation arm")
    encoder_state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    teacher = _build_model(arm, encoder_state)
    warm = torch.load(warm_path, map_location="cpu", weights_only=True)
    teacher.load_state_dict(warm["model_state"], strict=True)
    teacher = teacher.to(device)
    torch.cuda.reset_peak_memory_stats(device)

    paired_dataset = ResamplingRobustTeacherDataset(
        pair_cache, sampling, augmentation, config["preprocessing"], seed,
        int(training["steps_per_epoch"]) * int(training["batch_size"]),
    )
    dataset = ResamplingRepresentationDataset(paired_dataset, arm)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        persistent_workers=False,
    )
    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    positive_weight = torch.tensor(
        float(training["bce_positive_weight"]), device=device
    )
    epoch_records: list[dict[str, Any]] = []
    for epoch in range(int(training["epochs"])):
        dataset.set_epoch(epoch)
        teacher.train()
        losses: list[float] = []
        geometries: Counter[str] = Counter()
        sample_kinds: Counter[str] = Counter()
        for teacher_inputs, masks, geometry_names, sample_names in loader:
            teacher_inputs = teacher_inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=bool(training["amp"]),
            ):
                logits = teacher(teacher_inputs)
                bce = F.binary_cross_entropy_with_logits(
                    logits, masks, pos_weight=positive_weight
                )
                dice = _dice_loss(logits, masks)
                loss = float(training["bce_loss_weight"]) * bce + float(
                    training["dice_loss_weight"]
                ) * dice
            if not torch.isfinite(loss):
                raise RuntimeError("resampling teacher produced a non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                teacher.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            geometries.update(str(value) for value in geometry_names)
            sample_kinds.update(str(value) for value in sample_names)
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "geometry_counts": dict(sorted(geometries.items())),
            "sample_kind_counts": dict(sorted(sample_kinds.items())),
            "paper_evidence": False,
        }
        epoch_records.append(record)
        _write_jsonl(epoch_log_path, epoch_records)
        logging.info("epoch=%d metrics=%s", epoch + 1, record)

    _save_checkpoint(
        checkpoint_path,
        {
            "model_state": teacher.state_dict(),
            "epochs": int(training["epochs"]),
            "selected_epoch": int(training["epochs"]),
            "selection_rule": "fixed_final_epoch",
            "config_sha256": _sha256(config_path),
            "protocol_sha256": _sha256(protocol_path),
            "input_manifest_sha256": _sha256(manifest_path),
            "warm_start_checkpoint_sha256": _sha256(warm_path),
            "encoder_weights_sha256": _sha256(encoder_path),
            "architecture": model_config["architecture"],
            "representation_arm": arm,
            "teacher_input_channels": 9 if arm == "explicit_9ch" else (6 if arm in {"candidate_reference_6ch", "fc_siam_diff", "fc_siam_conc"} else 3),
            "augmentation": augmentation,
            "training_seed": seed,
            "multi_seed_stability_run": multi_seed_run,
        },
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    group_macro_ap: float | None = None
    authentic_scores: dict[str, np.ndarray] = {}
    selected: dict[str, Any] | None = None
    prediction_records: list[dict[str, Any]] = []
    if validation_cache:
        group_macro_ap, forged_results, forged_scores, forged_masks = _evaluate_forged(
            teacher,
            validation_cache,
            arm,
            device,
            training,
            config["preprocessing"],
            keep_scores=True,
        )
        authentic_scores = _evaluate_authentic(
            teacher,
            validation_cache,
            arm,
            device,
            training,
            config["preprocessing"],
        )
        operating = config["operating_point"]
        thresholds = np.arange(
            float(operating["candidate_min"]),
            float(operating["candidate_max"])
            + float(operating["candidate_step"]) / 2,
            float(operating["candidate_step"]),
        )
        selected = _select_operating_point(
            forged_scores,
            forged_masks,
            [str(record["source_group_id"]) for record in validation_cache],
            authentic_scores,
            thresholds,
            float(operating["max_authentic_fpr"]),
        )
        threshold = float(selected["threshold"])
        for record, result, scores, mask in zip(
            validation_cache, forged_results, forged_scores, forged_masks
        ):
            binary = scores >= threshold
            tp = int(np.count_nonzero(binary & mask))
            fp = int(np.count_nonzero(binary & ~mask))
            fn = int(np.count_nonzero(~binary & mask))
            prediction_records.append(
                {
                    "arm": arm,
                    "role": "validation",
                    "sample_kind": "forged_pair",
                    "sample_id": record["sample_id"],
                    "source_group_id": record["source_group_id"],
                    "average_precision": result["average_precision"],
                    "auroc": result["auroc"],
                    "threshold": threshold,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pixel_count": int(mask.size),
                    "paper_evidence": False,
                }
            )
        for group, scores in authentic_scores.items():
            prediction_records.append(
                {
                    "arm": arm,
                    "role": "validation",
                    "sample_kind": "authentic_pair",
                    "sample_id": f"{group}:authentic",
                    "source_group_id": group,
                    "threshold": threshold,
                    "false_positive_pixels": int(
                        np.count_nonzero(scores >= threshold)
                    ),
                    "pixel_count": int(scores.size),
                    "fpr": float(np.mean(scores >= threshold)),
                    "paper_evidence": False,
                }
            )
        if predictions_path is None or metrics_path is None:
            raise RuntimeError("continuation validation output paths are missing")
        _write_jsonl(predictions_path, prediction_records)
        _write_csv(
            metrics_path,
            [
                {
                    "arm": arm,
                    "optimizer_steps": expected_steps,
                    "validation_source_group_macro_pixel_ap": group_macro_ap,
                    "validation_threshold": threshold,
                    "validation_source_group_macro_forged_pixel_f1": selected[
                        "source_group_macro_forged_pixel_f1"
                    ],
                    "validation_unique_authentic_group_macro_pixel_fpr": selected[
                        "unique_authentic_group_macro_pixel_fpr"
                    ],
                    "paper_evidence": False,
                }
            ],
        )
    output = {
        "experiment": config["experiment"],
        "status": "resampling_robust_teacher_training_complete",
        "paper_evidence": False,
        "viewed_development_read": False,
        "unseen_development_read": False,
        "final_reserve_read": False,
        "multi_seed_authorized": multi_seed_run,
        "multi_seed_stability_run": multi_seed_run,
        "checkpoint_selection_used": False,
        "optimizer_steps_completed": expected_steps,
        "silent_failures": 0,
        "holdout_read": False,
        "representation_arm": arm,
        "fixed_final_epoch": int(training["epochs"]),
        "train_pairs": len(pair_cache),
        "validation_records_used": len(validation_cache),
        "validation_unique_authentic_groups": len(authentic_scores),
        "validation_prediction_records": len(prediction_records),
        "validation_source_group_macro_pixel_ap_model_resolution": group_macro_ap,
        "operating_point": selected,
        "pair_cache_hits": cache_hits,
        "protocol_sha256": _sha256(protocol_path),
        "config_sha256": _sha256(config_path),
        "trainer_code_sha256": _sha256(Path(__file__).resolve()),
        "input_manifest_sha256": _sha256(manifest_path),
        "warm_start_checkpoint_sha256": _sha256(warm_path),
        "epochs": epoch_records,
        "checkpoint": _portable_path(checkpoint_path, project_root),
        "checkpoint_sha256": checkpoint_sha256,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "outputs": {
            "epoch_log": str(epoch_log_path.relative_to(project_root)),
            "epoch_log_sha256": _sha256(epoch_log_path),
            "log": str(log_path.relative_to(project_root)),
            **(
                {
                    "prediction_records": str(
                        predictions_path.relative_to(project_root)
                    ),
                    "prediction_records_sha256": _sha256(predictions_path),
                    "metrics": str(metrics_path.relative_to(project_root)),
                    "metrics_sha256": _sha256(metrics_path),
                }
                if predictions_path is not None and metrics_path is not None
                else {}
            ),
        },
    }
    _write_json(summary_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
