from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


class VisualDiffAlignmentError(RuntimeError):
    """Raised when the explicitly logged VisualDiff-style alignment gate fails."""

    def __init__(self, reason: str, metadata: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = metadata


@dataclass(frozen=True)
class VisualDiffStyleResult:
    score_map: np.ndarray
    valid_mask: np.ndarray
    registered_reference: np.ndarray
    homography: np.ndarray
    metadata: dict[str, Any]


def _validate_rgb(image: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must be an HWC RGB image")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must use uint8 pixels")
    return array


def _sift_homography(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    nfeatures: int,
    lowe_ratio: float,
    minimum_matches: int,
    ransac_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    candidate_keypoints, candidate_descriptors = sift.detectAndCompute(
        candidate_gray, None
    )
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        reference_gray, None
    )
    metadata: dict[str, Any] = {
        "candidate_keypoints": len(candidate_keypoints),
        "reference_keypoints": len(reference_keypoints),
        "lowe_ratio": lowe_ratio,
        "minimum_matches": minimum_matches,
        "ransac_reprojection_threshold": ransac_threshold,
    }
    if candidate_descriptors is None or reference_descriptors is None:
        raise VisualDiffAlignmentError("sift_descriptors_missing", metadata)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    pairs = matcher.knnMatch(reference_descriptors, candidate_descriptors, k=2)
    retained = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < lowe_ratio * second.distance
    ]
    metadata["ratio_test_matches"] = len(retained)
    if len(retained) < minimum_matches:
        raise VisualDiffAlignmentError("insufficient_ratio_test_matches", metadata)
    source = np.asarray(
        [reference_keypoints[match.queryIdx].pt for match in retained],
        dtype=np.float32,
    )
    destination = np.asarray(
        [candidate_keypoints[match.trainIdx].pt for match in retained],
        dtype=np.float32,
    )
    cv2.setRNGSeed(20260725)
    homography, inliers = cv2.findHomography(
        source, destination, cv2.RANSAC, ransac_threshold
    )
    if homography is None or inliers is None or not np.all(np.isfinite(homography)):
        raise VisualDiffAlignmentError("homography_estimation_failed", metadata)
    inlier_count = int(np.count_nonzero(inliers))
    metadata["ransac_inliers"] = inlier_count
    metadata["ransac_inlier_fraction"] = inlier_count / len(retained)
    determinant = float(np.linalg.det(homography))
    condition = float(np.linalg.cond(homography))
    metadata["homography_determinant"] = determinant
    metadata["homography_condition"] = condition
    if (
        inlier_count < minimum_matches
        or not np.isfinite(condition)
        or abs(determinant) < 1e-10
    ):
        raise VisualDiffAlignmentError("homography_quality_gate_failed", metadata)
    return homography.astype(np.float64), metadata


def _dense_sift_distance(
    candidate: np.ndarray,
    registered_reference: np.ndarray,
    valid_mask: np.ndarray,
    *,
    grid_step: int,
    keypoint_size: int,
) -> tuple[np.ndarray, int]:
    height, width = candidate.shape[:2]
    margin = max(keypoint_size // 2, grid_step)
    xs = list(range(margin, max(margin + 1, width - margin), grid_step))
    ys = list(range(margin, max(margin + 1, height - margin), grid_step))
    if not xs or not ys:
        raise VisualDiffAlignmentError(
            "image_too_small_for_dense_sift",
            {"height": height, "width": width, "margin": margin},
        )
    keypoints = [
        cv2.KeyPoint(float(x), float(y), float(keypoint_size)) for y in ys for x in xs
    ]
    sift = cv2.SIFT_create()
    _, candidate_descriptors = sift.compute(
        cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY), keypoints
    )
    _, reference_descriptors = sift.compute(
        cv2.cvtColor(registered_reference, cv2.COLOR_RGB2GRAY), keypoints
    )
    if (
        candidate_descriptors is None
        or reference_descriptors is None
        or candidate_descriptors.shape != reference_descriptors.shape
        or candidate_descriptors.shape[0] != len(keypoints)
    ):
        raise VisualDiffAlignmentError(
            "dense_sift_descriptor_failure", {"dense_keypoints": len(keypoints)}
        )
    numerator = np.sum(candidate_descriptors * reference_descriptors, axis=1)
    denominator = np.linalg.norm(candidate_descriptors, axis=1) * np.linalg.norm(
        reference_descriptors, axis=1
    )
    cosine = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator, dtype=np.float32),
        where=denominator > 1e-12,
    )
    distance = np.clip(1.0 - cosine, 0.0, 1.0).astype(np.float32)
    grid = distance.reshape(len(ys), len(xs))
    score = cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
    score = cv2.medianBlur(score.astype(np.float32), 3)
    support_kernel = np.ones((keypoint_size + 1, keypoint_size + 1), np.uint8)
    descriptor_support = cv2.erode(
        valid_mask.astype(np.uint8), support_kernel, borderType=cv2.BORDER_CONSTANT
    ).astype(bool)
    score = score.astype(np.float32)
    score[~descriptor_support] = np.nan
    return score, len(keypoints)


def visualdiff_style_score(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    nfeatures: int = 5000,
    lowe_ratio: float = 0.75,
    minimum_matches: int = 12,
    ransac_threshold: float = 4.0,
    grid_step: int = 8,
    keypoint_size: int = 16,
) -> VisualDiffStyleResult:
    """Return a transparent VisualDiff-style score map, not an official reproduction."""

    candidate = _validate_rgb(candidate, "candidate")
    reference = _validate_rgb(reference, "reference")
    if nfeatures != 5000 or lowe_ratio != 0.75 or minimum_matches != 12:
        raise ValueError("VisualDiff-style feature/match policy changed")
    if ransac_threshold != 4.0 or grid_step != 8 or keypoint_size != 16:
        raise ValueError("VisualDiff-style dense-comparison policy changed")
    homography, metadata = _sift_homography(
        candidate,
        reference,
        nfeatures=nfeatures,
        lowe_ratio=lowe_ratio,
        minimum_matches=minimum_matches,
        ransac_threshold=ransac_threshold,
    )
    height, width = candidate.shape[:2]
    registered = cv2.warpPerspective(
        reference,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    support = cv2.warpPerspective(
        np.ones(reference.shape[:2], dtype=np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    score, dense_keypoints = _dense_sift_distance(
        candidate,
        registered,
        support,
        grid_step=grid_step,
        keypoint_size=keypoint_size,
    )
    valid = np.isfinite(score)
    metadata.update(
        {
            "implementation": "pairtrace_visualdiff_style_dense_sift_v1",
            "official_implementation": False,
            "dense_grid_step": grid_step,
            "dense_keypoint_size": keypoint_size,
            "dense_keypoints": dense_keypoints,
            "valid_support_fraction": float(np.mean(valid)),
        }
    )
    return VisualDiffStyleResult(
        score_map=score,
        valid_mask=valid,
        registered_reference=registered,
        homography=homography,
        metadata=metadata,
    )

