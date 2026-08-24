from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


IMPLEMENTATION_ID = "pairtrace_descan_edit_generator_v1"


class DescanEditError(RuntimeError):
    """Raised when a frozen deterministic edit cannot be materialized."""


@dataclass(frozen=True)
class DescanEditResult:
    candidate: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any]


def _validate_scan(scan: np.ndarray) -> np.ndarray:
    value = np.asarray(scan)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("scan must be an HWC uint8 RGB array")
    if min(value.shape[:2]) < 128:
        raise ValueError("scan dimensions must both be at least 128 pixels")
    return value


def _seed(group_id: str) -> int:
    payload = f"pairtrace-descan-edit-v1:{group_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _tie(group_id: str, attack: str, y: int, x: int) -> str:
    payload = f"{group_id}:{attack}:{y}:{x}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _patch_side(shape: tuple[int, int]) -> int:
    raw = int(round((0.10 * min(shape)) / 8.0) * 8)
    return max(32, min(128, raw))


def _origins(height: int, width: int, side: int) -> list[tuple[int, int]]:
    stride = max(8, side // 2)
    ys = list(range(side, height - 2 * side + 1, stride))
    xs = list(range(side, width - 2 * side + 1, stride))
    if not ys or not xs:
        ys = list(range(0, height - side + 1, stride))
        xs = list(range(0, width - side + 1, stride))
    origins = [(y, x) for y in ys for x in xs]
    if not origins:
        raise DescanEditError("no_valid_patch_origin")
    return origins


def _gradient(scan: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(scan, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(dx, dy)


def _window_rank(
    gradient: np.ndarray,
    origins: list[tuple[int, int]],
    side: int,
    group_id: str,
    attack: str,
) -> list[tuple[int, int]]:
    return sorted(
        origins,
        key=lambda point: (
            -float(
                gradient[
                    point[0] : point[0] + side,
                    point[1] : point[1] + side,
                ].mean()
            ),
            _tie(group_id, attack, *point),
        ),
    )


def _rectangles_separated(
    first: tuple[int, int], second: tuple[int, int], side: int, gap: int
) -> bool:
    ay, ax = first
    by, bx = second
    return (
        ax + side + gap <= bx
        or bx + side + gap <= ax
        or ay + side + gap <= by
        or by + side + gap <= ay
    )


def _result(
    scan: np.ndarray,
    candidate: np.ndarray,
    *,
    group_id: str,
    attack: str,
    destination: tuple[int, int, int, int],
    metadata: dict[str, Any],
) -> DescanEditResult:
    mask = np.any(candidate != scan, axis=2)
    if not mask.any():
        raise DescanEditError(f"{attack}_empty_changed_pixel_mask")
    y, x, height, width = destination
    allowed = np.zeros(mask.shape, dtype=bool)
    allowed[y : y + height, x : x + width] = True
    if np.any(mask & ~allowed):
        raise DescanEditError(f"{attack}_changed_pixels_outside_destination")
    return DescanEditResult(
        candidate=candidate,
        mask=mask,
        metadata={
            "implementation": IMPLEMENTATION_ID,
            "group_id": group_id,
            "attack": attack,
            "seed": _seed(group_id),
            "destination_xywh": [x, y, width, height],
            "changed_pixels": int(mask.sum()),
            "changed_fraction": float(mask.mean()),
            **metadata,
        },
    )


def copy_move_edit(scan: np.ndarray, group_id: str) -> DescanEditResult:
    scan = _validate_scan(scan)
    height, width = scan.shape[:2]
    side = _patch_side((height, width))
    origins = _origins(height, width, side)
    ranked_sources = _window_rank(
        _gradient(scan), origins, side, group_id, "copy_move_source"
    )
    for source_y, source_x in ranked_sources:
        source = scan[source_y : source_y + side, source_x : source_x + side]
        destinations = [
            point
            for point in origins
            if _rectangles_separated((source_y, source_x), point, side, 8)
        ]
        destinations.sort(
            key=lambda point: (
                -float(
                    np.mean(
                        np.abs(
                            source.astype(np.int16)
                            - scan[
                                point[0] : point[0] + side,
                                point[1] : point[1] + side,
                            ].astype(np.int16)
                        )
                    )
                ),
                _tie(group_id, "copy_move_destination", *point),
            )
        )
        for destination_y, destination_x in destinations:
            candidate = scan.copy()
            candidate[
                destination_y : destination_y + side,
                destination_x : destination_x + side,
            ] = source
            local_changed = np.any(candidate != scan, axis=2)[
                destination_y : destination_y + side,
                destination_x : destination_x + side,
            ]
            if float(local_changed.mean()) < 0.25:
                continue
            return _result(
                scan,
                candidate,
                group_id=group_id,
                attack="copy_move",
                destination=(destination_y, destination_x, side, side),
                metadata={
                    "patch_side": side,
                    "source_xywh": [source_x, source_y, side, side],
                    "minimum_changed_patch_fraction": 0.25,
                },
            )
    raise DescanEditError("copy_move_no_observable_nonoverlapping_destination")


def local_erase_edit(scan: np.ndarray, group_id: str) -> DescanEditResult:
    scan = _validate_scan(scan)
    height, width = scan.shape[:2]
    side = _patch_side((height, width))
    gradient = _gradient(scan)
    ranked = _window_rank(
        gradient, _origins(height, width, side), side, group_id, "local_erase"
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for y, x in ranked:
        local_gradient = gradient[y : y + side, x : x + side]
        threshold = float(np.percentile(local_gradient, 70.0))
        binary = (local_gradient >= threshold).astype(np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        if count <= 1:
            continue
        component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = (labels == component_index).astype(np.uint8)
        component = cv2.dilate(component, dilate_kernel).astype(bool)
        if int(component.sum()) < 32:
            continue
        ring = cv2.dilate(component.astype(np.uint8), ring_kernel).astype(bool)
        ring &= ~component
        if int(ring.sum()) < 16:
            continue
        local_scan = scan[y : y + side, x : x + side]
        background = np.median(local_scan[ring], axis=0).round().astype(np.uint8)
        candidate = scan.copy()
        local_candidate = candidate[y : y + side, x : x + side]
        local_candidate[component] = background
        changed = np.any(local_candidate != local_scan, axis=2)
        if int(changed.sum()) < 32 or float(changed[component].mean()) < 0.25:
            continue
        return _result(
            scan,
            candidate,
            group_id=group_id,
            attack="local_erase",
            destination=(y, x, side, side),
            metadata={
                "patch_side": side,
                "gradient_percentile": 70.0,
                "component_pixels": int(component.sum()),
                "background_rgb": background.tolist(),
                "minimum_changed_pixels": 32,
                "minimum_component_changed_fraction": 0.25,
            },
        )
    raise DescanEditError("local_erase_no_valid_high_gradient_component")


def generate_descan_edits(
    scan: np.ndarray, group_id: str
) -> dict[str, DescanEditResult]:
    return {
        "copy_move": copy_move_edit(scan, group_id),
        "local_erase": local_erase_edit(scan, group_id),
    }
