from __future__ import annotations

import cv2
import numpy as np
import pytest

from pairtrace_doc.descan_edits import (
    IMPLEMENTATION_ID,
    copy_move_edit,
    generate_descan_edits,
    local_erase_edit,
)


def _document_fixture() -> np.ndarray:
    image = np.full((320, 416, 3), 242, dtype=np.uint8)
    cv2.rectangle(image, (24, 22), (392, 296), (18, 28, 38), 3)
    for index, y in enumerate(range(48, 280, 22)):
        cv2.putText(
            image,
            f"INVOICE {index:02d}  1234567890",
            (38, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (25 + index, 35, 55),
            1,
            cv2.LINE_AA,
        )
    cv2.circle(image, (340, 230), 28, (130, 50, 25), 3)
    return image


@pytest.mark.parametrize("factory", [copy_move_edit, local_erase_edit])
def test_descan_edit_is_deterministic_exact_and_unmarked(factory) -> None:
    scan = _document_fixture()
    first = factory(scan, "fixture-001")
    second = factory(scan, "fixture-001")
    assert np.array_equal(first.candidate, second.candidate)
    assert np.array_equal(first.mask, second.mask)
    assert first.metadata == second.metadata
    assert first.metadata["implementation"] == IMPLEMENTATION_ID
    assert np.array_equal(first.mask, np.any(first.candidate != scan, axis=2))
    assert first.mask.any()
    x, y, width, height = first.metadata["destination_xywh"]
    allowed = np.zeros(first.mask.shape, dtype=bool)
    allowed[y : y + height, x : x + width] = True
    assert not np.any(first.mask & ~allowed)
    introduced_exact_green = np.all(first.candidate == (0, 255, 0), axis=2) & ~np.all(
        scan == (0, 255, 0), axis=2
    )
    assert not introduced_exact_green.any()


def test_both_attacks_are_emitted_with_distinct_masks() -> None:
    results = generate_descan_edits(_document_fixture(), "fixture-002")
    assert set(results) == {"copy_move", "local_erase"}
    assert not np.array_equal(results["copy_move"].mask, results["local_erase"].mask)


def test_scan_validation_rejects_too_small_or_float_input() -> None:
    with pytest.raises(ValueError, match="at least 128"):
        copy_move_edit(np.zeros((64, 64, 3), dtype=np.uint8), "small")
    with pytest.raises(ValueError, match="uint8"):
        local_erase_edit(np.zeros((256, 256, 3), dtype=np.float32), "float")
