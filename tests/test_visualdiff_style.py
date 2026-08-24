import cv2
import numpy as np
import pytest

from pairtrace_doc.baselines.visualdiff_style import (
    VisualDiffAlignmentError,
    visualdiff_style_score,
)


def _document_fixture() -> np.ndarray:
    image = np.full((256, 320, 3), 245, dtype=np.uint8)
    for index in range(9):
        y = 28 + 23 * index
        cv2.putText(
            image,
            f"Invoice {index} 12345 ABC",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (10 + index, 10 + index, 10 + index),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(image, (230, 20), (300, 75), (30, 30, 30), 2)
    cv2.line(image, (15, 235), (305, 235), (40, 40, 40), 2)
    return image


def test_visualdiff_style_is_deterministic_and_localizes_an_edit() -> None:
    reference = _document_fixture()
    candidate = reference.copy()
    candidate[105:135, 185:255] = 245
    first = visualdiff_style_score(candidate, reference)
    second = visualdiff_style_score(candidate, reference)
    assert np.allclose(first.homography, second.homography)
    assert np.array_equal(first.valid_mask, second.valid_mask)
    assert np.allclose(first.score_map, second.score_map, equal_nan=True)
    assert first.score_map.shape == candidate.shape[:2]
    assert first.metadata["official_implementation"] is False
    assert first.metadata["ratio_test_matches"] >= 12
    assert first.metadata["valid_support_fraction"] > 0.8
    edit_score = float(np.nanmean(first.score_map[105:135, 185:255]))
    background_score = float(np.nanmean(first.score_map[150:210, 20:100]))
    assert edit_score > background_score


def test_visualdiff_style_records_alignment_failure() -> None:
    blank = np.full((128, 160, 3), 255, dtype=np.uint8)
    with pytest.raises(VisualDiffAlignmentError) as error:
        visualdiff_style_score(blank, blank)
    assert error.value.reason == "sift_descriptors_missing"
    assert error.value.metadata["candidate_keypoints"] == 0
