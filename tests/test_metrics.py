import numpy as np

from pairtrace_doc.metrics import average_precision, binary_counts, precision_recall_f1_iou


def test_average_precision_is_one_for_perfect_ranking() -> None:
    score = average_precision(np.array([0.9, 0.8, 0.2]), np.array([1, 1, 0]))
    assert score == 1.0


def test_binary_metrics() -> None:
    counts = binary_counts(np.array([0.9, 0.4, 0.7, 0.1]), np.array([1, 1, 0, 0]), 0.5)
    metrics = precision_recall_f1_iou(counts)
    assert counts.tp == 1
    assert counts.fp == 1
    assert counts.fn == 1
    assert metrics["pixel_f1"] == 0.5
    assert metrics["pixel_iou"] == 1 / 3

