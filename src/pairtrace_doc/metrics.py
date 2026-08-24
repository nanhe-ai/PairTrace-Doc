from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryCounts:
    tp: int
    fp: int
    fn: int
    tn: int


def binary_counts(scores: np.ndarray, labels: np.ndarray, threshold: float) -> BinaryCounts:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have identical shapes")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    predictions = scores >= threshold
    positives = labels == 1
    return BinaryCounts(
        tp=int(np.sum(predictions & positives)),
        fp=int(np.sum(predictions & ~positives)),
        fn=int(np.sum(~predictions & positives)),
        tn=int(np.sum(~predictions & ~positives)),
    )


def precision_recall_f1_iou(counts: BinaryCounts) -> dict[str, float]:
    precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if counts.tp + counts.fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = counts.tp + counts.fp + counts.fn
    iou = counts.tp / union if union else 0.0
    return {"pixel_precision": precision, "pixel_recall": recall, "pixel_f1": f1, "pixel_iou": iou}


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have identical shapes")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("average precision requires at least one positive label")

    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    cumulative_tp = np.cumsum(ranked_labels)
    cumulative_fp = np.cumsum(1 - ranked_labels)
    threshold_ends = np.r_[ranked_scores[1:] != ranked_scores[:-1], True]
    tp = cumulative_tp[threshold_ends]
    fp = cumulative_fp[threshold_ends]
    recall = tp / positives
    precision = tp / (tp + fp)
    recall_step = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_step * precision))

