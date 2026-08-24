import numpy as np

from pairtrace_doc.pipelines.evaluate_visualdiff_paired_manifest import (
    _auc,
    _select_thresholds,
    _threshold_counts,
)


def test_threshold_counts_match_direct_binary_computation() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 1, 0, 1])
    thresholds = np.asarray([0.0, 0.5, 1.0])
    tp, fp, fn = _threshold_counts(scores, labels, thresholds)
    assert tp.tolist() == [2.0, 1.0, 0.0]
    assert fp.tolist() == [2.0, 1.0, 0.0]
    assert fn.tolist() == [0.0, 1.0, 2.0]


def test_worst_case_failures_and_fpr_gate_affect_threshold_selection() -> None:
    forged = [
        {
            "status": "ok",
            "valid_scores": np.asarray([0.1, 0.8, 0.9]),
            "valid_labels": np.asarray([0, 1, 1]),
            "image_score": 0.9,
        },
        {"status": "failed"},
    ]
    authentic = [
        {
            "status": "ok",
            "valid_scores": np.asarray([0.0, 0.0, 0.1]),
            "valid_labels": np.asarray([0, 0, 0]),
            "image_score": 0.1,
        }
    ]
    selected = _select_thresholds(
        forged,
        authentic,
        {
            "candidate_min": 0.0,
            "candidate_max": 1.0,
            "candidate_step": 0.1,
            "authentic_pixel_fpr_max": 0.01,
            "authentic_image_fpr_max": 0.01,
        },
    )
    assert selected["pixel"]["threshold"] > 0.1
    assert selected["selection_used_test_or_evaluation"] is False


def test_auc_handles_ties() -> None:
    assert _auc(np.asarray([0.9, 0.5, 0.5]), np.asarray([1, 1, 0])) == 0.75
