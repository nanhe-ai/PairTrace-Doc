# Direct Paired-Baseline Protocol

Frozen: 2026-07-25 UTC, before implementation-dependent scoring and before any new evaluation image is decoded.

## Objective

Test PairTrace against methods that receive the same candidate/reference information. This closes the gap left by the AIForge and TFR one-shot tables, whose released TruFor and ADCD-Net baselines are single-image systems.

## Implementation provenance gate

VisualDiff, Localized Document Image Change Detection, Hierarchical Patch Comparison, Text Change Detection, Document Image Verification, and SImProv were verified as real and relevant. At protocol freeze, no attributable, license-compatible public implementation and checkpoint was established for direct insertion into this repository. The paper must not imply otherwise.

The required classical baseline is therefore labelled **VisualDiff-style reimplementation**, not “VisualDiff” and not an official reproduction. It follows the paper's published high-level sequence—SIFT-based alignment followed by dense descriptor comparison—but its engineering choices and code provenance remain ours. If an official implementation is later found, it becomes a separately versioned arm after license and checksum freeze; it does not silently replace this baseline.

## Fixed method inventory

1. `pairtrace_9c_roundtrip`: each of the three frozen PairTrace-9C round-trip checkpoints.
2. `fc_siam_diff_roundtrip`: each of the three frozen FC-Siam-diff round-trip checkpoints.
3. `fc_siam_diff_identity_continuation`: the new same-start/same-budget controls after their training gate passes.
4. `raw_rgb_difference`: registered normalized absolute RGB difference.
5. `ssim_distance`: registered local SSIM distance.
6. `visualdiff_style_dense_sift`: the reimplementation specified below.

The architecture table reports the three learned seeds individually and as a family mean. Nonlearned baselines are deterministic and are not duplicated as pseudo-seeds.

## VisualDiff-style reimplementation

- Convert candidate and reference to grayscale uint8.
- Detect SIFT features with OpenCV using `nfeatures=5000` and otherwise library defaults.
- Match descriptors with two-nearest-neighbor L2 matching and Lowe ratio `0.75`.
- Require at least 12 retained matches; estimate a homography with RANSAC reprojection threshold `4.0` pixels. If the gate fails, emit an item-level failure/fallback record rather than silently omitting the pair.
- Warp the reference to candidate coordinates with bilinear interpolation and reflected borders.
- Place dense SIFT keypoints on an 8-pixel grid with keypoint size 16 pixels in both registered images.
- Compute per-grid cosine distance between corresponding 128-dimensional descriptors, clip to `[0, 1]`, bilinearly interpolate to candidate resolution, and apply a fixed 3x3 median filter.
- Pixels outside mutually valid warped support receive `NaN` for forged-pixel ranking and are tracked separately; document-level scoring also records valid-support fraction.

These values are fixed before the new test is opened. Any failure-driven change requires a dated amendment and a fresh untouched source.

## Common registration and score handling

PairTrace, FC-Siam-diff, raw RGB difference, and SSIM use the existing phase-correlation-initialized homography ECC front end. The VisualDiff-style arm uses its method-native SIFT homography; a secondary common-ECC descriptor result may be reported only as an explicitly labelled diagnostic.

Pixel AP is threshold-free. Each method's authentic pixel-area and image-alert thresholds are selected on the existing AIForge validation partition under its frozen maximum authentic-FPR rule, never on the new evaluation set. Threshold source and hash are written into every item record.

## Stage gates

1. Synthetic translation, perspective, localized edit, and authentic fixtures test SIFT alignment, dense score shape/range, deterministic output, support masking, and failure logging.
2. A three-pair toy run must complete for all deterministic methods.
3. A frozen pilot of 20 new source groups measures registration success, valid support, runtime, and catastrophic authentic false alarms. No model or threshold is selected.
4. Full scoring proceeds only if at least 90% of pilot pairs have >=90% valid support, every failure is recorded, and storage/runtime remain within the preregistered budget. Failure of this gate is a result and terminates full expansion.

## Endpoints

Primary endpoint: source-group-macro pixel AP on forged candidates. Secondary endpoints: authentic pixel-area FPR, authentic image FPR, image AUROC, registration success, valid-support fraction, runtime, and peak memory. Pairwise differences use 5,000 source-group bootstrap resamples with a fixed seed named by the new evaluation protocol.

The conclusion may say a reference helps only after separately distinguishing equal-information paired competitors from single-image operational baselines.
