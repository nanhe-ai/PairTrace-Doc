# DESCAN-18K Deterministic Edit-Generator Specification

Frozen: 2026-07-25 UTC, while the DESCAN-18K license gate was closed and before any archive byte or dataset image was accessed locally.

This document resolves the implementation choices left abstract in the prospective real-scan protocol. The generator never queries a detector, threshold, checkpoint, or evaluation score.

## Common rules

- Input is one uint8 RGB scan with both dimensions at least 128 pixels.
- The item seed is the first 64 bits of SHA-256 of `pairtrace-descan-edit-v1:` followed by the source-group ID. It is recorded even though version 1 uses hash-based tie breaking rather than a mutable random-number stream.
- The base patch side is the nearest multiple of eight to 10% of the shorter image side, clipped to `[32, 128]`.
- Candidate patch origins form a row-major grid with a one-patch margin and stride `max(8, patch_side // 2)`. Ranking ties are broken by SHA-256 of the group ID, attack name, and integer patch coordinates.
- Gradient magnitude is the Euclidean magnitude of 3x3 Sobel derivatives of grayscale RGB converted by OpenCV.
- The authoritative mask is computed only after editing as `any(candidate_rgb != input_scan_rgb, axis=2)`. It must be nonempty, must remain inside the declared destination, and must contain no pixels introduced merely for visualization.
- The output retains the original dimensions and uint8 RGB type. Empty or invalid edits are explicit item failures, never silent omissions.

## Copy-move

1. Rank source windows by descending mean gradient magnitude and keep the first valid window.
2. Consider destination windows that do not intersect the source window, including an eight-pixel exclusion gap.
3. Rank destinations by descending mean absolute RGB distance between the source patch and the destination patch. This makes the deterministic edit observable without examining any detector.
4. Copy the source pixels exactly into the destination. No border, feathering, color marker, or annotation is rendered.
5. Require at least 25% of the destination patch pixels to change.

The declared destination is the copied rectangle; the source region is not part of the positive mask because it is unchanged.

## Local erase

1. Rank windows by descending mean gradient magnitude and select the first valid window.
2. Within that window, threshold gradient magnitude at its 70th percentile, close the binary map with a 3x3 ellipse, retain the largest connected component, and dilate it with a 5x5 ellipse.
3. Estimate background independently for each RGB channel as the median of a five-pixel ring surrounding the component inside the selected window.
4. Replace only component pixels by the three-channel median. No outline, blending mask, or annotation is rendered.
5. Require at least 32 changed pixels and at least 25% of selected component pixels to change. This lower bound was fixed after the synthetic unit fixture showed that dilation intentionally includes local background pixels; no DESCAN byte had been accessed. If a selected window fails, continue through the frozen rank order; if all fail, emit a failure.

The declared destination is the selected window and the exact changed-pixel mask is generally nonrectangular.

## Versioning boundary

The implementation identifier is `pairtrace_descan_edit_generator_v1`. Its source checksum, this specification checksum, OpenCV version, and NumPy version must be recorded in the materialization summary. Any algorithm change after a real DESCAN image is decoded requires a new version, a dated protocol amendment, and a fresh untouched source population.
