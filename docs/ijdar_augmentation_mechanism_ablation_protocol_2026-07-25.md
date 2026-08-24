# Reference-Augmentation Mechanism Ablation Protocol

Frozen: 2026-07-25 UTC, before training any new ablation and before reading any new evaluation image or score.

## Question

Is the observed projective-stress gain specific to a reference-only warp/inverse-warp round trip, or can it be explained by extra optimization, generic interpolation, a single misalignment, reference degradation, or perturbing a different member of the pair?

## Common start and budget

All learned ablations use PairTrace-9C and seeds `20260747`, `20260763`, and `20260764`. Each starts from the corresponding frozen 3,000-step clean PairTrace-9C checkpoint and receives exactly 3,000 continuation steps. Data, crop sampling, matched brightness/contrast and JPEG policy already present in the base recipe, optimizer, scheduler, batch size, AMP, validation, and fixed-final-epoch selection are identical.

The existing identity and reference-round-trip continuations may be reused only if their hashes and recipes pass the new registry audit. New arms use new output paths and never overwrite frozen artifacts.

## Fixed ablation arms

| Arm | Perturbed image(s) | Operation | Purpose |
|---|---|---|---|
| `identity` | none | no geometric/resampling change | same-start extra-step control |
| `reference_roundtrip` | reference | draw translation/affine/perspective, warp, then inverse-warp | proposed mechanism |
| `reference_resize_restore` | reference | resize by a random factor in `[0.94, 1.06]`, then resize back to the fixed crop | generic resampling without a projective round trip |
| `reference_single_warp` | reference | one translation/affine/perspective warp with no inverse | distinguish interpolation exposure from learning under residual misalignment |
| `reference_blur` | reference | Gaussian blur with sigma drawn uniformly from `[0.5, 2.0]` | photometric/degradation control |
| `reference_jpeg` | reference | JPEG recompression with integer quality drawn from `[50, 90]` | compression control |
| `candidate_roundtrip` | candidate | the same warp/inverse-warp family applied only to the candidate | test side specificity while preserving target coordinates |
| `joint_roundtrip` | candidate and reference | identical sampled warp/inverse-warp applied separately to both | test whether shared resampling is sufficient |

For all non-identity arms, perturbation is applied with probability 0.75 and identity with probability 0.25. Bilinear interpolation and `BORDER_REFLECT_101` are fixed for geometric and resize operations. The candidate and mask coordinate system is never forward-warped; candidate-only and joint arms use round trips so the supervision remains in the original coordinate frame. JPEG and blur in this table are additional asymmetric operations, distinct from the matched mild preprocessing already applied to both pair members.

## Stage gates

1. Deterministic synthetic-image unit tests check shape, seed repeatability, side specificity, identity probability, mask invariance, and that non-identity operations alter their intended image.
2. Static config tests require a common warm start, identical 3,000-step budget, distinct output paths, and exactly one arm-specific augmentation policy.
3. GPU preflight uses 100 pairs and 100 steps for `reference_resize_restore`, `reference_single_warp`, `reference_blur`, `reference_jpeg`, `candidate_roundtrip`, and `joint_roundtrip` with seed `20260747`.
4. Pilot trains all eight arms for seed `20260747`. Required: finite losses, complete checkpoints and validation outputs, zero silent failures, and resource/cost report.
5. The other two seeds run only after the pilot execution gate passes. Validation ranking cannot remove an arm.
6. Confirmatory scoring is restricted to the new never-viewed paired evaluation frozen separately.

## Endpoints and claim rule

Primary contrasts are `reference_roundtrip - identity` and `reference_roundtrip - each alternative` under the three in-range projective stresses, using source-group paired bootstrap. Secondary endpoints are clean AP, blur/JPEG/nonrigid AP, authentic pixel-area FPR, and authentic image FPR.

The mechanism claim may be strengthened only if reference round trips outperform the identity control under projective stresses without an unacceptable authentic-FPR regression and the pattern is not matched by every generic degradation arm. Blur, JPEG, and nonrigid results remain out-of-range boundary tests unless their dedicated arms are explicitly shown to generalize on untouched data. No “general robustness” wording is permitted.
