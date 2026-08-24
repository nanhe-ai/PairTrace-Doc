# Same-Start, Same-Budget Architecture Completion Protocol

Frozen: 2026-07-25 UTC, before training any newly requested continuation and before accessing any new evaluation images or scores.

## Question

Does reference-only round-trip continuation improve projective-resampling robustness within each paired-localizer architecture when the comparison uses the same clean starting checkpoint and exactly the same additional optimization budget?

The previous TFR-authentic-120 study cannot answer this across architectures because its non-9C clean checkpoints stop after 3,000 steps while their round-trip variants receive another 3,000 steps. Only PairTrace-9C has an identity continuation. This protocol fills that control without changing or overwriting the old study.

## Fixed architecture and seed family

The new identity continuations cover:

- `signed_difference_3ch` (`signed_diff`)
- `absolute_difference_3ch` (`absolute_diff`)
- `candidate_reference_6ch` (`6ch`)
- `fc_siam_diff` (`fc_siam_diff`)
- `fc_siam_conc` (`fc_siam_conc`)

PairTrace-9C (`explicit_9ch`) uses its already completed identity continuations as the reference implementation. Seeds are fixed to `20260747`, `20260763`, and `20260764`. Every new run starts from the corresponding frozen `pairtrace_confirmatory_clean_<slug>_seed<seed>.pt` checkpoint. The exact path/digest inventory is stored in `outputs/manifests/ijdar_same_budget_warm_starts_20260725.json`.

## Only permitted recipe difference

For a given architecture and seed, the old round-trip continuation and new identity continuation must match in training data, sampling, preprocessing, optimizer, learning rate, scheduler, batch size, number of workers, AMP setting, gradient clipping, six epochs, 500 steps per epoch, fixed-final-epoch selection, and validation procedure.

The only permitted semantic difference is:

- round trip: geometry probabilities `{clean: 0.25, translation: 0.25, affine: 0.25, perspective: 0.25}` and `transform_application: reference_roundtrip_only`;
- identity continuation: geometry probabilities `{clean: 1.0, translation: 0.0, affine: 0.0, perspective: 0.0}` and `transform_application: reference_clean_identity_only`.

Architecture labels and output paths may differ. Optimizer state is reinitialized for both continuation regimes, as in the frozen training code; model weights are bound by SHA-256.

## Stage gates

1. **Static gate:** config merge, warm-start binding, step-budget equality, augmentation-only recipe difference, non-overwriting paths, and no-holdout flags pass unit tests.
2. **Toy CPU gate:** synthetic tensors exercise every representation and checkpoint-binding validator. No scientific metric is produced.
3. **GPU preflight:** one seed and one architecture, 100 pairs and 100 optimizer steps. Required: finite loss, exact step count, checkpoint reload, zero silent failures, peak device memory recorded, and validation output complete.
4. **Pilot:** all five architectures for seed `20260747`, full 3,000-step identity continuation. Required: all artifacts complete, no NaN/Inf, no omitted sample, and validation operating point satisfies authentic pixel FPR <= 0.01. Pilot validation may diagnose execution only and cannot choose an architecture.
5. **Full training:** remaining two seeds launch only after the pilot gate passes. All 15 identity continuations are retained regardless of validation ranking.
6. **Confirmatory scoring:** scoring is allowed only on a separately frozen, never-viewed evaluation source. The already opened TFR-authentic-120 set may be used for clearly labelled post-hoc diagnostics but cannot establish the repaired causal claim.

## Endpoints and inference

The primary within-architecture endpoint is the source-group paired difference in forged-image pixel AP, `roundtrip - identity_continuation`, under each preregistered projective stress. Secondary endpoints are clean AP, authentic pixel-area FPR at validation-frozen threshold, authentic image FPR, and the minimum stress AP. Results are reported for each seed and as a seed-family mean; no seed is selected.

Group bootstrap uses 5,000 resamples with a fixed seed recorded in the new evaluation protocol. A claim of improvement requires a positive mean paired difference and a 95% interval excluding zero for at least one stressed condition, without a predeclared unacceptable clean-AP or authentic-FPR regression. Exact acceptance margins belong to the new evaluation protocol because they depend on its mask semantics and group count.

## Evidence boundary

Until Gate 6 completes, the paper must say:

> PairTrace-9C has same-start causal evidence. Cross-architecture round-trip comparisons are descriptive and training-length-confounded.

Training completion alone does not repair the paper claim. Failed gates and partial runs remain logged and are not silently omitted.
