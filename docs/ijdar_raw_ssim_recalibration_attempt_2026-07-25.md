# Terminated Raw/SSIM Recalibration Attempt

Date: 2026-07-25 UTC.

The direct-baseline revision initially launched a full CPU replay of the existing AIForge development-100 raw-difference and SSIM calibration. The process exited without a Python exception after completing 5/100 groups. Its JSONL contains 20/400 planned method/item records and the log reports zero failures in those five groups. No summary, threshold, test score, or paper evidence was produced by this incomplete run, and it must not be cited.

Preserved artifacts:

- config SHA-256: `ac79560fc2c6b7a5a13a448178036132e5af7bc8d130bc91a955d3957caf280f`;
- partial predictions SHA-256: `3ca1e7dfe544389bd481293498a2d0e13a4fcbf856616e8b6e397953f759a1f1`;
- log SHA-256: `1bff8b867ef090207e9f97e9ab775537bbbf46e7ae2c682e3bf77cbf4035bacf`.

The accepted path is instead the explicit threshold-adoption gate in `outputs/tables/ijdar_raw_ssim_threshold_adoption_summary_20260725.json`. That gate verifies the already complete 100-group development run and its predictions/metrics, records that the old experiment did not authorize unscoped transfer, and binds the new direct-baseline protocol's authorization before any new evaluation is opened.
