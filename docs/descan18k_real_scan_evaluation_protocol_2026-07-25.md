# Prospective DESCAN-18K Real-Scan Paired Evaluation Protocol

Frozen: 2026-07-25 UTC. At freeze time no DESCAN archive bytes had been downloaded locally and no dataset image had been decoded or visually inspected for this study. Only public repository metadata, the dataset card, and the paper were read.

## Scientific purpose

Create a never-viewed, marker-free evaluation on real print--scan imagery. This addresses the absence of a prospective real-acquisition paired test without claiming that the deterministic local edits are naturally occurring fraud.

Two questions are kept separate:

1. Can paired methods localize deterministic edits on document imagery that has genuinely passed through printing and scanning?
2. What happens when the authentic reference is the aligned born-digital original while the candidate is its real scan, i.e. under global acquisition mismatch?

The second question is a hard boundary test and may fail. It is not used to tune a method.

## Source identity

- Dataset: DESCAN-18K, official release by the authors of “Descanning: From Scanned to the Original Images with a Color Correction Diffusion Model.”
- Repository: `ENCLab/DESCAN-18K` on Hugging Face.
- Repository revision observed before freeze: `b5617e5ef9217116daa0ed2740394cc1ef2e03d7`.
- Candidate archive: `Test.zip`.
- Reported archive size: 767,709,023 bytes.
- Reported Git-LFS SHA-256: `93c904410f409c1a393e5e79b35bf1748cb5d0ad2d18d1e1958d32213a9d8304`.
- Reported test population: 360 aligned scan/original pairs from scanner 1 and scanner 2, which are excluded from the dataset's train/validation scanners.
- Reported format: same-named 1024x1024 RGB TIFF files under `Test/scan` and `Test/clean`.

## License gate — currently closed

The official repository and paper state that the 11 Raspberry Pi Foundation magazine sources are CC BY-NC-SA 3.0 and present the release for research evaluation. However, the Hugging Face repository currently has no explicit dataset-license metadata, and the official GitHub repository has no license file or operative dataset-license text after its “License” table-of-contents entry.

Therefore archive download and image decoding are prohibited until one of the following is recorded:

- written clarification from the dataset authors that research evaluation and creation of nonredistributed derived masks/predictions are permitted; or
- documented institutional approval that the source license plus official release is sufficient for this noncommercial research use.

No dataset image, altered image, or crop will be redistributed. Only filenames or pseudonymous group IDs, checksums, transformation parameters, masks if legally redistributable, predictions, metrics, and code are candidates for release. A draft clarification request is stored separately; this protocol does not authorize sending it.

## Frozen population and stages after the license gate opens

After archive checksum verification, archive member names may be listed without decoding pixels. Pair membership is the exact basename intersection of `Test/scan` and `Test/clean`. Any duplicate basename, missing mate, corrupt member, dimension mismatch, or unexpected file type is logged and excluded by rule rather than discretion.

Groups are ordered by SHA-256 of `"pairtrace-descan-20260725:" + basename`.

- toy: first 3 eligible groups;
- pilot: first 20 eligible groups, inclusive of toy;
- full: all eligible groups, expected 360.

No image quality or method score affects membership or order.

## Candidate/reference conditions

For each group, create two deterministic forged scan candidates and retain an authentic scan candidate:

- `copy_move`: copy a texture/text-containing source patch to a nonoverlapping destination selected by the frozen deterministic placement algorithm;
- `local_erase`: replace a selected high-gradient connected region with a local background estimate.

The edit generator receives only the scan image and a seed derived from the group ID. It may not query any detector under evaluation. The exact changed-pixel mask is `any(candidate_rgb != original_scan_rgb)` after materialization; empty masks, excessive masks, invalid placement, or modifications outside the declared destination are failures. There is no colored boundary marker or other annotation written into the candidate.

Each candidate is scored in two reference conditions:

- `scan_reference`: reference is the unmodified real scan. This tests marker-free tamper localization on real-scanned content with same-rendering correspondence.
- `digital_reference`: reference is the aligned clean digital mate. This tests real print--scan acquisition mismatch.

The authentic controls are the unmodified scan paired with itself and with its clean digital mate, respectively. The identical scan/self control is acknowledged as an easy correspondence anchor and is never described as independent acquisition.

## Registration and admission diagnostics

Before model scoring, record dimensions, mutual valid support, phase-correlation shift, ECC convergence, homography determinant/condition, SSIM, normalized RGB difference, and edge overlap for each scan/digital pair. These measurements do not remove hard cases. Pilot expansion requires:

- all 20 group pairs decode and bind correctly;
- at least 90% of scan/digital registrations have >=90% mutual valid support;
- each attack succeeds for at least 18/20 pilot groups;
- no green boundary is introduced by the editor (byte-level color test plus visual audit of the three toy cases);
- every failure has an item record and reason;
- estimated full storage remains below 5 GB beyond the archive.

If the gate fails, full expansion stops and the failure is reported. Thresholds and model selection never use pilot or full scores.

## Frozen methods

- PairTrace-9C round-trip, three seeds.
- FC-Siam-diff round-trip, three seeds.
- FC-Siam-diff identity continuation, three seeds, if its same-budget training gate completes.
- registered raw normalized RGB difference.
- registered SSIM distance.
- VisualDiff-style dense-SIFT reimplementation after its unit gate completes.

If a method is unavailable at launch, the registry records it as blocked; it may not be added after any full-set score is read without declaring the addition post-hoc.

## Endpoints and uncertainty

Primary endpoint: source-group-macro exact changed-pixel AP on forged candidates, reported separately for `scan_reference` and `digital_reference`. Secondary endpoints: attack-macro AP, scanner-macro AP, authentic pixel-area FPR, authentic image FPR, image AUROC, registration success/support, and failure counts. No reference condition is averaged into a single headline number.

Pairwise effects use 5,000 bootstrap resamples over source groups with seed `20260725`. Learned-model seed families are reported without best-seed selection. All raw score maps or cache keys, item predictions, transformation parameters, failures, and aggregate tables are retained.

## Claim boundary

Passing this protocol would establish marker-free evaluation on real-scanned source imagery and an explicit real scan/digital-reference stress test. It would not establish naturally authored fraud, independent human editing, reliable cross-device deployment, or a calibrated document-level validator.
