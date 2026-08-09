# Decision Log — Goal 20260809-132635

## 2026-08-09 — New Goal rather than relaxing old evidence

Goal `20260809-112351` remains STOP. The user's new authorization is represented by this separate Goal with a prospectively different continuation rule.

## 2026-08-09 — Bounded formal default

Use d24 only, scales ×2/×3/×4, seeds 0/1/2, and exactly 500 FP32 updates. This matches the project's established fixed-budget recovery method and does not imply conventional 120k-step retraining.

## 2026-08-09 — Cloud-compatible recovery geometry

Freeze batch 8 and a 48×48 LR crop. Batch 8 matches the established bounded MVP; 48×48 matches the validated EDSR pilot and existing Featurize subimage geometry. This first EDSR run is not claimed to have the same LR-pixel budget as the historical RCAN 64×64 run.

## 2026-08-09 — Fatal-only continuation

Random crop losses cannot form a matched stability comparison. Their windows remain descriptive. Only provenance/data/static-artifact/integrity failures, NaN/Inf, or OOM terminate this Goal.

## 2026-08-09 — No silent inference fallback

Use full-image FP32 evaluation without chop, tiling, or self-ensemble. An OOM is preserved as a failure; no fallback is introduced after viewing results.

## 2026-08-09 — Release safety

All generated state goes directly to `/home/featurize/work`. Success and failure paths require a durable status plus a structured bundle whose internal members and sibling SHA are verified before `featurize instance release`. If that verification fails, a raw partial archive is attempted but the instance is deliberately left running with a billing warning for manual inspection. A shutdown fallback after repeated release-command failure is labeled billing-unsafe.

## 2026-08-09 — Final source freeze after review fixes

The first source freeze was exercised by smoke attempt 01. Independent review then required stronger completed-run/evaluation binding, exhaustive data checks, portable checksums, signal forwarding, and exact success-bundle membership. The corrected final freeze is protocol SHA `ae05cb0a...9ce4` and source-manifest SHA `9b5bde66...e51d`; exact-source smoke attempt 02 and final independent review both pass.
