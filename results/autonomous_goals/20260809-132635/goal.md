# Autonomous Goal 20260809-132635

## Title

One-command Featurize execution package and bounded formal EDSR-L ×2/×3/×4 static-depth recovery

## Status

**APPROVED BY THE USER ON 2026-08-09.** The user authorized preparing, testing, committing, and pushing a repository-native Featurize package, then intends to start the cloud instance and invoke it personally.

This is a new Goal. Completed Goal `20260809-112351` remains permanently STOP under its own frozen predicate and must not be edited, resumed, or reinterpreted.

## Objective

Produce a reproducible one-command cloud workflow that:

1. checks the saved Featurize image and already-present datasets without downloading or extracting datasets;
2. downloads only missing official canonical EDSR-L ×2/×3/×4 checkpoints and verifies their frozen byte sizes and full SHA-256 values;
3. constructs physical 32→24 Students using the already-verified endpoint-inclusive ordered transplant;
4. runs exactly 500 FP32 recovery updates for scales ×2/×3/×4 and seeds 0/1/2 with `0.5 L1(Student, Teacher) + 0.5 L1(Student, HR)`;
5. freezes pure final-step Student checkpoints, then evaluates Teacher, zero-step Student, and all recovered seeds on DIV2K validation plus Set5/Set14/BSD100/Urban100/Manga109 using frozen PSNR-Y/SSIM-Y definitions;
6. writes all state and evidence under `/home/featurize/work`, creates and verifies one downloadable archive, and calls `featurize instance release` on success or terminal failure.

## Prospective replacement of the old noisy Gate

The prior random-crop first/last loss comparison is not reused. It remains a binding failure only inside Goal `20260809-112351`. In this Goal, random-crop loss traces are descriptive. Execution stops only for a provenance/data/static-deployment/integrity failure, NaN/Inf, or CUDA OOM. Deterministic full-image before/after quality is reported after checkpoints freeze; it is not used to select a best checkpoint.

## Frozen experiment

The exact contract is `protocol.json`. Key points:

- Teacher: official canonical EDSR-L depth 32, width 256, `res_scale=0.1`;
- Student: physical depth 24 only;
- scales: 2, 3, 4;
- seeds: 0, 1, 2; seed 0 is prospectively the canonical deployment artifact;
- recovery: 500 updates, batch 8, 48×48 LR crop, Adam `1e-5`, cosine schedule, FP32;
- data: only DIV2K training IDs 0001–0760 from the existing DF2K subimage tree;
- checkpoint policy: final step only, no validation/benchmark selection;
- final metrics: full-image, no chop/tiling/self-ensemble, rounded `[0,255]`, scale-pixel shave, frozen Y/SSIM formulas;
- RTX 4090 throughput is descriptive training evidence, not formal deployment latency.

## Artifact ownership

```text
results/autonomous_goals/20260809-132635/
├── START_HERE.md
├── goal.md
├── protocol.json
├── source_manifest.json
├── progress.md
├── evidence.md
├── decision_log.md
├── final_report.md
├── executed_source/
│   ├── formal_recovery.py
│   └── run_featurize.sh
├── remote_artifacts/
└── subagents/
```

Large checkpoints, rolling optimizer states, and cloud output remain ignored under `/home/featurize/work`; only the verified result bundle is copied back before final scientific judgment.

## Phases

1. Implement and locally self-test the package.
2. Run a no-cost RTX 4060 smoke test without opening final benchmarks.
3. Independently review code, protocol, persistence, and release safety.
4. Commit and push only the new Goal/package and maintained state documentation.
5. User starts a Featurize RTX 4090 instance, pulls the commit, and invokes the one-command launcher.
6. Copy back and hash-verify the bundle; independently audit empirical facts.
7. Rerun frozen Student latency on the common RTX 4060 stack before any paper latency claim.

## Explicit non-goals

- No 120k-step conventional retraining.
- No d28 recovery in this first formal run.
- No use of old dynamic RCAN cloud launchers.
- No dataset download or extraction.
- No 4090 latency mixed into the 4060 deployment table.
- No paper-result promotion before copy-back and review.
