Turn budget wrap-up was requested after 4 assistant turns (soft limit 4, grace 1). Process-mode live steering is unavailable, so the child was warned at launch to wrap up by this budget. Output may be partial.

## Review

- **Blocker:** Frozen source manifest is stale. Every listed file mismatches its recorded SHA-256, including `results/autonomous_goals/20260809-132635/source_manifest.json` entries for `protocol.json`, `goal.md`, `START_HERE.md`, `executed_source/formal_recovery.py`, and `executed_source/run_featurize.sh`. `formal_recovery.py:135-140` rejects these mismatches, so the cloud preflight, training, evaluation, and structured bundling cannot run.
- **Blocker:** Output-location requirements are defaults, not enforced:
  - `formal_recovery.py:1292-1296` accepts any `--experiment-dir`, including paths outside repository `experiments/`.
  - `run_featurize.sh:184-186` only requires `RUN_ROOT` under `/home/featurize/work`, not under `$REPO_ROOT/experiments`.
  - Therefore environment overrides can violate both explicit persistence/location requirements despite the documented defaults being correct.
- **Note:** `scripts/eval/test_edsr_checkpoint.py:1-11` is not represented in `source_manifest.json`; consequently the promised standalone interface is not source-frozen or included in the verified source snapshot/archive. Its underlying evaluator is frozen, but the public wrapper is not attested.
- **Correct:** Architecture inference obtains contiguous physical depth and feature width, then strict-loading validates the complete scale-specific topology (`formal_recovery.py:1248-1259`, `1303-1308`). A wrong scale or incompatible tensor layout is rejected.
- **Correct:** Canonical and explicit datasets use normalized unique stem pairing with exact HR/LR key-set equality. Geometry is checked before metrics, with only bounded standard top-left modcrop allowed.
- **Correct:** Evaluation follows the frozen metric contract: rounded/clamped uint8 output, scale-pixel shave, specified Y conversion, per-image PSNR-Y/SSIM-Y, and arithmetic aggregate means.
- **Correct:** The standalone command emits per-image JSONL/CSV, human-readable per-image and aggregate logs, and a structured summary (`formal_recovery.py:1312-1399`); collision protection refuses nonempty target directories.
- **Correct:** The default formal run root is now repository-local and ignored by Git (`run_featurize.sh:13`; `.gitignore:21`). Structured archives verify member bytes and SHA-256 before release eligibility.
- **Residual risk:** Runtime testing was unavailable locally because PyTorch is not installed. Existing RTX 4060 smoke evidence predates this correction and does not exercise the new standalone command.

**NOT READY TO FREEZE**