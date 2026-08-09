Inherited decisions:
- `20260809-112351` remains permanently **STOP** under its frozen noisy loss predicate; it must not be edited, resumed, or reinterpreted.
- The user has now authorized a **new Goal** for packaging/pushing a Featurize workflow and intends to trigger the cloud run personally.
- Active method remains static EDSR-L 32→24, endpoint-inclusive ordered transplant, fixed-budget `0.5 KD-L1 + 0.5 GT-L1`, attachment-free Student.
- 4090 is for training/quality evaluation only; formal latency remains RTX 4060 same-stack measurement.

Diagnosis:
- The old STOP was procedural, not evidence of divergence: the pilot completed, improved held-out diagnostics, and produced a valid static checkpoint (`results/autonomous_goals/20260809-112351/final_report.md`).
- A new Goal may deliberately replace the noisy random-crop first/last loss gate without changing the old result.
- Existing Featurize tooling is not suitable unchanged:
  - **high:** `train_cloud.sh` launches the stopped dynamic RCAN/B2R-SR path.
  - **high:** `scripts/export/export_training_run.sh` is hard-coded to RCAN ×4/120000 and deletes an existing output with `rm -f`.
  - **high:** `scripts/eval/run_and_release.sh` copies only recent reports/tarballs, potentially omitting EDSR checkpoints and exact manifests.
  - **medium:** `docs/B2RSR_Featurize_Training_Guide_zh.md` is primarily legacy dynamic-RCAN guidance; its batch-48 conclusions must not be transferred to EDSR automatically.

Drift / contradiction check:
- Reusing the old Goal directory or its pilot checkpoint would contradict `docs/PROJECT_STATE.md`; every formal seed must start from a fresh official-checkpoint transplant.
- Treating 4090 timing as deployment latency would contradict `AGENTS.md`.
- Silently changing crop/batch after OOM would violate protocol freezing.
- Automatic benchmark evaluation is allowed only after the new Goal freezes scales, seeds, checkpoints, data split, metrics, and final-step policy.

Recommendation:
- Create a new approved Goal quoting the user’s authorization.
- Minimal defensible first formal run:
  - scales: ×2/×3/×4;
  - Teacher: canonical depth-32 official checkpoints with the already verified full hashes;
  - Student: depth-24 only, same frozen endpoint-inclusive mapping;
  - seeds: `0,1,2`;
  - every seed starts from the same fresh transplant, never the old ×4 pilot;
  - exactly 500 FP32 updates;
  - Adam `lr=1e-5`, betas `0.9/0.999`, cosine schedule over 500 steps;
  - loss: `0.5 L1(S,T) + 0.5 L1(S,HR)`;
  - batch 8, 48×48 LR crop as the safe cloud-image-compatible default;
  - recovery data restricted to frozen DIV2K training IDs, preferably `0001–0760`, even if stored inside DF2K;
  - final-step checkpoint only; no best-checkpoint selection;
  - seed 0 predeclared as canonical deployable checkpoint, with all three seeds used for aggregate statistics.
- Evaluate transplant-before-recovery, Teacher, and all recovered seeds on DIV2K validation; only after all checkpoints freeze, evaluate Set5/Set14/BSD100/Urban100/Manga109 for PSNR-Y/SSIM-Y and seed confidence intervals.
- Do not train d28 in this first run; existing latency describes it, and quality-Pareto expansion can follow after the requested three-scale d24 results.
- Replace the old stability gate with fatal-only termination:
  - checkpoint/hash/strict-load failure;
  - data pairing/leakage/geometry failure;
  - static deployment audit failure;
  - NaN/Inf/OOM;
  - persistence/export integrity failure.
  Random-crop loss windows become descriptive diagnostics only.
- Cloud wrapper must:
  1. perform check-only validation of existing datasets—no dataset download;
  2. download only missing official weights with resume and exact SHA-256 verification;
  3. write logs/checkpoints/resume states directly under `/home/featurize/work`;
  4. support idempotent skip/resume without overwriting completed runs;
  5. export exact source, protocol, environment, logs, metrics and final checkpoints with a verified manifest;
  6. call `featurize instance release`, not merely `shutdown`.
- On terminal failure, persist partial logs/state and still release by default. If release fails, shutdown may be attempted but must be reported as **billing potentially continuing**.
- Before push, stage only the new Goal/package and maintained documentation; the worktree already contains many unrelated changes.

Risks:
- Batch-8/48 differs from the earlier RCAN batch-8/64 pixel budget. It is defensible for the first EDSR result but is not a perfectly matched cross-backbone training-budget comparison.
- The saved cloud image’s exact filenames and benchmark directories remain unverified; the checker must normalize and hash pair mappings rather than trust counts alone.
- Full-image EDSR evaluation may OOM on unusually large images. Do not silently introduce chop/tiling; any evaluation fallback must be frozen beforehand.
- `featurize instance release` availability cannot be verified locally, so notification and billing-warning paths remain necessary.

Need from main agent:
- No user clarification is required for the safe 500-step, three-seed packaging default.
- A new decision is required only if the user actually means conventional long finetuning such as 120k steps, or if cloud data cannot support the frozen 48×48 protocol.

Suggested execution prompt:
- No separate worker handoff is warranted; the parent is already the implementation executor.