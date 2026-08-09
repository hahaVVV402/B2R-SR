## Review

### Correct

- The intended architecture and loss match the established method: physical EDSR-L d24, ordered transplant, and `0.5 L1(Student, Teacher) + 0.5 L1(Student, GT)`. The prior implementation proved strict loading and static construction at all three scales (`results/autonomous_goals/20260809-112351/final_report.md`, Phase 1–2).
- Full official checkpoint hashes are already attested in `results/autonomous_goals/20260809-112351/checkpoint_manifest.json`:
  - ×2: `0edfb8a3772c94485c939f81a8aeefe02cccdfbf45243c6d22b39c29eb7d46de`
  - ×3: `ea3ef2c6449845ac3bda7290480af76a53e5006beff60bf89c84ca62f986eeca`
  - ×4: `4f62e9ef1a4ec6a7d3da4ed837116cec641e4e1566f98187151d813e83a0c1c8`
- Final-step-only **model selection** is appropriate: evaluation should use the frozen step-500 checkpoint rather than select the best validation checkpoint.
- Keeping large datasets under `/home/featurize/data` and durable outputs under `/home/featurize/work` follows `docs/B2RSR_Featurize_Training_Guide_zh.md`, Sections 2 and 9.

### Blockers

- **Blocker — authorization:** The plan contradicts the active project state. `docs/PROJECT_STATE.md:3-4,20-23,91-99` says the EDSR feasibility goal ended **STOP**, formal EDSR recovery is not authorized, and any reconsideration requires a separately approved new goal with a newly frozen stability criterion. Nine cloud training runs and final benchmark opening must not begin until that goal exists. This also follows `AGENTS.md:73-80,83-87`.
- **Blocker — unresolved feasibility failure:** The previous pilot’s frozen stability check failed: final-10 mean loss `4.23538` exceeded first-10 `3.78766` (`results/autonomous_goals/20260809-112351/final_report.md:3-10`, Phase 4). The new protocol must prospectively replace or retain that condition—preferably using matched fixed crops or a deterministic diagnostic—not silently omit it.
- **Blocker — “final-step checkpoints” are insufficient for resume:** A step-500 inference checkpoint should be final evidence, but periodic atomic recovery state is still required. Persist optimizer, completed update, Python/NumPy/Torch/CUDA RNG, sampler/data cursor, scheduler/scaler if used, and configuration/source/data hashes. Otherwise any interruption restarts a seed and can produce duplicated or non-comparable updates.
- **Blocker — existing release wrapper does not attest persistence:** `scripts/eval/run_and_release.sh:29-38` treats failed or empty copying as success and only searches recent `*.tar.gz`/reports. It does not persist experiment checkpoints or plain `.tar` exports. It can therefore release an instance after losing the important artifacts.
- **Blocker — failure export is unsupported:** `scripts/export/export_training_run.sh:4-15` is hard-coded to RCAN X4 and refuses export without a final checkpoint. It cannot package a partial EDSR failure, resumable state, or the nine-run/evaluation matrix required by this plan.
- **Blocker — benchmark protocol is not frozen:** “PSNR/SSIM” is insufficient. Before any benchmark is opened, freeze RGB→Y conversion, `[0,255]` quantization, border shave, SSIM implementation/window, filename pairing, aggregation, and expected image counts. Otherwise results will not be comparable to official SR reporting.

### Notes and required plan corrections

#### Correctness and reproducibility

- **High — incomplete training freeze:** Freeze optimizer and hyperparameters, LR/batch/crop geometry, augmentation, precision/TF32 settings, data sampling, worker seeding, validation policy, and exact 500-update definition. Seeds `0/1/2` alone do not determine a reproducible run.
- **High — transplant policy must be exact:** Reuse the already attested endpoint-inclusive ordered d24 indices from `results/autonomous_goals/20260809-112351/protocol.json`, or prospectively declare another mapping. All three seeds at a scale must start from the same transplanted-state hash.
- **High — checkpoint verification:** Downloads must be checked against the known expected byte sizes and SHA-256 values, not merely assigned newly computed hashes. Preserve URL, effective URL, retrieval time, byte size, and resumable `.part` behavior as done in `checkpoint_manifest.json`.
- **High — Teacher/KD invariants:** Require Teacher `eval()`, `requires_grad=False`, detached outputs, strict checkpoint loading, exact aligned LR/HR crops, finite losses, and a strict finite-forward audit of every final Student.
- **Medium — deterministic completion matrix:** Each `{scale,seed}` needs a create-only identity, status manifest, initial-state hash, final-state hash, completed-update count, and explicit states such as pending/running/resumable/complete/failed. A valid completed run should be skipped; incompatible partial state should be rejected rather than overwritten.

#### Data layout

- **High — do not call `prepare_cloud_data.sh` for check-only behavior:** That script extracts archives (`prepare_cloud_data.sh:42-76`) and therefore violates the intended check-only contract.
- **High — stale marker risk:** `prepare_cloud_data.sh:31-40` exits solely because `.b2rsr_ready_x{scale}` exists, without revalidating directories. A new checker must inspect the actual data on every launch.
- **High — equal counts do not prove pairing:** `prepare_cloud_data.sh:94-135` checks only image counts. Validate matching stems, scale suffixes, readable images, and exact `HR = scale × LR` geometry.
- **High — benchmark layout is unspecified:** Freeze exact roots and expected counts for DIV2K-val and Set5/Set14/BSD100/Urban100/Manga109 at ×2/×3/×4. The Featurize guide documents DF2K and DIV2K validation, but not the five benchmark directory contract.
- **Medium — dataset evidence:** Export a sorted inventory and digest. Full per-file hashes are strongest; at minimum include relative paths, sizes, expected counts, and a manifest hash so resumed/evaluated runs cannot silently switch datasets.

#### Resume and interruption handling

- **High:** Save resumable state periodically to a temporary file, `fsync`, then atomically rename under `/home/featurize/work`. Keep it separate from the final pure Student state dictionary.
- **High:** On resume, verify goal/protocol hash, executed-source hash, official Teacher hash, scale, seed, transplanted initialization hash, data-manifest hash, and current step.
- **Medium:** Define signal handling for `EXIT`, `INT`, and `TERM`: capture status, persist logs/state, build a failure bundle, verify it, then invoke release. SIGKILL or host loss remains an unavoidable residual risk.
- **Medium:** Evaluation must not start until all intended final checkpoints have been hashed and frozen. A resumed run must not cause duplicate benchmark evaluation or overwrite earlier raw results.

#### Export and evidence

- **High:** Create a goal-specific exporter rather than reuse `export_training_run.sh`, whose config and checkpoint paths are RCAN-specific (`scripts/export/export_training_run.sh:4-15`).
- **High:** Produce both:
  1. an internal `SHA256SUMS` covering every bundle file; and
  2. a sibling SHA-256 for the completed archive under `/home/featurize/work`.
  Verify both after the archive reaches the persistent destination.
- **High:** Success and failure bundles should include protocol/source manifests, environment and Git state, data inventory, acquisition manifest, initial/final model hashes, resume-state inventory, complete loss traces, commands and logs, per-image evaluation JSON/CSV, aggregates, completion matrix, and the original task exit code.
- **Medium:** Do not rely on the existing “last 12 hours” collection (`scripts/eval/run_and_release.sh:31-36`); it can collect unrelated reports and miss older resumed artifacts. Export an explicit goal-owned file list.
- **Medium:** Avoid bundling the three large official Teacher files if their URLs, sizes, and verified hashes are included; final Student checkpoints and recovery evidence are the irreplaceable artifacts.

#### Billing and release

- **High — exit status corruption:** On successful release, `scripts/eval/run_and_release.sh:47-49` exits `0` even if the experiment failed. Persist and report the original workload status before release.
- **High — shutdown is not release:** The fallback at `scripts/eval/run_and_release.sh:49-51` explicitly may not stop billing. If automatic release is required, preflight the authenticated `featurize` CLI and make its absence a launch failure. On release failure, persist CLI output, notify prominently, and retry with a bounded policy; do not label shutdown as billing-safe.
- **High:** Invoke `featurize instance release` only after archive creation, archive listing, internal hash verification, destination copy, destination re-verification, and durable status logging.
- **Medium:** The existing 60-second cancellation window can leave an unattended billable instance. Omit it for the one-command production launcher or require an explicit `RELEASE=0` before launch.
- **Residual:** A release command may succeed after the machine becomes unavailable, so a post-release receipt cannot always be written locally. Record “release requested” and command output in persistent storage immediately before/during the call, then verify instance state externally.

#### Evaluation and paper evidence

- **High:** Evaluate at least official Teacher, zero-step transplanted d24, and recovered final d24. Reporting only recovered Students obscures the initial depth-transfer loss and the recovery gain.
- **High:** Preserve per-seed metrics and report mean, dispersion, and a prospectively selected confidence-interval method. Three seeds give weak interval estimates and should not be overstated.
- **High:** Final real-latency evidence must be rerun on the common target device and software stack; 4090 training latency cannot be mixed with the existing RTX 4060 table (`AGENTS.md:35-38`; `docs/PROJECT_STATE.md`, Sections 5 and 7).
- **Medium:** Parameters, MACs/FLOPs, p50, and p95 latency must remain separate. The existing no-training feasibility latency can support planning but should not substitute for a frozen final same-stack measurement.
- **Medium:** The goal-owned launcher and source belong under the **new** goal’s `executed_source/`; do not mutate or append runs to completed goal `20260809-112351`.