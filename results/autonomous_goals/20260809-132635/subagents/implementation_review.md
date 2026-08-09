Turn budget wrap-up was requested after 8 assistant turns (soft limit 8, grace 2). Process-mode live steering is unavailable, so the child was warned at launch to wrap up by this budget. Output may be partial.

## Review

### Correct
- Checkpoint provenance and transplant logic match the immutable prior EDSR evidence: full byte/SHA verification, strict loading, endpoint-inclusive indices, and body-tail convolution mapping are implemented at `formal_recovery.py:246-273,305-337`.
- Fresh-path all-nine gating is correct: evaluation scans every scale/seed checkpoint before opening benchmarks at `formal_recovery.py:992-1005`.
- Resume state captures Student, optimizer, scheduler, sampling RNG, NumPy/Torch/CUDA RNG, and truncates traces to the committed step at `formal_recovery.py:737-749,796-810,557-573`.
- Frozen PSNR-Y/SSIM-Y definitions, output rounding, scale shave, and seed t-interval are implemented consistently at `formal_recovery.py:896-977`.
- The launcher attempts `featurize instance release`, retries three times, and explicitly labels shutdown as billing-unsafe at `run_featurize.sh:92-114`.

### Blocker
- **Package is not launchable after the documented `git pull`.** `protocol.json:4-6` has `"frozen": false`, references a nonexistent `source_manifest.json`, and the entire goal is ignored by `.gitignore:25`. `formal_recovery.py:117-140` rejects an unfrozen protocol or missing manifest. Current `git status --ignored` shows `!! results/autonomous_goals/`, so none of the launcher files would arrive through `git pull` unless explicitly force-added, frozen, committed, and pushed. This is also recorded as pending at `progress.md:9`.
- **Failure finalization can release the instance without a successfully verified archive.** `finalize` disables error handling with `set +e` (`run_featurize.sh:116-125`). The fallback then runs `tar`, validation, move, and hashing as unchecked sequential commands (`run_featurize.sh:77-89`); any of those may fail, but the final successful `echo` makes the function appear successful. Execution then unconditionally reaches release (`run_featurize.sh:127-130`). Disk-full or tar failures can therefore falsely claim preservation and release contrary to the frozen export-integrity policy.

**Blockers remain; the package should not be launched yet.**

### High
- **Completed-run resume validation can accept a copied or incomplete run as another seed.** `completed_run_report` checks protocol/source hashes, step count, and checkpoint hash only (`formal_recovery.py:576-595`). It does not verify the report’s scale, seed, run-config hash/file, or trace hash/file. A copied `seed0` report/checkpoint under `seed1`, or a missing/corrupt trace, is accepted and skipped at `formal_recovery.py:666-675`, invalidating the three-seed claim.
- **Completed evaluation validation is similarly incomplete.** The already-complete path checks only protocol and final checkpoint hashes (`formal_recovery.py:1006-1010`). Missing or corrupted per-image JSONL files, wrong source-manifest hash, or incomplete scale/dataset summaries are not revalidated before a success bundle is permitted.
- **Data preflight does not enforce the frozen readability/geometry contract for all recovery files.** Only roughly 32 sampled pairs are decoded (`formal_recovery.py:411-427,447-450`), although `protocol.json:119-124` makes any readability/geometry failure fatal. Corrupt or undersized unsampled crops can survive preflight.
- **Evaluation silently changes HR geometry.** `read_pair(..., allow_modcrop=True)` crops HR to LR-derived dimensions (`formal_recovery.py:398-407,940`) instead of enforcing the protocol’s stated exact HR=`scale×LR` pairing. Either freeze/document standard modcrop explicitly or reject mismatches; the current implementation and protocol disagree.
- **Sibling archive checksum is not portable.** `sha256sum "$archive" > "$archive.sha256"` at `run_featurize.sh:69` and `:85` records the absolute Featurize path. After downloading, ordinary `sha256sum -c archive.sha256` will look for the now-nonexistent cloud path rather than the sibling tar.

### Medium
- **Signal handling is not fully bounded.** `INT`/`TERM` cause shell exit (`run_featurize.sh:133-135`), but the launcher does not retain or forward to the active Python child. Bash may defer its trap while waiting for that child, and a second signal during finalization uses default handling because traps are removed at `run_featurize.sh:120`. Actual process-group behavior needs an integration kill test.
- The bundle collector recursively includes every file under `RUN_ROOT` (`formal_recovery.py:1099-1113`) rather than the protocol’s stated explicit goal-owned allowlist. Stale or accidental files are hashed, but also exported.