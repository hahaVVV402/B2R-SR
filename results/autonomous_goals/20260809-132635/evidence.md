# Evidence — Goal 20260809-132635

Evidence will be appended without rewriting completed Goal `20260809-112351`.

## Inherited immutable facts

- Official EDSR-L ×2/×3/×4 strict import and local output equivalence: PASS.
- Physical d24 static construction and strict state round-trip: PASS.
- RTX 4060 ×4 d24 p50 speedup: `1.217595×`.
- Prior 100-step ×4 pilot: no NaN/OOM, deterministic held-out quality moved in the expected direction, strict final Student checkpoint PASS; old random-crop stability predicate FAIL and old Goal STOP.

Primary source: `results/autonomous_goals/20260809-112351/final_report.md`.

## New package evidence

### Initial pushed package freeze (superseded before cloud execution)

- Protocol SHA-256: `ae05cb0ada529f41fb2c56af253e3449eb90db90b6f0e10ddb22e05686ca9ce4`.
- Source-manifest SHA-256: `9b5bde664b958c2b51474cd55a65379fd8e7c071f8025289834b90257db4e51d`.
- Runner SHA-256: `ea7351aadbb954582e9bbc5222eeeb00eb1e1296eb2f906a19dd9fc78793c787`.
- Launcher SHA-256: `923f53d947565c8f86d16a51177e6a9e26b123e80b202eeed4a1a668bfecedc5`.
- `python3 -m py_compile`, `bash -n`, and the local tiny forward/backward/round-trip/metric self-test passed.
- A temporary structured partial bundle passed internal member hash verification and tar re-read.

### Exact-source RTX 4060 smoke

`remote_artifacts/phase0_4060_smoke_attempt02/smoke_report.json`:

- report SHA-256: `066e0923e268056dd16e17082b3584cad8b5b30463d3a7aa48bdcf3fa269145b`;
- device: NVIDIA GeForce RTX 4060 Laptop GPU;
- official ×4 checkpoint bytes/full SHA, strict load, and finite Teacher forward: PASS;
- physical d24 static audit: PASS;
- two FP32 CUDA forward/backward/Adam updates: PASS, finite losses `0.4706503153`, `0.7835155725`;
- temporary final Student strict round-trip and finite `[1,3,32,44]` forward: PASS;
- peak allocated/reserved: `846277120 / 954204160` bytes;
- report protocol and source-manifest hashes exactly match the frozen local files.

Attempt 01 used the first reviewed source freeze; attempt 02 bound the initially pushed package. Both are preserved, but the user-requested standalone-test revision below supersedes them before cloud execution.

### Independent reviews

- `subagents/protocol_oracle.md` and `cloud_plan_review.md` established the new-Goal/fatal-only/Featurize boundaries.
- `subagents/implementation_review.md` and `freeze_review.md` identified persistence, resume, evaluation, checksum, signal, and bundle-validation defects; all launch blockers/high findings were fixed before the final freeze.
- `subagents/final_package_review.md`: **READY FOR SELECTIVE COMMIT/PUSH**, no launch-blocking defect found.

### GitHub delivery

- Selective package commit: `681d0cd14a9eedad7cbb8d63140decbf8010b8de` (`Add Featurize EDSR recovery workflow`).
- Push: `origin/main` advanced from `979213f` to `681d0cd`.
- Post-push `git ls-remote origin refs/heads/main` returned the exact local commit hash.
- Pre-existing unrelated local worktree changes were neither staged nor committed.

### User-requested standalone-test revision

Current frozen hashes:

- protocol: `5c87880633a8f68d4b3aeb7c0e2f92a5f5d3ee1613e9542b6dcabfb34b3e77c6`;
- source manifest: `06e4180861235afa7c3fe06741e50aa0ba9d813eabd7c5deb6c4c5d1f8260268`;
- formal runner: `0702f666c6e2dc6ab826440b15d63ccaab4da0ada6a8b2ab144a7124ecd45981`;
- launcher: `42c7509f40f324e99140c89a9a054eedb72216808fa6006ea8e6e97dfb2b52ef`;
- standalone wrapper: `e2689dbdf13dd28308f33628ba7fb56658fafd9f3ead3197e5543fa03e3b9222`.

Validation:

- local synthetic d3/n_feats8 checkpoint test on CPU: strict architecture inference/load, paired input, PSNR-Y/SSIM-Y, CSV/JSONL/log/summary, optional image saving, and outside-`experiments/` rejection all PASS;
- `remote_artifacts/phase0_4060_smoke_attempt04/smoke_report.json` SHA-256 `84083e6ff892b63bc2b1de81df005b4e7399768c86b7cf7783f92f0c5b9700ff`: exact current protocol/source hashes, two finite ×4 CUDA updates, static audit, and strict round-trip PASS;
- `remote_artifacts/phase0_4060_standalone_test_attempt02/`: real d24 checkpoint SHA `a923f89a...c0d6`, inferred depth 24/width 256/×4, one non-benchmark training crop, PSNR-Y `41.8354488584` and SSIM-Y `0.9666962741`; summary, log, CSV and JSONL agree;
- temporary partial bundle includes the frozen standalone wrapper and passes member-hash/tar verification;
- `subagents/standalone_test_final_review.md`: **READY TO COMMIT/PUSH**, no blocker.

Revision delivery:

- selective source/evidence commit: `a886a719aacccc3265a743aaa01c3606c8c7fd02` (`Add standalone EDSR checkpoint evaluation`);
- push advanced `origin/main` from `d4a076b` to `a886a71`;
- unrelated pre-existing worktree changes were not staged.

Pending: the user-triggered Featurize run. No formal cloud quality result exists yet; the standalone CUDA check deliberately used a training crop, not a final benchmark.
