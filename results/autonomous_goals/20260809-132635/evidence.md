# Evidence — Goal 20260809-132635

Evidence will be appended without rewriting completed Goal `20260809-112351`.

## Inherited immutable facts

- Official EDSR-L ×2/×3/×4 strict import and local output equivalence: PASS.
- Physical d24 static construction and strict state round-trip: PASS.
- RTX 4060 ×4 d24 p50 speedup: `1.217595×`.
- Prior 100-step ×4 pilot: no NaN/OOM, deterministic held-out quality moved in the expected direction, strict final Student checkpoint PASS; old random-crop stability predicate FAIL and old Goal STOP.

Primary source: `results/autonomous_goals/20260809-112351/final_report.md`.

## New package evidence

### Frozen package

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

Attempt 01 used the prior reviewed source freeze and is preserved separately; attempt 02 is the binding package smoke.

### Independent reviews

- `subagents/protocol_oracle.md` and `cloud_plan_review.md` established the new-Goal/fatal-only/Featurize boundaries.
- `subagents/implementation_review.md` and `freeze_review.md` identified persistence, resume, evaluation, checksum, signal, and bundle-validation defects; all launch blockers/high findings were fixed before the final freeze.
- `subagents/final_package_review.md`: **READY FOR SELECTIVE COMMIT/PUSH**, no launch-blocking defect found.

Pending: selective commit/push and the user-triggered Featurize run. No formal cloud quality result exists yet.
