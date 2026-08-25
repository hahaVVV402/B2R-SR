# AGENTS.md

Operational contract for coding and experiment agents working in this repository.
`CLAUDE.md` describes the legacy code architecture and commands; this file owns current project state, hardware use, and artifact placement.

## Read first

1. `AGENTS.md`
2. `docs/PROJECT_STATE.md`
3. The active goal under `results/autonomous_goals/<YYYYMMDD-HHMMSS>/goal.md`, when one exists
4. Only then the specific code and evidence named by that goal

Do not infer the current research direction from archived reports or the stopped dynamic plugin.

## Current direction

The active paper direction is post-hoc **static depth transfer for pretrained CNN-SR checkpoints**:

`pretrained checkpoint -> topology-valid physical Student -> ordered weight transplant -> fixed-budget teacher-guided recovery -> quality/real-latency selection -> frozen static Student`

The deployed model has no Teacher, Router, Mask, patch scheduler, dynamic branch, keep-map lookup, or training hook. Dynamic routing, recoverability/TASD, segment composition, PConv replacement, width-48, and non-uniform-depth superiority claims are stopped historical directions. RCAN is the validated anchor; canonical EDSR-L is the proposed next backbone and must pass a no-training feasibility phase before long training.

## Execution environments

### Mac workspace

- Repository: `/Users/admin/Workspace/Research/DART-SR-Project/code/B2R-SR`
- Role: orchestration, editing, paper work, manifests, static checks, and lightweight CPU checks
- Do not assume the system Python has PyTorch/CUDA.

### RTX 4060 Laptop GPU over SSH

- Connect with the existing passwordless alias: `ssh 4060`
- Host: WSL2 on the RTX 4060 laptop
- Repository: `/home/jww/WorkSpace/Research/B2R-SR`
- Python: `/home/jww/miniconda3/envs/b2rsr/bin/python`
- PyTorch/CUDA baseline: PyTorch 2.3.0 + CUDA 12.1
- GPU check: `/usr/lib/wsl/lib/nvidia-smi`
- Role: free smoke tests, kill checks, short pilots, and target-device latency screening
- Hardware details and recovery steps: `docs/B2RSR_4060_WSL2_SSH_Setup_zh.md`

Safe preflight:

```bash
ssh 4060 'cd /home/jww/WorkSpace/Research/B2R-SR && \
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && \
  /home/jww/miniconda3/envs/b2rsr/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"'
```

### Rented GPU

Use a rented RTX 4090 only after local packaging and 4060 feasibility checks pass. Its role is training throughput, recovery training, and frozen quality evaluation. Do not mix 4090 latency with the existing 4060 latency table: formal latency comparisons must be rerun for every model on the same target device and software stack.

### Featurize cloud contract

- Persistent repository and outputs: `/home/featurize/work`; ephemeral instance data: `/home/featurize/data`.
- Prepare and review the exact launcher locally, then push it. On the instance, use `git pull --ff-only` and one goal-owned command; do not reconstruct formal commands interactively.
- A launcher must check existing data rather than silently download/extract it, strictly verify any downloaded checkpoint, keep resumable state and logs in `work`, and export a hash-verified goal bundle.
- Stop billing with `featurize instance release`; operating-system shutdown alone is not a confirmed billing stop. Persist success/failure status before requesting release.
- Active implementation record: `results/autonomous_goals/20260809-084626/goal.md`. The repository-native Featurize entrypoint is `scripts/cloud/run_featurize.sh -opt codes/options/run/run_EDSR_d24_formal.yml`; it is not promoted for formal execution until the RTX 4060 smoke and final review pass. Goal `20260809-132635` is the superseded, unexecuted 500-step package. Legacy `train_cloud.sh` must not launch the active EDSR method.

### Featurize SSH operating procedure

The instance is reachable through the existing `~/.ssh/config` entry, not a bare IP:

```
Host workspace.featurize.cn
  HostName workspace.featurize.cn
  User featurize
  Port 10742
```

The port is issued per rental. `Connection closed by <ip> port <n>` means no instance is currently running or the port changed; ask the user for the current SSH command from the Featurize console and update the config rather than guessing ports.

Standard sequence once the user reports an instance is up:

```bash
# 1. confirm identity, GPU, and free space before doing anything
ssh workspace.featurize.cn 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /home/featurize/work | tail -1'

# 2. bring the persistent clone to the reviewed commit (fast-forward only)
ssh workspace.featurize.cn 'cd /home/featurize/work/B2R-SR && git fetch origin && git status --porcelain=v1 --untracked-files=no && git pull --ff-only origin main && git log --oneline -1'

# 3. launch one goal-owned plan under nohup so an SSH drop cannot kill it;
#    keep RELEASE=0 for every run except the last
ssh workspace.featurize.cn 'cd /home/featurize/work/B2R-SR && export SR_DATA_ROOT=/home/featurize/data && \
  nohup env RELEASE=0 bash scripts/cloud/run_featurize.sh -opt <plan.yml> > /home/featurize/work/<plan>.log 2>&1 & echo started $!'

# 4. poll progress without holding the connection open
ssh workspace.featurize.cn 'tail -30 /home/featurize/work/<plan>.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

Rules that apply to every remote session:

- Always `nohup ... &` long runs; never hold training in a foreground SSH channel.
- `run_featurize.sh` refuses to start when tracked files are dirty, when the GPU is not an RTX 4090, or when less than 15 GiB is free under `/home/featurize/work`. Fix the cause; do not bypass the gate.
- `RELEASE=0` for intermediate runs, and only the final run releases the instance. After any release, verify with a failed SSH attempt and record the exit status.
- Never edit files on the instance. Change code locally, push, then `git pull --ff-only`.
- Copy every report and manifest back into the owning goal before releasing; the instance is not storage.

## Artifact placement

- Core model/training code: `codes/`
- Reusable data/evaluation tools: `scripts/data/` and `scripts/eval/`
- One-off experiment code: the goal's `executed_source/`; do not add scratch scripts to the repository root
- Autonomous experiment record: `results/autonomous_goals/<YYYYMMDD-HHMMSS>/`
  - minimum record: `goal.md`, `progress.md`, `evidence.md`, `decision_log.md`, `final_report.md`
  - place manifests, frozen protocol, executed-source snapshots, hashes, and copied remote reports inside the same goal
- Generated checkpoints/logs: ignored `experiments/` or `results/`; never commit large checkpoints
- Active handoff facts: `docs/PROJECT_STATE.md`
- Current explanatory documentation: `docs/`
- Stopped/obsolete narratives: `docs/archive/<dated-topic>/`; archive rather than delete
- Raw literature notes: keep inside the owning goal unless promoted to a maintained document
- Third-party code: `third_party/<project>/` with upstream URL, revision, and license
- Paper assets: `/Users/admin/Workspace/Research/DART-SR-Project/paper/`
- Temporary downloads, command output, and accidental artifacts: `/tmp`, never the repository root

The repository root is reserved for durable entrypoints and project-level configuration. Do not create scratch directories named after shell commands, flags, hosts, tensor sizes, or remote paths.

## Experiment lifecycle

1. Freeze data split, checkpoint hashes, candidate depths, seeds, quality metrics, latency protocol, and stop criteria.
2. Validate checkpoint provenance, strict loading, shapes, and a small forward pass.
3. Measure no-training target-device latency before long recovery training.
4. Run a short training/memory/throughput pilot.
5. Require explicit promotion before a long rented-GPU run or opening final benchmarks.
6. Copy remote manifests and reports back to the owning goal; never rely on the only copy remaining on a remote machine.

Internal PASS/STOP decisions belong in goal records. The paper should present supported methods and measured trade-offs, not experiment-management language.

## Safety

- Do not commit, push, stage, reset, stash, clean, or overwrite user changes unless explicitly requested.
- Do not delete historical experiment evidence. Move accidental local artifacts to `/tmp`; archive obsolete research narratives under `docs/archive/`.
- Do not run full training, rent hardware, or open final benchmarks without the frozen protocol and user approval.
- Parameters, MACs/FLOPs, and wall-clock latency are separate metrics; never use FLOPs as a latency claim.
