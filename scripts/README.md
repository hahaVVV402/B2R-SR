# B2R-SR operational scripts

> Current static-depth-transfer entrypoints are the model-neutral `codes/train.py`, `codes/test.py`, and `codes/run.py`, driven by YAML under `codes/options/`. Dynamic routing, LUT cascade, posterior routing, segment distillation, cheap-block, width, and non-uniform-depth scripts are retained for historical reproducibility and are not active paper methods. New one-off experiments belong in their autonomous goal's `executed_source/`; only promoted reusable tools remain here.

New operational helpers live here instead of adding more files to the repository root.
Existing root entrypoints (`train_cloud.sh`, `prepare_cloud_data.sh`, and
`prepare_pretrained.sh`) remain in place for compatibility.

## Active Featurize EDSR run

The formal EDSR YAMLs use the deterministic one-batch prefetch path validated by
the RTX 4060 sync/prefetch and interruption-resume checks. Progress logs report
input wait, GPU time, end-to-end step cadence, throughput, percentage, and ETA.

The repository-native workflow is documented in
`docs/EDSR_Static_Depth_Experiment_Workflow_zh.md`. The formal wrapper is:

```bash
bash scripts/cloud/run_featurize.sh \
  -opt codes/options/run/run_EDSR_d24_formal.yml
```

It checks existing datasets without downloading them, strictly acquires official
EDSR weights, delegates model-specific behavior to YAML-selected generic entrypoints,
writes all runtime evidence under ignored `experiments/`, exports a verified bundle,
and requests `featurize instance release`. Do not use legacy `train_cloud.sh` or the
superseded goal-owned 500-step package for the active 200,000-update configuration.

## Export a completed cloud run

Create one downloadable archive containing the final checkpoint, training logs,
TensorBoard events, configuration, environment metadata, and 12 final validation
samples:

```bash
./scripts/export/export_training_run.sh
```

The X4 defaults export step 120000 to:

```text
/home/featurize/work/B2RSR_RCAN_X4_120000_export.tar
```

Override the experiment, step, or output path with positional arguments:

```bash
./scripts/export/export_training_run.sh B2RSR_RCAN_X4 104000 \
  /home/featurize/work/B2RSR_RCAN_X4_104000_export.tar
```

## Data

Prepare the standard EDSR benchmark bundle (Set5, Set14, BSD100, Urban100,
Manga109; X2/X3/X4):

```bash
./scripts/data/prepare_sr_benchmarks.sh /home/featurize/data
```

The script prefers `/home/featurize/data/benchmark.tar`, validates all image
counts, and creates the canonical layout under
`/home/featurize/data/SRBenchmarks/`. If the archive is absent, it attempts a
resumable download from the official EDSR URL. Uploading `benchmark.tar` as a
Featurize dataset is the reliable fallback when external download fails.

## X4 quality evaluation

Check data and checkpoints without running inference:

```bash
./scripts/eval/run_b2rsr_rcan_x4.sh check \
  experiments/B2RSR_RCAN_X4/models/120000_G.pth
```

Run the full-width dense RCAN baseline and B2R-SR with identical datasets and
metric code:

```bash
./scripts/eval/run_b2rsr_rcan_x4.sh all \
  experiments/B2RSR_RCAN_X4/models/120000_G.pth
```

Use `dense` or `b2rsr` instead of `all` to run only one side. Logs and images
are written below:

```text
results/eval_RCAN_X4_dense/
results/eval_B2RSR_RCAN_X4/
```

`test.py` reports RGB PSNR/SSIM and Y-channel PSNR/SSIM. Use the Y-channel
numbers for comparison with the standard RCAN paper tables.

## Four decision gates with one checkpoint

On a GPU instance, attach these two files under `/home/featurize/data`:

```text
B2RSR_RCAN_X4_120000_export.tar
benchmark.tar
```

Then run one command:

```bash
python scripts/eval/run_gate_tests.py \
  --checkpoint /home/featurize/data/B2RSR_RCAN_X4_120000_export.tar \
  --archive /home/featurize/work/B2RSR_GATE_RESULTS.tar.gz
```

The script prepares the standard datasets, extracts the checkpoint temporarily,
and runs: dense-vs-all-keep equivalence, matched-K routing policies, a budget
sweep, and repeated dense/B2R latency. It writes JSON and Markdown reports and
produces the single archive passed through `--archive`.

Gate 1 includes a feature-delta teacher upper bound, but not the expensive GT
counterfactual oracle; Gate 3 is repeated forward timing without quality
matching. Both therefore remain screening results rather than automatic PASS
verdicts. These fixed-weight diagnostics do not replace formal retrained
ablations.

## Repeatable inference latency

Run quality evaluation first, then benchmark both configurations with the same
protocol:

```bash
python scripts/eval/benchmark_inference.py \
  -opt codes/options/test/test_RCAN_X4_dense.yml \
  --warmup 20 --runs 100 --max-images 5

python scripts/eval/benchmark_inference.py \
  -opt codes/options/test/test_B2RSR_RCAN_X4.yml \
  --checkpoint experiments/B2RSR_RCAN_X4/models/120000_G.pth \
  --warmup 20 --runs 100 --max-images 5
```

The benchmark disables the plugin's internal one-shot timer and reports median,
mean, p90, and standard deviation for repeated forward passes. The examples use
five images for a quick check; use `--max-images 0` for the complete datasets in
formal reporting. Compare rows only when dataset, image selection, hardware,
precision, and run arguments match.
