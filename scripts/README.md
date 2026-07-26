# B2R-SR operational scripts

New operational helpers live here instead of adding more files to the repository root.
Existing root entrypoints (`train_cloud.sh`, `prepare_cloud_data.sh`, and
`prepare_pretrained.sh`) remain in place for compatibility.

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
