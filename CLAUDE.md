# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core commands

Run commands from repository root unless noted.

### Environment / dependencies
- Install core deps: `pip install numpy opencv-python lmdb`
- Optional logging: `pip install tensorboardX`

### Train
- Generic SR / DART-SR training entrypoint:
  - `cd codes && python train.py -opt options/train/train_DARTSR_RCAN.yml`
  - `cd codes && python train.py -opt options/train/train_DARTSR_CARN.yml`
  - `cd codes && python train.py -opt options/train/train_DARTSR_SRResNet.yml`
- Original single-branch SR training:
  - `cd codes && python train.py -opt options/train/train_RCAN.yml`
  - `cd codes && python train.py -opt options/train/train_CARN.yml`
  - `cd codes && python train.py -opt options/train/train_SRResNet.yml`
  - `cd codes && python train.py -opt options/train/train_FSRCNN.yml`
- Original ClassSR training:
  - `cd codes && python train_ClassSR.py -opt options/train/train_ClassSR_RCAN.yml`
  - (or CARN / SRResNet / FSRCNN variants in `codes/options/train/`)
- Multi-GPU distributed training:
  - `cd codes && python -m torch.distributed.launch --nproc_per_node=2 train.py -opt options/train/train_DARTSR_RCAN_local.yml --launcher pytorch`

### Test / eval
- Generic SR / DART-SR testing entrypoint:
  - `cd codes && python test.py -opt options/test/test_DARTSR_RCAN.yml`
  - `cd codes && python test.py -opt options/test/test_DARTSR_CARN.yml`
  - `cd codes && python test.py -opt options/test/test_DARTSR_SRResNet.yml`
- Original single-branch SR testing:
  - `cd codes && python test.py -opt options/test/test_RCAN.yml`
  - (or CARN / SRResNet / FSRCNN variants)
- Original ClassSR testing:
  - `cd codes && python test_ClassSR.py -opt options/test/test_ClassSR_RCAN.yml`
  - (or CARN / SRResNet / FSRCNN variants)

### Data prep / sanity checks
- Common DIV2K prep scripts (from README workflow):
  - `cd codes/data_scripts && python data_augmentation.py`
  - `cd codes/data_scripts && python generate_mod_LR_bic.py`
  - `cd codes/data_scripts && python extract_subimages_train.py`
  - `cd codes/data_scripts && python divide_sub_images_train.py`
- Dataloader smoke test:
  - `cd codes/data_scripts && python test_dataloader.py`

### Utility scripts in repo root
- `./prepare_cloud_data.sh /home/featurize/data` (extracts and validates DF2K/DIV2K validation data)
- `./train_cloud.sh 2|3|4` (prepares data and launches the matching B2R-SR RCAN training config)

## High-level architecture

### 1) Entrypoints and runtime modes
- `codes/train.py` + `codes/test.py` are the main pipelines for **standard SR backbones** and **DART-SR plugin mode**.
- `codes/train_ClassSR.py` + `codes/test_ClassSR.py` are the original **ClassSR patch-routing** pipelines.
- Mode is selected by YAML (`model: sr` vs `model: ClassSR`, plus `network_G` settings).

### 2) YAML-driven configuration and path wiring
- All runs go through `codes/options/options.py`:
  - parses YAML,
  - sets `CUDA_VISIBLE_DEVICES` from `gpu_ids`,
  - expands paths,
  - auto-derives experiment/result directories under `experiments/` and `results/`.
- Most operational changes are done by editing option files under:
  - `codes/options/train/*.yml`
  - `codes/options/test/*.yml`

### 3) Model factory split (SR vs ClassSR)
- `codes/models/__init__.py` maps:
  - `model: sr` -> `SRModel` (`codes/models/SR_model.py`)
  - `model: ClassSR` -> `ClassSR_Model` (`codes/models/ClassSR_model.py`)
- `codes/models/networks.py` builds `network_G` by `which_model_G`.

### 4) DART-SR plugin integration (framework-level routing)
- Implemented in `codes/models/archs/dart_sr_plugin_arch.py`.
- `networks.define_G()` switches to plugin wrapper when `network_G.plugin.enable: true`.
- Plugin wraps a backbone (RCAN/CARN_M/MSRResNet), adds:
  - degradation estimator,
  - per-stage router heads,
  - window-level gating (`route_window`),
  - compute-control losses (`loss_budget`, `loss_sparse`, `loss_tv`, `loss_deg`),
  - runtime metrics (`keep_ratio`, `flops_estimated`, `latency_ms`).
- `SRModel` is plugin-aware:
  - splits network output into `(sr, plugin_info)`,
  - adds plugin losses during training,
  - logs/exports plugin metrics,
  - supports loading either full DART-SR checkpoints or plain pretrained backbone weights.

### 5) Original ClassSR path (patch-level routing)
- `ClassSR_Model` uses crop/combine inference over image patches (`patch_size`, `step`) and class-based branch routing behavior from the ClassSR architectures.
- Supports legacy FLOPs/class-count reporting and optional mask visualization output in `test_ClassSR.py`.

### 6) Data layer
- `codes/data/__init__.py` is the dataset/dataloader factory.
- Key SR dataset modes used here:
  - `LQGT` / `LQGT_rcan` for paired training/testing,
  - `LQ` and `LQ_label` for other flows.
- Dataset mode is selected in YAML (`datasets.*.mode`).

## Repository-specific conventions to keep in mind

- Pretrained model paths in configs/scripts commonly use `experiments/pre_trained_models/` (with underscore in `pre_trained`).
- DART-SR local bootstrap config exists at `codes/options/train/train_DARTSR_RCAN_local.yml` and is referenced by `start_training.sh`.
- There is no dedicated lint/test framework config (no `pytest`, `ruff`, or `requirements.txt` found); evaluation is done via the provided training/testing entry scripts and option files.