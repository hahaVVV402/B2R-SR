# B2R-SR Featurize 云端测试与训练指南

> **当前入口（2026-08-09）：** `results/autonomous_goals/20260809-132635/START_HERE.md`。该入口运行静态 EDSR-L 32→24 的冻结500-step三尺度恢复，并自动校验、导出和归还实例。本文后续 `train_cloud.sh`、B2R-SR plugin和120000-step RCAN内容是历史动态路线，不能用于当前方法；仅存储目录、环境检查和`featurize instance release`原则仍适用。

## 1. 推荐策略

由于 DF2K 超过 100GB，而 Featurize 的 `/home/featurize/data` 会在实例归还或重置后清空，不建议在便宜实例上完整解压数据后再更换 GPU。推荐分成两个阶段：

| 阶段 | 推荐 GPU | 用途 |
|---|---|---|
| 代码检查 | RTX 3060 12GB | 只运行无数据 CUDA smoke test，不下载 DF2K |
| 真实测试与正式训练 | RTX 4090 24GB | 准备真实数据、短训练、正式训练，保持同一实例 |

不推荐使用 PRO 6000 96GB：当前模型用不到这么大显存，成本过高。RTX 3080 10GB 和单卡 RTX 2080 Ti 11GB 对完整 64-channel RCAN 风险较高。

---

## 2. 存储目录

```text
/home/featurize/work/       # 持久化同步盘
└── B2R-SR/                 # 代码、checkpoint、日志、结果

/home/featurize/data/       # 实例本地高速盘，实例归还后清空
├── DF2K/
├── DIV2K_valid_HR.zip
├── DIV2K_valid_LR_bicubic_X2.zip
├── DIV2K_valid_LR_bicubic_X3.zip
└── DIV2K_valid_LR_bicubic_X4.zip
```

代码和模型输出放在 `/home/featurize/work`，训练数据放在 `/home/featurize/data`。

---

## 3. 阶段一：RTX 3060 无数据测试

### 3.1 创建实例并选择基础镜像

选择 RTX 3060 12GB 按量实例。这个阶段不需要添加 DF2K 数据集。

推荐选择 Featurize 官方的干净基础镜像：

```text
CUDA 12.4 + Miniconda Base
```

实例启动后不要先安装软件，第一条命令检查 GPU：

```bash
nvidia-smi
```

必须正常显示 RTX 3060。如果返回 `No devices were found`，说明该实例或镜像的 GPU 挂载异常，应更换实例/镜像，不要尝试在容器内重装 NVIDIA 驱动。

### 3.2 创建 B2R-SR Conda 环境

不要污染 Miniconda 的 `base` 环境：

```bash
conda create -n b2rsr python=3.10 pip -y
conda activate b2rsr
python -m pip install --upgrade pip
```

安装已经验证过的 PyTorch 2.2.2 + CUDA 12.1：

```bash
python -m pip install \
  torch==2.2.2 \
  torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

`torchvision` 是 `codes/utils/util.py` 的运行时依赖，版本必须与 PyTorch 匹配。`nvidia-smi` 显示的是驱动支持的最高 CUDA 版本；较新的 NVIDIA 驱动可以运行 cu121 wheel。无需另外安装系统 CUDA、`cudatoolkit` 或 NVIDIA 驱动。

安装项目依赖。仓库提供统一的 `requirements.txt`：

```bash
cd /home/featurize/work/B2R-SR
python -m pip install -r requirements.txt
```

`requirements.txt` 同样固定了这两个版本并配置 cu121 额外索引；已按上一步安装时，重复执行只会确认版本满足要求。

### 3.3 验证依赖和 PyTorch CUDA 环境

仓库提供检查脚本，会验证所有核心依赖、CUDA 可用性并执行一次 GPU 矩阵运算：

```bash
cd /home/featurize/work/B2R-SR
python codes/check_environment.py --cuda
```

预期至少包含：

```text
PyTorch: 2.2.2+cu121
PyTorch CUDA: 12.1
CUDA available: True
GPU count: 1
GPU: NVIDIA GeForce RTX 3060
CUDA tensor: cuda:0
```

### 3.4 拉取代码

```bash
cd /home/featurize/work
git clone https://github.com/hahaVVV402/ClassSR.git B2R-SR
cd B2R-SR
```

如果 GitHub 仓库已改名为 `B2R-SR`，使用：

```bash
git clone https://github.com/hahaVVV402/B2R-SR.git B2R-SR
```

如果仓库已经存在：

```bash
cd /home/featurize/work/B2R-SR
git pull origin main
```

### 3.5 运行 smoke test

```bash
python codes/test_b2rsr_flow.py --cuda
```

预期最后出现：

```text
ALL DART-SR / B2R-SR FLOW TESTS PASSED
```

通过后即可归还 RTX 3060 实例。

---

## 4. 阶段二：RTX 4090 真实数据流程

### 4.1 创建实例

选择 RTX 4090 24GB 按量实例。真实流程跑通后，可以在同一实例上继续训练或转长租，避免重新准备数据。

如果已经将阶段一的正常环境保存为自定义镜像，直接选择该镜像，并执行：

```bash
conda activate b2rsr
nvidia-smi
```

如果没有创建自定义镜像，则在 4090 实例上重复第 3.2、3.3 节的 Conda/PyTorch 环境安装与验证。无论使用什么镜像，都必须先确认 `nvidia-smi` 和 `torch.cuda.is_available()` 正常。

### 4.2 添加数据集

通过 Featurize 工作区将以下数据集下载到 `/home/featurize/data`：

```text
DF2K.zip
DIV2K_vaild100.zip
```

检查：

```bash
ls -lh /home/featurize/data
```

### 4.3 更新代码

```bash
cd /home/featurize/work/B2R-SR
git pull origin main
chmod +x prepare_pretrained.sh prepare_cloud_data.sh train_cloud.sh
```

### 4.4 准备原版 RCAN 权重

```bash
./prepare_pretrained.sh
```

脚本会从自己的 Google Drive 下载并校验权重，然后保存为：

```text
experiments/pre_trained_models/RCAN_BIX2.pt
experiments/pre_trained_models/RCAN_BIX3.pt
experiments/pre_trained_models/RCAN_BIX4.pt
```

这些文件位于 `/home/featurize/work`，实例归还后仍会保留。

### 4.5 只准备和检查 X4 数据

```bash
./prepare_cloud_data.sh /home/featurize/data 4
```

预期检查结果：

```text
训练 GT/LQ：138849 对
验证 GT/LQ：100 对
```

脚本支持：

- 已解压的 `DF2K/`；
- 只有 `DF2K.zip` 的情况；
- 验证集外层合集 `DIV2K_vaild100.zip`；
- 四个独立的 DIV2K validation ZIP；
- 同一实例二次运行时快速跳过。

---

## 5. 短训练测试

仓库提供独立的 X4 smoke 配置：

```text
codes/options/train/train_B2RSR_RCAN_X4_smoke.yml
```

它使用与正式训练一致的 `GT_size: 192`、`batch_size: 48` 训练 20 steps，并在第 10、20 step 验证和保存，不会修改或覆盖正式配置。B2R-SR 的退化代理目标使用 batch 内归一化，因此不能使用会使目标恒为 1 的 `batch_size: 1`。

在 tmux 中运行：

```bash
tmux new -s b2rsr-x4-smoke
cd /home/featurize/work/B2R-SR
./train_cloud.sh 4 smoke 2>&1 | tee train_x4_smoke.log
```

需要确认：

- [ ] RCAN X4 权重成功加载
- [ ] DF2K dataloader 成功创建
- [ ] 前向传播成功
- [ ] 反向传播成功
- [ ] B2R-SR plugin losses 正常输出
- [ ] DIV2K validation 能计算 PSNR
- [ ] `models/` 中生成 checkpoint
- [ ] `training_state/` 中生成 `.state`

退出 tmux 但保持任务运行：

```text
Ctrl+B，然后按 D
```

重新进入：

```bash
tmux attach -t b2rsr-x4-smoke
```

---

## 6. RTX 4090 训练几何基准

X4 下 `GT_size: 96` 只产生 24×24 的 LQ patch，并且每张图只有 9 个 8×8 路由窗口，不能仅沿用 ClassSR 参数。正式训练前运行独立 GPU 基准：

```bash
cd /home/featurize/work/B2R-SR/codes
python benchmark_b2rsr_training.py --phase soft
```

默认比较 `GT96×batch16/32`、`GT128×batch16` 和 `GT192×batch8/12/16`，每组 warmup 20 steps、计时 100 steps，不读取验证集也不保存 checkpoint。输出包括 step 时间、images/s、LR Mpix/s 和 CUDA 峰值显存。

选择保留显存在 22GiB 内、吞吐距离最佳值不超过 5%、且每图至少有 36 个路由窗口的配置。然后用同一候选测试 hard-routing：

```bash
python benchmark_b2rsr_training.py --phase hard --cases 192x8,192x12,192x16
```

最终 batch 和 GT size 必须依据这台 4090 的实测结果确定；正式方法与 baseline 还应保持相同的数据/像素预算。

需要一次测试 X2/X3/X4 时，运行仓库根目录的批量脚本：

```bash
cd /home/featurize/work/B2R-SR
./benchmark_all_scales.sh all
```

脚本为三个尺度统一使用 48×48 LR patch（对应 GT 96/144/192），分别测试 batch 16/24/32/40/48 的 soft 与 hard 路径，并生成六个 `benchmark_X{scale}_{phase}.log`。基准只使用合成张量测试真实模型前向/反向，不修改正式 YAML、不读取数据集、不保存 checkpoint。

RTX 4090 实测三个尺度均由 batch 48 取得最高 soft/hard 吞吐，且峰值 reserved 显存低于 19GB，因此正式配置为：

| 尺度 | GT size | LR patch | 路由窗口/图 | Batch |
|---:|---:|---:|---:|---:|
| X2 | 96 | 48×48 | 36 | 48 |
| X3 | 144 | 48×48 | 36 | 48 |
| X4 | 192 | 48×48 | 36 | 48 |

---

## 7. 正式训练

性能基准和 batch-48 smoke test 通过后，正式 X4 配置保存在：

```text
codes/options/train/train_B2RSR_RCAN_X4.yml
```

它使用 `GT_size: 192`、`batch_size: 48` 训练 120000 steps。RTX 4090 实测 soft-routing 为 188.50 images/s、约 18.1GB reserved，hard-routing 为 141.71 images/s、约 19.7GB reserved；两条路径都在 24GB 显存的安全范围内。X4 的 192 HR patch 对应 48×48 LR patch 和每图 36 个 8×8 路由窗口。

启动：

```bash
tmux new -s b2rsr-x4
cd /home/featurize/work/B2R-SR
./train_cloud.sh 4
```

框架会把带时间戳的正式日志直接写入实验目录，无需额外使用 `tee`。查看最新日志：

```bash
LOG=$(find experiments/B2RSR_RCAN_X4 -maxdepth 1 -name 'train_*.log' | sort | tail -1)
tail -f "$LOG"
```

模型和训练状态保存在：

```text
/home/featurize/work/B2R-SR/experiments/B2RSR_RCAN_X4/
├── models/
├── training_state/
├── val_images/
├── tensorboard/
│   └── events.out.tfevents...
└── train_B2RSR_RCAN_X4_时间戳.log
```

全新启动同名实验时，整个旧实验目录（包括 TensorBoard）会自动归档；从 `resume_state` 恢复时则继续写入同一实验目录。查看所有实验：

```bash
tensorboard --logdir experiments --host 127.0.0.1 --port 6006
```

---

## 8. 最终执行顺序

```text
RTX 3060
  1. 拉取代码
  2. 安装依赖
  3. 运行 test_b2rsr_flow.py --cuda
  4. 通过后归还

RTX 4090
  1. 添加 DF2K 和 DIV2K validation 数据集
  2. git pull
  3. prepare_pretrained.sh
  4. prepare_cloud_data.sh /home/featurize/data 4
  5. 20-step 短训练
  6. 恢复正式配置
  7. 在同一实例继续正式训练
```

## 9. 关键原则

- 不要把 100GB DF2K 放到 `/home/featurize/work`；
- 不要在便宜实例完整解压后再换 GPU；
- checkpoint、training state 和日志必须保存在 `/home/featurize/work`；
- 正式训练前必须完成一次真实数据短训练；
- X2/X3/X4 正式训练统一使用 48×48 LR patch 和经 RTX 4090 soft/hard 基准选出的 `batch_size: 48`，对应 GT size 为 96/144/192；真实数据若 OOM 再降到 batch 40。
