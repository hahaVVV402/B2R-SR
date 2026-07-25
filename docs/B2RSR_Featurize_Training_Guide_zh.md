# B2R-SR Featurize 云端测试与训练指南

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
  --index-url https://download.pytorch.org/whl/cu121
```

`nvidia-smi` 显示的是驱动支持的最高 CUDA 版本；较新的 NVIDIA 驱动可以运行 cu121 PyTorch wheel。无需另外安装系统 CUDA、`cudatoolkit` 或 NVIDIA 驱动。

安装项目依赖。仓库提供统一的 `requirements.txt`：

```bash
cd /home/featurize/work/B2R-SR
python -m pip install -r requirements.txt
```

PyTorch 因 CUDA/CPU/macOS 构建不同，不写入通用 requirements；应先按上一步单独安装。

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

`train_cloud.sh` 默认使用正式的 120000-step 配置。首次测试前，应临时编辑：

```text
codes/options/train/train_B2RSR_RCAN_X4.yml
```

建议短测试参数：

```yaml
name: B2RSR_RCAN_X4_SMOKE

datasets:
  train:
    n_workers: 2
    batch_size: 1

train:
  niter: 20
  val_freq: 10

logger:
  print_freq: 1
  save_checkpoint_freq: 10
```

在 tmux 中运行：

```bash
tmux new -s b2rsr-x4-smoke
cd /home/featurize/work/B2R-SR
./train_cloud.sh 4 2>&1 | tee train_x4_smoke.log
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

## 6. 正式训练

短测试通过后，将 X4 YAML 恢复为正式参数：

```yaml
datasets:
  train:
    n_workers: 4
    batch_size: 1  # 确认显存充足后再尝试 2 或 4

train:
  niter: 120000
  val_freq: 2000

logger:
  print_freq: 100
  save_checkpoint_freq: 2000
```

将实验名从 smoke 名称恢复为正式名称：

```yaml
name: B2RSR_RCAN_X4
```

启动：

```bash
tmux new -s b2rsr-x4
cd /home/featurize/work/B2R-SR
./train_cloud.sh 4 2>&1 | tee train_x4_launcher.log
```

查看训练输出：

```bash
tail -f train_x4_launcher.log
```

模型和训练状态保存在：

```text
/home/featurize/work/B2R-SR/experiments/B2RSR_RCAN_X4/
├── models/
├── training_state/
├── val_images/
└── train_B2RSR_RCAN_X4.log
```

---

## 7. 最终执行顺序

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

## 8. 关键原则

- 不要把 100GB DF2K 放到 `/home/featurize/work`；
- 不要在便宜实例完整解压后再换 GPU；
- checkpoint、training state 和日志必须保存在 `/home/featurize/work`；
- 正式训练前必须完成一次真实数据短训练；
- 首次使用完整 RCAN 时从 `batch_size: 1` 开始。
