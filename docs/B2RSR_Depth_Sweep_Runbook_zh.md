# B2R-SR 深度扫描实验操作手册（Featurize 版）

> 更新日期：2026-07-27
> 适用脚本：`scripts/eval/run_depth_sweep.py`、`scripts/eval/run_depth_sweep_and_shutdown.sh`
> 目的：用现有 `120000_G.pth` checkpoint，**零训练**完成 v2 动态深度方向的 Stop/Go 诊断
> （SG-0 诚实基线 / SG-1 深度扫描 / SG-2 组影响 / SG-3 oracle 准备 / SG-5 方案 C kill-check）。
> 预期总费用：按量 RTX 3060 约 1 小时（按分钟计费），跑完自动归还实例。

---

## 1. 一页速查（熟练后只看这节）

```bash
# ① 开按量实例 → 进终端 → 拉代码/确认数据

# ② 冒烟测试（2–3 分钟，盯着跑完）
python scripts/eval/run_depth_sweep.py --datasets Set5 --no-loo \
    --runs 20 --warmup 10 --skip-data-prepare

# ③ 冒烟通过 → 挂机 + 自动归还，断开 SSH 走人
nohup bash scripts/eval/run_depth_sweep_and_shutdown.sh \
    --datasets Set5,BSD100,Urban100 --ssim --skip-data-prepare \
    > ~/depth_sweep_run.log 2>&1 &

# ④ 第二天：微信收到 Featurize 公众号通知 → 网页控制台确认实例"已归还"
#    → 从云盘 work/b2rsr_results/ 下载 B2RSR_DEPTH_SWEEP_*.tar.gz → 丢给 AI 分析
```

---

## 2. 前置条件检查清单

开机后先逐项确认，任何一项不满足都不要启动挂机：

| # | 检查项 | 命令 | 期望 |
|---|---|---|---|
| 1 | GPU 可用 | `nvidia-smi` | 显示 RTX 3060（或所租卡型） |
| 2 | 代码就位 | `ls scripts/eval/run_depth_sweep.py` | 文件存在 |
| 3 | checkpoint 就位 | `ls experiments/remote_exports/B2RSR_RCAN_X4_120000_export/checkpoint/120000_G.pth`（或数据盘上的 export tar） | 存在其一即可，脚本会自动查找 |
| 4 | 数据集就位 | `ls /home/featurize/data/SRBenchmarks/Set5/HR` | 5 张 png |
| 5 | featurize CLI 可用 | `featurize --help` | 打印帮助（自动归还依赖它） |
| 6 | 云盘可写 | `touch /home/featurize/work/.test && rm /home/featurize/work/.test` | 无报错 |

- 数据集不在时：去掉 `--skip-data-prepare`，脚本会自动调用
  `scripts/data/prepare_sr_benchmarks.sh` 下载 EDSR benchmark.tar 并校验。
- checkpoint 不在时：用 `--checkpoint /path/to/120000_G.pth`（或 export .tar）显式指定。

---

## 3. 两段式运行（省钱的关键流程）

### 3.1 第一段：冒烟测试（你在线盯着，2–3 分钟）

```bash
python scripts/eval/run_depth_sweep.py --datasets Set5 --no-loo \
    --runs 20 --warmup 10 --skip-data-prepare
```

看终端输出，按下表决策：

| 观察结果 | 含义 | 动作 |
|---|---|---|
| `V2-G0 exactness: ... PASS` 且 depth 表打印出来 | 管道正常 | 进入第二段 |
| `V2-G0 ... FAIL` | d=G 与 dense 不等价，代码/权重有问题 | **停**，把日志发给 AI 排查，别烧钱跑全量 |
| d=8、d=9 的 ΔY 已经 < −1 dB | 悬崖假设成立，方向大概率死 | 可以只跑 Set5 全套确认后直接归还实例，全量不用跑 |
| 报错（路径/显存/依赖） | 环境问题 | 按报错修复，修不了就归还实例，别挂机 |

### 3.2 第二段：全量挂机 + 自动归还（断开 SSH 走人）

```bash
nohup bash scripts/eval/run_depth_sweep_and_shutdown.sh \
    --datasets Set5,BSD100,Urban100 --ssim --skip-data-prepare \
    > ~/depth_sweep_run.log 2>&1 &
```

启动后可以立即断开 SSH。预计时长约 40–50 分钟，之后实例自动归还、计费停止。

想临时观察进度（可选）：

```bash
tail -f ~/depth_sweep_run.log
```

想反悔取消自动归还（在跑的过程中或最后 60 秒缓冲期内）：

```bash
pkill -f run_depth_sweep_and_shutdown
```

### 3.3 第二天取结果

1. 微信收到 Featurize 公众号通知「B2RSR depth sweep 完成」；
2. 到网页控制台**确认实例状态为"已归还"**（第一次使用务必人工确认这一步！）；
3. 结果在云盘上，归还后不丢。开任意新实例或用网页文件管理下载：

```text
/home/featurize/work/b2rsr_results/
├── B2RSR_DEPTH_SWEEP_<时间戳>.tar.gz   ← 下载这一个文件即可
└── depth_sweep_run_<时间戳>.log        ← 出问题时看这个
```

4. 把 tar.gz 丢给 AI，可直接产出 SG-1（悬崖/平滑）、SG-2（组影响排序）、
   SG-5（方案 C 生死）三个 Stop/Go 判定。

---

## 4. 产物说明（tar.gz 内的 4 个文件）

| 文件 | 内容 | 用途 |
|---|---|---|
| `depth_sweep_report.md` | 人读摘要：等价性判定、逐深度 PSNR 表、LOO 组影响、oracle 阶梯、延迟表、fp16/channels_last 探针、CA gate 统计 | 你自己 5 分钟读结论 |
| `depth_sweep_report.json` | 上述全部 + 每张图逐深度明细 + 运行参数 | AI 分析的主输入 |
| `per_image_depth.csv` | (数据集, 图, 深度) → PSNR-Y/RGB、SSIM、MAE-vs-dense | 画曲线、算置信区间 |
| `ca_gates.npz` | 200 层 × 64 通道的 CA gate 均值/跨图标准差矩阵 | 方案 C（通道裁剪）深入分析 |

---

## 5. 常用参数速查（`run_depth_sweep.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--datasets` | `Set5` | 逗号分隔：`Set5,BSD100,Urban100`（名称需在 test yml 中存在） |
| `--max-images N` | 0（全部） | 每数据集图数上限，冒烟/省钱用 |
| `--ssim` | 关 | 加算 SSIM（CPU 耗时大户，正式跑再开） |
| `--no-loo` | LOO 默认开 | 关闭 leave-one-out 组影响分析 |
| `--loo-images N` | 20 | LOO 只用前 N 张图（省时；0=全部） |
| `--runs / --warmup` | 50 / 30 | 每深度延迟测量次数/预热次数 |
| `--latency-images N` | 3 | 参与延迟测量的图数 |
| `--eps` | `0.05,0.1,0.2,0.3` | oracle 质量约束阶梯（dB） |
| `--no-dense-probes` | 探针默认开 | 关闭 fp16/channels_last 诚实基线探针 |
| `--checkpoint` | 自动查找 | 显式指定 .pth 或 export .tar |
| `--skip-data-prepare` | 关 | 数据已就位时跳过下载校验 |
| `--output-dir / --archive` | 自动时间戳 | 自定义输出位置 |

包装脚本 `run_depth_sweep_and_shutdown.sh` 的环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `RELEASE=0` | 1（归还） | 跑完不归还实例（调试用，**会继续计费**） |
| `NOTIFY=0` | 1（通知） | 不发微信通知 |
| `PERSIST_DIR=...` | `/home/featurize/work/b2rsr_results` | 结果持久化目录 |

示例（调试模式，跑完不归还）：

```bash
RELEASE=0 bash scripts/eval/run_depth_sweep_and_shutdown.sh --datasets Set5 --no-loo
```

---

## 6. 计费与数据安全须知（Featurize 平台特性，已查证）

1. **计费终点是"归还实例"，不是关机。** 操作系统 `shutdown` 后实例可能仍在计费。
   因此包装脚本用官方 CLI `featurize instance release` 归还实例
   （等价于网页上点"归还"，调用平台 API `/virtual_machine/<id>/release`）。
2. **按量计费按分钟计**，不满一小时不扣一小时的钱。
3. **归还实例会删除除云盘外的所有数据**；只有 `/home/featurize/work`（云盘/同步盘）
   在归还后保留。所以结果必须先拷到 work 再归还——包装脚本已自动做这件事。
4. `/home/featurize/data` 是数据盘（放数据集用），**不要**把唯一一份结果只放在实例本地目录。
5. `featurize notify` 通知需要你微信关注 Featurize 公众号。
6. **第一次使用后务必到网页控制台人工确认实例已归还**，验证自动归还链路可靠后再放心挂机。

---

## 7. 脚本逻辑说明

### 7.1 `run_depth_sweep.py`（实验主体，纯推理、零训练）

**核心思想：一次前向拿到全部 11 个深度的结果。**

RCAN 的 body 是 10 个 residual group（RG）顺序堆叠 + 一个尾部 conv + 全局残差。
前缀深度 d 的定义是"只执行前 d 个 RG，其余整组 identity 跳过"。由于 d 的执行是
嵌套的（深度 d+1 的前 d 组计算与深度 d 完全相同），脚本这样做：

```text
instrumented_depth_outputs():
  x = sub_mean → head                        # 只算一次
  逐个 RG 前向，缓存每级中间特征 f_0..f_10    # 只算一次（= 1 次 dense body）
  对每个 f_d：body 尾 conv + 全局残差 + tail   # 11 次，但 tail 只占全网 <2% 计算
  → 得到 outputs[0..10]，其中 outputs[10] 与 dense 逐 bit 相等
```

因此每张图的成本 ≈ 1.1 次 dense 前向，而不是 11 次。这是整个实验能压进
一小时的关键。

**执行顺序与各步产出：**

| 步骤 | 做什么 | 对应 Stop/Go |
|---|---|---|
| ① V2-G0 等价性 | 第一张图上比较 `outputs[10]` 与原始 `rcan(lq)` 的 max\|diff\|；不为 0 且超容差则判 FAIL 并提示停止 | 正确性硬门（本地 CPU 已验证 = 0.0） |
| ② 延迟扫描 | 对 d=0…10，用 `truncated_forward`（真正只执行前 d 组的独立路径，不是缓存复用）× 3 图 × 50 次 CUDA events 计时，报 median/p90 | SG-1 的延迟分母 |
| ③ 质量扫描 | 每个数据集逐图跑 instrumented 前向，记录逐深度 PSNR-Y/RGB/SSIM、与 dense 的 MAE；同时第一个数据集顺带用 forward hook 采集全部 200 个 CALayer 的通道门控值 | SG-1 曲线 + SG-5 数据 |
| ④ LOO 组影响 | 前 20 张图上，每次恰好跳过 1 个 RG（其余 9 组照常），测 ΔPSNR-Y → 10 个组的重要性排序 | SG-2（非前缀对照） |
| ⑤ oracle 阶梯 | 纯离线计算：对每个 ε ∈ {0.05,0.1,0.2,0.3} dB，逐图找满足 `PSNR ≥ dense−ε` 的最小深度，结合②的延迟表得 oracle 平均延迟 | SG-3 准备（仅上界分析，不是部署结果） |
| ⑥ dense 探针 | 同一张图上测 eager fp32 / autocast fp16 / channels_last 三种 dense 配置的延迟与 PSNR 差 | SG-0（诚实基线初筛） |
| ⑦ 导出 | 全部结果写入时间戳目录（json + md + csv + npz），打成一个 tar.gz，终端打印下载路径 | — |

**设计要点：**

- backbone 从 `120000_G.pth` 中按 `backbone.` 前缀提取、strict 加载，全程 `eval()` +
  `requires_grad=False`，权重零改动；
- 延迟一律 CUDA events + 预热 + 同步，不用 FLOPs 替代；
- oracle 只做离线上界估计，报告中已注明"不是可部署路由器"；
- 兼容纯 RCAN checkpoint（无 `backbone.` 前缀时整体加载）。

### 7.2 `run_depth_sweep_and_shutdown.sh`（包装：持久化 + 自动归还）

```text
① cd 到仓库根目录，透传全部参数执行 run_depth_sweep.py
② 记录退出码（成功失败都继续，保证后续持久化一定执行）
③ 把 12 小时内生成的 B2RSR_DEPTH_SWEEP_*.tar.gz 拷到云盘
   /home/featurize/work/b2rsr_results/（归还实例后不丢）
④ 把运行日志也拷一份到云盘（失败时第二天能看 traceback）
⑤ NOTIFY=1 时：featurize notify 发微信公众号通知
⑥ RELEASE=1 时：sleep 60（给你反悔的窗口）
   → featurize instance release 归还实例（计费停止）
   → 若 release 失败：回退 shutdown 并在日志打出大写警告
     （注意：回退关机不保证停止计费，第二天要人工检查）
⑦ RELEASE=0 时：保持运行并明确打印"继续计费中"
```

容错设计：`set -uo pipefail` 但**不带 `-e`**——实验失败时脚本不会中途退出，
仍会执行持久化和归还，避免"实验半夜崩了、实例空转烧钱到天亮"这种最坏情况。

---

## 8. 故障排查

| 症状 | 可能原因 | 处理 |
|---|---|---|
| `未找到 checkpoint` | export 未拉取到实例 | `--checkpoint` 显式指定，或先 `featurize dataset extract <id>` |
| `未知数据集 X` | 名称与 test yml 不符 | 用 `Set5/Set14/BSD100/Urban100/Manga109`（区分大小写不敏感） |
| CUDA OOM（Urban100 大图） | 3060 12GB 一般够；若租了小显存卡 | 先 `--datasets Set5,BSD100`，Urban100 换卡再跑 |
| V2-G0 FAIL | 权重加载或模块顺序问题 | 停，发日志给 AI；此时其他一切结果无效 |
| 第二天实例还在计费 | release 失败走了 shutdown 回退 | 手动网页归还；把 `depth_sweep_run_*.log` 里 WARN 行发给 AI |
| 微信没收到通知 | 未关注公众号 / NOTIFY=0 | 不影响实验，直接看网页控制台 |
