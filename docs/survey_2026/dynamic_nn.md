# 动态神经网络 / 条件计算 / 自适应推理（2021–2026）：理论与工程现状调研

**核心问题：为何理论 FLOPs 节省常无法转为真实 wall-clock 加速？**

调研日期：2026-02。所有论文均经网络检索验证存在；未能核对细节处标注 [未核验]。

---

## TL;DR（一句话答案）

FLOPs 只统计乘加次数，而 GPU 延迟由 **内存带宽、kernel 启动开销、并行度/占用率、gather-scatter 数据搬运、动态控制流破坏静态图优化** 共同决定。动态网络省掉的往往是"便宜的"计算（compute），留下的是"昂贵的"访存与调度（memory + scheduling），因此细粒度动态性在 GPU 上通常兑现不了加速；只有**粗粒度（block/layer/channel-group 级）+ 硬件感知调度（latency 预测/查表）+ 静态化执行（CUDA Graphs / 融合 kernel）**三者齐备时才能兑现。

---

## 1. 综述与批判：FLOPs–latency 脱节的文献定位

| 论文 | 出处 | 关键内容 |
|---|---|---|
| **Dynamic Neural Networks: A Survey** — Han, Huang, Song, Yang, Wang, Wang | TPAMI 2021 (arXiv:2102.04906, DOI 10.1109/TPAMI.2021.3117837) | 动态网络三分类：sample-wise / spatial-wise / temporal-wise。综述本身按算法类别组织，**缺乏跨硬件的标准化延迟评测**；"实际加速"被列为开放问题而非已解决问题 |
| **Conditional Computation in Neural Networks: Principles and Research Trends** — Scardapane et al. | *Intelligenza Artificiale* 18(1), 2024 (arXiv:2403.07965) | 统一形式化框架；三大家族：MoE、token selection、early-exit。讨论效率之外的可解释性/迁移收益。同样承认工程落地依赖底层支持 |
| **LAUDNet** — Han et al. | TPAMI 2024 (arXiv:2308.15949) | **摘要中明确指出理论-实际效率差距的三个来源**：(1) 研究碎片化缺乏统一方法；(2) 重算法设计、轻调度策略（尤其 CUDA GPU 场景）；(3) 库以静态算子为主，动态算子延迟难以测量 |
| Pattern Recognition 2026 动态网络新综述（S0262885626000879） | ScienceDirect | 批评既有文献 GFLOPs vs latency 报告不一致，提出更完整 benchmark 模板 [未核验细节] |

**批判性共识**：FLOPs 是效率的弱代理；动态控制流引入路由决策、分支、kernel 启动低效、内存搬运、硬件利用率下降，理论节省可能变成零甚至负的延迟收益，除非与运行时调度和硬件感知实现协同设计。

---

## 2. 硬件友好粒度：哪种路由粒度能兑现 GPU 加速（定量证据）

### 2.1 关键论文与实测数字

| 论文 | 粒度 | 实测加速（GPU） | 备注 |
|---|---|---|---|
| **SBNet** (Ren et al., CVPR 2018, arXiv:1801.02108) | 空间 **block 级**（tiled block sparsity，块级 gather→dense conv→scatter） | ResNet block 最高 **2.3×**；LiDAR 检测 pipeline 端到端 **1.8×** | 最早证明：把空间稀疏聚合成规则大块，就能用 dense kernel 吃到加速。掩码需低分辨率、块状 |
| **DynConv** (Verelst & Tuytelaars, CVPR 2020, arXiv:1912.03458) | **像素级**空间稀疏（Gumbel 掩码 + 自研 sparse conv CUDA kernel） | 报告了相对 dense baseline 的实测加速（依赖专用 gather/scatter kernel），但加速远低于 FLOPs 节省比例 | 说明像素粒度必须写专用 kernel 才有正收益；通用框架下为负 |
| **LASNet** (Han et al., NeurIPS 2022, arXiv:2210.06223) | **可调空间粒度**（coarse patch 级，latency 指导选粒度） | ResNet-101/ImageNet, t=0.4：**V100 延迟 −36%（≈1.56×）；Jetson TX2 −46%（≈1.85×）**，精度无损 | 关键发现：**TX2（算力受限）上实际加速接近理论 FLOPs 节省；V100（带宽/并行度富余）上差距大**——同一算法，硬件不同兑现率不同 |
| **LAUDNet** (Han et al., TPAMI 2024, arXiv:2308.15949) | 统一三种粒度：spatial（patch size 可选 1/2/4/8…）、layer skipping、channel skipping（组级） | ResNet-101 在 **V100 / RTX3090 / TX2 上延迟降低超过 50%** | 用 **latency predictor** 而非 FLOPs 指导粒度选择；结论：**在强 GPU 上应选粗 patch（如 S=4/8）与 channel-group 粒度，细粒度（S=1）无法兑现** [具体各粒度表格数字未核验] |

### 2.2 Latency lookup table / latency predictor 方法论

- **来源谱系**：硬件感知 NAS（如 FBNet、OFA arXiv:1908.09791、AOWS CVPR 2020）以**每算子实测延迟查表**替代 FLOPs 代理 → LASNet/LAUDNet 把该思想引入动态网络：对每个（算子类型 × 粒度 × 稀疏率 × 硬件）组合建模/查表预测延迟，然后**在延迟空间而非 FLOPs 空间做粒度选择与训练目标**。
- LAUDNet 的 latency predictor 同时建模**调度策略**（mask 生成、gather/scatter、内存布局），因为动态算子延迟 ≠ 各静态算子延迟之和。
- 方法论要点：(1) 延迟必须按目标硬件逐一测/建模，不可迁移假设；(2) 预测器要覆盖稀疏率-延迟曲线的**非线性平台区**（稀疏率低于某阈值前延迟几乎不降）；(3) 粒度是搜索变量而非固定超参。

### 2.3 粒度结论（回答"哪些能兑现、哪些不能"）

**能兑现（有定量证据）**：
- 空间 **block/patch 级**（SBNet 2.3×、LASNet 1.5–1.85×、LAUDNet >2×）
- **layer 级跳层**（整层 skip = 少发射整段 kernel，无 gather/scatter；LAUDNet 证实有效）
- **channel-group 级**（组粒度保持规则内存访问；LAUDNet）
- **2:4 半结构化稀疏**（硬件原生支持，1.3–1.8×，见 §5）
- **early-exit（batch=1 或逐样本流式）**（直接砍掉后续全部计算）

**难以/不能兑现**：
- **像素级空间稀疏**（无专用 kernel 时为负收益；有专用 kernel 也远低于理论值——DynConv）
- **非结构化权重稀疏**（通用框架下几乎无加速，见 §5）
- **单 channel 级动态剪枝**（破坏 GEMM 规则性，索引开销 > 节省 [未核验专门对比论文，但为 LAUDNet/社区共识]）
- **大 batch 下的 per-sample early-exit / token 路由**（batch 内分歧导致 padding 到最慢样本；见 §6）

---

## 3. GPU 执行层：定量机制

### 3.1 Kernel launch overhead 与 batch=1 launch-bound

- 每个 kernel 的驱动侧启动开销约 **5–15 µs**（NVIDIA TensorRT Developer Guide）；GPU 侧小 kernel 本身可能只有 2–10 µs，此时 CPU enqueue 时间 ≥ GPU 计算时间，即 **enqueue-bound / launch-bound**。
- batch=1 推理由几十到几百个短 kernel 组成时，纯启动开销可累积到**数百 µs–数 ms**。
- 动态网络恰恰在此雪上加霜：路由决策常需 **GPU→CPU 同步**（读取 mask/决策），每次同步为数十 µs 且打断流水。

### 3.2 CUDA Graphs

- Ampere+ 上直线型 graph 重放开销从 ~2 µs + 200 ns/节点降至 **~2.5 µs + ~1 ns/节点**（NVIDIA 官方 blog），即整图重放近似常数时间。
- llama.cpp 在 batch=1 时默认启用 CUDA Graphs；实测研究报告 H100 上 batch=1 decode 延迟降低 **1.259×**（arXiv:2605.30571 [未核验全文]）。
- **与动态网络的冲突**：CUDA Graphs 要求静态拓扑与静态形状。数据依赖的分支/可变形状会导致 graph 失效或需多 graph 缓存（每种路由路径一张图，路径组合爆炸）。这是"动态性 vs 静态化优化"的根本张力——早退型（路径数少、前缀共享）比 token 级路由（组合多）更容易 graph 化。

### 3.3 Gather/scatter 代价

- Gather/compress/scatter kernel 本身是 **memory-bound**（PyTorch semi-structured sparsity 官方 blog 明确指出）；动态稀疏若不能减少总内存流量，光减 FLOPs 无效。
- SpInfer (EuroSys 2025, DOI 10.1145/3689031.3717481)：非结构化剪枝的收益是"elusive"的，需 bitmap 编码 + 定制 kernel 才拿到端到端 **≤1.58×**。
- 系统层共识：稀疏格式/kernel 协同设计是必要条件，否则索引与搬运开销吞掉全部节省。

### 3.4 算子融合

- 融合减少 kernel 数（→减启动开销）和中间张量读写（→减内存流量）。动态掩码/路由若插在两个可融合算子之间，会**破坏融合机会**，产生隐性代价——即使动态分支本身免费。SBNet 式做法（gather 后跑连续多层 dense 再 scatter）本质上是把融合区间做大。

---

## 4. 【重点】Early-exit 的统计风险控制

### 4.1 已验证的论文清单

| 论文 | 出处 | 保证类型 | 任务 |
|---|---|---|---|
| **CALM: Confident Adaptive Language Modeling** — Schuster et al. | NeurIPS 2022 | **Learn-then-Test / distribution-free risk control**：校准逐 token 早退阈值，使相对全模型的期望质量退化 ≤ 用户容忍度 δ（以概率 1−ε） | LLM 解码；报告最高 **~3× 加速**且"provably"保持性能 |
| **Fast yet Safe: Early-Exiting with Risk Control** — Jazbec, Timans, et al. | NeurIPS 2024 (arXiv:2405.20915) | **Conformal Risk Control (CRC) + UCB/LTT** 事后校准退出阈值 λ，控制"performance gap risk"（早退输出 vs 全网络输出的差距）期望 ≤ ε | 覆盖**分类、语义分割、语言建模（CALM 设置）、图像生成（早退扩散）**——即已含非分类/生成式任务 |
| **Early-Exit Neural Networks with Nested Prediction Sets** — Jazbec et al. | UAI 2024 (PMLR v244) | 嵌套预测集：各出口的 conformal 预测集单调嵌套，任意退出时刻集合仍有效（anytime-valid 方向） | 分类为主 |
| **SAFE-KD: Risk-Controlled Early-Exit Distillation for Vision Backbones** | arXiv:2602.03043 (2026) | 早退蒸馏 + conformal 式 selective-risk 控制 | 视觉骨干 [细节未核验，仅确认存在] |
| **CALM 前身：Consistent Accelerated Inference via Confident Adaptive Transformers (CAT)** | EMNLP 2021 (arXiv:2104.08803) | conformal 一致性保证（与原模型预测一致的概率 ≥1−ε） | NLP 分类 [编号未核验] |

### 4.2 回归任务 / PSNR 保证的先例

- **Image Super-Resolution with Guarantees via Conformal Generative Models** — Adame, Csillag, Goedert, NeurIPS 2025 (arXiv:2502.09664)：conformal 方法用于 SR，**明确证明可控制预测图像的 PSNR**。这是"conformal + 图像恢复 + PSNR 保证"的最直接先例——但它是**静态模型的不确定性量化，不是自适应计算的退出准则**。
- **Fast yet Safe (NeurIPS 2024)** 中的风险控制框架对损失函数是通用的（任何有界单调损失均可套 CRC），其图像生成（扩散早退）实验已是回归式输出；**但据检索所见，尚无工作把 CRC 直接用于"早退超分/图像恢复网络的 PSNR/失真保证"**。
- Conformal 回归基础工具链成熟：Conformalized Quantile Regression (Romano et al., NeurIPS 2019)、Distribution-Free Predictive Inference for Regression (Lei et al.)、Conformal Risk Control (Angelopoulos et al., ICLR 2024 [编号未核验])。

### 4.3 现状与空白（回答第三个必答问题）

**现状**：
- 分类/生成式 LLM 的早退风险控制已相对成熟（CAT→CALM→Fast yet Safe），标准配方是：held-out 校准集 + LTT/CRC 选阈值 + "相对全模型的 performance gap"作为风险量（而非绝对精度，从而免去 ground truth 标签或放宽要求）。
- 保证形式均为 **marginal**（对校准/测试分布上的期望或高概率控制），依赖 i.i.d./可交换性。

**空白**（对回归/图像恢复尤其明显）：
1. **回归型自适应计算的质量保证几乎空白**：没有已发表工作对"早退 SR/去噪网络"给出 per-image 或期望 PSNR/SSIM 下界的 distribution-free 保证。SR-with-guarantees（NeurIPS 2025）控制 PSNR 但不做自适应计算；Fast yet Safe 做自适应计算但未做 PSNR 任务。**两者的交集是一个明确的开放机会。**
2. **空间粒度的风险控制缺失**：现有保证都在 sample/token 级退出；对 patch 级/区域级动态路由（LASNet 式）的区域质量保证无先例 [未检索到]。
3. Conditional（per-input）而非 marginal 保证、分布漂移下的保证、以及"延迟-风险"联合控制（同时保证 P95 延迟与质量）均属空白或极初期。

---

## 5. 结构化 vs 非结构化稀疏：GPU 实测差异

| 稀疏类型 | 理论上限 | 实测 | 来源 |
|---|---|---|---|
| **2:4 半结构化**（Ampere Sparse Tensor Core） | 2× 数学吞吐 | GEMM 层面 **~1.39×**（A100，538→387 µs，PyTorch 官方）；ViT-L MLP fwd+bwd **~1.3×**；TensorRT 端到端 **1.3–1.8×**（视模型与 batch） | NVIDIA TensorRT sparsity blog；PyTorch blog "accelerating-neural-network-training" |
| **50% 非结构化** | 2× FLOPs 节省 | 通用框架下 **≈1×（无加速）**；需专用引擎（SparseRT、DeepSparse/CPU、SpInfer bitmap kernel ≤1.58×）才有部分兑现 | SparseRT arXiv:2008.11849；SpInfer EuroSys'25；ICLR 2024 DST 论文（OpenReview PxoFut3dWW，A6000 上直接对比 2:4 vs 50% 非结构化） |
| **Block sparsity**（≥16×16 块） | 随稀疏率 | 块足够大时接近 dense GEMM 效率（cuSPARSE blocked-ELL / Triton blocksparse） | NVIDIA block-sparse tensor core blog |

**结论**：2:4 真实收益约为理论的 65–90%（1.3–1.8×/2×），且**只加速 GEMM 本体**，端到端被非 GEMM 部分（Amdahl）进一步稀释；非结构化稀疏在 GPU 上默认不兑现。这与动态网络的启示一致：**规则性（regularity）是兑现加速的货币**。

---

## 6. Mixture-of-Depths / Slimmable / Anytime 的工程可行性

### 6.1 Mixture-of-Depths (MoD)

- 原论文 (Raposo et al., arXiv:2404.02258)：等 FLOPs 下匹配 baseline 的**训练 wall-clock**（因 top-k 路由保持静态张量形状——这是 MoD 的关键工程设计），采样阶段报告"最高 50% 更快"。
- **MoDification** (NAACL 2025)：现实检验——vanilla MoD 难以低成本 retrofit 进已有 LLM；改造后实测 **~1.2× 延迟加速、~1.8× 内存节省**，长上下文场景收益最大。
- 工程判断：MoD 的可行性恰来自其**放弃 per-token 自由路由、改用容量固定的 top-k**（静态形状 → 可融合/可 graph 化）；但 autoregressive 推理时 top-k 的非因果性带来额外复杂度。实际是"定向优化"而非普适大加速。

### 6.2 Slimmable / 可切换宽度

- Slimmable Networks (Yu et al., ICLR 2019, arXiv:1812.08928)：单模型多宽度 + switchable BN。每个宽度都是**规则 dense 子网**，切换后延迟完全可预测——工程上最稳妥的"动态性"（实为少数静态模式间切换）。
- 但宽度减半 ≠ 延迟减半（小宽度下 GPU 利用率不足）；实践上 OFA (arXiv:1908.09791) 式"训练超网 + 按设备/延迟预算特化子网"更常用。Dynamic Slimmable Network (CVPR 2021) 把宽度决策做成 per-input，则重新引入 batch 分歧问题。
- **结论：可行性高**，代价是没有 per-input 细粒度节省，只有模式级切换。

### 6.3 Anytime prediction / MSDNet 式早退

- MSDNet (ICLR 2018) 定义 anytime 与 budgeted-batch 两种设置。
- **batch=1 / 流式：可行且收益直接**（退出即停止全部后续 kernel）。
- **大 batch：困难**——batch 内退出深度分歧导致要么按最慢样本 pad，要么做动态重组 batch（复杂调度）；TMLR/OpenReview 结果指出**batch 越大早退延迟收益越差**；系统论文（arXiv:2204.05223, arXiv:2407.20272）将 batching+early-exit 联合调度列为专门难题。
- 服务化的现实方案：exit 后重新组 batch（continuous batching 思想，LLM serving 已内置这种机制，故 LLM 早退比 CNN 大 batch 早退更工程可行）。

---

## 7. 必答问题汇总

### Q1. 哪些路由粒度被实证能在 GPU 兑现加速？

**能（定量证据见 §2.1/§5）**：block/patch 级空间稀疏（SBNet 2.3×、LASNet 1.56–1.85×、LAUDNet >2×/−50%+）；整层跳过；channel-group；2:4（1.3–1.8×）；batch=1 早退（CALM ~3×）；固定容量 top-k 路由（MoD 训练期等 wall-clock、推理 ~1.2×）。

**不能/极难**：像素级掩码（无专用 kernel 为负）；非结构化权重稀疏（≈1×）；单 channel 动态剪枝；大 batch per-sample 早退；数据依赖形状导致 CUDA Graph/融合失效的任何细粒度方案。

**经验规律**：兑现率 = f(粒度规则性, 硬件算力/带宽比)。同一 LASNet 在 TX2（compute-bound）兑现率显著高于 V100（充裕并行度使省算无感、访存成瓶颈）。

### Q2. 动态计算拿到真实加速的必要条件清单

1. **粗且规则的粒度**：路由单元 ≥ 一个能喂饱 SM 的 dense tile（block/patch/layer/channel-group），保证剩余计算仍是规则 dense kernel。
2. **以延迟为目标函数**：训练/搜索用目标硬件的 latency lookup table 或 latency predictor（LASNet/LAUDNet 方法论），不用 FLOPs 代理。
3. **消除或摊薄决策同步**：路由决策留在 GPU 上（避免 device→host 同步），或决策频率远低于 kernel 频率。
4. **保持静态化优化可用**：路径数有限、形状固定（top-k 固定容量、bucketing、多 CUDA Graph 缓存），使融合与 graph capture 不失效。
5. **减少的是瓶颈资源**：在 memory-bound 场景须减内存流量（KV 剪枝、跳层），在 launch-bound 场景须减 kernel 数（跳层优于稀疏化单 kernel 内部）。
6. **匹配部署形态**：per-sample 动态性只在 batch=1/流式/continuous-batching 服务中兑现；大静态 batch 用模式级切换（slimmable/OFA）。
7. **gather/scatter 开销 < 节省**：稀疏率需越过硬件相关的盈亏平衡点（延迟-稀疏率曲线有平台区）。
8. **端到端 Amdahl 核算**：只加速部分算子时，用端到端延迟而非算子级 speedup 报告。

### Q3. Conformal / distribution-free 质量保证用于自适应计算：现状与空白

- **现状**：CAT (2021) → CALM (2022) → Fast yet Safe (2024, CRC/LTT, 覆盖分类/分割/LM/扩散生成) → Nested Prediction Sets (UAI 2024) → SAFE-KD (2026)。配方成熟：校准集 + LTT/CRC 定退出阈值，控制相对全模型的 gap 风险；LLM 上可拿 ~2–3× 加速且带 marginal 保证。
- **空白**：① **回归/图像恢复任务（PSNR/SSIM 保证）的自适应计算无先例**——最近的两块拼图（Fast yet Safe 的 CRC 早退框架、Adame et al. 2025 的 conformal-PSNR 控制）尚未被组合；② 空间/区域粒度的风险控制；③ conditional 保证与分布漂移鲁棒性；④ 延迟-质量联合保证。对做 SR 动态推理 + 统计保证的项目而言，这是清晰且可直接引用两侧文献支撑的 novelty 空间。

---

## 8. 论文清单（全部经检索验证存在）

**综述/框架**
1. Han et al., *Dynamic Neural Networks: A Survey*, TPAMI 2021. arXiv:2102.04906
2. Scardapane et al., *Conditional Computation in Neural Networks: Principles and Research Trends*, Intelligenza Artificiale 2024. arXiv:2403.07965

**硬件友好粒度**
3. Ren et al., *SBNet: Sparse Blocks Network for Fast Inference*, CVPR 2018. arXiv:1801.02108
4. Verelst & Tuytelaars, *Dynamic Convolutions: Exploiting Spatial Sparsity for Faster Inference*, CVPR 2020. arXiv:1912.03458
5. Han et al., *Latency-Aware Spatial-wise Dynamic Networks* (LASNet), NeurIPS 2022. arXiv:2210.06223
6. Han et al., *Latency-Aware Unified Dynamic Networks* (LAUDNet), TPAMI 2024. arXiv:2308.15949, DOI 10.1109/TPAMI.2024.3393530

**Early-exit 风险控制**
7. Schuster et al., *Confident Adaptive Language Modeling* (CALM), NeurIPS 2022
8. Jazbec, Timans et al., *Fast yet Safe: Early-Exiting with Risk Control*, NeurIPS 2024. arXiv:2405.20915
9. Jazbec et al., *Early-Exit NNs with Nested Prediction Sets*, UAI 2024 (PMLR v244)
10. *SAFE-KD: Risk-Controlled Early-Exit Distillation for Vision Backbones*, arXiv:2602.03043 [细节未核验]
11. Adame, Csillag, Goedert, *Image Super-Resolution with Guarantees via Conformal Generative Models*, NeurIPS 2025. arXiv:2502.09664
12. Romano et al., *Conformalized Quantile Regression*, NeurIPS 2019

**GPU 执行层/系统**
13. NVIDIA TensorRT Developer Guide（launch overhead 5–15 µs；enqueue-bound 判据）
14. NVIDIA CUDA Graphs blogs（graph 重放 ~2.5 µs + ~1 ns/node）
15. SpInfer, EuroSys 2025, DOI 10.1145/3689031.3717481
16. SparseRT, arXiv:2008.11849
17. PyTorch blog: *Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity*（A100 GEMM 1.39×）
18. NVIDIA blog: *Accelerating Inference with Sparsity Using Ampere and TensorRT*（端到端 1.3–1.8×）

**MoD / slimmable / anytime**
19. Raposo et al., *Mixture-of-Depths*, arXiv:2404.02258
20. *MoDification: Mixture of Depths Made Easy*, NAACL 2025（1.2× 延迟、1.8× 内存）
21. Yu et al., *Slimmable Neural Networks*, ICLR 2019. arXiv:1812.08928
22. Cai et al., *Once-for-All*, ICLR 2020. arXiv:1908.09791
23. Li et al., *Dynamic Slimmable Network*, CVPR 2021
24. Huang et al., *Multi-Scale Dense Networks* (MSDNet), ICLR 2018
25. Teerapittayanon et al., *BranchyNet*, arXiv:1709.01686

**未核验/低置信项**：SAFE-KD 具体保证形式；Pattern Recognition 2026 综述细节；arXiv:2605.30571（CUDA Graphs batch=1 decode 1.259×）；CAT 的 arXiv 编号；LAUDNet 各粒度逐项表格数字（论文存在与总体结论已核验）。
