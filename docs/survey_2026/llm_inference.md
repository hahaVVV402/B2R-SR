# LLM 推理加速核心技术综述（2021–2026）——面向图像超分辨率 CNN 推理加速的机制性启发

> 调研目标：提炼 LLM 推理加速的**机制性思想**（而非具体实现），评估每种机制的
> ① 输出等价性保证（bit-exact / 分布等价 / 仅近似）
> ② 真实 wall-clock 加速（非 FLOPs）
> ③ 生效前提（batch、序列长、硬件）
> ④ 向图像超分（SR CNN）迁移的具体映射与障碍。
>
> 所有论文均经网络检索核验（arXiv / 会议官网 / OpenReview / ACL Anthology），未能核验的信息标注 [未核验]。

---

## 总览表

| 方向 | 代表方法 | 等价性 | 实测 wall-clock | 迁移到 SR 的核心映射 |
|---|---|---|---|---|
| 1. 推测解码 | Leviathan et al., SpS, Medusa, EAGLE | **分布等价（数学证明，lossless）** | 2–2.5×（原始）；EAGLE-3 至 ~6.5× | 「小网起草 + 大网验证」→ 小 SR 网出图 + 大网局部验证/修正；**关键障碍：SR 无逐 token 串行结构，验证即等于跑一遍大网** |
| 2. Early exit / 层跳过 | CALM, LayerSkip, ShortGPT, MoD | 近似（有质量约束但非等价） | CALM ≤3×，LayerSkip ≤2.16× | 按图像块难度做深度自适应（easy patch 早退）——SR 有天然的空间难度不均匀性，映射非常自然 |
| 3. KV cache / prefix caching | H2O, SnapKV, vLLM APC | 复用=bit-exact；压缩=近似 | SnapKV 3.6×（16K 上下文） | 视频 SR 帧间特征复用、滑窗推理重叠区复用；单图 SR 无时序可复用维度 |
| 4. MoE 稀疏激活 | Switch, Mixtral, DeepSeek-V2 | 近似（训练进架构，非后处理） | Mixtral 官方称约 6× vs Llama2-70B（active 13B/47B） | 空间 MoE：按 patch 路由到不同容量分支（ClassSR 已验证此思路） |
| 5. 量化 / 2:4 稀疏 | GPTQ, AWQ, SmoothQuant, 2:4 | 近似（PPL 损失小） | 量化 2–4×（memory-bound 时）；2:4 端到端仅 ~1.1–1.6× | SR 卷积是 compute-bound，INT8 TensorRT 收益直接；2:4 需 GEMM 足够大 |
| 6. 先便宜后验证/修正 | BiLD, Speculative Cascades, FrugalGPT | 取决于验收阈值（可调 lossless↔lossy） | BiLD ~2×，cascades 优于两者单独用 | 与 SR 的 residual 结构天然契合：便宜路径 + 误差检测 + 选择性精修 |

---

## 1. Speculative Decoding（推测解码：小模型起草 + 大模型验证）

### 论文清单（已核验）

| 论文 | 会议/年份 | 链接 |
|---|---|---|
| Fast Inference from Transformers via Speculative Decoding (Leviathan, Kalman, Matias) | ICML 2023 | https://proceedings.mlr.press/v202/leviathan23a.html |
| Accelerating LLM Decoding with Speculative Sampling (Chen et al., DeepMind) | arXiv 2023 (2302.01318) | https://arxiv.org/abs/2302.01318 |
| Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads | arXiv 2024 (2401.10774)，后收录 ICML 2024 [ICML 收录状态未核验] | https://arxiv.org/abs/2401.10774 |
| EAGLE-2: Faster Inference of LLMs with Dynamic Draft Trees | EMNLP 2024 | https://aclanthology.org/2024.emnlp-main.422.pdf |
| EAGLE-3: Scaling up Inference Acceleration via Training-Time Test | NeurIPS 2025 | https://arxiv.org/abs/2503.01840 |

### 机制说明：draft-then-verify 如何保证 lossless

核心是**修正拒绝采样（modified rejection sampling）**（Leviathan et al. Thm 1；Chen et al. 独立同期证明）：

1. 小 draft 模型 q 自回归生成 γ 个候选 token x₁…x_γ；
2. 大 target 模型 p **一次并行前向**算出这 γ+1 个位置的分布；
3. 逐位置接受判定：以概率 min(1, p(x)/q(x)) 接受；一旦拒绝，从**残差分布 norm(max(0, p−q))** 重采样该 token，并丢弃后续草稿。

数学结论：无论 q 多差，最终 token 序列的分布**恰好等于单独用 p 采样的分布**（分布等价；greedy 解码下即输出逐 token 完全一致）。q 的质量只影响**速度**（接受率→期望每步产出 token 数），不影响**正确性**。加速来源：decode 阶段是 memory-bandwidth-bound，跑 1 次 batch=γ 的并行验证 ≈ 跑 1 次单 token 前向的时延，故用「并行验证」换「串行生成」。

- Medusa：不用独立 draft 模型，在 target 上加多个解码头 + tree attention 起草。
- EAGLE 系列：在 target 的特征层做轻量自回归起草，EAGLE-2 动态草稿树 3.05–4.26×（EMNLP 2024），EAGLE-3 报告至 ~6.5×。

### 四要素

- **等价性**：✅ **分布等价（有数学证明）**——这是 LLM 加速中唯一「免费午餐」级别的保证。注意：是分布相同，不是浮点 bit-exact（数值上并行 kernel 与串行 kernel 可有 float 差异；greedy 下通常逐 token 一致）。
- **wall-clock**：原始论文 2–2.5×（Chinchilla-70B，Chen et al.）/ 2–3× 级（Leviathan）；vLLM 生产环境实测：draft-model 约 ≤1.5×，n-gram 匹配任务 ≤2.8×（https://vllm.ai/blog/2024-10-17-spec-decode）；EAGLE-3 至 ~6.5×。
- **前提**：① decode 必须 memory-bound（**小 batch/低 QPS**才有效，大 batch 下 GPU 已饱和，收益消失甚至为负——vLLM 博客明确指出）；② draft 与 target 分布要够接近（接受率高）；③ 有并行验证的空闲算力。
- **迁移到 SR**：
  - **映射**：小 SR 网（如 4 层 CNN）快速产出 HR 草稿 → 大 SR 网对草稿做「验证」，仅在草稿不可信区域重算。逐像素/逐 patch 的「接受-拒绝」对应逐 token 验证。
  - **核心障碍（必须直面）**：LLM 中验证便宜的根本原因是**自回归串行性**——大模型一次并行前向可验证 γ 个 token，而串行生成需 γ 次前向。**SR 是单次前向的密集预测，没有串行瓶颈**：让大网「验证」全图 = 让大网跑一遍全图，加速恒等于 0。因此逐字面照搬不成立。
  - **可行的变体映射**：(a) 把「验证」降级为**便宜的误差预测器**（轻量网判断哪些 patch 草稿失真大），仅对失真 patch 跑大网——这实际上退化为方向 6 的 cascade / 方向 2 的空间自适应，**等价性保证随之丧失**（除非配 bit-exact 的保守判据）；(b) 在**扩散式 SR / 自回归图像生成 SR** 中串行性回归，speculative 思想可直接用（已有 CVPR 2026 的 Multi-Scale Local Speculative Decoding for Image Generation，https://openaccess.thecvf.com/content/CVPR2026/papers/Peruzzo_Multi-Scale_Local_Speculative_Decoding_for_Image_Generation_CVPR_2026_paper.pdf [细节未核验]）；(c) 迭代式/递归式 SR（多次 refinement 的网络）中，用小网跳过若干迭代、大网一次验证多步——结构上最接近原始 speculative decoding。
  - **最有价值的启发**：不是 draft-verify 算法本身，而是其设计哲学——**「用便宜计算决定昂贵计算花在哪，且用可证明的验收准则控制输出偏差」**。SR 若想要 lossless，可以设计保守的可证上界判据（如 draft 与 target 在低成本代理量上的偏差界），这是 speculative decoding 给 SR 的真正机制性遗产。

---

## 2. Early Exit / Layer Skipping / Depth-Adaptive

### 论文清单（已核验）

| 论文 | 会议/年份 | 链接 |
|---|---|---|
| CALM: Confident Adaptive Language Modeling | NeurIPS 2022 | https://arxiv.org/abs/2207.07061 |
| LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding | ACL 2024（arXiv 2404.16710）[ACL 收录经 arXiv 页面标注，主体已核验] | https://arxiv.org/abs/2404.16710 |
| ShortGPT: Layers in LLMs are More Redundant Than You Expect | arXiv 2024 (2403.03853) | https://arxiv.org/abs/2403.03853 |
| Mixture-of-Depths: Dynamically Allocating Compute in Transformer LMs | arXiv 2024 (2404.02258)，未见正式会议收录 | https://arxiv.org/abs/2404.02258 [arXiv 编号经列表页核验] |

### 机制说明

- **CALM**：token 级早退。每层出口用置信度（softmax 差 / hidden state 饱和度 / 分类器）决定是否提前输出；关键贡献是**统计校准的质量约束**——用 Learn-then-Test 框架给出「与全模型输出的一致性 ≥ 1−δ（高概率）」的可控保证。至 3× 加速（NeurIPS 2022）。
- **LayerSkip**：训练时 layer dropout + 早退损失，使浅层出口可用；推理时**self-speculative decoding**——前 E 层当 draft、剩余层做验证（复用前 E 层的 KV cache），把早退的近似性用 speculative 验证「洗回」lossless。实测 1.82–2.16×。**这是「近似方法 + 验证机制 = 无损加速」的范本，对 SR 最有参考价值。**
- **ShortGPT**：静态层剪枝。用 Block Influence（层输入输出的余弦相似度）发现深层高度冗余，直接删 ~25% 层保留 ~92% 性能。
- **Mixture-of-Depths**：训练时学习 top-k 路由，每层只处理部分 token，其余 token 走残差直连；容量预算静态、路由动态。post-training 采样报告至 ~50% 提速。

### 四要素

- **等价性**：❌ 均为近似。CALM 提供**统计校准的质量界**（最接近保证的一类）；LayerSkip 的 self-speculative 模式对最终输出**分布等价**（验证端是全模型）；ShortGPT/MoD 纯近似。
- **wall-clock**：CALM ≤3×（编码器-解码器、生成任务）；LayerSkip 1.82–2.16×；ShortGPT ~25% 计算削减（论文未报统一端到端倍数）；MoD 采样最高 ~50% faster。
- **前提**：早退收益依赖「token 难度分布不均」；动态早退与 batching 冲突（batch 内 token 退出层数不同导致 padding/同步开销——生产部署的主要障碍）；静态剪枝（ShortGPT）无此问题。
- **迁移到 SR**：
  - **映射（非常自然）**：SR 的空间难度极不均匀——平坦区域（天空、墙面）bicubic 就够，纹理/边缘区才需要深网。逐 patch 深度自适应 = 逐 token 早退。事实上 SR 社区已有同构工作：ClassSR（CVPR 2021, https://arxiv.org/abs/2103.04039 [编号未核验，论文真实性经社区共识确认]）按 patch 难度分类路由到不同容量子网；SMSR、AdaDSR 等做空间稀疏卷积/层跳过。
  - **可直接借鉴的三个 LLM 侧增量**：① **CALM 的统计校准**——SR 的早退阈值目前多为启发式，可引入 Learn-then-Test 式校准，给出「与全网输出 PSNR 差 ≤ ε 的概率 ≥ 1−δ」的可控保证，这是现有 SR 早退工作缺失的；② **LayerSkip 的 self-speculative 思路**——浅层出口产出草稿，深层仅对置信度低的空间位置继续算，且训练时就为早退做正则（layer dropout），而非事后剪；③ **ShortGPT 的 Block Influence 度量**——用输入输出余弦相似度诊断 SR 网络（尤其 EDSR/RCAN 这类深残差堆叠）哪些 block 冗余，指导静态删层。
  - **障碍**：GPU 上空间自适应计算的碎片化——patch 级路由造成不规则内存访问，FLOPs 降幅≫wall-clock 降幅（与 LLM 早退的 batching 问题同源）。需 patch 聚合 batching（ClassSR 做法）或 block-sparse kernel 支持。

---

## 3. KV Cache 压缩与 Prefix Caching：计算复用思想

### 论文清单（已核验）

| 论文/系统 | 会议/年份 | 链接 |
|---|---|---|
| H2O: Heavy-Hitter Oracle for Efficient Generative Inference | NeurIPS 2023 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/6034a661584af6c28fd97a6f23e56c0a-Abstract-Conference.html |
| SnapKV: LLM Knows What You are Looking for Before Generation | NeurIPS 2024 | https://arxiv.org/abs/2404.14469 |
| vLLM / PagedAttention（含 Automatic Prefix Caching） | SOSP 2023（PagedAttention 论文）；APC 见官方文档 | https://docs.vllm.ai/ ；https://arxiv.org/abs/2309.06180 [SOSP 收录经社区共识，编号未逐字核验] |
| DeepSeek-V2 的 MLA（KV 低秩压缩进架构） | arXiv 2024 (2405.04434) | https://arxiv.org/abs/2405.04434 |

### 机制说明

两类本质不同的思想：

1. **精确复用（bit-exact）**：prefix caching——相同前缀的 KV 张量是确定函数，可跨请求缓存复用，prefill 计算直接跳过。数学上完全无损（因果注意力下前缀 KV 不受后文影响）。
2. **有损压缩（近似）**：H2O 基于注意力分数累积识别 heavy-hitter token，驱逐低贡献 KV；SnapKV 在 prefill 末尾用观察窗口的注意力模式选择保留哪些位置的 KV。SnapKV 报告 16K 上下文下 3.6× 生成提速、8.2× 内存效率。

### 四要素

- **等价性**：prefix caching ✅ bit-exact；KV 驱逐/压缩 ❌ 近似（长文任务上有可测退化，检索类任务尤其敏感）。
- **wall-clock**：prefix cache 收益 = 被复用前缀占比（多轮对话/共享 system prompt 场景可省大半 prefill）；SnapKV 3.6×（长上下文场景）。
- **前提**：KV 优化只在**长序列 + decode 阶段 memory-bound** 时重要；prefix caching 需请求间真有共享前缀。
- **迁移到 SR**：
  - **本质思想**：「识别计算图中**跨调用不变的中间结果**并缓存」+「识别中间状态中**低贡献部分**并丢弃」。
  - **映射 A（精确复用，对应 prefix caching）**：① **视频 SR / 连拍 SR**：相邻帧大面积静止，静止区域的卷积特征可跨帧复用（增量推理），这与「Skip-Convolution」「DeltaCNN」（CVPR 2021/2022 [编号未核验]）方向同构，且和 prefix caching 一样 bit-exact（对完全未变的感受野）；② **滑窗 tile 推理**：大图切 tile 时重叠区特征重复计算，可缓存重叠带的卷积特征——障碍是卷积感受野随深度扩张，深层特征的「不变区」快速收缩，能精确复用的层数有限（LLM 因果掩码保证前缀严格不变，CNN 没有这个结构优势，这是根本差异）。
  - **映射 B（有损压缩，对应 H2O/SnapKV）**：通道/特征图剪枝的运行时版本——按输入内容动态丢弃低响应通道。SR 侧对应动态通道剪枝类工作，近似性质与 KV 驱逐相同。
  - **主要障碍**：SR 单图推理无时序/请求间共享结构；卷积的空间感受野传播使「不变前缀」概念不如因果注意力干净。**该方向对视频 SR 价值最大，对单图 SR 价值有限。**

---

## 4. MoE 稀疏激活的推理效率

### 论文清单（已核验）

| 论文 | 会议/年份 | 链接 |
|---|---|---|
| Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | JMLR 2022（arXiv 2101.03961, 2021） | https://arxiv.org/abs/2101.03961 |
| Mixtral of Experts (Mixtral 8x7B) | arXiv 2024 (2401.04088) | https://arxiv.org/abs/2401.04088 |
| DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model | arXiv 2024 (2405.04434) | https://arxiv.org/abs/2405.04434 |

### 机制说明

条件计算：FFN 替换为 N 个 expert + 路由器，每 token 只激活 top-k（Switch 为 top-1，Mixtral 为 top-2）。**总参数 ≠ 激活参数**：Mixtral 47B 总参、每 token 仅激活 13B；官方称推理约 6× 快于 Llama2-70B（同等或更好质量）。收益本质是「用参数容量换单 token 计算量」——知识存储与计算解耦。

### 四要素

- **等价性**：❌ 不适用——MoE 是**训练进架构**的设计，不存在与某个稠密模型的等价性问题；它是「同质量下更快」而非「同模型无损加速」。
- **wall-clock**：Mixtral ~6×（vs 稠密 70B，官方博客 https://mistral.ai/news/mixtral-of-experts/）。注意成本：**全部参数须驻留显存**（47B 权重的显存占用换 13B 的计算量），大 batch 下 expert 负载不均引发通信/等待开销。
- **前提**：显存充足；batch 较小时路由碎片化不严重；分布式下需 expert parallelism。
- **迁移到 SR**：
  - **映射（自然且已被验证）**：**空间 MoE**——路由单位从 token 换成图像 patch/像素。不同 expert 专精不同图像内容（边缘、纹理、平坦区）。ClassSR（CVPR 2021）本质就是 3-expert 空间 MoE（难度分类器 = 路由器）；后续 ARM、Path-Restore 等同方向。
  - **LLM 侧可借鉴的增量**：① 端到端可微/负载均衡的路由训练（load balancing loss），SR 侧现多用两阶段训练；② top-k 软路由而非硬分类；③ 「参数扩容不增计算」的思路——SR 模型可用远大于当前的参数量（多 expert）而保持单 patch 计算量不变，提升质量-速度前沿。
  - **障碍**：patch 级路由的 GPU 碎片化（同方向 2）；SR 部署常在边缘设备，MoE 的「显存换计算」交换在显存受限设备上不利。

---

## 5. 量化（GPTQ / AWQ / SmoothQuant）与 2:4 结构化稀疏

### 论文清单（已核验）

| 论文 | 会议/年份 | 链接 |
|---|---|---|
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | ICLR 2023（arXiv 2210.17323）[会议收录经社区共识，未逐字核验] | https://arxiv.org/abs/2210.17323 [编号未逐字核验] |
| AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | MLSys 2024（Best Paper）[获奖信息未核验]（arXiv 2306.00978）[编号未逐字核验] | https://arxiv.org/abs/2306.00978 |
| SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs | ICML 2023（arXiv 2211.10438）[编号未逐字核验] | https://arxiv.org/abs/2211.10438 |
| Wanda: A Simple and Effective Pruning Approach for LLMs（含 2:4 实测） | ICLR 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/file/160adf2dc118a920e7858484b92a37d8-Paper-Conference.pdf |
| NVIDIA Ampere 2:4 结构化稀疏（官方技术文） | NVIDIA Dev Blog | https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/ |

### 机制说明与实测收益（重点：FLOPs vs wall-clock 的落差）

- **GPTQ**：逐层二阶（Hessian 感知）权重量化到 3–4 bit；weight-only，加速来自**权重读带宽降低**（decode memory-bound 时收益大）。
- **AWQ**：观察到少数「salient channel」主导误差，按激活幅度对权重做逐通道缩放再量化；HuggingFace 实测基准中 AWQ 生成吞吐快于 GPTQ（A100 + Mistral-7B，https://huggingface.co/docs/transformers/v4.35.2/main_classes/quantization），但**无统一的普适倍数**。W4 类方法社区常见实测 ~2–3× decode 提速 [具体倍数因 kernel/形状而异，未逐一核验]。
- **SmoothQuant**：把激活的离群值难度经数学等价变换迁移到权重上，实现 W8A8 全 INT8——**这一变换本身是数学等价的（变换前后 FP 网络严格同函数）**，损失只来自随后的 INT8 舍入。W8A8 可用 INT8 TensorCore 同时加速 compute-bound 的 prefill。
- **2:4 结构化稀疏**：每 4 个连续权重保留 2 个，Ampere+ Sparse TensorCore 硬件支持，理论 2×。**实测严重缩水**：线性层级 ~1.6×（ICLR 2024, https://openreview.net/pdf?id=PxoFut3dWW）；Wanda 论文附录端到端 LLaMA-7B 生成仅 37→42 tok/s ≈ **1.14×**（A100, bs=1）；NVIDIA 论坛有用户实测 decode 阶段**反而慢 12%**（RTX 4090, LLaMA2-7B, https://forums.developer.nvidia.com/t/why-am-i-2-4-sparse-slower-than-dense-in-the-decode-stage-of-llama2-7b/340931）——因为 decode 是 GEMV/小 GEMM，稀疏 TensorCore 吃不到收益。

### 四要素

- **等价性**：❌ 全部为近似（PPL 小幅上升）。但注意 SmoothQuant 的「等价变换 + 局部近似」分解值得学习。
- **wall-clock**：量化 weight-only 约 2–3×（memory-bound 场景）[未逐一核验]；W8A8 对 prefill/大 batch 也有效；2:4 端到端 **1.1–1.6×，可能为负**——**FLOPs 减半 ≠ 时间减半的最典型反例**。
- **前提**：量化收益取决于瓶颈类型（weight-only 只帮 memory-bound；W8A8 帮 compute-bound）；2:4 需要足够大的 GEMM 形状 + Ampere 以上硬件 + 优化 kernel。
- **迁移到 SR**：
  - **映射（最直接、工程上最成熟的方向）**：SR CNN 与 LLM decode 的瓶颈**相反**——SR 卷积在高分辨率特征图上是 **compute-bound**。因此：① weight-only 4bit（GPTQ/AWQ 思路）对 SR 收益有限（权重小、激活大）；② **W8A8 / INT8 全量化（SmoothQuant 思路）恰好对口**——TensorRT INT8 SR 模型是业界标配，2× 左右实测提速常见 [具体数字依模型/硬件，未逐一核验]。AWQ 的「激活感知的通道重要性」可指导 SR 的混合精度分配（对纹理敏感通道保高精度）。
  - **2:4 对 SR**：SR 卷积可 im2col 成大 GEMM，形状比 LLM decode 的 GEMV 有利，理论上 2:4 在 SR 上比在 LLM decode 上**更容易兑现**（接近 kernel 级 1.6× 一侧而非 1.1× 一侧）——这是一个反向迁移的机会点，但需实测验证。
  - **障碍**：SR 是回归任务，量化误差直接体现为可见 artifact（尤其低 bit 时的 banding）；对 PSNR/感知质量的容忍度比 LLM 的 PPL 更苛刻，通常需 QAT 或逐层敏感度分析。

---

## 6. 其他「先便宜后验证 / 修正」范式

### 论文清单（已核验）

| 论文 | 会议/年份 | 链接 |
|---|---|---|
| Speculative Decoding with Big Little Decoder (BiLD) | NeurIPS 2023 | https://papers.neurips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html |
| FrugalGPT: How to Use LLMs While Reducing Cost and Improving Performance | TMLR 2024（arXiv 2305.05176, 2023） | https://arxiv.org/abs/2305.05176 |
| Faster Cascades via Speculative Decoding (Speculative Cascades) | ICLR 2025 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/6f43166f50f26e8d8f3edc5545b0749f-Abstract-Conference.html |
| Cascade Speculative Drafting for Even Faster LLM Inference | NeurIPS 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/file/9cb5b083ba4f5ca6bd05dd307a2fb354-Paper-Conference.pdf |

### 机制说明：一条从 lossy 到 lossless 的连续谱

- **Cascade（FrugalGPT）**：便宜模型先答，置信度低才升级到贵模型。**无等价性保证**（deferral 规则是启发式/学习的），但允许「小模型答得比大模型好」的情况被保留——质量可以超过单独用大模型。
- **BiLD（NeurIPS 2023）**：小模型持续生成，大模型在小模型「不确定」时介入并回滚修正（fallback + rollback 两个策略）。介于 cascade 与 speculative 之间：不严格 lossless，但偏差可通过阈值控制，实测 ~2× 且质量退化极小 [具体数字未逐字核验，量级经摘要确认]。
- **Speculative Cascades（ICLR 2025）**：统一视角——speculative decoding 是「验收规则 = 严格匹配大模型分布」的特例，cascade 是「验收规则 = 学习的 deferral」的特例；把两者的验收规则参数化，可在 lossless（慢）↔ lossy（快甚至更准）之间连续调节，实测优于纯 speculative 和纯 cascade。
- **等价性总结**：该家族揭示了关键设计变量是**「验收/回退准则的严格程度」**：严格按大模型分布验收 → 分布等价；按质量代理验收 → 近似但更快，且可能更准。

### 迁移到 SR（本方向是可迁移性最高的范式）

- **映射**：SR 的「便宜路径」候选极多——bicubic、轻量 CNN、浅层出口；「验证/修正」= 误差图预测器或大网选择性精修。具体架构模板：
  1. 轻量网 F_s 全图产出草稿 HR₀ + 逐 patch 不确定度图 U；
  2. 验收规则 A(U, τ)：U ≤ τ 的 patch 直接采纳草稿；
  3. 仅对拒绝 patch 跑大网 F_l（带 context padding），拼合输出。
- **来自 LLM 的三条机制性教训**：
  - ① **验收准则显式化、可调节化**（Speculative Cascades 的核心）：SR 侧应把 τ 做成用户可调的「速度-保真度旋钮」，并且明确声明处于谱的哪个位置（τ→0 时逼近纯大网输出，可给出 PSNR 偏差上界）。
  - ② **校准的统计保证优于启发式阈值**（CALM 的 Learn-then-Test）：可对 U 做 conformal 校准，得到「采纳 patch 与大网输出差 ≤ ε 的概率 ≥ 1−δ」——这将是 SR 加速文献中罕见的**带保证的近似加速**。
  - ③ **回滚思想**（BiLD）：草稿 patch 被采纳后，若相邻精修 patch 与其边界出现不连续（可低成本检测），触发局部回滚重算，控制拼接 artifact。
- **障碍**：① patch 级验收导致边界 artifact（LLM token 无空间连续性约束，SR 有——这是 SR 独有的新问题）；② 不确定度预测器自身的成本与可靠性（预测误差的误差）；③ GPU 上不规则 patch 集合的高效 batch 执行；④ 与 speculative decoding 不同，**没有免费的 lossless**——大网不介入的区域永远没被大网算过，等价性只能是统计意义的，不可能是逐像素精确的。

---

## 综合结论：五条可迁移的机制性原则

1. **「验证」在 LLM 中便宜是因为串行性，SR 没有这个结构** → speculative decoding 的算法不可直接搬，但其**「便宜草稿 + 显式验收准则 + 可控偏差」**的框架可搬（迭代式/扩散式 SR 除外，那里可直接用）。
2. **空间难度不均匀是 SR 版的「token 难度不均匀」** → early-exit/MoD/MoE 的按需分配深度/容量思想迁移最自然，SR 已有 ClassSR/SMSR 等先例；LLM 侧的增量贡献是**校准的质量保证（CALM）**和**训练时为早退做正则（LayerSkip）**。
3. **FLOPs ≠ wall-clock 是贯穿性教训**：2:4 稀疏理论 2× 实测 1.1×；早退受 batching 碎片化拖累；MoE 受路由开销拖累。SR 侧任何空间自适应方案必须以 GPU kernel 友好性为一等设计约束（patch 聚合、block-sparse、静态形状）。
4. **瓶颈类型决定手段**：LLM decode 是 memory-bound（故 weight-only 量化、KV 压缩、speculative 有效）；SR 卷积是 compute-bound（故 W8A8 全量化、结构化稀疏、空间跳算更对口）。迁移前先做 roofline 定位。
5. **等价性是设计谱而非二元属性**：bit-exact（prefix 复用/帧间复用）→ 分布等价（speculative）→ 统计校准界（CALM）→ 纯近似（量化/剪枝）。SR 加速研究目前几乎全在「纯近似」一端；把**统计校准保证**引入 SR 空间自适应加速（如「99% 的 patch 与全网输出差 < 0.1dB」）是本调研识别出的最具新颖性的迁移机会。

---

## 附：核验状态说明

- 已核验（检索到 arXiv 摘要页 / 会议 proceedings / 官方文档）：Leviathan et al. (ICML 2023)、Chen et al. (2302.01318)、Medusa (2401.10774)、EAGLE-2 (EMNLP 2024)、EAGLE-3 (2503.01840, NeurIPS 2025)、CALM (NeurIPS 2022, 2207.07061)、LayerSkip (2404.16710)、ShortGPT (2403.03853)、MoD、H2O (NeurIPS 2023)、SnapKV (NeurIPS 2024, 2404.14469)、Switch (2101.03961)、Mixtral (2401.04088)、DeepSeek-V2 (2405.04434)、BiLD (NeurIPS 2023)、FrugalGPT (2305.05176, TMLR)、Speculative Cascades (ICLR 2025)、Cascade Speculative Drafting (NeurIPS 2024)、Wanda 2:4 实测 (ICLR 2024 proceedings PDF)、vLLM spec-decode 博客、NVIDIA 论坛 2:4 decode 变慢实测。
- 标注 [未核验] 项：GPTQ/AWQ/SmoothQuant 的 arXiv 编号与部分会议收录细节、ClassSR/DeltaCNN 的 arXiv 编号、Medusa 的 ICML 收录、AWQ Best Paper、量化具体倍数——这些论文本身真实存在（社区广泛引用），但本次检索未逐字确认上述细节字段。
