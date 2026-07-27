# B2R-SR 外部评审：v2 方向判断与替代方案（含 Web 核验）

> 撰写日期：2026-07-27
> 输入材料：`B2RSR_Improvement_Plan_zh.md`、`B2RSR_v1_Gate_Diagnostic_Milestone_zh.md`、`B2RSR_Literature_Survey_zh.md`、`B2RSR_Paper_Story_Strategy_zh.md`、`dart_sr_plugin_arch.py`、`RCAN_arch.py`、`gate_report.json (20260726-084439)`。
> 所有涉及外部论文的判断均经过本次 Web Search 核验，核验来源见第 8 节；**未能核验的判断均显式标注 [未核验]**。
> 遵守硬约束：all-active 必须先等价 dense；只承认质量匹配的 wall-clock latency；不用 FLOPs 替代 latency；feature-delta proxy 不称 GT oracle；不称任意 backbone 通用插件；一次只改一个主要变量。

---

## 0. 一页结论

1. **“完整图 residual-group 动态深度”值得继续，但只值得以“一天级免训练扫描”的成本继续。** 它是当前唯一同时满足「all-active 精确等价 dense」「无 gather/scatter」「减少 kernel 数而非增加」的方向。但必须预注册一个高失败概率：冻结 RCAN 在**没有中间监督**的情况下直接截断深度，PSNR 很可能出现悬崖式下降而不是平滑退化（详见 §2.1）。
2. **“动态深度”作为概念的新颖性已经为零。** AdaDSR、Dynamic-CSD、Adaptive Depth Networks（NeurIPS 2024）、LLM 侧的 ShortGPT / 深层剪枝已覆盖“深度可变推理”的几乎所有常规叙述。当前文档提出的差异假设（post-hoc、冻结 backbone、exact dense endpoint、设备延迟选深度）是**工程属性组合，不是算法贡献**，单独不足以支撑 CCF-C 以上的方法论文。
3. **真正可能形成贡献的缺口是“带统计保证的质量风险控制 + 实测延迟的深度/路径选择”**：分类领域已有 risk-controlled early exit（NeurIPS 2024 "Fast yet Safe"），SR 领域已有 conformal 不确定性掩码（NeurIPS 2025），但**“对回归型 SR 的自适应计算做 distribution-free 质量风险控制”尚未见直接先例**（本次检索未命中；投稿前仍需系统检索确认）。这是把现有 v2 主线从“又一个动态深度”升级为可辩护贡献的最现实路径。
4. **在做任何动态方法之前，必须先建立“诚实 dense 基线” Gate**：RTX 3060 上 batch=1 的小图 RCAN 推理大概率是 kernel-launch-bound 而非 compute-bound；`torch.compile` / CUDA Graphs / fp16 可能让 dense 本身快 1.5–3×。如果动态方法赢不过优化后的 dense，加速主张会在评审中直接崩塌。
5. **若坚持空间路由，RCAN 是最差的宿主之一**（全图 CA），应换 EDSR（全局部算子，halo 可精确等价）或原生窗口化的 SwinIR。RCAN 上唯一语义正确的空间稀疏方案就是文档 §11 的“全图 CA + scatter 残差”算子，它应保持 stretch 地位。

---

## 1. 问题一：完整图 residual-group 动态深度是否值得继续？

**结论：GO，但仅以免训练扫描的成本 GO，且带一个预注册的“悬崖假设”。**

### 1.1 支持继续的理由

- **执行语义正确**：跳过整个 residual group 使用其外层 identity，是 RCAN 计算图中天然存在的路径，不改变任何已执行算子的语义。`d=G` 逐 bit 复用 dense 调用顺序即可满足 V2-G0，这是 v1 窗口方案永远做不到的。
- **GPU 行为正确**：深度跳过是**减少 kernel 数量**的稀疏模式，与 v1“增加 gather/scatter/小 kernel”相反。RCAN body 占绝对主导计算，跳 k 组的延迟节省接近线性、可预测，不需要任何定制算子。
- **成本极低**：整个 `d=0…10` 扫描 + per-image oracle 用现有 checkpoint 一天内可完成，符合“先免训练验证”的要求。

### 1.2 必须预注册的失败模式（悬崖假设）

Veit et al.（NeurIPS 2016，已核验）证明分类 ResNet 删层后性能平滑退化，这常被引用来支持“深度可跳”。但该证据**不能直接迁移**到 RCAN：

1. RCAN 的 `body` 末端有一个 conv（`modules_body.append(conv(...))`），它在训练中只见过“10 组之后”的特征分布；把第 d 组（d<10）的输出直接喂给它是分布外输入。
2. SR 是逐像素回归任务，对特征分布偏移的容忍度低于分类 logits。
3. AdaDSR（已核验）虽在 RCAN/EDSR 上做动态深度，但它是**带监督重新训练**的；它的成功不能证明免训练截断可行。
4. LLM 侧 ShortGPT、"Unreasonable Ineffectiveness of the Deeper Layers"（均已核验）表明 transformer 深层高度冗余、免训练删层可行——但两文都发现**必须选对层**（浅层/末层不可删），且后者仍需少量 finetune "healing"。RCAN 是否有同样的冗余结构完全未知，这正是扫描要回答的问题。

因此预注册两个观测分支：

```text
分支 A（平滑退化）：存在 d*<10 使 PSNR ≥ dense − 0.1 dB
  → 进入固定深度前沿 + oracle 分析（原计划 §6 决策规则）
分支 B（悬崖）：d=9 已经掉 >0.3 dB，或 PSNR-d 曲线在 d<10 全线崩
  → 免训练路线终止；唯一低成本救援是「冻结 backbone + 每深度一个
    轻量 exit adapter（1×conv）」，adapter 训练量小、d=G 走原路径
    保持 exact endpoint。这算“修改一个主要变量”，需另立 Gate。
```

### 1.3 一个应补进扫描的免费变量：深度跳过的位置

嵌套前缀（跳后面的组）只是 11 个选择中的一族。以同样成本可离线枚举“跳哪 k 个组”的少量对照（如跳前 k 组、跳中间 k 组、按 ShortGPT 式 Block-Influence 分数选组）。LLM 文献一致显示**跳哪里比跳几层更重要**。如果非前缀模式显著更好，主线叙述应从“nested depth”改成“influence-ranked group subset”，同样保持 exact endpoint。这不增加训练成本，只增加扫描组合数（可限制在 ~30 个组合）。

---

## 2. 问题二：与 AdaDSR / Dynamic-CSD 等的真实差异与新颖性风险

### 2.1 已核验的重叠地图

| 工作 | 已核验事实 | 与 v2 主线的重叠 |
|---|---|---|
| AdaDSR（ECCVW 2020） | 像素级 depth map + 稀疏卷积，RCAN/EDSR 实例，内容与资源自适应 | 覆盖“SR + 动态深度 + RCAN”本体；粒度更细（像素级） |
| Adaptive Depth Networks with Skippable Sub-Paths（NeurIPS 2024） | 单网络训练出可跳子路径，测试时即时调深度，CNN 与 Transformer 通用 | 覆盖“嵌套可跳深度 + 延迟控制”，但需专门训练 |
| Leveraging Stochastic Depth Training for Adaptive Inference（2025 preprint） | 用随机深度训练得到零开销单模型自适应深度 | 同上，训练路线 |
| ShortGPT（ACL Findings 2025）/ Unreasonable Ineffectiveness of the Deeper Layers（2024） | LLM 免训练层删除；Block Influence 度量；深层冗余 | 覆盖“post-hoc 免训练深度削减 + 冻结权重”概念本身（不同领域） |
| APE（ECCV 2022） | patch 级 incremental capacity 回归 + 早退 | 覆盖“预测继续算下去值不值” |
| ENAF（WACV 2025） | 多出口 + patch 融合 | 同类 |
| Dynamic-CSD / DCS-RISR（PR 2024 / Neural Networks 2025，已核验存在） | 对 EDSR/RCAN/CARN 类骨干做动态通道/分支选择 | 覆盖“同一骨干内难度感知动态子网” |

### 2.2 对当前差异假设的判断

文档 §七 的差异假设是：**post-hoc + 完全冻结 RCAN + d=G exact endpoint + 目标设备延迟选深度**。逐项拆解：

- “post-hoc / 免训练”：ShortGPT 一族已把该属性在 LLM 上做成大方向；迁移到 SR 是**应用转移**，评审会问“除了换任务还有什么”。
- “exact dense endpoint”：这是正确性要求，不是贡献（文档 §13.1 自己也已承认）。
- “设备延迟选深度”：Compiler-aware NAS、LAUDNet 等已把 latency-aware 选择做成常规配置。
- “冻结 backbone”：AdaDSR/Adaptive Depth Networks 都要动 backbone，这一点是真实差异——但它是**约束更强而结果通常更弱**的设定，只有在“免训练也几乎不掉点”的实验事实成立时才有卖点。

**结论：这四个属性的组合只能构成一篇“负担得起的 workshop/短文级观察”，除非叠加一个真正的方法学增量。** 下节给出我认为最可行的增量。

### 2.3 可辩护的缺口：risk-controlled adaptive compute for SR

已核验的边界：

- "Fast yet Safe: Early-Exiting with Risk Control"（NeurIPS 2024）：对 early-exit 网络给出 distribution-free 风险控制的退出规则——但对象是分类/生成任务的 EENN，需要多出口结构。
- SAFE-KD（2026 preprint）：conformal risk control + 多出口蒸馏——分类骨干。
- "Image Super-Resolution with Guarantees via Conformalized Generative Models"（NeurIPS 2025）与 "Conformal Prediction for Reliable Image Super-Resolution"（PMLR v266）：SR 的 conformal 工作只做**不确定性掩码/置信区间**，不做计算分配。

即：**「用 conformal risk control 给自适应计算的 SR 提供 per-image 质量损失保证（例如“以 ≥1−α 概率，PSNR 损失 ≤ ε dB”），并在保证约束下最小化实测延迟」在本次检索中没有直接先例。** [投稿前需再做一轮系统检索确认]

这个增量的好处：

1. 与冻结 backbone 完全兼容：校准只需要在 held-out 集上跑全部深度，纯离线；
2. 把“PSNR 不单调、oracle 只是上界”这些当前的尴尬事实转化为方法的一部分（风险控制天然处理非单调与不确定性）；
3. 直接回应动态 SR 文献的公开痛点：所有 patch/depth 路由方法都无法保证最坏情况质量；
4. 论文故事从“更快”变成“**可保证地更快**”，即使加速比不惊艳（1.2–1.5×）也有话可说。

---

## 3. 替代方案（5 个）

评分维度按要求：all-active 等价性 / 冻结 RCAN / 真实 GPU 加速潜力 / 实现复杂度 / 训练成本 / 发表价值。

### 方案 A：深度扫描 → 风险控制深度选择器（推荐主线，v2 的升级版）

```text
Step 1  免训练：d=0…10 扫描 + 非前缀组子集对照（§1.3）
Step 2  免训练：per-image oracle 上界 + conformal 校准曲线
        （在校准集上求：给定 ε, α，每张图允许的最小深度）
Step 3  仅当 oracle 有自适应空间：训练轻量 ordinal depth predictor，
        用 conformal/risk-control 包裹其决策（预测错时 fallback 更深）
```

| 维度 | 评估 |
|---|---|
| all-active 等价 | **精确**（d=G 复用 dense 调用序） |
| 冻结 RCAN | 是（predictor 独立小网络） |
| 真实 GPU 加速 | 高置信：跳组线性减 kernel 数与 FLOPs，无不规则内存访问；预计 d=7 时接近 0.7× 延迟 [待实测] |
| 实现复杂度 | 低（前缀执行入口 + 校准脚本） |
| 训练成本 | Step 1–2 零训练；Step 3 仅小 predictor（数 GPU 时） |
| 发表价值 | 无风险控制：CCF-C 边缘。加风险控制且结果成立：扎实 CCF-C，多 backbone + 多硬件后可试 CCF-B |

**Stop/Go**：
- G-A1（免训练）：若无 d<10 满足 `dense−0.1 dB` 且 p50 speedup CI 下界 >1.05×（对**优化后的 dense**，见方案 E），停止整条线；
- G-A2：若 oracle 对最佳固定 d 的配对增益 CI 下界 ≤0，只保留 fixed-d 工程结论，不训练 predictor；
- G-A3：predictor + 风险控制在 3 seeds 下必须满足预注册的 (ε, α) 覆盖率，否则删除 guarantee claim。

### 方案 B：图像级级联升级（cheap-first cascade with exact escalation）

先跑一个便宜网络（CARN-M 或 fixed-d 浅前缀），一个小判别器预测“该图是否需要完整 RCAN”；需要则**整图**重跑 dense RCAN。这是 ClassSR 思想收缩到图像粒度 + 精确 fallback。

| 维度 | 评估 |
|---|---|
| all-active 等价 | **精确**（“全部升级”= 纯 dense RCAN） |
| 冻结 RCAN | 是 |
| 真实 GPU 加速 | 取决于数据集难度分布：easy 图直接省 ~90% 时间，hard 图付出 cheap-pass 税（约 +10–20%）。在 Test2K/4K 大图混合负载上期望值好；Set5 这类全 hard 小集上可能变慢 |
| 实现复杂度 | 最低（无任何 backbone 改动） |
| 训练成本 | 只训练一个二分类/回归 gate（可用现成 NR-IQA 特征） |
| 发表价值 | 单独不够（级联/escalation 是老思想）；作为方案 A 的“深度=0 或 G 的极端特例”并入风险控制框架最合理 |

**Stop/Go**：在 DIV2K val + Test2K 上统计 oracle 级联（GT 决定是否升级）的平均延迟；若 oracle 都拿不到 ≥1.2× 质量匹配加速，放弃。全程免训练可先测。

### 方案 C：CA-gate 引导的免训练动态通道裁剪（新假设，需文献复核）

观察：RCAN 每个 RCAB 的 CA 层本身就输出 per-channel、per-image 的 0–1 门控值——**backbone 自带一个免费的、输入自适应的通道重要性信号**。假设：对给定图像，长期低门控的通道可被物理裁掉（对 conv 权重做通道切片），得到每图专属的窄网络；保留全部通道时精确等于 dense。

| 维度 | 评估 |
|---|---|
| all-active 等价 | **精确**（keep=全通道即 dense；CA 分母是通道维不受影响——注意：裁剪后 CA 的 1×1 conv 也需一致切片，需推导） |
| 冻结 RCAN | 是（只做权重切片视图，不改值） |
| 真实 GPU 加速 | 中等置信：通道裁剪产生**连续更小的 GEMM**，是 GPU 友好稀疏；但切片/重建开销与逐图不同通道集需 batch=1；3060 小图上可能被 launch 开销吃掉，大图上更有戏 |
| 实现复杂度 | 中（需一个“按通道集实例化窄 RCAN”的构建器 + CA 统计采集 pass） |
| 训练成本 | 第一阶段零训练（阈值裁剪）；若有效可加轻量校准 |
| 发表价值 | 若成立，“预训练注意力即免费剪枝信号、免训练、输入自适应”是有记忆点的故事；**但动态通道剪枝文献密集（FBS、DCS-RISR、FMP 等），且“用 SE/CA 统计指导剪枝”方向存在既有工作，本次未逐一核验 [未核验]**——启动前必须专门检索 |
| 关键风险 | 裁通道 c 会同时改变下一层卷积的输入统计，误差沿 20 个 RCAB × 10 组累积；免训练下可能同样悬崖 |

**Stop/Go（一天级免训练探针）**：统计 120000_G.pth 在 DIV2K val 上每层 CA gate 分布。若不存在“大量通道门控稳定接近 0”的现象（例如 <10% 通道的平均 gate <0.2），该方案直接死亡，无需实现裁剪器。这是所有方案中最便宜的 kill-check。

### 方案 D：exactness-preserving 晚期稀疏残差算子（保持 stretch 地位，补充两点）

即文档 §11 的方向（active blocks + 精确 halo + 全图 CA + scatter）。维持“非主线”的判断，补充：

1. **TLC（ECCV 2022，已核验）必须进 related work**：它证明“全局池化的 train/test 统计不一致”是已知问题且可测试时修正。TLC 的方向与你相反（把全局 op 变局部以匹配训练 patch 统计），但评审会立刻联想到它；你的算子的卖点必须写成“保持 dense 推理语义的稀疏化”，与 TLC 的“修改语义以提质”区分。
2. **SBNet（CVPR 2018，已核验）是 gather-conv-scatter 的直接先例**，必须作为实现基线引用；LAUDNet 提供 latency-model 方法论。
3. 只有方案 A 全 Gate 通过、且团队确认有写 fused kernel 的工程预算时才启动。CCF-B 潜力真实存在（“首个对含全局注意力 CNN 的 exact 空间稀疏推理算子”[未核验，启动前需检索]），但工程风险是全项目最高。

### 方案 E（前置义务，不是可选项）：诚实 dense 基线 Gate

已核验事实：TensorRT/CUDA Graphs 文档明确指出多小 kernel 模型在 batch=1 时 launch-bound，单次 kernel launch 约 5–15 µs 主机开销；RCAN 10 组 × 20 RCAB × (2 conv + CA 的 2 conv + pooling) ≈ **>800 个 kernel/前向**，在 Set5 尺寸输入上极可能 launch-bound。

```text
义务实验（半天，免训练）：
dense RCAN 分别测：
  eager fp32（当前基线 106 ms）
  + channels_last
  + fp16/bf16 autocast
  + torch.compile (mode="reduce-overhead" → CUDA Graphs)
输入分别用 Set5 与 720p / 2K
```

**规则：此后一切“加速”主张的分母都是上表中最快的 dense 配置。** 如果 compile+fp16 把 dense 拉到 ~40 ms，而深度跳过在优化栈上只省 20%，故事就必须重估。反之，如果动态深度在优化栈上仍保持比例节省（理论上应该，因为它是减 kernel 数的模式），主张反而更硬。同时这回答了 GPU 执行角度：**v1 失败的深层原因是它在 launch-bound 的机器状态上增加了 launch 数；任何 v2 方案必须先声明自己在 launch-bound 与 compute-bound 两种状态下的行为。**

---

## 4. 方案对比总表

| | A 深度+风险控制 | B 图像级级联 | C CA-gate 通道裁剪 | D 稀疏残差算子 | E 诚实基线 |
|---|---|---|---|---|---|
| all-active = dense | 精确 | 精确 | 精确（需推导 CA 切片） | 按容差等价 | 不适用 |
| 冻结 RCAN | ✔ | ✔ | ✔ | ✔ | ✔ |
| GPU 加速置信度 | 高 | 中（分布依赖） | 中低 | 未知（需 kernel） | — |
| 实现复杂度 | 低 | 最低 | 中 | 高 | 极低 |
| 训练成本 | 0 → 小 | 小 | 0 → 小 | 中 | 0 |
| 免训练可先验证 | ✔ | ✔（oracle 级联） | ✔（gate 分布统计） | ✘ | ✔ |
| 发表上限 | CCF-C 稳 / B 可冲 | 并入 A | 高方差（先查重） | CCF-B 潜力/高风险 | 论文诚实性基础 |
| 首个 kill-check 用时 | 1 天 | 1 天 | 0.5 天 | — | 0.5 天 |

**推荐执行顺序（每步只改一个变量）**：

```text
E（诚实 dense 基线）
→ A-Step1（d 扫描 + 非前缀对照）      ← 与 C 的 gate 分布统计可并行（互不影响）
→ A-Step2（oracle + conformal 校准）
→ 决策点：A 成立 → 训练 predictor + 风险控制（主线论文）
          A 崩（悬崖）→ 评估 exit-adapter 救援 or 转向 C/B
→ D 永远排在 A 全部 Gate 之后
```

---

## 5. 问题七：RCAN 是否适合空间动态路由？

**不适合。** `CALayer` 的 `AdaptiveAvgPool2d(1)` 使每个 RCAB（共 200 个）都消费全图统计；任何空间划分要么破坏语义（v1 已证明 -1.24 dB），要么必须保留全图特征驻留（方案 D 的代价）。若空间路由是研究目标本身，建议：

1. **EDSR**：全部算子局部（conv+ReLU，无 CA、无 BN）→ 固定 halo + valid-center crop 可实现**数学精确**的 tile 等价；空间稀疏方案在它上面才有干净的 Gate 0。AdaDSR 也以 EDSR 为实例，可直接对标。
2. **SwinIR / 窗口 Transformer**：窗口是原生计算单元，token/窗口裁剪不需要重新发明分块语义；但 shifted window 依赖需要处理，且高效 ViT token-pruning 文献极其拥挤。
3. **CARN-M**：级联拼接使跨 stage 依赖复杂，不推荐作为空间路由宿主。

务实建议：**深度/通道等“非空间”稀疏留给 RCAN（方案 A/C），空间稀疏若要做就换 EDSR，并把“同一方法在 local-only vs global-CA 骨干上的可行性对比”本身写成论文的分析章节**——这能把 v1 的失败转化为有引用价值的负结果证据。

---

## 6. 论文叙事建议（若方案 A 走通）

- 标题方向：*"Certified Compute Reduction for Pretrained Super-Resolution: Risk-Controlled Depth Selection without Retraining"*（示意）。
- 三个贡献（不许多于三个）：
  1. 冻结 RCAN 的免训练深度前沿刻画 + 非前缀组子集分析（含悬崖/平滑的实证结论，无论正负都可写）；
  2. distribution-free 质量风险控制的深度选择器：给定 (ε, α)，保证 PSNR 损失超 ε 的概率 ≤ α，同时最小化实测延迟；
  3. launch-bound vs compute-bound 两种设备状态下的诚实延迟协议与 break-even 分析（以优化后 dense 为分母）。
- 必比基线：AdaDSR、APE/ENAF（协议可统一时）、最佳 fixed-d、优化后 dense、静态浅 RCAN（直接训一个 d 组的小 RCAN——**这是最致命的对照**：如果重训小模型全面优于冻结截断，"frozen" 卖点只剩“省训练成本”，必须提前面对）。
- 失败退路：若风险控制覆盖率成立但加速平庸（<1.2×），转投“可靠性/可信推理”角度的 venue 而不是效率角度。

---

## 7. 全部 Stop/Go 清单（可直接预注册）

| ID | 实验 | GO 条件 | FAIL 动作 |
|---|---|---|---|
| SG-0 | 优化 dense 基线（E） | 得到最快 dense 配置的 p50/p95 | 无 FAIL；结果冻结为全项目分母 |
| SG-1 | d=0…10 前缀扫描（免训练） | ∃d<10：PSNR ≥ dense−0.1 dB 且 speedup CI 下界 >1.05×（对 SG-0 分母） | 触发悬崖分支：评估 exit-adapter 或停线 |
| SG-2 | 非前缀组子集对照（≤30 组合） | 记录性实验，无 GO 门槛 | 若显著优于前缀 → 改叙事为 influence-ranked subset |
| SG-3 | per-image oracle vs 最佳 fixed-d | 配对增益 95% CI 下界 >0 | 不训练 predictor，只报 fixed-d |
| SG-4 | conformal 校准可行性（离线） | 目标 (ε=0.1 dB, α=0.1) 下平均深度 < 最佳 fixed-d | 删除 guarantee 叙事，退回普通 predictor 或停 |
| SG-5 | CA gate 分布统计（C 的 kill-check） | 存在稳定低门控通道群 | 方案 C 死亡，不实现裁剪器 |
| SG-6 | oracle 级联延迟（B 的 kill-check，Test2K） | oracle ≥1.2× 质量匹配加速 | 方案 B 死亡 |
| SG-7 | 静态浅 RCAN 对照（重训 d* 组小模型） | 冻结截断 + predictor 在质量-延迟上不被全面支配 | “frozen”卖点降级为训练成本论述 |

---

## 8. 本次 Web 核验来源

| 主题 | 来源（已访问） |
|---|---|
| AdaDSR 像素级动态深度、RCAN/EDSR 实例 | arXiv:2004.03915；github.com/csmliu/AdaDSR |
| Adaptive Depth Networks with Skippable Sub-Paths | NeurIPS 2024 proceedings；arXiv:2312.16392；github.com/wchkang/depth |
| Stochastic depth 自适应推理 | Korol et al. 2025 preprint（TU Dresden） |
| ShortGPT 免训练层删除 | arXiv:2403.03853；ACL Findings 2025；github.com/icip-cas/ShortGPT |
| 深层冗余 + 删层后 healing | "The Unreasonable Ineffectiveness of the Deeper Layers", arXiv:2403.17887 |
| 残差网络似浅网络集成（删层容忍） | Veit et al., NeurIPS 2016, arXiv:1605.06431 |
| APE patch 早退 / incremental capacity | ECCV 2022, arXiv:2203.11589 |
| ARM anytime 子网 | ECCV 2022, arXiv:2203.10812 |
| ENAF 多出口 patch 融合 | WACV 2025（CVF open access） |
| ClassSR / PCSR | arXiv:2103.04039；arXiv:2407.21448 |
| CSD / DCS-RISR 动态通道 | arXiv:2105.11683；arXiv:2212.07613；Neural Networks 2025 (PMID 39798352) |
| TLC 全局池化 train/test 不一致的测试时修正 | ECCV 2022, arXiv:2112.04491；github.com/megvii-research/TLC |
| 全局 op 分块伪影相关（Dense Normalization 等） | ECCV 2024 论文 06171；arXiv:2407.04245 |
| risk-controlled early exit | "Fast yet Safe", NeurIPS 2024, arXiv:2405.20915；SAFE-KD, arXiv:2602.03043 |
| SR conformal 不确定性（非计算分配） | NeurIPS 2025, arXiv:2502.09664；PMLR v266 (Chakraborti et al.) |
| batch=1 launch-bound、CUDA Graphs、TensorRT | NVIDIA TensorRT 性能文档；PyTorch/TensorRT CUDA Graphs 教程；Kosaian et al., ICML 2021 |
| SBNet gather/scatter 先例 | CVPR 2018（文档 §12 已录，本次未重复抓取） |
| 静态剪枝对照（ISS-P/SRP/FMP） | ICCV 2023；ICLR 2022；ICML 2024（openreview/CVF） |
| RCAN 结构复核 | arXiv:1807.02758；github.com/yulunzhang/RCAN；rcan-it |

**明确标注的未核验项**：(a) “SE/CA 统计指导剪枝”方向的既有工作密度（方案 C 启动前必查）；(b) “risk-controlled adaptive compute for SR 无直接先例”仅基于本次检索，投稿前需系统复查；(c) 方案 D 的“首个 exact 空间稀疏 CNN-with-global-attention 算子”未做专门检索。

---

## 9. 最后一句话

> v2 动态深度值得跑，但它作为论文只有在两件事之一发生时才成立：要么免训练截断出乎意料地不掉点（一个有趣的实证发现），要么给自适应计算装上 distribution-free 的质量保证（一个方法贡献）。在按 SG-0 → SG-1 的顺序花两天拿到这两个答案之前，不要写任何一行 controller 代码。
