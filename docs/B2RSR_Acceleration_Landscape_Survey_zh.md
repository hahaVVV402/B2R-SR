# 轻量化图像超分加速：2023–2026 全景调研与 B2R-SR 转向建议

> 撰写日期：2026-07-27
> 背景：三轮免训练诊断已证伪"冻结 RCAN + 免训练加速"的全部路线（空间窗口 / 前缀深度 /
> 非前缀组子集 / CA 通道裁剪 / 深度截断级联 oracle）。本报告调研领域内**真正拿到
> wall-clock 加速**的方法族，评估每条路线与现有资产（RCAN checkpoint、Gate 体系、
> 3060 延迟协议）的兼容性，给出转向建议。
> 全部外部结论经本次 Web Search 核验，来源见各节；未核验处显式标注。

---

## 0. 一个先于所有方案的调研结论

把这次和之前所有检索放在一起看，可以提炼出一条对你最重要的行业规律：

> **2023 年以后真正在 GPU/端侧拿到实测加速的工作，几乎全部走"静态重设计 + 部署优化"
> 路线（重参数化、LUT、量化、蒸馏、新算子），而"对已有大模型做动态/条件计算"路线
> 的论文（含 CAMixerSR 这类 CVPR 正会工作）主要赢在 FLOPs 和大图场景，
> 在中小分辨率单图上的 wall-clock 优势普遍很小甚至为负。**

证据链（已核验）：
- CAMixerSR 官方补充材料自己承认 attention 分支占 67.8% 延迟、routing 是为大图设计的；其收益主要在 2K–8K 输入（CVPR 2024 supplemental）。
- 动态网络综述（Conditional computation, Scardapane 2024；Dynamic NN survey 2025）明确指出动态架构与 batch 计算、硬件调度不兼容是普遍问题。
- NTIRE 2024/2025 Efficient SR 挑战赛 runtime 赛道的获胜方案（SPAN、及各队报告）全部是静态网络 + 重参数化 + 部署技巧，没有一个条件计算方法进入 runtime 前列。
- 你自己的 v1（0.844×）和 v2 诊断（oracle 上界仅 1.02×）是同一规律的本地复现。

**含义：如果目标是"真实加速 + 可发表"，方向本身应该从"动态路由"迁移到
"静态可部署效率"或"动态思想的少数真正有效场景（大图/视频/生成模型）"。**

---

## 1. 六大方法族全景（按真实加速可信度排序）

### 族 A：重参数化静态轻量网络（真实加速最可信）

| 工作 | 出处 | 核心机制 | 实测证据 |
|---|---|---|---|
| ECBSR | ACM MM 2021 | 训练期多分支 → 推理期合并为单个 3×3 conv | 移动端 NPU 实时，明确批判 FLOPs≠速度 |
| RepSR | ACM MM 2022 | VGG-style + BN 的重参数化配方 | 同上路线 |
| SPAN | CVPR-W 2024，NTIRE ESR **winner** | 无参数注意力 + 重参数化 | runtime 赛道冠军，公开 code |
| PLKSR | 2024 | 部分大核 CNN，直接反驳"transformer 更高效" | 直接效率指标全面对比 |

**特点**：all-active 无意义（本来就是静态图）；不保留 RCAN；加速真实且极大
（NTIRE runtime 级别是毫秒级）；训练成本中等（从头训，但模型小、收敛快）。
**论文空间**：单纯复现无新颖性；新颖性要来自新的重参数化结构或训练配方。竞争极度激烈。

### 族 B：LUT 化推理（端侧最快，GPU 上另类）

| 工作 | 出处 | 机制 |
|---|---|---|
| SR-LUT → MuLUT | ECCV 2022 / T-PAMI 2024 | 受限感受野网络 → 查表；多表协作扩大感受野 |
| AutoLUT | CVPR 2025 | 自动采样点 + 自适应残差学习 |
| IM-LUT | ICCV 2025 | 插值混合 LUT，扩展到任意尺度 |
| 百 KB 级 LUT | 2023 | 存储压到 hundred-KB 量级 |

**特点**：彻底绕开卷积计算，"加速"以数量级计（但主战场是 CPU/边缘设备，GPU 上
不一定优于 tensor-core 卷积）；训练成本低；**2025 年仍在 CVPR/ICCV 正会产出**，
说明该分支远未饱和。质量上限低于 CNN（受限感受野），差距是研究空间。

### 族 C：量化 + 部署栈（与任何架构正交）

| 工作 | 出处 | 机制 |
|---|---|---|
| 2DQuant | NeurIPS 2024 | SR 专用低 bit PTQ |
| Outlier-Aware PTQ | ICCV 2025 | 处理 SR 激活离群值 |
| QuantSR | NeurIPS 2023 Spotlight | 低 bit QAT |
| 2:4 半结构稀疏 | NVIDIA 官方 / PyTorch 教程 | Ampere Sparse Tensor Cores，实测 ~1.1–1.3× GEMM 提速 |

**关键事实（对你直接相关）**：你的 3060 是 Ampere，**支持 2:4 sparse tensor cores 和
INT8**——但你实测 fp16 autocast 反而更慢，说明 PyTorch eager 是瓶颈；这类方法必须
走 TensorRT/编译栈才能兑现。量化是工程强、论文新颖性弱的路线，适合作为任何主线的
"部署章节"而非独立贡献。

### 族 D：动态/条件计算（你刚离开的领域——幸存场景收窄为两个）

调研确认：单图中小分辨率上该族已基本出清。**幸存场景**：
1. **大图（2K–8K）patch 级路由**：CAMixerSR、PCSR、ENAF 都把主战场设在 Test2K/4K/8K——patch 数量大才摊薄调度开销。你的 B 方案若要复活，唯一出路是 **Test4K 级大图 + patch 级（而非整图级）级联**。
2. **生成式 SR 的步数/模块动态**：diffusion SR 的计算量比 CNN 大两个数量级，动态化收益空间也大两个数量级（见族 F）。

### 族 E：新算子骨干（Mamba / 大核 / 混合）

| 工作 | 出处 | 状态 |
|---|---|---|
| MambaIR / MambaIRv2 | ECCV 2024 / CVPR 2025 | 线性复杂度全局感受野；正会热点 |
| TSP-Mamba | CVPR 2025 | 扫描路径优化 |
| MambaLiteSR | 2025 | 低秩 Mamba + 蒸馏做边缘部署 |

**特点**：属于"设计新 backbone"赛道，与"加速已有 backbone"的项目初衷不同；
竞争白热化（每个 CVPR 十几篇），单人业余时间入场风险高。

### 族 F：一步扩散 SR 的效率化（当前最热的效率新边疆）

| 工作 | 出处 | 机制 |
|---|---|---|
| OSEDiff | NeurIPS 2024 | 蒸馏到一步扩散 Real-ISR |
| TSD-SR | CVPR 2025 | target score distillation 一步化 |
| QArtSR | 2025 | 一步扩散 SR 的低 bit 量化 |
| InfVSR / Stream DiffVSR | 2025 | 流式/自回归视频扩散 SR |

**特点**：模型大（SD 级），效率问题真实且远未解决——量化、缓存复用、区域自适应
步数都是开放题。**这是"动态/预算思想"最可能复活的地方**：对扩散 SR 做
region-adaptive 计算（平滑区少算、纹理区多算）尚未饱和 [需专项检索确认]。
门槛：需要 SD 级显存（3060 12GB 勉强推理、训练需租更大卡）。

---

## 2. 结合你的现实约束的三条候选主线

约束回顾：单人、实习中、只有晚上/周末、3060 预算级、已有资产 =
RCAN checkpoint + 完整 Gate/延迟协议 + 三轮高质量负结果 + 全套 benchmark 数据。

### 主线 α：大图 patch 级混合推理（B 方案的正确形态）★推荐首查

**命题**：在 Test2K/4K 大图上，patch 级"LUT/超轻网络先行 + 难 patch 升级到
RCAN/SPAN"的级联，以你的 Gate 协议验证质量匹配的 wall-clock 加速。

- 为什么现在可信了：大图 patch 数以百计，(1) 难度分布真正长尾（大量天空/平坦区），(2) 批量 patch 摊薄调度开销，(3) CAMixerSR/PCSR/ENAF 已证明该场景收益真实——但它们的便宜路径都还是神经网络，**"LUT 作为便宜路径 + 神经网络作为重路径"的混合体在本次检索中未发现直接先例**[投稿前需系统确认]。
- 免训练 kill-check（下次开机，~1 小时）：Test2K/4K 图 → 切 patch → 逐 patch 算 bicubic/超轻路径与 dense RCAN 的 PSNR 差 → oracle 级联延迟上界。**这次 oracle 若还不过 1.3×，α 也关闭**。
- 与现有资产完美衔接：延续 wall-clock 纪律、oracle 方法论、RCAN checkpoint 继续当重路径。
- 发表定位：CCF-C 稳（若 oracle 强 + 实现扎实），故事是"预算下的质量保证级联"，可嫁接之前调研的 conformal risk control 增量冲 B。

### 主线 β：LUT 系创新（换赛道，天花板清晰但赛道不挤）

**命题**：在 MuLUT/AutoLUT/IM-LUT 谱系上做增量（如内容自适应表选择、
LUT+微型残差网络混合、跨通道 LUT）。
- 优点：训练成本极低（小时级）、实验周期短（适合业余时间）、2025 仍在正会产出、3060 完全够用。
- 缺点：与"加速已有 backbone"的项目叙事断裂，等于开新项目；需要重建文献功底。

### 主线 γ：一步扩散 SR 的区域自适应计算（高风险高回报）

**命题**：把你的"预算/收益路由"思想搬到 OSEDiff 类一步扩散 SR 上：
区域自适应地决定 VAE/UNet 的计算深度或缓存复用。
- 优点：效率问题真实且未饱和，"动态思想"在这里还有处女地；冲 CCF-B/A 的可能性最高。
- 缺点：显存与训练成本超出 3060 预算级；单人业余时间风险大。适合作为**下一个阶段**（如实习结束后）的储备方向。

**淘汰说明**：族 A（重参数化）与族 E（Mamba）为红海全职赛道，不建议单人业余入场；
族 C（量化）建议作为任一主线的部署加分章节而非主线。

---

## 3. 建议的决策顺序

```text
第 1 步（下次开机，免训练，~1 小时）：
  主线 α 的 oracle kill-check——Test2K/4K 大图 patch 级级联上界
  ├─ oracle ≥1.3× → α 立项，进入便宜路径选型（bicubic / SR-LUT / SPAN-tiny）
  └─ oracle <1.3× → α 关闭，在 β 与 γ 之间做战略选择（β=稳，γ=赌）

第 2 步（若 α 立项）：
  便宜路径落地 + 难度判别器训练（首次真训练，几 GPU 时级）
  + 嫁接 conformal 质量保证（前次调研的差异化增量）

任何时候：负结果资产不浪费——
  "冻结 RCAN 不可免训练压缩"的三轮诊断 + Gate 方法论
  可作为 α 论文的 motivation/分析章节，或独立整理为 workshop 短文。
```

---

## 4. 可复用的深度调研提示词库

以下按主题给出可直接投给检索工具（Web Search / Perplexity / 学术检索）的提示词，
中英混排，英文为主（文献主体是英文）。使用建议：每次取一个主题的 2–4 条并行查，
避免一次塞太多导致结果稀释。

### 4.1 主线 α 立项前的查新（最优先）

```text
1. "hybrid look-up table neural network cascade super-resolution large image
   patch difficulty routing wall-clock"
2. "LUT as fast path CNN as refinement path image super-resolution 2024 2025"
3. "patch-level model selection 4K 8K super-resolution latency measured GPU
   not FLOPs"
4. "conformal prediction / risk control adaptive computation image restoration
   quality guarantee latency budget"  ← 确认差异化增量仍无人做
5. site:openaccess.thecvf.com CAMixerSR PCSR ENAF 后继工作 2025 2026 large
   image efficient SR
```

### 4.2 便宜路径选型

```text
6. "SR-LUT MuLUT AutoLUT inference latency GPU vs CPU benchmark comparison"
7. "SPAN NTIRE 2024 runtime winner architecture details reproduction"
8. "bicubic vs lightweight network quality gap smooth region sky patch
   super-resolution statistics"
```

### 4.3 若转 β（LUT 赛道）

```text
9. "look-up table super-resolution open problems receptive field color
   channel interaction survey 2025"
10. "IM-LUT AutoLUT limitations failure cases future work"
11. "LUT image restoration denoising deblocking video in-loop filter 2025"
    ← 看该技术向邻域扩张的空间
```

### 4.4 若转 γ（扩散 SR 效率）

```text
12. "one-step diffusion super-resolution region adaptive computation spatial
    mask 2025 2026"
13. "OSEDiff TSD-SR follow-up efficiency VAE encoder bottleneck latency
    breakdown"
14. "diffusion model spatially adaptive inference token merging cache reuse
    image restoration"
15. "quantization one-step diffusion SR QArtSR beyond INT8 INT4"
```

### 4.5 通用查新模板（投稿前必跑）

```text
16. "[你的方法一句话英文描述]" —— 原样整句搜，看是否已有同名工作
17. site:arxiv.org [核心关键词组合] 2025..2026 —— 限最近两年
18. [最近邻方法名] "cited by" / semantic scholar citations 2025 2026
    —— 顺引文找最新后继，防止投稿时撞车
```

---

## 5. 本次调研核验来源（关键项）

| 主题 | 来源 |
|---|---|
| 重参数化实测加速 | ECBSR (ACM MM 2021, polyu 官方 PDF)；RepSR (ACM MM 2022)；SPAN (github/hongyuanyu/SPAN, NTIRE 2024 winner, CVPR-W Oral) |
| NTIRE 效率挑战赛 | NTIRE 2024/2025 ESR Challenge Reports（CVF open access） |
| LUT 谱系 | SR-LUT→MuLUT (ECCV 2022/T-PAMI 2024)；AutoLUT (CVPR 2025)；IM-LUT (ICCV 2025) |
| 量化 | 2DQuant (NeurIPS 2024)；Outlier-Aware PTQ (ICCV 2025)；NVIDIA 2:4 sparsity 官方文档；PyTorch semi-structured sparse 教程 |
| CAMixerSR 延迟自白 | CVPR 2024 supplemental（attention branch 67.8% runtime） |
| 动态网络的硬件批判 | Scardapane 2024 conditional computation；Dynamic NN surveys (2024/2025) |
| Mamba 系 | MambaIR/v2 (ECCV 2024/CVPR 2025)；TSP-Mamba (CVPR 2025) |
| 一步扩散 SR | OSEDiff (NeurIPS 2024)；TSD-SR (CVPR 2025)；QArtSR、InfVSR (2025 preprints) |
| PLKSR | arXiv 2404.11848 / IEEE Access |

**未核验/需专项确认**：(a) "LUT 便宜路径 + NN 重路径混合级联"无直接先例——仅基于
本次检索，α 立项前必须用 §4.1 的 1–3 号提示词专项查新；(b) 扩散 SR 的
region-adaptive 计算饱和度——转 γ 前用 §4.4 专项查。

---

## 6. 一句话总结

> 行业证据与你的本地实验指向同一结论：**在中小图上给现成 CNN 做动态计算拿不到真实
> 加速；真实加速属于静态重设计（重参数化/LUT/量化）和"计算量足够大"的场景（大图、
> 扩散）。** 你的资产（wall-clock 纪律 + oracle 方法论 + RCAN 重路径）最平滑的
> 落点是主线 α——大图 patch 级 LUT/NN 混合级联；下次开机一小时的 oracle
> kill-check 就能定它的生死。
