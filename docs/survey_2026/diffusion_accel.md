# 扩散模型推理加速（2021–2026）核心技术调研
## —— 面向「CNN 图像超分辨率推理加速」的可迁移启发

> 调研日期：2026 年。所有论文均经 arXiv / GitHub / CVF / NeurIPS proceedings 检索核实存在；
> 数字来自论文原文、官方 repo 或官方博客；无法直接核实的数字标注 **[未核验]**。

---

## 0. 总览：六类机制一页表

| 类别 | 代表工作 | 典型 wall-clock 加速 | 是否免训练 | 质量控制机制 | 对 CNN-SR 的可迁移性 |
|---|---|---|---|---|---|
| 步数蒸馏 | PD, CM, LCM, InstaFlow, DMD | 10–30×（步数 50→1~4） | 否（需蒸馏训练） | 蒸馏损失对齐教师分布 | ★★★（映射为"迭代式SR→单次前向"或深模型→浅模型蒸馏） |
| 缓存复用 | DeepCache, Faster Diffusion, Block Caching, TGATE | 1.5–4× | 是 | 缓存间隔/分支选择、周期性全量刷新 | ★★★★（时间步相似性 → 帧间/patch间/尺度间/层间相似性） |
| 空间自适应 | ToMe for SD, RAS, QDM, PatchScaler | 1.5–2×（同预算下提质） | ToMe/RAS 是 | 合并率/区域刷新率可调，误差可控 | ★★★★★（平坦区少算、细节区多算，SR 最天然的适配） |
| 一步扩散 SR | SinSR, OSEDiff, AddSR, TSD-SR | 10–100×（vs 多步扩散SR） | 否 | VSD/目标分数蒸馏/对抗蒸馏 | ★★★（证明"单前向即可达感知质量"，为 CNN-SR 的对标基线） |
| 量化与编译 | SVDQuant, TensorRT, torch.compile | 1.5–3×（编译）；~3× kernel 级（W4A4） | 量化近似免训练(PTQ) | 低秩分支吸收离群值；LPIPS 监控 | ★★★★（与架构无关，CNN-SR 可直接用） |
| 自适应步数 | AdaDiff(step-wise), AdaDiff(step selection) | 25–40% 计算节省 | 否（需训练策略/退出头） | 不确定性阈值 / 奖励驱动策略 | ★★★（映射为早退网络 / 难度感知路由，SR 已有 ClassSR 等先例） |

---

## 1. 步数蒸馏（Step Distillation）

### 1.1 论文清单（均已核实）

| 论文 | 出处 | 核心机制 |
|---|---|---|
| Progressive Distillation for Fast Sampling of Diffusion Models (Salimans & Ho) | ICLR 2022, arXiv:2202.00512 | 学生每次学习教师两步的等效映射，反复减半：8192→…→4 步 |
| Consistency Models (Song et al.) | ICML 2023, arXiv:2303.01469 | 学习 ODE 轨迹上任意点→原点的一致性映射，支持 1–2 步采样 |
| Latent Consistency Models (LCM, Luo et al.) | arXiv:2310.04378 | 在潜空间对 SD 做一致性蒸馏 + skipping-step，2–4 步生成 768×768 |
| InstaFlow (Liu et al.) | ICLR 2024, arXiv:2309.06380 | Rectified Flow reflow 拉直 ODE 轨迹后蒸馏成一步 |
| DMD: Distribution Matching Distillation (Yin et al.) | CVPR 2024, arXiv:2311.18828 | 分布匹配（近似 KL 的双 score 梯度）+ 回归损失，一步生成 |
| DMD2 (Yin et al.) | NeurIPS 2024, arXiv:2405.14867 | 去掉回归损失、加 GAN 项与多步蒸馏，质量超越 DMD |

### 1.2 真实加速数字

- **Progressive Distillation**：CIFAR-10/ImageNet 64 上把采样从 8192 步降到 **4 步** 而 FID 接近教师；wall-clock 加速 ≈ 步数比（网络单步成本不变）。
- **LCM**：SD 类模型 50 步（DPM-Solver 20+ 步）→ **2–4 步**，即约 **5–12× 端到端加速**；768×768 少步生成 SOTA（论文自述）。
- **InstaFlow**：SD1.5 级质量的**一步**生成，官方报告单图 **~0.09 s（512×512，A100）**，vs SD 25 步约 0.88 s，约 **10×**。[0.09s 数字来自官方 repo/论文，A100 条件见原文；细节未核验具体 GPU 型号变体]
- **DMD**：官方页面自述"与 SD v1.5 质量相当，**30× faster**"；一步生成，FP16 下报告约 20 FPS（512×512）[未核验精确硬件]。

### 1.3 质量控制

- PD：逐级减半，每级重训收敛，退化被限制在每次减半的近似误差内。
- CM/LCM：一致性损失 + 允许多步"一致性采样"回补质量（步数-质量可调旋钮）。
- DMD/DMD2：分布级匹配（而非逐轨迹回归），避免一步映射的模糊平均；DMD2 用 GAN 项抑制退化。
- 共同代价：需要相当的蒸馏训练算力；蒸馏后模型对分布外输入可能退化更明显。

### 1.4 迁移到 CNN 超分

CNN-SR 没有时间步，但"步数蒸馏"抽象后是**「把 N 次迭代计算的效果蒸馏进 1 次前向」**：

1. **迭代/递归 SR 的展开蒸馏**：若 SR 系统含迭代细化（如 back-projection、递归网络、多阶段 restoration），可用 PD 式的"两步并一步"逐级折半。
2. **深→浅蒸馏 = 步数蒸馏的深度版**：把大 SR 模型（或 diffusion-SR 教师如 OSEDiff）蒸馏进轻量 CNN；DMD 的启示是**用分布匹配/GAN 损失代替纯像素回归**，避免 L1/L2 蒸馏导致的过平滑——这正是 CNN-SR 感知质量瓶颈所在。
3. **DMD 的"fake score"思想** ≈ 在 SR 中训练一个判别器/评分网络在特征分布层面对齐师生输出，比 feature-L2 蒸馏更能保住高频细节。

---

## 2. 【重点】缓存复用（Feature Caching / Reuse）

核心观察：**相邻时间步的深层特征高度相似**，可以隔步复用、只更新便宜的浅层。

### 2.1 论文清单与机制（均已核实）

| 论文 | 出处 | 机制要点 |
|---|---|---|
| DeepCache (Ma, Fang, Wang) | CVPR 2024, arXiv:2312.00858 | 利用 U-Net skip 连接：缓存高层（深层）特征 N 步复用，只重算最浅分支；cache_interval / cache_branch_id 两个旋钮 |
| Faster Diffusion (Li et al.) | NeurIPS 2024, arXiv:2312.09608 | 实证发现 **encoder 特征随 t 变化平缓、decoder 变化剧烈** → 隔步复用 encoder 输出（encoder propagation），且使多个相邻步的 decoder 可并行 |
| Block Caching: "Cache Me if You Can" (Wimbauer et al., Meta) | CVPR 2024, arXiv:2312.03209 | 逐 block 分析输出随时间的变化，变化小的 block 直接复用输出；配自动缓存调度 + 轻量 scale/shift 调整层减轻伪影 |
| TGATE (Zhang, Liu et al.) | arXiv:2404.02747 | 观察到 **cross-attention 在若干步后收敛到不动点** → 收敛后缓存其输出并跳过计算（"语义规划阶段"后 cross-attn 冗余） |
| （延伸）FasterCache (Lv et al.) | arXiv:2410.19355 | 视频 DiT 上缓存注意力特征，CFG 两分支输出复用；diffusers 已集成 |
| （延伸）PAB: Pyramid Attention Broadcast | arXiv:2408.12588 | 按注意力类型以不同间隔广播复用（视频） |

### 2.2 真实 wall-clock 数字

- **DeepCache**（官方论文/repo）：
  - SD v1.5，50 PLMS 步：**2.3×**，CLIP Score 仅 -0.05（512×512）。
  - LDM-4-G ImageNet 256，250 DDIM 步：**4.1×**（FID +0.22）；激进设置最高 **7.0×**。
  - 免训练、即插即用；与步数减少正交（少步 sampler 上仍有 ~1.5–2× 余量，但间隔越大退化越明显）。
- **Faster Diffusion**：SD 上约 **1.8×**（DDIM 50 步）、DeepFloyd-IF 提及 ~1.5×；论文报告 SD 采样时间降低约 24–41%（不同任务/配置）[逐项配置数字未核验，量级来自论文与 repo]。可叠加 DDIM/DPM-Solver。
- **Block Caching**：SD/EMU 类模型上在**相同 FLOPs 预算下质量更好**，或同质量下约 **1.5–1.8×** 加速 [具体倍数未核验，论文以"同预算提质"为主要叙述]。
- **TGATE**：官方 repo 与 diffusers 文档：**10–50% 加速**（取决于 pipeline 与 gate step），免训练，可与 DeepCache 叠加。

### 2.3 质量控制机制

- **间隔与深度旋钮**：DeepCache 的 cache_interval（隔几步刷新）与 cache_branch_id（保留多浅的实时分支）直接权衡速度/质量；间隔=2~3 几乎无损，=5+ 明显退化。
- **非均匀刷新**：DeepCache 支持非均匀间隔（在特征变化快的时段刷新更密）。
- **误差校正**：Block Caching 引入按 timestep 的轻量 scale-shift 校正层，抑制缓存导致的伪影。
- **阶段划分**：TGATE 只在"保真阶段"（cross-attn 收敛后）跳过，规划阶段不动，从机制上界定了安全区。
- 共同点：**误差不累积失控的原因是每隔 N 步做一次全量计算"锚定"**。

### 2.4 迁移到 CNN 超分（本调研最重要一节）

CNN-SR 单次前向、无时间步，"相邻步相似性"需重新寻找**冗余轴**。可行映射：

1. **时间轴 → 视频/连拍帧轴（最直接）**：视频 SR 中相邻帧内容高度相似 ⇔ 相邻时间步特征相似。可缓存上一帧的深层特征，本帧只计算浅层 + 运动补偿对齐后复用深层（DeepCache 的"深层复用、浅层重算"结构可原封照搬到 recurrent VSR）。
2. **时间轴 → 空间 patch 轴**：同一图内相似 patch（重复纹理、平坦天空）深层特征近似 ⇔ 相邻步特征近似。做 patch 级特征缓存/查表：对与已计算 patch 相似度高的 patch 复用其深层特征、只跑浅层残差修正。这本质是 ToMe 的缓存版（见 §3）。风险：相似度检索本身有开销，需 hash/低分辨率代理特征。
3. **时间轴 → 尺度/级联轴**：多尺度或级联 SR（×2 再 ×2、LAPAR/ProSR 式金字塔）中，上一尺度的深层语义特征对下一尺度大体不变 → **跨尺度缓存复用**：低分辨率阶段算好的语义特征上采样后直接注入高分辨率阶段，高分辨率阶段只保留浅层高频分支。这与 DeepCache "高层复用 + 低层便宜更新" 同构，且 U-Net skip 结构在 SR 网络（如 U-shaped restoration 网络）中现成可用。
4. **时间轴 → 层间轴（网络内部）**：Faster Diffusion 发现"encoder 变化平缓"→ CNN-SR 对应发现是**深层特征沿网络传播时相邻 block 输出高度相似**（残差网络的迭代精化视角，ResNet ≈ 离散 ODE）。可做 block 级 early-exit / 隔块复用：对"容易"的输入或区域跳过部分残差 block（与 §6 自适应深度合流）。
5. **TGATE 的启示——组件级收敛检测**：SR 网络中某些组件（如 non-local/attention 模块、退化估计分支）的输出可能在浅层就"收敛"，对后续层近似常量 → 计算一次后广播复用，而不是每个 block 都算。对含 transformer 块的混合 SR 网络（如 HAT、SwinIR 变体）尤其适用：cross-window attention 图可跨层复用。
6. **质量保证的迁移**：锚定思想照搬——每 N 帧/每 N 个 patch 簇/每个尺度起点做一次全量计算；对复用区域加轻量校正头（Block Caching 的 scale-shift）；用便宜代理（浅层特征差异范数）在线判断缓存是否失效。

> **小结**：缓存复用是六类机制中对 CNN-SR 启发密度最高的：它不依赖"扩散"本身，只依赖"存在一个变化缓慢的冗余轴 + 深浅层计算成本不对称"。CNN-SR 中该冗余轴天然存在于 **帧间、patch 间、尺度间、block 间** 四个维度。

---

## 3. 空间自适应（Spatially Adaptive Computation）

### 3.1 论文清单（均已核实）

| 论文 | 出处 | 机制 |
|---|---|---|
| ToMe for SD: Token Merging for Fast Stable Diffusion (Bolya & Hoffman) | CVPRW 2023, arXiv:2303.17604 | 在 SD 的 transformer 块内合并冗余 token（合并前算、输出再 unmerge），免训练 |
| RAS: Region-Adaptive Sampling for Diffusion Transformers (Liu et al., Microsoft) | CVPR 2026, arXiv:2502.10389 | 每步只更新模型当前关注的语义区域 token，其余区域复用缓存噪声输出；关注区跨步连续 |
| QDM: Quadtree-Based Region-Adaptive Sparse Diffusion for SR | arXiv:2503.12015 | 由 LQ 输入构建四叉树，细节区细粒度去噪、平坦区粗粒度/跳过 |
| PatchScaler: Patch-Independent Diffusion for SR | arXiv:2405.17158 | 按 patch 重建难度分配不同的采样步数 |

### 3.2 数字

- **ToMe for SD**：token 减 60%，生成 **最高 2×** 加速、内存最高降 5.6×，FID 几乎不变（512×512，免训练）；与 xFormers 叠加在高分辨率下最高 5.4×。
- **RAS**：Lumina-Next-T2I / SD3 上官方报告最高 **2.36× / 2.51×** 加速，人评质量退化很小（官方页面数字）。
- QDM / PatchScaler：SR 场景计算量随平坦区占比下降，具体倍数依赖内容 [逐项加速未核验]。

### 3.3 质量控制

- ToMe：合并率 r 连续可调；只合并（相似度高的）冗余 token，输出侧 unmerge 保持分辨率。
- RAS：关注区由模型自身注意力/噪声变化决定（自监督的难度信号）；周期性全量更新防止盲区漂移。

### 3.4 迁移到 CNN 超分

这是**对 SR 最天然适配**的一类，且 SR 社区已有先驱（可作对照）：ClassSR (CVPR 2021)、APE、MASA 等 patch 难度路由工作。扩散侧的新增启发：

1. **ToMe → SR transformer 块的 token 合并**：SwinIR/HAT 类 SR 中平坦区 window 的 token 可先合并再算 attention/MLP、算完 unmerge。SR 输入是自然图（比生成中间态更结构化），token 冗余度往往更高，预计收益 ≥ 生成场景。注意：SR 对像素级保真敏感，合并需限制在深层/低频通路，浅层高频路径保持全分辨率。
2. **RAS → 区域难度的"模型自报"信号**：不用外部分类器（ClassSR 的做法），而用**网络自身中间量**（残差幅值、注意力熵）在线决定哪些区域走完整深网络、哪些区域早退到轻量上采样。RAS 证明这种自监督难度信号足够可靠且几乎免费。
3. **QDM 四叉树 → 显式的多粒度执行计划**：CNN-SR 可在推理前由 LQ 图梯度/方差构建四叉树，平坦大块直接 bicubic+轻修正、细节小块走满网络——把稀疏计算从"每像素 mask"（GPU 不友好）变成"块级批处理"（GPU 友好），解决空间自适应落地的最大工程痛点。

---

## 4. 一步扩散超分（One-Step Diffusion SR）—— 效率对标

### 4.1 论文清单与效率数字（均已核实）

| 论文 | 出处 | 步数 | 实测效率 |
|---|---|---|---|
| SinSR (Wang et al.) | CVPR 2024, arXiv:2311.14760 | 1（从 ResShift 15 步蒸馏） | 512×512 单步 **~0.13 s**（A100 级，第三方汇总表）；vs ResShift 15 步 ~0.71 s，约 **5.4×**；vs LDM-SR 百步方案 >10× |
| OSEDiff (Wu et al.) | NeurIPS 2024, arXiv:2406.08177 | 1（VSD 正则 + LoRA 微调 SD） | 官方：A100、512×512 输入 **~0.1 s/图**（repo 提供测时脚本）；vs StableSR(200 步)/SeeSR(50 步) 数十倍加速；可训练参数仅 8.5 M |
| AddSR (Xie et al.) | arXiv:2404.01717 (ECCV 2024) | 1–4（对抗扩散蒸馏 ADD） | 4 步 **0.8 s** 内完成高感知质量 BSR（论文 Fig.1）；论文自述比 SeeSR 快约 7×（AddSR-1）[7× 未核验精确条件] |
| TSD-SR (Dong et al.) | CVPR 2025, arXiv:2411.18263 | 1（Target Score Distillation + 蒸馏采样优化） | 基于 SD3；论文自述推理速度优于同类一步法（约 0.1s 级，512×512）[精确毫秒数未核验] |

参考基线：StableSR ~200 步、DiffBIR/SeeSR ~50 步、ResShift 15 步——一步法相对它们是 **15–200× 的 NFE 缩减**，wall-clock 上 5–100× 不等（取决于基线）。

### 4.2 质量控制

- SinSR：确定性映射蒸馏（teacher 的 ODE 轨迹端点回归）+ consistency 保持损失。
- OSEDiff：**变分分数蒸馏 (VSD)** 把预训练 SD 当作分布先验正则，防止一步生成塌向过平滑；LoRA 只训少量参数以保留先验。
- AddSR：ADD（对抗 + 蒸馏）+ 提出预测输出作为下一步条件的 timestep-aware 策略，控制 1 步（保真偏向）到 4 步（感知偏向）的权衡。
- TSD-SR：目标分数蒸馏缓解 VSD 的偏差 + 面向细节的蒸馏采样。

### 4.3 迁移到 CNN 超分

1. **对标意义**：一步扩散 SR 在 A100 上 512×512 约 0.1 s，仍比高效 CNN-SR（同分辨率毫秒级）慢 1–2 个数量级。CNN-SR 的效率叙事应强调这一差距；同时一步法给出了"感知质量上界"参照。
2. **师生框架直接可用**：把 OSEDiff/TSD-SR 当教师，向轻量 CNN 蒸馏其感知质量（VSD/分数蒸馏损失同样可以作用于 CNN 学生——学生不必是扩散模型，DMD 系工作已证明学生只需是个生成器）。
3. **LoRA-式参数高效微调**：OSEDiff 仅训 8.5M 参数即可改造教师，提示 CNN-SR 的场景自适应（按退化类型/域切换 LoRA 分支）可以极低成本实现，不必重训整网。

---

## 5. 量化与编译

### 5.1 论文/工具与实测数字（均已核实）

| 工作 | 出处 | 实测收益 |
|---|---|---|
| SVDQuant (Li et al., MIT Han Lab) | ICLR 2025, arXiv:2411.05007 | W4A4：FLUX.1-dev 12B 显存 **3.5–3.6× ↓**；在 16GB laptop RTX 4090 上（消除 CPU offload 后）端到端 **8.7× 加速**（111.7s→12.9s，25 步，LPIPS 0.223）；纯 kernel 层面相对 BF16 约 3× [注：8.7× 含免 offload 收益，纯计算收益约 3×]。Nunchaku 推理引擎开源 |
| Torch-TensorRT（NVIDIA 官方博客） | developer.nvidia.com | 扩散模型（SD/FLUX 类）相对原生 PyTorch 约 **2×**（官方标题级结论，FP8/优化 kernel） |
| torch.compile + diffusers（PyTorch 官方博客） | pytorch.org | SD/FLUX pipeline 显著提速；社区常见数字为单模型 1.1–1.5×，与 regional compilation 组合降低编译时间 [具体倍数依模型/GPU，未逐项核验] |

### 5.2 质量控制

- SVDQuant 核心：**16-bit 低秩分支吸收权重/激活的离群值**，残差再做 4-bit 量化；低秩分支与 4-bit 分支 kernel 融合避免额外访存。质量用 LPIPS/ImageReward 监控（FLUX 上 LPIPS 0.223，优于 NF4 W4A16 的 0.272）。
- 编译类（TensorRT/torch.compile）数值上近似无损（同精度）或受控（FP8/FP16 混合）。

### 5.3 迁移到 CNN 超分

1. **完全架构无关，直接适用**：INT8 PTQ + TensorRT 是 CNN-SR 部署的成熟路线；SVDQuant 的新增启发是**当想压到 W4A4 时，用低秩旁路吸收离群值**——SR 网络（尤其含 attention 的）同样存在激活离群值，此技巧可平移。
2. **注意 SR 的特殊敏感性**：SR 输出是像素回归，量化误差直接表现为 banding/色偏，比生成任务更可见；建议对首尾卷积与上采样层保持高精度（扩散量化工作也普遍保护 in/out 层）。
3. torch.compile / CUDA graph 对小 batch、多次调用的 SR 服务收益显著（消除 Python/launch 开销），是"免费"的第一步。

---

## 6. 自适应步数 / 自适应计算量

### 6.1 论文清单（均已核实；两篇同名 AdaDiff 是不同工作）

| 论文 | 出处 | 机制 |
|---|---|---|
| AdaDiff: Adaptive Step Selection for Fast Diffusion Models (Zhang et al.) | AAAI 2025, arXiv:2311.14768 | 轻量步数选择策略网络，按 prompt 丰富度决定该样本用多少去噪步；奖励 = 质量-步数权衡的策略梯度训练 |
| AdaDiff: Accelerating Diffusion Models through Step-Wise Adaptive Computation (Tang et al.) | ECCV 2024, arXiv:2309.17074 | 每层挂 timestep-aware 不确定性估计头，不确定性低时跳过该层（步内早退/层跳过） |
| （相关）RAS（见 §3） | CVPR 2026 | 区域级的自适应更新频率，可视为空间维的自适应步数 |

### 6.2 数字与质量控制

- Step Selection AdaDiff：论文报告在保持质量指标（FID/CLIP）近乎不变下平均步数明显下降，加速比依数据集约 **1.3–1.6×** [精确数字未核验，量级来自论文叙述]。
- Step-Wise AdaDiff (ECCV24)：论文报告最高约 **40% 计算量节省** 且指标基本持平（文中出现 40% 上限表述）。
- 质量控制：不确定性阈值可调；策略训练时质量项直接进奖励。

### 6.3 迁移到 CNN 超分

1. **"步数" → "深度/分支"**：无时间步的 CNN-SR 中，等价物是**样本级/区域级早退**：容易样本走浅路径。SR 已有直接对应物（ClassSR、Path-Restore、动态深度网络），扩散侧的增量启发是：
   - 用**模型内生不确定性**（ECCV24 AdaDiff 的思路）而非外部分类器做路由，几乎零额外成本；
   - 用**策略梯度直接优化"质量-计算"奖励**（AAAI25 AdaDiff），而不是启发式阈值。
2. **"步数" → "细化迭代次数"**：若 SR 采用 iterative refinement（如 flow/recurrent 细化），可让停止时刻由残差收敛判据决定——直接照搬自适应步数。
3. 落地要点：批处理下动态深度会造成 warp divergence / 打包开销，建议块级（quadtree/patch-batch）而非像素级路由（同 §3.4 第 3 条）。

---

## 7. 汇总：对 CNN-SR 最值得落地的五条启发（按性价比排序）

1. **跨尺度/跨块的"深层缓存 + 浅层更新"**（DeepCache/Faster Diffusion 同构迁移）：在级联或 U 形 SR 网络中，语义/深层特征算一次、多处复用，浅层高频分支全量计算。免训练原型可行，预期 1.5–2×。
2. **块级空间自适应路由**（RAS/QDM + ClassSR 传统）：四叉树块划分 + 模型内生难度信号 + 块级批处理执行；平坦区走 bicubic+轻修正。内容依赖，自然图上通常 1.5–3×。
3. **分布匹配式蒸馏代替 L1 蒸馏**（DMD/OSEDiff 启发）：以一步扩散 SR 为教师、用 VSD/GAN 型分布损失训练轻量 CNN 学生，兼得 CNN 速度与扩散感知质量。
4. **量化 + 编译作为无风险底座**（SVDQuant/TensorRT）：INT8+TensorRT 起步（~2×），需要更激进时借鉴低秩旁路吸收离群值做 W4A4。
5. **组件级收敛复用**（TGATE 启发）：SR 网络中 attention 图/退化估计等"慢变量"组件计算一次后跨层广播，尤其适用于 window-attention 型 SR。

关键差异提醒：扩散加速的误差被后续去噪步"洗掉"一部分（自我修复），而 CNN-SR 单前向没有这种容错——**所有复用/跳过都必须配显式校正机制**（轻量残差头、周期性全量锚定、失效检测阈值），这是迁移时最不能省略的部分。

---

## 附：论文核验状态清单

全部经检索验证真实存在：PD (2202.00512)、CM (2303.01469)、LCM (2310.04378)、InstaFlow (2309.06380)、DMD (2311.18828)、DMD2 (2405.14867)、DeepCache (2312.00858, CVPR24)、Faster Diffusion (2312.09608, NeurIPS24)、Block Caching (2312.03209, CVPR24)、TGATE (2404.02747)、FasterCache (2410.19355)、ToMe for SD (2303.17604, CVPRW23)、RAS (2502.10389, CVPR26)、QDM (2503.12015)、PatchScaler (2405.17158)、SinSR (2311.14760, CVPR24)、OSEDiff (2406.08177, NeurIPS24)、AddSR (2404.01717)、TSD-SR (2411.18263, CVPR25)、SVDQuant (2411.05007, ICLR25)、AdaDiff-步选择 (2311.14768, AAAI25)、AdaDiff-层跳过 (2309.17074, ECCV24)。

[未核验] 标注项：InstaFlow 0.09s 的 GPU 变体细节；Block Caching 精确加速倍数；Faster Diffusion 各任务逐项百分比；AddSR 7× 对比条件；TSD-SR 精确毫秒；torch.compile 逐模型倍数；AdaDiff(AAAI25) 精确加速比；QDM/PatchScaler 加速倍数。
