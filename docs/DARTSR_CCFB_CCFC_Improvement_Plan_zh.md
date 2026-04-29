# 面向 CCF-B 冲击与 CCF-C 保底的 DART-SR 改进方案

## 1. 方法定位

当前 DART-SR 更适合定位为一个 **即插即用的超分辨率加速框架**，而不是一个全新的超分网络模型。

两者的区别是：

- 新的超分模型通常会重新设计主干网络结构。
- DART-SR 保留已有超分主干，例如 RCAN、CARN、SRResNet、EDSR、SwinIR 或 HAT，只在其外部或中间插入轻量级动态推理控制器，用于减少不必要的计算。

因此，改进后的方法可以表述为：

> DART++ 是一种面向高效图像超分辨率的、与主干网络无关的动态推理加速框架。它包装已有预训练超分模型，并在可控计算预算下，根据预测的重建收益为不同空间窗口自适应分配计算量。

这个定位很关键。ClassSR、FADN、AdaDSR、AdaFormer 以及近期一些高频先验掩码方法，已经覆盖了区域难度、频率感知和 token 稀疏化等超分加速思路。若要冲击 CCF-B，同时保底 CCF-C，改进后的 DART-SR 需要从“启发式跳过窗口”升级为更有说服力的“收益感知计算分配框架”。

## 2. 当前 DART-SR 的主要不足

当前实现已经具备一个可运行的原型，但从论文创新性的角度看，还存在几个明显短板。

第一，当前路由分数偏启发式。它主要由 router logits、局部方差和退化感知阈值组成：

```text
score = router_logits + var_weight * local_variance - tau
tau = tau0 + alpha * degradation_score
```

这容易被解释为“复杂区域多计算，简单区域少计算”，与已有的区域难度、频率感知、token 重要性方法存在明显重叠。

第二，退化分数目前主要用于调节阈值，并没有显式分配每个 stage 的计算预算，也不能自然支持可控的速度-质量权衡。

第三，被跳过的窗口主要走 identity 路径。对于低层视觉任务来说，这可能过于激进，因为平滑区域也可能需要低频校正，窗口边界也可能出现不连续。

第四，FLOPs 下降不一定等于真实推理延迟下降。窗口选择、索引、scatter/gather 等操作本身会引入开销。若目标是 CCF-B，需要严谨报告 GPU 同步延迟，并尽可能设计硬件友好的窗口打包策略。

## 3. 核心升级：从难度路由到收益路由

最重要的改进方向是：将路由目标从 **区域难度估计** 升级为 **边际重建收益估计**。

当前方法更像是在问：

> 这个 window 复杂吗？

改进后的方法应该问：

> 如果当前 stage 处理这个 window，相比跳过它，能带来多大的重建收益？

对于每个 stage 和每个 window，路由器预测：

```text
benefit_score = BenefitRouter(feature_window, degradation_code, stage_id, budget_code)
```

高收益窗口进入原始重型 SR block，低收益窗口走跳过路径或轻量补偿路径。

这样论文主张会更强：

> DART++ 并不是简单根据图像复杂度或高频强度分配计算，而是根据预测的边际重建收益进行动态计算分配。

## 4. 改进后的整体框架

改进后的框架仍然应该保持“已有 backbone + 加速插件”的形式，而不是设计一个新的整体网络。

```text
LR image
  |
  +--> 预训练 SR backbone
  |       |
  |       +--> stage 1
  |       +--> stage 2
  |       +--> ...
  |       +--> stage K
  |
  +--> DART++ acceleration plugin
          |
          +--> 退化与复杂度描述器
          +--> 预算分配器
          +--> 每个 stage 的收益路由器
          +--> 路由 mask

选择性执行：
  高收益 window -> 原始 heavy SR block
  低收益 window -> identity 或 cheap adapter

输出：
  SR reconstruction
```

主干网络仍然可以替换。DART++ 应该作为一个动态推理控制器，包装不同 SR 架构，而不是成为一个与特定 backbone 深度绑定的新网络。

## 5. 关键模块设计

### 5.1 退化与复杂度描述器

当前的 `DegradationEstimator` 可以扩展为更可靠的轻量描述器。

输入可以包括：

- LR 图像；
- backbone 的浅层特征；
- 频率线索，例如 Laplacian 响应、wavelet 响应或 Gaussian 高频残差。

输出可以包括：

- 全局退化编码；
- 图像复杂度统计；
- 可选的 stage 条件路由上下文。

这个模块必须保持轻量。它不能变成主要计算模块，否则 DART++ 的“加速框架”定位会被削弱。

### 5.2 预算分配器

预算分配器根据退化编码、复杂度编码和用户给定预算，输出每个 stage 的 keep ratio：

```text
[r1, r2, ..., rK] = BudgetAllocator(degradation_code, complexity_code, user_budget)
```

这样可以实现可控推理：

- 高质量模式：更大的 keep ratio；
- 均衡模式：中等 keep ratio；
- 快速模式：更小的 keep ratio。

这个模块能让论文从“一个固定加速点”升级为“可控速度-质量 Pareto 框架”。

### 5.3 收益路由器

收益路由器负责预测每个 stage 中每个 window 的边际重建收益：

```text
benefit_map_k = BenefitRouter_k(feature_k, degradation_code, budget_code)
```

训练监督可以来自 dense teacher。比如：

```text
benefit_proxy = loss(skip_output, GT) - loss(dense_output, GT)
```

也可以使用更轻量的特征或输出差异：

```text
benefit_proxy = || dense_stage_output - skipped_stage_output ||
```

第一种方式更强，但计算开销更大；第二种方式实现简单，可以作为辅助监督。

### 5.4 Top-k 路由与 STE

不再使用统一固定阈值，而是让每个 stage 按照预算分配器给出的 keep ratio 选择收益最高的 top-k windows：

```text
mask_k = TopK(benefit_map_k, keep_ratio=r_k)
```

训练时可以使用直通估计器：

```text
mask = hard_mask + soft_score - soft_score.detach()
```

这样既能保持路由可训练，又能让训练行为更接近推理时的 hard routing。

### 5.5 Heavy/Cheap 双路径执行

对于每个被路由的 stage，可以采用双路径：

```text
if mask = 1:
    output = heavy SR block(input)
else:
    output = cheap adapter(input) or identity(input)
```

cheap adapter 可以设计为：

- identity 路径，速度最快；
- 共享的 `1x1 conv + depthwise 3x3 conv`；
- low-rank residual adapter；
- 简单低频校正路径。

若目标是冲击 CCF-B，建议保留一个极轻量 cheap path。它可以减少直接跳过带来的重建误差和边界伪影，也能让方法更稳健。

## 6. 训练目标

总损失可以设计为：

```text
L = L_sr
  + lambda_budget * L_budget
  + lambda_sparse * L_sparse
  + lambda_benefit * L_benefit
  + lambda_tv * L_mask_tv
  + lambda_distill * L_distill
```

各项含义：

- `L_sr`：L1 或 Charbonnier 重建损失；
- `L_budget`：约束实际 keep ratio 接近预算分配器给出的目标；
- `L_sparse`：鼓励在可接受质量下减少计算；
- `L_benefit`：让预测收益与 teacher/proxy benefit 对齐；
- `L_mask_tv`：约束 mask 空间平滑，减少窗口边界伪影；
- `L_distill`：让动态模型输出接近 dense teacher 输出。

其中最关键的是 `L_benefit`。它支撑论文的核心主张：路由依据不是简单复杂度，而是预测重建收益。

## 7. 硬件友好加速

如果要让“加速”主张有说服力，必须从理论 FLOPs 走向真实延迟。

建议改进：

- 推理计时前后使用 `torch.cuda.synchronize()`；
- 在固定硬件上报告真实 latency；
- 将 active windows 打包成紧凑 batch 后再进入 heavy block；
- 减少 scatter/gather 开销；
- 报告 batch size、输入分辨率、GPU 型号、PyTorch/CUDA 版本和 warm-up 次数。

对于 CNN backbone，可以将 active windows 打包成 batch 统一处理。对于 Transformer backbone，可以在 window attention 中裁剪低收益 tokens。

这个点也可以作为论文的次级贡献：

> 提出硬件友好的窗口打包策略，将路由稀疏性转化为真实 GPU 延迟收益。

## 8. CCF-C 保底实验配置

若以 CCF-C 为保底，最低实验包建议包括：

- Backbones：CARN、SRResNet、RCAN、EDSR；
- Datasets：DIV2K validation、Set5、Set14、B100、Urban100、Manga109；
- Baselines：原始 dense backbone、ClassSR、FADN、AdaDSR、当前 DART-SR，以及一个近期高频掩码方法；
- Metrics：PSNR、SSIM、FLOPs、真实 latency、GPU memory；
- Ablations：
  - 方差路由 vs 收益路由；
  - 固定阈值 vs 预算分配器；
  - identity skip vs cheap adapter；
  - soft routing vs hard routing；
  - 有无退化描述器。

CCF-C 层面的主张可以是：

> DART++ 为 CNN-based SR backbone 提供了一种通用、可控的动态推理插件，并在速度-质量权衡上优于已有区域感知加速方法。

## 9. CCF-B 冲击实验配置

若要冲击 CCF-B，需要进一步扩展实验范围。

建议增加：

- Transformer SR backbone，例如 SwinIR 或 HAT；
- 大图测试集，例如 Test2K、Test4K、Test8K 或 DIV8K crops；
- 未见退化鲁棒性测试，例如 blur、noise、JPEG、mixed real-world degradation；
- 多个用户预算下的质量-速度 Pareto 曲线；
- mask 可视化和 routing-benefit correlation 分析；
- 不同输入分辨率下的真实 latency 对比，而不仅是 FLOPs 对比。

CCF-B 层面的主张应该升级为：

> DART++ 是一个统一的收益感知计算分配框架，可同时适配 CNN 与 Transformer 超分主干，在多种退化条件下实现可控动态推理，并带来真实硬件加速。

## 10. 论文贡献表述建议

论文可以组织为以下贡献：

1. 提出一种即插即用的图像超分动态推理加速框架，可包装已有 SR backbone，而不需要重新设计主干网络。
2. 提出收益感知路由机制，预测每个空间窗口在每个 stage 的边际重建收益。
3. 提出退化条件预算分配器，实现不同图像和不同退化程度下的可控速度-质量权衡。
4. 提出 heavy/cheap 双路径执行策略，并结合硬件友好的窗口打包，将路由稀疏性转化为实际推理加速。
5. 在 CNN 和 Transformer SR 模型、标准和大分辨率数据集、合成和未见退化场景中进行系统验证。

## 11. 实现路线图

建议按以下顺序推进：

1. 重构当前 DART-SR plugin，将路由逻辑与 backbone-specific forward 解耦。
2. 加入可靠的同步 latency 测量和日志记录。
3. 将固定阈值路由替换为 stage-wise top-k 路由。
4. 增加 dense-vs-skip 差异生成 benefit proxy。
5. 实现 `BenefitRouter` 和 `L_benefit`。
6. 实现 `BudgetAllocator`，支持用户可控 keep ratio。
7. 增加可选 cheap adapter 路径。
8. 从 RCAN/CARN/SRResNet 扩展到 EDSR。
9. 若目标是 CCF-B，进一步扩展到 SwinIR/HAT。
10. 构建完整 ablation 和 Pareto curve 评估脚本。

## 12. 核心表达

不建议将方法描述为：

> 一个由退化编码器、频率编码器、路由器和 adapter 组成的新超分网络。

更建议描述为：

> 一个面向已有超分网络的通用加速控制器。它预测计算在哪里最有重建收益，并在可控预算下只执行必要的空间窗口。

这样既能保留“通用加速框架”的身份，又能显著提高方法创新性，为冲击 CCF-B 和保底 CCF-C 提供更清晰的论文故事。
