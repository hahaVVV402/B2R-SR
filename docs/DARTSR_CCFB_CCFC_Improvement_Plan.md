# DART-SR Improvement Plan for CCF-B Target and CCF-C Backup

## 1. Positioning

The current DART-SR implementation should be positioned as a **plug-and-play super-resolution acceleration framework**, not as a standalone super-resolution network.

The key distinction is:

- A new SR model redesigns the main restoration backbone.
- DART-SR keeps an existing SR backbone, such as RCAN, CARN, SRResNet, EDSR, SwinIR, or HAT, and attaches a lightweight dynamic inference controller to reduce unnecessary computation.

Therefore, the improved method should be described as:

> DART++ is a backbone-agnostic dynamic inference framework for efficient image super-resolution. It wraps pretrained SR backbones and allocates computation to spatial windows according to predicted reconstruction benefit under a controllable computation budget.

This positioning is important because related works such as ClassSR, FADN, AdaDSR, AdaFormer, and recent high-frequency masking methods already cover different forms of region-aware or token-aware SR acceleration. To target CCF-B and keep CCF-C as a safer fallback, the improvement should emphasize a stronger and clearer contribution than heuristic window skipping.

## 2. Limitation of the Current DART-SR

The current implementation is a reasonable prototype, but its paper-level novelty is limited by several factors.

First, the routing score is mostly heuristic. It combines router logits, local variance, and a degradation-conditioned threshold:

```text
score = router_logits + var_weight * local_variance - tau
tau = tau0 + alpha * degradation_score
```

This can be interpreted as "complex regions receive more computation", which overlaps with existing SR acceleration ideas based on region difficulty, frequency, and token importance.

Second, the degradation score only adjusts the threshold. It does not explicitly allocate a stage-wise computation budget or provide a controllable speed-quality tradeoff.

Third, skipped windows currently rely mainly on identity forwarding. This may be too aggressive for low-level restoration because smooth regions can still require low-frequency correction and boundary consistency.

Fourth, FLOPs reduction does not automatically imply real latency reduction. Window selection, indexing, and scatter/gather operations may introduce overhead. For a stronger paper, the framework must report synchronized GPU latency and ideally include hardware-aware window packing.

## 3. Core Upgrade: From Difficulty Routing to Benefit Routing

The central improvement should be changing the routing objective from **difficulty estimation** to **marginal reconstruction benefit estimation**.

Instead of asking:

> Is this window complex?

the router should ask:

> How much will this window benefit if the current stage is executed instead of skipped?

For each stage and each window, the router predicts:

```text
benefit_score = BenefitRouter(feature_window, degradation_code, stage_id, budget_code)
```

High-benefit windows are processed by the original heavy SR block. Low-benefit windows are skipped or processed by a cheap compensation path.

This gives a stronger claim:

> DART++ allocates computation according to predicted reconstruction gain, not merely image complexity or high-frequency magnitude.

## 4. Proposed Framework

The improved framework should keep the existing backbone as the main restoration network and insert a lightweight controller around its stages.

```text
LR image
  |
  +--> pretrained SR backbone
  |       |
  |       +--> stage 1
  |       +--> stage 2
  |       +--> ...
  |       +--> stage K
  |
  +--> DART++ acceleration plugin
          |
          +--> degradation and complexity descriptor
          +--> budget allocator
          +--> benefit router for each stage
          +--> routing masks

Selective execution:
  high-benefit window -> original heavy SR block
  low-benefit window  -> identity or cheap adapter

Output:
  SR reconstruction
```

The backbone remains replaceable. DART++ should be implemented as a controller that can wrap different SR architectures rather than a new monolithic network.

## 5. Main Modules

### 5.1 Degradation and Complexity Descriptor

The current `DegradationEstimator` can be expanded into a more reliable descriptor module.

Inputs:

- LR image.
- Optional shallow backbone features.
- Frequency cues such as Laplacian response, wavelet response, or Gaussian high-frequency residual.

Outputs:

- Global degradation code.
- Image-level complexity statistics.
- Optional stage-conditioned routing context.

The descriptor should remain lightweight. It should not become the main network, otherwise the framework positioning becomes weaker.

### 5.2 Budget Allocator

The budget allocator converts degradation and user budget into stage-wise keep ratios.

```text
[r1, r2, ..., rK] = BudgetAllocator(degradation_code, complexity_code, user_budget)
```

This enables controllable inference:

- High-quality mode: larger keep ratio.
- Balanced mode: medium keep ratio.
- Fast mode: smaller keep ratio.

This is useful for paper framing because it gives a Pareto curve instead of a single operating point.

### 5.3 Benefit Router

The benefit router predicts window-level marginal reconstruction benefit for each stage.

```text
benefit_map_k = BenefitRouter_k(feature_k, degradation_code, budget_code)
```

Training supervision can be derived from a dense teacher:

```text
benefit_proxy = loss(skip_output, GT) - loss(dense_output, GT)
```

or from feature/output discrepancy:

```text
benefit_proxy = || dense_stage_output - skipped_stage_output ||
```

The first option is stronger but more expensive. The second option is simpler and can be used as an auxiliary target.

### 5.4 Top-k Routing with STE

Instead of using a fixed threshold for all images, each stage selects the top-k windows according to the allocated keep ratio.

```text
mask_k = TopK(benefit_map_k, keep_ratio=r_k)
```

During training, a straight-through estimator can be used:

```text
mask = hard_mask + soft_score - soft_score.detach()
```

This keeps routing trainable while matching hard inference behavior.

### 5.5 Heavy and Cheap Dual Path

For each routed stage:

```text
if mask = 1:
    output = heavy SR block(input)
else:
    output = cheap adapter(input) or identity(input)
```

The cheap adapter can be:

- identity path for maximum speed;
- shared `1x1 conv + depthwise 3x3 conv`;
- low-rank residual adapter;
- simple low-frequency correction path.

For CCF-B targeting, a lightweight cheap path is recommended because it reduces artifacts caused by direct skipping and makes the method more robust.

## 6. Training Objective

The total loss can include:

```text
L = L_sr
  + lambda_budget * L_budget
  + lambda_sparse * L_sparse
  + lambda_benefit * L_benefit
  + lambda_tv * L_mask_tv
  + lambda_distill * L_distill
```

Recommended terms:

- `L_sr`: L1 or Charbonnier reconstruction loss.
- `L_budget`: keep ratio should match the allocated budget.
- `L_sparse`: encourages lower computation when possible.
- `L_benefit`: aligns predicted benefit with teacher/proxy benefit.
- `L_mask_tv`: encourages spatially smooth masks to reduce boundary artifacts.
- `L_distill`: aligns dynamic output with dense teacher output.

The most important new loss is `L_benefit`, because it supports the main claim that routing is based on predicted reconstruction gain.

## 7. Hardware-Aware Acceleration

To make the acceleration claim credible, the implementation should move beyond theoretical FLOPs.

Required improvements:

- Use `torch.cuda.synchronize()` before and after timing.
- Report real latency on fixed hardware.
- Pack active windows into a compact batch before running heavy blocks.
- Minimize scatter/gather overhead.
- Report batch size, input resolution, GPU model, PyTorch/CUDA version, and warm-up iterations.

For CNN backbones, active windows can be packed and processed as a batch. For Transformer backbones, inactive window tokens can be pruned from local attention computation.

This can become a secondary contribution:

> A hardware-aware window packing strategy that converts routing sparsity into measurable GPU latency reduction.

## 8. Experiments for CCF-C Backup

The minimum experiment package for a CCF-C submission should include:

- Backbones: CARN, SRResNet, RCAN, EDSR.
- Datasets: DIV2K validation, Set5, Set14, B100, Urban100, Manga109.
- Baselines: original dense backbone, ClassSR, FADN, AdaDSR if available, current DART-SR, and one recent high-frequency masking method if reproducible.
- Metrics: PSNR, SSIM, FLOPs, real latency, GPU memory.
- Ablations:
  - variance routing vs benefit routing;
  - fixed threshold vs budget allocator;
  - identity skip vs cheap adapter;
  - soft routing vs hard routing;
  - with and without degradation descriptor.

The CCF-C-level claim can be:

> DART++ provides a general and controllable dynamic inference plugin for CNN-based SR backbones and achieves better speed-quality tradeoffs than prior region-aware acceleration methods.

## 9. Experiments for CCF-B Target

To make the work competitive for CCF-B, the experiment scope should be expanded.

Additional requirements:

- Add Transformer SR backbones such as SwinIR or HAT.
- Add large-resolution benchmarks: Test2K, Test4K, Test8K, or DIV8K crops.
- Evaluate unseen degradation robustness: blur, noise, JPEG, mixed real-world degradation.
- Provide quality-speed Pareto curves across multiple user budgets.
- Include mask visualization and routing-benefit correlation analysis.
- Compare real latency under multiple resolutions, not only FLOPs.

The CCF-B-level claim should be:

> DART++ is a unified benefit-aware computation allocation framework for both CNN and Transformer SR backbones, providing controllable acceleration under diverse degradations with real hardware speedup.

## 10. Suggested Paper Contributions

The paper can present the following contributions:

1. A plug-and-play dynamic inference framework for efficient image super-resolution that wraps existing SR backbones without redesigning the backbone architecture.
2. A benefit-aware routing mechanism that predicts the marginal reconstruction gain of executing each stage on each spatial window.
3. A degradation-conditioned budget allocator that enables controllable speed-quality tradeoffs across images and degradation levels.
4. A heavy/cheap dual-path execution strategy with hardware-aware window packing for translating routing sparsity into practical latency reduction.
5. Extensive evaluation on CNN and Transformer SR models, standard and large-resolution datasets, and synthetic and unseen degradation settings.

## 11. Implementation Roadmap

Recommended implementation stages:

1. Refactor the current DART-SR plugin so routing logic is separated from backbone-specific forward functions.
2. Add synchronized latency measurement and reliable metric logging.
3. Replace fixed-threshold routing with stage-wise top-k routing.
4. Add benefit proxy generation using dense-vs-skip discrepancy.
5. Add `BenefitRouter` and `L_benefit`.
6. Add `BudgetAllocator` for user-controllable keep ratios.
7. Add optional cheap adapter path.
8. Extend from RCAN/CARN/SRResNet to EDSR.
9. Add SwinIR/HAT support if targeting CCF-B.
10. Build full ablation and Pareto-curve evaluation scripts.

## 12. Key Message

The method should not be described as:

> A new SR network with degradation encoder, frequency encoder, router, and adapter.

It should be described as:

> A general acceleration controller for existing SR networks. It predicts where computation is beneficial and dynamically executes only the necessary spatial windows under a controllable computation budget.

This is the cleanest way to preserve the "general framework" identity while raising the novelty enough to aim for CCF-B and keep CCF-C as a realistic fallback.
