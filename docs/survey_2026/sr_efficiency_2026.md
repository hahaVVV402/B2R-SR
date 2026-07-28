# 高效图像超分辨率与图像复原（2023–2026）调研报告
### 重点：真实 wall-clock 加速工作、2025–2026 最新进展、「patch 级级联 + 质量保证」查新

> 检索方式：多轮 web 检索（arXiv / CVF Open Access / IJCAI / NeurIPS / ACM DL / IEEE / MMSys / MobiCom），所有条目均经检索验证；不确定处标注 **[未核验]**。
> 生成日期以检索时环境为准。

---

## 1. NTIRE Efficient SR Challenge（2024 / 2025 / 2026）

### 1.1 NTIRE 2024（第九届，CVPRW 2024）
- **报告**：*The Ninth NTIRE 2024 Efficient Super-Resolution Challenge Report* — [arXiv:2404.10343](https://arxiv.org/abs/2404.10343)，[CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Ren_The_Ninth_NTIRE_2024_Efficient_Super-Resolution_Challenge_Report_CVPRW_2024_paper.html)；官方 repo：[Amazingren/NTIRE2024_ESR](https://github.com/Amazingren/NTIRE2024_ESR)
- **结果**：主赛道（overall）第一为 **XiaomiMM**（SPAN 系改进，runtime 约 5.6 ms 量级）；runtime 子赛道由 **SPANF**（SPAN 变体）居首。检索到的二手综述亦提到 MLP-SR 与 runtime 赛道相关 **[两处来源对 runtime 赛道第一名表述不一致，具体排名建议核对报告 Table 1；标注：部分细节未核验]**。
- **技术趋势**：SPAN/RLFN 系轻量 CNN + 结构重参数化 + 蒸馏 + 剪枝为绝对主流；出现 Mamba 蒸馏方案 **DVMSR**（[CVF PDF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Lei_DVMSR_Distillated_Vision_Mamba_for_Efficient_Super-Resolution_CVPRW_2024_paper.pdf)）。

### 1.2 NTIRE 2025（第十届，CVPRW 2025）
- **报告**：*The Tenth NTIRE 2025 Efficient Super-Resolution Challenge Report* — [arXiv:2504.10686](https://arxiv.org/abs/2504.10686)，[CVF PDF](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Ren_The_Tenth_NTIRE_2025_Efficient_Super-Resolution_Challenge_Report_CVPRW_2025_paper.pdf)；官方 repo：[Amazingren/NTIRE2025_ESR](https://github.com/Amazingren/NTIRE2025_ESR)
- **冠军（overall）**：SJTU 团队，方法 **DSCLoRA**（[arXiv:2504.11271](https://arxiv.org/abs/2504.11271)）：在预训练 SPAN 卷积层内嵌 SConvLB（LoRA 式低秩分支，可合并、零推理开销）+ 空间关系蒸馏 + 像素级蒸馏（SJTU 新闻确认获 Winner Award：[链接](https://news.sjtu.edu.cn/jdyw/20250624/212058.html)）。报告提到另有 EMSR 等方法名，runtime-only 最优方法在 overall 排第三 **[runtime 赛道具体第一名团队名未核验]**。
- **代表方案**：**ESPAN (Expanded SPAN)**（[CVF](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Wang_Expanded_SPAN_for_Efficient_Super-Resolution_CVPRW_2025_paper.pdf)）：general re-parameterization (GRep) + 自蒸馏 + 渐进训练，三件套即 2025 年高效 SR 竞赛标准配方。
- **趋势**：几乎所有前排方案 = "SPAN/RLFN 骨干 + 重参数化 + 蒸馏 + 微调/低秩适配"；EFDN 作为 baseline。

### 1.3 NTIRE 2026（第十一届，CVPRW 2026）
- **报告**：*The Eleventh NTIRE 2026 Efficient Super-Resolution Challenge Report* — [arXiv:2604.03198](https://arxiv.org/abs/2604.03198)（DOI: 10.48550/arXiv.2604.03198）；官方 repo：[Amazingren/NTIRE2026_ESR](https://github.com/Amazingren/NTIRE2026_ESR)；挑战页：[ntire-sr.github.io/2026](https://ntire-sr.github.io/2026/)
- **设置**：95 队注册、15 队有效提交；目标在 DIV2K_LSDIR_valid 保持 ~26.90 dB / test 26.99 dB 的同时压缩 runtime/params/FLOPs。
- **获胜方法**：报告摘要仅给出总体框架描述（轻量架构 + 部署级工程优化的组合）；作者列表含 Xiaomi（Hongyuan Yu 等，SPAN 原班人马）与多个高校团队。相关 2026 年新方案如 **IAFMNet**（information-aware feature modulation，CVPR 2026 论文笔记出现）**[获胜团队/方法名细节未核验，需读报告正文]**。
- **旁证趋势**：AIM 2025（ICCV 2025 workshop）设立 *Efficient Perceptual Image SR* 基准（[CVF](https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Longarela_Efficient_Perceptual_Image_Super_Resolution_AIM_2025_Study_and_Benchmark_ICCVW_2025_paper.pdf)），SRC-B（三星）表现突出——高效 SR 竞赛开始从 fidelity 转向感知质量 + 效率双目标。

**Runtime 赛道三年趋势总结**：① 骨干收敛到 SPAN 类"极浅 CNN + 参数无关注意力"；② 训练侧堆料（重参数化、蒸馏、LoRA 合并、渐进训练），推理侧折叠为 plain conv；③ 比拼从架构创新转向工程化（TensorRT/half precision、算子融合）；④ 2025–2026 增量趋小，PSNR 阈值固定下 runtime 已压至个位数 ms（540p→4K 级输入）。

---

## 2. 重参数化路线（ECBSR → RepSR → SPAN → PLKSR 之后）

| 工作 | 时间/venue | 要点 | 链接 |
|---|---|---|---|
| **SPAN** | CVPRW 2024 | 参数无关注意力，NTIRE 2024 冠军骨干 | [arXiv:2311.12770](https://arxiv.org/abs/2311.12770) / [GitHub](https://github.com/hongyuanyu/SPAN) |
| **PLKSR** | IEEE Access 2024 | 部分大核 CNN | [GitHub](https://github.com/dslisleedh/PLKSR) |
| **PlainUSR** | ACCV 2024 | 追求纯 plain conv 快速推理，含重参数化局部注意力 | [CVF PDF](https://openaccess.thecvf.com/content/ACCV2024/papers/Wang_PlainUSR_Chasing_Faster_ConvNet_for_Efficient_Super-Resolution_ACCV_2024_paper.pdf) |
| **ESPAN (Expanded SPAN)** | CVPRW/NTIRE 2025 | GRep 通用重参数化 + 自蒸馏 + 渐进训练 | [CVF](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Wang_Expanded_SPAN_for_Efficient_Super-Resolution_CVPRW_2025_paper.pdf) |
| **DSCLoRA** | CVPRW 2025（NTIRE 2025 冠军） | LoRA 低秩分支合并进 SPAN 卷积 = "重参数化的低秩版"，蒸馏增强，零额外推理成本 | [arXiv:2504.11271](https://arxiv.org/abs/2504.11271) |
| Re-parameterized kernel recalibration | KBS 2025 | 重参数化核再校准的轻量 SR | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705125009220) |
| 结构重参数化 + 特征重用网络 | Sensors 2025 | LR-Layer/FS/LFS Block，多算子融合为单算子 | [MDPI](https://www.mdpi.com/1424-8220/25/19/5989) |
| **REPVSR** | 2025 (SciTePress) | 视频 SR 的结构重参数化 | [DOI](https://www.scitepress.org/Link.aspx?doi=10.5220/0013186900003912) |

**判断**：纯重参数化作为独立"新架构"的论文创新空间已很小；2025–2026 的实际演化方向是 **重参数化 + 蒸馏 + 低秩适配（LoRA-merge）的训练配方化**，以及与 Mamba/token 选择等其他机制的组合。真实 wall-clock 收益仍然可靠（合并后为 plain conv，GPU/NPU 友好）。

---

## 3. LUT 路线（MuLUT → AutoLUT / IM-LUT 之后）【重点】

### 3.1 纯 LUT 主线时间轴
| 工作 | venue | 要点 | 链接 |
|---|---|---|---|
| SR-LUT | CVPR 2021 | 开山：网络转 LUT | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Jo_Practical_Single-Image_Super-Resolution_Using_Look-Up_Table_CVPR_2021_paper.html) |
| SPLUT | ECCV 2022 | 串/并联级联多 LUT | [arXiv:2207.12987](https://arxiv.org/abs/2207.12987) |
| MuLUT / DNN-of-LUTs | ECCV 2022 / TPAMI 2024 | 多 LUT 协作 | [项目页](https://mulut.pages.dev/), [arXiv:2303.14506](https://arxiv.org/abs/2303.14506) |
| RC-LUT | ICCV 2023 | 重构卷积模块 LUT，扩大感受野 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Reconstructed_Convolution_Module_Based_Look-Up_Tables_for_Efficient_Image_Super-Resolution_ICCV_2023_paper.html) |
| EC-LUT (Expanded-CNN LUT) | AAAI 2024 | 扩展卷积网络导出 LUT | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28495) |
| SPF-LUT + DFC 压缩 | CVPR 2024 | LUT 压缩用于通用复原 | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Look-Up_Table_Compression_for_Efficient_Image_Restoration_CVPR_2024_paper.pdf) |
| **HKLUT** | IJCAI 2024 | 百 KB 级 LUT，可入片上 cache | [IJCAI](https://www.ijcai.org/proceedings/2024/95), [arXiv:2312.06101](https://arxiv.org/abs/2312.06101) |
| **TinyLUT** | NeurIPS 2024 | 可分离映射 + 动态离散化，存储为 MuLUT 的 4.1%，树莓派4B 上比 FSRCNN 快 5× | [NeurIPS](https://papers.nips.cc/paper_files/paper/2024/hash/9b01c4a7d3fc49875dad3c13848bcd9e-Abstract-Conference.html), [代码](https://github.com/Jonas-KD/TinyLUT) |
| **AutoLUT** | CVPR 2025 | 自动采样点学习 + 自适应残差 | [arXiv:2503.01565](https://arxiv.org/abs/2503.01565), [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_AutoLUT_LUT-Based_Image_Super-Resolution_with_Automatic_Sampling_and_Adaptive_Residual_CVPR_2025_paper.pdf) |
| **IM-LUT** | ICCV 2025 | 学习混合多种插值函数的 LUT，任意尺度 | [arXiv:2507.09923](https://arxiv.org/abs/2507.09923), [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_IM-LUT_Interpolation_Mixing_Look-Up_Tables_for_Image_Super-Resolution_ICCV_2025_paper.pdf) |
| **DnLUT** | CVPR 2025 | 通道感知 LUT 彩色去噪：~500KB、DnCNN 0.1% 能耗、20× 加速 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_DnLUT_Ultra-Efficient_Color_Image_Denoising_via_Channel-Aware_Lookup_Tables_CVPR_2025_paper.html) |
| Anisotropic Pooling for LUT-realizable CNN restoration | arXiv 2025 | 让 CNN 复原"可 LUT 化" | [arXiv:2510.21437](https://arxiv.org/abs/2510.21437) |
| **IQ-LUT** | arXiv 2026-04 | 插值+量化整合进单输入多输出 ECNN + 残差学习 + KD 引导非均匀量化，比 ECNN 存储降 50× | [arXiv:2604.07000](https://arxiv.org/abs/2604.07000) |
| **ISRLUT** | ACM (TODAES/TECS 系) 2025/2026 | 整数化 FHD SR：神经 LUT + 近存计算（硬件向） | [DOI:10.1145/3770759](https://doi.org/10.1145/3770759) |

### 3.2 LUT × 神经网络 混合/级联（专项检索："LUT as fast path"、"hybrid LUT network cascade"）
**结论：存在"LUT+conv 混合"工作，但均为特征级/模块级混合或时序级复用，没有发现"LUT 作为 patch 级快路径 + CNN 作为按需精修路径"的显式级联路由工作。**

已核验的最近邻：
1. **Online Streaming VSR with Convolutional LUT**（[arXiv:2303.00334](https://arxiv.org/abs/2303.00334)，TIP 2024, vol.33, pp.2305–2317，[PubMed](https://pubmed.ncbi.nlm.nih.gov/38470585/)）— **最接近的混合工作**：为在线流式视频 SR 提出卷积-LUT 混合模型改善延迟/质量折中。但它是**模块级混合**（同一前向内 conv 与 LUT 协同），非难度路由级联。
2. **Online Video Streaming SR with Adaptive LUT Fusion** [未核验]（ResearchGate 条目，2023）— LUT 融合方向的同组衍生。
3. **Online Video Quality Enhancement with Spatial-Temporal LUT** [未核验]（ResearchGate 2024）。
4. **Hybrid Mamba + sparse LUT** 水下增强（Neurocomputing 2025，[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231225001237)）— 特征级混合。
5. **VoLUT**（MLSys 2025，[PDF](https://zhan6841.github.io/assets/pdf/paper/volut-mlsys25.pdf)）— LUT 加速体积视频上采样的系统工作 [细节未核验]。
6. **AutoLUT 的 AdaRL**：LUT 内部的自适应残差连接——是"残差修正"思想，但仍在纯 LUT 框架内。
7. **PatternNet/角度**：SPLUT、MuLUT 的"级联"均指 **LUT 级联 LUT**（扩大感受野），不是 LUT→NN 的异构级联。

**明确的空白**：以「LUT 输出 + 便宜质量估计器判定 → 少数难 patch 走 CNN/Transformer 精修」为核心机制、并给出 wall-clock 端到端加速的图像 SR 论文，在本轮穷尽检索中**未发现**。检索式 "LUT fast path"、"hybrid LUT network cascade"、"LUT CNN cascade super-resolution" 均未命中直接匹配。

---

## 4. 大图（2K–8K）patch 级自适应方法（CAMixerSR / PCSR / ENAF 之后）

| 工作 | venue | 粒度/机制 | 链接 |
|---|---|---|---|
| ClassSR（背景） | CVPR 2021 | patch 难度分类 → 三档子网 | [CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf) |
| ARM（背景） | ECCV 2022 | supernet 子网 + **Edge-to-PSNR 查找表**选子网 | [arXiv:2203.10812](https://arxiv.org/abs/2203.10812) |
| CABM | CVPR 2023 | patch 级内容感知位宽 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Tian_CABM_Content-Aware_Bit_Mapping_for_Single_Image_Super-Resolution_Network_With_CVPR_2023_paper.html) |
| CAMixerSR | CVPR 2024 | token 级内容感知路由（conv vs attention） | [arXiv:2402.19289](https://arxiv.org/abs/2402.19289) |
| PCSR | ECCV 2024 | **像素级**难度分类 → 不同容量上采样器 | [arXiv:2407.21448](https://arxiv.org/abs/2407.21448), [ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00370.pdf) |
| ENAF | WACV 2025 | 多出口网络 + **tiny PSNR 估计器** + 自适应 patch 融合 | [CVF PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Nguyen_ENAF_A_Multi-Exit_Network_with_an_Adaptive_Patch_Fusion_for_WACV_2025_paper.pdf) |
| **PatchScaler** | ICCV 2025 | patch 独立扩散 SR：patch 难度感知的 Patch-Independent Reverse Process | [CVF PDF](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Liu_PatchScaler_An_Efficient_Patch-Independent_Diffusion_Model_for_Image_Super-Resolution_ICCV_2025_paper.pdf) |
| PTSR | PRL 2025-09 | patch 翻译器模型 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167865525001503) |
| **Pure-Pass** | arXiv 2025-10 | 自适应掩码的动态 token-mixing 路由，声称超 CAMixer 类基线 | [arXiv:2510.01997](https://arxiv.org/abs/2510.01997) |
| **E2L-CAMixerSR** | ICT Express 2026 | CAMixerSR 直接后继：early-to-late 组摘要融合 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405959526001062) |
| 自适应 token 选择轻量 SR | KBS 2026 | token 选择 + 特征增强，引 CAMixerSR | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705125020933) |
| ESSR 加速器 | arXiv 2025 | 8K@30FPS SR 硬件加速器 | [arXiv:2503.20245](https://arxiv.org/abs/2503.20245) |

**趋势**：2025–2026 的 patch 自适应工作向两个极端分化——(a) 更细粒度（像素级 PCSR、token 级 Pure-Pass）；(b) 转入扩散模型（PatchScaler）。**在"patch 路由 + 异构模型池 + wall-clock 验证"这个 ClassSR 式经典框架上，2025–2026 学术界几乎没有新的顶会工作**（多为期刊小改进），这是可利用的空档。

---

## 5. 任意尺度 / 成本-质量可控 SR

- **AnySR**：*Realizing Image Super-Resolution as Any-Scale, Any-Resource*（[arXiv:2407.04241](https://arxiv.org/abs/2407.04241)，TIP 2025 [期刊版未核验]）— any-scale + any-resource，子网可伸缩。
- **Test-Time Cost-and-Quality Controllable Arbitrary-Scale SR with Variable Fourier Components**（AAAI 2025，[AAAI 页](https://ojs.aaai.org/index.php/AAAI/article/view/32369) [venue 细节未核验]）与其扩展 **Efficient Cost-and-Quality Controllable ASSR with Fourier Constraints**（[arXiv:2510.23978](https://arxiv.org/abs/2510.23978)）— 测试时逐级预测傅里叶分量，随算量增加质量单调提升。
- **IM-LUT**（ICCV 2025，见 §3）也支持任意尺度（插值混合天然 scale-agnostic）。
- ARM（ECCV 2022）仍是 any-time SR 的经典参照。

**趋势**：可控性研究从"训练多个模型"转向"单模型测试时连续可控"（Fourier 分量、子网切换、token 丢弃率）。但这些工作的"控制"是**开环**的（用户给预算），**没有闭环质量保证**（不检查输出实际质量后决定是否追加计算）。

---

## 6. 【核心查新】「patch 级难度路由 + 轻量快路径 + 重模型精修 + 质量保证」组合

### 6.1 最近邻工作逐项分析

| 工作 | patch 路由 | 轻/重异构级联 | 质量保证机制 | 空白 |
|---|---|---|---|---|
| **ClassSR** (CVPR 2021) | ✅ 难度分类 | ✅ 三档子网（同架构不同宽度） | ❌ 分类器 argmax，无质量校验，误分类无兜底 | 无回退；子网同源非异构；无 wall-clock 优先设计 |
| **ARM** (ECCV 2022) | ✅ edge score | ✅ 权重共享子网 | ⚠️ Edge-to-PSNR LUT 是**先验预测**，非输出后验证 | 预测错误无纠正；子网共享权重，非独立快/慢模型 |
| **PCSR** (ECCV 2024) | 像素级 | ✅ 多容量上采样头 | ❌ 无 | 无质量兜底 |
| **ENAF** (WACV 2025) | ✅ patch | ✅ 多出口（同网深浅） | ⚠️ tiny PSNR 估计器决定出口——**最接近"质量感知路由"** | 单网多出口而非独立轻/重模型；估计器仍是前置预测，无"输出不达标→升级重算"闭环；无形式化保证 |
| **PatchScaler** (ICCV 2025) | ✅ patch 难度→扩散步数 | ⚠️ 同一扩散模型变步数 | ❌ | 扩散范畴，非 fidelity SR |
| **MobiSR** (MobiCom 2019) | ✅ 难度感知 patch 调度 | ✅ 双模型（轻/重）映射到异构处理器 | ⚠️ total-variation 难度代理，满足质量目标为软约束 | 系统侧调度，无学习型路由，无输出后验证；2019 年后该线沉寂 |
| **NEMO** (MobiCom 2020) | anchor 帧选择（视频） | ❌ | ✅ **提供相对逐帧 SR 的质量退化保证**（[KAIST](https://ina.kaist.ac.kr/projects/nemo/)） | 视频流媒体缓存场景，非单图 patch 级联 |
| **Palantir** (MMSys 2025 / [arXiv:2408.06152](https://arxiv.org/abs/2408.06152)) | ✅ **patch 级 anchor 调度**，DAG 质量估计 | ❌（SR 模型单一，决定"做/不做 SR"） | ⚠️ 轻量质量估计驱动调度，"不牺牲质量"为经验结论 | 是"SR vs 复用"二选一，不是轻/重 SR 模型级联；直播系统绑定编码器信息 |
| **AdaDSR** (2020) | 像素级深度预测 | 同网变深 | ❌ | — |
| **Saliency-aware dynamic routing**（遥感，[arXiv:2210.07598](https://arxiv.org/abs/2210.07598)） | ✅ | ✅ | ❌ | 领域受限 |
| 推测执行式 draft-verify（LLM 领域） | — | ✅ draft→verify 范式成熟 | ✅ 接受/拒绝机制 | **该范式尚未被移植到 SR patch 级**（专项检索未命中） |

### 6.2 新颖性结论

**主流路线（2025–2026 高效 SR 三大主线）**：
1. **训练配方化的极轻 CNN**（SPAN 系 + 重参数化 + 蒸馏 + LoRA-merge）— NTIRE runtime 赛道统治者，真实 wall-clock 最硬；
2. **LUT 家族**（TinyLUT/AutoLUT/IM-LUT/IQ-LUT/DnLUT）— 边缘设备、存储与能耗极限；
3. **内容自适应稀疏计算**（CAMixerSR→Pure-Pass 的 token 路由、PCSR 像素路由、Mamba 骨干）— 大图场景。

**「patch 级级联 + 质量保证」剩余新颖性评估**：
- **已被占据的部分**：patch 难度路由（ClassSR/ARM/MobiSR）、多出口+PSNR 估计（ENAF）、patch 级质量估计调度（Palantir）、退化保证概念（NEMO，视频）。单独任何一项都不新。
- **仍然空白的组合**（本轮穷尽检索未发现占据者）：
  1. **异构级联**：以真正独立的快路径（尤其是 **LUT** 或 SPAN-tiny）+ 独立重模型（Transformer/大 CNN）构成级联，而非同网子网/多出口；
  2. **输出后验证的闭环质量保证**：先跑快路径，用便宜的质量估计器/代理检查**实际输出**，不达标的 patch 才升级重算（speculative-execution / draft-verify 语义）——SR 领域无人做；
  3. **统计意义上的质量保证**（如 conformal 风险控制：保证 ≥1−α 比例 patch 的 PSNR 损失 ≤ ε）——完全空白；
  4. LUT-as-fast-path 与上述任一结合——完全空白。
- **风险最大的三个最近邻**（写论文必须重点区分）：**ENAF**（质量估计器+多出口）、**ARM**（Edge-to-PSNR 表）、**Palantir**（patch 级质量估计调度，系统会议）。与它们的差异点应落在：独立异构模型级联、后验校验而非前验预测、形式化/统计质量保证、单图 wall-clock。
- **新颖性余量估计**：组合级新颖性 **中等偏高**；机制级（后验验证 + 统计保证）新颖性 **高**。需在相关工作中诚实覆盖 ClassSR/ARM/ENAF/PCSR/MobiSR/NEMO/Palantir。

---

## 7. 公开可用的轻量 SR 预训练模型（快路径候选）

| 模型 | 参数量 (×4) | 公开报告 PSNR (×4) | 权重来源 | 备注 |
|---|---|---|---|---|
| **SPAN** | ~498K（挑战版 SPAN-tiny 更小）[参数量未核验精确值] | 挑战 DIV2K 上 ~27.09 dB（NTIRE 设定）；Set5 ≈32.2 dB [Set5 值未核验] | [github.com/hongyuanyu/SPAN](https://github.com/hongyuanyu/SPAN)（repo 含模型） | NTIRE 2024 冠军骨干，spandrel 已收录 |
| **RLFN** | 543K | Set5 32.24 / Set14 28.62 / B100 27.60 / Urban100 26.17（[来源表](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chen_Large_Kernel_Frequency-enhanced_Network_for_Efficient_Single_Image_Super-Resolution_CVPRW_2024_paper.pdf)） | [github.com/bytedance/RLFN [未核验链接]](https://github.com/bytedance/RLFN)；论文 [arXiv:2205.07514](https://arxiv.org/abs/2205.07514) | NTIRE 2022 冠军 |
| **PLKSR / PLKSR-tiny** | ~7.4M / tiny 更小 [精确值未核验] | 论文报告优于同级；DF2K 预训练已放出（2024-05-22） | [github.com/dslisleedh/PLKSR](https://github.com/dslisleedh/PLKSR)（`pretrained_models`） | IEEE Access 2024 |
| **ECBSR** (M4C16) | ~10–52K（变体） | Set5 31.04–31.92 / Urban100 24.79–25.81（变体差异，[综述表](https://guangweigao.github.io/paper/ACM-CSUR-Survey.pdf)） | [github.com/xindongzhang/ECBSR [未核验链接]](https://github.com/xindongzhang/ECBSR) | 移动端重参数化经典 |
| **SAFMN** | 240K（轻量版）/ 5.60M（大版，Set5 32.65） | [ICCV 2023 supp](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Sun_Spatially-Adaptive_Feature_Modulation_ICCV_2023_supplemental.pdf) | [github.com/sunny2109/SAFMN [未核验链接]](https://github.com/sunny2109/SAFMN) | ICCV 2023 |
| **TinyLUT** | LUT 存储 ~百 KB 级（非参数量口径） | 竞争级（较 MuLUT 精度持平/略升，存储 4.1%） | [github.com/Jonas-KD/TinyLUT](https://github.com/Jonas-KD/TinyLUT) | 树莓派实测 5×+ 快于 FSRCNN，**LUT 快路径首选** |
| **AutoLUT** | LUT 级 | 优于 MuLUT/SPF-LUT（CVPR 2025 表） | 论文称代码公开 [仓库链接未核验] | CVPR 2025 |
| **DSCLoRA (SPAN 微调)** | 与 SPAN 同量级 | NTIRE 2025 冠军指标 | [HF paper 页](https://huggingface.co/papers/2504.11271)，NTIRE2025_ESR repo 含各队模型 | 可合并权重 |
| **EFDN** | ~276K [未核验] | NTIRE baseline 级 | NTIRE2024/2025_ESR 官方 repo 内含 baseline 权重 | 官方挑战 baseline，取权重最方便 |

> 实用提示：`NTIRE2024_ESR` / `NTIRE2025_ESR` / `NTIRE2026_ESR` 官方 GitHub 仓库集中收录了**所有参赛队的模型定义与权重**，是级联快路径候选一站式来源。另 [spandrel](https://chainner.app/spandrel/) 库统一封装了 SPAN 等社区权重加载。

---

## 8. 必答问题小结

**Q1 2025–2026 主流技术路线**：见 §6.2 三大主线（配方化极轻 CNN / LUT 家族 / 内容自适应稀疏计算），另加感知效率赛道（AIM 2025）与 Mamba 骨干作为第四支线。

**Q2 「patch 级级联 + 质量保证」新颖性还剩多少**：粗粒度组合已被 ClassSR/ARM/ENAF/MobiSR/Palantir 分别占据局部；**"独立异构快慢模型 + 输出后验校验 + 统计质量保证（±LUT 快路径）"的完整组合无人占据**。最近邻 = ENAF、ARM、Palantir、PCSR、MobiSR、NEMO。新颖性余量：机制级高、组合级中高，但相关工作必须严格切割。

**Q3 可用轻量预训练模型**：见 §7 表；首选快路径候选 = TinyLUT（极端低延迟）、SPAN/DSCLoRA（GPU 最优质量-速度）、ECBSR（移动端）、EFDN（基线对照）。

---

## 附：残留不确定项
- NTIRE 2024 runtime 子赛道第一名（SPANF vs MLP-SR）两处来源表述不一致，需读报告 Table。
- NTIRE 2026 获胜团队与方法名未从报告正文提取（仅摘要可得）。
- 表中标 [未核验] 的 GitHub 链接为业界周知仓库名，但本轮未逐一打开验证。
- 2026 年 arXiv 编号（2604.xxxxx）按检索返回原样记录。
