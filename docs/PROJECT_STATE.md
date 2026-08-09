# B2R-SR 当前项目状态（新会话唯一交接上下文）

> 更新：2026-08-09
> 状态：EDSR-L有界可行性Goal `20260809-112351`仍按冻结协议永久结束为STOP；用户已另行批准新Goal `20260809-132635`，用于三尺度500-step恢复与Featurize一键执行包。
> 规则：先读根目录`AGENTS.md`和新Goal；新Goal不得修改、续跑或反向改写旧Goal事实。

## 1. 当前主线

B2R-SR 已从动态空间路由转向：

> **将已有高质量SR checkpoint转换为单一静态、零部署附件、质量损失可控且真实硬件更快的Student。**

动态patch/window路由、深度折叠、recoverability proxy和segment/function-transfer蒸馏均已由实验关闭。不要复活这些路线，也不要继续堆叠新的蒸馏loss。

F2/F1、PConv、宽度和非均匀深度matched-latency实验均已完成，未产生可推广的新结构算法。当前论文不再围绕“候选失败/拒绝”组织，而定位为：

> **Hardware-Calibrated Static Depth Transfer for Pretrained Super-Resolution Networks**

论文方法是架构感知的物理减深、顺序权重移植、固定预算Teacher-guided recovery和质量—真实延迟Pareto选择。RCAN仍是已验证锚点。canonical EDSR-L（32→24 blocks，×2/×3/×4）的Autonomous Goal `20260809-112351`已完成：三尺度checkpoint对齐、静态构造和RTX 4060无训练延迟均通过，×4 d24 p50 speedup为`1.217595×`；但100步Pilot的冻结loss稳定性条件失败，故旧Goal最终决策为**STOP**。

用户随后显式批准新Goal `20260809-132635`。新Goal前瞻性取消不匹配随机crop的首尾loss Gate，改为仅在provenance/data/static-artifact/integrity失败、NaN/Inf或OOM时停止；固定运行×2/×3/×4、seeds 0/1/2、每组500 FP32 updates，并在全部final-step checkpoint冻结后统一评测。该授权不改变旧Goal的STOP记录，也不是120k-step长训练授权。

## 2. 已冻结的实验事实

### 动态路由

- 8×8窗口all-keep相对dense RCAN平均约−1.2406 dB；
- RTX 3060筛查speedup为0.844×，即慢于dense；
- 原DART-SR动态空间路由主线：**STOP**。

### RCAN静态物理裁薄

- Teacher：10 residual groups × 20 RCAB；
- Student：10 × 15 RCAB；
- 参数量：约15.59M→11.87M，减少约24%；
- DIV2K validation gap：0.168006 dB；
- RTX 4060 p50 speedup：1.293320×；
- 该速度约为20→15深度变化FLOPs理想上限（约1.33×）的97%；
- state_dict、部署图和零附件Gate：PASS。

### MSRResNet静态裁薄

- 16→12 residual blocks；
- DIV2K validation gap：0.238542 dB；
- RTX 4060 p50 speedup：1.115648×；
- 原1.15×速度Gate：FAIL。

### 质量恢复方法

- Recoverability-aware选择：RCAN/MSR相对uniform仅约+0.017210/+0.014216 dB，STOP；
- TASD最佳early gain均值约+0.011542 dB且随训练缩小，0 arms promoted，STOP；
- Common-input segment composition：0/4 seed×monitor cells通过，12/12比较低于+0.01 dB；正序不胜逆序/right-anchor，STOP；
- Alignment/function-transfer Research Direction：STOP；Paper Story：STOP；
- Checkpoint-to-deployment Framework：PIVOT/继续。

## 3. F2/F1之后的新判断

系统调研：`results/autonomous_goals/20260802-001700/final_report.md`；后续实验：`results/autonomous_goals/20260803-004155/final_report.md`与`results/autonomous_goals/20260803-215622/final_report.md`。

冻结结论：

- 参数/MACs和同设备真实延迟必须分开报告；
- 全200-site PConv RCAN比已接受的15-block Student慢，未训练；
- width-48虽然接近部分延迟目标，但两seed质量损失均超过1.23 dB；
- 最佳Fisher非均匀深度候选仍比均匀15-block控制点低约0.119 dB；
- cheap-block、宽度和非均匀深度均未promotion，不写成论文核心算法；
- 可防守贡献是从预训练checkpoint到零附件静态Student的转换与实测部署闭环，而非宣称剪枝、block评分或KD本身新颖。

## 4. RCAN为什么仍是首个实验锚点

RCAN不是唯一允许的模型，也不能代表全部SR骨干。它先用于主实验，是因为：

- 它是经典的大型residual-in-residual CNN SR模型；
- 200个规则RCAB适合做受控block替换；
- 已有完整Teacher、数据、训练、延迟和20→15控制点；
- F1需要与这个已知实测延迟点做因果清晰的matched-latency比较。

定向调研曾将canonical EDSR-L列为第二骨干首选：32个同形、无BN、residual scale=0.1的宽Residual Blocks，官方有×2/×3/×4 checkpoint，适合32→24物理减深且与RCAN形成flat/wide与nested/attention的结构互补。`20260809-112351`验证了其严格导入、物理静态Student和目标硬件延迟杠杆，但冻结Pilot稳定性Gate未通过；不得只引用正向延迟/诊断结果而忽略旧Goal的最终STOP。新Goal `20260809-132635`是用户另行批准的前瞻性重审，只运行固定d24三尺度恢复，不扩新骨干、d28 recovery、Transformer/SSM或新loss。MSRResNet仍只是历史低成本后备；CARN和RDN因递归/级联或dense fusion依赖不作为本轮减深骨干；RLFN/SPAN仅作现代高效SR参考基线。

## 5. 当前论文证据标准

- 主结果必须覆盖×2/×3/×4及Set5、Set14、BSD100、Urban100、Manga109的PSNR/SSIM；
- 至少报告RCAN与EDSR-L两个骨干，若EDSR可行性不足则收窄论文适用范围，不伪称任意Backbone通用；
- 参数、MACs/FLOPs和p50/p95真实延迟分表报告；Teacher、Student和对照使用同一计时栈；
- 需要深度—质量—延迟Pareto、固定恢复预算、多seed和置信区间；
- 论文正文呈现支持的方法与trade-off，不使用内部FAIL/Reject语言组织主故事；
- 在模型、数据、seed和协议冻结前不得打开最终benchmark。

## 6. 下一会话允许与禁止

### 当前新Goal允许

- `20260809-132635`允许准备、review、commit并push Featurize执行包；由用户启动云实例后，执行冻结的×2/×3/×4、三seed、每组500-step恢复与冻结后质量评测；
- 不得在`20260809-112351`下追加更新、续训或重跑Pilot；
- RTX 4090只记录训练吞吐和质量，正式Student延迟随后回到统一RTX 4060栈重测；
- 如需500步以外长训练、d28恢复或新骨干，仍需新的明确批准与冻结协议；
- 按Abstract、Introduction、Related Work、Method、Experiments、Conclusion顺序重写论文；
- 在`/Users/admin/Workspace/Research/DART-SR-Project/paper/figures/`维护draw.io/SVG/PDF论文图源。

### 继续禁止

- 使用未完成任务的中间结果；
- 重新包装TASD/segment KD为原创；
- 推理期patch/pixel routing；
- 用FLOPs代替真实延迟，或混用4060与4090延迟；
- 在模型冻结前打开最终benchmarks；
- 同时扩展INT8、Transformer、SSM或新loss；
- 未批准即长训练、租卡、commit、push、删除实验结果或覆盖用户修改。

## 7. 硬件与目录约定

- Mac仓库：`/Users/admin/Workspace/Research/DART-SR-Project/code/B2R-SR`，负责编辑与编排；本地Python不可假定有Torch/CUDA。
- RTX 4060 Laptop：`ssh 4060`；远端仓库`/home/jww/WorkSpace/Research/B2R-SR`；Python`/home/jww/miniconda3/envs/b2rsr/bin/python`；`/usr/lib/wsl/lib/nvidia-smi`。用于免费筛查、短pilot和目标延迟。
- Featurize RTX 4090：仓库与输出在`/home/featurize/work`，数据在`/home/featurize/data`；当前入口见`results/autonomous_goals/20260809-132635/START_HERE.md`。结果验证后用`featurize instance release`归还；最终延迟必须回到统一目标设备完整重测。
- 完整操作与产物放置规则见根目录`AGENTS.md`；SSH恢复指南见`docs/B2RSR_4060_WSL2_SSH_Setup_zh.md`。

## 8. 新会话建议读取顺序

1. `AGENTS.md`；
2. `results/autonomous_goals/20260809-132635/goal.md`与`protocol.json`；
3. `docs/PROJECT_STATE.md`（本文）；
4. `results/autonomous_goals/20260809-112351/final_report.md`（不可改写的EDSR-L有界可行性STOP）；
5. `results/autonomous_goals/20260803-215622/final_report.md`；
6. `results/autonomous_goals/20260803-004155/final_report.md`；
7. `results/autonomous_goals/20260802-001700/final_report.md`；
8. `results/autonomous_goals/20260801-110247/final_report.md`；
9. `docs/B2RSR_Advisor_Progress_Update_20260802_zh.md`。

旧叙事和旧计划已移至`docs/archive/pre_f1_20260802/`，仅供追溯，不得作为当前方案依据。实验结果、审计、原始调研和代码均未删除。
