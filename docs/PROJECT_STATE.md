# B2R-SR 项目状态快照（供新会话/压缩后恢复上下文）

> 更新：2026-07-29 深夜。本文件是唯一权威的"当前状态"，新会话应先读本文。
> 深度背景按需读：`B2RSR_CrossDomain_Acceleration_Survey_zh.md`（最新方向）、
> `B2RSR_Acceleration_Landscape_Survey_zh.md`、`B2RSR_v1_Gate_Diagnostic_Milestone_zh.md`。

## 一、大方向（已定）

- **大论文叙事**：论文1（已发表，重参数化模型级加速）→ 论文2（框架级：patch 级异构级联 + 质量保证）。"零件 → 系统"递进。
- **论文2 当前主线 = 方案 A「Draft-Verify SR」**：轻量快路径全图出草稿 → **后验**误差检测（非前验难度预测）→ 少数 patch 升级 RCAN 重算 → conformal 校准给统计质量保证。
- 依据：三份独立调研共同指向的空白——回归任务自适应计算的统计质量保证无先例；"输出后验校验闭环"SR 无人做；LUT-as-fast-path 空白。最危险近邻：ENAF(WACV25)/ARM/Palantír(MMSys25)，差异点=后验vs前验+统计保证+异构级联。
- 目标 venue：CV 会议（CCF-C 稳、B 冲）；系统会议已排除。

## 二、已证伪路线（不要重провер）

1. v1 空间 8×8 窗口路由：all-keep −1.24 dB、3060 上 0.844×（Gate 报告）；
2. 免训练前缀深度截断：d=9 即 −1.59 dB（悬崖）；非前缀 LOO 最佳单组 −0.108 dB 仅 1.11×；
3. CA gate 通道剪枝：跨图 std 0.019，无输入自适应信号；<0.1 门控通道仅 2%；
4. bicubic 作快路径的级联：修正 BGR bug 后 oracle 仅 1.27×（cheap 8.5%）FAIL；
5. 3060 上 fp16/channels_last 反而更慢（launch-bound），eager fp32 即诚实分母。

⚠️ 历史 bug 教训：run_cascade_oracle 曾喂 BGR 给需 RGB 的 RCAN（已修，commit 见 log）；
bicubic 曾用 numpy imresize_np 计延迟 57ms/patch（已改 cv2 0.5ms）。

## 三、已验证事实

- 手工前验特征（lap_var 等 6 个）AUC = **0.807**，combo 0.807；recall99% 仅 12.1% cheap → 1.32×；
- 图级质量预算（0.05dB）下 cheap 24% → 1.53×，但 worst-img loss 1.11dB（尾部失控 → conformal 动机）；
- V2-G0：深度截断 d=G 与 dense 逐 bit 相等（执行机制正确）；
- CARN-M 官方权重已内嵌加载器（OfficialCARNM in run_cascade_oracle.py），strict 加载验证过。

## 四、硬件与环境（全部就绪）

- **4060 笔记本（主力筛查机）**：`ssh 4060`（Mac ~/.ssh/config 已配，免密）。
  仓库 `/home/jww/WorkSpace/Research/B2R-SR`；conda env `b2rsr`（torch 2.3.0+cu121，
  用 `zsh -ilc "conda activate b2rsr && ..."` 调用）；数据 `~/data/SRBenchmarks/`
  （DIV2K_valid_2K 官方LR + Set5/14/BSD100/Urban100/Manga109 全套 X2/3/4）；
  checkpoint 已在仓库标准位置；check_env.py 9/9 PASS。
- 云端 Featurize 3060：历史延迟基线设备；`featurize instance release` 停止计费；
  云盘 /home/featurize/work 持久。4090 只在论文正式数字阶段租。
- 用户边界：**只在用户明确给出的目录内操作；改动大动作先征求同意；脚本写完先过目再跑**。

## 五、下一步（唯一进行中任务）

**SG-A kill-check**：写 `scripts/eval/analyze_posterior_signal.py`（写完给用户过目）：
- DIV2K_valid_2K，64px patch；快路径 CARN-M 出草稿；真值 RCAN；
- 对照：前验 6 特征（analyze_router_features.py 里有实现，AUC 0.807）；
- 实验组：后验信号（|draft−bicubic| 统计、draft 高频能量、局部残差方差）；
- 输出：AUC 对比 + conformal 阈值下 cheap%/speedup 推算。
- **预注册判据：后验 AUC ≥ 0.90 且显著 > 0.807 → 方案 A 立项；否则转方案 B（LUT 快路径，TinyLUT 有开源码）**。

## 六、可复用脚本索引（scripts/eval/）

- check_env.py：环境自检（9 项）
- run_depth_sweep.py：深度扫描诊断（已完成使命）
- run_cascade_oracle.py：级联 oracle（含 OfficialCARNM 加载器，BGR 已修）
- analyze_router_features.py：前验特征 AUC + 质量预算工作点（对照基线）
- generate_router_labels.py / train_patch_router.py：标签生成 + router 训练（阶段2用，
  标签需改为后验误差；含真实级联合成与端到端计时）
- run_and_release.sh / run_depth_sweep_and_shutdown.sh：云端自动归还包装
