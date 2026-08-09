# Featurize 启动入口 — Goal 20260809-132635

本 Goal 运行 canonical EDSR-L 32→24 静态 Student 的 ×2/×3/×4、三 seed、每组 500-step 固定预算恢复。它不是旧的动态 RCAN `train_cloud.sh`，也不是 120k-step 重训。

## 云端预期资产

脚本只检查、不下载数据集。Featurize 镜像必须已有：

```text
/home/featurize/data/DF2K/DF2K_train_HR_sub
/home/featurize/data/DF2K/DF2K_train_LR_bicubic/X2_sub
/home/featurize/data/DF2K/DF2K_train_LR_bicubic/X3_sub
/home/featurize/data/DF2K/DF2K_train_LR_bicubic/X4_sub
/home/featurize/data/DIV2K_valid_HR
/home/featurize/data/DIV2K_valid_LR_bicubic/X2|X3|X4
/home/featurize/data/SRBenchmarks/{Set5,Set14,BSD100,Urban100,Manga109}
```

官方 EDSR 权重若缺失，脚本会续传下载并检查冻结的 bytes 与完整 SHA-256。

## 启动

在 RTX 4090 实例中：

```bash
cd /home/featurize/work/B2R-SR
git pull --ff-only origin main
git status --short

nohup env RELEASE=1 NOTIFY=1 \
  bash results/autonomous_goals/20260809-132635/executed_source/run_featurize.sh \
  > /home/featurize/work/edsr_20260809-132635_bootstrap.log 2>&1 &
```

查看日志：

```bash
tail -f /home/featurize/work/B2R-SR/experiments/EDSR_static_depth_20260809-132635/launcher.log
```

训练checkpoint、逐步loss、最终测试指标和完整日志首先写入仓库内（该目录被Git忽略）：

```text
/home/featurize/work/B2R-SR/experiments/EDSR_static_depth_20260809-132635/
```

成功或终止失败后，再在 `/home/featurize/work/b2rsr_exports/` 生成并验证一个 `.tar` 与同名 `.sha256`，因此仓库实验目录和独立导出归档两处都有记录。归档验证后才执行 `featurize instance release`。若状态文件或带内部哈希清单的结构化归档无法验证，脚本会保留实例并发出计费警告，避免归还前丢失证据。只想调试而不自动归还时，必须在启动前显式设 `RELEASE=0`；此时实例会继续计费。

脚本可重复执行：已完成且哈希一致的 `{scale,seed}` 会跳过；未完成运行从原子 `resume.pt` 恢复。任何不一致的旧产物会导致停止，而不是覆盖。

## 独立测试单个checkpoint

这与训练后的九任务批量汇总分开。可指定一个训练好的EDSR checkpoint、尺度和测试集：

```bash
python scripts/eval/test_edsr_checkpoint.py \
  --checkpoint experiments/EDSR_static_depth_20260809-132635/training/x4/seed0/student_final.pt \
  --scale 4 \
  --dataset Set5 \
  --data-root /home/featurize/data \
  --experiment-dir experiments/EDSR_d24_x4_seed0
```

输出在`experiments/EDSR_d24_x4_seed0/test/X4/Set5/`：`summary.json`、`test.log`、逐图`per_image_metrics.csv`与JSONL。需要保存SR图片时附加`--save-images`。自定义测试集可改用成对的`--hr-dir`和`--lr-dir`。
