#!/bin/bash
# 下载 RCAN 预训练模型的脚本

mkdir -p experiments/pre_trained_models

# 你需要从以下来源之一获取 RCAN 预训练模型：
# 1. 原始 RCAN 论文的官方模型
# 2. ClassSR 项目提供的 branch1 模型
# 3. 自己训练的 RCAN 模型

echo "请手动下载 RCAN 预训练模型到："
echo "  experiments/pre_trained_models/RCAN_BIX4.pth"
echo ""
echo "可能的下载源："
echo "  - https://github.com/yulunzhang/RCAN"
echo "  - https://github.com/Lornatang/RCAN-PyTorch"
