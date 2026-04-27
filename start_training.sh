#!/bin/bash
# DART-SR 训练启动脚本

echo "=========================================="
echo "DART-SR 训练准备检查"
echo "=========================================="

# 1. 检查数据集
echo ""
echo "1. 检查数据集..."
if [ ! -d "datasets/DIV2K" ]; then
    echo "❌ 数据集不存在: datasets/DIV2K"
    echo ""
    echo "请下载 DIV2K 数据集："
    echo "  - 训练集 HR: https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    echo "  - 训练集 LR (X4): https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"
    echo ""
    echo "解压后放到："
    echo "  datasets/DIV2K/HR/"
    echo "  datasets/DIV2K/LR_bicubic/X4/"
    echo ""
else
    echo "✓ 数据集目录存在"
fi

# 2. 检查预训练模型
echo ""
echo "2. 检查预训练模型..."
if [ ! -f "experiments/pre_trained_models/RCAN_BIX4.pth" ]; then
    echo "❌ 预训练模型不存在: experiments/pre_trained_models/RCAN_BIX4.pth"
    echo ""
    echo "请下载 RCAN 预训练模型："
    echo "  选项 1: ClassSR 提供的模型"
    echo "    https://drive.google.com/drive/folders/1jzAFazbaGxHb-xL4vmxc-hHbR1J-uek_"
    echo "    下载 RCAN_branch1.pth 并重命名为 RCAN_BIX4.pth"
    echo ""
    echo "  选项 2: 原始 RCAN 模型"
    echo "    https://github.com/yulunzhang/RCAN"
    echo ""
else
    echo "✓ 预训练模型存在"
fi

# 3. 检查配置文件
echo ""
echo "3. 检查配置文件..."
if [ ! -f "codes/options/train/train_DARTSR_RCAN_local.yml" ]; then
    echo "❌ 配置文件不存在"
else
    echo "✓ 配置文件存在"
    echo ""
    echo "⚠️  请编辑配置文件，修改以下路径："
    echo "  codes/options/train/train_DARTSR_RCAN_local.yml"
    echo ""
    echo "  需要修改的字段："
    echo "    - dataroot_GT: 训练集 HR 路径"
    echo "    - dataroot_LQ: 训练集 LR 路径"
    echo "    - val dataroot_GT: 验证集 HR 路径"
    echo "    - val dataroot_LQ: 验证集 LR 路径"
    echo "    - pretrain_model_G: 预训练模型路径"
fi

# 4. 检查 GPU
echo ""
echo "4. 检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "⚠️  未检测到 nvidia-smi，请确认 CUDA 环境"
fi

echo ""
echo "=========================================="
echo "准备完成后，运行以下命令开始训练："
echo "=========================================="
echo ""
echo "cd codes"
echo "python train.py -opt options/train/train_DARTSR_RCAN_local.yml"
echo ""
echo "或使用多卡训练（如果有多张 GPU）："
echo "python -m torch.distributed.launch --nproc_per_node=2 train.py -opt options/train/train_DARTSR_RCAN_local.yml --launcher pytorch"
echo ""
