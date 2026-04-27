#!/bin/bash
# 快速测试 DART-SR pipeline

echo "=========================================="
echo "DART-SR Pipeline Smoke Test"
echo "=========================================="
echo ""

cd codes

# 检查依赖
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo "Error: PyTorch not installed"; exit 1; }
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" || { echo "Error: NumPy not installed"; exit 1; }

echo ""
echo "Running pipeline tests..."
echo ""

# 运行测试
python test_pipeline.py

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✓ Pipeline test completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Prepare your dataset (DIV2K, Set5, etc.)"
    echo "  2. Update paths in codes/options/train/train_DARTSR_RCAN_local.yml"
    echo "  3. Run: cd codes && python train.py -opt options/train/train_DARTSR_RCAN_local.yml"
else
    echo "✗ Pipeline test failed. Please check the error messages above."
fi

exit $exit_code
