#!/usr/bin/env bash
set -euo pipefail

SCALE="${1:-4}"
[[ "$SCALE" =~ ^(2|3|4)$ ]] || { echo "用法: $0 2|3|4" >&2; exit 1; }

ROOT=$(cd "$(dirname "$0")" && pwd)
CONFIG="options/train/train_DARTSRPP_RCAN_X${SCALE}.yml"
CHECKPOINT="$ROOT/experiments/pre_trained_models/RCAN_BIX${SCALE}.pth"

[[ -f "$CHECKPOINT" ]] || {
  echo "缺少对应尺度的预训练模型: $CHECKPOINT" >&2
  exit 1
}

"$ROOT/prepare_cloud_data.sh" /home/featurize/data
cd "$ROOT/codes"
python train.py -opt "$CONFIG"
