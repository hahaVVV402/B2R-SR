#!/usr/bin/env bash
set -euo pipefail

SCALE="${1:-4}"
MODE="${2:-train}"
[[ "$SCALE" =~ ^(2|3|4)$ ]] || { echo "用法: $0 2|3|4 [train|smoke]" >&2; exit 1; }
[[ "$MODE" =~ ^(train|smoke)$ ]] || { echo "用法: $0 2|3|4 [train|smoke]" >&2; exit 1; }
[[ "$MODE" != "smoke" || "$SCALE" == "4" ]] || { echo "smoke 配置目前仅支持 X4。" >&2; exit 1; }

ROOT=$(cd "$(dirname "$0")" && pwd)
SUFFIX=""
[[ "$MODE" == "smoke" ]] && SUFFIX="_smoke"
CONFIG="options/train/train_B2RSR_RCAN_X${SCALE}${SUFFIX}.yml"
CHECKPOINT="$ROOT/experiments/pre_trained_models/RCAN_BIX${SCALE}.pt"

"$ROOT/prepare_pretrained.sh"
[[ -f "$CHECKPOINT" ]] || {
  echo "缺少对应尺度的预训练模型: $CHECKPOINT" >&2
  exit 1
}

"$ROOT/prepare_cloud_data.sh" /home/featurize/data "$SCALE"
cd "$ROOT/codes"
python train.py -opt "$CONFIG"
