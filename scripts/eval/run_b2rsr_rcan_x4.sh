#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DEFAULT_CHECKPOINT="$ROOT/experiments/B2RSR_RCAN_X4/models/120000_G.pth"
CHECKPOINT="${2:-$DEFAULT_CHECKPOINT}"
DATA_ROOT="/home/featurize/data"

[[ "$MODE" =~ ^(all|dense|b2rsr|check)$ ]] || {
  echo "用法: $0 [all|dense|b2rsr|check] [B2R checkpoint]" >&2
  exit 1
}

"$ROOT/scripts/data/prepare_sr_benchmarks.sh" "$DATA_ROOT"
"$ROOT/prepare_pretrained.sh"

DENSE_WEIGHT="$ROOT/experiments/pre_trained_models/RCAN_BIX4.pt"
[[ -f "$DENSE_WEIGHT" ]] || { echo "缺少 dense RCAN 权重: $DENSE_WEIGHT" >&2; exit 1; }
if [[ "$MODE" == "all" || "$MODE" == "b2rsr" || "$MODE" == "check" ]]; then
  [[ -f "$CHECKPOINT" ]] || { echo "缺少 B2R-SR checkpoint: $CHECKPOINT" >&2; exit 1; }
  CHECKPOINT=$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")
fi

if [[ "$MODE" == "check" ]]; then
  echo "评测环境检查通过。"
  echo "  数据: $DATA_ROOT/SRBenchmarks"
  echo "  Dense: $DENSE_WEIGHT"
  echo "  B2R-SR: $CHECKPOINT"
  exit 0
fi

cd "$ROOT/codes"
if [[ "$MODE" == "all" || "$MODE" == "dense" ]]; then
  python test.py -opt options/test/test_RCAN_X4_dense.yml
fi

if [[ "$MODE" == "all" || "$MODE" == "b2rsr" ]]; then
  TMP_CONFIG=$(mktemp "${TMPDIR:-/tmp}/b2rsr-eval.XXXXXX.yml")
  trap 'rm -f "$TMP_CONFIG"' EXIT
  python - "$ROOT/codes/options/test/test_B2RSR_RCAN_X4.yml" "$TMP_CONFIG" "$CHECKPOINT" <<'PY'
from pathlib import Path
import sys

source, target, checkpoint = map(Path, sys.argv[1:])
text = source.read_text()
old = "  pretrain_model_G: ../experiments/B2RSR_RCAN_X4/models/120000_G.pth"
if old not in text:
    raise SystemExit("无法在 B2R-SR 测试配置中定位默认 checkpoint。")
target.write_text(text.replace(old, "  pretrain_model_G: " + str(checkpoint), 1))
PY
  python test.py -opt "$TMP_CONFIG"
fi

echo "评测完成。日志和输出图像位于:"
echo "  $ROOT/results/eval_RCAN_X4_dense"
echo "  $ROOT/results/eval_B2RSR_RCAN_X4"
