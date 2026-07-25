#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
[[ "$MODE" =~ ^(soft|hard|all)$ ]] || {
  echo "用法: $0 [soft|hard|all]" >&2
  exit 1
}

ROOT=$(cd "$(dirname "$0")" && pwd)
phases=(soft hard)
[[ "$MODE" == "all" ]] || phases=("$MODE")

for scale in 2 3 4; do
  gt=$((48 * scale))
  cases="${gt}x16,${gt}x24,${gt}x32,${gt}x40,${gt}x48"
  for phase in "${phases[@]}"; do
    log="$ROOT/benchmark_X${scale}_${phase}.log"
    echo
    echo "=== X${scale} ${phase}: LR=48, GT=${gt}, batches=16/24/32/40/48 ==="
    (
      cd "$ROOT/codes"
      python benchmark_b2rsr_training.py \
        --opt "options/train/train_B2RSR_RCAN_X${scale}.yml" \
        --phase "$phase" \
        --cases "$cases"
    ) 2>&1 | tee "$log"
  done
done

echo
echo "全部基准完成。日志：$ROOT/benchmark_X{2,3,4}_{soft,hard}.log"
