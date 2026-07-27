#!/usr/bin/env bash
# 跑 depth sweep → 结果拷到云盘(/home/featurize/work) → featurize instance release 归还实例停止计费。
#
# 用法（在 Featurize 实例上）：
#   nohup bash scripts/eval/run_depth_sweep_and_shutdown.sh [run_depth_sweep.py 的参数...] \
#       > ~/depth_sweep_run.log 2>&1 &
# 之后可直接断开 SSH，跑完自动归还实例（= 停止计费）。
#
# 环境变量开关：
#   RELEASE=0     跑完不归还实例（调试用）
#   PERSIST_DIR   结果持久化目录，默认 /home/featurize/work/b2rsr_results（云盘，归还后不丢）
#   NOTIFY=1      跑完发微信公众号通知（需关注 Featurize 公众号）

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PERSIST_DIR="${PERSIST_DIR:-/home/featurize/work/b2rsr_results}"
RELEASE="${RELEASE:-1}"
NOTIFY="${NOTIFY:-1}"

echo "==== depth sweep started at $(date) ===="

cd "$REPO_ROOT"
python scripts/eval/run_depth_sweep.py "$@"
STATUS=$?

echo "==== sweep exit code: $STATUS ===="

# 无论成功失败，都把结果归档 + 日志拷到云盘（/home/featurize/work 归还后仍保留）
mkdir -p "$PERSIST_DIR"
find "$REPO_ROOT/results/depth_sweep" -name "B2RSR_DEPTH_SWEEP_*.tar.gz" \
    -newermt "-12 hours" -exec cp -v {} "$PERSIST_DIR/" \; 2>/dev/null
cp -v ~/depth_sweep_run.log "$PERSIST_DIR/depth_sweep_run_$(date +%Y%m%d-%H%M%S).log" 2>/dev/null || true

echo "==== results persisted to $PERSIST_DIR ===="
ls -lh "$PERSIST_DIR"

if [[ "$NOTIFY" == "1" ]] && command -v featurize >/dev/null 2>&1; then
    featurize notify -t "B2RSR depth sweep 完成 (exit=$STATUS)，结果已存云盘，实例即将归还" || true
fi

if [[ "$RELEASE" == "1" ]]; then
    echo "==== releasing instance in 60s (Ctrl-C / pkill 可取消) ===="
    sleep 60
    if command -v featurize >/dev/null 2>&1; then
        featurize instance release && echo "instance released" && exit 0
        echo "WARN: featurize instance release 失败，回退到关机（注意：关机可能不停止计费！）"
    else
        echo "WARN: 未找到 featurize CLI，回退到关机（注意：关机可能不停止计费！）"
    fi
    sudo shutdown -h now || sudo poweroff || shutdown -h now
else
    echo "==== RELEASE=0, 实例保持运行（继续计费中！） ===="
fi
