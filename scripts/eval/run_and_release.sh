#!/usr/bin/env bash
# 通用包装：跑任意命令 → 归档 results/ 下新产物到云盘 → featurize instance release。
#
# 用法：
#   nohup bash scripts/eval/run_and_release.sh \
#       python scripts/eval/analyze_router_features.py > ~/run.log 2>&1 &
#
# 环境变量：
#   RELEASE=0     跑完不归还（默认 1 = 归还）
#   NOTIFY=0      不发微信通知（默认 1）
#   PERSIST_DIR   默认 /home/featurize/work/b2rsr_results

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PERSIST_DIR="${PERSIST_DIR:-/home/featurize/work/b2rsr_results}"
RELEASE="${RELEASE:-1}"
NOTIFY="${NOTIFY:-1}"

echo "==== run_and_release: $* ===="
echo "==== started at $(date) ===="

cd "$REPO_ROOT"
"$@"
STATUS=$?
echo "==== exit code: $STATUS ===="

mkdir -p "$PERSIST_DIR"
# 归档最近 12 小时内 results/ 下生成的打包与报告
find "$REPO_ROOT/results" \( -name "*.tar.gz" -o -name "*_report.md" -o -name "*_report.json" \) \
    -newermt "-12 hours" -exec cp -v --parents {} "$PERSIST_DIR/" \; 2>/dev/null || \
find "$REPO_ROOT/results" \( -name "*.tar.gz" -o -name "*_report.md" -o -name "*_report.json" \) \
    -newermt "-12 hours" -exec cp -v {} "$PERSIST_DIR/" \;

echo "==== results persisted to $PERSIST_DIR ===="

if [[ "$NOTIFY" == "1" ]] && command -v featurize >/dev/null 2>&1; then
    featurize notify -t "B2RSR 任务完成 (exit=$STATUS)：$1 ${2:-}" || true
fi

if [[ "$RELEASE" == "1" ]]; then
    echo "==== releasing instance in 60s (pkill -f run_and_release 可取消) ===="
    sleep 60
    if command -v featurize >/dev/null 2>&1; then
        featurize instance release && echo "instance released" && exit 0
        echo "WARN: release 失败，回退关机（可能不停止计费，次日请人工检查！）"
    fi
    sudo shutdown -h now || sudo poweroff || shutdown -h now
else
    echo "==== RELEASE=0，实例保持运行（计费中） ===="
fi
