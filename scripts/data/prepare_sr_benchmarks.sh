#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/home/featurize/data}"
ARCHIVE="${2:-$DATA_ROOT/benchmark.tar}"
TARGET="$DATA_ROOT/SRBenchmarks"
URL="https://cv.snu.ac.kr/research/EDSR/benchmark.tar"

command -v tar >/dev/null || { echo "缺少 tar。" >&2; exit 1; }
mkdir -p "$DATA_ROOT"

count_images() {
  find "$1" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' '
}

validate() {
  local dataset expected scale hr lr hr_count lr_count
  local datasets=(Set5 Set14 BSD100 Urban100 Manga109)
  local counts=(5 14 100 100 109)

  for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    expected="${counts[$i]}"
    hr="$TARGET/$dataset/HR"
    [[ -d "$hr" ]] || { echo "缺少目录: $hr" >&2; return 1; }
    hr_count=$(count_images "$hr")
    [[ "$hr_count" -eq "$expected" ]] || {
      echo "$dataset HR 应为 $expected 张，实际为 $hr_count" >&2
      return 1
    }
    for scale in 2 3 4; do
      lr="$TARGET/$dataset/LR_bicubic/X$scale"
      [[ -d "$lr" ]] || { echo "缺少目录: $lr" >&2; return 1; }
      lr_count=$(count_images "$lr")
      [[ "$lr_count" -eq "$expected" ]] || {
        echo "$dataset X$scale 应为 $expected 张，实际为 $lr_count" >&2
        return 1
      }
    done
  done
}

if validate 2>/dev/null; then
  touch "$DATA_ROOT/.b2rsr_benchmarks_ready"
  echo "标准测试集已通过检查: $TARGET"
  exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
  command -v curl >/dev/null || {
    echo "缺少 $ARCHIVE，且系统没有 curl。请把 EDSR benchmark.tar 上传到 $DATA_ROOT。" >&2
    exit 1
  }
  echo "未找到 benchmark.tar，尝试从 EDSR 官方地址续传下载。"
  echo "若云端网络失败，请手动上传到: $ARCHIVE"
  curl -fL --retry 5 --retry-delay 5 --continue-at - -o "$ARCHIVE" "$URL"
fi

tar -tf "$ARCHIVE" >/dev/null || { echo "压缩包损坏或格式错误: $ARCHIVE" >&2; exit 1; }
TMP=$(mktemp -d "$DATA_ROOT/.b2rsr-benchmark.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

echo "解压 $(basename "$ARCHIVE")"
tar -xf "$ARCHIVE" -C "$TMP"

find_dataset() {
  local name=$1
  find "$TMP" -type d -name "$name" -exec test -d '{}/HR' \; -print -quit
}

mkdir -p "$TARGET"
for dataset in Set5 Set14 Urban100 Manga109; do
  src=$(find_dataset "$dataset")
  [[ -n "$src" ]] || { echo "压缩包中未找到 $dataset/HR" >&2; exit 1; }
  mkdir -p "$TARGET/$dataset"
  cp -a "$src/." "$TARGET/$dataset/"
done

src=$(find_dataset B100)
[[ -n "$src" ]] || src=$(find_dataset BSD100)
[[ -n "$src" ]] || { echo "压缩包中未找到 B100/HR 或 BSD100/HR" >&2; exit 1; }
mkdir -p "$TARGET/BSD100"
cp -a "$src/." "$TARGET/BSD100/"

validate
touch "$DATA_ROOT/.b2rsr_benchmarks_ready"
echo "标准测试集准备完成: $TARGET"
