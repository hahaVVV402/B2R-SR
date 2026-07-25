#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
DEST="$ROOT/experiments/pre_trained_models"
URL="${RCAN_WEIGHTS_URL:-https://drive.usercontent.google.com/download?id=1EQytAXfTptfz-pI4JHfq40P-CcdGfCjZ&export=download&confirm=t}"
LOCAL_ARCHIVE="${RCAN_WEIGHTS_ARCHIVE:-/home/featurize/data/models_ECCV2018RCAN.zip}"
EXPECTED_SHA256="${RCAN_WEIGHTS_SHA256:-44d97388bb4d94f629cb11d9a26134b056878d6f5d56c134d0179d6acd5ef80e}"

mkdir -p "$DEST"
if [[ -s "$DEST/RCAN_BIX2.pt" && -s "$DEST/RCAN_BIX3.pt" && -s "$DEST/RCAN_BIX4.pt" ]]; then
  echo "原版 RCAN X2/X3/X4 权重已存在，跳过下载。"
  exit 0
fi

command -v unzip >/dev/null || { echo "缺少 unzip。" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "缺少 sha256sum。" >&2; exit 1; }

REMOVE_ARCHIVE=false
if [[ -f "$LOCAL_ARCHIVE" ]]; then
  ARCHIVE="$LOCAL_ARCHIVE"
  echo "使用本地 RCAN 权重包：$ARCHIVE"
else
  command -v curl >/dev/null || { echo "缺少 curl。" >&2; exit 1; }
  ARCHIVE="$DEST/models_ECCV2018RCAN.zip.part"
  REMOVE_ARCHIVE=true
  echo "未找到本地权重包，开始下载（约 288 MB，支持断点续传）"
  curl -L -C - --fail --retry 3 --progress-bar "$URL" -o "$ARCHIVE"
fi

if ! echo "$EXPECTED_SHA256  $ARCHIVE" | sha256sum -c -; then
  $REMOVE_ARCHIVE && rm -f "$ARCHIVE"
  echo "权重包校验失败，请检查文件后重新运行：$ARCHIVE" >&2
  exit 1
fi

for scale in 2 3 4; do
  unzip -j -o "$ARCHIVE" "models_ECCV2018RCAN/RCAN_BIX${scale}.pt" -d "$DEST" >/dev/null
  [[ -s "$DEST/RCAN_BIX${scale}.pt" ]] || { echo "缺少 RCAN_BIX${scale}.pt" >&2; exit 1; }
done
$REMOVE_ARCHIVE && rm -f "$ARCHIVE"

echo "权重已保存到持久化仓库目录：$DEST"
