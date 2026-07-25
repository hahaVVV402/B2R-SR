#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/home/featurize/data}"
SCALE="${2:-all}"
[[ "$SCALE" == "all" || "$SCALE" =~ ^(2|3|4)$ ]] || {
  echo "用法: $0 [数据目录] [2|3|4|all]" >&2
  exit 1
}
command -v unzip >/dev/null || { echo "缺少 unzip，请先安装。" >&2; exit 1; }
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

scales=(2 3 4)
[[ "$SCALE" == "all" ]] || scales=("$SCALE")

ready=true
for scale in "${scales[@]}"; do
  [[ -f "$DATA_ROOT/.b2rsr_ready_x$scale" ]] || ready=false
done
if $ready; then
  echo "数据已通过检查，无需重新解压：${DATA_ROOT}（尺度: ${scales[*]}）"
  exit 0
fi

# Featurize 新实例可能提供已解压目录，也可能只下载 DF2K.zip。
if [[ ! -d "$DATA_ROOT/DF2K/DF2K_train_HR_sub" ]]; then
  [[ -f "$DATA_ROOT/DF2K.zip" ]] || {
    echo "缺少 DF2K 目录或 $DATA_ROOT/DF2K.zip；请先从 Featurize 数据集下载到实例。" >&2
    exit 1
  }
  echo "开始解压大型训练集 DF2K.zip（只在新实例首次运行时执行）"
  unzip -q -n "$DATA_ROOT/DF2K.zip" -d "$DATA_ROOT"
else
  echo "DF2K 已解压，跳过大型压缩包。"
fi

extract_if_missing() {
  local archive=$1 marker=$2
  [[ -e "$marker" ]] && { echo "已存在，跳过: $marker"; return; }
  [[ -f "$DATA_ROOT/$archive" ]] || { echo "缺少文件: $DATA_ROOT/$archive" >&2; exit 1; }
  echo "解压 $archive"
  unzip -q -n "$DATA_ROOT/$archive" -d "$DATA_ROOT"
}

extract_if_missing DIV2K_valid_HR.zip "$DATA_ROOT/DIV2K_valid_HR"
for scale in "${scales[@]}"; do
  extract_if_missing "DIV2K_valid_LR_bicubic_X${scale}.zip" \
    "$DATA_ROOT/DIV2K_valid_LR_bicubic/X$scale"
done

TRAIN_GT="$DATA_ROOT/DF2K/DF2K_train_HR_sub"
VAL_GT="$DATA_ROOT/DIV2K_valid_HR"
[[ -d "$TRAIN_GT" ]] || { echo "未找到训练 GT: $TRAIN_GT" >&2; exit 1; }
[[ -d "$VAL_GT" ]] || { echo "未找到验证 GT: $VAL_GT" >&2; exit 1; }

count_images() {
  find "$1" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' '
}

TRAIN_GT_COUNT=$(count_images "$TRAIN_GT")
VAL_GT_COUNT=$(count_images "$VAL_GT")
[[ "$TRAIN_GT_COUNT" -gt 0 ]] || { echo "训练 GT 目录为空: $TRAIN_GT" >&2; exit 1; }
[[ "$VAL_GT_COUNT" -eq 100 ]] || { echo "验证 GT 应为 100 张，实际为 $VAL_GT_COUNT" >&2; exit 1; }

printf '\n%-5s %-8s %-8s %s\n' "尺度" "训练对" "验证对" "训练 LQ"
for scale in "${scales[@]}"; do
  TRAIN_LQ="$DATA_ROOT/DF2K/DF2K_train_LR_bicubic/X${scale}_sub"
  VAL_LQ="$DATA_ROOT/DIV2K_valid_LR_bicubic/X$scale"
  [[ -d "$TRAIN_LQ" ]] || { echo "未找到训练 LQ: $TRAIN_LQ" >&2; exit 1; }
  [[ -d "$VAL_LQ" ]] || { echo "未找到验证 LQ: $VAL_LQ" >&2; exit 1; }

  TRAIN_LQ_COUNT=$(count_images "$TRAIN_LQ")
  VAL_LQ_COUNT=$(count_images "$VAL_LQ")
  [[ "$TRAIN_LQ_COUNT" -eq "$TRAIN_GT_COUNT" ]] || {
    echo "X$scale 训练集不配对: GT=$TRAIN_GT_COUNT, LQ=$TRAIN_LQ_COUNT" >&2
    exit 1
  }
  [[ "$VAL_LQ_COUNT" -eq "$VAL_GT_COUNT" ]] || {
    echo "X$scale 验证集不配对: GT=$VAL_GT_COUNT, LQ=$VAL_LQ_COUNT" >&2
    exit 1
  }
  touch "$DATA_ROOT/.b2rsr_ready_x$scale"
  printf 'X%-4s %-8s %-8s %s\n' "$scale" "$TRAIN_GT_COUNT" "$VAL_GT_COUNT" "$TRAIN_LQ"
done

echo "数据准备完成：${DATA_ROOT}（尺度: ${scales[*]}）"
