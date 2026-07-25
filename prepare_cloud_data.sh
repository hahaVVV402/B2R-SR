#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/home/featurize/data}"

command -v unzip >/dev/null || { echo "缺少 unzip，请先安装。" >&2; exit 1; }
cd "$DATA_ROOT"

for archive in \
  DIV2K_valid_HR.zip \
  DIV2K_valid_LR_bicubic_X2.zip \
  DIV2K_valid_LR_bicubic_X3.zip \
  DIV2K_valid_LR_bicubic_X4.zip; do
  [[ -f "$archive" ]] || { echo "缺少文件: $DATA_ROOT/$archive" >&2; exit 1; }
  echo "解压 $archive"
  unzip -q -n "$archive" -d "$DATA_ROOT"
done

find_dir() {
  for path in "$@"; do
    [[ -d "$path" ]] && { printf '%s\n' "$path"; return; }
  done
  return 1
}

# 优先使用已切好的子图，减少训练时读取和裁剪大图的开销。
TRAIN_GT=$(find_dir \
  "$DATA_ROOT/DF2K/DF2K_train_HR_sub" \
  "$DATA_ROOT/DF2K/DF2K_train_HR" \
  "$DATA_ROOT/DF2K/HR") || {
  echo "未找到 DF2K HR 目录，请检查 $DATA_ROOT/DF2K 的结构。" >&2
  exit 1
}
VAL_GT=$(find_dir "$DATA_ROOT/DIV2K_valid_HR") || {
  echo "未找到 DIV2K_valid_HR。" >&2
  exit 1
}

count_images() {
  find "$1" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' '
}

TRAIN_GT_COUNT=$(count_images "$TRAIN_GT")
VAL_GT_COUNT=$(count_images "$VAL_GT")
[[ "$TRAIN_GT_COUNT" -gt 0 ]] || { echo "DF2K HR 目录为空: $TRAIN_GT" >&2; exit 1; }
[[ "$VAL_GT_COUNT" -eq 100 ]] || { echo "验证集 HR 应为 100 张，实际为 $VAL_GT_COUNT: $VAL_GT" >&2; exit 1; }

printf '\n%-5s %-8s %-8s %s\n' "尺度" "训练GT" "验证GT" "LQ目录"
for scale in 2 3 4; do
  TRAIN_LQ=$(find_dir \
    "$DATA_ROOT/DF2K/DF2K_train_LR_bicubic/X${scale}_sub" \
    "$DATA_ROOT/DF2K/DF2K_train_LR_bicubic/X$scale" \
    "$DATA_ROOT/DF2K/LR_bicubic/X$scale") || {
      echo "未找到 DF2K X$scale LQ 目录。" >&2
      exit 1
    }
  VAL_LQ=$(find_dir "$DATA_ROOT/DIV2K_valid_LR_bicubic/X$scale") || {
    echo "未找到 DIV2K validation X$scale LQ 目录。" >&2
    exit 1
  }

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
  printf 'X%-4s %-8s %-8s %s\n' "$scale" "$TRAIN_GT_COUNT" "$VAL_GT_COUNT" "$TRAIN_LQ"
done

cat <<EOF

数据准备完成。YAML 路径：
训练 GT: $TRAIN_GT
验证 GT: $VAL_GT
训练 LQ: $DATA_ROOT/DF2K/DF2K_train_LR_bicubic/X{2,3,4}_sub
验证 LQ: $DATA_ROOT/DIV2K_valid_LR_bicubic/X{2,3,4}
qEOF
