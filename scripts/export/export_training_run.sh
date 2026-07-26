#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EXPERIMENT="${1:-B2RSR_RCAN_X4}"
STEP="${2:-120000}"
EXP="$ROOT/experiments/$EXPERIMENT"
CONFIG="$ROOT/codes/options/train/train_B2RSR_RCAN_X4.yml"
CHECKPOINT="$EXP/models/${STEP}_G.pth"
OUTPUT="${3:-/home/featurize/work/${EXPERIMENT}_${STEP}_export.tar}"
BUNDLE="${EXPERIMENT}_${STEP}_export"

[[ -d "$EXP" ]] || { echo "缺少实验目录: $EXP" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "缺少训练配置: $CONFIG" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "缺少 checkpoint: $CHECKPOINT" >&2; exit 1; }
command -v tar >/dev/null || { echo "缺少 tar。" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "缺少 sha256sum。" >&2; exit 1; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/b2rsr-export.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
DEST="$TMP/$BUNDLE"
mkdir -p "$DEST/metadata" "$DEST/checkpoint" "$DEST/val_samples"

cp "$CONFIG" "$DEST/metadata/"
cp "$CHECKPOINT" "$DEST/checkpoint/"

find "$EXP" -maxdepth 1 -type f -name 'train_*.log' -exec cp {} "$DEST/metadata/" \;
[[ -d "$EXP/tensorboard" ]] && cp -a "$EXP/tensorboard" "$DEST/"

LOG=$(find "$EXP" -maxdepth 1 -type f -name 'train_*.log' | sort | tail -1)
if [[ -n "$LOG" ]]; then
  grep '# Validation # PSNR' "$LOG" > "$DEST/metadata/validation_psnr.txt" || true
  tail -n 200 "$LOG" > "$DEST/metadata/final_log_tail.txt"
fi

{
  echo "experiment=$EXPERIMENT"
  echo "step=$STEP"
  echo "created_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "checkpoint=$CHECKPOINT"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$DEST/metadata/manifest.txt"

git -C "$ROOT" status --short > "$DEST/metadata/git_status.txt" 2>/dev/null || true
git -C "$ROOT" diff > "$DEST/metadata/uncommitted.patch" 2>/dev/null || true

{
  python --version 2>&1 || true
  python - <<'PY' 2>&1 || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("cudnn:", torch.backends.cudnn.version())
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
except Exception as exc:
    print("torch environment unavailable:", exc)
PY
  command -v nvidia-smi >/dev/null && nvidia-smi || true
} > "$DEST/metadata/environment.txt"

if [[ -d "$EXP/val_images" ]]; then
  find "$EXP/val_images" -type f -name "*_${STEP}.png" | sort | sed -n '1,12p' |
    while IFS= read -r image; do cp "$image" "$DEST/val_samples/"; done
fi

(
  cd "$DEST"
  find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
)

mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"
tar -cf "$OUTPUT" -C "$TMP" "$BUNDLE"
tar -tf "$OUTPUT" >/dev/null

printf '\n导出完成，请从 Featurize 下载这一个文件：\n%s\n\n' "$OUTPUT"
printf '文件大小: '
du -h "$OUTPUT" | awk '{print $1}'
printf 'SHA-256: '
sha256sum "$OUTPUT" | awk '{print $1}'
