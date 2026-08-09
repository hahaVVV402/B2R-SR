#!/usr/bin/env bash
# One command: verify Featurize image -> recover EDSR x2/x3/x4 -> evaluate -> export -> release.
set -Eeuo pipefail

GOAL_ID="20260809-132635"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GOAL_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$GOAL_ROOT/../../.." && pwd)
PROTOCOL="$GOAL_ROOT/protocol.json"
RUNNER="$SCRIPT_DIR/formal_recovery.py"
DATA_ROOT="${DATA_ROOT:-/home/featurize/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/experiments/pre_trained_models/EDSR}"
RUN_ROOT="${RUN_ROOT:-/home/featurize/work/b2rsr_results/$GOAL_ID}"
EXPORT_ROOT="${EXPORT_ROOT:-/home/featurize/work/b2rsr_exports}"
RELEASE="${RELEASE:-1}"
NOTIFY="${NOTIFY:-1}"
PYTHON="${PYTHON:-}"
FINALIZING=0
BUNDLE_DONE=0
ARCHIVE_VERIFIED=0
STATUS_RECORDED=0
CHILD_PID=""
WORKLOAD_STATUS=1

choose_python() {
  local candidates=() candidate
  [[ -n "$PYTHON" ]] && candidates+=("$PYTHON")
  command -v python >/dev/null 2>&1 && candidates+=("$(command -v python)")
  candidates+=(
    "/home/featurize/miniconda3/envs/b2rsr/bin/python"
    "/home/featurize/miniconda/envs/b2rsr/bin/python"
    "$HOME/miniconda3/envs/b2rsr/bin/python"
    "/opt/conda/envs/b2rsr/bin/python"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import cv2,numpy,torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
      PYTHON="$candidate"
      return
    fi
  done
  echo "ERROR: 找不到同时具有 torch/CUDA、OpenCV 和 NumPy 的 Python。可用 PYTHON=/path/to/python 指定。" >&2
  return 1
}

run_foreground() {
  local result
  "$@" &
  CHILD_PID=$!
  if wait "$CHILD_PID"; then result=0; else result=$?; fi
  CHILD_PID=""
  return "$result"
}

handle_signal() {
  local code=$1
  trap - INT TERM
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
    CHILD_PID=""
  fi
  exit "$code"
}

write_status() {
  local status=$1 phase=$2
  local temporary="$RUN_ROOT/launcher_status.txt.tmp"
  {
    echo "goal_id=$GOAL_ID"
    echo "phase=$phase"
    echo "workload_exit_status=$status"
    echo "recorded_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "run_root=$RUN_ROOT"
  } > "$temporary" || return 1
  mv "$temporary" "$RUN_ROOT/launcher_status.txt" || return 1
  sync || return 1
  STATUS_RECORDED=1
}

archive_and_verify() {
  local partial=$1 stamp archive
  stamp=$(date -u '+%Y%m%dT%H%M%SZ')
  archive="$EXPORT_ROOT/${GOAL_ID}_$([[ "$partial" == "1" ]] && echo partial || echo complete)_${stamp}.tar"
  mkdir -p "$EXPORT_ROOT" || return 1
  local options=()
  [[ "$partial" == "1" ]] && options+=(--allow-partial)
  run_foreground "$PYTHON" "$RUNNER" bundle \
    --protocol "$PROTOCOL" \
    --run-root "$RUN_ROOT" \
    --archive "$archive" \
    "${options[@]}" || return 1
  (cd "$(dirname "$archive")" && sha256sum "$(basename "$archive")" > "$(basename "$archive.sha256")") || return 1
  (cd "$(dirname "$archive")" && sha256sum -c "$(basename "$archive.sha256")") || return 1
  printf '%s\n' "$archive" > "$RUN_ROOT/latest_export_path.txt" || return 1
  sync || return 1
  BUNDLE_DONE=1
  ARCHIVE_VERIFIED=1
  echo "==== verified export: $archive ===="
}

fallback_partial_archive() {
  local stamp archive
  stamp=$(date -u '+%Y%m%dT%H%M%SZ')
  archive="$EXPORT_ROOT/${GOAL_ID}_raw_partial_${stamp}.tar"
  mkdir -p "$EXPORT_ROOT" || return 1
  tar -cf "$archive.part" -C "$(dirname "$RUN_ROOT")" "$(basename "$RUN_ROOT")" || return 1
  tar -tf "$archive.part" >/dev/null || return 1
  mv "$archive.part" "$archive" || return 1
  (cd "$(dirname "$archive")" && sha256sum "$(basename "$archive")" > "$(basename "$archive.sha256")") || return 1
  (cd "$(dirname "$archive")" && sha256sum -c "$(basename "$archive.sha256")") || return 1
  printf '%s\n' "$archive" > "$RUN_ROOT/latest_export_path.txt" || return 1
  sync || return 1
  BUNDLE_DONE=1
  echo "WARN: structured partial bundle failed; preserved raw partial archive, but no release will occur without the structured internal manifest: $archive" >&2
}

release_instance() {
  local status=$1 attempt
  if [[ "$RELEASE" != "1" ]]; then
    echo "==== RELEASE=$RELEASE; 实例保持运行并继续计费 ===="
    return
  fi
  if ! command -v featurize >/dev/null 2>&1; then
    echo "BILLING WARNING: featurize CLI 不存在，不能确认停止计费；尝试关机但必须在网页人工归还。" | tee -a "$RUN_ROOT/release.log" >&2
  else
    for attempt in 1 2 3; do
      echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') release_requested attempt=$attempt workload_status=$status" | tee -a "$RUN_ROOT/release.log"
      sync
      if featurize instance release 2>&1 | tee -a "$RUN_ROOT/release.log"; then
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') release_command_succeeded attempt=$attempt" | tee -a "$RUN_ROOT/release.log"
        sync
        return
      fi
      sleep 10
    done
    echo "BILLING WARNING: featurize instance release 三次失败；关机不保证停止计费，必须在网页人工归还。" | tee -a "$RUN_ROOT/release.log" >&2
  fi
  sudo shutdown -h now || sudo poweroff || shutdown -h now || true
}

finalize() {
  local status=$?
  [[ "$FINALIZING" == "1" ]] && exit "$status"
  FINALIZING=1
  trap - EXIT
  trap '' INT TERM
  set +e
  WORKLOAD_STATUS=$status
  write_status "$status" "$([[ "$status" == "0" ]] && echo complete || echo failed)"
  if [[ "$BUNDLE_DONE" != "1" ]]; then
    archive_and_verify 1 || fallback_partial_archive || true
  fi
  if [[ "$ARCHIVE_VERIFIED" != "1" || "$STATUS_RECORDED" != "1" ]]; then
    echo "BILLING WARNING: 状态或结构化归档未验证；为防止证据丢失，不自动归还实例。请人工检查后在网页归还。" >&2
    if [[ "$NOTIFY" == "1" ]] && command -v featurize >/dev/null 2>&1; then
      featurize notify -t "EDSR $GOAL_ID 需要人工处理：归档未验证，实例仍在计费" || true
    fi
    [[ "$status" == "0" ]] && status=97
    exit "$status"
  fi
  if [[ "$NOTIFY" == "1" ]] && command -v featurize >/dev/null 2>&1; then
    featurize notify -t "EDSR $GOAL_ID 完成 (exit=$status)，结果已验证并准备归还" || true
  fi
  release_instance "$status"
  exit "$status"
}
trap finalize EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

mkdir -p "$RUN_ROOT" "$CHECKPOINT_DIR" "$EXPORT_ROOT"
exec > >(tee -a "$RUN_ROOT/launcher.log") 2>&1

echo "==== EDSR formal Featurize run $GOAL_ID ===="
echo "started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "repo=$REPO_ROOT"
echo "data=$DATA_ROOT"
echo "run=$RUN_ROOT"
echo "release=$RELEASE"

[[ "$RELEASE" =~ ^[01]$ ]] || { echo "RELEASE 只能是 0 或 1" >&2; exit 2; }
[[ "$NOTIFY" =~ ^[01]$ ]] || { echo "NOTIFY 只能是 0 或 1" >&2; exit 2; }
[[ "$REPO_ROOT" == /home/featurize/work/* ]] || { echo "ERROR: 仓库必须位于持久化 /home/featurize/work 下。" >&2; exit 2; }
[[ "$RUN_ROOT" == /home/featurize/work/* ]] || { echo "ERROR: RUN_ROOT 必须位于持久化 /home/featurize/work 下。" >&2; exit 2; }
[[ "$EXPORT_ROOT" == /home/featurize/work/* ]] || { echo "ERROR: EXPORT_ROOT 必须位于持久化 /home/featurize/work 下。" >&2; exit 2; }
for command in curl git sha256sum stat tar; do
  command -v "$command" >/dev/null 2>&1 || { echo "ERROR: 缺少命令 $command" >&2; exit 2; }
done
choose_python
if [[ "$RELEASE" == "1" ]]; then
  command -v featurize >/dev/null 2>&1 || { echo "ERROR: RELEASE=1 但找不到 featurize CLI。" >&2; exit 2; }
  featurize --help > "$RUN_ROOT/featurize_cli_help.txt" 2>&1 || { echo "ERROR: featurize CLI 不可调用。" >&2; exit 2; }
fi
run_foreground "$PYTHON" "$RUNNER" self-test

# Preserve the exact frozen goal contract beside the generated artifacts.
while IFS= read -r relative; do
  source="$GOAL_ROOT/$relative"
  destination="$RUN_ROOT/source_snapshot/$relative"
  mkdir -p "$(dirname "$destination")"
  if [[ -e "$destination" ]]; then
    cmp -s "$source" "$destination" || { echo "ERROR: existing source snapshot differs: $destination" >&2; exit 2; }
  else
    cp "$source" "$destination"
  fi
done < <("$PYTHON" - "$GOAL_ROOT/source_manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for path in sorted(manifest["files"]):
    print(path)
PY
)
manifest_snapshot="$RUN_ROOT/source_snapshot/source_manifest.json"
if [[ -e "$manifest_snapshot" ]]; then
  cmp -s "$GOAL_ROOT/source_manifest.json" "$manifest_snapshot" || {
    echo "ERROR: existing source manifest snapshot differs: $manifest_snapshot" >&2
    exit 2
  }
else
  cp "$GOAL_ROOT/source_manifest.json" "$manifest_snapshot"
fi

ACQUISITION="$RUN_ROOT/checkpoint_acquisition_history.tsv"
if [[ ! -f "$ACQUISITION" ]]; then
  printf 'checked_utc\tscale\tstatus\turl\teffective_url\thttp_code\tfilename\tbytes\tsha256\n' > "$ACQUISITION"
fi

while IFS=$'\t' read -r scale url filename expected_bytes expected_sha; do
  destination="$CHECKPOINT_DIR/$filename"
  part="$destination.part"
  status="reused_verified"
  effective_url="$url"
  http_code=""
  if [[ -f "$destination" ]]; then
    observed_bytes=$(stat -c '%s' "$destination")
    observed_sha=$(sha256sum "$destination" | awk '{print $1}')
    [[ "$observed_bytes" == "$expected_bytes" && "$observed_sha" == "$expected_sha" ]] || {
      echo "ERROR: 已有 checkpoint 与冻结 bytes/SHA 不符，拒绝覆盖: $destination" >&2
      exit 3
    }
  else
    status="downloaded"
    if [[ -f "$part" ]] \
      && [[ "$(stat -c '%s' "$part")" == "$expected_bytes" ]] \
      && [[ "$(sha256sum "$part" | awk '{print $1}')" == "$expected_sha" ]]; then
      status="recovered_verified_part"
      mv "$part" "$destination"
    else
      curl_output=$(curl --silent --show-error --fail --location --retry 5 --retry-delay 2 \
        --continue-at - --output "$part" --write-out $'%{url_effective}\t%{http_code}' "$url")
      IFS=$'\t' read -r effective_url http_code <<< "$curl_output"
      observed_bytes=$(stat -c '%s' "$part")
      observed_sha=$(sha256sum "$part" | awk '{print $1}')
      [[ "$observed_bytes" == "$expected_bytes" && "$observed_sha" == "$expected_sha" ]] || {
        echo "ERROR: 下载的 x$scale checkpoint bytes/SHA 不符；保留 .part 供诊断。" >&2
        exit 3
      }
      mv "$part" "$destination"
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$scale" "$status" "$url" "$effective_url" "$http_code" \
    "$filename" "$expected_bytes" "$expected_sha" >> "$ACQUISITION"
done < <("$PYTHON" - "$PROTOCOL" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1], encoding="utf-8"))
for scale in protocol["training"]["scales"]:
    spec = protocol["checkpoints"]["scales"][str(scale)]
    print(scale, spec["url"], spec["filename"], spec["bytes"], spec["sha256"], sep="\t")
PY
)

run_foreground "$PYTHON" -u "$RUNNER" check \
  --protocol "$PROTOCOL" \
  --repo-root "$REPO_ROOT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/cloud_preflight.json"

while IFS= read -r scale; do
  while IFS= read -r seed; do
    echo "==== train x${scale} seed${seed} ===="
    run_foreground "$PYTHON" -u "$RUNNER" train \
      --protocol "$PROTOCOL" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --data-root "$DATA_ROOT" \
      --run-root "$RUN_ROOT" \
      --scale "$scale" \
      --seed "$seed"
  done < <("$PYTHON" - "$PROTOCOL" <<'PY'
import json, sys
print(*json.load(open(sys.argv[1], encoding="utf-8"))["training"]["seeds"], sep="\n")
PY
)
done < <("$PYTHON" - "$PROTOCOL" <<'PY'
import json, sys
print(*json.load(open(sys.argv[1], encoding="utf-8"))["training"]["scales"], sep="\n")
PY
)

run_foreground "$PYTHON" -u "$RUNNER" evaluate \
  --protocol "$PROTOCOL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT"

WORKLOAD_STATUS=0
write_status 0 complete
archive_and_verify 0
exit 0
