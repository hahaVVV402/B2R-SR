#!/usr/bin/env bash
# Featurize wrapper: preflight -> generic run plan -> verified export -> release.
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PLAN=""
RELEASE="${RELEASE:-1}"
NOTIFY="${NOTIFY:-1}"
EXPORT_ROOT="${EXPORT_ROOT:-/home/featurize/work/b2rsr_exports}"
PYTHON="${PYTHON:-}"
CHILD_PID=""
FINALIZING=0
STATUS_RECORDED=0
ARCHIVE_VERIFIED=0
BOOTSTRAP_ACTIVE=1

bootstrap_finalize() {
  local code=$? stamp root status archive verified=0
  [[ "$BOOTSTRAP_ACTIVE" == 1 && "$code" != 0 ]] || return "$code"
  trap - EXIT INT TERM
  set +e
  stamp=$(date -u '+%Y%m%dT%H%M%SZ')
  root=/home/featurize/work/b2rsr_bootstrap_failures
  mkdir -p "$root"
  status="$root/bootstrap_failure_${stamp}.txt"
  printf 'exit_code=%s\nrepo=%s\nplan=%s\n' "$code" "$REPO_ROOT" "$PLAN" > "$status"
  sync
  if command -v tar >/dev/null 2>&1 && command -v sha256sum >/dev/null 2>&1; then
    archive="$status.tar"
    tar -cf "$archive.part" -C "$root" "$(basename "$status")" \
      && tar -tf "$archive.part" >/dev/null \
      && mv "$archive.part" "$archive" \
      && (cd "$root" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256") \
      && (cd "$root" && sha256sum -c "$(basename "$archive").sha256") \
      && sync && verified=1
  fi
  if [[ "$verified" == 1 && "$RELEASE" == 1 ]] && command -v featurize >/dev/null 2>&1; then
    local released=0 attempt
    for attempt in 1 2 3; do
      if featurize instance release; then released=1; break; fi
      sleep $((attempt * 5))
    done
    [[ "$released" == 1 ]] || echo "BILLING WARNING: bootstrap release failed; release manually; evidence: $archive" >&2
  else
    echo "BILLING WARNING: bootstrap failure was not releasable; inspect $status" >&2
  fi
  exit "$code"
}
trap bootstrap_finalize EXIT

usage() {
  echo "Usage: $0 -opt codes/options/run/run_EDSR_d24_formal.yml" >&2
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    -opt) PLAN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$PLAN" ]] || { usage; exit 2; }
[[ "$PLAN" = /* ]] || PLAN="$REPO_ROOT/$PLAN"

choose_python() {
  local candidate
  for candidate in "$PYTHON" \
      "$(command -v python 2>/dev/null || true)" \
      /environment/miniconda3/envs/b2rsr/bin/python \
      /home/featurize/miniconda3/envs/b2rsr/bin/python \
      /home/featurize/miniconda/envs/b2rsr/bin/python \
      "$HOME/miniconda3/envs/b2rsr/bin/python" \
      /opt/conda/envs/b2rsr/bin/python; do
    if [[ -n "$candidate" && -x "$candidate" ]] \
        && "$candidate" -c 'import cv2,numpy,torch,yaml; assert torch.cuda.is_available()' >/dev/null 2>&1; then
      PYTHON="$candidate"
      return
    fi
  done
  echo "ERROR: no CUDA Python with torch/cv2/numpy/yaml" >&2
  exit 2
}

run_child() {
  setsid "$@" &
  CHILD_PID=$!
  set +e
  wait "$CHILD_PID"
  local code=$?
  set -e
  CHILD_PID=""
  return "$code"
}

handle_signal() {
  local signal=$1 code=$2
  if [[ -n "$CHILD_PID" ]]; then
    kill -"$signal" -- "-$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
    CHILD_PID=""
  fi
  exit "$code"
}
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

write_status() {
  local code=$1 state=$2
  "$PYTHON" - "$OUTPUT_ROOT/cloud_status.json" "$code" "$state" "$(git -C "$REPO_ROOT" rev-parse HEAD)" <<'PY' || return 1
import json,os,sys,tempfile
from datetime import datetime,timezone
path,code,state,commit=sys.argv[1:]
payload={"status":state,"exit_code":int(code),"commit":commit,
         "updated_at_utc":datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")}
tmp=path+".tmp"
with open(tmp,"w",encoding="utf-8") as f:
 json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,path)
PY
  STATUS_RECORDED=1
}

bundle_and_verify() {
  local partial=$1 stamp kind archive snapshot
  stamp=$(date -u '+%Y%m%dT%H%M%SZ')
  kind=$([[ "$partial" == 1 ]] && echo partial || echo complete)
  archive="$EXPORT_ROOT/${PLAN_NAME}_${kind}_${stamp}.tar"
  snapshot="$OUTPUT_ROOT/launcher_snapshot.log"
  cp "$OUTPUT_ROOT/launcher.log" "$snapshot" || return 1
  sync || return 1
  "$PYTHON" - "$OUTPUT_ROOT" "$archive" "$partial" <<'PY' || return 1
import hashlib,json,os,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path
root=Path(sys.argv[1]).resolve(); archive=Path(sys.argv[2]).resolve(); partial=bool(int(sys.argv[3]))
archive.parent.mkdir(parents=True,exist_ok=True)
files=[]
for path in sorted(root.rglob("*")):
 if not path.is_file() or path.name in {"launcher.log","bundle_receipt.json"} or path.suffix==".tmp": continue
 if not partial and path.name=="resume.pt" and path.parent.name=="training_state": continue
 files.append(path)
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
manifest={"schema_version":1,"partial":partial,
 "created_at_utc":datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z"),
 "files":{str(p.relative_to(root)): {"bytes":p.stat().st_size,"sha256":sha(p)} for p in files}}
manifest_bytes=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
temporary=archive.with_suffix(archive.suffix+".part")
with tarfile.open(temporary,"w") as tar:
 for path in files: tar.add(path,arcname=str(Path(root.name)/path.relative_to(root)),recursive=False)
 info=tarfile.TarInfo(str(Path(root.name)/"bundle_manifest.json")); info.size=len(manifest_bytes)
 import io; tar.addfile(info,io.BytesIO(manifest_bytes))
os.replace(temporary,archive)
with tarfile.open(archive,"r") as tar:
 members={m.name:m for m in tar.getmembers() if m.isfile()}
 expected={str(Path(root.name)/rel) for rel in manifest["files"]}|{str(Path(root.name)/"bundle_manifest.json")}
 if set(members)!=expected: raise RuntimeError("archive member set mismatch")
 for rel,spec in manifest["files"].items():
  data=tar.extractfile(members[str(Path(root.name)/rel)]).read()
  if len(data)!=spec["bytes"] or hashlib.sha256(data).hexdigest()!=spec["sha256"]:
   raise RuntimeError("archive member hash mismatch: "+rel)
print(json.dumps({"archive":str(archive),"members":len(expected),"partial":partial},sort_keys=True))
PY
  (cd "$(dirname "$archive")" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256") || return 1
  (cd "$(dirname "$archive")" && sha256sum -c "$(basename "$archive").sha256") || return 1
  printf '%s\n' "$archive" > "$OUTPUT_ROOT/latest_export_path.txt" || return 1
  sync || return 1
  "$PYTHON" - "$OUTPUT_ROOT/bundle_receipt.json" "$archive" <<'PY' || return 1
import hashlib,json,os,sys
p,a=sys.argv[1:]; data=open(a,"rb").read(); payload={"archive":a,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"verified":True}
t=p+".tmp"; open(t,"w").write(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(t,p)
PY
  ARCHIVE_VERIFIED=1
}

release_instance() {
  local code=$1 attempt
  [[ "$RELEASE" == 1 ]] || { echo "RELEASE=0: instance remains billable" >&2; return "$code"; }
  for attempt in 1 2 3; do
    featurize instance release && return "$code"
    sleep $((attempt * 5))
  done
  echo "BILLING WARNING: featurize instance release failed; release manually" >&2
  return 96
}

finalize() {
  local code=$?
  [[ "$FINALIZING" == 1 ]] && exit "$code"
  FINALIZING=1
  trap - EXIT INT TERM
  set +e
  write_status "$code" "$([[ "$code" == 0 ]] && echo complete || echo failed)"
  local status_result=$?
  bundle_and_verify "$([[ "$code" == 0 ]] && echo 0 || echo 1)"
  local archive_result=$?
  set -e
  if [[ "$status_result" != 0 || "$archive_result" != 0 \
        || "$STATUS_RECORDED" != 1 || "$ARCHIVE_VERIFIED" != 1 ]]; then
    echo "BILLING WARNING: status/archive not verified; instance intentionally left running" >&2
    exit 97
  fi
  if [[ "$NOTIFY" == 1 ]] && command -v featurize >/dev/null 2>&1; then
    featurize notify -t "${PLAN_NAME} finished (exit=${code}); export verified" || true
  fi
  release_instance "$code"
  exit $?
}

[[ "$RELEASE" =~ ^[01]$ && "$NOTIFY" =~ ^[01]$ ]] || { echo "RELEASE/NOTIFY must be 0 or 1" >&2; exit 2; }
for command in curl df git setsid sha256sum stat tar; do command -v "$command" >/dev/null || { echo "Missing $command" >&2; exit 2; }; done
choose_python
[[ "$REPO_ROOT" == /home/featurize/work/* ]] || { echo "Repository must be under /home/featurize/work" >&2; exit 2; }
[[ -f "$PLAN" ]] || { echo "Missing plan: $PLAN" >&2; exit 2; }
if [[ ! -f "$REPO_ROOT/.env" ]]; then
  printf '%s\n' 'SR_DATA_ROOT=/home/featurize/data' > "$REPO_ROOT/.env"
fi
PLAN_INFO=$(cd "$REPO_ROOT" && "$PYTHON" - "$PLAN" <<'PY'
import sys
sys.path.insert(0,"codes")
from options import options
p=options.load(sys.argv[1]); print(p["name"]+"\t"+p["output_root"])
PY
)
IFS=$'\t' read -r PLAN_NAME PLAN_OUTPUT <<< "$PLAN_INFO"
DATA_ROOT=$(cd "$REPO_ROOT" && "$PYTHON" - <<'PY'
import os,sys
sys.path.insert(0,"codes")
from options import options
options.load_dotenv(); print(os.environ.get("SR_DATA_ROOT", ""))
PY
)
[[ "$DATA_ROOT" == /home/featurize/data* && -d "$DATA_ROOT" ]] || {
  echo "SR_DATA_ROOT must resolve to existing /home/featurize/data on Featurize" >&2
  exit 2
}
OUTPUT_ROOT=$(cd "$REPO_ROOT" && "$PYTHON" - "$PLAN_OUTPUT" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
)
EXPERIMENTS_ROOT="$REPO_ROOT/experiments"
[[ "$OUTPUT_ROOT" == "$EXPERIMENTS_ROOT"/* ]] || { echo "Plan output escapes experiments/" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT" "$EXPORT_ROOT" "$REPO_ROOT/experiments/pre_trained_models/EDSR"
exec > >(tee -a "$OUTPUT_ROOT/launcher.log") 2>&1
BOOTSTRAP_ACTIVE=0
trap finalize EXIT

echo "==== Featurize plan: $PLAN_NAME ===="
echo "repo=$REPO_ROOT"
echo "plan=$PLAN"
echo "output=$OUTPUT_ROOT"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=no)" ]] || { echo "Tracked repository changes detected" >&2; exit 2; }
[[ "$("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))')" == *"RTX 4090"* ]] || { echo "RTX 4090 required" >&2; exit 2; }
[[ "$RELEASE" == 0 ]] || command -v featurize >/dev/null || { echo "Featurize CLI missing" >&2; exit 2; }
FREE_KB=$(df -Pk /home/featurize/work | awk 'NR==2 {print $4}')
(( FREE_KB >= 15 * 1024 * 1024 )) || { echo "Need at least 15 GiB free under /home/featurize/work" >&2; exit 2; }

# Check every configured dataset directory before any training.
while IFS= read -r directory; do
  [[ -d "$directory" ]] || { echo "Missing configured dataset directory: $directory" >&2; exit 3; }
done < <(cd "$REPO_ROOT" && "$PYTHON" - "$PLAN" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,"codes")
from options import options
plan=options.load(sys.argv[1]); paths=set()
for run in plan["runs"]:
 cfg=options.load(str(Path(run["train_opt"])))
 for dataset in cfg["datasets"].values():
  paths.update((dataset["dataroot_GT"],dataset["dataroot_LQ"]))
print(*sorted(paths),sep="\n")
PY
)

# Acquire every scale-specific official Teacher from the train YAMLs.
while IFS=$'\t' read -r path url bytes sha; do
  mkdir -p "$(dirname "$path")"
  if [[ ! -f "$path" ]]; then
    run_child curl --fail --location --retry 5 --continue-at - --output "$path.part" "$url"
    [[ "$(stat -c '%s' "$path.part")" == "$bytes" ]] || { echo "Checkpoint byte mismatch: $path" >&2; exit 3; }
    [[ "$(sha256sum "$path.part" | awk '{print $1}')" == "$sha" ]] || { echo "Checkpoint SHA mismatch: $path" >&2; exit 3; }
    mv "$path.part" "$path"
  fi
  [[ "$(stat -c '%s' "$path")" == "$bytes" && "$(sha256sum "$path" | awk '{print $1}')" == "$sha" ]] \
    || { echo "Existing checkpoint provenance mismatch: $path" >&2; exit 3; }
done < <(cd "$REPO_ROOT" && "$PYTHON" - "$PLAN" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,"codes")
from options import options
plan=options.load(sys.argv[1]); seen=set()
for run in plan["runs"]:
 cfg=options.load(str(Path(run["train_opt"])))
 path=str(Path(cfg["path"]["teacher_checkpoint"]).resolve())
 if path in seen: continue
 seen.add(path); spec=cfg["teacher"]["checkpoint"]
 print(path,spec["url"],spec["bytes"],spec["sha256"],sep="\t")
PY
)

run_child "$PYTHON" -m unittest discover -s codes/tests -p 'test_static_depth.py' -v
run_child "$PYTHON" -u "$REPO_ROOT/codes/run.py" -opt "$PLAN" --preflight
run_child "$PYTHON" -u "$REPO_ROOT/codes/run.py" -opt "$PLAN"
exit 0
