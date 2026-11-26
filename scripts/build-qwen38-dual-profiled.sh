#!/usr/bin/env bash
# Build a sealed dual-target Qwen binary. This file is a template: it does
# nothing without an explicit clean source commit/tree and DFlash generation.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: build-qwen38-dual-profiled.sh \
  --source DIR --commit 40HEX --tree 40HEX --dflash-generation v14|v15 \
  [--workspace-root DIR] [--flashinfer-home DIR] \
  [--artifact-root DIR] [--target-dir DIR] [--fi-manifest FILE] [--profiles-only]

The script acquires the workspace build lock only after all source and profile
checks pass. Compilation does not reserve a GPU and never overwrites an
existing artifact directory.
EOF
}

die() {
  printf 'BUILD_REFUSED %s\n' "$*" >&2
  exit 2
}

source_dir=
expected_commit=
expected_tree=
dflash_generation=
workspace_root=
flashinfer_home=${FLASHINFER_HOME:-}
artifact_root=
target_dir=
fi_manifest=
profiles_only=0
while (($#)); do
  case "$1" in
    --source) source_dir=${2-}; shift 2 ;;
    --commit) expected_commit=${2-}; shift 2 ;;
    --tree) expected_tree=${2-}; shift 2 ;;
    --dflash-generation) dflash_generation=${2-}; shift 2 ;;
    --workspace-root) workspace_root=${2-}; shift 2 ;;
    --flashinfer-home) flashinfer_home=${2-}; shift 2 ;;
    --artifact-root) artifact_root=${2-}; shift 2 ;;
    --target-dir) target_dir=${2-}; shift 2 ;;
    --fi-manifest) fi_manifest=${2-}; shift 2 ;;
    --profiles-only) profiles_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -d "$source_dir/.git" || -f "$source_dir/.git" ]] || die "--source must be a git worktree"
source_dir=$(cd "$source_dir" && pwd -P)
workspace_root=${workspace_root:-$(cd "$source_dir/.." && pwd -P)}
artifact_root=${artifact_root:-${FLUX_ARTIFACT_ROOT:-$workspace_root/flux-dual-builds}}
target_dir=${target_dir:-${FLUX_DUAL_TARGET_DIR:-$workspace_root/flux-dual-target}}
[[ "$dflash_generation" == v14 || "$dflash_generation" == v15 ]] || {
  die "--dflash-generation must be v14 or v15"
}
[[ -n "$flashinfer_home" ]] || die "set --flashinfer-home or FLASHINFER_HOME"
[[ "$workspace_root" == /* && "$flashinfer_home" == /* \
  && "$artifact_root" == /* && "$target_dir" == /* ]] \
  || die "workspace, FlashInfer, and output paths must be absolute"
[[ ! -L "$flashinfer_home" ]] || die "FlashInfer home must not be a symlink"
flashinfer_home=$(realpath -- "$flashinfer_home")
artifact_root=$(realpath -m -- "$artifact_root")
target_dir=$(realpath -m -- "$target_dir")
case "$artifact_root/" in "$source_dir/"*) die "artifact root must be outside the source worktree" ;; esac
case "$target_dir/" in "$source_dir/"*) die "target directory must be outside the source worktree" ;; esac
[[ -d "$flashinfer_home/include" && -d "$flashinfer_home/3rdparty/cccl" ]] \
  || die "FlashInfer home must contain include and 3rdparty/cccl"

profile_dir="$source_dir/crates/spark-server/src/recipe/builtin"
prefill_profile="$profile_dir/qwen3.8-flash-next-prefill4k-champion.yaml"
mtp_profile="$profile_dir/qwen3.8-flash-next-native-mtp-safe-local.yaml"
if [[ "$dflash_generation" == v14 ]]; then
  dflash_profile="$profile_dir/qwen3.8-27b-dflash2-safe-local.yaml"
else
  dflash_profile="$profile_dir/qwen3.8-27b-dflash2-v15-safe-local.yaml"
fi
build_track="dense-qwen-dflash-$dflash_generation"

validate_profile() {
  local file=$1 expected_kind=$2 expected_track=${3-}
  [[ -f "$file" && ! -L "$file" ]] || die "profile is absent or a symlink: $file"
  local policy kind declared track computed duplicates malformed env_sections
  policy=$(awk '$1 == "environment_policy:" { print $2 }' "$file")
  kind=$(awk '$1 == "profile_kind:" { print $2 }' "$file")
  declared=$(awk '$1 == "environment_sha256:" { print $2 }' "$file")
  track=$(awk '$1 == "build_track:" { print $2 }' "$file")
  [[ "$policy" == sealed-v1 ]] || die "$file is not sealed-v1"
  [[ "$kind" == "$expected_kind" ]] || die "$file profile kind mismatch"
  [[ "$declared" =~ ^[0-9a-f]{64}$ ]] || die "$file has no exact environment receipt"
  [[ "$track" == "$expected_track" ]] || die "$file build track mismatch"
  env_sections=$(grep -c '^  env:$' "$file" || true)
  [[ "$env_sections" == 1 ]] || die "$file must have exactly one atlas.env map"
  malformed=$(awk '
    /^  env:/ { env=1; next }
    env && (/^[^ ]/ || /^  [^ ]/) { env=0 }
    env && /^    / && $0 !~ /^    (#.*|ATLAS_[A-Z0-9_]+: "[^"]*")$/ { print }
  ' "$file")
  [[ -z "$malformed" ]] || die "$file has a non-canonical environment entry"
  duplicates=$(awk '
    /^  env:/ { env=1; next }
    env && (/^[^ ]/ || /^  [^ ]/) { env=0 }
    env && /^    ATLAS_[A-Z0-9_]+:/ { key=$1; sub(/:$/, "", key); print key }
  ' "$file" | LC_ALL=C sort | uniq -d)
  [[ -z "$duplicates" ]] || die "$file has duplicate environment keys: $duplicates"
  computed=$(awk '
    /^  env:/ { env=1; next }
    env && (/^[^ ]/ || /^  [^ ]/) { env=0 }
    env && /^    ATLAS_[A-Z0-9_]+: "[^"]*"$/ {
      line=$0; sub(/^    /, "", line); sub(/: "/, "=", line); sub(/"$/, "", line); print line
    }
  ' "$file" | LC_ALL=C sort | sha256sum | awk '{print $1}')
  [[ "$computed" == "$declared" ]] || die "$file environment receipt mismatch"
}

validate_profile "$prefill_profile" flash-next-deterministic-prefill ""
validate_profile "$mtp_profile" flash-next-native-mtp-memory-safe ""
validate_profile "$dflash_profile" "$build_track" "$build_track"
if ((profiles_only)); then
  printf 'PROFILE_CHECK_PASS generation=%s track=%s\n' "$dflash_generation" "$build_track"
  exit 0
fi

[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || die "--commit must be an exact 40-hex commit"
[[ "$expected_tree" =~ ^[0-9a-f]{40}$ ]] || die "--tree must be an exact 40-hex tree"
actual_commit=$(git -C "$source_dir" rev-parse HEAD)
actual_tree=$(git -C "$source_dir" rev-parse 'HEAD^{tree}')
[[ "$actual_commit" == "$expected_commit" ]] || die "source commit mismatch"
[[ "$actual_tree" == "$expected_tree" ]] || die "source tree mismatch"
[[ -z "$(git -C "$source_dir" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "source worktree is dirty, including untracked files"

fi_manifest_sha="none"
fi_count=0
if [[ -n "$fi_manifest" ]]; then
  [[ -f "$fi_manifest" && ! -L "$fi_manifest" ]] \
    || die "FlashInfer build-input manifest is absent or a symlink: $fi_manifest"
  fi_manifest_sha=$(sha256sum -- "$fi_manifest" | awk '{print $1}')
  while IFS=$'\t' read -r relative expected_sha expected_size extra; do
    [[ -n "$relative" && -n "$expected_sha" && -n "$expected_size" && -z "$extra" ]] \
      || die "malformed FlashInfer build-input manifest"
    case "$relative" in
      include/*|3rdparty/cccl/*) ;;
      *) die "unsafe FlashInfer build-input path: $relative" ;;
    esac
    [[ "$relative" != *"/../"* && "$relative" != ../* && "$relative" != */.. \
      && "$relative" != *"//"* && "$relative" != ./* ]] \
      || die "unsafe FlashInfer build-input path: $relative"
    fi_path="$flashinfer_home/$relative"
    [[ -f "$fi_path" && ! -L "$fi_path" ]] \
      || die "FlashInfer build input is absent or a symlink: $relative"
    [[ "$(realpath -- "$fi_path")" == "$fi_path" ]] \
      || die "FlashInfer build-input path contains a symlink: $relative"
    [[ "$(stat -Lc '%s' "$fi_path")" == "$expected_size" ]] \
      || die "FlashInfer build-input size mismatch: $relative"
    [[ "$(sha256sum -- "$fi_path" | awk '{print $1}')" == "$expected_sha" ]] \
      || die "FlashInfer build-input SHA-256 mismatch: $relative"
    fi_count=$((fi_count + 1))
  done <"$fi_manifest"
  [[ "$fi_count" -gt 0 ]] || die "FlashInfer build-input manifest is empty"
fi

short=${expected_commit:0:8}
artifact_dir="$artifact_root/qwen38-dual-$dflash_generation-$short"
binary_name="spark-qwen38-dual-$dflash_generation-$short"
[[ ! -e "$artifact_dir" ]] || die "sealed artifact already exists: $artifact_dir"
[[ ! -e "$target_dir" ]] || die "target directory must not already exist"

# Non-blocking acquisition prevents concurrent builds from colliding.
exec 9>"$workspace_root/.build.lock"
flock -n 9 || die "workspace build lock is busy"

mkdir -p "$artifact_root"
mkdir -m 0755 "$target_dir"
mkdir -m 0755 "$artifact_dir"
log="$artifact_dir/build.log"
started=$(date -u +%FT%TZ)
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
build_command="ATLAS_TARGET_MODEL=qwen3.8-27b,qwen3.8-flash-next ATLAS_RELEASE_PROFILE_TRACK=$build_track cargo build --locked --offline --release -j2 -p spark-server --bin spark"
cargo_bin_dir=$(dirname -- "$(command -v cargo)")
{
  printf '%s dual build start commit=%s tree=%s track=%s\n' \
    "$started" "$expected_commit" "$expected_tree" "$build_track"
  cd "$source_dir"
  setsid nice -n 19 env -u ATLAS_SKIP_BUILD -u SKIP_ATLAS_BUILD -u ATLAS_SKIP_NATIVE_BUILD \
    -u ATLAS_FI_GDN_ARCH \
    PATH="${cuda_home}/bin:$cargo_bin_dir:/usr/local/bin:/usr/bin:/bin" \
    CUDA_HOME="${cuda_home}" CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13000}" \
    CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$target_dir" \
    ATLAS_BUILD_COMMIT="$expected_commit" ATLAS_RELEASE_PROFILE_TRACK="$build_track" \
    ATLAS_TARGET_HW=gb10 \
    ATLAS_TARGET_MODEL=qwen3.8-27b,qwen3.8-flash-next \
    ATLAS_TARGET_QUANT=nvfp4 \
    FLASHINFER_HOME="$flashinfer_home" \
    timeout -s TERM -k 30s 1800s \
    cargo build --locked --offline --release -j2 -p spark-server --bin spark
  printf '%s dual build complete\n' "$(date -u +%FT%TZ)"
} >"$log" 2>&1

[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$expected_commit" ]] || die "source commit changed during build"
[[ "$(git -C "$source_dir" rev-parse 'HEAD^{tree}')" == "$expected_tree" ]] || die "source tree changed during build"
[[ -z "$(git -C "$source_dir" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "source worktree changed during build"
target_rows=$(grep -E 'compiled [0-9]+ kernels for target [0-9]+ \(gb10, .* nvfp4\)' "$log" || true)
[[ $(wc -l <<<"$target_rows") -eq 2 ]] || die "build log does not contain exactly two kernel targets"
grep -Eq 'target 0 \(gb10, qwen3\.8-27b, nvfp4\)' <<<"$target_rows" || die "dense target missing"
grep -Eq 'target 1 \(gb10, qwen3\.8-flash-next, nvfp4\)' <<<"$target_rows" || die "Flash-Next target missing"

binary="$artifact_dir/$binary_name"
install -m 0555 "$target_dir/release/spark" "$binary"
[[ $(stat -c '%a:%h' "$binary") == 555:1 ]] || die "sealed binary mode/link count invalid"
finished=$(date -u +%FT%TZ)

python3 -B - "$artifact_dir/build-receipt.json" "$source_dir" "$expected_commit" \
  "$expected_tree" "$build_track" "$started" "$finished" "$build_command" "$log" \
  "$binary" "$prefill_profile" "$mtp_profile" "$dflash_profile" \
  "$fi_manifest_sha" "$fi_count" <<'PY'
import hashlib, json, os, subprocess, sys

(out, source, commit, tree, track, started, finished, command, log, binary,
 prefill, mtp, dflash, fi_manifest_sha, fi_count) = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
def scalar(path, key):
    prefix = f"  {key}: "
    values = [line[len(prefix):].strip() for line in open(path, encoding="utf-8") if line.startswith(prefix)]
    if len(values) != 1:
        raise SystemExit(f"{path}: expected exactly one {key}")
    return values[0].strip('"')
receipt = {
    "schema": "atlas-qwen38-dual-build-v1",
    "source": {"path": source, "commit": commit, "tree": tree, "status": "clean"},
    "build": {"track": track, "command": command, "started_utc": started,
              "finished_utc": finished, "log": log, "log_sha256": sha(log),
              "flashinfer_input_manifest_sha256": fi_manifest_sha,
              "flashinfer_input_count": int(fi_count)},
    "kernel_targets": ["gb10/qwen3.8-27b/nvfp4", "gb10/qwen3.8-flash-next/nvfp4"],
    "profiles": [{"path": p, "sha256": sha(p),
                  "profile_kind": scalar(p, "profile_kind"),
                  "environment_sha256": scalar(p, "environment_sha256")}
                 for p in (prefill, mtp, dflash)],
    "binary": {"path": binary, "sha256": sha(binary),
               "build_id": subprocess.check_output(["readelf", "-n", binary], text=True).split("Build ID: ", 1)[1].split()[0],
               "bytes": os.stat(binary).st_size, "mode": "0555", "nlink": 1},
}
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
with os.fdopen(os.open(out, flags, 0o444), "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush(); os.fsync(handle.fileno())
PY
chmod 0444 "$log"
printf 'BUILD_PASS binary=%s receipt=%s\n' "$binary" "$artifact_dir/build-receipt.json"
