#!/usr/bin/env bash
# Create the deterministic source payload used by SOURCE_BUNDLE_MANIFEST.json.
# Run only against the final clean release commit (the script refuses a dirty
# tree); the caller writes the separate canonical JSON inventory afterward.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RELEASE_COMMIT FLASHINFER_WHEEL OUTPUT_TAR_GZ WORK_DIR" >&2
  exit 64
fi

release_commit=$1
flashinfer_wheel=$2
output_tar_gz=$3
work_dir=$4
expected_wheel_sha=d124369346a3d48eac67e31c42f7a3c813bcc0abc10e2e36db413b7b3dfd97df
repo_root=$(git rev-parse --show-toplevel)

if [[ -n $(git -C "$repo_root" status --porcelain=v1) ]]; then
  echo "refusing to bundle a dirty worktree" >&2
  exit 65
fi
if [[ $(git -C "$repo_root" rev-parse "$release_commit^{commit}") != "$(git -C "$repo_root" rev-parse HEAD)" ]]; then
  echo "release commit must be the checked-out HEAD" >&2
  exit 66
fi
if [[ $(sha256sum "$flashinfer_wheel" | awk '{print $1}') != "$expected_wheel_sha" ]]; then
  echo "FlashInfer wheel hash mismatch" >&2
  exit 67
fi
if [[ -e "$work_dir" ]]; then
  echo "work directory must not already exist" >&2
  exit 68
fi

mkdir -p "$work_dir/flux-engine" "$work_dir/third_party/flashinfer-python-0.6.14"
git -C "$repo_root" archive --format=tar "$release_commit" \
  | tar -xf - -C "$work_dir/flux-engine"
unzip -q "$flashinfer_wheel" -d "$work_dir/third_party/flashinfer-python-0.6.14"

bad_type=$(find "$work_dir" ! -type d ! -type f -print -quit)
if [[ -n "$bad_type" ]]; then
  echo "unsupported archive member type: $bad_type" >&2
  exit 69
fi
if grep -RIEl --exclude='*.o' --exclude='*.png' --exclude='*.jpg' --exclude='*.jpeg' \
    '(^|[^A-Za-z0-9_])/(home|Users)/[A-Za-z0-9._-]+/' "$work_dir" | head -n 1 | grep -q .; then
  echo "machine-local path found in source payload" >&2
  exit 70
fi

mkdir -p "$(dirname -- "$output_tar_gz")"
tar -C "$work_dir" --sort=name --format=posix --mtime='@0' \
  --owner=0 --group=0 --numeric-owner \
  --pax-option=delete=atime,delete=ctime -cf - . \
  | gzip -n -9 > "$output_tar_gz"
sha256sum "$output_tar_gz"
