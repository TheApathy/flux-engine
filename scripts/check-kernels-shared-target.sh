#!/usr/bin/env bash
# Two worktrees, one CARGO_TARGET_DIR: each atlas-kernels build must embed its
# own tree's kernels. Before the source-root key, cargo treated the second
# tree's build script as fresh (its rerun-if-changed paths were absolute into
# the first tree) and shipped the first tree's PTX.
#
# Cheap and CPU-only: a fake nvcc copies each .cu into its .ptx, so a marker
# appended to one kernel shows which tree a PTX came from. Both trees are
# back-dated an hour so mtimes alone cannot rescue a stale reuse.
#
# Usage:
#   scripts/check-kernels-shared-target.sh [REV]    # default HEAD
#
# Exit 0: every build embedded its own tree, a same-tree no-op build stayed
# Fresh, and switching back reused that tree's cached PTX without recompiling.

set -euo pipefail

rev="${1:-HEAD}"
repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/atlas-shared-target.XXXXXX")"
marker_cu="kernels/gb10/common/argmax_bf16.cu"

cleanup() {
  for tree in a b; do
    git -C "$repo" worktree remove --force "$scratch/$tree" 2>/dev/null || true
  done
  rm -rf "$scratch"
}
trap cleanup EXIT

mkdir -p "$scratch/cuda/bin"
cat > "$scratch/cuda/bin/nvcc" <<'EOF'
#!/usr/bin/env bash
src=""; out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    *.cu) src="$1"; shift ;;
    *) shift ;;
  esac
done
cp "$src" "$out"
echo "$src" >> "$FAKE_NVCC_LOG"
EOF
chmod +x "$scratch/cuda/bin/nvcc"

for tree in a b; do
  git -C "$repo" worktree add --detach --quiet "$scratch/$tree" "$rev"
  printf '\n// atlas-shared-target-marker tree-%s\n' "$tree" >> "$scratch/$tree/$marker_cu"
  find "$scratch/$tree" -path "$scratch/$tree/.git" -prune -o -type f -exec touch -d '1 hour ago' {} +
done

export CARGO_TARGET_DIR="$scratch/target"
export CUDA_HOME="$scratch/cuda"
export FAKE_NVCC_LOG="$scratch/nvcc.log"
unset ATLAS_SKIP_BUILD SKIP_ATLAS_BUILD ATLAS_BUILD_SOURCE_ROOT

ptx_dir() {
  # The newest build-script output names the PTX directory the crate embeds.
  local output
  output="$(ls -t "$CARGO_TARGET_DIR"/release/build/atlas-kernels-*/output | head -1)"
  sed -n 's/^cargo:rustc-env=ATLAS_PTX_DIR=//p' "$output"
}

fail=0
build_and_check() {
  local tree="$1"
  : > "$FAKE_NVCC_LOG"
  (cd "$scratch/$tree" && cargo build --release -q -p atlas-kernels 2>/dev/null)
  local compiled embedded
  compiled="$(wc -l < "$FAKE_NVCC_LOG")"
  embedded="$(grep -ho 'atlas-shared-target-marker tree-[ab]' "$(ptx_dir)"/t0__argmax_bf16.ptx || echo none)"
  echo "build $tree: embedded '$embedded', nvcc ran $compiled times"
  if [ "$embedded" != "atlas-shared-target-marker tree-$tree" ]; then
    echo "FAIL: tree $tree embedded another tree's PTX"
    fail=1
  fi
  LAST_COMPILED="$compiled"
}

build_and_check a
build_and_check b
build_and_check a
if [ "$LAST_COMPILED" -ne 0 ]; then
  echo "FAIL: switching back to tree a recompiled $LAST_COMPILED kernels"
  fail=1
fi
if ! (cd "$scratch/a" && cargo build --release -v -p atlas-kernels 2>&1) | grep -q 'Fresh atlas-kernels'; then
  echo "FAIL: a same-tree no-op build reran the atlas-kernels build script"
  fail=1
fi

[ "$fail" -eq 0 ] && echo "PASS: shared CARGO_TARGET_DIR keeps each worktree's kernels"
exit "$fail"
