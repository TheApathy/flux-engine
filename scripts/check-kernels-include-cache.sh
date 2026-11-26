#!/usr/bin/env bash
# The atlas-kernels PTX cache against a real nvcc when a quoted #include
# vanishes behind an unchanged wrapper.
#
# A probe kernel goes in a scratch worktree of REV. The real nvcc compiles
# only that probe; every other kernel is copied by a stand-in, so a run takes
# about a minute on CPU (no GPU). The probe includes one header and,
# under `#if 0`, one header that does not exist. Checks:
#   1. a clean build compiles the probe;
#   2. a rebuild for an unrelated edit reuses it (the inactive missing include
#      does not defeat the cache);
#   3. deleting the active header, with the probe's mtime unchanged, makes the
#      build fail with nvcc's missing-include error instead of reusing PTX;
#   4. restoring the header builds again;
#   5. renaming an older header over the include recompiles (mv keeps mtime);
#   6. a same-size in-place overwrite that keeps the mtime (cp -p) recompiles
#      once the build script reruns (cargo's own rerun check is mtime-only).
#
# Known gap (by design): cargo reruns build.rs on mtimes only, so a cp -p
# overwrite alone is caught at the next rerun for any reason. Release builds
# are fresh clones with their own target and compile every kernel.
#
# Usage:
#   scripts/check-kernels-include-cache.sh [REV]    # default HEAD

set -euo pipefail

rev="${1:-HEAD}"
repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
real_cuda="${CUDA_HOME:-/usr/local/cuda}"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/atlas-include-cache.XXXXXX")"
tree="$scratch/tree"
common="$tree/kernels/gb10/common"

cleanup() {
  git -C "$repo" worktree remove --force "$tree" 2>/dev/null || true
  rm -rf "$scratch"
}
trap cleanup EXIT

mkdir -p "$scratch/cuda/bin"
cat > "$scratch/cuda/bin/nvcc" <<EOF
#!/usr/bin/env bash
src=""; out=""; args=("\$@")
while [ \$# -gt 0 ]; do
  case "\$1" in
    -o) out="\$2"; shift 2 ;;
    *.cu) src="\$1"; shift ;;
    *) shift ;;
  esac
done
if [ "\$(basename "\$src")" = zz_include_cache_probe.cu ]; then
  echo "\$src" >> "\$FAKE_NVCC_LOG"
  exec "$real_cuda/bin/nvcc" "\${args[@]}"
fi
cp "\$src" "\$out"
EOF
chmod +x "$scratch/cuda/bin/nvcc"

git -C "$repo" worktree add --detach --quiet "$tree" "$rev"
mkdir -p "$common/zz_include_cache_probe"
cat > "$common/zz_include_cache_probe.cu" <<'EOF'
#include "zz_include_cache_probe/active.cuh"
#if 0
#include "zz_include_cache_probe/never_exists.cuh"
#endif
extern "C" __global__ void zz_include_cache_probe(float* x) { x[0] = ZZ_PROBE_VALUE; }
EOF
echo '#define ZZ_PROBE_VALUE 1.0f' > "$common/zz_include_cache_probe/active.cuh"
find "$tree" -path "$tree/.git" -prune -o -type f -exec touch -d '1 hour ago' {} +

export CARGO_TARGET_DIR="$scratch/target"
export CUDA_HOME="$scratch/cuda"
export FAKE_NVCC_LOG="$scratch/nvcc.log"
unset ATLAS_SKIP_BUILD SKIP_ATLAS_BUILD

fail=0
build() {
  : > "$FAKE_NVCC_LOG"
  (cd "$tree" && cargo build --release -p atlas-kernels > "$scratch/build.log" 2>&1)
}
probe_compiles() { grep -c zz_include_cache_probe "$FAKE_NVCC_LOG" || true; }

if build && [ "$(probe_compiles)" = 1 ]; then
  echo "1 clean build: probe compiled by the real nvcc"
else
  echo "FAIL 1: clean build did not compile the probe"; tail -20 "$scratch/build.log"; exit 1
fi

# An unrelated kernel edit reruns the build script; the probe must stay cached.
echo '// unrelated edit' >> "$common/bf16_add.cu"
if build && [ "$(probe_compiles)" = 0 ]; then
  echo "2 unrelated rebuild: probe reused (inactive missing include is harmless)"
else
  echo "FAIL 2: probe was recompiled (or the build failed) on an unrelated edit"; fail=1
fi

# Delete the active include. The probe itself is untouched and older than its PTX.
rm "$common/zz_include_cache_probe/active.cuh"
if build; then
  echo "FAIL 3: build succeeded after deleting an active include (stale PTX reused)"; fail=1
elif grep -q 'active.cuh' "$scratch/build.log"; then
  echo "3 deleted include: build failed with nvcc's missing-include error"
else
  echo "FAIL 3: build failed, but not on the missing include"; tail -20 "$scratch/build.log"; fail=1
fi

echo '#define ZZ_PROBE_VALUE 1.0f' > "$common/zz_include_cache_probe/active.cuh"
touch -d '1 hour ago' "$common/zz_include_cache_probe/active.cuh"
if build && [ "$(probe_compiles)" = 1 ]; then
  echo "4 restored include: probe recompiled and the build succeeds"
else
  echo "FAIL 4: restored include did not rebuild cleanly"; tail -20 "$scratch/build.log"; fail=1
fi

# Rename another, older header over the include. mv keeps its mtime, so every
# input still predates the cached PTX; only the file identity changed.
printf '#define ZZ_PROBE_VALUE 2.0f\n' > "$common/zz_include_cache_probe/other.cuh"
touch -d '2 hours ago' "$common/zz_include_cache_probe/other.cuh"
mv "$common/zz_include_cache_probe/other.cuh" "$common/zz_include_cache_probe/active.cuh"
if build && [ "$(probe_compiles)" = 1 ]; then
  echo "5 header renamed over the include: probe recompiled"
else
  echo "FAIL 5: a header renamed over the include reused cached PTX"; fail=1
fi

# Overwrite the include in place with same-size bytes and keep its mtime
# (cp -p / rsync -t): same path, same inode, old mtime, same size.
printf '#define ZZ_PROBE_VALUE 3.0f\n' > "$scratch/same_size.cuh"
touch -d '3 hours ago' "$scratch/same_size.cuh"
cp -p "$scratch/same_size.cuh" "$common/zz_include_cache_probe/active.cuh"
# Cargo's rerun-if-changed is mtime-only, so nothing reruns the build script
# for this alone; any other kernel edit does, and then the PTX cache itself
# must notice the new bytes.
echo '// unrelated edit 2' >> "$common/bf16_add.cu"
if build && [ "$(probe_compiles)" = 1 ]; then
  echo "6 same-size in-place overwrite (cp -p): probe recompiled"
else
  echo "FAIL 6: a same-size in-place overwrite reused cached PTX"; fail=1
fi

[ "$fail" -eq 0 ] && echo "PASS: a vanished, renamed or overwritten quoted include cannot reuse cached PTX"
exit "$fail"
