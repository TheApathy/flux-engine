#!/usr/bin/env bash
# Build the Laguna release binary.
#
#   CUTLASS_HOME=/path/to/cutlass bash bench/laguna/build_cutlass.sh
#
# A bare `cargo build --release` is NOT equivalent and is the single most common
# way to spend an afternoon debugging a healthy tree. It produces a `spark` that
# defaults to a different kernel target, cannot load Laguna, and fails only at
# serve time -- long after the build looked clean. The three ATLAS_TARGET_*
# variables below are what select the Laguna NVFP4 kernel set, and CUTLASS_HOME
# is what lets the grouped MoE GEMMs compile at all.
#
# Note also that a clean `cargo build --release` never compiles examples or
# benches; a stale kernel caller in examples/ can survive a green build. Use
# `--features="cuda gpu-examples"` when you need those checked.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

# CUTLASS checkout. No default is right on someone else's machine; if you built
# vLLM from source you already have one under its .deps directory.
: "${CUTLASS_HOME:?set CUTLASS_HOME to a CUTLASS source checkout}"
export CUTLASS_HOME

# Architecture is resolved from kernels/$ATLAS_TARGET_HW/HARDWARE.toml (gb10 -> sm_121f).
export ATLAS_TARGET_HW="${ATLAS_TARGET_HW:-gb10}"
export ATLAS_TARGET_MODEL="${ATLAS_TARGET_MODEL:-laguna-s-2.1}"
export ATLAS_TARGET_QUANT="${ATLAS_TARGET_QUANT:-nvfp4}"

cd "$REPO" || { echo "FATAL: repo root $REPO missing"; exit 3; }

# Editing a .cu does not reliably trigger a kernel rebuild, and the "compiled N
# kernels" line is itself cached, so it will happily report success over stale
# PTX. Clearing the fingerprint is the only reliable way to force one.
if [ "${LAGUNA_FORCE_KERNELS:-0}" = 1 ]; then
  echo "forcing kernel rebuild (clearing atlas-kernels fingerprint)"
  rm -rf target/release/.fingerprint/atlas-kernels-*
fi

echo "BUILD START $(date -u +%H:%M:%S) CUTLASS=$CUTLASS_HOME HW=$ATLAS_TARGET_HW MODEL=$ATLAS_TARGET_MODEL"
cargo build --release --bin spark 2>&1
rc=$?
echo "BUILD END rc=$rc $(date -u +%H:%M:%S)"
[ "$rc" = 0 ] && echo "binary: $REPO/target/release/spark  (export LAGUNA_BIN to point the harness at it)"
exit $rc
