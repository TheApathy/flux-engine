#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Fast parallel Cargo driver for Flux Engine worktrees.
#
# Accelerates local iteration by:
#   1. Enabling sccache automatically when installed (`RUSTC_WRAPPER`).
#   2. Optionally seeding an isolated per-worktree `CARGO_TARGET_DIR` via
#      hardlinks (`cp -al`) from a warm base target directory when
#      `ATLAS_ISOLATED_TARGET=1` is set, avoiding `.cargo-lock` contention and
#      `ATLAS_BUILD_SOURCE_ROOT` thrashing across concurrent worktrees.
#
# Optional environment variables:
#   CUDA_HOME              CUDA toolkit root (default: /usr/local/cuda)
#   FLASHINFER_HOME        Optional FlashInfer include root
#   ATLAS_ISOLATED_TARGET  Set to 1 to use a per-worktree target directory
#   FLUX_TARGET_LANES_DIR  Parent dir for isolated per-worktree targets
#                          (default: ${HOME}/.cache/flux-target-lanes)
#   FLUX_WARM_TARGET_DIR   Optional warm target/ dir to hardlink-seed from

set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CARGO_BIN_DIR="${CARGO_HOME:-${HOME}/.cargo}/bin"
export PATH="${HOME}/.local/bin:${CARGO_BIN_DIR}:${CUDA_HOME}/bin:${PATH}"
export CARGO_INCREMENTAL=0
export PAGER=cat
export NEXTEST_NO_PAGER=1
export ATLAS_TARGET_HW="${ATLAS_TARGET_HW:-gb10}"
export ATLAS_TARGET_MODEL="${ATLAS_TARGET_MODEL:-qwen3.8-flash-next}"
export ATLAS_TARGET_QUANT="${ATLAS_TARGET_QUANT:-nvfp4}"

if command -v sccache >/dev/null 2>&1; then
  export RUSTC_WRAPPER="$(command -v sccache)"
fi

# Seed an isolated per-worktree target dir via hardlinks if requested via ATLAS_ISOLATED_TARGET=1
if [[ "${ATLAS_ISOLATED_TARGET:-0}" == "1" && -z "${CARGO_TARGET_DIR:-}" ]]; then
  WT_ROOT="$(git rev-parse --show-toplevel)"
  WT_NAME="$(basename "$WT_ROOT")"
  LANES_DIR="${FLUX_TARGET_LANES_DIR:-${HOME}/.cache/flux-target-lanes}"
  DEST_TARGET="${LANES_DIR}/${WT_NAME}"
  WARM_TARGET="${FLUX_WARM_TARGET_DIR:-}"
  if [[ -n "$WARM_TARGET" && ! -d "$DEST_TARGET/debug/deps" && -d "$WARM_TARGET/debug/deps" ]]; then
    mkdir -p "$DEST_TARGET/debug"
    cp -al "$WARM_TARGET/debug/deps" "$DEST_TARGET/debug/deps" 2>/dev/null || true
    cp -al "$WARM_TARGET/debug/build" "$DEST_TARGET/debug/build" 2>/dev/null || true
    rm -f "$DEST_TARGET/debug/.cargo-lock" "$DEST_TARGET/.cargo-lock" 2>/dev/null || true
  fi
  export CARGO_TARGET_DIR="$DEST_TARGET"
fi

exec "$@"
