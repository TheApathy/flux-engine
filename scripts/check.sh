#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Fast Rust-only iteration loop. Skips the multi-minute PTX compile
# (ATLAS_SKIP_BUILD=1 / SKIP_ATLAS_BUILD=1) and bypasses cudarc's nvcc probe
# (CUDARC_CUDA_VERSION=13000). Use for `cargo check`, `cargo clippy`,
# `cargo clippy --tests`, etc. when you only care about Rust correctness.
#
# Usage:
#   scripts/check.sh                       # cargo check on the workspace
#   scripts/check.sh clippy --tests        # cargo clippy --tests
#   scripts/check.sh clippy -p spark-server
#
# Anything that needs to actually launch a kernel (Docker build, perf
# tests, runtime smoke) MUST go through the real build path — the stub
# registry produced under ATLAS_SKIP_BUILD has zero PTX.

set -euo pipefail

export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13000}"
export ATLAS_SKIP_BUILD="${ATLAS_SKIP_BUILD:-1}"
export SKIP_ATLAS_BUILD="${SKIP_ATLAS_BUILD:-1}"

CARGO_BIN="${CARGO:-cargo}"
if ! command -v "$CARGO_BIN" >/dev/null 2>&1; then
  if [[ -x "/workspace/.cargo/bin/cargo" ]]; then
    CARGO_BIN="/workspace/.cargo/bin/cargo"
  else
    echo "ERROR: cargo not found in PATH (override with CARGO=/path/to/cargo)" >&2
    exit 127
  fi
fi

# If the first arg is a known cargo subcommand, pass through verbatim.
# Otherwise default to `check` and forward all args (e.g. `check.sh -p
# spark-server` runs `cargo check -p spark-server`).
case "${1:-}" in
  ""|build|check|clippy|test|fmt|doc|run|tree|metadata|update|fix)
    exec "$CARGO_BIN" "${@:-check}"
    ;;
  *)
    exec "$CARGO_BIN" check "$@"
    ;;
esac
