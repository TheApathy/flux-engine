#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Execute standalone AWS Kani 0.68.0 (`cbmc` + `CaDiCaL 3.0.0`) Rust formal verification harnesses.
set -euo pipefail

KANI_HOME="${KANI_HOME:-${HOME}/.kani/kani-0.68.0}"
CARGO_HOME="${CARGO_HOME:-${HOME}/.cargo}"
TOOLCHAIN="$(tr -d '[:space:]' < "${KANI_HOME}/rust-toolchain-version")"
TOOLCHAIN_ROOT="$(rustup run "${TOOLCHAIN}" rustc --print sysroot)"

export PATH="${KANI_HOME}/bin:${TOOLCHAIN_ROOT}/bin:${CARGO_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${TOOLCHAIN_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export KANI_SYSROOT="${KANI_HOME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(mktemp -d -t atlas-kani-XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT
cd "${WORK_DIR}"

"${KANI_HOME}/bin/kani-driver" "${SCRIPT_DIR}/kani_proofs.rs" "$@"
