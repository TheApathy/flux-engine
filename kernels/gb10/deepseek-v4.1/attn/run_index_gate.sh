#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
# GPU job: indexer gate. Queue line, flock, START/END, positive DONE line with the exit code.
ATLAS_ROOT="${ATLAS_ROOT:-$(git rev-parse --show-toplevel)}"
Q="${FLUX_QUEUE_LOG:-/dev/null}"
LOCK_FILE="${FLUX_GPU_LOCK:-}"
DIR="${ATLAS_ROOT}/kernels/gb10/deepseek-v4.1/attn"
BIN=$DIR/sparse_index_gate
[ -x "$BIN" ] || { echo "DONE rc=127 (binary missing)"; exit 127; }
echo "$(date -u +%FT%TZ) dsv41-attention indexer gate v2 (12 small fixtures + pool cases, <1 min) QUEUED pid=$$" >> "$Q"
if [ -n "$LOCK_FILE" ]; then exec 9>"$LOCK_FILE"; flock 9; fi
echo "$(date -u +%FT%TZ) dsv41-attention indexer gate START pid=$$" >> "$Q"
cd "$DIR"
"$BIN" idx_fixtures/runD_L20_kernel_L02_0 idx_fixtures/runD_L20_kernel_L02_1 idx_fixtures/runD_L20_kernel_L14_1 \
     idx_fixtures/runD_L20_kernel_L20_0 idx_fixtures/runD_L20_kernel_L20_1 idx_fixtures/runD_L20_kernel_L24_0 \
     idx_fixtures/runE_torch_L02_0 idx_fixtures/runE_torch_L02_1 idx_fixtures/runE_torch_L14_1 \
     idx_fixtures/runE_torch_L20_0 idx_fixtures/runE_torch_L20_1 idx_fixtures/runE_torch_L24_0 > index_gate.log 2>&1 && rc=0 || rc=$?
echo "$(date -u +%FT%TZ) dsv41-attention indexer gate END rc=$rc pid=$$" >> "$Q"
if [ -n "${LOCK_FILE:-}" ]; then flock -u 9; fi
echo "DONE rc=$rc"
exit "$rc"
