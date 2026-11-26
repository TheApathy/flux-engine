#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
# GPU job: attention seam gate (dsv41_attn_seam) vs runF_faithful (4 layers of attention weights, <2 min).
ATLAS_ROOT="${ATLAS_ROOT:-$(git rev-parse --show-toplevel)}"
Q="${FLUX_QUEUE_LOG:-/dev/null}"
LOCK_FILE="${FLUX_GPU_LOCK:-}"
BIN="${ATLAS_ROOT}/target/release/examples/dsv41_attn_seam"
LOG="${SEAM_LOG:-${ATLAS_ROOT}/kernels/gb10/deepseek-v4.1/attn/seam_gate.log}"
[ -x "$BIN" ] || { echo "DONE rc=127 (binary missing)"; exit 127; }
echo "$(date -u +%FT%TZ) dsv41-attention seam gate vs runF (compress/index/attention, <2 min) QUEUED pid=$$" >> "$Q"
if [ -n "$LOCK_FILE" ]; then exec 9>"$LOCK_FILE"; flock 9; fi
echo "$(date -u +%FT%TZ) dsv41-attention seam gate START pid=$$" >> "$Q"
"$BIN" "${RUN:-runF_faithful}" > "$LOG" 2>&1 && rc=0 || rc=$?
echo "$(date -u +%FT%TZ) dsv41-attention seam gate END rc=$rc pid=$$" >> "$Q"
if [ -n "${LOCK_FILE:-}" ]; then flock -u 9; fi
echo "DONE rc=$rc"
exit "$rc"
