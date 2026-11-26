#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
# GPU job: sparse attention gate on the REAL (runC_2048 L2) and SYNTHETIC (all-masked row) fixtures.
ATLAS_ROOT="${ATLAS_ROOT:-$(git rev-parse --show-toplevel)}"
Q="${FLUX_QUEUE_LOG:-/dev/null}"
LOCK_FILE="${FLUX_GPU_LOCK:-}"
DIR="${ATLAS_ROOT}/kernels/gb10/deepseek-v4.1/attn"
BIN=$DIR/sparse_attn
[ -x "$BIN" ] || { echo "DONE rc=127 (binary missing)"; exit 127; }
echo "$(date -u +%FT%TZ) dsv41-attention sparse_attn gate + production entry (2 fixtures, <1 min) QUEUED pid=$$" >> "$Q"
if [ -n "$LOCK_FILE" ]; then exec 9>"$LOCK_FILE"; flock 9; fi
echo "$(date -u +%FT%TZ) dsv41-attention sparse_attn gate START pid=$$" >> "$Q"
cd "$DIR/real" && "$BIN" > "$DIR/attn_gate_real.log" 2>&1 && rc1=0 || rc1=$?
cd "$DIR/sk1"  && "$BIN" > "$DIR/attn_gate_synth.log" 2>&1 && rc2=0 || rc2=$?
echo "$(date -u +%FT%TZ) dsv41-attention sparse_attn gate END rc=$rc1,$rc2 pid=$$" >> "$Q"
if [ -n "${LOCK_FILE:-}" ]; then flock -u 9; fi
echo "DONE rc_real=$rc1 rc_synth=$rc2"
[ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]
