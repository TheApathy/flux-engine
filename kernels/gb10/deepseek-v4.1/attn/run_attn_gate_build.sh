#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
# GPU-lock job: compile the attention gate UNDER the lock, then run it on both fixtures.
ATLAS_ROOT="${ATLAS_ROOT:-$(git rev-parse --show-toplevel)}"
Q="${FLUX_QUEUE_LOG:-/dev/null}"
LOCK_FILE="${FLUX_GPU_LOCK:-}"
DIR="${ATLAS_ROOT}/kernels/gb10/deepseek-v4.1/attn"
echo "$(date -u +%FT%TZ) dsv41-attention nvcc (under lock) + sparse_attn gate (2 fixtures, <2 min) QUEUED pid=$$" >> "$Q"
if [ -n "$LOCK_FILE" ]; then exec 9>"$LOCK_FILE"; flock 9; fi
echo "$(date -u +%FT%TZ) dsv41-attention nvcc+attn gate START pid=$$" >> "$Q"
cd "$DIR"
PATH=/usr/local/cuda/bin:$PATH nvcc -O3 -std=c++17 -arch=sm_121a --fmad=false -DDSV41_ATTN_GATE ../cb3/dsv41_sparse_attn.cu -o sparse_attn > nvcc.log 2>&1 && rc0=0 || rc0=$?
rc1=-1; rc2=-1
if [ $rc0 -eq 0 ]; then
  cd "$DIR/real" && "$DIR/sparse_attn" > "$DIR/attn_gate_real.log" 2>&1 && rc1=0 || rc1=$?
  cd "$DIR/sk1"  && "$DIR/sparse_attn" > "$DIR/attn_gate_synth.log" 2>&1 && rc2=0 || rc2=$?
fi
echo "$(date -u +%FT%TZ) dsv41-attention nvcc+attn gate END rc=nvcc:$rc0,real:$rc1,synth:$rc2 pid=$$" >> "$Q"
if [ -n "${LOCK_FILE:-}" ]; then flock -u 9; fi
echo "DONE nvcc=$rc0 real=$rc1 synth=$rc2"
[ "$rc0" -eq 0 ] && [ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]
