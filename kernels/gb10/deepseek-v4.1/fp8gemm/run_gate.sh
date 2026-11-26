#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
# GPU-lock job: compile (under the lock) and run the fused FP8 GEMM gate on real layer-2 weights.
ATLAS_ROOT="${ATLAS_ROOT:-$(git rev-parse --show-toplevel)}"
Q="${FLUX_QUEUE_LOG:-/dev/null}"
LOCK_FILE="${FLUX_GPU_LOCK:-}"
D="${ATLAS_ROOT}/kernels/gb10/deepseek-v4.1/fp8gemm"
echo "$(date -u +%FT%TZ) dsv41-attention nvcc (under lock) + fused fp8 GEMM gate (7 shapes, <1 min) QUEUED pid=$$" >> "$Q"
if [ -n "$LOCK_FILE" ]; then exec 9>"$LOCK_FILE"; flock 9; fi
echo "$(date -u +%FT%TZ) dsv41-attention fp8-gemm gate START pid=$$" >> "$Q"
cd "$D"
PATH=/usr/local/cuda/bin:$PATH nvcc -O3 -std=c++17 -arch=sm_121a --fmad=false fp8_gemm_gate.cu -lcublasLt -o fp8_gemm_gate > nvcc.log 2>&1 && rc0=0 || rc0=$?
rc=-1
[ $rc0 -eq 0 ] && { ./fp8_gemm_gate fx/wq_b fx/wo_b fx/w1 fx/wq_a fx/wkv fx/w2 fx/wo_a_g0 > gate.log 2>&1 && rc=0 || rc=$?; }
echo "$(date -u +%FT%TZ) dsv41-attention fp8-gemm gate END rc=nvcc:$rc0,gate:$rc pid=$$" >> "$Q"
if [ -n "${LOCK_FILE:-}" ]; then flock -u 9; fi
echo "DONE nvcc=$rc0 gate=$rc"
[ $rc0 -eq 0 ] && [ $rc -eq 0 ]
