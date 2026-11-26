#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
# GPU-lock job: BUILD under the lock (so no other lane's timing window overlaps the build),
# then the core gates on runF_faithful and runG_replay. Positive DONE line with all exit codes.
ATLAS_ROOT="${ATLAS_ROOT:-$(git rev-parse --show-toplevel)}"
Q="${FLUX_QUEUE_LOG:-/dev/null}"
LOCK_FILE="${FLUX_GPU_LOCK:-}"
W="${ATLAS_ROOT}"
echo "$(date -u +%FT%TZ) dsv41-attention build (under lock, no GPU) + core gates runF/runG/runI/runJ + profile (~5 min) QUEUED pid=$$" >> "$Q"
if [ -n "$LOCK_FILE" ]; then exec 9>"$LOCK_FILE"; flock 9; fi
echo "$(date -u +%FT%TZ) dsv41-attention build+core-gate START pid=$$" >> "$Q"
cd "$W"
export PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_MODEL=deepseek-v4.1 ATLAS_TARGET_QUANT=cb3
cargo build --release -j 16 -p spark-model --example dsv41_attn_seam --example dsv41_attn_replay_gate --example dsv41_attn_runj_gate --example dsv41_forward --features cuda,gpu-examples > $W/kernels/gb10/deepseek-v4.1/attn/build.log 2>&1 && rb=0 || rb=$?
r1=-1; r2=-1; r3=-1; r4=-1; r5=-1
if [ $rb -eq 0 ]; then
  cd "$W/kernels/gb10/deepseek-v4.1/attn"
  "$W/target/release/examples/dsv41_attn_seam" runF_faithful > core_gate_runF_faithful.log 2>&1 && r1=0 || r1=$?
  "$W/target/release/examples/dsv41_attn_seam" runG_replay > core_gate_runG_replay.log 2>&1 && r2=0 || r2=$?
  "$W/target/release/examples/dsv41_attn_replay_gate" runI_plus20 > replay_gate_runI.log 2>&1 && r3=0 || r3=$?
  "$W/target/release/examples/dsv41_attn_runj_gate" runJ_decode > runj_gate.log 2>&1 && r4=0 || r4=$?
  ATLAS_DSV41_CORE_PROF=1 "$W/target/release/examples/dsv41_forward" --run runH_feed40 --layers 40 --path model --attn-real --prof > core_profile.log 2>&1 && r5=0 || r5=$?
fi
echo "$(date -u +%FT%TZ) dsv41-attention build+core-gate END rc=build:$rb,runF:$r1,runG:$r2,runI:$r3,runJ:$r4,prof:$r5 pid=$$" >> "$Q"
if [ -n "${LOCK_FILE:-}" ]; then flock -u 9; fi
echo "DONE build=$rb runF=$r1 runG=$r2 runI=$r3 runJ=$r4 prof=$r5"
[ "$rb" -eq 0 ] && [ "$r1" -eq 0 ] && [ "$r2" -eq 0 ] && [ "$r3" -eq 0 ] && [ "$r4" -eq 0 ] && [ "$r5" -eq 0 ]
