#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash on ONE GB10 with Atlas (single-node, EP=1).
#
# Why single-node works here: the stock NVFP4 / FP8 checkpoints are 155-179 GB
# and need EP=2 across two GB10s. A single GB10 has ~119 GiB usable, so the
# only way onto one node is a checkpoint that fits. This script defaults to a
# REAP expert-pruned FP8 checkpoint (model_type=deepseek_v4, 144/160 of 256
# experts, ue8m0 block-scaled FP8 — a format Atlas's deepseek_v4 loader reads).
#
# Atlas runs EP=1 by default (the MoE all-reduce is a no-op with one rank), so
# there is no --ep flag to set. What matters is kv-cache-dtype=fp8 (BF16 KV
# produces corrupted output on this checkpoint) and a gpu-memory-utilization high enough to
# hold weights + KV + scratch.
#
#   MODEL_DIR=/path/to/checkpoint bash bench/deepseek-v4/serve_single.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

export PATH="/usr/local/cuda/bin:$PATH"

# REQUIRED for single-node E8M0 (MXFP4) routed experts. DeepSeek-V4's native
# E8M0 experts only have a *transposed* prefill GEMM (moe_..._t_k64_e8m0); the
# non-transposed fallback misreads the [N,K/32] E8M0 scales as NVFP4 [N,K/16]
# and panics. Building the transposed tables the normal way needs cost_full GB
# of headroom, which a single-node 87 GB checkpoint (~22 GB free) does not have.
# Unified layout transposes phase-by-phase, freeing the untransposed copies as
# it goes, so it fits the tight budget. See factory/m2_setup.rs.
export ATLAS_UNIFIED_MOE_LAYOUT="${ATLAS_UNIFIED_MOE_LAYOUT:-1}"

MODELS_DIR="${FLUX_MODELS_DIR:-${HOME}/models}"
MODEL_DIR="${MODEL_DIR:-${MODELS_DIR}/DeepSeek-V4-Flash-162B}"
BIN="${DS4_BIN:-$REPO/target/release/spark}"
PORT="${PORT:-8899}"
HOST="${HOST:-127.0.0.1}"
KV_DTYPE="${KV_DTYPE:-fp8}"                       # fp8 REQUIRED for coherence
GPU_MEM="${GPU_MEM:-0.94}"                        # 0.94*119 ~= 112 GB budget
# 12288 (not 16384): the attention mirrors (~2.2 GB) + the BF16
# prefill fallbacks squeeze the KV pool at gpu_mem 0.94-0.96; 16384 no longer
# boots. Freeing the BF16 fallbacks (switch prefill to the FP8 GEMMs) is the
# open memory item that buys the headroom back.
MAX_SEQ="${MAX_SEQ:-12288}"
MAX_BATCH="${MAX_BATCH:-1}"
LOG="${LOG:-$REPO/serve-deepseek-single.log}"

[ -x "$BIN" ] || { echo "FATAL: spark binary not found/executable at $BIN (build first: ATLAS_TARGET_MODEL=deepseek-v4.1 cargo build --release --bin flux-server)"; exit 3; }
[ -f "$MODEL_DIR/config.json" ] || { echo "FATAL: no config.json under MODEL_DIR=$MODEL_DIR"; exit 3; }

echo "serve: $BIN"
echo "model: $MODEL_DIR"
echo "port : $HOST:$PORT   kv=$KV_DTYPE  gpu_mem=$GPU_MEM  max_seq=$MAX_SEQ  batch=$MAX_BATCH"
echo "log  : $LOG"

# EXTRA_ARGS is how perf variants are driven (--speculative, --num-drafts N,
# etc.) without forking this script per experiment.
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
[ ${#EXTRA[@]} -gt 0 ] && echo "extra: ${EXTRA[*]}"

exec > >(tee "$LOG") 2>&1
exec "$BIN" serve "$MODEL_DIR" \
  --model-from-path "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --kv-cache-dtype "$KV_DTYPE" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-seq-len "$MAX_SEQ" \
  --max-batch-size "$MAX_BATCH" \
  "${EXTRA[@]}"
