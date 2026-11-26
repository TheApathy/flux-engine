#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Qwen3.8-Flash-Next dedicated lossless K=5 MTP speculative decode profile for
# GB10 (DGX Spark, sm_121a, 128 GB unified LPDDR5X).
#
# Reproduces the `k5-lossless-decode` (`decode-9ba/k5all.env` + K11-DEC) profile:
#   - Coding peak:   57.4 tok/s client max | 56.3 tok/s client median (1.70x vs K=1)
#   - Mean accept:   3.02 - 3.15 tokens/cycle (K=5 / 4 Inferact MTP drafts)
#   - Token parity:  LOSSLESS 10/10 (2,560 / 2,560 greedy tokens bit-identical to K=1)
#   - Uses state-deterministic GDN_NORM_FP8=1, conservative 8,192-token prefill
#     chunks, and disables the FP8 projection weight cache (`FAST_PROJ_FP8_WEIGHT_CACHE=0`)
#     to maximize L2/UMA bandwidth and headroom for 5-row speculative verify.
#
# Usage (built-in checkpoint MTP head — JMNI-Labs, Blackfrost-AI, or nvidia):
#   MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
#   ./bench/qwen38-flash-next-gb10/serve-k5-decode.sh
set -euo pipefail

DRY_RUN=0
FORWARD_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run|--verify-only)
      DRY_RUN=1
      ;;
    *)
      FORWARD_ARGS+=("$arg")
      ;;
  esac
done

: "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-Flash-Next NVFP4 checkpoint directory}"
BIN="${BIN:-target/release/spark}"
PORT="${PORT:-8899}"
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
MODEL_NAME="${MODEL_NAME:-qwen38-native-mtp-frontier-v1}"
KTARGET="${KTARGET:-qwen3.8-flash-next}"
FLUX_MODE="${FLUX_MODE:-k5-decode}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-135168}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-${FLUX_QWEN4_PREFILL_WIDE:-${ATLAS_QWEN4_PREFILL_WIDE:-8192}}}"
PREFILL_WIDE="${FLUX_QWEN4_PREFILL_WIDE:-${ATLAS_QWEN4_PREFILL_WIDE:-$MAX_PREFILL_TOKENS}}"
NUM_DRAFTS="${NUM_DRAFTS:-4}"

[[ -d "$MODEL_DIR" ]] || {
  echo "serve-k5-decode.sh: MODEL_DIR does not exist or is not a directory: $MODEL_DIR" >&2
  exit 2
}
MTP_ARGS=()
if [[ -n "${MTP_DIR:-}" ]]; then
  [[ -d "$MTP_DIR" ]] || {
    echo "serve-k5-decode.sh: MTP_DIR does not exist or is not a directory: $MTP_DIR" >&2
    exit 2
  }
  MTP_VOCAB="${MTP_VOCAB:-96000}"
  MTP_ARGS=(--mtp-from-path "$MTP_DIR")
else
  MTP_VOCAB="${MTP_VOCAB:-0}"
fi
[[ -x "$BIN" ]] || {
  echo "serve-k5-decode.sh: BIN is missing or not executable: $BIN" >&2
  exit 2
}

if ! python3 - "$BIN" "$KTARGET" "qwen4-qsa" "mtp-from-path" "ATLAS_QWEN4_K5_HYBRID" "ATLAS_QWEN4_K5_NATIVE_MOE" "ATLAS_QWEN4_MTP_CHAIN" "ATLAS_QWEN4_VERIFY_GRAPH" "ATLAS_QWEN4_VERIFY_FAST_HOST" "ATLAS_QWEN4_MTP_BATCH_REPLAY" <<'PY'
import pathlib, sys
blob = pathlib.Path(sys.argv[1]).read_bytes()
for sym in sys.argv[2:]:
    if sym.encode("utf-8") not in blob:
        sys.stderr.write(f"serve-k5-decode.sh: binary {sys.argv[1]} is missing required symbol '{sym}'\n")
        sys.exit(2)
PY
then
  exit 2
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'QWEN38_FLASH_NEXT_VERIFY_OK profile=%s mode=%s wide=%s drafts=%s bin=%s\n' \
    "serve-k5-decode.sh" "$FLUX_MODE" "$PREFILL_WIDE" "$NUM_DRAFTS" "$BIN"
  exit 0
fi

exec env \
  -u ATLAS_PROFILE \
  -u ATLAS_FULL_PROFILE \
  FLUX_MODE="$FLUX_MODE" \
  FLUX_QWEN4_K5_HYBRID=1 \
  FLUX_QWEN4_K5_NATIVE_MOE=1 \
  FLUX_QWEN4_MTP_CHAIN=1 \
  FLUX_QWEN4_MTP_CONF=0.6 \
  FLUX_QWEN4_MTP_CONF_EARLY=1 \
  FLUX_QWEN4_VERIFY_GRAPH=1 \
  FLUX_QWEN4_VERIFY_FAST_HOST=1 \
  ATLAS_BOUNDED_HOST_COPY=1 \
  ATLAS_CUBLAS_TUNED=0 \
  ATLAS_DFLASH_ASYNC=0 \
  ATLAS_DFLASH_SPEC_CYCLE_V2=1 \
  ATLAS_MOE_T_LANES=1 \
  ATLAS_MTP_THINK_SPEC=1 \
  ATLAS_PLE_CACHE_MB=0 \
  ATLAS_PLE_HALF_PAGE=1 \
  ATLAS_QWEN4_HC_GEMV_V2=1 \
  ATLAS_QWEN4_HYBRID_ALL_K=1 \
  ATLAS_QWEN4_K16_EXACT=1 \
  ATLAS_QWEN4_K4_HYBRID=1 \
  ATLAS_QWEN4_K5_BATCH_ATTN_QKV=1 \
  ATLAS_QWEN4_K5_BATCH_GDN=1 \
  ATLAS_QWEN4_K5_BATCH_HYPER=1 \
  ATLAS_QWEN4_K5_BATCH_SSM=1 \
  ATLAS_QWEN4_K5_HYBRID=1 \
  ATLAS_QWEN4_K5_NATIVE_MOE=1 \
  ATLAS_QWEN4_MOE_DEDUP=1 \
  ATLAS_QWEN4_MOE_DOWN_ACT=1 \
  ATLAS_QWEN4_MTP_CHAIN=1 \
  ATLAS_QWEN4_PLE_IO_QUEUE_DEPTH=256 \
  ATLAS_QWEN4_PLE_IO_TIMEOUT_MS=30000 \
  ATLAS_QWEN4_PLE_PREFILL_BATCH=1 \
  ATLAS_QWEN4_PLE_PREFILL_COMBINE=1 \
  ATLAS_QWEN4_PLE_PREFILL_STREAM=3 \
  ATLAS_QWEN4_PLE_WARM_READ=2 \
  ATLAS_QWEN4_PREFILL_ATTN_CORE16=1 \
  ATLAS_QWEN4_PREFILL_ATTN_CORE32=0 \
  ATLAS_QWEN4_PREFILL_ATTN_DEVICE16=1 \
  ATLAS_QWEN4_PREFILL_ATTN_FI=1 \
  ATLAS_QWEN4_PREFILL_ATTN_FLASH=1 \
  ATLAS_QWEN4_PREFILL_ATTN_FULLROW=1 \
  ATLAS_QWEN4_PREFILL_ATTN_GQA=1 \
  ATLAS_QWEN4_PREFILL_ATTN_HC16=1 \
  ATLAS_QWEN4_PREFILL_ATTN_O16=1 \
  ATLAS_QWEN4_PREFILL_ATTN_PREP_FUSED=1 \
  ATLAS_QWEN4_PREFILL_ATTN_QKV16=1 \
  ATLAS_QWEN4_PREFILL_BA_GEMM=1 \
  ATLAS_QWEN4_PREFILL_CONV_PARALLEL=1 \
  ATLAS_QWEN4_PREFILL_EMBED_BATCH=1 \
  ATLAS_QWEN4_PREFILL_FAST_PROJ=all \
  ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8=ssm,attn \
  ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SCALE=tensor \
  ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SHARED_ACT=1 \
  ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=0 \
  ATLAS_QWEN4_PREFILL_GDN_FI=2 \
  ATLAS_QWEN4_PREFILL_GDN_LAZYFINAL=4 \
  ATLAS_QWEN4_PREFILL_GDN_NORM_FP8=1 \
  ATLAS_QWEN4_PREFILL_HC_EXACT=1 \
  ATLAS_QWEN4_PREFILL_HC_FUSED=2 \
  ATLAS_QWEN4_PREFILL_MHC_FUSED="inject,cross,combine,nostage,upmix,injgemm" \
  ATLAS_QWEN4_PREFILL_MOE_BATCH=1 \
  ATLAS_QWEN4_PREFILL_MOE_COMPACT=1 \
  ATLAS_QWEN4_PREFILL_MOE_COMPACT_K32=1 \
  ATLAS_QWEN4_PREFILL_MOE_COMPACT_T=1 \
  ATLAS_QWEN4_PREFILL_MOE_COMPACT_V2=8 \
  ATLAS_QWEN4_PREFILL_MOE_STATUS_ASYNC=1 \
  ATLAS_QWEN4_PREFILL_MOE_W4A8=1 \
  ATLAS_QWEN4_PREFILL_MOE_W4A8_K64=1 \
  ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE=2 \
  ATLAS_QWEN4_PREFILL_MULTI_CHUNK=1 \
  ATLAS_QWEN4_PREFILL_QSA=1 \
  ATLAS_QWEN4_PREFILL_QSA_KEY_GEMM=1 \
  ATLAS_QWEN4_PREFILL_QSA_TILE=1 \
  ATLAS_QWEN4_PREFILL_QSA_V2=1 \
  ATLAS_QWEN4_PREFILL_SHARED_W4A8=1 \
  ATLAS_QWEN4_PREFILL_SSM_EXACT=1 \
  ATLAS_QWEN4_PREFILL_SSM_GRID32=1 \
  ATLAS_QWEN4_PREFILL_WIDE="$PREFILL_WIDE" \
  ATLAS_QWEN4_QSA_ATTN_V2=1 \
  ATLAS_QWEN4_QSA_GRAPH_SAFE=1 \
  ATLAS_QWEN4_QSA_SCORE_V2=1 \
  ATLAS_QWEN4_QSA_SELECT_PAR=1 \
  ATLAS_QWEN4_RELEASE_BF16_QKVZ=1 \
  ATLAS_QWEN4_RELEASE_SPEC_UNUSED=1 \
  ATLAS_QWEN4_RELEASE_UNUSED=1 \
  ATLAS_QWEN4_VERIFY_GRAPH=1 \
  ATLAS_SSM_GDN_LAZY=1 \
  ATLAS_UNIFIED_MOE_LAYOUT=1 \
  ATLAS_W4A16_EXACT_ROWPAR=1 \
  ATLAS_W4A16_EXACT_SWK=1 \
  ATLAS_W4A16_GEMV_RT2=1 \
  RUST_LOG="${RUST_LOG:-info,spark::scheduler::spec_accept_log=debug}" \
  ATLAS_QWEN4_PLE_VERIFY_BATCH=1 \
  ATLAS_QWEN4_K5_BATCH_CONV=1 \
  ATLAS_QWEN4_MTP_CONF=0.6 \
  ATLAS_QWEN4_MTP_CONF_EARLY=1 \
  ATLAS_MOE_T_LANES_ROWS_DOWN_T8=1 \
  ATLAS_W4A16_EXACT_SW4=1 \
  ATLAS_VERIFY_ARGMAX_ROWS=1 \
  ATLAS_QWEN4_VERIFY_OPROJ_ROWS=1 \
  ATLAS_QWEN4_HC_DUAL_ROWS=1 \
  ATLAS_QWEN4_VERIFY_GNORM_ROWS=1 \
  ATLAS_QWEN4_MTP_UNIFIED_T=1 \
  ATLAS_SSM_DUMMY_STATE_ONLY=1 \
  ATLAS_QWEN4_VERIFY_FAST_HOST=1 \
  ATLAS_QWEN4_GDN_COMMIT_NO_MIRROR=1 \
  ATLAS_QWEN4_MTP_KEEP_LAST_ROW=1 \
  ATLAS_QWEN4_MTP_BATCH_REPLAY=1 \
  ATLAS_QWEN4_QSA_VERIFY_ROWS=1 \
  ATLAS_QWEN4_HC_TAIL_ROWS=1 \
  ATLAS_QWEN4_MTP_KV_ONLY_REPLAY=1 \
  ATLAS_MOE_T_LANES_DOWN_T8=1 \
  "$BIN" serve \
    --no-tui \
    --bind "$BIND_ADDR" \
    --port "$PORT" \
    --model-from-path "$MODEL_DIR" \
    --model-name "$MODEL_NAME" \
    --kernel-target "$KTARGET" \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype bf16 \
    --block-size 16 \
    --max-seq-len "$MAX_SEQ_LEN" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --max-batch-size 1 \
    --max-num-seqs 1 \
    --ssm-cache-slots 0 \
    --qwen4-qsa \
    --oom-guard-mb 4096 \
    --request-timeout 300 \
    --disable-simhash-watchdog \
    --enable-prefix-caching false \
    --speculative \
    "${MTP_ARGS[@]}" \
    --num-drafts "$NUM_DRAFTS" \
    --mtp-vocab "$MTP_VOCAB" \
    --mtp-quantization bf16 \
    "${FORWARD_ARGS[@]}"
