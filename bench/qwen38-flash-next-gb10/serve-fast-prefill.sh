#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Qwen3.8-Flash-Next pure maximum-velocity prefill-only operating profile for
# GB10 (DGX Spark, sm_121a, 128 GB unified LPDDR5X).
#
# Reproduces the `fast-prefill` (`fn-unified-fast.env`) lineage:
#   - Activates store-warp E4M3 GDN output normalization (`GDN_NORM_FP8=3` Ultra tier),
#     7-stage mHC fusion (`inject,cross,combine,nostage,upmix,injgemm,postpre`),
#     MXFP8 MoE slot quantization (`MOE_SLOT_MXFP8=1`), XQ activation reuse
#     (`MOE_XQ_REUSE=1`), and last-layer crop (`LAST_LAYER_CROP=1`).
#   - Intended for prefill-only batch ingestion, RAG indexing, rubric scoring,
#     or single-token TTFT sweeps (target-only by default). For combined
#     serving with 10/10 lossless K=5 decode, use `serve.sh` (`GDN_NORM_FP8=2`).
#
# Usage:
#   MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-Offload \
#   ./bench/qwen38-flash-next-gb10/serve-fast-prefill.sh
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

: "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-Flash-Next-NVFP4-Offload checkpoint directory}"
BIN="${BIN:-target/release/spark}"
PORT="${PORT:-8899}"
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
MODEL_NAME="${MODEL_NAME:-qwen38-native-mtp-frontier-v1}"
KTARGET="${KTARGET:-qwen3.8-flash-next}"
FLUX_MODE="${FLUX_MODE:-fast-prefill}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-135168}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-${FLUX_QWEN4_PREFILL_WIDE:-${ATLAS_QWEN4_PREFILL_WIDE:-16384}}}"
PREFILL_WIDE="${FLUX_QWEN4_PREFILL_WIDE:-${ATLAS_QWEN4_PREFILL_WIDE:-$MAX_PREFILL_TOKENS}}"
GDN_NORM_FP8="${FLUX_QWEN4_PREFILL_GDN_NORM_FP8:-${ATLAS_QWEN4_PREFILL_GDN_NORM_FP8:-3}}"

[[ -d "$MODEL_DIR" ]] || {
  echo "serve-fast-prefill.sh: MODEL_DIR does not exist or is not a directory: $MODEL_DIR" >&2
  exit 2
}
[[ -x "$BIN" ]] || {
  echo "serve-fast-prefill.sh: BIN is missing or not executable: $BIN" >&2
  exit 2
}

if ! python3 - "$BIN" "$KTARGET" "qwen4-qsa" "ATLAS_QWEN4_PREFILL_WIDE" "ATLAS_QWEN4_PREFILL_GDN_NORM_FP8" "ATLAS_QWEN4_PREFILL_MOE_SLOT_MXFP8" "ATLAS_QWEN4_PREFILL_MOE_XQ_REUSE" "ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP" <<'PY'
import pathlib, sys
blob = pathlib.Path(sys.argv[1]).read_bytes()
for sym in sys.argv[2:]:
    if sym.encode("utf-8") not in blob:
        sys.stderr.write(f"serve-fast-prefill.sh: binary {sys.argv[1]} is missing required symbol '{sym}'\n")
        sys.exit(2)
PY
then
  exit 2
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'QWEN38_FLASH_NEXT_VERIFY_OK profile=%s mode=%s wide=%s gdn_norm_fp8=%s bin=%s\n' \
    "serve-fast-prefill.sh" "$FLUX_MODE" "$PREFILL_WIDE" "$GDN_NORM_FP8" "$BIN"
  exit 0
fi

exec env \
  -u ATLAS_PROFILE \
  -u ATLAS_FULL_PROFILE \
  FLUX_MODE="$FLUX_MODE" \
  ATLAS_DISABLE_SPECULATION=1 \
  FLUX_QWEN4_PREFILL_WIDE="$PREFILL_WIDE" \
  FLUX_QWEN4_PREFILL_GDN_NORM_FP8="$GDN_NORM_FP8" \
  FLUX_QWEN4_PREFILL_MOE_SLOT_MXFP8=1 \
  FLUX_QWEN4_PREFILL_MOE_XQ_REUSE=1 \
  FLUX_QWEN4_PREFILL_LAST_LAYER_CROP=1 \
  FLUX_QWEN4_PREFILL_MHC_FUSED="inject,cross,combine,nostage,upmix,injgemm,postpre" \
  ATLAS_CUBLAS_TUNED=0 \
  ATLAS_DFLASH_ASYNC=0 \
  ATLAS_PLE_CACHE_MB=0 \
  ATLAS_PLE_HALF_PAGE=1 \
  ATLAS_QWEN4_K16_EXACT=1 \
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
  ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=1 \
  ATLAS_QWEN4_PREFILL_GDN_FI=2 \
  ATLAS_QWEN4_PREFILL_GDN_LAZYFINAL=4 \
  ATLAS_QWEN4_PREFILL_GDN_NORM_FP8="$GDN_NORM_FP8" \
  ATLAS_QWEN4_PREFILL_HC_EXACT=1 \
  ATLAS_QWEN4_PREFILL_HC_FUSED=2 \
  ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP=1 \
  ATLAS_QWEN4_PREFILL_MHC_FUSED="inject,cross,combine,nostage,upmix,injgemm,postpre" \
  ATLAS_QWEN4_PREFILL_MOE_BATCH=1 \
  ATLAS_QWEN4_PREFILL_MOE_COMPACT=1 \
  ATLAS_QWEN4_PREFILL_MOE_COMPACT_K32=1 \
  ATLAS_QWEN4_PREFILL_MOE_COMPACT_T=1 \
  ATLAS_QWEN4_PREFILL_MOE_COMPACT_V2=8 \
  ATLAS_QWEN4_PREFILL_MOE_SLOT_MXFP8=1 \
  ATLAS_QWEN4_PREFILL_MOE_STATUS_ASYNC=1 \
  ATLAS_QWEN4_PREFILL_MOE_W4A8=1 \
  ATLAS_QWEN4_PREFILL_MOE_W4A8_K64=1 \
  ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE=2 \
  ATLAS_QWEN4_PREFILL_MOE_XQ_REUSE=1 \
  ATLAS_QWEN4_PREFILL_MULTI_CHUNK=1 \
  ATLAS_QWEN4_PREFILL_QSA=1 \
  ATLAS_QWEN4_PREFILL_QSA_KEY_GEMM=1 \
  ATLAS_QWEN4_PREFILL_QSA_TILE=1 \
  ATLAS_QWEN4_PREFILL_QSA_V2=1 \
  ATLAS_QWEN4_PREFILL_SHARED_W4A8=1 \
  ATLAS_QWEN4_PREFILL_SSM_EXACT=1 \
  ATLAS_QWEN4_PREFILL_SSM_GRID32=1 \
  ATLAS_QWEN4_PREFILL_WIDE="$PREFILL_WIDE" \
  ATLAS_QWEN4_QSA_SELECT_PAR=1 \
  ATLAS_QWEN4_RELEASE_BF16_QKVZ=1 \
  ATLAS_QWEN4_RELEASE_UNUSED=1 \
  ATLAS_UNIFIED_MOE_LAYOUT=1 \
  ATLAS_W4A16_GEMV_RT2=1 \
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
    "${FORWARD_ARGS[@]}"
