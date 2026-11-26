#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Qwen3.8-Flash-Next interactive low-memory 4K-chunk prefill + lossless K=5 MTP
# decode profile for GB10 (DGX Spark, sm_121a, 128 GB unified LPDDR5X).
#
# Reproduces the certified `v9-w4096` (`interactive-w4096`) operating profile
# from unified13 (`05f-serve-v9-w4096-gate` & LIL v0.7.9 w4096 campaign):
#   - 8K prefill:    5,276.3 tok/s peak | 5,266.8 tok/s median (1,555.4 ms/8K)
#   - 16K prefill:   5,176.9 tok/s peak | 5,164.5 tok/s median (1,586.2 ms/8K)
#   - 64K prefill:   5,026.5 tok/s peak | 5,016.1 tok/s median (1,633.1 ms/8K)
#   - 128K prefill:  4,899.8 tok/s peak | 4,891.7 tok/s median (1,674.7 ms/8K)
#   - K=5 decode:    57.4 tok/s peak    | 56.3 tok/s median (LOSSLESS 10/10 = 2,560/2,560)
#   - Memory floor:  11,907 - 12,695 MiB MemAvailable (+4.2 GiB extra UMA headroom vs w16384)
#
# Usage (built-in checkpoint MTP head — JMNI-Labs, Blackfrost-AI, or nvidia):
#   MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
#   ./bench/qwen38-flash-next-gb10/serve-w4096.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FLUX_MODE="${FLUX_MODE:-interactive-w4096}"
export FLUX_QWEN4_PREFILL_WIDE="${FLUX_QWEN4_PREFILL_WIDE:-4096}"
export ATLAS_QWEN4_PREFILL_WIDE="${ATLAS_QWEN4_PREFILL_WIDE:-4096}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}"

exec "$SCRIPT_DIR/serve.sh" "$@"
