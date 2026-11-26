#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# ==============================================================================
# Flux Engine — Turnkey Runner for local-inference-lab/llm-inference-bench
# Upstream Benchmark Repo: https://github.com/local-inference-lab/llm-inference-bench
#
# Usage:
#   # 1. One-word preset profiles against a running Flux Engine server:
#   ./bench/llm-inference-bench/run-lil-bench.sh matrix       # 0..128K context + decode matrix (C=1)
#   ./bench/llm-inference-bench/run-lil-bench.sh prefill      # Standalone cold-prefill sweep (8K..128K)
#   ./bench/llm-inference-bench/run-lil-bench.sh estonia      # 136,576-token 6-hop Estonia needle benchmark
#   ./bench/llm-inference-bench/run-lil-bench.sh coding       # Burst C=1 decode throughput probe
#   ./bench/llm-inference-bench/run-lil-bench.sh repetition   # Greedy 8K needle-checksum stability probe
#   ./bench/llm-inference-bench/run-lil-bench.sh gsm8k        # Full 1,319-item GSM8K accuracy benchmark
#   ./bench/llm-inference-bench/run-lil-bench.sh mmlu-pro     # Stratified 1,000-item MMLU-Pro accuracy benchmark
#   ./bench/llm-inference-bench/run-lil-bench.sh gpqa         # 198-item GPQA Diamond accuracy benchmark
#   ./bench/llm-inference-bench/run-lil-bench.sh tool-eval    # Community tool-eval-bench suite
#
#   # 2. Or pass any llm_decode_bench.py flags directly:
#   ./bench/llm-inference-bench/run-lil-bench.sh --contexts 0,16384,65536 --concurrency 1 --duration 30
# ==============================================================================

set -euo pipefail
export LLM_BENCH_NO_UPDATE_CHECK="${LLM_BENCH_NO_UPDATE_CHECK:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LIL_REPO_URL="${LIL_REPO_URL:-https://github.com/local-inference-lab/llm-inference-bench.git}"
LIL_DEFAULT_CACHE="${XDG_CACHE_HOME:-${HOME}/.cache}/flux-engine/llm-inference-bench"

HOST="${FLUX_HOST:-${HOST:-127.0.0.1}}"
PORT="${FLUX_PORT:-${PORT:-}}"
MODEL="${FLUX_SERVED_MODEL:-${MODEL:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: ./bench/llm-inference-bench/run-lil-bench.sh [PROFILE_OR_FLAGS...]

Built-in Turnkey Profiles:
  matrix          Run 0..128K prefill + C=1 sustained decode matrix (--contexts 0,8192,16384,32768,65536,131072 --concurrency 1)
  prefill         Run standalone cold-prefill sweep across 8k,16k,32k,64k,128k (--prefill-only --standalone-prefill)
  estonia         Run the 136,576-token 6-hop Estonia needle benchmark (--test-profile estonia --profile-concurrency 1 --profile-runs 5 --max-tokens 5120)
  coding          Run C=1 burst decode benchmark (--skip-prefill --contexts 0 --concurrency 1 --request-count 5 --warmup-request-count 1 --max-tokens 1024)
  repetition      Run the 8K greedy numerical stability probe (--test-profile needle-checksum)
  hotel-lights    Run the compact numeric reasoning profile (--test-profile hotel-lights)
  gsm8k           Run the pinned 1,319-item GSM8K accuracy benchmark (--test-profile gsm8k)
  mmlu-pro        Run the pinned 1,000-item MMLU-Pro accuracy benchmark (--test-profile mmlu-pro)
  gpqa            Run the 198-item GPQA Diamond accuracy benchmark (--test-profile gpqa-diamond)
  tool-eval       Run the community tool-eval-bench suite (--tool-eval)

Environment Overrides:
  HOST / FLUX_HOST          Server host (default: 127.0.0.1)
  PORT / FLUX_PORT          Server port (auto-probed across 8899, 8896, 8888, 8000 if unset)
  MODEL / FLUX_SERVED_MODEL Model ID (auto-detected from GET /v1/models if unset)
  LIL_BENCH_DIR             Path to local checkout of local-inference-lab/llm-inference-bench
  OUTPUT_DIR                Directory for JSON output receipts (default: bench/llm-inference-bench/runs)
  DRY_RUN=1                 Print the resolved command without executing
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

# 1. Resolve or auto-clone local-inference-lab/llm-inference-bench
resolve_lil_script() {
    local candidates=(
        "${LIL_BENCH_DIR:-}"
        "${REPO_ROOT}/ext/llm-inference-bench"
        "/opt/llm-inference-bench"
        "${LIL_DEFAULT_CACHE}"
    )
    for dir in "${candidates[@]}"; do
        if [[ -n "${dir}" && -f "${dir}/llm_decode_bench.py" ]]; then
            echo "${dir}/llm_decode_bench.py"
            return 0
        fi
    done

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "${LIL_DEFAULT_CACHE}/llm_decode_bench.py"
        return 0
    fi

    echo "[lil-bench] Cloning ${LIL_REPO_URL} into ${LIL_DEFAULT_CACHE} ..." >&2
    mkdir -p "$(dirname "${LIL_DEFAULT_CACHE}")"
    git clone --depth 1 "${LIL_REPO_URL}" "${LIL_DEFAULT_CACHE}" >&2
    echo "${LIL_DEFAULT_CACHE}/llm_decode_bench.py"
}

# 2. Auto-probe active Flux Engine port if PORT is not explicitly provided
resolve_port() {
    if [[ -n "${PORT}" ]]; then
        return 0
    fi
    for candidate_port in 8899 8896 8888 8000 5000; do
        if curl -sf --max-time 2 "http://${HOST}:${candidate_port}/health" >/dev/null 2>&1 || \
           curl -sf --max-time 2 "http://${HOST}:${candidate_port}/v1/models" >/dev/null 2>&1; then
            PORT="${candidate_port}"
            return 0
        fi
    done
    PORT="8899"
}

# 3. Query /v1/models to auto-detect model name and verify server readiness
preflight_server() {
    local need_estonia_seq_len="${1:-0}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        PORT="${PORT:-8899}"
        MODEL="${MODEL:-qwen38-native-mtp-frontier-v1}"
        return 0
    fi

    resolve_port
    local models_json=""
    if ! models_json="$(curl -sf --max-time 5 "http://${HOST}:${PORT}/v1/models" 2>/dev/null)"; then
        echo "[lil-bench] ERROR: Could not reach Flux Engine server at http://${HOST}:${PORT}/v1/models" >&2
        echo "  Start the server first, for example:" >&2
        echo "    MODEL_DIR=/path/to/models/Qwen3.8-Flash-Next-NVFP4 ./bench/qwen38-flash-next-gb10/serve.sh" >&2
        echo "  Or via the universal container:" >&2
        echo "    docker run --rm --gpus all --ipc=host -p 8899:8899 -v /path/to/models:/models:ro ghcr.io/theapathy/flux-engine:gb10 serve /models/Qwen3.8-Flash-Next-NVFP4-Offload" >&2
        exit 1
    fi

    if [[ -z "${MODEL}" ]]; then
        MODEL="$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d["data"][0]["id"])' <<<"${models_json}" 2>/dev/null || echo "default")"
    fi

    # If running Estonia (136,576 prompt tokens + 5,120 max_tokens = 141,696 tokens),
    # check local server process cmdline (if running on localhost) to warn on --max-seq-len < 143360
    if [[ "${need_estonia_seq_len}" == "1" ]]; then
        local max_seq_len=""
        max_seq_len="$(ps -eo args 2>/dev/null | grep -E '(spark|flux-server).*--model' | grep -v grep | sed -n 's/.*--max-seq-len[= ]\([0-9]*\).*/\1/p' | head -n 1 || true)"
        if [[ -n "${max_seq_len}" && "${max_seq_len}" -lt 143360 ]]; then
            echo "[lil-bench] WARNING: Running --test-profile estonia (136,576 prompt tokens + 5,120 output tokens)" >&2
            echo "  against a server started with --max-seq-len ${max_seq_len} (< 143360)." >&2
            echo "  To avoid context-overflow rejection, launch with FLUX_MODE=long-context-136k:" >&2
            echo "    source scripts/flux_env_bridge.sh --mode long-context-136k" >&2
            echo "    # and pass --max-seq-len 147456 --request-timeout 1800" >&2
        fi
    fi
}

PROFILE="${1:-matrix}"
EXTRA_ARGS=()
NEED_ESTONIA_CHECK=0
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi

case "${PROFILE}" in
    matrix)
        shift || true
        OUT_FILE="${OUTPUT_DIR}/lil-matrix-${STAMP}.json"
        EXTRA_ARGS=(
            --contexts "0,8192,16384,32768,65536,131072"
            --prefill-contexts "8k,16k,32k,64k,128k"
            --concurrency "1"
            --duration 30
            --max-tokens 2048
            --output "${OUT_FILE}"
            "$@"
        )
        ;;
    prefill)
        shift || true
        OUT_FILE="${OUTPUT_DIR}/lil-prefill-${STAMP}.json"
        EXTRA_ARGS=(
            --prefill-only
            --standalone-prefill
            --prefill-contexts "8k,16k,32k,64k,128k"
            --display-mode plain
            --output "${OUT_FILE}"
            "$@"
        )
        ;;
    estonia)
        shift || true
        NEED_ESTONIA_CHECK=1
        OUT_FILE="${OUTPUT_DIR}/lil-estonia-${STAMP}.json"
        EXTRA_ARGS=(
            --test-profile estonia
            --profile-concurrency 1
            --profile-runs 5
            --max-tokens 5120
            --completion-stats-seed 20261016
            --completion-stats-stall-timeout 600
            --display-mode plain
            --output "${OUT_FILE}"
            "$@"
        )
        ;;
    coding|burst)
        shift || true
        OUT_FILE="${OUTPUT_DIR}/lil-coding-${STAMP}.json"
        EXTRA_ARGS=(
            --skip-prefill
            --contexts "0"
            --concurrency "1"
            --request-count 5
            --warmup-request-count 1
            --max-tokens 1024
            --display-mode plain
            --output "${OUT_FILE}"
            "$@"
        )
        ;;
    repetition|needle-checksum)
        shift || true
        OUT_FILE="${OUTPUT_DIR}/lil-needle-checksum-${STAMP}.json"
        EXTRA_ARGS=(
            --test-profile needle-checksum
            --display-mode plain
            --output "${OUT_FILE}"
            "$@"
        )
        ;;
    hotel-lights)
        shift || true
        OUT_FILE="${OUTPUT_DIR}/lil-hotel-lights-${STAMP}.json"
        EXTRA_ARGS=(
            --test-profile hotel-lights
            --display-mode plain
            --output "${OUT_FILE}"
            "$@"
        )
        ;;
    gsm8k)
        shift || true
        OUT_FILE="${OUTPUT_DIR}/lil-gsm8k-${STAMP}.json"
        EXTRA_ARGS=(
            --test-profile gsm8k
            --display-mode plain
            --output "${OUT_FILE}"
            "$@"
        )
        ;;
    mmlu-pro)
        shift || true
        OUT_FILE="${OUTPUT_DIR}/lil-mmlu-pro-${STAMP}.json"
        EXTRA_ARGS=(
            --test-profile mmlu-pro
            --display-mode plain
            --output "${OUT_FILE}"
            "$@"
        )
        ;;
    gpqa|gpqa-diamond)
        shift || true
        OUT_FILE="${OUTPUT_DIR}/lil-gpqa-diamond-${STAMP}.json"
        EXTRA_ARGS=(
            --test-profile gpqa-diamond
            --display-mode plain
            --output "${OUT_FILE}"
            "$@"
        )
        ;;
    tool-eval)
        shift || true
        OUT_FILE="${OUTPUT_DIR}/lil-tool-eval-${STAMP}.json"
        EXTRA_ARGS=(
            --tool-eval
            --output "${OUT_FILE}"
            "$@"
        )
        ;;
    *)
        # Raw passthrough to llm_decode_bench.py
        if [[ "$*" == *"estonia"* ]]; then
            NEED_ESTONIA_CHECK=1
        fi
        EXTRA_ARGS=("$@")
        ;;
esac

LIL_SCRIPT="$(resolve_lil_script)"
preflight_server "${NEED_ESTONIA_CHECK}"

CMD=(
    python3 "${LIL_SCRIPT}"
    --host "${HOST}"
    --port "${PORT}"
    --model "${MODEL}"
    "${EXTRA_ARGS[@]}"
)

echo "[lil-bench] Endpoint: http://${HOST}:${PORT} (model: ${MODEL})" >&2
echo "[lil-bench] Command:  ${CMD[*]}" >&2

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

exec "${CMD[@]}"
