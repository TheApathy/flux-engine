#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="${FLUX_WORKTREE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TARGET_DIR="${CARGO_TARGET_DIR:-$WORKTREE/target}"
BIN="$TARGET_DIR/release/flux-server"
MODEL_DIR="${FLUX_MODEL_DIR:-${HOME}/models/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid}"
QUAL_DIR="$WORKTREE/qualification/accuracy"
REPORTS_DIR="$QUAL_DIR/reports-smoke"
PORT=8899

mkdir -p "$QUAL_DIR" "$REPORTS_DIR"

echo "=== Milestone 3: Full Accuracy Parity Smoke Check Run ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Export CUDA environment
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="/usr/local/cuda-13.0/bin:$PATH"

SERVER_PID=""
MEM_SAMPLER_PID=""

cleanup() {
    echo "[*] Running cleanup trap..."
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[*] Terminating server PID $SERVER_PID..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in {1..30}; do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[!] Force killing server PID $SERVER_PID..."
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
    fi
    if [[ -n "$MEM_SAMPLER_PID" ]] && kill -0 "$MEM_SAMPLER_PID" 2>/dev/null; then
        kill -TERM "$MEM_SAMPLER_PID" 2>/dev/null || true
    fi
    # Reclaim memory
    echo "[*] Waiting for memory reclamation (MemAvailable >= 100 GiB)..."
    for _ in {1..60}; do
        ma_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        ma_gib=$(( ma_kib / 1048576 ))
        if (( ma_gib >= 100 )); then break; fi
        sleep 1
    done
    echo "[+] Memory reclaimed: ${ma_gib} GiB available."
}

trap cleanup EXIT

# 1. Acquire GPU Lock
LOCK_FILE="${FLUX_LOCK_FILE:-${TMPDIR:-/tmp}/flux-gpu.lock}"
echo "[*] Waiting for GPU lock on ${LOCK_FILE}..."
exec 9>"$LOCK_FILE"
flock -w 7200 9
echo "[+] Acquired GPU lock."

# 2. Check Pre-flight Box Safety (MemAvailable >= 100 GiB)
echo "[*] Checking pre-flight MemAvailable..."
mem_avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
mem_avail_gib=$(( mem_avail_kib / 1048576 ))
echo "    Current MemAvailable: ${mem_avail_gib} GiB (${mem_avail_kib} KiB)"
if (( mem_avail_gib < 100 )); then
    echo "[-] ERROR: MemAvailable (${mem_avail_gib} GiB) < 100 GiB! Waiting..."
    while true; do
        sleep 5
        mem_avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        mem_avail_gib=$(( mem_avail_kib / 1048576 ))
        if (( mem_avail_gib >= 100 )); then
            echo "[+] MemAvailable reached ${mem_avail_gib} GiB."
            break
        fi
    done
fi

# 3. Start background memory sampler (close fd 9 so child does not hold GPU flock)
MEM_CSV="$REPORTS_DIR/memory_samples.csv"
echo "timestamp,mem_available_mib" > "$MEM_CSV"
(
    while true; do
        ts=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
        ma_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        ma_mib=$(( ma_kib / 1024 ))
        echo "$ts,$ma_mib" >> "$MEM_CSV"
        sleep 0.25
    done
) 9>&- &
MEM_SAMPLER_PID=$!
echo "[+] Memory sampler started (PID: $MEM_SAMPLER_PID)."

wait_for_health() {
    local max_wait=300
    local elapsed=0
    echo "[*] Waiting for server at http://127.0.0.1:$PORT/health to become ready..."
    while (( elapsed < max_wait )); do
        if curl -s -f "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            echo "[+] Server is ready (elapsed: ${elapsed}s)."
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[-] Server died unexpectedly during startup!"
            return 1
        fi
        sleep 2
        elapsed=$(( elapsed + 2 ))
    done
    echo "[-] Timed out waiting for server to become ready."
    return 1
}

# -----------------------------------------------------------------------------
# Arm 1: Plain (no speculation, K=1)
# -----------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "=== Starting Arm 1: Plain (no speculation, K=1) ==="
echo "========================================================================="

set -a
# shellcheck disable=SC1090
source "$QUAL_DIR/recipe-plain.env"
set +a

SERVER_LOG_PLAIN="$QUAL_DIR/server_plain.log"
echo "[*] Launching flux-server (Plain arm)..."

"$BIN" serve \
  --no-tui \
  --bind 127.0.0.1 \
  --port "$PORT" \
  --model-from-path "$MODEL_DIR" \
  --model-name qwen38 \
  --kernel-target qwen3.8-flash-next \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype bf16 \
  --block-size 16 \
  --max-seq-len 262144 \
  --max-prefill-tokens 4096 \
  --max-batch-size 1 \
  --max-num-seqs 1 \
  --ssm-cache-slots 0 \
  --swap-space-gb 0 \
  --qwen4-qsa \
  --oom-guard-mb 4096 \
  --request-timeout 14340 \
  --disable-simhash-watchdog \
  --enable-prefix-caching false > "$SERVER_LOG_PLAIN" 2>&1 9>&- &
SERVER_PID=$!
echo "[+] Server launched with PID $SERVER_PID."

wait_for_health

echo "[*] Running Plain GSM8K (4 items)..."
python3 "$WORKTREE/repro-kit/lil-bench" accuracy gsm8k \
  --concurrency 1 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model qwen38 \
  --output "$REPORTS_DIR/plain-gsm8k.json" \
  --profile-runs 4

echo "[*] Running Plain MMLU-Pro (4 items)..."
python3 "$WORKTREE/repro-kit/lil-bench" accuracy mmlu-pro \
  --concurrency 1 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model qwen38 \
  --output "$REPORTS_DIR/plain-mmlu-pro.json" \
  --profile-runs 4

# Check metrics
PLAIN_SPEC_METRIC=$(curl -s "http://127.0.0.1:$PORT/metrics" | grep 'atlas_spec_decode_verify_total' || true)
echo "[*] Plain arm spec decode verify metrics: $PLAIN_SPEC_METRIC"

echo "[*] Stopping Plain server PID $SERVER_PID..."
kill -TERM "$SERVER_PID" 2>/dev/null || true
for _ in {1..30}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 1
done
if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[!] Force killing Plain server PID $SERVER_PID..."
    kill -9 "$SERVER_PID" 2>/dev/null || true
fi
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

echo "[*] Waiting for memory reclamation after Plain arm..."
while true; do
    ma_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    ma_gib=$(( ma_kib / 1048576 ))
    if (( ma_gib >= 100 )); then break; fi
    sleep 2
done
echo "[+] Memory reclaimed: ${ma_gib} GiB available."

# -----------------------------------------------------------------------------
# Arm 2: Native-MTP K=5
# -----------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "=== Starting Arm 2: Native-MTP K=5 ==="
echo "========================================================================="

set -a
# shellcheck disable=SC1090
source "$QUAL_DIR/recipe-k5.env"
set +a

SERVER_LOG_K5="$QUAL_DIR/server_k5.log"
echo "[*] Launching flux-server (Native-MTP K=5 arm)..."

"$BIN" serve \
  --no-tui \
  --bind 127.0.0.1 \
  --port "$PORT" \
  --model-from-path "$MODEL_DIR" \
  --model-name qwen38 \
  --kernel-target qwen3.8-flash-next \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype bf16 \
  --block-size 16 \
  --max-seq-len 262144 \
  --max-prefill-tokens 4096 \
  --max-batch-size 1 \
  --max-num-seqs 1 \
  --ssm-cache-slots 0 \
  --swap-space-gb 0 \
  --qwen4-qsa \
  --oom-guard-mb 4096 \
  --request-timeout 14340 \
  --disable-simhash-watchdog \
  --enable-prefix-caching false \
  --speculative \
  --num-drafts 4 \
  --mtp-vocab 0 \
  --mtp-quantization bf16 > "$SERVER_LOG_K5" 2>&1 9>&- &
SERVER_PID=$!
echo "[+] Server launched with PID $SERVER_PID."

wait_for_health

echo "[*] Running Native-MTP K=5 GSM8K (4 items)..."
python3 "$WORKTREE/repro-kit/lil-bench" accuracy gsm8k \
  --concurrency 1 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model qwen38 \
  --output "$REPORTS_DIR/k5-gsm8k.json" \
  --profile-runs 4

echo "[*] Running Native-MTP K=5 MMLU-Pro (4 items)..."
python3 "$WORKTREE/repro-kit/lil-bench" accuracy mmlu-pro \
  --concurrency 1 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model qwen38 \
  --output "$REPORTS_DIR/k5-mmlu-pro.json" \
  --profile-runs 4

# Check metrics
K5_SPEC_METRIC=$(curl -s "http://127.0.0.1:$PORT/metrics" | grep 'atlas_spec_decode_verify_total' || true)
echo "[*] K5 arm spec decode verify metrics: $K5_SPEC_METRIC"

echo "[*] Stopping K5 server PID $SERVER_PID..."
kill -TERM "$SERVER_PID" 2>/dev/null || true
for _ in {1..30}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 1
done
if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[!] Force killing K5 server PID $SERVER_PID..."
    kill -9 "$SERVER_PID" 2>/dev/null || true
fi
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

# Stop memory sampler
if [[ -n "$MEM_SAMPLER_PID" ]] && kill -0 "$MEM_SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$MEM_SAMPLER_PID" 2>/dev/null || true
    MEM_SAMPLER_PID=""
fi

# Wait for memory reclamation
echo "[*] Waiting for final memory reclamation..."
while true; do
    ma_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    ma_gib=$(( ma_kib / 1048576 ))
    if (( ma_gib >= 100 )); then break; fi
    sleep 2
done
echo "[+] Memory reclaimed: ${ma_gib} GiB available."

# Release GPU lock
echo "[*] Releasing GPU lock..."
exec 9>&-
echo "[+] GPU lock released."

echo ""
echo "=== Smoke Check Complete. Now verifying parity & zero intervention ledger ==="
python3 "$QUAL_DIR/verify_smoke_results.py" --reports-dir "$REPORTS_DIR" --memory-csv "$MEM_CSV"
