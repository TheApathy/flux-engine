#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="${FLUX_WORKTREE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TARGET_DIR="${CARGO_TARGET_DIR:-$WORKTREE/target}"
MODEL_DIR="${FLUX_MODEL_DIR:-${HOME}/models/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid}"
QUAL_DIR="$WORKTREE/qualification/accuracy"
REPORTS_DIR="$QUAL_DIR/reports"
PORT=8899
IMAGE_TAG="flux-engine:r17-candidate"

CONCURRENCY_PLAIN="${CONCURRENCY_PLAIN:-8}"
PROFILE_RUNS_OVERRIDE="${PROFILE_RUNS_OVERRIDE:-}"

mkdir -p "$QUAL_DIR" "$REPORTS_DIR"

echo "=== Step C: Full Accuracy Qualification (Plain K=1 Arm — Fast-Track Scope) ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Concurrency Plain: $CONCURRENCY_PLAIN (K=5 arm omitted in fast-track scope)"
echo "Image Tag: $IMAGE_TAG"

CONTAINER_NAME=""
MEM_SAMPLER_PID=""

cleanup() {
    echo "[*] Running cleanup trap..."
    if [[ -n "$CONTAINER_NAME" ]]; then
        echo "[*] Stopping container $CONTAINER_NAME..."
        docker stop -t 10 "$CONTAINER_NAME" 2>/dev/null || true
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    fi
    docker rm -f test-step-c flux-step-c-plain flux-step-c-k5 2>/dev/null || true
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

# Clean any existing containers
docker rm -f test-step-c flux-step-c-plain flux-step-c-k5 2>/dev/null || true

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
MEM_CSV="$QUAL_DIR/memory_samples.csv"
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
    echo "[*] Waiting for container server at http://127.0.0.1:$PORT/health to become ready..."
    sleep 3
    while (( elapsed < max_wait )); do
        if curl -s -f "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            echo "[+] Container server is ready (elapsed: ${elapsed}s)."
            return 0
        fi
        if ! docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
            echo "[-] Container died unexpectedly during startup! Logs:"
            docker logs "$CONTAINER_NAME" | tail -n 50 || true
            return 1
        fi
        sleep 2
        elapsed=$(( elapsed + 2 ))
    done
    echo "[-] Timed out waiting for container server to become ready."
    return 1
}

EXTRA_LIL_ARGS=( "--max-tokens" "16384" )
if [[ -n "$PROFILE_RUNS_OVERRIDE" ]]; then
    EXTRA_LIL_ARGS+=( "--profile-runs" "$PROFILE_RUNS_OVERRIDE" )
fi

# -----------------------------------------------------------------------------
# Arm 1: Plain (no speculation, K=1)
# -----------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "=== Starting Arm 1: Plain (no speculation, K=1) in Container ==="
echo "========================================================================="

CONTAINER_NAME="flux-step-c-plain"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v "${MODEL_DIR}:/models:ro" \
  --env-file "$QUAL_DIR/recipe-plain.env" \
  --entrypoint /opt/flux/bin/flux-server-real \
  "$IMAGE_TAG" \
  serve \
    --no-tui \
    --bind 127.0.0.1 \
    --port "$PORT" \
    --model-from-path /models \
    --model-name qwen38 \
    --kernel-target qwen3.8-flash-next \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype bf16 \
    --block-size 16 \
    --max-seq-len 32768 \
    --max-prefill-tokens 4096 \
    --max-batch-size 8 \
    --max-num-seqs 8 \
    --ssm-cache-slots 0 \
    --swap-space-gb 0 \
    --qwen4-qsa \
    --oom-guard-mb 4096 \
    --request-timeout 14340 \
    --disable-simhash-watchdog \
    --disable-loop-watchdog \
    --disable-thinking-loop-watchdog \
    --disable-confidence-early-stop \
    --enable-prefix-caching false

wait_for_health

echo "[*] Running Plain GSM8K (concurrency: $CONCURRENCY_PLAIN)..."
python3 "$WORKTREE/repro-kit/lil-bench" accuracy gsm8k \
  --concurrency "$CONCURRENCY_PLAIN" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model qwen38 \
  --output "$REPORTS_DIR/plain-gsm8k.json" \
  "${EXTRA_LIL_ARGS[@]}"

echo "[*] Running Plain MMLU-Pro (concurrency: $CONCURRENCY_PLAIN)..."
python3 "$WORKTREE/repro-kit/lil-bench" accuracy mmlu-pro \
  --concurrency "$CONCURRENCY_PLAIN" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model qwen38 \
  --output "$REPORTS_DIR/plain-mmlu-pro.json" \
  "${EXTRA_LIL_ARGS[@]}"

# Save Plain server logs
docker logs "$CONTAINER_NAME" > "$QUAL_DIR/server_plain.log" 2>&1 || true

echo "[*] Stopping Plain container $CONTAINER_NAME..."
docker stop -t 15 "$CONTAINER_NAME" 2>/dev/null || true
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
CONTAINER_NAME=""

echo "[*] Waiting for memory reclamation after Plain arm..."
while true; do
    ma_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    ma_gib=$(( ma_kib / 1048576 ))
    if (( ma_gib >= 100 )); then break; fi
    sleep 2
done
echo "[+] Memory reclaimed: ${ma_gib} GiB available."

# Stop memory sampler
if [[ -n "$MEM_SAMPLER_PID" ]] && kill -0 "$MEM_SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$MEM_SAMPLER_PID" 2>/dev/null || true
    wait "$MEM_SAMPLER_PID" 2>/dev/null || true
    MEM_SAMPLER_PID=""
fi

# Release GPU lock
echo "[*] Releasing GPU lock..."
exec 9>&-
echo "[+] GPU lock released."

echo "[+] Plain arm (GSM8K 1319 + MMLU-Pro 1000) complete (Plain-only fast-track scope; K=5 arm omitted)."
exit 0
