#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="${FLUX_WORKTREE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGET_DIR="${CARGO_TARGET_DIR:-$WORKTREE/target}"
BIN="${FLUX_SERVER_BIN:-$TARGET_DIR/release/flux-server}"
MODEL_DIR="${FLUX_MODEL_DIR:?Set FLUX_MODEL_DIR to the Qwen3.8-Flash-Next checkpoint directory}"
RECIPE="${FLUX_RECIPE:-$WORKTREE/repro-kit/recipes/flux-qwen4-native-262k-w4096-bf16-owner16.env}"
QUAL_DIR="${FLUX_QUAL_DIR:-$WORKTREE/qualification/native262k}"
LOCK_FILE="${FLUX_LOCK_FILE:-${TMPDIR:-/tmp}/flux-gpu.lock}"
PORT="${FLUX_PORT:-8899}"

mkdir -p "$QUAL_DIR"

echo "=== Milestone 2: Native 262K Qualification Run ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Acquire GPU Lock
exec 9>"$LOCK_FILE"
echo "[*] Waiting for GPU lock..."
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

# 3. Source certified recipe
echo "[*] Sourcing recipe: $RECIPE"
set -a
# shellcheck disable=SC1090
source "$RECIPE"
set +a

# 4. Start memory sampler
echo "[*] Starting background memory sampler..."
MEM_CSV="$QUAL_DIR/memory_samples.csv"
rm -f "$MEM_CSV"
echo "timestamp,mem_available_mib" > "$MEM_CSV"
(
    while true; do
        ts=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
        ma_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        ma_mib=$(( ma_kib / 1024 ))
        echo "$ts,$ma_mib" >> "$MEM_CSV"
        sleep 0.25
    done
) &
MEM_SAMPLER_PID=$!
echo "    Memory sampler PID: $MEM_SAMPLER_PID"

SERVER_PID=""
cleanup() {
    echo -e "\n[*] Running teardown..."
    if [[ -n "$MEM_SAMPLER_PID" ]] && kill -0 "$MEM_SAMPLER_PID" 2>/dev/null; then
        kill "$MEM_SAMPLER_PID" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[*] Terminating flux-server (PID $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[!] Force killing flux-server..."
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "[*] Verifying memory reclamation..."
    for _ in $(seq 1 45); do
        post_avail=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
        if (( post_avail >= 100 )); then
            echo "[+] Memory reclaimed: ${post_avail} GiB MemAvailable."
            break
        fi
        sleep 1
    done
}
trap cleanup EXIT

# 5. Launch flux-server
echo "[*] Launching flux-server on port $PORT..."
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
  --request-timeout 900 \
  --disable-simhash-watchdog \
  --enable-prefix-caching false > "$QUAL_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "    flux-server PID: $SERVER_PID"

# 6. Poll readiness
echo "[*] Polling http://127.0.0.1:$PORT/health..."
ready=0
for i in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[-] ERROR: flux-server exited prematurely!" >&2
        tail -n 60 "$QUAL_DIR/server.log" >&2
        exit 1
    fi
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[+] flux-server is ready in ${i}s!"
        ready=1
        break
    fi
    sleep 1
done

if (( ready != 1 )); then
    echo "[-] ERROR: Timed out waiting for flux-server readiness!" >&2
    tail -n 60 "$QUAL_DIR/server.log" >&2
    exit 1
fi

# 7. Execute 262K Qualification Battery
echo "[*] Running 262K test battery..."
python3 "$WORKTREE/repro-kit/m2_262k_harness.py" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model-name qwen38 \
  --output-dir "$QUAL_DIR"

echo "[+] 262K test battery completed successfully."

# Teardown will run via EXIT trap
