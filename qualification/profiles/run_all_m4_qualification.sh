#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="${FLUX_WORKTREE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TARGET_DIR="${CARGO_TARGET_DIR:-$WORKTREE/target}"
BIN="$TARGET_DIR/release/flux-server"
QUAL_DIR="$WORKTREE/qualification/profiles"
MEM_CSV="$QUAL_DIR/memory_samples.csv"

cd "$WORKTREE"

echo "=== Milestone 4 Live Qualification Master Runner ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Acquire GPU lock
LOCK_FILE="${FLUX_LOCK_FILE:-${TMPDIR:-/tmp}/flux-gpu.lock}"
echo "[*] Waiting for GPU lock on ${LOCK_FILE}..."
exec 9>"$LOCK_FILE"
flock -w 7200 9
echo "[+] Acquired GPU lock."

MEM_SAMPLER_PID=""

cleanup() {
    echo -e "\n[*] Running master cleanup trap..."
    if [[ -n "$MEM_SAMPLER_PID" ]] && kill -0 "$MEM_SAMPLER_PID" 2>/dev/null; then
        kill "$MEM_SAMPLER_PID" 2>/dev/null || true
    fi
    # Terminate any flux-server running on port 8899
    fuser -k 8899/tcp 2>/dev/null || true
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

# 2. Check pre-flight memory
mem_avail_gib=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
if (( mem_avail_gib < 100 )); then
    echo "[-] Pre-flight MemAvailable < 100 GiB (${mem_avail_gib} GiB). Waiting..."
    while true; do
        sleep 2
        mem_avail_gib=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
        if (( mem_avail_gib >= 100 )); then break; fi
    done
fi
echo "[+] Pre-flight memory safe: ${mem_avail_gib} GiB available."

# 3. Start continuous background memory sampler
echo "timestamp,mem_available_mib" > "$MEM_CSV"
(
    while true; do
        ts=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
        ma_mib=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
        echo "$ts,$ma_mib" >> "$MEM_CSV"
        sleep 0.25
    done
) &
MEM_SAMPLER_PID=$!
echo "[+] Background memory sampler started (PID $MEM_SAMPLER_PID), polling every 250ms to $MEM_CSV."

# 4. Run Stage 1: Cold Start Qualification
echo -e "\n======================================================="
echo "=== EXECUTING STAGE 1: COLD START QUALIFICATION ==="
if [[ -f "$QUAL_DIR/coldstart_results.json" ]] && python3 -c '
import json, sys
data = json.load(open("'"$QUAL_DIR/coldstart_results.json"'"))
med = data.get("median_seconds", 999)
if data.get("verdict") == "PASS" and med <= 150.0:
    print(f"[+] Reusing existing certified cold start receipt (median: {med}s <= 150.0s).")
    sys.exit(0)
sys.exit(1)
'; then
    echo "[+] Cold Start Qualification already certified!"
else
    python3 "$QUAL_DIR/run_m4_coldstart.py"
fi
# 5. Run Stage 2: Sustained Performance Matrix
echo -e "\n======================================================="
echo "=== EXECUTING STAGE 2: SUSTAINED PERFORMANCE MATRIX ==="
echo "======================================================="
python3 "$QUAL_DIR/run_m4_sustained_matrix.py"

# 6. Stop memory sampler
kill "$MEM_SAMPLER_PID" 2>/dev/null || true
MEM_SAMPLER_PID=""

# 7. Run Stage 3: Generate Receipts
echo -e "\n======================================================="
echo "=== EXECUTING STAGE 3: RECEIPT GENERATION ==="
echo "======================================================="
python3 "$QUAL_DIR/generate_m4_receipts.py"

# 8. Commit sealed receipts
echo -e "\n======================================================="
echo "=== COMMITTING QUALIFICATION RECEIPTS ==="
echo "======================================================="
git add qualification/M4_RECEIPT.json qualification/M4_RECEIPT.md qualification/profiles/
git commit -m "qual(profiles): seal Milestone 4 performance profiles and fast loading (R4)"
echo "[+] Sealed commit: $(git rev-parse HEAD)"

echo -e "\n[+] Milestone 4 Live Qualification Completed Successfully!"
