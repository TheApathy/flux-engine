#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
# runJ oracle capture: runI's prompt (1044 tok = 512+512+20 encoder chunks),
# checkpoint numerics (DENSE_FP4=off, bf16 head), SWA replay ON, fp32-P torch attention, FULL
# taps at layers 2,20,24,28,32,36 for the prefill AND the first 8 greedy DECODE steps
# (--tap-decode 8: non-speculative decode, so each step's T=1 forward is tapped).
# Closes: replay indexers L32/L36, the 3-chunk L20 tail with mixed candidate widths, and the
# decode gate's data. NEVER co-load: preflight refuses if a model is resident or < 90 GB free;
# a watchdog kills the container below 12 GB. capture_ref.py writes the manifest LAST.
Q="${FLUX_QUEUE_LOG:-/dev/null}"
ORACLE_DIR="${DSV41_ORACLE_DIR:?Set DSV41_ORACLE_DIR}"
OUT="${ORACLE_DIR}/ref/runJ_decode"
NEED_KB=$((90 * 1024 * 1024)); ABORT_KB=$((12 * 1024 * 1024))   # Guardrail: >= 12 GB free (runJ low-water ~15 GB)
memavail() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
echo "$(date -u +%FT%TZ) dsv41-attention runJ capture (runI prompt + 8 tapped decode steps, L2/20/24/28/32/36; python engine, K124 arena ~72GB, ~15min) QUEUED pid=$$" >> "$Q"
[ -e "$OUT/manifest.json" ] && { echo "DONE runJ rc=96 ($OUT already complete; refusing to overwrite)"; exit 96; }

# preflight BEFORE queueing on the lock: refuse if another model container is resident or < 90 GB free.
others=$(docker ps --format '{{.Names}}' | grep -Ei 'dsv41|atlas|spark|vllm' | grep -v buildkit || true)
avail=$(memavail)
echo "preflight (queue time): MemAvailable $((avail/1024/1024)) GB; containers: [${others}]"
if [ -n "$others" ] || [ "$avail" -lt "$NEED_KB" ]; then
  echo "REFUSING runJ: MemAvailable=$((avail/1024/1024))GB (need 90GB) containers=[$others]" >&2
  exit 97
fi

export CONTAINER_NAME="dsv41-attention-runJ-$$"
export EXTRA_DOCKER_ARGS="-e DSV41_DEV_SKIP_VERIFY=1 -e DSV41_MEM_FLOOR_GB=5.0 -e DSV41_DENSE_FP4=off -e DSV41_HEAD_FMT=bf16 -e DSV41_SWA_REPLAY=1 -e DSV41_PREFILL_CHUNK=512 --mount type=bind,src=${ORACLE_DIR},dst=/oracle"
# in_image.sh takes the GPU flock itself for exactly the container's lifetime. This watcher writes
# START once the container runs (= lock held), then samples MemAvailable every 0.5 s: it records the
# low-water mark and KILLS the container below 12 GB. It also re-checks the preflight at START,
# because the lock -- not queue position -- decides when we actually run.
LOWF=$(mktemp)
( until docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; do sleep 1; kill -0 $$ 2>/dev/null || exit 0; done
  avail=$(memavail)
  start_others=$(docker ps --format '{{.Names}}' | grep -Ei 'dsv41|atlas|spark|vllm' | grep -v buildkit | grep -vx "$CONTAINER_NAME" || true)
  echo "$(date -u +%FT%TZ) dsv41-attention runJ window START (lock held, container running, MemAvailable $((avail/1024/1024)) GB, others=[$start_others]) pid=$$" >> "$Q"
  if [ -n "$start_others" ] || [ "$avail" -lt "$NEED_KB" ]; then
    echo "$(date -u +%FT%TZ) dsv41-attention runJ START PREFLIGHT REFUSED MemAvailable=$((avail/1024/1024))GB others=[$start_others] -> docker kill" >> "$Q"
    docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
    exit 0
  fi
  low=$avail
  while docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; do
    m=$(memavail); [ "$m" -lt "$low" ] && low=$m; echo "$low" > "$LOWF"
    if [ "$m" -lt "$ABORT_KB" ]; then
      echo "$(date -u +%FT%TZ) dsv41-attention runJ WATCHDOG MemAvailable $((m/1024)) MB -> docker kill" >> "$Q"
      docker kill "$CONTAINER_NAME" >/dev/null 2>&1
    fi
    sleep 0.5
  done ) &
WATCH=$!

"${DSV41_REF_IMAGE_RUNNER:?Set DSV41_REF_IMAGE_RUNNER}" /oracle/capture_ref.py \
  --out /oracle/ref/runJ_decode --attn torch --tap-logits --tap-decode 8 \
  --prompt-ids-json /oracle/runG_plus20.json --prompt-name runG_plus20 \
  --max-prompt-tokens 1044 --max-seq 8192 --layers 2,20,24,28,32,36 && rc=0 || rc=$?
kill "$WATCH" 2>/dev/null || true
low=$(cat "$LOWF" 2>/dev/null || echo 0); rm -f "$LOWF"
# Post-check: the decode steps must actually have been tapped (the engine's tap hook only fires
# on paths that reach Model.attention). L02: 3 prefill chunks (.000-.002) + 8 steps (.003-.010);
# L24: 1 replay (.000) + 8 steps (.001-.008).
chk="manifest=$([ -e "$OUT/manifest.json" ] && echo yes || echo NO)"
for f in L02.topk.010 L02.attn_o_pre_inverse_rope.010 L24.topk.008 L36.attn_o_pre_inverse_rope.008 L20.cand_out.010; do
  chk="$chk $f=$([ -e "$OUT/$f.bin" ] && echo yes || echo NO)"
done
echo "$(date -u +%FT%TZ) dsv41-attention runJ window END rc=$rc lowwater_GB=$((low/1024/1024)) [$chk] pid=$$" >> "$Q"
echo "DONE runJ rc=$rc lowwater_GB=$((low/1024/1024)) $chk"
exit "$rc"
