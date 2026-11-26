#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
# GPU job: whole attention block (projections + sparse attention core) through the
# driver on runH_feed40, layers 0..20, MoE fed. Arm A: core fed from the capture (baseline);
# arm B: the real core. Positive DONE line with both exit codes.
ATLAS_ROOT="${ATLAS_ROOT:-$(git rev-parse --show-toplevel)}"
Q="${FLUX_QUEUE_LOG:-/dev/null}"
LOCK_FILE="${FLUX_GPU_LOCK:-}"
BIN="${ATLAS_ROOT}/target/release/examples/dsv41_forward"
OUT="${OUT_DIR:-$(mktemp -d)}"
[ -x "$BIN" ] || { echo "DONE rc=127 (binary missing)"; exit 127; }
rm -rf -- "$OUT/fed" "$OUT/real"
mkdir -p "$OUT/fed" "$OUT/real"
echo "$(date -u +%FT%TZ) dsv41-attention whole-block gate (21 layers dense, MoE fed, 2 arms, ~3 min) QUEUED pid=$$" >> "$Q"
if [ -n "$LOCK_FILE" ]; then exec 9>"$LOCK_FILE"; flock 9; fi
echo "$(date -u +%FT%TZ) dsv41-attention whole-block gate START pid=$$" >> "$Q"
"$BIN" --run runH_feed40 --layers 21 --feed attn-core,moe --engram live --core fed  --tap-dir "$OUT/fed" > "$OUT/fed.log" 2>&1 && rc1=0 || rc1=$?
"$BIN" --run runH_feed40 --layers 21 --feed attn-core,moe --engram live --core real --tap-dir "$OUT/real" > "$OUT/real.log" 2>&1 && rc2=0 || rc2=$?
echo "$(date -u +%FT%TZ) dsv41-attention whole-block gate END rc=$rc1,$rc2 pid=$$" >> "$Q"
if [ -n "${LOCK_FILE:-}" ]; then flock -u 9; fi
echo "DONE rc_fed=$rc1 rc_real=$rc2"
[ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]
