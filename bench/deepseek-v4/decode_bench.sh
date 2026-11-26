#!/usr/bin/env bash
# Measure single-stream decode throughput against a running Atlas serve.
#
# Decode tok/s here is the number the DeepSeek-V4 single-node work is judged on
# (the ds4-on-spark reference does ~20 plain / 27.7 DSpark-mean on one Spark).
# The server already reports tok/s and TTFT per request in its own log line, so
# this just drives a fixed-length greedy completion and echoes the usage block.
#
#   PORT=8899 NTOK=256 bash bench/deepseek-v4/decode_bench.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
NTOK="${NTOK:-256}"
PROMPT="${PROMPT:-Write a detailed technical explanation of how virtual memory paging works in a modern operating system.}"
MODELS_DIR="${FLUX_MODELS_DIR:-${HOME}/models}"
MODEL="${MODEL:-${MODELS_DIR}/DeepSeek-V4-Flash-162B}"

req() {
  curl -s -m 600 "http://$HOST:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c '
import json,os,sys
print(json.dumps({
  "model": os.environ["MODEL"],
  "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
  "max_tokens": int(os.environ["NTOK"]),
  "temperature": 0.0,
  "stream": False,
}))' )"
}

export MODEL PROMPT NTOK

# The server logs its own (prefill-excluded) tok/s per request. Remember where
# the log ends BEFORE the runs: reading the whole file and tailing it printed a
# stale line from an earlier serve whenever LOG pointed at a log this run never
# appended to, which silently passed off an old number as "authoritative".
SERVE_LOG="${LOG:-${REPO}/serve-deepseek-single.log}"
LOG_MARK=0
[ -f "$SERVE_LOG" ] && LOG_MARK=$(wc -c <"$SERVE_LOG")

# One warm-up: the first request after boot pays lazy-init and (when graphs are
# on) the capture pass, neither of which belongs in a steady-state number.
echo "warmup..." >&2
req >/dev/null

for i in $(seq 1 "${REPS:-3}"); do
  t0=$(date +%s.%N)
  out=$(req)
  t1=$(date +%s.%N)
  echo "$out" | ELAPSED="$(echo "$t1 - $t0" | bc)" python3 -c '
import json,sys,os
t=float(os.environ["ELAPSED"])
raw=sys.stdin.read()
try:
    d=json.loads(raw)
except Exception:
    print("  [FAIL] non-JSON response:", raw[:200]); sys.exit(0)
if "error" in d: print("  [FAIL]", str(d["error"])[:200]); sys.exit(0)
u=d.get("usage",{})
ct=u.get("completion_tokens",0)
print(f"  run: {ct} tok in {t:.2f}s = {ct/t:.1f} tok/s (wall, incl. prefill)")
'
done

echo "-- server-reported (excludes HTTP + prefill), from $SERVE_LOG:"
if [ ! -f "$SERVE_LOG" ]; then
  echo "  [no serve log at $SERVE_LOG — set LOG= to the log this serve is writing]"
else
  lines=$( { tail -c "+$((LOG_MARK + 1))" "$SERVE_LOG" | grep -a "tok/s, TTFT" || true; } | tail -n "${REPS:-3}")
  if [ -z "$lines" ]; then
    echo "  [no new completion lines since this run started — wrong LOG for the running serve?]"
  else
    echo "$lines"
  fi
fi
