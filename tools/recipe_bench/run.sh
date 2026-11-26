#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Launch a built-in recipe end to end and measure it: prefill at 2048/8192
# (warm median, rep 1 dropped) and decode at 256 tokens on a coding and a
# chat prompt. Writes one JSON result file per model. Includes process-group
# cleanup on exit and optional host lockfile / memory preflight guards.
#
# Usage:
#   run.sh <lane-name> <label> <recipe.yaml> <model_dir> <model_name> <bin> \
#          <prefill_req_2048.json> <prefill_req_8192.json> [port]
#
# Optional environment variables:
#   RECIPE_BENCH_LOCK_FILE   Path to flock file for shared-GPU serialization (opt-in)
#   RECIPE_BENCH_QUEUE_FILE  Path to queue log file (opt-in)
#   RECIPE_BENCH_MIN_MEM_GB  Minimum MemAvailable in GiB required before launch (default: 0 = disabled)
#   RECIPE_BENCH_WAIT_GPU    Set to 1 to wait for 0 active GPU compute apps before launch (default: 0)
#   CUDA_HOME                CUDA toolkit root (default: /usr/local/cuda)
#
# The prefill request files are pre-tokenized (`prompt_token_ids`) `/v1/completions`
# bodies with `max_tokens: 1` built with the target model's tokenizer.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Q="${RECIPE_BENCH_QUEUE_FILE:-}"
LOCK="${RECIPE_BENCH_LOCK_FILE:-}"
MIN_MEM_GB="${RECIPE_BENCH_MIN_MEM_GB:-0}"
WAIT_GPU="${RECIPE_BENCH_WAIT_GPU:-0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

LANE=${1:?lane name}
LABEL=${2:?label}
YAML=${3:?recipe yaml path}
MODEL_DIR=${4:?model dir}
MODEL_NAME=${5:?model name}
BIN=${6:?spark binary path}
REQ_2048=${7:?prefill request file, 2048 tokens}
REQ_8192=${8:?prefill request file, 8192 tokens}
PORT=${9:-8897}

OUT="$HERE/results/$LABEL"
mkdir -p "$OUT"

log_queue() {
  if [[ -n "$Q" ]]; then
    echo "$(date -u +%FT%TZ) $LANE recipe_bench:$LABEL $*" >> "$Q"
  fi
}

if [[ -n "$LOCK" ]]; then
  log_queue "(${YAML##*/}) QUEUED pid=$$"
  exec 9>"$LOCK"
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 1; }
  log_queue "window START pid=$$"
fi

# Guaranteed cleanup on every exit path: kill the whole process group the
# server runs in and wait for it to exit before returning.
#
# NOT trapping ERR: bash fires the ERR trap on any simple command returning
# nonzero (such as a routine `grep -q` with no match or a non-2xx health probe),
# which would prematurely terminate a healthy server. EXIT already covers
# normal return and explicit exits cleanly without false positives.
CURRENT_PID=""
CLEANED_UP=0
cleanup() {
  [ "$CLEANED_UP" = 1 ] && return 0
  CLEANED_UP=1
  if [ -n "$CURRENT_PID" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    kill -INT -- -"$CURRENT_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$CURRENT_PID" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$CURRENT_PID" 2>/dev/null || true
    for _ in $(seq 1 15); do kill -0 "$CURRENT_PID" 2>/dev/null || break; sleep 1; done
  fi
  if [[ "$WAIT_GPU" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    for _ in $(seq 1 30); do
      nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break
      sleep 2
    done
  fi
  log_queue "window END"
}
trap cleanup EXIT INT TERM

if [[ "$WAIT_GPU" == "1" || "$MIN_MEM_GB" -gt 0 ]]; then
  APPS=0
  MA=0
  for _ in $(seq 1 60); do
    if [[ "$WAIT_GPU" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
      APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    else
      APPS=0
    fi
    MA=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [[ "$APPS" -eq 0 && "$MA" -ge "$MIN_MEM_GB" ]]; then
      break
    fi
    sleep 5
  done
  echo "preflight apps=$APPS memavail=${MA}GB (min=${MIN_MEM_GB}GB)"
  if [[ "$APPS" -ne 0 || "$MA" -lt "$MIN_MEM_GB" ]]; then
    echo "PREFLIGHT FAILED"
    exit 2
  fi
fi

# Generate argv/env from the recipe YAML; force --port to $PORT so the
# health-check target always matches the launched server.
python3 "$HERE/generate.py" "$YAML" "$MODEL_DIR" "$MODEL_NAME" env > "$OUT/gen_env.txt" \
  || { echo "ABORT: generate.py env failed" >&2; exit 3; }
python3 "$HERE/generate.py" "$YAML" "$MODEL_DIR" "$MODEL_NAME" argv "port=$PORT" > "$OUT/gen_argv.txt" \
  || { echo "ABORT: generate.py argv failed" >&2; exit 3; }
if ! grep -A1 -- "--port" "$OUT/gen_argv.txt" | tail -1 | grep -qx "$PORT"; then
  echo "ABORT: generated argv does not carry --port $PORT"; cat "$OUT/gen_argv.txt"; exit 3
fi

mapfile -t GEN_ENV < "$OUT/gen_env.txt"
mapfile -t GEN_ARGV < "$OUT/gen_argv.txt"

if [[ -n "$LOCK" ]]; then
  setsid env -i HOME="$HOME" LANG=C.UTF-8 \
    PATH="${CUDA_HOME}/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/local/cuda/lib64" \
    RUST_LOG=info \
    "${GEN_ENV[@]}" \
    timeout -s INT -k 60 1200 "$BIN" "${GEN_ARGV[@]}" 9>&- > "$OUT/server.log" 2>&1 &
else
  setsid env -i HOME="$HOME" LANG=C.UTF-8 \
    PATH="${CUDA_HOME}/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/local/cuda/lib64" \
    RUST_LOG=info \
    "${GEN_ENV[@]}" \
    timeout -s INT -k 60 1200 "$BIN" "${GEN_ARGV[@]}" > "$OUT/server.log" 2>&1 &
fi
CURRENT_PID=$!
echo "$CURRENT_PID" > "$OUT/pid"

READY=0
for _ in $(seq 1 400); do
  curl -sf -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ready && { READY=1; break; }
  kill -0 "$CURRENT_PID" 2>/dev/null || { echo "SERVER DIED during load"; tail -25 "$OUT/server.log"; break; }
  sleep 2
done
if [ "$READY" != 1 ]; then
  echo '{"label":"'"$LABEL"'","status":"load_failed"}' > "$OUT/result.json"
  echo "LOAD FAILED"
  exit 1
fi
echo "READY"

# Verify the launched server process saw the exact generated ATLAS_*/FLUX_* environment.
REAL_PID=$(pgrep -P "$CURRENT_PID" | tail -1 || true)
if [ -z "$REAL_PID" ]; then
  REAL_PID="$CURRENT_PID"
fi
if [ -n "$REAL_PID" ] && [ -r "/proc/$REAL_PID/environ" ]; then
  tr '\0' '\n' < "/proc/$REAL_PID/environ" | grep -E '^(ATLAS|FLUX)_' | sort > "$OUT/actual_env.txt"
  ENV_MATCH=$(diff <(grep -E '^(ATLAS|FLUX)_' "$OUT/gen_env.txt" | sort) "$OUT/actual_env.txt" > /dev/null 2>&1 && echo true || echo false)
else
  ENV_MATCH=unknown
fi
echo "environ proof: match=$ENV_MATCH"

prefill_warm_median() {
  local req=$1 n=$2
  for i in 1 2 3 4; do
    curl -s -m 300 -o "$OUT/prefill-$n-$i.json" -H 'Content-Type: application/json' --data-binary @"$req" "http://127.0.0.1:$PORT/v1/completions"
  done
}

decode_reps() {
  local dlabel=$1 req=$2
  for i in warmup 1; do
    S=$(date +%s.%N)
    curl -s -m 180 -o "$OUT/decode-$dlabel-$i.json" -H 'Content-Type: application/json' --data-binary @"$req" "http://127.0.0.1:$PORT/v1/chat/completions"
    E=$(date +%s.%N); echo "$S $E" > "$OUT/decode-$dlabel-$i.wall"
  done
}

prefill_warm_median "$REQ_2048" 2048
prefill_warm_median "$REQ_8192" 8192
decode_reps coding "$HERE/prompts/coding.json"
decode_reps chat "$HERE/prompts/chat.json"

grep -a -iE 'panic|CUDA_ERROR|error' "$OUT/server.log" | grep -v -i 'error_rate\|0 error' > "$OUT/errors.txt" || true

python3 - "$OUT" "$LABEL" "$ENV_MATCH" <<'PY'
import hashlib, json, statistics, sys

out, label, env_match = sys.argv[1], sys.argv[2], sys.argv[3]

def prefill(n):
    rows = []
    for i in [1, 2, 3, 4]:
        try:
            with open(f"{out}/prefill-{n}-{i}.json") as fh:
                r = json.load(fh)
            if "error" in r:
                rows.append({"rep": i, "error": r["error"]})
                continue
            u = r["usage"]
            rows.append({"rep": i, "prompt_tokens": u["prompt_tokens"], "ttft_ms": u["time_to_first_token_ms"]})
        except Exception as e:
            rows.append({"rep": i, "parse_error": str(e)})
    warm = [row["ttft_ms"] for row in rows if row.get("rep") != 1 and "ttft_ms" in row]
    result = {"reps": rows}
    if warm:
        med = statistics.median(warm)
        pt = next((row["prompt_tokens"] for row in rows if "prompt_tokens" in row), None)
        result["warm_median_ttft_ms"] = med
        if pt:
            result["tok_s"] = pt / med * 1000
    return result

def decode(label_):
    try:
        with open(f"{out}/decode-{label_}-1.json") as fh:
            r = json.load(fh)
        if "error" in r:
            return {"error": r["error"]}
        u = r["usage"]; msg = r["choices"][0]["message"]; text = msg.get("content") or ""
        with open(f"{out}/decode-{label_}-1.wall") as fh:
            s, e = map(float, fh.read().split())
        wall = (e - s) * 1000
        ttft = u.get("time_to_first_token_ms", 0.0)
        ct = u["completion_tokens"]
        dec = (ct - 1) / ((wall - ttft) / 1000) if wall > ttft and ct > 1 else None
        return {
            "completion_tokens": ct,
            "finish_reason": r["choices"][0]["finish_reason"],
            "decode_tok_s": dec,
            "text_sha256_16": hashlib.sha256(text.encode()).hexdigest()[:16],
        }
    except Exception as e:
        return {"parse_error": str(e)}

result = {
    "label": label,
    "status": "ok",
    "env_proof_match": env_match,
    "prefill_2048": prefill(2048),
    "prefill_8192": prefill(8192),
    "decode_coding": decode("coding"),
    "decode_chat": decode("chat"),
}
with open(f"{out}/result.json", "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps(result, indent=2))
PY

echo "DONE"
