#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# CPU-only regression test for tools/recipe_bench/run.sh trap behavior:
# 1. Verifies run.sh's actual `trap cleanup ...` line does NOT trap ERR.
# 2. Executes that exact trap specification in a subshell with a background
#    dummy process and runs genuinely failing commands (without `|| true`)
#    to prove the dummy process survives nonzero exit codes.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="${HERE}/run.sh"

TRAP_LINE="$(grep -E '^[[:space:]]*trap cleanup ' "$RUN_SH" | head -n 1 || true)"
if [[ -z "$TRAP_LINE" ]]; then
  echo "FAIL: could not find 'trap cleanup ...' in $RUN_SH" >&2
  exit 1
fi

if [[ "$TRAP_LINE" =~ [[:space:]]ERR([[:space:]]|$) ]]; then
  echo "FAIL: run.sh traps ERR ($TRAP_LINE), which kills healthy servers on nonzero commands" >&2
  exit 1
fi

# Extract the exact signals registered by run.sh
SIGNALS="${TRAP_LINE#*trap cleanup }"

sleep 300 &
CURRENT_PID=$!
CLEANED_UP=0
KILLED_EARLY=0

cleanup() {
  [ "$CLEANED_UP" = 1 ] && return 0
  CLEANED_UP=1
  if [ -n "$CURRENT_PID" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    kill -9 "$CURRENT_PID" 2>/dev/null || true
  fi
}

# Register the exact trap signals extracted from run.sh
# shellcheck disable=SC2064,SC2086
trap cleanup $SIGNALS

# Trigger real nonzero exits (without `|| true`) under `set +e` (matching run.sh).
# If ERR were trapped, these would invoke cleanup and kill $CURRENT_PID.
grep -q "pattern-that-does-not-exist" /dev/null
false

if ! kill -0 "$CURRENT_PID" 2>/dev/null; then
  echo "FAIL: dummy server was killed by a failing command mid-script" >&2
  KILLED_EARLY=1
else
  echo "PASS: run.sh trap ($SIGNALS) keeps dummy server (pid $CURRENT_PID) alive across nonzero commands"
fi

kill -9 "$CURRENT_PID" 2>/dev/null || true
exit "$KILLED_EARLY"
