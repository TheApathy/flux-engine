#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Stream a chat completion from Flux Engine via SSE.
# Flux Engine exposes the OpenAI-compatible /v1/chat/completions endpoint,
# so any OpenAI streaming client works.
#
# Usage:
#   bash examples/curl/streaming.sh
#   FLUX_URL=http://127.0.0.1:8888 FLUX_MODEL=Qwen3.8-Flash-Next-NVFP4 \
#     bash examples/curl/streaming.sh "Write a haiku about Blackwell GPUs."

set -euo pipefail

FLUX_URL="${FLUX_URL:-${ATLAS_URL:-http://127.0.0.1:8888}}"
FLUX_MODEL="${FLUX_MODEL:-${ATLAS_MODEL:-Qwen3.8-Flash-Next-NVFP4}}"
PROMPT="${1:-Write a haiku about the ocean.}"

echo "Streaming from $FLUX_URL with $FLUX_MODEL ..."
echo "Prompt: $PROMPT"
echo

# Build valid, properly escaped JSON (handles quotes, backslashes, and newlines).
if command -v jq >/dev/null 2>&1; then
  PAYLOAD=$(jq -n \
    --arg model "$FLUX_MODEL" \
    --arg prompt "$PROMPT" \
    '{model: $model, messages: [{role: "user", content: $prompt}], max_tokens: 256, temperature: 0.7, stream: true}')
else
  PAYLOAD=$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": True,
}))
' "$FLUX_MODEL" "$PROMPT")
fi

CURL_HEADERS=(-H "Content-Type: application/json")
if [[ -n "${FLUX_API_KEY:-}" ]]; then
  CURL_HEADERS+=(-H "Authorization: Bearer $FLUX_API_KEY")
fi

# -N disables curl buffering so SSE deltas arrive live.
if command -v jq >/dev/null 2>&1; then
  curl -sS -N -X POST "$FLUX_URL/v1/chat/completions" \
    "${CURL_HEADERS[@]}" \
    -d "$PAYLOAD" | while IFS= read -r line; do
      line="${line%$'\r'}"
      [[ "$line" == "data: [DONE]" ]] && { echo; break; }
      if [[ "$line" == data:\ * ]]; then
        printf '%s\n' "${line#data: }" | jq --unbuffered -rj '.choices[0].delta.content // empty'
      fi
    done
else
  python3 -u -c '
import json, sys
for raw in sys.stdin:
    line = raw.rstrip("\r\n")
    if line == "data: [DONE]":
        print(flush=True)
        break
    if line.startswith("data: "):
        chunk = json.loads(line[6:])
        choices = chunk.get("choices") or []
        delta = (choices[0].get("delta") or {}) if choices else {}
        content = delta.get("content")
        if content:
            sys.stdout.write(content)
            sys.stdout.flush()
' < <(curl -sS -N -X POST "$FLUX_URL/v1/chat/completions" "${CURL_HEADERS[@]}" -d "$PAYLOAD")
fi
