<!-- SPDX-License-Identifier: AGPL-3.0-only -->

<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <a href="README.md"><b>Documentation (<code>docs/</code>)</b></a> · <b>Coding Agents &amp; Claude Code Guide (<code>docs/CLAUDE_CODE.md</code>)</b></p>

# Connecting Claude Code & Coding Agents to Flux Engine (`docs/CLAUDE_CODE.md`)

Flux Engine ships native **Anthropic Messages** (`/v1/messages`,
`/v1/messages/count_tokens`) and **OpenAI** (`/v1/chat/completions`,
`/v1/responses`, `/v1/models`) endpoints with streaming tool calling
(`tool_use` / `input_json_delta`), live reasoning streams (`thinking_delta`),
and `xgrammar` schema enforcement—so coding agents connect directly with zero
translation proxy.

---

## 1. Start Flux Engine Locally

### Option A — Universal Container (Port `8899` by default)

```bash
docker run --rm -it --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8899:8899 \
  -v "$HOME/models:/models:ro" \
  -e FLUX_API_KEY="flux-local-dev" \
  ghcr.io/theapathy/flux-engine:v1.0.3
```

### Option B — Native `flux-server` Binary (Port `8888` by default)

```bash
FLUX_API_KEY="flux-local-dev" \
./target/release/flux-server serve \
  --model-from-path "$HOME/models/Qwen3.8-Flash-Next-NVFP4-Offload" \
  --served-model-name qwen3.8-flash-next \
  --kernel-target qwen3.8-flash-next \
  --bind 127.0.0.1 \
  --port 8888 \
  --max-seq-len 135168 \
  --kv-cache-dtype bf16 \
  --gpu-memory-utilization 0.55 \
  --speculative --num-drafts 4
```

Verify the server is ready before launching your coding agent:

```bash
curl -sf http://127.0.0.1:8899/health && echo " ready"
curl -sf -H "x-api-key: ${FLUX_API_KEY}" http://127.0.0.1:8899/v1/models
```

*(Adjust `8899` to `8888` if using the raw `flux-server` default port, or pass `--port 8000` to bind port `8000`.)*

---

## 2. Claude Code

Claude Code speaks the Anthropic Messages API natively. Point
`ANTHROPIC_BASE_URL` at Flux Engine (without a trailing `/v1`):

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8899"
export ANTHROPIC_API_KEY="flux-local-dev"
export ANTHROPIC_MODEL="qwen3.8-flash-next"

claude
```

- **Streaming thinking & tools:** Flux Engine's `/v1/messages` translator
  (`crates/spark-server/src/anthropic/translator.rs`) emits standard Anthropic
  `thinking_delta`, `text_delta`, and `input_json_delta` SSE blocks so Claude
  Code renders reasoning and multi-tool edits natively.
- **Exact token budgeting:** Claude Code calls `POST /v1/messages/count_tokens`,
  which Flux Engine serves using the loaded checkpoint's exact tokenizer and
  chat template.
- **Non-loopback hosts:** If Claude Code connects to a remote workstation over
  a LAN/VPN, bind Flux Engine with `--bind 0.0.0.0` behind TLS and use
  `--auth-tokens-file` (see [`docs/API.md`](API.md#2-authentication-cors--security-controls)).

---

## 3. OpenCode

Configure a custom OpenAI-compatible or Anthropic-compatible provider in your
OpenCode configuration (`opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "flux-local": {
      "npm": "@ai-sdk/anthropic",
      "name": "Flux Engine (Local)",
      "options": {
        "baseURL": "http://127.0.0.1:8899/v1",
        "apiKey": "flux-local-dev"
      },
      "models": {
        "qwen3.8-flash-next": {
          "name": "Qwen3.8-Flash-Next (GB10)",
          "limit": { "context": 135168, "output": 32768 }
        }
      }
    }
  }
}
```

---

## 4. Cursor, Continue & Cline

All three editors connect directly via Flux Engine's OpenAI-compatible API:

| Client | Provider Setting | Base URL | API Key | Model Name |
|---|---|---|---|---|
| **Cursor** | OpenAI API Key → Override OpenAI Base URL | `http://127.0.0.1:8899/v1` | `flux-local-dev` | `qwen3.8-flash-next` (from `/v1/models`) |
| **Continue** | `provider: openai` | `http://127.0.0.1:8899/v1` | `flux-local-dev` | `qwen3.8-flash-next` |
| **Cline** | API Provider: `OpenAI Compatible` | `http://127.0.0.1:8899/v1` | `flux-local-dev` | `qwen3.8-flash-next` |

---

## 5. Python SDKs (Anthropic & OpenAI)

### Anthropic Python SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:8899",
    api_key="flux-local-dev",
)

with client.messages.stream(
    model="qwen3.8-flash-next",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Review this diff for off-by-one bugs."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8899/v1",
    api_key="flux-local-dev",
)

resp = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[{"role": "user", "content": "Write unit tests for a ring buffer."}],
    stream=True,
)
for chunk in resp:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

---

## 6. Practical Tips for Agentic Coding Sessions

1. **Prefix caching ON for interactive sessions:** Unlike cold-prefill
   benchmarks (which pass `--enable-prefix-caching false --ssm-cache-slots 0`),
   keep prefix caching enabled for multi-turn coding sessions so repeated system
   prompts and repository context stay warm in KV + Marconi SSM caches.
2. **Context window & UMA headroom on 128 GB GB10:** Use `--mode interactive`
   (`w4096`) when running large editors or multiple local tools alongside the
   server (`11.9–12.7 GiB` free UMA floor), or `--mode 256k` when sessions
   regularly exceed `135K` tokens. Keep `--gpu-memory-utilization <= 0.55` on
   GB10 UMA.
3. **Thinking controls:** Leave `FLUX_THINKING` unset so individual agent
   requests can opt into reasoning via `thinking` (Anthropic) or
   `chat_template_kwargs` (OpenAI). See [`FLAGS.md`](FLAGS.md) for
   `FLUX_THINKING_BUDGET` and `FLUX_THINK_NUDGE`.
