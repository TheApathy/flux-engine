<!-- SPDX-License-Identifier: AGPL-3.0-only -->

<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <a href="README.md"><b>Documentation (<code>docs/</code>)</b></a> · <b>HTTP API Reference (<code>docs/API.md</code>)</b></p>

# Flux Engine HTTP API & Compatibility Matrix (`docs/API.md`)

This reference is verified directly against `crates/spark-server/src/main_modules/serve_router.rs`,
`crates/spark-server/src/cli.rs`, `crates/spark-server/src/main_modules/middleware.rs`,
and `crates/spark-server/src/{anthropic,openai}/`.

- **Default raw binary address (`flux-server serve`):** `http://127.0.0.1:8888` (`--bind` default `127.0.0.1`, `--port` default `8888`).
- **Default container address (`flux serve`):** `http://0.0.0.0:8899` (`FLUX_PORT=8899`).
- **Default max request body size:** `128 MiB` (override via `FLUX_MAX_BODY_BYTES` / `ATLAS_MAX_BODY_BYTES`).

---

## 1. Endpoint Inventory

### 1.1 OpenAI-Compatible Endpoints

| Method & Path | Streaming (SSE) | Description |
|---|---|---|
| `POST /v1/chat/completions` | Yes (`stream: true`) | Chat completions with tool calling (`tools`, `tool_choice`), structured outputs (`response_format`), reasoning/thinking (`reasoning`, `reasoning_effort`, `chat_template_kwargs`), multimodal vision (`image_url`), `store: true`, and opt-in `"report_interventions": true`. |
| `GET /v1/chat/completions/{id}` | No | Retrieve a previously stored chat completion (`store: true`). |
| `POST /v1/completions` | Yes (`stream: true`) | Legacy prompt text completions. |
| `POST /v1/responses` | Yes (`stream: true`) | OpenAI Responses API (`input`, tools, reasoning, stateful `conversation` linkage). |
| `GET /v1/responses/{id}` | No | Retrieve a stored Response object. |
| `DELETE /v1/responses/{id}` | No | Delete a stored Response object. |
| `GET /v1/responses/{id}/input_items` | No | List input items associated with a stored Response. |
| `POST /v1/responses/{id}/cancel` | No | Returns `400 response_not_cancellable` (Flux completes responses synchronously; `background: true` is not supported). |
| `POST /v1/conversations` | No | Create a conversation container for `/v1/responses`. |
| `GET / POST / DELETE /v1/conversations/{id}` | No | Read, update, or delete a conversation. |
| `POST / GET /v1/conversations/{id}/items` | No | Append or list conversation items. |
| `GET / DELETE /v1/conversations/{id}/items/{item_id}` | No | Read or delete a single conversation item. |
| `GET /v1/models` | No | List active served model metadata. |
| `GET /v1/models/{*model_id}` | No | Fetch metadata for a specific model ID. |

### 1.2 Anthropic Messages API

| Method & Path | Streaming (SSE) | Description |
|---|---|---|
| `POST /v1/messages` | Yes (`stream: true`) | Native Anthropic Messages endpoint supporting `system`, `messages`, `tools` (`tool_use` / `tool_result`), `thinking` blocks, `output_config` / `output_format` structured JSON schemas, and inline base64 / data-URI `image` content blocks (no remote HTTP image fetching). Emits Anthropic SSE events: `message_start`, `content_block_start`, `content_block_delta` (`text_delta`, `thinking_delta`, `input_json_delta`), `content_block_stop`, `message_delta`, `message_stop`, and `error`. |
| `POST /v1/messages/count_tokens` | No | Exact token count for an Anthropic Messages payload using the loaded model's tokenizer and chat template. |

### 1.3 Ollama-Compatible Discovery Endpoints

| Method & Path | Description |
|---|---|
| `GET /api/tags` | Returns the loaded model in Ollama's `/api/tags` JSON schema for IDE/client auto-discovery. |
| `POST /api/show` | Returns model capabilities and context metadata in Ollama's `/api/show` format. |

*(Only `/api/tags` and `/api/show` are exposed for discovery; inference uses OpenAI `/v1/*` or Anthropic `/v1/messages`.)*

### 1.4 Tokenization, Templates, Telemetry & Disaggregated Serving

| Method & Path | Description |
|---|---|
| `POST /tokenize` · `POST /detokenize` | Raw tokenizer encode/decode endpoints for the loaded checkpoint. |
| `POST /v1/apply-template` | Render a chat request through the server's Jinja template without running inference. |
| `POST /v1/debug/prompt` | Inspect rendered prompt tokens (DeepSeek-V4.1 / diagnostic inspection). |
| `GET /` | Built-in Web Recipe Studio UI. |
| `GET /flux/modes` · `GET /recipes` · `GET /v1/flux/modes` | Inspect active server state, scheduler telemetry, and built-in operating profiles. |
| `GET / PATCH /v1/flux/settings` | Inspect or update live runtime request defaults (persistable via `FLUX_SETTINGS_FILE`). |
| `POST /v1/decisions` | Logit / next-token decision inspection endpoint. |
| `GET /v1/memory` · `GET /v1/experts/usage` | Live VRAM/UMA memory telemetry and MoE expert routing histograms. |
| `GET /v1/disagg/status` · `POST /v1/disagg/{prefill,decode,generate}` | Disaggregated prefill/decode coordination endpoints (`1P+1D` / multi-node). |

### 1.5 Health, Readiness & Prometheus Metrics

| Method & Path | Auth Exempt | Description |
|---|---|---|
| `GET /health` · `GET /ready` | Yes | Readiness probe (`200 OK` once model load completes; returns unhealthy during graceful drain configured by `FLUX_SHUTDOWN_READINESS_DELAY_SECS`, `0..=60`). |
| `GET /health/live` | Yes | Process liveness probe. |
| `GET /metrics` | Yes | Prometheus exposition metrics (prefill/decode throughput, KV block utilization, speculative verify counters `atlas_spec_decode_verify_total`). |

### 1.6 Explicit `501 Not Implemented` Stubs

To keep auto-probing SDKs (LangChain, Helicone, Vercel AI SDK) from failing on
opaque `404` responses, these routes return structured OpenAI-format `501`
errors: `/v1/embeddings`, `/v1/batches*`, `/v1/files*`, `/v1/audio/{transcriptions,translations,speech}`,
`/v1/images/{generations,edits,variations}`, and `/v1/moderations`.

---

## 2. Authentication, CORS & Security Controls

### 2.1 Bearer & `x-api-key` Authentication

By default, `flux-server` binds to `127.0.0.1` with authentication disabled for
local single-user convenience. Authentication activates automatically whenever
any token is configured, or explicitly via `--require-auth`:

- **Token sources:**
  - `--auth-tokens-file <PATH>` (recommended for production; one token per line, `#` comments ignored, `chmod 0600`)
  - `FLUX_API_KEY=<TOKEN>` environment variable
  - `--auth-token <TOKEN>` (alias `--api-key <TOKEN>`, convenient for local testing)
- **Accepted client headers:**
  - `Authorization: Bearer <token>` (OpenAI SDKs)
  - `x-api-key: <token>` (Anthropic SDKs / Claude Code)
- **Protected routes:** `/v1/*` (including `/v1/flux/modes`, `/v1/flux/settings`, `/v1/disagg/*`), `/tokenize`, `/detokenize`, `/api/tags`, `/api/show`.
- **Scoped tokens (optional):** Tokens in `--auth-tokens-file` or `--auth-token` may carry optional scope prefixes (`read:`, `inference:`, `admin:`); unprefixed tokens grant full access. Insufficient scope returns `403 Forbidden` (`insufficient_scope`). `PATCH /v1/flux/settings` and `POST /v1/disagg/*` require `admin` scope.
- **Open probe/UI routes:** `/`, `/flux/modes`, `/recipes`, `/health`, `/ready`, `/health/live`, `/metrics`.
- **Reverse-proxy client IP:** Rate limiting uses the socket peer address by default; set `FLUX_TRUST_PROXY=1` only behind a trusted reverse proxy to read `X-Forwarded-For`.

### 2.2 Loopback-Only Default CORS

By default, `serve_router.rs` permits cross-origin browser requests **only** from
loopback origins (`http(s)://localhost[:port]`, `127.0.0.1`, `[::1]`). Set
`FLUX_CORS_ALLOW_ORIGINS` (comma-separated origins, or `*`) to widen CORS
explicitly when serving a browser frontend on another host.

---

## 3. Feature Capabilities

| Feature | OpenAI `/v1/chat/completions` & `/v1/responses` | Anthropic `/v1/messages` | Notes |
|---|---|---|---|
| **Streaming SSE** | Yes | Yes | Live token, reasoning, and tool-call argument deltas. |
| **Tool / Function Calling** | Yes (`tools`, `tool_choice`) | Yes (`tools`, `tool_use`, `tool_result`) | Per-model tool parsers (`hermes`, `qwen`, `glm`, `gemma4`, `dsv41`, etc.). |
| **Structured Outputs (`xgrammar`)** | Yes (`response_format`: `json_object`, `json_schema`) | Yes (`output_config` / `output_format` JSON schema + tool schema) | Safe Rust pushdown-automaton grammar mask in `crates/xgrammar`. |
| **Reasoning / Thinking Streams** | Yes (`reasoning`, `reasoning_effort`, `chat_template_kwargs`) | Yes (`thinking` blocks + `thinking_delta`) | Budget capped by `--max-thinking-budget` / `MODEL.toml` (`262144` on Qwen3.8). |
| **Vision / Multimodal** | Yes (`image_url` base64 / data URI) | Yes (`image` content blocks) | Supported on VLM checkpoints (`Qwen3-VL`, `Qwen3.6`, `Gemma-4`, `Qwen3.8-Flash-Next`). |
| **Intervention Ledger** | Yes (`"report_interventions": true`) | — | Opt-in report-only audit ledger (`FLAGS.md` §2.1). |

---

## 4. Quick Examples

### OpenAI Chat Completion (`curl`)

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${FLUX_API_KEY:-local}" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Explain CUDA warp shuffles in 3 bullets."}],
    "max_tokens": 256,
    "temperature": 0.0
  }'
```

### Anthropic Messages (`curl`)

```bash
curl -s http://127.0.0.1:8888/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${FLUX_API_KEY:-local}" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Write a minimal Rust LRU cache trait."}]
  }'
```

For end-to-end coding agent setup (Claude Code, OpenCode, Cursor, Continue,
Cline), see [`docs/CLAUDE_CODE.md`](CLAUDE_CODE.md).
