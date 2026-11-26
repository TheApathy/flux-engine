# Qwen3.8 Coding Reproduction

This guide describes the public, secret-free reproduction interface for **Flux
Engine** (`TheApathy/flux-engine`). It records benchmark evidence without
publishing model weights, credentials, host paths, process state, full
generated responses, or machine-specific logs.

## Current Status

The endpoint verification harnesses and sanitized reference evidence live under
[`repro/qwen38-coding/`](../../repro/qwen38-coding/) and
[`repro/qwen38-flash-next/`](../../repro/qwen38-flash-next/). These commands
verify an already-running compatible `flux-server` instance; for full container
and host launch instructions see [`REPRODUCING.md`](../../REPRODUCING.md) and
[`GETTING_STARTED.md`](../../GETTING_STARTED.md).

## Reproduce Against a Running Server

```bash
git clone --depth 1 https://github.com/TheApathy/flux-engine.git
cd flux-engine
python3 repro/qwen38-coding/run.py
```

Target a custom endpoint when necessary:

```bash
python3 repro/qwen38-coding/run.py \
  --endpoint http://127.0.0.1:8896/v1/chat/completions
```

The coding gate executes three sequential deterministic requests (`temperature:
0.0`, `"chat_template_kwargs": {"enable_thinking": false}`, `max_tokens: 1500`)
and verifies one stable output hash and the configured throughput floor. Do
**not** pass `reasoning_effort` in the request payload — Qwen3.8's Jinja chat
template rejects `reasoning_effort` with HTTP `400`. Results are written under
ignored `runs/` and contain hashes and metrics, not raw response text.

## Evidence Policy

Tracked evidence may contain:

- canonical request hashes
- token counts and finish reason
- content and stable-output hashes
- per-run server decode TPS and client wall time
- deterministic pass/fail state

The checked-in gamma sweep retains per-run server TPS values, the common output
fingerprint, and SHA-256 digests of the original verification run artifacts
without exposing raw logs or local filesystem paths.

Tracked evidence must not contain:

- API tokens or credential files
- absolute user or model-cache paths
- model weights or quantized sidecars
- full prompts from private workloads
- full generated responses
- hostnames, process IDs, environment dumps, or raw serve logs

Run the public-safety leak check before committing:

```bash
python3 scripts/verify_no_leaks.py
```
