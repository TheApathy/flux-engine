# flux-router: multi-Spark data parallelism (DP)

N independent single-Spark flux-servers behind one HAProxy front door. **This is not native single-engine C64**
(rows64 P2: a 64-row MoE and dense pass on one Spark). It is DP: each request runs entirely on one Spark, and
aggregate throughput scales with the number of boxes. There is no engine change. 2 × C32 is modeled at about
394 tok/s aggregate (`concurrency-scale/rows64`); that is not measured yet.

```
FLUX_ROUTER_BACKENDS="spark-a:8888=32,spark-b:8888=32" repro-kit/router/flux-router --port 8888
repro-kit/router/flux-router --backends spark-a:8888=32 --skip-backend-checks --render-only   # inspect
```

## Backends and caps
- `host:port=cap`, where cap is `1..128` (the router/launcher per-backend cap; live engine hard ceiling is `256`
  in `crates/spark-model/src/layers/mod.rs`). Hosts are hostnames, IPv4, or `[IPv6]`. Anything
  else is refused, because the value is written into the HAProxy config.
- The cap must equal that server's resolved scheduler batch width (`FLUX_CONCURRENCY`, published as
  `serving.concurrency`, NOT `serving.ingress_queue` / `--max-num-seqs`). Pick each backend's mode
  per server: target-only C17-128, batched K5 C2-32, or opt-in adaptive-K C9-64.
- Startup refuses when a backend is unreachable, the backends serve different `/v1/models` ids, or a cap
  contradicts the server's published `serving.concurrency` (below).

### Engine endpoint: `/flux/modes` → `serving`
`api/flux_ui.rs` publishes the mode catalog, scheduler counters, and resolved
serving concurrency/speculation metadata in the `modes_json` body:
```json
"serving": {
  "concurrency": 32,
  "ingress_queue": 32,
  "max_seq_len": 65536,
  "speculation": "target-only",
  "num_drafts": 0,
  "world_size": 1,
  "source": "resolved-at-load"
}
```
- `speculation` is one of `target-only` | `native-mtp` | `batched-k5` | `adaptive-k` | `dflash`.
- `concurrency`, `ingress_queue` (`--max-num-seqs`), and `max_seq_len` are the effective serve args after the launcher's clamps.

Built at load from resolved scheduler facts:
- `concurrency` is the batch width the scheduler actually runs, after the EP clamp.
- `ingress_queue` (`--max-num-seqs`) only sizes the request channel, and the router never uses it as a cap.
- `speculation` counts only an installed proposer.
- If a backend is given without a cap (`spark-a:8888`), the router uses the published value.
- If a backend is given with a cap, the router refuses a mismatch.
- If the field is absent, as on servers today, an explicit cap is required.

## Behaviour
- **Balancing.** Least outstanding requests, counted per HTTP request, not per TCP connection, so keep-alive
  clients still spread. Each backend receives at most `cap` requests at once.
  - The rest wait at the router for up to `FLUX_ROUTER_QUEUE_TIMEOUT_S` (default 120 s), then get a 503.
  - Backends do not answer 429 from router traffic alone. Three things can still produce one:
    - traffic sent to a backend directly;
    - a backend's own rate limit;
    - a burst of cancellations larger than the backend's pending-queue slack.
- **Health.** `GET /health` every 2 s with a 3 s timeout. A backend that is loading, faulted, refusing connections,
  or hung leaves rotation after 2 failed checks and rejoins after 3 passing ones.
  - Limit: `/health` reports model-ready and fault state, not decode progress.
  - A GPU stall that the engine doesn't report as a fault keeps answering 200, and requests sent there wait until
    `timeout server`. Detecting that needs a progress-based health signal from the engine.
- **Retries.** Only connection failures are retried, once, on the other backend. Once response bytes have been
  sent, a request is never replayed, so if a backend dies mid-generation that client's request fails.
- **Cancellation.** A client that disconnects mid-stream frees its router slot at once. The backend frees its decode
  slot at its next token.
  - In that gap the backend may briefly hold `cap + 1` requests.
  - flux-server's pending queue, `max(4·C, 32)` (`scheduler/mod_helpers.rs`), absorbs this. Keep that slack.
- **Streaming.** SSE passes through unbuffered. The `X-Flux-Backend` header names the server that answered
  (`b0`, `b1`, ...).
- **Client timeouts.** 30 s to send request headers and 10 s idle keep-alive. Generation itself may take up to
  1800 s.
- **Not routable (501).**
  - The 501 covers `/v1/conversations*`, `/v1/files*`, `/v1/batches*`, `/v1/chat/completions/{id}`, and
    `/v1/responses/{id}...`.
  - `POST /v1/responses` whose body sets a top-level `previous_response_id` or `conversation` to a non-null value
    also gets 501, because both resolve against one server's local store.
    - `router/flux_state.lua`, run inside HAProxy, parses the body's top-level JSON members and decodes names, so an
      escaped name like `previous\u005fresponse_id` is caught.
    - It enforces the JSON grammar the way serde_json rejects bodies: RFC 8259 numbers, paired surrogates, valid
      UTF-8, and no raw control bytes below 0x20.
    - It is tested on 34 cases, including the adversarial JSON parser set, under local Lua and HAProxy's Lua 5.3.
    - It is not a proof of full serde equivalence; anything it can't parse is refused, never forwarded.
    - Nested keys, such as `metadata.conversation`, are not state.
    - A body that isn't one complete JSON object gets 400, so it is never forwarded unchecked. That includes
      chunked request bodies.
  - A `/v1/responses` body larger than 256 KiB gets 413, because it can't be inspected in one buffer.
    A body of exactly 256 KiB is inspected. Use
    `/v1/chat/completions` for large prompts.
  - Stateless `POST /v1/responses` and `/v1/chat/completions` are routed. However, an object created with
    `store: true` through the router can't be fetched back through it, because it lives on one server under a random id.
  - Use a server directly if you need stored objects.
- **No prefix affinity.** Prefix caching is off, so there is nothing to keep warm.
- **Stats.** `--stats-port N` serves HAProxy's stats page on 127.0.0.1 only.

## Requirements
- `haproxy` 2.4 or newer built with Lua (`USE_LUA=1`, which the official image and Ubuntu's package have) on PATH, or
  the `haproxy:2.8` image. The config is checked with `haproxy -c` in the tests.
- The same image, checkpoint, recipe and API keys on every backend. `Authorization` passes through unchanged.
- **Authenticated backends** (`--require-auth`): the router's own startup checks need a key. Set
  `FLUX_ROUTER_DISCOVERY_KEY_FILE=/path/to/key`, preferred, or `FLUX_ROUTER_DISCOVERY_KEY`. The key is sent only on
  the router's `/v1/models` and `/flux/modes` startup requests. It is never written to the HAProxy config or the
  logs.

## Tests
`python3 -m pytest repro-kit/tests/test_flux_router.py` runs 41 tests. The integration tests need docker and
`haproxy:2.8`. They cover:
- 3 × overload: each backend filled to exactly its cap, never over, with zero 429s;
- keep-alive spreading;
- 501 routes, with stateless `/v1/responses` routed;
- SSE first byte under 1.5 s;
- client cancel freeing capacity;
- no replay after response bytes;
- dead and hung backends marked DOWN, with all traffic moving to the survivor;
- queue timeout returning 503 without backend 429s;
- config injection refused, including a newline before the port;
- `/v1/responses` state references, including escaped names and chunked bodies (501);
- null references and nested metadata (routed), unparsable bodies (400), oversized bodies (413), and an
  exactly-at-limit body (routed);
- the Lua scanner's 34 cases, run both locally and inside the HAProxy image (`tests/lua/test_flux_state.lua`);
- discovery auth header sent, and never rendered into the config.

## Two-Spark validation checklist
- Parity: C64 through the router byte-identical to C1 per backend, using the strict 32-case suite.
- Throughput: C8, C16, C32, C48, C64, with a 2 × C16 adaptive-K arm.
- Overload at C96, kill-one-backend failover, and TTFT overhead under 2 ms.
