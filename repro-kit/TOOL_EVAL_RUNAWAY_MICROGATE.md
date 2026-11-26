# Tool-eval runaway microgate

`tool_eval_runaway_microgate.py` is a bounded correctness check for the QAD5500
tool-call path. It makes two otherwise identical greedy requests against an
already-running server: SSE first, then blocking. Both calls share one deadline
of at most 45 seconds. The script does not start a server or execute a tool.

A pass requires all of the following:

- the launch contract declares C1 `interactive-w4096`, BF16 KV, speculation
  off, and `qwen3_coder`;
- the request sets `parallel_tool_calls=false`, deterministic greedy sampling,
  an explicit token cap of at most 512, `return_token_ids=true`, and
  `report_interventions=true`;
- the SSE stream is strict UTF-8, structurally valid, has exactly one final
  usage object in Atlas's complete wire shape (token counts and breakdowns,
  TTFT, and decode rate), and exact terminal token IDs, and ends with exactly
  one `[DONE]` marker; the blocking response must carry the identical token-ID
  vector;
- both responses carry exactly one terminal `atlas.interventions.v1` ledger
  with `policy=report_only` and no events; missing, malformed, or non-empty
  ledgers fail the gate, including any schema-backfill/type-coercion or path
  normalization repair to a published tool call and any noncanonical
  `qwen3_coder` parser recovery;
- every manifest, SSE event, blocking body, and decoded tool-argument object is
  strict JSON: duplicate keys, `NaN`/`Infinity`, and exponent overflow such as
  `1e9999` fail closed;
- streaming content, reasoning content, tool name, and raw argument bytes match
  the blocking response;
- exactly one `get_weather` call is emitted and its decoded arguments are
  exactly `{"city":"Paris"}` before any response artifact can be written;
- every response has Atlas's exact `fp_atlas` system fingerprint, response IDs
  have the native `chatcmpl-<lowercase UUID>` shape, and tool-call IDs have the
  native `call_<16 lowercase hex>` shape;
- the response stays below the token cap, no arm ends with
  `finish_reason=length`, and neither arm contains four consecutive blank
  lines.

## Offline self-test

```bash
python3 repro-kit/tool_eval_runaway_microgate.py \
  --fixture repro-kit/tests/fixtures/tool_eval_runaway_micro.json \
  --out-dir /tmp/flux-tool-microgate-offline
```

The committed fixture is deterministic and does not contact a server.

## Live run

The example manifest is the sealed test case. Do not replace its prompt, tool,
schema, expected tool, or expected `{"city":"Paris"}` argument. Changing those
fields creates a different test and the validator fails closed.

Copy
`examples/tool-eval-runaway-live-provenance.example.json` and replace every
placeholder digest with the exact value for this run. The schema is exact:

- `runner.git_tree`, `runner.git_blob`, and `runner.sha256` bind the committed
  microgate source and its current bytes;
- `source.git_tree` must equal `runner.git_tree`;
- `execution.mode` is `host-binary` or `container-image`;
- `execution.binary_sha256` is always required; `execution.image_digest` is
  `null` for `host-binary` and `sha256:<64 lowercase hex>` for
  `container-image`;
- `checkpoint.id`, `checkpoint.revision`, and `checkpoint.served_model` are
  required, and `served_model` must equal the manifest request model.

The runner validates its own committed identity. It cannot independently
observe the process owning the HTTP listener, the executable that process
mapped, the backing container, or its opened checkpoint inventory. Therefore a
live receipt is deliberately `authoritative=false`, has
`authority_scope=live-parser-only-unverified-runtime`, keeps the operator input
under `declared_identity`, and sets every `runtime_binding` field to `false`.
It cannot be used as a release-qualified runtime identity receipt.

```bash
cp repro-kit/examples/tool-eval-runaway-manifest.example.json /tmp/tool-manifest.json
cp repro-kit/examples/tool-eval-runaway-live-provenance.example.json /tmp/tool-live-provenance.json
# Replace every placeholder digest in /tmp/tool-live-provenance.json first.
python3 repro-kit/tool_eval_runaway_microgate.py \
  --manifest /tmp/tool-manifest.json \
  --live-provenance /tmp/tool-live-provenance.json \
  --endpoint http://127.0.0.1:8899/v1/chat/completions \
  --out-dir /tmp/flux-tool-microgate-live
```

If authentication is enabled, place the bearer value in `FLUX_API_KEY`; the
value is used for the request but never written to an artifact. Use
`--api-key-env NAME` to select another environment variable.

The output directory must not already exist. It contains the canonical input
`manifest.json`, exact streaming and blocking requests, raw `response-stream.sse`, raw
`response-blocking.json`, and a canonical `receipt.json` with hashes, timing,
UTF-8 hex, token IDs, tool fragments, usage, finish reasons, and the verdict.
A nonzero exit status means the gate did not pass.

This bounded parser/wire microgate does not replace a fresh-process full
tool-eval replay, GPQA replay, model-quality qualification, or an independently
bound listener/container/checkpoint provenance gate.

## Rendered-prompt evidence

The runner generates distinct 128-bit nonces for the streaming and blocking
requests and sends each as `rendered_prompt_evidence_nonce`. The server decodes
the exact final prompt-token vector with special tokens preserved, hashes those
UTF-8 bytes, immediately discards the decoded text, and returns only its SHA-256
in `atlas.rendered_prompt.v1` evidence on that request's terminal response. The
evidence field contains no raw prompt bytes and adds no prompt logging. Evidence
must match the request nonce and response ID; an
operator-supplied manifest string is not accepted.
Any response field containing prompt material, including a top-level or nested
`rendered_prompt_utf8`, `raw_prompt_bytes`, or `prompt_text`, fails closed. The
only prompt-named response fields allowed are the four-field evidence object
and Atlas usage token counters. This violation aborts before response artifacts
or a receipt are written, so prompt text is never persisted by the runner.

The gate constructs the sole allowed model-specific prompt from the sealed
request and hashes that complete byte-exact envelope. Tool JSON serialization,
native Qwen instructions, Atlas safety block, sole user message, role
boundaries, and the final assistant prefix with checkpoint thinking explicitly
pinned off through `chat_template_kwargs.enable_thinking=false` are exact. Extra messages,
injected/trailing text, duplicate protocols, alternate prefixes, a Hermes
marker, or stream/block digest disagreement fail closed. The generated nonces
are sealed in `request-stream.json` and `request-blocking.json`; only terminal
digests remain in the raw response artifacts and receipt.
