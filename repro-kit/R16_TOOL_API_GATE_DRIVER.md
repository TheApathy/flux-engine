# Tool-API streaming/blocking release gate driver

`repro-kit/r15-gates/tool-api-streaming-blocking` produces the `tool-api-streaming-blocking` evidence for the
exact-image release suite. It sends real `/v1/chat/completions` tool-call requests, one SSE and one blocking,
to one fresh, owned, target-only server in the immutable release container. It never ingests prior reports,
never accepts a PASS flag, and never synthesizes a missing proof.

## Reuse map

| Concern | Reused as-is from | How |
|---|---|---|
| Sealed tool case: prompt, `get_weather` schema, `tool_choice=required`, `parallel_tool_calls=false`, greedy sampling, cap 64, `enable_thinking=false`, expected `{"city":"Paris"}` | `repro-kit/examples/tool-eval-runaway-manifest.example.json` | exact bytes pinned (`MANIFEST_SHA256`), then `validate_manifest` |
| Nonced request pair, HTTP capture with per-line SSE timings, strict SSE parser with tool-call delta assembly, blocking parser, privacy and wire-shape validation, stream-vs-blocking comparison, rendered-prompt digest | `repro-kit/tool_eval_runaway_microgate.py` | pinned bytes (`MICROGATE_SHA256`) loaded under a private name; `request_bodies`, `http_post`, `parse_sse`, `parse_blocking`, `_bind_summary_to_manifest`, `compare_summaries`, `expected_rendered_prompt`, `validate_rendered_prompt` are called unchanged |
| Parent-suite seal, pinned module loader, container inspection, PID discovery and pidfd ownership, health probe, runtime directory creation | `r15-gates/mmlu-pro` (`LIFECYCLE_SHA256`, the same pin as native-262k and memory-floor) | `verify_parent_suite`, `load_module`, `inspect_container`, `find_pid`, `claim_pid`, `request`, `make_directory` |
| Live process proof (executable hash, recipe environment, `/models` bind, namespace, cgroup, PID-owned listener, mode `none`) | `repro-kit/r15-container-suite.py` `verify_live_server` | before and after the requests; the two proofs must be identical |

The driver does not call `evaluate_captures`. That function's live mode needs a git runner identity and an
operator-declared provenance file, and it deliberately records the runtime as unobserved
(`authoritative=false`). This driver observes the runtime itself, then runs the same checks in the same order.

## Inputs

```text
--container-id @CONTAINER_ID@     --image-id @IMAGE_ID@     --image-digest @IMAGE_DIGEST@
--base-url @BASE_URL@             --run-id @RUN_ID@
--receipt-fd @GATE_RECEIPT_FD@    --raw-log-fd @RAW_LOG_FD@
--runtime-dir <the exact suite runtime directory>
--suite-program <tracked repro-kit/r15-container-suite.py>   --suite-sha256 <suite.program_sha256>
--lifecycle-program <tracked repro-kit/r15-gates/mmlu-pro>
--microgate-program <tracked repro-kit/tool_eval_runaway_microgate.py>
--tool-manifest <tracked repro-kit/examples/tool-eval-runaway-manifest.example.json>
--container-port 8888   (default)
```

Plan entry (replace the zero digest with the driver's real SHA-256):

```json
"tool-api-streaming-blocking": {
  "program_path": "/results/r15-source/clean-tree/repro-kit/r15-gates/tool-api-streaming-blocking",
  "program_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "argv": [
    "/results/r15-source/clean-tree/repro-kit/r15-gates/tool-api-streaming-blocking",
    "--container-id", "@CONTAINER_ID@", "--image-id", "@IMAGE_ID@", "--image-digest", "@IMAGE_DIGEST@",
    "--base-url", "@BASE_URL@", "--run-id", "@RUN_ID@",
    "--receipt-fd", "@GATE_RECEIPT_FD@", "--raw-log-fd", "@RAW_LOG_FD@",
    "--runtime-dir", "/results/r15-runtime",
    "--suite-program", "/results/r15-source/clean-tree/repro-kit/r15-container-suite.py",
    "--suite-sha256", "0000000000000000000000000000000000000000000000000000000000000000",
    "--lifecycle-program", "/results/r15-source/clean-tree/repro-kit/r15-gates/mmlu-pro",
    "--microgate-program", "/results/r15-source/clean-tree/repro-kit/tool_eval_runaway_microgate.py",
    "--tool-manifest", "/results/r15-source/clean-tree/repro-kit/examples/tool-eval-runaway-manifest.example.json"
  ],
  "timeout_seconds": 7200,
  "receipt_relpath": "gates/01-tool-api-streaming-blocking.json",
  "stdout_relpath": "stdout/01-tool-api-streaming-blocking.log",
  "stderr_relpath": "stderr/01-tool-api-streaming-blocking.log"
}
```

The parent must be the sealed suite memfd, the same pin as the accuracy drivers.

## Execution

1. **Launch.** Start the fresh process with `docker exec <cid> /usr/bin/env <image recipe> ATLAS_DISABLE_SPECULATION=1 <binary> serve ...`.
   - It uses the canonical interactive w4096 argv (`recipes/flux-serve-w4096.argv`) without speculation.
   - It adds `--model-from-path /models --model-name qad5500 --bind 0.0.0.0 --port 8888 --tool-call-parser qwen3_coder`.
   - The model name is the sealed request model. The microgate binds every response's `model` to it.
2. **Readiness and proof.** Two `/health` probes, then `verify_live_server` in mode `none`.
3. **Live contract check.** The process's `/proc` argv and environment must carry exactly:
   - `--kv-cache-dtype bf16`, `--max-batch-size 1`, `--max-num-seqs 1`, `--max-prefill-tokens 4096`;
   - `--tool-call-parser qwen3_coder`, `--kernel-target qwen3.8-flash-next`;
   - `--enable-prefix-caching false`, `--ssm-cache-slots 0`;
   - `--model-name qad5500` and a single `--no-tui`;
   - an effective `ATLAS_DISABLE_SPECULATION=1`;
   - no speculation, draft, MTP or chat-template override options.

   This is the microgate's `launch_contract` (C1, interactive-w4096, BF16, speculation none, `qwen3_coder`, checkpoint-native template), observed rather than declared. It is checked before and after the requests.
4. **Requests.** The microgate's exact nonced pair goes to `<base-url>/v1/chat/completions`, SSE first, then blocking, under its 45-second shared deadline. The pidfd is checked before, between and after them.
5. **Evaluation.** The microgate's parsers and comparisons run unchanged, followed by the gate's own checks:
   - **8-key cases.** Two cases (blocking, streaming) with exactly the keys `_recompute_tool_evidence` accepts.
   - **`exact_arguments`.** One `get_weather` call whose decoded arguments equal the sealed `{"city":"Paris"}`.
   - **`no_blank_line_runaway`.** Fewer than four consecutive blank lines, no `finish_reason=length`, and completion tokens below the cap.
   - **`no_all_tools_runaway`.** No more calls than expected tools, and no call to a tool outside the expected set.
   - **`thinking_spacing_valid`.** The server-rendered prompt digest equals the exact `enable_thinking=false` envelope; reasoning is empty; and no `<think>`/`</think>` appears in content, emitted text or arguments.
   - **Stream-vs-blocking consistency.** Tool calls (name, raw argument bytes, parsed arguments), finish reason, content bytes, terminal token IDs and rendered-prompt digest must all be identical.
6. **PASS rule.** PASS requires zero microgate errors, every case key true, `finish_reason=tool_calls` in both arms, observed calls equal to expected, and every consistency field true. Otherwise the driver writes `tool-api/failure.json` (the error class and a digest of the check list, never response text), exits 2, and writes no receipt.
7. **Order of writes.** Raw captures are written only after the microgate's privacy-safe parsers accepted them; then the validator, then the receipt.

## Artifacts (under `tool-api/`, mode 0400)

- `request-stream.json` and `request-blocking.json`: the exact canonical request bodies.
- `response-stream.sse` and `response-blocking.json`: the raw response bytes.
- `validator.json`: the `flux-r15-gate-validator/1` document. Its `raw` carries:
  - `cases` (the wrapper census);
  - `consistency`, `tool_calls` (stream and blocking), `finish_reasons`, `usage`, `terminal_token_ids`;
  - `stream_events`, `stream_tool_fragments` (the SSE delta assembly), `stream_done_seen`, `response_ids`;
  - `rendered_prompt` (digest, per-arm match, audit), `interventions`, `content_types`;
  - `requests` (request SHA-256s);
  - `timings` (pair, per-arm elapsed and first byte, first tool-call delta);
  - `artifacts` (each raw file's locator, SHA-256, bytes, status, content type and timings);
  - `process_bindings`, `live_controls`, and the manifest, microgate and lifecycle pins.
- `failure.json`: on failure only.

## Evidence keys

The wrapper's exact 11-key contract:
- `blocking_cases 1`, `streaming_cases 1`, `tool_calls_expected 2`, `tool_calls_observed 2`;
- `exact_arguments`, `finish_reasons_valid`, `no_blank_line_runaway`, `no_all_tools_runaway`, `thinking_spacing_valid`;
- `validator_artifact "tool-api/validator.json"`, `validator_receipt_sha256`.

## Offline validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 5 prlimit --as=94371840 \
  python3 -m pytest -q -p no:cacheprovider repro-kit/tests/test_r16_tool_api_gate_driver.py
```

A loopback mock answers real HTTP requests with the microgate's committed real-shaped Atlas captures,
substituting each request's nonce. The tests then check:
- the driver's evidence passes the wrapper's `_validate_tool_evidence`;
- tampered census or boolean evidence is rejected, as is a modified or missing validator artifact;
- a raw case with a wrong finish reason, a missing case or an extra per-case key is rejected;
- a wrong container ID or a non-PASS document is rejected;
- streamed-argument drift, a missing `[DONE]` marker, a blank-line runaway and stream/blocking content drift produce no PASS;
- each case rule fails on its own hazard;
- the live launch-contract check refuses a missing parser, speculation, a template override, fp8 KV, a wider batch, speculation enabled in the environment, and a different model;
- the three helper pins equal the tracked bytes.

These tests mock the server; they do not qualify an image, engine, model or GPU.

## Limitations

- The tool case is the single sealed microgate case. Additional tool scenarios would be new, separately
  sealed tests.
- The live `launch_contract` profile is bound through the recipe hash that `verify_live_server` proves, plus the
  argv/environment check above. The driver does not re-derive the profile name from the recipe.
- Timings are diagnostic, not throughput claims.
