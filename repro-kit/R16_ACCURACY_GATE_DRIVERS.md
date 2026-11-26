# Real MMLU-Pro and GSM8K release drivers

`r15-gates/mmlu-pro` and `r15-gates/gsm8k` collect actual paired accuracy
measurements against the immutable container owned by the release suite. They
do not ingest previously completed reports, accept evaluator-supplied PASS
flags, or synthesize missing measurements. Offline tests explicitly mock all
Docker, server, and HTTP execution and never qualify an image.

## Inputs and dependencies

Each driver accepts the suite identity and inherited receipt/log descriptors:

```text
--container-id @CONTAINER_ID@
--image-id @IMAGE_ID@
--image-digest @IMAGE_DIGEST@
--base-url @BASE_URL@
--run-id @RUN_ID@
--receipt-fd @GATE_RECEIPT_FD@
--raw-log-fd @RAW_LOG_FD@
--runtime-dir <the exact suite runtime directory>
--suite-program <the tracked source r15-container-suite.py>
--suite-sha256 <the exact suite.program_sha256>
--lil-program <the tracked llm-inference-bench/llm_decode_bench.py>
--container-port 8888
```

The port is optional and defaults to 8888. The base URL must be explicit HTTP
IPv4 loopback with a port, no credentials, and no path/query/fragment. The
source-pinned suite must include the live listener/model/native-mode broker
upgrade (`ccc03dc3`). Its bytes must also match the actual sealed suite memfd
in the launching parent's command line. Supplying an arbitrary verifier and
its self-attested hash cannot substitute a different qualification authority.
Direct standalone invocation is intentionally not a release path.

Lil is pinned to v0.7.9, scorer version 2, and source SHA-256
`11f36dcf276b0d3ef7da3144d16b50c4bf6737fee211ded4f13b348962545fc2`.
The normal evaluator dependencies (`httpx` and `rich`) must be installed on the
qualification host. No dependency installation occurs in the driver. The
dataset bytes are verified against the existing pinned hashes, copied into a
sealed memfd, and supplied directly to Lil's dataset builder. Its cache and
download resolver is replaced before invocation. Missing data, modules, or
unsupported evaluator/broker versions fail closed without a gate receipt.

The implementation reuses Lil's actual `BUILTIN_TEST_PROFILES`,
`load_benchmark_dataset_items`, `extract_final_answer`, and
`score_completion_profile` APIs. It does not create Lil summary reports or
claim this blocking exact-token arm is identical to Lil's streaming benchmark
transport. It uses the same pinned dataset prompts and scoring rules.

## Execution contract

The exact container ID/image and read-only root are inspected. Binary and
recipe paths/hashes come from immutable image labels already bound by the
outer wrapper. The literal recipe is read without symlink traversal through
the container init's host `/proc` root and checked against its label hash.

For each dataset, two distinct fresh C1 processes run sequentially:

- Plain: `ATLAS_DISABLE_SPECULATION=1`, no speculative CLI selector.
- Native K5: `ATLAS_DISABLE_SPECULATION=0 --speculative --num-drafts 4`.

Both use `--no-tui --model-from-path /models`, a unique model name, BF16 KV,
`--qwen4-qsa --max-batch-size 1 --max-num-seqs 1 --max-seq-len 262144`, and the declared
container port. No `--mtp-gate`, external drafter, or prefix-cache selector is
added. Each startup requires two successful `/health` probes and an independent
suite-owned proof of actual executable/recipe/environment, model mount,
namespace, cgroup, native selectors, and PID-owned endpoint. The proof is
repeated after the arm. Positive K=5 verification telemetry is required in the
native arm, while the plain arm must have zero K=5 events.

Both managed accuracy launches set one explicit finite
`--request-timeout 14340` (3 hours 59 minutes) per request. The HTTP client
deadline remains 14,400 seconds, leaving 60 seconds for a terminal timeout
response. This avoids inheriting the ordinary 300-second serving default for
131,072-token output budgets. It does not change the engine's global CLI
default, disable timeouts, extend readiness or cleanup deadlines, or make a
truncated response acceptable. Timeout/non-stop responses still fail the gate.
The shared memory-floor lifecycle inherits this explicit launch deadline;
native-262k keeps its separately defined launch policy. Updating the lifecycle
bytes also requires reviewed literal-pin updates in those dependent drivers.

The required metric is `atlas_spec_decode_verify_total{k="5",...}` from
actual verifier execution. At base `9b9023e4`, the shared K-gamma native route
instead emits the legacy `k="dflash"` label. Integration must add truthful
numeric/native K5 telemetry before a live native accuracy arm can qualify.
The driver deliberately does not treat that legacy label or launch argv as
proof of a native five-row target verification frame. The numeric label must
come from the actual Qwen4 native proposer, proposer-origin drafts, and observed
target input-row count. It does not prove a single physical wide GPU kernel:
the backend can use a serial fallback. Nor does the counter alone prove target
state commit or accuracy. Full metric snapshots are retained in the validator,
and the paired token/output/accuracy gates remain independently mandatory.

Local Linux Docker, pidfds, and sufficient `/proc` and signal permissions are
required. A root-owned engine may require a privileged qualification
supervisor. Server ownership is established from its unique model-name token,
container cgroup, actual executable hash, and pidfd. Cleanup signals only that
claimed process identity. Failure to stop gracefully, forced termination, or
identity drift fails the gate. A launch/ownership failure still fails the gate
and the outer wrapper remains responsible for exact container teardown. No
remote Docker or weaker process-proof fallback is supported.

Every pinned item is submitted once per arm to blocking
`/v1/chat/completions` with greedy temperature 0, a 131,072 output budget,
`return_token_ids=true`, `report_interventions=true`, and a fresh
`rendered_prompt_evidence_nonce`. A successful measurement requires:

- One terminal `stop` choice with nonempty visible content and no tool calls.
- Exact integer prompt/completion/total usage and explicit `cached_tokens=0`.
- All scheduler-owned token IDs, with count equal to completion usage.
- An exact empty report-only intervention ledger.
- A unique response ID and a matching server-produced prompt nonce/response ID.
- A final, parseable answer scored by the real pinned Lil scorer.

Plain/K5 comparison requires identical parsed answers, visible content and
reasoning hashes, rendered prompt hashes, exact token-sequence hashes, token
counts, finish reasons, and correctness for every item. Full token IDs remain
in the response journal, not reconstructed from text. All 1,000 MMLU-Pro and
1,319 GSM8K items are mandatory. Provisional sanity floors remain 0.65 and 0.95
respectively. Aggregate equality cannot hide swapped per-item outcomes.

## Artifacts and interpretation

Artifacts use no-follow exclusive creation below private mode-0700 directories
`accuracy/mmlu-pro/` and `accuracy/gsm8k/`. Files use mode 0400. Each contains:

- `responses.jsonl`: actual request and response for every measured item,
  including full token IDs, content, usage, interventions, and prompt evidence.
- `validator.json`: the strict gate-validator document with the required
  per-item recomputation rows, normalized measurements, journal hash/locator,
  evaluator hash, and independent process bindings.

The journal is bounded, fsynced, identity-checked, and rehashed before sealing
the validator. The driver emits the exact existing accuracy evidence schema
only after complete coverage, parity, accuracy, live proof, and cleanup pass.
It writes that receipt solely to the inherited descriptor. Failure preserves
partial private observations for diagnosis but writes no PASS receipt.

The outer wrapper validates the strict validator rows and its artifact hash.
Detailed observations and response-journal bindings are retained for independent
audit, not new automatically traversed outer-wrapper artifact types. A final
release review should verify the retained journal hash and inspect real
runtime receipts. These source/offline changes are not accuracy scores,
container qualification, or publication authority.

Offline validation:

```bash
python3 -m unittest discover -s repro-kit/tests \
  -p 'test_r16_real_accuracy_gate_drivers.py'
```

Both executable files intentionally carry the same standalone implementation,
apart from a fixed gate identity. Sealed driver execution cannot safely locate
an unpinned sibling implementation through `__file__`. A synchronization test
prevents the two copies from drifting.

## Real C1 memory-floor driver

`r15-gates/memory-floor` measures memory during two fresh C1 processes, Plain
and native K5. It reuses the exact reviewed accuracy lifecycle, including
mandatory QSA, actual engine pidfds, live broker proofs before and after each
arm, and graceful cleanup. The standalone file embeds the reviewed
`mmlu-pro` source digest. A different helper cannot be authorized by supplying
its own hash. Updating that helper requires an explicit pin update and review.

The identity/descriptor inputs are the same as the accuracy drivers, replacing
`--lil-program` with `--lifecycle-program <tracked r15-gates/mmlu-pro>`.
`--suite-sha256` must match the actual sealed parent suite. Runtime and launch
identities, binary/recipe pins, and checkpoint provenance are checked against
the actual parent's sealed plan, not a caller-supplied completed report.
The inherited raw-log descriptor remains read-only.

Every arm submits source-pinned synthetic token-vector workloads at 8,192,
131,072, and 262,144 total tokens, each reserving exactly 64 output tokens.
Each context uses a different first token. Requests force `ignore_eos`, greedy
sampling, and exact scheduler token IDs. The measured usage must match the
exact prompt/output geometry, report zero cached tokens, and finish at the
output budget. The mounted QAD5500 tokenizer bytes must match the embedded
source pin. Native K5 telemetry must increase after every native workload;
Plain must have zero K5 events.

Host `/proc/meminfo` and the exact container's local cgroup-v2 hierarchy are
sampled every 250 ms during startup and requests. At least one complete
sample must fall inside each actual startup/request interval. Missing reads,
identity drift, unsupported cgroup counters, or more than 10,000 samples fail
closed. The minimum available memory must be at least 6,656 MiB. Available
memory is the smaller of host `MemAvailable` and every finite non-root
ancestor's remaining `memory.max - memory.current`. Used memory is host
`MemTotal - MemAvailable`, not GPU allocation or engine-only RSS. Current/peak
container counters and the hierarchy remain in the sample journal. New OOM
events in that hierarchy fail the run. This sampler is not a watchdog or a
guarantee that sub-250-ms transients cannot occur.

Complete engine stdout/stderr are retained together, never discarded.
Readable current-boot kernel-journal access is mandatory. The driver anchors
to an actual journal cursor and verifies that the cursor is still present
when collecting all subsequent kernel events. Log tails are never used for
qualification. Allocator warnings, CUDA OOM/fatal/illegal-address errors,
generic Linux OOM/oom-kill events, NVRM and Xid fail the gate. Complete measured
engine/kernel messages are also forwarded to suite-captured stderr, without
writing the read-only raw-log FD. Size limits fail closed without truncation.

Private mode-0700 `memory/` retains mode-0400 request/response and sample
journals, full engine/kernel logs, and their hashes. The strict validator
keeps the existing six-key raw samples and exact memory evidence schema,
plus workload, process, source-pin and artifact audit details. No saved PASS
receipts are consumed. A PASS qualifies only this synthetic-token C1 memory
workload, not semantic retrieval, Plain/K5 token parity, MMLU/GSM8K, C8 memory,
native batching, or performance. Those remain separate mandatory gates.

Final artifact hashes must match the exact kernel records consumed during
validation, the measured messages derived from those records, the complete
engine bytes scanned, and the sample/request journals actually produced by
the workload. Rewriting any artifact between validation and hashing fails
without a validator or PASS receipt. Kernel baseline identity is compared as
a complete record, not only as a cursor string.

Offline tests mock every Docker, HTTP, server, kernel and proc action:

```bash
python3 -m unittest discover -s repro-kit/tests \
  -p 'test_r16_memory_floor_driver.py'
```
