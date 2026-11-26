# Paired plain/native-K5 long-context release driver

`r15-gates/plain-k5-long-context` is the operational driver for the
`plain-k5-long-context` suite gate. It collects the issue #13 post-fix runtime
receipt: fresh-process greedy identity between target-only plain decode and
native K5 (`--speculative --num-drafts 4`) at every required long-context rung,
with exact scheduler-owned token IDs, finish reasons, usage, a target-state
receipt, and a subsequent ordinary target token. It consumes no earlier report
and accepts no evaluator-supplied PASS. Offline tests mock every Docker, server,
and HTTP interaction and never qualify an image.

## Inputs and dependencies

The driver takes the same suite identity and inherited descriptors as the
MMLU-Pro and GSM8K drivers plus one directory argument:

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
--qualifier-dir <the tracked source repro-kit directory>
--container-port 8888
```

The checked-in plan example carries this exact argument set, and the offline
test parses that entry against `parse_args`. The suite program must be the
sealed launching suite (parent memfd pin) and must include the live
listener/model/native-mode broker. From `--qualifier-dir` the driver loads three
tracked modules by pinned SHA-256 from checked bytes, in dependency order:
`qwen4_endpoint_admission.py`, `r9_source_to_binary.py`, and
`qwen4_mtp_long_context_qualify.py`. A hash mismatch fails before any module
executes, and a test asserts the pins match the tracked sources.

Reused by import from the pinned qualifier, not rewritten:

- the deterministic known-answer workload (`build_request_bundle`,
  `assemble_prompt`, `completion_body`, tokenizer parity),
- exact-ID response validation (`validate_completion_response`,
  `validate_token_vector`), `compare_arms`, and `execute_requests`,
- per-request engagement proof (`validate_request_log_delta`,
  `parse_mtp_accept_frames`, `parse_completed_spec_cycles`),
- the endpoint gate (`execute_endpoint_admission_probes`: 262143+1 accepted at
  the inclusive endpoint, 262144+1 rejected with HTTP 400 and no dispatch,
  post-rejection health canary and exact prefill metric deltas, final
  whole-log dispatch census),
- the graph-disabled rollback/slot-reuse arm and the QSA route telemetry v3
  consumer (`execute_rollback_slot_reuse`, `validate_qsa_route_receipts`,
  `validate_log_contract`).

## Execution contract

The container, image labels, read-only root, and literal recipe are inspected
exactly as the accuracy drivers do. Three fresh C1 processes run sequentially
under the canonical broker argv (`serve --no-tui --model-from-path /models`,
unique model name, `--bind 0.0.0.0`, declared port, `--max-batch-size 1
--max-num-seqs 1 --max-seq-len 262144 --kv-cache-dtype bf16`, no `--mtp-gate`,
no foreign or auxiliary drafter selector):

1. `plain`: `ATLAS_DISABLE_SPECULATION=1`, no speculative selector.
2. `k5`: `ATLAS_DISABLE_SPECULATION=0 --speculative --num-drafts 4`.
3. `k5_eager_churn`: the K5 argv plus `ATLAS_DFLASH_DEBUG_NO_GRAPH=1` and
   `ATLAS_QWEN4_QSA_ROUTE_TELEMETRY=1`.

Each process requires two `/health` probes, the suite-owned live process proof
(mode `none` for plain, `native-mtp-k5` for both K5 arms), the same proof again
after the arm, pidfd-bound cleanup, and distinct PID/start-time identities.
The server's output is captured through the `docker exec` client into a sealed
per-arm log beneath the runtime directory and echoed to the gate's stderr so
the suite's complete raw-log census still sees it.

The gate needs receipt logging from the live process: `ATLAS_MTP_ACCEPT_LOG=1`,
`ATLAS_DFLASH_SPEC_CYCLE_V2=1`, `ATLAS_DFLASH_ASYNC=0`, and a `RUST_LOG`
directive enabling `spec_accept_log=debug`. The live broker compares every
recipe assignment against the process environment, so these are added only
when the release recipe leaves them unpinned or pins the same value; a recipe
that pins any of them to another value, or pins `RUST_LOG` without the
`spec_accept_log=debug` directive, fails closed. The K5 arm additionally needs
the recipe's `ATLAS_QWEN4_VERIFY_GRAPH=1` route (the pinned log contract
requires a verify CUDA-graph capture marker) and `ATLAS_QWEN4_QSA_VERIFY_ROWS`
engaged. The eager arm's `ATLAS_DFLASH_DEBUG_NO_GRAPH=1` forces the eager
verifier regardless of the graph selector without touching a recipe key.

The recipe selects one of two native-K5 draft policies (`k5_draft_policy`,
recorded in the validator evidence). `fixed-k5`: every ATLAS/FLUX alias of
`MTP_CONF`, `MTP_CONF_EARLY` and `MTP_VOCAB_PRUNE` is absent or exactly `0`,
and the K5 arms run with all six forced to `0`. `confidence-gated-k5`: both
aliases of each selector are present and equal, `MTP_CONF` is a decimal
`0.d{1,4}` strictly above zero, `MTP_CONF_EARLY` is `0` or `1`,
`MTP_VOCAB_PRUNE` is `0`, `ATLAS_QWEN4_QSA_JOURNAL_ALL_K` is unset or `1`, and
`ATLAS_QWEN4_MTP_GATE_QSA_MIN_WIDTH` is unset. The K5 arms then keep the
recipe's values. The confidence gate only shortens draft chains, and verify
still decides every committed token. This applies only to the C=1 serve this
gate runs. Batched K5 (C>1) keeps the server's fixed-K5 contract. Any other
combination fails closed, including either alias of `MTP_CONF_FILE`, which is
always refused.

The gate passes the policy to the pinned qualifier
(`configure_k5_draft_policy`) before any server starts. Under
`confidence-gated-k5` the log parsers accept `MTP_ACCEPT` frames with 1..4
verified drafts (still `num_drafts=4` requested), `SPEC_CYCLE_V2` records with
`gamma` 1..4 and `k = gamma + 1`, and QSA route frames of 2..5 rows. Each route
frame must have one more row than its paired MTP frame has drafts. Chains
shorter than 4 run the K2..K4 verify steps, which emit no `SPEC_CYCLE_V2`
record. A request whose first frame is full width must still bind its
completed cycle. Fixed K5 parsing is unchanged.

Per arm, in order:

- `plain`: tokenize and seal the request bundle; run every paired case
  (2,017, 2,043-2,050, 2,112, 4,096, 7,336, 7,710, 8,192, 30,647, 32,768,
  65,536, 114,755, 122,413, 131,072, 238,938, the stopped 262,080 semantic
  case, and the `262080+64 ignore_eos` endpoint fill) twice each; run the
  continuation probes; run the endpoint admission probes.
- `k5`: tokenizer parity, the same cases with per-request native-MTP
  engagement required at the exact first-speculative geometry, continuation
  probes, a real `262143+1` one-token boundary control, and a per-case
  acceptance-depth census.
- `k5_eager_churn`: the 2,043-2,050 boundary rows with per-request QSA route
  receipts (exact 12-layer set, K=5 rows only, monotone totals from a zero
  baseline), then the 2,017 / 2,050 / 2,017 rollback and single-slot reuse
  sequence.

A continuation probe sends the paired prompt plus the arm's returned completion
IDs with `max_tokens=1`, `ignore_eos=true`, and requires one exact
scheduler-owned token, `finish_reason="length"`, `cached_tokens=0`, and one
scheduler dispatch. It is skipped, identically in both arms, when the extended
geometry would exceed the 262,144-token endpoint. Plain and K5 continuation
tokens must be identical.

## Exact target-state comparisons

The driver arms `ATLAS_QWEN4_TARGET_STATE_RECEIPT=1` and requires one complete,
strictly parsed schema-1 receipt for every successful completion. A recipe
that disables or limits these receipts fails closed. No receipt is permitted
in the rejected `262144+1` request delta. A whole-log census rejects missing,
duplicate, failed, malformed, or unassigned state receipts.

The target-fed boundary is deliberately one token behind the emitted output:
`seq_len == token_count == prompt_tokens + completion_tokens - 1`. The exact
fed vector is `prompt_token_ids + completion_token_ids[:-1]`. This follows
ordinary decode's input-token commit and
`verify_dflash_step::reconcile_terminal_verify_state`, not the HTTP usage
total. The driver independently recomputes the emitter's framed
`atlas-qwen4-fed-tokens-v1` digest from that vector. Equal but incorrect fed
hashes in two arms do not satisfy the check.

Each receipt must retain the real source mode. Plain is
`unloaded-target-only / ordinary-target-only / proposer_loaded=false`.
Both K5 processes are `engaged-native-mtp / native-mtp / proposer_loaded=true`,
including a one-token completion with no verify frame. These fields identify
the installed proposer and scheduler control. They do not prove that a K5
cycle executed for that particular one-token request. Actual verification
engagement remains subject to the unchanged request-scoped MTP/SPEC_CYCLE
checks where the request budget permits it. Loaded-serial receipts cannot
substitute for either arm.

The emitter must include production classifier repair
`d31873532f7fd10a890677d30e78fdd2637c5c57` (isolated follow-up dependency
`802e213d`). Without it, the legitimate Plain combination of no proposer and
`ATLAS_DISABLE_SPECULATION=1` fails state capture. No loader, scheduler,
receipt-schema, qualifier, or wrapper relaxation is made by this driver.

The comparator checks GDN/attention census, live PLE plus logical BF16 KV
coverage, exact fed-token ownership, buffer/byte census, session metadata,
and the final target-state hash:

- Every repetition of every paired case compares with its Plain counterpart,
  including stopped retrieval and the `262080+64 ignore_eos` fill.
- Every non-skipped continuation compares target state with its Plain
  continuation, as well as the subsequent ordinary output token.
- Every graph-disabled boundary request compares against both Plain and
  graph-enabled K5 controls.
- All three rollback/slot-reuse requests compare response and target state
  against their canonical Plain controls, including the rollback trigger.
- The real native `262143+1` boundary request compares with the Plain
  admission request. Both states contain 262143 fed tokens and retain the
  final emitted token as pending. This does not claim a 262144-token fed
  state.
- The post-overflow Plain completion canary compares response and state with
  the pre-overflow Plain control.

Request evidence binds each receipt to its actual log offset, length, and
digest. Byte-identical Plain repetitions claim different offsets exactly
once. Before sealing, claimed ranges must remain disjoint and unchanged.
The raw validator retains these receipts and comparisons without changing
the wrapper's exact 13 evidence keys or 10-key rung rows. Any unsupported
fed-token geometry or failed comparison stops the run rather than inferring
or normalizing a matching digest.

After each arm the complete server log is scanned. Any CUDA fatal, CUDA out of
memory, illegal memory access, NVRM, Xid, KV-cache exhaustion, overflow
truncation, or panic marker makes the run VOID (exit 3, no receipt). An
ERROR-level line fails closed unless it lies inside the correlated overflow
rejection delta. Every `/metrics` snapshot is retained; the plain arm must show
zero `atlas_spec_decode_verify_total{k="5"}` events and both K5 arms must
show more than zero. The counter is engagement evidence only.

## Route policy

Route receipts are classified against an explicit release policy. Qwen4
switches to sparse QSA at position 2,048. At this source revision the native
MTP verifier attends row by row for every frame (`SPARSE_ROWS_V2_QUALIFIED =
False`), so every telemetry frame must report `row-serial` for all 12 layers;
a `sparse-rows-v2` or `sparse-rows-legacy` frame fails the gate because an
unqualified route served the release. When the sparse multi-row route is
qualified, flipping the constant makes the policy require `sparse-rows-v2` for
frames whose rows are all sparse (`seq_len >= 2048`) and reject a row-serial
fallback there; the offline test covers both branches.

## Evidence

Per-rung booleans mean exactly:

- `token_identity`: identical completion token IDs in every repetition and an
  identical subsequent ordinary target token.
- `text_identity`, `finish_reason_identity`, `usage_identity`: identical text,
  finish reason, and exact prompt/completion/total usage.
- `native_k5_engaged`: every K5 request has an `MTP_ACCEPT` frame and a
  completed `SPEC_CYCLE_V2` record at the exact first-speculative position
  bound to exactly one runtime request ID.
- `partial_accept_cycle`: at least one completed K5 cycle in the request
  accepted fewer than four drafts, so the rollback and commit path completed.
- `qsa_verify_rows_engaged`: the multi-row QSA verify selector marker is in
  the K5 process log and every request's completed cycles carried five target
  rows.

Both 262,080-token cases fold into one row (AND of the booleans, the full
endpoint's 262,144 total); the 4,096 ladder case is retained under
`additional_cases`. The gate receipt carries the outer wrapper's exact 13
evidence keys, and the sealed `flux-r15-gate-validator/1` document's `raw.cases`
rows are the exact 10-key rows the wrapper recomputes. The validator also
retains per-arm results, acceptance depth, continuation identity, endpoint
admission, rollback/slot-reuse, route policy, process bindings, metrics, and
the qualifier pins.

Artifacts (mode 0400, no-follow exclusive creation under
`long-context/plain-k5/`): `requests.json` (the request manifest and every
exact prompt body), `responses.jsonl` (every journaled request hash and
response), `plain.server.log`, `k5.server.log`, `k5_eager_churn.server.log`,
and `validator.json`.

## Relationship to sibling gates

The native-262K gate (`native-262k`) proves BF16 retrieval targets and the
boundary/overflow admission on its own target-only process. This driver runs
the endpoint probes itself on its plain process because suite gates run
sequentially with sealed per-gate receipts and cannot consume one another's
receipt descriptors. The two receipts are independent evidence of the same
admission contract.

Offline validation:

```bash
python3 -m unittest discover -s repro-kit/tests \
  -p 'test_r16_plain_k5_long_context_driver.py'
```

The historical state-comparison checkpoint passed 37/37 focused tests,
including hostile
fed/state-digest mismatches at every covered phase, strict mode/coverage and
one-token-behind checks, offset ownership, late log mutation, and wrapper
artifact verification. The unchanged qualifier suite passes 36/36,
classifier source/proof checks 6/6, live-process broker checks 22/22, and
`git diff --check` passes. Every Docker, process, HTTP, state buffer, and
target-state hash in the focused integration tests is mocked. The fake state
hash is explicitly an offline fixture, not a GPU receipt. The later lifecycle
checkpoint and its 58-test coverage are described below.

This source and its offline tests are not an issue #13 runtime receipt. The
receipt exists only after the sealed suite runs this driver against the exact
immutable image with claimed GPU resources.

## Lifecycle Review and Runtime Hold

The setup-window and failed-claim containment blockers are closed at
source/offline scope by reviewed donors `d10f2d60` and `1623ebd4`. The
in-container supervisor waits for GO until output setup succeeds, owns its
direct child through a pidfd, retires it on control EOF, and requires a
nonce-bound cleanup acknowledgement. Cleanup syscall failures preserve the
failure while attempting bounded child retirement.

The reviewed focused driver suite passes 58/58 offline tests, including
partial setup, failed ownership claims, missing acknowledgements,
disconnects, forced cleanup, and pidfd signal failures. These mocked results
do not qualify real Docker handoff, GPU state parity, or a container.

Issue #13 and release qualification remain on runtime HOLD until the sealed
suite passes the complete paired workload against the exact immutable image
with claimed resources and no disqualifying runtime signals. The complete
raw-log gate remains unchanged, including its NVRM checks.
