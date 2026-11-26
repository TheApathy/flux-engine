# Flux Lens

Flux Lens is an original, Python-standard-library-only harness for exact-token
blocking completion experiments. It does not import another benchmark's source
or infer engine capabilities from a user-supplied arm name.

The first increment provides offline workload planning, finite concurrent
bursts, private request/response journals, locally replayable integrity hashes,
exact-token comparison, and reusable reports. Only `run` sends HTTP requests.
There is no automatic deployment, model download, upload, server reconfiguration,
warmup, retry, or endpoint discovery.

## Corpus

Use an explicit token-ID corpus. Tokenization and checkpoint selection are
outside this MVP. Each case is a separate matrix row, so a short default burst
cannot silently omit the rest of the corpus.

```json
{
  "schema": "flux-lens.corpus.v1",
  "model": {
    "name": "served-model-name",
    "revision": "declared-checkpoint-revision-or-digest"
  },
  "cases": [
    {
      "id": "small-exact-example",
      "prompt_token_ids": [10, 11, 12],
      "max_tokens": 8,
      "expected_token_ids": [30, 31]
    }
  ]
}
```

These IDs are schema examples, not a meaningful model prompt or expected
answer. Supply independently established IDs for the checkpoint being tested.
Revision/name fields are declarations, not verified model provenance.

Every case needs a nonempty prompt, positive output budget, and unique ID.
Expected output IDs are optional. When supplied they must be the complete,
nonempty expected completion, including any returned structural/EOS tokens.
No text normalization, prefix-only matching, or manual acceptance is used.
An unscored case never receives a quality-pass headline.

## Plan, Run, Compare, Report

From the repository root, plan a case x concurrency x burst-size x repetition
matrix without contacting any endpoint:

```sh
python3 repro-kit/flux_lens.py plan \
  --corpus corpus.json \
  --concurrency 1,2,4,8 \
  --burst-sizes 8,16 \
  --repeats 3 \
  --out lens-plan.json
```

Each burst size is the total number of requests in that cell, not a per-worker
count. It must be at least the largest planned concurrency. Every cell repeats
its selected case across all burst requests. A start gate releases C client
workers, each assigned a deterministic strided subset of that burst. Cells
execute sequentially. This is finite offered client concurrency, not proof of
C simultaneous native engine rows or sustained throughput. The plan is bounded
to C <= 128 and 10,000 total requests, including every case and repetition.

Only the following explicit command contacts an endpoint:

```sh
python3 repro-kit/flux_lens.py run \
  --plan lens-plan.json \
  --endpoint http://127.0.0.1:8080/v1/completions \
  --outdir lens-baseline
```

The endpoint must be the exact `/v1/completions` path, without credentials in
the URL, a query, or fragment. A bearer key may be read with
`--api-key-env FLUX_LENS_API_KEY`. The key and request headers are not written
into artifacts. An optional `--identity identity.json` accepts a flat map of
declared string metadata, such as engine/image revisions or a profile name.
This metadata stays private and never proves the executed binary, model,
speculative mode, physical K, or native batch width.

The socket timeout defaults to 300 seconds and can be set with `--timeout`.
It is a transport/socket timeout, not an independent hard wall-clock kill of
the remote request. There are no retries, redirect follows, environment proxy
handlers, health probes, or automatic server mutations. A failed/interrupted
run cannot be resumed or spliced into a fresh one. Choose a new output directory.

Run a separately configured candidate against the same plan, then compare:

```sh
python3 repro-kit/flux_lens.py run \
  --plan lens-plan.json \
  --endpoint http://127.0.0.1:8081/v1/completions \
  --outdir lens-candidate
python3 repro-kit/flux_lens.py compare lens-baseline lens-candidate
python3 repro-kit/flux_lens.py report lens-candidate
```

All run directories must be new. Files are created exclusively, not overwritten.
`compare` requires separate run directories, identical case/token corpus, model
declarations, generation controls, matrix, finish policy, assignment contract,
and harness source hash. Every request must validate, every paired full output
token-ID array and finish reason must match, and supplied expected IDs must not
mismatch. Missing IDs, matching token counts with different IDs, error payloads,
or incomplete journals cannot pass comparison. Relative rates are observed E2E
ratios with token parity, not a quality-qualified or native-MTP speedup claim.

Exit codes: 0 for a completed operation with no failed request/expected-ID
mismatch, 1 for a recorded failed run or expected-ID mismatch, and 2 for refused
inputs, a failed comparison, an incomplete/invalid receipt, or an IO error.
Exit 0 on an unscored run means collection completed, not that accuracy passed.

## Request And Response Contract

Requests use `prompt_token_ids`, `return_token_ids=true`, blocking `stream=false`,
one completion, no echo, temperature 0, top-k 1, top-p 1, min-p 0, seed 0,
repetition penalty 1, and presence/frequency penalties 0. These controls are
captured in the workload fingerprint. The harness does not assert that an
unverified server honored every control or had no additional interventions.

Responses must have HTTP 200, the declared model name, exactly one indexed
choice, nonblank text, nonempty full `choices[0].token_ids`, and matching usage:

- `prompt_tokens` equals the exact input ID count, with no extra template tokens.
- `completion_tokens` equals the full returned completion ID count.
- `total_tokens` equals prompt plus completion, within the planned output budget.
- `usage.prompt_tokens_details.cached_tokens` is explicitly the integer zero.

Absent cache details are UNKNOWN, not zero, and fail this strict MVP. Explicit
zero is a reported-cache observation only. It does not prove a fresh process,
empty GPU caches, unloaded prefix state, no warmup, or cold TTFT.

By default only `finish_reason=stop` is admitted. If a fixed output budget is
intentional, set `--allow-length` during planning. This policy is fingerprinted,
and a `length` completion is accepted only with exactly `max_tokens` IDs. It
does not prove the answer is complete or correct. An expected-ID score requires
full exact equality even on a planned length finish. Empty, truncated JSON,
error, unknown-finish, over-budget, malformed-type, bool-as-int, and nonfinite
payloads fail closed. The body limit is 64 MiB.

## Metrics And Accuracy Limits

The throughput metric is **E2E aggregate output tok/s**: the sum of valid output
IDs across a complete cell divided by the interval from releasing its client
start gate to completion of its last HTTP response body. It includes dispatch,
network, queueing, prefill, and output generation. It is NOT decode tok/s,
prefill tok/s, TTFT, inter-token latency, or a native batching measurement.
Invalid cells have a null aggregate rate rather than a partial-success rate.

Per-cell E2E request latency p50/p95 uses the nearest-rank method on valid
requests' monotonic start-to-body-completion intervals. It is explicitly NOT
TTFT or inter-token latency. Every cell includes its expected-token scoring
state. A measured rate remains only an observation if a case is unscored,
mismatched, or incomplete. The harness does not issue a correctness-qualified
performance headline.

An expected-ID match is a narrow corpus score. Even all expected IDs matching
does not establish MMLU-Pro/GSM8K/BFCL quality, semantic correctness, no
interventions, safe tool behavior, greedy equivalence on other prompts,
correctness at another context, or readiness to release a container.

The completions endpoint does not supply an intervention ledger. Intervention
attribution is always `unverified` here. Model/binary/image identities remain
`declared_not_verified`, physical native MTP width and engine active batch width
remain `unknown`, and cold-process state remains `unverified`. Separate
request-bracketed engine-side proof is required to make those claims.

## Private Receipts And Reusable Output

Run and raw directories are mode 0700. Plans and artifact files are mode 0600.
Raw requests, response bodies, declared endpoints/identities, monotonic/UTC
timestamps, and the append-only attempt journal remain local. No artifact is
uploaded automatically. Treat even token-ID-only prompts as sensitive data.

`manifest.json` hashes the exact artifact files. `report` checks their hashes,
revalidates every request/response and timeline, and recomputes `result.json`
rather than trusting saved totals. A missing final manifest/result means the
run is unsealed/incomplete and is refused. Hashes provide locally replayable
integrity checks, NOT a digital signature, trusted attestation, immutable
storage, or protection against someone who can rewrite the owned directory
and recompute all hashes. This is an owned-local-artifact workflow, not an
adversarial filesystem service.

Reusable CLI JSON deliberately omits free-form case IDs, endpoint URLs,
local paths/usernames, identity values, response text, and exception messages.
It uses numeric case indices, context/budget sizes, fixed status codes, and
fingerprints. Raw journals are not public reports. Review before sharing any
artifact, especially if your workload sizes or fingerprints are sensitive.

## Validation And Extensions

The MVP test suite uses injected/mock HTTP transports and forbids real sockets:

```sh
python3 -m unittest discover -s repro-kit/tests -p test_flux_lens.py -v
```

Mock tests establish harness behavior only. They are not an engine, model,
container, native-MTP, 262K-context, throughput, or accuracy receipt.

### Flux-Specific Qualification Roadmap

These extensions are planned, not capabilities established by this MVP. Their
purpose is to identify usable performance gains without hiding accuracy,
context, state-isolation, or serving regressions. Each needs its own tests and
evidence contract before it can affect a release decision.

1. **Evidence-bound performance comparisons.** Connect trial requests to the
   existing plain/native-K5 accuracy and intervention gates through verified
   request, source, model, image, and mode identities. A saved PASS from another
   run must not qualify a rate. Missing provenance or intervention accounting
   must remain unknown, and broad quality scoring must remain distinct from
   the MVP's narrow expected-token matches.
2. **Boundary and state campaigns.** Add reusable 2,048-boundary and inclusive
   262,144-total-token campaigns, BF16 KV proof, and explicit over-limit refusal.
   Pair plain/K5 outputs with recurrent-state and subsequent-target-token
   checks. Extend to cancellation, order changes, slot replacement, and reuse
   against independent C1 controls. Prompt length alone must not establish
   successful long-context retrieval or isolated native concurrency.
3. **Streaming and sustained measurements.** Attribute full scheduler-owned
   token IDs while measuring TTFT, ITL, and steady output windows. Preserve
   transport arrival timing separately from engine prefill/decode timestamps.
   SSE events and text fragments must never substitute for counted tokens.
   Disconnects, timeouts, and incomplete streams must invalidate the relevant
   comparison rather than disappear from successful-request statistics.
4. **Scheduler pressure and recovery.** Exercise closed/open-loop arrival
   schedules, mixed long-prefill/short-decode traffic, backpressure, overload,
   aborts, and clean drain. Bind offered concurrency to observed active engine
   rows, native proposer/verifier engagement, physical K, and queue depth.
   Client worker counts must not be relabeled as native batching support.
   Add tool and structured-output workloads with full intervention accounting.
5. **Evidence-linked reports.** Build privacy-safe context/profile cards and
   performance frontiers from comparable, quality-qualified runs. Include
   repeated paired experiments, randomized A/B order, uncertainty, explicit
   warm/cold state, and failures alongside successful cells. Neither a best
   sample nor an unscored exact-output match should become a release headline.
