# Matched Profile Driver: Source/Offline Candidate, Release HOLD

`r15-gates/prefill-decode-profiles` repairs the five reviewed profile, timing,
streaming, and child-lifecycle gaps. This is not a performance result or a
release qualification. No host measurements are supplied by this change.
Offline tests use synthetic fixtures and mock every subprocess/server,
Docker operation, HTTP request, and clock. Their positive arithmetic fixture
explicitly bypasses the production profile-policy verifier.
That fixture binds a temporary test clock/source contract to the local source
bytes so an isolated donor remains standalone when integration source moves.
This test-only replacement is not production provenance or a release proof.
The synthetic pipeline checks its row/evidence arithmetic directly and does
not invoke or mock the independent production wrapper's `/3` closure.

The real invocation remains HOLD because the immutable-image broker pins
one recipe. The integrated sealed suite plan now requires the profile
contract, and the independent wrapper recomputes the `/3` journal, clocks,
source/provenance, process bindings and all host bands. Distinct CLI scheduling
policies still do not prove the real interactive/high-prefill recipes.
See `R16_PROFILE_EVIDENCE_CONTRACT.md` for the integrated wrapper boundary.

A separate geometry HOLD remains: canonical high-prefill sets WIDE=16384,
which is real per-chunk row geometry. The production validator rejects that
recipe with this contract's required 4096 scheduler chunk. Dual-recipe image
support alone will not close this mismatch. Preserve the validator; use a
new reviewed per-profile chunk contract and host measurements, or a distinct
source-pinned 4096/larger-arena recipe with its own baseline.

## Workload Unchanged

The grid remains 8,192 / 16,384 / 32,768 / 65,536 / 131,072 / 262,080 prompt
tokens, 4,096-token scheduler chunks, C1, BF16 KV, and native capacity 262,144.
Plain disables speculation. Native K5 uses `--speculative --num-drafts 4`.
The greedy seed, sampling parameters, no-thinking framing, three passkeys,
5/50/95-percent needle positions, and request ordering are unchanged.
Each context runs an exact blocking completion, a blocking `max_tokens=1`
request, and an SSE completion. Output budgets remain 256, or 64 at 262,080.
Exact token IDs, output counts, `length` finish reasons, no cached prompt
tokens, text identity, passkeys, fresh processes, double health checks,
initial/final live proofs, and K5 telemetry remain required.

## Explicit Clock Scopes

- Decode is `scheduler-decode-time:completion-tokens-minus-one/v1`.
  Source `api/completions.rs` emits terminal `usage.response_token/s` as
  `(completion_tokens - 1) / (decode_time_ms / 1000)`. The driver reconstructs
  that scheduler duration as `(completion_tokens - 1) / response_token/s`.
  Terminal usage must prove the exact prompt/output budget and zero cache
  hits. Raw usage, finish geometry, and the measured rate are retained.
- Prompt rate is `client-monotonic-blocking-max-tokens-one/v1`:
  `prompt_tokens / (finish - start)` for the one-token blocking request.
  This is a request-wall prompt-rate / TTFT proxy, not engine-only prefill
  throughput. Both finite monotonic stamps and their exact difference are
  retained in the observation and journal. The host band and published rate
  use this same wall. Engine TTFT is only a bounded diagnostic.
- First/last text-event arrival span is transport-only diagnostic data.
  SSE text does not expose token boundaries. A first event containing forty
  tokens cannot turn 216 later token intervals into 255 timed tokens. No
  client rate is inferred from that span, and the former 20-percent
  cross-clock tolerance is removed. A healthy N-token/N-text-event stream
  is accepted; the former N-minus-one text-event bound is gone.

All three host bands remain `abs(100 * (host - measured) / measured) <= 5`:
plain scheduler decode, K5 scheduler decode, and request-wall TTFT proxy.
No bands are widened. Clocks are never silently substituted for one another.

## Host Reference /3

Existing arguments are unchanged, including `--host-reference` and its
SHA-256. Existing `/2` example/reference files are intentionally rejected.
The new reference must be canonical JSON: exactly `suite.canonical_bytes`
of its parsed object, without a trailing newline. Consequently its retained
parsed document can be independently rehashed without losing original-byte
identity. The shape is:

```text
schema: flux-prefill-decode-host-reference/3
chunk_tokens: 4096
binary_sha256: exact release executable SHA-256
decode_budget_tokens: 256
measurement_contract:
  decode_clock: scheduler-decode-time:completion-tokens-minus-one/v1
  prefill_clock: client-monotonic-blocking-max-tokens-one/v1
  completions_source_sha256: exact pinned API source SHA-256
  cli_source_sha256: exact pinned CLI/defaults source SHA-256
provenance:
  source_revision, source_tree, source_bundle_manifest_sha256,
  checkpoint_id, checkpoint_revision, checkpoint_manifest_sha256
profiles:
  interactive / high-prefill:
    argv_overlay, host_recipe_sha256, rows
```

Each profile has exactly six rows. Each row carries `context_tokens`,
`decode_tokens` (255 or 63), `k5_decode_wall_seconds`,
`plain_decode_wall_seconds`, `prefill_wall_seconds`, `prompt_sha256`, raw
`k5_usage`, raw `plain_usage`, and `prefill_client_monotonic` with float
`start`/`finish`. Declared decode walls must exactly equal the source-defined
raw-usage inversion. The host TTFT-proxy wall must exactly equal its clock
proof. Prompt hashes must match actual tokenizer-built requests.

The driver reads the actual launching suite's fully sealed `--plan-fd`
through the parent's `/proc` descriptor, using the source-pinned suite's
`load_plan`. Host source and checkpoint provenance must match that plan.
The binary and recipe must match immutable image labels and the sealed plan.
Image source revision/tree must match too. API and CLI source bytes under
the plan's source root must match the driver's pinned hashes. These checks
do not create a source-to-binary or checkpoint proof: they consume the
wrapper's already independently verified plan/provenance, and that trust
boundary still needs the root-owned wiring audit.

The reviewed completion-handler pin is
`baf0f06930419df9bedba2318de591e74a6f0a95fd5f4bebe014f7bebdb3ae27`,
the exact bytes at integration `3e125d78`. Relative to the original donor,
the handler adds opt-in intervention-ledger reporting and request tests.
Exact token-ID prompts, terminal usage/finish geometry, both decode-rate
formulas, and TTFT scope remain unchanged. Reporting defaults to false, and
the response field is omitted when reporting is absent. The CLI pin remains
`6fc6122f1b501f66009964da953a7d3d156283d39858fceeaaaaae322b0a5318`.
Production still requires exact source bytes; no old-pin fallback exists.

## Profile Identity And Required Wiring

Only source-known paired selectors are accepted: `--max-prefill-tokens`,
`--max-seq-len`, `--scheduling-policy`, and `--tbt-deadline-ms`. Duplicates,
unknown flags, malformed values, and alternate assignment grammar fail.
Canonicalization normalizes order, integers, default FIFO, and FIFO's unused
deadline. Equivalent configurations HOLD even when argv lists differ.

The required sealed plan `/2` contract has this exact shape:

```text
performance_profile_contract:
  schema: flux-performance-profile-contract/1
  profiles:
    interactive / high-prefill:
      recipe_relpath
      recipe_sha256
      argv_overlay
```

The paths must name the source-known recipes
`repro-kit/recipes/flux-v9-w4096-interactive.env` and
`repro-kit/recipes/flux-v9-max-prefill.env`. Checked recipe bytes must encode
both public and internal `QWEN4_PREFILL_WIDE` values 4096/16384 and the known
`FLUX_MODE` values. Hashes and effective overlays must match the host
reference. They must also match the actual immutable image/broker recipe;
one current recipe cannot satisfy both different canonical recipe hashes.
Names, a JSON PASS flag, or FIFO versus SLAI cannot bypass this check.

The integrated wrapper requires `/3` raw measurements, exact clock and
source/checkpoint provenance, canonical host-reference bytes, a source-bound
profile contract and the exact context/output census. It independently
replays terminal usage, client clock operands, output identity, all 72 journal
roles, the three five-percent bands and aggregate flags. Dependent journal
bytes are rehashed after cleanup and immediately before sealing. The suite
plan example now uses `/2`; its unset hashes remain deliberately inert.

Before any production PASS, the remaining root-owned work is:

1. Define how two real source recipes are selected and source/image-byte
   verified. Update immutable recipe selection and live broker checks without
   weakening recipe/environment identity. The present single-recipe policy
   intentionally cannot pass this driver for both recipes.
2. Produce a real canonical `/3` host reference and refreshed source-inventory
   seals without inventing measurements. The checked-in `/2` host-reference
   example is legacy and intentionally rejected, not a measured baseline.
   Produce real matched host and
   immutable-image receipts with the same checkpoint, binary, generation,
   prompt hashes, recipe bytes, and clock scopes.

Legacy twelve-key row/evidence arithmetic is retained only for compatibility
testing; it is not independent /3 closure. `raw` now explicitly carries the
/3 measurement schema/contract, profile contract, host reference/hash,
observations, all host bands, process proofs, and response journal binding.
Every journal entry names its profile, mode, context, kind, prompt hash, and
request selectors. Blocking responses retain their parsed JSON and finite
client monotonic clock; streaming entries retain relative event timestamps,
event JSON text, and that client clock. The journal has exactly 72 requests.
Each observation names its profile, context, prompt hash, and the plain/native
arms' exact, one-token, and stream measurements.

`process_bindings` has four explicitly named entries with `profile`, `mode`,
`proof`, `launch_argv`, and `launch_argv_sha256`. The argv hash uses canonical
JSON bytes. `verification_metrics` has four named entries with `profile`,
`mode`, `k5_events`, `metrics_sha256`, and the raw `metrics` text. Profile
identity retains the source revision/tree/bundle hash, contracted recipe
path/hash, host recipe hash, exact and effective overlays, and each mode's
SHA-256 of its canonical live proof. The PID, start-time, and broker identity
inside each proof must be independently checked; matching hashes alone do
not qualify a proof.

## Launch-Owned Cleanup

`docker exec -i` runs an inline Python supervisor sealed within the driver.
Engine argv is encoded so its model name cannot cause supervisor/process
census confusion. The engine's exact argv/environment are unchanged. Logs
stream through suite-owned stderr; supervisor stdout holds only a bounded,
nonce-bound child-retirement acknowledgement. A GO byte enables child launch.
Before initial and final live proof, exact `/proc/PID/cmdline` bytes must equal
the intended launch argv while the same claimed pidfd remains live. This
binds actual chunk/context, C1, BF16, speculation, and profile selectors rather
than recording only an intended launch configuration.
EOF on the driver-owned control pipe requests cleanup even after a failed
claim. The supervisor owns an unreaped direct child and pidfd, never a stale
numeric shell PID. TERM/KILL pidfd failures use bounded `Popen` direct-child
fallbacks and cannot skip wait/reap. Errors, forced termination, missing ack,
or an unproven Docker-client disconnect fail closed. No whole-container kill
is used. Real Docker behavior remains unqualified by these offline tests.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
python3 -B -m pytest -p no:cacheprovider -q \
  repro-kit/tests/test_r16_prefill_decode_profiles_driver.py
```
