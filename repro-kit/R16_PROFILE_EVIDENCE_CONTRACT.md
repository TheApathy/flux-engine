# Profile Evidence Contract

Source and CPU validation are not a performance or container qualification.
The current immutable-image live broker admits one recipe. The source-defined
interactive and high-prefill recipes have different bytes and environments.
The performance gate and independent outer wrapper therefore retain a real
**dual-recipe image/broker HOLD**. Changing scheduler flags or profile labels
cannot close that HOLD.

Image-side provenance is now implemented: strict launch manifest `/2`, four
fixed profile labels, equality to the proven source recipe contract, and actual
standoff image recipe byte hashes before and after the suite. Startup binding
and broker `/1` remain unchanged. The remaining live-broker HOLD is deliberate;
image labels or copied recipe bytes do not prove the selected server environment.

There is also a separate **profile geometry HOLD**. In the current engine,
WIDE defines actual per-chunk row geometry, not just arena capacity. The
production validator requires it to match `--max-prefill-tokens`. Canonical
high-prefill selects WIDE=16384, while this `/3` measurement contract requires
chunk 4096 for both profiles. Packaging both recipes alone cannot make that
configuration launch. Do not remove the validator or reinterpret recipe bytes
to manufacture a match.

## Clock And Workload

Raw metrics use `flux-profile-metrics/3`. Host references use
`flux-prefill-decode-host-reference/3`. Legacy references are rejected.

- Decode uses `scheduler-decode-time:completion-tokens-minus-one/v1`.
  The source-defined terminal API rate is
  `(completion_tokens - 1) / (decode_time_ms / 1000.0)`. The retained rate is
  inverted in that operation order to obtain a source-rate-derived wall. It
  is not an independently captured engine timer. SSE text arrival times are
  diagnostic only and do not define token boundaries or throughput.
- Prefill uses `client-monotonic-blocking-max-tokens-one/v1`. This is the
  prompt count divided by the client monotonic request wall for one blocking
  output token. It is a request-wall prompt-rate/TTFT proxy, not kernel-only
  prefill speed. Host tolerances and published rates use this same wall.
- The contexts are 8192, 16384, 32768, 65536, 131072 and 262080 prompt tokens.
  The first five request 256 output tokens and time 255 decode intervals.
  The final context requests 64 outputs and times 63 intervals within the
  native 262144-token window. Chunk selectors stay 4096.

The measurement contract pins the exact completions API and CLI source files.
The outer wrapper binds both to the proven source inventory. A source change
requires a reviewed pin refresh even when the clock formula is unchanged.

## Sealed Policy And Host Reference

The sealed suite plan uses `flux-r15-container-suite-plan/2` and rejects `/1`.
It requires `performance_profile_contract`, schema
`flux-performance-profile-contract/1`. Its exact profile bindings contain
`recipe_relpath`, `recipe_sha256` and `argv_overlay`. Recipe paths must be the
source-defined `flux-v9-w4096-interactive.env` and `flux-v9-max-prefill.env`
under `repro-kit/recipes`. The outer wrapper independently checks their
inventory hashes, literal assignments, FLUX mode and both wide selectors
(4096 and 16384 respectively). Those WIDE values are per-chunk geometry, so
the present universal 4096 scheduler requirement is incompatible with the
canonical high-prefill recipe. Independent arena capacity is selected through
`ATLAS_MAX_BATCH_TOKENS`, not by reinterpreting WIDE.

A future canonical interactive/high-prefill comparison needs explicit
per-profile scheduler chunks (4096/16384), a versioned measurement/host
contract and fresh matched host rows. Alternatively, a separately named and
source-pinned WIDE=4096/larger-arena recipe needs its own host baseline.
Historical w4096 results must not be relabeled as w16384 measurements.

Only reviewed serve selectors are accepted. Reordered flags, explicit FIFO
defaults and unused FIFO deadlines do not establish distinct profiles. The
example FIFO/SLAI overlays illustrate syntax only. They are not proof of the
interactive/high-prefill environment or a performance advantage.

The plan's `--host-reference-sha256` pins canonical host-reference JSON bytes,
with no trailing newline. The retained document must reproduce that exact
hash. It includes exact source revision, tree and bundle-inventory hash,
checkpoint ID/revision/inventory hash and binary hash. Each named profile
includes its recipe hash, exact overlay and six context rows. Each row retains
the prompt SHA-256, plain/K5 terminal usage, recomputable decode walls and the
prefill client monotonic start/finish operands. All three measured bands
(plain decode, K5 decode and request-wall prefill) must be within five percent.

No measured host reference is supplied by this change. Build one only from
real retained host measurements using the declared clocks and exact workload.
The all-zero example-plan hashes are deliberately inert and must be replaced
from verified artifacts. Do not turn historical recipe comments or synthetic
unit-test data into measured baseline numbers.

## Independent Recensus

The outer wrapper reopens and hashes the complete canonical response journal.
Exactly 72 unique roles must exist: two profiles, two modes, six contexts and
three request kinds (blocking exact IDs, one-token wall and streaming usage).
It rederives retained observations and raw rows from those records, checks
the exact deterministic request controls, prompt and output budgets, cache
rejection, unique response IDs, terminal usage and finish geometry. Coalesced
text events and one event per output token are both valid, including equal
arrival timestamps.

Named process bindings retain the observed exact launch argv, argv hash and
the source-pinned broker proof. The wrapper checks four distinct process
identities, C1, BF16 KV, disabled prefix reuse, plain versus native K5 selectors
and the linkage from each profile identity to its process proof. Named metric
records independently prove zero K5 verify events for plain and positive
events for native K5. Raw journal bytes, derived metadata and published
evidence must agree. Free-standing `all_*` flags do not qualify a run.
The journal is retained as an independent dependent artifact, with a globally
unique locator, actual byte count and SHA-256. Its bytes are rehashed after
container cleanup and again immediately before the outer seal. A mutation
that leaves gate and validator JSON unchanged still prevents qualification.
Blocking exact and one-token responses both require choice index zero as an
exact integer and `finish_reason=length` for their `ignore_eos` budgets.

CPU tests use explicitly synthetic fixtures and scope one mock only around
the known-incompatible runtime recipe binding when testing arithmetic or
orchestration. Separate unmocked tests require the real dual-recipe HOLD.
Neither test class is a runtime, quality, throughput or release receipt.
