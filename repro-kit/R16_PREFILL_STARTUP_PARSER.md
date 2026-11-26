# R16 Prefill Startup Parser

Source-only descendant of Plain QSA closure
`c032dfbb3e880b3670dd65d881c2ccb2e9147dd7`, tree
`a92828cceab730bc13c446536a20b37a08fd809e`.

## Scope And Audit Correction

The original WIDE finding was partly stale. The parent already validates the
closed width set, exact multi-chunk selection and equality to the scheduler
chunk. That guard ran only after weight loading, immediately before factory
construction. This change invokes the same validator during preflight, directly
after shared budget resolution, before arena-size computation and weight
loading. The existing late defensive call is retained. GPU initialization still
precedes this preflight, so this does not claim pre-GPU argument rejection.

`ATLAS_MAX_BATCH_TOKENS` previously treated a non-Unicode value as absent and
warned/fell back for malformed or undersized values. Those cases now return an
error from the shared budget resolver rather than silently use another arena
geometry. The caller already propagates resolution failure.

Only `serve_phases/kv_cache.rs` and the one early call in
`serve_phases/preflight.rs` change production behavior. No model, scheduler,
profile, floor, resource controller, grouped-loader or proposer policy changes.

## Exact Contract

| Input | Behavior |
| --- | --- |
| WIDE absent or exactly `0` | Disabled, with the multi-chunk selector inert |
| WIDE `4096`, `8192`, `16384` or `32768` on Qwen4 | Requires multi-chunk exactly `1` and matching resolved prefill budget |
| Any other WIDE text or non-Unicode WIDE on Qwen4 | Error |
| Non-Unicode multi-chunk with active WIDE on Qwen4 | Error |
| WIDE on another model family | Inert, including uninterpretable selector bytes |
| MAX_BATCH_TOKENS absent | Existing resolved default remains unchanged |
| MAX_BATCH_TOKENS present | Nonempty ASCII decimal digits, representable as `usize`, at least the resolved minimum |
| MAX_BATCH_TOKENS empty, non-Unicode, signed, spaced, malformed, overflowing or smaller | Error |

The resolved minimum remains the maximum of prefill plus logical decode slots,
the speculative-row allowance and padded physical decode rows. Existing
prefix-checkpoint and high-speed-swap clamps are unchanged. Leading decimal
zeros remain accepted for MAX_BATCH_TOKENS. Zero is not a documented disable
sentinel for this override and is below every ordinary resolved minimum.

No arbitrary maximum or new memory allowance is introduced. Successfully
parsing a large value does not prove its downstream allocation arithmetic,
memory fit or safety. Existing memory checks and measured floors are unchanged.
This does not repair the model helper's independent cached environment reader
or redesign environment mutation during startup.

## Evidence And Limits

Ten new Rust test definitions cover supported/invalid WIDE values, scoped
non-Unicode handling, disabled behavior, MAX_BATCH_TOKENS grammar and bounds,
prefix-checkpoint and speculative minima. Existing tests additionally assert
rejection beneath C128, high-speed-swap and padded concurrency minima.

Nine bounded Python source-wiring checks bind environment reads, strict parser
error propagation, shared minimum derivation, server/model width tables, early
and late guard placement and the new native regression definitions. Mutants
exercise lossy reads, absent lower bounds, missing propagation and late/missing
or incorrectly bound WIDE validation. These tests inspect source, not compiled
Rust behavior.

Offline checks run with bytecode disabled, a 90 MiB address-space cap, five CPU
seconds and a finite 50-second external timeout. Rust tests are not compiled or
executed. No native build, model load, GPU run, quality/performance qualification,
image build or GHCR promotion is implied.
