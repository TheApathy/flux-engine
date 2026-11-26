# R16 QSA Row-Serial Verify Journal

Base: `c032dfbb3e880b3670dd65d881c2ccb2e9147dd7`.

This is an isolated source repair, not runtime qualification. It does not
include the separate Qwen4 fixed-K3/K4 delegation patch.

## Failure And Repair

Qwen4 generic verification defaults unqualified physical widths, including K4,
to `RowSerialOracle`. That route already saved recurrent H/conv intermediates
and used ordinary Qwen4 decode, PLE and final mixing. It did not snapshot QSA's
physical raw-ring and compressed-key destinations. The existing commit path
nevertheless restored the journal for those widths, so a rejected suffix could
leave QSA writes behind.

The oracle now uses the existing canonical predicate:

```rust
metadata.qwen4_qsa_required && crate::layers::qwen4_qsa::verify_journal_rows(k)
```

That preserves existing 2..=9 coverage and the explicit all-K diagnostic opt-out.
K5 remains journaled by the existing predicate. No hybrid selector is enabled,
no width limit is changed, and no environment or serving profile is changed.

Object-safe `TransformerLayer` hooks prepare the existing undo journal once
before the row loop and snapshot each physical destination before its decode
writes. Non-QSA layer defaults do nothing. The Qwen3 attention implementation
delegates to existing indexer APIs and reports a requested but absent indexer
as an error, matching the batched route's fail-closed behavior.

The snapshot receives the exact row-offset metadata slot, chronological row,
backend and caller stream. This must happen before decode, not after it:
the kernel saves overwritten values, and row zero resets the valid mask.
Preparation and snapshot errors propagate with `?`; a failed row does not
proceed to the next row. Existing recurrent snapshots stay after completed
decode and existing commit/abort code is unchanged.

## Side-Stream Failure Ordering

Simply adding snapshots is insufficient when ordinary QSA overlap is enabled.
Plain decode can launch indexer writes on a side stream before QKV work, then
join later. A QKV failure before that join could leave writes racing rollback.

The oracle therefore uses a narrow `decode_qwen4_verified_row` hook. Its default
delegates ordinary decode for non-QSA layer types. Qwen3 attention runs the same
common decoder with `transactional_qsa=true`, using the existing attention
guard to suppress QSA side-stream overlap during verification. Both common
attention branches forward that flag. Ordinary Plain decode, the KV-only
fallback and the prefill fallback continue to pass `false`, preserving their
original overlap behavior.
No new stream, transaction state, or alternative math path is introduced.

## Executed Checks

```sh
python3 -B -m unittest discover -s repro-kit/tests -p test_r16_qsa_oracle_journal_source.py -v
git diff --check
```

The 14 source/mutation checks pass. They pin object-safe hook wiring, exact
predicate reuse, prepare/snapshot/decode ordering, metadata offsets, preserved
recurrent snapshots, explicit error propagation, unchanged route selection,
verified-row transaction flags, unchanged Plain/KV-only/prefill flags and a
common-decoder callsite census. Mutation
probes reject removed hooks, unguarded hooks, wrong slots/rows/streams,
swallowed errors, post-decode snapshots, and dropped transaction flags.

These are source checks, not executed Rust or injected device failures.
The existing GPU mock records launches but does not execute QSA undo kernels,
so it cannot establish cache rollback parity. No compiler, native test, GPU,
model, serving, container or performance action ran for this repair.

## Remaining Qualification

- Compile and independently review the combined source with fixed-K3/K4
  delegation before any executable claim.
- Force each journaled oracle width, especially default K4, with every accepted
  prefix and each compression-group alignment. Compare committed raw-ring and
  compressed-key bytes against ordinary Plain decode.
- Check contexts below, across and above 2048, repeated fresh-process token
  parity, H/conv/PLE state, hidden-row selection, and sequence history.
- Inject preparation, snapshot, attention/QKV and commit failures. Confirm
  zero-accept cleanup does not race outstanding QSA writes and no later row runs.
- Qualify eager and admitted graph capture/replay separately, with overlap
  configured on and off. Runtime evidence, accuracy and speed remain unproven.

Widths outside the existing canonical journal predicate, including experimental
physical K16, are not newly covered by this repair.
