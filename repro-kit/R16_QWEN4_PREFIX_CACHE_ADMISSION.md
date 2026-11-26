# Qwen4 Prefix Snapshot Admission

## Source-Confirmed Gap

At C1, Qwen4 previously admitted prefix caching with positive Marconi SSM
snapshot slots. A partial-prefix hit can restore GDN state and skip prefill
without restoring the separately allocated PLE convolution history.

The production policy now rejects only this combination before the C1 early
return. This is a fail-closed boundary, not an implementation of complete
snapshot restoration or a GPU accuracy claim.

| C1 prefix caching | SSM snapshot slots | Admission |
| --- | --- | --- |
| false | 0 | Unchanged |
| true | 0 | Unchanged, paired-cache lookup misses and prefills from zero |
| false | positive | Unchanged, no active prefix restore |
| true | positive | Rejected until PLE state round-trips |

The existing C2+ lifecycle restrictions remain unchanged. The C128 limit is
labeled an admission ceiling, not measured runtime qualification.

## Reachability

All paths here are relative to the source tree. This chain was reviewed
against integration base `103a0fdfdae80438d8845dbbd69c8300d981617c`.

1. `main_modules/serve_load.rs` invokes the production Qwen4 policy before
   GPU initialization. The previous C1 early return bypassed cache checks.
2. With prefix caching, positive snapshots, block size 16 and checkpoint
   interval 256, `main_modules/serve_phases/kv_cache.rs` can align the default
   prefill budget to 4096 tokens.
3. `scheduler/prefill_a_step.rs` calls `Model::prefill_chunk`. Ordinary
   Qwen4 prefill with the fast-family selectors off reaches the prefix lookup
   in `model/trait_impl/prefill_b.rs`.
4. `prefill_b/save_checkpoint.rs` saves an intermediate checkpoint at a chunk
   boundary. `model/ssm_snapshot.rs` copies GDN h/conv state only.
5. On sequence free and allocation, PLE history is cleared separately.
   A later same-session partial extension can restore the GDN checkpoint in
   `prefill_b/prefix_lookup.rs` and set a positive `marconi_skip_to`.
6. `prefill_b/forward_layers.rs` calls PLE only for the suffix, with reset
   false. Its token history supports n-gram lookup but cannot reconstruct
   residual-dependent convolution state for skipped rows.

The model source paths above are under `crates/spark-model/src`. Server
source paths are under `crates/spark-server/src`. Runtime numerical impact
has not been reproduced on GPU by this review.

## Upstream Provenance

- Atlas-Inf PR163 was inspected at exact head
  `80d14290fc3c91dafcbfda99b36a74629594bfdb`.
- Its optional exact-prefix mode is not a drop-in accuracy fix. It compares
  cache-on/off with exact mode enabled in both arms, changes cold pass
  layout, and has no C2+, vision, or multi-rank qualification.
- Complete auxiliary-state infrastructure predates that PR in upstream
  commit `e77ae6f053205a924b3ffec5653af399a0b5a84f`. It attaches PLE/QSA
  state to Marconi snapshots and declines non-restorable hits.
- The local standalone PLE implementation differs. Broadly copying the
  upstream format would not prove local state restoration.

Two reusable invariants are complete-state restoration at the same prefix
boundary and equal cold/warm pass layout for any optional exact-replay mode.
Neither invariant relaxes lossless MTP-versus-plain requirements.

## Related Lifecycles

Decode-boundary ring snapshots also contain only GDN state. The current sole
watchdog caller stops after rewinding, so this review does not claim resumed
decode corruption through that route.

Legacy sequence swap in `model/trait_impl/sequence.rs` stores KV and GDN but
not PLE. C2+ Qwen4 admission already rejects swap. At C1, request drain
capacity is zero while one sequence is active or prefilling. The sole
swap-out loop requires both a newly admitted request and an active victim,
so no production Qwen4 legacy swap roundtrip was shown reachable. High-speed
per-block storage is a separate lifecycle and is not qualified by this review.

## Verification Boundary

The patch includes direct Rust tests of `validate_qwen4_concurrency` and
targeted source-only Python checks, including mutations of the guard's
condition and ordering. No Rust compiler, GPU, engine, server, container,
or shared integration source was used by this lane.

Run the offline checks with:

```sh
python3 -m unittest discover -s repro-kit/tests -p test_r16_qwen4_prefix_snapshot_admission.py -v
rustfmt --edition 2024 --check crates/spark-server/src/main_modules/qwen4_concurrency.rs
```

The authorized build owner must still execute the production-policy Rust
tests. Runtime qualification remains separate from source admission.
