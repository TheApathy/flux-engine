# Mixed Forward Scratch Isolation

## Provenance

Adapted from Atlas-Inf/atlas PR162 and its original fix commit
`20d9b1fb4350de3a474d12b7c197492e1002accb` by AzeezIsh. The merge commit is
`6aae708078e3d662232f28f6aa9d83e50f309761`.

The upstream fix reserves routing storage for the larger of the prefill row
count and padded decode row count. This adaptation additionally checks the
routing, alignment, metadata, scratch-arena, and pinned-staging extents before
the fused path begins GPU work. It does not change scheduling or arithmetic.

## Local Reachability

At the integration base `1603e33bf9b16bcbc45f2687e229862c0cfddfa5`, a
non-mHC MoE attention layer can process two decode rows before a one-token
prefill tail. With top-k eight, decode routing writes scratch bytes `[0, 128)`.
The previous placement put prefill positions at byte 64 and slots at byte 72.
They remained live across the decode call and could be overwritten before
prefill consumed them. The fixed placement starts metadata at byte 128.

The upstream C31/tail21 GPU crash is upstream evidence, not a local
reproduction. Wider local generic MoE decode uses per-row routing in paths
that differ from upstream. The conservative reservation also covers those
widths without assuming that their dispatch is identical.

Qwen4/mHC is excluded from this fused route by the normal standard and
batched-prefill scheduler gates. The concrete multi-prefill mixed method
retains its sequential decode-then-prefill behavior. This change does not
enable Qwen4 concurrency or qualify any additional concurrency width.

## Verification

The Cargo regression `mixed_forward_scratch_geometry` imports the actual
production layout helper. It covers the historical byte overwrite and the
fixed sentinel preservation, exact-fit and one-byte-short arena limits,
alignment, paged/nonpaged upload sizes, and checked arithmetic overflows.

Run after acquiring the coordinated compiler resources:

```sh
cargo test -p spark-model --test mixed_forward_scratch_geometry
```

Compilation and test execution are intentionally deferred while the root
integration build owns the compiler resources. Formatting and diff checks
are the only local checks performed during preparation of this port.

Remaining qualification is a fresh-process GPU mixed-forward tail workload
on a non-mHC MoE model with two live decode rows and a one-token prefill tail,
with metadata/cache integrity and CUDA errors checked. CPU sentinel geometry
alone is not a GPU correctness, serving, or performance receipt.
