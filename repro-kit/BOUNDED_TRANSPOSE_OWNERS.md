# Bounded Transposed Owners: Source Candidate

Status: SOURCE_AND_OFFLINE_ONLY. Rust compilation, native CPU tests, real
weight transforms, loading, READY, generation, quality, throughput and container
qualification are NOT_RUN. This candidate is not integrated or promoted.

Parent: `e82bf90c5e745ed105476fa8c876f29e7a15ad9a`.
Parent tree: `022e9cfe871f9a2ca691a3164daeadc062015479`.

## Selector And Scope

`ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS` is default-off. Absent or exactly `0`
retains the existing destination allocation path. Exactly `1` opts in. Empty,
whitespace-padded, textual boolean, non-Unicode and other values fail closed.

Opt-in requires the existing canonical Qwen4 routed geometry: 48 layers, 512
experts, hidden width 2560, intermediate width 640, EP world size 1 and EP rank
0. It requires unified layout, and refuses a hybrid request even if the
original hybrid admission would have fallen back to unified. Validation occurs
in the factory before transpose allocation/fallback and again at the unified
layer entry point. There is no allocation-failure fallback to legacy slabs,
individual tensors or managed memory.

The exact one-layer native-MTP submodel produced by `qwen4_mtp_config(target)`
is outside this target-only flag. The shared layer helper preserves that
submodel's existing individual unified-transpose path; unsupported target
geometry still fails at the factory. This is an explicit scope exclusion,
not an allocation-error fallback, and is covered by a dedicated regression.

This changes only routed transposed destination owner geometry. It does not
change BF16 KV, KV budget/addressability, context, chunk size, quantization,
MTP/Plain dispatch, scalar values, arithmetic kernels, startup floor values,
strict kernel stops or source grouped ownership. It does not introduce VMM or
UVM. Destination bases remain ordinary backend-tracked model-lifetime
allocations, not claimed grouped-source owners.

## Exact Requested Geometry

Each gate, up and down projection has identical canonical byte counts despite
their swapped matrix dimensions. Packed data is 819200 bytes per expert and
scale data is 102400 bytes. Both are 256-byte aligned and their pair stride is
921600 bytes. An owner has an unexposed 256-byte front guard.

| Quantity | Value |
| --- | ---: |
| Owner cap | 67108864 bytes |
| Expert pairs per full owner | 72 |
| Full owner request | 66355456 bytes |
| Full owners per projection | 7 |
| Tail expert pairs | 8 |
| Tail owner request | 7373056 bytes |
| Owners per projection / layer / model | 8 / 24 / 1152 |
| Packed/scales component views per model | 147456 |
| Routed payload per layer | 1415577600 bytes |
| Routed payload per model | 67947724800 bytes |
| Existing two-slab requested bytes per layer | 1415578112 bytes |
| Bounded requested bytes per layer | 1415583744 bytes |
| Existing two-slab requested bytes per model | 67947749376 bytes |
| Bounded requested bytes per model | 67948019712 bytes |
| Additional requested bytes per layer / model | 5632 / 270336 |

73 pairs plus the front guard would request 67277056 bytes, exceeding the cap.
Exact expert membership is 0 through 511, in table order. Boundaries 71/72,
503/504 and final expert 511 are covered explicitly. Within each owner, packed
and scale ranges are disjoint, aligned and in bounds; no table exposes an
owning base. Addresses naturally differ from the previous slabs. Table
equivalence means expert order, pointed-to bytes and associated scalar/input
pointer values, not numerical equality of old/new device addresses.

Unified admission preserves its existing complete-layer-plus-host-transient
bound and floor, with no credit for Phase-B frees. Only opt-in adds 5632 bytes
to account for 24 front guards instead of two. Requested byte counts do not
prove physical backing granularity, contiguity, fragmentation or reduced
driver overhead. There are 12 times as many destination owners as the existing
96-slab model, though 128 times fewer than individual packed/scale allocations.

## Lifetime And Failure Behavior

Every successful destination allocation is retained in the layer immediately,
before range validation or DMA. Allocation or copy failure returns an error
without releasing, overwriting or retrying prior destination owners. No new
destructor calls the driver. Existing context teardown owns their storage.
Any repeated unified/hybrid call over a partial bounded-owner set is refused.

Gate/up destination owners are constructed before Phase B. The existing
Phase-B source retirement block is unchanged: synchronize the default stream,
release Gate owners, null Gate aliases, release Up owners, then null Up aliases.
Down destination allocation remains after the complete Phase-B block. Phase D
likewise synchronizes, releases Down owners and only then nulls Down aliases.
Sources are never retired per new destination chunk, because their original
group boundaries differ.

The existing source-group registry and backend quarantine errors/panics before
uncertain frees, tombstone aliases, and refuse retries. These implementations
and their existing tests are unchanged. Destination owners must not be claimed
through the grouped-source API: that would change bulk teardown semantics.

## Downstream Layout Audit

Owning slab fields are construction/lifetime state, not dispatch addresses.
Dispatch continues to use `ExpertPtrTable` for every projection. Reviewed
consumers index the table with the selected expert rather than deriving expert
addresses from one globally contiguous base:

- Plain decode: `forward_phase.rs` and
  `kernels/gb10/common/moe_shared_expert_fused_t.cu`.
- Batched/K5: `forward_batched.rs`, `forward_k5.rs` and the per-expert pointer
  loads in the exact-K16 implementation.
- Prefill: `forward_prefill_routed.rs`, the Flash-Next grouped-GEMM wrapper and
  its included `qwen3-next-80b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu`.
- Worklist: `moe_w4a16_worklist.cuh`.
- Compact transposed variants: `moe_w4a16_orig_i640_compact_prefill_gemm.cuh`
  indexes `packed_ptrs[item.expert]` and `scale_ptrs[item.expert]` after the
  per-variant transposed-layout definitions.

No global contiguous-owner requirement was found in this source audit. This
does not replace native execution, all-profile review or CUDA sanitizer work.
Per-expert pointer-table consumers and boundary modules remain decoupled from
bounded-transpose internals (verified by live structural source contracts rather
than historical E82 byte-identity snapshots).

## Verification Boundaries

The offline Python suite is run with:

```sh
ulimit -v 92160
ulimit -t 5
set -e
PYTHONDONTWRITEBYTECODE=1 python3 repro-kit/test_bounded_transpose_owners.py
```

It is a source-contract and independent model test, not execution of Rust or
CUDA. It covers arithmetic, membership, pointer-table order, tiny byte fixtures,
scalar bits, overflow, selector refusal, simulated partial-allocation lifetime,
source-free quarantine/alias retirement, and owner-cap geometry/tracing contracts.
Structural source-contract assertions verify Phase B/D synchronization order,
admission guard wiring, pointer-table builder identity, and decoupling from
boundary/consumer files against current HEAD. Source inspection, modeled tests
and native results must remain separate claims.

Observed offline result: 90 tests PASS under the 90 MiB address-space and
five-CPU-second limits. All 512 experts are covered across distinct tiny Gate,
Up and Down byte/table/scalar fixtures, not real weights.

Eight new Rust CPU unit-test definitions cover selector/geometry rejection,
non-Unicode values, exact canonical geometry, overflow/cap rejection, partial
allocation retention, all-512-expert tiny payload/table/scalar equivalence and
the 5632-byte admission delta, plus the exact native-MTP scope exclusion.
They are NOT_RUN. Existing runtime tests for
failed source frees, pending H2D, backend panics, tombstones and alias retirement
are also NOT_RUN in this task.

Before a real run, independently review the frozen commit/diff, compile and
execute the CPU Rust definitions under separately granted native authority,
then use an exact-source build and a separately reviewed strict smoke
controller. Preserve BF16, context 135168, W4096, declared memory floors and
all strict stops. A successful READY would establish startup only. It would
not establish Plain/K5 accuracy, longer-context retrieval, speed, concurrency
or container readiness.

The observed late-startup kernel warnings do not establish which allocation
failed: kernel source and journal receipt clock domains are not calibrated.
This is a bounded allocation-topology hypothesis, not a claimed root cause or
prediction of READY.
