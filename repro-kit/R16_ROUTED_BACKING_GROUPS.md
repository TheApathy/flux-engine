# R16 Routed Original Backing Groups

Status: isolated source candidate, originally based on `4a6d1626`, now composed
with the canonical `bf3782d2` one-slot repair and opt-in eager donor `caabee82`.
See `R16_EAGER_GROUPED_ONE_SLOT_COMPOSITION.md` for the current exact input
commits and scope. `R16_GROUPED_ONE_SLOT_COMPOSITION.md` records the earlier
non-eager composition.
Not compiled, loaded, GPU-qualified, image-qualified or a demonstrated
driver-warning fix.

## Admission And Layout

`ATLAS_FAST_LOAD_ROUTED_GROUPS` is OFF when absent or exactly `0`. Exactly `1`
opts in. Other values, including non-Unicode bytes, fail closed. Serving
requires the fast loader, Qwen4 target architecture, 48 layers, 512 experts,
hidden size 2560, intermediate size 640 and resolved TP1/EP1 with both ranks
zero. Plain and native MTP (`--speculative`, with in-checkpoint MTP or
`--mtp-from-path`) are admitted; `--dflash`, `--self-speculative` and
`--ngram-speculative` are rejected, and `ATLAS_DISABLE_SPECULATION` does not
widen that. Native MTP is safe because groups hold only
`model.language_model.layers.*` routed originals: no `mtp.*` tensor is ever
planned (the plan also refuses one), the MTP layer loads under weight prefix
`mtp` so it never attaches a group and frees its own originals individually,
the `--mtp-from-path` sidecar loads ungrouped and cannot merge groups, and every
MTP read (`mtp.*` names only) follows the target groups' release in
`build_model`. A target routed read after release fails closed (tombstone).
The backend must declare explicit grouped ownership support before any grouped
allocation.

Before any weight allocation, the grouped loader parses the complete selected
shard inventory and requires all 147,456 unique canonical routed packed/scale
members. Packed tensors are U8 `[N,K/2]` and scales are F8_E4M3 `[N,K/16]`.
Names, indices, shapes, dtypes, byte extents and selected shard membership are
checked. Supplemental weights cannot replace grouped target aliases. Runtime
shard metadata is checked against the preflight plan again before copying.
This is not a file-identity/TOCTOU protection scheme and does not attest payload
contents. Existing loader file-validation limitations are not closed here.

Each backing is confined to one shard, layer and projection, with a maximum
requested size of 64 MiB including an unexposed 256-byte guard. Views are
256-byte aligned, disjoint and bounded. The default lazy path avoids eager
whole-projection allocations for tensors in future shards, advancing up to
one bounded group's later-in-shard residency at first use.

`ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER` (unset/`0`/`1` only; `1` requires
`ATLAS_FAST_LOAD_ROUTED_GROUPS=1`) instead allocates, registers and claims
every planned base after the loader pre-flight and before the first shard
payload read (header/inventory reads already occurred during preflight),
logging one `FAST_WEIGHT_ROUTED_GROUPS_EAGER` line with the base count and
bytes. The admission arithmetic and planned final group bytes are unchanged.
Earlier residency can change actual pressure and transient peak, which require
runtime evidence. Unset, the path is the lazy one above.
Both paths register before claiming; a refused registration frees the base
and a refused claim unregisters and attempts one ordinary free before aborting
the load. Current CUDA claim errors occur before its grouped-state transition.
A failed ordinary rollback free follows the existing backend restore behavior,
not grouped-free quarantine. It is not proof of in-process recovery or universal
uncertain-free protection. No DMA or public view exists during this rollback.

Actual cardinality depends on the selected metadata's shard placement and
file order. `FAST_WEIGHT_ROUTED_GROUP_PLAN` emits the exact planned backing
count, component count, payload, requested bytes and guard/alignment bytes.
It does not report an allocator quantum or a physical-memory saving.

An independent streaming single-shard canonical fixture yields 1,152 bases,
67,947,724,800 payload bytes and 294,912 guard bytes, replacing 147,456
individual packed/scale bases in that synthetic layout. This is not a measured
checkpoint result. Pathological shard placement can retain 147,456 groups.
The historical ideal 144-group layout is NOT implemented. Requested payload
is unchanged and guards add bytes. No physical GiB saving, warning causality,
startup success or inference-rate improvement is established.

## Ownership And DMA

The registry tracks a not-yet-published group before its explicit backend claim.
The claim succeeds before plan-base or tensor-view publication and before DMA.
A panic after registry registration retains that record through failed-load
Drop. Unsupported backends reject; grouped allocations have no UVM fallback.
This opt-in path can therefore refuse an allocation where the default
individual/UVM path might continue. Shared experts, scalar metadata, MTP,
attention and all other tensors retain their established allocation paths.
Existing scalar slabs and transposed expert slabs are unchanged.

`copy_into_owned_range` retains the exact backing base with the checked view
plus chunk offset, while submitting to its interior destination. Flight state
and base retention precede submission. Submission/record errors, panics,
timeouts and uncertain event progress retain the pinned source and exact base
lease. Only positive completion of a recorded event releases that lease.
Legacy zero-offset `copy` delegates to the same API. The composed candidate
preserves the canonical one-slot selector for a nonempty tensor/scalar image
fitting one 64 MiB chunk. Empty and multi-chunk geometries retain three slots.
Each copy still restarts at slot zero and waits before reusing its buffer.

WeightStore stores explicit group owners and rejects individual consumption of
group views. MoeLayer validates every projection pair before attaching shared
owner handles. Unified transpose Phase B synchronizes, frees gate then up
groups and nulls their aliases. Phase D separately synchronizes, frees down
groups and nulls their aliases. Shared expert frees remain individual.
Hybrid mode keeps grouped originals live, as it already keeps originals live.

Group release tombstones every member before the backend call. CUDA keeps its
allocation-table mutex across the single grouped driver free, preventing a
successful-free/address-reuse race. Backend quarantine is installed before
the driver call and even before refusing a still-pending H2D lease. Failed or
uncertain grouped frees cannot be retried by grouped free, ordinary free or
bulk teardown. Future H2D retention on a quarantined base fails. Existing
recorded DMA may still complete and release its lease without unquarantining
the allocation. Ordinary free rejects both live grouped bases and interior
views. Bulk free skips all explicitly grouped owners and reports retention,
including healthy groups, rather than invalidating live owner handles.

No destructor calls GPU free. Failed-load and live/quarantined registry owner
records are deliberately retained until process/context teardown. There is
no in-process recovery/retry API for quarantined groups. The current model-swap
path uses exec/restart rather than `free_all_allocations`; the bulk-API guards
are lifecycle safety, not proof that today's swap invokes bulk teardown.

No CUDA kernel ABI or default profile/kernel selection changes are included.
Individual allocations retain their existing free/failed-free restore behavior.

## Validation And Next Gates

Bounded source/reference check command:

```sh
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v \
  /usr/bin/timeout --signal=KILL 50s \
  /usr/bin/prlimit --as=94371840 --cpu=5 -- \
  /usr/bin/python3 -m unittest -q \
  repro-kit/tests/test_routed_backing_groups_source.py \
  repro-kit/tests/test_grouped_bounce_ranges_source.py \
  repro-kit/tests/test_pinned_bounce_lifetime_source.py
```

The first composed run passed 36/36, with 0.17 seconds wall time and 50,820 KiB
peak RSS. These checks inspect source wiring and an independent streaming
layout reference. They do not compile or execute Rust, CUDA or a model.
The follow-up admission repair adds resolved-TP and all speculative-flag
negative source/reference regressions, including loaded-serial override
refusal. It keeps the existing guard inline and does not add native test
definitions for these predicates.
The composed follow-up run passed 40/40, with 0.15 seconds wall time and
51,276 KiB peak RSS under the same bounds. Whitespace checks passed.

Rust definitions cover strict base-only frees, guards and membership, complete
model aliases, phase scoping, tombstones, DMA uncertainty, backend driver
failure/panic, pending leases, ordinary/bulk retry refusal and address reuse.
The strict GpuBackend mock uses the production allocation-table transitions
and records every driver-free attempt. These native definitions remain
uncompiled and unexecuted, including the full canonical inventory fixtures.

Required next gates are independent frozen-source review, authorized Rust
compilation and native mock execution, then a separately claimed fresh host
load with unchanged warning policy and exact source/ELF provenance. Group
counts, allocator observations and startup behavior require real receipts.
Serving, accuracy, native K5, long-context, concurrency, performance and image
qualification remain separate gates. No runtime authority is transferred by
this source candidate or its tests.
