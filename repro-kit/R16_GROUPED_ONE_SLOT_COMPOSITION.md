# R16 Grouped Loader And One-Slot Composition

Status: isolated source-only merge candidate, 2026-10-22. No compiler, native
test, model, GPU, container, resource claim or runtime action was performed by
this lane. This is not a release-qualified binary or a demonstrated startup
fix.

## Frozen Inputs

- First parent: canonical `bf3782d23a06a64fa6280a9e7c96eaa86f0900b4`, tree
  `ccd7dd26c82167c9ad32c5584978318546c6ef30`.
- Second parent: grouped candidate `41975ff4c081fb1816fd30eb13c6b04d0a5c8a0d`,
  tree `d0f2c28ec42179c09f9989a4198435e524cdb344`.
- Common ancestor: `4a6d1626d02637601029030d340789c6d29276a0`.

The canonical worktree and frozen donor were not edited. The root-owned dirty
canonical handoff was not included. Composition uses the new isolated branch
`integration/r16-grouped-one-slot-20261022`.

## Preserved Behavior

The canonical constructor sizes each pinned chunk from the largest selected
tensor or scalar-slab image, capped at 64 MiB. A nonempty single-chunk image
uses one slot. Empty or multi-chunk geometry retains three slots. Copy/event,
timeout, drain and Drop retention behavior remains unchanged. For the recorded
shard-17 geometry, this changes requested pinned bytes from 3,539,712 to
1,179,904. These are source-derived requested bytes, not physical memory or a
successful load receipt.

The grouped diagnostic remains OFF by default and Plain-only with resolved
TP1/EP1, both ranks zero and all four speculative flags disabled. Its complete
147,456-member preflight, per-shard/layer/projection bounded groups, backend
ownership claim, quarantined one-attempt group free and Phase B/D tombstones
are preserved. Base-plus-checked-offset DMA retains the exact owning base.
One-slot reuse still waits for positive completion before buffer overwrite.
No kernel ABI or default profile changes are included.

Actual checkpoint group cardinality still requires the selected shard plan.
The earlier metadata accounting of 2,304 groups is not a runtime result and
must not be replaced by the historical ideal 144-group layout. Guards add
requested bytes, so fewer allocations do not establish physical GiB savings.

The sole textual merge conflict was in
`tests/test_pinned_bounce_lifetime_source.py`. Resolution preserves the
canonical constructor checks with whitespace compaction and the one-slot
regression, alongside the donor's exact-base/range retention assertions.
Both production copier files merged without textual conflicts. Their diffs
against each parent were inspected to confirm both changes remain.

## Bounded Offline Validation

```sh
ulimit -v 92160
ulimit -t 5
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -q \
  repro-kit/tests/test_routed_backing_groups_source.py \
  repro-kit/tests/test_grouped_bounce_ranges_source.py \
  repro-kit/tests/test_pinned_bounce_lifetime_source.py
```

Result: 41/41 source/reference checks passed in 0.150 seconds. These checks
inspect Rust wiring and an independent Python layout reference. Rust native
definitions remain uncompiled and unexecuted. Whitespace validation passed.

Independent frozen-source review, authorized compilation/native tests, then a
separately claimed exact-source fresh load under the unchanged warning STOP
policy are still required. Readiness, protocol/semantic smoke, Plain/native-K5
parity, repeatability, 262K retrieval, concurrency, quality suites, performance
and immutable-image qualification remain separate gates. No historical speed
is attributed to this composition.
