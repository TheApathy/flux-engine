# R16 Eager Grouped Loader And One-Slot Composition

Status: isolated source-only merge candidate, 2026-10-22. No compiler, native
test, model, GPU, container, resource claim, cache operation or runtime action
was performed by this lane. This is not a demonstrated startup or accuracy
fix and establishes no prefill/decode rate.

## Exact Inputs And Source Closure

- First parent: accepted grouped/one-slot composition
  `bae0b7d07ac7b3facfd7ef2e0a582dbdca8674ee`, tree
  `87b4ba70e55ff087ecaba2df2b63cda9c2ed3080`.
- Second parent: eager donor `caabee820853a6bfc43e82149d8452623da7822d`,
  tree `298dc6773ea6a778a841baea454f61051cc8093c`.
- Common ancestor: grouped guard repair
  `41975ff4c081fb1816fd30eb13c6b04d0a5c8a0d`.
- New isolated branch: `integration/r16-eager-grouped-one-slot-20261022`.

The merge had no textual conflicts. All four donor-modified production files
are byte-identical to the frozen eager donor. Both production copier and its
native test definitions are byte-identical to accepted `bae0b7d0`. These six
hashes are enforced in `tests/test_eager_grouped_one_slot_source.py`.
Only this composition's new scope/test files and the current grouped-loader
scope document are added/updated beyond those inputs. No GDN, profile, kernel
or unrelated donor is integrated. The older composition scope document is a
historical record of `bae0b7d0`, not the current eager materialization order.
Canonical and frozen donor worktrees were not edited.

The first worktree-add attempt failed inside the 90 MiB address-space cap
because Git attempted a large pack mapping, before checkout or merge. The
already-created new branch was reused with 1 MiB packed-Git windows and an
8 MiB mapping limit, preserving the same cap. No host setting or memory/cache
operation was used to recover.

## Admission And Allocation Order

`ATLAS_FAST_LOAD_ROUTED_GROUPS` and its new sub-selector
`ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER` are OFF when absent or exactly `0`, and
accept only exact ASCII `1` to enable. Malformed/non-Unicode values fail.
EAGER=1 requires GROUPS=1. The existing canonical target geometry and fast
loader guard remain mandatory: Plain, resolved TP1/EP1, both ranks zero and
all four speculative flags false. Loaded-serial speculation overrides do not
bypass this guard. Both loader constructors leave eager mode disabled.

Complete selected metadata is prevalidated before any grouped allocation:
147,456 canonical routed packed/scale components, exact target geometry,
unambiguous names and membership. Each backing remains confined to one
shard/layer/projection, aligned and bounded to 64 MiB including its guard.

When eager is enabled, the existing OOM-preflight refusal happens before
`allocate_all`. Every planned backing is then materialized in plan order,
before the shard payload loop. Header/inventory reads already occurred during
preflight, so this does not mean before every file read. First tensor views
reuse the existing bases, without allocating them twice. Allocation failure
stops the Result collection at that prefix and aborts loading, with no
automatic retry or managed-memory fallback for grouped storage.

The planned final bytes and preflight arithmetic remain unchanged. Earlier
residency can change actual pressure and transient peak. Neither physical
memory savings, reduced driver warnings nor successful readiness is proved.
Actual group counts still depend on shard placement, not the historical ideal
144-group layout. Original tensor/scalar images still determine the pinned
copier chunk and one-slot selection, not the size of an entire backing group.

## Ownership, Rollback And DMA Limits

The donor refactors both eager and opt-in lazy materialization to allocate,
register an unpublished owner, claim the exact backend base, then publish the
base into the plan. No tensor view or DMA is published before successful
claim. Registry failure attempts one ordinary free of the unclaimed base.
Claim failure first unregisters only the latest matching live registry entry,
removing its reserved names, then attempts one ordinary free and aborts.

This rollback relies on the current CUDA claim contract: every error return
precedes its only `grouped=true` transition, which is followed directly by
success. It does not assume arbitrary third-party backends have transactional
claim errors. A panic after registration retains its registry owner through
failed-load Drop, rather than submitting a rollback free on an uncertain
claim. Previously claimed prefix groups also retain their explicit ownership
until context/process teardown.

An ordinary rollback free that returns an error follows the inherited backend
restore semantics. It is not a grouped-quarantine transition and is not proof
against every possible future ordinary/bulk-free retry. There is no DMA or
public alias at this rollback boundary. The current load returns an error,
does not retry, and process/context teardown is the terminal boundary. The
candidate does not introduce a general in-process failed-load recovery API.

Registered/claimed group release otherwise retains the accepted backend
quarantine-before-driver-call, allocation mutex, Phase B/D tombstones, interior
and bulk-free rejection, and no destructor GPU frees. The unchanged copier
retains the exact owning base while submitting to checked view/chunk offsets.
One-slot reuse still waits before overwrite. Submission, record, poll, timeout
and Drop uncertainty retain source/event/base ownership. Eager allocation
does not change kernel ABI, inference profile, or these DMA state transitions.

## Bounded Offline Checks And Next Gates

```sh
ulimit -v 92160
ulimit -t 5
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -q \
  repro-kit/tests/test_routed_backing_groups_source.py \
  repro-kit/tests/test_grouped_bounce_ranges_source.py \
  repro-kit/tests/test_pinned_bounce_lifetime_source.py \
  repro-kit/tests/test_eager_grouped_one_slot_source.py
```

Result: 53/53 source/reference checks passed in 0.130 seconds. Whitespace
validation passed. The seven new checks cover exact donor/copier bytes,
negative eager/scope selectors, preflight-before-eager-before-payload order,
current CUDA claim error ordering, rollback/panic owner retention, tiny
independent prefix-failure ownership references and one-slot owned-range DMA
wiring. Python references do not execute Rust or a driver. Native definitions
remain uncompiled and unexecuted in this lane.

Next gates are independent exact frozen-source review, an authorized
exact-source build/native-test receipt and a separately claimed no-cache host
diagnostic under the unchanged warning STOP policy. Readiness, strict protocol,
natural-stop numerical quality, Plain/native-K5 parity, repeatability, 262K
retrieval, concurrency, GSM8K/MMLU-Pro, matched rates and immutable-image
qualification remain separate. No issue closure or GHCR promotion follows
from this source candidate.
