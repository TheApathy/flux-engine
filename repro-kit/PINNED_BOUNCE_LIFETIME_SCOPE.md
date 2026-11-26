# Pinned Bounce-Buffer Lifetime Repair

Exact base: `a67954024f57f15ae382cab74a2c0c8773663da8`.

This is an isolated source-only candidate. No canonical integration, compiler,
native execution, model, package, GPU or server action is authorized by this
receipt. The shared server was live while this source work was performed.

## Repair

The inherited copier took an event handle out of a slot before a bounded wait
succeeded. A timeout/error could then retry and overwrite the same pinned bytes
while DMA still read them. Ordinary error unwind also dropped the pinned owner
and could call `cuMemFreeHost` under a pending copy. Copy submission followed by
event creation/record failure had no reliable completion ownership at all.

Each slot now has an explicit Idle, Untracked or Recorded state. It keeps its
handle and exact owning device base until positive completion. The event is
created before submission. An exact-base/range device retention reference and
the Untracked slot state are installed before entering the async driver call.
Submission errors/panics and record errors therefore retain both DMA owners.
An unrecorded event is never polled: a fresh CUDA event may report ready without
proving anything about the preceding copy.

The copier is poisoned before entering copy/drain backend operations. A failed
operation or a caught panic forbids reuse and retry. A later successful drain
may release a positively completed transfer, but does not unpoison the copier.
There is no automatic retry and no blocking stream synchronization or explicit
process exit/abort introduced here. CUDA graph capture is refused at creation.

The copier borrows its backend through Drop. Explicit drain and Drop attempt
every slot, including after an earlier slot error or panic, with one shared
30-second polling deadline per drain. A still-uncertain slot deliberately forgets
its pinned owner instead of freeing its memory. Completed slots release their
source normally, even if event destruction fails after positive completion.
Failed release/destruction can conservatively retain a completed destination or
event handle; failure still propagates and the copier remains poisoned.

CUDA destination ownership uses checked pending-H2D references in the existing
allocation table. Exact owning bases, checked subranges and address arithmetic
are required. Retention and free selection hold the same table mutex. Individual
free removes an eligible allocation before entering the driver, so a concurrent
retain cannot succeed during a free. Bulk free atomically removes only eligible
allocations, leaves retained allocations tracked and returns an error when it
skips any. Unrelated allocations may still be freed. Failed driver frees restore
their allocation records. Reference overflow/underflow refuses mutation.

The GpuBackend hooks default to no-ops only for its synchronous pinned-copy
fallback. An implementation that overrides that fallback to enqueue actual
async work must also implement the retention hooks. Production CUDA does so.

## Limits

- Polling is bounded, not arbitrary driver entry latency. A driver API call or
  allocator can itself stall; this does not claim an interruptible CUDA driver.
- Quarantine is intentionally retained until context/process teardown when no
  completion proof remains. There is no background retry or reclamation daemon.
- At most three 64 MiB pinned slots and at most three active destination
  references exist per copier. Exact destination allocations can be larger than
  a chunk; their full owning allocations are retained. Repeated failed loads
  can accumulate retained memory and must not be presented as healthy retries.
- Device pointers remain raw views, not universally owning Rust handles. This
  repair covers allocations made and freed through this CUDA backend table,
  not arbitrary foreign driver frees or external context destruction.
- The loader's inherited detached-reader/raw-fd error-unwind issue is not fixed
  in this seal. A separately reviewed reader-owned-descriptor follow-up remains
  necessary. No reader-thread termination or bounded kernel-I/O claim follows.
- No loading speed, model parity, accuracy, quality, container or release claim
  follows. This is not runtime or native qualification.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s repro-kit/tests -p test_pinned_bounce_lifetime_source.py -v
```

Result: 12/12 source wiring checks PASS. These inspect source, not native
behavior. Eleven small native mock/state-machine regression definitions cover
success, timeout, create/retain/submit/record/poll/release/destroy failures,
caught submission panic, all-slot draining after error/panic, same-destination
references, exact-base/range refusal and reference overflow/underflow. Their
synthetic pinned buffers are four bytes each. Native definitions remain unrun.

Scoped `rustfmt --check` and `git diff --check` PASS. The source-check command
measured a maximum RSS of 15,352 KiB with `/usr/bin/time -v`. This measurement
is for Python source inspection, not a compiler or native test run.
Independent source review and an authorized compiler/native-test window remain
required before any integration or runtime validation.
