# Container lifecycle-v3 integration contract

`container_lifecycle_v3.py` is the fail-closed lifecycle primitive for a
release qualification runner. It is source-only and does not invoke Docker,
the GPU, a server, or the network by itself. The concrete R12 integration is
`r12_container_release_runner.py`; it replaces the private cleanup state
machine carried by the prior R8 release GPU runner.

## Required launch registration

Before starting `docker run -d`, create one `LaunchIdentity` with:

- the exact container name;
- the immutable `sha256:` image ID, never a mutable tag;
- the run ID;
- a random per-launch ownership token;
- the cleanup-contract version; and
- a run-local cidfile path.

Put the run ID, ownership token, and cleanup version into the three label keys
exported by the module. Pass the cidfile path to `docker run --cidfile`. Keep
the exact detached `Popen` object until its stdout/stderr have been reaped.

The adapter's label sweep must query all three exact labels and return every
matching container. Its `inspect` projection must include the full container
ID, exact name, immutable image ID, complete labels, and an exact boolean
`State.Running`. A missing/non-boolean running state is unresolved, not
stopped. The only accepted inspect names are exactly `name` and `/name`;
additional leading slashes, prefixes, or suffixes are identity conflicts.

The R12 runner owns `--detach`, `--name`, `--cidfile`, the three ownership
labels, and the immutable image argument. It refuses caller overrides,
pre-existing cidfiles, and mutable image tags. It retains the exact detached
client object until it is reaped. Its Docker adapter first obtains a clean
container index, then inspects an exact full ID, so a daemon or inspect error
can never be confused with absence. A failed graceful stop is followed by an
exact-ID kill; failure of both operations remains quarantined.

## Required wrapper

Call `run_with_cleanup_v3(body, adapter, identity, release_lock)` while the
process owns the qualification flock. Do not put lock release in a separate
outer `finally` block. The wrapper catches every body `BaseException`, moves
into signal-deferred reconciliation, and calls `release_lock` only after:

1. the tracked detached launch client has settled;
2. the cidfile, exact name, remembered IDs, and exact-label sweep are all
   error-free;
3. any exact-owned running container has been stopped;
4. diagnostics are attempted only after a later inspect confirms stopped; and
5. repeated complete sweeps prove either stable absence or stable stopped
   state.

Unreadable or malformed cidfiles, daemon/inspect failures, foreign identity
collisions, multiple owned matches, and cleanup implementation exceptions all
retain the process and flock in an unbounded reconciliation loop. Restore the
daemon or identity evidence and let that same process finish. Do not kill the
wrapper or remove the lock file.

Signal handling changes state before raising the first body signal. This makes
the body-return-to-cleanup transition fail-closed. Signals received during
cleanup or lock release are deduplicated and deferred until after a safe
unlock. Error and event evidence is deduplicated, text- and count-bounded, and
the lifecycle enforces a minimum retry backoff even if an adapter's wait hook
returns immediately or raises.

`reconcile_before_unlock(max_cycles=...)` is only for bounded offline tests.
The production wrapper intentionally exposes no bounded-cleanup option.

## Memory-floor evidence

Start `MemoryFloorMonitor` immediately before an arm and call `finish` in the
arm's `finally` path. Persist the returned object as the `MEMFLOOR` assertion.
The status meanings are:

- `PASS`: complete sampler coverage and every sample meets the floor;
- `FAIL`: complete sampler coverage but the measured minimum is below the
  floor;
- `ERROR`: the arm aborted, the sampler raised, died early, failed to join,
  produced no samples, or did not cover the start/end window.

Never synthesize a memory pass from a partial sample list.

## Offline verification

These commands do not invoke Docker, a GPU, a server, a build, or the network:

```bash
python3 -B -m unittest -v repro-kit/tests/test_container_lifecycle_v3.py
python3 -B -m unittest -v repro-kit/tests/test_r12_container_release_runner.py
python3 -B -m py_compile \
  repro-kit/container_lifecycle_v3.py \
  repro-kit/r12_container_release_runner.py \
  repro-kit/tests/test_container_lifecycle_v3.py \
  repro-kit/tests/test_r12_container_release_runner.py
```

The tests cover delayed container creation after an initial absent inspect,
stop-before-diagnostics ordering, cidfile errors, identity conflicts,
`BaseException` quarantine, unresolved-wrapper lock retention, tracked launch
settlement, fatal sampler errors, clean premature sampler death, and real
low-memory failure. They also inject signals at the body, transition, cleanup,
and unlock phases; bound an unresolved quarantine; reject malformed exact
names; exercise stop/kill and diagnostic failures; distinguish daemon and
inspect errors from absence; reject foreign collisions; and run the complete
R12 path through an argv-level fake Docker backend.

## Evidence boundary

Offline tests prove the lifecycle state machine and concrete adapter command
contract only. They are not Docker, GPU, server, memory-floor, correctness,
performance, or release evidence. The adapter still needs an independent code
review followed by a fresh exact-image fault-injection run under the real
qualification flock.
