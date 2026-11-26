# Fast-Loader Direct-I/O Diagnostic Selector

Base: `58d04dd678162a7b658f13f5e73beb5a4f1a4d08`.

This source-only candidate adds an opt-in selector for a controlled load
experiment. It is not a validated loading, accuracy, performance, container
or release fix. It grants no build, native, GPU, model or server authority.

## Selection

When unset, `ATLAS_FAST_LOAD_DIRECT_IO_TENSOR_CAP` leaves the loader's
configured cap unchanged. The default remains 5000 selected tensors per
shard. An explicit `try_direct_io=false` still disables direct I/O.

For a separately authorized experiment, setting the variable to `20000`
allows an O_DIRECT open attempt on 13,824-tensor expert shards that the
default heuristic reads through the buffered path. The count is evaluated
after the existing index and skip filters. The same effective cap applies
to normal shards and `extra_weights.safetensors`.

The value is read once at load entry, before shard resolution or GPU
operations within this loader. The serving application may already have
initialized its GPU backend. Only a nonempty ASCII decimal positive `usize` is accepted.
Zero, signs, whitespace, separators, overflow, non-ASCII digits and
non-Unicode values fail closed. Error messages do not echo invalid bytes.
Leading zeros are accepted when the value is positive.

The override log is only a diagnostic selection receipt. The filesystem
can still reject O_DIRECT, in which case the existing buffered fallback
remains. Selection alone does not establish actual direct reads, lower
page-cache pressure, successful loading or throughput. The historical
5000-tensor heuristic exists because buffered reads can be faster on
expert-heavy shards, so raising it may increase loading time.

## Unchanged Boundaries

This selector does not change allocation strategy or limits. Scalar-slab
construction, individual non-scalar device allocations, managed fallback,
pinned DMA staging, aligned host reader buffers and metadata allocations
remain. It does not fix or diagnose driver-internal allocation failures.
It preserves the reader-owned descriptor, memory guards and the existing
owned-file page-cache advice. It introduces no global cache eviction,
kernel setting or lower admission/runtime memory floor.

## Verification Scope

Four Rust unit-test definitions cover the pure selector without mutating
the process environment. They are uncompiled and unrun under this
source-only assignment. Eight bounded Python checks inspect wiring and
retained safety contracts. These are source checks, not executions of the
Rust parser, filesystem fallback, loader or model.

The selector source suite passes 8/8, with measured maximum RSS 15,404 KiB
and 0.02 seconds wall time under a 90 MiB address-space / 5 CPU-second
limit. The existing reader-descriptor and pinned-bounce source suites
also pass 6/6 and 12/12. `git diff --check` passes. `rustfmt --check`
could not start within that address-space limit because its LLVM shared
library could not be mapped. No formatting, compiler or native result
is implied by the Python checks.

Independent source review found no confirmed blocker in this three-file
candidate. It inspected the source checks without executing them and
confirmed that the selector does not touch the separate pinned-bounce
constructor candidate. Final composition still requires review.

Run the source checks from the checkout root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s repro-kit/tests -p test_fast_direct_io_selector_source.py -v
```

Independent review, a separately authorized compile/native test window,
and a fresh controlled load with actual I/O/memory receipts are required
before deciding whether the selector helps. No speed or accuracy gain is
claimed.
