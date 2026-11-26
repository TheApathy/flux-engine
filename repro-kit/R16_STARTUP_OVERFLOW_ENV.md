# R16 startup capacity and environment repair

Status: source/offline checks only. This change has not been compiled or executed
as Rust. It does not establish binary, model, GPU, accuracy, speed or container
qualification.

Base commit: `82e429db05ae80ca170a84e2e871427e45dfadb6`.

## Scope

- `crates/spark-server/src/main.rs`
- `crates/spark-server/src/main_modules/serve_phases/kv_cache.rs`
- `repro-kit/tests/test_r16_startup_overflow_env_source.py`
- This document

No speculative mode, WIDE selector, chunk-size, KV-cache dtype or serving default
is enabled or changed. The asynchronous application body is unchanged, apart
from moving the alias-sync call out of that body into synchronous `main`.

## Capacity admission

The strict ASCII-decimal parser remains responsible for syntax, `usize`
representability and the existing minimum. Its ability to parse `usize::MAX`
does not make that value valid resolved geometry.

The shared preflight/runtime resolver now checks:

```text
default_min = max(checked(prefill_budget + max_batch_size),
                  speculative_rows, padded_decode_rows(max_batch_size))
sequence_capacity = checked(max_seq_len * max_batch_size)
upper_bound = max(default_min, sequence_capacity)
default_min <= resolved_max_batch_tokens <= upper_bound
```

An overflowing sequence-capacity product fails whether an override is present
or absent. A representable but oversized explicit override fails before its
acceptance log and before this resolver returns geometry to its consumers.
The `max(default_min, ...)` preserves tiny-context speculative and padded decode
floors that intentionally exceed the logical sequence product.

Examples covered by Rust regression definitions:

- Chunk 4096, sequence 4096, C128: default 4224 remains unchanged. Exact sequence
  capacity 524288 is accepted, while 524289 and `usize::MAX` are rejected.
- Sequence `usize::MAX`, C2: the checked product rejects both absent and ordinary
  explicit overrides.
- Sequence 1, C1 with four speculative drafts: minimum and bound remain 6.
- Sequence 1 at C33/C65/C97: physical padded bounds remain 48/96/128.

This is a bounded admission fix. It does not prove that every later
`BufferSizes`, model-config, checkpoint, high-speed-swap or speculative-count
arithmetic operation is checked, or that every admitted shape fits device memory.

## Environment aliasing and startup order

Environment capture uses `vars_os()` and retains each value as `OsString`.
Only keys are inspected as Unicode, by the pure `flux_env_alias_key` helper.
Unrelated non-Unicode values therefore cannot panic a Unicode-only snapshot.
Non-Unicode keys are ignored because they cannot name one of the supported
Unicode selectors. Relevant non-Unicode alias values are copied without loss so
their existing strict selector parsers can reject them explicitly.

Existing behavior is preserved:

- The seven existing `FLUX_*` META names and the empty suffix do not alias.
- Generic names map to `ATLAS_<suffix>`.
- Both special WIDE and fast-projection-cache-reserve mappings remain unchanged.
- Any existing ATLAS value wins, including empty or non-Unicode values, because
  presence is checked with `var_os(...).is_some()` before each write.
- Snapshot iteration and first-created-alias behavior remain unchanged when
  more than one FLUX spelling targets the same ATLAS selector.

Synchronous `main` performs alias synchronization and then invokes the separate
`#[tokio::main]` application function. Tokio runtime construction therefore no
longer precedes this environment mutation. Pure Rust tests exercise key mapping,
META exclusions, absent-only selection and raw non-Unicode values without
mutating the test process environment.

## Offline verification

The following two suites pass, 9 checks each, plus `git diff --check`:

```sh
python3 -B -m unittest discover -s repro-kit/tests -p 'test_r16_startup_overflow_env_source.py' -v
python3 -B -m unittest discover -s repro-kit/tests -p 'test_r16_prefill_startup_parser_source.py' -q
```

The new suite inspects bounded source contracts and rejects mutations to checked
capacity arithmetic, the minimum floor, comparison direction, raw environment
capture, ATLAS presence checks and synchronous-before-runtime ordering. It also
checks that the corresponding Rust regression definitions exist. These are not
executions of those Rust regressions.

No compiler, native fixture, controller, GPU, model, server, container, resource
claim or public write was used. Native compilation and independent review are
still required before integration qualification. Accuracy and matched
prefill/decode receipts remain separate acceptance gates.
