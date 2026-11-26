# R16 Checked Arena Arithmetic

Source/offline repair based on frozen startup-overflow donor
`b94ab71abaf2c6724433c8f0369ddfce34972be7`. This is not a native build,
model-load, accuracy, serving, throughput, or container-promotion receipt.

## Scope

- `BufferSizes::from_config` and `BufferSizes::total_bytes` now return
  `anyhow::Result`. Arena products, additions, alignments, work-list counts,
  split-K extents, merged-FFN output, and the 21-buffer total are checked.
- Local `ArenaWidths` computes fallible residual, linear-attention, and
  Mamba projection widths. Global `ModelConfig` getters are unchanged.
- The constructor validates its aggregate total before returning. The
  allocator checks that total again before its first `gpu.alloc` call.
  The model factory estimate and serving preflight propagate both errors.
- `DecodeMetaLayout::checked_meta_bytes` checks the metadata prefix,
  block-table product, and their sum. Arena admission uses it. Legacy
  metadata helpers and unrelated upload callers are deliberately unchanged.
- One pure checked GDN helper replaces both the two-phase preflight estimate
  and its diagnostic per-token duplicate. Its valid per-row formula remains
  `2 * (2 * key_width + value_width) + 8 * value_heads + 4 * value_width`.
  No-SSM and zero-convolution cases remain zero.

Representable valid geometry retains its existing dtype choices, sentinel
allocations, C32 floor, C128 padding, expert-major mixer slicing, MRoPE
metadata, FlashInfer cap, and opt-in selectors. Zero KV block size is now
rejected instead of silently substituting an arbitrary block-table size.
Overflow now returns a labeled capacity error rather than wrapping,
saturating the residual width, or panicking in a sizing helper.

The allocator's initialization counter remains a subset sum of the same
21 fields whose aggregate total has already been checked. The preflight
diagnostic fixed-cost sum is likewise a subset of the earlier checked
reserve total. This patch does not claim all arithmetic elsewhere in the
engine is checked, nor does it add allocation-failure rollback.

## Validation

Run each command with a 90 MiB virtual-memory cap and a five-second CPU cap:

```sh
ulimit -v 92160
ulimit -t 5
set -e
python3 -B -m unittest discover -s repro-kit/tests -p 'test_r16_checked_arena_source.py' -q
python3 -B -m unittest discover -s repro-kit/tests -p 'test_r16_prefill_startup_parser_source.py' -q
python3 -B -m unittest discover -s repro-kit/tests -p 'test_r16_startup_overflow_env_source.py' -q
git --no-optional-locks -c core.packedGitWindowSize=1m -c core.packedGitLimit=8m diff --check
```

Latest local result: 11 new source/mutation checks PASS, both existing
startup suites PASS at 9 checks each, and whitespace validation PASS.
These checks inspect source contracts and reject deliberate source mutants.
They do not execute Rust or establish binary behavior.

There are 16 new Rust regression definitions, all **NOT_RUN**:

- Four checked-arithmetic/local-width tests cover exact boundaries,
  malformed alignment, derived products, sums, and ordinary-width parity.
- Seven arena tests cover huge row counts, block tables, padded decode
  metadata, zero block size, malformed widths, aggregate-only overflow,
  public-field mutation, C128, and MRoPE. Invalid allocation cases assert
  zero mock allocations.
- One metadata test and one split-K/work-list test cover layout parity,
  zero divisors, exact capacity, and overflow.
- Three GDN tests cover ordinary geometry, exact capacity, dimension/row
  overflow, and inactive behavior.

The first source-suite attempt reported three test errors because its
normalizer did not account for Rust trailing commas. The next attempt
reported one error that exposed the remaining unchecked diagnostic GDN
duplicate. Both failures are superseded by the later passing result, not
reclassified as passing runs. The duplicate now calls the checked helper.

## Remaining Gates

Independent source review, an authorized compiler/native test run, and the
existing full accuracy, state-parity, long-context, serving, performance,
and container gates are still required. No compiler, formatter, GPU,
server, model, or container was run for this donor. This patch provides no
new prefill/decode measurement and does not qualify speculative decoding.
