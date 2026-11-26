# R16 Accuracy And Startup Composition

This candidate combines reviewed source repairs. It has not qualified model
accuracy, startup memory, serving, performance or a container image.

## Source Provenance

- Base `3b8af27beb582881b53859bcefcefa8bf071ec79` combines the eager grouped
  loader, one-slot bounce ownership, targeted native test definitions and GDN
  bridge hygiene. The earlier eager binary does not contain these later fixes.
- Plain QSA graph policy donor `c032dfbb3e880b3670dd65d881c2ccb2e9147dd7`
  binds Qwen4 identity, QSA need, the graph-safe selector and the 2048 boundary
  to both ordinary and PLE suffix graph admission.
- Shortened native-MTP donor `18d9a57e50b70d61e745aee854c24fca2f8c46e7`
  routes Qwen4 fixed-K3/K4 entries to the existing architecture-aware generic
  verifier before effects, preserving physical width and propagated errors.
- Grouped native-MTP donor `10895002a84f72957cccc3c3dd27490bf04f9e9f`
  admits native MTP only under existing exact grouped-loader geometry, while
  MTP tensors remain individually owned. Other draft methods stay refused.
- Startup parser donor `72ab9a39ea9c719650bfca325f2326c033ed59c9` rejects
  invalid batch-token overrides and validates WIDE before weight loading,
  without changing defaults, widths or resolved buffer minima.
- QSA oracle donor `ee7981de22d9b76768bf5e1b158fbaa78176ad8e` prepares
  existing undo storage and snapshots each row before writes. Its verified-row
  hook disables QSA side-stream overlap using the existing transaction flag.
  Plain, KV-only and prefill overlap defaults remain unchanged.

The shortened-frame and oracle-journal repairs are composed together. Neither
source tests nor correct output on full K5 can establish shortened-frame
partial-accept correctness.

The early-unused-MTP source regression now checks the actual individual
allocation, bounded copy, managed fallback and allocation audit after grouped
destination selection. Mutation checks reject removed copies, swallowed
allocation errors, zero-byte managed copies and missing audits. This repairs
a stale destructuring-string assertion, not production allocation behavior.

## Executed Source Checks

All nine bounded suites pass on the composed source:

| Suite | Checks |
| --- | ---: |
| Grouped bounce ranges | 6 |
| Routed backing groups | 27 |
| Eager grouped one-slot | 7 |
| Native-MTP-safe grouping | 6 |
| Early unused-MTP release | 9 |
| Plain QSA graph admission | 8 |
| Shortened Qwen4 verification | 7 |
| Prefill startup parser | 9 |
| QSA oracle journaling | 14 |
| Total | 93 |

These are source/reference and mutation checks, not executed Rust or device
parity. Prior CPU_NATIVE_ONLY 77/77 evidence applies to the separately pinned
loader-test source, not this candidate. The two native-MTP ownership tests,
six graph policy tests and ten new parser tests remain unexecuted here.

The separately guarded FP16 GDN host-stub v2 run compiled its objects, then
failed host linking because its shim omitted three compiler-emitted CUDA ABI
symbols. All 13 behavioral cases remain NOT_RUN. Preserve that evidence and
qualify a new fixture rather than changing the frozen run or counting PASS.

## Remaining Release Gates

1. Independent combined-source review, native regression execution, exact
   real-CUDA build and native closure census using fresh sealed inputs.
2. No-cache Plain readiness, strict API/raw-ID checks, finite hotel completion
   with natural stop and correct final answer, then the 8K request grid.
3. Forced shortened MTP frames at every accept count, compression alignment,
   2047/2048 transitions, repeated-greedy and committed-state parity.
4. True 262144-token retrieval, LF009, GSM8K and MMLU-Pro accuracy.
5. Real native concurrency and matched interactive/high-prefill profiles.
   Serial fallback is not native aggregate-throughput qualification.
6. Exact-image build, identity-bound end-to-end qualification, then GHCR
   promotion. No image, alias or public issue is promoted by this source work.

Keep native/model/image resource claims separate. Preserve strict warning
stops, the native 100 GiB floor, model 8192 MiB ongoing floor and finite caps.
The one-off cache-drop/compaction authorization remains consumed.
