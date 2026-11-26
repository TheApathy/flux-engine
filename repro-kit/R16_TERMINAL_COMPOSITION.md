# R16 Terminal And Startup Source Composition

This is a source candidate, not a serving/container release. Native model
loading, accuracy, throughput and exact-image gates remain open.

## Provenance

Base `82e429db05ae80ca170a84e2e871427e45dfadb6` retains the prior eager loader,
MTP-safe grouping, Plain graph policy, shortened K3/K4 delegation and QSA oracle
journal composition. New donors:

| Donor | Applied Commit | Scope |
| --- | --- | --- |
| `70370490d8ff8701c4a128169f04759bd24b26e0` | ancestor | K3/K4 abort, classified-row terminal commit, postcommit failures |
| `b94ab71abaf2c6724433c8f0369ddfce34972be7` | `9e673aad` | Bounded startup override and raw aliases before runtime |
| `4984556873f8f5cd756fcee081f9785f692bf00f` | `129e2781` | Early exact-width/tree QSA journal admission |
| `1e83127f8d392105a5dcf5fd4893fda59f9a66ad` | `88e57b03` | Qwen4 fixed-K2 delegation and recursive legacy refusal |
| `b8d35bdb217d2038a5868b1072a1842915b269db` | `78517a82` | K2 scheduler abort, terminal and postcommit boundaries |

The only cherry-pick conflict was in the `verify_d.rs` test-module imports and
neighboring test insertion point. Both donors' imports and all eight new test
definitions are retained. No production conflict required manual resolution.
Composition commit `6d59fa82` updates the admission source test to bind the
replacement nonrecursive legacy guard and explicitly reject a restored back-edge.
The pre-existing terminal-error source assertion is updated for the K2 helper
wiring, with exact direct-notification and rollback-notification counts.

## Executed Checks

All 148 focused source/reference/mutation methods pass under the unchanged
90 MiB address-space and five-second per-command caps:

| Suite | Checks |
| --- | ---: |
| Grouped bounce | 6 |
| Routed backing ownership | 27 |
| Eager one-slot grouping | 7 |
| Native-MTP-safe grouping | 6 |
| Early MTP release | 9 |
| Plain QSA graphs | 8 |
| Shortened model delegation | 7 |
| Startup strict parser | 9 |
| Oracle QSA journal | 14 |
| Startup overflow and raw aliases | 9 |
| QSA frame admission | 11 |
| K2 model delegation | 9 |
| K3/K4 short-frame boundaries | 8 |
| K2 scheduler cleanup | 10 |
| Existing speculative terminal notifications | 8 |
| Total | 148 |

The final suite is function-based rather than unittest-based. Its eight methods
were explicitly collected and called; a zero-method unittest invocation is not
counted. An earlier exploratory run found zero free functions in class-based
suites; that run established no coverage and was replaced by actual executions.

Seventeen new scheduler recording definitions (K3/K4 nine, K2 eight) are
registered but NOT_RUN. Other new model/startup Rust definitions likewise await
native compilation/execution. Older CPU_NATIVE_ONLY 77/77 is not evidence for
these added definitions. Rustfmt could not map LLVM under the source cap, so
no cap was weakened and the files were manually formatted.

## Explicit Remaining Holds

- Generic/tree/batched terminal paths and policy walkers still derive retained
  input count from visible output delta. A separate isolated donor is repairing
  hidden control rows and preclassification cancellation.
- The startup capacity bound is not universal overflow protection. Extreme CLI
  default geometry can still reach unchecked BufferSizes/preflight arithmetic.
  A separate checked-arena donor is in progress.
- A failed backend commit can already have overwritten canonical checkpoints.
  Error abort remains best-effort, terminal and noncacheable, not exact device
  rollback proof. EP fixed-width terminal protocol is also unqualified.
- No current source build, exact-binary readiness/quality, Plain/K5 state parity,
  LF009, true262144 retrieval, GSM8K/MMLU-Pro, native concurrency, matched profile
  or exact-image E2E is established by this document. No GHCR promotion follows.

Preserve all frozen donors, failed native receipts and older controls. Complete
and independently review the remaining source repairs before pinning a final
build controller/snapshot. Keep native/model/image ownership separate.
