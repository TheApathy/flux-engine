# R15 Flash-Next release checklist

This checklist is the release gate for the Qwen3.8-Flash-Next QAD5500 hybrid.
A source patch, static test, direct-binary smoke, and container result are
different evidence stages. A box is checked only at the stage named by that
row; none of the source-only rows imply a releasable container.

## Integrated source and offline gates

- [x] R16 source-candidate release-tool Python suite passed 529/529 after the
  operational native-262K and memory-floor drivers and their evidence-binding
  repairs, plus the PLE prefix-snapshot admission regressions. These are
  offline/mock tests, not model scores or a live container
  result. The earlier 446/446 native-linked CPU/mock scheduler result predates
  the newer Rust changes and does not qualify them.
- [x] Restore a clean full combined offline-suite result. At `7778c864`, all
  597 tests passed in 144.71 seconds with third-party pytest plugin autoload
  disabled. Independently reviewed repairs establish deterministic mocked
  admissions and bind terminal/forward state to target-fed token geometry.
  The earlier `f86e8d61` result remains 585 passed / one fixture failure, not
  a PASS. JUnit receipt: `work/r16-offline-7778c864-20261021T2038.xml`, SHA-256
  `c2d28aad5e7a88880e2d2eaadb6dc8fb8ca0af290449a72316d6ec6dfd6483ff`.
  This is offline/mock evidence, not GPU or container qualification.
- [x] Recheck the complete offline suite after the S4 comparison/lifecycle
  integration. At `a6219cec`, all 655 tests passed in 167.00 seconds with
  plugin autoload disabled. The focused S4 suite passed 58/58 separately.
  JUnit receipt: `work/r16-offline-s4-20261021T2349.xml`, SHA-256
  `59b3e63f717b17e8615c6881e291c1b9716e57d6dbf1ca67628467aa6c80fd54`.
  Runtime and container gates remain unchecked.
- [x] Repeat the 655-test offline suite after the narrow admission calibration
  at `2f6823df`: 655/655 pass in 169.05 seconds. Receipt:
  `work/r16-offline-admission-20261021T2354.xml`, SHA-256
  `b58212251f9e4a8939f2f258e5d60b9d6b927694a519d39e5957375cc29ab774`.
  This Python result does not execute the changed Rust function. The native
  test claim refused a busy build lock before compilation and remains open.
- [x] Native-262K and C1 memory-floor operational drivers are integrated with
  source-pinned lifecycle helpers, real process-proof contracts, complete
  kernel windows, exact endpoint admission checks, and consumed-byte-bound
  evidence artifacts. Independent fault tests reject late artifact rewrites.
  See `R16_NATIVE262K_GATE_DRIVER.md` and `R16_ACCURACY_GATE_DRIVERS.md`.
- [x] Independently reviewed PR162 adaptation protects mixed-prefill metadata
  from padded decode routing writes (`5e968b40`). Rust regression definitions
  passed 3/3 in the native-linked CPU checkpoint `9e3c69f9`. GPU mixed-tail
  correctness remains pending.
- [x] Claim-first slot compaction includes retained-hidden preflight rollback
  (`ed317287`, `276886d7`). Destination claims cannot be stranded by a
  pre-copy hidden validation error. Source review is complete, but the added
  pool, hidden-frame and scheduler regressions passed under compiler ownership
  at `9e3c69f9`. This is CPU ownership evidence, not GPU migration parity.
- [x] Plain launcher repairs cover both normal chunk profiles and fast-prefill
  (`bb177584`, `6dab6508`). Four execution-shim tests pass. They do not prove
  engine startup. Historical sidecar performance is explicitly separate from
  qualification of the current checkpoint-embedded MTP head.
- [x] Narrow Qwen4 transpose-before-MTP startup move is integrated after two
  independent source reviews (`4aeee685`). Its source verdict covers the
  canonical unified, non-hybrid recipe, not measured peak-memory reduction.
- [x] Independently reviewed narrow NVFP4 allocation-admission calibration is
  integrated as `d8d95713` (donor `f5aa0f55`). The 1.165 multiplier tightens
  the old 1.02 estimate without changing reserves or allocations. Source
  formatting and diff checks pass. This is not memory reclaimed or a complete
  startup-peak model. The existing `9e3c69f9` ELF does not contain this change;
  native execution and a rebuilt serving binary remain pending. Broader
  startup/cache/ledger changes remain on source-review HOLD.
- [x] Independently reviewed failed-bootstrap retirement (`f1a3351b`) rejects
  final cache admission, consumes the terminal sink without early resource
  free, and prevents a later successful completion. Injected actual-step Rust
  tests cover all seven failure edges, both response kinds and normal-stop
  controls and passed in the actual server test binary at `9e3c69f9`.
- [x] Independently reviewed Qwen4 admission repair (`3e93b5f1`) rejects
  prefix caching plus positive SSM snapshot slots before the C1 early return,
  because those snapshots omit PLE convolution history. Canonical
  prefix-disabled launches are unchanged. Six new source-only regressions
  pass; all eight direct Rust admission tests also passed at `9e3c69f9`.
  Fresh-process accuracy checks remain pending.
  See `R16_QWEN4_PREFIX_CACHE_ADMISSION.md`.
- [x] Six additional public forks and relevant topic branches were inspected at
  pinned commits (archived in historical triage notes). Source leads were
  classified as locally actionable, already addressed, ABI-incompatible or
  research-only. This survey supplies no new runtime performance or
  native-262K receipt.
- [x] Original Flux Lens provides offline exact-token workload planning,
  explicit endpoint runs, complete token/finish comparisons, private journals,
  and replay-verified reusable reports. Its context x offered-client-concurrency
  matrix supports C up to 128, not a claim of native engine width. All 48
  mock-only harness tests pass. E2E output throughput and E2E request latency
  are not prefill, decode-only, TTFT, or ITL measurements. Source CLI and image
  wiring are present; the published R8 image does not contain this addition.
  See `FLUX_LENS.md`.
- [x] Real paired MMLU-Pro/GSM8K gate drivers, the checkpoint/listener-bound
  startup-proof broker, and the whole-file boundary-safe raw-log gate are
  integrated and offline-tested. Native wide-verifier metrics distinguish
  numeric native-proposer widths from the legacy DFlash label. None of these
  source changes supplies a live accuracy score or proves a physical wide
  GPU kernel. The external startup/tool drivers have a parent-proof protocol
  candidate but remain on independent-review hold for process identity,
  failure cleanup, and complete example-plan wiring.
- [x] Long-context native-MTP source repair and fail-closed QSA admission are
  in the integration ancestry (`a2fbf9c6`, `99bc2eaf`). The scheduler-owned
  exact-token reporter matrix is also present (`691aabb5`). This is source and
  offline-gate evidence, not a post-fix runtime receipt for issue #13.
- [x] Integrate the independently reviewed S4 paired Plain/K5 driver with
  exact fed-token and target-state comparisons, setup containment, GO/EOF
  handoff, nonce-bound cleanup acknowledgement, and signal-failure retirement.
  The reviewed focused suite passes 58/58 offline tests. These mocked results
  do not close issue #13 or qualify real Docker cleanup or GPU accuracy.
  See `R16_PLAIN_K5_LONG_CONTEXT_GATE.md`.
- [x] The paired plain/native-K5 long-context gate driver
  (`r15-gates/plain-k5-long-context`) is integrated and offline-tested: every
  required rung, the `262080+64 ignore_eos` paired exact-ID arm, the
  262143+1/262144+1 endpoint probes, continuation tokens, the graph-disabled
  2,048-boundary, rollback/slot-reuse and route-telemetry arms, VOID on fatal
  runtime signals. See `R16_PLAIN_K5_LONG_CONTEXT_GATE.md`. It is not an
  issue #13 runtime receipt until the sealed suite runs it on the exact image.
- [x] Streaming and blocking tool-call termination reaches scheduler
  cancellation; tool output is validated before publication; UTF-8 and
  whitespace streaming state is retained; reasoning/content boundaries are
  explicit (`7828d7bb` in this lane).
- [x] Flash-Next sampling categories carry the vendor temperature, top-p,
  top-k, and presence-penalty rows. `min_p` and `top_n_sigma` preserve
  request-over-category-over-server precedence, including explicit zero.
- [x] The Flash-Next profile defaults to BF16 KV and conservative 12-repeat / 18
  fuzzy-tolerance watchdog thresholds. The full native target built 203 kernels
  and the focused sampling suite passed 15/15 before integration (`9df92698`).
- [x] Per-sequence MTP gate evidence is normalized by active width, with width
  transitions invalidating mixed evidence (`b6c2f041`).
- [x] Native plain C8 has a direct-binary token-parity smoke covering forward
  and reversed waves, cancellation, replacement, and reuse. This is a smoke,
  not container, K5, full-logit, or long-context qualification.
- [x] C8 validation candidates for stable native SSM pointer-table reuse and
  batched native KV writes are integrated behind explicit selectors
  (`87a88c9d`, `64927834`, `b99bf9ff`). Their 7-test offline contract passes;
  neither selector has GPU parity or performance credit yet.
- [x] Container lifecycle-v3 final independent audit passed on the exact
  committed diff (`4dc3ef04`): 55/55 full offline tests and 120/120 repeated
  high-risk signal/launch/discovery/backoff probes. This authorizes staged live
  validation; it is not itself a Docker or container-runtime receipt.
- [x] The R15 exact-image outer wrapper binds the full source-to-binary closure,
  immutable image ID and digest, ordered root filesystem layers, OCI source and
  recipe labels, all required suite receipts, the complete raw-log census, and
  lifecycle-v3 cleanup. Its dry run and offline tests are harness evidence only.
  See `R15_EXACT_IMAGE_RELEASE.md`.
- [x] The tracked R15 suite orchestrator and plan schema run from sealed memfds,
  seal every standalone gate driver, reject surviving process descendants,
  and preserve only safe relative evidence locators. This is offline harness
  infrastructure. The example plan still names unresolved gate-driver paths and
  therefore cannot qualify an image.
- [x] Review-hold repairs are wired offline-tested end to end in
  `r15_exact_image_release.py` and `r15-container-suite.py`: the suite plan
  binds the exact proven source root and tracked executable inventory as well
  as the source-bundle manifest (`source_bundle_manifest_sha256`),
  each gate evidence block is recomputed by the wrapper from a signed-expression
  `flux-r15-gate-validator/1` validator artifact, the proven checkpoint and the
  runtime tree must be disjoint even through bind aliases, launch path parents
  remain anchored by held descriptors with wrapper-owned O_EXCL diagnostics,
  and binary/recipe copies use unmounted standoff containers. Independent live
  startup proofs additionally bind actual executing bytes, read-only recipe,
  environment, namespace, cgroup, and process identity. The raw-log census is
  case-insensitive and boundary-safe with per-pattern counts. Linux subreaper
  and pidfd containment drains descendants even after `setsid` and outer-suite
  timeout. Empty retrieval answers and per-item plain/K5 accuracy disagreement
  are rejected. These are offline harness improvements; runtime container
  qualification still requires the real gate drivers and claimed resources.

## Accuracy and behavior runtime gates

- [x] Execute the newer native-linked CPU Rust regressions at `9e3c69f9`:
  completion 41/41 (including three real handler/channel tests), scheduler
  466/466 (including 11 mandatory-verifier error tests and the MTP gate),
  Qwen4 admission 8/8, mixed scratch 3/3, actual pool-slot policy 9/9,
  native hidden-frame policy 9/9, and target-state classifier/digest 5/5.
  The separate proposer-pairing source test passed 2/2. All matched counts
  were nonzero and all three compiler locks were held. These test artifacts
  use `ATLAS_SKIP_BUILD=1` CUDA-kernel stubs and are not serving binaries.
- [x] Build a real serving ELF at production checkpoint `9e3c69f9` with
  173 dense and 203 Flash-Next PTX files, five storage PTX files and native
  GDN/attention archives. Input-manifest checks and CLI help pass. Preserve
  the post-build reporting failure separately from compilation success.
  This does not qualify startup, inference or the later integration HEAD.
- [ ] Requalify fresh Plain/native-K5 startup after the early transpose move.
  The old `dc7d2925` host smoke stopped before readiness: first the Plain CLI
  flag was invalid, then both corrected launch arms failed the fast-loader
  memory admission check. No inference probes or new performance receipts
  resulted. A new `9e3c69f9` Plain diagnostic passed the unchanged guard,
  loaded all 40 shards and entered construction, but was deliberately stopped
  before readiness after new driver-internal `NV_ERR_NO_MEMORY` warnings.
  No surfaced CUDA allocation failure or managed-memory fallback was logged.
  This is an incomplete diagnostic, not proof of fatal engine OOM or a startup
  PASS. Do not bypass the memory guard or
  stop other sessions implicitly.
- [ ] Preserve canonical memory controls during startup qualification. The
  current launcher and tracked release argv use a 4,096 MiB OOM reserve.
  The historical 2,048 MiB and later 1,024 MiB smoke overrides are not matched
  canonical runs. Early target transpose occurs after the initial fast-loader
  guard and cannot clear that earlier admission failure.
- [ ] Fresh-process tool-eval replay of the reported runaway cases, including
  TC-03 and TC-51, in both streaming and blocking APIs. Prove cancellation
  completes in tens of tokens rather than at the 16K output cap.
- [ ] Fresh-process GPQA replay with intervention counts and watchdog fires
  recorded. Do not compare the old 74/100 and vLLM 91/100 as matched evidence.
- [ ] Matched 2,048 versus 57,344 thinking-budget A/B on the hybrid. Keep 2,048
  as the default until the larger budget improves quality without runaways.
- [ ] Align `FLUX_THINKING_BUDGET` documentation and launcher behavior with the
  selected, runtime-validated model default. The current generic documentation
  says 1,024 while this target profile currently selects 2,048.
- [ ] Add or validate one benchmark-control mode that disables conversation
  rewriting (F7, F23/F30, F28, F31) and watchdog intervention, or returns a
  complete intervention ledger in every benchmark response.
- [ ] Missing-space reproduction using identical returned token IDs in streamed
  and non-streamed responses, with reflection suppression independently
  disabled. Prove whether the defect is generation or rendering.
- [ ] Tool-grammar A/B for spurious extra calls, with all conversation-rewriting
  interventions disabled or explicitly reported.
- [ ] MMLU-Pro on the integrated release candidate, with immutable binary/model
  provenance and a sealed result artifact.
- [ ] GSM8K on the same candidate and provenance.

## MTP K5 and long-context gates

- [ ] Close issue #13 only after fresh-process greedy identity between plain and
  native K5 at the 2,048 QSA boundary and long-context rungs. Minimum prompt
  rungs: 2,017, every 2,043-2,050 boundary row, 2,112, 7,336, 7,710, 8,192,
  30,647, 32,768, 65,536, 114,755, 122,413, 131,072, and 238,938.
- [ ] At every rung compare token IDs, finish reason, target-state receipt, and
  a subsequent ordinary target token. A correct final string alone is not
  enough.
- [ ] Prove the model-native QAD5500 drafter is the loaded proposer and K=5 is
  physically active; record acceptance depth without treating acceptance as
  correctness.
- [ ] True 262K BF16 retrieval with exact tokenizer count and request geometry.
  Capacity or a 136K receipt does not satisfy this gate. Prove `262080+64` and
  `262143+1` reach the inclusive 262,144 total-token endpoint, and prove
  `262144+1` is rejected before dispatch.
- [ ] Add the missing paired plain/K5 `262080+64`, `ignore_eos=true`, exact-ID
  arm. The current issue matrix reaches a 238,938-token prompt and the current
  native-262K gate is target-only; neither proves K5 identity at the endpoint.
- [ ] Add graph-disabled 2,048-boundary, rollback/slot-reuse, and route-telemetry
  arms. Distinguish rows-v2 launches from the row-serial long-context fallback.
- [ ] Repeat the 262K retrieval in the container with the exact release image,
  no prefix-cache hit, and complete uncapped NVRM/Xid/OOM logs.

## Concurrency and performance gates

- [ ] Native plain C2/C4/C8 full-logit and recurrent-state parity against
  independent C1 controls. The existing C8 token smoke is insufficient.
- [x] Repair the concurrency validator's production-geometry mismatch in
  source/offline tests (`57609732`, `ea60429c`, `7778c864`). Normal final
  target-fed state is `prompt + completion - 1`, and
  frame pre-step length is `prompt + logits_index - 1`. Cancellation is a
  distinct phase: a forward can compute one additional unemitted row before
  the cancellation check. Faithful driver fixtures and deterministic
  admissions cover these differences; focused probe/matrix/driver tests
  passed 107/107 and the full offline suite passed 597/597. The operational
  driver is integrated, not runtime-qualified by its mocked passes.
- [ ] Native K5 plus concurrency. The current target-only C2-C8 path rejects an
  active proposer, and the current K5 route remains per-sequence past the
  scheduler's limited batched depths.
- [ ] Implement stable per-sequence QSA rollback journals, hidden-state stashes,
  proposer state, and sequence-major CxK verification before enabling C8 K5.
- [ ] Qualify C16, C32, C64, and C128. Refuse unsupported widths rather than
  silently serializing while reporting native concurrency.
- [ ] Optimize C8 only after a matched stage profile. Track HTTP whole-wave and
  decode-only throughput separately; the current observed values are roughly
  67 tok/s whole-wave and 79 tok/s inferred decode-only.
- [ ] Qualify the SSM pointer-cache and batched-KV selectors independently and
  together: C8-to-C7-to-C8 width churn, same-width replacement, cancellation,
  row removal, slot reuse, forward/reverse order, selector-off A/B, KV-byte
  parity where feasible, terminal-token parity, and per-frame full logits.
- [ ] Reproduce host high-prefill and interactive profile numbers in-container
  at 8K, 16K, 32K, 64K, 128K, and native 262K using 4,096-token chunking.
- [ ] Seal plain and K5 decode at each context with the same prompt, timing
  window, sampler, and checkpoint. Do not mix replay, direct-binary, and HTTP
  numbers on one performance claim.
- [ ] Repair performance-driver metric and ownership holes before live use.
  Independent CPU counterexamples against isolated candidate `144ae7b3`
  still accept first-event-coalesced 50 tok/s decode as 59.0278 tok/s,
  reject a healthy 256-text-event stream, and admit engine/client prefill
  clock mismatches. List-inequality profile checks accept reordered identical
  flags or explicit default FIFO as distinct recipes. Numeric-PID shell
  TERM/sleep/KILL ownership also remains unresolved. Candidate stays HOLD;
  no source/mock PASS can substitute for matched workload/timing evidence.

## Container and publication gates

- [ ] Build one immutable R15 image from a clean integrated commit and record
  source commit, binary hash, image ID, model identity, recipe hash, and launch
  argv.
- [ ] Run fresh-process container startup, readiness, tool APIs, K5/plain
  accuracy, long-context retrieval, concurrency, performance, memory-floor,
  cancellation, and clean-teardown gates.
- [ ] Require complete raw log scans with zero CUDA fatal, OOM, NVRM, and Xid
  matches. Bounded log tails are display evidence only.
- [ ] Publish to GHCR only after every required release gate above passes.
  Record the immutable digest and update the version tag to that digest.
- [ ] Post objective issue updates without host usernames, local paths, tokens,
  or unverified performance claims. Close issue #13 only with its sealed
  post-fix runtime receipt.

