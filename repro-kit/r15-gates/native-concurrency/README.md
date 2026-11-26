# Live native-concurrency C2-C8 release gate driver

`r15-gates/native-concurrency/native-concurrency-c2-c8` is the operational
layer that produces the `native-concurrency-c2-c8` gate evidence for the
exact-image release suite. It launches fresh in-container server processes,
runs the already-integrated qualifier's request plan and validators against
each one, seals the plain nine-arm matrix with the untouched matrix sealer,
runs the native-K5 nine-arm matrix under the engaged-native-mtp state
contract, and emits the wrapper's 18-arm evidence. It never ingests prior
reports, never accepts a PASS flag, and never synthesizes a missing proof.

Offline tests mock every Docker, server, HTTP, and process binding. They do
not qualify an image.

## Reuse map

| Concern | Reused as-is from | How the driver calls it |
|---|---|---|
| Request plan, C1 controls, forward/reverse waves, C8 cancel/replace/drain/reuse, full-logits and terminal-state parity, frame ownership, evidence-line grammar, memory sampler, log cursor | `repro-kit/qwen4_native_concurrency_probe.py` | loaded from pinned bytes under a private module name with `r9_provenance` injected; `make_cases`, `stream_request`, `run_wave`, `run_api_churn`, `parse_log`, `validate_wave`, `validate_full_churn`, `validate_reuse_phase`, `validate_reference_receipt`, `validate_log_binding`, `validate_checkpoint_inventory`, `validate_r15_source_closure` are called unchanged |
| Nine-arm plain matrix seal (replay of every token/logit/state/session/slot/sequence chain, fresh process and log identities, Git-captured source closure) | `repro-kit/qwen4_native_concurrency_matrix.py` | loaded from pinned bytes with `__file__` set to its tracked path so its import-time Git capture runs; `load_arms`, `validate_matrix`, `revalidate_reviewed_source_closure` are called unchanged |
| R15 source-to-binary closure | `repro-kit/r9_source_to_binary.py` | pinned bytes, injected into the probe |
| Live process proof (executable hash, recipe environment, `/models` bind, namespace, cgroup, PID-owned listener, mode) | `repro-kit/r15-container-suite.py` `verify_live_server` | before and after every arm, exactly as the accuracy drivers do |
| Container inspection, pidfd ownership, cleanup, sealed-parent pin, pinned-module loader, artifact creation beneath the runtime | `r15-gates/mmlu-pro` pattern | mirrored |

Nothing in the probe or sealer is weakened. The probe's direct-binary
`validate_provenance` is the one function the driver does not call, because it
resolves `argv[0]`, cwd, checkpoint, and bundle paths on the probe host. The
driver performs each of its checks in container terms instead:

- executable: `/proc/PID/exe` hash equals the image label and the release
  provenance `binary_sha256` (also re-proven by `verify_live_server`)
- argv, cwd, and the `ATLAS_`/`FLUX_`/`RUST_LOG` environment: read from the
  host `/proc` view and compared with the per-arm launch manifest exactly
- log binding: `probe.validate_log_binding` on the host path of the
  `/results` bind, which the shell redirect makes the server's fd 1 and fd 2
- source closure: `probe.validate_r15_source_closure` on the release launch
  manifest's provenance (host paths the wrapper already verified)
- checkpoint: `probe.validate_checkpoint_inventory` with full byte hashing once
  per driver run, then the probe's identity recheck before and after each arm;
  the `/models` bind source must equal the provenance checkpoint path

## Inputs

```text
--container-id @CONTAINER_ID@
--image-id @IMAGE_ID@
--image-digest @IMAGE_DIGEST@
--base-url @BASE_URL@
--run-id @RUN_ID@
--receipt-fd @GATE_RECEIPT_FD@
--raw-log-fd @RAW_LOG_FD@
--runtime-dir <the exact suite runtime directory, bound at /results>
--suite-program <tracked repro-kit/r15-container-suite.py>
--suite-sha256 <the exact suite.program_sha256>
--probe-program <tracked repro-kit/qwen4_native_concurrency_probe.py>
--probe-sha256 <its SHA-256; must equal the Git blob the sealer captures>
--r9-program <tracked repro-kit/r9_source_to_binary.py>
--r9-sha256 <its SHA-256>
--matrix-program <tracked repro-kit/qwen4_native_concurrency_matrix.py>
--matrix-sha256 <its SHA-256>
--launch-manifest <the exact-image launch manifest with the 21-key provenance>
--launch-manifest-sha256 <its SHA-256>
--container-port 8888            (default)
--container-results-path /results (default; in-container mount of runtime-dir)
--container-cwd /                 (default)
--max-tokens 64                   (16..256)
--minimum-full-frames 8           (cannot go below 8)
--memory-floor-mib 6656           (probe default; plan may raise it)
--timeout 900
--cancel-token 20                 (P8 mid-row cancellation boundary)
--modes none,native-mtp-k5        (a subset never produces a full-scope PASS receipt)
--scope full                      (default; "plain" requires --modes none, see below)
--ladder                          (optional C9-C32 rungs)
--ladder-widths 9-32
--ladder-profiles native-reference
```

The parent must be the sealed suite memfd (same pin as the accuracy drivers).
The matrix sealer must run from a Git worktree whose HEAD blobs for the three
reviewed files equal the pinned bytes; the driver refuses otherwise.

## Execution

Per mode (`none` first, then `native-mtp-k5`), per width (2, 4, 8), per
selector profile (`native-reference`, `native-optimized`, `serial-control`):

1. A fresh process via `docker exec -w <cwd> <cid> /bin/sh -c 'exec
   /usr/bin/env <recipe + arm env> <binary> serve ... >>"$0" 2>&1'
   /results/<arm>.server.log`. The server's stdout and stderr are the arm's
   own log file; `exec` means the shell and `env` are replaced, so the live
   process argv is the manifest argv.
2. Arm argv: `serve --no-tui --model-from-path /models --model-name <alias>
   --bind 0.0.0.0 --port <port> --qwen4-qsa --max-batch-size W
   --max-num-seqs W --kernel-target qwen3.8-flash-next --world-size 1
   --tp-size 1 --ep-size 1 --kv-cache-dtype bf16 --block-size 16
   --max-seq-len 2048 --max-prefill-tokens 1024 --ssm-cache-slots 0
   --swap-space-gb 0 --enable-prefix-caching false --request-timeout 900`,
   plus `--speculative --num-drafts 4` in K5 mode.
3. Arm environment: the image recipe, the profile's six selectors
   (`ATLAS_QWEN4_NATIVE_BATCH` plus the five optimization selectors), the
   three diagnostics (`ATLAS_LOGITS_FNV=1`,
   `ATLAS_QWEN4_NATIVE_BATCH_RECEIPT=1`, `ATLAS_QWEN4_TARGET_STATE_RECEIPT=1`),
   and `ATLAS_DISABLE_SPECULATION=1` (plain) or `0` (K5). The recipe must
   not pin any of those keys and its `RUST_LOG` must start at `info` or more
   verbose, because the receipts are `tracing::info!` lines. One public model
   alias is shared by every arm of a run, as the sealer's common contract
   requires; the pid is still located by the binary path plus that alias.
4. Readiness: two `/health` probes, `verify_live_server`, then the manifest
   binding above. The proof is repeated after the arm and must be identical.
5. Probe phases, unchanged: isolated C1 for every prompt (plus the replacement
   prompt at C8), forward wave, reverse wave, and at C8 the eight-chunk
   cancellation of the final admission, replacement into the retired highest
   slot, full-width restoration, drain, and post-drain reuse.
6. Driver phases, appended in the same process:
   - same-prompt wave: W identical prompts; terminal token/finish/usage parity
     against the C1 control and the executed-width census (sessions alias, so
     this phase is token-level by construction)
   - P8 mid-row cancellation at C8: row `W/2-1` cancelled after
     `--cancel-token` content chunks; survivors and the replacement must match
     C1 tokens, full logits, and terminal state, the replacement must enter
     after the cancelled state retired and reuse the retired slot
   - executed-width census for every phase: native and serial frame counts by
     row count from the phase's log window; a native arm fails when the
     claimed width was not the maximum native width executed or any C2+ serial
     frame appears (collapse to serial), a serial control fails on any native
     frame
   - TTFT, decode seconds, and tokens-per-second per request from a hook around
     the probe's `urlopen`; diagnostic only
   - first-divergence token index per request and forward/reverse
     self-consistency, recorded before the probe's parity check raises
7. Arm close: qualified-log evidence-line grammar, forbidden-marker census (any hit
   is a VOID verdict), exactly one `QWEN4_NATIVE_BATCH_ENGAGED rows=W`
   (none for the serial control), optimization-selector engagement counts,
   MemAvailable floor, `/metrics` K5 counter (zero in plain, positive in K5),
   provenance and process identity rechecks, final proof, SIGTERM through the
   pidfd. Forced cleanup fails the gate.
8. After the nine plain arms: the matrix sealer produces
   `concurrency/native-concurrency-c2-c8/matrix/none.json`.
9. Native-K5 arms run the identical phases. The probe's state-receipt checks
   are applied through a contract shim that demands
   `mode=engaged-native-mtp scheduler_path=native-mtp proposer_loaded=true`
   and then delegates every other check to the probe unchanged. These arms
   are validated by the driver, not sealed by the matrix, and their receipts
   carry that limitation. The engine at this base refuses the native batched
   path with an active proposer, so the K5 half fails closed today; the plain
   seal and every private observation remain on disk for diagnosis.
10. Optional ladder: one fresh native process per width C9-C32 with C1
    controls and forward/reverse waves replayed by a width-agnostic copy of
    the wave checks. Ladder rows are reported separately from the 18-arm
    census.

Every arm's complete server log is appended from its owned descriptor to the
inherited raw log as an `R15_LOG_FRAME`, including readiness, request, proof,
and cleanup failures. Publication runs after process cleanup. A failed launcher
still closes and publishes its empty log. The suite's final uncapped scan
covers the complete log, including shutdown output.
The final artifact pathname must still identify that owned descriptor.

## Artifacts

Below `concurrency/native-concurrency-c2-c8/` in the runtime directory, all
mode 0400 after sealing:

- `arms/<mode>/<profile>-c<W>.manifest.json`: the per-arm launch manifest
  (probe launch schema) with the live environment prediction
- `arms/<mode>/<profile>-c<W>.server.log`: the complete server log
- `arms/<mode>/<profile>-c<W>.json`: the probe-schema arm receipt
- `matrix/none.json`: the sealed plain matrix
- `ladder/<profile>-c<W>.*`: ladder manifests, logs, receipts
- `validator.json`: the `flux-r15-gate-validator/1` document
- `failure-<hex>.json`: on failure only; failure class (`FAIL`/`VOID`),
  error, timing rows, diagnostics, process bindings

`validator.json` `raw` carries the wrapper's exact census (`arms` with the
seven exact keys per arm, `available_mib_samples` as first/floor/last per arm,
the three churn booleans) plus `plain_matrix_seal`, `arm_observations`
(executed widths, same-prompt and mid-cancel evidence, K5 counters, receipt
and log locators with hashes), `process_bindings`, `timing_rows`,
`diagnostics`, `ladder`, `provenance`, `model`, `max_tokens`, `cancel_token`.

## Evidence keys

The receipt evidence is the wrapper's exact 13-key contract:
`levels [2,4,8]`, `modes ["none","native-mtp-k5"]`, `fresh_processes 18`,
`minimum_full_frames`, `forward_reverse_order_all_frames`,
`per_frame_logit_identity`, `per_frame_state_identity`,
`cancellation_replacement_pass`, `final_highest_slot_reused`,
`post_drain_clean`, `minimum_available_mib`, `validator_artifact`,
`validator_receipt_sha256`.

## Offline validation

```bash
python3 -m unittest discover -s repro-kit/tests \
  -p 'test_r16_native_concurrency_gate_driver.py'
```

The tests drive the real probe validators and the real Git-captured sealer
with a fake lockstep engine, and include hostile fixtures for serial collapse,
token drift (with first-divergence diagnosis), logits drift, engagement-width
mismatch, replacement into a fresh slot, duplicated physical slots, missing
state receipts, NVRM/Xid markers (VOID), plain receipts in a K5 arm, inverted
K5 counters, and a plain-only mode subset.

## Source-review follow-up

The middle-row cancellation check now validates the cancelled request's full
receipt metadata through the same mode contract as survivors. Its output count
must cover the content chunks already received and remain strictly below the
C1 completion budget. Logit and state sessions must exactly equal the original
wave plus replacement. Native-K5 cancellation cannot substitute a plain-mode
receipt.

The optional ladder checks each frame's absolute sequence length against
`prompt_tokens + output_index`, as well as complete logits, exact session
membership, route metadata, width, stable slots, and terminal state. A
stepwise but uniformly shifted sequence-length history is not accepted.

Artifact hashes distinguish two scopes. The probe receipt's
`server_log_sha256` covers its explicitly bounded qualified prefix, ending at
`server_log_qualified_end_offset`. The observation's `server_log.sha256`
covers the complete, closed artifact, including shutdown output. Failed arms
do not create PASS receipts, but their complete logs remain in the raw stream.

### Plain emitter dependency

The canonical plain launch has no proposer and retains
`ATLAS_DISABLE_SPECULATION=1`, as required by the unchanged live-process
verifier. At original driver commit
`1d3a0c4be8687f5b54c84fb56fde003ff2996806`, the real target-state emitter
rejects `(None, true)`. A fake that unconditionally manufactures a plain
receipt would hide that incompatibility.

This driver requires production classifier repair
`d31873532f7fd10a890677d30e78fdd2637c5c57` (isolated-lane cherry-pick
`2b65347d`) in
`crates/spark-model/src/model/qwen4_target_state_receipt.rs`:
`classify_capture_mode(None, true)` must truthfully identify the same
`unloaded-target-only / ordinary-target-only / proposer_loaded=false`
state as `(None, false)`. No mode normalization, loader relaxation, alternate
receipt schema, or proof-verifier change is used here. Loaded-serial controls
remain distinct and still require a loaded native proposer.

Offline regressions read the original source bytes to demonstrate rejection
and require the repaired production classifier as an explicit source
dependency for acceptance. Those are Python/source-contract checks, not Rust
execution or GPU qualification. Other successful fake-engine fixtures model
the complete requested protocol, not a claim that the current engine supports
concurrent native K5.

The matrix also requires an immutable image recipe that does not assign the
six qualification selector keys. A recipe pinning optimization selectors
cannot be silently overridden across reference and optimized arms. The
existing selector-pin rejection and recipe-bound live proof remain intact.
In particular, `repro-kit/recipes/flux-qwen4-native-262k-bf16.env` assigns
`ATLAS_ATTN_QKV_EXACT_STRIDED`, `ATLAS_ATTN_QK_NORM_BATCHED`,
`ATLAS_ATTN_GATE_BATCHED`, `ATLAS_ATTN_KV_WRITE_BATCHED`, and
`ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE`. That recipe is an operational HOLD for
this gate until a separately reviewed qualification recipe or protocol
resolves the conflict. This driver does not alter the recipe.

### Qualification recipe

Every serving recipe assigns `ATLAS_QWEN4_PREFILL_WIDE` (4096, 8192 or
16384), and startup refuses a WIDE value that differs from the resolved
prefill chunk (`crates/spark-server/src/main_modules/serve_phases/kv_cache.rs`).
The arm argv pins `--max-prefill-tokens 1024`, so every arm launched from a
serving recipe stops before READY. The live proof requires the live
environment to equal the recipe, so the driver cannot override WIDE.

`repro-kit/recipes/flux-qwen4-native-concurrency-c8.env` is
`flux-v9-max-prefill.env` with `FLUX_QWEN4_PREFILL_WIDE` removed and
`ATLAS_QWEN4_PREFILL_WIDE=0`. WIDE has no value of 1024, so matching the
budget instead would require changing the probe's and driver's pinned
`--max-prefill-tokens`. WIDE=0 keeps the fast-prefill family for every native
prompt: each is one chunk at offset 0 ending at or below 1024 rows, inside the
legacy 2048-row window. Build the concurrency image with
`--build-arg FLUX_RECIPE_NAME=flux-qwen4-native-concurrency-c8.env`.

For the QAD5500 checkpoint with the existing bounded-owner16 loading path,
build a distinct qualification image using
`--build-arg FLUX_RECIPE_NAME=flux-qwen4-native-concurrency-c8-owner16.env`.
This dedicated recipe preserves the generic short-context geometry and adds
the C1 eager grouped-load controls, bounded 16 MiB transpose owners, and the
8192 MiB allocator floor. The generic recipe is unchanged. Bind the dedicated
recipe's actual SHA-256 and in-container path in both the immutable image
labels and launch provenance. The driver hashes the selected recipe bytes,
merges the profile selectors, and verifies the complete live environment.
Changing only the label name does not establish the intended recipe bytes.

Use `--scope plain --modes none --memory-floor-mib 8192`. This runs nine fresh
arms in C2/C4/C8 order, starting with C2 native-reference, and stops on the
first failed arm. It does not qualify native K5 concurrency, long-context QSA,
or performance. Keep the gate-owned selectors and diagnostics out of the
recipe. Require the bounded-owner startup census in addition to the route,
quality, ownership, cancellation and cleanup receipts before accepting this
checkpoint-specific configuration. No runtime pass is recorded by this recipe.

The direct host probe is not interchangeable with this container path: its
strict provenance function permits only its selector/diagnostic environment,
so it rejects the extra owner16 and load controls. Keep that contract intact.
A host owner16 diagnostic needs a separately reviewed controller that binds
the full effective environment. The container driver already binds the full
selected recipe and process environment.

### Plain-only scope

At this source the engine refuses `--speculative` with `--max-batch-size`
above 1 (`main_modules/qwen4_concurrency.rs`), so the default 18-arm scope
cannot pass. `--scope plain --modes none` runs the nine plain arms, seals
`matrix/none.json` exactly as before, and emits a receipt for the distinct gate
`native-concurrency-c2-c8-plain`. That receipt carries
`scope=plain`, `covered_modes=["none"]`, `not_covered_modes=["native-mtp-k5"]`
and `native_k5=NOT_COVERED`. The release wrapper's gate set does not contain
the plain gate name, so this receipt can never satisfy
`native-concurrency-c2-c8`. The default scope, its modes and its receipt bytes
are unchanged. `--scope plain` with any other `--modes` value is refused.

### Repeat diagnostic and shipped-PTX census

Two read-only companions sit beside the gate. Neither sets
`release_qualified`.

- `repro-kit/qwen4_native_concurrency_repeat.py run` drives one fresh
  direct-binary process per width (same launch contract as the probe, plus
  `ATLAS_LOGITS_FNV=1` and `ATLAS_QWEN4_NATIVE_BATCH_RECEIPT=1`). It runs two
  isolated C1 passes, then at least 20 identical waves, and keeps every raw
  log window. `... export --run DIR SHA` (once per width 2, 4, 8) recomputes
  per-slot token-ID, finish and logits-chain identity against C1 from the raw
  files and writes one `flux-qwen4-native-concurrency-repeat-export/1` record:
  identity rates, per-slot and per-wave divergence counts, a first-divergence
  histogram and the first divergences with token windows. Per the batching
  policy, bit-exact identity is diagnostic data and never gates; the record
  always carries `gating=false` and `release_qualified=false`, and the
  qualification lane owns the accuracy and contamination verdict. Only an
  evidence gap (hash, census, route collapse, slot instability, process
  replacement) makes the record `DATA_INCOMPLETE`. The schema is
  `EXPORT_FIELDS`/`WIDTH_FIELDS` and the check is `validate_export`, both in
  `repro-kit/qwen4_native_concurrency_repeat_data.py`. Two waves miss a
  one-in-nineteen flip at C2 with probability 0.81; twenty waves, 0.11.
- `repro-kit/qwen4_native_gdn_ptx_census.py --binary <ELF> --source-root <tree>`
  streams the shipped ELF for the embedded `gated_delta_rule_decode_f32_multi_seq`
  PTX entry and reports `FIX_PRESENT` only when exactly one clamp-free twin is
  embedded and the tree proves the Flash-Next override is the sole definer of
  `ATLAS_GDN_MULTI_SEQ_NO_STATE_NORM`.

### Offline validation

The repaired driver suite passes 38/38 tests. Neighboring source/proof suites
also pass: classifier 6/6, probe 21/21, matrix 37/37, concurrency contract 8/8,
and live-process broker 22/22. Scoped classifier `rustfmt --check` and
`git diff --check` pass. These results do not execute Rust or qualify a GPU,
engine, server, or container.

Run the driver and matrix suites sequentially. The matrix suite deliberately
rewrites and restores pinned source paths for hostile provenance tests;
overlapping it with the driver suite correctly trips source-identity guards.

## Limitations

- The arm receipts keep the sealer's verbatim limitation list, including
  "direct-binary endpoint evidence"; container scope is recorded in the
  validator and the gate receipt (`execution_scope`), not inside the sealed
  arm receipts, because the sealer requires that list exactly.
- The sealed matrix covers plain mode only; the K5 matrix is driver-validated.
- The native-K5 C2+ engine admission limitation remains a release blocker.
  A plain-scope PASS does not change that.
- Policy (2026-10-23): batching is qualified by accuracy panels plus
  contamination checks, and bit-exact identity against C1 is diagnostic. The
  probe and matrix sealer used by this gate still fail an arm on any token,
  logits or state mismatch against C1. Re-scoping that sealed verdict is not
  part of this change and needs separate approval; until then a PASS here is
  stricter than the policy requires, and a FAIL here is not by itself a
  batching disqualification.
- Same-prompt waves are token-level evidence (identical session hashes).
- Timing rows are diagnostic, not throughput claims: the diagnostics add
  hashing and logging overhead to every step.
- The driver is one standalone file by the same necessity as the accuracy
  drivers: sealed execution cannot locate sibling modules.
