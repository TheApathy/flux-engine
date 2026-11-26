# R16 target-state receipt plain-control repair

Base: `3919ff6e4d223edbcce162b4046a1c306b43984e`.

## Scope and invariant

The strict release process proof requires effective
`ATLAS_DISABLE_SPECULATION=1` for a Plain control. The target-only concurrency
launch also omits speculative proposers. Previously, arming target-state
capture for that legitimate combination failed because the emitter rejected
`active_proposer=None` plus `speculation_forced_disabled=true`.

`classify_capture_mode` is now the production classifier used directly by
`capture_enabled`. With no installed proposer, both immutable kill-switch
settings truthfully report the existing `unloaded-target-only` mode,
`ordinary-target-only` scheduler path, and `proposer_loaded=false`. The kill
switch changes scheduling, not proposer ownership. This does not relabel a
loaded proposer as unloaded.

The other cases remain unchanged:

- Native Qwen4 proposer plus kill switch: `loaded-serial`, ordinary target
  scheduling, `proposer_loaded=true`.
- Native Qwen4 proposer without kill switch: `engaged-native-mtp`, native-MTP
  scheduling, `proposer_loaded=true`.
- DFlash and other proposer kinds: rejected with either control setting.

The shared process-proof predicate accepts both unloaded Plain and actual
loaded-serial Plain controls. Individual gate launch contracts and the
emitted mode distinguish them. Neither that proof predicate nor a loader,
scheduler, or consumer guard changes here.

## Preserved evidence contract

Receipt schema 1, both v1 hash domains, exact target-fed token ownership,
GDN/PLE/logical BF16 KV coverage and census, capture limits, and emitted log
grammar are unchanged. The secondary-stream commit wait remains exclusive to
`engaged-native-mtp`; both target-only modes drain the default producer
stream. The new classification reads only the existing immutable model
control and actual proposer kind, not launcher labels or mutable environment.

No schema extension is necessary: these fields describe the same physical
proposer/scheduler state as before. The separate live process proof remains
responsible for proving the exact kill-switch environment. A future need to
encode the control spelling inside a standalone state receipt would require
an explicit metadata and consumer-contract change, not an inferred meaning
for the existing mode.

## Validation and limits

Rust regression definitions call the exact production classifier:

- `capture_mode_preserves_plain_loaded_serial_and_engaged_identity`: all four
  admitted proposer/control combinations.
- `capture_mode_rejects_foreign_proposers_with_either_control`: both foreign
  kinds under both control settings.
- `only_engaged_native_capture_uses_the_secondary_commit_wait`: the four
  admitted combinations retain the existing wait decision.

The focused Python suite checks source wiring, mode separation, secondary
wait exclusivity, and selected unchanged state/provenance guards. It also
executes the unchanged live-mode proof helper to require the real kill
switch for both unloaded and loaded-serial Plain launches. These are
source/proof-helper checks, not execution of Rust or a GPU receipt.

Executed offline checks:

- `test_r16_target_state_plain_control.py`: 6/6 pass.
- Existing `test_r16_live_process_broker.py`: 22/22 pass.
- Existing `test_qwen4_native_concurrency_probe.py`: 21/21 pass.
- Scoped `rustfmt --edition 2024 --check` and `git diff --check`: pass.

The three new production-classifier regressions subsequently compiled and
passed within the five-test target-state suite at integrated `9e3c69f9`.

This repair does not qualify Plain, loaded-serial, native K5, native
concurrency, or a container. The S4 consumer now independently reconstructs
fed-token digests and compares complete target-state receipts against Plain
controls across paired requests, continuations, eager verification,
rollback/slot reuse, and endpoint controls. Its comparison and lifecycle
paths pass source/offline review, but fresh sealed runtime qualification
remains on HOLD.

Concurrent K5 still requires its independent implementation and runtime
qualification. The isolated classifier lane performed no compiler, GPU,
model, server, or container action. The later root CPU checkpoint is
documented separately.
