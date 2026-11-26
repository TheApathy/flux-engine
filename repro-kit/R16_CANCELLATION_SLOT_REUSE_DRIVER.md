# R16 cancellation-slot-reuse gate driver

`repro-kit/r15-gates/cancellation-slot-reuse` produces the
`cancellation-slot-reuse` evidence that `_validate_cancellation_evidence` in
`r15_exact_image_release.py` recomputes. It is a live, in-container driver.
Offline tests mock every server and process binding and do not qualify an
image.

## What runs

For each mode, `none` first and then `native-mtp-k5`, the driver launches one
fresh C8 `native-reference` process. It uses the pinned
`native-concurrency-c2-c8` driver, loaded from exact bytes, for container
inspection, launch, the live process proof, provenance, log binding, memory
sampling, the forbidden-marker census and pidfd cleanup. Against that
endpoint it runs these phases:

1. **C1 references.** Each of the eight prompts, and the replacement prompt,
   runs alone in a fresh slot. These runs record the token ids, full logits
   and terminal state.
2. **Churn.** Eight streaming requests are admitted in order. The client
   disconnects the final admission, which holds the highest live slot, after
   eight content chunks. The replacement starts only after that response has
   closed. The probe's `validate_full_churn` must pass on the engine
   receipts:
   - the cancelled target-state receipt precedes the replacement's first
     frame;
   - the replacement's first frame uses the retired physical slot;
   - that slot is the highest in the full C8 frame;
   - the full logits and terminal state of the survivors and the replacement
     equal C1.
3. **Post-drain reuse.** The replacement prompt runs again alone. It must
   produce no batch frame and must match C1 exactly.
4. **Release census.** The `atlas_requests_active` gauge must reach 0 at
   three points: before the churn, after the drain and after the reuse.

The driver recomputes every census boolean itself from the observed rows.
Survivors and the replacement must equal their fresh-slot C1 references in
token ids, sha, finish reason and usage. Each completion exchange records:
- the request-body SHA-256 and size;
- the SHA-256 and size of the streamed response bytes;
- whether `[DONE]` arrived, so the cancelled stream is proven to be a
  mid-stream client disconnect;
- the TTFT and elapsed time.

## Inputs

These are every flag of `native-concurrency-c2-c8`; see its README. Two flags
are added:

```text
--concurrency-driver-program <tracked r15-gates/native-concurrency/native-concurrency-c2-c8>
--concurrency-driver-sha256  <its SHA-256>
```

The parent must be the sealed suite, which is checked by the pinned driver's
parser. `--ladder` is refused. A subset of `--modes` never produces a PASS.

## Artifacts

All artifacts are below `cancellation/cancellation-slot-reuse/` in the
runtime directory:
- `arms/<mode>-native-reference-c8.{manifest.json,server.log,json}`;
- `validator.json`, the `flux-r15-gate-validator/1` document;
- `failure-<hex>.json`, written only on failure.

The `raw` object carries the wrapper census:
- `concurrency`, `slots`, `replacement_requests` and `replacement_survivor`;
- `plain_and_k5`, `exact_survivor_identity`, `final_highest_slot_reused` and
  `post_drain_clean`;
- `mode_observations` with the per-request exchange rows, plus the process
  bindings and provenance.

The receipt evidence is the wrapper's exact 10-key contract.

## Known limit at this base

The native-concurrency README records that the engine refuses the native
batched path while a proposer is active. The `native-mtp-k5` C8 arm is
therefore expected to fail closed until that engine path exists. In that case
no PASS receipt is written, and the plain arm's artifacts stay on disk.

## Offline validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 5 prlimit --as=94371840 \
  python3 -m pytest -q -p no:cacheprovider \
  repro-kit/tests/test_r16_cancellation_slot_reuse_driver.py
```
