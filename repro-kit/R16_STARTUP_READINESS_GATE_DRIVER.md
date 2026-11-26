# Startup-readiness release gate driver

`r15-gates/startup-readiness` produces the `startup-readiness` evidence that
`r15_exact_image_release.py` `_validate_startup_evidence` consumes. It starts
one fresh in-container server per mode, `none` first and then
`native-mtp-k5`, and records real request and process evidence from each. It
never accepts a PASS flag and never synthesizes a missing proof.

## Contract

- Invocation is the memory-floor contract plus `--process-proof-fd
  @PROCESS_PROOF_FD@`, which the suite requires exactly once for this gate.
  See the entry in `examples/r15-container-suite-plan.example.json`.
- The lifecycle is the pinned `r15-gates/mmlu-pro` (`LIFECYCLE_SHA256`). Its
  `server()` launches, claims the PID by pidfd and executable hash, waits for
  `/health`, and calls `verify_live_server` before and after the arm. The
  sealed parent plan and the suite pin are verified as in `memory-floor`.
- For each mode, the driver sends `flux-r15-live-process-request/1` over the
  sequenced-packet channel. The suite re-verifies the live process and
  returns its proof, which must equal the lifecycle proof. After both modes
  the channel is closed, which tells the suite that both proofs were sent.

## Evidence per fresh process

| Validator field | Derived from |
|---|---|
| `ready_http_status` | a fresh `GET /health` after the readiness proof; the body must be `{"status":"ready","model":<alias>}` |
| `model_identity_match` | that `/health` body, plus `GET /v1/models` listing exactly one entry with `id` = alias, `object` = `model`, `owned_by` = `atlas-spark` |
| `checkpoint_identity_match` | the live `/proc/PID/root/models` has the same (dev, ino) as the read-only `/models` bind source, the same `config.json` bytes, and the same checkpoint in both modes |
| `plain_scheduler_ready` | exactly one `Atlas Spark starting...`, `Scheduler started (batched mode, max_batch=1, ...)` and `Listening on 0.0.0.0:<container port>` anchor, with the startup anchor first |
| `k5_scheduler_ready` | the mode-exact speculative scheduler state: `none` has `mtp=false` and no admission line; `native-mtp-k5` has `mtp=true`, `num_drafts=4` and exactly one `QWEN4_SPECULATION_ADMISSION_ENGAGED mode=native-only` |
| `rootfs_read_only` | `HostConfig.ReadonlyRootfs` (lifecycle) and the live process's own `/` mount being `ro` |
| `process_executable_sha256` | the brokered proof's `executable_sha256`, which must equal the image binary hash in both modes |

Any NVRM, Xid, CUDA fatal/error, illegal-access, out-of-memory, OOM-kill,
panic or assertion text in a startup window or in the complete engine log is a
STOP. Distinct (PID, start ticks) are required for the two modes.

## Artifacts

All are mode 0400, below `startup/` in the runtime directory:

- `engine.log`: the complete server output of both modes.
- `responses.jsonl`: the raw `/health` and `/v1/models` responses with statuses and timestamps.
- `validator.json`: a `flux-r15-gate-validator/1` document.

`raw` carries the validator's exact `attempts`, `rootfs_read_only` and
`process_executable_sha256`, plus `observations`. Each observation holds:

- the brokered proof;
- launch, ready, health and models timestamps;
- every lifecycle readiness probe, with its outcome and response hash;
- the log segment offset and hash, and the anchor census;
- the checkpoint identity.

`raw` also records the artifact hashes, the field semantics, the suite plan
hash, the lifecycle pin and the checkpoint id and revision.

## Offline validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 5 prlimit --as=94371840 \
  python3 -m pytest -q -p no:cacheprovider \
  repro-kit/tests/test_r16_startup_readiness_gate_driver.py
```

Docker, HTTP, `/proc` and the server are mocked. The tests feed a real-shaped
run through the driver, check that the outer validator accepts it, and then
check that the validator rejects every tampered or missing piece. They also
check that every driver fault leaves no PASS receipt. They do not qualify an
image.
