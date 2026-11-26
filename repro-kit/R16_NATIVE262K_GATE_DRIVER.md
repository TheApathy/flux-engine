# Operational native-262K release driver

`r15-gates/native-262k` implements the immutable-container native-262K gate.
It does not ingest an earlier qualification receipt, accept saved measurements,
or turn context capacity alone into a PASS. Offline mock tests validate the
driver, not the engine, checkpoint, container, accuracy, or throughput.

## Launch contract

The sealed R15 suite launches this standalone executable with:

```text
--container-id @CONTAINER_ID@
--image-id @IMAGE_ID@
--image-digest @IMAGE_DIGEST@
--base-url @BASE_URL@
--run-id @RUN_ID@
--receipt-fd @GATE_RECEIPT_FD@
--raw-log-fd @RAW_LOG_FD@
--runtime-dir <the exact private suite runtime directory>
--suite-program <the tracked r15-container-suite.py>
--suite-sha256 <the exact launching suite program hash>
--lifecycle-program <the tracked r15-gates/mmlu-pro>
--endpoint-program <the tracked qwen4_endpoint_admission.py>
--container-port 8888
```

The optional port defaults to 8888. The URL must be explicit HTTP IPv4 loopback.
The inherited raw-log descriptor is never written. Server stdout and stderr
remain inherited from the suite's regular-file captures, not `/dev/null`.

The source-pinned lifecycle helper is the exact `mmlu-pro` implementation with
the mandatory QSA launch repair and finite managed accuracy deadline, SHA-256
`dfae597a00c668ee5e3b2f02042c39df6035fa48cbb887cea53a3cb50a41b032`.
This gate uses its own server launch policy rather than the accuracy helper's
server context manager; the updated pin does not change that launch policy.
The endpoint helper is pinned to SHA-256
`81d336c6bc477b6efafeb100e8e72b78c8e8b7cffc54c8880a77227c3fa84832`.
These are literal code pins, not caller-supplied helper hashes. Every loader
hashes and executes the same bytes and refuses changed files. The lifecycle
helper verifies the actual fully sealed suite executable in the launching
parent before the driver loads the source copy of that exact suite. Direct
standalone invocation is not a release path. No sibling is imported through a
mutable Python search path.

## Actual execution

One fresh target-only C1 server runs inside the already-owned immutable image.
Its explicit launch selects the proven binary and `/models`, `--no-tui`, the
Flash-Next kernel target, physical QSA, BF16 KV, a 262,144-token capacity,
16,384-token prefill chunks, block size 16, no prefix caching, zero SSM cache
slots, and zero swap. The real speculation kill switch is set to 1.

The existing live broker verifies executable, recipe, checkpoint mount,
namespace, cgroup, PID/start time, read-only paths and PID-owned listener before
and after the requests. The driver also checks the actual `/proc` arguments,
effective QSA/BF16-related environment, and mounted tokenizer bytes twice. It
requires the calibrated tokenizer SHA-256
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
No YaRN/RoPE extension, truncation, quantized QSA operands, cache or swap can
satisfy this gate.

The workload preserves `qwen4_native_262k_qualify.py`'s exact component texts,
u32 token assembly, three passkeys and three calibrated prompt hashes:

- Retrieval at 258,000, 261,888 and 262,080 prompt tokens, with each label
  mapped exactly once to its complete passkey in the actual returned text.
  Wrong extensions, prefixes, swapped labels, duplicates and extra prose fail.
- An exact greedy repeat at 262,080, comparing full token IDs, text, finish and
  completion counts.
- A forced 262,080 + 64 completion ending at exactly 262,144 total tokens, with
  the same exact labeled passkeys retained and a complete 64-ID output.
- A forced 262,143 + 1 completion reaching the inclusive native boundary, whose
  scheduler-owned token equals the 262,080 + 1 control.
- A correlated 262,144 + 1 rejection with the exact HTTP 400 context error,
  exactly one echoed request ID, unchanged scheduler prefill counters and no
  dispatch/runtime-request markers in the measured log suffix.
- Post-rejection health and a fresh 262,080 + 1 canary with token identity and
  exact prefill token/chunk counter advances.

Every successful completion requires the declared per-process model identity,
a unique response ID, exact integer usage, all scheduler-owned token IDs,
explicit zero cached tokens, a nonblank returned answer, the expected finish
and a complete empty report-only `atlas.interventions.v1` ledger. Semantic
retrieval requires `stop`; forced fills require `length` at the exact budget.
Missing completion-ledger support fails closed, not as an inferred empty ledger.

After graceful PID-bound server cleanup, the full captured streams must contain
the physical QSA marker and parsed complete BF16 terminal chunk receipts for all
four accepted prompt lengths. Each receipt requires exact start/rows/end
geometry, 16,384-token chunks, tile rows 10, and the exact BF16 core identifier.
Partial numeric or core-name matches cannot qualify. The final whole-stream
dispatch census must match all eight accepted requests and contain no overflow
dispatch. Late dispatch, quantized sparse-QSA, fatal GPU/OOM/driver markers and
forced cleanup fail.
The complete-log scan includes CUDA enum/camel-case allocation and illegal
address errors, separator variants, host OOM/kill events and allocator warnings.

Before server entry, the driver captures one actual current-boot kernel journal
record and its cursor, checks its boot ID against `/proc`, then replays from
that same cursor after server cleanup. The original baseline record must still
be present unchanged, every cursor must be unique, and all records plus the
current host boot must retain the original boot identity. Unreadable/incomplete
journals, lost cursors, malformed events, boot changes and unsafe measured
kernel messages fail closed. The retained baseline is not counted as a new
event. Kernel evidence remains separate from server stdout/stderr and does not
claim to cover other release gates outside this native arm.
Kernel hashes bind the exact baseline/full journal bytes parsed and measured
bytes scanned. Each artifact is re-read before sealing and must still equal
those consumed bytes, rather than merely hashing whatever a later read finds.

## Artifacts and remaining qualification

Artifacts live under private `native262k/` with exclusive no-follow creation:
`responses.jsonl`, complete `server-stream-N.log` captures, `kernel-start.jsonl`,
`kernel-full.jsonl`, `kernel-measured.log`, and `validator.json`.
Files are mode 0400. The response journal retains exact request bodies, raw
response bytes as base64, headers, statuses and byte hashes, including rejected
responses. All measured HTTP observations are journaled before interpretation.
Partial private journals and server captures remain on failure, with no PASS
receipt. The canonical suite raw log remains owned by the suite.

The emitted evidence and validator conform to the existing strict native-262K
release schema. Extra validator observations bind complete cases, process
proofs, tokenizer/controls, journals, stream hashes, overflow counters and the
post-rejection canary. A distinct kernel-window observation binds the actual
baseline cursor, boot identity, measured event count, monotonic window and all
three kernel artifact hashes. These do not claim native batching, native-MTP
accuracy, MMLU-Pro/GSM8K quality, or any throughput result.

A real release still requires a clean integrated source closure, the exact
source-built immutable image, an audited duplicate-free BF16 recipe, truthful
raw-completion intervention reporting, sufficient local `/proc`/pidfd access
and read permission for the complete current-boot kernel journal,
explicit resource claims, and an actual successful outer release run. Neither
these source changes nor the mock suite is that live receipt.

```sh
python3 -m unittest discover -s repro-kit/tests \
  -p test_r16_native262k_gate_driver.py -v
```
