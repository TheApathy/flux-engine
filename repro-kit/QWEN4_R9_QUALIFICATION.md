# Qwen3.8 Flash-Next R15 qualification

The historical filename is retained so existing links remain valid. Every
release command and checked-in launch example below is pinned to
`v0.9.13-r15`; the generic `r9_source_to_binary.py` utility still supports
older releases when their version is supplied explicitly.

This runbook defines the release gates for native 262K context, native MTP, native concurrency, and container quality. A compiler pass is not a runtime pass. A host binary pass is not a container pass. A logical multi-request scheduler is not native GPU batching.

## Release gates

R15 is releasable only when all of these gates pass against the same source revision, binary hash, QAD5500 checkpoint inventory, recipe hash, and immutable image digest.

1. Source and CPU gates pass.
2. The direct binary passes the native 262K qualification.
3. Native MTP K=5 matches target-only greedy output across the Issue #13 boundary matrix.
4. Native C2 and C4 execute one batched model forward with isolated KV, SSM, PLE, and logits state.
5. The supported concurrency ladder passes correctness and load tests. Do not advertise a rung that falls back to repeated C1 forwards.
6. The exact container image repeats the binary gates.
7. Lil Bench completes all 1,319 GSM8K items and all 1,000 pinned MMLU-Pro items.
8. The release image and immutable version tag are pushed only after every required receipt is sealed.

## Native 262K binary qualification

The 262K gate uses BF16 KV, target-only decode, C1, 16,384-token prefill chunks, physical paged QSA, no YaRN, no prefix cache, no swap, and no context truncation.

Start from a new detached worktree at the revision being qualified. The source
worktree must remain clean, including untracked files. Create the complete
engine and pinned FlashInfer build-input bundle outside that worktree:

```bash
python3 repro-kit/r9_source_to_binary.py bundle \
  --source-root /path/to/clean-r15-source \
  --source-revision <40-hex-r15-commit> \
  --source-tree <40-hex-r15-tree> \
  --flashinfer-home /path/to/verified-flashinfer-home \
  --output-dir /results/r15-source \
  --release-version v0.9.13-r15
```

Build into a new target directory. The builder rechecks the clean commit and
tree before and after compilation, verifies every FlashInfer input against the
tracked manifest, and emits `build-receipt.json` beside the sealed binary:

```bash
scripts/build-qwen38-dual-profiled.sh \
  --source /path/to/clean-r15-source \
  --commit <40-hex-r15-commit> \
  --tree <40-hex-r15-tree> \
  --dflash-generation v15 \
  --flashinfer-home /path/to/verified-flashinfer-home \
  --artifact-root /results/r15-builds \
  --target-dir /results/r15-target
```

Bind that exact build receipt and binary to the reopened source archive. Do not
write a source-to-binary receipt by hand:

```bash
python3 repro-kit/r9_source_to_binary.py receipt \
  --source-root /path/to/clean-r15-source \
  --source-revision <40-hex-r15-commit> \
  --source-tree <40-hex-r15-tree> \
  --source-manifest /results/r15-source/SOURCE_BUNDLE_MANIFEST.json \
  --source-bundle /results/r15-source/flux-engine-source-v0.9.13-r15.tar.gz \
  --build-receipt /results/r15-builds/<sealed-build>/build-receipt.json \
  --binary /results/r15-builds/<sealed-build>/<sealed-binary> \
  --output /results/SOURCE_TO_BINARY_RECEIPT.json \
  --release-version v0.9.13-r15
```

The live 262K harness reopens all six provenance inputs: the clean Git
worktree, source manifest, source archive, build receipt, binary, and final
source-to-binary receipt. A digest-only or self-consistent hand-written receipt
is rejected. This is local deterministic provenance, not a signed remote-build
attestation. It also requires `tokenizer.json` SHA-256
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
That pin is part of the correctness oracle, not merely checkpoint metadata.

Create a content inventory outside the checkpoint directory:

```bash
python3 repro-kit/qwen4_checkpoint_inventory.py \
  --checkpoint-root /models/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
  --checkpoint-id JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
  --checkpoint-revision <immutable-checkpoint-revision> \
  --output /results/qad5500-checkpoint-inventory.json
```

Copy [`examples/qwen4-native-262k-binary-launch.example.json`](examples/qwen4-native-262k-binary-launch.example.json) to `/results`, then replace every zero hash with the value from the R15 build and checkpoint receipts. Keep `provenance.release_version` exactly `v0.9.13-r15`.

Review the exact plan without launching a server:

```bash
python3 repro-kit/qwen4_native_262k_qualify.py \
  --manifest /results/qwen4-native-262k-binary-launch.json \
  --dry-run
```

Run the fresh-process gate:

```bash
python3 repro-kit/qwen4_native_262k_qualify.py \
  --manifest /results/qwen4-native-262k-binary-launch.json \
  --receipt /results/qwen4-native-262k-binary-receipt.json \
  --timeout 3600
```

The gate proves all of the following:

- semantic retrieval of all three fixed passkeys through the calibrated
  QAD5500 Hybrid manual no-thinking chat frame at 258,000, 261,888, and
  262,080 prompt tokens
- the prior-comparable prompt IDs at those three lengths have exact little-endian
  u32 SHA-256 values `05d689a3149f158b2af5047421a152f03cc93c78f5377712801bf0d0f50ca2ad`,
  `9692378175dbb8126e65d7e4ee022d7150ffe8a06608fb9c936159c05ba4e806`,
  and `a664d90f8924cc85fd3862c7617b9f9894a4c22b031757c93fd22d8aac00c3c0`
- deterministic repeated output at 262,080 prompt tokens
- a forced 64-token decode that reaches 262,144 total tokens
- a 262,143-token prompt plus one generated token whose scheduler-owned token
  ID is identical to the one-token 262,080 control
- rejection of a 262,144-token prompt plus one requested output token
- an echoed request ID, unchanged scheduler-owned prefill-progress counters,
  and no request-log dispatch for that rejection
- a post-rejection health check and a fresh 262,080 plus one completion whose
  exact counter deltas are sealed
- a final whole-log dispatch census with zero 262,144-token dispatches
- BF16 persistent KV and BF16 sparse-QSA operands
- exact sparse-QSA endpoint receipts at 258,000, 261,888, 262,080, and 262,143
- no inherited unrecorded `ATLAS_` or `FLUX_` selectors
- a source receipt that binds the binary hash to a reopened clean commit and
  tree, exact source archive, pinned FlashInfer inputs, and build receipt
- a full checkpoint inventory that binds the selected weight path

Retrieval passes when every fixed passkey value is present in the generated
text. The receipt records a canonical exact-format comparison separately, but
formatting is advisory and cannot turn a semantic success into a failure. This
avoids rejecting a correct answer solely because it used commas or omitted the
requested labels.

The harness never discards a response before scoring it. Each HTTP body is
stored losslessly as base64, with its parsed JSON and SHA-256, in the PASS
receipt. If any launch, accounting, semantic, determinism, boundary, overflow,
or log gate fails, the same evidence collected so far is sealed at
`<receipt>.failure.json`; the server log remains at `<receipt>.server.log`.
Neither artifact weakens the gate, and an existing failure artifact blocks an
accidental overwrite just like an existing PASS receipt.

The receipt says `direct-binary-not-container`. It must never be used as container evidence.

## Exact-image container gate

Build an immutable R15 image from a clean committed source tree. Record the image ID, repo digest, root filesystem layers, in-container binary hash, source-to-binary receipt, recipe hash, and checkpoint mount inventory. Run the same 262K workload through the container endpoint. Use the lifecycle-v3 runner so an interrupted Docker launch cannot leave a running container after the GB10 lock is released.

Do not reuse an R8 performance receipt for R15. Do not overwrite the stable tag before the immutable R15 tag passes.

## Native MTP Issue #13 gate

Compare `FLUX_SPEC_MODE=none` with native K=5 at temperature zero. Use fresh
processes and the same in-memory token-ID request bodies. Compare scheduler-owned
token IDs, not normalized text. The v4 harness owns three sequential processes:
plain, normal K=5, and a forced-eager K=5 boundary and churn control.

Required reported-length analogues are 2,017, 2,112, 7,336, 7,710, 30,647,
114,755, 122,413, and 238,938 tokens. The fixed ladder also includes 4,096,
8,192, 32,768, 65,536, and 131,072 tokens. Test every preverify length from
2,043 through 2,050 so K=5 crosses both the verifier endpoint and the 2,048
token sparse-QSA boundary. The ordinary matrix requests 96 output tokens and
repeats every row twice. Endpoint semantics and endpoint filling are deliberately
separate requests. The stopped semantic row sends 262,080 prompt tokens with
`max_tokens=64`, `ignore_eos=false`, and `stop=["\n"]`; it must return exactly
one canonical structured answer with no trailing or contradictory assignment.
The fill-identity row sends the same length with `max_tokens=64`,
`ignore_eos=true`, `stop=[]`, and `return_token_ids=true`. It passes only at
exactly 262,144 total tokens, with 64 scheduler token IDs and
`finish_reason=length`, and with exact plain/K5 text, usage, and token-ID
identity. Forced fill text is not scored as a semantic answer. Every arm must
use `--max-seq-len 262144`.

The admission probes run on the plain arm only. They are capacity and recovery
checks, not speculative-decode checks. The accepted probe sends 262,143 prompt
tokens plus one forced token and must finish at exactly 262,144 total tokens.
The rejected probe sends 262,144 prompt tokens plus one requested token. It must
receive the exact HTTP 400 context-length error. Between `/metrics` snapshots,
the scheduler-owned prefill-progress chunk and token counters must remain
unchanged. The gate does not require lazy chat/TUI prompt, generation, or decode
counters, which need not be registered by `/v1/completions`. A unique supplied
`x-request-id` must be echoed exactly once
on the HTTP 400 response, positively binding that response to the probe. Its
request-scoped log delta must have no scheduler dispatch marker or runtime
request ID. A `/health` request and a fresh 2,017-token completion must then pass.
The completion's response geometry must pass and its scheduler-owned prefill
chunk and token counters must advance by the exact expected amounts. After
process shutdown, a whole-log census must contain
every expected accepted dispatch and no 262,144-token dispatch, closing a late
asynchronous-dispatch false pass. These admission probes remain plain-arm only.

Pass requires identical greedy token IDs, no state drift, no cache reuse, and no
multi-row sparse-QSA verify above the dense endpoint. The forced-eager arm reruns
the exact canonical 2,043 through 2,050 bodies with both
`ATLAS_QWEN4_VERIFY_GRAPH=0` and `ATLAS_DFLASH_DEBUG_NO_GRAPH=1`; any verify
graph-capture marker fails that arm. This is a forced-eager correctness control,
not a measured graph-on/off performance comparison. Performance is reported only
after parity passes.

Every required K=5 request owns a request-scoped log delta. Each delta must bind
exactly one prefill dispatch, at least one `MTP_ACCEPT frame`, at least one
completed `SPEC_CYCLE_V2` record, and exactly one runtime request ID. The frame
and completed record must each begin at the request's first speculative position,
which is prompt length plus the first prefill-produced token. A global startup
marker or a frame from another request cannot satisfy this gate. The
normal K=5 arm remains graph-on. Only the forced-eager control disables graphs.
The normal arm must contain at least one successful verify CUDA-graph
capture/use marker; the configured environment alone is not engagement proof.
The forced-eager arm is also a clean follow-up telemetry hook. Its manifest pins
C1, world size 1, TP1, EP1, and HSS off, and its arm-local environment contains
only native K5, graph-disable controls, and
`ATLAS_QWEN4_QSA_ROUTE_TELEMETRY=1`. Every request-scoped delta must contain a
strictly contiguous frame sequence matching its `MTP_ACCEPT` geometries. Each
frame must carry exactly one executed-route receipt for all twelve ordered QSA
layer ordinals, with exact per-row route counts and monotone cumulative totals.
Missing, extra, malformed, cross-request, or graph-on-arm route receipts fail
closed. These receipts prove successful host dispatch, not GPU completion,
correctness, latency, or throughput; correctness still comes from the paired
token and semantic gates above.

The common environment must pin `ATLAS_QWEN4_QSA_VERIFY_ROWS=1`,
`ATLAS_QWEN4_VERIFY_GRAPH=1`, `ATLAS_QWEN4_QSA_GRAPH_SAFE=1`,
`ATLAS_DFLASH_ASYNC=0`, `ATLAS_DFLASH_SPEC_CYCLE_V2=1`, and
`ATLAS_MTP_ACCEPT_LOG=1`. Keep `ATLAS_DFLASH_DEBUG_NO_GRAPH` and
`ATLAS_QWEN4_QSA_ROUTE_TELEMETRY` out of the common environment; they belong
only to the forced-eager arm. `MTP_ACCEPT frame` proves
the proposer route engaged but is emitted before commit. Rollback proof therefore
comes from a completed `SPEC_CYCLE_V2` record with schema 2, K=5, gamma=4,
`accepted<4`, C=1, async off, and no added synchronizations. The record is emitted
only after commit, proposer-state, and finalize phases.

The churn control sends a 2,017-token canary, a 2,050-token trigger, then the
identical 2,017-token canary. It requires distinct completed runtime request IDs,
a completed partial-accept rollback on the intervening trigger, and exact
semantic and scheduler-token-ID identity between the canaries. With
`--max-batch-size 1`, `--max-num-seqs 1`, prefix caching disabled, zero cached
tokens, and sequential blocking requests, the final canary structurally reuses
the only model-owned slot. This is not a direct numeric `slot=0` telemetry claim.

The launch manifest, HTTP JSON, checkpoint inventory, and source-to-binary
receipt use a strict loader that rejects duplicate keys and non-finite numbers.
The R15 manifest must carry the complete clean-source closure: commit, tree,
source root, source archive and manifest, build receipt, final binary, and the
generated source-to-binary receipt with every SHA-256. The MTP gate reopens
that closure with the same verifier as the native-262K gate; a shortened or
self-attested PASS receipt is not release evidence.
The runner snapshots the raw and canonical launch manifest, reopens and verifies
the manifest and every bound artifact before each arm, and repeats that check
before the final seal. A changed manifest, recipe, binary, checkpoint inventory,
or source receipt fails closed rather than producing a PASS artifact.

Copy [`examples/qwen4-mtp-long-context-pair-launch.example.json`](examples/qwen4-mtp-long-context-pair-launch.example.json) outside the source tree. Replace every placeholder path, revision, and zero hash with the exact R15 build and checkpoint values. Keep `provenance.release_version` exactly `v0.9.13-r15`. The harness owns both loopback server processes and constructs one canonical request bundle in memory for both arms.

Review the launch and request contract without starting any process:

```bash
python3 repro-kit/qwen4_mtp_long_context_qualify.py \
  --manifest /results/qwen4-mtp-long-context-pair-launch.json \
  --dry-run
```

Run and seal the direct-binary receipt:

```bash
python3 repro-kit/qwen4_mtp_long_context_qualify.py \
  --manifest /results/qwen4-mtp-long-context-pair-launch.json \
  --receipt /results/qwen4-mtp-long-context-pair-receipt.json \
  --timeout 7200
```

This focused gate uses deterministic exact-token known-answer analogues for the
full matrix above. Each receipt explicitly says the reporter's original prompt
bytes were unavailable, so matching a disclosed length is never represented as
a byte-for-byte reproduction. A dry run or the offline unit suite proves only
the fail-closed harness contract. It does not prove model accuracy, rollback, or
262,144-token endpoint execution. A runtime PASS additionally requires correct
plain and K=5 answers, the exact pair comparisons, native route engagement, the
accepted and rejected endpoint admission proofs, post-rejection recovery,
completed rollback and sole-slot-reuse evidence, and no fatal server-log marker.
All JSON inputs reject duplicate keys, JSON non-finite constants, and finite
syntax whose exponent overflows to a non-finite host float. Integer response
fields use exact built-in integer checks, so booleans cannot satisfy index,
usage, cached-token, or census fields.
The receipt is direct-binary evidence only. Repeat the same gate against the
immutable container before updating or closing Issue #13.

Suggested issue update after the immutable image passes:

> We reproduced the long-prompt native-MTP failure and traced it to the QSA draft and verify path. R15 changes the sparse rank walk, prevents unsafe multi-row sparse verification past the dense endpoint, and keeps native K=5 on the exact row-by-row path. We validated target-only and K=5 greedy token parity across the 2K boundary and the long-context matrix on the immutable release image. The attached receipt identifies the image digest, binary hash, checkpoint inventory, recipe, prompts, and token outputs.

Do not post this wording before the runtime matrix passes.

## Native concurrency gate

Use these terms consistently:

- Logical concurrency means multiple requests are admitted and isolated, but the model may still execute C1 once per request.
- Native concurrency means active rows share one model forward and one set of batched kernels while retaining independent request state.

The source preview admits every exact active width from C2 through C32 on short target-only BF16 decode. It covers batched attention, HC state, SSM state, PLE state, MoE execution, the final mixer, the LM head, sampling, and one full-vocabulary logits row per request. Padding rows are forbidden. KV, SSM, and PLE state aliases are rejected before the first request-owned GPU mutation. C10-C32 MoE work uses exact active-row tiles of at most nine rows within the single model traversal.

The route is the default for Qwen4 at `--max-batch-size` 2-32 (`ATLAS_QWEN4_NATIVE_BATCH` unset). `ATLAS_QWEN4_NATIVE_BATCH=0` is the serial opt-out; qualification runs set `1`, which refuses to start when a process fact rules the route out instead of producing a serial receipt. Each serialized C2+ frame carries a typed reason (`spark_model::qwen4_batch_policy::SerialReason`). The tracked matrix currently qualifies C2/C4/C8; extending that matrix is required before claiming runtime C9-C32 support. The route is eager-only, TP1/EP1, full-vocabulary, BF16 KV, and requires speculation, prefix caching, profiling, and high-speed swap to be off. It uses the exact unified MoE row kernels directly and does not use the verifier-only `ATLAS_QWEN4_K5_NATIVE_MOE` selector. The scheduler treats Qwen4's `hc_count` as its highway capability and keeps mixed prefill and physical slot compaction away from unsupported ordinary-residual paths.

Context capacity and native concurrency are independent claims. A request whose immutable prompt plus output budget can reach token 2,048 is marked QSA-required before prefill and stays on serialized C1 forwards for its entire lifetime. This preserves the complete per-request QSA side index. A server configured for 262,144 tokens can therefore admit concurrent requests, but the current native one-forward preview does not batch their long-context decode. With native batching enabled (the default), C33-C128 fails admission; with `ATLAS_QWEN4_NATIVE_BATCH=0`, the ordinary serialized path remains available.

Current implementation and evidence status:

| Active width | Execution with selector on | Evidence required before release |
|---|---|---|
| C1 | Existing single-request path | Matched control |
| C2-C32, all rows below QSA horizon | One eager target model forward | Source-contract tests exist. GPU parity, churn, memory, and throughput receipts are still required for every width |
| C2-C32, any request QSA-required | Serialized C1 forwards | Long-context correctness control |
| C33-C128 | Refused with native selector enabled; serialized with it disabled | Native orchestration is not implemented or claimed |

Compare every width C2 through C32 against repeated isolated C1 runs using identical prompts, positions, token IDs, and full logits. Use a fresh process for each width so the `QWEN4_NATIVE_BATCH_ENGAGED rows=N` marker binds the actual rung. Then test heterogeneous short sequence lengths, adversarial row order, mixed finish times, cancellation, removal from every row, non-contiguous physical slots, slot reuse, and repopulation from C2 back to C32. Require exact output parity, zero state aliases, zero serial fallback in the measured native frames, no CUDA graph capture, and no error after request-owned mutation.

C16 and C32 have source admission and exact-row tiling, but still need matched whole-model state, PLE, recurrent-pointer, arena, and kernel runtime proofs. C33-C128 require new or explicitly serial-per-row exact projection fallbacks, PLE scratch beyond 32 rows, recurrent pointer capacity beyond 32 rows, and a separate memory and scheduler qualification. Do not infer native C128 from the launcher's logical `FLUX_CONCURRENCY=128` ceiling.

Lil Bench concurrency traffic alone does not prove native batching. The server receipt must show one batched model forward for the active rows and no serial fallback.

## GSM8K and MMLU-Pro

The image bundles pinned Lil Bench v0.7.9 datasets. Network update checks are disabled.

Run each mode from its own fresh container and server process. The server reads
`FLUX_SPEC_MODE` at startup, so changing the variable in `docker exec` does not
change an already-running engine. Run the full C1 profiles against the plain
container first:

```bash
docker exec <plain-container> lil-bench accuracy gsm8k \
  --concurrency 1 \
  --model qwen38-qad5500-r15 \
  --output /results/r15-plain-gsm8k.json

docker exec <plain-container> lil-bench accuracy mmlu-pro \
  --concurrency 1 \
  --model qwen38-qad5500-r15 \
  --output /results/r15-plain-mmlu-pro.json

docker exec <k5-container> lil-bench accuracy gsm8k \
  --concurrency 1 \
  --model qwen38-qad5500-r15 \
  --output /results/r15-k5-gsm8k.json

docker exec <k5-container> lil-bench accuracy mmlu-pro \
  --concurrency 1 \
  --model qwen38-qad5500-r15 \
  --output /results/r15-k5-mmlu-pro.json
```

Start `<plain-container>` with `FLUX_SPEC_MODE=none` and `<k5-container>` with
the native K=5 recipe. The authoritative gate is full paired parity from the
same immutable image and checkpoint. Every item ID, correctness value, parsed
answer, final answer, and completion-token count must match.

The provisional absolute sanity floors are 95 percent for GSM8K and 65 percent for the pinned MMLU-Pro subset. These floors are not substitutes for the paired gate because earlier reference results used different evaluators or sampling settings.

The accuracy receipt must reject partial runs. It requires all expected unique item IDs, zero request errors, zero timeouts, zero stalls, zero truncations, exact usage rather than estimated tokens, and all 14 MMLU-Pro categories totaling 1,000 items.
The built-in GSM8K and MMLU-Pro profiles also request the report-only
intervention ledger. Every item must carry the exact
`atlas.interventions.v1` schema with an empty `events` array; a missing,
malformed, or non-empty ledger invalidates the report.

After all four Lil reports exist, copy [`examples/lil-accuracy-pair-envelope.example.json`](examples/lil-accuracy-pair-envelope.example.json) to `/results`. Bind the immutable image ID and digest, the in-container binary, source inventory, checkpoint inventory, both recipes, both server argument arrays, and all four report hashes. Then seal the authoritative paired receipt:

```bash
python3 repro-kit/lil_accuracy_qualify.py \
  --envelope /results/lil-accuracy-pair-envelope.json \
  --receipt /results/lil-accuracy-pair-receipt.json
```

The validator refuses to overwrite an existing receipt. Preserve the envelope, its referenced files, and the read-only receipt together.

## Handoff checklist

Every handoff must include:

- worktree and branch
- exact commit and tree hashes
- changed file list
- build and test commands with exit status
- image ID and digest when a container exists
- checkpoint inventory and hash
- recipe and server argv hashes
- receipt paths and hashes
- resource lock state
- remaining blockers

Stop and report instead of weakening a gate when any identity, file set, token count, or runtime path is ambiguous.
