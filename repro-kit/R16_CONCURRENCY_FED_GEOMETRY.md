# R16 concurrency target-fed geometry

This is an offline qualifier repair, not a new GPU, accuracy, performance, or
container qualification receipt. It does not change scheduler execution.

## Normal terminal state

Let `P` be prompt tokens and `C` be published output tokens. Prefill consumes
the prompt and samples the first output without feeding that output back into
the target. Each decode consumes the preceding pending output before sampling
the next output. Ordinary termination does not perform an extra forward.

The exact normal terminal target-fed extent is `P + C - 1`, including when the
last published output is EOS. Receipt `seq_len` and `token_count` both describe
that fed prefix. `kv_token_positions` remains `seq_len * attention_layers`.

The raw `LOGITS_FNV.idx` is output count before publishing the next sampled
output. A frame bound to index `idx` therefore has pre-forward extent
`P + idx - 1`, not `P + idx`. The forward produces extent `P + idx`.

Source anchors are `scheduler/prefill_b_step.rs`,
`scheduler/decode_logits_seq.rs`, `scheduler/decode_logits_step.rs`, and
`scheduler/lifecycle.rs` under `crates/spark-server/src`. The target receipt
uses `seq.tokens`, checked against `seq.seq_len`, in
`crates/spark-model/src/model/qwen4_target_state_receipt.rs`.

## Cancellation is a distinct terminal transition

Ordinary decode can observe cancellation after forwarding the pending output
and logging its new logits but before publishing the new sample. A stream
send failure can instead retire after publishing an output, leaving that
output pending. The validator must not assume the two states are identical.

The unmodified contiguous raw logits rows determine which exact transition
occurred. `cancelled.terminal_geometry` persists the derived discriminator.

| Geometry | Raw decode indices | Target-fed terminal extent |
| --- | --- | --- |
| `pending-output` | `1 .. C-1` inclusive | `P + C - 1` |
| `computed-unemitted` | `1 .. C` inclusive | `P + C` |

All raw rows, including the computed but unpublished row `C`, must match the
exact C1 logits prefix. The final canceled raw row must be present in the last
frame carrying that session. The terminal fed extent must equal that frame's
pre-forward extent plus one. Missing rows, gaps, invented rows, mislabeled
geometry, length drift, and state retirement before the final frame fail closed.

This does not accept a permissive range of receipt lengths. A specific raw-row
sequence determines one exact fed extent. Frame/session ownership, stable
slots, retirement-before-reuse, digest coverage, and native-K5 mode guards are
unchanged. The canceled-state helper delegates common validation through the
existing four-argument state validator so the driver's K5 wrapper still runs.

## Evidence limits

Canceled `fed_tokens_sha256` and `target_state_sha256` are preserved as emitted.
These artifacts do not contain a separately sealed C1 target-state snapshot at
each truncated cancellation prefix. They therefore do not independently prove
that the canceled state digest equals a C1 digest at the same shorter extent.
No digest is synthesized or relabeled as such a proof. Completed survivors,
the replacement, and post-drain reuse still require full terminal C1 identity.

The focused tests use production-derived geometry and hostile off-by-one,
missing-extra-row, extra-row hash, mislabeled-transition, and frame-binding
cases. Their pass result is source/CPU evidence only. Real-kernel binary,
GPU correctness, long-context accuracy, throughput, and container E2E gates
remain separate requirements.
