# Native MTP Hidden Handoff Migration

External Rust callers driving native Qwen4 MTP proposal must use
`Model::save_hidden_for_mtp_sequence(accepted_row, &seq, stream)` instead of
`Model::save_hidden_for_mtp(accepted_row, stream)`.

Save after committing the accepted target prefix and before another request or
proposer overwrites the shared target-hidden scratch. Pass the same sequence and
its unchanged committed cursor to `run_mtp_propose` or `run_mtp_propose_multi`.
The native handoff is consumed once. A missing frame, changed session, reused
physical slot, or mismatched cursor fails rather than falling back to shared
hidden bytes. Save a fresh frame before making another native proposal.

The legacy method remains available for other model implementations. Calling it
alone does not publish the sequence-owned frame required by native Qwen4 MTP.
The scheduler's native save sites use the sequence-aware method.

Retained storage follows the production SSM intermediate capacity, including
its spare row. Storage capacity is not a kernel active-row limit and does not
permit reads beyond the actual target-hidden source arena.

This change isolates retained target-hidden prefixes. It does not enable native
speculative concurrency or isolate target/proposer execution scratch. Existing
target-only native batching guards remain unchanged.
