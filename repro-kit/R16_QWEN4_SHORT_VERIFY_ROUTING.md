# Qwen4 shortened native-MTP verification

Source-only fix. It still needs native compilation, exact-token and committed
state parity, and exact-container validation. It does not establish that MTP
accuracy is fixed.

Confidence or grammar shortening can leave two or three draft tokens, which
the scheduler dispatches through its physical K3 or K4 entry. Those model
entries previously used the ordinary hidden-size row layout without Qwen4's
four-stream residual geometry, PLE injection or final mixer. They also owned
generic graph caches without QSA-specific capture admission.

Both sized entries now delegate Qwen4 frames to the existing K-generic
Qwen4-aware verifier before embeddings, pre-verify copies, metadata uploads,
graph lookup or other frame effects. Other model families retain their
existing paths. The delegation preserves the actual physical width. It does
not pad, repeat C1 calls, bypass existing route gates or enable experimental
hybrids. The returned vector must convert to exactly three or four rows.
Errors propagate without retrying a partly mutated frame.

The generic verifier handles residual-width rows, PLE, final mixing,
per-row recurrent intermediates and graph admission. Its only sized fallback
is a separately selected legacy K2 diagnostic, so K3/K4 delegation does not
recurse. However, source review found that its default K4 row-serial oracle
does not prepare or snapshot the QSA undo journal before ordinary decode
mutates the side cache. Accepted-prefix commit already consumes that journal
for the canonical widths two through nine. This delegation must not be
qualified or integrated alone. It requires the separately reviewed oracle
journal repair and forced partial-accept state parity.

Validation must force physical K3 and K4, not just default full K5. Cover
confidence and grammar shortening, all acceptance/rejection prefixes, slot
reuse, 2047/2048 transitions, long context, logprob row layout, and comparison
of exact greedy tokens plus committed target state against Plain and the
generic verifier. Offline source-wiring checks are not runtime proof.
