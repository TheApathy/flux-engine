// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request grammar matching state.

use xgrammar::{CompiledGrammar, GrammarMatcher, allocate_token_bitmask, reset_token_bitmask};

use super::engine::GrammarError;

// ── GrammarState ───────────────────────────────────────────────────────

/// How many of the costliest token-masks to pre-warm when a grammar
/// state is created (xgrammar Tier 2, overlapped mask generation).
///
/// `CompiledGrammar::compile_top_k_masks` ranks reachable scanable
/// parser states by first-character scan breadth and eagerly populates
/// the JIT mask cache for the top `k`. Called from [`GrammarState::new`]
/// — which runs during the prefill phase of a grammar-constrained
/// request, while the GPU is busy with the prompt — so the first decode
/// steps never pay a cold mask-computation stall.
///
/// `8` is a deliberately small constant: a tool-call structural-tag
/// grammar has only a handful of genuinely expensive states (the JSON
/// string / value scanners), and the call is `<= ranked.len()` work, so
/// over-provisioning `k` is harmless. The mask cache is per-grammar and
/// shared (`Arc`) across every request that reuses the cached
/// `CompiledGrammar`, so this warm-up amortizes across requests too.
const FORCED_TOKEN_TOP_K: usize = 8;

/// Per-request grammar matching state.
///
/// Wraps a [`GrammarMatcher`] with its own bitmask buffer. The bitmask
/// is reused across decode steps to avoid re-allocation.
pub struct GrammarState {
    matcher: GrammarMatcher,
    /// Bitmask buffer: `Box<[i32]>` of shape `(1, ceil(vocab_size / 32))`.
    bitmask_data: Box<[i32]>,
    vocab_size: usize,
}

impl GrammarState {
    /// Create a new per-request grammar state from a compiled grammar.
    ///
    /// `vocab_size` must match the tokenizer vocabulary used during compilation.
    ///
    /// As part of construction this pre-warms the [`FORCED_TOKEN_TOP_K`]
    /// costliest token-masks via [`CompiledGrammar::compile_top_k_masks`]
    /// (xgrammar Tier 2). Construction happens during prefill, so the
    /// warm-up overlaps the prompt forward pass and the first decode
    /// steps never stall on a cold mask computation. The warm-up only
    /// populates a cache — it cannot change matcher behavior — so it is
    /// safe unconditionally.
    pub fn new(compiled: &CompiledGrammar, vocab_size: usize) -> Result<Self, GrammarError> {
        let matcher = GrammarMatcher::new(
            compiled, None,  // use stop tokens from compiled grammar
            false, // require stop token for proper termination
            -1,    // unlimited rollback
        )
        .map_err(GrammarError::Compilation)?;

        // Tier 2 (overlapped mask generation): eagerly compute the
        // costliest masks so they are warm before the first decode
        // step. Pure cache population — no behavioral effect.
        let warmed = compiled.compile_top_k_masks(FORCED_TOKEN_TOP_K);
        tracing::debug!(
            warmed,
            requested = FORCED_TOKEN_TOP_K,
            "Grammar: pre-warmed top-k token masks during prefill"
        );

        let bitmask_data = allocate_token_bitmask(1, vocab_size);

        Ok(Self {
            matcher,
            bitmask_data,
            vocab_size,
        })
    }

    /// Fill the allowed-token bitmask for the next decode step.
    ///
    /// Returns `true` if the bitmask constrains at least one token (i.e., is not
    /// all-ones). When `false`, the caller can skip bitmask application.
    ///
    /// `fill_bitmask` is read-only with respect to parser state (trial probes
    /// are rolled back before returning), so it only needs to be called when
    /// sampling needs the allowed-token bitmask.
    pub fn fill_bitmask(&mut self) -> bool {
        // Once the matcher has accepted its stop token, the grammar is already
        // satisfied and imposes no further constraint; skip bitmask filling.
        if self.matcher.is_terminated() {
            return false;
        }
        reset_token_bitmask(&mut self.bitmask_data);
        self.matcher
            .fill_next_token_bitmask(&mut self.bitmask_data, 0, false)
    }

    /// Raw bitmask data: `ceil(vocab_size / 32)` i32 words.
    ///
    /// Bit `token_id` is at `data[token_id / 32] & (1 << (token_id % 32))`.
    /// A set bit means the token is allowed.
    pub fn bitmask_data(&self) -> &[i32] {
        &self.bitmask_data
    }

    /// Check if a specific token is allowed by the current bitmask.
    pub fn is_token_allowed(&self, token_id: u32) -> bool {
        let word = (token_id / 32) as usize;
        let bit = token_id % 32;
        if word >= self.bitmask_data.len() {
            return false;
        }
        (self.bitmask_data[word] & (1i32 << bit)) != 0
    }

    /// Accept a sampled token and advance the grammar state.
    ///
    /// Returns `true` if the token was accepted by the grammar.
    /// Returns `false` if the token violates the grammar (should not happen
    /// if the bitmask was applied correctly).
    ///
    /// Short-circuits with `true` once the matcher has reached its
    /// terminated (accepting) state so trailing tokens in speculative draft
    /// runs do not emit post-termination warnings or get treated as grammar
    /// rejections by `truncate_drafts_at_grammar_boundary`. Tokens past a
    /// completed grammar are handled independently by the EOS handler.
    pub fn accept_token(&mut self, token_id: u32) -> bool {
        if self.matcher.is_terminated() {
            return true;
        }
        self.matcher.accept_token(token_id as i32)
    }

    /// The single grammar-forced next token, if the current state
    /// admits exactly one legal token (xgrammar Tier 3b, Coalescence).
    ///
    /// Returns `Some(token_id)` when the constrained grammar leaves no
    /// choice — the token is fully determined, so the model sampling
    /// step (and the full vocab-wide mask fill) for this position can be
    /// skipped and `token_id` emitted directly. Returns `None` when the
    /// continuation is a genuine choice, the state is dead, or the
    /// matcher has terminated.
    ///
    /// CORRECTNESS: `forced_token` computes the same authoritative
    /// next-token bitmask [`Self::fill_bitmask`] would and reports a
    /// token only when it is the *sole* set bit — so it is, by
    /// construction, the only grammar-legal token. The normal path could
    /// only ever have sampled that exact token (every other token is
    /// masked to `-inf`). The matcher state is left unchanged: the caller
    /// must still feed the returned token back through
    /// [`Self::accept_token`], exactly as for a sampled token.
    ///
    /// Returns `None` once the matcher has terminated (no further
    /// constraint) — symmetric with [`Self::fill_bitmask`]'s guard.
    pub fn forced_token(&mut self) -> Option<i32> {
        if self.matcher.is_terminated() {
            return None;
        }
        self.matcher.forced_token()
    }

    /// Whether the grammar has been fully matched (all required structure generated).
    pub fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }

    /// True when the grammar admits nothing but a stop token at this
    /// position: either a stop token was already accepted, or the root rule
    /// is complete and every non-stop token is masked. A `stop_after_first`
    /// structural tag reaches this state on its close token, so the
    /// scheduler can end the sequence there instead of spending one more
    /// decode step on a grammar-forced EOS. A looping (`auto`,
    /// `parallel_tool_calls=true`) tag grammar still admits prose or another
    /// call after a close, so it reports `false`.
    ///
    /// Refills the bitmask buffer; the fill is a pure function of the matcher
    /// state, so a later [`Self::fill_bitmask`] for sampling is unaffected.
    pub fn only_stop_tokens_remain(&mut self) -> bool {
        if self.matcher.is_terminated() {
            return true;
        }
        if !self.matcher.is_grammar_completed() {
            return false;
        }
        if !self.fill_bitmask() {
            // Unconstrained next step: every token is legal.
            return false;
        }
        let stop_ids = self.matcher.stop_token_ids();
        let mut has_allowed_stop = false;
        for (word_idx, &word) in self.bitmask_data.iter().enumerate() {
            let mut bits = word as u32;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                let token_id = (word_idx as u32) * 32 + bit;
                if (token_id as usize) < self.vocab_size {
                    if !stop_ids.contains(&(token_id as i32)) {
                        return false;
                    }
                    has_allowed_stop = true;
                }
            }
        }
        has_allowed_stop
    }

    /// Rollback the grammar state by `n` tokens.
    ///
    /// Used for MTP speculative decode: when draft tokens are rejected,
    /// the grammar state must be rewound to match.
    pub fn rollback(&mut self, n: usize) {
        self.matcher.rollback(n as i32);
    }

    /// Return the number of matcher-history entries currently available for
    /// rollback.
    ///
    /// Speculative grammar validation may keep accepting a draft suffix after
    /// the matcher has terminated. [`Self::accept_token`] deliberately treats
    /// those trailing tokens as valid, but they do not advance the matcher and
    /// therefore do not create history entries. Callers doing a transient
    /// accept/rollback pass must use the history delta rather than the number
    /// of logically accepted draft tokens.
    pub(crate) fn num_history_steps(&self) -> usize {
        self.matcher.num_history_steps()
    }

    /// Reset the grammar state to the initial position.
    pub fn reset(&mut self) {
        self.matcher.reset();
    }

    /// Apply the current bitmask to a slice of f32 logits in-place.
    ///
    /// Masked tokens (disallowed by grammar) are set to `f32::NEG_INFINITY`.
    /// This is the CPU-side application; for GPU-side, a CUDA kernel would
    /// be needed (future optimization).
    pub fn apply_bitmask_to_logits(&self, logits: &mut [f32]) {
        let n = logits.len().min(self.vocab_size);
        for token_id in 0..n {
            let word = token_id / 32;
            let bit = token_id % 32;
            if word < self.bitmask_data.len() && (self.bitmask_data[word] & (1i32 << bit)) == 0 {
                logits[token_id] = f32::NEG_INFINITY;
            }
        }
    }
}

// ── Vocabulary extraction helper ───────────────────────────────────────

// F72 helpers (`decoded_vocab_bytes`, `compute_trigger_breakers`)
// were removed in F73 / fix42. The byte-level partial-trigger anchor
// at the sampler level hung the server in production despite passing
// isolated tests. The xgrammar non-anchored TagDispatch limitation is
// now handled at the streaming-sanitizer + parser layer (envelope_open
// markers in `LeakMarkers`, plus `<minimax:_call>` → `<tool_call>`
// normalisation in `parse_tool_calls`).
