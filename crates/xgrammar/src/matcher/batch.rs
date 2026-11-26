// SPDX-License-Identifier: AGPL-3.0-only
//
// BatchGrammarMatcher — batched bitmask fill / token acceptance.
//
// Port of `class BatchGrammarMatcher` from `cpp/grammar_matcher.cc`.
// The C++ version uses a hand-rolled `ThreadPool`; this port uses the
// crate's existing `rayon` dependency for the parallel fill path.
//
// SIMPLIFICATION vs C++
// ---------------------
// The C++ `BatchFillNextTokenBitmask` takes an optional `indices`
// permutation mapping matcher `i` to an arbitrary tensor slice. This
// port exposes the natural mapping (`matchers[i]` -> slice `i`) so the
// per-thread mutable slices are provably disjoint without `unsafe`.

use std::sync::Arc;

use rayon::prelude::*;

use super::bitmask::bitmask_size;
use super::fill::FillError;
use super::matcher::GrammarMatcher;

/// A batched driver over many [`GrammarMatcher`]s — fills one packed
/// bitmask tensor for a whole decode batch in parallel.
#[derive(Debug, Clone)]
pub struct BatchGrammarMatcher {
    /// Upper bound on worker threads; `1` forces the sequential path.
    max_threads: usize,
    /// Dedicated Rayon pool enforcing `max_threads` when `max_threads > 1`.
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl BatchGrammarMatcher {
    /// Construct a batched matcher.
    ///
    /// `max_threads`: `None` picks `available_parallelism / 2` (the C++
    /// `"auto"` default); `Some(n)` caps worker threads at `n` (must be >= 1).
    #[must_use]
    pub fn new(max_threads: Option<usize>) -> Self {
        let max_threads = match max_threads {
            Some(n) => {
                assert!(n >= 1, "max_threads must be >= 1");
                n
            }
            None => std::thread::available_parallelism()
                .map(|n| (n.get() / 2).max(1))
                .unwrap_or(1),
        };
        let pool = if max_threads > 1 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build()
                .ok()
                .map(Arc::new)
        } else {
            None
        };
        Self { max_threads, pool }
    }

    /// The configured worker-thread cap.
    #[must_use]
    pub fn max_threads(&self) -> usize {
        self.max_threads
    }

    /// Fill `bitmask` for every matcher in `matchers`.
    ///
    /// `bitmask` is one flat buffer of `matchers.len() *
    /// bitmask_size(vocab)` words; matcher `i` writes slice `i`. Every
    /// matcher must share the same vocabulary size. Port of
    /// `BatchFillNextTokenBitmask` (natural-index form).
    ///
    /// Returns one `Result` per matcher (same order).
    pub fn fill_next_token_bitmask(
        &self,
        matchers: &mut [GrammarMatcher],
        bitmask: &mut [i32],
        debug: bool,
    ) -> Vec<Result<bool, FillError>> {
        if matchers.is_empty() {
            return Vec::new();
        }
        let vocab = matchers[0].tokenizer_info().vocab_size();
        let words = bitmask_size(vocab);
        if words == 0 {
            return matchers
                .iter_mut()
                .map(|m| m.fill_next_token_bitmask(&mut [], 0, debug))
                .collect();
        }
        assert!(
            bitmask.len() >= matchers.len() * words,
            "bitmask buffer too small for the batch"
        );
        let used = &mut bitmask[..matchers.len() * words];

        if self.max_threads <= 1 || matchers.len() == 1 {
            return matchers
                .iter_mut()
                .zip(used.chunks_mut(words))
                .map(|(m, slice)| m.fill_next_token_bitmask(slice, 0, debug))
                .collect();
        }

        // Parallel path: matcher `i` <-> disjoint slice `i`, bounded by
        // `self.pool` so `max_threads` is enforced.
        let mut run = || {
            matchers
                .par_iter_mut()
                .zip(used.par_chunks_mut(words))
                .map(|(m, slice)| m.fill_next_token_bitmask(slice, 0, debug))
                .collect()
        };
        match &self.pool {
            Some(pool) => pool.install(run),
            None => run(),
        }
    }

    /// Accept one token per matcher. `token_ids[i]` is fed to
    /// `matchers[i]`. Port of `BatchAcceptToken`.
    pub fn accept_token(
        matchers: &mut [GrammarMatcher],
        token_ids: &[i32],
        debug: bool,
    ) -> Vec<bool> {
        assert_eq!(
            matchers.len(),
            token_ids.len(),
            "matchers and token_ids must have equal length"
        );
        matchers
            .iter_mut()
            .zip(token_ids)
            .map(|(m, &t)| m.accept_token(t, debug))
            .collect()
    }

    /// Accept one string per matcher. Port of `BatchAcceptString`.
    pub fn accept_string(
        matchers: &mut [GrammarMatcher],
        input_strs: &[&str],
        debug: bool,
    ) -> Vec<bool> {
        assert_eq!(
            matchers.len(),
            input_strs.len(),
            "matchers and input_strs must have equal length"
        );
        matchers
            .iter_mut()
            .zip(input_strs)
            .map(|(m, &s)| m.accept_string(s, debug))
            .collect()
    }
}
