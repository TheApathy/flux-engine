// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill chunk sizing while other sequences are decoding.
//!
//! Every scheduler tick runs at most one prefill chunk and then one decode
//! step, and the decode step waits for that chunk on the GPU
//! (`complete_prefill_handoff`, `scheduler/mod.rs`). While a 4096- or
//! 8192-token chunk executes, no active decode row emits a token. Two levers
//! bound that stall:
//!
//! - **SLO-budgeted interleave** (`--decode-gap-target-ms`, default 150;
//!   `prefill_slo`). Engaged only when the model reports prepared
//!   fast-prefill widths ([`Model::fast_prefill_chunk_widths`]). While a row
//!   decodes, `prefill_slo` decides each tick whether the pending prompt gets
//!   a chunk and how long it is ([`SloChunk`]); this module turns that
//!   decision into the call's chunk length.
//! - **Fixed cap** (`--prefill-chunk-with-active-decode`, default off): caps
//!   every chunk issued while a row decodes; under interleave it bounds the
//!   widths the SLO may pick.
//!
//! With no decode row the historical `remaining.min(max_prefill_tokens)` +
//! WY4 sizing runs unchanged. The one addition: a request whose offset was
//! left off the `max_prefill_tokens` boundary by earlier bounded chunks
//! re-grows through the largest aligned prepared width (512 -> 1024 -> 2048
//! -> W), because a full chunk at that offset would leave the fast family.
//!
//! Hazards (also logged at scheduler start): chunking changes prefill
//! numerics (chunk 2+ reads earlier context from the KV cache); without
//! interleave a fixed cap below the Qwen4 fast geometry `W` sends capped
//! requests to the ordinary path.

use anyhow::{Result, bail};

/// Capped chunks are multiples of this many tokens so every row-tiled prefill
/// kernel (WY4 GDN chunks of 4, the 8-row HC inject GEMM tiers, 16-row
/// attention tiles) sees an aligned intermediate boundary.
pub const PREFILL_CHUNK_ALIGN: usize = 16;

/// GDN WY4 boundary that every intermediate chunk has always been aligned to.
const WY4_ALIGN: usize = 4;

/// Most prepared widths carried (512..=262144 is 10).
pub const MAX_PREPARED_WIDTHS: usize = 12;

/// Resolve the operator's `--prefill-chunk-with-active-decode` value against
/// the resolved `--max-prefill-tokens` chunk.
///
/// `0` (the default) disables the cap. Any other value must be a positive
/// multiple of [`PREFILL_CHUNK_ALIGN`]; a value at or above the full chunk is
/// a no-op and resolves to `None` so the scheduler log says so.
pub fn resolve_chunk_cap(max_prefill_tokens: usize, requested: usize) -> Result<Option<usize>> {
    if requested == 0 {
        return Ok(None);
    }
    if !requested.is_multiple_of(PREFILL_CHUNK_ALIGN) {
        bail!(
            "--prefill-chunk-with-active-decode must be 0 or a positive multiple of \
             {PREFILL_CHUNK_ALIGN}; got {requested}"
        );
    }
    if max_prefill_tokens == 0 || requested >= max_prefill_tokens {
        // Chunking is off, or the cap cannot shorten any chunk.
        return Ok(None);
    }
    Ok(Some(requested))
}

/// The model's prepared fast-prefill widths (ascending, validated).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedWidths {
    widths: [usize; MAX_PREPARED_WIDTHS],
    len: usize,
}

impl PreparedWidths {
    /// Nonempty, strictly ascending, each a positive multiple of
    /// [`PREFILL_CHUNK_ALIGN`], at most [`MAX_PREPARED_WIDTHS`].
    pub fn new(widths: &[usize]) -> Option<Self> {
        let valid = !widths.is_empty()
            && widths.len() <= MAX_PREPARED_WIDTHS
            && widths
                .iter()
                .all(|&w| w > 0 && w.is_multiple_of(PREFILL_CHUNK_ALIGN))
            && widths.windows(2).all(|pair| pair[0] < pair[1]);
        if !valid {
            return None;
        }
        let mut stored = [0; MAX_PREPARED_WIDTHS];
        stored[..widths.len()].copy_from_slice(widths);
        Some(Self {
            widths: stored,
            len: widths.len(),
        })
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.widths[..self.len]
    }

    pub fn smallest(&self) -> usize {
        self.widths[0]
    }

    /// Prepared widths `<= limit` that `offset` is a multiple of, largest first.
    pub fn aligned_desc(&self, offset: usize, limit: usize) -> impl Iterator<Item = usize> + '_ {
        self.as_slice()
            .iter()
            .rev()
            .copied()
            .filter(move |&w| w <= limit && offset.is_multiple_of(w))
    }

    /// Largest prepared width `<= limit` that `offset` is a multiple of.
    pub fn largest_aligned(&self, offset: usize, limit: usize) -> Option<usize> {
        self.aligned_desc(offset, limit).next()
    }
}

/// This tick's SLO decision for one pending prompt: `rows` tokens for the
/// prompt whose unprefilled remainder is `remaining`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SloChunk {
    pub rows: usize,
    pub remaining: usize,
}

/// Per-tick chunk-sizing configuration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChunkPlan {
    /// Resolved cap from [`resolve_chunk_cap`].
    pub cap_with_active_decode: Option<usize>,
    /// Prepared widths when interleave is engaged; `None` keeps the
    /// historical sizing exactly.
    pub widths: Option<PreparedWidths>,
    /// The SLO decision for the prompt it was computed for.
    pub slo: Option<SloChunk>,
    /// Bounded width for any other prompt (a prompt the decision was not
    /// computed for): the smallest prepared width, or the parity width.
    pub fallback_width: usize,
    /// Chunk boundaries off a prepared boundary are multiples of this
    /// (lcm of the 16-row tiles and the KV block size).
    pub carry_align: usize,
}

/// Chunk-size inputs for one prefill tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkRequest {
    /// Prompt tokens already prefilled (the next chunk's start offset).
    pub offset: usize,
    /// Prompt tokens not yet prefilled for this request.
    pub remaining: usize,
    /// Resolved `--max-prefill-tokens`.
    pub max_prefill_tokens: usize,
    /// Cap and interleave configuration for this tick.
    pub plan: ChunkPlan,
    /// At least one sequence is in decode this tick.
    pub decode_rows_active: bool,
    /// The model cannot attend across chunk boundaries (MLA without a paged
    /// prefill kernel): the whole remainder must go in one chunk.
    pub single_chunk_required: bool,
}

/// Number of prompt tokens the next prefill call should cover.
///
/// Intermediate chunks keep the historical WY4 (4-token) alignment; a capped
/// or bounded chunk is additionally a multiple of [`PREFILL_CHUNK_ALIGN`].
/// The final chunk is never aligned (it must reach the prompt end).
pub fn prefill_chunk_len(req: ChunkRequest) -> usize {
    if req.single_chunk_required {
        return req.remaining;
    }
    if let Some(widths) = req.plan.widths
        && let Some(len) = interleave_chunk_len(widths, &req)
    {
        return len;
    }
    let mut effective_max = req.max_prefill_tokens;
    if req.decode_rows_active
        && let Some(cap) = req.plan.cap_with_active_decode
    {
        effective_max = effective_max.min(cap);
    }
    let mut chunk_len = req.remaining.min(effective_max);
    let is_last = chunk_len >= req.remaining;
    if !is_last && chunk_len >= WY4_ALIGN {
        chunk_len = (chunk_len / WY4_ALIGN) * WY4_ALIGN;
    }
    chunk_len
}

/// The interleave sizing, or `None` for the historical sizing.
fn interleave_chunk_len(widths: PreparedWidths, req: &ChunkRequest) -> Option<usize> {
    let max = req.max_prefill_tokens;
    if max == 0 || req.remaining == 0 {
        return None;
    }
    if let Some(slo) = req.plan.slo
        && slo.remaining == req.remaining
    {
        return Some(slo.rows.min(req.remaining));
    }
    if req.plan.slo.is_none() && !req.decode_rows_active {
        // No decode row: the historical full chunk whenever the offset sits
        // on its boundary (always, for a request that never ran a bounded
        // chunk); otherwise re-grow through the largest aligned width.
        if req.offset.is_multiple_of(max) {
            return None;
        }
        return widths
            .largest_aligned(req.offset, max)
            .map(|w| w.min(req.remaining));
    }
    Some(bounded_len(
        widths,
        req.plan,
        req.offset,
        req.remaining,
        max,
    ))
}

/// A bounded chunk for a prompt the SLO did not decide this tick: the
/// largest prepared width `<= fallback` aligned to `offset`, or, off every
/// prepared boundary, `fallback` rounded down to the carry alignment.
pub fn bounded_len(
    widths: PreparedWidths,
    plan: ChunkPlan,
    offset: usize,
    remaining: usize,
    max_prefill_tokens: usize,
) -> usize {
    let limit = max_prefill_tokens
        .min(plan.fallback_width.max(widths.smallest()))
        .min(plan.cap_with_active_decode.unwrap_or(usize::MAX));
    let align = plan.carry_align.max(PREFILL_CHUNK_ALIGN);
    let len = widths
        .largest_aligned(offset, limit)
        .unwrap_or((limit / align * align).max(align));
    len.min(remaining)
}

/// Budget for a request's first chunk: the explicit prefill ceiling, never
/// the arena's extra decode-slot capacity; under interleave the SLO decision
/// for this prompt (or the bounded fallback width), and otherwise the
/// active-decode cap when a decode row exists at admission time (chunk 0 runs
/// inside the same tick as that row's decode step).
///
/// A single-chunk model (`Model::is_mla()`: GLM-5.3 refuses any partial
/// prompt, and a TransformerModel MLA prefill split in two silently corrupts
/// attention) never takes the cap or a bounded width, exactly as
/// [`prefill_chunk_len`] already does for later chunks (Review A HIGH on
/// 606c60ea).
pub fn initial_chunk_budget(
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    plan: ChunkPlan,
    decode_rows_active: bool,
    single_chunk_required: bool,
    prompt_len: usize,
) -> usize {
    let mut budget = max_prefill_tokens.min(max_batch_tokens);
    if single_chunk_required {
        return budget;
    }
    if let Some(widths) = plan.widths
        && (decode_rows_active || plan.slo.is_some())
    {
        return match plan.slo {
            Some(slo) if slo.remaining == prompt_len => slo.rows.min(budget),
            _ => bounded_len(widths, plan, 0, budget, budget),
        };
    }
    if decode_rows_active && let Some(cap) = plan.cap_with_active_decode {
        budget = budget.min(cap);
    }
    budget
}

/// One-line operator warning when a fixed cap is below the model's
/// fast-prefill chunk geometry and interleave is not engaged (every capped
/// chunk then runs on the ordinary path).
pub fn fast_path_warning(
    cap: Option<usize>,
    fast_rows: Option<usize>,
    interleave_engaged: bool,
) -> Option<String> {
    let (cap, rows) = (cap?, fast_rows?);
    (cap < rows && !interleave_engaged).then(|| {
        format!(
            "--prefill-chunk-with-active-decode {cap} is below the model's fast-prefill chunk \
             geometry ({rows} rows): every chunk issued while another sequence decodes is \
             refused by the fast-prefill family and that request finishes on the ordinary \
             prefill path (correct, several times slower per token). Raise the cap to {rows}, \
             serve the fast family at a narrower ATLAS_QWEN4_PREFILL_WIDE, or enable \
             --decode-gap-target-ms"
        )
    })
}

#[cfg(test)]
#[path = "prefill_chunk_policy_tests.rs"]
mod tests;
