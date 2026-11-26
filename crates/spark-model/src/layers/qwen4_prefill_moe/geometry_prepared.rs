// SPDX-License-Identifier: AGPL-3.0-only

//! Prepared chunk widths of the fast-prefill family.
//!
//! The wide geometry (`ATLAS_QWEN4_PREFILL_WIDE=W`) used to admit exactly one
//! chunk shape: `W` rows at a `W`-aligned offset (plus the prompt's final
//! remainder). Every kernel and scratch buffer on the fast path already runs
//! any row count up to `W` (scratch is sized at `max(rows, W)`, attention and
//! HC tiles need 16-row multiples, the injgemm tier has its own pinned-algo
//! predicate and falls back per component), so the lock was the admission
//! predicate alone. A bounded interleave chunk (power-of-two prepared
//! capacities) needs a smaller chunk that stays on the fast path: the wide
//! geometry now also prepares every power of two from [`MIN_PREPARED_ROWS`]
//! up to `W`. A chunk is admitted when it is exactly one prepared width `B`
//! at a `B`-aligned offset, or the prompt's final remainder inside one.
//!
//! The legacy geometry (WIDE unset: 2048-row chunks of an exactly-8192-token
//! prompt) prepares only its own width, so its admission is unchanged.

use super::ChunkGeometry;

/// Smallest prepared chunk width. Every prepared width is a power of two at
/// or above this (so a multiple of 128 and of the 16-row attention tile), or
/// the geometry's own `rows`.
pub(crate) const MIN_PREPARED_ROWS: usize = 512;

impl ChunkGeometry {
    /// Whether `width` is a prepared chunk width of this geometry: `rows`
    /// itself, and for the wide geometry every power of two in
    /// `MIN_PREPARED_ROWS..rows`.
    pub(crate) fn is_prepared(&self, width: usize) -> bool {
        width == self.rows
            || (self.tail
                && width >= MIN_PREPARED_ROWS
                && width < self.rows
                && width.is_power_of_two())
    }

    /// Prepared widths, ascending; the last one is `rows`.
    pub(crate) fn prepared_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = if self.tail {
            std::iter::successors(Some(MIN_PREPARED_ROWS), |w| w.checked_mul(2))
                .take_while(|&w| w < self.rows)
                .collect()
        } else {
            Vec::new()
        };
        widths.push(self.rows);
        widths
    }

    /// Chunk `[start, start + rows)` of a `prompt_rows`-token prompt is one
    /// prepared chunk: exactly one prepared width `B` at a `B`-aligned
    /// offset, or the prompt's final remainder inside such a chunk.
    pub(crate) fn chunk_admissible(&self, start: usize, rows: usize, prompt_rows: usize) -> bool {
        rows > 0
            && self
                .prepared_widths()
                .into_iter()
                .any(|width| self.chunk_rows_at(start, prompt_rows, width) == Some(rows))
    }

    /// Prompts the fast lineage may carry across several prepared chunks:
    /// every prompt the `W` lineage owns ([`Self::lineage_prompt`]) plus, for
    /// the wide geometry, a prompt of at most `W` rows that bounded chunks
    /// split (longer than the smallest prepared width).
    pub(crate) fn chunked_prompt(&self, prompt_rows: usize) -> bool {
        self.lineage_prompt(prompt_rows)
            || (self.tail
                && prompt_rows > MIN_PREPARED_ROWS.min(self.rows)
                && prompt_rows <= self.total)
    }

    /// Window form of [`Self::chunk_admissible`] for layer-local admission,
    /// which does not see the prompt length: `rows` fit inside one prepared
    /// width aligned at `start`. Legacy: exactly `rows` at a `rows`-aligned
    /// offset (the historical rule).
    pub(crate) fn window_rows_admitted(&self, rows: usize, start: usize) -> bool {
        if !self.tail {
            return rows == self.rows && start.is_multiple_of(self.rows);
        }
        (2..=self.rows).contains(&rows)
            && self
                .prepared_widths()
                .into_iter()
                .any(|width| rows <= width && start.is_multiple_of(width))
    }
}

#[cfg(test)]
#[path = "geometry_prepared_tests.rs"]
mod tests;
