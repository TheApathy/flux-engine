// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_PLE_QUEUE_PREFETCH=1` (default off): while the scheduler is
//! busy with another request, hand the pending-queue head's first-chunk
//! tokens to [`Model::begin_prefill_queue_prefetch`] once, so a model with
//! host-side per-token reads (Flash-Next PLE) can start chunk 0's reads
//! before that request is admitted. Admission adopts the read when its chunk
//! 0 has the same tokens; nothing changes what is computed.

use spark_model::traits::Model;

const SELECTOR: &str = "ATLAS_QWEN4_PLE_QUEUE_PREFETCH";

fn parse_selected(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

/// Identity of one handed-over first chunk: its length and an FNV-1a hash of
/// its tokens (a repeat hand-over of the same head is skipped; a collision
/// only skips one prefetch, admission still reads).
fn chunk_key(tokens: &[u32]) -> (usize, u64) {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    (tokens.len(), hash)
}

pub(super) struct QueuePrefetch {
    selected: bool,
    last: Option<(usize, u64)>,
}

impl QueuePrefetch {
    pub(super) fn from_env() -> Self {
        Self::new(parse_selected(std::env::var(SELECTOR).ok().as_deref()))
    }

    fn new(selected: bool) -> Self {
        Self {
            selected,
            last: None,
        }
    }

    pub(super) fn selected(&self) -> bool {
        self.selected
    }

    /// The first-chunk length to hand over for `head`, or `None`: off, idle
    /// (admission is imminent and starts the read itself), no head, an empty
    /// prompt, or this exact first chunk was already handed over.
    fn plan(&mut self, busy: bool, head: Option<&[u32]>, chunk_budget: usize) -> Option<usize> {
        if !self.selected || !busy {
            return None;
        }
        let head = head?;
        let len = head.len().min(chunk_budget);
        if len == 0 {
            return None;
        }
        let key = chunk_key(&head[..len]);
        if self.last == Some(key) {
            return None;
        }
        self.last = Some(key);
        Some(len)
    }

    /// One scheduler tick: hand the pending head over if [`Self::plan`] says so.
    pub(super) fn tick(
        &mut self,
        model: &dyn Model,
        busy: bool,
        head: Option<&[u32]>,
        chunk_budget: usize,
    ) {
        if let (Some(len), Some(tokens)) = (self.plan(busy, head, chunk_budget), head) {
            model.begin_prefill_queue_prefetch(tokens, len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{QueuePrefetch, chunk_key, parse_selected};

    #[test]
    fn selector_is_exact_and_default_off() {
        assert!(parse_selected(Some("1")));
        for off in [None, Some(""), Some("0"), Some("true"), Some("1 ")] {
            assert!(!parse_selected(off), "{off:?}");
        }
        let mut off = QueuePrefetch::new(false);
        assert_eq!(off.plan(true, Some(&[1, 2, 3]), 16), None);
    }

    #[test]
    fn hands_over_the_busy_head_once_per_first_chunk() {
        let mut q = QueuePrefetch::new(true);
        let a: Vec<u32> = (0..20_000).collect();
        // Idle: admission is next and issues the read itself.
        assert_eq!(q.plan(false, Some(&a), 16_384), None);
        // Busy: the head's first chunk, capped at the chunk budget.
        assert_eq!(q.plan(true, Some(&a), 16_384), Some(16_384));
        // Every later tick with the same head is a no-op.
        assert_eq!(q.plan(true, Some(&a), 16_384), None);
        assert_eq!(q.plan(true, Some(&a), 16_384), None);
        // A short prompt is handed over whole.
        let b = [7u32, 8, 9];
        assert_eq!(q.plan(true, Some(&b), 16_384), Some(3));
        // No head, or an empty prompt: nothing.
        assert_eq!(q.plan(true, None, 16_384), None);
        assert_eq!(q.plan(true, Some(&[]), 16_384), None);
    }

    #[test]
    fn a_new_head_or_a_new_chunk_budget_is_a_new_hand_over() {
        let mut q = QueuePrefetch::new(true);
        let a: Vec<u32> = (0..20_000).collect();
        let mut b = a.clone();
        b[100] = 1;
        assert_eq!(q.plan(true, Some(&a), 16_384), Some(16_384));
        assert_eq!(q.plan(true, Some(&b), 16_384), Some(16_384));
        // The interleaver shrank the budget: admission will read 8,192 tokens.
        assert_eq!(q.plan(true, Some(&b), 8_192), Some(8_192));
        // Tokens past the first chunk do not change the hand-over.
        let mut c = b.clone();
        c[19_000] = 5;
        assert_eq!(q.plan(true, Some(&c), 8_192), None);
    }

    #[test]
    fn chunk_key_separates_length_and_content() {
        assert_ne!(chunk_key(&[1, 2]), chunk_key(&[2, 1]));
        assert_ne!(chunk_key(&[1, 2]), chunk_key(&[1, 2, 0]));
        assert_eq!(chunk_key(&[]), (0, 0xcbf2_9ce4_8422_2325));
    }
}
