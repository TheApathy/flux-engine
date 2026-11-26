// SPDX-License-Identifier: AGPL-3.0-only

//! Pure sizing terms for the startup fit planner (longctx-plan's
//! `fit_plan.rs`): what one saved state costs, what the RAM tier keeps free,
//! and how much KV a slot hold-back reserves. No device or env access.

/// QSA side cache per token: raw ring + compressed keys, 1 KiB each per
/// 16-token block per attention layer (qwen4_qsa.rs QsaCacheLayout).
pub const QSA_BYTES_PER_TOKEN_PER_ATTN_LAYER: u64 = 128;
/// Native-MTP head K/V per token (1 layer, bf16, 2 KV heads x 256 x K+V).
pub const MTP_KV_BYTES_PER_TOKEN: u64 = 2048;
/// RAM-tier margin default (strixite's measured 4 GiB; kept MemAvailable floor
/// for the conversation cache, `ATLAS_QWEN4_SEQSTATE_CACHE_RAM_MARGIN_GIB`).
pub const DEFAULT_RAM_MARGIN_BYTES: u64 = 4 << 30;
/// Default slots whose full-depth KV the plan holds back (lead default Q2).
pub const DEFAULT_HOLD_SLOTS: u64 = 1;

/// Inputs that fix one sequence's saved-state size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlobGeometry {
    /// Target K+V bytes per token across all attention layers, as stored.
    pub kv_bytes_per_token: u64,
    pub attn_layers: u64,
    /// Exact mode (swap) carries the MTP K/V; Resume mode (cache) does not.
    pub with_mtp: bool,
    /// Per-slot fixed state: GDN h + conv of every linear layer, PLE live +
    /// checkpoint history (`ssm_pool.h_bytes/conv_bytes`, PLE stride x 2).
    pub fixed_bytes: u64,
}

impl BlobGeometry {
    pub fn bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
            + self.attn_layers * QSA_BYTES_PER_TOKEN_PER_ATTN_LAYER
            + if self.with_mtp {
                MTP_KV_BYTES_PER_TOKEN
            } else {
                0
            }
    }

    /// Bytes of one blob at `tokens` positions, block-rounded, plus the
    /// fixed state and a header allowance (tokens + section table).
    pub fn blob_bytes(&self, tokens: u64, block_size: u64) -> u64 {
        let rounded = tokens.div_ceil(block_size.max(1)) * block_size.max(1);
        let header = 4096 + 8 * tokens;
        rounded * self.bytes_per_token() + self.fixed_bytes + header
    }
}

/// KV bytes the plan subtracts from MemAvailable for `hold_slots` slots at
/// full depth. Under A1 the pool is committed whole, so this equals the
/// committed pool when `hold_slots` = slots; under A2 (VMM, deferred) it is
/// the reserve on top of what is mapped.
pub fn kv_hold_bytes(kv_bytes_per_token: u64, max_seq_len: u64, hold_slots: u64) -> u64 {
    kv_bytes_per_token * max_seq_len * hold_slots
}

/// The conversation cache's startup rule (strixite memory_plan.hpp):
/// MemAvailable after load must cover the margin plus one full-capacity save.
pub fn cache_fits(mem_available: u64, margin: u64, full_state_bytes: u64) -> bool {
    mem_available >= margin.saturating_add(full_state_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FN_BF16: BlobGeometry = BlobGeometry {
        kv_bytes_per_token: 24_576,
        attn_layers: 12,
        with_mtp: true,
        fixed_bytes: 0,
    };

    #[test]
    fn per_token_sizes_match_the_design() {
        assert_eq!(FN_BF16.bytes_per_token(), 28_160);
        let nvfp4 = BlobGeometry {
            kv_bytes_per_token: 6_912,
            ..FN_BF16
        };
        assert_eq!(nvfp4.bytes_per_token(), 10_496);
        let resume = BlobGeometry {
            with_mtp: false,
            ..nvfp4
        };
        assert_eq!(resume.bytes_per_token(), 8_448);
    }

    #[test]
    fn blob_rounds_to_blocks_and_adds_fixed_state() {
        let g = BlobGeometry {
            fixed_bytes: 1000,
            ..FN_BF16
        };
        assert_eq!(g.blob_bytes(17, 16), 32 * 28_160 + 1000 + 4096 + 8 * 17);
    }

    #[test]
    fn hold_and_fit_rules() {
        assert_eq!(kv_hold_bytes(6_912, 524_288, 1), 6_912 * 524_288);
        assert!(cache_fits(10, 4, 6));
        assert!(!cache_fits(9, 4, 6));
        assert!(!cache_fits(1, u64::MAX, 1));
    }
}
