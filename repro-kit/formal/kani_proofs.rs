// SPDX-License-Identifier: AGPL-3.0-only
//! AWS Kani Rust Verifier (`cbmc` + `kissat`) formal proof harnesses for
//! Flux / Atlas `v0.9.16` critical Rust invariants.
//!
//! Run via `repro-kit/formal/run_kani.sh`. Every `#[kani::proof]` harness
//! symbolically explores the full Rust MIR state space (checked arithmetic,
//! slice bounds, assertion reachability, panic freedom).

#![allow(dead_code)]

const THINK_DEFER_BUDGET_FACTOR: u32 = 3;
const THINK_DEFER_ABS_CEILING: u32 = 2_048;
const CARVEOUT_ALIGN: usize = 64 << 10;

pub fn min_answer_runway(max_output_tokens: usize) -> usize {
    if max_output_tokens == 0 {
        return 0;
    }
    (max_output_tokens / 2)
        .min(256)
        .max(max_output_tokens / 10)
        .max(1)
}

pub fn compute_think_defer_hard_override(
    thinking_tokens: u32,
    thinking_budget: Option<u32>,
    output_tokens_len: usize,
    max_output_tokens: usize,
) -> bool {
    let budget_overrun = match thinking_budget {
        Some(b) => thinking_tokens >= b.saturating_mul(THINK_DEFER_BUDGET_FACTOR),
        None => thinking_tokens >= THINK_DEFER_ABS_CEILING,
    };
    let min_runway = min_answer_runway(max_output_tokens);
    let runway_exhausted =
        max_output_tokens > 0 && output_tokens_len >= max_output_tokens.saturating_sub(min_runway);
    budget_overrun || runway_exhausted
}

pub const fn should_inject_think_end(
    force_end_thinking: bool,
    in_code_fence: bool,
    hard_override: bool,
) -> bool {
    force_end_thinking && (!in_code_fence || hard_override)
}

pub fn crosses_interval(
    chunk_start: usize,
    end_token: usize,
    block_size: usize,
    interval_blocks: usize,
) -> bool {
    if interval_blocks == 0 || block_size == 0 {
        return false;
    }
    let Some(span) = interval_blocks.checked_mul(block_size) else {
        return false;
    };
    end_token / span > chunk_start / span
}

pub const fn can_batch_mixed_step(
    q12_dispatch_disabled: bool,
    serialize_hc: bool,
    prefilling_len: usize,
    active_len: usize,
    spec_active: bool,
    is_ep: bool,
) -> bool {
    !q12_dispatch_disabled
        && !serialize_hc
        && prefilling_len >= 2
        && active_len > 0
        && !spec_active
        && !is_ep
}

pub const fn carveout_footprint(bytes: usize) -> usize {
    let b = if bytes == 0 { 1 } else { bytes };
    b.next_multiple_of(CARVEOUT_ALIGN)
}

pub const fn splitmix64_mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub const fn position_base(seed: u64, abs_pos: u64) -> u64 {
    splitmix64_mix(
        splitmix64_mix(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
            ^ abs_pos.wrapping_mul(0xD1B5_4A32_D192_ED03),
    )
}

pub const fn key_bits(seed: u64, abs_pos: u64, token_id: u32) -> u64 {
    splitmix64_mix(position_base(seed, abs_pos) ^ (token_id as u64))
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn prove_thinking_runway_and_defer_override() {
        let thinking_tokens: u32 = kani::any();
        let has_budget: bool = kani::any();
        let raw_budget: u32 = kani::any();
        let thinking_budget = if has_budget { Some(raw_budget) } else { None };
        let output_tokens_len: usize = kani::any();
        let max_output_tokens: usize = kani::any();
        let in_code_fence: bool = kani::any();

        // Must never panic or overflow for ANY u32 / usize combination.
        let hard = compute_think_defer_hard_override(
            thinking_tokens,
            thinking_budget,
            output_tokens_len,
            max_output_tokens,
        );
        let inject = should_inject_think_end(true, in_code_fence, hard);

        if max_output_tokens >= 2 {
            let runway = min_answer_runway(max_output_tokens);
            assert!(runway >= 1 && runway < max_output_tokens);
            if output_tokens_len >= max_output_tokens - runway {
                assert!(hard);
                assert!(inject);
            }
        }
    }

    #[kani::proof]
    fn prove_ckpt_interval_crosses_interval_panic_free() {
        let chunk_start: usize = kani::any();
        let end_token: usize = kani::any();
        let block_size: usize = kani::any();
        let interval_blocks: usize = kani::any();

        // Panic-freedom over the entire usize^4 domain (including 0 and usize::MAX).
        let fired = crosses_interval(chunk_start, end_token, block_size, interval_blocks);
        if block_size == 0 || interval_blocks == 0 {
            assert!(!fired);
        }
    }

    #[kani::proof]
    fn prove_can_batch_mixed_step_blocks_spec() {
        let q12_off: bool = kani::any();
        let hc: bool = kani::any();
        let pref: usize = kani::any();
        let act: usize = kani::any();
        let spec: bool = kani::any();
        let ep: bool = kani::any();

        let admit = can_batch_mixed_step(q12_off, hc, pref, act, spec, ep);
        if spec || hc || ep || q12_off || pref < 2 || act == 0 {
            assert!(!admit);
        } else {
            assert!(admit);
        }
    }

    #[kani::proof]
    fn prove_carveout_footprint_aligned_and_bounded() {
        let bytes: usize = kani::any();
        kani::assume(bytes <= (usize::MAX - CARVEOUT_ALIGN));
        let fp = carveout_footprint(bytes);
        assert!(fp >= CARVEOUT_ALIGN);
        assert!(fp >= bytes);
        assert!(fp.is_multiple_of(CARVEOUT_ALIGN));
        if bytes > 0 {
            assert!(fp - bytes < CARVEOUT_ALIGN);
        }
    }

    #[kani::proof]
    fn prove_splitmix64_key_bits_panic_free() {
        let seed: u64 = kani::any();
        let abs_pos: u64 = kani::any();
        let tok: u32 = kani::any();
        let _ = key_bits(seed, abs_pos, tok);
    }
}
