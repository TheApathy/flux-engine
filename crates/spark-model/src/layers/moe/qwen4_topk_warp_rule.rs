// SPDX-License-Identifier: AGPL-3.0-only

//! Admission rule of `ATLAS_QWEN4_PREFILL_MOE_TOPK_WARP` (default off): the warp-per-token router
//! top-k twin `moe_topk_softmax_batched_w32` (kernels/gb10/common/moe_topk.cu) in place of the
//! 256-thread-per-token `moe_topk_softmax_batched` on the N-token prefill softmax router.
//! Bit-identical indices and weights: the twin reproduces the block kernel's selection order
//! ([`rank`]), its full-softmax summation tree and its normalisation, with the same expressions.
//! Dependency-free (std only) so `tests/moe_topk_warp_static.rs` runs it with `rustc --test`.

/// Default-off selector: absent/0 off, 1 on; anything else fails the first prefill.
pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_TOPK_WARP";
/// The twin's entry in the `moe_topk` module.
pub(crate) const KERNEL: &str = "moe_topk_softmax_batched_w32";
/// The twin holds one 512-expert row per warp (two 16-byte loads per lane).
pub(crate) const EXPERTS: u32 = 512;
/// Top-k values and indices of a row live in a [8][16] shared array.
pub(crate) const MAX_TOP_K: u32 = 16;
/// One token per warp, 256-thread blocks.
pub(crate) const TOKENS_PER_BLOCK: u32 = 8;
pub(crate) const BLOCK_THREADS: u32 = 256;

pub(crate) fn parse(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{SELECTOR} must be absent, 0 or 1; got {other:?}")),
    }
}

/// Why the twin may not replace the block kernel for this dispatch (`None` = it may). Only the
/// softmax router reaches this check; the sigmoid/correction-bias router keeps its own kernel.
pub(crate) fn refusal(
    num_experts: u32,
    top_k: u32,
    num_tokens: u32,
    logits_addr: u64,
) -> Option<&'static str> {
    if num_experts != EXPERTS {
        return Some("num_experts_not_512");
    }
    if top_k == 0 || top_k > MAX_TOP_K {
        return Some("top_k_not_1_to_16");
    }
    if num_tokens == 0 {
        return Some("no_tokens");
    }
    // Rows are 1024 B, so an aligned base aligns every row's two uint4 loads.
    if !logits_addr.is_multiple_of(16) {
        return Some("logits_not_16b_aligned");
    }
    None
}

pub(crate) fn grid(num_tokens: u32) -> u32 {
    num_tokens.div_ceil(TOKENS_PER_BLOCK)
}

// The tie-order model below is the contract the CUDA twin implements; only the standalone test
// (`tests/moe_topk_warp_static.rs`) calls it, so it carries no runtime cost.
#[allow(dead_code)]
pub(crate) fn bitrev5(x: u32) -> u32 {
    (x & 31).reverse_bits() >> 27
}

/// The block kernel's tie order among equal logits (lower rank wins): thread `tid = e % 256` scans
/// `e = tid` before `tid + 256` (j), its warp's shfl_down tournament keeps the receiving lane
/// (bit-reversed lane order), and thread 0 folds warps 0..7 in order.
#[allow(dead_code)]
pub(crate) fn rank(expert: u32) -> u32 {
    (((expert >> 5) & 7) << 6) | (bitrev5(expert & 31) << 1) | (expert >> 8)
}

/// Inverse of [`rank`] on 0..512.
#[allow(dead_code)]
pub(crate) fn expert_of_rank(rank: u32) -> u32 {
    let w = rank >> 6;
    let l = bitrev5((rank >> 1) & 31);
    let j = rank & 1;
    (j << 8) | (w << 5) | l
}

/// Lane of the twin's warp that holds `expert`, and its register slot (`x[8 j + c]`).
#[allow(dead_code)]
pub(crate) fn owner_of(expert: u32) -> (u32, u32) {
    let lane = (expert & 255) >> 3;
    let slot = ((expert >> 8) << 3) | (expert & 7);
    (lane, slot)
}
