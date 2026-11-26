// SPDX-License-Identifier: AGPL-3.0-only

//! Admission rule of `ATLAS_QWEN4_PREFILL_GDN_CONV_V2` (default off): the FI-route conv/SiLU/L2
//! kernel `qwen4_conv1d_l2norm_{f16,bf16}_prefill_v2` (qwen4_gdn_fi_glue.cu) in place of
//! `..._prefill_parallel`. Byte-identical output: per element the same taps, order and rounding;
//! the per-head L2 sum on v1's reduction tree. 8 channels per lane, 32-token strips per warp.
//! Dependency-free (std only) so `tests/gdn_conv_v2_static.rs` runs it with `rustc --test`.

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_CONV_V2";
pub(crate) const V3_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_CONV_V3";
/// Tokens per warp strip (`QWEN4_FI_CONV_V2_TOKENS` in the kernel).
pub(crate) const TOKENS: u32 = 32;
/// Warps (strips) per 256-thread CTA; one warp spans 256 channels.
pub(crate) const WARPS: u32 = 8;
pub(crate) const BLOCK_THREADS: u32 = 256;
pub(crate) const CHANNELS_PER_CTA: u32 = 256;

pub(crate) fn parse(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{SELECTOR} must be absent, 0 or 1; got {other:?}")),
    }
}

pub(crate) fn parse_v3(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "{V3_SELECTOR} must be absent, 0 or 1; got {other:?}"
        )),
    }
}

/// The shape and pointer facts v2 relies on; `None` = admissible.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refusal(
    rows: u32,
    dim: u32,
    qk_channels: u32,
    head_dim: u32,
    d_conv: u32,
    input_stride: u32,
    output_stride: u32,
    ptrs: [u64; 3],
) -> Option<&'static str> {
    if rows == 0 {
        return Some("no_rows");
    }
    if !dim.is_multiple_of(CHANNELS_PER_CTA)
        || !qk_channels.is_multiple_of(CHANNELS_PER_CTA)
        || qk_channels > dim
    {
        return Some("channels_not_256_blocks");
    }
    if head_dim != 128 {
        return Some("head_dim_not_128");
    }
    if d_conv != 4 {
        return Some("d_conv_not_4");
    }
    // 16-byte row loads/stores: 8 bf16 / 8 16-bit outputs per lane.
    if !input_stride.is_multiple_of(8)
        || !output_stride.is_multiple_of(8)
        || input_stride < dim
        || output_stride < dim
    {
        return Some("row_stride_not_16b");
    }
    if ptrs.iter().any(|p| !p.is_multiple_of(16)) {
        return Some("pointer_not_16b_aligned");
    }
    None
}

/// Launch grid `[dim / 256, ceil(rows / (8 * TOKENS)), 1]`.
pub(crate) fn grid(dim: u32, rows: u32) -> [u32; 3] {
    [dim / CHANNELS_PER_CTA, rows.div_ceil(WARPS * TOKENS), 1]
}
