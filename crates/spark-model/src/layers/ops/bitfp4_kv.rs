// SPDX-License-Identifier: AGPL-3.0-only

//! Native Blackwell (SM120/SM121) BitDecoding-FP4 + 6-op SWAR RaZeR-KV kernel
//! descriptors and launch wrappers.
//!
//! All 7 CUDA entrypoints live in `kernels/gb10/common/paged_decode_attn_bitfp4_sm120.cu`
//! (auto-discovered as module `"paged_decode_attn_bitfp4_sm120"` by `atlas-kernels`).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Auto-discovered module name for `kernels/gb10/common/paged_decode_attn_bitfp4_sm120.cu`.
pub const BITFP4_KV_MODULE: &str = "paged_decode_attn_bitfp4_sm120";

/// Head dimension supported by the SM120/SM121 BitDecoding-FP4 kernels.
pub const BITFP4_KV_HEAD_DIM: u32 = 256;

/// Paged KV block size aligned to `NVFP4_GROUP_SIZE = 16` and GateSight eviction.
pub const BITFP4_KV_BLOCK_SIZE: u32 = 16;

/// Query sub-tile width processed per CTA in single/multi-seq decode.
pub const BITFP4_DECODE_Q_SUB: u32 = 8;

/// Query sub-tile width processed per CTA in `K = gamma` multi-query verify.
pub const BITFP4_KGAMMA_Q_SUB: u32 = 16;

const BITFP4_FLAG_APPLY_WHT: u32 = 1 << 0;
const BITFP4_FLAG_FOUR_OVER_SIX: u32 = 1 << 1;
const BITFP4_FLAG_RAZER_KV: u32 = 1 << 2;

/// Whether orthogonal 256-point Walsh-Hadamard rotation (`K' = H(K - mu_k)`,
/// `Q' = H Q`) is applied to suppress channel outliers under token-wise K scaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitFp4WhtMode {
    Disabled,
    Enabled,
}

/// Block-16 scale selection strategy for K/V quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitFp4ScaleSearch {
    /// Standard NVFP4 `amax / 6` scale.
    StandardMax6,
    /// Four-Over-Six (`{amax / 6, amax / 4}`) MSE scale search.
    FourOverSix,
}

/// Whether 6-op branchless SWAR RaZeR-KV (`-0.0 -> ±4.0 + ±1.0`, `neg_sv` stored
/// in bit 7 of `.ue4m3` scale bytes) is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitFp4RazerMode {
    Disabled,
    Enabled,
}

/// Bitmask flags passed to the BitDecoding-FP4 KV write and decode kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitFp4QuantFlags {
    pub wht: BitFp4WhtMode,
    pub scale_search: BitFp4ScaleSearch,
    pub razer: BitFp4RazerMode,
}

impl BitFp4QuantFlags {
    /// Default BitDecoding-FP4 configuration: WHT + Four-Over-Six + optional RaZeR.
    #[must_use]
    pub const fn new(razer: BitFp4RazerMode) -> Self {
        Self {
            wht: BitFp4WhtMode::Enabled,
            scale_search: BitFp4ScaleSearch::FourOverSix,
            razer,
        }
    }

    /// Encode into the `u32` bitmask expected by `paged_decode_attn_bitfp4_sm120.cu`.
    #[must_use]
    pub const fn to_mask(self) -> u32 {
        let mut mask = 0u32;
        if matches!(self.wht, BitFp4WhtMode::Enabled) {
            mask |= BITFP4_FLAG_APPLY_WHT;
        }
        if matches!(self.scale_search, BitFp4ScaleSearch::FourOverSix) {
            mask |= BITFP4_FLAG_FOUR_OVER_SIX;
        }
        if matches!(self.razer, BitFp4RazerMode::Enabled) {
            mask |= BITFP4_FLAG_RAZER_KV;
        }
        mask
    }
}

/// Optional BF16 attention-sink sidecar buffers merged in split 0's prologue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitFp4Bf16Sink {
    pub k_sink_bf16: DevicePtr,
    pub v_sink_bf16: DevicePtr,
    pub num_sink_tokens: u32,
}

impl Default for BitFp4Bf16Sink {
    fn default() -> Self {
        Self {
            k_sink_bf16: DevicePtr::NULL,
            v_sink_bf16: DevicePtr::NULL,
            num_sink_tokens: 0,
        }
    }
}

/// Resolved kernel handles for the native SM120/SM121 BitDecoding-FP4 KV path.
#[derive(Clone, Copy, Debug)]
pub struct BitFp4KvKernels {
    pub decode: KernelHandle,
    pub decode_razer: KernelHandle,
    pub kgamma: KernelHandle,
    pub kgamma_razer: KernelHandle,
    pub reduce: KernelHandle,
    pub write_k: KernelHandle,
    pub promote_v: KernelHandle,
}

impl BitFp4KvKernels {
    /// Probe all 7 optional entrypoints from `"paged_decode_attn_bitfp4_sm120"`.
    /// Returns `None` if the base BitDecoding-FP4 set (write_k, promote_v, decode,
    /// kgamma, reduce) is unavailable.
    #[must_use]
    pub fn load_optional(gpu: &dyn GpuBackend) -> Option<Self> {
        let decode =
            crate::layers::try_kernel(gpu, BITFP4_KV_MODULE, "paged_decode_attn_bitfp4_sm120");
        let decode_razer = crate::layers::try_kernel(
            gpu,
            BITFP4_KV_MODULE,
            "paged_decode_attn_bitfp4_razer_sm120",
        );
        let kgamma = crate::layers::try_kernel(
            gpu,
            BITFP4_KV_MODULE,
            "paged_decode_attn_bitfp4_kgamma_sm120",
        );
        let kgamma_razer = crate::layers::try_kernel(
            gpu,
            BITFP4_KV_MODULE,
            "paged_decode_attn_bitfp4_kgamma_razer_sm120",
        );
        let reduce =
            crate::layers::try_kernel(gpu, BITFP4_KV_MODULE, "paged_decode_attn_bitfp4_reduce");
        let write_k =
            crate::layers::try_kernel(gpu, BITFP4_KV_MODULE, "reshape_and_cache_bitfp4_k_block16");
        let promote_v =
            crate::layers::try_kernel(gpu, BITFP4_KV_MODULE, "promote_block16_bitfp4_v_transposed");

        let handles = Self {
            decode,
            decode_razer,
            kgamma,
            kgamma_razer,
            reduce,
            write_k,
            promote_v,
        };
        if handles.is_complete_for(BitFp4RazerMode::Disabled) {
            Some(handles)
        } else {
            None
        }
    }

    /// Returns `true` when every required symbol for `razer` mode is non-null.
    #[must_use]
    pub const fn is_complete_for(&self, razer: BitFp4RazerMode) -> bool {
        let base_ok = self.write_k.0 != 0
            && self.promote_v.0 != 0
            && self.decode.0 != 0
            && self.kgamma.0 != 0
            && self.reduce.0 != 0;
        match razer {
            BitFp4RazerMode::Disabled => base_ok,
            BitFp4RazerMode::Enabled => {
                base_ok && self.decode_razer.0 != 0 && self.kgamma_razer.0 != 0
            }
        }
    }
}

/// Write `num_tokens` of K (token-wise `H(K - mu_k)` + 4/6 + optional RaZeR)
/// and V (self-contained block-16 channel-transposed `V^T`) into paged NVFP4
/// cache pools.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_bitfp4_kv_block16(
    gpu: &dyn GpuBackend,
    kernels: &BitFp4KvKernels,
    key: DevicePtr,
    value: DevicePtr,
    mu_k: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    key_stride: u32,
    value_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    flags: BitFp4QuantFlags,
    stream: u64,
) -> Result<()> {
    if num_tokens == 0 || num_kv_heads == 0 {
        return Ok(());
    }
    anyhow::ensure!(
        key_stride.is_multiple_of(8),
        "BitDecoding-FP4 K write requires 16-byte aligned key_stride (multiple of 8 BF16 elems, got {key_stride})"
    );
    let mask = flags.to_mask();

    // 1. Token-wise K write: 1 warp per (token, kv_head)
    KernelLaunch::new(gpu, kernels.write_k)
        .grid([num_tokens, num_kv_heads, 1])
        .block([32, 1, 1])
        .arg_ptr(key)
        .arg_ptr(mu_k)
        .arg_ptr(k_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(key_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .arg_u32(mask)
        .launch(stream)?;

    // 2. Race-free block-16 transposed V^T write: 1 CTA (256 threads = HDIM) per
    //    contiguous block-run; non-head tokens return immediately.
    KernelLaunch::new(gpu, kernels.promote_v)
        .grid([num_tokens, num_kv_heads, 1])
        .block([BITFP4_KV_HEAD_DIM, 1, 1])
        .arg_ptr(value)
        .arg_ptr(v_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_tokens)
        .arg_u32(num_kv_heads)
        .arg_u32(value_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .arg_u32(mask)
        .launch(stream)
}

/// Launch native SM120/SM121 BitDecoding-FP4 GQA-fused paged decode attention.
///
/// When `num_splits == 1`, writes final BF16 output directly to `output`.
/// When `num_splits > 1`, writes base-2 partial `(o[256], m, l)` to `workspace`
/// and the caller must follow with [`paged_decode_attn_bitfp4_reduce`].
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bitfp4_sm120(
    gpu: &dyn GpuBackend,
    kernels: &BitFp4KvKernels,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    workspace: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_seqs: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    inv_sqrt_d: f32,
    num_splits: u32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    flags: BitFp4QuantFlags,
    sink: BitFp4Bf16Sink,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        num_kv_heads > 0 && num_q_heads.is_multiple_of(num_kv_heads),
        "BitDecoding-FP4 requires num_q_heads ({num_q_heads}) divisible by num_kv_heads ({num_kv_heads})"
    );
    anyhow::ensure!(
        num_splits <= 1 || !workspace.is_null(),
        "BitDecoding-FP4 split-K (num_splits={num_splits}) requires a non-null workspace"
    );
    anyhow::ensure!(
        q_stride.is_multiple_of(8),
        "BitDecoding-FP4 decode requires 16-byte aligned q_stride (multiple of 8 BF16 elems, got {q_stride})"
    );
    if num_seqs == 0 {
        return Ok(());
    }
    let gqa_ratio = num_q_heads / num_kv_heads;
    let q_groups_per_kv = div_ceil(gqa_ratio, BITFP4_DECODE_Q_SUB);
    let grid_x = num_kv_heads
        .checked_mul(q_groups_per_kv)
        .ok_or_else(|| anyhow::anyhow!("BitDecoding-FP4 decode grid.x overflow"))?;
    let kernel = match flags.razer {
        BitFp4RazerMode::Disabled => kernels.decode,
        BitFp4RazerMode::Enabled => kernels.decode_razer,
    };

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, num_splits.max(1), num_seqs])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(workspace)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_f32(inv_sqrt_d)
        .arg_u32(num_splits.max(1))
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .arg_u32(flags.to_mask())
        .arg_ptr(sink.k_sink_bf16)
        .arg_ptr(sink.v_sink_bf16)
        .arg_u32(sink.num_sink_tokens)
        .launch(stream)
}

/// Launch native SM120/SM121 BitDecoding-FP4 `K = gamma` multi-query verify
/// attention (16 stacked `(q_row, q_head)` queries per CTA sharing one KV scan).
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bitfp4_kgamma_sm120(
    gpu: &dyn GpuBackend,
    kernels: &BitFp4KvKernels,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    workspace: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    inv_sqrt_d: f32,
    num_splits: u32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    num_qtile: u32,
    flags: BitFp4QuantFlags,
    sink: BitFp4Bf16Sink,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        num_kv_heads > 0 && num_q_heads.is_multiple_of(num_kv_heads),
        "BitDecoding-FP4 kgamma requires num_q_heads ({num_q_heads}) divisible by num_kv_heads ({num_kv_heads})"
    );
    anyhow::ensure!(
        num_splits <= 1 || !workspace.is_null(),
        "BitDecoding-FP4 kgamma split-K (num_splits={num_splits}) requires a non-null workspace"
    );
    anyhow::ensure!(
        q_stride.is_multiple_of(8),
        "BitDecoding-FP4 kgamma requires 16-byte aligned q_stride (multiple of 8 BF16 elems, got {q_stride})"
    );
    if num_qtile == 0 {
        return Ok(());
    }
    let gqa_ratio = num_q_heads / num_kv_heads;
    let total_q = num_qtile
        .checked_mul(gqa_ratio)
        .ok_or_else(|| anyhow::anyhow!("BitDecoding-FP4 kgamma total query count overflow"))?;
    let num_q_chunks = div_ceil(total_q, BITFP4_KGAMMA_Q_SUB);
    let kernel = match flags.razer {
        BitFp4RazerMode::Disabled => kernels.kgamma,
        BitFp4RazerMode::Enabled => kernels.kgamma_razer,
    };

    KernelLaunch::new(gpu, kernel)
        .grid([num_kv_heads, num_q_chunks, num_splits.max(1)])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(workspace)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_f32(inv_sqrt_d)
        .arg_u32(num_splits.max(1))
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .arg_u32(num_qtile)
        .arg_u32(flags.to_mask())
        .arg_ptr(sink.k_sink_bf16)
        .arg_ptr(sink.v_sink_bf16)
        .arg_u32(sink.num_sink_tokens)
        .launch(stream)
}

/// Base-2 LSE reduction combining `num_splits` BitDecoding-FP4 partial outputs
/// (`[num_rows, num_q_heads, num_splits, 258]` F32) into final BF16 `output`.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bitfp4_reduce(
    gpu: &dyn GpuBackend,
    reduce_kernel: KernelHandle,
    workspace: DevicePtr,
    output: DevicePtr,
    seq_lens: DevicePtr,
    num_q_heads: u32,
    num_splits: u32,
    num_rows: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        num_splits > 0 && !workspace.is_null(),
        "BitDecoding-FP4 reduce requires num_splits > 0 (got {num_splits}) and a non-null workspace"
    );
    if num_q_heads == 0 || num_rows == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, reduce_kernel)
        .grid([num_q_heads, num_rows, 1])
        .block([32, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_ptr(seq_lens)
        .arg_u32(num_q_heads)
        .arg_u32(num_splits)
        .launch(stream)
}
