// SPDX-License-Identifier: AGPL-3.0-only

//! Native Blackwell SM120/SM121 (with SM103a fallback) 1D GDN correction-record
//! verify, device-driven cross-layer replay commit, and tail-row zeroing ops.
//!
//! All 3 CUDA entrypoints live in
//! `kernels/gb10/common/gdn_correction_replay_sm120.cu` (auto-discovered by
//! `atlas-kernels` under module `"gdn_correction_replay_sm120"`):
//!
//! - `gdn_verify_record_1d_sm120`: reads FP32 `h_state` once (`L2::evict_first`)
//!   without mutating it, computes exact unroll-4 FP32 GDN outputs with optional
//!   explicit inline PTX `cvt.rn.bf16.f32` rounding boundaries for `beta` and
//!   `o = S q`, emits compact 1D correction records `(delta, k, decay)`
//!   (`63.75x`–`95.4x` smaller than 2D `h_state_intermediates`), and zeros
//!   inactive tail rows `[live_tokens, max_tokens)`.
//! - `gdn_commit_replay_1d_sm120`: single cross-layer GPU `_commit` launch over
//!   `[num_v_heads, batch_size, num_ssm_layers]` reading device-resident
//!   `d_num_accepted` (`0` host sync), enforcing fail-closed `Generation` and
//!   `sharing_count <= 1` descriptor guards on host and device, and replaying
//!   `H = decay * H + k (x) delta` in 0-ULP identical FP32 arithmetic order.
//! - `gdn_zero_tail_rows_sm120`: explicit standalone inactive tail-row zeroing
//!   `[live_tokens, max_tokens)` across any non-null subset of output/record
//!   buffers.
//!
//! Opt-in gate: `ATLAS_GDN_1D_REPLAY=1` (default OFF = bit-identical baseline,
//! strict `ZERO_OR_ONE` fail-closed parsing with `VarError::NotUnicode` warned
//! and forced OFF).

use std::collections::HashSet;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Auto-discovered `atlas-kernels` module name for
/// `kernels/gb10/common/gdn_correction_replay_sm120.cu`.
pub const GDN_1D_REPLAY_MODULE: &str = "gdn_correction_replay_sm120";

/// Environment variable gating the opt-in 1D GDN correction-record verify &
/// cross-layer replay commit path.
pub const GDN_1D_REPLAY_ENV_VAR: &str = "ATLAS_GDN_1D_REPLAY";

/// Fixed GDN key/value head dimension supported by the SM120/SM121 1D replay kernels.
pub const GDN_1D_HEAD_DIM: u32 = 128;

/// Verify kernel flag: compute `beta = sigmoid(raw_b)` from raw logits.
pub const GDN_1D_FLAG_BETA_FROM_RAW_LOGIT: u32 = 1;

/// Verify kernel flag: apply explicit inline PTX `cvt.rn.bf16.f32` rounding to `beta`.
pub const GDN_1D_FLAG_ROUND_BETA_BF16: u32 = 2;

/// Verify kernel flag: apply explicit inline PTX `cvt.rn.bf16.f32` rounding to
/// `o = (S q) * inv_sqrt_d` before downstream FP32 RMSNorm.
pub const GDN_1D_FLAG_ROUND_OUT_BF16: u32 = 4;

/// Commit status bitmask: all descriptors valid.
pub const GDN_1D_STATUS_OK: u32 = 0;

/// Commit status fault bit: null pointer in [`GdnCommitSlotDesc`].
pub const GDN_1D_STATUS_NULL_PTR: u32 = 1;

/// Commit status fault bit: `generation != expected_generation` (stale record).
pub const GDN_1D_STATUS_GEN_MISMATCH: u32 = 2;

/// Commit status fault bit: `sharing_count > 1` (shared/aliased slot).
pub const GDN_1D_STATUS_SHARED_SLOT: u32 = 4;

/// Commit status fault bit: `d_num_accepted[seq] > desc.max_tokens`.
pub const GDN_1D_STATUS_ACCEPTED_OOB: u32 = 8;

/// Commit status fault bit: bad geometry, alignment, or nonzero reserved flags.
pub const GDN_1D_STATUS_BAD_GEOMETRY: u32 = 16;

/// Exact IEEE-754 single-precision `1.0f / sqrtf(128.0f)` (`0x3db504f3`).
pub const GDN_1D_INV_SQRT_128: f32 = 0.088_388_346_f32;

const FP32_BYTES: usize = 4;
const H_HEAD_ELEMS: usize = (GDN_1D_HEAD_DIM as usize) * (GDN_1D_HEAD_DIM as usize);

/// Runtime policy resolved from `ATLAS_GDN_1D_REPLAY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Gdn1dReplayPolicy {
    /// Default OFF: preserve the baseline GDN verify & commit path bit-identically.
    Disabled,
    /// Explicit `ATLAS_GDN_1D_REPLAY=1`: emit 1D correction records during verify
    /// and replay in-place on device-driven cross-layer commit; fail closed when
    /// incompatible.
    Enabled,
}

/// How the `beta` buffer passed to `gdn_verify_record_1d_sm120` is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GdnBetaInputMode {
    /// `beta` already contains post-sigmoid FP32 values (Flux `gates_buf` layout).
    PrecomputedBeta,
    /// `beta` contains raw pre-sigmoid logits `b` and the kernel computes
    /// `1.0 / (1.0 + exp(-b))`.
    RawLogitSigmoid,
}

/// Rounding discipline applied to `beta` inside `gdn_verify_record_1d_sm120`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GdnBetaRoundingMode {
    /// Preserve exact FP32 `beta` (0-ULP match with Flux FP32 sequence verify).
    ExactFp32,
    /// Explicitly round `beta` through inline PTX `cvt.rn.bf16.f32` to match
    /// BF16-bounded prefill numerics without `fp_contract` FMA folding.
    RoundToBf16Ptx,
}

/// Rounding discipline applied to `o = (S q) * inv_sqrt_d` before FP32 RMSNorm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GdnOutputRoundingMode {
    /// Emit exact FP32 `o` (0-ULP match with Flux FP32 sequence verify).
    ExactFp32,
    /// Explicitly round `o` through inline PTX `cvt.rn.bf16.f32` before writing
    /// FP32 output for downstream FP32 RMSNorm (exact prefill-vs-verify parity).
    RoundToBf16Ptx,
}

/// Numeric boundary configuration for `gdn_verify_record_1d_sm120`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GdnVerifyNumericConfig {
    pub beta_input: GdnBetaInputMode,
    pub beta_rounding: GdnBetaRoundingMode,
    pub output_rounding: GdnOutputRoundingMode,
}

impl GdnVerifyNumericConfig {
    /// Pure FP32 mode: 0-ULP bit-exact with Flux's FP32 sequence verify path.
    #[must_use]
    pub const fn exact_fp32() -> Self {
        Self {
            beta_input: GdnBetaInputMode::PrecomputedBeta,
            beta_rounding: GdnBetaRoundingMode::ExactFp32,
            output_rounding: GdnOutputRoundingMode::ExactFp32,
        }
    }

    /// Prefill BF16-boundary parity mode: explicit `cvt.rn.bf16.f32` rounding
    /// for `beta = sigmoid(raw_b)` and `o = S q` before FP32 RMSNorm.
    #[must_use]
    pub const fn prefill_bf16_parity(beta_input: GdnBetaInputMode) -> Self {
        Self {
            beta_input,
            beta_rounding: GdnBetaRoundingMode::RoundToBf16Ptx,
            output_rounding: GdnOutputRoundingMode::RoundToBf16Ptx,
        }
    }

    /// Encode into the CUDA kernel `flags` bitmask.
    #[must_use]
    pub const fn to_kernel_flags(self) -> u32 {
        let mut flags = 0u32;
        if matches!(self.beta_input, GdnBetaInputMode::RawLogitSigmoid) {
            flags |= GDN_1D_FLAG_BETA_FROM_RAW_LOGIT;
        }
        if matches!(self.beta_rounding, GdnBetaRoundingMode::RoundToBf16Ptx) {
            flags |= GDN_1D_FLAG_ROUND_BETA_BF16;
        }
        if matches!(self.output_rounding, GdnOutputRoundingMode::RoundToBf16Ptx) {
            flags |= GDN_1D_FLAG_ROUND_OUT_BF16;
        }
        flags
    }
}

/// Fail-closed status bitmask emitted by `gdn_commit_replay_1d_sm120`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GdnCommitStatus(pub u32);

impl GdnCommitStatus {
    /// True when no fail-closed fault bits were set.
    #[must_use]
    pub const fn is_ok(self) -> bool {
        self.0 == GDN_1D_STATUS_OK
    }

    /// Raw status bitmask value.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// True when all bits in `flag` are set.
    #[must_use]
    pub const fn has_flag(self, flag: u32) -> bool {
        (self.0 & flag) == flag
    }

    /// Return `Ok(())` if `is_ok()`, or a detailed fail-closed error listing every
    /// triggered fault bit.
    ///
    /// # Errors
    /// Returns an error if any `GDN_1D_STATUS_*` fault bit is set.
    pub fn ensure_ok(self) -> Result<()> {
        if self.is_ok() {
            return Ok(());
        }
        let mut reasons = Vec::new();
        if self.has_flag(GDN_1D_STATUS_NULL_PTR) {
            reasons.push("NULL_PTR");
        }
        if self.has_flag(GDN_1D_STATUS_GEN_MISMATCH) {
            reasons.push("GEN_MISMATCH");
        }
        if self.has_flag(GDN_1D_STATUS_SHARED_SLOT) {
            reasons.push("SHARED_SLOT");
        }
        if self.has_flag(GDN_1D_STATUS_ACCEPTED_OOB) {
            reasons.push("ACCEPTED_OOB");
        }
        if self.has_flag(GDN_1D_STATUS_BAD_GEOMETRY) {
            reasons.push("BAD_GEOMETRY");
        }
        bail!(
            "gdn_commit_replay_1d_sm120 rejected commit fail-closed (status={:#x}: {})",
            self.0,
            reasons.join("|")
        );
    }
}

/// Parse an `ATLAS_GDN_1D_REPLAY` environment lookup using fail-closed
/// `ZERO_OR_ONE` rules.
///
/// - Unset (`NotPresent`) or `"0"` -> `Ok(Disabled)`.
/// - `"1"` -> `Ok(Enabled)`.
/// - `NotUnicode` -> emits `tracing::warn!` and returns `Err(...)` (fail closed,
///   matching `recipe_contract` non-UTF-8 refusal).
/// - Any other UTF-8 value -> returns `Err(...)` (fail closed).
///
/// # Errors
/// Returns an error if the variable is non-UTF-8 or set to any UTF-8 value
/// other than `"0"` or `"1"`.
pub fn parse_gdn_1d_replay_env(
    value: std::result::Result<String, std::env::VarError>,
) -> Result<Gdn1dReplayPolicy> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(Gdn1dReplayPolicy::Disabled),
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                env = GDN_1D_REPLAY_ENV_VAR,
                "ATLAS_GDN_1D_REPLAY is not valid UTF-8; refusing fail-closed"
            );
            bail!("{GDN_1D_REPLAY_ENV_VAR} must be valid UTF-8 and exactly \"0\" or \"1\"")
        }
        Ok(raw) => match raw.as_str() {
            "0" => Ok(Gdn1dReplayPolicy::Disabled),
            "1" => Ok(Gdn1dReplayPolicy::Enabled),
            _ => bail!(
                "{GDN_1D_REPLAY_ENV_VAR} must be exactly \"0\" or \"1\" (fail-closed ZERO_OR_ONE), got {raw:?}"
            ),
        },
    }
}

/// Read and cache `ATLAS_GDN_1D_REPLAY` fail-closed.
///
/// # Errors
/// Returns an error if `ATLAS_GDN_1D_REPLAY` is set to an invalid UTF-8 value
/// other than `"0"` or `"1"`.
pub fn gdn_1d_replay_policy() -> Result<Gdn1dReplayPolicy> {
    static GATE: OnceLock<std::result::Result<Gdn1dReplayPolicy, String>> = OnceLock::new();
    match GATE.get_or_init(|| {
        parse_gdn_1d_replay_env(std::env::var(GDN_1D_REPLAY_ENV_VAR)).map_err(|e| e.to_string())
    }) {
        Ok(policy) => Ok(*policy),
        Err(msg) => bail!("{msg}"),
    }
}

/// ABI-pinned 64-byte per-`(layer, seq)` commit descriptor shared with
/// `GdnCommitSlotDesc` in `gdn_correction_replay_sm120.cu`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GdnCommitSlotDesc {
    /// Device pointer to FP32 `h_state` `[num_v_heads, 128, 128]` (16-byte aligned).
    pub h_state_ptr: u64,
    /// Device pointer to FP32 1D `delta` records `[max_tokens, num_v_heads, 128]`.
    pub delta_ptr: u64,
    /// Device pointer to FP32 1D `k` records `[max_tokens, num_k_heads, 128]`.
    pub k_ptr: u64,
    /// Device pointer to FP32 1D `decay` records `[max_tokens, num_v_heads]`.
    pub decay_ptr: u64,
    /// Live slot generation counter at commit time.
    pub generation: u64,
    /// Generation counter captured when the 1D correction record was written.
    pub expected_generation: u64,
    /// Active reference/sharing count on `h_state_ptr`; must be `<= 1`.
    pub sharing_count: u32,
    /// Verify window capacity `W`; `d_num_accepted[seq]` must satisfy `<= max_tokens`.
    pub max_tokens: u32,
    /// Reserved descriptor flags; must be `0`.
    pub flags: u32,
    /// Explicit alignment padding; must be `0`.
    pub pad: u32,
}

const _: () = {
    assert!(std::mem::size_of::<GdnCommitSlotDesc>() == 64);
    assert!(std::mem::align_of::<GdnCommitSlotDesc>() == 8);
};

impl GdnCommitSlotDesc {
    /// Byte size of one serialized [`GdnCommitSlotDesc`] matching the CUDA ABI.
    pub const ABI_BYTES: usize = 64;

    /// Serialize this descriptor into exact little-endian bytes for H2D upload
    /// without `unsafe` pointer transmutation.
    #[must_use]
    pub fn to_le_bytes(self) -> [u8; Self::ABI_BYTES] {
        let mut out = [0u8; Self::ABI_BYTES];
        out[0..8].copy_from_slice(&self.h_state_ptr.to_le_bytes());
        out[8..16].copy_from_slice(&self.delta_ptr.to_le_bytes());
        out[16..24].copy_from_slice(&self.k_ptr.to_le_bytes());
        out[24..32].copy_from_slice(&self.decay_ptr.to_le_bytes());
        out[32..40].copy_from_slice(&self.generation.to_le_bytes());
        out[40..48].copy_from_slice(&self.expected_generation.to_le_bytes());
        out[48..52].copy_from_slice(&self.sharing_count.to_le_bytes());
        out[52..56].copy_from_slice(&self.max_tokens.to_le_bytes());
        out[56..60].copy_from_slice(&self.flags.to_le_bytes());
        out[60..64].copy_from_slice(&self.pad.to_le_bytes());
        out
    }

    /// Deserialize a 64-byte little-endian ABI record back into [`Self`].
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8; Self::ABI_BYTES]) -> Self {
        let read_u64 = |off: usize| -> u64 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[off..off + 8]);
            u64::from_le_bytes(buf)
        };
        let read_u32 = |off: usize| -> u32 {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[off..off + 4]);
            u32::from_le_bytes(buf)
        };
        Self {
            h_state_ptr: read_u64(0),
            delta_ptr: read_u64(8),
            k_ptr: read_u64(16),
            decay_ptr: read_u64(24),
            generation: read_u64(32),
            expected_generation: read_u64(40),
            sharing_count: read_u32(48),
            max_tokens: read_u32(52),
            flags: read_u32(56),
            pad: read_u32(60),
        }
    }
}

/// Checked geometry and arena byte layout for 1D GDN correction records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Gdn1dRecordGeometry {
    pub batch_size: u32,
    pub max_tokens: u32,
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub delta_bytes: usize,
    pub k_bytes: usize,
    pub decay_bytes: usize,
    pub k_offset_bytes: usize,
    pub decay_offset_bytes: usize,
    pub total_arena_bytes: usize,
    pub legacy_2d_h_inter_bytes: usize,
}

fn align_up_16(value: usize) -> Result<usize> {
    value
        .checked_add(15)
        .map(|v| v & !15)
        .context("1D GDN correction-record alignment overflow")
}

impl Gdn1dRecordGeometry {
    /// Validate `(batch_size, max_tokens, num_k_heads, num_v_heads)` and compute
    /// exact 16-byte-aligned arena offsets and memory savings vs 2D `h_inter`.
    ///
    /// # Errors
    /// Returns an error if any dimension is zero, `num_v_heads` is not divisible
    /// by `num_k_heads`, or any byte calculation overflows `usize`.
    pub fn try_new(
        batch_size: u32,
        max_tokens: u32,
        num_k_heads: u32,
        num_v_heads: u32,
    ) -> Result<Self> {
        ensure!(batch_size > 0, "GDN 1D replay requires batch_size > 0");
        ensure!(max_tokens > 0, "GDN 1D replay requires max_tokens > 0");
        ensure!(num_k_heads > 0, "GDN 1D replay requires num_k_heads > 0");
        ensure!(num_v_heads > 0, "GDN 1D replay requires num_v_heads > 0");
        ensure!(
            num_v_heads.is_multiple_of(num_k_heads),
            "GDN 1D replay requires num_v_heads ({num_v_heads}) divisible by num_k_heads ({num_k_heads})"
        );

        let total_rows = (batch_size as usize)
            .checked_mul(max_tokens as usize)
            .context("GDN 1D replay total_rows overflow")?;
        let head_dim = GDN_1D_HEAD_DIM as usize;

        let delta_bytes = total_rows
            .checked_mul(num_v_heads as usize)
            .and_then(|v| v.checked_mul(head_dim))
            .and_then(|v| v.checked_mul(FP32_BYTES))
            .context("GDN 1D replay delta_bytes overflow")?;
        let k_bytes = total_rows
            .checked_mul(num_k_heads as usize)
            .and_then(|v| v.checked_mul(head_dim))
            .and_then(|v| v.checked_mul(FP32_BYTES))
            .context("GDN 1D replay k_bytes overflow")?;
        let decay_bytes = total_rows
            .checked_mul(num_v_heads as usize)
            .and_then(|v| v.checked_mul(FP32_BYTES))
            .context("GDN 1D replay decay_bytes overflow")?;

        let k_offset_bytes = align_up_16(delta_bytes)?;
        let decay_offset_bytes = align_up_16(
            k_offset_bytes
                .checked_add(k_bytes)
                .context("GDN 1D replay decay offset overflow")?,
        )?;
        let total_arena_bytes = align_up_16(
            decay_offset_bytes
                .checked_add(decay_bytes)
                .context("GDN 1D replay total arena byte overflow")?,
        )?;

        let legacy_2d_h_inter_bytes = total_rows
            .checked_mul(num_v_heads as usize)
            .and_then(|v| v.checked_mul(H_HEAD_ELEMS))
            .and_then(|v| v.checked_mul(FP32_BYTES))
            .context("GDN 2D h_inter comparison byte overflow")?;

        Ok(Self {
            batch_size,
            max_tokens,
            num_k_heads,
            num_v_heads,
            delta_bytes,
            k_bytes,
            decay_bytes,
            k_offset_bytes,
            decay_offset_bytes,
            total_arena_bytes,
            legacy_2d_h_inter_bytes,
        })
    }

    /// Exact logical payload bytes `(delta + k + decay)` without arena alignment padding.
    #[must_use]
    pub const fn logical_record_bytes(&self) -> usize {
        self.delta_bytes + self.k_bytes + self.decay_bytes
    }

    /// Compression ratio of 2D `h_state_intermediates` over 1D correction records
    /// (`>= 63.75x` for MHA and `~95.4x` for Qwen3.8 GQA 16:48).
    #[must_use]
    pub fn compression_ratio_vs_2d_inter(&self) -> f64 {
        (self.legacy_2d_h_inter_bytes as f64) / (self.logical_record_bytes() as f64)
    }

    /// Compute the per-sequence byte offsets `(delta_off, k_off, decay_off)`
    /// relative to the respective base pointers for batch slot `seq`.
    ///
    /// # Errors
    /// Returns an error if `seq >= self.batch_size`.
    pub fn seq_record_offsets(&self, seq: u32) -> Result<(usize, usize, usize)> {
        ensure!(
            seq < self.batch_size,
            "sequence index {seq} out of range for batch_size {}",
            self.batch_size
        );
        let rows_before = (seq as usize) * (self.max_tokens as usize);
        let head_dim = GDN_1D_HEAD_DIM as usize;
        let delta_off = rows_before * (self.num_v_heads as usize) * head_dim * FP32_BYTES;
        let k_off = rows_before * (self.num_k_heads as usize) * head_dim * FP32_BYTES;
        let decay_off = rows_before * (self.num_v_heads as usize) * FP32_BYTES;
        Ok((delta_off, k_off, decay_off))
    }

    /// Compute the exact byte size of FP32 `h_state` for one batch (`[B, nv, 128, 128]`).
    ///
    /// # Errors
    /// Returns an error on arithmetic overflow.
    pub fn h_state_bytes(&self) -> Result<usize> {
        (self.batch_size as usize)
            .checked_mul(self.num_v_heads as usize)
            .and_then(|v| v.checked_mul(H_HEAD_ELEMS))
            .and_then(|v| v.checked_mul(FP32_BYTES))
            .context("GDN h_state byte size overflow")
    }
}

/// Resolved CUDA kernel handles for `gdn_correction_replay_sm120.cu`.
#[derive(Debug, Clone, Copy)]
pub struct GdnCorrectionReplayKernels {
    pub verify_record_1d: KernelHandle,
    pub commit_replay_1d: KernelHandle,
    pub zero_tail_rows: KernelHandle,
}

impl GdnCorrectionReplayKernels {
    /// Probe all 3 optional entrypoints from `"gdn_correction_replay_sm120"`.
    #[must_use]
    pub fn load_optional(gpu: &dyn GpuBackend) -> Self {
        Self {
            verify_record_1d: crate::layers::try_kernel(
                gpu,
                GDN_1D_REPLAY_MODULE,
                "gdn_verify_record_1d_sm120",
            ),
            commit_replay_1d: crate::layers::try_kernel(
                gpu,
                GDN_1D_REPLAY_MODULE,
                "gdn_commit_replay_1d_sm120",
            ),
            zero_tail_rows: crate::layers::try_kernel(
                gpu,
                GDN_1D_REPLAY_MODULE,
                "gdn_zero_tail_rows_sm120",
            ),
        }
    }

    /// True when `gdn_verify_record_1d_sm120` is loaded.
    #[must_use]
    pub const fn has_verify_record_1d(&self) -> bool {
        self.verify_record_1d.0 != 0
    }

    /// True when `gdn_commit_replay_1d_sm120` is loaded.
    #[must_use]
    pub const fn has_commit_replay_1d(&self) -> bool {
        self.commit_replay_1d.0 != 0
    }

    /// True when `gdn_zero_tail_rows_sm120` is loaded.
    #[must_use]
    pub const fn has_zero_tail_rows(&self) -> bool {
        self.zero_tail_rows.0 != 0
    }

    /// True when all 3 SM120/SM121 1D replay entrypoints are loaded.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.has_verify_record_1d() && self.has_commit_replay_1d() && self.has_zero_tail_rows()
    }
}

fn ensure_aligned_non_null(ptr: DevicePtr, align: u64, label: &str) -> Result<()> {
    ensure!(!ptr.is_null(), "GDN 1D replay {label} pointer is null");
    ensure!(
        ptr.0.is_multiple_of(align),
        "GDN 1D replay {label} pointer ({:#x}) must be {align}-byte aligned",
        ptr.0
    );
    Ok(())
}

fn ensure_ranges_disjoint(
    a_ptr: DevicePtr,
    a_bytes: usize,
    a_label: &str,
    b_ptr: DevicePtr,
    b_bytes: usize,
    b_label: &str,
) -> Result<()> {
    if a_bytes == 0 || b_bytes == 0 {
        return Ok(());
    }
    let a_end = a_ptr
        .0
        .checked_add(a_bytes as u64)
        .with_context(|| format!("address range overflow for {a_label}"))?;
    let b_end = b_ptr
        .0
        .checked_add(b_bytes as u64)
        .with_context(|| format!("address range overflow for {b_label}"))?;
    let disjoint = a_end <= b_ptr.0 || b_end <= a_ptr.0;
    ensure!(
        disjoint,
        "GDN 1D replay buffers {a_label} ([{:#x}, {:#x})) and {b_label} ([{:#x}, {:#x})) overlap",
        a_ptr.0,
        a_end,
        b_ptr.0,
        b_end
    );
    Ok(())
}

/// Host-side fail-closed validation of `[num_ssm_layers * batch_size]` commit
/// descriptors prior to launching `gdn_commit_replay_1d_sm120`.
///
/// Enforces:
/// - Exact descriptor count `num_ssm_layers * batch_size` (`> 0`).
/// - Non-null, 16-byte-aligned `h_state_ptr`, non-null 4-byte-aligned record pointers.
/// - `generation == expected_generation`.
/// - `sharing_count <= 1`.
/// - `max_tokens > 0`, `flags == 0`, `pad == 0`.
/// - Strict pairwise distinctness of `h_state_ptr` across batch slots within each
///   SSM layer (prevents duplicate-slot aliasing even if `sharing_count` was not
///   incremented).
///
/// # Errors
/// Returns an error on the first violated fail-closed invariant.
pub fn validate_commit_slot_descs(
    descs: &[GdnCommitSlotDesc],
    num_ssm_layers: u32,
    batch_size: u32,
) -> Result<()> {
    ensure!(
        num_ssm_layers > 0 && batch_size > 0,
        "GDN 1D commit requires num_ssm_layers > 0 and batch_size > 0"
    );
    let expected_len = (num_ssm_layers as usize)
        .checked_mul(batch_size as usize)
        .context("GDN 1D commit descriptor count overflow")?;
    ensure!(
        descs.len() == expected_len,
        "GDN 1D commit expected {expected_len} descriptors (layers={num_ssm_layers} x batch={batch_size}), got {}",
        descs.len()
    );

    // Check pairwise distinctness across ALL (layer, seq) slots: the commit
    // grid `(num_v_heads, batch_size, num_ssm_layers)` updates every slot in
    // parallel, so either intra-layer batch aliasing or cross-layer aliasing of
    // `h_state_ptr` would cause a concurrent GPU data race.
    let mut seen_h_ptrs = HashSet::with_capacity(expected_len);
    for layer in 0..num_ssm_layers as usize {
        for seq in 0..batch_size as usize {
            let desc = &descs[layer * (batch_size as usize) + seq];
            ensure_aligned_non_null(
                DevicePtr(desc.h_state_ptr),
                16,
                &format!("desc[layer={layer}, seq={seq}].h_state_ptr"),
            )?;
            ensure_aligned_non_null(
                DevicePtr(desc.delta_ptr),
                4,
                &format!("desc[layer={layer}, seq={seq}].delta_ptr"),
            )?;
            ensure_aligned_non_null(
                DevicePtr(desc.k_ptr),
                4,
                &format!("desc[layer={layer}, seq={seq}].k_ptr"),
            )?;
            ensure_aligned_non_null(
                DevicePtr(desc.decay_ptr),
                4,
                &format!("desc[layer={layer}, seq={seq}].decay_ptr"),
            )?;
            ensure!(
                desc.generation == desc.expected_generation,
                "GDN 1D commit generation mismatch at layer={layer}, seq={seq}: live={} != expected={}",
                desc.generation,
                desc.expected_generation
            );
            ensure!(
                desc.sharing_count <= 1,
                "GDN 1D commit refused shared slot at layer={layer}, seq={seq}: sharing_count={} (> 1)",
                desc.sharing_count
            );
            ensure!(
                desc.max_tokens > 0,
                "GDN 1D commit requires max_tokens > 0 at layer={layer}, seq={seq}"
            );
            ensure!(
                desc.flags == 0 && desc.pad == 0,
                "GDN 1D commit requires reserved flags=0 and pad=0 at layer={layer}, seq={seq}"
            );
            ensure!(
                seen_h_ptrs.insert(desc.h_state_ptr),
                "GDN 1D commit detected duplicate/aliased h_state_ptr {:#x} at layer={layer}, seq={seq}",
                desc.h_state_ptr
            );
        }
    }
    Ok(())
}

/// Launch parameters for [`launch_gdn_verify_record_1d_sm120`].
#[derive(Debug, Clone, Copy)]
pub struct GdnVerifyRecord1dLaunchArgs {
    pub h_state: DevicePtr,
    pub query: DevicePtr,
    pub key: DevicePtr,
    pub value: DevicePtr,
    pub gate: DevicePtr,
    pub beta: DevicePtr,
    pub output: DevicePtr,
    pub delta_out: DevicePtr,
    pub k_out: DevicePtr,
    pub decay_out: DevicePtr,
    pub geometry: Gdn1dRecordGeometry,
    pub live_tokens: u32,
    pub qk_stride: u32,
    pub v_stride: u32,
    pub gb_stride: u32,
    pub out_stride: u32,
    pub numeric_config: GdnVerifyNumericConfig,
}

/// Validate all pointers, strides, alignment, and non-aliasing invariants, then
/// launch `gdn_verify_record_1d_sm120`.
///
/// # Errors
/// Returns an error without enqueuing a kernel if any handle, pointer, stride,
/// alignment, or disjointness precondition fails.
pub fn launch_gdn_verify_record_1d_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &GdnVerifyRecord1dLaunchArgs,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "gdn_verify_record_1d_sm120 kernel handle is null"
    );
    let geom = args.geometry;
    // Re-validate geometry in case caller constructed fields manually.
    let checked = Gdn1dRecordGeometry::try_new(
        geom.batch_size,
        geom.max_tokens,
        geom.num_k_heads,
        geom.num_v_heads,
    )?;
    ensure!(
        checked == geom,
        "GdnVerifyRecord1dLaunchArgs geometry fields are inconsistent with try_new"
    );
    ensure!(
        args.live_tokens <= geom.max_tokens,
        "gdn_verify_record_1d_sm120 live_tokens ({}) exceeds max_tokens ({})",
        args.live_tokens,
        geom.max_tokens
    );

    let min_qk = geom.num_k_heads.saturating_mul(GDN_1D_HEAD_DIM);
    let min_v = geom.num_v_heads.saturating_mul(GDN_1D_HEAD_DIM);
    ensure!(
        args.qk_stride >= min_qk,
        "gdn_verify_record_1d_sm120 qk_stride ({}) < num_k_heads * 128 ({min_qk})",
        args.qk_stride
    );
    ensure!(
        args.v_stride >= min_v,
        "gdn_verify_record_1d_sm120 v_stride ({}) < num_v_heads * 128 ({min_v})",
        args.v_stride
    );
    ensure!(
        args.gb_stride >= geom.num_v_heads,
        "gdn_verify_record_1d_sm120 gb_stride ({}) < num_v_heads ({})",
        args.gb_stride,
        geom.num_v_heads
    );
    ensure!(
        args.out_stride >= min_v,
        "gdn_verify_record_1d_sm120 out_stride ({}) < num_v_heads * 128 ({min_v})",
        args.out_stride
    );

    ensure_aligned_non_null(args.h_state, 16, "h_state")?;
    ensure_aligned_non_null(args.query, 4, "query")?;
    ensure_aligned_non_null(args.key, 4, "key")?;
    ensure_aligned_non_null(args.value, 4, "value")?;
    ensure_aligned_non_null(args.gate, 4, "gate")?;
    ensure_aligned_non_null(args.beta, 4, "beta")?;
    ensure_aligned_non_null(args.output, 4, "output")?;
    ensure_aligned_non_null(args.delta_out, 4, "delta_out")?;
    ensure_aligned_non_null(args.k_out, 4, "k_out")?;
    ensure_aligned_non_null(args.decay_out, 4, "decay_out")?;

    let h_bytes = geom.h_state_bytes()?;
    let total_rows = (geom.batch_size as usize)
        .checked_mul(geom.max_tokens as usize)
        .context("verify output rows overflow")?;
    let out_span_bytes = total_rows
        .checked_mul(args.out_stride as usize)
        .and_then(|v| v.checked_mul(FP32_BYTES))
        .context("verify output byte span overflow")?;

    ensure_ranges_disjoint(
        args.h_state,
        h_bytes,
        "h_state",
        args.output,
        out_span_bytes,
        "output",
    )?;
    ensure_ranges_disjoint(
        args.h_state,
        h_bytes,
        "h_state",
        args.delta_out,
        geom.delta_bytes,
        "delta_out",
    )?;
    ensure_ranges_disjoint(
        args.h_state,
        h_bytes,
        "h_state",
        args.k_out,
        geom.k_bytes,
        "k_out",
    )?;
    ensure_ranges_disjoint(
        args.h_state,
        h_bytes,
        "h_state",
        args.decay_out,
        geom.decay_bytes,
        "decay_out",
    )?;
    ensure_ranges_disjoint(
        args.delta_out,
        geom.delta_bytes,
        "delta_out",
        args.k_out,
        geom.k_bytes,
        "k_out",
    )?;
    ensure_ranges_disjoint(
        args.delta_out,
        geom.delta_bytes,
        "delta_out",
        args.decay_out,
        geom.decay_bytes,
        "decay_out",
    )?;
    ensure_ranges_disjoint(
        args.k_out,
        geom.k_bytes,
        "k_out",
        args.decay_out,
        geom.decay_bytes,
        "decay_out",
    )?;
    ensure_ranges_disjoint(
        args.output,
        out_span_bytes,
        "output",
        args.delta_out,
        geom.delta_bytes,
        "delta_out",
    )?;
    ensure_ranges_disjoint(
        args.output,
        out_span_bytes,
        "output",
        args.k_out,
        geom.k_bytes,
        "k_out",
    )?;
    ensure_ranges_disjoint(
        args.output,
        out_span_bytes,
        "output",
        args.decay_out,
        geom.decay_bytes,
        "decay_out",
    )?;

    KernelLaunch::new(gpu, kernel)
        .grid([geom.num_v_heads, geom.batch_size, 1])
        .block([GDN_1D_HEAD_DIM, 1, 1])
        .arg_ptr(args.h_state)
        .arg_ptr(args.query)
        .arg_ptr(args.key)
        .arg_ptr(args.value)
        .arg_ptr(args.gate)
        .arg_ptr(args.beta)
        .arg_ptr(args.output)
        .arg_ptr(args.delta_out)
        .arg_ptr(args.k_out)
        .arg_ptr(args.decay_out)
        .arg_u32(geom.batch_size)
        .arg_u32(args.live_tokens)
        .arg_u32(geom.max_tokens)
        .arg_u32(geom.num_k_heads)
        .arg_u32(geom.num_v_heads)
        .arg_u32(args.qk_stride)
        .arg_u32(args.v_stride)
        .arg_u32(args.gb_stride)
        .arg_u32(args.out_stride)
        .arg_u32(args.numeric_config.to_kernel_flags())
        .launch(stream)
}

/// Launch parameters for [`launch_gdn_commit_replay_1d_sm120`].
#[derive(Debug, Clone, Copy)]
pub struct GdnCommitReplay1dLaunchArgs<'a> {
    /// Device buffer holding `[num_ssm_layers * batch_size]` [`GdnCommitSlotDesc`] entries.
    pub descs_dev: DevicePtr,
    /// Optional host copy of the descriptor table for pre-launch fail-closed validation.
    pub host_descs: Option<&'a [GdnCommitSlotDesc]>,
    /// Device buffer holding `[batch_size]` `u32` accepted token counts (`0` host sync).
    pub d_num_accepted: DevicePtr,
    /// Optional device `u32` fault status word (`atomicOr` on failure); pass `None` if unused.
    pub status_out: Option<DevicePtr>,
    pub num_ssm_layers: u32,
    pub batch_size: u32,
    pub num_k_heads: u32,
    pub num_v_heads: u32,
}

/// Pre-launch fail-closed validation of [`GdnCommitReplay1dLaunchArgs`] (usable
/// before H2D descriptor upload so invalid launches perform zero device writes).
///
/// # Errors
/// Returns an error if the kernel handle, geometry, pointers, alignment,
/// disjointness, or host descriptors fail validation.
pub fn validate_commit_replay_1d_launch_args(
    kernel: KernelHandle,
    args: &GdnCommitReplay1dLaunchArgs<'_>,
) -> Result<DevicePtr> {
    ensure!(
        kernel.0 != 0,
        "gdn_commit_replay_1d_sm120 kernel handle is null"
    );
    ensure!(
        args.num_ssm_layers > 0 && args.batch_size > 0,
        "gdn_commit_replay_1d_sm120 requires num_ssm_layers > 0 and batch_size > 0"
    );
    ensure!(
        args.num_k_heads > 0
            && args.num_v_heads > 0
            && args.num_v_heads.is_multiple_of(args.num_k_heads),
        "gdn_commit_replay_1d_sm120 requires num_v_heads ({}) divisible by num_k_heads ({}) > 0",
        args.num_v_heads,
        args.num_k_heads
    );
    ensure_aligned_non_null(args.descs_dev, 8, "descs_dev")?;
    ensure_aligned_non_null(args.d_num_accepted, 4, "d_num_accepted")?;

    let desc_bytes = (args.num_ssm_layers as usize)
        .checked_mul(args.batch_size as usize)
        .and_then(|v| v.checked_mul(GdnCommitSlotDesc::ABI_BYTES))
        .context("gdn_commit_replay_1d_sm120 descriptor table byte size overflow")?;
    let accepted_bytes = (args.batch_size as usize)
        .checked_mul(4)
        .context("gdn_commit_replay_1d_sm120 d_num_accepted byte size overflow")?;

    let status_ptr = match args.status_out {
        Some(ptr) => {
            ensure_aligned_non_null(ptr, 4, "status_out")?;
            ensure_ranges_disjoint(
                ptr,
                4,
                "status_out",
                args.descs_dev,
                desc_bytes,
                "descs_dev",
            )?;
            ensure_ranges_disjoint(
                ptr,
                4,
                "status_out",
                args.d_num_accepted,
                accepted_bytes,
                "d_num_accepted",
            )?;
            ptr
        }
        None => DevicePtr::NULL,
    };

    if let Some(host_descs) = args.host_descs {
        validate_commit_slot_descs(host_descs, args.num_ssm_layers, args.batch_size)?;
        if let Some(status) = args.status_out {
            let h_bytes = (args.num_v_heads as usize)
                .checked_mul(H_HEAD_ELEMS)
                .and_then(|v| v.checked_mul(FP32_BYTES))
                .context("gdn_commit_replay_1d_sm120 h_state byte size overflow")?;
            for (idx, desc) in host_descs.iter().enumerate() {
                ensure_ranges_disjoint(
                    status,
                    4,
                    "status_out",
                    DevicePtr(desc.h_state_ptr),
                    h_bytes,
                    &format!("desc[{idx}].h_state_ptr"),
                )?;
            }
        }
    }

    Ok(status_ptr)
}

/// Validate host descriptors (when supplied), alignment, and geometry, then
/// launch ONE cross-layer `gdn_commit_replay_1d_sm120` kernel over
/// `(num_v_heads, batch_size, num_ssm_layers)`.
///
/// # Errors
/// Returns an error without launching if any invariant fails.
pub fn launch_gdn_commit_replay_1d_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &GdnCommitReplay1dLaunchArgs<'_>,
    stream: u64,
) -> Result<()> {
    let status_ptr = validate_commit_replay_1d_launch_args(kernel, args)?;

    KernelLaunch::new(gpu, kernel)
        .grid([args.num_v_heads, args.batch_size, args.num_ssm_layers])
        .block([GDN_1D_HEAD_DIM, 1, 1])
        .arg_ptr(args.descs_dev)
        .arg_ptr(args.d_num_accepted)
        .arg_ptr(status_ptr)
        .arg_u32(args.num_ssm_layers)
        .arg_u32(args.batch_size)
        .arg_u32(args.num_k_heads)
        .arg_u32(args.num_v_heads)
        .launch(stream)
}

/// Launch parameters for [`launch_gdn_zero_tail_rows_sm120`].
#[derive(Debug, Clone, Copy)]
pub struct GdnZeroTailRowsLaunchArgs {
    pub output: Option<DevicePtr>,
    pub delta_out: Option<DevicePtr>,
    pub k_out: Option<DevicePtr>,
    pub decay_out: Option<DevicePtr>,
    pub geometry: Gdn1dRecordGeometry,
    pub live_tokens: u32,
    pub out_stride: u32,
}

/// Validate parameters and launch `gdn_zero_tail_rows_sm120` to explicitly zero
/// inactive tail rows `[live_tokens, max_tokens)`.
///
/// When `live_tokens == geometry.max_tokens`, returns `Ok(())` without launching
/// a no-op kernel.
///
/// # Errors
/// Returns an error if the kernel handle is null, `live_tokens > max_tokens`,
/// all 4 target buffers are `None`, or any non-null buffer is misaligned.
pub fn launch_gdn_zero_tail_rows_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &GdnZeroTailRowsLaunchArgs,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "gdn_zero_tail_rows_sm120 kernel handle is null"
    );
    let geom = args.geometry;
    let checked = Gdn1dRecordGeometry::try_new(
        geom.batch_size,
        geom.max_tokens,
        geom.num_k_heads,
        geom.num_v_heads,
    )?;
    ensure!(
        checked == geom,
        "GdnZeroTailRowsLaunchArgs geometry fields are inconsistent with try_new"
    );
    ensure!(
        args.live_tokens <= geom.max_tokens,
        "gdn_zero_tail_rows_sm120 live_tokens ({}) exceeds max_tokens ({})",
        args.live_tokens,
        geom.max_tokens
    );
    ensure!(
        args.output.is_some()
            || args.delta_out.is_some()
            || args.k_out.is_some()
            || args.decay_out.is_some(),
        "gdn_zero_tail_rows_sm120 requires at least one non-null target buffer"
    );

    if let Some(out) = args.output {
        ensure_aligned_non_null(out, 4, "output")?;
        let min_v = geom.num_v_heads.saturating_mul(GDN_1D_HEAD_DIM);
        ensure!(
            args.out_stride >= min_v,
            "gdn_zero_tail_rows_sm120 out_stride ({}) < num_v_heads * 128 ({min_v})",
            args.out_stride
        );
    }
    if let Some(delta) = args.delta_out {
        ensure_aligned_non_null(delta, 4, "delta_out")?;
    }
    if let Some(k) = args.k_out {
        ensure_aligned_non_null(k, 4, "k_out")?;
    }
    if let Some(decay) = args.decay_out {
        ensure_aligned_non_null(decay, 4, "decay_out")?;
    }

    if args.live_tokens == geom.max_tokens {
        return Ok(());
    }

    KernelLaunch::new(gpu, kernel)
        .grid([geom.num_v_heads, geom.batch_size, 1])
        .block([GDN_1D_HEAD_DIM, 1, 1])
        .arg_ptr(args.output.unwrap_or(DevicePtr::NULL))
        .arg_ptr(args.delta_out.unwrap_or(DevicePtr::NULL))
        .arg_ptr(args.k_out.unwrap_or(DevicePtr::NULL))
        .arg_ptr(args.decay_out.unwrap_or(DevicePtr::NULL))
        .arg_u32(geom.batch_size)
        .arg_u32(args.live_tokens)
        .arg_u32(geom.max_tokens)
        .arg_u32(geom.num_k_heads)
        .arg_u32(geom.num_v_heads)
        .arg_u32(args.out_stride)
        .launch(stream)
}

// ── Bit-Exact CPU Mathematical Reference ──────────────────────────────────

/// Exact IEEE-754 round-to-nearest-even FP32 -> BF16 -> FP32 boundary matching
/// PTX `cvt.rn.bf16.f32` followed by 16-bit left shift.
#[must_use]
pub fn gdn_1d_round_f32_to_bf16_rn(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = bits & 0x7f80_0000;
    let frac = bits & 0x007f_ffff;
    if exp == 0x7f80_0000 {
        // Infinity stays infinity; NaN is quieted in the upper 16 bits.
        if frac != 0 {
            let quiet_bf16 = ((bits >> 16) | 0x0040) as u16;
            return f32::from_bits(u32::from(quiet_bf16) << 16);
        }
        return f32::from_bits(bits & 0xffff_0000);
    }
    // Round-to-nearest-even on bit 16: add 0x7fff + lsb(bit 16).
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    f32::from_bits(rounded & 0xffff_0000)
}

/// Exact unroll-4 single-accumulator dot product matching `gdn_1d_col_dot`
/// under `--fmad=false` (`__fmul_rn` / `__fadd_rn`).
#[inline]
fn cpu_col_dot_128(h_col: &[f32; 128], vec: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    let mut j = 0usize;
    while j < 128 {
        let p0 = h_col[j] * vec[j];
        let p1 = h_col[j + 1] * vec[j + 1];
        let p2 = h_col[j + 2] * vec[j + 2];
        let p3 = h_col[j + 3] * vec[j + 3];
        let s01 = p0 + p1;
        let s012 = s01 + p2;
        let s0123 = s012 + p3;
        acc += s0123;
        j += 4;
    }
    acc
}

/// Exact rank-1 column update matching `gdn_1d_rank1_update_col` under `--fmad=false`.
#[inline]
fn cpu_rank1_update_col_128(h_col: &mut [f32; 128], k_vec: &[f32], decay: f32, delta_i: f32) {
    let mut j = 0usize;
    while j < 128 {
        h_col[j] = (decay * h_col[j]) + (k_vec[j] * delta_i);
        h_col[j + 1] = (decay * h_col[j + 1]) + (k_vec[j + 1] * delta_i);
        h_col[j + 2] = (decay * h_col[j + 2]) + (k_vec[j + 2] * delta_i);
        h_col[j + 3] = (decay * h_col[j + 3]) + (k_vec[j + 3] * delta_i);
        j += 4;
    }
}

/// Inputs for [`cpu_gdn_verify_record_1d`] and [`cpu_gdn_2d_snapshot_reference`].
#[derive(Debug, Clone, Copy)]
pub struct CpuGdnVerify1dInputs<'a> {
    /// Initial FP32 `h_state` `[B, nv, 128, 128]` (never mutated by verify).
    pub h_state: &'a [f32],
    /// FP32 queries `[B * max_tokens, nk, 128]`.
    pub query: &'a [f32],
    /// FP32 keys `[B * max_tokens, nk, 128]`.
    pub key: &'a [f32],
    /// FP32 values `[B * max_tokens, nv, 128]`.
    pub value: &'a [f32],
    /// FP32 decay gates `[B * max_tokens, nv]`.
    pub gate: &'a [f32],
    /// FP32 beta (post-sigmoid or raw logits depending on `numeric_config`) `[B * max_tokens, nv]`.
    pub beta: &'a [f32],
    pub geometry: Gdn1dRecordGeometry,
    pub live_tokens: u32,
    pub numeric_config: GdnVerifyNumericConfig,
}

/// Outputs produced by [`cpu_gdn_verify_record_1d`].
#[derive(Debug, Clone)]
pub struct CpuGdnVerify1dOutputs {
    /// FP32 GDN outputs `[B * max_tokens, nv, 128]` (tail rows zeroed).
    pub output: Vec<f32>,
    /// 1D FP32 `delta` correction records `[B * max_tokens, nv, 128]`.
    pub delta: Vec<f32>,
    /// 1D FP32 `k` correction records `[B * max_tokens, nk, 128]`.
    pub k: Vec<f32>,
    /// 1D FP32 `decay` correction records `[B * max_tokens, nv]`.
    pub decay: Vec<f32>,
}

/// Outputs produced by [`cpu_gdn_2d_snapshot_reference`].
#[derive(Debug, Clone)]
pub struct CpuGdn2dSnapshotOutputs {
    /// FP32 GDN outputs `[B * max_tokens, nv, 128]` (tail rows zeroed).
    pub output: Vec<f32>,
    /// 2D snapshots `h_snapshots[seq][step]` for `step in 0..=live_tokens`, where
    /// `step == 0` is initial `h_state` and `step == t + 1` is post-token-`t` `h_state`.
    pub h_snapshots: Vec<Vec<Vec<f32>>>,
}

fn validate_cpu_verify_inputs(inputs: &CpuGdnVerify1dInputs<'_>) -> Result<()> {
    let geom = inputs.geometry;
    let checked = Gdn1dRecordGeometry::try_new(
        geom.batch_size,
        geom.max_tokens,
        geom.num_k_heads,
        geom.num_v_heads,
    )?;
    ensure!(checked == geom, "inconsistent Gdn1dRecordGeometry");
    ensure!(
        inputs.live_tokens <= geom.max_tokens,
        "live_tokens ({}) > max_tokens ({})",
        inputs.live_tokens,
        geom.max_tokens
    );
    let b = geom.batch_size as usize;
    let total_rows = b * (geom.max_tokens as usize);
    let nk = geom.num_k_heads as usize;
    let nv = geom.num_v_heads as usize;
    let d = GDN_1D_HEAD_DIM as usize;
    ensure!(
        inputs.h_state.len() == b * nv * d * d,
        "h_state len mismatch"
    );
    ensure!(
        inputs.query.len() == total_rows * nk * d,
        "query len mismatch"
    );
    ensure!(inputs.key.len() == total_rows * nk * d, "key len mismatch");
    ensure!(
        inputs.value.len() == total_rows * nv * d,
        "value len mismatch"
    );
    ensure!(inputs.gate.len() == total_rows * nv, "gate len mismatch");
    ensure!(inputs.beta.len() == total_rows * nv, "beta len mismatch");
    Ok(())
}

/// Bit-exact CPU reference for `gdn_verify_record_1d_sm120`.
///
/// # Errors
/// Returns an error if input slice lengths or geometry are invalid.
pub fn cpu_gdn_verify_record_1d(
    inputs: &CpuGdnVerify1dInputs<'_>,
) -> Result<CpuGdnVerify1dOutputs> {
    validate_cpu_verify_inputs(inputs)?;
    let geom = inputs.geometry;
    let b = geom.batch_size as usize;
    let max_t = geom.max_tokens as usize;
    let live_t = inputs.live_tokens as usize;
    let nk = geom.num_k_heads as usize;
    let nv = geom.num_v_heads as usize;
    let head_repeat = nv / nk;
    let d = GDN_1D_HEAD_DIM as usize;
    let total_rows = b * max_t;

    // Pre-poison buffers with sentinel non-zero values to verify tail-row zeroing.
    let mut output = vec![f32::from_bits(0x7fc0_1234); total_rows * nv * d];
    let mut delta = vec![f32::from_bits(0x7fc0_1234); total_rows * nv * d];
    let mut k_out = vec![f32::from_bits(0x7fc0_1234); total_rows * nk * d];
    let mut decay_out = vec![f32::from_bits(0x7fc0_1234); total_rows * nv];

    for seq in 0..b {
        for vh in 0..nv {
            let kh = vh / head_repeat;
            let sub = vh - kh * head_repeat;
            let h_head_base = (seq * nv + vh) * d * d;

            if live_t > 0 {
                // Transpose per-column working state into `[tid][j]` matching `H_reg[j]` per thread `tid`.
                let mut h_cols = vec![[0.0_f32; 128]; 128];
                for j in 0..d {
                    for tid in 0..d {
                        h_cols[tid][j] = inputs.h_state[h_head_base + j * d + tid];
                    }
                }

                for t in 0..live_t {
                    let row = seq * max_t + t;
                    let qk_base = (row * nk + kh) * d;
                    let v_base = (row * nv + vh) * d;
                    let gb_idx = row * nv + vh;

                    let q_slice = &inputs.query[qk_base..qk_base + d];
                    let k_slice = &inputs.key[qk_base..qk_base + d];
                    let v_slice = &inputs.value[v_base..v_base + d];

                    let decay = inputs.gate[gb_idx].clamp(0.0, 1.0);
                    let mut bt = inputs.beta[gb_idx];
                    if matches!(
                        inputs.numeric_config.beta_input,
                        GdnBetaInputMode::RawLogitSigmoid
                    ) {
                        bt = 1.0_f32 / (1.0_f32 + (-bt).exp());
                    }
                    if matches!(
                        inputs.numeric_config.beta_rounding,
                        GdnBetaRoundingMode::RoundToBf16Ptx
                    ) {
                        bt = gdn_1d_round_f32_to_bf16_rn(bt);
                    }

                    for tid in 0..d {
                        let hk_dot = cpu_col_dot_128(&h_cols[tid], k_slice);
                        let delta_i = (v_slice[tid] - (decay * hk_dot)) * bt;
                        cpu_rank1_update_col_128(&mut h_cols[tid], k_slice, decay, delta_i);
                        let q_dot = cpu_col_dot_128(&h_cols[tid], q_slice);
                        let mut out_val = q_dot * GDN_1D_INV_SQRT_128;
                        if matches!(
                            inputs.numeric_config.output_rounding,
                            GdnOutputRoundingMode::RoundToBf16Ptx
                        ) {
                            out_val = gdn_1d_round_f32_to_bf16_rn(out_val);
                        }
                        output[v_base + tid] = out_val;
                        delta[v_base + tid] = delta_i;
                        if sub == 0 {
                            k_out[qk_base + tid] = k_slice[tid];
                        }
                    }
                    decay_out[gb_idx] = decay;
                }
            }

            // Explicit tail-row zeroing for `[live_t, max_t)`.
            for t in live_t..max_t {
                let row = seq * max_t + t;
                let qk_base = (row * nk + kh) * d;
                let v_base = (row * nv + vh) * d;
                let gb_idx = row * nv + vh;
                for tid in 0..d {
                    output[v_base + tid] = 0.0;
                    delta[v_base + tid] = 0.0;
                    if sub == 0 {
                        k_out[qk_base + tid] = 0.0;
                    }
                }
                decay_out[gb_idx] = 0.0;
            }
        }
    }

    Ok(CpuGdnVerify1dOutputs {
        output,
        delta,
        k: k_out,
        decay: decay_out,
    })
}

/// Traditional 2D `h_state_intermediates` snapshot reference for proving 0-ULP
/// bit-for-bit parity against 1D correction-record replay.
///
/// # Errors
/// Returns an error if input slice lengths or geometry are invalid.
pub fn cpu_gdn_2d_snapshot_reference(
    inputs: &CpuGdnVerify1dInputs<'_>,
) -> Result<CpuGdn2dSnapshotOutputs> {
    validate_cpu_verify_inputs(inputs)?;
    let geom = inputs.geometry;
    let b = geom.batch_size as usize;
    let max_t = geom.max_tokens as usize;
    let live_t = inputs.live_tokens as usize;
    let nk = geom.num_k_heads as usize;
    let nv = geom.num_v_heads as usize;
    let head_repeat = nv / nk;
    let d = GDN_1D_HEAD_DIM as usize;
    let total_rows = b * max_t;
    let seq_h_elems = nv * d * d;

    let mut output = vec![0.0_f32; total_rows * nv * d];
    let mut h_snapshots = Vec::with_capacity(b);

    for seq in 0..b {
        let mut seq_snaps = Vec::with_capacity(live_t + 1);
        let mut h_cur = inputs.h_state[seq * seq_h_elems..(seq + 1) * seq_h_elems].to_vec();
        seq_snaps.push(h_cur.clone());

        for t in 0..live_t {
            let row = seq * max_t + t;
            for vh in 0..nv {
                let kh = vh / head_repeat;
                let qk_base = (row * nk + kh) * d;
                let v_base = (row * nv + vh) * d;
                let gb_idx = row * nv + vh;
                let h_base = vh * d * d;

                let q_slice = &inputs.query[qk_base..qk_base + d];
                let k_slice = &inputs.key[qk_base..qk_base + d];
                let v_slice = &inputs.value[v_base..v_base + d];

                let decay = inputs.gate[gb_idx].clamp(0.0, 1.0);
                let mut bt = inputs.beta[gb_idx];
                if matches!(
                    inputs.numeric_config.beta_input,
                    GdnBetaInputMode::RawLogitSigmoid
                ) {
                    bt = 1.0_f32 / (1.0_f32 + (-bt).exp());
                }
                if matches!(
                    inputs.numeric_config.beta_rounding,
                    GdnBetaRoundingMode::RoundToBf16Ptx
                ) {
                    bt = gdn_1d_round_f32_to_bf16_rn(bt);
                }

                for tid in 0..d {
                    let mut h_col = [0.0_f32; 128];
                    for j in 0..d {
                        h_col[j] = h_cur[h_base + j * d + tid];
                    }
                    let hk_dot = cpu_col_dot_128(&h_col, k_slice);
                    let delta_i = (v_slice[tid] - (decay * hk_dot)) * bt;
                    cpu_rank1_update_col_128(&mut h_col, k_slice, decay, delta_i);
                    let q_dot = cpu_col_dot_128(&h_col, q_slice);
                    let mut out_val = q_dot * GDN_1D_INV_SQRT_128;
                    if matches!(
                        inputs.numeric_config.output_rounding,
                        GdnOutputRoundingMode::RoundToBf16Ptx
                    ) {
                        out_val = gdn_1d_round_f32_to_bf16_rn(out_val);
                    }
                    output[v_base + tid] = out_val;
                    for j in 0..d {
                        h_cur[h_base + j * d + tid] = h_col[j];
                    }
                }
            }
            seq_snaps.push(h_cur.clone());
        }
        h_snapshots.push(seq_snaps);
    }

    Ok(CpuGdn2dSnapshotOutputs {
        output,
        h_snapshots,
    })
}

/// CPU-side commit slot descriptor mirroring [`GdnCommitSlotDesc`] for exact
/// reference testing of `gdn_commit_replay_1d_sm120`.
#[derive(Debug)]
pub struct CpuGdnCommitSlot<'a> {
    pub h_state: &'a mut [f32],
    pub delta: &'a [f32],
    pub k: &'a [f32],
    pub decay: &'a [f32],
    pub generation: u64,
    pub expected_generation: u64,
    pub sharing_count: u32,
    pub max_tokens: u32,
    pub flags: u32,
}

/// Bit-exact CPU reference for `gdn_commit_replay_1d_sm120`.
///
/// Mirrors the device kernel's fail-closed status accumulation: any slot failing
/// generation, sharing, bounds, or flags leaves its `h_state` completely
/// untouched and sets the exact corresponding `GDN_1D_STATUS_*` bit(s).
///
/// # Errors
/// Returns an error if outer slice dimensions (`slots`, `d_num_accepted`, head counts)
/// are structurally mismatched.
pub fn cpu_gdn_commit_replay_1d(
    slots: &mut [CpuGdnCommitSlot<'_>],
    d_num_accepted: &[u32],
    num_ssm_layers: u32,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
) -> Result<GdnCommitStatus> {
    ensure!(
        num_ssm_layers > 0 && batch_size > 0,
        "cpu_gdn_commit_replay_1d requires num_ssm_layers > 0 and batch_size > 0"
    );
    ensure!(
        num_k_heads > 0 && num_v_heads > 0 && num_v_heads.is_multiple_of(num_k_heads),
        "cpu_gdn_commit_replay_1d requires num_v_heads divisible by num_k_heads > 0"
    );
    ensure!(
        slots.len() == (num_ssm_layers as usize) * (batch_size as usize),
        "slots length mismatch"
    );
    ensure!(
        d_num_accepted.len() == batch_size as usize,
        "d_num_accepted length mismatch"
    );

    let nk = num_k_heads as usize;
    let nv = num_v_heads as usize;
    let head_repeat = nv / nk;
    let d = GDN_1D_HEAD_DIM as usize;
    let mut status_bits = GDN_1D_STATUS_OK;

    for layer in 0..num_ssm_layers as usize {
        for seq in 0..batch_size as usize {
            let slot = &mut slots[layer * (batch_size as usize) + seq];
            let accepted = d_num_accepted[seq];
            let mut err = GDN_1D_STATUS_OK;
            if slot.h_state.is_empty()
                || slot.delta.is_empty()
                || slot.k.is_empty()
                || slot.decay.is_empty()
            {
                err |= GDN_1D_STATUS_NULL_PTR;
            }
            if slot.generation != slot.expected_generation {
                err |= GDN_1D_STATUS_GEN_MISMATCH;
            }
            if slot.sharing_count > 1 {
                err |= GDN_1D_STATUS_SHARED_SLOT;
            }
            if accepted > slot.max_tokens {
                err |= GDN_1D_STATUS_ACCEPTED_OOB;
            }
            if slot.flags != 0 {
                err |= GDN_1D_STATUS_BAD_GEOMETRY;
            }
            if err != GDN_1D_STATUS_OK {
                status_bits |= err;
                continue;
            }
            if accepted == 0 {
                continue;
            }
            let max_t = slot.max_tokens as usize;
            ensure!(
                slot.h_state.len() == nv * d * d,
                "slot h_state len mismatch"
            );
            ensure!(
                slot.delta.len() >= max_t * nv * d,
                "slot delta len too small"
            );
            ensure!(slot.k.len() >= max_t * nk * d, "slot k len too small");
            ensure!(slot.decay.len() >= max_t * nv, "slot decay len too small");

            for vh in 0..nv {
                let kh = vh / head_repeat;
                let h_base = vh * d * d;
                for tid in 0..d {
                    let mut h_col = [0.0_f32; 128];
                    for j in 0..d {
                        h_col[j] = slot.h_state[h_base + j * d + tid];
                    }
                    for t in 0..accepted as usize {
                        let k_base = (t * nk + kh) * d;
                        let delta_i = slot.delta[(t * nv + vh) * d + tid];
                        let decay = slot.decay[t * nv + vh];
                        cpu_rank1_update_col_128(
                            &mut h_col,
                            &slot.k[k_base..k_base + d],
                            decay,
                            delta_i,
                        );
                    }
                    for j in 0..d {
                        slot.h_state[h_base + j * d + tid] = h_col[j];
                    }
                }
            }
        }
    }

    Ok(GdnCommitStatus(status_bits))
}

/// Bit-exact CPU reference for `gdn_zero_tail_rows_sm120`.
///
/// # Errors
/// Returns an error if `live_tokens > geometry.max_tokens` or any provided slice
/// has an incompatible length.
pub fn cpu_gdn_zero_tail_rows(
    mut output: Option<&mut [f32]>,
    mut delta: Option<&mut [f32]>,
    mut k: Option<&mut [f32]>,
    mut decay: Option<&mut [f32]>,
    geometry: Gdn1dRecordGeometry,
    live_tokens: u32,
) -> Result<()> {
    ensure!(
        live_tokens <= geometry.max_tokens,
        "live_tokens ({live_tokens}) > max_tokens ({})",
        geometry.max_tokens
    );
    let b = geometry.batch_size as usize;
    let max_t = geometry.max_tokens as usize;
    let live_t = live_tokens as usize;
    let nk = geometry.num_k_heads as usize;
    let nv = geometry.num_v_heads as usize;
    let d = GDN_1D_HEAD_DIM as usize;
    let total_rows = b * max_t;

    if let Some(buf) = output.as_deref() {
        ensure!(buf.len() == total_rows * nv * d, "output len mismatch");
    }
    if let Some(buf) = delta.as_deref() {
        ensure!(buf.len() == total_rows * nv * d, "delta len mismatch");
    }
    if let Some(buf) = k.as_deref() {
        ensure!(buf.len() == total_rows * nk * d, "k len mismatch");
    }
    if let Some(buf) = decay.as_deref() {
        ensure!(buf.len() == total_rows * nv, "decay len mismatch");
    }

    for seq in 0..b {
        for t in live_t..max_t {
            let row = seq * max_t + t;
            if let Some(buf) = output.as_deref_mut() {
                buf[row * nv * d..(row + 1) * nv * d].fill(0.0);
            }
            if let Some(buf) = delta.as_deref_mut() {
                buf[row * nv * d..(row + 1) * nv * d].fill(0.0);
            }
            if let Some(buf) = k.as_deref_mut() {
                buf[row * nk * d..(row + 1) * nk * d].fill(0.0);
            }
            if let Some(buf) = decay.as_deref_mut() {
                buf[row * nv..(row + 1) * nv].fill(0.0);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use std::env::VarError;
    use std::ffi::OsString;
    #[cfg(unix)]
    use std::os::unix::ffi::OsStringExt;

    #[test]
    fn env_parser_is_fail_closed_zero_or_one_and_warns_not_unicode_off() {
        assert_eq!(
            parse_gdn_1d_replay_env(Err(VarError::NotPresent)).unwrap(),
            Gdn1dReplayPolicy::Disabled
        );
        assert_eq!(
            parse_gdn_1d_replay_env(Ok("0".to_string())).unwrap(),
            Gdn1dReplayPolicy::Disabled
        );
        assert_eq!(
            parse_gdn_1d_replay_env(Ok("1".to_string())).unwrap(),
            Gdn1dReplayPolicy::Enabled
        );
        for bad in ["", "2", "true", "false", "on", " 1 ", "-1"] {
            assert!(
                parse_gdn_1d_replay_env(Ok(bad.to_string())).is_err(),
                "invalid value {bad:?} must fail closed"
            );
        }
        #[cfg(unix)]
        let bad_os = OsString::from_vec(vec![0xff, 0xfe]);
        #[cfg(not(unix))]
        let bad_os = OsString::from("bad");
        assert!(
            parse_gdn_1d_replay_env(Err(VarError::NotUnicode(bad_os))).is_err(),
            "NotUnicode must fail closed with Err"
        );
    }

    #[test]
    fn inv_sqrt_128_constant_matches_exact_ieee_f32() {
        assert_eq!(
            GDN_1D_INV_SQRT_128.to_bits(),
            (1.0_f32 / 128.0_f32.sqrt()).to_bits()
        );
        assert_eq!(GDN_1D_INV_SQRT_128.to_bits(), 0x3db5_04f3);
    }

    #[test]
    fn commit_slot_desc_abi_roundtrip_is_exact_64_bytes() {
        let desc = GdnCommitSlotDesc {
            h_state_ptr: 0x1122_3344_5566_7780,
            delta_ptr: 0x2233_4455_6677_8890,
            k_ptr: 0x3344_5566_7788_99a0,
            decay_ptr: 0x4455_6677_8899_aab0,
            generation: 42,
            expected_generation: 42,
            sharing_count: 1,
            max_tokens: 16,
            flags: 0,
            pad: 0,
        };
        let bytes = desc.to_le_bytes();
        assert_eq!(bytes.len(), 64);
        assert_eq!(GdnCommitSlotDesc::from_le_bytes(&bytes), desc);
    }

    #[test]
    fn geometry_proves_64x_to_95x_memory_reduction_over_2d_h_inter() {
        // Qwen3.8-27B / Qwen3.8-Flash-Next GQA 16:48, W=16, B=1
        let gqa = Gdn1dRecordGeometry::try_new(1, 16, 16, 48).unwrap();
        assert_eq!(gqa.legacy_2d_h_inter_bytes, 16 * 48 * 128 * 128 * 4);
        assert_eq!(
            gqa.logical_record_bytes(),
            16 * (48 * 128 * 4 + 16 * 128 * 4 + 48 * 4)
        );
        assert!(
            gqa.compression_ratio_vs_2d_inter() > 95.0,
            "expected >95x GQA compression, got {:.2}x",
            gqa.compression_ratio_vs_2d_inter()
        );

        // Worst-case MHA 1:1 (nk == nv == 48): 65,536 B vs 1,028 B = 63.75x
        let mha = Gdn1dRecordGeometry::try_new(4, 16, 48, 48).unwrap();
        assert!(
            mha.compression_ratio_vs_2d_inter() > 63.7,
            "expected >=63.75x MHA compression, got {:.2}x",
            mha.compression_ratio_vs_2d_inter()
        );
        assert_eq!(mha.k_offset_bytes % 16, 0);
        assert_eq!(mha.decay_offset_bytes % 16, 0);
        assert_eq!(mha.total_arena_bytes % 16, 0);

        // Hostile dimensions rejected.
        assert!(Gdn1dRecordGeometry::try_new(0, 16, 16, 48).is_err());
        assert!(Gdn1dRecordGeometry::try_new(1, 0, 16, 48).is_err());
        assert!(Gdn1dRecordGeometry::try_new(1, 16, 0, 48).is_err());
        assert!(Gdn1dRecordGeometry::try_new(1, 16, 16, 47).is_err());
    }

    #[test]
    fn bf16_rne_rounding_matches_ieee_ties_to_even_and_specials() {
        // Exact tie halfway between 1.0 (0x3f80_0000, even) and 1.0 + 2^-7 (0x3f81_0000, odd).
        let tie_even = f32::from_bits(0x3f80_8000);
        assert_eq!(gdn_1d_round_f32_to_bf16_rn(tie_even).to_bits(), 0x3f80_0000);
        // Exact tie halfway between 0x3f81_0000 (odd) and 0x3f82_0000 (even) -> rounds up to even.
        let tie_odd = f32::from_bits(0x3f81_8000);
        assert_eq!(gdn_1d_round_f32_to_bf16_rn(tie_odd).to_bits(), 0x3f82_0000);
        // Above halfway rounds up.
        let above = f32::from_bits(0x3f80_8001);
        assert_eq!(gdn_1d_round_f32_to_bf16_rn(above).to_bits(), 0x3f81_0000);
        // Specials preserve sign/inf and quiet NaN.
        assert_eq!(
            gdn_1d_round_f32_to_bf16_rn(f32::NEG_INFINITY).to_bits(),
            f32::NEG_INFINITY.to_bits()
        );
        assert_eq!(
            gdn_1d_round_f32_to_bf16_rn(-0.0_f32).to_bits(),
            (-0.0_f32).to_bits()
        );
        assert!(gdn_1d_round_f32_to_bf16_rn(f32::NAN).is_nan());
    }

    #[test]
    fn launchers_validate_hostile_inputs_and_dispatch_on_mock_gpu() {
        let gpu = MockGpuBackend::new();
        let geom = Gdn1dRecordGeometry::try_new(2, 8, 2, 4).unwrap();
        let verify_args = GdnVerifyRecord1dLaunchArgs {
            h_state: DevicePtr(0x0100_0000),
            query: DevicePtr(0x0200_0000),
            key: DevicePtr(0x0200_0400),
            value: DevicePtr(0x0200_0800),
            gate: DevicePtr(0x0300_0000),
            beta: DevicePtr(0x0300_0010),
            output: DevicePtr(0x0200_1000),
            delta_out: DevicePtr(0x0400_0000),
            k_out: DevicePtr(0x0400_0000 + geom.k_offset_bytes as u64),
            decay_out: DevicePtr(0x0400_0000 + geom.decay_offset_bytes as u64),
            geometry: geom,
            live_tokens: 5,
            qk_stride: 2 * 128 + 2 * 128 + 4 * 128 + 4 * 128,
            v_stride: 2 * 128 + 2 * 128 + 4 * 128 + 4 * 128,
            gb_stride: 8,
            out_stride: 2 * 128 + 2 * 128 + 4 * 128 + 4 * 128,
            numeric_config: GdnVerifyNumericConfig::prefill_bf16_parity(
                GdnBetaInputMode::PrecomputedBeta,
            ),
        };
        launch_gdn_verify_record_1d_sm120(&gpu, KernelHandle(1), &verify_args, 7).unwrap();

        let descs = [
            GdnCommitSlotDesc {
                h_state_ptr: 0x0100_0000,
                delta_ptr: 0x0400_0000,
                k_ptr: 0x0400_0000 + geom.k_offset_bytes as u64,
                decay_ptr: 0x0400_0000 + geom.decay_offset_bytes as u64,
                generation: 9,
                expected_generation: 9,
                sharing_count: 1,
                max_tokens: 8,
                flags: 0,
                pad: 0,
            },
            GdnCommitSlotDesc {
                h_state_ptr: 0x0110_0000,
                delta_ptr: 0x0410_0000,
                k_ptr: 0x0420_0000,
                decay_ptr: 0x0430_0000,
                generation: 11,
                expected_generation: 11,
                sharing_count: 0,
                max_tokens: 8,
                flags: 0,
                pad: 0,
            },
        ];
        let commit_args = GdnCommitReplay1dLaunchArgs {
            descs_dev: DevicePtr(0x0500_0000),
            host_descs: Some(&descs),
            d_num_accepted: DevicePtr(0x0510_0000),
            status_out: Some(DevicePtr(0x0520_0000)),
            num_ssm_layers: 1,
            batch_size: 2,
            num_k_heads: 2,
            num_v_heads: 4,
        };
        launch_gdn_commit_replay_1d_sm120(&gpu, KernelHandle(2), &commit_args, 7).unwrap();

        let zero_args = GdnZeroTailRowsLaunchArgs {
            output: Some(DevicePtr(0x0200_1000)),
            delta_out: Some(DevicePtr(0x0400_0000)),
            k_out: Some(DevicePtr(0x0400_0000 + geom.k_offset_bytes as u64)),
            decay_out: Some(DevicePtr(0x0400_0000 + geom.decay_offset_bytes as u64)),
            geometry: geom,
            live_tokens: 5,
            out_stride: 4 * 128,
        };
        launch_gdn_zero_tail_rows_sm120(&gpu, KernelHandle(3), &zero_args, 7).unwrap();
        // When live_tokens == max_tokens, zero_tail_rows is a clean no-op (no launch).
        launch_gdn_zero_tail_rows_sm120(
            &gpu,
            KernelHandle(3),
            &GdnZeroTailRowsLaunchArgs {
                live_tokens: 8,
                ..zero_args
            },
            7,
        )
        .unwrap();
        assert_eq!(gpu.launch_count(), 3);

        // Hostile rejections must NEVER enqueue a kernel.
        assert!(launch_gdn_verify_record_1d_sm120(&gpu, KernelHandle(0), &verify_args, 0).is_err());
        assert!(
            launch_gdn_verify_record_1d_sm120(
                &gpu,
                KernelHandle(1),
                &GdnVerifyRecord1dLaunchArgs {
                    h_state: DevicePtr(0x0100_0008),
                    ..verify_args
                },
                0,
            )
            .is_err()
        );
        assert!(
            launch_gdn_verify_record_1d_sm120(
                &gpu,
                KernelHandle(1),
                &GdnVerifyRecord1dLaunchArgs {
                    delta_out: verify_args.h_state,
                    ..verify_args
                },
                0,
            )
            .is_err()
        );
        assert!(
            launch_gdn_verify_record_1d_sm120(
                &gpu,
                KernelHandle(1),
                &GdnVerifyRecord1dLaunchArgs {
                    live_tokens: 9,
                    ..verify_args
                },
                0,
            )
            .is_err()
        );

        // Commit rejections: generation mismatch, sharing > 1, duplicate h_state_ptr.
        let mut stale = descs;
        stale[1].expected_generation = 99;
        assert!(
            launch_gdn_commit_replay_1d_sm120(
                &gpu,
                KernelHandle(2),
                &GdnCommitReplay1dLaunchArgs {
                    host_descs: Some(&stale),
                    ..commit_args
                },
                0,
            )
            .is_err()
        );

        let mut shared = descs;
        shared[0].sharing_count = 2;
        assert!(
            launch_gdn_commit_replay_1d_sm120(
                &gpu,
                KernelHandle(2),
                &GdnCommitReplay1dLaunchArgs {
                    host_descs: Some(&shared),
                    ..commit_args
                },
                0,
            )
            .is_err()
        );

        let mut dup_h = descs;
        dup_h[1].h_state_ptr = dup_h[0].h_state_ptr;
        assert!(
            launch_gdn_commit_replay_1d_sm120(
                &gpu,
                KernelHandle(2),
                &GdnCommitReplay1dLaunchArgs {
                    host_descs: Some(&dup_h),
                    ..commit_args
                },
                0,
            )
            .is_err()
        );

        // Cross-layer duplicate h_state_ptr and status_out aliasing must fail closed.
        assert!(
            launch_gdn_commit_replay_1d_sm120(
                &gpu,
                KernelHandle(2),
                &GdnCommitReplay1dLaunchArgs {
                    host_descs: Some(&dup_h),
                    num_ssm_layers: 2,
                    batch_size: 1,
                    ..commit_args
                },
                0,
            )
            .is_err()
        );
        assert!(
            launch_gdn_commit_replay_1d_sm120(
                &gpu,
                KernelHandle(2),
                &GdnCommitReplay1dLaunchArgs {
                    status_out: Some(commit_args.descs_dev),
                    ..commit_args
                },
                0,
            )
            .is_err()
        );

        assert_eq!(gpu.launch_count(), 3);
    }

    #[test]
    fn cuda_source_contains_required_entrypoints_and_ptx_contracts() {
        let cuda =
            include_str!("../../../../../kernels/gb10/common/gdn_correction_replay_sm120.cu");
        for entry in [
            "gdn_verify_record_1d_sm120(",
            "gdn_commit_replay_1d_sm120(",
            "gdn_zero_tail_rows_sm120(",
        ] {
            assert!(cuda.contains(entry), "missing CUDA entry {entry}");
        }
        assert!(cuda.contains("cvt.rn.bf16.f32"));
        assert!(cuda.contains("L2::evict_first"));
        assert!(cuda.contains("__CUDA_ARCH__ >= 1200"));
        assert!(cuda.contains("__shared__ alignas(16) float smem_k[GDN_1D_DIM]"));
        assert!(cuda.contains("__shared__ alignas(16) float smem_q[GDN_1D_DIM]"));
        assert!(cuda.contains("static_assert(sizeof(GdnCommitSlotDesc) == 64"));
    }
}
