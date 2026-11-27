// SPDX-License-Identifier: AGPL-3.0-only

//! Phases 3-6: per-sequence RoPE, KV-cache write, batched paged
//! attention, gate multiply + O projection.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::ctx::MultiSeqCtx;
use super::qkv::{raw_attn_proj_tc_override, resolve_attn_proj_tc_for_sm_with_policy};
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactAttnOProjDispatch {
    Existing,
    Exact,
    PerRowK1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttnGateBatchRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttnOProjM17AStageRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
    Conflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttnKvWriteBatchRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

/// Atomic admission for the native-Qwen4 BF16 KV-write consolidation.  The
/// existing `reshape_and_cache_flash` ABI already indexes tokens with
/// `blockIdx.x` and accepts independent K/V row strides, so the batched route
/// changes only launch grouping: it writes the same BF16 source bytes to the
/// same slots as the per-row loop.
const fn attn_kv_write_batch_route(
    requested: bool,
    native_qwen4: bool,
    bf16_kv: bool,
    rows: usize,
    has_kernel: bool,
) -> AttnKvWriteBatchRoute {
    if !requested {
        return AttnKvWriteBatchRoute::Disabled;
    }
    if !native_qwen4 || !bf16_kv || rows < 2 || rows > crate::layers::QWEN4_NATIVE_MAX_ROWS {
        return AttnKvWriteBatchRoute::Ineligible;
    }
    if has_kernel {
        AttnKvWriteBatchRoute::Complete
    } else {
        AttnKvWriteBatchRoute::Missing
    }
}

fn parse_attn_kv_write_batched(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_ATTN_KV_WRITE_BATCHED must be exactly 0 or 1"),
    }
}

fn attn_kv_write_batched_requested() -> Result<bool> {
    static GATE: std::sync::OnceLock<std::result::Result<bool, &'static str>> =
        std::sync::OnceLock::new();
    (*GATE.get_or_init(|| match std::env::var("ATLAS_ATTN_KV_WRITE_BATCHED") {
        Ok(value) => parse_attn_kv_write_batched(Some(value.as_str())),
        Err(std::env::VarError::NotPresent) => parse_attn_kv_write_batched(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_ATTN_KV_WRITE_BATCHED must be valid UTF-8 and exactly 0 or 1")
        }
    }))
    .map_err(anyhow::Error::msg)
}

fn log_attn_kv_write_batched_engagement_once(rows: usize) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!("ENGAGED ATLAS_ATTN_KV_WRITE_BATCHED: multi_seq M={rows}");
    });
}

/// Atomic production decision for the ABI-separated activation-staged O
/// projection. A requested eligible route never silently falls through: a
/// stale/missing symbol or an independently requested projection override is
/// reported before gate/output buffers are mutated.
const fn attn_o_proj_m17_astage_route(
    requested: bool,
    ordinary_nvfp4: bool,
    gated: bool,
    rows: usize,
    has_kernel: bool,
    rt2_requested: bool,
    serial_override: bool,
) -> AttnOProjM17AStageRoute {
    if !requested {
        return AttnOProjM17AStageRoute::Disabled;
    }
    if !ordinary_nvfp4 || !gated || rows < 9 || rows > 17 {
        return AttnOProjM17AStageRoute::Ineligible;
    }
    if rt2_requested || serial_override {
        return AttnOProjM17AStageRoute::Conflict;
    }
    if has_kernel {
        AttnOProjM17AStageRoute::Complete
    } else {
        AttnOProjM17AStageRoute::Missing
    }
}

fn parse_attn_o_proj_m17_astage(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_ATTN_O_PROJ_EXACT_M17_ASTAGE must be exactly 0 or 1"),
    }
}

fn attn_o_proj_m17_astage_requested() -> Result<bool> {
    static GATE: std::sync::OnceLock<std::result::Result<bool, &'static str>> =
        std::sync::OnceLock::new();
    (*GATE.get_or_init(
        || match std::env::var("ATLAS_ATTN_O_PROJ_EXACT_M17_ASTAGE") {
            Ok(value) => parse_attn_o_proj_m17_astage(Some(value.as_str())),
            Err(std::env::VarError::NotPresent) => parse_attn_o_proj_m17_astage(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("ATLAS_ATTN_O_PROJ_EXACT_M17_ASTAGE must be valid UTF-8 and exactly 0 or 1")
            }
        },
    ))
    .map_err(anyhow::Error::msg)
}

#[cfg(test)]
const fn attn_gate_batch_route(
    requested: bool,
    gated: bool,
    rows: usize,
    dim: u32,
    gate_stride: u32,
    has_kernel: bool,
) -> AttnGateBatchRoute {
    if !requested {
        return AttnGateBatchRoute::Disabled;
    }
    if !gated
        || rows < 5
        || rows > 17
        || dim == 0
        || gate_stride < dim
        || (rows as u64) * (dim as u64) > u32::MAX as u64
    {
        return AttnGateBatchRoute::Ineligible;
    }
    if has_kernel {
        AttnGateBatchRoute::Complete
    } else {
        AttnGateBatchRoute::Missing
    }
}

const ATTN_GATE_BATCH_WIDE_MAX_ROWS: usize = 256;

const fn attn_gate_batch_route_wide(
    mode: Option<bool>,
    sm_count: u32,
    gated: bool,
    rows: usize,
    dim: u32,
    gate_stride: u32,
    has_kernel: bool,
) -> AttnGateBatchRoute {
    let requested = match mode {
        Some(v) => v,
        None => sm_count > 64,
    };
    if !requested {
        return AttnGateBatchRoute::Disabled;
    }
    if !gated
        || rows < 2
        || rows > ATTN_GATE_BATCH_WIDE_MAX_ROWS
        || dim == 0
        || gate_stride < dim
        || (rows as u64) * (gate_stride as u64) > u32::MAX as u64
    {
        return AttnGateBatchRoute::Ineligible;
    }
    if has_kernel {
        AttnGateBatchRoute::Complete
    } else if matches!(mode, Some(true)) {
        AttnGateBatchRoute::Missing
    } else {
        AttnGateBatchRoute::Ineligible
    }
}

fn parse_attn_gate_batched(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_ATTN_GATE_BATCHED must be exactly 0 or 1"),
    }
}

fn attn_gate_batched_mode() -> Result<Option<bool>> {
    static GATE: OnceLock<std::result::Result<Option<bool>, &'static str>> = OnceLock::new();
    (*GATE.get_or_init(|| match std::env::var("ATLAS_ATTN_GATE_BATCHED") {
        Ok(value) => parse_attn_gate_batched(Some(value.as_str())).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_ATTN_GATE_BATCHED must be valid UTF-8 and exactly 0 or 1")
        }
    }))
    .map_err(anyhow::Error::msg)
}

fn attn_gate_batched_requested() -> Result<bool> {
    Ok(attn_gate_batched_mode()? == Some(true))
}

#[derive(Clone, Copy)]
pub(in super::super) struct MultiSeqAttnBatchKernels {
    pub(in super::super) mrope_multi_seq: KernelHandle,
    pub(in super::super) mrope_yarn_multi_seq: KernelHandle,
    pub(in super::super) cache_bf16_multi_seq: KernelHandle,
    pub(in super::super) cache_fp8_multi_seq: KernelHandle,
    pub(in super::super) cache_nvfp4_multi_seq: KernelHandle,
}

pub(in super::super) fn cached_multiseq_attn_batch_kernels(
    gpu: &dyn GpuBackend,
) -> MultiSeqAttnBatchKernels {
    static KERNELS: OnceLock<Mutex<HashMap<usize, MultiSeqAttnBatchKernels>>> = OnceLock::new();
    let map = KERNELS.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (gpu as *const dyn GpuBackend).cast::<()>() as usize;
    let load = || MultiSeqAttnBatchKernels {
        mrope_multi_seq: crate::layers::try_kernel(
            gpu,
            "rope_mrope_interleaved",
            "rope_forward_mrope_interleaved_multi_seq",
        ),
        mrope_yarn_multi_seq: crate::layers::try_kernel(
            gpu,
            "rope_mrope_interleaved",
            "rope_forward_mrope_interleaved_yarn_multi_seq",
        ),
        cache_bf16_multi_seq: crate::layers::try_kernel(
            gpu,
            "reshape_and_cache",
            "reshape_and_cache_flash_multi_seq",
        ),
        cache_fp8_multi_seq: crate::layers::try_kernel(
            gpu,
            "reshape_and_cache",
            "reshape_and_cache_flash_fp8_multi_seq",
        ),
        cache_nvfp4_multi_seq: crate::layers::try_kernel(
            gpu,
            "reshape_and_cache",
            "reshape_and_cache_flash_nvfp4_multi_seq",
        ),
    };
    let Ok(mut guard) = map.lock() else {
        return load();
    };
    *guard.entry(key).or_insert_with(load)
}

#[inline]
fn should_auto_serialize_paged_split_boundary(
    rows: usize,
    has_kv_indir: bool,
    has_ddtree_parent_ids: bool,
    crosses_split_boundary: bool,
) -> bool {
    rows > 1 && !has_kv_indir && !has_ddtree_parent_ids && crosses_split_boundary
}

/// Compute split-K partition count for `ATLAS_FLASH_ATTN_KGAMMA_SPLITK`.
///
/// Targets one CTA per SM (`sm_count / nq`), bounded by both
/// `ceil(max_seq_len_host / SPLIT_TILE)` and `MAX_SPLITS_CAP = 64` so the
/// extended `BufferSizes::splitk_workspace` (`decode_rows * q_heads *
/// max_seq_len.div_ceil(512).clamp(2, 64)`) never overflows even on 188+ SM GPUs.
fn compute_kgamma_splitk_splits(nq: u32, max_seq_len_host: u32, sm_count: u32) -> u32 {
    const SPLIT_TILE: u32 = 512;
    const MAX_SPLITS_CAP: u32 = 64;
    let occupancy_target = (sm_count.max(1) / nq.max(1)).max(2);
    let seq_target = max_seq_len_host.div_ceil(SPLIT_TILE);
    occupancy_target.min(seq_target.max(1)).min(MAX_SPLITS_CAP)
}

const fn bf16_kgamma_geometry(rows: usize, head_dim: u32, flat: bool, symbol: bool) -> bool {
    rows >= 8 && rows <= 32 && head_dim == 256 && flat && symbol
}

const fn fp8_bc4_kgamma_geometry(rows: usize, head_dim: u32, flat: bool, symbol: bool) -> bool {
    rows == 15 && head_dim == 128 && flat && symbol
}

#[cfg(test)]
#[path = "attn_route_tests.rs"]
mod route_tests;

const fn exact_attn_o_proj_dispatch(
    ordinary_nvfp4: bool,
    route: Option<ops::ExactLmHeadRoute>,
) -> ExactAttnOProjDispatch {
    if !ordinary_nvfp4 {
        return ExactAttnOProjDispatch::Existing;
    }
    match route {
        Some(ops::ExactLmHeadRoute::Exact(_)) => ExactAttnOProjDispatch::Exact,
        Some(ops::ExactLmHeadRoute::SerialK1(_)) => ExactAttnOProjDispatch::PerRowK1,
        None => ExactAttnOProjDispatch::Existing,
    }
}

/// Cached `ATLAS_ATTN_QKV_MEGA` env-var lookup. When `1`/`true` (or
/// auto-enabled on wide-SM GPUs with `sm_count > 64`), the multi-seq RoPE and
/// KV-cache-write phases each collapse from N sequential launches into a
/// single batched launch. Default off on `<= 64` SMs. Bit-identical to the
/// sequential path when the strided RoPE kernel is present.
pub(super) fn attn_qkv_mega_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ATLAS_ATTN_QKV_MEGA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

impl Qwen3AttentionLayer {
    /// Resolve native-target tail selectors before any hyperconnection, KV, or
    /// recurrent state can be modified. `C2..=ATTN_GATE_BATCH_WIDE_MAX_ROWS`
    /// (covering `2..=256`, including the legacy `C5-C17` tier) are eligible
    /// for the exact strided gate kernel; an explicit request with a stale
    /// kernel bundle must fail here instead of after paged attention has run.
    /// The M17 O-projection experiment is ineligible below C9, but its strict
    /// parser is still evaluated so a malformed selector cannot surface mid-frame.
    pub(super) fn preflight_qwen4_native_tail_selectors(&self, rows: usize) -> Result<()> {
        let gate_requested = attn_gate_batched_requested()?;
        // Covers the full wide gate range `2..=ATTN_GATE_BATCH_WIDE_MAX_ROWS`
        // (strict superset of legacy `(5..=17).contains(&rows)`).
        if gate_requested && (2..=ATTN_GATE_BATCH_WIDE_MAX_ROWS).contains(&rows) {
            anyhow::ensure!(
                self.gated && self.sigmoid_gate_mul_batched_k.0 != 0,
                "ATLAS_ATTN_GATE_BATCHED=1 requires sigmoid_gate_mul_batched for native Qwen4 C{rows}"
            );
        }
        match attn_kv_write_batch_route(
            attn_kv_write_batched_requested()?,
            true,
            matches!(self.kv_dtype, KvCacheDtype::Bf16),
            rows,
            self.reshape_cache_k.0 != 0,
        ) {
            AttnKvWriteBatchRoute::Disabled | AttnKvWriteBatchRoute::Complete => {}
            AttnKvWriteBatchRoute::Ineligible => anyhow::bail!(
                "ATLAS_ATTN_KV_WRITE_BATCHED=1 requires native Qwen4 C2-C{} with BF16 KV",
                crate::layers::QWEN4_NATIVE_MAX_ROWS
            ),
            AttnKvWriteBatchRoute::Missing => {
                anyhow::bail!("ATLAS_ATTN_KV_WRITE_BATCHED=1 requires reshape_and_cache_flash")
            }
        }
        let _ = attn_o_proj_m17_astage_requested()?;
        Ok(())
    }

    fn ms_o_proj_nvfp4_k1(
        &self,
        c: &MultiSeqCtx<'_>,
        attn_out: DevicePtr,
        o_out: DevicePtr,
    ) -> Result<()> {
        for i in 0..c.n {
            let attn_out_i = attn_out.offset(i * c.q_dim as usize * c.bf16);
            let o_out_i = o_out.offset(i * c.h * c.bf16);
            ops::w4a16_decode_gemv(
                c.fwd.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                self.gemv_sw,
                attn_out_i,
                &self.attn.o_proj,
                o_out_i,
                c.h as u32,
                c.nq * c.hd,
                c.stream,
            )?;
        }
        Ok(())
    }

    /// Phase 3: per-token RoPE (each sequence has its own position).
    ///
    /// When `ATLAS_ATTN_QKV_MEGA=1` and the strided kernel is present,
    /// all N tokens are RoPE'd in a single kernel launch via
    /// `rope_forward_strided_b3`. Otherwise the original per-token loop
    /// runs (1 launch per token).
    pub(super) fn ms_phase_rope(&self, c: &MultiSeqCtx<'_>, meta: AttnMetadataDev) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;

        let mega_requested = attn_qkv_mega_enabled();
        let batch_k = if mega_requested || (n > 1 && atlas_core::device::active_sm_count() > 64) {
            Some(cached_multiseq_attn_batch_kernels(fwd.gpu))
        } else {
            None
        };
        if mega_requested
            && self.yarn.is_some()
            && !(self.mrope_interleaved && batch_k.is_some_and(|k| k.mrope_yarn_multi_seq.0 != 0))
        {
            anyhow::bail!(
                "ATLAS_ATTN_QKV_MEGA=1 is incompatible with YaRN; \
                 no YaRN-scaled multi-seq RoPE kernel is available for this layer"
            );
        }

        let batch_rope = n > 1 && (mega_requested || atlas_core::device::active_sm_count() > 64);
        if batch_rope && per_seq_qkv.is_multiple_of(bf16) && q_proj_bytes.is_multiple_of(bf16) {
            let qkv_stride_bf16 = (per_seq_qkv / bf16) as u32;
            let k_offset_bf16 = (q_proj_bytes / bf16) as u32;
            let rotary_dim = self
                .rotary_dim_override
                .unwrap_or(fwd.config.rotary_dim() as u32);
            let theta = self
                .rope_theta_override
                .unwrap_or(fwd.config.rope_theta as f32);
            if self.mrope_interleaved {
                let batch_k =
                    batch_k.unwrap_or_else(|| cached_multiseq_attn_batch_kernels(fwd.gpu));
                if let Some(yarn) = self.yarn {
                    if batch_k.mrope_yarn_multi_seq.0 != 0 {
                        return ops::rope_mrope_interleaved_yarn_multi_seq(
                            fwd.gpu,
                            batch_k.mrope_yarn_multi_seq,
                            qkv_buf,
                            meta.positions,
                            meta.positions_h,
                            meta.positions_w,
                            n as u32,
                            qkv_stride_bf16,
                            k_offset_bf16,
                            nq,
                            nkv,
                            hd,
                            rotary_dim,
                            theta,
                            yarn.factor,
                            yarn.correction_low,
                            yarn.correction_high,
                            yarn.attention_factor,
                            stream,
                        );
                    }
                } else if batch_k.mrope_multi_seq.0 != 0 {
                    return ops::rope_mrope_interleaved_multi_seq(
                        fwd.gpu,
                        batch_k.mrope_multi_seq,
                        qkv_buf,
                        meta.positions,
                        meta.positions_h,
                        meta.positions_w,
                        n as u32,
                        qkv_stride_bf16,
                        k_offset_bf16,
                        nq,
                        nkv,
                        hd,
                        rotary_dim,
                        theta,
                        stream,
                    );
                }
            } else if self.yarn.is_none() && self.rope_strided_b3_k.0 != 0 {
                return ops::rope_strided_b3(
                    fwd.gpu,
                    self.rope_strided_b3_k,
                    qkv_buf,
                    meta.positions,
                    n as u32,
                    qkv_stride_bf16,
                    k_offset_bf16,
                    nq,
                    nkv,
                    hd,
                    rotary_dim,
                    theta,
                    stream,
                );
            }
        }

        // Fallback: per-token loop (N launches).
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let pos_i = meta.positions.offset(i * 4); // u32 per position
            if self.mrope_interleaved {
                self.apply_mrope(
                    fwd.gpu,
                    q_out_i,
                    k_out_i,
                    pos_i,
                    meta.positions_h.offset(i * 4),
                    meta.positions_w.offset(i * 4),
                    1,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(fwd.config.rotary_dim() as u32),
                    self.rope_theta_override
                        .unwrap_or(fwd.config.rope_theta as f32),
                    stream,
                )?;
            } else {
                anyhow::ensure!(
                    self.yarn.is_none(),
                    "YaRN RoPE is not supported on non-MRoPE layers without precomputed qwen4_yarn_inv_freq"
                );
                ops::rope(
                    fwd.gpu,
                    self.rope_k,
                    q_out_i,
                    k_out_i,
                    pos_i,
                    1,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(fwd.config.rotary_dim() as u32),
                    self.rope_theta_override
                        .unwrap_or(fwd.config.rope_theta as f32),
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Phase 4: per-token KV cache write.
    ///
    /// When `ATLAS_ATTN_QKV_MEGA=1` or on wide-SM GPUs (`sm_count > 64`), all N
    /// tokens write in a single kernel launch for BF16/FP8/NVFP4 KV caches
    /// (unless FP8 calibration is actively observing contiguous rows). Turbo
    /// WHT layouts always take the per-token path below so each row's K/V is
    /// contiguous.
    pub(super) fn ms_phase_cache_write(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nkv,
            hd,
            bs,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let kv_stride = nkv * hd;

        // Batched KV cache write: single head-parallel launch covering all N
        // sequences (`grid = (N, num_kv_heads)`). Auto-enabled on wide-SM GPUs
        // (`sm_count > 64`) and under `ATLAS_ATTN_QKV_MEGA=1`.
        // Skipped for Turbo WHT layouts (which expect contiguous K/V) and while
        // FP8 calibration is actively observing contiguous rows.
        let batch_kv =
            n > 1 && (attn_qkv_mega_enabled() || atlas_core::device::active_sm_count() > 64);
        let fp8_calibrating = !fwd.graph_capture
            && self
                .fp8_calibration
                .as_ref()
                .is_some_and(|cal| cal.is_calibrating());
        if batch_kv && per_seq_qkv.is_multiple_of(bf16) {
            let per_token_stride_bf16 = (per_seq_qkv / bf16) as u32;
            let k_base = qkv_buf.offset(q_proj_bytes);
            let v_base = k_base.offset((nkv * hd) as usize * bf16);
            let batch_k = cached_multiseq_attn_batch_kernels(fwd.gpu);
            match self.kv_dtype {
                KvCacheDtype::Bf16 => {
                    if batch_k.cache_bf16_multi_seq.0 != 0 {
                        return ops::reshape_and_cache_multi_seq(
                            fwd.gpu,
                            batch_k.cache_bf16_multi_seq,
                            k_base,
                            v_base,
                            kv_cache.k_pool_ptr(self.attn_layer_idx),
                            kv_cache.v_pool_ptr(self.attn_layer_idx),
                            meta.slot,
                            n as u32,
                            nkv,
                            hd,
                            bs,
                            per_token_stride_bf16,
                            per_token_stride_bf16,
                            stream,
                        );
                    }
                    return self.write_kv_cache(
                        fwd.gpu,
                        k_base,
                        v_base,
                        kv_cache,
                        meta.slot,
                        n as u32,
                        nkv,
                        hd,
                        bs,
                        per_token_stride_bf16,
                        per_token_stride_bf16,
                        stream,
                        fwd.graph_capture,
                    );
                }
                KvCacheDtype::Fp8 if !fp8_calibrating => {
                    if batch_k.cache_fp8_multi_seq.0 != 0 {
                        let (k_scale, v_scale) = self.effective_fp8_scales();
                        return ops::reshape_and_cache_fp8_multi_seq(
                            fwd.gpu,
                            batch_k.cache_fp8_multi_seq,
                            k_base,
                            v_base,
                            kv_cache.k_pool_ptr(self.attn_layer_idx),
                            kv_cache.v_pool_ptr(self.attn_layer_idx),
                            meta.slot,
                            n as u32,
                            nkv,
                            hd,
                            bs,
                            k_scale,
                            v_scale,
                            per_token_stride_bf16,
                            per_token_stride_bf16,
                            kv_cache.cache_stride() as u64,
                            stream,
                        );
                    }
                    return self.write_kv_cache(
                        fwd.gpu,
                        k_base,
                        v_base,
                        kv_cache,
                        meta.slot,
                        n as u32,
                        nkv,
                        hd,
                        bs,
                        per_token_stride_bf16,
                        per_token_stride_bf16,
                        stream,
                        fwd.graph_capture,
                    );
                }
                KvCacheDtype::Nvfp4 => {
                    if self.active_bitfp4_kv(hd, bs).is_none()
                        && batch_k.cache_nvfp4_multi_seq.0 != 0
                    {
                        return ops::reshape_and_cache_nvfp4_multi_seq(
                            fwd.gpu,
                            batch_k.cache_nvfp4_multi_seq,
                            k_base,
                            v_base,
                            kv_cache.k_pool_ptr(self.attn_layer_idx),
                            kv_cache.v_pool_ptr(self.attn_layer_idx),
                            meta.slot,
                            n as u32,
                            nkv,
                            hd,
                            bs,
                            per_token_stride_bf16,
                            per_token_stride_bf16,
                            kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                            kv_cache.nvfp4_data_bytes() as u64,
                            stream,
                        );
                    }
                    return self.write_kv_cache(
                        fwd.gpu,
                        k_base,
                        v_base,
                        kv_cache,
                        meta.slot,
                        n as u32,
                        nkv,
                        hd,
                        bs,
                        per_token_stride_bf16,
                        per_token_stride_bf16,
                        stream,
                        fwd.graph_capture,
                    );
                }
                _ => {}
            }
        }

        // Fallback: N sequential launches.
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);
            let slot_i = meta.slot.offset(i * 8); // i64 per slot
            self.write_kv_cache(
                fwd.gpu,
                k_out_i,
                v_out_i,
                kv_cache,
                slot_i,
                1,
                nkv,
                hd,
                bs,
                kv_stride,
                kv_stride,
                stream,
                fwd.graph_capture,
            )?;
        }
        Ok(())
    }

    /// Native-Qwen4 wrapper for the exact BF16 KV-write consolidation.
    ///
    /// The selector is resolved during whole-layer preflight, before HC/QKV/
    /// RoPE mutate any buffer.  Execution reuses that cached decision and logs
    /// engagement only after the single strided write has launched.
    pub(super) fn ms_phase_cache_write_qwen4_native(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        if !attn_kv_write_batched_requested()? {
            return self.ms_phase_cache_write(c, kv_cache, meta);
        }

        let route = attn_kv_write_batch_route(
            true,
            c.fwd.config.is_qwen4_exp(),
            matches!(self.kv_dtype, KvCacheDtype::Bf16),
            c.n,
            self.reshape_cache_k.0 != 0,
        );
        anyhow::ensure!(
            route == AttnKvWriteBatchRoute::Complete,
            "ATLAS_ATTN_KV_WRITE_BATCHED execution escaped native Qwen4 preflight"
        );
        anyhow::ensure!(
            c.per_seq_qkv.is_multiple_of(c.bf16),
            "native Qwen4 QKV row is not BF16 aligned"
        );
        let row_stride_bf16 = u32::try_from(c.per_seq_qkv / c.bf16)
            .map_err(|_| anyhow::anyhow!("native Qwen4 QKV row stride exceeds u32"))?;
        let k_base = c.qkv_buf.offset(c.q_proj_bytes);
        let v_base = k_base.offset((c.nkv * c.hd) as usize * c.bf16);
        self.write_kv_cache(
            c.fwd.gpu,
            k_base,
            v_base,
            kv_cache,
            meta.slot,
            c.n as u32,
            c.nkv,
            c.hd,
            c.bs,
            row_stride_bf16,
            row_stride_bf16,
            c.stream,
            c.fwd.graph_capture,
        )?;
        log_attn_kv_write_batched_engagement_once(c.n);
        Ok(())
    }

    /// Phase 5: build contiguous Q buffer + run BATCHED paged decode.
    /// Returns the attn_out buffer pointer for downstream phases.
    pub(super) fn ms_phase_paged_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        self.ms_phase_paged_decode_into(c, kv_cache, meta, c.fwd.buffers.attn_output())
    }

    /// Phase 5 targeting an explicit output slice `attn_out` (`[n, q_dim]` BF16)
    /// so cross-sequence batched DFlash verify can gather each sequence's `K`
    /// rows into disjoint slices of `buffers.attn_output()` with zero copies.
    pub(super) fn ms_phase_paged_decode_into(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
        attn_out: DevicePtr,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            hd,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let (k_dtype, _) = self.kv_dtype.kv_pair();
        let k_is_turbo = crate::layers::qwen3_attention::helpers::kv_dtype_is_turbo(k_dtype);
        let total_q_heads = u32::try_from(n)
            .ok()
            .and_then(|rows| rows.checked_mul(nq))
            .ok_or_else(|| anyhow::anyhow!("multi-seq Q head count overflow"))?;
        // Strided Q read: every paged_decode_attn* kernel takes `q_stride` in
        // BF16 elements (`Q + seq_idx * q_stride + q_head * hd`). When
        // `per_seq_qkv` is 16-byte aligned and K is not Turbo-rotated, read Q
        // directly from `qkv_buf` with `q_stride_elems = (per_seq_qkv / bf16)`,
        // avoiding N `cuMemcpyDtoDAsync` calls per attention layer. When K is
        // Turbo-rotated, pack Q contiguously first so `apply_turbo_q_wht_and_innerq`
        // rotates only `[n * nq, hd]` without mutating interleaved gate/K/V slices.
        let (q_contiguous, q_stride_elems) = if per_seq_qkv.is_multiple_of(16) && !k_is_turbo {
            (qkv_buf, (per_seq_qkv / bf16) as u32)
        } else {
            let q_contig = fwd.buffers.ssm_qkvz();
            for i in 0..n {
                let q_out_i = qkv_buf.offset(i * per_seq_qkv);
                fwd.gpu.copy_d2d_async(
                    q_out_i,
                    q_contig.offset(i * q_dim as usize * bf16),
                    q_dim as usize * bf16,
                    stream,
                )?;
            }
            (q_contig, nq * hd)
        };
        if k_is_turbo {
            self.apply_turbo_q_wht_and_innerq(fwd.gpu, q_contiguous, total_q_heads, hd, stream)?;
        }
        self.ms_phase_paged_decode_dispatch(
            c,
            kv_cache,
            meta,
            q_contiguous,
            q_stride_elems,
            attn_out,
        )?;
        self.apply_turbo_out_iwht(fwd.gpu, attn_out, total_q_heads, hd, stream)?;
        Ok(attn_out)
    }

    fn ms_phase_paged_decode_dispatch(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
        q_contiguous: DevicePtr,
        q_stride_elems: u32,
        attn_out: DevicePtr,
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            q_dim,
            max_seq_len_host,
            ..
        } = *c;
        let inv_sqrt_d = self.effective_attn_scale(hd);
        // ATLAS_TREE_AWARE_ATTN: pull optional KV indirection from the
        // forward context (populated by `verify_d.rs` when tree mode is
        // active). NULL keeps the kernel in legacy chain-mode.
        //
        // CUDA graph fix: `kv_indir_base_ptr` is a 1×i32 device buffer
        // (not the raw scalar) so a captured graph sees the fresh
        // `seq.seq_len` on each replay. Host writes the value into the
        // buffer before each step in `verify_d.rs`.
        let (kv_indir, kv_indir_base_ptr, kv_indir_stride) = match fwd.tree_aware_attn {
            Some(t) => (t.kv_indir, t.kv_indir_base_ptr, t.kv_indir_stride),
            None => (
                spark_runtime::gpu::DevicePtr::NULL,
                spark_runtime::gpu::DevicePtr::NULL,
                0u32,
            ),
        };

        // Preserve independent K1 split geometry. The explicit diagnostic
        // always takes this path; production does so only when a flat K-window
        // straddles an aggressive split threshold that a single batch-wide
        // split count cannot represent. Each serialized row receives
        // `max_seq_len_host` as a safe upper bound on its split-K geometry
        // (the kernel reads the exact per-row length from `meta.seq_len`),
        // avoiding underestimates when concurrent sequences have independent lengths.
        let serial_requested = n > 1 && crate::layers::attn_paged_serial_enabled();
        let split_boundary = should_auto_serialize_paged_split_boundary(
            n,
            kv_indir != spark_runtime::gpu::DevicePtr::NULL,
            fwd.ddtree_parent_ids_dev.is_some(),
            self.flat_window_needs_k1_paged_rows(nq, n as u32, max_seq_len_host),
        );
        if serial_requested || split_boundary {
            if kv_indir != spark_runtime::gpu::DevicePtr::NULL {
                anyhow::bail!(
                    "DFLASH_CONTROL_PATH_PROOF path=attn_paged requested=true engaged=false \
                     requirement=serial paged attention requires a flat chain"
                );
            }
            let block_table_row_bytes = meta.max_blocks_per_seq as usize * 4;
            let q_row_bytes = q_dim as usize * bf16;
            let q_in_row_bytes = q_stride_elems as usize * bf16;
            for i in 0..n {
                self.run_paged_decode(
                    fwd.gpu,
                    q_contiguous.offset(i * q_in_row_bytes),
                    kv_cache,
                    attn_out.offset(i * q_row_bytes),
                    meta.block_table.offset(i * block_table_row_bytes),
                    meta.seq_len.offset(i * 4),
                    meta.max_blocks_per_seq,
                    1,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    nq * hd,
                    fwd.buffers.splitk_workspace(),
                    spark_runtime::gpu::DevicePtr::NULL,
                    spark_runtime::gpu::DevicePtr::NULL,
                    0,
                    max_seq_len_host,
                    stream,
                )?;
            }
            if serial_requested {
                crate::model::control_engagement::engage(
                    crate::model::control_engagement::ControlPath::AttnPaged,
                )?;
            }
            return Ok(());
        }

        // ATLAS_TREE_KV_PACK two-pool fast path:
        // 1. Scatter ancestor KV into per-layer scratch (one CTA per
        //    (seq, ancestor slot), each copies one token's K+V).
        // 2. Call the existing paged_decode_attn_fp8 kernel with BOTH the
        //    real cache (for prior-linear context, fast BC=4) AND the
        //    pack pool (for the tree window, fast BC=4). The kernel uses
        //    `pos < kv_indir_base` → real cache, `pos >= kv_indir_base`
        //    → pack pool. Indirection table is still passed but the
        //    pack-pool branch takes precedence inside the kernel.
        //
        // This keeps linear context attention identical to chain mode
        // (so outputs stay coherent) while restoring BC=4 batching for
        // the tree window. FP8 only at present; NVFP4 path falls through
        // to the slow indirected fallback.
        if let Some(t) = fwd.tree_aware_attn
            && let Some(pack) = t.pack
            && (self.attn_layer_idx as u32) < pack.num_attn_layers
            && matches!(self.kv_dtype, KvCacheDtype::Fp8)
        {
            self.run_packed_kv_attn_fp8(
                fwd.gpu,
                q_contiguous,
                q_stride_elems,
                attn_out,
                n as u32,
                nq,
                nkv,
                hd,
                bs,
                inv_sqrt_d,
                kv_cache,
                meta,
                t,
                pack,
                stream,
            )?;
            return Ok(());
        }

        // Qwen3.8 FP8 fused-query verify. Strict M=15/HDIM=128/non-tree
        // contract; any mismatch or missing PTX falls through unchanged.
        if crate::layers::flash_attn_kgamma_fp8_bc4_enabled()
            && fp8_bc4_kgamma_geometry(
                n,
                hd,
                kv_indir == spark_runtime::gpu::DevicePtr::NULL
                    && fwd.ddtree_parent_ids_dev.is_none(),
                self.paged_decode_kgamma_fp8_bc4_k.is_some(),
            )
            && matches!(self.kv_dtype, KvCacheDtype::Fp8)
            && let Some(kernel) = self.paged_decode_kgamma_fp8_bc4_k
        {
            static UNSAFE_FP8_KGAMMA: std::sync::Once = std::sync::Once::new();
            UNSAFE_FP8_KGAMMA.call_once(|| {
                tracing::error!(
                    "UNSAFE unverified FP8 Kgamma exact candidate engaged: M={n} HDIM={hd}; \
                     device raw-bit parity is not yet proven"
                );
            });
            let (k_scale, v_scale) = self.effective_fp8_scales();
            ops::paged_decode_attn_kgamma_fp8_bc4(
                fwd.gpu,
                kernel,
                q_contiguous,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                nq,
                nkv,
                hd,
                bs,
                inv_sqrt_d,
                k_scale,
                v_scale,
                q_stride_elems,
                kv_cache.cache_stride() as u64,
                n as u32,
                stream,
            )?;
            return Ok(());
        }

        // BF16 K-gamma fused route. Flat-only: all speculative rows share the
        // block table and differ only by their causal seq_len cutoff. A missing
        // symbol or any unsupported geometry falls through byte-for-byte to
        // the existing paged-decode implementation.
        if crate::layers::flash_attn_kgamma_bf16_enabled()
            && bf16_kgamma_geometry(
                n,
                hd,
                kv_indir == spark_runtime::gpu::DevicePtr::NULL
                    && fwd.ddtree_parent_ids_dev.is_none(),
                self.paged_decode_kgamma_bf16_k.is_some(),
            )
            && matches!(self.kv_dtype, KvCacheDtype::Bf16)
            && self.sliding_window.is_none()
            && let Some(kernel) = self.paged_decode_kgamma_bf16_k
        {
            ops::paged_decode_attn_kgamma_bf16(
                fwd.gpu,
                kernel,
                q_contiguous,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                nq,
                nkv,
                hd,
                bs,
                inv_sqrt_d,
                q_stride_elems,
                kv_cache.cache_stride() as u64,
                n as u32,
                stream,
            )?;
            return Ok(());
        }

        // Native Blackwell SM120/SM121 BitDecoding-FP4 decode & K=gamma verify path
        // (opt-in via `ATLAS_BITFP4_KV=1`, plus optional `ATLAS_BITFP4_RAZER_KV=1`).
        //
        // Once BitDecoding-FP4 is active on this layer, K is stored WHT-rotated and
        // V is stored block-16 channel-transposed (`V^T`), so execution MUST NEVER
        // fall through into legacy token-major NVFP4 kernels (`paged_decode_attn_kgamma_nvfp4*`):
        //   - Tree-aware / DDTree indirection -> fail closed (`anyhow::bail!`).
        //   - Single-sequence flat K=gamma (`c.shared_block_table && 2..=32`) ->
        //     `paged_decode_attn_bitfp4_kgamma_sm120` (16 stacked queries/CTA sharing
        //     row 0's block table).
        //   - Independent concurrent requests (`!c.shared_block_table`), `n == 1`, or
        //     `n > 32` -> route directly to `self.run_paged_decode` ->
        //     `paged_decode_attn_bitfp4_sm120` (indexes `block_tables[blockIdx.z]`
        //     per sequence, preventing cross-tenant KV reads).
        if let Some((bitfp4_k, flags)) = self.active_bitfp4_kv(hd, bs) {
            if kv_indir != spark_runtime::gpu::DevicePtr::NULL
                || fwd.ddtree_parent_ids_dev.is_some()
            {
                anyhow::bail!(
                    "ATLAS_BITFP4_KV=1 does not support tree-aware or DDTree KV indirection \
                     (layer {}); unset ATLAS_BITFP4_KV or disable tree speculation",
                    self.attn_layer_idx
                );
            }
            if crate::layers::qwen3_attention::bitfp4_kv::bitfp4_kgamma_eligible(
                n,
                c.shared_block_table,
            ) {
                let workspace = fwd.buffers.splitk_workspace();
                let slot_bytes = n
                    .saturating_mul(nq as usize)
                    .saturating_mul(hd as usize + 2)
                    .saturating_mul(std::mem::size_of::<f32>());
                let max_splits_cap =
                    if workspace == spark_runtime::gpu::DevicePtr::NULL || slot_bytes == 0 {
                        1
                    } else {
                        (fwd.buffers.sizes().splitk_workspace / slot_bytes).clamp(1, 64) as u32
                    };
                let num_splits =
                    crate::layers::qwen3_attention::bitfp4_kv::compute_bitfp4_kgamma_splits(
                        nq,
                        nkv,
                        n as u32,
                        max_seq_len_host,
                        atlas_core::device::active_sm_count(),
                        max_splits_cap,
                    );
                ops::paged_decode_attn_bitfp4_kgamma_sm120(
                    fwd.gpu,
                    bitfp4_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    workspace,
                    meta.block_table,
                    meta.seq_len,
                    meta.max_blocks_per_seq,
                    nq,
                    nkv,
                    inv_sqrt_d,
                    num_splits,
                    q_stride_elems,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    n as u32,
                    flags,
                    ops::BitFp4Bf16Sink::default(),
                    stream,
                )?;
                if num_splits > 1 {
                    ops::paged_decode_attn_bitfp4_reduce(
                        fwd.gpu,
                        bitfp4_k.reduce,
                        workspace,
                        attn_out,
                        meta.seq_len,
                        nq,
                        num_splits,
                        n as u32,
                        stream,
                    )?;
                }
                return Ok(());
            }
            return self.run_paged_decode(
                fwd.gpu,
                q_contiguous,
                kv_cache,
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                n as u32,
                nq,
                nkv,
                hd,
                bs,
                inv_sqrt_d,
                q_stride_elems,
                fwd.buffers.splitk_workspace(),
                spark_runtime::gpu::DevicePtr::NULL,
                spark_runtime::gpu::DevicePtr::NULL,
                0,
                max_seq_len_host,
                stream,
            );
        }

        // FlashAttention-v2 inspired K=γ-fused path. Active when:
        //   - `ATLAS_FLASH_ATTN_KGAMMA=1`
        //   - num_seqs ≥ 8 (K=γ verify shape; chain/draft stay on legacy)
        //   - num_seqs ≤ 32 (kernel QTILE_MAX bound)
        //   - NVFP4 KV cache + HDIM=256 + kgamma kernel resolved
        //   - No tree-aware indirection (legacy kernel handles tree mode)
        //
        // Collapses 17×4=68 CTAs → 4 CTAs (one per q_head), loads each
        // KV vector once per warp and reuses across 2-3 owned queries.
        // On wide discrete GPUs (>64 SMs) at short contexts (`< 1024`), single-CTA
        // per q_head kgamma (24 CTAs) starves the GPU while `run_paged_decode`
        // below launches `n * nq = 192` CTAs; keep kgamma for `>= 1024` where
        // split-K fans out across all SMs. GB10 (`<= 64` SMs) is unchanged.
        let kgamma_ctx_ok = atlas_core::device::active_sm_count() <= 64 || max_seq_len_host >= 1024;
        if crate::layers::flash_attn_kgamma_enabled()
            && kgamma_ctx_ok
            && (8..=32).contains(&n)
            && matches!(self.kv_dtype, KvCacheDtype::Nvfp4)
            && hd == 256
            && kv_indir == spark_runtime::gpu::DevicePtr::NULL
            && let Some(kgamma_k) = self.paged_decode_kgamma_k
        {
            // Split-K fork: when seq_len is long enough that a single-CTA per
            // q_head leaves SMs idle, partition the KV history across
            // `num_splits` CTAs per q_head and follow with a reduce kernel.
            // With num_q_heads=4 and num_splits=12 the grid becomes 48 CTAs —
            // one per SM on GB10 sm_120.
            //
            // Gates:
            //   - ATLAS_FLASH_ATTN_KGAMMA_SPLITK=1 (off by default)
            //   - max_seq_len_host >= 1024 (short context: single-CTA wins)
            //   - both splitk + reduce kernels resolved
            //
            // num_splits formula targets ~active_sm_count() / num_q_heads CTAs
            // (12 for 4-head on 48-SM GB10, up to 47 on 188-SM discrete GPUs),
            // capped so each split processes at least SPLIT_TILE positions and
            // never exceeds MAX_SPLITS_CAP = 64.
            // Warn only on the SYMBOL being absent — the `max_seq_len_host` and
            // `num_splits >= 2` conditions below are legitimate runtime
            // decisions, not a stale kernel cache, and warning on those would
            // be the noise that trains people to skip the log.
            if crate::layers::flash_attn_kgamma_splitk_enabled()
                && (self.paged_decode_kgamma_splitk_k.is_none()
                    || self.paged_decode_kgamma_reduce_k.is_none())
            {
                static WARNED: std::sync::Once = std::sync::Once::new();
                crate::layers::warn_kernel_fallback(
                    &WARNED,
                    "ATLAS_FLASH_ATTN_KGAMMA_SPLITK=1",
                    "paged_decode_attn_kgamma_nvfp4_splitk / _reduce_nvfp4",
                    "the K=γ paged decode attention stays single-CTA-per-head — \
                     nq CTAs with no split of the KV axis",
                );
            }
            if crate::layers::flash_attn_kgamma_splitk_enabled()
                && max_seq_len_host >= 1024
                && let Some(splitk_k) = self.paged_decode_kgamma_splitk_k
                && let Some(reduce_k) = self.paged_decode_kgamma_reduce_k
            {
                let num_splits = compute_kgamma_splitk_splits(
                    nq,
                    max_seq_len_host,
                    atlas_core::device::active_sm_count(),
                );
                if num_splits >= 2 {
                    let workspace = fwd.buffers.splitk_workspace();
                    ops::paged_decode_attn_kgamma_nvfp4_splitk(
                        fwd.gpu,
                        splitk_k,
                        q_contiguous,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        workspace,
                        meta.block_table,
                        meta.seq_len,
                        meta.max_blocks_per_seq,
                        nq,
                        nkv,
                        hd,
                        bs,
                        inv_sqrt_d,
                        num_splits,
                        q_stride_elems,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.nvfp4_data_bytes() as u64,
                        n as u32,
                        stream,
                    )?;
                    ops::paged_decode_attn_kgamma_reduce_nvfp4(
                        fwd.gpu, reduce_k, workspace, attn_out, nq, num_splits, n as u32, stream,
                    )?;
                    return Ok(());
                }
            }

            // FA2-grafted fast path: cp.async double-buffered KV tile
            // staging in shared memory. Same caller contract as the
            // baseline single-CTA kgamma kernel. Gated by
            // ATLAS_FA2_KGAMMA=1 (off by default). Takes precedence over
            // VEC when both are enabled — they share the same kernel
            // shape but FA2 stages KV through SMEM with a 2-stage
            // pipeline whereas VEC dequants direct-from-global in pairs.
            if crate::layers::flash_attn_kgamma_fa2_enabled()
                && self.paged_decode_kgamma_fa2_k.is_none()
            {
                static WARNED: std::sync::Once = std::sync::Once::new();
                crate::layers::warn_kernel_fallback(
                    &WARNED,
                    "ATLAS_FA2_KGAMMA=1",
                    "paged_decode_attn_kgamma_nvfp4_fa2",
                    "the K=γ paged decode attention stays on the baseline \
                     single-CTA kgamma kernel with no cp.async SMEM staging",
                );
            }
            if crate::layers::flash_attn_kgamma_fa2_enabled()
                && let Some(fa2_k) = self.paged_decode_kgamma_fa2_k
            {
                // One-time INFO log to confirm the FA2 path is active.
                static FA2_FIRST_DISPATCH: std::sync::Once = std::sync::Once::new();
                FA2_FIRST_DISPATCH.call_once(|| {
                    tracing::info!(
                        "ATLAS_FA2_KGAMMA: dispatching paged_decode_attn_kgamma_nvfp4_fa2 (n={n})"
                    );
                });
                ops::paged_decode_attn_kgamma_nvfp4_fa2(
                    fwd.gpu,
                    fa2_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    meta.seq_len,
                    meta.max_blocks_per_seq,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    q_stride_elems,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    n as u32,
                    stream,
                )?;
                return Ok(());
            }

            // VEC fast path: 16 warps/CTA + 2-position dequant batching.
            // Same caller contract as the baseline single-CTA kgamma kernel.
            // Gated by ATLAS_KGAMMA_VECDEQUANT=1 (off by default).
            if crate::layers::flash_attn_kgamma_vecdequant_enabled()
                && let Some(vec_k) = self.paged_decode_kgamma_vec_k
            {
                ops::paged_decode_attn_kgamma_nvfp4_vec(
                    fwd.gpu,
                    vec_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    meta.seq_len,
                    meta.max_blocks_per_seq,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    q_stride_elems,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    n as u32,
                    stream,
                )?;
                return Ok(());
            }

            ops::paged_decode_attn_kgamma_nvfp4(
                fwd.gpu,
                kgamma_k,
                q_contiguous,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                nq,
                nkv,
                hd,
                bs,
                inv_sqrt_d,
                q_stride_elems,
                kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                kv_cache.nvfp4_data_bytes() as u64,
                n as u32,
                stream,
            )?;
            return Ok(());
        }

        let tree_or_indir =
            kv_indir != spark_runtime::gpu::DevicePtr::NULL || fwd.ddtree_parent_ids_dev.is_some();
        if self.try_run_gqa12_ragged_paged_decode(
            fwd.gpu,
            q_contiguous,
            kv_cache,
            attn_out,
            fwd.buffers.ssm_deinterleaved(),
            fwd.buffers.sizes().ssm_deinterleaved,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            n as u32,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            q_stride_elems,
            tree_or_indir,
            stream,
        )? {
            return Ok(());
        }

        self.run_paged_decode(
            fwd.gpu,
            q_contiguous,
            kv_cache,
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            n as u32,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            q_stride_elems,
            fwd.buffers.splitk_workspace(),
            kv_indir,
            kv_indir_base_ptr,
            kv_indir_stride,
            max_seq_len_host,
            stream,
        )?;
        Ok(())
    }

    /// ATLAS_TREE_KV_PACK two-pool dispatcher (FP8). Scatters tree-window
    /// KV into per-layer scratch, then runs the existing FP8 attn kernel
    /// with the real cache as `K_cache/V_cache` (linear-context, fast
    /// BC=4) and the scratch as `K_pack_pool/V_pack_pool` (tree-window,
    /// fast BC=4 over contiguous slots). The kernel routes positions
    /// `< kv_indir_base` to the real cache and `>= kv_indir_base` to the
    /// pack pool — restores BC=4 batching for the entire query range.
    #[allow(clippy::too_many_arguments)]
    fn run_packed_kv_attn_fp8(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        q_contiguous: DevicePtr,
        q_stride_elems: u32,
        attn_out: DevicePtr,
        n: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        bs: u32,
        inv_sqrt_d: f32,
        kv_cache: &PagedKvCache,
        meta: crate::layer::AttnMetadataDev,
        tree: crate::layer::TreeAwareAttn,
        pack: crate::layer::TreeKvPack,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;

        let src_k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        let src_v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
        // SAFETY: `scratch_k_ptrs` / `scratch_v_ptrs` reference a `Vec<DevicePtr>`
        // owned by `TransformerModel` (allocated at init, never resized).
        // Bounds-checked by the caller (`attn_layer_idx < num_attn_layers`).
        let dst_k_pool = unsafe { *pack.scratch_k_ptrs.add(self.attn_layer_idx) };
        let dst_v_pool = unsafe { *pack.scratch_v_ptrs.add(self.attn_layer_idx) };

        // Scatter: grid (n, max_chain_len). Each CTA copies one ancestor's
        // K+V from the real paged cache → scratch.
        let scatter_k = pack.scatter_fp8_kernel;
        if scatter_k.0 == 0 {
            anyhow::bail!("tree_kv_pack: scatter_fp8 kernel not loaded");
        }
        let max_chain_len = pack.block_size;
        KernelLaunch::new(gpu, scatter_k)
            .grid([n, max_chain_len, 1])
            .block([256, 1, 1])
            .arg_ptr(src_k_pool)
            .arg_ptr(src_v_pool)
            .arg_ptr(meta.block_table)
            .arg_ptr(tree.kv_indir)
            .arg_ptr(dst_k_pool)
            .arg_ptr(dst_v_pool)
            .arg_u32(n)
            .arg_u32(max_chain_len)
            .arg_u32(nkv)
            .arg_u32(hd)
            .arg_u32(pack.cache_block_size)
            .arg_u32(pack.cache_max_blocks_per_seq)
            .arg_u32(tree.kv_indir_stride)
            // CUDA graph fix: `abs_base` is now a device-buffer pointer so a
            // captured graph reads the fresh `seq.seq_len` on each replay.
            .arg_ptr(pack.abs_base_ptr)
            .launch(stream)?;

        // Now run the standard FP8 attn — same metadata as chain mode, but
        // hand the scratch in as the pack-pool. Kernel will:
        //   pos in [0..kv_indir_base)        → real cache, BC=4 batched
        //   pos in [kv_indir_base..seq_len)  → scratch, BC=4 batched
        // and feed both into the same online-softmax accumulator.
        use crate::layers::ops;
        let (k_scale, v_scale) = self.effective_fp8_scales();
        ops::paged_decode_attn_fp8(
            gpu,
            self.paged_decode_k,
            q_contiguous,
            src_k_pool,
            src_v_pool,
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            n,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            k_scale,
            v_scale,
            q_stride_elems,
            kv_cache.cache_stride() as u64,
            // Indirection still passed (legacy contract); the kernel's
            // pack-pool branch takes precedence when pack_pool != null.
            // CUDA graph fix: kv_indir_base is now a device-buffer ptr.
            tree.kv_indir,
            tree.kv_indir_base_ptr,
            tree.kv_indir_stride,
            // Pack pool: routes tree-window reads through the scratch.
            dst_k_pool,
            dst_v_pool,
            max_chain_len,
            stream,
        )?;
        Ok(())
    }

    /// Phase 6: gate multiply (when gated) + O projection. Writes to
    /// `o_out`. Returns the o_out buffer pointer.
    pub(super) fn ms_phase_o_proj(
        &self,
        c: &MultiSeqCtx<'_>,
        attn_out: DevicePtr,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            hd,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;

        // Resolve the experimental projection before the gate mutates
        // `attn_out`. A requested stale symbol or incompatible projection
        // override therefore fails without a partial O-phase effect.
        let ordinary_nvfp4 = self
            .o_weight
            .as_ref()
            .and_then(|weight| weight.as_fp8())
            .is_none()
            && self.o_dense_bf16_decode.is_none()
            && self.o_dense_bf16.is_none()
            && !self.attn.o_proj.is_null();
        let serial_o_proj = n > 1 && crate::layers::attn_out_serial_enabled();
        let o_proj_m17_astage_route = attn_o_proj_m17_astage_route(
            attn_o_proj_m17_astage_requested()?,
            ordinary_nvfp4,
            self.gated,
            n,
            self.w4a16_exact_o_proj_kernels.m17_astage_present(),
            ops::w4a16_gemv_rt2_enabled(),
            serial_o_proj,
        );
        match o_proj_m17_astage_route {
            AttnOProjM17AStageRoute::Missing => anyhow::bail!(
                "ATLAS_ATTN_O_PROJ_EXACT_M17_ASTAGE=1 requires \
                 w4a16_gemv_batch_logits_exact_m17_astage for eligible gated ordinary-NVFP4 M=9..17"
            ),
            AttnOProjM17AStageRoute::Conflict => anyhow::bail!(
                "ATLAS_ATTN_O_PROJ_EXACT_M17_ASTAGE=1 conflicts with \
                 ATLAS_W4A16_GEMV_RT2=1 or the serial attention-O control"
            ),
            AttnOProjM17AStageRoute::Disabled
            | AttnOProjM17AStageRoute::Ineligible
            | AttnOProjM17AStageRoute::Complete => {}
        }

        if self.gated {
            let gate_stride = if per_seq_qkv.is_multiple_of(bf16) {
                u32::try_from(per_seq_qkv / bf16).unwrap_or(0)
            } else {
                0
            };
            match attn_gate_batch_route_wide(
                attn_gate_batched_mode()?,
                atlas_core::device::active_sm_count(),
                self.gated,
                n,
                q_dim,
                gate_stride,
                self.sigmoid_gate_mul_batched_k.0 != 0,
            ) {
                AttnGateBatchRoute::Complete => {
                    static BATCHED_GATE: std::sync::Once = std::sync::Once::new();
                    BATCHED_GATE.call_once(|| {
                        tracing::info!(
                            "ENGAGED ATLAS_ATTN_GATE_BATCHED: multi_seq M={n} dim={q_dim} stride={gate_stride}"
                        );
                    });
                    ops::sigmoid_gate_mul_batched(
                        fwd.gpu,
                        self.sigmoid_gate_mul_batched_k,
                        attn_out,
                        qkv_buf.offset(q_dim as usize * bf16),
                        attn_out,
                        q_dim,
                        gate_stride,
                        n as u32,
                        stream,
                    )?;
                }
                AttnGateBatchRoute::Missing => {
                    anyhow::bail!(
                        "ATLAS_ATTN_GATE_BATCHED=1 requires sigmoid_gate_mul_batched for eligible gated M=2..={ATTN_GATE_BATCH_WIDE_MAX_ROWS}"
                    );
                }
                AttnGateBatchRoute::Disabled | AttnGateBatchRoute::Ineligible => {
                    for i in 0..n {
                        let gate_i = qkv_buf.offset(i * per_seq_qkv + q_dim as usize * bf16);
                        let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                        ops::sigmoid_gate_mul(
                            fwd.gpu,
                            self.sigmoid_gate_mul_k,
                            attn_out_i,
                            gate_i,
                            attn_out_i,
                            q_dim,
                            stream,
                        )?;
                    }
                }
            }
        }

        let o_out = fwd.buffers.moe_output();
        let exact_o_dispatch = exact_attn_o_proj_dispatch(
            ordinary_nvfp4,
            self.w4a16_exact_o_proj_kernels.route_for_rows(n as u32),
        );
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
        let sm_count = atlas_core::device::active_sm_count();
        // Native Qwen4 (`qwen4_attn_hyper.is_some()`) preflights and requires
        // the bit-exact O-projection tier, so exclude it from the non-exact
        // SM120 TC/cuBLASLt fast path.
        let o_proj_sm120_fast = ordinary_nvfp4
            && self.qwen4_attn_hyper.is_none()
            && !serial_o_proj
            && o_proj_m17_astage_route != AttnOProjM17AStageRoute::Complete
            && resolve_attn_proj_tc_for_sm_with_policy(
                raw_attn_proj_tc_override(),
                sm_count,
                n,
                det_batch,
            );
        let mut o_proj_done = false;
        if o_proj_sm120_fast {
            let sm120_max_m = crate::layers::dense_ffn::sm120_coalesced_max_m();
            if n <= sm120_max_m
                && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                    fwd.gpu,
                    attn_out,
                    &self.attn.o_proj,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    sm120_max_m,
                    stream,
                )?
            {
                o_proj_done = true;
            } else if n >= crate::layers::dense_ffn::CUBLASLT_NVFP4_MIN_M as usize
                && ops::try_cublaslt_nvfp4_multiseq_single(
                    fwd.gpu,
                    attn_out,
                    &self.attn.o_proj,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    h as u32,
                    fwd.graph_capture,
                    "attn_o_proj_multi_seq",
                    stream,
                )?
            {
                o_proj_done = true;
            } else if n <= 32
                && sm_count > 64
                && self.w4a16_gemm_t_m32_n64_splitk_k.0 != 0
                && self.reduce_splitk_k.0 != 0
                && let Some(nvfp4_t) = self.o_nvfp4_t.as_ref()
                && let Some(ws_ptr) = *self
                    .qkv_splitk_workspace
                    .lock()
                    .map_err(|_| anyhow::anyhow!("qkv_splitk_workspace lock poisoned"))?
            {
                let splits = crate::layers::attn_qkv_splitk().clamp(4, 8);
                ops::w4a16_gemm_n64_m32_splitk(
                    fwd.gpu,
                    self.w4a16_gemm_t_m32_n64_splitk_k,
                    self.reduce_splitk_k,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    ws_ptr,
                    n as u32,
                    h as u32,
                    nq * hd,
                    h as u32,
                    splits,
                    stream,
                )?;
                o_proj_done = true;
            } else if self.w4a16_gemm_t_m32_n64_k.0 != 0
                && let Some(nvfp4_t) = self.o_nvfp4_t.as_ref()
            {
                ops::w4a16_gemm_n64_m32(
                    fwd.gpu,
                    self.w4a16_gemm_t_m32_n64_k,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
                o_proj_done = true;
            }
        }
        if o_proj_done {
            // Already dispatched through coalesced MMA, cuBLASLt NVFP4, or m32_n64 TC above.
        } else if serial_o_proj {
            if self
                .o_weight
                .as_ref()
                .and_then(|weight| weight.as_fp8())
                .is_some()
                || self.o_dense_bf16_decode.is_some()
                || self.o_dense_bf16.is_some()
                || self.attn.o_proj.is_null()
            {
                anyhow::bail!(
                    "DFLASH_CONTROL_PATH_PROOF path=attn_out requested=true engaged=false \
                     requirement=serial output requires the ordinary K1 NVFP4 weight"
                );
            }
            // Losslessness diagnostic: match `attention_forward_oproj`
            // exactly, including its software-GEMV dispatch decision.
            self.ms_o_proj_nvfp4_k1(c, attn_out, o_out)?;
            crate::model::control_engagement::engage(
                crate::model::control_engagement::ControlPath::AttnOut,
            )?;
        } else if o_proj_m17_astage_route == AttnOProjM17AStageRoute::Complete {
            ops::w4a16_gemv_batch_logits_exact_m17_astage(
                fwd.gpu,
                self.w4a16_exact_o_proj_kernels.m17_astage(),
                attn_out,
                &self.attn.o_proj,
                o_out,
                n as u32,
                h as u32,
                nq * hd,
                stream,
            )?;
            static O_PROJ_M17_ASTAGE: std::sync::Once = std::sync::Once::new();
            O_PROJ_M17_ASTAGE.call_once(|| {
                tracing::info!(
                    "ENGAGED ATLAS_ATTN_O_PROJ_EXACT_M17_ASTAGE: multi_seq M={n} N={h} K={}",
                    nq * hd
                );
            });
        } else if let Some(o_bf16) = self
            .o_dense_bf16_decode
            .as_ref()
            .or(self.o_dense_bf16.as_ref())
        {
            // BF16 dense O projection (decode-only override or BF16 dense models).
            if n > 1
                && self.dense_gemm_k.0 != 0
                && !crate::layers::deterministic_batch::deterministic_batch_enabled()
            {
                ops::dense_gemm(
                    fwd.gpu,
                    self.dense_gemm_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            } else {
                for i in 0..n {
                    let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                    let o_out_i = o_out.offset(i * h * bf16);
                    ops::dense_gemv(
                        fwd.gpu,
                        self.dense_gemv_k,
                        attn_out_i,
                        o_bf16,
                        o_out_i,
                        h as u32,
                        nq * hd,
                        stream,
                    )?;
                }
            }
        } else if let Some(o_fp8) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
            // FP8 native: per-token w8a16_gemv for O projection.
            for i in 0..n {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                ops::w8a16_gemv(
                    fwd.gpu,
                    self.w8a16_gemv_k,
                    attn_out_i,
                    o_fp8.weight,
                    o_fp8.row_scale,
                    o_out_i,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            }
        } else if exact_o_dispatch == ExactAttnOProjDispatch::Exact {
            ops::w4a16_gemv_batch_logits_exact(
                fwd.gpu,
                self.w4a16_exact_o_proj_kernels,
                attn_out,
                &self.attn.o_proj,
                o_out,
                n as u32,
                h as u32,
                nq * hd,
                stream,
            )?;
        } else if exact_o_dispatch == ExactAttnOProjDispatch::PerRowK1 {
            self.ms_o_proj_nvfp4_k1(c, attn_out, o_out)?;
        } else {
            let has_m16_enabled = self.w4a16_gemm_t_m16_k.0 != 0
                && crate::layers::tc_nvfp4_m16_enabled()
                && super::qkv::tc_nvfp4_m16_ms_attn_enabled();
            match select_ms_o_proj_existing_fallback_route_with_policy(
                n,
                sm_count,
                h,
                !self.attn.o_proj.is_null(),
                self.o_nvfp4_t.is_some(),
                self.w4a16_gemm_t_m32_n64_k.0 != 0,
                self.w4a16_gemm_t_m128_k.0 != 0,
                has_m16_enabled,
                det_batch,
            ) {
                MsOProjFallbackRoute::PerRowK1 => {
                    self.ms_o_proj_nvfp4_k1(c, attn_out, o_out)?;
                }
                MsOProjFallbackRoute::Batch3 => {
                    ops::w4a16_gemv_batch3(
                        fwd.gpu,
                        self.w4a16_gemv_batch3_k,
                        attn_out,
                        &self.attn.o_proj,
                        o_out,
                        h as u32,
                        nq * hd,
                        stream,
                    )?;
                }
                MsOProjFallbackRoute::Batch2 => {
                    ops::w4a16_gemv_batch2(
                        fwd.gpu,
                        self.w4a16_gemv_batch2_k,
                        attn_out,
                        &self.attn.o_proj,
                        o_out,
                        h as u32,
                        nq * hd,
                        stream,
                    )?;
                }
                MsOProjFallbackRoute::TcM32N64 => {
                    if let Some(nvfp4_t) = self.o_nvfp4_t.as_ref() {
                        ops::w4a16_gemm_n64_m32(
                            fwd.gpu,
                            self.w4a16_gemm_t_m32_n64_k,
                            attn_out,
                            nvfp4_t,
                            o_out,
                            n as u32,
                            h as u32,
                            nq * hd,
                            stream,
                        )?;
                    }
                }
                MsOProjFallbackRoute::TcM128 => {
                    if let Some(nvfp4_t) = self.o_nvfp4_t.as_ref() {
                        ops::w4a16_gemm_n128_m128(
                            fwd.gpu,
                            self.w4a16_gemm_t_m128_k,
                            attn_out,
                            nvfp4_t,
                            o_out,
                            n as u32,
                            h as u32,
                            nq * hd,
                            stream,
                        )?;
                    }
                }
                MsOProjFallbackRoute::TcM16 => {
                    if let Some(nvfp4_t) = self.o_nvfp4_t.as_ref() {
                        ops::w4a16_gemm_n128_m16(
                            fwd.gpu,
                            self.w4a16_gemm_t_m16_k,
                            attn_out,
                            nvfp4_t,
                            o_out,
                            n as u32,
                            h as u32,
                            nq * hd,
                            stream,
                        )?;
                    }
                }
                MsOProjFallbackRoute::SerialGemv => {
                    for i in 0..n {
                        let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                        let o_out_i = o_out.offset(i * h * bf16);
                        ops::w4a16_gemv(
                            fwd.gpu,
                            self.w4a16_gemv_k,
                            attn_out_i,
                            &self.attn.o_proj,
                            o_out_i,
                            h as u32,
                            nq * hd,
                            stream,
                        )?;
                    }
                }
            }
        }
        Ok(o_out)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MsOProjFallbackRoute {
    PerRowK1,
    Batch3,
    Batch2,
    TcM32N64,
    TcM128,
    TcM16,
    SerialGemv,
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn select_ms_o_proj_existing_fallback_route_with_policy(
    n: usize,
    sm_count: u32,
    h: usize,
    has_o_proj: bool,
    has_o_nvfp4_t: bool,
    has_m32_n64: bool,
    has_m128: bool,
    has_m16_enabled: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> MsOProjFallbackRoute {
    if det_batch.is_enabled() && has_o_proj {
        return MsOProjFallbackRoute::PerRowK1;
    }
    if !det_batch.is_enabled() && n == 3 && has_o_proj {
        return MsOProjFallbackRoute::Batch3;
    }
    if !det_batch.is_enabled() && n == 2 && has_o_proj {
        return MsOProjFallbackRoute::Batch2;
    }
    if !det_batch.is_enabled() && has_o_nvfp4_t {
        if n > 3 && n <= 32 && has_m32_n64 {
            return MsOProjFallbackRoute::TcM32N64;
        }
        if n > 32 && sm_count > 64 && h <= 8192 && has_m32_n64 {
            return MsOProjFallbackRoute::TcM32N64;
        }
        if n > 32 && has_m128 {
            return MsOProjFallbackRoute::TcM128;
        }
        if n > 32 && has_m32_n64 {
            return MsOProjFallbackRoute::TcM32N64;
        }
        if n > 3 && n <= 32 && has_m16_enabled {
            return MsOProjFallbackRoute::TcM16;
        }
    }
    MsOProjFallbackRoute::SerialGemv
}

#[cfg(test)]
mod deterministic_batch_tests {
    use super::{MsOProjFallbackRoute, select_ms_o_proj_existing_fallback_route_with_policy};
    use crate::layers::deterministic_batch::DeterministicBatchPolicy;

    #[test]
    fn deterministic_batch_pins_o_proj_existing_fallback_to_per_row_k1_across_1_to_128() {
        for sm in [48u32, 188u32] {
            for m in [1usize, 2, 3, 4, 8, 16, 32, 64, 128] {
                assert_eq!(
                    select_ms_o_proj_existing_fallback_route_with_policy(
                        m,
                        sm,
                        5120,
                        true,
                        true,
                        true,
                        true,
                        true,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    MsOProjFallbackRoute::PerRowK1,
                    "O-proj existing fallback drifted at M={m} sm={sm}"
                );
            }
        }
        assert_eq!(
            select_ms_o_proj_existing_fallback_route_with_policy(
                2,
                48,
                5120,
                true,
                true,
                true,
                true,
                true,
                DeterministicBatchPolicy::Disabled,
            ),
            MsOProjFallbackRoute::Batch2
        );
        assert_eq!(
            select_ms_o_proj_existing_fallback_route_with_policy(
                3,
                48,
                5120,
                true,
                true,
                true,
                true,
                true,
                DeterministicBatchPolicy::Disabled,
            ),
            MsOProjFallbackRoute::Batch3
        );
        assert_eq!(
            select_ms_o_proj_existing_fallback_route_with_policy(
                8,
                48,
                5120,
                true,
                true,
                true,
                true,
                true,
                DeterministicBatchPolicy::Disabled,
            ),
            MsOProjFallbackRoute::TcM32N64
        );
        assert_eq!(
            select_ms_o_proj_existing_fallback_route_with_policy(
                64,
                48,
                5120,
                true,
                true,
                true,
                true,
                true,
                DeterministicBatchPolicy::Disabled,
            ),
            MsOProjFallbackRoute::TcM128
        );
    }
}

#[cfg(test)]
mod bf16_kgamma_tests {
    use super::bf16_kgamma_geometry;

    #[test]
    fn only_flat_supported_kverify_geometry_engages() {
        assert!(bf16_kgamma_geometry(15, 256, true, true));
        assert!(!bf16_kgamma_geometry(7, 256, true, true));
        assert!(!bf16_kgamma_geometry(33, 256, true, true));
        assert!(!bf16_kgamma_geometry(15, 128, true, true));
        assert!(!bf16_kgamma_geometry(15, 256, false, true));
        assert!(!bf16_kgamma_geometry(15, 256, true, false));
    }

    #[test]
    fn cuda_and_rust_abi_keep_the_same_ordered_tail() {
        let cuda = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/common/paged_decode_attn_kgamma_bf16.cu"
        ));
        let ops = include_str!("../../../ops/prefill_attn_b.rs");
        for needle in ["cache_stride", "num_qtile"] {
            assert!(cuda.contains(needle));
            assert!(ops.contains(needle));
        }
        assert!(cuda.contains("head_dim != HDIM"));
        assert!(cuda.contains("const int* table = block_tables"));
    }
}

#[cfg(test)]
mod fp8_bc4_kgamma_tests {
    use super::fp8_bc4_kgamma_geometry;

    #[test]
    fn only_qwen38_m15_flat_geometry_engages() {
        assert!(fp8_bc4_kgamma_geometry(15, 128, true, true));
        assert!(!fp8_bc4_kgamma_geometry(14, 128, true, true));
        assert!(!fp8_bc4_kgamma_geometry(15, 256, true, true));
        assert!(!fp8_bc4_kgamma_geometry(15, 128, false, true));
        assert!(!fp8_bc4_kgamma_geometry(15, 128, true, false));
    }

    #[test]
    fn cuda_and_rust_abi_retain_bc4_and_scales() {
        let cuda = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/common/paged_decode_attn_fp8.cu"
        ));
        let ops = include_str!("../../../ops/prefill_attn_b.rs");
        assert!(cuda.contains("paged_decode_attn_kgamma_fp8_bc4"));
        assert!(cuda.contains("#define BC 4"));
        assert!(cuda.contains("paged_decode_attn_kgamma_fp8_bc4_exact"));
        assert!(cuda.contains("global_warp / NUM_WARPS"));
        assert!(cuda.contains("global_warp % NUM_WARPS"));
        assert!(cuda.contains("for(int stride=NUM_WARPS/2;stride>0;stride>>=1)"));
        for needle in ["k_scale", "v_scale", "cache_stride", "num_qtile"] {
            assert!(cuda.contains(needle));
            assert!(ops.contains(needle));
        }
    }

    fn legacy_scalar_fixture(scores: &[f32], values: &[f32]) -> f32 {
        let chunk = scores.len().div_ceil(8);
        let mut states = [(f32::NEG_INFINITY, 0.0f32, 0.0f32); 8];
        for (warp, state) in states.iter_mut().enumerate() {
            let begin = (warp * chunk).min(scores.len());
            let end = (begin + chunk).min(scores.len());
            for block in (begin..end).collect::<Vec<_>>().chunks(4) {
                let mut next_m = state.0;
                for &i in block {
                    next_m = next_m.max(scores[i]);
                }
                let old = (state.0 - next_m).exp();
                state.1 *= old;
                state.2 *= old;
                for &i in block {
                    let e = (scores[i] - next_m).exp();
                    state.1 += e;
                    state.2 += e * values[i];
                }
                state.0 = next_m;
            }
        }
        for stride in [4, 2, 1] {
            for warp in 0..stride {
                let rhs = states[warp + stride];
                if rhs.1 > 0.0 {
                    let lhs = states[warp];
                    let m = lhs.0.max(rhs.0);
                    let a = (lhs.0 - m).exp();
                    let b = (rhs.0 - m).exp();
                    states[warp] = (m, lhs.1 * a + rhs.1 * b, lhs.2 * a + rhs.2 * b);
                }
            }
        }
        states[0].2 / states[0].1
    }

    fn grouped_paired_fixture(groups: [(&[f32], &[f32]); 2]) -> [f32; 2] {
        let chunks = groups.map(|(scores, _)| scores.len().div_ceil(8));
        let max_blocks = groups
            .iter()
            .zip(chunks)
            .map(|((scores, _), chunk)| {
                (0..8)
                    .map(|warp| {
                        let begin = (warp * chunk).min(scores.len());
                        let end = (begin + chunk).min(scores.len());
                        (end - begin).div_ceil(4)
                    })
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let mut states = [[(f32::NEG_INFINITY, 0.0f32, 0.0f32); 8]; 2];
        // Interleave BC=4 block steps across both warp-groups while keeping
        // each group's running online-softmax state disjoint.
        for block_idx in 0..max_blocks {
            for warp in 0..8 {
                for ((&(scores, values), chunk), group_states) in
                    groups.iter().zip(chunks).zip(states.iter_mut())
                {
                    let begin = (warp * chunk).min(scores.len());
                    let end = (begin + chunk).min(scores.len());
                    let b_start = begin + block_idx * 4;
                    if b_start >= end {
                        continue;
                    }
                    let b_end = (b_start + 4).min(end);
                    let state = &mut group_states[warp];
                    let mut next_m = state.0;
                    for &score in &scores[b_start..b_end] {
                        next_m = next_m.max(score);
                    }
                    let old = (state.0 - next_m).exp();
                    state.1 *= old;
                    state.2 *= old;
                    for (&score, &value) in
                        scores[b_start..b_end].iter().zip(&values[b_start..b_end])
                    {
                        let e = (score - next_m).exp();
                        state.1 += e;
                        state.2 += e * value;
                    }
                    state.0 = next_m;
                }
            }
        }
        for stride in [4, 2, 1] {
            for group_states in &mut states {
                for warp in 0..stride {
                    let rhs = group_states[warp + stride];
                    if rhs.1 > 0.0 {
                        let lhs = group_states[warp];
                        let m = lhs.0.max(rhs.0);
                        let a = (lhs.0 - m).exp();
                        let b = (rhs.0 - m).exp();
                        group_states[warp] = (m, lhs.1 * a + rhs.1 * b, lhs.2 * a + rhs.2 * b);
                    }
                }
            }
        }
        [
            states[0][0].2 / states[0][0].1,
            states[1][0].2 / states[1][0].1,
        ]
    }

    #[test]
    fn paired_groups_preserve_raw_legacy_bits() {
        let q0s: Vec<f32> = (0..73)
            .map(|i| ((i * 17 % 31) as f32 - 15.0) / 7.0)
            .collect();
        let q1s: Vec<f32> = (0..74)
            .map(|i| ((i * 11 % 29) as f32 - 14.0) / 9.0)
            .collect();
        let q0v: Vec<f32> = (0..73).map(|i| (i as f32).sin()).collect();
        let q1v: Vec<f32> = (0..74).map(|i| (i as f32).cos()).collect();
        let legacy = [
            legacy_scalar_fixture(&q0s, &q0v),
            legacy_scalar_fixture(&q1s, &q1v),
        ];
        // The grouped kernel interleaves execution across both groups while
        // owning disjoint per-group warp state.
        let grouped = grouped_paired_fixture([(&q0s, &q0v), (&q1s, &q1v)]);
        assert_eq!(legacy.map(f32::to_bits), grouped.map(f32::to_bits));
    }
}

#[cfg(test)]
mod kgamma_splitk_dynamic_sm_tests {
    use super::compute_kgamma_splitk_splits;

    #[test]
    fn kgamma_splitk_preserves_gb10_and_scales_safely_on_high_sm_gpus() {
        // 48-SM GB10 parity: 4 Q heads -> occupancy target 12.
        assert_eq!(compute_kgamma_splitk_splits(4, 1024, 48), 2);
        assert_eq!(compute_kgamma_splitk_splits(4, 8192, 48), 12);
        // 188-SM RTX PRO 6000: 4 Q heads -> occupancy target 47, bounded by seq_len/512 and 64.
        assert_eq!(compute_kgamma_splitk_splits(4, 8192, 188), 16);
        assert_eq!(compute_kgamma_splitk_splits(4, 32_768, 188), 47);
        // Extreme SM count still never exceeds MAX_SPLITS_CAP (64) or extended workspace cap.
        for sm_count in [48u32, 160, 170, 188, 512] {
            for seq_len in [1024u32, 4096, 16_384, 131_072] {
                let splits = compute_kgamma_splitk_splits(2, seq_len, sm_count);
                let workspace_max = seq_len.div_ceil(512).clamp(2, 64);
                assert!(
                    splits <= workspace_max && splits <= 64,
                    "sm={sm_count} seq_len={seq_len} splits={splits} workspace_max={workspace_max}"
                );
            }
        }
    }
}

#[cfg(test)]
mod sm120_multiseq_attn_tests {
    use super::super::qkv::{
        AttnProjTcOverride, parse_attn_proj_tc_override, resolve_attn_proj_tc_for_sm,
    };
    use super::{AttnGateBatchRoute, attn_gate_batch_route_wide};

    #[test]
    fn wide_gate_batch_auto_enables_on_sm120_and_supports_up_to_256_rows() {
        // GB10 (48 SMs) stays Disabled when env unset.
        assert_eq!(
            attn_gate_batch_route_wide(None, 48, true, 32, 6144, 14336, true),
            AttnGateBatchRoute::Disabled
        );
        // Explicit "0" disables even on SM120.
        assert_eq!(
            attn_gate_batch_route_wide(Some(false), 188, true, 64, 6144, 14336, true),
            AttnGateBatchRoute::Disabled
        );
        // SM120 auto-enables for 2..=256 when kernel present, gracefully falls back if absent.
        for rows in [2usize, 4, 16, 17, 18, 32, 64, 128, 256] {
            assert_eq!(
                attn_gate_batch_route_wide(None, 188, true, rows, 6144, 14336, true),
                AttnGateBatchRoute::Complete,
                "rows={rows}"
            );
            assert_eq!(
                attn_gate_batch_route_wide(None, 188, true, rows, 6144, 14336, false),
                AttnGateBatchRoute::Ineligible,
                "rows={rows}"
            );
            assert_eq!(
                attn_gate_batch_route_wide(Some(true), 48, true, rows, 6144, 14336, false),
                AttnGateBatchRoute::Missing,
                "rows={rows}"
            );
        }
        assert_eq!(
            attn_gate_batch_route_wide(None, 188, true, 257, 6144, 14336, true),
            AttnGateBatchRoute::Ineligible
        );
    }

    #[test]
    fn o_proj_tc_policy_preserves_gb10_and_enables_sm120_at_m4_plus() {
        assert!(!resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            48,
            64
        ));
        assert!(!resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            188,
            3
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            188,
            4
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            188,
            8
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            188,
            16
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            188,
            64
        ));
        assert!(!resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Disabled,
            188,
            64
        ));
        assert!(!resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Enabled,
            48,
            8
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Enabled,
            48,
            16
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Enabled,
            188,
            8
        ));
        assert_eq!(
            parse_attn_proj_tc_override(Some("0")),
            AttnProjTcOverride::Disabled
        );
        assert_eq!(
            parse_attn_proj_tc_override(Some("1")),
            AttnProjTcOverride::Enabled
        );
        assert_eq!(parse_attn_proj_tc_override(None), AttnProjTcOverride::Auto);
    }
}
