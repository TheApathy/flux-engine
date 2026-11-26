// SPDX-License-Identifier: AGPL-3.0-only

//! Exact K1-order Qwen4 K=5 attention projection staging.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

const QWEN4_K16_EXACT_ENV: &str = "ATLAS_QWEN4_K16_EXACT";
const WIDE_EXACT_QKV_OUTS_PER_BLOCK: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum WideExactQkvTier {
    M64,
    M80,
    M128,
}

impl WideExactQkvTier {
    const fn for_rows(rows: usize) -> Option<Self> {
        match rows {
            33..=64 => Some(Self::M64),
            65..=80 => Some(Self::M80),
            81..=128 => Some(Self::M128),
            _ => None,
        }
    }

    const fn qg_symbol(self) -> &'static str {
        match self {
            Self::M64 => "w4a16_gemv_qg_exact_m64",
            Self::M80 => "w4a16_gemv_qg_exact_m80",
            Self::M128 => "w4a16_gemv_qg_exact_m128",
        }
    }

    const fn dual_kv_symbol(self) -> &'static str {
        match self {
            Self::M64 => "w4a16_gemv_dual_kv_exact_m64",
            Self::M80 => "w4a16_gemv_dual_kv_exact_m80",
            Self::M128 => "w4a16_gemv_dual_kv_exact_m128",
        }
    }
}

fn wide_exact_qkv_kernels(
    gpu: &dyn GpuBackend,
    tier: WideExactQkvTier,
) -> Option<(KernelHandle, KernelHandle)> {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        parking_lot::Mutex<HashMap<(usize, WideExactQkvTier), (KernelHandle, KernelHandle)>>,
    > = OnceLock::new();
    let key = ((gpu as *const dyn GpuBackend).cast::<()>() as usize, tier);
    let map = CACHE.get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
    let (qg, dual_kv) = *map.lock().entry(key).or_insert_with(|| {
        let qg = gpu
            .kernel("w4a16_gemv_exact_attention", tier.qg_symbol())
            .unwrap_or(KernelHandle(0));
        let dual_kv = gpu
            .kernel("w4a16_gemv_exact_attention", tier.dual_kv_symbol())
            .unwrap_or(KernelHandle(0));
        (qg, dual_kv)
    });
    (qg.0 != 0 && dual_kv.0 != 0).then_some((qg, dual_kv))
}

#[allow(clippy::too_many_arguments)]
fn launch_wide_qg_exact(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (33..=128).contains(&rows),
        "wide exact attention QG rows must be in 33..=128, got {rows}"
    );
    ensure!(
        n == 2 * num_heads * head_dim,
        "gated-Q output width mismatch"
    );
    ensure!(
        k > 0 && k.is_multiple_of(16),
        "gated-Q K must be a positive multiple of 16"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, WIDE_EXACT_QKV_OUTS_PER_BLOCK), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(out_stride)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn launch_wide_dual_kv_exact(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    k_weight: &QuantizedWeight,
    k_output: DevicePtr,
    v_weight: &QuantizedWeight,
    v_output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (33..=128).contains(&rows),
        "wide exact attention dual-KV rows must be in 33..=128, got {rows}"
    );
    ensure!(n > 0, "exact attention dual-KV width must be non-zero");
    ensure!(
        k > 0 && k.is_multiple_of(16),
        "dual-KV K must be a positive multiple of 16"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, WIDE_EXACT_QKV_OUTS_PER_BLOCK), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(k_weight.weight)
        .arg_ptr(k_weight.weight_scale)
        .arg_f32(k_weight.weight_scale_2)
        .arg_ptr(k_output)
        .arg_ptr(v_weight.weight)
        .arg_ptr(v_weight.weight_scale)
        .arg_f32(v_weight.weight_scale_2)
        .arg_ptr(v_output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen4ExactQkvRowsRoute {
    LegacyK5K9,
    K16,
    K32,
}

const QWEN4_K16_EXACT_RECEIPT: &str = "ENGAGED ATLAS_QWEN4_K16_EXACT: M16 qg+dual_kv+qk_norm";
const QWEN4_K32_EXACT_RECEIPT: &str =
    "ENGAGED ATLAS_QWEN4_PREFILL_ATTN_CORE32: two ordered M16 qg+dual_kv projections";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen4QkvProjectionExtents {
    q_dim: u32,
    q_proj_dim: u32,
    kv_dim: u32,
    q_proj_bytes: usize,
    kv_bytes: usize,
    row_stride_bf16: u32,
    row_bytes: usize,
    total_bytes: usize,
}

const fn qwen4_k16_target_attention_geometry(
    is_qwen4_exp: bool,
    hidden_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> bool {
    is_qwen4_exp && hidden_size == 2560 && num_q_heads == 24 && num_kv_heads == 2 && head_dim == 256
}

fn qwen4_qkv_projection_extents(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rows: usize,
) -> Option<Qwen4QkvProjectionExtents> {
    let q_dim = num_q_heads.checked_mul(head_dim)?;
    let q_proj_dim = q_dim.checked_mul(2)?;
    let kv_dim = num_kv_heads.checked_mul(head_dim)?;
    let row_stride_bf16 = q_proj_dim.checked_add(kv_dim.checked_mul(2)?)?;
    let q_proj_bytes = (q_proj_dim as usize).checked_mul(2)?;
    let kv_bytes = (kv_dim as usize).checked_mul(2)?;
    let row_bytes = (row_stride_bf16 as usize).checked_mul(2)?;
    let total_bytes = rows.checked_mul(row_bytes)?;
    Some(Qwen4QkvProjectionExtents {
        q_dim,
        q_proj_dim,
        kv_dim,
        q_proj_bytes,
        kv_bytes,
        row_stride_bf16,
        row_bytes,
        total_bytes,
    })
}

const fn qwen4_exact_qkv_receipt(route: Qwen4ExactQkvRowsRoute) -> Option<&'static str> {
    match route {
        Qwen4ExactQkvRowsRoute::LegacyK5K9 => None,
        Qwen4ExactQkvRowsRoute::K16 => Some(QWEN4_K16_EXACT_RECEIPT),
        Qwen4ExactQkvRowsRoute::K32 => Some(QWEN4_K32_EXACT_RECEIPT),
    }
}

/// Keep the already-qualified K5/K9 switch independent from the experimental
/// gamma-15 route. In particular, enabling either switch must not widen the
/// other one's admitted row set.
const fn qwen4_exact_qkv_rows_route(
    rows: usize,
    legacy_k5_k9_enabled: bool,
    k16_enabled: bool,
) -> Option<Qwen4ExactQkvRowsRoute> {
    match rows {
        5 | 9 if legacy_k5_k9_enabled => Some(Qwen4ExactQkvRowsRoute::LegacyK5K9),
        16 if k16_enabled => Some(Qwen4ExactQkvRowsRoute::K16),
        _ => None,
    }
}

fn parse_exact_k16_env_value(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{QWEN4_K16_EXACT_ENV} must be exactly 0 or 1, got {other:?}"),
    }
}

fn qwen4_k16_exact_requested() -> Result<bool> {
    static CACHED: std::sync::OnceLock<std::result::Result<bool, String>> =
        std::sync::OnceLock::new();
    CACHED
        .get_or_init(|| match std::env::var(QWEN4_K16_EXACT_ENV) {
            Ok(value) => parse_exact_k16_env_value(Some(&value)).map_err(|e| e.to_string()),
            Err(std::env::VarError::NotPresent) => {
                parse_exact_k16_env_value(None).map_err(|e| e.to_string())
            }
            Err(std::env::VarError::NotUnicode(_)) => Err(format!(
                "{QWEN4_K16_EXACT_ENV} must be valid UTF-8 and exactly 0 or 1"
            )),
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

fn qwen4_k5_batch_attn_qkv_env() -> Option<&'static str> {
    static CACHED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CACHED
        .get_or_init(|| std::env::var("ATLAS_QWEN4_K5_BATCH_ATTN_QKV").ok())
        .as_deref()
}

fn qwen4_batched_k5_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_QWEN4_BATCHED_K5").ok().as_deref() == Some("1"))
}

impl Qwen3AttentionLayer {
    /// Projects an admitted set of contiguous Qwen4 attention inputs into a strided
    /// `[Q|gate|K|V]` arena. The projection kernels preserve the ordinary K1
    /// reduction order; Q/K normalization remains the same per-head K1 call.
    ///
    /// Returns `None` unless the explicit Qwen4 K5/K9 flag or the independent
    /// strict K16 flag is enabled. A requested route fails closed when the
    /// layer geometry or exact kernels do not match.
    pub(in super::super) fn qwen4_k5_project_qkv_exact(
        &self,
        normed: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, usize)>> {
        let legacy_k5_k9_enabled = qwen4_k5_batch_attn_qkv_env() == Some("1")
            || crate::layers::qwen4_verify_fast_host_selected();
        let route = if rows == 32 && crate::layers::qwen4_prefill_moe::attn16::core32_selected()? {
            Some(Qwen4ExactQkvRowsRoute::K32)
        } else if ((2..=crate::layers::QWEN4_HYBRID_MAX_ROWS).contains(&rows)
            && legacy_k5_k9_enabled
            && crate::layers::qwen4_hybrid_rows(rows))
            || ((2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&rows)
                && legacy_k5_k9_enabled
                && qwen4_batched_k5_enabled())
        {
            // K4 (ATLAS_QWEN4_K4_HYBRID) and every K in 2..=9 under
            // ATLAS_QWEN4_HYBRID_ALL_K share the K5/K9 exact-rows family.
            Some(Qwen4ExactQkvRowsRoute::LegacyK5K9)
        } else {
            qwen4_exact_qkv_rows_route(rows, legacy_k5_k9_enabled, qwen4_k16_exact_requested()?)
        };
        let Some(route) = route else {
            return Ok(None);
        };

        ensure!(self.gated, "Qwen4 exact QKV requires gated attention");
        ensure!(
            self.attn.q_norm_full.is_none()
                && self.attn.k_norm_full.is_none()
                && self.v_norm_weight.is_none(),
            "Qwen4 exact QKV requires per-head Q/K norms and no V norm"
        );
        let q_weight = self
            .q_weight
            .as_ref()
            .and_then(|weight| weight.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact Q requires ordinary NVFP4"))?;
        let k_weight = self
            .k_weight
            .as_ref()
            .and_then(|weight| weight.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact K requires ordinary NVFP4"))?;
        let v_weight = self
            .v_weight
            .as_ref()
            .and_then(|weight| weight.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact V requires ordinary NVFP4"))?;

        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        if matches!(
            route,
            Qwen4ExactQkvRowsRoute::K16 | Qwen4ExactQkvRowsRoute::K32
        ) {
            let selector = match route {
                Qwen4ExactQkvRowsRoute::K32 => {
                    crate::layers::qwen4_prefill_moe::attn16::CORE32_SELECTOR
                }
                _ => QWEN4_K16_EXACT_ENV,
            };
            ensure!(
                qwen4_k16_target_attention_geometry(ctx.config.is_qwen4_exp(), h, nq, nkv, hd),
                "{selector} requires exact Flash-Next attention geometry"
            );
        }
        let extents = qwen4_qkv_projection_extents(nq, nkv, hd, rows)
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact QKV extent overflow"))?;
        let qkv = ctx.buffers.qkv_output();
        ensure!(
            extents.total_bytes <= ctx.buffers.sizes().qkv_output,
            "Qwen4 exact QKV staging exceeds QKV arena"
        );
        // Preflight the widest slab first before issuing any GPU work.
        let first_slab = if route == Qwen4ExactQkvRowsRoute::K32 {
            16
        } else if rows == 33 {
            17
        } else {
            rows.min(32)
        };
        let check_rows = first_slab.max(4);
        ensure!(
            self.w4a16_exact_qkv_kernels.qg_for_rows(check_rows).0 != 0
                && self.w4a16_exact_qkv_kernels.dual_kv_for_rows(check_rows).0 != 0,
            "Qwen4 exact QKV kernels are unavailable"
        );

        let wide_pair = if route != Qwen4ExactQkvRowsRoute::K32 {
            WideExactQkvTier::for_rows(rows).and_then(|tier| wide_exact_qkv_kernels(ctx.gpu, tier))
        } else {
            None
        };
        if let Some((qg_kernel, dual_kv_kernel)) = wide_pair {
            launch_wide_qg_exact(
                ctx.gpu,
                qg_kernel,
                normed,
                q_weight,
                qkv,
                rows as u32,
                extents.q_proj_dim,
                h,
                nq,
                hd,
                extents.row_stride_bf16,
                stream,
            )?;
            let wide_k_base = qkv.offset(extents.q_proj_bytes);
            let wide_v_base = wide_k_base.offset(extents.kv_bytes);
            launch_wide_dual_kv_exact(
                ctx.gpu,
                dual_kv_kernel,
                normed,
                k_weight,
                wide_k_base,
                v_weight,
                wide_v_base,
                rows as u32,
                extents.kv_dim,
                h,
                extents.row_stride_bf16,
                stream,
            )?;
        } else {
            let mut row_start = 0usize;
            while row_start < rows {
                let rem = rows - row_start;
                let projection_rows = if route == Qwen4ExactQkvRowsRoute::K32 {
                    16.min(rem)
                } else if rem == 33 {
                    17
                } else {
                    rem.min(32)
                };
                // The M4 tier keeps MAX_M=4 accumulators and guards every row with
                // `row < M`, so M=2..=3 run it bit-exactly with idle accumulators.
                let kernel_rows = projection_rows.max(4);
                let qg_kernel = self.w4a16_exact_qkv_kernels.qg_for_rows(kernel_rows);
                let dual_kv_kernel = self.w4a16_exact_qkv_kernels.dual_kv_for_rows(kernel_rows);
                ensure!(
                    qg_kernel.0 != 0 && dual_kv_kernel.0 != 0,
                    "Qwen4 exact QKV kernels are unavailable for slab M={projection_rows}"
                );
                let chunk_input = normed.offset(row_start * h as usize * 2);
                let chunk_output = qkv.offset(row_start * extents.row_bytes);
                ops::w4a16_gemv_qg_exact(
                    ctx.gpu,
                    qg_kernel,
                    chunk_input,
                    q_weight,
                    chunk_output,
                    projection_rows as u32,
                    extents.q_proj_dim,
                    h,
                    nq,
                    hd,
                    extents.row_stride_bf16,
                    stream,
                )?;
                let k_base = chunk_output.offset(extents.q_proj_bytes);
                let v_base = k_base.offset(extents.kv_bytes);
                ops::w4a16_gemv_dual_kv_exact(
                    ctx.gpu,
                    dual_kv_kernel,
                    chunk_input,
                    k_weight,
                    k_base,
                    v_weight,
                    v_base,
                    projection_rows as u32,
                    extents.kv_dim,
                    h,
                    extents.row_stride_bf16,
                    stream,
                )?;
                row_start += projection_rows;
            }
        }

        let eps = ctx.config.rms_norm_eps as f32;
        if self.rms_norm_qk_batch3_k.0 != 0
            && !self.attn.q_norm.weight.is_null()
            && !self.attn.k_norm.weight.is_null()
        {
            ops::rms_norm_qk_batched(
                ctx.gpu,
                self.rms_norm_qk_batch3_k,
                qkv,
                &self.attn.q_norm,
                &self.attn.k_norm,
                extents.row_stride_bf16,
                extents.q_dim,
                extents.kv_dim,
                extents.q_proj_dim,
                hd,
                rows as u32,
                eps,
                stream,
            )?;
        } else {
            for row in 0..rows {
                let q_out = qkv.offset(row * extents.row_bytes);
                let k_out = q_out.offset(extents.q_proj_bytes);
                if !self.attn.q_norm.weight.is_null() {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        q_out,
                        &self.attn.q_norm,
                        q_out,
                        nq,
                        hd,
                        eps,
                        stream,
                    )?;
                }
                if !self.attn.k_norm.weight.is_null() {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        k_out,
                        &self.attn.k_norm,
                        k_out,
                        nkv,
                        hd,
                        eps,
                        stream,
                    )?;
                }
            }
        }
        if let Some(receipt) = qwen4_exact_qkv_receipt(route) {
            crate::model::k16_route_receipt::mark_qkv_layer()?;
            static K16_EXACT_ENGAGED: std::sync::Once = std::sync::Once::new();
            static K32_EXACT_ENGAGED: std::sync::Once = std::sync::Once::new();
            let once = match route {
                Qwen4ExactQkvRowsRoute::K32 => &K32_EXACT_ENGAGED,
                _ => &K16_EXACT_ENGAGED,
            };
            once.call_once(|| tracing::warn!("{receipt}"));
        }
        Ok(Some((qkv, extents.row_bytes)))
    }

    /// Projects `rows` (`2..=9`) MTP inputs into the strided `[Q|gate|K|V]`
    /// arena: projects K and V for all `rows` rows in one exact launch and
    /// projects Q+gate (and `q_norm`) only for the final row when
    /// `emit_last_row` is true.
    pub(in super::super) fn qwen4_k5_project_mtp_qkv_exact(
        &self,
        normed: DevicePtr,
        rows: usize,
        emit_last_row: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, usize)>> {
        if !(2..=crate::layers::QWEN4_HYBRID_MAX_ROWS).contains(&rows)
            || !crate::layers::qwen4_hybrid_rows(rows)
            || qwen4_k5_batch_attn_qkv_env() == Some("0")
            || !self.gated
            || self.attn.q_norm_full.is_some()
            || self.attn.k_norm_full.is_some()
            || self.v_norm_weight.is_some()
        {
            return Ok(None);
        }
        let (Some(q_weight), Some(k_weight), Some(v_weight)) = (
            self.q_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) else {
            return Ok(None);
        };
        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let Some(extents) = qwen4_qkv_projection_extents(nq, nkv, hd, rows) else {
            return Ok(None);
        };
        if extents.total_bytes > ctx.buffers.sizes().qkv_output {
            return Ok(None);
        }
        let kernel_rows = rows.max(4);
        let dual_kv_kernel = self.w4a16_exact_qkv_kernels.dual_kv_for_rows(kernel_rows);
        if dual_kv_kernel.0 == 0 || (emit_last_row && self.w4a16_gemv_qg_k.0 == 0) {
            return Ok(None);
        }
        let qkv = ctx.buffers.qkv_output();
        let k_base = qkv.offset(extents.q_proj_bytes);
        let v_base = k_base.offset(extents.kv_bytes);
        ops::w4a16_gemv_dual_kv_exact(
            ctx.gpu,
            dual_kv_kernel,
            normed,
            k_weight,
            k_base,
            v_weight,
            v_base,
            rows as u32,
            extents.kv_dim,
            h,
            extents.row_stride_bf16,
            stream,
        )?;
        if emit_last_row {
            let last = rows - 1;
            ops::w4a16_gemv_qg(
                ctx.gpu,
                self.w4a16_gemv_qg_k,
                normed.offset(last * h as usize * 2),
                q_weight,
                qkv.offset(last * extents.row_bytes),
                extents.q_proj_dim,
                h,
                nq,
                hd,
                stream,
            )?;
        }
        let eps = ctx.config.rms_norm_eps as f32;
        if self.rms_norm_qk_batch3_k.0 != 0
            && !self.attn.q_norm.weight.is_null()
            && !self.attn.k_norm.weight.is_null()
        {
            ops::rms_norm_qk_batched(
                ctx.gpu,
                self.rms_norm_qk_batch3_k,
                qkv,
                &self.attn.q_norm,
                &self.attn.k_norm,
                extents.row_stride_bf16,
                0,
                extents.kv_dim,
                extents.q_proj_dim,
                hd,
                rows as u32,
                eps,
                stream,
            )?;
            if emit_last_row {
                let q_last = qkv.offset((rows - 1) * extents.row_bytes);
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    q_last,
                    &self.attn.q_norm,
                    q_last,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
        } else {
            for row in 0..rows {
                let q_out = qkv.offset(row * extents.row_bytes);
                let k_out = q_out.offset(extents.q_proj_bytes);
                if emit_last_row && row == rows - 1 && !self.attn.q_norm.weight.is_null() {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        q_out,
                        &self.attn.q_norm,
                        q_out,
                        nq,
                        hd,
                        eps,
                        stream,
                    )?;
                }
                if !self.attn.k_norm.weight.is_null() {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        k_out,
                        &self.attn.k_norm,
                        k_out,
                        nkv,
                        hd,
                        eps,
                        stream,
                    )?;
                }
            }
        }
        Ok(Some((qkv, extents.row_bytes)))
    }

    /// Apply per-head Q/K RMSNorm in place on the strided `[Q|G|K|V]` arena.
    #[allow(clippy::too_many_arguments)]
    fn apply_qkv_qk_norm(
        &self,
        qkv: DevicePtr,
        rows: usize,
        nq: u32,
        nkv: u32,
        hd: u32,
        extents: Qwen4QkvProjectionExtents,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let eps = ctx.config.rms_norm_eps as f32;
        if self.rms_norm_qk_batch3_k.0 != 0
            && !self.attn.q_norm.weight.is_null()
            && !self.attn.k_norm.weight.is_null()
        {
            ops::rms_norm_qk_batched(
                ctx.gpu,
                self.rms_norm_qk_batch3_k,
                qkv,
                &self.attn.q_norm,
                &self.attn.k_norm,
                extents.row_stride_bf16,
                extents.q_dim,
                extents.kv_dim,
                extents.q_proj_dim,
                hd,
                rows as u32,
                eps,
                stream,
            )
        } else {
            for row in 0..rows {
                let q_out = qkv.offset(row * extents.row_bytes);
                let k_out = q_out.offset(extents.q_proj_bytes);
                if !self.attn.q_norm.weight.is_null() {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        q_out,
                        &self.attn.q_norm,
                        q_out,
                        nq,
                        hd,
                        eps,
                        stream,
                    )?;
                }
                if !self.attn.k_norm.weight.is_null() {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        k_out,
                        &self.attn.k_norm,
                        k_out,
                        nkv,
                        hd,
                        eps,
                        stream,
                    )?;
                }
            }
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_direct_nvfp4_gemm(
        &self,
        route: crate::layers::qwen4_fast_proj::DirectNvfp4GemmRoute,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::qwen4_fast_proj::DirectNvfp4GemmRoute;
        match route {
            DirectNvfp4GemmRoute::PipeM128N128 => ops::w4a16_gemm_pipe_m128n128(
                gpu,
                self.w4a16_gemm_pipe_m128n128_k,
                input,
                weight,
                output,
                m as u32,
                n as u32,
                k as u32,
                stream,
            ),
            DirectNvfp4GemmRoute::Pipe => ops::w4a16_gemm_pipe(
                gpu,
                self.w4a16_gemm_pipe_k,
                input,
                weight,
                output,
                m as u32,
                n as u32,
                k as u32,
                stream,
            ),
            DirectNvfp4GemmRoute::Base => ops::w4a16_gemm(
                gpu,
                self.w4a16_gemm_k,
                input,
                weight,
                output,
                m as u32,
                n as u32,
                k as u32,
                stream,
            ),
        }
    }

    /// Direct NVFP4 Tensor Core QG + dual-KV projection for `rows <= 256`
    /// (`QWEN4_NATIVE_MAX_ROWS`): projects QG into contiguous `ssm_deinterleaved`
    /// via single-sweep K-coalesced `w4a16_gemm_coalesced*` (default ON for
    /// `rows >= DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS` (`4`) via
    /// `ATLAS_QWEN4_BATCHED_VERIFY_COALESCED_NVFP4`, or opt-in
    /// `w4a16_gemm_pipe*` via `ATLAS_QWEN4_BATCHED_VERIFY_DIRECT_NVFP4=1`),
    /// scatters into strided `qkv_output`, deinterleaves Q/G in place, and runs
    /// exact dual-KV (or coalesced KV) directly into `qkv_output`.
    #[allow(clippy::too_many_arguments)]
    fn try_qwen4_qkv_direct_nvfp4(
        &self,
        normed: DevicePtr,
        q_weight: &QuantizedWeight,
        k_weight: &QuantizedWeight,
        v_weight: &QuantizedWeight,
        rows: usize,
        h: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        extents: Qwen4QkvProjectionExtents,
        fp8_qkv_ready: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let coalesced_enabled = crate::layers::qwen4_fast_proj::coalesced_nvfp4_dense_enabled();
        let direct_enabled = crate::layers::qwen4_fast_proj::direct_nvfp4_dense_enabled();
        let Some(q_mode) = select_qkv_direct_q_mode(
            coalesced_enabled,
            direct_enabled,
            rows,
            h as usize,
            fp8_qkv_ready,
        ) else {
            return Ok(false);
        };
        if !qkv_direct_nvfp4_eligible(
            rows,
            h as usize,
            extents.q_proj_dim as usize,
            extents.kv_dim as usize,
            hd as usize,
            extents.q_proj_bytes,
            ctx.buffers.sizes().ssm_deinterleaved,
        ) || self.deinterleave_qg_k.0 == 0
            || [q_weight, k_weight, v_weight].iter().any(|w| {
                w.weight.is_null() || w.weight_scale.is_null() || !w.weight_scale_2.is_finite()
            })
        {
            return Ok(false);
        }
        let qkv = ctx.buffers.qkv_output();
        let q_scratch = ctx.buffers.ssm_deinterleaved();
        if q_scratch.is_null() || q_scratch == qkv || q_scratch == normed {
            return Ok(false);
        }
        let strided_copy_k = strided_copy_rows_kernel(ctx.gpu);
        let wide_dual_kv = WideExactQkvTier::for_rows(rows)
            .and_then(|tier| wide_exact_qkv_kernels(ctx.gpu, tier))
            .map(|(_, dual_kv)| dual_kv);
        let kv_span = (rows * extents.kv_bytes).next_multiple_of(16);
        let coalesced_kv_eligible = wide_dual_kv.is_none()
            && qkv_direct_prefers_coalesced_kv(
                coalesced_enabled,
                rows,
                h as usize,
                extents.kv_dim as usize,
                extents.kv_bytes,
                extents.row_bytes,
                extents.q_proj_bytes,
                ctx.buffers.sizes().ssm_deinterleaved,
                strided_copy_k.0 != 0,
            )
            && normed.0.is_multiple_of(16)
            && q_scratch.0.is_multiple_of(16)
            && qkv.0.is_multiple_of(16)
            && k_weight.weight.0.is_multiple_of(16)
            && v_weight.weight.0.is_multiple_of(16)
            && k_weight.weight_scale.0.is_multiple_of(2)
            && v_weight.weight_scale.0.is_multiple_of(2);
        let dual_kv_slabs_loaded = if wide_dual_kv.is_some() {
            true
        } else {
            let mut ok = true;
            let mut row_start = 0usize;
            while row_start < rows {
                let slab = direct_nvfp4_dual_kv_slab(rows - row_start);
                if self.w4a16_exact_qkv_kernels.dual_kv_for_rows(slab.max(4)).0 == 0 {
                    ok = false;
                    break;
                }
                row_start += slab;
            }
            ok
        };
        // Preflight K/V availability BEFORE launching QG or mutating `qkv_output`:
        // when `dual_kv_slabs_loaded` is false, verify a coalesced KV kernel is
        // actually present on `ctx.gpu` so Step 4 cannot fail after Step 2/3 wrote `qkv`.
        let coalesced_kv_ready = coalesced_kv_eligible && {
            let sym = crate::layers::qwen4_fast_proj::coalesced_nvfp4_dispatch_symbol(
                rows,
                extents.kv_dim as usize,
                h as usize,
            );
            crate::layers::try_kernel(ctx.gpu, "w4a16", sym).0 != 0
                || crate::layers::try_kernel(ctx.gpu, "w4a16_gemm", sym).0 != 0
        };
        if wide_dual_kv.is_none() && !coalesced_kv_ready && !dual_kv_slabs_loaded {
            return Ok(false);
        }

        // 1. Direct NVFP4 Tensor Core QG into contiguous `q_scratch[rows, q_proj_dim]`.
        let mut ran_coalesced = false;
        let q_route_label = if q_mode == QkvDirectQMode::Coalesced
            && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                ctx.gpu,
                normed,
                q_weight,
                q_scratch,
                rows,
                extents.q_proj_dim as usize,
                h as usize,
                stream,
            )? {
            ran_coalesced = true;
            crate::layers::qwen4_fast_proj::coalesced_nvfp4_dispatch_symbol(
                rows,
                extents.q_proj_dim as usize,
                h as usize,
            )
        } else if direct_enabled {
            let Some(q_route) = crate::layers::qwen4_fast_proj::select_direct_nvfp4_gemm_route(
                rows,
                extents.q_proj_dim as usize,
                h as usize,
                normed.0.is_multiple_of(16),
                q_weight.weight.0.is_multiple_of(16),
                q_scratch.0.is_multiple_of(16),
                self.w4a16_gemm_pipe_m128n128_k.0 != 0,
                self.w4a16_gemm_pipe_k.0 != 0,
                self.w4a16_gemm_k.0 != 0,
            ) else {
                return Ok(false);
            };
            self.launch_direct_nvfp4_gemm(
                q_route,
                normed,
                q_weight,
                q_scratch,
                rows,
                extents.q_proj_dim as usize,
                h as usize,
                ctx.gpu,
                stream,
            )?;
            q_route.label()
        } else {
            return Ok(false);
        };

        // 2+3. Fused scatter + gated Q/Gate deinterleave (`q_scratch[rows, q_proj_dim]`
        //      -> `qkv[rows, row_stride_bf16]` with `[Q_all | G_all]`): replaces
        //      `strided_copy_rows_16` + in-place `deinterleave_qg` with 1 launch
        //      (`scatter_qkv_multi_seq` with `kv_dim = 0, gated = 1`), saving 12
        //      launches and 12 full `[rows, 12288]` BF16 round-trips per round
        //      with 0-ULP bit-exact parity.
        let scatter_qg_k = scatter_qkv_multi_seq_kernel(ctx.gpu);
        if scatter_qg_k.0 != 0 && extents.q_proj_dim == 2 * nq * hd {
            KernelLaunch::new(ctx.gpu, scatter_qg_k)
                .grid([div_ceil(extents.q_proj_dim, 256), rows as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(q_scratch)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(qkv)
                .arg_u32(rows as u32)
                .arg_u32(extents.q_proj_dim)
                .arg_u32(0)
                .arg_u32(extents.row_stride_bf16)
                .arg_u32(nq)
                .arg_u32(hd)
                .arg_u32(1)
                .launch(stream)?;
        } else {
            if strided_copy_k.0 != 0
                && q_scratch.0.is_multiple_of(16)
                && qkv.0.is_multiple_of(16)
                && extents.q_proj_bytes.is_multiple_of(16)
                && extents.row_bytes.is_multiple_of(16)
            {
                ops::strided_copy_rows(
                    ctx.gpu,
                    strided_copy_k,
                    q_scratch,
                    qkv,
                    rows as u32,
                    extents.q_proj_bytes as u32,
                    extents.q_proj_bytes as u64,
                    extents.row_bytes as u64,
                    stream,
                )?;
            } else {
                for row in 0..rows {
                    ctx.gpu.copy_d2d_async(
                        q_scratch.offset(row * extents.q_proj_bytes),
                        qkv.offset(row * extents.row_bytes),
                        extents.q_proj_bytes,
                        stream,
                    )?;
                }
            }
            ops::deinterleave_qg(
                ctx.gpu,
                self.deinterleave_qg_k,
                qkv,
                rows as u32,
                nq,
                hd,
                extents.row_stride_bf16,
                stream,
            )?;
        }

        // 4. Dual-KV / Coalesced-KV direct NVFP4 into strided `qkv` K/V slices.
        //    For `rows = 129..=256` (`wide_dual_kv == None`), run single-launch
        //    coalesced NVFP4 W4A16 GEMMs for K and V into the now-dead `q_scratch`
        //    arena + `strided_copy_rows`, avoiding 6–8 serial 32-row scalar launches.
        let mut ran_coalesced_kv = false;
        if let Some(dual_kv_kernel) = wide_dual_kv {
            let k_base = qkv.offset(extents.q_proj_bytes);
            let v_base = k_base.offset(extents.kv_bytes);
            launch_wide_dual_kv_exact(
                ctx.gpu,
                dual_kv_kernel,
                normed,
                k_weight,
                k_base,
                v_weight,
                v_base,
                rows as u32,
                extents.kv_dim,
                h,
                extents.row_stride_bf16,
                stream,
            )?;
        } else {
            if coalesced_kv_eligible {
                let k_scratch = q_scratch;
                let v_scratch = q_scratch.offset(kv_span);
                let k_ok = crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    normed,
                    k_weight,
                    k_scratch,
                    rows,
                    extents.kv_dim as usize,
                    h as usize,
                    stream,
                )?;
                let v_ok = k_ok
                    && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                        ctx.gpu,
                        normed,
                        v_weight,
                        v_scratch,
                        rows,
                        extents.kv_dim as usize,
                        h as usize,
                        stream,
                    )?;
                if v_ok {
                    let k_base = qkv.offset(extents.q_proj_bytes);
                    let v_base = k_base.offset(extents.kv_bytes);
                    ops::strided_copy_rows(
                        ctx.gpu,
                        strided_copy_k,
                        k_scratch,
                        k_base,
                        rows as u32,
                        extents.kv_bytes as u32,
                        extents.kv_bytes as u64,
                        extents.row_bytes as u64,
                        stream,
                    )?;
                    ops::strided_copy_rows(
                        ctx.gpu,
                        strided_copy_k,
                        v_scratch,
                        v_base,
                        rows as u32,
                        extents.kv_bytes as u32,
                        extents.kv_bytes as u64,
                        extents.row_bytes as u64,
                        stream,
                    )?;
                    ran_coalesced_kv = true;
                }
            }
            if !ran_coalesced_kv {
                anyhow::ensure!(
                    dual_kv_slabs_loaded,
                    "direct NVFP4 QKV rows={rows} coalesced K/V unavailable and dual_kv slabs not loaded"
                );
                let mut row_start = 0usize;
                while row_start < rows {
                    let slab = direct_nvfp4_dual_kv_slab(rows - row_start);
                    let dual_kv_kernel = self.w4a16_exact_qkv_kernels.dual_kv_for_rows(slab.max(4));
                    let chunk_in = normed.offset(row_start * h as usize * 2);
                    let chunk_qkv = qkv.offset(row_start * extents.row_bytes);
                    let k_base = chunk_qkv.offset(extents.q_proj_bytes);
                    let v_base = k_base.offset(extents.kv_bytes);
                    ops::w4a16_gemv_dual_kv_exact(
                        ctx.gpu,
                        dual_kv_kernel,
                        chunk_in,
                        k_weight,
                        k_base,
                        v_weight,
                        v_base,
                        slab as u32,
                        extents.kv_dim,
                        h,
                        extents.row_stride_bf16,
                        stream,
                    )?;
                    row_start += slab;
                }
            }
        }

        // 5. Batched per-head Q/K RMSNorm.
        self.apply_qkv_qk_norm(qkv, rows, nq, nkv, hd, extents, ctx, stream)?;

        let kv_route = if wide_dual_kv.is_some() {
            "w4a16_gemv_dual_kv_exact_wide"
        } else if ran_coalesced_kv {
            "w4a16_gemm_coalesced_kv"
        } else {
            "w4a16_gemv_dual_kv_exact"
        };
        if ran_coalesced {
            static QKV_COALESCED_LOGGED: std::sync::Once = std::sync::Once::new();
            QKV_COALESCED_LOGGED.call_once(|| {
                tracing::info!(
                    selector = crate::layers::qwen4_fast_proj::COALESCED_NVFP4_DENSE_ENV,
                    component = "attn_qkv",
                    q_route = q_route_label,
                    kv_route,
                    rows,
                    q_proj_dim = extents.q_proj_dim,
                    kv_dim = extents.kv_dim,
                    row_stride = extents.row_stride_bf16,
                    "QWEN4_BATCHED_VERIFY_COALESCED_NVFP4_ENGAGED"
                );
            });
        } else {
            static QKV_DIRECT_LOGGED: std::sync::Once = std::sync::Once::new();
            QKV_DIRECT_LOGGED.call_once(|| {
                tracing::info!(
                    selector = crate::layers::qwen4_fast_proj::DIRECT_NVFP4_DENSE_ENV,
                    component = "attn_qkv",
                    q_route = q_route_label,
                    kv_route,
                    rows,
                    q_proj_dim = extents.q_proj_dim,
                    kv_dim = extents.kv_dim,
                    row_stride = extents.row_stride_bf16,
                    "QWEN4_BATCHED_VERIFY_DIRECT_NVFP4_ENGAGED"
                );
            });
        }
        Ok(true)
    }

    fn oproj_prefers_fp8_cached(
        &self,
        gpu: &dyn GpuBackend,
        weight: &QuantizedWeight,
        rows: usize,
        h: usize,
        attn_width: usize,
        graph_capture: bool,
    ) -> bool {
        if !crate::layers::qwen4_fast_proj::coalesced_nvfp4_prefers_fp8_cached(rows, h) {
            return false;
        }
        let nvfp4_mode = crate::layers::qwen4_fast_proj::nvfp4::mode_or_off();
        let verify_min_m = crate::layers::qwen4_fast_proj::nvfp4::verify_min_m();
        if nvfp4_mode.admits_weight(h, attn_width, None) {
            if !nvfp4_mode.admits_call(rows, h, attn_width, h, None, true, verify_min_m) {
                return false;
            }
            return !graph_capture
                || (crate::layers::qwen4_fast_proj::nvfp4::quant_kernels_ready(gpu)
                    && crate::layers::qwen4_fast_proj::nvfp4::has_cached_weight_for_gpu(
                        gpu, weight, h, attn_width, None,
                    )
                    && crate::layers::qwen4_fast_proj::nvfp4::scratch_ready_for_capture_for_gpu(
                        gpu, rows, attn_width,
                    ));
        }
        !graph_capture
            || (crate::layers::qwen4_fast_proj::fp8_tensor_scales()
                && crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                    weight, h, attn_width, None,
                )
                && crate::layers::qwen4_fast_proj::fp8_scratch_ready_for_capture(
                    h,
                    rows.max(crate::layers::qwen4_fast_proj::resolve_fp8_pad_m(rows, h)),
                    attn_width,
                ))
    }

    /// Direct NVFP4 Tensor Core O projection
    /// (`output[rows, h] = input[rows, attn_width] @ W_o^T`) for batched verify
    /// and cross-seq MTP (`rows <= 256`, `QWEN4_NATIVE_MAX_ROWS`).
    ///
    /// Dispatches single-sweep K-coalesced `w4a16_gemm_coalesced*` (`n32` ->
    /// `80` CTAs at `h = 2560`) by default for `rows in 4..=256`
    /// (`DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS..=QWEN4_NATIVE_MAX_ROWS`,
    /// `ATLAS_QWEN4_BATCHED_VERIFY_COALESCED_NVFP4 != 0`). When disabled (`=0`),
    /// falls back to opt-in `ATLAS_QWEN4_BATCHED_VERIFY_DIRECT_NVFP4=1`
    /// (`w4a16_gemm_pipe*`), or returns `Ok(false)` so the caller falls through
    /// to FP8 cuBLASLt / exact GEMV.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn try_qwen4_oproj_direct_nvfp4(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        h: usize,
        attn_width: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if rows == 0
            || rows > crate::layers::QWEN4_NATIVE_MAX_ROWS
            || self.o_dense_bf16.is_some()
            || self.o_weight.as_ref().and_then(|w| w.as_fp8()).is_some()
        {
            return Ok(false);
        }
        let weight = &self.attn.o_proj;
        if weight.weight.is_null()
            || weight.weight_scale.is_null()
            || !weight.weight_scale_2.is_finite()
        {
            return Ok(false);
        }
        if crate::layers::qwen4_fast_proj::coalesced_nvfp4_dense_enabled()
            && rows >= crate::layers::qwen4_hyper::DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS
            && !self.oproj_prefers_fp8_cached(
                ctx.gpu,
                weight,
                rows,
                h,
                attn_width,
                ctx.graph_capture,
            )
            && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                ctx.gpu, input, weight, output, rows, h, attn_width, stream,
            )?
        {
            static OPROJ_COALESCED_LOGGED: std::sync::Once = std::sync::Once::new();
            OPROJ_COALESCED_LOGGED.call_once(|| {
                let route = crate::layers::qwen4_fast_proj::coalesced_nvfp4_dispatch_symbol(
                    rows, h, attn_width,
                );
                tracing::info!(
                    selector = crate::layers::qwen4_fast_proj::COALESCED_NVFP4_DENSE_ENV,
                    component = "attn_o_proj",
                    route,
                    rows,
                    n = h,
                    k = attn_width,
                    "QWEN4_BATCHED_VERIFY_COALESCED_NVFP4_ENGAGED"
                );
            });
            return Ok(true);
        }
        if !crate::layers::qwen4_fast_proj::direct_nvfp4_dense_enabled() {
            return Ok(false);
        }
        let Some(route) = crate::layers::qwen4_fast_proj::select_direct_nvfp4_gemm_route(
            rows,
            h,
            attn_width,
            input.0.is_multiple_of(16),
            weight.weight.0.is_multiple_of(16),
            output.0.is_multiple_of(16),
            self.w4a16_gemm_pipe_m128n128_k.0 != 0,
            self.w4a16_gemm_pipe_k.0 != 0,
            self.w4a16_gemm_k.0 != 0,
        ) else {
            return Ok(false);
        };
        self.launch_direct_nvfp4_gemm(
            route, input, weight, output, rows, h, attn_width, ctx.gpu, stream,
        )?;
        static OPROJ_DIRECT_LOGGED: std::sync::Once = std::sync::Once::new();
        OPROJ_DIRECT_LOGGED.call_once(|| {
            tracing::info!(
                selector = crate::layers::qwen4_fast_proj::DIRECT_NVFP4_DENSE_ENV,
                component = "attn_o_proj",
                route = route.label(),
                rows,
                n = h,
                k = attn_width,
                "QWEN4_BATCHED_VERIFY_DIRECT_NVFP4_ENGAGED"
            );
        });
        Ok(true)
    }

    /// Tensor-core batched-verify QG/K/V projection into the exact same
    /// strided `[Q (nq*hd) | G (nq*hd) | K (nkv*hd) | V (nkv*hd)]` arena as
    /// [`Self::qwen4_k5_project_qkv_exact`].
    ///
    /// Defaults to single-sweep K-coalesced NVFP4 `w4a16_gemm_coalesced*` +
    /// `deinterleave_qg` + exact `dual_kv` for `rows in 4..=48`
    /// (`DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS..=QKV_COALESCED_MAX_ROWS`, and
    /// up to `QWEN4_NATIVE_MAX_ROWS` (`256`) on cold capture when FP8 is not
    /// ready; `ATLAS_QWEN4_BATCHED_VERIFY_COALESCED_NVFP4 != 0`, zero FP8 weight
    /// cache RAM, CUDA-graph capture safe). Above `48` rows when FP8 is ready,
    /// hands off to `qwen4_fast_proj`'s shared-activation cached FP8 cuBLASLt
    /// path. When disabled (`=0`), falls back to opt-in
    /// `ATLAS_QWEN4_BATCHED_VERIFY_DIRECT_NVFP4=1` or FP8 cuBLASLt.
    ///
    /// Returns `Ok(None)` without writing `qkv_output` when TC dense is off,
    /// `rows < min_rows` (`DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS = 4`),
    /// geometry does not match, or kernels are unavailable so the caller falls
    /// back cleanly to [`Self::qwen4_k5_project_qkv_exact`].
    pub(in super::super) fn qwen4_k5_project_qkv_tc_dense(
        &self,
        normed: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, usize)>> {
        if !crate::layers::qwen4_hyper::batched_verify_tc_component_active(
            crate::layers::qwen4_hyper::BatchedVerifyTcComponent::Fp8Dense,
            rows,
            ctx.graph_capture,
        ) || !self.gated
            || self.attn.q_norm_full.is_some()
            || self.attn.k_norm_full.is_some()
            || self.v_norm_weight.is_some()
        {
            return Ok(None);
        }
        let (Some(q_weight), Some(k_weight), Some(v_weight)) = (
            self.q_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) else {
            return Ok(None);
        };
        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let Some(extents) = qwen4_qkv_projection_extents(nq, nkv, hd, rows) else {
            return Ok(None);
        };
        if extents.total_bytes > ctx.buffers.sizes().qkv_output {
            return Ok(None);
        }

        let q_proj_dim = extents.q_proj_dim as usize;
        let kv_dim = extents.kv_dim as usize;
        let h_usize = h as usize;
        let nq_usize = nq as usize;
        let hd_usize = hd as usize;
        let row_stride_usize = extents.row_stride_bf16 as usize;
        let nvfp4_mode = crate::layers::qwen4_fast_proj::nvfp4::mode_or_off();
        let nvfp4_verify_min_m = crate::layers::qwen4_fast_proj::nvfp4::verify_min_m();
        let nvfp4_kernels_ready = nvfp4_mode.is_enabled()
            && crate::layers::qwen4_fast_proj::nvfp4::quant_kernels_ready(ctx.gpu);
        let all_nvfp4_handles_qkv = nvfp4_kernels_ready
            && nvfp4_mode.admits_call(
                rows,
                q_proj_dim,
                h_usize,
                row_stride_usize,
                Some((nq_usize, hd_usize)),
                true,
                nvfp4_verify_min_m,
            )
            && nvfp4_mode.admits_call(
                rows,
                kv_dim,
                h_usize,
                row_stride_usize,
                None,
                true,
                nvfp4_verify_min_m,
            );
        let fp8_qkv_available = qkv_tc_dense_fp8_available(
            ctx.gpu,
            [q_weight, k_weight, v_weight],
            rows,
            h_usize,
            q_proj_dim,
            kv_dim,
            hd_usize,
            all_nvfp4_handles_qkv,
        );
        let fp8_qkv_capture_safe = qkv_shared_act_capture_safe(QkvCaptureReadiness {
            nvfp4_mode,
            verify_min_m: nvfp4_verify_min_m,
            rows,
            q_proj_dim,
            kv_dim,
            h: h_usize,
            row_stride: row_stride_usize,
            nq: nq_usize,
            hd: hd_usize,
            fp8_tensor_scales: crate::layers::qwen4_fast_proj::fp8_tensor_scales(),
            fp8_scratch_ready: crate::layers::qwen4_fast_proj::fp8_scratch_ready_for_capture(
                q_proj_dim,
                rows.max(crate::layers::qwen4_fast_proj::resolve_fp8_pad_m(
                    rows, q_proj_dim,
                )),
                h_usize,
            ),
            nvfp4_scratch_ready:
                crate::layers::qwen4_fast_proj::nvfp4::scratch_ready_for_capture_for_gpu(
                    ctx.gpu, rows, h_usize,
                ),
            nvfp4_kernels_ready,
            q_fp8_cached: crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                q_weight,
                q_proj_dim,
                h_usize,
                Some((nq_usize, hd_usize)),
            ),
            k_fp8_cached: crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                k_weight, kv_dim, h_usize, None,
            ),
            v_fp8_cached: crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                v_weight, kv_dim, h_usize, None,
            ),
            q_nvfp4_cached: crate::layers::qwen4_fast_proj::nvfp4::has_cached_weight_for_gpu(
                ctx.gpu,
                q_weight,
                q_proj_dim,
                h_usize,
                Some((nq_usize, hd_usize)),
            ),
            k_nvfp4_cached: crate::layers::qwen4_fast_proj::nvfp4::has_cached_weight_for_gpu(
                ctx.gpu, k_weight, kv_dim, h_usize, None,
            ),
            v_nvfp4_cached: crate::layers::qwen4_fast_proj::nvfp4::has_cached_weight_for_gpu(
                ctx.gpu, v_weight, kv_dim, h_usize, None,
            ),
        });
        let fp8_qkv_ready = fp8_qkv_available && (!ctx.graph_capture || fp8_qkv_capture_safe);

        let qkv = ctx.buffers.qkv_output();
        if self.try_qwen4_qkv_direct_nvfp4(
            normed,
            q_weight,
            k_weight,
            v_weight,
            rows,
            h,
            nq,
            nkv,
            hd,
            extents,
            fp8_qkv_ready,
            ctx,
            stream,
        )? {
            return Ok(Some((qkv, extents.row_bytes)));
        }

        if !fp8_qkv_ready {
            return Ok(None);
        }

        let row_stride = extents.row_stride_bf16 as usize;
        crate::layers::qwen4_fast_proj::clear_prequant_fp8_act();
        crate::layers::qwen4_fp8_act_amax::clear();
        let tensor_scales = crate::layers::qwen4_fast_proj::fp8_tensor_scales();
        let shared_act = tensor_scales && crate::layers::qwen4_fast_proj::fp8_shared_act_selected();
        if shared_act {
            use crate::layers::qwen4_fast_proj::SharedActOut;
            crate::layers::qwen4_fast_proj::fp8_gemm_shared_act(
                ctx.gpu,
                normed,
                &[
                    SharedActOut {
                        weight: k_weight,
                        out: qkv.offset(extents.q_proj_bytes),
                        n: extents.kv_dim as usize,
                        qg: None,
                    },
                    SharedActOut {
                        weight: v_weight,
                        out: qkv.offset(extents.q_proj_bytes + extents.kv_bytes),
                        n: extents.kv_dim as usize,
                        qg: None,
                    },
                    SharedActOut {
                        weight: q_weight,
                        out: qkv,
                        n: extents.q_proj_dim as usize,
                        qg: Some((nq as usize, hd as usize)),
                    },
                ],
                rows,
                h as usize,
                row_stride,
                stream,
            )?;
        } else {
            crate::layers::qwen4_fast_proj::fp8_gemm_ex(
                ctx.gpu,
                normed,
                q_weight,
                qkv,
                rows,
                extents.q_proj_dim as usize,
                h as usize,
                row_stride,
                Some((nq as usize, hd as usize)),
                stream,
            )?;
            crate::layers::qwen4_fast_proj::fp8_gemm_ex(
                ctx.gpu,
                normed,
                k_weight,
                qkv.offset(extents.q_proj_bytes),
                rows,
                extents.kv_dim as usize,
                h as usize,
                row_stride,
                None,
                stream,
            )?;
            crate::layers::qwen4_fast_proj::fp8_gemm_ex(
                ctx.gpu,
                normed,
                v_weight,
                qkv.offset(extents.q_proj_bytes + extents.kv_bytes),
                rows,
                extents.kv_dim as usize,
                h as usize,
                row_stride,
                None,
                stream,
            )?;
        }

        self.apply_qkv_qk_norm(qkv, rows, nq, nkv, hd, extents, ctx, stream)?;
        static QKV_TC_LOGGED: std::sync::Once = std::sync::Once::new();
        QKV_TC_LOGGED.call_once(|| {
            tracing::info!(
                selector = crate::layers::qwen4_hyper::BATCHED_VERIFY_TC_DENSE_ENV,
                component = "attn_qkv",
                route = if shared_act {
                    "qwen4_fast_proj_fp8_shared_act"
                } else {
                    "qwen4_fast_proj_fp8"
                },
                tensor_scales,
                cached_q = crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                    q_weight,
                    extents.q_proj_dim as usize,
                    h as usize,
                    Some((nq as usize, hd as usize)),
                ),
                rows,
                q_proj_dim = extents.q_proj_dim,
                kv_dim = extents.kv_dim,
                row_stride,
                "QWEN4_BATCHED_VERIFY_TC_DENSE_ENGAGED"
            );
        });
        Ok(Some((qkv, extents.row_bytes)))
    }
}

/// Default SM dynamic shared memory cap for `deinterleave_qg` without opt-in carveout.
const DEINTERLEAVE_QG_MAX_SMEM_BYTES: usize = 49_152;

/// Maximum row count where batched-verify attention QKV prefers direct
/// coalesced NVFP4 (`w4a16_gemm_coalesced*` + `deinterleave_qg` + `dual_kv`).
/// Rows `<= 3` stay off coalesced NVFP4, coalesced NVFP4 covers `4..=48`
/// (`DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS..=QKV_COALESCED_MAX_ROWS`), and
/// rows `49..=256` (`..=QWEN4_NATIVE_MAX_ROWS`) yield to shared-activation FP8
/// cuBLASLt when the FP8 QKV cache is ready and fall back to coalesced NVFP4
/// when `!fp8_qkv_ready`.
const QKV_COALESCED_MAX_ROWS: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QkvDirectQMode {
    Coalesced,
    LegacyDirect,
}

#[inline]
#[must_use]
const fn select_qkv_direct_q_mode(
    coalesced_enabled: bool,
    direct_enabled: bool,
    rows: usize,
    h: usize,
    fp8_qkv_ready: bool,
) -> Option<QkvDirectQMode> {
    if coalesced_enabled
        && rows >= crate::layers::qwen4_hyper::DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS
        && rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS
        && h > 0
        && h.is_multiple_of(32)
        && (rows <= QKV_COALESCED_MAX_ROWS || !fp8_qkv_ready)
    {
        Some(QkvDirectQMode::Coalesced)
    } else if direct_enabled
        && rows > 0
        && rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS
        && h > 0
        && h.is_multiple_of(16)
    {
        Some(QkvDirectQMode::LegacyDirect)
    } else {
        None
    }
}

/// Whether `try_qwen4_qkv_direct_nvfp4` should compute `k_proj` and `v_proj`
/// via single-launch coalesced NVFP4 W4A16 GEMMs (`try_coalesced_nvfp4_w4a16_gemm`)
/// into the dead `q_scratch` arena + `strided_copy_rows`, eliminating 6–8 serial
/// 32-row scalar `dual_kv` launches for `rows = 129..=256`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn qkv_direct_prefers_coalesced_kv(
    coalesced_enabled: bool,
    rows: usize,
    h: usize,
    kv_dim: usize,
    kv_bytes: usize,
    row_bytes: usize,
    q_proj_bytes: usize,
    scratch_bytes: usize,
    has_strided_copy: bool,
) -> bool {
    if !coalesced_enabled
        || !has_strided_copy
        || rows <= 128
        || rows > crate::layers::QWEN4_NATIVE_MAX_ROWS
        || h == 0
        || !h.is_multiple_of(32)
        || kv_dim == 0
        || !kv_bytes.is_multiple_of(16)
        || !q_proj_bytes.is_multiple_of(16)
        || !row_bytes.is_multiple_of(16)
    {
        return false;
    }
    let Some(raw_span) = rows.checked_mul(kv_bytes) else {
        return false;
    };
    let Some(aligned_span) = raw_span.checked_next_multiple_of(16) else {
        return false;
    };
    match aligned_span.checked_mul(2) {
        Some(need) => need <= scratch_bytes,
        None => false,
    }
}

/// Per-projection cache/dispatch status for deciding whether batched-verify QKV
/// (`q_proj`, `k_proj`, `v_proj`) can safely execute on the shared-activation
/// FP8 / native cuBLASLt NVFP4 path inside a CUDA graph capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QkvCaptureReadiness {
    nvfp4_mode: crate::layers::qwen4_fast_proj::nvfp4::Nvfp4FastProjMode,
    verify_min_m: usize,
    rows: usize,
    q_proj_dim: usize,
    kv_dim: usize,
    h: usize,
    row_stride: usize,
    nq: usize,
    hd: usize,
    fp8_tensor_scales: bool,
    fp8_scratch_ready: bool,
    nvfp4_scratch_ready: bool,
    nvfp4_kernels_ready: bool,
    q_fp8_cached: bool,
    k_fp8_cached: bool,
    v_fp8_cached: bool,
    q_nvfp4_cached: bool,
    k_nvfp4_cached: bool,
    v_nvfp4_cached: bool,
}

#[must_use]
const fn qkv_shared_act_capture_safe(r: QkvCaptureReadiness) -> bool {
    let q_qg = Some((r.nq, r.hd));
    let q_nvfp4_weight = r.nvfp4_mode.admits_weight(r.q_proj_dim, r.h, q_qg);
    let k_nvfp4_weight = r.nvfp4_mode.admits_weight(r.kv_dim, r.h, None);
    let v_nvfp4_weight = r.nvfp4_mode.admits_weight(r.kv_dim, r.h, None);

    let q_uses_nvfp4 = r.nvfp4_mode.admits_call(
        r.rows,
        r.q_proj_dim,
        r.h,
        r.row_stride,
        q_qg,
        true,
        r.verify_min_m,
    );
    let k_uses_nvfp4 = r.nvfp4_mode.admits_call(
        r.rows,
        r.kv_dim,
        r.h,
        r.row_stride,
        None,
        true,
        r.verify_min_m,
    );
    let v_uses_nvfp4 = r.nvfp4_mode.admits_call(
        r.rows,
        r.kv_dim,
        r.h,
        r.row_stride,
        None,
        true,
        r.verify_min_m,
    );

    // If NVFP4 claims a weight (preventing FP8 caching) but refuses this row
    // count (`rows < verify_min_m`), shared-act would fall back to uncached
    // FP8 dequantization — refuse capture so Coalesced W4A16 handles it.
    if (q_nvfp4_weight && !q_uses_nvfp4)
        || (k_nvfp4_weight && !k_uses_nvfp4)
        || (v_nvfp4_weight && !v_uses_nvfp4)
    {
        return false;
    }

    let any_nvfp4 = q_uses_nvfp4 || k_uses_nvfp4 || v_uses_nvfp4;
    let any_fp8 = !q_uses_nvfp4 || !k_uses_nvfp4 || !v_uses_nvfp4;

    if any_nvfp4 && (!r.nvfp4_kernels_ready || !r.nvfp4_scratch_ready) {
        return false;
    }
    if any_fp8 && (!r.fp8_tensor_scales || !r.fp8_scratch_ready) {
        return false;
    }

    let q_ok = if q_uses_nvfp4 {
        r.q_nvfp4_cached
    } else {
        r.q_fp8_cached
    };
    let k_ok = if k_uses_nvfp4 {
        r.k_nvfp4_cached
    } else {
        r.k_fp8_cached
    };
    let v_ok = if v_uses_nvfp4 {
        r.v_nvfp4_cached
    } else {
        r.v_fp8_cached
    };

    q_ok && k_ok && v_ok
}

#[inline]
#[must_use]
const fn direct_nvfp4_dual_kv_slab(rem: usize) -> usize {
    if rem == 33 {
        17
    } else if rem > 32 {
        32
    } else {
        rem
    }
}

#[inline]
#[must_use]
const fn qkv_direct_nvfp4_eligible(
    rows: usize,
    h: usize,
    q_proj_dim: usize,
    kv_dim: usize,
    hd: usize,
    q_proj_bytes: usize,
    ssm_deinterleaved_bytes: usize,
) -> bool {
    rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS
        && qkv_tc_dense_fp8_shape_valid(rows, h, q_proj_dim, kv_dim, hd)
        && q_proj_bytes <= DEINTERLEAVE_QG_MAX_SMEM_BYTES
        && match rows.checked_mul(q_proj_bytes) {
            Some(need) => need <= ssm_deinterleaved_bytes,
            None => false,
        }
}

struct StridedCopyKernelCache {
    primary: std::sync::OnceLock<(usize, (KernelHandle, KernelHandle))>,
    fallback: std::sync::OnceLock<
        parking_lot::Mutex<std::collections::HashMap<usize, (KernelHandle, KernelHandle)>>,
    >,
}

impl StridedCopyKernelCache {
    const fn new() -> Self {
        Self {
            primary: std::sync::OnceLock::new(),
            fallback: std::sync::OnceLock::new(),
        }
    }

    fn probe(gpu: &dyn GpuBackend) -> (KernelHandle, KernelHandle) {
        (
            crate::layers::try_kernel(gpu, "strided_copy_rows", "strided_copy_rows_16"),
            crate::layers::try_kernel(gpu, "strided_copy_rows", "scatter_qkv_multi_seq"),
        )
    }

    fn get_or_load(&self, gpu: &dyn GpuBackend) -> (KernelHandle, KernelHandle) {
        let key = std::ptr::from_ref(gpu) as *const () as usize;
        if let Some(&(cached_key, cached_val)) = self.primary.get() {
            if cached_key == key {
                return cached_val;
            }
        } else {
            let val = Self::probe(gpu);
            let &(stored_key, stored_val) = self.primary.get_or_init(|| (key, val));
            if stored_key == key {
                return stored_val;
            }
            let map = self
                .fallback
                .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
            return *map.lock().entry(key).or_insert(val);
        }
        let map = self
            .fallback
            .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
        if let Some(val) = map.lock().get(&key).copied() {
            return val;
        }
        let val = Self::probe(gpu);
        *map.lock().entry(key).or_insert(val)
    }
}

fn strided_copy_rows_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    static CACHE: StridedCopyKernelCache = StridedCopyKernelCache::new();
    CACHE.get_or_load(gpu).0
}

fn scatter_qkv_multi_seq_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    static CACHE: StridedCopyKernelCache = StridedCopyKernelCache::new();
    CACHE.get_or_load(gpu).1
}

#[inline]
#[must_use]
const fn qkv_tc_dense_fp8_shape_valid(
    rows: usize,
    h: usize,
    q_proj_dim: usize,
    kv_dim: usize,
    hd: usize,
) -> bool {
    rows > 0
        && h > 0
        && q_proj_dim > 0
        && kv_dim > 0
        && hd > 0
        && h.is_multiple_of(16)
        && q_proj_dim.is_multiple_of(16)
        && kv_dim.is_multiple_of(16)
        && hd.is_multiple_of(8)
}

#[allow(clippy::too_many_arguments)]
fn qkv_tc_dense_fp8_available(
    gpu: &dyn GpuBackend,
    weights: [&QuantizedWeight; 3],
    rows: usize,
    h: usize,
    q_proj_dim: usize,
    kv_dim: usize,
    hd: usize,
    all_nvfp4_handles_qkv: bool,
) -> bool {
    if !qkv_tc_dense_fp8_shape_valid(rows, h, q_proj_dim, kv_dim, hd)
        || weights.iter().any(|w| {
            w.weight.is_null() || w.weight_scale.is_null() || !w.weight_scale_2.is_finite()
        })
    {
        return false;
    }
    if all_nvfp4_handles_qkv {
        return true;
    }
    if crate::layers::qwen4_fast_proj::fp8_tensor_scales() {
        crate::layers::qwen4_fast_proj::fp8_tensor_kernels(gpu).is_ok()
    } else {
        crate::layers::qwen4_hyper::fp8_row_scale_kernels_available(gpu)
    }
}

#[cfg(test)]
#[path = "qwen4_k16_projection_tests.rs"]
mod qwen4_k16_projection_tests;

#[cfg(test)]
mod wide_exact_qkv_tests {
    use super::WideExactQkvTier;

    #[test]
    fn wide_exact_qkv_tiers_cover_33_to_128_in_single_sweep() {
        for rows in [0, 1, 2, 16, 25, 32, 129, usize::MAX] {
            assert_eq!(WideExactQkvTier::for_rows(rows), None, "rows={rows}");
        }
        for (rows, tier, qg, kv) in [
            (
                33,
                WideExactQkvTier::M64,
                "w4a16_gemv_qg_exact_m64",
                "w4a16_gemv_dual_kv_exact_m64",
            ),
            (
                40,
                WideExactQkvTier::M64,
                "w4a16_gemv_qg_exact_m64",
                "w4a16_gemv_dual_kv_exact_m64",
            ),
            (
                64,
                WideExactQkvTier::M64,
                "w4a16_gemv_qg_exact_m64",
                "w4a16_gemv_dual_kv_exact_m64",
            ),
            (
                65,
                WideExactQkvTier::M80,
                "w4a16_gemv_qg_exact_m80",
                "w4a16_gemv_dual_kv_exact_m80",
            ),
            (
                80,
                WideExactQkvTier::M80,
                "w4a16_gemv_qg_exact_m80",
                "w4a16_gemv_dual_kv_exact_m80",
            ),
            (
                81,
                WideExactQkvTier::M128,
                "w4a16_gemv_qg_exact_m128",
                "w4a16_gemv_dual_kv_exact_m128",
            ),
            (
                128,
                WideExactQkvTier::M128,
                "w4a16_gemv_qg_exact_m128",
                "w4a16_gemv_dual_kv_exact_m128",
            ),
        ] {
            assert_eq!(WideExactQkvTier::for_rows(rows), Some(tier), "rows={rows}");
            assert_eq!(tier.qg_symbol(), qg, "rows={rows}");
            assert_eq!(tier.dual_kv_symbol(), kv, "rows={rows}");
        }
    }

    #[test]
    fn tc_dense_qkv_shape_and_extents_match_split_qg_kv_layout() {
        use super::{qkv_tc_dense_fp8_shape_valid, qwen4_qkv_projection_extents};
        // Flash-Next TP1 (nq=24, nkv=2, hd=256) and TP2 (nq=12, nkv=1, hd=256).
        // Legacy direct NVFP4 stays opt-in while default batched-verify QKV uses
        // coalesced NVFP4 (4..=48) and shared-act FP8 cuBLASLt (> 48 when ready).
        assert!(!crate::layers::qwen4_fast_proj::parse_direct_nvfp4_dense(
            None
        ));
        for (nq, nkv) in [(24u32, 2u32), (12u32, 1u32)] {
            for rows in [6usize, 7, 8, 10, 12, 16, 20, 40, 80, 128] {
                let ext = qwen4_qkv_projection_extents(nq, nkv, 256, rows).unwrap();
                assert_eq!(ext.q_proj_dim, ext.q_dim * 2);
                assert_eq!(ext.row_stride_bf16, ext.q_proj_dim + 2 * ext.kv_dim);
                assert_eq!(ext.q_proj_bytes, ext.q_proj_dim as usize * 2);
                assert_eq!(ext.kv_bytes, ext.kv_dim as usize * 2);
                assert_eq!(ext.row_bytes, ext.row_stride_bf16 as usize * 2);
                assert!(qkv_tc_dense_fp8_shape_valid(
                    rows,
                    2_560,
                    ext.q_proj_dim as usize,
                    ext.kv_dim as usize,
                    256,
                ));
            }
        }
        assert!(!qkv_tc_dense_fp8_shape_valid(0, 2_560, 12_288, 512, 256));
        assert!(!qkv_tc_dense_fp8_shape_valid(20, 2_560, 12_288, 512, 252));
        assert!(!qkv_tc_dense_fp8_shape_valid(20, 2_552, 12_288, 512, 256));
    }

    #[test]
    fn direct_nvfp4_qkv_eligibility_and_dual_kv_slabs_cover_6_to_128() {
        use super::{
            DEINTERLEAVE_QG_MAX_SMEM_BYTES, direct_nvfp4_dual_kv_slab, qkv_direct_nvfp4_eligible,
            qwen4_qkv_projection_extents,
        };
        let arena = 4 * 1024 * 1024; // 4 MiB ssm_deinterleaved floor at 128 tokens
        for (nq, nkv) in [(24u32, 2u32), (12u32, 1u32)] {
            for rows in [6usize, 7, 8, 10, 12, 16, 20, 40, 64, 80, 128] {
                let ext = qwen4_qkv_projection_extents(nq, nkv, 256, rows).unwrap();
                assert!(qkv_direct_nvfp4_eligible(
                    rows,
                    2_560,
                    ext.q_proj_dim as usize,
                    ext.kv_dim as usize,
                    256,
                    ext.q_proj_bytes,
                    arena,
                ));
            }
        }
        // Reject > 256 rows, oversized shared memory (> 48 KiB), or insufficient scratch.
        assert!(!qkv_direct_nvfp4_eligible(
            257, 2_560, 12_288, 512, 256, 24_576, arena
        ));
        assert!(!qkv_direct_nvfp4_eligible(
            20,
            2_560,
            24_576,
            512,
            256,
            DEINTERLEAVE_QG_MAX_SMEM_BYTES + 16,
            arena,
        ));
        assert!(!qkv_direct_nvfp4_eligible(
            80,
            2_560,
            12_288,
            512,
            256,
            24_576,
            80 * 24_576 - 1
        ));

        // Dual-KV fallback slabs never leave a 1-row tail for any rows in 2..=128.
        for rows in 2usize..=128 {
            let mut rem = rows;
            while rem > 0 {
                let slab = direct_nvfp4_dual_kv_slab(rem);
                assert!(
                    (2..=32).contains(&slab),
                    "rows={rows} rem={rem} slab={slab}"
                );
                rem -= slab;
            }
        }
    }

    #[test]
    fn coalesced_nvfp4_qkv_mode_selects_coalesced_for_4_to_48_and_fp8_above_48_when_ready() {
        use super::{QkvDirectQMode, select_qkv_direct_q_mode};

        assert!(crate::layers::qwen4_fast_proj::parse_coalesced_nvfp4_dense(
            None
        ));
        // Rows <= 3 stay off coalesced; rows >= 4 engage Coalesced NVFP4 inside batched verify
        // (eliminating the C=2 r=4..6 scalar GEMV cliff while C=1 stays unscoped on strict 0-ULP).
        for rows in 0..4 {
            assert_eq!(
                select_qkv_direct_q_mode(true, false, rows, 2_560, true),
                None
            );
            assert_eq!(
                select_qkv_direct_q_mode(true, false, rows, 2_560, false),
                None
            );
        }
        // Rows 4..=48 with K % 32 == 0 select Coalesced NVFP4 even when FP8 QKV cache is ready.
        for rows in [4usize, 5, 6, 7, 8, 9, 10, 12, 16, 20, 25, 32, 40, 48] {
            assert_eq!(
                select_qkv_direct_q_mode(true, false, rows, 2_560, true),
                Some(QkvDirectQMode::Coalesced),
                "rows={rows}"
            );
            assert_eq!(
                select_qkv_direct_q_mode(true, false, rows, 2_560, false),
                Some(QkvDirectQMode::Coalesced),
                "rows={rows}"
            );
            assert_eq!(
                select_qkv_direct_q_mode(true, true, rows, 2_560, true),
                Some(QkvDirectQMode::Coalesced),
                "rows={rows}"
            );
            // Disabling coalesced falls back to LegacyDirect only when opted in, else None (FP8).
            assert_eq!(
                select_qkv_direct_q_mode(false, true, rows, 2_560, true),
                Some(QkvDirectQMode::LegacyDirect),
                "rows={rows}"
            );
            assert_eq!(
                select_qkv_direct_q_mode(false, false, rows, 2_560, true),
                None
            );
        }
        // Rows 49..=128 yield to shared-act FP8 cuBLASLt when ready, and fall back to
        // Coalesced NVFP4 when FP8 QKV cache is not ready.
        for rows in [49usize, 64, 80, 128] {
            assert_eq!(
                select_qkv_direct_q_mode(true, false, rows, 2_560, true),
                None,
                "rows={rows}"
            );
            assert_eq!(
                select_qkv_direct_q_mode(true, false, rows, 2_560, false),
                Some(QkvDirectQMode::Coalesced),
                "rows={rows}"
            );
            assert_eq!(
                select_qkv_direct_q_mode(true, true, rows, 2_560, true),
                Some(QkvDirectQMode::LegacyDirect),
                "rows={rows}"
            );
            assert_eq!(
                select_qkv_direct_q_mode(true, true, rows, 2_560, false),
                Some(QkvDirectQMode::Coalesced),
                "rows={rows}"
            );
            assert_eq!(
                select_qkv_direct_q_mode(false, true, rows, 2_560, true),
                Some(QkvDirectQMode::LegacyDirect),
                "rows={rows}"
            );
            assert_eq!(
                select_qkv_direct_q_mode(false, false, rows, 2_560, true),
                None
            );
        }
        // K % 32 != 0 skips Coalesced (falling back to LegacyDirect if opted in).
        assert_eq!(select_qkv_direct_q_mode(true, false, 20, 48, false), None);
        assert_eq!(
            select_qkv_direct_q_mode(true, true, 20, 48, false),
            Some(QkvDirectQMode::LegacyDirect)
        );
        // > 256 rows or zero K fail closed.
        assert_eq!(
            select_qkv_direct_q_mode(true, true, 257, 2_560, false),
            None
        );
        assert_eq!(select_qkv_direct_q_mode(true, true, 20, 0, false), None);
    }

    #[test]
    fn fp8_or_nvfp4_qkv_capture_safe_accepts_cached_nvfp4_q_and_covers_192_256() {
        use super::{
            QkvCaptureReadiness, QkvDirectQMode, qkv_direct_prefers_coalesced_kv,
            qkv_shared_act_capture_safe, qwen4_qkv_projection_extents, select_qkv_direct_q_mode,
        };
        use crate::layers::qwen4_fast_proj::nvfp4::Nvfp4FastProjMode;

        let ext = qwen4_qkv_projection_extents(24, 2, 256, 192).unwrap();
        let base_wide = QkvCaptureReadiness {
            nvfp4_mode: Nvfp4FastProjMode::Wide,
            verify_min_m: 64,
            rows: 192,
            q_proj_dim: ext.q_proj_dim as usize,
            kv_dim: ext.kv_dim as usize,
            h: 2_560,
            row_stride: ext.row_stride_bf16 as usize,
            nq: 24,
            hd: 256,
            fp8_tensor_scales: true,
            fp8_scratch_ready: true,
            nvfp4_scratch_ready: true,
            nvfp4_kernels_ready: true,
            // Under FAST_PROJ_NVFP4=wide, warmup caches Q in NVFP4_WEIGHT_CACHE
            // under qg=Some((24, 256)) and never populates FP8 WEIGHT_CACHE for Q!
            q_fp8_cached: false,
            k_fp8_cached: true,
            v_fp8_cached: true,
            q_nvfp4_cached: true,
            k_nvfp4_cached: false,
            v_nvfp4_cached: false,
        };

        // Wide mode at rows = 64, 128, 192, 256: capture-safe with NVFP4 Q + FP8 K/V!
        for rows in [64usize, 128, 192, 256] {
            let r = QkvCaptureReadiness { rows, ..base_wide };
            assert!(
                qkv_shared_act_capture_safe(r),
                "wide mode rows={rows} must accept NVFP4-cached Q + FP8-cached K/V"
            );
            // When shared-act is capture-safe, rows > 48 stay on shared-act (not Coalesced).
            assert_eq!(
                select_qkv_direct_q_mode(true, false, rows, 2_560, true),
                None,
                "rows={rows} must stay on shared-act when capture-safe"
            );
        }

        // Wide mode missing NVFP4 Q cache or scratch/kernels fails closed.
        assert!(!qkv_shared_act_capture_safe(QkvCaptureReadiness {
            q_nvfp4_cached: false,
            ..base_wide
        }));
        assert!(!qkv_shared_act_capture_safe(QkvCaptureReadiness {
            nvfp4_scratch_ready: false,
            ..base_wide
        }));
        assert!(!qkv_shared_act_capture_safe(QkvCaptureReadiness {
            nvfp4_kernels_ready: false,
            ..base_wide
        }));
        assert!(!qkv_shared_act_capture_safe(QkvCaptureReadiness {
            k_fp8_cached: false,
            ..base_wide
        }));

        // Wide mode at rows = 56 (< verify_min_m = 64): NVFP4 refuses rows < 64
        // while FP8 didn't cache Q -> must fail closed so Coalesced W4A16 runs!
        assert!(!qkv_shared_act_capture_safe(QkvCaptureReadiness {
            rows: 56,
            ..base_wide
        }));

        // All mode at rows = 192, 256: capture-safe with all 3 weights in NVFP4_WEIGHT_CACHE,
        // even when FP8 weight cache and FP8 scratch are completely unpopulated!
        let base_all = QkvCaptureReadiness {
            nvfp4_mode: Nvfp4FastProjMode::All,
            fp8_tensor_scales: false,
            fp8_scratch_ready: false,
            q_fp8_cached: false,
            k_fp8_cached: false,
            v_fp8_cached: false,
            q_nvfp4_cached: true,
            k_nvfp4_cached: true,
            v_nvfp4_cached: true,
            ..base_wide
        };
        for rows in [64usize, 192, 256] {
            assert!(
                qkv_shared_act_capture_safe(QkvCaptureReadiness { rows, ..base_all }),
                "all mode rows={rows} must be capture-safe with NVFP4-cached Q/K/V alone"
            );
        }
        assert!(!qkv_shared_act_capture_safe(QkvCaptureReadiness {
            v_nvfp4_cached: false,
            ..base_all
        }));

        // Off mode (default / GB10): requires FP8 weight cache for all 3 weights.
        let base_off = QkvCaptureReadiness {
            nvfp4_mode: Nvfp4FastProjMode::Off,
            q_fp8_cached: true,
            k_fp8_cached: true,
            v_fp8_cached: true,
            q_nvfp4_cached: false,
            nvfp4_scratch_ready: false,
            nvfp4_kernels_ready: false,
            ..base_wide
        };
        assert!(qkv_shared_act_capture_safe(base_off));
        assert!(!qkv_shared_act_capture_safe(QkvCaptureReadiness {
            q_fp8_cached: false,
            ..base_off
        }));

        // For rows = 129..=256 (e.g. 192, 256) on the direct/coalesced fallback path,
        // k_proj/v_proj select single-launch coalesced NVFP4 instead of 6-8 serial
        // 32-row scalar dual_kv launches.
        let arena_256 = 256 * 16_384 * 2; // 8 MiB ssm_deinterleaved at 256 rows
        for rows in [129usize, 160, 192, 256] {
            let e = qwen4_qkv_projection_extents(24, 2, 256, rows).unwrap();
            assert_eq!(
                select_qkv_direct_q_mode(true, false, rows, 2_560, false),
                Some(QkvDirectQMode::Coalesced),
                "cold capture fallback rows={rows}"
            );
            assert!(
                qkv_direct_prefers_coalesced_kv(
                    true,
                    rows,
                    2_560,
                    e.kv_dim as usize,
                    e.kv_bytes,
                    e.row_bytes,
                    e.q_proj_bytes,
                    arena_256,
                    true,
                ),
                "rows={rows} must use coalesced NVFP4 K/V instead of 6-8 serial dual_kv launches"
            );
        }
        // Rows <= 128 keep WideExactQkvTier / exact dual_kv; > 256 or disabled fails closed.
        for rows in [32usize, 64, 128, 257] {
            let e = qwen4_qkv_projection_extents(24, 2, 256, rows.min(256)).unwrap();
            assert!(
                !qkv_direct_prefers_coalesced_kv(
                    true,
                    rows,
                    2_560,
                    e.kv_dim as usize,
                    e.kv_bytes,
                    e.row_bytes,
                    e.q_proj_bytes,
                    arena_256,
                    true,
                ),
                "rows={rows} must not engage >128 coalesced K/V path"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_parity_fused_scatter_deinterleave_qg_bit_exact() {
        use super::{
            ops, qwen4_qkv_projection_extents, scatter_qkv_multi_seq_kernel,
            strided_copy_rows_kernel,
        };
        use spark_runtime::cuda_backend::AtlasCudaBackend;
        use spark_runtime::gpu::{DevicePtr, GpuBackend};
        use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

        let Some(ptx) = atlas_kernels::ptx_for_model("flash-next") else {
            return;
        };
        let Ok(gpu) = AtlasCudaBackend::new(0, &ptx.modules) else {
            return;
        };
        let strided_k = strided_copy_rows_kernel(&gpu);
        let scatter_k = scatter_qkv_multi_seq_kernel(&gpu);
        let deinter_k = crate::layers::try_kernel(&gpu, "ssm_preprocess", "deinterleave_qg");
        if strided_k.0 == 0 || scatter_k.0 == 0 || deinter_k.0 == 0 {
            return;
        }

        let stream = gpu.default_stream();
        let rows = 8usize;
        let nq = 24u32;
        let nkv = 2u32;
        let hd = 256u32;
        let ext = qwen4_qkv_projection_extents(nq, nkv, hd, rows).unwrap();

        let src_host: Vec<u16> = (0..(rows * ext.q_proj_dim as usize))
            .map(|i| ((i.wrapping_mul(257) ^ (i >> 3)) & 0xFFFF) as u16)
            .collect();
        // SAFETY: `src_host` is an initialized `Vec<u16>` of POD elements with no padding bytes;
        // `u8` has alignment 1 and `src_host` outlives the copy below.
        let src_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                src_host.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&src_host[..]),
            )
        };
        let sentinel: u16 = 0x5A5A;
        let init_qkv = vec![sentinel; rows * ext.row_stride_bf16 as usize];
        // SAFETY: `init_qkv` is an initialized `Vec<u16>` of POD elements with no padding bytes;
        // `u8` has alignment 1 and `init_qkv` outlives the copies below.
        let init_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                init_qkv.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&init_qkv[..]),
            )
        };

        let src_dev = gpu.alloc(src_bytes.len()).unwrap();
        let ref_dev = gpu.alloc(init_bytes.len()).unwrap();
        let fused_dev = gpu.alloc(init_bytes.len()).unwrap();
        gpu.copy_h2d_async(src_bytes, src_dev, stream).unwrap();
        gpu.copy_h2d_async(init_bytes, ref_dev, stream).unwrap();
        gpu.copy_h2d_async(init_bytes, fused_dev, stream).unwrap();

        // Reference: strided_copy_rows_16 + in-place deinterleave_qg.
        ops::strided_copy_rows(
            &gpu,
            strided_k,
            src_dev,
            ref_dev,
            rows as u32,
            ext.q_proj_bytes as u32,
            ext.q_proj_bytes as u64,
            ext.row_bytes as u64,
            stream,
        )
        .unwrap();
        ops::deinterleave_qg(
            &gpu,
            deinter_k,
            ref_dev,
            rows as u32,
            nq,
            hd,
            ext.row_stride_bf16,
            stream,
        )
        .unwrap();

        // Fused: single launch of scatter_qkv_multi_seq (kv_dim = 0, gated = 1).
        KernelLaunch::new(&gpu, scatter_k)
            .grid([div_ceil(ext.q_proj_dim, 256), rows as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(src_dev)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(fused_dev)
            .arg_u32(rows as u32)
            .arg_u32(ext.q_proj_dim)
            .arg_u32(0)
            .arg_u32(ext.row_stride_bf16)
            .arg_u32(nq)
            .arg_u32(hd)
            .arg_u32(1)
            .launch(stream)
            .unwrap();

        gpu.synchronize(stream).unwrap();

        let mut ref_out = vec![0u16; init_qkv.len()];
        let mut fused_out = vec![0u16; init_qkv.len()];
        // SAFETY: `ref_out` and `fused_out` are exclusively borrowed initialized `Vec<u16>`
        // buffers of POD elements; any byte pattern written by `copy_d2h` is a valid `u16`.
        unsafe {
            gpu.copy_d2h(
                ref_dev,
                std::slice::from_raw_parts_mut(
                    ref_out.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(&ref_out[..]),
                ),
            )
            .unwrap();
            gpu.copy_d2h(
                fused_dev,
                std::slice::from_raw_parts_mut(
                    fused_out.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(&fused_out[..]),
                ),
            )
            .unwrap();
        }
        assert_eq!(
            fused_out, ref_out,
            "fused scatter_qkv_multi_seq must be 100% bit-identical to strided_copy_rows + deinterleave_qg"
        );
        // Verify K/V tail slices stayed untouched (sentinel intact).
        for r in 0..rows {
            let row_base = r * ext.row_stride_bf16 as usize;
            for col in (ext.q_proj_dim as usize)..(ext.row_stride_bf16 as usize) {
                assert_eq!(fused_out[row_base + col], sentinel);
            }
        }
        for p in [src_dev, ref_dev, fused_dev] {
            let _ = gpu.free(p);
        }
    }
}
