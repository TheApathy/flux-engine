// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in fusion of the FlashInfer route's gated RMS norm with the FP8
//! tensor-scale quantization of the GDN out-projection activation
//! (`ATLAS_QWEN4_PREFILL_GDN_NORM_FP8`, default off).
//!
//! The FP8 tensor-scale out projection today writes BF16 normed rows, re-reads
//! them for the activation absmax, and re-reads them again to write e4m3
//! (`qwen4_fast_proj::fp8_gemm_ex`). Here the norm kernels compute the same
//! BF16-rounded values and write e4m3 directly (`qwen4_gdn_norm_fp8.cu`):
//! - `=1` exact: one absmax pass + one quantizing pass; e4m3 rows and scale are
//!   bit-identical to the unfused FP8 tensor-scale path (0 of 50.3M bytes differ
//!   on FN L0/L22/L46 norm weights, 8192 rows).
//! - `=2` static: the activation scale is the analytic bound
//!   sqrt(head_dim) * max|norm weight| (|x * rsqrt(mean x^2)| <= sqrt(d),
//!   sigmoid <= 1), so the absmax pass is skipped. NOT bit-identical: the scale
//!   is 1.07-1.16x the measured absmax on FN layers; dequantized rel-L2 vs the
//!   BF16 rows is unchanged (2.65e-2 either way).
//! - `=3` static, in the FI kernel: the FP16 FI export's store warps apply the
//!   norm to each O tile in shared memory and write the e4m3 rows directly
//!   (`fi_gdn_bridge_f16_nf8.cu`, bench-prefill/fn_gdn_fi/nf8_variant), so the
//!   16-bit GDN output never reaches DRAM. Same numerics class as `=2`: H state
//!   bit-identical, 3-5 of 50.3M e4m3 bytes one step off (reduction order), at
//!   1.65-1.71 vs 2.60-2.67 ms/layer for FI16 + `=2` (nf8b1, real L22/L46).
//!   Needs `ATLAS_QWEN4_PREFILL_GDN_FI=2`.
//!
//! Requires the FI route, `ATLAS_QWEN4_PREFILL_FAST_PROJ` with ssm,
//! `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8` with ssm and
//! `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SCALE=tensor`; fails closed otherwise.

use spark_runtime::kernel_args::KernelLaunch;

use super::qwen4_gdn_fi::FiDtype;
use super::qwen4_prefill_exact_plan::Plan;
use super::*;
use crate::layers::qwen4_fast_proj;

pub(super) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_NORM_FP8";
pub(super) const V2_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_NORM_FP8_V2";
pub(super) const L2_DISCARD_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_L2_DISCARD";

/// L2 cache line granularity required by `discard.global.L2 [addr], 128` on SM121a.
pub(super) const L2_DISCARD_LINE_BYTES: usize = 128;
/// `discard_flags` bitmask matching `QWEN4_NF8_DISC_*` in `qwen4_gdn_norm_fp8.cu`.
pub(super) const DISC_FLAG_X: u32 = 1 << 0;
pub(super) const DISC_FLAG_G: u32 = 1 << 1;
pub(super) const DISC_FLAG_QKV: u32 = 1 << 2;

/// Rows processed per CTA in `qwen4_gdn_norm_fp8_{bf16,f16}_v2_strip`
/// (`ROWS_PER_CTA = 8u` in `qwen4_gdn_norm_fp8.cu`): 512-row strips launch
/// `[3, 64, 1] = 192 CTAs` (exact 1 wave @ 4 CTAs/SM on 48 SMs) and 640-row
/// tail-absorbed strips launch `[3, 80, 1] = 240 CTAs` (exact 1 wave @ 5 CTAs/SM).
pub(super) const QUANT_V2_STRIP_ROWS_PER_CTA: u32 = 8;

/// Launch grid `[3, ceil(strip_rows / 8), 1]` for `qwen4_gdn_norm_fp8_*_v2_strip`.
#[inline]
#[must_use]
pub(super) const fn quant_v2_strip_grid(strip_rows: u32) -> [u32; 3] {
    [3, strip_rows.div_ceil(QUANT_V2_STRIP_ROWS_PER_CTA), 1]
}

/// Policy controlling `discard.global.L2` on consumed GDN scratch and gate cache lines.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum L2DiscardPolicy {
    Disabled,
    Enabled,
}

impl L2DiscardPolicy {
    pub(super) const fn as_flag_str(self) -> &'static str {
        match self {
            Self::Disabled => "0",
            Self::Enabled => "1",
        }
    }
}

fn parse_l2_discard(value: Option<&str>) -> Result<L2DiscardPolicy> {
    match value.map(str::trim) {
        None | Some("") => Ok(L2DiscardPolicy::Enabled),
        Some(v) => match v.to_ascii_lowercase().as_str() {
            "1" | "on" | "true" => Ok(L2DiscardPolicy::Enabled),
            "0" | "off" | "false" => Ok(L2DiscardPolicy::Disabled),
            other => anyhow::bail!(
                "{L2_DISCARD_SELECTOR}: unknown value {other:?} (expected 0, 1, off, on, false, true)"
            ),
        },
    }
}

/// Default-on (`ATLAS_QWEN4_PREFILL_GDN_L2_DISCARD=0` disables L2 discard on
/// consumed `gdn_out_strip`, `qkv_strip`, and `gate` cache lines).
pub(super) fn l2_discard_policy() -> Result<L2DiscardPolicy> {
    static SEL: std::sync::OnceLock<Result<L2DiscardPolicy, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| match std::env::var(L2_DISCARD_SELECTOR) {
        Err(std::env::VarError::NotPresent) => Ok(L2DiscardPolicy::Enabled),
        Ok(v) => parse_l2_discard(Some(&v)).map_err(|e| e.to_string()),
        Err(e) => Err(format!("invalid {L2_DISCARD_SELECTOR}: {e}")),
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

/// Device buffer pointers and strides for one `quant_v2_strip` launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StripL2DiscardBuffers {
    pub qkv_strip: DevicePtr,
    pub gdn_out_strip: DevicePtr,
    pub z_row: DevicePtr,
    pub conv_dim: usize,
    pub value_dim: usize,
    pub qkvz_stride: usize,
}

/// Resolved per-strip L2 discard arguments for `qwen4_gdn_norm_fp8_*_v2_strip`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StripL2DiscardArgs {
    pub qkv_discard: DevicePtr,
    pub discard_flags: u32,
}

/// Compute 128-byte-aligned L2 discard flags for the non-strip `quant_v2` launch.
#[must_use]
pub(super) fn quant_v2_discard_flags(
    policy: L2DiscardPolicy,
    gdn_out: DevicePtr,
    value_dim: usize,
) -> u32 {
    if !matches!(policy, L2DiscardPolicy::Enabled) {
        return 0;
    }
    let align = L2_DISCARD_LINE_BYTES as u64;
    if !gdn_out.is_null()
        && gdn_out.0.is_multiple_of(align)
        && (value_dim * 2).is_multiple_of(L2_DISCARD_LINE_BYTES)
    {
        DISC_FLAG_X
    } else {
        0
    }
}

/// Compute 128-byte-aligned L2 discard arguments for one `quant_v2_strip` launch.
///
/// Applies to every strip including the final strip because `qkv_strip`,
/// `gdn_out_strip`, and the Z columns of `deinterleaved` are dead scratch data
/// once `quant_v2_strip` consumes them.
#[must_use]
pub(super) fn resolve_strip_l2_discard(
    policy: L2DiscardPolicy,
    bufs: StripL2DiscardBuffers,
) -> StripL2DiscardArgs {
    if !matches!(policy, L2DiscardPolicy::Enabled) {
        return StripL2DiscardArgs {
            qkv_discard: DevicePtr::NULL,
            discard_flags: 0,
        };
    }
    let align = L2_DISCARD_LINE_BYTES as u64;
    let mut discard_flags = 0u32;
    if !bufs.gdn_out_strip.is_null()
        && bufs.gdn_out_strip.0.is_multiple_of(align)
        && (bufs.value_dim * 2).is_multiple_of(L2_DISCARD_LINE_BYTES)
    {
        discard_flags |= DISC_FLAG_X;
    }
    if !bufs.z_row.is_null()
        && bufs.z_row.0.is_multiple_of(align)
        && (bufs.qkvz_stride * 2).is_multiple_of(L2_DISCARD_LINE_BYTES)
    {
        discard_flags |= DISC_FLAG_G;
    }
    let qkv_discard = if !bufs.qkv_strip.is_null()
        && bufs.qkv_strip.0.is_multiple_of(align)
        && bufs.conv_dim == Plan::CONV_DIM
        && (bufs.conv_dim * 2).is_multiple_of(L2_DISCARD_LINE_BYTES)
    {
        discard_flags |= DISC_FLAG_QKV;
        bufs.qkv_strip
    } else {
        DevicePtr::NULL
    };
    StripL2DiscardArgs {
        qkv_discard,
        discard_flags,
    }
}

/// Default-off (`ATLAS_QWEN4_PREFILL_GDN_NORM_FP8_V2=1` enables the 16-head/CTA vectorized v2 kernel).
pub(super) fn v2_selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| match std::env::var(V2_SELECTOR) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" => Ok(false),
            "1" | "on" | "true" => Ok(true),
            other => Err(format!(
                "{V2_SELECTOR}: unknown value {other:?} (expected 0, 1, off, on, false, true)"
            )),
        },
        Err(e) => Err(format!("invalid {V2_SELECTOR}: {e}")),
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Mode {
    Exact,
    Static,
    FiStatic,
}

impl Mode {
    const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact_amax_bit_identical",
            Self::Static => "static_bound_non_bit_exact",
            Self::FiStatic => "static_bound_in_fi_store_warps_non_bit_exact",
        }
    }
}

fn parse(value: Option<&str>) -> Result<Option<Mode>> {
    match value.map(str::trim) {
        None | Some("") | Some("0") => Ok(None),
        Some("1") => Ok(Some(Mode::Exact)),
        Some("2") => Ok(Some(Mode::Static)),
        Some("3") => Ok(Some(Mode::FiStatic)),
        Some(other) => anyhow::bail!("{SELECTOR}: unknown value {other:?} (expected 0, 1, 2, 3)"),
    }
}

/// Parsed once per process; a malformed or unsupported selection fails the
/// first prefill.
pub(super) fn mode() -> Result<Option<Mode>> {
    static SEL: std::sync::OnceLock<Result<Option<Mode>, String>> = std::sync::OnceLock::new();
    let mode = SEL
        .get_or_init(|| parse(std::env::var(SELECTOR).ok().as_deref()).map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| anyhow::anyhow!(e))?;
    if mode.is_some() {
        let _ = v2_selected()?;
        let _ = l2_discard_policy()?;
        #[cfg(not(atlas_fi_gdn))]
        {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "{SELECTOR} requested, but this build was compiled without FlashInfer GDN AOT objects (cfg(atlas_fi_gdn) disabled); falling back to native CUDA GDN norm path"
                );
            });
            return Ok(None);
        }
        #[cfg(atlas_fi_gdn)]
        {
            ensure!(
                super::qwen4_gdn_fi::selected(),
                "{SELECTOR} requires {}",
                super::qwen4_gdn_fi::SELECTOR
            );
            ensure!(
                qwen4_fast_proj::selection()?.ssm
                    && qwen4_fast_proj::fp8_selection()?.ssm
                    && qwen4_fast_proj::fp8_tensor_scales(),
                "{SELECTOR} requires {}=ssm, {}=ssm and ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SCALE=tensor",
                qwen4_fast_proj::SELECTOR,
                qwen4_fast_proj::FP8_SELECTOR
            );
            if mode == Some(Mode::FiStatic) {
                ensure!(
                    super::qwen4_gdn_fi::dtype() == Some(FiDtype::F16),
                    "{SELECTOR}=3 requires {}=2 (the FP16 export)",
                    super::qwen4_gdn_fi::SELECTOR
                );
            }
        }
    }
    Ok(mode)
}

#[derive(Clone, Copy)]
struct Kernels {
    amax: KernelHandle,
    quant: KernelHandle,
    amax_v2: KernelHandle,
    quant_v2: KernelHandle,
    quant_v2_strip: KernelHandle,
    norm_amax: KernelHandle,
    norm_amax_v2: KernelHandle,
}

fn kernels(gpu: &dyn GpuBackend, dtype: FiDtype) -> Result<Kernels> {
    type Slot = std::sync::OnceLock<Result<Kernels, String>>;
    static BF16: Slot = Slot::new();
    static F16: Slot = Slot::new();
    let (slot, suffix) = match dtype {
        FiDtype::Bf16 => (&BF16, "bf16"),
        FiDtype::F16 => (&F16, "f16"),
    };
    slot.get_or_init(|| {
        let load = |name: String| {
            gpu.kernel("qwen4_gdn_norm_fp8", &name)
                .map_err(|e| e.to_string())
        };
        Ok(Kernels {
            amax: load(format!("qwen4_gdn_norm_amax_{suffix}"))?,
            quant: load(format!("qwen4_gdn_norm_fp8_{suffix}"))?,
            amax_v2: load(format!("qwen4_gdn_norm_amax_{suffix}_v2"))?,
            quant_v2: load(format!("qwen4_gdn_norm_fp8_{suffix}_v2"))?,
            quant_v2_strip: load(format!("qwen4_gdn_norm_fp8_{suffix}_v2_strip"))?,
            norm_amax: load(format!("qwen4_gated_rms_norm_{suffix}in_amax"))?,
            norm_amax_v2: load(format!("qwen4_gated_rms_norm_{suffix}in_amax_v2"))?,
        })
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: kernel unavailable: {e}"))
}

pub(super) fn quant_v2_strip_kernel(gpu: &dyn GpuBackend, dtype: FiDtype) -> Result<KernelHandle> {
    Ok(kernels(gpu, dtype)?.quant_v2_strip)
}

/// Producer::GdnNorm: writes BF16 normed output into `norm_out` AND reduces tensor-wide absmax
/// into `qwen4_fp8_act_amax::slot` so the following `ssm out_proj` skips its `qwen4_fp8_act_amax` pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_gated_norm_16in_amax(
    gpu: &dyn GpuBackend,
    dtype: FiDtype,
    s: super::qwen4_prefill_exact_plan::SsmShape,
    gdn_out: DevicePtr,
    z: DevicePtr,
    norm_weight: DevicePtr,
    norm_out: DevicePtr,
    rows: usize,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let k = kernels(gpu, dtype)?;
    let rows_u32 = u32::try_from(rows)?;
    let amax_slot = crate::layers::qwen4_fp8_act_amax::slot(gpu)?;
    gpu.memset_async(amax_slot, 0, 4, stream)?;
    let use_v2 = v2_selected()?
        && rows_u32 > 0
        && s.nv.is_multiple_of(16)
        && [gdn_out.0, z.0, norm_weight.0, norm_out.0]
            .iter()
            .all(|p| p.is_multiple_of(16));
    if use_v2 {
        KernelLaunch::new(gpu, k.norm_amax_v2)
            .grid([(s.nv / 16) as u32, rows_u32, 1])
            .block([256, 1, 1])
            .arg_ptr(gdn_out)
            .arg_ptr(z)
            .arg_ptr(norm_weight)
            .arg_ptr(norm_out)
            .arg_ptr(amax_slot)
            .arg_u32(128)
            .arg_f32(eps)
            .arg_u32(s.value_dim as u32)
            .arg_u32(s.qkvz as u32)
            .arg_u32(s.value_dim as u32)
            .arg_u32(rows_u32)
            .launch(stream)?;
    } else {
        KernelLaunch::new(gpu, k.norm_amax)
            .grid([s.nv as u32, rows_u32, 1])
            .block([128, 1, 1])
            .arg_ptr(gdn_out)
            .arg_ptr(z)
            .arg_ptr(norm_weight)
            .arg_ptr(norm_out)
            .arg_ptr(amax_slot)
            .arg_u32(128)
            .arg_f32(eps)
            .arg_u32(s.value_dim as u32)
            .arg_u32(s.qkvz as u32)
            .arg_u32(s.value_dim as u32)
            .launch(stream)?;
    }
    crate::layers::qwen4_fp8_act_amax::note_ready(crate::layers::qwen4_fp8_act_amax::Ready {
        act: norm_out.0,
        rows,
        cols: s.value_dim,
        stream,
        producer: crate::layers::qwen4_fp8_act_amax::Producer::GdnNorm,
    });
    Ok(())
}

/// FP8 scratch of the `=3` path: e4m3 activation rows written by the FI
/// kernel, then the out-projection GEMM.
#[derive(Clone, Copy)]
pub(super) struct Nf8Scratch {
    w8: DevicePtr,
    ws: DevicePtr,
    a8: DevicePtr,
}

impl Nf8Scratch {
    /// (e4m3 rows, scale) for the FI kernel; same slots as fp8_gemm_ex's
    /// activation (a8, ws + 12).
    pub(super) fn fi_outputs(self) -> (DevicePtr, DevicePtr) {
        (self.a8, self.ws.offset(12))
    }
}

impl Qwen3SsmLayer {
    pub(super) fn qwen4_gdn_nf8_scratch(
        &self,
        rows: usize,
        ctx: &ForwardContext,
    ) -> Result<Nf8Scratch> {
        let shape = super::qwen4_prefill_exact_plan::SsmShape::from_heads(
            ctx.config.linear_num_key_heads,
            ctx.config.linear_num_value_heads,
        );
        let (w8, ws, a8, _) =
            qwen4_fast_proj::fp8_scratch(ctx.gpu, Plan::H, rows, shape.value_dim)?;
        Ok(Nf8Scratch { w8, ws, a8 })
    }

    /// `=3`: the FP8 out projection of the e4m3 rows the FI kernel wrote.
    pub(super) fn qwen4_gdn_nf8_out_proj(
        &self,
        s: Nf8Scratch,
        output: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let shape = super::qwen4_prefill_exact_plan::SsmShape::from_heads(
            ctx.config.linear_num_key_heads,
            ctx.config.linear_num_value_heads,
        );
        let tk = qwen4_fast_proj::fp8_tensor_kernels(ctx.gpu)?;
        let weight = &self.ssm.out_proj;
        ensure!(
            !weight.weight.is_null()
                && !weight.weight_scale.is_null()
                && weight.weight_scale_2.is_finite(),
            "{SELECTOR}: invalid NVFP4 out_proj weight"
        );
        // Weight absmax slot only: the FI kernel wrote the activation scale at ws + 12.
        // When the weight tensor is already cached, fp8_weight_tensor_gemm issues the 4-byte fence.
        if !qwen4_fast_proj::has_cached_weight_tensor(weight, Plan::H, shape.value_dim, None) {
            ctx.gpu.memset_async(s.ws, 0, 4, stream)?;
        }
        let m_gemm = qwen4_fast_proj::resolve_fp8_pad_m(rows, Plan::H);
        if m_gemm > rows {
            ctx.gpu.memset_async(
                s.a8.offset(rows * shape.value_dim),
                0,
                (m_gemm - rows) * shape.value_dim,
                stream,
            )?;
        }
        qwen4_fast_proj::fp8_weight_tensor_gemm(
            ctx.gpu,
            &tk,
            weight,
            output,
            m_gemm,
            Plan::H,
            shape.value_dim,
            Plan::H,
            None,
            s.w8,
            s.ws,
            s.a8,
            true,
            stream,
        )?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "SSM_PREFILL_GDN_NORM_FP8_ENGAGED selector={SELECTOR} rows={rows} m_gemm={m_gemm} mode={}",
                Mode::FiStatic.name()
            );
        });
        Ok(())
    }

    /// Gated RMS norm of the FI route's 16-bit GDN output (`gdn_out`, stride
    /// VALUE_DIM; gate rows `z`, stride QKVZ) quantized straight to e4m3, then
    /// the FP8 tensor-scale out projection into `output` (`[rows, H]`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen4_gdn_norm_fp8_out_proj(
        &self,
        mode: Mode,
        gdn_out: DevicePtr,
        z: DevicePtr,
        output: DevicePtr,
        rows: usize,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let dtype = super::qwen4_gdn_fi::dtype().ok_or_else(|| {
            anyhow::anyhow!(
                "{SELECTOR}: {} is not selected",
                super::qwen4_gdn_fi::SELECTOR
            )
        })?;
        let k = kernels(ctx.gpu, dtype)?;
        let tk = qwen4_fast_proj::fp8_tensor_kernels(ctx.gpu)?;
        let weight = &self.ssm.out_proj;
        ensure!(
            !weight.weight.is_null()
                && !weight.weight_scale.is_null()
                && weight.weight_scale_2.is_finite(),
            "{SELECTOR}: invalid NVFP4 out_proj weight"
        );
        let s = super::qwen4_prefill_exact_plan::SsmShape::from_heads(
            ctx.config.linear_num_key_heads,
            ctx.config.linear_num_value_heads,
        );
        let (n, kd) = (Plan::H, s.value_dim);
        let rows_u32 = u32::try_from(rows)?;
        let (w8, ws, a8, _) = qwen4_fast_proj::fp8_scratch(ctx.gpu, n, rows, kd)?;
        // Same scratch layout as fp8_gemm_ex: [w amax bits, a amax bits, w scale, a scale];
        // in Mode::Static with cached weight, neither ws nor ws+4 is read as an atomicMax slot
        // and fp8_weight_tensor_gemm issues the 4-byte ordering fence on ws before the GEMM.
        let (a_amax, a_scale) = (ws.offset(4), ws.offset(12));
        if matches!(mode, Mode::Exact)
            || !qwen4_fast_proj::has_cached_weight_tensor(weight, n, kd, None)
        {
            ctx.gpu.memset_async(ws, 0, 8, stream)?;
        }
        let use_v2 = v2_selected()?
            && rows_u32 > 0
            && s.nv.is_multiple_of(16)
            && [gdn_out.0, z.0, self.ssm.norm.weight.0]
                .iter()
                .all(|p| p.is_multiple_of(16))
            && a8.0.is_multiple_of(8);
        let amax_bits = match mode {
            Mode::Exact => {
                if use_v2 {
                    KernelLaunch::new(ctx.gpu, k.amax_v2)
                        .grid([(s.nv / 16) as u32, rows_u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(gdn_out)
                        .arg_ptr(z)
                        .arg_ptr(self.ssm.norm.weight)
                        .arg_ptr(a_amax)
                        .arg_u32(128)
                        .arg_f32(eps)
                        .arg_u32(s.value_dim as u32)
                        .arg_u32(s.qkvz as u32)
                        .arg_u32(rows_u32)
                        .launch(stream)?;
                } else {
                    KernelLaunch::new(ctx.gpu, k.amax)
                        .grid([s.nv as u32, rows_u32, 1])
                        .block([128, 1, 1])
                        .arg_ptr(gdn_out)
                        .arg_ptr(z)
                        .arg_ptr(self.ssm.norm.weight)
                        .arg_ptr(a_amax)
                        .arg_u32(128)
                        .arg_f32(eps)
                        .arg_u32(s.value_dim as u32)
                        .arg_u32(s.qkvz as u32)
                        .launch(stream)?;
                }
                a_amax
            }
            Mode::Static | Mode::FiStatic => DevicePtr::NULL,
        };
        if use_v2 {
            let discard_flags = quant_v2_discard_flags(l2_discard_policy()?, gdn_out, s.value_dim);
            KernelLaunch::new(ctx.gpu, k.quant_v2)
                .grid([(s.nv / 16) as u32, rows_u32.div_ceil(4), 1])
                .block([256, 1, 1])
                .arg_ptr(gdn_out)
                .arg_ptr(z)
                .arg_ptr(self.ssm.norm.weight)
                .arg_ptr(amax_bits)
                .arg_ptr(a8)
                .arg_ptr(a_scale)
                .arg_u32(128)
                .arg_f32(eps)
                .arg_u32(s.value_dim as u32)
                .arg_u32(s.qkvz as u32)
                .arg_u32(s.value_dim as u32)
                .arg_u32(rows_u32)
                .arg_u32(discard_flags)
                .launch(stream)?;
        } else {
            KernelLaunch::new(ctx.gpu, k.quant)
                .grid([s.nv as u32, rows_u32, 1])
                .block([128, 1, 1])
                .arg_ptr(gdn_out)
                .arg_ptr(z)
                .arg_ptr(self.ssm.norm.weight)
                .arg_ptr(amax_bits)
                .arg_ptr(a8)
                .arg_ptr(a_scale)
                .arg_u32(128)
                .arg_f32(eps)
                .arg_u32(s.value_dim as u32)
                .arg_u32(s.qkvz as u32)
                .arg_u32(s.value_dim as u32)
                .launch(stream)?;
        }
        let m_gemm = qwen4_fast_proj::resolve_fp8_pad_m(rows, n);
        if m_gemm > rows {
            ctx.gpu
                .memset_async(a8.offset(rows * kd), 0, (m_gemm - rows) * kd, stream)?;
        }
        qwen4_fast_proj::fp8_weight_tensor_gemm(
            ctx.gpu, &tk, weight, output, m_gemm, n, kd, n, None, w8, ws, a8, true, stream,
        )?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "SSM_PREFILL_GDN_NORM_FP8_ENGAGED selector={SELECTOR} rows={rows} m_gemm={m_gemm} mode={} fi_dtype={dtype:?} v2={use_v2}",
                mode.name()
            );
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn parse_selector() {
        assert_eq!(parse(None).unwrap(), None);
        assert_eq!(parse(Some("0")).unwrap(), None);
        assert_eq!(parse(Some(" ")).unwrap(), None);
        assert_eq!(parse(Some("1")).unwrap(), Some(Mode::Exact));
        assert_eq!(parse(Some("2")).unwrap(), Some(Mode::Static));
        assert_eq!(parse(Some("3")).unwrap(), Some(Mode::FiStatic));
        assert!(parse(Some("4")).is_err());
        assert!(parse(Some("tensor")).is_err());
    }

    #[test]
    fn parse_l2_discard_policy() {
        assert_eq!(parse_l2_discard(None).unwrap(), L2DiscardPolicy::Enabled);
        assert_eq!(
            parse_l2_discard(Some("")).unwrap(),
            L2DiscardPolicy::Enabled
        );
        assert_eq!(
            parse_l2_discard(Some(" 1 ")).unwrap(),
            L2DiscardPolicy::Enabled
        );
        assert_eq!(
            parse_l2_discard(Some("ON")).unwrap(),
            L2DiscardPolicy::Enabled
        );
        assert_eq!(
            parse_l2_discard(Some("true")).unwrap(),
            L2DiscardPolicy::Enabled
        );
        assert_eq!(
            parse_l2_discard(Some("0")).unwrap(),
            L2DiscardPolicy::Disabled
        );
        assert_eq!(
            parse_l2_discard(Some("off")).unwrap(),
            L2DiscardPolicy::Disabled
        );
        assert_eq!(
            parse_l2_discard(Some("False")).unwrap(),
            L2DiscardPolicy::Disabled
        );
        assert!(parse_l2_discard(Some("2")).is_err());
        assert!(parse_l2_discard(Some("maybe")).is_err());
    }

    #[test]
    fn resolve_strip_discard_alignment_and_opt_out() {
        let scratch = DevicePtr(0x1000_0000);
        let max_strip_rows = 640usize;
        let qkv_strip = scratch;
        let gdn_out_strip = scratch.offset(max_strip_rows * Plan::CONV_DIM * 2);
        let deinterleaved = DevicePtr(0x2000_0000);
        let z_row = deinterleaved.offset(Plan::CONV_DIM * 2 + 512 * Plan::QKVZ * 2);
        let bufs = StripL2DiscardBuffers {
            qkv_strip,
            gdn_out_strip,
            z_row,
            conv_dim: Plan::CONV_DIM,
            value_dim: Plan::VALUE_DIM,
            qkvz_stride: Plan::QKVZ,
        };

        // Enabled and 128B-aligned: all three buffers are eligible (including on last_strip).
        let enabled = resolve_strip_l2_discard(L2DiscardPolicy::Enabled, bufs);
        assert_eq!(enabled.qkv_discard, qkv_strip);
        assert_eq!(
            enabled.discard_flags,
            DISC_FLAG_X | DISC_FLAG_G | DISC_FLAG_QKV
        );

        // Opt-out disables every discard bit and nulls qkv_discard.
        let disabled = resolve_strip_l2_discard(L2DiscardPolicy::Disabled, bufs);
        assert_eq!(disabled.qkv_discard, DevicePtr::NULL);
        assert_eq!(disabled.discard_flags, 0);

        // Misaligned individual buffers drop only their own discard bit.
        let bad_x = resolve_strip_l2_discard(
            L2DiscardPolicy::Enabled,
            StripL2DiscardBuffers {
                gdn_out_strip: gdn_out_strip.offset(16),
                ..bufs
            },
        );
        assert_eq!(bad_x.qkv_discard, qkv_strip);
        assert_eq!(bad_x.discard_flags, DISC_FLAG_G | DISC_FLAG_QKV);

        let bad_g = resolve_strip_l2_discard(
            L2DiscardPolicy::Enabled,
            StripL2DiscardBuffers {
                z_row: z_row.offset(64),
                ..bufs
            },
        );
        assert_eq!(bad_g.qkv_discard, qkv_strip);
        assert_eq!(bad_g.discard_flags, DISC_FLAG_X | DISC_FLAG_QKV);

        let bad_qkv = resolve_strip_l2_discard(
            L2DiscardPolicy::Enabled,
            StripL2DiscardBuffers {
                qkv_strip: qkv_strip.offset(32),
                ..bufs
            },
        );
        assert_eq!(bad_qkv.qkv_discard, DevicePtr::NULL);
        assert_eq!(bad_qkv.discard_flags, DISC_FLAG_X | DISC_FLAG_G);

        // Non-strip quant_v2 alignment and opt-out checks.
        assert_eq!(
            quant_v2_discard_flags(L2DiscardPolicy::Enabled, gdn_out_strip, Plan::VALUE_DIM),
            DISC_FLAG_X
        );
        assert_eq!(
            quant_v2_discard_flags(L2DiscardPolicy::Disabled, gdn_out_strip, Plan::VALUE_DIM),
            0
        );
        assert_eq!(
            quant_v2_discard_flags(
                L2DiscardPolicy::Enabled,
                gdn_out_strip.offset(16),
                Plan::VALUE_DIM
            ),
            0
        );
    }

    #[test]
    fn strip_l2_discard_covers_exact_disjoint_cache_lines() {
        let scratch_base = 0x1000_0000usize;
        let max_strip_rows = 640usize;
        let qkv_base = scratch_base;
        let gdn_out_base = scratch_base + max_strip_rows * Plan::CONV_DIM * 2;
        let deinterleaved_base = 0x2000_0000usize;
        let z_col_byte_offset = Plan::CONV_DIM * 2;
        let qkvz_row_bytes = Plan::QKVZ * 2;
        let qkv_row_bytes = Plan::CONV_DIM * 2;
        let gdn_out_row_bytes = Plan::VALUE_DIM * 2;

        assert!(qkv_base.is_multiple_of(L2_DISCARD_LINE_BYTES));
        assert!(gdn_out_base.is_multiple_of(L2_DISCARD_LINE_BYTES));
        assert!(z_col_byte_offset.is_multiple_of(L2_DISCARD_LINE_BYTES));
        assert!(qkvz_row_bytes.is_multiple_of(L2_DISCARD_LINE_BYTES));
        assert!(qkv_row_bytes.is_multiple_of(L2_DISCARD_LINE_BYTES));
        assert!(gdn_out_row_bytes.is_multiple_of(L2_DISCARD_LINE_BYTES));

        // Verify exact bijective coverage for non-multiple-of-8 tails (1, 7, 9, 129, 513, 517),
        // exact 8-row multiple (8), full (512), and tail-absorbed (640) strip heights.
        let rows_per_cta = QUANT_V2_STRIP_ROWS_PER_CTA as usize;
        for ns in [1usize, 7, 8, 9, 129, 512, 513, 517, 640] {
            let grid = quant_v2_strip_grid(u32::try_from(ns).expect("strip height fits u32"));
            let grid_y = grid[1] as usize;
            assert_eq!(grid[0], 3);
            assert_eq!(grid[2], 1);
            if ns == 512 || ns == 640 {
                assert!(
                    (grid[0] * grid[1]).is_multiple_of(48),
                    "SM48 exact wave for ns={ns}"
                );
            }
            let mut qkv_lines = HashSet::new();
            let mut gdn_out_lines = HashSet::new();
            let mut gate_lines = HashSet::new();

            for by in 0..grid_y {
                let row0 = (grid_y - 1 - by) * rows_per_cta;
                let r_end = rows_per_cta.min(ns - row0);
                for bx in 0..3usize {
                    // 1) qkv_strip discard: matches qwen4_nf8_quant_v2_body<InT, true>
                    //    (<= 2 passes of 768 lines covering r_end * 160 <= 1280 lines).
                    let total_lines = r_end * 160;
                    for tid in 0..256usize {
                        let tid_grp = bx * 256 + tid;
                        for line in [tid_grp, tid_grp + 768] {
                            if line < total_lines {
                                let addr = qkv_base + (row0 * 160 + line) * L2_DISCARD_LINE_BYTES;
                                assert!(
                                    addr.is_multiple_of(L2_DISCARD_LINE_BYTES),
                                    "qkv_strip addr {addr:#x} not 128B aligned"
                                );
                                assert!(
                                    addr + L2_DISCARD_LINE_BYTES <= qkv_base + ns * qkv_row_bytes,
                                    "qkv_strip addr {addr:#x} out of bounds for ns={ns}"
                                );
                                assert!(
                                    addr + L2_DISCARD_LINE_BYTES <= gdn_out_base,
                                    "qkv_strip discard overlaps gdn_out_strip for ns={ns}"
                                );
                                assert!(
                                    qkv_lines.insert(addr),
                                    "duplicate qkv_strip discard at {addr:#x} for ns={ns}"
                                );
                            }
                        }
                    }

                    // 2) gdn_out_strip & gate (z) per-row inline discards
                    for tid in 0..256usize {
                        let warp = tid >> 5;
                        let lane = tid & 31;
                        let sub = lane >> 4;
                        let q = lane & 15;
                        if q != 0 && q != 8 {
                            continue;
                        }
                        let head = bx * 16 + warp * 2 + sub;
                        for r in 0..r_end {
                            let row = row0 + r;
                            let elem_col = head * 128 + q * 8;
                            let x_addr = gdn_out_base + (row * Plan::VALUE_DIM + elem_col) * 2;
                            assert!(
                                x_addr.is_multiple_of(L2_DISCARD_LINE_BYTES),
                                "gdn_out_strip addr {x_addr:#x} not 128B aligned"
                            );
                            assert!(
                                x_addr + L2_DISCARD_LINE_BYTES
                                    <= gdn_out_base + ns * gdn_out_row_bytes,
                                "gdn_out_strip addr {x_addr:#x} out of bounds for ns={ns}"
                            );
                            assert!(
                                gdn_out_lines.insert(x_addr),
                                "duplicate gdn_out_strip discard at {x_addr:#x} for ns={ns}"
                            );

                            let g_addr = deinterleaved_base
                                + z_col_byte_offset
                                + (row * Plan::QKVZ + elem_col) * 2;
                            assert!(
                                g_addr.is_multiple_of(L2_DISCARD_LINE_BYTES),
                                "gate addr {g_addr:#x} not 128B aligned"
                            );
                            let row_start = deinterleaved_base + row * qkvz_row_bytes;
                            assert!(
                                g_addr >= row_start + z_col_byte_offset
                                    && g_addr + L2_DISCARD_LINE_BYTES <= row_start + qkvz_row_bytes,
                                "gate discard {g_addr:#x} touches Q|K|V columns for ns={ns}"
                            );
                            assert!(
                                gate_lines.insert(g_addr),
                                "duplicate gate discard at {g_addr:#x} for ns={ns}"
                            );
                        }
                    }
                }
            }

            assert_eq!(
                qkv_lines.len(),
                ns * qkv_row_bytes / L2_DISCARD_LINE_BYTES,
                "qkv_strip line count mismatch for ns={ns}"
            );
            assert_eq!(
                gdn_out_lines.len(),
                ns * gdn_out_row_bytes / L2_DISCARD_LINE_BYTES,
                "gdn_out_strip line count mismatch for ns={ns}"
            );
            assert_eq!(
                gate_lines.len(),
                ns * gdn_out_row_bytes / L2_DISCARD_LINE_BYTES,
                "gate line count mismatch for ns={ns}"
            );
        }

        // Cold 16K prefill (36 SSM layers x 32 strips x 512 rows) writeback elimination accounting.
        let layers = 36usize;
        let strips = 32usize;
        let strip_rows = 512usize;
        let qkv_total_bytes = layers * strips * strip_rows * qkv_row_bytes;
        let qkv_last_strip_bytes = layers * strip_rows * qkv_row_bytes;
        let gdn_out_total_bytes = layers * strips * strip_rows * gdn_out_row_bytes;
        assert_eq!(qkv_total_bytes, 12_079_595_520);
        assert_eq!(qkv_last_strip_bytes, 377_487_360);
        assert_eq!(gdn_out_total_bytes, 7_247_757_312);
    }

    #[cfg(not(atlas_fi_gdn))]
    #[test]
    fn mode_without_aot_falls_back_cleanly() {
        // Without FlashInfer GDN AOT objects (`cfg(not(atlas_fi_gdn))`), recipes
        // that request `ATLAS_QWEN4_PREFILL_GDN_NORM_FP8=1|2|3` must gracefully
        // fall back to `Ok(None)` rather than failing every prefill request.
        assert_eq!(
            mode().expect("cfg(not(atlas_fi_gdn)) mode() must not hard-error"),
            None
        );
    }
}
