// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in FlashInfer SM121a chunked GDN for Flash-Next prefill
//! (`ATLAS_QWEN4_PREFILL_GDN_FI=1` BF16 export, `=2` FP16 export; default off).
//! The FP16 export is the same kernel with 10 mantissa bits for q/k/v/o and
//! its in-kernel MMA operands: ~8x lower error than BF16 at the same speed on
//! Flash-Next shapes.
//!
//! Replaces the exact path's token-serial register recurrence (one value head
//! per CTA, ~3.9 us per token per layer) with the exported CuTe-DSL chunked
//! delta-rule kernel through the bridge in `csrc/fi_gdn/` (shared with the
//! Qwen3.8-27B lane: same h_qk=16 / h_v=48 / D=128 geometry, FP32 [HV][K][V]
//! state, multiplicative decay gates interleaved [T, 2*HV]).
//!
//! NOT bit-exact: Q/K/V and the GDN output cross the kernel boundary in 16 bits,
//! where the exact path keeps them in FP32 rows. The conv/L2 and gated-norm
//! arithmetic is otherwise the exact kernels' (`qwen4_gdn_fi_glue.cu`).

use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::*;

pub(super) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_FI";

#[cfg(atlas_fi_gdn)]
type BridgeFn = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    i32,
    i32,
    *mut core::ffi::c_void,
) -> i32;

#[cfg(atlas_fi_gdn)]
mod ffi {
    unsafe extern "C" {
        pub fn atlas_gdn_fi_sm121a_prefill(
            qkv: *mut core::ffi::c_void,
            gate_beta: *mut core::ffi::c_void,
            output: *mut core::ffi::c_void,
            state: *mut core::ffi::c_void,
            total_len: i32,
            conv_dim: i32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        pub fn atlas_gdn_fi_sm121a_prefill_f16(
            qkv: *mut core::ffi::c_void,
            gate_beta: *mut core::ffi::c_void,
            output: *mut core::ffi::c_void,
            state: *mut core::ffi::c_void,
            total_len: i32,
            conv_dim: i32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        pub fn atlas_gdn_fi_sm121a_prefill_f16_strip(
            qkv: *mut core::ffi::c_void,
            gate_in: *mut core::ffi::c_void,
            beta_in: *mut core::ffi::c_void,
            output: *mut core::ffi::c_void,
            state: *mut core::ffi::c_void,
            strip_len: i32,
            conv_dim: i32,
            strip_flags: i32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        /// FP16 export with the gated norm + static e4m3 quantization in its
        /// store warps (csrc/fi_gdn/fi_gdn_bridge_f16_nf8.cu).
        pub fn atlas_gdn_fi_sm121a_prefill_f16_nf8(
            qkv: *mut core::ffi::c_void,
            gate_beta: *mut core::ffi::c_void,
            output: *mut core::ffi::c_void,
            state: *mut core::ffi::c_void,
            total_len: i32,
            conv_dim: i32,
            z: *mut core::ffi::c_void,
            z_stride: i64,
            normw: *mut core::ffi::c_void,
            out8: *mut core::ffi::c_void,
            scale_out: *mut core::ffi::c_void,
            eps: f32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        /// Strip-mined L2-resident FP16 export with gated norm + static e4m3
        /// quantization in its store warps (csrc/fi_gdn/fi_gdn_bridge_f16_nf8.cu).
        pub fn atlas_gdn_fi_sm121a_prefill_f16_nf8_strip(
            qkv: *mut core::ffi::c_void,
            gate_in: *mut core::ffi::c_void,
            beta_in: *mut core::ffi::c_void,
            output: *mut core::ffi::c_void,
            state: *mut core::ffi::c_void,
            strip_len: i32,
            conv_dim: i32,
            strip_flags: i32,
            z: *mut core::ffi::c_void,
            z_stride: i64,
            normw: *mut core::ffi::c_void,
            out8: *mut core::ffi::c_void,
            scale_out: *mut core::ffi::c_void,
            eps: f32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
        /// Load-time sizing of each bridge's buffers (module, state,
        /// tensormaps, cu_seqlens, gate/beta for `tokens` rows).
        pub fn atlas_gdn_fi_sm121a_prefill_reserve(tokens: i64) -> i32;
        pub fn atlas_gdn_fi_sm121a_prefill_f16_reserve(tokens: i64) -> i32;
        pub fn atlas_gdn_fi_sm121a_prefill_f16_nf8_reserve(tokens: i64) -> i32;
    }
}

/// 16-bit format of q/k/v/o across the FlashInfer kernel boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FiDtype {
    #[cfg_attr(
        not(atlas_fi_gdn),
        expect(
            dead_code,
            reason = "only constructed when cfg(atlas_fi_gdn) is enabled"
        )
    )]
    Bf16,
    F16,
}

impl FiDtype {
    const fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        }
    }

    #[cfg(atlas_fi_gdn)]
    fn bridge(self) -> BridgeFn {
        match self {
            Self::Bf16 => ffi::atlas_gdn_fi_sm121a_prefill,
            Self::F16 => ffi::atlas_gdn_fi_sm121a_prefill_f16,
        }
    }
}

#[cfg(atlas_fi_gdn)]
pub(super) fn dtype() -> Option<FiDtype> {
    static SEL: std::sync::OnceLock<Option<FiDtype>> = std::sync::OnceLock::new();
    *SEL.get_or_init(|| match std::env::var(SELECTOR).ok().as_deref() {
        Some("1") => Some(FiDtype::Bf16),
        Some("2") => Some(FiDtype::F16),
        _ => None,
    })
}

#[cfg(not(atlas_fi_gdn))]
pub(super) fn dtype() -> Option<FiDtype> {
    static WARNED: std::sync::Once = std::sync::Once::new();
    if matches!(
        std::env::var(SELECTOR).ok().as_deref(),
        Some("1") | Some("2")
    ) {
        WARNED.call_once(|| {
            tracing::warn!(
                "{SELECTOR} requested, but this build was compiled without FlashInfer GDN AOT objects (cfg(atlas_fi_gdn) disabled); falling back to native CUDA GDN prefill"
            );
        });
    }
    None
}

pub(super) fn selected() -> bool {
    dtype().is_some()
}

/// Default strip height (8 x 64 exact FlashInfer GDN chunks) for strip-mined
/// L2-resident GDN prefill on GB10 (`<= 64` SMs). At 512 rows, `qkv_strip`
/// (10.0 MiB) + recurrent state ping-pong (6.0 MiB) + `out_strip` (6.0 MiB) +
/// gates (0.19 MiB) = ~22.2 MiB stays inside GB10's verified 24.0 MiB L2 cache
/// (whereas 768 rows reaches 31.2 MiB and spills L2).
const DEFAULT_STRIP_ROWS: usize = 512;
/// Default strip height (32 x 64 exact FlashInfer GDN chunks) for high-SM
/// Blackwell GPUs (`active_sm_count() > 64`, e.g. SM120 188 SMs with 96 MiB L2)
/// in serial strip mode: 2048 rows keeps `qkv_strip` + state + `out_strip`
/// (~70 MiB) inside the 96 MiB L2 cache while cutting strip-loop launch overhead 4x.
const DEFAULT_STRIP_ROWS_HIGH_SM: usize = 2048;
/// Default strip height (16 x 64 exact FlashInfer GDN chunks) when ping-pong
/// double-buffered strip overlap is active on high-SM Blackwell GPUs
/// (`active_sm_count() > 64`): two active slots = `2 * (20 + 12) = 64 MiB` +
/// `6 MiB` recurrent state = `70 MiB <= 96 MiB` SM120 L2 cache while halving
/// non-overlapped prologue/epilogue fraction vs 2048-row strips.
const DEFAULT_PING_PONG_STRIP_ROWS_HIGH_SM: usize = 1024;
/// SM-count threshold separating GB10-class (`<= 64` SMs) from high-SM Blackwell
/// GPUs (`> 64` SMs) when choosing the default strip height and overlap policy.
const HIGH_SM_THRESHOLD: u32 = 64;
/// Tail-absorption margin added to `strip_rows` so the final strip always has
/// `129..=strip_rows + 128` rows (strictly above `SHORT_FALLBACK_MAX_ROWS = 64`).
const STRIP_TAIL_MARGIN: usize = 128;
/// Selector for double-buffered ping-pong strip overlap (`conv_v4_strip(s+1)` +
/// `quant(s-1)` on a secondary stream concurrently with 48-CTA `fi_gdn_f16_strip(s)`
/// on the main stream).
/// - unset / `""`: auto-ON on `active_sm_count() > 64` (SM120/SM103); OFF on GB10 (`<= 64`).
/// - `"0"` / `"off"` / `"false"` / `"no"`: disabled.
/// - `"1"` / `"on"` / `"true"` / `"yes"`: enabled.
const STRIP_OVERLAP_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_STRIP_OVERLAP";
/// Opt-in selector to enable fused `f16_nf8_strip` instead of `f16_strip`
/// paired with `quant_v2_strip` (`1` / `on` / `true` enables; default disabled
/// because `f16_strip + quant_v2_strip` is faster on GB10 sm_121a).
const NF8_STRIP_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_NF8_STRIP";
/// Optional override for strip height (must be >= 128 and a multiple of 64).
const STRIP_ROWS_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_FI_STRIP_ROWS";
/// Opt-out selector for `qwen4_conv1d_l2norm_{f16,bf16}_prefill_v4_strip`
/// (16B `ld128`/`st128`, cooperative 4-warp SMEM ring buffer, and safe
/// `discard.global.L2` on consumed `new_input` QKV rows; default on when
/// strip-mined conv is active; `0` / `off` / `false` reverts to `v3_strip`).
const CONV_V4_STRIP_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_CONV_V4_STRIP";

/// Launch geometry for `qwen4_conv1d_l2norm_{f16,bf16}_prefill_v3_strip`,
/// matching `QWEN4_FI_CONV_V3_STRIP_CHANNELS_PER_CTA`, `QWEN4_FI_CONV_V3_STRIP_WARPS`,
/// and `QWEN4_FI_CONV_V3_STRIP_TOKENS` in `qwen4_gdn_fi_glue.cu` (128 channels/CTA =
/// 1 Q/K head, 4 warps x up to 48 tokens = 192 rows/CTA, balanced evenly across
/// `gridDim.y * 4` warps so 512-row strips at `conv_dim = 10240` launch
/// `(80, 3) = 240 CTAs = 5.00` exact waves on 48 SMs).
const CONV_V3_STRIP_CHANNELS_PER_CTA: u32 = 128;
const CONV_V3_STRIP_WARPS: u32 = 4;
const CONV_V3_STRIP_TOKENS: u32 = 48;
const CONV_V3_STRIP_BLOCK_THREADS: u32 = CONV_V3_STRIP_WARPS * 32;
const CONV_V3_STRIP_ROWS_PER_CTA: u32 = CONV_V3_STRIP_WARPS * CONV_V3_STRIP_TOKENS;

/// Launch geometry for `qwen4_conv1d_l2norm_{f16,bf16}_prefill_v4_strip`,
/// matching `QWEN4_FI_CONV_V4_STRIP_CHANNELS_PER_CTA`, `QWEN4_FI_CONV_V4_STRIP_WARPS`,
/// `QWEN4_FI_CONV_V4_STRIP_ROWS_PER_CTA`, and `QWEN4_FI_CONV_V4_STRIP_DISC_INPUT` in
/// `qwen4_gdn_fi_glue.cu` (256 channels/CTA = 2 Q/K heads, 8 channels/lane = 16B
/// `ld128`/`st128`, 4 warps cooperatively stepping 4 rows/iter via a 16-slot SMEM
/// ring buffer; 512-row strips at `conv_dim = 10240` launch `(40, 6) = 240 CTAs =
/// 5.00` exact waves on 48 SMs).
const CONV_V4_STRIP_CHANNELS_PER_CTA: u32 = 256;
const CONV_V4_STRIP_WARPS: u32 = 4;
const CONV_V4_STRIP_ROWS_PER_CTA: u32 = 96;
const CONV_V4_STRIP_BLOCK_THREADS: u32 = CONV_V4_STRIP_WARPS * 32;
const CONV_V4_STRIP_DISC_INPUT: u32 = 1;

/// Non-power-of-2 256-byte-aligned skew inserted between `qkv_strip` and
/// `gdn_out_strip` inside the strip scratch slab (3 x 512 B = 6 x 256 B =
/// 12 x 128 B). Breaks 64 KiB / 256 KiB L2 set aliasing between `gdn_out_strip`,
/// `z`, and `out_fp8` on power-of-2 512-row strips while preserving 256-B
/// alignment for vector loads/stores and 128-B `discard.global.L2` lines.
const GDN_OUT_STRIP_L2_SKEW_BYTES: usize = 1536;

const fn gdn_out_strip_offset(max_strip_rows: usize, conv_dim: usize) -> usize {
    max_strip_rows * conv_dim * 2 + GDN_OUT_STRIP_L2_SKEW_BYTES
}

fn nf8_strip_enabled() -> bool {
    static SEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SEL.get_or_init(|| match std::env::var(NF8_STRIP_SELECTOR) {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true"),
        Err(_) => false,
    })
}

#[inline]
#[must_use]
fn resolve_strip_overlap_enabled(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim) {
        Some(v)
            if v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("no")
                || v.eq_ignore_ascii_case("off") =>
        {
            false
        }
        Some(v)
            if v == "1"
                || v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("yes")
                || v.eq_ignore_ascii_case("on") =>
        {
            true
        }
        _ => sm_count > HIGH_SM_THRESHOLD,
    }
}

fn strip_overlap_enabled() -> bool {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(STRIP_OVERLAP_SELECTOR).ok());
    resolve_strip_overlap_enabled(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[inline]
#[must_use]
const fn default_strip_rows_for_sm(sm_count: u32) -> usize {
    if sm_count > HIGH_SM_THRESHOLD {
        DEFAULT_STRIP_ROWS_HIGH_SM
    } else {
        DEFAULT_STRIP_ROWS
    }
}

#[inline]
#[must_use]
fn resolve_strip_rows_config(override_raw: Option<&str>, sm_count: u32) -> (usize, usize) {
    let strip_rows = override_raw
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&r| r >= 128 && r.is_multiple_of(64))
        .unwrap_or_else(|| default_strip_rows_for_sm(sm_count));
    (strip_rows, strip_rows + STRIP_TAIL_MARGIN)
}

#[inline]
#[must_use]
fn resolve_strip_rows_config_for_mode(
    override_raw: Option<&str>,
    sm_count: u32,
    overlap_enabled: bool,
) -> (usize, usize) {
    let strip_rows = override_raw
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&r| r >= 128 && r.is_multiple_of(64))
        .unwrap_or_else(|| {
            if sm_count > HIGH_SM_THRESHOLD && overlap_enabled {
                DEFAULT_PING_PONG_STRIP_ROWS_HIGH_SM
            } else {
                default_strip_rows_for_sm(sm_count)
            }
        });
    (strip_rows, strip_rows + STRIP_TAIL_MARGIN)
}

fn strip_rows_config() -> (usize, usize) {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(STRIP_ROWS_SELECTOR).ok());
    resolve_strip_rows_config(raw.as_deref(), atlas_core::device::active_sm_count())
}

fn strip_rows_config_for_mode(overlap_enabled: bool) -> (usize, usize) {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(STRIP_ROWS_SELECTOR).ok());
    resolve_strip_rows_config_for_mode(
        raw.as_deref(),
        atlas_core::device::active_sm_count(),
        overlap_enabled,
    )
}

fn conv_v3_selected() -> bool {
    use super::qwen4_gdn_conv_v2 as v2;
    v2::parse_v3(std::env::var(v2::V3_SELECTOR).ok().as_deref()).unwrap_or(false)
}

const fn conv_v3_strip_grid(conv_dim: u32, strip_rows: u32) -> [u32; 3] {
    [
        conv_dim / CONV_V3_STRIP_CHANNELS_PER_CTA,
        strip_rows.div_ceil(CONV_V3_STRIP_ROWS_PER_CTA),
        1,
    ]
}

fn parse_conv_v4_strip(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim) {
        None | Some("") => Ok(true),
        Some(v) => match v.to_ascii_lowercase().as_str() {
            "1" | "on" | "true" => Ok(true),
            "0" | "off" | "false" => Ok(false),
            other => anyhow::bail!(
                "{CONV_V4_STRIP_SELECTOR}: unknown value {other:?} (expected 0, 1, off, on, false, true)"
            ),
        },
    }
}

fn conv_v4_strip_enabled() -> Result<bool> {
    static SEL: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| match std::env::var(CONV_V4_STRIP_SELECTOR) {
        Err(std::env::VarError::NotPresent) => Ok(true),
        Ok(v) => parse_conv_v4_strip(Some(&v)).map_err(|e| e.to_string()),
        Err(e) => Err(format!("invalid {CONV_V4_STRIP_SELECTOR}: {e}")),
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

const fn conv_v4_strip_grid(conv_dim: u32, strip_rows: u32) -> [u32; 3] {
    [
        conv_dim / CONV_V4_STRIP_CHANNELS_PER_CTA,
        strip_rows.div_ceil(CONV_V4_STRIP_ROWS_PER_CTA),
        1,
    ]
}

#[must_use]
fn conv_v4_strip_discard_flags(
    policy: super::qwen4_gdn_norm_fp8::L2DiscardPolicy,
    deinterleaved: DevicePtr,
    input_stride: usize,
) -> u32 {
    if !matches!(policy, super::qwen4_gdn_norm_fp8::L2DiscardPolicy::Enabled) {
        return 0;
    }
    let align = super::qwen4_gdn_norm_fp8::L2_DISCARD_LINE_BYTES;
    if !deinterleaved.is_null()
        && deinterleaved.0.is_multiple_of(align as u64)
        && (input_stride * 2).is_multiple_of(align)
    {
        CONV_V4_STRIP_DISC_INPUT
    } else {
        0
    }
}

/// Bytes one bridge holds for `rows`: gate + beta (FP32 [rows, HV]), the FP32
/// state and its transposed initial copy, the tensormaps and cu_seqlens.
const fn bridge_bytes(rows: usize) -> usize {
    rows * 48 * 4 * 2 + 2 * 48 * 128 * 128 * 4 + 48 * 128 + 16
}

#[cfg(atlas_fi_gdn)]
fn invoke_reserve(dtype: FiDtype, max_rows: usize) -> Result<(&'static str, usize)> {
    use super::qwen4_gdn_norm_fp8::{Mode, mode};
    let (bridge, reserve, reserve_rows): (&str, unsafe extern "C" fn(i64) -> i32, usize) =
        match (dtype, mode()) {
            (FiDtype::F16, Ok(Some(Mode::FiStatic))) => (
                "f16_nf8",
                ffi::atlas_gdn_fi_sm121a_prefill_f16_nf8_reserve,
                max_rows,
            ),
            (FiDtype::F16, Ok(Some(Mode::Static)))
                if nf8_strip_enabled()
                    && conv_v3_selected()
                    && std::env::var_os("ATLAS_QWEN4_GDN_FI_DUMP_DIR").is_none() =>
            {
                let (_, max_strip_rows) = strip_rows_config();
                (
                    "f16_nf8",
                    ffi::atlas_gdn_fi_sm121a_prefill_f16_nf8_reserve,
                    max_rows.min(max_strip_rows),
                )
            }
            (FiDtype::F16, _) => (
                "f16",
                ffi::atlas_gdn_fi_sm121a_prefill_f16_reserve,
                max_rows,
            ),
            (FiDtype::Bf16, _) => ("bf16", ffi::atlas_gdn_fi_sm121a_prefill_reserve, max_rows),
        };
    let tokens = i64::try_from(reserve_rows)?;
    // SAFETY: the reserve entry only allocates and initialises the bridge's own
    // buffers under its mutex; no caller pointer crosses.
    let rc = unsafe { reserve(tokens) };
    ensure!(
        rc == 0,
        "{SELECTOR}: {bridge} bridge reserve failed rc={rc} rows={reserve_rows}"
    );
    Ok((bridge, reserve_rows))
}

#[cfg(not(atlas_fi_gdn))]
fn invoke_reserve(_dtype: FiDtype, _max_rows: usize) -> Result<(&'static str, usize)> {
    anyhow::bail!("{SELECTOR}: FlashInfer GDN AOT bridge is not available in this build")
}

#[cfg(atlas_fi_gdn)]
#[allow(clippy::too_many_arguments)]
fn invoke_prefill(
    dtype: FiDtype,
    qkv: DevicePtr,
    gates: DevicePtr,
    gdn_out: DevicePtr,
    h_state: DevicePtr,
    rows: usize,
    conv_dim_arg: i32,
    z_base: DevicePtr,
    z_stride: i64,
    norm_weight: DevicePtr,
    nf8: Option<(DevicePtr, DevicePtr)>,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let ptr = |p: DevicePtr| p.0 as usize as *mut core::ffi::c_void;
    let rc = if let Some((out8, scale_out)) = nf8 {
        // `Some`: the store warps write the gated norm as static-scale
        // e4m3 rows into `out8` (scale at `scale_out`); `normed` is unused.
        ensure!(
            dtype == FiDtype::F16,
            "{SELECTOR}: the norm-fp8 FI export is FP16 only (=2)"
        );
        // SAFETY: live device buffers sized by the caller's ensure! checks;
        // z rows are the Z columns of `deinterleaved` (stride QKVZ), out8
        // holds rows x VALUE_DIM bytes, scale_out 4.
        unsafe {
            ffi::atlas_gdn_fi_sm121a_prefill_f16_nf8(
                ptr(qkv),
                ptr(gates),
                ptr(gdn_out),
                ptr(h_state),
                i32::try_from(rows)?,
                conv_dim_arg,
                ptr(z_base),
                z_stride,
                ptr(norm_weight),
                ptr(out8),
                ptr(scale_out),
                eps,
                stream as usize as *mut core::ffi::c_void,
            )
        }
    } else {
        let bridge = dtype.bridge();
        // SAFETY: live device buffers sized by the caller's ensure! checks and
        // the exact-route preflight; the bridge only enqueues work on `stream`.
        unsafe {
            bridge(
                ptr(qkv),
                ptr(gates),
                ptr(gdn_out),
                ptr(h_state),
                i32::try_from(rows)?,
                conv_dim_arg,
                stream as usize as *mut core::ffi::c_void,
            )
        }
    };
    ensure!(
        rc == 0,
        "{SELECTOR}: atlas_gdn_fi_sm121a_prefill failed rc={rc} rows={rows}"
    );
    Ok(())
}

#[cfg(not(atlas_fi_gdn))]
#[allow(clippy::too_many_arguments)]
fn invoke_prefill(
    _dtype: FiDtype,
    _qkv: DevicePtr,
    _gates: DevicePtr,
    _gdn_out: DevicePtr,
    _h_state: DevicePtr,
    _rows: usize,
    _conv_dim_arg: i32,
    _z_base: DevicePtr,
    _z_stride: i64,
    _norm_weight: DevicePtr,
    _nf8: Option<(DevicePtr, DevicePtr)>,
    _eps: f32,
    _stream: u64,
) -> Result<()> {
    anyhow::bail!("{SELECTOR}: FlashInfer GDN AOT bridge is not available in this build")
}

#[cfg(atlas_fi_gdn)]
#[allow(clippy::too_many_arguments)]
fn invoke_prefill_f16_nf8_strip(
    qkv_strip: DevicePtr,
    gate_in: DevicePtr,
    beta_in: DevicePtr,
    h_state: DevicePtr,
    ns: usize,
    r0: usize,
    rows: usize,
    conv_dim_i32: i32,
    strip_flags: i32,
    z_strip: DevicePtr,
    z_stride_i64: i64,
    norm_weight: DevicePtr,
    a8_strip: DevicePtr,
    a_scale: DevicePtr,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let ptr = |p: DevicePtr| p.0 as usize as *mut core::ffi::c_void;
    let ns_i32 = i32::try_from(ns)?;
    // SAFETY: all device buffers are live, aligned, and sized for `ns <= max_active_strip`
    // by the caller; `qkv_strip` sizes the unused O descriptor (`conv_dim > value_dim`);
    // `z_strip` points to the Z columns with stride `z_stride_i64`; `a8_strip` holds
    // `ns * value_dim` bytes; `a_scale` holds 4 bytes; the bridge only enqueues GPU work
    // on `stream`.
    let rc = unsafe {
        ffi::atlas_gdn_fi_sm121a_prefill_f16_nf8_strip(
            ptr(qkv_strip),
            ptr(gate_in),
            ptr(beta_in),
            ptr(qkv_strip),
            ptr(h_state),
            ns_i32,
            conv_dim_i32,
            strip_flags,
            ptr(z_strip),
            z_stride_i64,
            ptr(norm_weight),
            ptr(a8_strip),
            ptr(a_scale),
            eps,
            stream as usize as *mut core::ffi::c_void,
        )
    };
    ensure!(
        rc == 0,
        "{SELECTOR}: atlas_gdn_fi_sm121a_prefill_f16_nf8_strip failed rc={rc} r0={r0} ns={ns} rows={rows}"
    );
    Ok(())
}

#[cfg(not(atlas_fi_gdn))]
#[allow(clippy::too_many_arguments)]
fn invoke_prefill_f16_nf8_strip(
    _qkv_strip: DevicePtr,
    _gate_in: DevicePtr,
    _beta_in: DevicePtr,
    _h_state: DevicePtr,
    _ns: usize,
    _r0: usize,
    _rows: usize,
    _conv_dim_i32: i32,
    _strip_flags: i32,
    _z_strip: DevicePtr,
    _z_stride_i64: i64,
    _norm_weight: DevicePtr,
    _a8_strip: DevicePtr,
    _a_scale: DevicePtr,
    _eps: f32,
    _stream: u64,
) -> Result<()> {
    anyhow::bail!("{SELECTOR}: FlashInfer GDN AOT bridge is not available in this build")
}

#[cfg(atlas_fi_gdn)]
#[allow(clippy::too_many_arguments)]
fn invoke_prefill_f16_strip(
    qkv_strip: DevicePtr,
    gate_in: DevicePtr,
    beta_in: DevicePtr,
    gdn_out_strip: DevicePtr,
    h_state: DevicePtr,
    ns: usize,
    r0: usize,
    rows: usize,
    conv_dim: i32,
    strip_flags: i32,
    stream: u64,
) -> Result<()> {
    let ptr = |p: DevicePtr| p.0 as usize as *mut core::ffi::c_void;
    // SAFETY: device buffers are live, aligned, and sized for `ns <= max_strip_rows`
    // by the caller; the bridge only enqueues GPU work on `stream`.
    let rc = unsafe {
        ffi::atlas_gdn_fi_sm121a_prefill_f16_strip(
            ptr(qkv_strip),
            ptr(gate_in),
            ptr(beta_in),
            ptr(gdn_out_strip),
            ptr(h_state),
            ns as i32,
            conv_dim,
            strip_flags,
            stream as usize as *mut core::ffi::c_void,
        )
    };
    ensure!(
        rc == 0,
        "{SELECTOR}: atlas_gdn_fi_sm121a_prefill_f16_strip failed rc={rc} r0={r0} ns={ns} rows={rows}"
    );
    Ok(())
}

#[cfg(not(atlas_fi_gdn))]
#[allow(clippy::too_many_arguments)]
fn invoke_prefill_f16_strip(
    _qkv_strip: DevicePtr,
    _gate_in: DevicePtr,
    _beta_in: DevicePtr,
    _gdn_out_strip: DevicePtr,
    _h_state: DevicePtr,
    _ns: usize,
    _r0: usize,
    _rows: usize,
    _conv_dim: i32,
    _strip_flags: i32,
    _stream: u64,
) -> Result<()> {
    anyhow::bail!("{SELECTOR}: FlashInfer GDN AOT bridge is not available in this build")
}

/// Load-time sizing of the bridge the prefill calls (the norm-fp8 FP16 export
/// under `ATLAS_QWEN4_PREFILL_GDN_NORM_FP8=3` or `=2` with `nf8_strip` active,
/// else the selected dtype's), so no request allocates, grows or frees its
/// buffers (`ATLAS_PREFILL_SCRATCH_INIT` applies). No-op when the FI route is
/// not selected.
pub(crate) fn reserve_bridge(max_rows: usize) -> Result<()> {
    let Some(dtype) = dtype() else {
        return Ok(());
    };
    let (bridge, reserve_rows) = invoke_reserve(dtype, max_rows)?;
    tracing::info!(
        "QWEN4_PREFILL_SCRATCH_RESERVED site=fi_gdn bridge={bridge} rows={reserve_rows} bytes={} never_freed=1",
        bridge_bytes(reserve_rows)
    );
    Ok(())
}

#[derive(Clone, Copy)]
struct Kernels {
    conv_16: KernelHandle,
    conv_commit: KernelHandle,
    gated_norm_16in: KernelHandle,
}

fn kernels(gpu: &dyn GpuBackend, dtype: FiDtype) -> Result<Kernels> {
    type Slot = std::sync::OnceLock<Result<Kernels, String>>;
    static BF16: Slot = Slot::new();
    static F16: Slot = Slot::new();
    let slot = match dtype {
        FiDtype::Bf16 => &BF16,
        FiDtype::F16 => &F16,
    };
    slot.get_or_init(|| {
        let load = |module: &str, name: &str| gpu.kernel(module, name).map_err(|e| e.to_string());
        let (conv, norm) = match dtype {
            FiDtype::Bf16 => (
                "qwen4_conv1d_l2norm_bf16_prefill_parallel",
                "qwen4_gated_rms_norm_bf16in_multi_seq",
            ),
            FiDtype::F16 => (
                "qwen4_conv1d_l2norm_f16_prefill_parallel",
                "qwen4_gated_rms_norm_f16in_multi_seq",
            ),
        };
        Ok(Kernels {
            conv_16: load("qwen4_gdn_fi_glue", conv)?,
            conv_commit: load(
                "causal_conv1d_prefill_parallel",
                "causal_conv1d_prefill_state_commit",
            )?,
            gated_norm_16in: load("qwen4_gdn_fi_glue", norm)?,
        })
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: kernel unavailable: {e}"))
}

/// Active opt-in conv/SiLU/L2 prefill kernel variant (`v2` vs `v3`).
#[derive(Clone, Copy)]
enum ConvV2Variant {
    V2,
    V3,
}

impl ConvV2Variant {
    const fn selector(self) -> &'static str {
        match self {
            Self::V2 => super::qwen4_gdn_conv_v2::SELECTOR,
            Self::V3 => super::qwen4_gdn_conv_v2::V3_SELECTOR,
        }
    }
}

/// `ATLAS_QWEN4_PREFILL_GDN_CONV_V2` / `ATLAS_QWEN4_PREFILL_GDN_CONV_V3` (default off): the v2 or
/// v3 conv/SiLU/L2 entry of the selected dtype alongside its active variant, loaded only when the
/// selector is on (`None` = v1, the shipping kernel).
fn conv_v2_kernel(
    gpu: &dyn GpuBackend,
    dtype: FiDtype,
) -> Result<Option<(KernelHandle, ConvV2Variant)>> {
    use super::qwen4_gdn_conv_v2 as v2;
    static V3_SEL: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    let v3_on = V3_SEL
        .get_or_init(|| v2::parse_v3(std::env::var(v2::V3_SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)?;
    if v3_on {
        type Slot = std::sync::OnceLock<Result<KernelHandle, String>>;
        static BF16_V3: Slot = Slot::new();
        static F16_V3: Slot = Slot::new();
        let (slot, name) = match dtype {
            FiDtype::Bf16 => (&BF16_V3, "qwen4_conv1d_l2norm_bf16_prefill_v3"),
            FiDtype::F16 => (&F16_V3, "qwen4_conv1d_l2norm_f16_prefill_v3"),
        };
        return slot
            .get_or_init(|| {
                gpu.kernel("qwen4_gdn_fi_glue", name)
                    .map_err(|e| e.to_string())
            })
            .clone()
            .map(|handle| Some((handle, ConvV2Variant::V3)))
            .map_err(|e| anyhow::anyhow!("{}: kernel {name} unavailable: {e}", v2::V3_SELECTOR));
    }
    static SEL: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    let on = SEL
        .get_or_init(|| v2::parse(std::env::var(v2::SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)?;
    if !on {
        return Ok(None);
    }
    type Slot = std::sync::OnceLock<Result<KernelHandle, String>>;
    static BF16: Slot = Slot::new();
    static F16: Slot = Slot::new();
    let (slot, name) = match dtype {
        FiDtype::Bf16 => (&BF16, "qwen4_conv1d_l2norm_bf16_prefill_v2"),
        FiDtype::F16 => (&F16, "qwen4_conv1d_l2norm_f16_prefill_v2"),
    };
    slot.get_or_init(|| {
        gpu.kernel("qwen4_gdn_fi_glue", name)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map(|handle| Some((handle, ConvV2Variant::V2)))
    .map_err(|e| anyhow::anyhow!("{}: kernel {name} unavailable: {e}", v2::SELECTOR))
}

fn conv_v2_commit_kernel(
    gpu: &dyn GpuBackend,
    dtype: FiDtype,
    variant: ConvV2Variant,
) -> Result<KernelHandle> {
    use super::qwen4_gdn_conv_v2 as v2;
    type Slot = std::sync::OnceLock<Result<KernelHandle, String>>;
    if matches!(variant, ConvV2Variant::V3) {
        static BF16_V3_COMMIT: Slot = Slot::new();
        static F16_V3_COMMIT: Slot = Slot::new();
        let (slot, name) = match dtype {
            FiDtype::Bf16 => (
                &BF16_V3_COMMIT,
                "qwen4_conv1d_l2norm_bf16_prefill_v3_commit",
            ),
            FiDtype::F16 => (&F16_V3_COMMIT, "qwen4_conv1d_l2norm_f16_prefill_v3_commit"),
        };
        return slot
            .get_or_init(|| {
                gpu.kernel("qwen4_gdn_fi_glue", name)
                    .map_err(|e| e.to_string())
            })
            .clone()
            .map_err(|e| anyhow::anyhow!("{}: kernel {name} unavailable: {e}", v2::V3_SELECTOR));
    }
    static BF16: Slot = Slot::new();
    static F16: Slot = Slot::new();
    let (slot, name) = match dtype {
        FiDtype::Bf16 => (&BF16, "qwen4_conv1d_l2norm_bf16_prefill_v2_commit"),
        FiDtype::F16 => (&F16, "qwen4_conv1d_l2norm_f16_prefill_v2_commit"),
    };
    slot.get_or_init(|| {
        gpu.kernel("qwen4_gdn_fi_glue", name)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{}: kernel {name} unavailable: {e}", v2::SELECTOR))
}

impl Qwen3SsmLayer {
    /// Conv + L2 + chunked GDN + gated RMS norm for `rows` tokens of one
    /// chunk. Reads the packed QKVZ rows (`deinterleaved`, stride QKVZ) and
    /// the interleaved gate/beta rows (`gates`, stride GATES); carries the
    /// conv and H state; writes `[rows, VALUE_DIM]` BF16 into `normed`.
    /// `scratch` is the FP32 conv arena, reused for 16-bit Q|K|V and the 16-bit
    /// GDN output.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen4_gdn_fi_core(
        &self,
        state: &mut SsmLayerState,
        deinterleaved: DevicePtr,
        gates: DevicePtr,
        scratch: DevicePtr,
        normed: Option<DevicePtr>,
        nf8: Option<(DevicePtr, DevicePtr)>,
        rows: usize,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let dtype = dtype().ok_or_else(|| anyhow::anyhow!("{SELECTOR} is not selected"))?;
        let k = kernels(ctx.gpu, dtype)?;
        let s = super::qwen4_prefill_exact_plan::SsmShape::from_heads(
            ctx.config.linear_num_key_heads,
            ctx.config.linear_num_value_heads,
        );
        let qkv_bytes = rows * s.conv_dim * 2;
        let out_bytes = if nf8.is_some() {
            0
        } else {
            rows * s.value_dim * 2
        };
        ensure!(
            rows > 0 && qkv_bytes + out_bytes <= ctx.buffers.sizes().ssm_conv_out_f32,
            "{SELECTOR}: 16-bit Q/K/V + GDN output exceed the conv arena"
        );
        ensure!(
            !state.h_is_f16 && !state.h_state.is_null() && !state.conv_state.is_null(),
            "{SELECTOR} requires live FP32 recurrent state"
        );
        if let Some((out8, scale_out)) = nf8
            && let Some((strip_rows, max_strip_rows)) = self.try_qwen4_gdn_fi_nf8_strip_core(
                state,
                s,
                deinterleaved,
                gates,
                scratch,
                out8,
                scale_out,
                rows,
                eps,
                ctx,
                stream,
            )?
        {
            spark_runtime::cublaslt::stage_timer::mark("gdn_fi", stream);
            static NF8_CORE_STRIP_LOGGED: std::sync::Once = std::sync::Once::new();
            NF8_CORE_STRIP_LOGGED.call_once(|| {
                tracing::info!(
                    "SSM_PREFILL_GDN_FI_ENGAGED selector={SELECTOR} rows={rows} kernel=flashinfer-gdn-sm121a-f16-nf8-strip strip_rows={strip_rows} max_strip_rows={max_strip_rows} dtype={} non_bit_exact=16bit_qkv_and_output",
                    dtype.name()
                );
            });
            return Ok(());
        }
        let qkv = scratch;
        let gdn_out = if nf8.is_some() {
            qkv
        } else {
            scratch.offset(qkv_bytes)
        };
        let rows_u32 = u32::try_from(rows)?;
        let mut conv_committed = false;

        // ATLAS_QWEN4_PREFILL_GDN_CONV_V2 / ATLAS_QWEN4_PREFILL_GDN_CONV_V3:
        // the byte-identical v2/v3 conv (8 channels per lane, 32-token strips)
        // when admitted; v1 otherwise.
        let conv_launch = match conv_v2_kernel(ctx.gpu, dtype)? {
            Some((v2_kernel, variant)) => {
                use super::qwen4_gdn_conv_v2 as v2;
                let refused = v2::refusal(
                    rows_u32,
                    s.conv_dim as u32,
                    (s.key_dim * 2) as u32,
                    128,
                    4,
                    s.qkvz as u32,
                    s.conv_dim as u32,
                    [deinterleaved.0, qkv.0, self.ssm.conv1d.weight.0],
                );
                match refused {
                    None => {
                        static ENGAGED: std::sync::Once = std::sync::Once::new();
                        ENGAGED.call_once(|| {
                            tracing::info!(
                                "QWEN4_PREFILL_GDN_CONV_V2_ENGAGED selector={} dtype={} rows={rows} tokens_per_warp={} byte_identical_to=qwen4_conv1d_l2norm_{}_prefill_parallel",
                                variant.selector(),
                                dtype.name(),
                                v2::TOKENS,
                                dtype.name()
                            );
                        });
                        let kernel = if rows_u32 >= 4 && state.conv_state.0.is_multiple_of(16) {
                            conv_committed = true;
                            conv_v2_commit_kernel(ctx.gpu, dtype, variant)?
                        } else {
                            v2_kernel
                        };
                        KernelLaunch::new(ctx.gpu, kernel)
                            .grid(v2::grid(s.conv_dim as u32, rows_u32))
                            .block([v2::BLOCK_THREADS, 1, 1])
                    }
                    Some(reason) => {
                        static REFUSED: std::sync::Once = std::sync::Once::new();
                        REFUSED.call_once(|| {
                            tracing::warn!(
                                "QWEN4_PREFILL_GDN_CONV_V2_REFUSED selector={} reason={reason} rows={rows} fallback=v1",
                                variant.selector()
                            );
                        });
                        KernelLaunch::new(ctx.gpu, k.conv_16)
                            .grid([div_ceil(s.conv_dim as u32, 256), rows_u32, 1])
                            .block([256, 1, 1])
                    }
                }
            }
            None => KernelLaunch::new(ctx.gpu, k.conv_16)
                .grid([div_ceil(s.conv_dim as u32, 256), rows_u32, 1])
                .block([256, 1, 1]),
        };
        conv_launch
            .arg_ptr(state.conv_state)
            .arg_ptr(deinterleaved)
            .arg_ptr(self.ssm.conv1d.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(qkv)
            .arg_u32(rows_u32)
            .arg_u32(s.conv_dim as u32)
            .arg_u32(4)
            .arg_u32((s.key_dim * 2) as u32)
            .arg_u32(128)
            .arg_f32(1e-6)
            .arg_u32(s.qkvz as u32)
            .arg_u32(s.conv_dim as u32)
            .launch(stream)?;
        // Commit the shift register only after every conv reader has run.
        if !conv_committed {
            KernelLaunch::new(ctx.gpu, k.conv_commit)
                .grid([div_ceil(s.conv_dim as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(state.conv_state)
                .arg_ptr(deinterleaved)
                .arg_u32(rows_u32)
                .arg_u32(s.conv_dim as u32)
                .arg_u32(4)
                .arg_u32(s.qkvz as u32)
                .launch(stream)?;
        }

        // Stage split (ATLAS_QWEN4_PREFILL_STAGES=1 only): gdn_conv + gdn_fi +
        // the caller's gdn_conv_norm = the former gdn_conv_norm.
        spark_runtime::cublaslt::stage_timer::mark("gdn_conv", stream);

        let conv_dim_arg = if super::qwen4_ba_gemm::uses_planar_fi_gates(rows) {
            -(s.conv_dim as i32)
        } else {
            s.conv_dim as i32
        };
        invoke_prefill(
            dtype,
            qkv,
            gates,
            gdn_out,
            state.h_state,
            rows,
            conv_dim_arg,
            deinterleaved.offset(s.conv_dim * 2),
            s.qkvz as i64,
            self.ssm.norm.weight,
            nf8,
            eps,
            stream,
        )?;
        spark_runtime::cublaslt::stage_timer::mark("gdn_fi", stream);

        // `None`: the caller runs the gated norm fused with the out-projection
        // FP8 quantization (qwen4_gdn_norm_fp8.rs) on the GDN output left in
        // `scratch` after the Q|K|V rows.
        if let (Some(normed), None) = (normed, nf8) {
            let fuse_amax = super::qwen4_prefill_exact::check_selected().is_ok_and(|c| !c)
                && crate::layers::qwen4_fast_proj::selection().is_ok_and(|s| s.ssm)
                && crate::layers::qwen4_fast_proj::fp8_selection().is_ok_and(|s| s.ssm)
                && crate::layers::qwen4_fast_proj::fp8_tensor_scales()
                && crate::layers::qwen4_fp8_act_amax::gdn_norm_selected()?;
            if fuse_amax {
                super::qwen4_gdn_norm_fp8::launch_gated_norm_16in_amax(
                    ctx.gpu,
                    dtype,
                    s,
                    gdn_out,
                    deinterleaved.offset(s.conv_dim * 2),
                    self.ssm.norm.weight,
                    normed,
                    rows,
                    eps,
                    stream,
                )?;
            } else {
                KernelLaunch::new(ctx.gpu, k.gated_norm_16in)
                    .grid([s.nv as u32, rows_u32, 1])
                    .block([128, 1, 1])
                    .arg_ptr(gdn_out)
                    .arg_ptr(deinterleaved.offset(s.conv_dim * 2))
                    .arg_ptr(self.ssm.norm.weight)
                    .arg_ptr(normed)
                    .arg_u32(128)
                    .arg_f32(eps)
                    .arg_u32(s.value_dim as u32)
                    .arg_u32(s.qkvz as u32)
                    .arg_u32(s.value_dim as u32)
                    .launch(stream)?;
            }
        }
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "SSM_PREFILL_GDN_FI_ENGAGED selector={SELECTOR} rows={rows} kernel=flashinfer-gdn-sm121a-aot dtype={} non_bit_exact=16bit_qkv_and_output",
                dtype.name()
            );
        });
        Ok(())
    }

    /// Shared strip-mined `conv_v4_strip -> atlas_gdn_fi_sm121a_prefill_f16_nf8_strip`
    /// loop (reverts to `conv_v3_strip` when `ATLAS_QWEN4_PREFILL_GDN_CONV_V4_STRIP=0`)
    /// used by `ATLAS_QWEN4_PREFILL_GDN_NORM_FP8=2` and `=3` when opted in via
    /// `ATLAS_QWEN4_PREFILL_GDN_NF8_STRIP=1` (default OFF). Fuses gated RMS norm +
    /// static FP8 E4M3 quantization into the FlashInfer store warp, eliminating
    /// `gdn_out_strip` and all `quant_v2_strip` launches while keeping the active
    /// strip working set (~16.2 MiB @ 512 rows) inside GB10's verified 24.0 MiB L2 cache.
    /// Supports both TP=1 (`conv_dim=10240`) and TP=2 (`conv_dim=5120`).
    #[allow(clippy::too_many_arguments)]
    fn try_qwen4_gdn_fi_nf8_strip_core(
        &self,
        state: &mut SsmLayerState,
        s: super::qwen4_prefill_exact_plan::SsmShape,
        deinterleaved: DevicePtr,
        gates: DevicePtr,
        scratch: DevicePtr,
        a8: DevicePtr,
        a_scale: DevicePtr,
        rows: usize,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(usize, usize)>> {
        use super::qwen4_gdn_conv_v2 as v2;

        if dtype() != Some(FiDtype::F16)
            || !nf8_strip_enabled()
            || rows <= super::qwen4_gdn_fi_policy::SHORT_FALLBACK_MAX_ROWS
            || std::env::var_os("ATLAS_QWEN4_GDN_FI_DUMP_DIR").is_some()
        {
            return Ok(None);
        }
        let Some(conv_strip_k) = conv_strip_kernel(ctx.gpu, FiDtype::F16)? else {
            return Ok(None);
        };
        let rows_u32 = u32::try_from(rows)?;
        if v2::refusal(
            rows_u32,
            s.conv_dim as u32,
            (s.key_dim * 2) as u32,
            128,
            4,
            s.qkvz as u32,
            s.conv_dim as u32,
            [deinterleaved.0, scratch.0, self.ssm.conv1d.weight.0],
        )
        .is_some()
        {
            return Ok(None);
        }
        if state.h_is_f16
            || state.h_state.is_null()
            || state.conv_state.is_null()
            || !state.conv_state.0.is_multiple_of(16)
            || !scratch.0.is_multiple_of(128)
            || !deinterleaved.0.is_multiple_of(128)
            || !self.ssm.norm.weight.0.is_multiple_of(16)
            || !a8.0.is_multiple_of(16)
            || a_scale.is_null()
        {
            return Ok(None);
        }
        let (strip_rows, max_strip_rows) = strip_rows_config();
        let max_active_strip = rows.min(max_strip_rows);
        let strip_arena_bytes = max_active_strip * s.conv_dim * 2;
        if strip_arena_bytes > ctx.buffers.sizes().ssm_conv_out_f32 {
            return Ok(None);
        }

        let l2_discard = super::qwen4_gdn_norm_fp8::l2_discard_policy()?;
        let conv_discard_flags = conv_v4_strip_discard_flags(l2_discard, deinterleaved, s.qkvz);
        let qkv_strip = scratch;
        let z_base = deinterleaved.offset(s.conv_dim * 2);
        let planar = super::qwen4_ba_gemm::uses_planar_fi_gates(rows);
        let conv_dim_i32 = i32::try_from(s.conv_dim)?;
        let z_stride_i64 = i64::try_from(s.qkvz)?;

        let mut r0 = 0usize;
        while r0 < rows {
            let rem = rows - r0;
            let ns = if rem <= max_strip_rows {
                rem
            } else {
                strip_rows
            };
            let ns_u32 = u32::try_from(ns)?;
            let first_strip = r0 == 0;
            let last_strip = r0 + ns == rows;

            conv_strip_k.launch(
                ctx.gpu,
                StripConvLaunch {
                    conv_state: state.conv_state,
                    deinterleaved,
                    weight: self.ssm.conv1d.weight,
                    qkv_strip,
                    strip_rows: ns_u32,
                    row_offset: r0 as u32,
                    total_rows: rows_u32,
                    conv_dim: s.conv_dim as u32,
                    qk_channels: (s.key_dim * 2) as u32,
                    input_stride: s.qkvz as u32,
                    discard_flags: conv_discard_flags,
                },
                stream,
            )?;

            let gate_in = if planar {
                gates.offset(r0 * s.nv * 4)
            } else {
                gates.offset(r0 * s.gates * 4)
            };
            let beta_in = if planar {
                gates.offset((rows * s.nv + r0 * s.nv) * 4)
            } else {
                DevicePtr::NULL
            };
            let strip_flags = i32::from(first_strip) | (i32::from(last_strip) << 1);
            invoke_prefill_f16_nf8_strip(
                qkv_strip,
                gate_in,
                beta_in,
                state.h_state,
                ns,
                r0,
                rows,
                conv_dim_i32,
                strip_flags,
                z_base.offset(r0 * s.qkvz * 2),
                z_stride_i64,
                self.ssm.norm.weight,
                a8.offset(r0 * s.value_dim),
                a_scale,
                eps,
                stream,
            )?;

            r0 += ns;
        }
        Ok(Some((strip_rows, max_strip_rows)))
    }

    /// Strip-mined L2-resident SSM core followed by the FP8 tensor-scale `out_proj` GEMM.
    /// Defaults to `conv_v4_strip -> fi_gdn_f16_strip -> quant_v2_strip` at `512`-row
    /// strips on GB10 (`8 * 64` exact GDN chunks, ~22.2 MiB inside GB10's verified 24.0 MiB
    /// L2 cache; `ATLAS_QWEN4_PREFILL_GDN_CONV_V4_STRIP=0` reverts to `conv_v3_strip`);
    /// opts into fused `conv_v4_strip -> fi_gdn_f16_nf8_strip` (~16.2 MiB L2 working set,
    /// zero `gdn_out_strip` / `quant_v2_strip`) when `ATLAS_QWEN4_PREFILL_GDN_NF8_STRIP=1`
    /// (default OFF). Strip height honors `ATLAS_QWEN4_PREFILL_GDN_FI_STRIP_ROWS`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_qwen4_gdn_fi_strip_norm_fp8_out_proj(
        &self,
        state: &mut SsmLayerState,
        deinterleaved: DevicePtr,
        gates: DevicePtr,
        scratch: DevicePtr,
        output: DevicePtr,
        rows: usize,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        use super::qwen4_gdn_conv_v2 as v2;
        use super::qwen4_prefill_exact_plan::Plan;
        use crate::layers::qwen4_fast_proj;

        if dtype() != Some(FiDtype::F16)
            || std::env::var_os("ATLAS_QWEN4_GDN_FI_DUMP_DIR").is_some()
        {
            return Ok(false);
        }

        // Preferred path: fused `f16_nf8_strip` (eliminates `gdn_out_strip` and all
        // `quant_v2_strip` launches; supports TP=1/TP=2 and all rows > 64).
        if nf8_strip_enabled()
            && conv_v3_selected()
            && rows > super::qwen4_gdn_fi_policy::SHORT_FALLBACK_MAX_ROWS
        {
            let s = super::qwen4_prefill_exact_plan::SsmShape::from_heads(
                ctx.config.linear_num_key_heads,
                ctx.config.linear_num_value_heads,
            );
            let weight = &self.ssm.out_proj;
            ensure!(
                !weight.weight.is_null()
                    && !weight.weight_scale.is_null()
                    && weight.weight_scale_2.is_finite(),
                "{}: invalid NVFP4 out_proj weight",
                super::qwen4_gdn_norm_fp8::SELECTOR
            );
            let (n, kd) = (Plan::H, s.value_dim);
            let (w8, ws, a8, _) = qwen4_fast_proj::fp8_scratch(ctx.gpu, n, rows, kd)?;
            let a_scale = ws.offset(12);
            if let Some((strip_rows, max_strip_rows)) = self.try_qwen4_gdn_fi_nf8_strip_core(
                state,
                s,
                deinterleaved,
                gates,
                scratch,
                a8,
                a_scale,
                rows,
                eps,
                ctx,
                stream,
            )? {
                if !qwen4_fast_proj::has_cached_weight_tensor(weight, n, kd, None) {
                    ctx.gpu.memset_async(ws, 0, 4, stream)?;
                }
                spark_runtime::cublaslt::stage_timer::mark("gdn_fi", stream);
                spark_runtime::cublaslt::stage_timer::mark("gdn_conv_norm", stream);

                let tk = qwen4_fast_proj::fp8_tensor_kernels(ctx.gpu)?;
                let m_gemm = qwen4_fast_proj::resolve_fp8_pad_m(rows, n);
                if m_gemm > rows {
                    ctx.gpu
                        .memset_async(a8.offset(rows * kd), 0, (m_gemm - rows) * kd, stream)?;
                }
                qwen4_fast_proj::fp8_weight_tensor_gemm(
                    ctx.gpu, &tk, weight, output, m_gemm, n, kd, n, None, w8, ws, a8, true, stream,
                )?;

                static NF8_STRIP_LOGGED: std::sync::Once = std::sync::Once::new();
                NF8_STRIP_LOGGED.call_once(|| {
                    tracing::info!(
                        "SSM_PREFILL_GDN_FI_STRIP_ENGAGED selector={SELECTOR} mode=nf8_strip rows={rows} strip_rows={strip_rows} max_strip_rows={max_strip_rows} m_gemm={m_gemm} dtype=f16"
                    );
                });
                return Ok(true);
            }
        }

        let want_overlap = strip_overlap_enabled()
            && !ctx.profile
            && !ctx.graph_capture
            && spark_runtime::gpu_capture_region::active_capture_region().is_none()
            && !ctx.gpu.stream_is_capturing(stream);
        let (strip_rows, max_strip_rows) = strip_rows_config_for_mode(want_overlap);

        if ctx.config.tp_world_size > 1
            || rows <= max_strip_rows
            || !super::qwen4_gdn_norm_fp8::v2_selected()?
        {
            return Ok(false);
        }
        let Some(conv_strip_k) = conv_strip_kernel(ctx.gpu, FiDtype::F16)? else {
            return Ok(false);
        };
        let rows_u32 = u32::try_from(rows)?;
        if v2::refusal(
            rows_u32,
            Plan::CONV_DIM as u32,
            (Plan::KEY_DIM * 2) as u32,
            128,
            4,
            Plan::QKVZ as u32,
            Plan::CONV_DIM as u32,
            [deinterleaved.0, scratch.0, self.ssm.conv1d.weight.0],
        )
        .is_some()
        {
            return Ok(false);
        }
        if state.h_is_f16
            || state.h_state.is_null()
            || state.conv_state.is_null()
            || !state.conv_state.0.is_multiple_of(16)
            || !scratch.0.is_multiple_of(128)
            || !deinterleaved.0.is_multiple_of(128)
            || !self.ssm.norm.weight.0.is_multiple_of(16)
        {
            return Ok(false);
        }
        let gdn_out_off = gdn_out_strip_offset(max_strip_rows, Plan::CONV_DIM);
        let strip_arena_bytes = gdn_out_off + max_strip_rows * Plan::VALUE_DIM * 2;
        if strip_arena_bytes > ctx.buffers.sizes().ssm_conv_out_f32 {
            return Ok(false);
        }

        let weight = &self.ssm.out_proj;
        ensure!(
            !weight.weight.is_null()
                && !weight.weight_scale.is_null()
                && weight.weight_scale_2.is_finite(),
            "{}: invalid NVFP4 out_proj weight",
            super::qwen4_gdn_norm_fp8::SELECTOR
        );
        let (n, kd) = (Plan::H, Plan::VALUE_DIM);
        let l2_discard = super::qwen4_gdn_norm_fp8::l2_discard_policy()?;
        let conv_discard_flags = conv_v4_strip_discard_flags(l2_discard, deinterleaved, Plan::QKVZ);

        let target = if let Some(nvfp4_target) = try_prepare_strip_nvfp4_target(
            ctx,
            weight,
            rows,
            n,
            kd,
            deinterleaved,
            gates,
            scratch,
            output,
            stream,
        )? {
            nvfp4_target
        } else {
            let (w8, ws, a8, _) = qwen4_fast_proj::fp8_scratch(ctx.gpu, n, rows, kd)?;
            if !a8.0.is_multiple_of(8) {
                return Ok(false);
            }
            let tk = qwen4_fast_proj::fp8_tensor_kernels(ctx.gpu)?;
            let quant_strip_k =
                super::qwen4_gdn_norm_fp8::quant_v2_strip_kernel(ctx.gpu, FiDtype::F16)?;
            if !qwen4_fast_proj::has_cached_weight_tensor(weight, n, kd, None) {
                ctx.gpu.memset_async(ws, 0, 8, stream)?;
            }
            StripOutTarget::Fp8 {
                quant_strip_k,
                tk,
                w8,
                ws,
                a8,
                a_scale: ws.offset(12),
            }
        };

        let strips = build_strip_ranges(rows, strip_rows, max_strip_rows);
        let z_base = deinterleaved.offset(Plan::CONV_DIM * 2);
        let planar = super::qwen4_ba_gemm::uses_planar_fi_gates(rows);

        let ping_pong = if want_overlap && strips.len() >= 2 {
            compute_ping_pong_layout(
                scratch,
                max_strip_rows,
                Plan::CONV_DIM,
                kd,
                ctx.buffers.sizes().ssm_conv_out_f32,
            )
            .and_then(|layout| {
                get_or_init_strip_overlap_runtime(ctx.gpu, stream)
                    .ok()
                    .flatten()
                    .map(|rt| (layout, rt))
            })
        } else {
            None
        };

        if let Some((layout, ov)) = ping_pong {
            let (r0_0, ns_0) = strips[0];
            conv_strip_k.launch(
                ctx.gpu,
                StripConvLaunch {
                    conv_state: state.conv_state,
                    deinterleaved,
                    weight: self.ssm.conv1d.weight,
                    qkv_strip: layout.slot_qkv[0],
                    strip_rows: ns_0 as u32,
                    row_offset: r0_0 as u32,
                    total_rows: rows_u32,
                    conv_dim: Plan::CONV_DIM as u32,
                    qk_channels: (Plan::KEY_DIM * 2) as u32,
                    input_stride: Plan::QKVZ as u32,
                    discard_flags: conv_discard_flags,
                },
                stream,
            )?;
            // `start_event` orders `side_stream` strictly after `deinterleaved`, `ws` memset,
            // and `conv(0)`'s read of `conv_state` (so even a 2-strip final `conv(1)` writing
            // `conv_state` cannot race with strip 0).
            ctx.gpu.record_event(ov.start_event, stream)?;
            ctx.gpu.stream_wait_event(ov.side_stream, ov.start_event)?;
            let mut join_guard =
                StripOverlapJoinGuard::new(ctx.gpu, stream, ov.side_stream, ov.side_done);

            let num_strips = strips.len();
            for s_idx in 0..num_strips {
                let cur = s_idx & 1;
                let (r0, ns) = strips[s_idx];
                let first_strip = s_idx == 0;
                let last_strip = s_idx + 1 == num_strips;

                if s_idx > 0 {
                    ctx.gpu.stream_wait_event(stream, ov.conv_done[cur])?;
                }
                let gate_in = if planar {
                    gates.offset(r0 * 48 * 4)
                } else {
                    gates.offset(r0 * 96 * 4)
                };
                let beta_in = if planar {
                    gates.offset((rows * 48 + r0 * 48) * 4)
                } else {
                    DevicePtr::NULL
                };
                let strip_flags = i32::from(first_strip) | (i32::from(last_strip) << 1);
                invoke_prefill_f16_strip(
                    layout.slot_qkv[cur],
                    gate_in,
                    beta_in,
                    layout.slot_out[cur],
                    state.h_state,
                    ns,
                    r0,
                    rows,
                    Plan::CONV_DIM as i32,
                    strip_flags,
                    stream,
                )?;
                if s_idx + 1 < num_strips {
                    ctx.gpu.record_event(ov.fi_done[cur], stream)?;
                }

                // While 48-CTA `fi_gdn(s_idx)` runs on `stream`, execute `quant(s_idx - 1)`
                // followed by `conv(s_idx + 1)` on `ov.side_stream` across the 140 idle SMs
                // using the opposite ping-pong buffer slot `(s_idx - 1) & 1 == (s_idx + 1) & 1`.
                if s_idx > 0 {
                    let prev = (s_idx - 1) & 1;
                    let (prev_r0, prev_ns) = strips[s_idx - 1];
                    ctx.gpu
                        .stream_wait_event(ov.side_stream, ov.fi_done[prev])?;
                    join_guard.mark_launched();
                    launch_strip_quant(
                        ctx.gpu,
                        &target,
                        StripQuantLaunch {
                            qkv_strip: layout.slot_qkv[prev],
                            gdn_out_strip: layout.slot_out[prev],
                            z_row: z_base.offset(prev_r0 * Plan::QKVZ * 2),
                            norm_weight: self.ssm.norm.weight,
                            row_offset: prev_r0,
                            strip_rows: prev_ns,
                            total_rows: rows,
                            value_dim: kd,
                            eps,
                            l2_discard,
                            last_strip: false,
                            allow_early_scale_write: false,
                        },
                        ov.side_stream,
                    )?;
                }
                if s_idx + 1 < num_strips {
                    let next = (s_idx + 1) & 1;
                    let (next_r0, next_ns) = strips[s_idx + 1];
                    join_guard.mark_launched();
                    conv_strip_k.launch(
                        ctx.gpu,
                        StripConvLaunch {
                            conv_state: state.conv_state,
                            deinterleaved,
                            weight: self.ssm.conv1d.weight,
                            qkv_strip: layout.slot_qkv[next],
                            strip_rows: next_ns as u32,
                            row_offset: next_r0 as u32,
                            total_rows: rows_u32,
                            conv_dim: Plan::CONV_DIM as u32,
                            qk_channels: (Plan::KEY_DIM * 2) as u32,
                            input_stride: Plan::QKVZ as u32,
                            discard_flags: conv_discard_flags,
                        },
                        ov.side_stream,
                    )?;
                    ctx.gpu.record_event(ov.conv_done[next], ov.side_stream)?;
                }
            }

            // Join `ov.side_stream` (`quant(S-2)`) back onto `stream` before the final strip
            // quant and `ssm.out_proj` GEMM.
            join_guard.join()?;

            let last_idx = num_strips - 1;
            let (last_r0, last_ns) = strips[last_idx];
            launch_strip_quant(
                ctx.gpu,
                &target,
                StripQuantLaunch {
                    qkv_strip: layout.slot_qkv[last_idx & 1],
                    gdn_out_strip: layout.slot_out[last_idx & 1],
                    z_row: z_base.offset(last_r0 * Plan::QKVZ * 2),
                    norm_weight: self.ssm.norm.weight,
                    row_offset: last_r0,
                    strip_rows: last_ns,
                    total_rows: rows,
                    value_dim: kd,
                    eps,
                    l2_discard,
                    last_strip: true,
                    allow_early_scale_write: false,
                },
                stream,
            )?;

            static OVERLAP_LOGGED: std::sync::Once = std::sync::Once::new();
            OVERLAP_LOGGED.call_once(|| {
                tracing::info!(
                    "GDN_PREFILL_STRIP_OVERLAP_ENGAGED selector={STRIP_OVERLAP_SELECTOR} rows={rows} strips={num_strips} strip_rows={strip_rows} max_strip_rows={max_strip_rows} conv={} out_proj={} slot_bytes={}",
                    conv_strip_k.variant_name(),
                    target.label(),
                    layout.total_bytes / 2,
                );
            });
        } else {
            let qkv_strip = scratch;
            let gdn_out_strip = scratch.offset(gdn_out_off);
            let num_strips = strips.len();
            for (s_idx, &(r0, ns)) in strips.iter().enumerate() {
                let ns_u32 = ns as u32;
                let first_strip = s_idx == 0;
                let last_strip = s_idx + 1 == num_strips;

                conv_strip_k.launch(
                    ctx.gpu,
                    StripConvLaunch {
                        conv_state: state.conv_state,
                        deinterleaved,
                        weight: self.ssm.conv1d.weight,
                        qkv_strip,
                        strip_rows: ns_u32,
                        row_offset: r0 as u32,
                        total_rows: rows_u32,
                        conv_dim: Plan::CONV_DIM as u32,
                        qk_channels: (Plan::KEY_DIM * 2) as u32,
                        input_stride: Plan::QKVZ as u32,
                        discard_flags: conv_discard_flags,
                    },
                    stream,
                )?;

                let gate_in = if planar {
                    gates.offset(r0 * 48 * 4)
                } else {
                    gates.offset(r0 * 96 * 4)
                };
                let beta_in = if planar {
                    gates.offset((rows * 48 + r0 * 48) * 4)
                } else {
                    DevicePtr::NULL
                };
                let strip_flags = i32::from(first_strip) | (i32::from(last_strip) << 1);
                invoke_prefill_f16_strip(
                    qkv_strip,
                    gate_in,
                    beta_in,
                    gdn_out_strip,
                    state.h_state,
                    ns,
                    r0,
                    rows,
                    Plan::CONV_DIM as i32,
                    strip_flags,
                    stream,
                )?;

                launch_strip_quant(
                    ctx.gpu,
                    &target,
                    StripQuantLaunch {
                        qkv_strip,
                        gdn_out_strip,
                        z_row: z_base.offset(r0 * Plan::QKVZ * 2),
                        norm_weight: self.ssm.norm.weight,
                        row_offset: r0,
                        strip_rows: ns,
                        total_rows: rows,
                        value_dim: kd,
                        eps,
                        l2_discard,
                        last_strip,
                        allow_early_scale_write: true,
                    },
                    stream,
                )?;
            }
        }

        spark_runtime::cublaslt::stage_timer::mark("gdn_fi", stream);
        spark_runtime::cublaslt::stage_timer::mark("gdn_conv_norm", stream);

        let m_gemm = match target {
            StripOutTarget::Fp8 { tk, w8, ws, a8, .. } => {
                let m_gemm = qwen4_fast_proj::resolve_fp8_pad_m(rows, n);
                if m_gemm > rows {
                    ctx.gpu
                        .memset_async(a8.offset(rows * kd), 0, (m_gemm - rows) * kd, stream)?;
                }
                qwen4_fast_proj::fp8_weight_tensor_gemm(
                    ctx.gpu, &tk, weight, output, m_gemm, n, kd, n, None, w8, ws, a8, true, stream,
                )?;
                m_gemm
            }
            StripOutTarget::Nvfp4 {
                a_packed,
                a_scale,
                alpha_dev,
                w_scale_swizzled,
                ..
            } => {
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    weight.weight.0,
                    w_scale_swizzled.0,
                    output.0,
                    rows_u32,
                    u32::try_from(n)?,
                    u32::try_from(kd)?,
                    u32::try_from(n)?,
                    spark_runtime::cublaslt::BlockScaleScalars::device_alpha(alpha_dev.0),
                    stream,
                )?;
                static NVFP4_OUT_LOGGED: std::sync::Once = std::sync::Once::new();
                NVFP4_OUT_LOGGED.call_once(|| {
                    tracing::info!(
                        "GDN_PREFILL_OUT_PROJ_NVFP4_ENGAGED selector={} rows={rows} n={n} k={kd} strip_rows={strip_rows} max_strip_rows={max_strip_rows} ping_pong={}",
                        qwen4_fast_proj::nvfp4::NVFP4_SELECTOR,
                        ping_pong.is_some(),
                    );
                });
                rows
            }
        };

        static STRIP_LOGGED: std::sync::Once = std::sync::Once::new();
        STRIP_LOGGED.call_once(|| {
            tracing::info!(
                "SSM_PREFILL_GDN_FI_STRIP_ENGAGED selector={SELECTOR} mode=f16_strip conv={} rows={rows} strip_rows={strip_rows} max_strip_rows={max_strip_rows} m_gemm={m_gemm} dtype=f16 out_proj={} ping_pong={} l2_discard={}",
                conv_strip_k.variant_name(),
                target.label(),
                ping_pong.is_some(),
                l2_discard.as_flag_str()
            );
        });
        Ok(true)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FiGdnPingPongLayout {
    slot_qkv: [DevicePtr; 2],
    slot_out: [DevicePtr; 2],
    total_bytes: usize,
}

#[must_use]
fn compute_ping_pong_layout(
    scratch: DevicePtr,
    max_strip_rows: usize,
    conv_dim: usize,
    value_dim: usize,
    arena_bytes: usize,
) -> Option<FiGdnPingPongLayout> {
    if scratch.is_null() || !scratch.0.is_multiple_of(128) || max_strip_rows == 0 {
        return None;
    }
    let gdn_out_off = gdn_out_strip_offset(max_strip_rows, conv_dim);
    let slot_stride = (gdn_out_off
        .checked_add(max_strip_rows.checked_mul(value_dim)?.checked_mul(2)?)?
        .checked_add(GDN_OUT_STRIP_L2_SKEW_BYTES)?)
    .next_multiple_of(256);
    let total_bytes = slot_stride.checked_mul(2)?;
    if total_bytes > arena_bytes {
        return None;
    }
    Some(FiGdnPingPongLayout {
        slot_qkv: [scratch, scratch.offset(slot_stride)],
        slot_out: [
            scratch.offset(gdn_out_off),
            scratch.offset(slot_stride + gdn_out_off),
        ],
        total_bytes,
    })
}

#[must_use]
fn build_strip_ranges(
    rows: usize,
    strip_rows: usize,
    max_strip_rows: usize,
) -> Vec<(usize, usize)> {
    let mut strips = Vec::with_capacity(rows.div_ceil(strip_rows.max(1)));
    let mut r0 = 0usize;
    while r0 < rows {
        let rem = rows - r0;
        let ns = if rem <= max_strip_rows {
            rem
        } else {
            strip_rows
        };
        strips.push((r0, ns));
        r0 += ns;
    }
    strips
}

#[derive(Clone, Copy, Debug)]
struct FiGdnOverlapStream {
    side_stream: u64,
    start_event: u64,
    conv_done: [u64; 2],
    fi_done: [u64; 2],
    side_done: u64,
}

#[inline]
fn gpu_backend_key(gpu: &dyn GpuBackend) -> usize {
    std::ptr::from_ref(gpu) as *const () as usize
}

fn get_or_init_strip_overlap_runtime(
    gpu: &dyn GpuBackend,
    main_stream: u64,
) -> Result<Option<FiGdnOverlapStream>> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static RUNTIME: Mutex<Option<HashMap<usize, Option<FiGdnOverlapStream>>>> = Mutex::new(None);
    let key = gpu_backend_key(gpu);
    {
        let guard = RUNTIME
            .lock()
            .expect("GDN strip overlap runtime lock poisoned");
        if let Some(map) = guard.as_ref()
            && let Some(&cached) = map.get(&key)
        {
            return Ok(cached.filter(|rt| rt.side_stream != main_stream));
        }
    }
    let side_stream = gpu.create_stream()?;
    if side_stream == 0 {
        let mut guard = RUNTIME
            .lock()
            .expect("GDN strip overlap runtime lock poisoned");
        guard.get_or_insert_with(HashMap::new).insert(key, None);
        return Ok(None);
    }
    let start_event = gpu.create_event()?;
    let conv_done = [gpu.create_event()?, gpu.create_event()?];
    let fi_done = [gpu.create_event()?, gpu.create_event()?];
    let side_done = gpu.create_event()?;
    let created = if start_event != 0
        && conv_done[0] != 0
        && conv_done[1] != 0
        && fi_done[0] != 0
        && fi_done[1] != 0
        && side_done != 0
    {
        Some(FiGdnOverlapStream {
            side_stream,
            start_event,
            conv_done,
            fi_done,
            side_done,
        })
    } else {
        None
    };
    let mut guard = RUNTIME
        .lock()
        .expect("GDN strip overlap runtime lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    let entry = *map.entry(key).or_insert(created);
    Ok(entry.filter(|rt| rt.side_stream != main_stream))
}

/// RAII join guard ensuring any side-stream conv/quant work launched inside the
/// ping-pong strip loop is joined back onto `main_stream` before returning — on
/// normal completion via [`Self::join`], early `Err` return, or panic unwind.
struct StripOverlapJoinGuard<'a> {
    gpu: &'a dyn GpuBackend,
    main_stream: u64,
    side_stream: u64,
    side_done: u64,
    side_launched: bool,
    joined: bool,
}

impl<'a> StripOverlapJoinGuard<'a> {
    #[must_use]
    const fn new(
        gpu: &'a dyn GpuBackend,
        main_stream: u64,
        side_stream: u64,
        side_done: u64,
    ) -> Self {
        Self {
            gpu,
            main_stream,
            side_stream,
            side_done,
            side_launched: false,
            joined: false,
        }
    }

    fn mark_launched(&mut self) {
        self.side_launched = true;
    }

    fn join(mut self) -> Result<()> {
        self.joined = true;
        if !self.side_launched {
            return Ok(());
        }
        self.gpu.record_event(self.side_done, self.side_stream)?;
        self.gpu.stream_wait_event(self.main_stream, self.side_done)
    }
}

impl Drop for StripOverlapJoinGuard<'_> {
    fn drop(&mut self) {
        if self.side_launched && !self.joined {
            let ok = self
                .gpu
                .record_event(self.side_done, self.side_stream)
                .and_then(|()| self.gpu.stream_wait_event(self.main_stream, self.side_done))
                .is_ok();
            if !ok {
                let _ = self.gpu.synchronize(self.side_stream);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct StripNvfp4Kernels {
    norm_nvfp4_f16_strip: KernelHandle,
    swizzle_weight: KernelHandle,
}

fn strip_nvfp4_kernels(gpu: &dyn GpuBackend) -> Option<StripNvfp4Kernels> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static CACHE: Mutex<Option<HashMap<usize, Option<StripNvfp4Kernels>>>> = Mutex::new(None);
    let key = gpu_backend_key(gpu);
    let mut guard = CACHE
        .lock()
        .expect("GDN strip NVFP4 kernel cache lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(&cached) = map.get(&key) {
        return cached;
    }
    let norm_k = crate::layers::try_kernel(
        gpu,
        "qwen4_gdn_norm_fp8",
        "qwen4_gdn_norm_nvfp4_f16_v2_strip",
    );
    let swizzle_k = crate::layers::try_kernel(
        gpu,
        "qwen4_nvfp4_act_quant",
        "qwen4_nvfp4_swizzle_weight_scales_128x4",
    );
    let resolved = (norm_k.0 != 0 && swizzle_k.0 != 0).then_some(StripNvfp4Kernels {
        norm_nvfp4_f16_strip: norm_k,
        swizzle_weight: swizzle_k,
    });
    map.insert(key, resolved);
    resolved
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StripNvfp4ActLayout {
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    alpha_dev: DevicePtr,
    total_bytes: usize,
}

#[must_use]
fn ranges_disjoint(a: (u64, usize), b: (u64, usize)) -> bool {
    if a.1 == 0 || b.1 == 0 {
        return true;
    }
    let Some(a_end) = a.0.checked_add(a.1 as u64) else {
        return false;
    };
    let Some(b_end) = b.0.checked_add(b.1 as u64) else {
        return false;
    };
    a_end <= b.0 || b_end <= a.0
}

#[must_use]
fn compute_strip_nvfp4_act_layout(
    base: DevicePtr,
    capacity: usize,
    rows: usize,
    k: usize,
    peers: &[(DevicePtr, usize)],
) -> Option<StripNvfp4ActLayout> {
    if base.is_null() || !base.0.is_multiple_of(256) || rows == 0 || k == 0 || !k.is_multiple_of(32)
    {
        return None;
    }
    let packed_bytes = rows.checked_mul(k / 2)?.next_multiple_of(256);
    let scale_bytes = spark_runtime::cublaslt::block_scale_padded_bytes(
        u32::try_from(rows).ok()?,
        u32::try_from(k).ok()?,
        spark_runtime::cublaslt::NVFP4_BLOCK_SIZE,
    )
    .next_multiple_of(256);
    let total_bytes = packed_bytes.checked_add(scale_bytes)?.checked_add(256)?;
    if total_bytes > capacity {
        return None;
    }
    let mine = (base.0, total_bytes);
    if peers
        .iter()
        .any(|&(ptr, len)| !ptr.is_null() && !ranges_disjoint(mine, (ptr.0, len)))
    {
        return None;
    }
    Some(StripNvfp4ActLayout {
        a_packed: base,
        a_scale: base.offset(packed_bytes),
        alpha_dev: base.offset(packed_bytes + scale_bytes),
        total_bytes,
    })
}

fn get_or_swizzle_out_proj_scales(
    gpu: &dyn GpuBackend,
    swizzle_k: KernelHandle,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<Option<DevicePtr>> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    type CacheKey = (usize, u64, u64, u32, usize, usize);
    static SCALES: Mutex<Option<HashMap<CacheKey, DevicePtr>>> = Mutex::new(None);
    let key: CacheKey = (
        gpu_backend_key(gpu),
        weight.weight.0,
        weight.weight_scale.0,
        weight.weight_scale_2.to_bits(),
        n,
        k,
    );
    {
        let guard = SCALES
            .lock()
            .expect("GDN out_proj NVFP4 scale cache lock poisoned");
        if let Some(map) = guard.as_ref()
            && let Some(&cached) = map.get(&key)
        {
            return Ok(Some(cached));
        }
    }
    if spark_runtime::gpu_capture_region::active_capture_region().is_some()
        || gpu.stream_is_capturing(stream)
    {
        return Ok(None);
    }
    let scale_bytes = spark_runtime::cublaslt::block_scale_padded_bytes(
        u32::try_from(n)?,
        u32::try_from(k)?,
        spark_runtime::cublaslt::NVFP4_BLOCK_SIZE,
    )
    .next_multiple_of(256);
    let dst = gpu.alloc(scale_bytes)?;
    let groups = (k / 16) as u32;
    let padded_groups = groups.next_multiple_of(4);
    let padded_n = (n as u32).next_multiple_of(128);
    KernelLaunch::new(gpu, swizzle_k)
        .grid([div_ceil(padded_groups, 128), padded_n.min(32_768), 1])
        .block([128, 1, 1])
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(dst)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(0u32)
        .arg_u32(0u32)
        .launch(stream)?;
    let mut guard = SCALES
        .lock()
        .expect("GDN out_proj NVFP4 scale cache lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    let entry = *map.entry(key).or_insert(dst);
    Ok(Some(entry))
}

#[derive(Clone, Copy)]
enum StripOutTarget {
    Fp8 {
        quant_strip_k: KernelHandle,
        tk: crate::layers::qwen4_fast_proj::Fp8TensorKernels,
        w8: DevicePtr,
        ws: DevicePtr,
        a8: DevicePtr,
        a_scale: DevicePtr,
    },
    Nvfp4 {
        norm_nvfp4_k: KernelHandle,
        a_packed: DevicePtr,
        a_scale: DevicePtr,
        alpha_dev: DevicePtr,
        w_scale_swizzled: DevicePtr,
        weight_scale_2: f32,
    },
}

impl StripOutTarget {
    const fn label(self) -> &'static str {
        match self {
            Self::Fp8 { .. } => "fp8",
            Self::Nvfp4 { .. } => "nvfp4",
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn try_prepare_strip_nvfp4_target(
    ctx: &ForwardContext,
    weight: &QuantizedWeight,
    rows: usize,
    n: usize,
    kd: usize,
    deinterleaved: DevicePtr,
    gates: DevicePtr,
    scratch: DevicePtr,
    output: DevicePtr,
    stream: u64,
) -> Result<Option<StripOutTarget>> {
    use crate::layers::qwen4_fast_proj::nvfp4;
    let nvfp4_mode = nvfp4::mode()?;
    let scope_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active();
    if !nvfp4_mode.admits_call(rows, n, kd, n, None, scope_active, nvfp4::verify_min_m())
        || !weight.weight.0.is_multiple_of(16)
        || !output.0.is_multiple_of(16)
        || crate::layers::qwen4_prefill_native::fp8_lookup(weight, n, kd, None)?.is_some()
    {
        return Ok(None);
    }
    let Some(ks) = strip_nvfp4_kernels(ctx.gpu) else {
        return Ok(None);
    };
    let sizes = ctx.buffers.sizes();
    let Some(act_layout) = compute_strip_nvfp4_act_layout(
        ctx.buffers.ssm_qkvz(),
        sizes.ssm_qkvz,
        rows,
        kd,
        &[
            (deinterleaved, sizes.ssm_deinterleaved),
            (gates, sizes.ssm_gates),
            (scratch, sizes.ssm_conv_out_f32),
            (output, sizes.moe_output),
        ],
    ) else {
        return Ok(None);
    };
    let Some(w_scale_swizzled) =
        get_or_swizzle_out_proj_scales(ctx.gpu, ks.swizzle_weight, weight, n, kd, stream)?
    else {
        return Ok(None);
    };
    Ok(Some(StripOutTarget::Nvfp4 {
        norm_nvfp4_k: ks.norm_nvfp4_f16_strip,
        a_packed: act_layout.a_packed,
        a_scale: act_layout.a_scale,
        alpha_dev: act_layout.alpha_dev,
        w_scale_swizzled,
        weight_scale_2: weight.weight_scale_2,
    }))
}

struct StripQuantLaunch {
    qkv_strip: DevicePtr,
    gdn_out_strip: DevicePtr,
    z_row: DevicePtr,
    norm_weight: DevicePtr,
    row_offset: usize,
    strip_rows: usize,
    total_rows: usize,
    value_dim: usize,
    eps: f32,
    l2_discard: super::qwen4_gdn_norm_fp8::L2DiscardPolicy,
    last_strip: bool,
    allow_early_scale_write: bool,
}

fn launch_strip_quant(
    gpu: &dyn GpuBackend,
    target: &StripOutTarget,
    p: StripQuantLaunch,
    stream: u64,
) -> Result<()> {
    use super::qwen4_prefill_exact_plan::Plan;
    let ns_u32 = p.strip_rows as u32;
    let discard = super::qwen4_gdn_norm_fp8::resolve_strip_l2_discard(
        p.l2_discard,
        super::qwen4_gdn_norm_fp8::StripL2DiscardBuffers {
            qkv_strip: p.qkv_strip,
            gdn_out_strip: p.gdn_out_strip,
            z_row: p.z_row,
            conv_dim: Plan::CONV_DIM,
            value_dim: p.value_dim,
            qkvz_stride: Plan::QKVZ,
        },
    );
    match *target {
        StripOutTarget::Fp8 {
            quant_strip_k,
            a8,
            a_scale,
            ..
        } => {
            let scale_arg = if p.last_strip
                || (p.allow_early_scale_write && p.row_offset + 1280 >= p.total_rows)
            {
                a_scale
            } else {
                DevicePtr::NULL
            };
            KernelLaunch::new(gpu, quant_strip_k)
                .grid(super::qwen4_gdn_norm_fp8::quant_v2_strip_grid(ns_u32))
                .block([256, 1, 1])
                .arg_ptr(p.gdn_out_strip)
                .arg_ptr(p.z_row)
                .arg_ptr(p.norm_weight)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(a8.offset(p.row_offset * p.value_dim))
                .arg_ptr(scale_arg)
                .arg_u32(128)
                .arg_f32(p.eps)
                .arg_u32(p.value_dim as u32)
                .arg_u32(Plan::QKVZ as u32)
                .arg_u32(p.value_dim as u32)
                .arg_u32(ns_u32)
                .arg_ptr(discard.qkv_discard)
                .arg_u32(discard.discard_flags)
                .launch(stream)
        }
        StripOutTarget::Nvfp4 {
            norm_nvfp4_k,
            a_packed,
            a_scale,
            alpha_dev,
            weight_scale_2,
            ..
        } => {
            let alpha_arg = if p.last_strip {
                alpha_dev
            } else {
                DevicePtr::NULL
            };
            KernelLaunch::new(gpu, norm_nvfp4_k)
                .grid(super::qwen4_gdn_norm_fp8::quant_v2_strip_grid(ns_u32))
                .block([256, 1, 1])
                .arg_ptr(p.gdn_out_strip)
                .arg_ptr(p.z_row)
                .arg_ptr(p.norm_weight)
                .arg_ptr(a_packed)
                .arg_ptr(a_scale)
                .arg_ptr(alpha_arg)
                .arg_f32(weight_scale_2)
                .arg_u32(128)
                .arg_f32(p.eps)
                .arg_u32(p.value_dim as u32)
                .arg_u32(Plan::QKVZ as u32)
                .arg_u32(p.value_dim as u32)
                .arg_u32(ns_u32)
                .arg_u32(p.row_offset as u32)
                .arg_u32(p.total_rows as u32)
                .arg_ptr(discard.qkv_discard)
                .arg_u32(discard.discard_flags)
                .launch(stream)
        }
    }
}

#[derive(Clone, Copy)]
enum StripConvKernel {
    V3(KernelHandle),
    V4(KernelHandle),
}

struct StripConvLaunch {
    conv_state: DevicePtr,
    deinterleaved: DevicePtr,
    weight: DevicePtr,
    qkv_strip: DevicePtr,
    strip_rows: u32,
    row_offset: u32,
    total_rows: u32,
    conv_dim: u32,
    qk_channels: u32,
    input_stride: u32,
    discard_flags: u32,
}

impl StripConvKernel {
    const fn variant_name(self) -> &'static str {
        match self {
            Self::V3(_) => "v3_strip",
            Self::V4(_) => "v4_strip",
        }
    }

    fn launch(self, gpu: &dyn GpuBackend, p: StripConvLaunch, stream: u64) -> Result<()> {
        match self {
            Self::V3(kernel) => KernelLaunch::new(gpu, kernel)
                .grid(conv_v3_strip_grid(p.conv_dim, p.strip_rows))
                .block([CONV_V3_STRIP_BLOCK_THREADS, 1, 1])
                .arg_ptr(p.conv_state)
                .arg_ptr(p.deinterleaved)
                .arg_ptr(p.weight)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(p.qkv_strip)
                .arg_u32(p.strip_rows)
                .arg_u32(p.row_offset)
                .arg_u32(p.total_rows)
                .arg_u32(p.qk_channels)
                .arg_f32(1e-6)
                .arg_u32(p.input_stride)
                .arg_u32(p.conv_dim)
                .launch(stream),
            Self::V4(kernel) => KernelLaunch::new(gpu, kernel)
                .grid(conv_v4_strip_grid(p.conv_dim, p.strip_rows))
                .block([CONV_V4_STRIP_BLOCK_THREADS, 1, 1])
                .arg_ptr(p.conv_state)
                .arg_ptr(p.deinterleaved)
                .arg_ptr(p.weight)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(p.qkv_strip)
                .arg_u32(p.strip_rows)
                .arg_u32(p.row_offset)
                .arg_u32(p.total_rows)
                .arg_u32(p.qk_channels)
                .arg_f32(1e-6)
                .arg_u32(p.input_stride)
                .arg_u32(p.conv_dim)
                .arg_u32(p.discard_flags)
                .launch(stream),
        }
    }
}

fn conv_strip_kernel(gpu: &dyn GpuBackend, dtype: FiDtype) -> Result<Option<StripConvKernel>> {
    use super::qwen4_gdn_conv_v2 as v2;
    static V3_SEL: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    let v3_on = V3_SEL
        .get_or_init(|| v2::parse_v3(std::env::var(v2::V3_SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)?;
    if !v3_on {
        return Ok(None);
    }
    type Slot = std::sync::OnceLock<Result<KernelHandle, String>>;
    if conv_v4_strip_enabled()? {
        static BF16_V4_STRIP: Slot = Slot::new();
        static F16_V4_STRIP: Slot = Slot::new();
        let (slot, name) = match dtype {
            FiDtype::Bf16 => (&BF16_V4_STRIP, "qwen4_conv1d_l2norm_bf16_prefill_v4_strip"),
            FiDtype::F16 => (&F16_V4_STRIP, "qwen4_conv1d_l2norm_f16_prefill_v4_strip"),
        };
        return slot
            .get_or_init(|| {
                gpu.kernel("qwen4_gdn_fi_glue", name)
                    .map_err(|e| e.to_string())
            })
            .clone()
            .map(|h| Some(StripConvKernel::V4(h)))
            .map_err(|e| {
                anyhow::anyhow!("{CONV_V4_STRIP_SELECTOR}: kernel {name} unavailable: {e}")
            });
    }
    static BF16_STRIP: Slot = Slot::new();
    static F16_STRIP: Slot = Slot::new();
    let (slot, name) = match dtype {
        FiDtype::Bf16 => (&BF16_STRIP, "qwen4_conv1d_l2norm_bf16_prefill_v3_strip"),
        FiDtype::F16 => (&F16_STRIP, "qwen4_conv1d_l2norm_f16_prefill_v3_strip"),
    };
    slot.get_or_init(|| {
        gpu.kernel("qwen4_gdn_fi_glue", name)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map(|h| Some(StripConvKernel::V3(h)))
    .map_err(|e| anyhow::anyhow!("{}: kernel {name} unavailable: {e}", v2::V3_SELECTOR))
}

/// Selector for FlashInfer CuTe-DSL BF16 chunked GDN prefill on dense Qwen3.8
/// models (e.g. `Qwen3.8-27B` where `h_qk=16, h_v=48, kd=128, vd=128`).
///
/// - `"1"` / `"true"` / `"yes"` / `"on"` -> enabled
/// - `"0"` / `"false"` / `"no"` / `"off"` -> disabled
/// - unset -> auto-enabled when `active_sm_count() > 64` (SM120/SM103 discrete
///   GPUs where 48-CTA WY32 under-occupies 170–188 SMs; GB10 `<=64` SMs stays
///   on its default WY32 route unless explicitly opted in).
const DENSE_GDN_FI_SELECTOR: &str = "ATLAS_GDN_PREFILL_FI";

#[inline]
fn resolve_dense_gdn_fi_enabled(env: Option<&str>, sm_count: u32) -> bool {
    match env.map(str::trim) {
        Some("0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF") => false,
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON") => true,
        _ => sm_count > 64,
    }
}

#[inline]
fn dense_gdn_fi_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let env = std::env::var(DENSE_GDN_FI_SELECTOR).ok();
        let sm_count = atlas_core::device::active_sm_count();
        resolve_dense_gdn_fi_enabled(env.as_deref(), sm_count)
    })
}

/// Pure shape & state guard for the BF16 AOT GDN prefill bridge.
#[inline]
fn dense_gdn_fi_shape_eligible(
    rows: u32,
    nk: usize,
    nv: usize,
    kd: usize,
    vd: usize,
    conv_dim: usize,
    h_is_f16: bool,
    h_state_non_null: bool,
) -> bool {
    usize::try_from(rows).is_ok_and(|r| r > super::qwen4_gdn_fi_policy::SHORT_FALLBACK_MAX_ROWS)
        && !h_is_f16
        && h_state_non_null
        && kd == 128
        && vd == 128
        && matches!(
            (nk, nv, conv_dim),
            (16, 48, 10240) | (8, 24, 5120) | (4, 12, 2560)
        )
}

/// Try running the FlashInfer CuTe-DSL BF16 chunked GDN prefill bridge for
/// dense Qwen3.8 SSM layers (`Qwen3.8-27B` TP=1/2/4). Returns `Ok(true)` when
/// the bridge executed, `Ok(false)` when ineligible or unavailable without AOT.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_dense_gdn_fi_prefill(
    qkv: DevicePtr,
    gate_beta: DevicePtr,
    output: DevicePtr,
    h_state: DevicePtr,
    h_is_f16: bool,
    rows: u32,
    nk: usize,
    nv: usize,
    kd: usize,
    vd: usize,
    conv_dim: usize,
    stream: u64,
) -> Result<bool> {
    if !dense_gdn_fi_enabled()
        || !dense_gdn_fi_shape_eligible(
            rows,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            h_is_f16,
            !h_state.is_null(),
        )
    {
        return Ok(false);
    }

    #[cfg(atlas_fi_gdn)]
    {
        let ptr = |p: DevicePtr| p.0 as usize as *mut core::ffi::c_void;
        let rows_i32 = i32::try_from(rows)?;
        let conv_dim_i32 = i32::try_from(conv_dim)?;
        // SAFETY: `qkv` is `[rows, conv_dim]` BF16, `gate_beta` is interleaved
        // `[rows, 2*nv]` FP32, `output` is `[rows, nv*vd]` BF16, and `h_state`
        // is `[nv, kd, vd]` FP32, all live device allocations sized by the
        // prefill arena; the bridge only enqueues work on `stream`.
        let rc = unsafe {
            ffi::atlas_gdn_fi_sm121a_prefill(
                ptr(qkv),
                ptr(gate_beta),
                ptr(output),
                ptr(h_state),
                rows_i32,
                conv_dim_i32,
                stream as usize as *mut core::ffi::c_void,
            )
        };
        ensure!(
            rc == 0,
            "{DENSE_GDN_FI_SELECTOR}: atlas_gdn_fi_sm121a_prefill failed rc={rc} rows={rows} conv_dim={conv_dim}"
        );
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::info!(
                "ENGAGED {DENSE_GDN_FI_SELECTOR}: FlashInfer BF16 chunked GDN prefill M={rows} nk={nk} nv={nv} conv_dim={conv_dim}"
            );
        });
        Ok(true)
    }

    #[cfg(not(atlas_fi_gdn))]
    {
        let _ = (qkv, gate_beta, output, stream);
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::super::qwen4_gdn_norm_fp8::{
        DISC_FLAG_G, DISC_FLAG_QKV, DISC_FLAG_X, L2_DISCARD_LINE_BYTES, L2DiscardPolicy,
        StripL2DiscardBuffers, resolve_strip_l2_discard,
    };
    use super::super::qwen4_prefill_exact_plan::Plan;
    use super::*;

    #[test]
    fn strip_loop_discards_all_strips_including_last_strip() {
        let strip_rows = DEFAULT_STRIP_ROWS;
        let max_strip_rows = strip_rows + STRIP_TAIL_MARGIN;
        let scratch = DevicePtr(0x1000_0000);
        let deinterleaved = DevicePtr(0x2000_0000);
        let qkv_strip = scratch;
        let gdn_out_off = gdn_out_strip_offset(max_strip_rows, Plan::CONV_DIM);
        let gdn_out_strip = scratch.offset(gdn_out_off);
        let z_base = deinterleaved.offset(Plan::CONV_DIM * 2);

        assert!(gdn_out_strip.0.is_multiple_of(256));
        assert!(gdn_out_strip.0.is_multiple_of(L2_DISCARD_LINE_BYTES as u64));
        assert!(z_base.0.is_multiple_of(L2_DISCARD_LINE_BYTES as u64));
        assert_eq!(gdn_out_off % 65_536, GDN_OUT_STRIP_L2_SKEW_BYTES);
        assert_eq!(gdn_out_off % 262_144, GDN_OUT_STRIP_L2_SKEW_BYTES);

        for rows in [641usize, 1024, 16_384, 16_512] {
            let mut r0 = 0usize;
            let mut strips = 0usize;
            while r0 < rows {
                let rem = rows - r0;
                let ns = if rem <= max_strip_rows {
                    rem
                } else {
                    strip_rows
                };
                let z_row = z_base.offset(r0 * Plan::QKVZ * 2);
                let discard = resolve_strip_l2_discard(
                    L2DiscardPolicy::Enabled,
                    StripL2DiscardBuffers {
                        qkv_strip,
                        gdn_out_strip,
                        z_row,
                        conv_dim: Plan::CONV_DIM,
                        value_dim: Plan::VALUE_DIM,
                        qkvz_stride: Plan::QKVZ,
                    },
                );
                assert_eq!(
                    discard.qkv_discard, qkv_strip,
                    "qkv_strip must be discarded on strip r0={r0} ns={ns} rows={rows}"
                );
                assert_eq!(
                    discard.discard_flags,
                    DISC_FLAG_X | DISC_FLAG_G | DISC_FLAG_QKV,
                    "all strip buffers must be discarded on strip r0={r0} ns={ns} rows={rows}"
                );
                r0 += ns;
                strips += 1;
            }
            assert_eq!(r0, rows);
            if rows == 16_384 {
                assert_eq!(strips, 32);
            }
        }
    }

    #[test]
    fn conv_v3_strip_grid_hits_exact_sm48_waves_and_covers_tails() {
        let g512 = conv_v3_strip_grid(Plan::CONV_DIM as u32, DEFAULT_STRIP_ROWS as u32);
        assert_eq!(g512, [80, 3, 1]);
        assert_eq!((g512[0] * g512[1]) % 48, 0);
        assert_eq!(conv_v3_strip_grid(Plan::CONV_DIM as u32, 640), [80, 4, 1]);
        assert_eq!(conv_v3_strip_grid(Plan::CONV_DIM as u32, 128), [80, 1, 1]);
        assert_eq!(conv_v3_strip_grid(Plan::CONV_DIM as u32, 1), [80, 1, 1]);
        assert_eq!(conv_v3_strip_grid(5120, 512), [40, 3, 1]);
    }

    #[test]
    fn conv_v4_strip_grid_hits_exact_sm48_waves_and_covers_tails() {
        let g512 = conv_v4_strip_grid(Plan::CONV_DIM as u32, DEFAULT_STRIP_ROWS as u32);
        assert_eq!(g512, [40, 6, 1]);
        assert_eq!((g512[0] * g512[1]) % 48, 0);
        assert_eq!(conv_v4_strip_grid(Plan::CONV_DIM as u32, 640), [40, 7, 1]);
        assert_eq!(conv_v4_strip_grid(Plan::CONV_DIM as u32, 128), [40, 2, 1]);
        assert_eq!(conv_v4_strip_grid(Plan::CONV_DIM as u32, 1), [40, 1, 1]);
        assert_eq!(conv_v4_strip_grid(5120, 512), [20, 6, 1]);
    }

    #[test]
    fn quant_v2_strip_grid_hits_exact_sm48_waves_and_covers_tails() {
        use super::super::qwen4_gdn_norm_fp8::quant_v2_strip_grid;

        let g512 = quant_v2_strip_grid(DEFAULT_STRIP_ROWS as u32);
        assert_eq!(g512, [3, 64, 1]);
        assert!((g512[0] * g512[1]).is_multiple_of(48));

        let g640 = quant_v2_strip_grid((DEFAULT_STRIP_ROWS + STRIP_TAIL_MARGIN) as u32);
        assert_eq!(g640, [3, 80, 1]);
        assert!((g640[0] * g640[1]).is_multiple_of(48));

        assert_eq!(quant_v2_strip_grid(1), [3, 1, 1]);
        assert_eq!(quant_v2_strip_grid(7), [3, 1, 1]);
        assert_eq!(quant_v2_strip_grid(8), [3, 1, 1]);
        assert_eq!(quant_v2_strip_grid(9), [3, 2, 1]);
        assert_eq!(quant_v2_strip_grid(129), [3, 17, 1]);
        assert_eq!(quant_v2_strip_grid(513), [3, 65, 1]);
    }

    #[test]
    fn parse_conv_v4_strip_defaults_on_and_handles_opt_out() {
        assert!(parse_conv_v4_strip(None).expect("None defaults on"));
        assert!(parse_conv_v4_strip(Some("")).expect("empty defaults on"));
        assert!(parse_conv_v4_strip(Some("1")).expect("1 enables"));
        assert!(parse_conv_v4_strip(Some("on")).expect("on enables"));
        assert!(parse_conv_v4_strip(Some("TRUE")).expect("TRUE enables"));
        assert!(!parse_conv_v4_strip(Some("0")).expect("0 disables"));
        assert!(!parse_conv_v4_strip(Some("off")).expect("off disables"));
        assert!(!parse_conv_v4_strip(Some("false")).expect("false disables"));
        assert!(parse_conv_v4_strip(Some("invalid")).is_err());
    }

    #[test]
    fn conv_v4_strip_discard_flags_respects_policy_and_alignment() {
        let aligned = DevicePtr(0x2000_0000);
        let unaligned = DevicePtr(0x2000_0010);
        assert_eq!(
            conv_v4_strip_discard_flags(L2DiscardPolicy::Enabled, aligned, Plan::QKVZ),
            CONV_V4_STRIP_DISC_INPUT
        );
        assert_eq!(
            conv_v4_strip_discard_flags(L2DiscardPolicy::Enabled, aligned, 8192),
            CONV_V4_STRIP_DISC_INPUT
        );
        assert_eq!(
            conv_v4_strip_discard_flags(L2DiscardPolicy::Disabled, aligned, Plan::QKVZ),
            0
        );
        assert_eq!(
            conv_v4_strip_discard_flags(L2DiscardPolicy::Enabled, DevicePtr::NULL, Plan::QKVZ),
            0
        );
        assert_eq!(
            conv_v4_strip_discard_flags(L2DiscardPolicy::Enabled, unaligned, Plan::QKVZ),
            0
        );
        assert_eq!(
            conv_v4_strip_discard_flags(L2DiscardPolicy::Enabled, aligned, Plan::QKVZ + 8),
            0
        );
    }

    #[test]
    fn strip_rows_config_scales_for_high_sm_and_honors_override() {
        assert_eq!(resolve_strip_rows_config(None, 48), (512, 640));
        assert_eq!(resolve_strip_rows_config(None, 64), (512, 640));
        assert_eq!(resolve_strip_rows_config(None, 65), (2048, 2176));
        assert_eq!(resolve_strip_rows_config(None, 188), (2048, 2176));

        // Explicit valid override wins over both GB10 and SM120 defaults.
        assert_eq!(resolve_strip_rows_config(Some("512"), 188), (512, 640));
        assert_eq!(resolve_strip_rows_config(Some(" 1024 "), 188), (1024, 1152));
        assert_eq!(resolve_strip_rows_config(Some("128"), 48), (128, 256));

        // Invalid overrides (< 128, non-multiple of 64, unparseable) fall back to the SM default.
        for bad in ["", "0", "64", "192x", "300", "auto"] {
            assert_eq!(
                resolve_strip_rows_config(Some(bad), 48),
                (512, 640),
                "bad={bad:?} sm=48"
            );
            assert_eq!(
                resolve_strip_rows_config(Some(bad), 188),
                (2048, 2176),
                "bad={bad:?} sm=188"
            );
        }
    }

    #[test]
    fn dense_gdn_fi_policy_and_shape_guards() {
        assert!(!resolve_dense_gdn_fi_enabled(None, 48));
        assert!(!resolve_dense_gdn_fi_enabled(None, 64));
        assert!(resolve_dense_gdn_fi_enabled(None, 65));
        assert!(resolve_dense_gdn_fi_enabled(None, 188));
        assert!(!resolve_dense_gdn_fi_enabled(Some("0"), 188));
        assert!(!resolve_dense_gdn_fi_enabled(Some("false"), 188));
        assert!(resolve_dense_gdn_fi_enabled(Some("1"), 48));
        assert!(resolve_dense_gdn_fi_enabled(Some("true"), 48));

        // TP=1 / TP=2 / TP=4 Qwen3.8-27B geometries (strictly above SHORT_FALLBACK_MAX_ROWS=64).
        assert!(dense_gdn_fi_shape_eligible(
            65, 16, 48, 128, 128, 10240, false, true
        ));
        assert!(dense_gdn_fi_shape_eligible(
            7942, 8, 24, 128, 128, 5120, false, true
        ));
        assert!(dense_gdn_fi_shape_eligible(
            128, 4, 12, 128, 128, 2560, false, true
        ));

        // Reject <= SHORT_FALLBACK_MAX_ROWS (64), FP16 recurrent state, null state, or unsupported shapes.
        assert!(!dense_gdn_fi_shape_eligible(
            32, 16, 48, 128, 128, 10240, false, true
        ));
        assert!(!dense_gdn_fi_shape_eligible(
            64, 16, 48, 128, 128, 10240, false, true
        ));
        assert!(!dense_gdn_fi_shape_eligible(
            65, 16, 48, 128, 128, 10240, true, true
        ));
        assert!(!dense_gdn_fi_shape_eligible(
            65, 16, 48, 128, 128, 10240, false, false
        ));
        assert!(!dense_gdn_fi_shape_eligible(
            65, 16, 32, 128, 128, 8192, false, true
        ));
        assert!(!dense_gdn_fi_shape_eligible(
            65, 16, 48, 64, 128, 10240, false, true
        ));
    }

    #[test]
    fn strip_overlap_policy_and_ping_pong_strip_height_preserve_gb10_defaults() {
        for sm in [0u32, 48, 64] {
            assert!(
                !resolve_strip_overlap_enabled(None, sm),
                "sm={sm} must default strip overlap OFF"
            );
            assert!(!resolve_strip_overlap_enabled(Some(""), sm));
            assert!(resolve_strip_overlap_enabled(Some("1"), sm));
            assert!(resolve_strip_overlap_enabled(Some("ON"), sm));
            assert!(resolve_strip_overlap_enabled(Some("true"), sm));
            assert_eq!(
                resolve_strip_rows_config_for_mode(None, sm, false),
                (512, 640)
            );
            assert_eq!(
                resolve_strip_rows_config_for_mode(None, sm, true),
                (512, 640)
            );
        }
        for sm in [65u32, 110, 170, 188] {
            assert!(
                resolve_strip_overlap_enabled(None, sm),
                "sm={sm} must auto-enable strip overlap"
            );
            assert!(resolve_strip_overlap_enabled(Some(""), sm));
            assert!(!resolve_strip_overlap_enabled(Some("0"), sm));
            assert!(!resolve_strip_overlap_enabled(Some("off"), sm));
            assert!(!resolve_strip_overlap_enabled(Some("False"), sm));
            assert!(!resolve_strip_overlap_enabled(Some("no"), sm));
            assert_eq!(
                resolve_strip_rows_config_for_mode(None, sm, true),
                (1024, 1152),
                "ping-pong high-SM default must stay <= 1024 so 2 slots fit 96 MiB L2"
            );
            assert_eq!(
                resolve_strip_rows_config_for_mode(None, sm, false),
                (2048, 2176),
                "serial high-SM default preserves 2048-row strips"
            );
            assert_eq!(
                resolve_strip_rows_config_for_mode(Some("512"), sm, true),
                (512, 640),
                "explicit override wins over ping-pong default"
            );
        }
    }

    #[test]
    fn ping_pong_layout_fits_ssm_conv_out_f32_and_skews_all_four_buffers() {
        let scratch = DevicePtr(0x1000_0000);
        // Active 1024-row ping-pong slots fit inside SM120 96 MiB L2 (2 * (20 + 12) = 64 MiB).
        let active_two_slot_bytes = 2 * 1024 * (Plan::CONV_DIM + Plan::VALUE_DIM) * 2;
        assert_eq!(active_two_slot_bytes, 64 << 20);
        assert!(active_two_slot_bytes <= 96 << 20);

        for &(max_strip_rows, rows) in &[(640usize, 641usize), (1152, 1153), (1152, 16_384)] {
            let arena_bytes = rows * Plan::QKVZ * 4;
            let layout = compute_ping_pong_layout(
                scratch,
                max_strip_rows,
                Plan::CONV_DIM,
                Plan::VALUE_DIM,
                arena_bytes,
            )
            .expect("2 ping-pong slots must fit ssm_conv_out_f32 for rows > max_strip_rows");

            let qkv_bytes = max_strip_rows * Plan::CONV_DIM * 2;
            let out_bytes = max_strip_rows * Plan::VALUE_DIM * 2;
            let regions = [
                (layout.slot_qkv[0].0, qkv_bytes),
                (layout.slot_out[0].0, out_bytes),
                (layout.slot_qkv[1].0, qkv_bytes),
                (layout.slot_out[1].0, out_bytes),
            ];
            for (i, &(base_a, len_a)) in regions.iter().enumerate() {
                assert!(
                    base_a.is_multiple_of(256),
                    "region {i} must be 256B aligned"
                );
                assert!(
                    base_a.is_multiple_of(L2_DISCARD_LINE_BYTES as u64),
                    "region {i} must be 128B aligned for L2 discard"
                );
                assert!(
                    base_a + len_a as u64 <= scratch.0 + arena_bytes as u64,
                    "region {i} must stay within ssm_conv_out_f32"
                );
                for (j, &(base_b, len_b)) in regions.iter().enumerate().skip(i + 1) {
                    assert!(
                        ranges_disjoint((base_a, len_a), (base_b, len_b)),
                        "regions {i} and {j} must be disjoint"
                    );
                    assert_ne!(
                        base_a % 65_536,
                        base_b % 65_536,
                        "regions {i} and {j} must occupy distinct 64 KiB L2 set offsets"
                    );
                }
            }
        }

        // Undersized or misaligned scratch fails closed.
        assert!(
            compute_ping_pong_layout(
                DevicePtr::NULL,
                1152,
                Plan::CONV_DIM,
                Plan::VALUE_DIM,
                1 << 30
            )
            .is_none()
        );
        assert!(
            compute_ping_pong_layout(
                scratch.offset(64),
                1152,
                Plan::CONV_DIM,
                Plan::VALUE_DIM,
                1 << 30
            )
            .is_none()
        );
        assert!(
            compute_ping_pong_layout(scratch, 1152, Plan::CONV_DIM, Plan::VALUE_DIM, 10 << 20)
                .is_none()
        );
    }

    #[test]
    fn strip_ranges_cover_rows_and_absorb_short_tails() {
        for &(rows, strip_rows) in &[
            (641usize, 512usize),
            (1153, 1024),
            (2048, 1024),
            (4096, 1024),
            (16_384, 1024),
            (16_512, 1024),
        ] {
            let max_strip = strip_rows + STRIP_TAIL_MARGIN;
            let strips = build_strip_ranges(rows, strip_rows, max_strip);
            assert!(strips.len() >= 2, "rows={rows} strip={strip_rows}");
            let mut cursor = 0usize;
            for &(r0, ns) in &strips {
                assert_eq!(r0, cursor);
                assert!(ns > super::super::qwen4_gdn_fi_policy::SHORT_FALLBACK_MAX_ROWS);
                assert!(ns <= max_strip);
                cursor += ns;
            }
            assert_eq!(cursor, rows);
        }
    }

    #[test]
    fn strip_nvfp4_act_layout_fits_ssm_qkvz_and_rejects_aliasing() {
        let rows = 16_384usize;
        let ssm_qkvz_bytes = rows * Plan::QKVZ * 2;
        let base = DevicePtr(0x4000_0000);
        let peers = [
            (DevicePtr(0x1000_0000), rows * Plan::QKVZ * 2),
            (DevicePtr(0x2000_0000), rows * Plan::GATES * 4),
            (DevicePtr(0x8000_0000), rows * Plan::QKVZ * 4),
        ];
        let layout =
            compute_strip_nvfp4_act_layout(base, ssm_qkvz_bytes, rows, Plan::VALUE_DIM, &peers)
                .expect("NVFP4 activation layout must fit inside ssm_qkvz");
        assert!(layout.a_packed.0.is_multiple_of(256));
        assert!(layout.a_scale.0.is_multiple_of(256));
        assert!(layout.alpha_dev.0.is_multiple_of(256));
        assert!(layout.total_bytes < (60 << 20));
        assert!(layout.total_bytes <= ssm_qkvz_bytes);

        // Aliased peer must fail closed.
        let aliased = [(layout.a_scale, 4096)];
        assert!(
            compute_strip_nvfp4_act_layout(base, ssm_qkvz_bytes, rows, Plan::VALUE_DIM, &aliased)
                .is_none()
        );
        // Undersized capacity must fail closed.
        assert!(
            compute_strip_nvfp4_act_layout(
                base,
                layout.total_bytes - 1,
                rows,
                Plan::VALUE_DIM,
                &peers
            )
            .is_none()
        );
    }

    #[test]
    fn strip_overlap_join_guard_joins_on_explicit_join_and_early_drop() {
        use std::sync::Mutex;

        struct RecordingGpu {
            inner: spark_runtime::gpu::mock::MockGpuBackend,
            records: Mutex<Vec<(u64, u64)>>,
            waits: Mutex<Vec<(u64, u64)>>,
        }

        impl RecordingGpu {
            fn new() -> Self {
                Self {
                    inner: spark_runtime::gpu::mock::MockGpuBackend::new(),
                    records: Mutex::new(Vec::new()),
                    waits: Mutex::new(Vec::new()),
                }
            }

            fn take_events(&self) -> (Vec<(u64, u64)>, Vec<(u64, u64)>) {
                (
                    std::mem::take(&mut *self.records.lock().expect("records poisoned")),
                    std::mem::take(&mut *self.waits.lock().expect("waits poisoned")),
                )
            }
        }

        impl GpuBackend for RecordingGpu {
            fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
                self.inner.alloc(bytes)
            }
            fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
                self.inner.alloc_managed(bytes)
            }
            fn free(&self, ptr: DevicePtr) -> Result<()> {
                self.inner.free(ptr)
            }
            fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
                self.inner.copy_h2d(src, dst)
            }
            fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
                self.inner.copy_d2h(src, dst)
            }
            fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
                self.inner.copy_d2d(src, dst, bytes)
            }
            fn launch(
                &self,
                func: KernelHandle,
                grid: [u32; 3],
                block: [u32; 3],
                shared_mem: u32,
                stream: u64,
                params: &mut [*mut std::ffi::c_void],
            ) -> Result<()> {
                self.inner
                    .launch(func, grid, block, shared_mem, stream, params)
            }
            fn synchronize(&self, stream: u64) -> Result<()> {
                self.inner.synchronize(stream)
            }
            fn default_stream(&self) -> u64 {
                self.inner.default_stream()
            }
            fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
                self.inner.kernel(module, func_name)
            }
            fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
                self.inner.memset(ptr, value, bytes)
            }
            fn memset_async(
                &self,
                ptr: DevicePtr,
                value: u8,
                bytes: usize,
                stream: u64,
            ) -> Result<()> {
                self.inner.memset_async(ptr, value, bytes, stream)
            }
            fn total_memory(&self) -> Result<usize> {
                self.inner.total_memory()
            }
            fn free_memory(&self) -> Result<usize> {
                self.inner.free_memory()
            }
            fn record_event(&self, event: u64, stream: u64) -> Result<()> {
                self.records
                    .lock()
                    .expect("records poisoned")
                    .push((event, stream));
                Ok(())
            }
            fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
                self.waits
                    .lock()
                    .expect("waits poisoned")
                    .push((stream, event));
                Ok(())
            }
        }

        let gpu = RecordingGpu::new();
        // No side launch -> no events recorded/waited.
        {
            let guard = StripOverlapJoinGuard::new(&gpu, 11, 22, 99);
            guard.join().expect("empty join");
        }
        assert_eq!(gpu.take_events(), (vec![], vec![]));

        // Normal explicit join records side_done on side_stream and waits on main_stream.
        {
            let mut guard = StripOverlapJoinGuard::new(&gpu, 11, 22, 99);
            guard.mark_launched();
            guard.join().expect("explicit join");
        }
        assert_eq!(gpu.take_events(), (vec![(99, 22)], vec![(11, 99)]));

        // Early error drop still joins side_stream cleanly onto main_stream.
        {
            let mut guard = StripOverlapJoinGuard::new(&gpu, 11, 22, 99);
            guard.mark_launched();
            drop(guard);
        }
        assert_eq!(gpu.take_events(), (vec![(99, 22)], vec![(11, 99)]));
    }

    #[cfg(not(atlas_fi_gdn))]
    #[test]
    fn unavailable_without_aot_falls_back_cleanly() {
        assert_eq!(dtype(), None);
        assert!(!selected());
        reserve_bridge(512).expect("reserve_bridge must be a clean no-op without AOT");
    }
}
