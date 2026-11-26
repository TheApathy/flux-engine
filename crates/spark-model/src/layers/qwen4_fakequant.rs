// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off FAKE activation quantization for the Flash-Next prefill quality
//! gate. `ATLAS_QWEN4_PREFILL_FAKEQUANT=<spec>` rounds the BF16 operands that
//! feed chosen prefill GEMMs to FP8 / MXFP8 / NVFP4 and back to BF16
//! (`qwen4_prefill_fakequant.cu`), so the quality gate can price FP8/FP4
//! activations before anyone writes an FP8/FP4 GEMM. The GEMMs are unchanged.
//!
//! Spec grammar: comma-separated `format[:scope[+scope...]]` items, later items
//! override earlier ones; `off`/`0`/empty disables. `file:<path>` re-reads the
//! spec from `<path>` whenever its mtime changes, so one server load can run
//! several gate arms.
//! - formats: `fp8` (E4M3, per-row FP32 scale), `fp8b128` (E4M3, 1x128 FP32
//!   scales), `mxfp8` (E4M3, UE8M0 per 32: b12x w4a8), `nvfp4` (E2M1, E4M3
//!   per 16, per-tensor FP32 global scale), `nvfp4row` (per-row global scale),
//!   `e2m1raw` (no scales; a deliberately bad control).
//! - MoE kernel switches (need the V2 compact path, any level >= 1): `v7` runs the
//!   V7 kernels (128-row planner, fused gate/up/SwiGLU, exact weights) instead of the
//!   loaded set; `w4a8` selects the routed-expert W4A8 kernels (`spec.moe_kernel_w4a8`);
//!   `b12xw` runs the twins that round the expert weights exactly as
//!   b12x's w4a8 NVFP4 path does (UE8M0/32 exponent + E4M3 residual,
//!   `e4m3(e2m1 * residual)`); `b12x` is `mxfp8:moe` plus `b12xw`, i.e. b12x's w4a8
//!   numerics end to end. `sharedv7` runs the shared expert through the V7 kernels
//!   (same path as ATLAS_QWEN4_PREFILL_SHARED_V7=1).
//! - scopes (default `moe`): `moe_in` (routed gate/up input), `moe_down`
//!   (SwiGLU output into down), `moe` (both), `proj` (SSM/attention fast
//!   projections), `hc` (mHC fast GEMMs, i.e. K or N equal to the 4-stream
//!   residual width), `dense` (`proj+hc`), `all`.
//!
//! The router, the shared expert and every non-fast-projection op keep BF16.

use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_FAKEQUANT";
const HC_RESIDUAL_WIDTH: usize = 10240;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Fp8Row,
    Fp8B128,
    Mxfp8,
    Nvfp4,
    Nvfp4Row,
    E2m1Raw,
}

impl Format {
    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "fp8" => Self::Fp8Row,
            "fp8b128" => Self::Fp8B128,
            "mxfp8" => Self::Mxfp8,
            "nvfp4" => Self::Nvfp4,
            "nvfp4row" => Self::Nvfp4Row,
            "e2m1raw" => Self::E2m1Raw,
            other => bail!(
                "{SELECTOR}: unknown format {other:?} (fp8, fp8b128, mxfp8, nvfp4, nvfp4row, e2m1raw)"
            ),
        })
    }

    /// `(mode, block)` of `qwen4_fakequant_rows`.
    fn kernel_mode(self) -> (u32, u32) {
        match self {
            Self::Fp8Row => (0, 0),
            Self::Fp8B128 => (0, 128),
            Self::Mxfp8 => (1, 32),
            Self::Nvfp4 | Self::Nvfp4Row => (2, 16),
            Self::E2m1Raw => (3, 0),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Spec {
    pub moe_in: Option<Format>,
    pub moe_down: Option<Format>,
    pub proj: Option<Format>,
    pub hc: Option<Format>,
    pub moe_weights_b12x: bool,
    pub moe_kernel_v7: bool,
    pub moe_shared_v7: bool,
    /// Routed experts through the V10 W4A8 kernels (b12x numerics, MXFP8 activations
    /// quantised inside the kernels): the per-dispatch twin of ATLAS_QWEN4_PREFILL_MOE_W4A8=1.
    pub moe_kernel_w4a8: bool,
}

impl Spec {
    fn any(self) -> bool {
        self != Self::default()
    }
}

pub(crate) fn parse(value: &str) -> Result<Spec> {
    let mut spec = Spec::default();
    let value = value.trim();
    if value.is_empty() || value == "0" || value == "off" {
        return Ok(spec);
    }
    for item in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (format, scopes) = item.split_once(':').unwrap_or((item, "moe"));
        let format = format.trim();
        if format == "sharedv7" {
            anyhow::ensure!(
                scopes.split('+').all(|s| s.trim() == "moe"),
                "{SELECTOR}: sharedv7 selects the shared-expert kernels (scope moe)"
            );
            spec.moe_shared_v7 = true;
            continue;
        }
        if format == "w4a8" {
            anyhow::ensure!(
                scopes.split('+').all(|s| s.trim() == "moe"),
                "{SELECTOR}: w4a8 selects the routed-expert kernels (scope moe)"
            );
            spec.moe_kernel_w4a8 = true;
            continue;
        }
        if format == "v7" {
            anyhow::ensure!(
                scopes.split('+').all(|s| s.trim() == "moe"),
                "{SELECTOR}: v7 selects the routed-expert kernels (scope moe)"
            );
            spec.moe_kernel_v7 = true;
            continue;
        }
        if format == "b12xw" || format == "b12x" {
            anyhow::ensure!(
                scopes.split('+').all(|s| s.trim() == "moe"),
                "{SELECTOR}: {format} applies to the routed experts only (scope moe)"
            );
            spec.moe_weights_b12x = true;
            if format == "b12x" {
                spec.moe_in = Some(Format::Mxfp8);
                spec.moe_down = Some(Format::Mxfp8);
            }
            continue;
        }
        let format = Format::parse(format)?;
        for scope in scopes.split('+').map(str::trim) {
            match scope {
                "moe" => {
                    spec.moe_in = Some(format);
                    spec.moe_down = Some(format);
                }
                "moe_in" => spec.moe_in = Some(format),
                "moe_down" => spec.moe_down = Some(format),
                "proj" => spec.proj = Some(format),
                "hc" => spec.hc = Some(format),
                "dense" => {
                    spec.proj = Some(format);
                    spec.hc = Some(format);
                }
                "all" => {
                    spec.moe_in = Some(format);
                    spec.moe_down = Some(format);
                    spec.proj = Some(format);
                    spec.hc = Some(format);
                }
                other => bail!(
                    "{SELECTOR}: unknown scope {other:?} (moe, moe_in, moe_down, proj, hc, dense, all)"
                ),
            }
        }
    }
    Ok(spec)
}

struct FileState {
    path: String,
    mtime: Option<SystemTime>,
    spec: Spec,
}

/// The active spec. Static selectors are parsed once; `file:` selectors are
/// re-parsed whenever the file's mtime changes (a malformed file fails the
/// prefill that reads it).
pub(crate) fn current() -> Result<Spec> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let Some(raw) = RAW.get_or_init(|| std::env::var(SELECTOR).ok()) else {
        return Ok(Spec::default());
    };
    if let Some(path) = raw.strip_prefix("file:") {
        static FILE: Mutex<Option<FileState>> = Mutex::new(None);
        let mtime = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .with_context(|| format!("{SELECTOR}: stat {path}"))?;
        let mut guard = FILE.lock().unwrap();
        if let Some(state) = guard.as_ref()
            && state.path == path
            && state.mtime == Some(mtime)
        {
            return Ok(state.spec);
        }
        let text =
            std::fs::read_to_string(path).with_context(|| format!("{SELECTOR}: read {path}"))?;
        let spec = parse(&text)?;
        tracing::info!(
            "QWEN4_FAKEQUANT spec={:?} source=file:{path} parsed={spec:?}",
            text.trim()
        );
        *guard = Some(FileState {
            path: path.to_string(),
            mtime: Some(mtime),
            spec,
        });
        return Ok(spec);
    }
    static STATIC: OnceLock<Result<Spec, String>> = OnceLock::new();
    STATIC
        .get_or_init(|| {
            let spec = parse(raw).map_err(|e| e.to_string());
            if let Ok(spec) = &spec {
                tracing::info!("QWEN4_FAKEQUANT spec={raw:?} parsed={spec:?}");
            }
            spec
        })
        .clone()
        .map_err(|e| anyhow::anyhow!(e))
}

#[derive(Clone, Copy)]
struct Kernels {
    rows: KernelHandle,
    absmax: KernelHandle,
}

fn kernels(gpu: &dyn GpuBackend) -> Result<Kernels> {
    static K: OnceLock<Result<(KernelHandle, KernelHandle), String>> = OnceLock::new();
    let (rows, absmax) = K
        .get_or_init(|| {
            let load = |f: &str| {
                gpu.kernel("qwen4_prefill_fakequant", f)
                    .map_err(|e| e.to_string())
            };
            Ok((
                load("qwen4_fakequant_rows")?,
                load("qwen4_fakequant_absmax")?,
            ))
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("{SELECTOR}: kernel unavailable: {e}"))?;
    anyhow::ensure!(
        rows.0 != 0 && absmax.0 != 0,
        "{SELECTOR}: NULL kernel handle"
    );
    Ok(Kernels { rows, absmax })
}

/// Grow-only device scratch, one per purpose. Every caller runs on the single
/// prefill stream, so reuse is stream ordered; a grown-out block is leaked
/// rather than freed while it may still be in flight.
fn scratch(
    gpu: &dyn GpuBackend,
    slot: &Mutex<Option<(DevicePtr, usize)>>,
    bytes: usize,
) -> Result<DevicePtr> {
    let mut guard = slot.lock().unwrap();
    if let Some((ptr, cap)) = *guard
        && cap >= bytes
    {
        return Ok(ptr);
    }
    let cap = bytes.next_multiple_of(1 << 20);
    let ptr = gpu
        .alloc(cap)
        .with_context(|| format!("{SELECTOR}: alloc {cap} B scratch"))?;
    tracing::info!("QWEN4_FAKEQUANT scratch allocated bytes={cap}");
    *guard = Some((ptr, cap));
    Ok(ptr)
}

static MOE_BACKUP: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
static PROJ_OPERAND: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
static ABSMAX: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);

/// `dst[rows][cols] = dequant(quant(src))`; `src == dst` is allowed.
fn quantize(
    gpu: &dyn GpuBackend,
    format: Format,
    src: DevicePtr,
    dst: DevicePtr,
    rows: usize,
    cols: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        rows > 0 && cols > 0,
        "{SELECTOR}: empty operand {rows}x{cols}"
    );
    let k = kernels(gpu)?;
    let tensor_absmax = if format == Format::Nvfp4 {
        let bits = scratch(gpu, &ABSMAX, 4)?;
        gpu.memset_async(bits, 0, 4, stream)?;
        KernelLaunch::new(gpu, k.absmax)
            .grid([(rows * cols).div_ceil(256 * 16).clamp(1, 1024) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_u64((rows * cols) as u64)
            .arg_ptr(bits)
            .launch(stream)?;
        bits
    } else {
        DevicePtr::NULL
    };
    let (mode, block) = format.kernel_mode();
    KernelLaunch::new(gpu, k.rows)
        .grid([rows.min(8192) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(rows as u32)
        .arg_u32(cols as u32)
        .arg_u32(mode)
        .arg_u32(block)
        .arg_ptr(tensor_absmax)
        .launch(stream)
}

/// Restores the routed-expert input that [`moe_in_begin`] quantized in place.
#[must_use]
pub(crate) struct MoeInRestore {
    input: DevicePtr,
    backup: DevicePtr,
    bytes: usize,
}

/// Quantizes the routed gate/up input `[rows][2560]` in place, keeping a BF16
/// backup: the router already consumed it, but the shared-expert gate blend
/// reads it again after the routed GEMMs.
pub(crate) fn moe_in_begin(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    rows: usize,
    cols: usize,
    stream: u64,
) -> Result<Option<MoeInRestore>> {
    let Some(format) = current()?.moe_in else {
        return Ok(None);
    };
    let bytes = rows * cols * 2;
    let backup = scratch(gpu, &MOE_BACKUP, bytes)?;
    gpu.copy_d2d_async(input, backup, bytes, stream)?;
    quantize(gpu, format, input, input, rows, cols, stream)?;
    Ok(Some(MoeInRestore {
        input,
        backup,
        bytes,
    }))
}

pub(crate) fn moe_in_end(
    gpu: &dyn GpuBackend,
    restore: Option<MoeInRestore>,
    stream: u64,
) -> Result<()> {
    match restore {
        Some(r) => gpu.copy_d2d_async(r.backup, r.input, r.bytes, stream),
        None => Ok(()),
    }
}

/// Quantizes the SwiGLU output `[pairs][inter]` that feeds the down GEMM, in place.
pub(crate) fn moe_down(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    pairs: usize,
    inter: usize,
    stream: u64,
) -> Result<()> {
    match current()?.moe_down {
        Some(format) => quantize(gpu, format, act, act, pairs, inter, stream),
        None => Ok(()),
    }
}

/// The A operand a fast projection GEMM `[m][k] x [n][k]^T` should read: `act`
/// itself, or a fake-quantized copy (the source may have other readers).
pub(crate) fn proj_operand(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<DevicePtr> {
    let spec = current()?;
    if !spec.any() {
        return Ok(act);
    }
    let hc = k == HC_RESIDUAL_WIDTH || n == HC_RESIDUAL_WIDTH;
    let Some(format) = (if hc { spec.hc } else { spec.proj }) else {
        return Ok(act);
    };
    let dst = scratch(gpu, &PROJ_OPERAND, m * k * 2)?;
    quantize(gpu, format, act, dst, m, k, stream)?;
    Ok(dst)
}

/// `ATLAS_QWEN4_MOE_HIST_DUMP=<file>` (diagnostic, default off): appends each
/// compact routed dispatch's expert offsets — the exact per-expert row
/// histogram the GEMM grid is built from — as little-endian u32 records
/// `[0x54534948 "HIST", seq, rows, experts + 1, offsets...]`. `seq` counts
/// dispatches in this process (48 per prefill chunk on Flash-Next, in layer
/// order). Call it after the stream has been synchronized.
pub(crate) fn moe_hist_dump(
    gpu: &dyn GpuBackend,
    offsets: DevicePtr,
    experts: usize,
    rows: usize,
    stream: u64,
) -> Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU32, Ordering};
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let Some(path) = PATH.get_or_init(|| std::env::var("ATLAS_QWEN4_MOE_HIST_DUMP").ok()) else {
        return Ok(());
    };
    let mut bytes = vec![0u8; (experts + 1) * 4];
    gpu.copy_d2h_on_stream(offsets, &mut bytes, stream)?;
    let header = [
        0x5453_4948u32,
        SEQ.fetch_add(1, Ordering::Relaxed),
        rows as u32,
        (experts + 1) as u32,
    ];
    let mut record: Vec<u8> = header.iter().flat_map(|w| w.to_le_bytes()).collect();
    record.extend_from_slice(&bytes);
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(&record))
        .with_context(|| format!("ATLAS_QWEN4_MOE_HIST_DUMP: append {path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_specs() {
        assert_eq!(parse("").unwrap(), Spec::default());
        assert_eq!(parse("off").unwrap(), Spec::default());
        assert_eq!(parse("0").unwrap(), Spec::default());
        let moe = parse("fp8").unwrap();
        assert_eq!(moe.moe_in, Some(Format::Fp8Row));
        assert_eq!(moe.moe_down, Some(Format::Fp8Row));
        assert_eq!(moe.proj, None);
        assert_eq!(moe.hc, None);
        let mixed = parse("mxfp8:moe, fp8:proj").unwrap();
        assert_eq!(mixed.moe_down, Some(Format::Mxfp8));
        assert_eq!(mixed.proj, Some(Format::Fp8Row));
        assert_eq!(mixed.hc, None);
        let all = parse("nvfp4:all").unwrap();
        assert_eq!(all.hc, Some(Format::Nvfp4));
        let split = parse("fp8:moe_in+dense,nvfp4:moe_down").unwrap();
        assert_eq!(split.moe_in, Some(Format::Fp8Row));
        assert_eq!(split.moe_down, Some(Format::Nvfp4));
        assert_eq!(split.hc, Some(Format::Fp8Row));
        assert!(parse("fp16").is_err());
        assert!(parse("fp8:mlp").is_err());
        let b12x = parse("b12x").unwrap();
        assert!(b12x.moe_weights_b12x);
        assert_eq!(b12x.moe_in, Some(Format::Mxfp8));
        assert_eq!(b12x.proj, None);
        let w_only = parse("b12xw:moe").unwrap();
        assert!(w_only.moe_weights_b12x);
        assert_eq!(w_only.moe_in, None);
        assert!(parse("b12xw:proj").is_err());
        let v7 = parse("v7,fp8:proj").unwrap();
        assert!(v7.moe_kernel_v7 && !v7.moe_weights_b12x);
        assert_eq!(v7.proj, Some(Format::Fp8Row));
        assert_eq!(v7.moe_in, None);
        let w4a8 = parse("w4a8,fp8:proj").unwrap();
        assert!(w4a8.moe_kernel_w4a8 && !w4a8.moe_kernel_v7);
        assert_eq!(w4a8.moe_in, None);
        assert!(parse("w4a8:proj").is_err());
    }

    #[test]
    fn kernel_modes_match_the_cuda_contract() {
        assert_eq!(Format::Fp8Row.kernel_mode(), (0, 0));
        assert_eq!(Format::Fp8B128.kernel_mode(), (0, 128));
        assert_eq!(Format::Mxfp8.kernel_mode(), (1, 32));
        assert_eq!(Format::Nvfp4.kernel_mode(), (2, 16));
        assert_eq!(Format::Nvfp4Row.kernel_mode(), (2, 16));
        assert_eq!(Format::E2m1Raw.kernel_mode(), (3, 0));
    }
}
