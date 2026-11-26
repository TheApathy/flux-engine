// SPDX-License-Identifier: AGPL-3.0-only

//! `fused_v2` launches of the MXFP8 decode GEMV (`dense_gemv_mxfp8.cu`,
//! `dense_gemv_mxfp8_v2_*`). Bit-identical to the reference
//! `dense_gemv_mxfp8` per row (see the kernel header for the argument):
//! hardware E4M3 decode on chunks where the weight product is provably
//! exact, 32 or 64 threads per output row, and fused qkv|z, k|v and
//! gate|up(+silu_mul) launches.
//!
//! `ATLAS_QWEN4_DENSE_FP8_GEMV=ref` keeps the reference kernel (one launch
//! per matrix); unset or `fused_v2` selects this path.
//! `ATLAS_QWEN4_DENSE_FP8_TPO=32|64` pins threads per output; unset or
//! `auto` picks from K. Both are read only when the FP8 arm is engaged.

use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::mxfp8::MXFP8_BLOCK;
use super::proj::{GEMV_MODULE, MxFp8Proj};

pub(crate) const GEMV_VARIANT_ENV: &str = "ATLAS_QWEN4_DENSE_FP8_GEMV";
pub(crate) const TPO_ENV: &str = "ATLAS_QWEN4_DENSE_FP8_TPO";

const BLOCK: u32 = 256;
/// Largest K served with one warp per row under `auto` (K=2560 hidden and
/// K=640 shared down: <= 3 chunk iterations per reference lane).
pub(crate) const TPO32_MAX_K: u32 = 3072;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemvVariant {
    Reference,
    FusedV2,
}

impl GemvVariant {
    pub(crate) fn name(self) -> &'static str {
        match self {
            GemvVariant::Reference => "ref",
            GemvVariant::FusedV2 => "fused_v2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TpoPolicy {
    Auto,
    Fixed(u32),
}

pub(crate) fn parse_variant(raw: Option<&str>) -> Result<GemvVariant> {
    match raw {
        None | Some("") | Some("fused_v2") => Ok(GemvVariant::FusedV2),
        Some("ref") => Ok(GemvVariant::Reference),
        Some(v) => bail!("{GEMV_VARIANT_ENV} must be fused_v2 or ref, got {v:?}"),
    }
}

pub(crate) fn parse_tpo(raw: Option<&str>) -> Result<TpoPolicy> {
    match raw {
        None | Some("") | Some("auto") => Ok(TpoPolicy::Auto),
        Some("32") => Ok(TpoPolicy::Fixed(32)),
        Some("64") => Ok(TpoPolicy::Fixed(64)),
        Some(v) => bail!("{TPO_ENV} must be auto, 32 or 64, got {v:?}"),
    }
}

/// Threads cooperating on one output row for a `[*, k]` matrix.
pub(crate) fn threads_per_out(policy: TpoPolicy, k: u32) -> u32 {
    match policy {
        TpoPolicy::Fixed(t) => t,
        TpoPolicy::Auto if k <= TPO32_MAX_K => 32,
        TpoPolicy::Auto => 64,
    }
}

fn env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(v) => Ok(Some(v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} is not valid Unicode"),
    }
}

fn read_selection() -> Result<(GemvVariant, TpoPolicy)> {
    let variant = parse_variant(env(GEMV_VARIANT_ENV)?.as_deref())?;
    let tpo = parse_tpo(env(TPO_ENV)?.as_deref())?;
    Ok((variant, tpo))
}

/// Read once per process so every layer sees one answer.
pub(crate) fn selected() -> Result<(GemvVariant, TpoPolicy)> {
    static SEL: OnceLock<Result<(GemvVariant, TpoPolicy), String>> = OnceLock::new();
    SEL.get_or_init(|| read_selection().map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// The four `fused_v2` entry points, resolved once per layer at load.
#[derive(Debug, Clone, Copy)]
pub(crate) struct V2Kernels {
    pub(crate) seg_t32: KernelHandle,
    pub(crate) seg_t64: KernelHandle,
    pub(crate) gus_t32: KernelHandle,
    pub(crate) gus_t64: KernelHandle,
    pub(crate) tpo: TpoPolicy,
}

impl V2Kernels {
    pub(crate) fn resolve(gpu: &dyn GpuBackend, tpo: TpoPolicy) -> Result<Self> {
        let get = |func: &str| -> Result<KernelHandle> {
            let h = gpu.kernel(GEMV_MODULE, func).with_context(|| {
                format!("{GEMV_VARIANT_ENV}=fused_v2 needs {func} in this target's PTX")
            })?;
            ensure!(h.0 != 0, "{func} resolved a null kernel handle");
            Ok(h)
        };
        Ok(Self {
            seg_t32: get("dense_gemv_mxfp8_v2_t32")?,
            seg_t64: get("dense_gemv_mxfp8_v2_t64")?,
            gus_t32: get("dense_gemv_mxfp8_v2_gus_t32")?,
            gus_t64: get("dense_gemv_mxfp8_v2_gus_t64")?,
            tpo,
        })
    }

    fn tpo_for(&self, k: u32) -> u32 {
        threads_per_out(self.tpo, k)
    }

    /// Rows `[a.n]` of `a` into `out_a`, then (optionally) `[b.n]` rows of `b`
    /// into `out_b`, in one launch. Both share `k` and the activation.
    pub(crate) fn seg(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        (a, out_a): (&MxFp8Proj, DevicePtr),
        b: Option<(&MxFp8Proj, DevicePtr)>,
        stream: u64,
    ) -> Result<()> {
        let k = a.k();
        let (b, out_b, n1) = match b {
            Some((p, out)) => {
                ensure!(
                    p.k() == k,
                    "fused MXFP8 segments need one k, got {k} and {}",
                    p.k()
                );
                (p, out, p.n())
            }
            None => (a, out_a, 0),
        };
        let total = a
            .n()
            .checked_add(n1)
            .context("fused MXFP8 rows overflow u32")?;
        let tpo = self.tpo_for(k);
        let kernel = if tpo == 32 {
            self.seg_t32
        } else {
            self.seg_t64
        };
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(total, BLOCK / tpo), 1, 1])
            .block([BLOCK, 1, 1])
            .arg_ptr(input)
            .arg_ptr(a.weight())
            .arg_ptr(a.scale())
            .arg_ptr(out_a)
            .arg_u32(a.n())
            .arg_ptr(b.weight())
            .arg_ptr(b.scale())
            .arg_ptr(out_b)
            .arg_u32(n1)
            .arg_u32(k)
            .arg_u32(k / MXFP8_BLOCK as u32)
            .launch(stream)
    }

    /// `act = bf16(silu(gate x) * up x)` and `up_out = up x` in one launch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gate_up_silu(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        gate: &MxFp8Proj,
        up: &MxFp8Proj,
        act: DevicePtr,
        up_out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            gate.n() == up.n() && gate.k() == up.k(),
            "fused gate/up MXFP8 shapes differ: [{},{}] vs [{},{}]",
            gate.n(),
            gate.k(),
            up.n(),
            up.k()
        );
        let (n, k) = (gate.n(), gate.k());
        let tpo = self.tpo_for(k);
        let kernel = if tpo == 32 {
            self.gus_t32
        } else {
            self.gus_t64
        };
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n, BLOCK / tpo / 2), 1, 1])
            .block([BLOCK, 1, 1])
            .arg_ptr(input)
            .arg_ptr(gate.weight())
            .arg_ptr(gate.scale())
            .arg_ptr(up.weight())
            .arg_ptr(up.scale())
            .arg_ptr(act)
            .arg_ptr(up_out)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(k / MXFP8_BLOCK as u32)
            .launch(stream)
    }
}

/// The load receipt line for one K (pure, so its format is testable).
pub(crate) fn receipt_line(variant: GemvVariant, tpo: Option<u32>, k: u32) -> String {
    match tpo {
        Some(t) => format!(
            "DENSE_FP8_GEMV variant={} threads_per_out={t} k={k}",
            variant.name()
        ),
        None => format!(
            "DENSE_FP8_GEMV variant={} threads_per_out=64 k={k}",
            variant.name()
        ),
    }
}

/// One INFO line per distinct K at load.
pub(crate) fn emit_receipt(v2: Option<&V2Kernels>, ks: &[u32]) {
    static SEEN: Mutex<BTreeSet<u32>> = Mutex::new(BTreeSet::new());
    let mut seen = SEEN.lock().unwrap_or_else(|p| p.into_inner());
    for &k in ks {
        if seen.insert(k) {
            let line = match v2 {
                Some(v) => receipt_line(GemvVariant::FusedV2, Some(v.tpo_for(k)), k),
                None => receipt_line(GemvVariant::Reference, None, k),
            };
            tracing::info!("{line}");
        }
    }
}

#[cfg(test)]
#[path = "gemv_v2_tests.rs"]
pub(crate) mod tests;
