// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::KernelHandle;

pub(super) const K: u32 = 17_408;
pub(super) const DOWN_N: u32 = 128;
pub(super) const SCALE2: f32 = 1.0 / 896.0;
pub(super) const REDZONE: usize = 4096;
pub(super) const SCHEMA: &str = "qwen38-fused-silu-physical-raw-v1";
pub(super) const CUDA_SHA: &str =
    "3231570d94d89d25a20da1faf4584c6d95d184f293cd4cc2c03a2070a3a3d5ba";
pub(super) const WRAPPER_SHA: &str =
    "d0082ca50bfb1ad6d77fd2d9b4370b5378dfcaf9c69662f0a88b8f2b4745d920";
pub(super) const ROUTE_SHA: &str =
    "26a30cfcccad0ecead5cdcfe55b4745ea8208e933e6394ec007de60fc94ea90a";
pub(super) const SILU_SHA: &str =
    "4b82914f131e84d1aea37f7c70152cec0ae12dc63bd9e975c046c6806788bdfc";
pub(super) const PHYSICAL_SHA: &str =
    "61c648cac4fbb81e1cefcee85af32050fc6dba8584219bf4c0397b4b8fdb741a";
pub(super) const DOWN_SHA: &str =
    "0a936fb42fb8b42d3c205a5560ad005c89686c4646a2ef4a0a1af8164737f301";
pub(super) const DIRECT_MODULES: [&str; 4] = [
    "moe_silu_mul",
    "quantize_bf16_to_nvfp4_cutlass",
    "quantize_nvfp4",
    "nvfp4_cutlass",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BinaryIdentity {
    pub(super) path: String,
    pub(super) sha256: String,
    pub(super) profile: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BundleIdentity {
    pub(super) target: String,
    pub(super) module_count: usize,
    pub(super) sha256: String,
    pub(super) direct_modules: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    pub(super) m: u32,
    pub(super) padded: u32,
    pub(super) elements: usize,
    pub(super) packed: usize,
    pub(super) scales: usize,
    pub(super) tail: usize,
    pub(super) reps: usize,
}

impl Plan {
    pub(super) fn checked(m: u32, k: u32, reps: usize, scale2: f32) -> Result<Self> {
        ensure!(
            matches!(m, 2_079 | 8_192) && k == K,
            "requires exact M=2079|8192,K=17408"
        );
        ensure!(
            scale2.is_finite() && scale2 > 0.0,
            "scale2 must be finite and positive"
        );
        let minimum = if m == 2_079 { 21 } else { 31 };
        ensure!(
            reps >= minimum,
            "M={m} requires at least {minimum} timing pairs"
        );
        let padded = m.checked_add(127).context("row padding overflow")? / 128 * 128;
        let elements = usize::try_from(m)?
            .checked_mul(usize::try_from(k)?)
            .context("element overflow")?;
        let scales = usize::try_from(padded)?
            .checked_mul(usize::try_from(k / 16)?)
            .context("scale overflow")?;
        let tail = usize::try_from(padded - m)? * usize::try_from(k / 16)?;
        ensure!(
            m != 2_079 || tail == 105_536,
            "M2079 physical tail extent changed"
        );
        ensure!(m != 8_192 || tail == 0, "M8192 must not have a padded tail");
        Ok(Self {
            m,
            padded,
            elements,
            packed: elements / 2,
            scales,
            tail,
            reps,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct Kernels {
    pub(super) silu: KernelHandle,
    pub(super) parent: KernelHandle,
    pub(super) fused: KernelHandle,
    pub(super) down: KernelHandle,
}
