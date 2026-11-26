// SPDX-License-Identifier: AGPL-3.0-only

//! Side tables that pair each NVFP4 prefill weight with its checkpoint-native
//! source. Keyed by the NVFP4 weight's device pointers and shape, so the layer
//! structs, the exact-prefill `ensure!` chain and every decode path keep the
//! NVFP4 copy they use today; prefill call sites only consult the table.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Module;
use super::e4m3::{ConversionStats, quantize_pow2_tensor};
use crate::weight_map::QuantizedWeight;

/// NVFP4 packed and scale pointers, `scale2` bits and shape. `scale2` guards a
/// freed-and-reused address of the same shape (a reloaded model re-derives it
/// from different weights); qwen4 never frees these copies while loaded.
pub(crate) type Key = (u64, u64, u32, usize, usize);

pub(crate) fn key(weight: &QuantizedWeight, n: usize, k: usize) -> Key {
    (
        weight.weight.0,
        weight.weight_scale.0,
        weight.weight_scale_2.to_bits(),
        n,
        k,
    )
}

/// Resident E4M3 rows (destination order) and their device f32 tensor scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NativeFp8 {
    pub w8: DevicePtr,
    pub scale: DevicePtr,
    pub scale_value: f32,
    pub qg: Option<(usize, usize)>,
    pub module: Module,
}

#[derive(Default)]
pub(crate) struct Registry {
    fp8: HashMap<Key, NativeFp8>,
    bf16: HashMap<Key, (DevicePtr, Module)>,
    router: HashSet<u64>,
    registered: BTreeMap<&'static str, usize>,
    pub(crate) fp8_bytes: usize,
    pub(crate) bf16_alias_bytes: usize,
    pub(crate) stats: ConversionStats,
    pub(crate) worst_rel_fro: f64,
}

/// Rows, then the f32 scale on a 256-byte boundary (as the FP8 weight cache).
fn fp8_alloc_bytes(n: usize, k: usize) -> (usize, usize) {
    let rows = (n * k).next_multiple_of(256);
    (rows, rows + 256)
}

fn read_bf16(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    elements: usize,
    stream: u64,
) -> Result<Vec<u16>> {
    let mut host = vec![0u8; elements * 2];
    gpu.synchronize(stream)?;
    gpu.copy_d2h(src, &mut host)?;
    Ok(host
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

impl Registry {
    fn count(&mut self, module: Module) {
        *self.registered.entry(module.name()).or_insert(0) += 1;
    }

    pub(crate) fn registered(&self) -> &BTreeMap<&'static str, usize> {
        &self.registered
    }

    pub(crate) fn has_family(&self, family: super::Family) -> bool {
        Module::ALL
            .iter()
            .any(|m| m.family() == family && self.registered.contains_key(m.name()))
    }

    /// Build the E4M3 copy of `bf16` (the exact MXFP8 dequantization the NVFP4
    /// copy was quantized from) and pair it with `nvfp4`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn insert_mxfp8(
        &mut self,
        gpu: &dyn GpuBackend,
        module: Module,
        nvfp4: &QuantizedWeight,
        bf16: DevicePtr,
        n: usize,
        k: usize,
        qg: Option<(usize, usize)>,
        stream: u64,
    ) -> Result<NativeFp8> {
        ensure!(
            module.is_fp8(),
            "{} is not an MXFP8-shipped module",
            module.name()
        );
        ensure!(
            n > 0 && k > 0 && n.is_multiple_of(16) && k.is_multiple_of(16),
            "{}: unsupported native FP8 shape [{n}, {k}]",
            module.name()
        );
        ensure!(
            !bf16.is_null() && !nvfp4.weight.is_null(),
            "{}: null weight",
            module.name()
        );
        let key = key(nvfp4, n, k);
        if self.fp8.contains_key(&key) {
            bail!("{}: NVFP4 weight registered twice", module.name());
        }
        let q = quantize_pow2_tensor(&read_bf16(gpu, bf16, n * k, stream)?, n, k, qg)?;
        let (rows_bytes, bytes) = fp8_alloc_bytes(n, k);
        let base = gpu.alloc(bytes)?;
        gpu.copy_h2d(&q.bytes, base)?;
        gpu.copy_h2d(&q.scale.to_le_bytes(), base.offset(rows_bytes))?;
        let entry = NativeFp8 {
            w8: base,
            scale: base.offset(rows_bytes),
            scale_value: q.scale,
            qg,
            module,
        };
        self.fp8.insert(key, entry);
        self.fp8_bytes += bytes;
        self.stats = self.stats.merge(q.stats);
        self.worst_rel_fro = self.worst_rel_fro.max(q.stats.rel_fro());
        self.count(module);
        Ok(entry)
    }

    /// Pair `nvfp4` with the resident BF16 tensor it was quantized from.
    pub(crate) fn insert_bf16(
        &mut self,
        module: Module,
        nvfp4: &QuantizedWeight,
        bf16: DevicePtr,
        n: usize,
        k: usize,
    ) -> Result<()> {
        ensure!(
            !module.is_fp8(),
            "{} is MXFP8-shipped, not BF16",
            module.name()
        );
        ensure!(
            !bf16.is_null() && !nvfp4.weight.is_null(),
            "{}: null weight",
            module.name()
        );
        if self.bf16.insert(key(nvfp4, n, k), (bf16, module)).is_some() {
            bail!("{}: NVFP4 weight registered twice", module.name());
        }
        self.bf16_alias_bytes += n * k * 2;
        self.count(module);
        Ok(())
    }

    pub(crate) fn insert_router(&mut self, bf16_gate: DevicePtr, n: usize, k: usize) -> Result<()> {
        ensure!(!bf16_gate.is_null(), "router: null BF16 gate");
        if !self.router.insert(bf16_gate.0) {
            bail!("router: BF16 gate registered twice");
        }
        self.bf16_alias_bytes += n * k * 2;
        self.count(Module::Router);
        Ok(())
    }

    /// The native E4M3 operand for an FP8 prefill projection, if registered.
    /// A registered weight requested with a different gated-Q permutation is
    /// an error: its rows were laid out for the registered one.
    pub(crate) fn fp8(
        &self,
        weight: &QuantizedWeight,
        n: usize,
        k: usize,
        qg: Option<(usize, usize)>,
    ) -> Result<Option<NativeFp8>> {
        let Some(entry) = self.fp8.get(&key(weight, n, k)).copied() else {
            return Ok(None);
        };
        let qg = qg.filter(|&(h, _)| h > 0);
        ensure!(
            entry.qg == qg,
            "{}: native rows were built for gated-Q {:?}, prefill asked for {qg:?}",
            entry.module.name(),
            entry.qg
        );
        Ok(Some(entry))
    }

    pub(crate) fn bf16(
        &self,
        weight: &QuantizedWeight,
        n: usize,
        k: usize,
    ) -> Option<(DevicePtr, Module)> {
        self.bf16.get(&key(weight, n, k)).copied()
    }

    pub(crate) fn router(&self, bf16_gate: DevicePtr) -> bool {
        self.router.contains(&bf16_gate.0)
    }
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
