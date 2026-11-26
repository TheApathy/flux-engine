// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_PREFILL_NATIVE=1`, family `shared`: the Flash-Next shared
//! expert (gate/up `[640, 2560]`, down `[2560, 640]`, 48 target layers,
//! shipped MXFP8 block-32) runs prefill from its exact BF16 dequantization on
//! cuBLASLt BF16 GEMMs with BF16 activations, instead of the load-time NVFP4
//! re-quantization (W4A8 with MXFP8 activations under
//! `ATLAS_QWEN4_PREFILL_SHARED_W4A8=1`, W4A16/FP8 otherwise).
//!
//! Why BF16 and not the E4M3 pow2-tensor FP8 GEMM the GDN/attention families
//! use: the only FP8 GEMM with a device tensor scale (`fp8_gemm_device_scaled`)
//! takes per-tensor E4M3 activations. On the checkpoint the weight side of that
//! route is 1.7e-6..7.7e-6, but the per-tensor E4M3 activations (input and the
//! SwiGLU intermediate) leave the shared-expert output at ~5e-2 relative error,
//! against 3.2e-3 for BF16 weights + BF16 activations and 1.1e-1..1.8e-1 for
//! NVFP4 x MXFP8. The BF16 operand reproduces every MXFP8 value exactly, so the
//! weight error is 0.
//!
//! Decode is untouched: the NVFP4 shared copy, its transposed tables and every
//! decode/admission path keep what they use today. The native copies are keyed
//! by the layer's resident `shared_expert_gate` pointer (the unified loader
//! frees and nulls the original NVFP4 shared pointers, so those are not keys).
//! Receipts: `QWEN4_PREFILL_NATIVE engaged family=shared` at load,
//! `QWEN4_PREFILL_NATIVE_CALL family=shared` on the first call.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::{WeightDtype, WeightStore};

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_NATIVE";
pub(crate) const FAMILY: &str = "shared";
const PATH: &str = "bf16_cublaslt_exact_mxfp8";
const PROJECTIONS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// Shared selector for all native prefill weight families.
pub(crate) fn selected() -> Result<bool> {
    super::selected()
}

fn on() -> bool {
    selected().unwrap_or(false)
}

/// `[rows, k]` of gate, up, down.
pub(crate) fn shapes(hidden: usize, inter: usize) -> [(usize, usize); 3] {
    [(inter, hidden), (inter, hidden), (hidden, inter)]
}

/// Resident bytes of one layer's three BF16 tensors.
pub(crate) fn layer_bytes(hidden: usize, inter: usize) -> usize {
    shapes(hidden, inter).iter().map(|(n, k)| n * k * 2).sum()
}

/// Weight error of the native operand versus exact MXFP8 (proven by the
/// exhaustive unit test: every MXFP8 value is a BF16 value).
pub(crate) fn weight_rel_err_vs_mxfp8() -> f64 {
    0.0
}

/// What the checkpoint ships for one projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Shipped {
    pub weight: Option<WeightDtype>,
    pub scale: Option<WeightDtype>,
    pub shape: Option<Vec<usize>>,
}

impl Shipped {
    fn read(store: &WeightStore, prefix: &str) -> Self {
        let weight = store.get(&format!("{prefix}.weight")).ok();
        let scale = store.get(&format!("{prefix}.weight_scale")).ok();
        Self {
            weight: weight.map(|t| t.dtype),
            scale: scale.map(|t| t.dtype),
            shape: weight.map(|t| t.shape.clone()),
        }
    }
}

/// MXFP8 = E4M3 values + UInt8 (UE8M0) block scales; block-128 FP8 ships
/// BF16/F32 scales and is refused (the receipt claims an exact MXFP8 source).
pub(crate) fn check_shipped(
    projection: &str,
    shipped: &Shipped,
    (n, k): (usize, usize),
) -> Result<()> {
    ensure!(
        shipped.weight == Some(WeightDtype::FP8E4M3) && shipped.scale == Some(WeightDtype::UInt8),
        "{SELECTOR}: shared_expert.{projection} is not shipped as MXFP8 (weight {:?}, scale {:?}); \
         the native shared route needs the exact MXFP8 source",
        shipped.weight,
        shipped.scale
    );
    ensure!(
        shipped.shape.as_deref() == Some(&[n, k][..]),
        "{SELECTOR}: shared_expert.{projection} shape {:?} is not [{n}, {k}]",
        shipped.shape
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    /// Built here at load from the MXFP8 checkpoint tensors.
    Owned,
    /// `ATLAS_QWEN4_SHARED_BF16=1` tensors (feat/fnd-shared), 0 new bytes.
    Alias,
}

impl Source {
    fn name(self) -> &'static str {
        match self {
            Self::Owned => "owned",
            Self::Alias => "alias_shared_bf16",
        }
    }
}

/// One layer's resident row-major BF16 gate/up `[inter, hidden]`, down `[hidden, inter]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SharedBf16 {
    pub gate: DevicePtr,
    pub up: DevicePtr,
    pub down: DevicePtr,
    pub hidden: usize,
    pub inter: usize,
    pub source: Source,
}

#[derive(Default)]
pub(crate) struct Registry {
    by_gate: HashMap<u64, SharedBf16>,
    pub(crate) owned_bytes: usize,
    pub(crate) alias_bytes: usize,
}

impl Registry {
    pub(crate) fn insert(
        &mut self,
        shared_expert_gate: DevicePtr,
        entry: SharedBf16,
    ) -> Result<()> {
        ensure!(
            !shared_expert_gate.is_null(),
            "{SELECTOR}: shared: null shared_expert_gate key"
        );
        ensure!(
            [entry.gate, entry.up, entry.down]
                .iter()
                .all(|p| !p.is_null()),
            "{SELECTOR}: shared: null BF16 weight"
        );
        ensure!(
            !self.by_gate.contains_key(&shared_expert_gate.0),
            "{SELECTOR}: shared: layer registered twice"
        );
        let bytes = layer_bytes(entry.hidden, entry.inter);
        match entry.source {
            Source::Owned => self.owned_bytes += bytes,
            Source::Alias => self.alias_bytes += bytes,
        }
        self.by_gate.insert(shared_expert_gate.0, entry);
        Ok(())
    }

    pub(crate) fn get(&self, shared_expert_gate: DevicePtr) -> Option<SharedBf16> {
        self.by_gate.get(&shared_expert_gate.0).copied()
    }

    pub(crate) fn layers(&self) -> usize {
        self.by_gate.len()
    }

    fn source_summary(&self) -> &'static str {
        let owned = self.by_gate.values().any(|e| e.source == Source::Owned);
        let alias = self.by_gate.values().any(|e| e.source == Source::Alias);
        match (owned, alias) {
            (true, true) => "mixed",
            (false, true) => Source::Alias.name(),
            _ => Source::Owned.name(),
        }
    }
}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();

fn registry() -> MutexGuard<'static, Registry> {
    REGISTRY
        .get_or_init(|| Mutex::new(Registry::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn free_all(gpu: &dyn GpuBackend, ptrs: &[DevicePtr]) {
    for p in ptrs {
        if let Err(e) = gpu.free(*p) {
            tracing::warn!("{SELECTOR}: shared: free after a failed load: {e:#}");
        }
    }
}

/// Dequantize the three MXFP8 projections; frees what it built on failure.
fn build_owned(store: &WeightStore, prefix: &str, gpu: &dyn GpuBackend) -> Result<[DevicePtr; 3]> {
    let mut built: Vec<DevicePtr> = Vec::with_capacity(3);
    for projection in PROJECTIONS {
        match crate::weight_map::dequant_mxfp8_to_bf16(
            store,
            &format!("{prefix}.{projection}"),
            gpu,
        ) {
            Ok(w) => built.push(w.weight),
            Err(e) => {
                free_all(gpu, &built);
                return Err(e)
                    .with_context(|| format!("{SELECTOR}: shared: dequant {prefix}.{projection}"));
            }
        }
    }
    Ok([built[0], built[1], built[2]])
}

/// Load-time hook, once per MoE layer after `load_moe_qwen35`.
/// `alias`: the layer's `ATLAS_QWEN4_SHARED_BF16` gate/up/down when that
/// control kept them; `None` builds an owned copy.
pub(crate) fn register_layer(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    layer_prefix: &str,
    shared_expert_gate: DevicePtr,
    alias: Option<[DevicePtr; 3]>,
) -> Result<()> {
    if !super::active_plan(config)?.is_some_and(|p| p.covers(super::plan::Family::Shared)) {
        return Ok(());
    }
    // Same `inter` as `load_moe_qwen35` and `qwen4_shared_bf16`, which load the
    // shared expert from `moe_intermediate_size` (640 on Flash-Next, equal to
    // `shared_expert_intermediate_size` there); reading the other field would
    // mis-size the alias/owned copy on a checkpoint where they differ.
    let (hidden, inter) = (config.hidden_size, config.moe_intermediate_size);
    ensure!(inter > 0, "{SELECTOR}: shared: model has no shared expert");
    let prefix = format!("{layer_prefix}.mlp.shared_expert");
    for (projection, shape) in PROJECTIONS.iter().zip(shapes(hidden, inter)) {
        check_shipped(
            projection,
            &Shipped::read(store, &format!("{prefix}.{projection}")),
            shape,
        )?;
    }
    let (ptrs, source) = match alias {
        Some(ptrs) => (ptrs, Source::Alias),
        None => (build_owned(store, &prefix, gpu)?, Source::Owned),
    };
    let entry = SharedBf16 {
        gate: ptrs[0],
        up: ptrs[1],
        down: ptrs[2],
        hidden,
        inter,
        source,
    };
    let inserted = registry().insert(shared_expert_gate, entry);
    if inserted.is_err() && source == Source::Owned {
        free_all(gpu, &ptrs);
    }
    inserted
}

/// Every Flash-Next target layer is MoE with a shared expert, so a planned
/// family must register all of them (a loader route that skips the hook
/// would otherwise leave that layer on NVFP4 without a receipt).
pub(crate) fn check_complete(registered: usize, target_layers: usize) -> Result<()> {
    ensure!(
        registered > 0 && registered == target_layers,
        "{SELECTOR}: family {FAMILY} registered {registered} of {target_layers} target layers; \
         a shared-expert load route did not reach its hook"
    );
    Ok(())
}

/// Load receipt; fails when the family is planned but a layer did not register.
pub(crate) fn log_load_receipt(config: &ModelConfig) -> Result<()> {
    if !super::active_plan(config)?.is_some_and(|p| p.covers(super::plan::Family::Shared)) {
        return Ok(());
    }
    let reg = registry();
    check_complete(reg.layers(), config.num_hidden_layers)?;
    tracing::info!(
        "QWEN4_PREFILL_NATIVE engaged selector={SELECTOR} family={FAMILY} layers={} source={} path={PATH} \
         bf16_resident_mib={} bf16_alias_mib={} weight_rel_err_vs_mxfp8={:e} activations=bf16 decode=unchanged_nvfp4",
        reg.layers(),
        reg.source_summary(),
        reg.owned_bytes >> 20,
        reg.alias_bytes >> 20,
        weight_rel_err_vs_mxfp8(),
    );
    Ok(())
}

/// Prefill lookup (a cheap no-op while the selector is off).
pub(crate) fn lookup(shared_expert_gate: DevicePtr) -> Option<SharedBf16> {
    if !on() {
        return None;
    }
    registry().get(shared_expert_gate)
}

/// Output buffers of the default shared route (the blend reads `down_out`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scratch {
    pub gate_out: DevicePtr,
    pub up_out: DevicePtr,
    pub down_out: DevicePtr,
}

/// gate/up GEMM -> SiLU(gate) * up -> down GEMM, all BF16, alpha 1.
pub(crate) fn run(
    gpu: &dyn GpuBackend,
    native: &SharedBf16,
    act_mul: KernelHandle,
    input: DevicePtr,
    rows: usize,
    scratch: Scratch,
    stream: u64,
) -> Result<()> {
    ensure!(
        act_mul.0 != 0,
        "{SELECTOR}: shared: activation kernel is not loaded"
    );
    ensure!(rows > 0, "{SELECTOR}: shared: empty prefill");
    let to_u32 = |v: usize| u32::try_from(v).context("shared native GEMM dimension exceeds u32");
    let (m, h, i) = (to_u32(rows)?, to_u32(native.hidden)?, to_u32(native.inter)?);
    let elements = to_u32(rows * native.inter)?;
    use spark_runtime::cublaslt::bf16_gemm_act_weight_t;
    bf16_gemm_act_weight_t(input.0, native.gate.0, scratch.gate_out.0, m, i, h, stream)?;
    bf16_gemm_act_weight_t(input.0, native.up.0, scratch.up_out.0, m, i, h, stream)?;
    crate::layers::ops::silu_mul(
        gpu,
        act_mul,
        scratch.gate_out,
        scratch.up_out,
        scratch.gate_out,
        elements,
        stream,
    )?;
    bf16_gemm_act_weight_t(
        scratch.gate_out.0,
        native.down.0,
        scratch.down_out.0,
        m,
        h,
        i,
        stream,
    )?;
    note_call(rows, native);
    Ok(())
}

static CALLS: AtomicUsize = AtomicUsize::new(0);

fn note_call(rows: usize, native: &SharedBf16) {
    if CALLS.fetch_add(1, Ordering::Relaxed) == 0 {
        tracing::info!(
            "QWEN4_PREFILL_NATIVE_CALL selector={SELECTOR} family={FAMILY} module=shared_expert path={PATH} \
             m={rows} n={} k={} source={} weight_scale=1 first_call",
            native.inter,
            native.hidden,
            native.source.name()
        );
    }
}

/// Refuse a diagnostic that asks for a different shared route on a layer
/// the native plan engaged (logged by the caller's error).
pub(crate) fn refuse_conflict(fakequant_shared_v7: bool) -> Result<()> {
    ensure!(
        !fakequant_shared_v7,
        "{SELECTOR}=1 (family {FAMILY}) conflicts with the fake-quant moe_shared_v7 arm; unset one"
    );
    Ok(())
}

#[cfg(test)]
#[path = "shared_tests.rs"]
mod tests;
