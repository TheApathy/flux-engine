// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_PREFILL_NATIVE=1` (default off; absent or `0` changes nothing):
//! Flash-Next prefill projections read checkpoint-native weights instead of
//! the load-time NVFP4 re-quantization.
//!
//! - MXFP8-shipped GDN `in_proj_qkvz` / `out_proj` and attention `q/k/v/o`:
//!   the FP8 tensor-scale prefill GEMM (`ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8`
//!   with `_SCALE=tensor`) multiplies E4M3 rows built at load from the exact
//!   MXFP8 dequantization with one power-of-two tensor scale (see `e4m3`),
//!   not E4M3 rows re-rounded from the NVFP4 copy. Activations are unchanged
//!   (tensor-scale E4M3).
//! - BF16-shipped hyper-connection down/up/inject (and the final mixer): the
//!   fast-projection dequant helpers hand out the resident BF16 tensor with
//!   scale 1 instead of `LUT * fp8` + scale2.
//! - BF16-shipped router (`mlp.gate`): the prefill gate GEMM takes the BF16
//!   `dense_gemm` arm on the resident tensor.
//!
//! Decode is untouched: every NVFP4 copy, layer field and exact-prefill
//! admission check stays as it is. Receipts: `QWEN4_PREFILL_NATIVE engaged`
//! once at load, `QWEN4_PREFILL_NATIVE_CALL` on the first call per module
//! (from the launching code), `QWEN4_PREFILL_NATIVE_REFUSED` where a planned
//! family meets a route that still reads NVFP4.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::weight_map::QuantizedWeight;

#[path = "qwen4_prefill_native/e4m3.rs"]
pub(crate) mod e4m3;
#[path = "qwen4_prefill_native/launch.rs"]
mod launch;
#[path = "qwen4_prefill_native/plan.rs"]
pub(crate) mod plan;
#[path = "qwen4_prefill_native/registry.rs"]
pub(crate) mod registry;
#[path = "qwen4_prefill_native/shared.rs"]
pub(crate) mod shared;

pub(crate) use launch::{bf16_dequant_gemm, native_fp8_gemm, router_dense_gemm};
pub(crate) use plan::{Family, Plan};

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_NATIVE";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Module {
    GdnQkvz,
    GdnOut,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    HcDown,
    HcUp,
    HcInject,
    FinalDown,
    FinalUp,
    Router,
}

impl Module {
    pub(crate) const ALL: [Module; 12] = [
        Self::GdnQkvz,
        Self::GdnOut,
        Self::AttnQ,
        Self::AttnK,
        Self::AttnV,
        Self::AttnO,
        Self::HcDown,
        Self::HcUp,
        Self::HcInject,
        Self::FinalDown,
        Self::FinalUp,
        Self::Router,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::GdnQkvz => "gdn_qkvz",
            Self::GdnOut => "gdn_out",
            Self::AttnQ => "attn_q",
            Self::AttnK => "attn_k",
            Self::AttnV => "attn_v",
            Self::AttnO => "attn_o",
            Self::HcDown => "hc_down",
            Self::HcUp => "hc_up",
            Self::HcInject => "hc_inject",
            Self::FinalDown => "final_down",
            Self::FinalUp => "final_up",
            Self::Router => "router",
        }
    }

    pub(crate) fn family(self) -> Family {
        match self {
            Self::GdnQkvz | Self::GdnOut => Family::Ssm,
            Self::AttnQ | Self::AttnK | Self::AttnV | Self::AttnO => Family::Attn,
            Self::HcDown | Self::HcUp | Self::HcInject | Self::FinalDown | Self::FinalUp => {
                Family::Hc
            }
            Self::Router => Family::Router,
        }
    }

    pub(crate) fn is_fp8(self) -> bool {
        matches!(self.family(), Family::Ssm | Family::Attn)
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|m| *m == self).unwrap_or(0)
    }

    /// The attention projection a `{p}.self_attn.{q,k,v,o}_proj` prefix names.
    pub(crate) fn attention(prefix: &str) -> Option<Module> {
        let tail = prefix.rsplit_once(".self_attn.")?.1;
        match tail {
            "q_proj" => Some(Self::AttnQ),
            "k_proj" => Some(Self::AttnK),
            "v_proj" => Some(Self::AttnV),
            "o_proj" => Some(Self::AttnO),
            _ => None,
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn parse_selector(raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{SELECTOR} must be exactly 0 or 1, got {other:?}"),
    }
}

/// Read once; a malformed value fails the first load-time hook.
pub(crate) fn selected() -> Result<bool> {
    // Umbrella-aware: ATLAS_QWEN4_DENSE_NATIVE=1 implies this control unless
    // its own env is explicitly 0 (qwen4_native_selectors).
    crate::qwen4_native_selectors::selected(crate::qwen4_native_selectors::Knob::Prefill)
}

fn on() -> bool {
    selected().unwrap_or(false)
}

static REGISTRY: OnceLock<Mutex<registry::Registry>> = OnceLock::new();

fn registry() -> MutexGuard<'static, registry::Registry> {
    REGISTRY
        .get_or_init(|| Mutex::new(registry::Registry::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Set by the load receipt. The registry, plan and selector are process
/// statics keyed by device pointers; an in-process model reload (TUI swap)
/// would mix a stale plan with reused addresses, so it is refused instead.
static SEALED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn ensure_not_reloaded(sealed: bool) -> Result<()> {
    ensure!(
        !sealed,
        "{SELECTOR}=1 does not support an in-process model reload; restart the server to load another model"
    );
    Ok(())
}

/// The process plan for `config` (target layers only; MTP layers never register).
pub(super) fn active_plan(config: &ModelConfig) -> Result<Option<Plan>> {
    if !selected()? || config.weight_prefix == "mtp" {
        return Ok(None);
    }
    Ok(Some(plan::process_plan(config)?))
}

/// [`active_plan`] for a registration: also refuses a second model load.
fn registering_plan(config: &ModelConfig) -> Result<Option<Plan>> {
    let plan = active_plan(config)?;
    if plan.is_some() {
        ensure_not_reloaded(SEALED.load(Ordering::Acquire))?;
    }
    Ok(plan)
}

// ── Load-time hooks (one call per weight from the loaders) ──────────────

/// True when the checkpoint ships `{prefix}.weight` as MXFP8: E4M3 values with
/// a UInt8 (UE8M0) block-scale tensor. Block-128 FP8 ships BF16/F32 scales.
pub(crate) fn ships_mxfp8(store: &spark_runtime::weights::WeightStore, prefix: &str) -> bool {
    let dtype = |name: String| store.get(&name).ok().map(|t| t.dtype);
    shipped_as_mxfp8(
        dtype(format!("{prefix}.weight")),
        dtype(format!("{prefix}.weight_scale")),
    )
}

pub(crate) fn shipped_as_mxfp8(
    weight: Option<spark_runtime::weights::WeightDtype>,
    scale: Option<spark_runtime::weights::WeightDtype>,
) -> bool {
    use spark_runtime::weights::WeightDtype;
    weight == Some(WeightDtype::FP8E4M3) && scale == Some(WeightDtype::UInt8)
}

/// MXFP8-shipped GDN / attention weight: build its native E4M3 copy.
/// `shipped_mxfp8` is the caller's check of the checkpoint tensors; a planned
/// family whose weight is not MXFP8 is refused (the receipt claims an MXFP8 source).
#[allow(clippy::too_many_arguments)]
pub(crate) fn register_mxfp8(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    module: Module,
    nvfp4: &QuantizedWeight,
    bf16: DevicePtr,
    n: usize,
    k: usize,
    shipped_mxfp8: bool,
    stream: u64,
) -> Result<()> {
    let Some(plan) = registering_plan(config)? else {
        return Ok(());
    };
    if !plan.covers(module.family()) {
        return Ok(());
    }
    ensure!(
        shipped_mxfp8,
        "{SELECTOR}: {} is not shipped as MXFP8 (E4M3 + UE8M0); the native FP8 source would not be exact",
        module.name()
    );
    let qg = match module {
        Module::AttnQ => {
            let (heads, head_dim) = (config.num_attention_heads, config.head_dim);
            ensure!(
                2 * heads * head_dim == n,
                "{SELECTOR}: q_proj rows {n} are not gated 2 x {heads} x {head_dim}"
            );
            Some((heads, head_dim))
        }
        _ => None,
    };
    registry().insert_mxfp8(gpu, module, nvfp4, bf16, n, k, qg, stream)?;
    Ok(())
}

/// True when `ptr` is the WeightStore's own BF16 tensor `name`: memory the
/// store owns for the model's lifetime (only `release_consumed_alias`, used
/// for the MTP and LM-head tensors, ever frees a store alias). Dropping a
/// `DenseWeight` handle, as the HC_BF16=0 loader does, frees nothing.
pub(crate) fn store_retains_bf16(
    store: &spark_runtime::weights::WeightStore,
    name: &str,
    ptr: DevicePtr,
) -> bool {
    let stored = store.get(name).ok().map(|t| (t.dtype, t.ptr));
    retained_alias(stored, ptr)
}

pub(crate) fn retained_alias(
    stored: Option<(spark_runtime::weights::WeightDtype, DevicePtr)>,
    ptr: DevicePtr,
) -> bool {
    !ptr.is_null() && stored == Some((spark_runtime::weights::WeightDtype::BF16, ptr))
}

/// BF16-shipped hyper-connection weight. `retained` is the caller's
/// [`store_retains_bf16`] check; a planned family whose source is not a
/// store-owned alias (e.g. a loader-owned dequant buffer that may be freed)
/// is refused rather than keyed to a pointer that could dangle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn register_bf16(
    config: &ModelConfig,
    module: Module,
    nvfp4: &QuantizedWeight,
    bf16: DevicePtr,
    n: usize,
    k: usize,
    retained: bool,
) -> Result<()> {
    let Some(plan) = registering_plan(config)? else {
        return Ok(());
    };
    if plan.covers(module.family()) {
        ensure!(
            retained,
            "{SELECTOR}: {} BF16 source is not a store-owned BF16 tensor; refusing a pointer that may be freed",
            module.name()
        );
        registry().insert_bf16(module, nvfp4, bf16, n, k)?;
    }
    Ok(())
}

/// The resident BF16 router of a layer that also has an NVFP4 router copy.
pub(crate) fn register_router(
    config: &ModelConfig,
    bf16_gate: DevicePtr,
    n: usize,
    k: usize,
    retained: bool,
) -> Result<()> {
    let Some(plan) = registering_plan(config)? else {
        return Ok(());
    };
    if plan.covers(Family::Router) {
        ensure!(
            retained,
            "{SELECTOR}: router BF16 gate is not a store-owned BF16 tensor"
        );
        registry().insert_router(bf16_gate, n, k)?;
    }
    Ok(())
}

/// Load receipt; fails when a planned family registered no weight (a loader
/// route no longer reaches its hook).
pub(crate) fn log_load_receipt(config: &ModelConfig) -> Result<()> {
    let Some(plan) = active_plan(config)? else {
        return Ok(());
    };
    ensure_not_reloaded(SEALED.swap(true, Ordering::AcqRel))?;
    let reg = registry();
    // Family `shared` lives in `shared::Registry`; `shared::log_load_receipt`
    // checks it against every target layer.
    for family in Family::ALL.into_iter().filter(|&f| f != Family::Shared) {
        ensure!(
            !plan.covers(family) || reg.has_family(family),
            "{SELECTOR}: family {} is planned but no weight reached its load hook",
            family.name()
        );
    }
    let modules = reg
        .registered()
        .iter()
        .map(|(m, c)| format!("{m}:{c}"))
        .collect::<Vec<_>>()
        .join(",");
    tracing::info!(
        "QWEN4_PREFILL_NATIVE engaged selector={SELECTOR} modules={modules} plan={} \
         fp8_resident_mib={} bf16_alias_mib={} fp8_scale=pow2_tensor fp8_source=mxfp8_exact_bf16 \
         fp8_inexact_frac={:.6} fp8_rel_fro={:.3e} fp8_worst_rel_fro={:.3e} decode=unchanged_nvfp4 activations=unchanged",
        plan.summary(),
        reg.fp8_bytes >> 20,
        reg.bf16_alias_bytes >> 20,
        reg.stats.inexact_fraction(),
        reg.stats.rel_fro(),
        reg.worst_rel_fro,
    );
    Ok(())
}

// ── Prefill-time lookups (cheap no-ops while the selector is off) ───────

pub(crate) fn fp8_lookup(
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    qg: Option<(usize, usize)>,
) -> Result<Option<registry::NativeFp8>> {
    if !on() {
        return Ok(None);
    }
    registry().fp8(weight, n, k, qg)
}

pub(crate) fn bf16_lookup(
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
) -> Option<(DevicePtr, Module)> {
    if !on() {
        return None;
    }
    registry().bf16(weight, n, k)
}

pub(crate) fn router_bf16(bf16_gate: DevicePtr) -> bool {
    on() && registry().router(bf16_gate)
}

// ── Per-call receipts ────────────────────────────────────────────────────

static CALLS: [AtomicUsize; 12] = [const { AtomicUsize::new(0) }; 12];
static TOTAL: AtomicUsize = AtomicUsize::new(0);

/// Count one launched native projection; the first per module is logged with
/// its path, and the per-module counts at every power-of-two total from 64.
pub(crate) fn note_call(
    module: Module,
    path: &'static str,
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
) {
    if CALLS[module.index()].fetch_add(1, Ordering::Relaxed) == 0 {
        tracing::info!(
            "QWEN4_PREFILL_NATIVE_CALL selector={SELECTOR} module={} path={path} m={m} n={n} k={k} weight_scale={scale:e} first_call",
            module.name()
        );
    }
    let total = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    if total >= 64 && total.is_power_of_two() {
        let counts = Module::ALL
            .iter()
            .map(|m| format!("{}:{}", m.name(), calls(*m)))
            .collect::<Vec<_>>()
            .join(",");
        tracing::info!("QWEN4_PREFILL_NATIVE_CALLS total={total} {counts}");
    }
}

pub(crate) fn calls(module: Module) -> usize {
    CALLS[module.index()].load(Ordering::Relaxed)
}

/// A planned family met a route that still reads NVFP4 (logged once per reason).
pub(crate) fn note_refused(family: Family, reason: &'static str, rows: usize) {
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    if !on() {
        return;
    }
    let mut seen = SEEN.lock().unwrap_or_else(|p| p.into_inner());
    // Only for a family the plan engaged (it registered weights at load).
    if seen.contains(&reason) || !registry().has_family(family) {
        return;
    }
    seen.push(reason);
    tracing::warn!(
        "QWEN4_PREFILL_NATIVE_REFUSED selector={SELECTOR} family={} reason={reason} rows={rows} fallback=nvfp4",
        family.name()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_model_load_is_refused() {
        assert!(ensure_not_reloaded(false).is_ok());
        let err = ensure_not_reloaded(true).unwrap_err().to_string();
        assert!(err.contains("in-process model reload"), "{err}");
    }

    #[test]
    fn selector_is_exact() {
        assert!(!parse_selector(None).unwrap());
        assert!(!parse_selector(Some("0")).unwrap());
        assert!(parse_selector(Some("1")).unwrap());
        for bad in ["", "true", " 1", "2"] {
            assert!(parse_selector(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn mxfp8_source_needs_e4m3_values_and_ue8m0_scales() {
        use spark_runtime::weights::WeightDtype as D;
        assert!(shipped_as_mxfp8(Some(D::FP8E4M3), Some(D::UInt8)));
        assert!(
            !shipped_as_mxfp8(Some(D::FP8E4M3), Some(D::BF16)),
            "block-128 FP8"
        );
        assert!(!shipped_as_mxfp8(Some(D::BF16), None));
        assert!(!shipped_as_mxfp8(None, None));
    }

    #[test]
    fn bf16_source_must_be_the_store_owned_alias() {
        use spark_runtime::weights::WeightDtype as D;
        let p = DevicePtr(4096);
        assert!(retained_alias(Some((D::BF16, p)), p));
        assert!(
            !retained_alias(Some((D::BF16, DevicePtr(8192))), p),
            "loader-owned copy"
        );
        assert!(
            !retained_alias(Some((D::FP8E4M3, p)), p),
            "dequantized from FP8"
        );
        assert!(!retained_alias(None, p), "tombstoned / released");
        assert!(!retained_alias(
            Some((D::BF16, DevicePtr::NULL)),
            DevicePtr::NULL
        ));
    }

    #[test]
    fn attention_prefixes_map_to_modules() {
        let p = "model.language_model.layers.3.self_attn";
        assert_eq!(
            Module::attention(&format!("{p}.q_proj")),
            Some(Module::AttnQ)
        );
        assert_eq!(
            Module::attention(&format!("{p}.o_proj")),
            Some(Module::AttnO)
        );
        assert_eq!(
            Module::attention(&format!("{p}.indexer.index_qk_proj")),
            None
        );
        assert_eq!(
            Module::attention("model.layers.0.mlp.shared_expert.gate_proj"),
            None
        );
    }

    #[test]
    fn module_table_is_consistent() {
        for (i, m) in Module::ALL.iter().enumerate() {
            assert_eq!(m.index(), i);
            assert_eq!(m.is_fp8(), matches!(m.family(), Family::Ssm | Family::Attn));
        }
        assert_eq!(CALLS.len(), Module::ALL.len());
    }

    #[test]
    fn lookups_are_inert_while_the_selector_is_off() {
        // The test process never sets the selector.
        if selected().unwrap_or(true) {
            return;
        }
        let w = QuantizedWeight {
            weight: DevicePtr(4096),
            weight_scale: DevicePtr(8192),
            weight_scale_2: 1.0,
            input_scale: DevicePtr::NULL,
        };
        assert_eq!(fp8_lookup(&w, 16, 16, None).unwrap(), None);
        assert_eq!(bf16_lookup(&w, 16, 16), None);
        assert!(!router_bf16(DevicePtr(4096)));
    }
}
