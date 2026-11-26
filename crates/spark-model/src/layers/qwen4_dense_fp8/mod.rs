// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_DENSE_FP8=1`: decode-only native 8-bit path for the qwen4_exp
//! dense modules the checkpoint ships as MXFP8 (default off).
//!
//! Absent or `0` changes nothing: GDN qkvz/out_proj, attention q/k/v/o and
//! the shared expert keep today's NVFP4 requant (one scale2 per tensor) on
//! every path. With the selector set, plain C1 decode runs those matrices
//! through `dense_gemv_mxfp8`, which reads the checkpoint's E4M3 codes and
//! UE8M0 block-32 exponents as shipped (aliased from the weight store, which
//! keeps them resident). No requantization happens, so the weight error
//! against the exact MXFP8 -> BF16 dequant is zero (`mxfp8.rs`). Prefill,
//! prefill-as-decode and every multi-row route keep the NVFP4 operands.
//!
//! The decode arm only fires inside [`DecodeScope`], which
//! `decode_dispatch` (the single-token model decode) opens. A decode entry
//! that does not open it simply keeps NVFP4, and the per-layer receipt
//! counters then read zero, so a missed entry is visible rather than a
//! silent prefill change.
//!
//! Qualified only for C1 target-only: native MTP, DFlash, self-speculation,
//! multi-row decode, native batch and TP/EP are refused at load, mirroring
//! `layers/moe/qwen4_router_bf16.rs`.

pub(crate) mod gemv_v2;
pub(crate) mod mxfp8;
#[cfg(all(test, feature = "cuda", target_os = "linux"))]
#[path = "mxfp8_gpu_tests.rs"]
mod mxfp8_gpu_tests;
pub(crate) mod proj;

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail, ensure};

pub(crate) const QWEN4_DENSE_FP8_ENV: &str = "ATLAS_QWEN4_DENSE_FP8";

/// Launch inputs the rule reads, captured once so the rule is pure.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DenseFp8Launch {
    pub(crate) is_qwen4: bool,
    pub(crate) use_native_mtp: bool,
    pub(crate) has_dflash: bool,
    pub(crate) self_speculative: bool,
    pub(crate) multi_batch: bool,
    pub(crate) native_batch: bool,
    pub(crate) multi_rank: bool,
    /// First P1 BF16 reference selector set alongside this one, if any.
    pub(crate) bf16_arm: Option<&'static str>,
}

/// The P1 BF16 reference arms for the same modules. Each must be mutually
/// exclusive with the FP8 arm: in attention the BF16 arm would mask FP8,
/// in `ssm_forward` FP8 would silently win.
pub(crate) const BF16_ARM_SELECTORS: [&str; 3] = [
    "ATLAS_QWEN4_GDN_BF16",
    "ATLAS_QWEN4_ATTN_BF16",
    "ATLAS_QWEN4_SHARED_BF16",
];

/// A selector counts as set unless it is absent, empty or exactly `0`.
pub(crate) fn first_set_bf16_arm(read: impl Fn(&str) -> Option<String>) -> Option<&'static str> {
    BF16_ARM_SELECTORS
        .into_iter()
        .find(|name| read(name).is_some_and(|v| !v.is_empty() && v != "0"))
}

pub(crate) fn parse_selector(raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("{QWEN4_DENSE_FP8_ENV} must be exactly 0 or 1"),
    }
}

/// Read once; admission, every layer and the decode arms see one answer.
pub(crate) fn selected() -> Result<bool> {
    static SEL: OnceLock<Result<bool, String>> = OnceLock::new();
    SEL.get_or_init(|| {
        let raw = match std::env::var(QWEN4_DENSE_FP8_ENV) {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(format!("{QWEN4_DENSE_FP8_ENV} is not valid Unicode"));
            }
        };
        parse_selector(raw.as_deref()).map_err(|e| e.to_string())
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// `Ok(true)` = build the FP8 copies; `Ok(false)` = default path.
pub(crate) fn decide(selected: bool, launch: DenseFp8Launch) -> Result<bool> {
    if !selected {
        return Ok(false);
    }
    ensure!(
        launch.is_qwen4,
        "{QWEN4_DENSE_FP8_ENV}=1 is a qwen4_exp control; unset it for this model"
    );
    let refusals = [
        (
            launch.use_native_mtp,
            "native MTP verify rows run the NVFP4 dense projections; disable speculation",
        ),
        (
            launch.has_dflash,
            "DFlash verify rows run the NVFP4 dense projections; drop --dflash",
        ),
        (
            launch.self_speculative,
            "self-speculative verify runs the NVFP4 dense projections",
        ),
        (
            launch.multi_batch,
            "multi-row decode runs the NVFP4 dense projections; use --max-batch-size 1",
        ),
        (
            launch.native_batch,
            "native Qwen4 batch decode requires NVFP4 operands; unset ATLAS_QWEN4_NATIVE_BATCH",
        ),
        (
            launch.multi_rank,
            "the FP8 copies are not sharded; run TP1/EP1",
        ),
    ];
    if let Some((_, reason)) = refusals.iter().find(|(blocked, _)| *blocked) {
        bail!("{QWEN4_DENSE_FP8_ENV}=1 refused: {reason}");
    }
    if let Some(name) = launch.bf16_arm {
        bail!(
            "{QWEN4_DENSE_FP8_ENV}=1 refused: {name} selects the BF16 reference arm for the same \
             module; the FP8 and BF16 arms are mutually exclusive, unset one"
        );
    }
    Ok(true)
}

/// Load-time gate before any layer is built. Reads no other variable unless
/// the selector is set, so the default path touches only the selector.
pub(crate) fn admit_launch(
    is_qwen4: bool,
    use_native_mtp: bool,
    has_dflash: bool,
    self_speculative: bool,
    max_batch_size: usize,
    multi_rank: bool,
) -> Result<bool> {
    let selected = selected()?;
    if !selected {
        return Ok(false);
    }
    decide(
        true,
        DenseFp8Launch {
            is_qwen4,
            use_native_mtp,
            has_dflash,
            self_speculative,
            multi_batch: max_batch_size > 1,
            native_batch: std::env::var("ATLAS_QWEN4_NATIVE_BATCH").ok().as_deref() == Some("1"),
            multi_rank,
            // Through the umbrella plan, not raw env: under
            // ATLAS_QWEN4_DENSE_NATIVE=1 the BF16 envs are absent but on.
            bf16_arm: first_set_bf16_arm(crate::qwen4_native_selectors::plan()?.reader()),
        },
    )
}

/// Per-layer rule inside the shared Qwen3.5/Qwen4 loader. The MTP layer
/// only loads under native MTP, which admission refuses; refusing it here
/// keeps a bypass loud.
pub(crate) fn layer_engaged(selected: bool, is_qwen4: bool, is_mtp_layer: bool) -> Result<bool> {
    if !selected {
        return Ok(false);
    }
    ensure!(
        is_qwen4,
        "{QWEN4_DENSE_FP8_ENV}=1 is a qwen4_exp control; unset it for this model"
    );
    ensure!(
        !is_mtp_layer,
        "{QWEN4_DENSE_FP8_ENV}=1 refused: the MTP layer verifies through NVFP4 projections"
    );
    Ok(true)
}

thread_local! {
    static DECODE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// RAII marker for the single-token model decode. The FP8 arms run only
/// while one is alive on this thread.
#[must_use = "the scope ends when the guard drops"]
pub(crate) struct DecodeScope(());

impl DecodeScope {
    pub(crate) fn enter() -> Self {
        DECODE_DEPTH.with(|d| d.set(d.get().saturating_add(1)));
        DecodeScope(())
    }
}

impl Drop for DecodeScope {
    fn drop(&mut self) {
        DECODE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

pub(crate) fn in_decode_scope() -> bool {
    DECODE_DEPTH.with(|d| d.get() > 0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Module {
    Gdn,
    Attn,
    Shared,
}

impl Module {
    pub(crate) const ALL: [Module; 3] = [Module::Gdn, Module::Attn, Module::Shared];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Module::Gdn => "gdn",
            Module::Attn => "attn",
            Module::Shared => "shared",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

static MODULE_DISPATCHES: [AtomicU64; 3] =
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Process-wide host dispatch totals per module (eager calls plus CUDA graph
/// captures; a graph replay re-runs the captured FP8 kernels without host code).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn module_dispatches(module: Module) -> u64 {
    MODULE_DISPATCHES[module.index()].load(Ordering::Relaxed)
}

/// Per-layer receipt: counts every host dispatch of this layer's FP8 arm.
/// It is incremented by the same call that launches the kernel.
#[derive(Debug)]
pub(crate) struct LayerReceipt {
    module: Module,
    layer: usize,
    dispatches: AtomicU64,
}

impl LayerReceipt {
    pub(crate) fn new(module: Module, layer: usize) -> Self {
        Self {
            module,
            layer,
            dispatches: AtomicU64::new(0),
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn dispatches(&self) -> u64 {
        self.dispatches.load(Ordering::Relaxed)
    }

    /// One INFO line per layer on its first FP8 dispatch, so a log proves
    /// every installed layer took the arm (gdn 36 + attn 12 + shared 48 lines
    /// on Flash-Next), never just the first one.
    pub(crate) fn record(&self) {
        let first_for_layer = self.dispatches.fetch_add(1, Ordering::Relaxed) == 0;
        let module_total =
            MODULE_DISPATCHES[self.module.index()].fetch_add(1, Ordering::Relaxed) + 1;
        if first_for_layer {
            tracing::info!(
                "QWEN4_DENSE_FP8_DECODE_ENGAGED module={} layer={} kernel=dense_gemv_mxfp8 module_dispatches={module_total}",
                self.module.name(),
                self.layer
            );
        }
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
