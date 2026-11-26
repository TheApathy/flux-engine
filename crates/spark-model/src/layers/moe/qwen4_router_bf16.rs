// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_ROUTER_BF16=1`: opt-in MoE router precision control for
//! qwen4_exp (default-off diagnostic).
//!
//! Absent or `0` leaves every router as it is today: qwen4_exp re-quantizes
//! the checkpoint BF16 `mlp.gate.weight` to NVFP4 at load (one per-tensor
//! scale2) and `predequant_for_prefill` re-encodes that copy as FP8. With the
//! selector set, a qwen4_exp target layer gets `gate_nvfp4 = None`, so neither
//! copy exists: decode takes the existing `dense_gemv` arm (`forward.rs`) and
//! prefill the existing `dense_gemm` arm (`forward_prefill.rs`), both on the
//! stored BF16 tensor, which also changes the GEMV/GEMM arithmetic and
//! reduction order. The exact K2..K9 MoE and native C2..C32 batch need the
//! NVFP4 router and error per call, so every launch that reaches them is
//! refused here at load. Qualified only for the fixed C1, no-speculation,
//! no-n-gram recipe; n-gram drafting is decided outside the factory and is
//! not covered.

use anyhow::{Result, bail, ensure};
use spark_runtime::weights::WeightDtype;

pub(crate) const QWEN4_ROUTER_BF16_ENV: &str = "ATLAS_QWEN4_ROUTER_BF16";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterPrecision {
    Nvfp4,
    Bf16,
}

/// Every launch input the rule reads, captured once so the rule is pure.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RouterLaunch {
    pub(crate) is_qwen4: bool,
    pub(crate) use_native_mtp: bool,
    pub(crate) has_dflash: bool,
    pub(crate) self_speculative: bool,
    /// `max_batch_size > 1`: multi-row decode reaches `forward_k2`/`forward_k3`,
    /// which take the exact NVFP4-router MoE on the unified `_t_lanes` route.
    pub(crate) multi_batch: bool,
    /// `ATLAS_QWEN4_NATIVE_BATCH=1`: native C2-C32 preflight asserts NVFP4.
    pub(crate) native_batch: bool,
    /// TP or EP world size above 1: `forward_ep` and sharded routing are unqualified.
    pub(crate) multi_rank: bool,
}

/// Kept for the module's tests; the live read goes through
/// `qwen4_native_selectors`, which applies the same 0/1 rule.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn parse_selector(raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("{QWEN4_ROUTER_BF16_ENV} must be exactly 0 or 1"),
    }
}

/// Read once; the load gate, every layer and prefill admission see one answer.
/// Routed through `qwen4_native_selectors` so `ATLAS_QWEN4_DENSE_NATIVE=1`
/// implies it unless this env is explicitly `0`.
pub(crate) fn selected() -> Result<bool> {
    crate::qwen4_native_selectors::selected(crate::qwen4_native_selectors::Knob::Router)
}

pub(crate) fn decide_router_precision(
    selected: bool,
    launch: RouterLaunch,
) -> Result<RouterPrecision> {
    if !selected {
        return Ok(RouterPrecision::Nvfp4);
    }
    if !launch.is_qwen4 {
        bail!("{QWEN4_ROUTER_BF16_ENV}=1 is a qwen4_exp control; unset it for this model");
    }
    let refusals = [
        (
            launch.use_native_mtp,
            "native MTP verifies through the exact NVFP4-router MoE; disable speculation",
        ),
        (
            launch.has_dflash,
            "DFlash verify rows need the exact NVFP4-router MoE; drop --dflash",
        ),
        (
            launch.self_speculative,
            "self-speculative verify needs the exact NVFP4-router MoE",
        ),
        (
            launch.multi_batch,
            "multi-row decode needs the exact NVFP4-router MoE; use --max-batch-size 1",
        ),
        (
            launch.native_batch,
            "native Qwen4 batch decode requires the NVFP4 router; unset ATLAS_QWEN4_NATIVE_BATCH",
        ),
        (
            launch.multi_rank,
            "tensor/expert-parallel routing is not part of this control; run TP1/EP1",
        ),
    ];
    if let Some((_, reason)) = refusals.iter().find(|(blocked, _)| *blocked) {
        bail!("{QWEN4_ROUTER_BF16_ENV}=1 refused: {reason}");
    }
    Ok(RouterPrecision::Bf16)
}

fn env_is_one(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

/// Load-time gate, before any layer is built. Touches no other variable
/// unless the selector is set, so the default path reads only the selector.
/// BF16 router bytes kept resident under the control: one `[experts x hidden]`
/// BF16 `mlp.gate.weight` per MoE layer (qwen4_exp: every layer). The NVFP4
/// and prefill-FP8 copies it replaces are not built, so this is the gross
/// BF16 footprint, not a net delta.
pub(crate) fn extra_resident_bytes(experts: usize, hidden: usize, layers: usize) -> u64 {
    experts as u64 * hidden as u64 * 2 * layers as u64
}

/// Load-time receipt; docs grep this exact prefix.
pub(crate) fn engaged_line(bytes: u64) -> String {
    format!("QWEN4_ROUTER_BF16 engaged scope=decode+prefill extra_resident_bytes={bytes}")
}

pub(crate) fn admit_launch(
    is_qwen4: bool,
    use_native_mtp: bool,
    has_dflash: bool,
    self_speculative: bool,
    max_batch_size: usize,
    multi_rank: bool,
) -> Result<RouterPrecision> {
    let selected = selected()?;
    let mut launch = RouterLaunch {
        is_qwen4,
        ..RouterLaunch::default()
    };
    if selected {
        launch = RouterLaunch {
            use_native_mtp,
            has_dflash,
            self_speculative,
            multi_batch: max_batch_size > 1,
            native_batch: env_is_one("ATLAS_QWEN4_NATIVE_BATCH"),
            multi_rank,
            ..launch
        };
    }
    decide_router_precision(selected, launch)
}

/// Per-layer rule inside the shared Qwen3.5/Qwen4 loader. The MTP layer goes
/// through the same loader, but only under native MTP, which `admit_launch`
/// refuses; refusing it here too keeps a bypass loud.
pub(crate) fn layer_router_precision(
    selected: bool,
    is_qwen4: bool,
    is_mtp_layer: bool,
) -> Result<RouterPrecision> {
    if !selected {
        return Ok(RouterPrecision::Nvfp4);
    }
    ensure!(
        is_qwen4,
        "{QWEN4_ROUTER_BF16_ENV}=1 is a qwen4_exp control; unset it for this model"
    );
    ensure!(
        !is_mtp_layer,
        "{QWEN4_ROUTER_BF16_ENV}=1 refused: the MTP layer verifies through the exact NVFP4-router MoE"
    );
    Ok(RouterPrecision::Bf16)
}

/// The checkpoint router as the `WeightStore` holds it. `dtype` is the stored
/// safetensors dtype (`weights/loader/load_fns.rs:76,160`,
/// `fast_weights/header.rs:118-130`). `loaded_ptr` is `MoeWeights::gate`,
/// which `load_moe_qwen35` takes through `dense()`, a pointer alias.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StoredRouter<'a> {
    pub(crate) dtype: WeightDtype,
    pub(crate) shape: &'a [usize],
    pub(crate) stored_ptr: u64,
    pub(crate) loaded_ptr: u64,
}

/// The control is only meaningful for a router that is BF16 *in the
/// checkpoint* and is served from that very tensor, never from a copy.
pub(crate) fn validate_stored_bf16_router(
    key: &str,
    router: StoredRouter<'_>,
    num_experts: usize,
    hidden: usize,
) -> Result<()> {
    ensure!(
        router.dtype == WeightDtype::BF16,
        "{QWEN4_ROUTER_BF16_ENV}=1 refused: {key} is stored as {:?}, not BF16; \
         a dequantized router is not a BF16 control",
        router.dtype
    );
    ensure!(
        router.loaded_ptr == router.stored_ptr,
        "{QWEN4_ROUTER_BF16_ENV}=1 refused: the loaded router is not the stored {key} tensor"
    );
    ensure!(
        router.shape == [num_experts, hidden],
        "{QWEN4_ROUTER_BF16_ENV}=1 refused: {key} shape {:?} is not [{num_experts}, {hidden}]",
        router.shape
    );
    ensure!(
        router.loaded_ptr != 0 && router.loaded_ptr.is_multiple_of(16) && hidden.is_multiple_of(8),
        "{QWEN4_ROUTER_BF16_ENV}=1 refused: {key} is not laid out for dense_gemv_bf16 16-byte loads"
    );
    Ok(())
}

/// Qwen4 MoE-only prefill admission for a layer built without an NVFP4
/// router. Only the control may build one, the router is the BF16 tensor
/// through `dense_gemm`, and no FP8 router copy may exist. The shared expert
/// keeps today's rule: under the V7/W4A8 shared route it has no predequant
/// bundle, otherwise its three FP8 copies and the FP8 GEMM must be complete.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bf16_router_bundle_valid(
    selected: bool,
    gate_fp8: Option<u64>,
    bf16_gate: u64,
    dense_gemm: u64,
    shared_v7: bool,
    shared_fp8: [Option<u64>; 3],
    fp8_gemm: u64,
) -> bool {
    if !selected || gate_fp8.is_some() || bf16_gate == 0 || dense_gemm == 0 {
        return false;
    }
    shared_v7 || (shared_fp8.iter().all(|p| p.is_some_and(|p| p != 0)) && fp8_gemm != 0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn router_engaged_receipt_bytes() {
        // Flash-Next: 512 experts x 2560 hidden x 48 layers, BF16.
        let bytes = super::extra_resident_bytes(512, 2_560, 48);
        assert_eq!(bytes, 125_829_120);
        assert_eq!(super::extra_resident_bytes(512, 2_560, 0), 0);
        assert_eq!(
            super::engaged_line(bytes),
            "QWEN4_ROUTER_BF16 engaged scope=decode+prefill extra_resident_bytes=125829120"
        );
    }

    use super::{
        RouterLaunch, RouterPrecision, StoredRouter, WeightDtype, bf16_router_bundle_valid,
        decide_router_precision, layer_router_precision, parse_selector,
        validate_stored_bf16_router,
    };

    const SHAPE: [usize; 2] = [512, 2_560];

    fn stored(dtype: WeightDtype) -> StoredRouter<'static> {
        StoredRouter {
            dtype,
            shape: &SHAPE,
            stored_ptr: 0x1000,
            loaded_ptr: 0x1000,
        }
    }

    #[test]
    fn stored_router_must_be_checkpoint_bf16_alias() {
        let key = "model.language_model.layers.0.mlp.gate.weight";
        let check = |router| validate_stored_bf16_router(key, router, 512, 2_560);
        assert!(check(stored(WeightDtype::BF16)).is_ok());
        for dtype in [
            WeightDtype::FP8E4M3,
            WeightDtype::UInt8,
            WeightDtype::FP32,
            WeightDtype::FP8E8M0,
        ] {
            assert!(check(stored(dtype)).is_err(), "{dtype:?}");
        }
        let copied = StoredRouter {
            loaded_ptr: 0x2000,
            ..stored(WeightDtype::BF16)
        };
        assert!(check(copied).is_err());
        let transposed = StoredRouter {
            shape: &[2_560, 512],
            ..stored(WeightDtype::BF16)
        };
        assert!(check(transposed).is_err());
        let misaligned = StoredRouter {
            stored_ptr: 0x1008,
            loaded_ptr: 0x1008,
            ..stored(WeightDtype::BF16)
        };
        assert!(check(misaligned).is_err());
    }

    fn qwen4() -> RouterLaunch {
        RouterLaunch {
            is_qwen4: true,
            ..RouterLaunch::default()
        }
    }

    #[test]
    fn default_keeps_nvfp4_router() {
        for is_qwen4 in [false, true] {
            let launch = RouterLaunch {
                is_qwen4,
                use_native_mtp: true,
                multi_batch: true,
                native_batch: true,
                ..RouterLaunch::default()
            };
            assert_eq!(
                decide_router_precision(false, launch).unwrap(),
                RouterPrecision::Nvfp4
            );
            assert_eq!(
                layer_router_precision(false, is_qwen4, true).unwrap(),
                RouterPrecision::Nvfp4
            );
        }
    }

    #[test]
    fn selector_parse_is_exact() {
        assert!(!parse_selector(None).unwrap());
        assert!(!parse_selector(Some("0")).unwrap());
        assert!(parse_selector(Some("1")).unwrap());
        for bad in ["", "true", "01", " 1"] {
            assert!(parse_selector(Some(bad)).is_err());
        }
    }

    #[test]
    fn plain_qwen4_selects_bf16() {
        assert_eq!(
            decide_router_precision(true, qwen4()).unwrap(),
            RouterPrecision::Bf16
        );
        assert_eq!(
            layer_router_precision(true, true, false).unwrap(),
            RouterPrecision::Bf16
        );
    }

    #[test]
    fn selector_refuses_other_models_and_nvfp4_only_consumers() {
        assert!(decide_router_precision(true, RouterLaunch::default()).is_err());
        assert!(layer_router_precision(true, false, false).is_err());
        assert!(layer_router_precision(true, true, true).is_err());
        let blocked = [
            RouterLaunch {
                use_native_mtp: true,
                ..qwen4()
            },
            RouterLaunch {
                has_dflash: true,
                ..qwen4()
            },
            RouterLaunch {
                self_speculative: true,
                ..qwen4()
            },
            RouterLaunch {
                multi_batch: true,
                ..qwen4()
            },
            RouterLaunch {
                native_batch: true,
                ..qwen4()
            },
            RouterLaunch {
                multi_rank: true,
                ..qwen4()
            },
        ];
        for launch in blocked {
            assert!(decide_router_precision(true, launch).is_err(), "{launch:?}");
        }
    }

    #[test]
    fn prefill_bundle_admits_only_the_complete_control() {
        let shared = [Some(1); 3];
        assert!(bf16_router_bundle_valid(
            true, None, 1, 1, true, [None; 3], 0
        ));
        assert!(bf16_router_bundle_valid(true, None, 1, 1, false, shared, 1));
        assert!(!bf16_router_bundle_valid(
            false, None, 1, 1, true, shared, 1
        ));
        assert!(!bf16_router_bundle_valid(
            true,
            Some(1),
            1,
            1,
            true,
            shared,
            1
        ));
        assert!(!bf16_router_bundle_valid(true, None, 0, 1, true, shared, 1));
        assert!(!bf16_router_bundle_valid(true, None, 1, 0, true, shared, 1));
        assert!(!bf16_router_bundle_valid(
            true, None, 1, 1, false, shared, 0
        ));
        for missing in 0..3 {
            let mut partial = shared;
            partial[missing] = None;
            assert!(!bf16_router_bundle_valid(
                true, None, 1, 1, false, partial, 1
            ));
            partial[missing] = Some(0);
            assert!(!bf16_router_bundle_valid(
                true, None, 1, 1, false, partial, 1
            ));
        }
    }
}
