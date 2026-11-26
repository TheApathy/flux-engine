// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_HC_BF16=1`: opt-in hyper-connection (HC) precision control
//! for qwen4_exp decode (default-off diagnostic).
//!
//! Absent or `0` leaves every HC mixer as it is today: qwen4_exp
//! re-quantizes the checkpoint BF16 `input_mix_weight_down/up` and
//! `block_inject_weight` (96 per-layer mixers) plus the final
//! `hyper_connection_mixer` down/up to NVFP4 at load (one per-tensor
//! scale2 each, `quantize_to_nvfp4_cached`); decode takes `w4a16_gemv_pf_dual`
//! / `qwen4_hc_up_mix_fused` on that NVFP4 copy, and prefill reads the same
//! NVFP4 tensors directly (`prefill_fused.rs`).
//!
//! With the selector set, every admitted qwen4_exp layer *additionally*
//! keeps the loader's BF16 dense copies of down/up/inject (and the final
//! mixer's down/up) resident instead of letting them go out of scope once
//! quantized. The M=1 decode mixer (`Qwen4HyperConnection::prepare_decode_post_norm`)
//! then takes a `dense_gemv` arm on those BF16 tensors in place of the NVFP4
//! GEMVs. The NVFP4 copies are left exactly as they are and keep serving
//! prefill unchanged — this control has no BF16 prefill arm, so the several
//! prefill-row callers that reuse the M=1 decode entry point
//! (`prepare_prefill_exact_staged`'s tail tile, `prefill_qwen4_moe_batch`,
//! `prefill_moe_only`, the exact-prefill checks) call
//! `Qwen4HyperConnection::prepare_decode_nvfp4` instead of `prepare_decode`,
//! so prefill is never silently touched. Qualified only for the fixed C1, no-speculation,
//! no-n-gram recipe, same refusal set as `qwen4_router_bf16`: the exact
//! K2..K9 MoE path and native C2..C32 batch decode read the HC mixer under
//! routes this control was not measured against, so they are refused at
//! load before any layer is built.

use anyhow::{Result, bail, ensure};
use spark_runtime::weights::WeightDtype;

pub(crate) const QWEN4_HC_BF16_ENV: &str = "ATLAS_QWEN4_HC_BF16";

/// Every launch input the rule reads, captured once so the rule is pure.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HcLaunch {
    pub(crate) is_qwen4: bool,
    pub(crate) use_native_mtp: bool,
    pub(crate) has_dflash: bool,
    pub(crate) self_speculative: bool,
    /// `max_batch_size > 1`: multi-row decode reaches the batched-verify HC
    /// tail (`HC_DUAL_ROWS`/`HC_TAIL_ROWS`), which this control does not add
    /// a BF16 arm for.
    pub(crate) multi_batch: bool,
    /// `ATLAS_QWEN4_NATIVE_BATCH=1`: native C2-C32 preflight asserts NVFP4.
    pub(crate) native_batch: bool,
    /// TP or EP world size above 1: sharded HC mixing is unqualified.
    pub(crate) multi_rank: bool,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn parse_selector(raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("{QWEN4_HC_BF16_ENV} must be exactly 0 or 1"),
    }
}

/// Read once; the load gate and every layer see one answer.
pub(crate) fn selected() -> Result<bool> {
    // Umbrella-aware: ATLAS_QWEN4_DENSE_NATIVE=1 implies this control unless
    // its own env is explicitly 0 (qwen4_native_selectors).
    crate::qwen4_native_selectors::selected(crate::qwen4_native_selectors::Knob::Hc)
}

/// True when this launch keeps the HC mixers' BF16 copies resident for
/// decode. `false` (including every refusal reason) keeps today's
/// NVFP4-only behaviour.
pub(crate) fn decide_keep_bf16(selected: bool, launch: HcLaunch) -> Result<bool> {
    if !selected {
        return Ok(false);
    }
    if !launch.is_qwen4 {
        bail!("{QWEN4_HC_BF16_ENV}=1 is a qwen4_exp control; unset it for this model");
    }
    let refusals = [
        (
            launch.use_native_mtp,
            "native MTP verifies through the NVFP4 HC mixer; disable speculation",
        ),
        (
            launch.has_dflash,
            "DFlash verify rows need the NVFP4 HC mixer; drop --dflash",
        ),
        (
            launch.self_speculative,
            "self-speculative verify needs the NVFP4 HC mixer",
        ),
        (
            launch.multi_batch,
            "multi-row decode needs the NVFP4 HC batched-verify tail; use --max-batch-size 1",
        ),
        (
            launch.native_batch,
            "native Qwen4 batch decode requires the NVFP4 HC mixer; unset ATLAS_QWEN4_NATIVE_BATCH",
        ),
        (
            launch.multi_rank,
            "tensor/expert-parallel HC mixing is not part of this control; run TP1/EP1",
        ),
    ];
    if let Some((_, reason)) = refusals.iter().find(|(blocked, _)| *blocked) {
        bail!("{QWEN4_HC_BF16_ENV}=1 refused: {reason}");
    }
    Ok(true)
}

fn env_is_one(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

/// Load-time gate, before any layer is built. Touches no other variable
/// unless the selector is set, so the default path reads only the selector.
pub(crate) fn admit_launch(
    is_qwen4: bool,
    use_native_mtp: bool,
    has_dflash: bool,
    self_speculative: bool,
    max_batch_size: usize,
    multi_rank: bool,
) -> Result<bool> {
    let selected = selected()?;
    let mut launch = HcLaunch {
        is_qwen4,
        ..HcLaunch::default()
    };
    if selected {
        launch = HcLaunch {
            use_native_mtp,
            has_dflash,
            self_speculative,
            multi_batch: max_batch_size > 1,
            native_batch: env_is_one("ATLAS_QWEN4_NATIVE_BATCH"),
            multi_rank,
            ..launch
        };
    }
    decide_keep_bf16(selected, launch)
}

/// The checkpoint HC tensor as the `WeightStore` holds it, before the
/// loader decides whether to retain it. Mirrors `qwen4_router_bf16`'s
/// `StoredRouter`, generalised to any of the HC mixer's three matrices
/// (`down`, `up`, `inject`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct StoredHcTensor<'a> {
    pub(crate) dtype: WeightDtype,
    pub(crate) shape: &'a [usize],
}

/// The control is only meaningful for a tensor that is BF16 *in the
/// checkpoint*; a dequantized or mixed-precision copy is not a BF16
/// control, so it is refused rather than silently kept.
pub(crate) fn validate_stored_bf16_tensor(
    key: &str,
    tensor: StoredHcTensor<'_>,
    expected_shape: &[usize],
) -> Result<()> {
    ensure!(
        tensor.dtype == WeightDtype::BF16,
        "{QWEN4_HC_BF16_ENV}=1 refused: {key} is stored as {:?}, not BF16; \
         a dequantized tensor is not a BF16 control",
        tensor.dtype
    );
    ensure!(
        tensor.shape == expected_shape,
        "{QWEN4_HC_BF16_ENV}=1 refused: {key} shape {:?} is not {expected_shape:?}",
        tensor.shape
    );
    Ok(())
}

/// True extra GPU bytes this control allocates: always 0. `dense_auto`
/// returns a pointer into the `WeightStore`'s own resident tensor, never a
/// copy, so retaining it costs nothing beyond the pointer already on the
/// stack; see [`aliased_bf16_bytes`] for the (non-extra) byte count it
/// keeps reachable.
pub(crate) const EXTRA_RESIDENT_BYTES: u64 = 0;

/// BF16 bytes this control keeps a `DenseWeight` pointer alive for, across
/// every HC mixer (2 per hidden layer, plus the final mixer, which has no
/// `block_inject`), when engaged. These are an alias into the `WeightStore`
/// tensors the loader already holds (`dense_auto` returns the stored
/// `w.ptr` directly, no copy) and which are never freed on the default
/// path either — so this is **not** extra GPU memory, only extra pointers
/// this control keeps referencing past the point where today's code lets
/// them go out of scope ([`EXTRA_RESIDENT_BYTES`] is always 0). Reported in
/// the startup receipt only for visibility into how much of the checkpoint
/// stays reachable.
pub(crate) fn aliased_bf16_bytes(
    rank: usize,
    residual_width: usize,
    hc_count: usize,
    num_hidden_layers: usize,
) -> u64 {
    let down_up = 4 * rank * residual_width; // down [rank x r] + up [r x rank], 2 bytes/elem
    let inject = 2 * hc_count * residual_width; // block_inject [hc_count x r], 2 bytes/elem
    let per_layer_mixers = 2; // attn_hyper_connection + mlp_hyper_connection
    let per_layer = per_layer_mixers * (down_up + inject);
    let final_mixer = down_up; // no block_inject on the terminal mixer
    (per_layer * num_hidden_layers + final_mixer) as u64
}

/// Pure decision the loader applies at each HC call site: when `keep` is
/// `false` (the default path, or any refused launch), every slot is `None`
/// — byte-identical to today, no BF16 copy survives the quantize call.
/// When `true`, every slot the checkpoint actually has (down/up always,
/// inject only when the mixer takes injection) is retained.
pub(crate) fn resident_copies<T: Copy>(
    keep: bool,
    down: T,
    up: T,
    inject: Option<T>,
) -> (Option<T>, Option<T>, Option<T>) {
    if keep {
        (Some(down), Some(up), inject)
    } else {
        (None, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        HcLaunch, StoredHcTensor, WeightDtype, admit_launch, decide_keep_bf16, parse_selector,
        validate_stored_bf16_tensor,
    };

    fn qwen4() -> HcLaunch {
        HcLaunch {
            is_qwen4: true,
            ..HcLaunch::default()
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
    fn default_keeps_nvfp4_only() {
        let launch = HcLaunch {
            is_qwen4: true,
            use_native_mtp: true,
            multi_batch: true,
            native_batch: true,
            ..HcLaunch::default()
        };
        assert!(!decide_keep_bf16(false, launch).unwrap());
    }

    #[test]
    fn plain_qwen4_keeps_bf16() {
        assert!(decide_keep_bf16(true, qwen4()).unwrap());
    }

    #[test]
    fn selector_refuses_other_models_and_nvfp4_only_consumers() {
        assert!(decide_keep_bf16(true, HcLaunch::default()).is_err());
        let blocked = [
            HcLaunch {
                use_native_mtp: true,
                ..qwen4()
            },
            HcLaunch {
                has_dflash: true,
                ..qwen4()
            },
            HcLaunch {
                self_speculative: true,
                ..qwen4()
            },
            HcLaunch {
                multi_batch: true,
                ..qwen4()
            },
            HcLaunch {
                native_batch: true,
                ..qwen4()
            },
            HcLaunch {
                multi_rank: true,
                ..qwen4()
            },
        ];
        for launch in blocked {
            assert!(decide_keep_bf16(true, launch).is_err(), "{launch:?}");
        }
    }

    #[test]
    fn admit_launch_matches_decide_keep_bf16() {
        // No env set: NotPresent branch, selected() must resolve to false
        // regardless of other launch inputs.
        assert!(!admit_launch(true, true, true, true, 8, true).unwrap());
    }

    #[test]
    fn stored_tensor_must_be_checkpoint_bf16_with_expected_shape() {
        let shape = [320, 10_240];
        let key =
            "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight";
        let check = |dtype, s: &[usize]| {
            validate_stored_bf16_tensor(key, StoredHcTensor { dtype, shape: s }, &shape)
        };
        assert!(check(WeightDtype::BF16, &shape).is_ok());
        for dtype in [
            WeightDtype::FP8E4M3,
            WeightDtype::UInt8,
            WeightDtype::FP32,
            WeightDtype::FP8E8M0,
        ] {
            assert!(check(dtype, &shape).is_err(), "{dtype:?}");
        }
        assert!(check(WeightDtype::BF16, &[10_240, 320]).is_err());
    }

    #[test]
    fn aliased_bytes_matches_census_report_order_of_magnitude() {
        use super::aliased_bf16_bytes;

        // rank=320 (hc_lowrank), r=2_560*4=10_240 (residual_width), hc_count=4,
        // 48 hidden layers — the precision-census geometry for qwen4_exp.
        // This is the aliased (already-resident) byte count, not extra
        // allocation — see `aliased_bf16_bytes`'s doc comment.
        let bytes = aliased_bf16_bytes(320, 10_240, 4, 48);
        let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        // Order-of-magnitude sanity check on the aliased BF16 footprint for
        // 48 layers + final mixer (~0.95–1.2 GiB).
        assert!(
            (0.8..=1.4).contains(&gib),
            "{gib} GiB out of expected range"
        );
    }

    #[test]
    fn aliased_bytes_is_zero_for_zero_layers_and_zero_geometry() {
        use super::aliased_bf16_bytes;
        assert_eq!(aliased_bf16_bytes(0, 0, 0, 0), 0);
        assert_eq!(aliased_bf16_bytes(320, 10_240, 4, 0), 4 * 320 * 10_240); // final mixer only
    }

    #[test]
    fn extra_resident_bytes_is_always_zero() {
        // This control never allocates new GPU memory: it only keeps a
        // `DenseWeight` pointer into an already-resident `WeightStore`
        // tensor alive past where today's code drops it.
        assert_eq!(super::EXTRA_RESIDENT_BYTES, 0);
    }

    #[test]
    fn resident_copies_default_path_drops_every_bf16_slot() {
        use super::resident_copies;
        let (down, up, inject) = resident_copies(false, 1u64, 2u64, Some(3u64));
        assert_eq!((down, up, inject), (None, None, None));
    }

    #[test]
    fn resident_copies_engaged_path_keeps_every_present_slot() {
        use super::resident_copies;
        // Per-layer HC mixer: down/up/inject all present.
        assert_eq!(
            resident_copies(true, 1u64, 2u64, Some(3u64)),
            (Some(1u64), Some(2u64), Some(3u64))
        );
        // Final mixer: no block_inject slot in the checkpoint.
        assert_eq!(
            resident_copies(true, 1u64, 2u64, None::<u64>),
            (Some(1u64), Some(2u64), None)
        );
    }
}
