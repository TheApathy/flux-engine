// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_SHARED_BF16=1`: opt-in shared-expert precision control for
//! qwen4_exp (default-off diagnostic, also enabled by
//! `ATLAS_QWEN4_DENSE_NATIVE=1` unless explicitly overridden with `0`).
//!
//! Absent (without `ATLAS_QWEN4_DENSE_NATIVE=1`) or explicit `0` leaves the
//! shared expert as it is today: the checkpoint's MXFP8 gate/up/down
//! dequantizes exactly to BF16 at load
//! (`quant_helpers::modelopt_mixed_to_bf16`) and that BF16 tensor is
//! immediately re-quantized to NVFP4 (one per-tensor scale2,
//! `quant_helpers::quantized_modelopt_mixed`). With the selector set, the
//! loader builds `MoeWeights::shared_bf16` with the exact BF16 tensors
//! **in addition to** the usual NVFP4 `MoeWeights::shared_expert` — this
//! control is purely additive, never a null swap, because the NVFP4 copy is
//! still the only thing every prefill path (`forward_prefill_phase.rs`,
//! `qwen4_shared_v7.rs`) and every other decode branch (FP8, non-unified
//! NVFP4) reads. Decode's unified-layout dispatch
//! (`dispatch_unified_t_decode`, `layers/moe/forward_phase.rs`) runs the
//! fused NVFP4 shared+routed kernels exactly as before (their NVFP4 shared
//! answer is computed and discarded), then — only when `shared_bf16` is
//! `Some` — overwrites `shared_out` with three `ops::dense_gemv` calls plus
//! `ops::moe_silu_mul` against the BF16 tensors. The overwrite must happen
//! strictly after the fused call: with a null shared-table route those
//! kernels write *zero* to the shared outputs (not "leave untouched"), so an
//! earlier ordering was tried and found wrong (`moe_shared_expert_fused_t.cu`).
//!
//! Qualified only for the fixed C1, no-speculation, single-row decode
//! recipe, same hazard class as `qwen4_router_bf16`. Every consumer other
//! than `dispatch_unified_t_decode` (native MTP verify, DFlash verify,
//! self-speculative verify, multi-row decode, native C2..C32 batch, TP/EP,
//! every prefill path) keeps reading the NVFP4 `shared_expert` unchanged —
//! this control simply has no arm for them, so those launches are refused
//! here at load, not because a pointer would be null. The exact-prefill
//! fast family (`layers/qwen4_prefill_moe.rs`,
//! `ATLAS_QWEN4_PREFILL_MOE_BATCH=1`) can reuse decode-shaped per-row entry
//! points for some rows; it is refused here unless its own full exact
//! companion bundle is also set (`SharedLaunch::prefill_moe_batch_without_exact_bundle`).

use anyhow::{Result, bail, ensure};

pub(crate) const QWEN4_SHARED_BF16_ENV: &str = "ATLAS_QWEN4_SHARED_BF16";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SharedPrecision {
    Nvfp4,
    Bf16,
}

/// Every launch input the rule reads, captured once so the rule is pure.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SharedLaunch {
    pub(crate) is_qwen4: bool,
    pub(crate) use_native_mtp: bool,
    pub(crate) has_dflash: bool,
    pub(crate) self_speculative: bool,
    /// `max_batch_size > 1`: multi-row decode reaches the batched/K2/K3
    /// dispatch, which read `weights.shared_expert` as NVFP4 directly.
    pub(crate) multi_batch: bool,
    /// `ATLAS_QWEN4_NATIVE_BATCH=1`: native C2-C32 preflight asserts NVFP4.
    pub(crate) native_batch: bool,
    /// TP or EP world size above 1: sharded/EP shared-expert handling is
    /// not part of this control.
    pub(crate) multi_rank: bool,
    /// `ATLAS_QWEN4_PREFILL_MOE_BATCH=1` without its full exact companion
    /// bundle (`ATLAS_QWEN4_PREFILL_HC_EXACT`, `ATLAS_QWEN4_PREFILL_ATTN_QKV16`,
    /// `ATLAS_QWEN4_PREFILL_ATTN_O16`, `ATLAS_QWEN4_PREFILL_SSM_EXACT`,
    /// `ATLAS_QWEN4_PREFILL_FAST_PROJ=all`). This control has no prefill arm
    /// (see module doc); the fast prefill-batch family
    /// (`layers/qwen4_prefill_moe.rs`) reuses decode-shaped per-row entry
    /// points for some tail rows, and an incomplete exact bundle is the
    /// documented signal that a recipe is reaching one of those un-reviewed
    /// for this control. The complete bundle is its own established
    /// qualification gate (`qwen4_prefill_exact.rs:50-56`) and is not,
    /// itself, refused.
    pub(crate) prefill_moe_batch_without_exact_bundle: bool,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn parse_selector(raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("{QWEN4_SHARED_BF16_ENV} must be exactly 0 or 1"),
    }
}

/// Read once; the load gate and every layer see one answer.
pub(crate) fn selected() -> Result<bool> {
    // Umbrella-aware: ATLAS_QWEN4_DENSE_NATIVE=1 implies this control unless
    // its own env is explicitly 0 (qwen4_native_selectors).
    crate::qwen4_native_selectors::selected(crate::qwen4_native_selectors::Knob::Shared)
}

pub(crate) fn decide_shared_precision(
    selected: bool,
    launch: SharedLaunch,
) -> Result<SharedPrecision> {
    if !selected {
        return Ok(SharedPrecision::Nvfp4);
    }
    if !launch.is_qwen4 {
        bail!("{QWEN4_SHARED_BF16_ENV}=1 is a qwen4_exp control; unset it for this model");
    }
    let refusals = [
        (
            launch.use_native_mtp,
            "native MTP verifies through the exact NVFP4 shared-expert kernels; disable speculation",
        ),
        (
            launch.has_dflash,
            "DFlash verify rows need the exact NVFP4 shared-expert kernels; drop --dflash",
        ),
        (
            launch.self_speculative,
            "self-speculative verify needs the exact NVFP4 shared-expert kernels",
        ),
        (
            launch.multi_batch,
            "multi-row decode needs the exact NVFP4 shared-expert kernels; use --max-batch-size 1",
        ),
        (
            launch.native_batch,
            "native Qwen4 batch decode requires the NVFP4 shared expert; unset ATLAS_QWEN4_NATIVE_BATCH",
        ),
        (
            launch.multi_rank,
            "tensor/expert-parallel shared-expert handling is not part of this control; run TP1/EP1",
        ),
        (
            launch.prefill_moe_batch_without_exact_bundle,
            "ATLAS_QWEN4_PREFILL_MOE_BATCH=1 without the full exact bundle (ATLAS_QWEN4_PREFILL_HC_EXACT, ATLAS_QWEN4_PREFILL_ATTN_QKV16, ATLAS_QWEN4_PREFILL_ATTN_O16, ATLAS_QWEN4_PREFILL_SSM_EXACT, ATLAS_QWEN4_PREFILL_FAST_PROJ=all) may reach a decode-shaped prefill-row entry point this control has no arm for",
        ),
    ];
    if let Some((_, reason)) = refusals.iter().find(|(blocked, _)| *blocked) {
        bail!("{QWEN4_SHARED_BF16_ENV}=1 refused: {reason}");
    }
    Ok(SharedPrecision::Bf16)
}

fn env_is_one(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

/// `ATLAS_QWEN4_PREFILL_MOE_BATCH=1` set without every member of the exact
/// companion bundle also set — see `SharedLaunch::prefill_moe_batch_without_exact_bundle`.
fn prefill_moe_batch_without_exact_bundle() -> bool {
    if !env_is_one("ATLAS_QWEN4_PREFILL_MOE_BATCH") {
        return false;
    }
    let fast_proj_all = std::env::var("ATLAS_QWEN4_PREFILL_FAST_PROJ")
        .ok()
        .as_deref()
        == Some("all");
    !(env_is_one("ATLAS_QWEN4_PREFILL_HC_EXACT")
        && env_is_one("ATLAS_QWEN4_PREFILL_ATTN_QKV16")
        && env_is_one("ATLAS_QWEN4_PREFILL_ATTN_O16")
        && env_is_one("ATLAS_QWEN4_PREFILL_SSM_EXACT")
        && fast_proj_all)
}

/// Load-time gate, before any layer is built. Resolves the process-wide
/// `qwen4_native_selectors` plan and reads launch-refusal env vars
/// (`ATLAS_QWEN4_NATIVE_BATCH`, `ATLAS_QWEN4_PREFILL_*`) only when selected.
pub(crate) fn admit_launch(
    is_qwen4: bool,
    use_native_mtp: bool,
    has_dflash: bool,
    self_speculative: bool,
    max_batch_size: usize,
    multi_rank: bool,
) -> Result<SharedPrecision> {
    let selected = selected()?;
    let mut launch = SharedLaunch {
        is_qwen4,
        ..SharedLaunch::default()
    };
    if selected {
        launch = SharedLaunch {
            use_native_mtp,
            has_dflash,
            self_speculative,
            multi_batch: max_batch_size > 1,
            native_batch: env_is_one("ATLAS_QWEN4_NATIVE_BATCH"),
            multi_rank,
            prefill_moe_batch_without_exact_bundle: prefill_moe_batch_without_exact_bundle(),
            ..launch
        };
    }
    decide_shared_precision(selected, launch)
}

/// Per-layer rule inside the shared Qwen3.5/Qwen4 loader. The MTP layer goes
/// through the same loader, but only under native MTP, which `admit_launch`
/// refuses; refusing it here too keeps a bypass loud.
pub(crate) fn layer_shared_precision(
    selected: bool,
    is_qwen4: bool,
    is_mtp_layer: bool,
) -> Result<SharedPrecision> {
    if !selected {
        return Ok(SharedPrecision::Nvfp4);
    }
    ensure!(
        is_qwen4,
        "{QWEN4_SHARED_BF16_ENV}=1 is a qwen4_exp control; unset it for this model"
    );
    ensure!(
        !is_mtp_layer,
        "{QWEN4_SHARED_BF16_ENV}=1 refused: the MTP layer verifies through the exact NVFP4 shared-expert kernels"
    );
    Ok(SharedPrecision::Bf16)
}

/// Extra GPU-resident bytes the control adds per layer: three BF16 tensors
/// (gate `[inter, h]`, up `[inter, h]`, down `[h, inter]`) that would
/// otherwise be freed after the one-shot NVFP4 re-quantization, times the
/// number of qwen4 MoE layers the control actually touches (never the MTP
/// layer — see `layer_shared_precision`).
pub(crate) fn extra_resident_bytes(hidden: usize, intermediate: usize, layers: usize) -> u64 {
    let elements_per_layer = 2 * intermediate * hidden + hidden * intermediate;
    (elements_per_layer * layers * 2/* BF16 = 2 bytes/elem */) as u64
}

impl super::MoeLayer {
    /// Computes the shared expert's contribution directly from
    /// `MoeWeights::shared_bf16`'s exact BF16 tensors, writing into the same
    /// `shared_gate_scratch` / `shared_up_scratch` / `shared_out` buffers the
    /// fused NVFP4 kernels already filled. Call this AFTER
    /// `dispatch_unified_t_decode`'s fused ops have run, never before: those
    /// ops still take the real, resident NVFP4 shared tables (`shared_expert`
    /// is never null under this control) and write their own correct NVFP4
    /// answer into `shared_out`. This call overwrites that answer with the
    /// BF16 one, which is what the result `ops::moe_weighted_sum_blend` later
    /// consumes. Decode `M=1` only.
    pub(super) fn shared_decode_bf16(
        &self,
        sh: &crate::weight_map::SharedExpertBf16,
        ctx: &crate::layer::ForwardContext,
        expert_input: spark_runtime::gpu::DevicePtr,
        shared_gate_scratch: spark_runtime::gpu::DevicePtr,
        shared_up_scratch: spark_runtime::gpu::DevicePtr,
        shared_out: spark_runtime::gpu::DevicePtr,
        h: u32,
        inter: u32,
        stream: u64,
    ) -> Result<()> {
        super::ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv,
            expert_input,
            &sh.gate,
            shared_gate_scratch,
            inter,
            h,
            stream,
        )?;
        super::ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv,
            expert_input,
            &sh.up,
            shared_up_scratch,
            inter,
            h,
            stream,
        )?;
        super::ops::moe_silu_mul(
            ctx.gpu,
            self.moe_silu_mul,
            shared_gate_scratch,
            shared_up_scratch,
            shared_gate_scratch,
            inter,
            stream,
        )?;
        super::ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv,
            shared_gate_scratch,
            &sh.down,
            shared_out,
            h,
            inter,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SharedLaunch, SharedPrecision, decide_shared_precision, extra_resident_bytes,
        layer_shared_precision, parse_selector,
    };

    fn qwen4() -> SharedLaunch {
        SharedLaunch {
            is_qwen4: true,
            ..SharedLaunch::default()
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
    fn default_keeps_nvfp4_shared_expert() {
        for is_qwen4 in [false, true] {
            let launch = SharedLaunch {
                is_qwen4,
                use_native_mtp: true,
                multi_batch: true,
                native_batch: true,
                ..SharedLaunch::default()
            };
            assert_eq!(
                decide_shared_precision(false, launch).unwrap(),
                SharedPrecision::Nvfp4
            );
            assert_eq!(
                layer_shared_precision(false, is_qwen4, true).unwrap(),
                SharedPrecision::Nvfp4
            );
        }
    }

    #[test]
    fn plain_qwen4_selects_bf16() {
        assert_eq!(
            decide_shared_precision(true, qwen4()).unwrap(),
            SharedPrecision::Bf16
        );
        assert_eq!(
            layer_shared_precision(true, true, false).unwrap(),
            SharedPrecision::Bf16
        );
    }

    #[test]
    fn selector_refuses_other_models_and_nvfp4_only_consumers() {
        assert!(decide_shared_precision(true, SharedLaunch::default()).is_err());
        assert!(layer_shared_precision(true, false, false).is_err());
        assert!(layer_shared_precision(true, true, true).is_err());
        let blocked = [
            SharedLaunch {
                use_native_mtp: true,
                ..qwen4()
            },
            SharedLaunch {
                has_dflash: true,
                ..qwen4()
            },
            SharedLaunch {
                self_speculative: true,
                ..qwen4()
            },
            SharedLaunch {
                multi_batch: true,
                ..qwen4()
            },
            SharedLaunch {
                native_batch: true,
                ..qwen4()
            },
            SharedLaunch {
                multi_rank: true,
                ..qwen4()
            },
            SharedLaunch {
                prefill_moe_batch_without_exact_bundle: true,
                ..qwen4()
            },
        ];
        for launch in blocked {
            assert!(decide_shared_precision(true, launch).is_err(), "{launch:?}");
        }
    }

    #[test]
    fn extra_bytes_match_flash_next_geometry() {
        // Flash-Next: hidden=2560, shared_expert_intermediate=640, 48 MoE layers.
        // 2*640*2560 + 2560*640 = 3*640*2560 = 4_915_200 elements/layer.
        let bytes = extra_resident_bytes(2_560, 640, 48);
        assert_eq!(bytes, 4_915_200 * 48 * 2);
        assert_eq!(extra_resident_bytes(2_560, 640, 0), 0);
    }
}
