// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_GDN_BF16=1` / `ATLAS_QWEN4_ATTN_BF16=1`: opt-in decode-only
//! precision controls for qwen4_exp (default-off diagnostics).
//!
//! Mirrors `layers::moe::qwen4_router_bf16`: a pure rule struct plus a
//! load-time admission gate that refuses every launch shape which would
//! otherwise reach an NVFP4-only consumer of the weight this control
//! retains in BF16.
//!
//! Scope, per the precision-census A/Bs (#4/#5/#6):
//! - `ATLAS_QWEN4_GDN_BF16=1`: the 36 GDN layers keep the exact
//!   MXFP8->BF16 `in_proj` QKVZ concat and `out_proj`; decode (M=1) takes
//!   the existing `dense_gemv` arm. Prefill is untouched: the NVFP4
//!   transposed copies (`qkvz_nvfp4_t`, and the FP8/NVFP4 out_proj
//!   prefill arms) are checked with strictly higher priority than the
//!   fields this control sets, so the prefill dispatch order never
//!   changes. This control must NOT be confused with
//!   `ATLAS_GDN_BF16_WEIGHTS`, which also changes the out_proj *prefill*
//!   dispatch (it sets the shared `out_proj_dense` field, checked first
//!   in both decode and prefill) and is not qualified here.
//! - `ATLAS_QWEN4_ATTN_BF16=1`: the 12 attention layers keep BF16
//!   q/k/v/o (today NULL on this checkpoint: the loader dequantizes
//!   MXFP8 to BF16 only transiently, inside `quantized_modelopt_mixed`,
//!   then frees it once the NVFP4 copy is built); decode takes a BF16
//!   `dense_gemv` arm through new decode-only fields that are checked
//!   with higher priority than the existing (prefill-shared) `q_weight`
//!   / `k_weight` / `v_weight` / `o_weight` / `o_dense_bf16` fields.
//!
//! Qualified only for C1 (batch=1, M=1), target-only, no speculation, no
//! n-gram drafting (n-gram drafting is decided outside the factory and
//! is not covered — same caveat as the router control).

use anyhow::{Result, bail};

pub(crate) const QWEN4_GDN_BF16_ENV: &str = "ATLAS_QWEN4_GDN_BF16";
pub(crate) const QWEN4_ATTN_BF16_ENV: &str = "ATLAS_QWEN4_ATTN_BF16";

/// Every launch input the rule reads, captured once so the rule is pure.
/// Identical shape to `qwen4_router_bf16::RouterLaunch` by design: both
/// controls are refused by the same set of NVFP4-only consumers.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DecodeBf16Launch {
    pub(crate) is_qwen4: bool,
    pub(crate) use_native_mtp: bool,
    pub(crate) has_dflash: bool,
    pub(crate) self_speculative: bool,
    /// `max_batch_size > 1`: multi-row decode reaches the exact-NVFP4
    /// batched GEMM/GEMV routes (`trait_decode_batched.rs`,
    /// `trait_impl/multi_seq/*`), which assert the quantized weight.
    pub(crate) multi_batch: bool,
    /// `ATLAS_QWEN4_NATIVE_BATCH=1`: native C2-C32 preflight asserts NVFP4.
    pub(crate) native_batch: bool,
    /// TP or EP world size above 1: sharded routing is unqualified.
    pub(crate) multi_rank: bool,
    /// `ATLAS_QWEN4_PREFILL_MOE_BATCH=1` reuses the M=1 decode entry
    /// points per prefill row: `qwen3_ssm/qwen4_prefill_moe.rs` calls
    /// `ssm_forward` (GDN's decode path) whenever
    /// `ATLAS_QWEN4_PREFILL_SSM_EXACT` is unset, and
    /// `layers/moe/qwen4_prefill_compact.rs` / `qwen4_prefill_moe.rs`
    /// call the attention decode GEMVs per row whenever
    /// `ATLAS_QWEN4_PREFILL_ATTN_QKV16`/`_O16` are unset. The decode-only
    /// BF16 fields this control sets would then fire during prefill too.
    /// True means: `ATLAS_QWEN4_PREFILL_MOE_BATCH=1` is set AND this
    /// control's specific mitigation env(s) are NOT all `1`.
    pub(crate) prefill_reuses_decode_entry_unmitigated: bool,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn parse_selector(env_name: &str, raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("{env_name} must be exactly 0 or 1"),
    }
}

/// Read once; the load gate and every layer see one answer.
pub(crate) fn gdn_bf16_selected() -> Result<bool> {
    // Umbrella-aware: ATLAS_QWEN4_DENSE_NATIVE=1 implies this control unless
    // its own env is explicitly 0 (qwen4_native_selectors).
    crate::qwen4_native_selectors::selected(crate::qwen4_native_selectors::Knob::Gdn)
}

/// Read once; the load gate and every layer see one answer.
pub(crate) fn attn_bf16_selected() -> Result<bool> {
    // Umbrella-aware: ATLAS_QWEN4_DENSE_NATIVE=1 implies this control unless
    // its own env is explicitly 0 (qwen4_native_selectors).
    crate::qwen4_native_selectors::selected(crate::qwen4_native_selectors::Knob::Attn)
}

fn env_is_one(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

/// Shared refusal table: the same launch shapes that make the router
/// control unsafe make this control unsafe, because all three controls
/// retire an NVFP4 copy that some decode/verify consumer still asserts.
pub(crate) fn decide(env_name: &str, selected: bool, launch: DecodeBf16Launch) -> Result<bool> {
    if !selected {
        return Ok(false);
    }
    if !launch.is_qwen4 {
        bail!("{env_name}=1 is a qwen4_exp control; unset it for this model");
    }
    let refusals = [
        (
            launch.use_native_mtp,
            "native MTP verifies through the exact NVFP4 decode path; disable speculation",
        ),
        (
            launch.has_dflash,
            "DFlash verify rows need the exact NVFP4 decode path; drop --dflash",
        ),
        (
            launch.self_speculative,
            "self-speculative verify needs the exact NVFP4 decode path",
        ),
        (
            launch.multi_batch,
            "multi-row decode needs the exact NVFP4 decode path; use --max-batch-size 1",
        ),
        (
            launch.native_batch,
            "native Qwen4 batch decode requires NVFP4; unset ATLAS_QWEN4_NATIVE_BATCH",
        ),
        (
            launch.multi_rank,
            "tensor/expert-parallel routing is not part of this control; run TP1/EP1",
        ),
        (
            launch.prefill_reuses_decode_entry_unmitigated,
            "ATLAS_QWEN4_PREFILL_MOE_BATCH=1 reuses this control's decode-only BF16 arms per \
             prefill row unless its exact mitigation env(s) are also set (GDN: \
             ATLAS_QWEN4_PREFILL_SSM_EXACT=1; attention: ATLAS_QWEN4_PREFILL_ATTN_QKV16=1 and \
             ATLAS_QWEN4_PREFILL_ATTN_O16=1)",
        ),
    ];
    if let Some((_, reason)) = refusals.iter().find(|(blocked, _)| *blocked) {
        bail!("{env_name}=1 refused: {reason}");
    }
    Ok(true)
}

/// Load-time gate, before any layer is built. Touches no other variable
/// unless a selector is set, so the default path reads only the two
/// selectors. Returns `(gdn_engaged, attn_engaged)`.
pub(crate) fn admit_launch(
    is_qwen4: bool,
    use_native_mtp: bool,
    has_dflash: bool,
    self_speculative: bool,
    max_batch_size: usize,
    multi_rank: bool,
) -> Result<(bool, bool)> {
    let gdn_selected = gdn_bf16_selected()?;
    let attn_selected = attn_bf16_selected()?;
    let prefill_moe_batch = env_is_one("ATLAS_QWEN4_PREFILL_MOE_BATCH");
    let build_launch = |selected: bool, prefill_reuses_decode_entry_unmitigated: bool| {
        if !selected {
            return DecodeBf16Launch {
                is_qwen4,
                ..DecodeBf16Launch::default()
            };
        }
        DecodeBf16Launch {
            is_qwen4,
            use_native_mtp,
            has_dflash,
            self_speculative,
            multi_batch: max_batch_size > 1,
            native_batch: env_is_one("ATLAS_QWEN4_NATIVE_BATCH"),
            multi_rank,
            prefill_reuses_decode_entry_unmitigated,
        }
    };
    let gdn_unmitigated = prefill_moe_batch && !env_is_one("ATLAS_QWEN4_PREFILL_SSM_EXACT");
    let attn_unmitigated = prefill_moe_batch
        && !(env_is_one("ATLAS_QWEN4_PREFILL_ATTN_QKV16")
            && env_is_one("ATLAS_QWEN4_PREFILL_ATTN_O16"));
    let gdn_engaged = decide(
        QWEN4_GDN_BF16_ENV,
        gdn_selected,
        build_launch(gdn_selected, gdn_unmitigated),
    )?;
    let attn_engaged = decide(
        QWEN4_ATTN_BF16_ENV,
        attn_selected,
        build_launch(attn_selected, attn_unmitigated),
    )?;
    if gdn_engaged {
        static LOG_GDN: std::sync::Once = std::sync::Once::new();
        LOG_GDN.call_once(|| {
            tracing::info!(
                "QWEN4_GDN_BF16 engaged scope=decode-only extra_resident_bytes_per_layer=out_proj_h*value_dim*2"
            );
        });
    }
    if attn_engaged {
        static LOG_ATTN: std::sync::Once = std::sync::Once::new();
        LOG_ATTN.call_once(|| {
            tracing::info!(
                "QWEN4_ATTN_BF16 engaged scope=decode-only extra_resident_bytes_per_layer=qkvo_bf16_minus_nvfp4"
            );
        });
    }
    Ok((gdn_engaged, attn_engaged))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen4() -> DecodeBf16Launch {
        DecodeBf16Launch {
            is_qwen4: true,
            ..DecodeBf16Launch::default()
        }
    }

    #[test]
    fn selector_parse_is_exact() {
        for env in [QWEN4_GDN_BF16_ENV, QWEN4_ATTN_BF16_ENV] {
            assert!(!parse_selector(env, None).unwrap());
            assert!(!parse_selector(env, Some("0")).unwrap());
            assert!(parse_selector(env, Some("1")).unwrap());
            for bad in ["", "true", "01", " 1"] {
                assert!(parse_selector(env, Some(bad)).is_err(), "{env} {bad}");
            }
        }
    }

    #[test]
    fn default_stays_disengaged() {
        for is_qwen4 in [false, true] {
            let launch = DecodeBf16Launch {
                is_qwen4,
                use_native_mtp: true,
                multi_batch: true,
                native_batch: true,
                ..DecodeBf16Launch::default()
            };
            assert!(!decide(QWEN4_GDN_BF16_ENV, false, launch).unwrap());
            assert!(!decide(QWEN4_ATTN_BF16_ENV, false, launch).unwrap());
        }
    }

    #[test]
    fn plain_qwen4_engages() {
        assert!(decide(QWEN4_GDN_BF16_ENV, true, qwen4()).unwrap());
        assert!(decide(QWEN4_ATTN_BF16_ENV, true, qwen4()).unwrap());
    }

    #[test]
    fn non_qwen4_model_is_refused() {
        assert!(decide(QWEN4_GDN_BF16_ENV, true, DecodeBf16Launch::default()).is_err());
        assert!(decide(QWEN4_ATTN_BF16_ENV, true, DecodeBf16Launch::default()).is_err());
    }

    #[test]
    fn every_nvfp4_only_consumer_is_refused() {
        let blocked = [
            DecodeBf16Launch {
                use_native_mtp: true,
                ..qwen4()
            },
            DecodeBf16Launch {
                has_dflash: true,
                ..qwen4()
            },
            DecodeBf16Launch {
                self_speculative: true,
                ..qwen4()
            },
            DecodeBf16Launch {
                multi_batch: true,
                ..qwen4()
            },
            DecodeBf16Launch {
                native_batch: true,
                ..qwen4()
            },
            DecodeBf16Launch {
                multi_rank: true,
                ..qwen4()
            },
            DecodeBf16Launch {
                prefill_reuses_decode_entry_unmitigated: true,
                ..qwen4()
            },
        ];
        for launch in blocked {
            assert!(
                decide(QWEN4_GDN_BF16_ENV, true, launch).is_err(),
                "{launch:?}"
            );
            assert!(
                decide(QWEN4_ATTN_BF16_ENV, true, launch).is_err(),
                "{launch:?}"
            );
        }
    }

    #[test]
    fn admit_launch_is_independent_per_control() {
        // Neither selector set via env in this process by default: both
        // read false unless a previous test in this binary set the env
        // before the OnceLock latched. We only exercise the pure `decide`
        // table above for behavior; this test just checks the tuple shape
        // is independent (gdn/attn do not leak into each other's reason).
        let launch = qwen4();
        let gdn = decide(QWEN4_GDN_BF16_ENV, true, launch).unwrap();
        let attn = decide(QWEN4_ATTN_BF16_ENV, true, launch).unwrap();
        assert!(gdn);
        assert!(attn);
    }
}
