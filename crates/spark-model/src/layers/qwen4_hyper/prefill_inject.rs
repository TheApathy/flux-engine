// SPDX-License-Identifier: AGPL-3.0-only

//! Deferred hyperconnection injection for Flash-Next prefill (default off).
//!
//! `ATLAS_QWEN4_PREFILL_MHC_FUSED=<list>` selects levers, comma separated:
//! - `inject`: a core sublayer (GDN or attention) leaves its saved-scale
//!   injection pending for the MLP hyperconnection of the same layer;
//! - `cross`: the MLP hyperconnection leaves its injection pending for the
//!   next layer's attention hyperconnection.
//!
//! - `upmix`: the full-row fast HC path runs the mix in the up projection's
//!   epilogue (`qwen4_hc_upmix`, see [`super::prefill_upmix`]); on the
//!   HC_FUSED path the same tiling replaces its up+mix kernel byte-for-byte
//!   (`qwen4_hc_upmix_hcfused`);
//! - `nostage`: preparations whose consumer reads only the packed mixed rows
//!   (the MoE FFN, the full-row GDN core) skip `qwen4_hc_stage_mixed`;
//! - `combine` (with `cross`): the MoE FFN also leaves its output combine
//!   (unpermute + shared-expert blend) pending, and the next preparation
//!   forms the output rows in the same pass (`qwen4_hc_combine_inject_norm`);
//! - `injgemm`: the HC_FUSED preparation applies a pending core injection
//!   (or none), its rstd pass and its four down|inject projections in one
//!   kernel (`qwen4_hc_inject_gemm_ws[_l2h]` in `qwen4_hc_combine_gemm.cu`,
//!   see [`super::prefill_inject_gemm`]);
//! - `postpre` (with `combine` and `injgemm`): a pending MoE output is formed
//!   once per row and injected, normalised and projected in the same cluster
//!   kernel (`qwen4_hc_combine_moe_gemm[_mx]`), so the injected hyper rows are
//!   read once instead of twice at a layer boundary;
//! - `postsplit`: the same row formation as its own kernel
//!   (`qwen4_hc_moe_rows[_mx]`), then the core-injection pass. An
//!   alternative to `postpre`, not combinable with it;
//! - `gatepre`: consume precomputed shared-expert gate scalars (may also be
//!   enabled independently via `ATLAS_QWEN4_PREFILL_MHC_GATEPRE=1|on|true`).
//!
//! `1` or `all` selects every lever except `postpre`, `postsplit`, and
//! `gatepre`; absent or `0` keeps the unfused path.
//! The next [`Qwen4HyperConnection::prepare_prefill_exact`] on the full-row
//! fast HC path applies a pending injection together with its group RMS norm
//! in one pass (`qwen4_hc_inject_norm`), byte-identical to
//! `qwen4_hc_inject_saved` followed by `qwen4_hc_group_norm`. Every other
//! reader of the hyper rows applies it with the unfused kernel first
//! (see [`super::pending`]).

use std::sync::OnceLock;

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen4HyperConnection;
use super::pending::{self, MoeCombine, PendingInject};
use crate::layers::qwen4_fast_proj;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MHC_FUSED";

const FUSED_HIDDEN: usize = 2560;
const FUSED_STREAMS: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Selection {
    pub inject: bool,
    pub cross: bool,
    pub upmix: bool,
    pub nostage: bool,
    pub combine: bool,
    pub injgemm: bool,
    pub postpre: bool,
    pub postsplit: bool,
    pub gatepre: bool,
}

impl Selection {
    fn merge(&mut self, other: Self) {
        self.inject |= other.inject;
        self.cross |= other.cross;
        self.upmix |= other.upmix;
        self.nostage |= other.nostage;
        self.combine |= other.combine;
        self.injgemm |= other.injgemm;
        self.postpre |= other.postpre;
        self.postsplit |= other.postsplit;
        self.gatepre |= other.gatepre;
    }
}

const ALL: Selection = Selection {
    inject: true,
    cross: true,
    upmix: true,
    nostage: true,
    combine: true,
    injgemm: true,
    postpre: false,
    postsplit: false,
    gatepre: false,
};

/// Which hyperconnection boundary an injection crosses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Boundary {
    /// GDN/attention core into the same layer's MLP hyperconnection.
    Core,
    /// MoE output into the next layer's attention hyperconnection.
    Layer,
}

pub(crate) fn parse(value: Option<&str>) -> Result<Selection> {
    let mut sel = Selection::default();
    let Some(value) = value else { return Ok(sel) };
    let value = value.trim();
    if value.is_empty() || value == "0" {
        return Ok(sel);
    }
    for item in value.split(',') {
        match item.trim() {
            "1" | "all" => sel.merge(ALL),
            "inject" => sel.inject = true,
            "cross" => sel.cross = true,
            "upmix" => sel.upmix = true,
            "nostage" => sel.nostage = true,
            "combine" => sel.combine = true,
            "injgemm" => sel.injgemm = true,
            "postpre" => sel.postpre = true,
            "postsplit" => sel.postsplit = true,
            "gatepre" => sel.gatepre = true,
            other => bail!(
                "{SELECTOR}: unknown lever {other:?} (expected inject, cross, upmix, \
                 nostage, combine, injgemm, postpre, postsplit, gatepre, all, 1)"
            ),
        }
    }
    if sel.postpre && sel.postsplit {
        bail!("{SELECTOR}: postpre and postsplit are alternatives, select one");
    }
    Ok(sel)
}

pub(crate) const GATEPRE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MHC_GATEPRE";

/// Parsed once per process; a malformed selector fails the first prefill.
pub(crate) fn selection() -> Result<Selection> {
    static SEL: OnceLock<Result<Selection, String>> = OnceLock::new();
    SEL.get_or_init(|| {
        let raw = std::env::var(SELECTOR).ok();
        let mut sel = parse(raw.as_deref()).map_err(|e| e.to_string())?;
        if let Ok(v) = std::env::var(GATEPRE_SELECTOR) {
            match v.trim().to_ascii_lowercase().as_str() {
                "" | "0" | "off" | "false" => {}
                "1" | "on" | "true" => sel.gatepre = true,
                other => {
                    return Err(format!(
                        "{GATEPRE_SELECTOR}: unknown value {other:?} (expected 0, 1, off, on, false, true)"
                    ));
                }
            }
        }
        Ok(sel)
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

fn aligned16(ptr: DevicePtr) -> bool {
    !ptr.is_null() && ptr.0.is_multiple_of(16)
}

pub(super) fn disjoint(a: DevicePtr, a_bytes: usize, b: DevicePtr, b_bytes: usize) -> bool {
    a.0.saturating_add(a_bytes as u64) <= b.0 || b.0.saturating_add(b_bytes as u64) <= a.0
}

impl Qwen4HyperConnection {
    fn canonical_fused_geometry(&self) -> bool {
        self.hidden_size == FUSED_HIDDEN && self.hc_count == FUSED_STREAMS
    }

    fn deferral_selected(&self, boundary: Boundary) -> Result<bool> {
        let sel = selection()?;
        let lever = match boundary {
            Boundary::Core => sel.inject,
            Boundary::Layer => sel.cross,
        };
        let fusing_prepare = qwen4_fast_proj::selection()?.hc || super::prefill_fused::level() != 0;
        Ok(lever && fusing_prepare && self.canonical_fused_geometry())
    }

    /// Whether this (MLP) hyperconnection's MoE may leave its output combine
    /// pending: lever `combine` on top of a deferred layer-boundary injection.
    pub(crate) fn combine_deferrable(&self) -> Result<bool> {
        Ok(selection()?.combine && self.deferral_selected(Boundary::Layer)?)
    }

    /// Apply `hyper += core * saved_scale` now, or leave it pending for the
    /// next hyperconnection preparation when the lever for `boundary` is
    /// selected and that preparation can fuse it. `combine`: the MoE output
    /// rows in `core` still have to be formed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inject_saved_or_defer(
        &self,
        boundary: Boundary,
        hyper: DevicePtr,
        core: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        combine: Option<MoeCombine>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if self.deferral_selected(boundary)? {
            return pending::defer(PendingInject {
                hyper,
                core,
                residual,
                rows: num_tokens,
                combine,
            });
        }
        if let Some(c) = &combine {
            pending::materialize_combine(c, core, num_tokens, gpu, stream)?;
        }
        self.inject_saved_batched(hyper, core, residual, num_tokens, gpu, stream)
    }

    /// Whether the fused kernel can consume `p` for this preparation: the
    /// same rows, canonical geometry, 16-byte aligned rows and a core buffer
    /// disjoint from both hyper and residual.
    fn inject_norm_admitted(&self, p: &PendingInject) -> bool {
        let row = self.residual_width() * 2;
        let core_bytes = p.rows * self.hidden_size * 2;
        let combine_ok = p.combine.is_none_or(|c| {
            let gate_ok = if !c.gate_scalars.is_null() {
                c.gate_scalars.0.is_multiple_of(4)
            } else {
                !c.gate_input.is_null()
                    && c.gate_input.0.is_multiple_of(4)
                    && c.shared_gate.0.is_multiple_of(2)
            };
            c.topk > 0
                && c.topk <= 16
                && [c.expert_out, c.token_to_perm, c.topk_weights]
                    .into_iter()
                    .all(|ptr| !ptr.is_null() && ptr.0.is_multiple_of(4))
                && !c.shared_out.is_null()
                && c.shared_out.0.is_multiple_of(8)
                && gate_ok
        });
        combine_ok
            && self.canonical_fused_geometry()
            && self.inject.is_some()
            && p.rows > 0
            && p.rows <= u32::MAX as usize
            && [p.hyper, p.core, p.residual, self.norm.weight]
                .into_iter()
                .all(aligned16)
            && disjoint(p.core, core_bytes, p.hyper, p.rows * row)
            && disjoint(p.core, core_bytes, p.residual, p.rows * row)
    }

    /// [`Self::take_fusable_pending`] for the HC_FUSED preparation, which
    /// fuses the injection into its rstd pass instead of the group norm.
    pub(super) fn take_rstd_fusable_pending(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<PendingInject>> {
        let Some(p) = pending::take() else {
            return Ok(None);
        };
        if p.hyper == hyper
            && p.residual == residual
            && p.rows == num_tokens
            && self.inject_norm_admitted(&p)
        {
            return Ok(Some(p));
        }
        pending::apply(&p, gpu, stream)?;
        Ok(None)
    }

    /// Take the pending injection owed to exactly these hyper rows when the
    /// fast HC path will fuse it; any other pending injection is applied now.
    pub(super) fn take_fusable_pending(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<PendingInject>> {
        let Some(p) = pending::take() else {
            return Ok(None);
        };
        // qwen4_hc_combine_inject_norm reads bf16 slot rows only.
        if p.hyper == hyper
            && p.residual == residual
            && p.rows == num_tokens
            && qwen4_fast_proj::selection()?.hc
            && p.combine.is_none_or(|c| c.slot_scales.is_null())
            && self.inject_norm_admitted(&p)
        {
            return Ok(Some(p));
        }
        pending::apply(&p, gpu, stream)?;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lever_lists() {
        let none = Selection::default();
        assert_eq!(parse(None).unwrap(), none);
        assert_eq!(parse(Some("0")).unwrap(), none);
        assert_eq!(parse(Some("")).unwrap(), none);
        assert_eq!(parse(Some("1")).unwrap(), ALL);
        assert_eq!(parse(Some("all")).unwrap(), ALL);
        assert_eq!(
            parse(Some("inject")).unwrap(),
            Selection {
                inject: true,
                ..none
            }
        );
        assert_eq!(
            parse(Some("cross")).unwrap(),
            Selection {
                cross: true,
                ..none
            }
        );
        assert_eq!(
            parse(Some("upmix")).unwrap(),
            Selection {
                upmix: true,
                ..none
            }
        );
        assert_eq!(
            parse(Some("nostage")).unwrap(),
            Selection {
                nostage: true,
                ..none
            }
        );
        assert_eq!(
            parse(Some("combine")).unwrap(),
            Selection {
                combine: true,
                ..none
            }
        );
        assert_eq!(
            parse(Some("injgemm")).unwrap(),
            Selection {
                injgemm: true,
                ..none
            }
        );
        assert_eq!(
            parse(Some("inject, cross,upmix,nostage,combine,injgemm")).unwrap(),
            ALL
        );
        assert_eq!(
            parse(Some("all,postpre")).unwrap(),
            Selection {
                postpre: true,
                ..ALL
            }
        );
        assert_eq!(
            parse(Some("all,postpre,gatepre")).unwrap(),
            Selection {
                postpre: true,
                gatepre: true,
                ..ALL
            }
        );
        assert_eq!(
            parse(Some("postpre,all")).unwrap(),
            parse(Some("all,postpre")).unwrap()
        );
        assert_eq!(
            parse(Some("postsplit,all")).unwrap(),
            parse(Some("all,postsplit")).unwrap()
        );
        assert_eq!(
            parse(Some("gatepre,all")).unwrap(),
            parse(Some("all,gatepre")).unwrap()
        );
        assert_eq!(
            parse(Some("gatepre,1")).unwrap(),
            parse(Some("1,gatepre")).unwrap()
        );
        assert_eq!(
            parse(Some("postsplit")).unwrap(),
            Selection {
                postsplit: true,
                ..none
            }
        );
        assert!(parse(Some("postpre,postsplit")).is_err());
        assert!(parse(Some("fold")).is_err());
        assert!(parse(Some("yes")).is_err());
    }

    #[test]
    fn alignment_and_overlap() {
        assert!(aligned16(DevicePtr(0x1000)));
        assert!(!aligned16(DevicePtr(0x1008)));
        assert!(!aligned16(DevicePtr::NULL));
        assert!(disjoint(DevicePtr(0x1000), 0x100, DevicePtr(0x1100), 0x10));
        assert!(!disjoint(DevicePtr(0x1000), 0x100, DevicePtr(0x10f0), 0x20));
        assert!(!disjoint(DevicePtr(0x1100), 0x10, DevicePtr(0x1000), 0x101));
    }
}
