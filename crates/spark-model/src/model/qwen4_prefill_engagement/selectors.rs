// SPDX-License-Identifier: AGPL-3.0-only

use super::{ATTN_SELECTOR, PrefillPath, SSM_SELECTOR};
use crate::layers::qwen3_ssm::qwen4_prefill_gemm::{self, Mode};
use anyhow::Result;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Selectors {
    pub(super) attention: bool,
    pub(super) ssm: bool,
    pub(super) moe_only: bool,
    pub(super) hyper_exact: bool,
    pub(super) ssm_exact: bool,
    pub(super) moe_compact: bool,
    pub(super) ssm_gemm: Mode,
    pub(super) slot_mxfp8_r2: bool,
}

impl Selectors {
    pub(super) fn from_values(attention: Option<&str>, ssm: Option<&str>) -> Self {
        Self {
            attention: attention == Some("1"),
            ssm: ssm == Some("1"),
            moe_only: false,
            hyper_exact: false,
            ssm_exact: false,
            moe_compact: false,
            ssm_gemm: Mode::Off,
            slot_mxfp8_r2: false,
        }
    }

    pub(super) fn with_moe_only(mut self, selected: bool) -> Self {
        self.moe_only = selected;
        self.attention |= selected;
        self.ssm |= selected;
        self
    }

    pub(super) fn with_slot_mxfp8_r2(mut self, selected: bool) -> Self {
        // The raw slot selector is load-time configuration, while `moe_only`
        // is request-local and becomes false when the fast family is
        // suppressed.  Only arm an R2 receipt for an effectively selected
        // grouped-MoE request; suppressed requests must take the BF16 fallback.
        self.slot_mxfp8_r2 = self.moe_only && selected;
        self
    }

    pub(super) fn from_env() -> Result<Self> {
        let attention = std::env::var(ATTN_SELECTOR).ok();
        let ssm = std::env::var(SSM_SELECTOR).ok();
        let mut selectors = Self::from_values(attention.as_deref(), ssm.as_deref())
            .with_moe_only(crate::layers::qwen4_prefill_moe::selected()?);
        selectors.hyper_exact = crate::layers::qwen4_prefill_moe::hyper_selected()?;
        selectors.ssm_exact = crate::layers::qwen3_ssm::qwen4_prefill_exact::selected()?;
        selectors.moe_compact = crate::layers::moe::qwen4_prefill_compact::selected()?;
        selectors.ssm_gemm = qwen4_prefill_gemm::mode()?;
        let slot_mxfp8_r2 = crate::layers::moe::qwen4_prefill_compact::slot_mxfp8_r2_selected()?;
        Ok(selectors.with_slot_mxfp8_r2(slot_mxfp8_r2))
    }

    pub(super) const fn any(self) -> bool {
        self.attention || self.ssm || self.slot_mxfp8_r2
    }

    pub(super) const fn value(self, path: PrefillPath) -> bool {
        match path {
            PrefillPath::Attention => self.attention,
            PrefillPath::Ssm => self.ssm,
        }
    }
}
