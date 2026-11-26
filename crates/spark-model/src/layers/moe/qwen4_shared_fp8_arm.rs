// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_DENSE_FP8=1` decode arm of the shared expert. Inside the
//! decode scope `dispatch_unified_t_decode` passes null shared weights to
//! the fused routed kernels (which then write zeros to the shared slots)
//! and this arm writes `shared_out = down(silu(gate(x)) * up(x))` from the
//! native MXFP8 checkpoint tensors. Outside the scope nothing changes.

use anyhow::{Result, ensure};

use super::MoeLayer;
use crate::layers::qwen4_dense_fp8::proj::SharedFp8;

impl MoeLayer {
    pub(crate) fn set_qwen4_shared_fp8(&mut self, fp8: SharedFp8) {
        self.qwen4_shared_fp8 = Some(fp8);
    }

    /// The installed arm when this call must replace the fused shared expert.
    pub(super) fn qwen4_shared_fp8_active(&self) -> Option<&SharedFp8> {
        self.qwen4_shared_fp8.as_ref().filter(|fp8| fp8.active())
    }

    /// Only the unified-T decode route consumes the FP8 shared expert; any
    /// other M=1 route would silently keep NVFP4, so it is refused.
    pub(super) fn qwen4_shared_fp8_route_check(&self) -> Result<()> {
        if self.qwen4_shared_fp8_active().is_none() {
            return Ok(());
        }
        let native_fp8_route = self.fp8_gate_weight_ptrs.is_some()
            && self.fp8_up_weight_ptrs.is_some()
            && self.fp8_down_weight_ptrs.is_some()
            && self.fp8_shared_expert.is_some();
        ensure!(
            !native_fp8_route && self.use_t_layout_for_decode(),
            "ATLAS_QWEN4_DENSE_FP8=1 refused: this MoE decode route is not unified-T, \
             so the FP8 shared expert would not run"
        );
        Ok(())
    }
}
