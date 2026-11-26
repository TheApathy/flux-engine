// SPDX-License-Identifier: AGPL-3.0-only

//! BF16-router arm of the Qwen4 MoE-only prefill admission
//! (`qwen4_prefill_admission.rs`), split out to keep that file small.

use anyhow::{Result, ensure};

use super::MoeLayer;

impl MoeLayer {
    /// `ATLAS_QWEN4_ROUTER_BF16=1`: the router is the stored BF16 tensor
    /// through `dense_gemm`, with no NVFP4 or FP8 router copy.
    pub(super) fn validate_bf16_router_prefill(
        &self,
        gate_fp8: Option<u64>,
        shared_v7: bool,
    ) -> Result<()> {
        ensure!(
            super::qwen4_router_bf16::bf16_router_bundle_valid(
                super::qwen4_router_bf16::selected()?,
                gate_fp8,
                self.weights.gate.weight.0,
                self.dense_gemm.0,
                shared_v7,
                [
                    self.shared_gate_fp8.map(|p| p.0),
                    self.shared_up_fp8.map(|p| p.0),
                    self.shared_down_fp8.map(|p| p.0),
                ],
                self.fp8_gemm_k.0,
            ),
            "Qwen4 MoE-only prefill requires the complete BF16-router control bundle"
        );
        Ok(())
    }
}
