// SPDX-License-Identifier: AGPL-3.0-only

//! HDIM>256 contiguous prefill attention (Gemma-4 full-attention layers).
//!
//! Two kernels serve this head width: the scalar `inferspark_prefill_512`
//! reference (BR=16, no tensor cores) and, when `ATLAS_GEMMA_PREFILL_512_TC=1`,
//! the tensor-core `inferspark_prefill_512tc` (the common BR=32 kernel at
//! HDIM=512 / BC=16). The kernel name and the grid's BR are decided together
//! here; a grid built for the other kernel's BR does not fail, it computes the
//! wrong q-tiles.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use super::{WidePrefillAttnRoute, mark_wide_prefill_attn_tc_engaged, wide_prefill_attn_tc_route};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Whether this layer's contiguous prefill takes the HDIM>256 kernels.
    pub(super) fn has_wide_prefill_attention(&self, hd: u32) -> bool {
        hd > 256 && self.prefill_attn_512_k.0 != 0
    }

    /// Run the HDIM>256 contiguous prefill. Returns the profile label of the
    /// kernel that actually ran.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_wide(
        &self,
        ctx: &ForwardContext,
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        out: DevicePtr,
        n: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        inv_sqrt_d: f32,
        sliding_window: u32,
        stream: u64,
    ) -> Result<&'static str> {
        let route = wide_prefill_attn_tc_route(
            self.prefill_attn_512_tc_requested,
            hd,
            self.prefill_attn_512_tc_k.0 != 0,
        );
        match route {
            WidePrefillAttnRoute::Complete => {
                ops::prefill_attention_512_tc(
                    ctx.gpu,
                    self.prefill_attn_512_tc_k,
                    q,
                    k,
                    v,
                    out,
                    n,
                    1,
                    nq,
                    nkv,
                    hd,
                    inv_sqrt_d,
                    true,
                    sliding_window,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("prefill_512tc failed: n={n} nq={nq} nkv={nkv} hd={hd}: {e}")
                })?;
                mark_wide_prefill_attn_tc_engaged();
                Ok("flash_attn_512_tc")
            }
            WidePrefillAttnRoute::Missing => anyhow::bail!(
                "ATLAS_GEMMA_PREFILL_512_TC=1 selected HDIM={hd} prefill, but \
                 inferspark_prefill_512tc is missing from this kernel target; rebuild \
                 the gemma-4-31b kernel bundle before measuring"
            ),
            WidePrefillAttnRoute::Disabled | WidePrefillAttnRoute::Ineligible => {
                ops::prefill_attention(
                    ctx.gpu,
                    self.prefill_attn_512_k,
                    q,
                    k,
                    v,
                    out,
                    n,
                    1,
                    nq,
                    nkv,
                    hd,
                    inv_sqrt_d,
                    true,
                    sliding_window,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("prefill_512 failed: n={n} nq={nq} nkv={nkv} hd={hd}: {e}")
                })?;
                Ok("flash_attn_512_scalar")
            }
        }
    }
}
