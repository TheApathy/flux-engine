// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-resident EP split-phase overlapped MoE forward.
//!
//! Replaces the legacy CPU D2H routing-table scaffold with:
//! 1. Zero-D2H, zero-host-sync device routing (`gate` → `topk`) on the replicated token.
//! 2. Local routed-expert execution (excluding shared expert) + weighted sum into `moe_output`.
//! 3. Split-phase overlapped all-reduce (`comm.all_reduce_begin`) concurrent with
//!    shared-expert execution on the compute stream, synchronized via
//!    `comm.all_reduce_end` before blending the shared expert once.

use spark_comm::CommBackend;
#[cfg(test)]
use spark_comm::RoutePacketLayout;
#[cfg(test)]
use spark_runtime::kernel_args::div_ceil;

use super::*;

/// Block width used when launching `ep_pack_selected_routes_kernel` and
/// `ep_unpack_selected_routes_kernel`.
#[cfg(test)]
pub const EP_ROUTE_PACK_BLOCK_DIM: u32 = 128;

/// Precomputed byte geometry for one GPU-resident EP selected-route exchange + all-reduce.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpRouteExchangePlan {
    /// Selected compact packet wire layout (`TopK6` = 48B, `TopK8` = 64B, `TopK10` = 80B).
    pub layout: RoutePacketLayout,
    /// Number of local tokens contributed by each rank.
    pub local_tokens: usize,
    /// Participating EP world size (`>= 1`).
    pub world_size: usize,
    /// Local rank (`0..world_size`).
    pub rank: usize,
    /// Bytes contributed by one rank (`local_tokens * layout.packet_bytes()`).
    pub rank_route_bytes: usize,
    /// Total gathered route buffer size across all ranks (`rank_route_bytes * world_size`).
    pub total_route_bytes: usize,
    /// Byte offset of this rank's in-place slot inside the gather buffer (`rank * rank_route_bytes`).
    pub self_slot_byte_offset: usize,
    /// BF16 output all-reduce byte count (`local_tokens * hidden_size * 2`).
    pub output_allreduce_bytes: usize,
}

/// Compute and validate the route-packet exchange and all-reduce byte geometry.
///
/// # Errors
/// Returns an error if `top_k` is outside `1..=10`, `world_size == 0`, `rank >= world_size`,
/// or any byte calculation overflows `usize`.
#[cfg(test)]
pub fn plan_ep_route_exchange(
    local_tokens: usize,
    top_k: usize,
    hidden_size: usize,
    rank: usize,
    world_size: usize,
) -> Result<EpRouteExchangePlan> {
    let layout = RoutePacketLayout::for_top_k(top_k).ok_or_else(|| {
        anyhow::anyhow!("EP selected-route layout unsupported for top_k={top_k} (expected 1..=10)")
    })?;
    anyhow::ensure!(world_size >= 1, "EP world_size must be >= 1");
    anyhow::ensure!(
        rank < world_size,
        "EP rank {rank} out of range for world_size {world_size}"
    );

    let rank_route_bytes = local_tokens
        .checked_mul(layout.packet_bytes())
        .ok_or_else(|| anyhow::anyhow!("EP route rank byte count overflow"))?;
    let total_route_bytes = rank_route_bytes
        .checked_mul(world_size)
        .ok_or_else(|| anyhow::anyhow!("EP route total gather byte count overflow"))?;
    let self_slot_byte_offset = rank
        .checked_mul(rank_route_bytes)
        .ok_or_else(|| anyhow::anyhow!("EP route self-slot byte offset overflow"))?;
    let output_allreduce_bytes = local_tokens
        .checked_mul(hidden_size)
        .and_then(|elems| elems.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("EP output all-reduce byte count overflow"))?;

    Ok(EpRouteExchangePlan {
        layout,
        local_tokens,
        world_size,
        rank,
        rank_route_bytes,
        total_route_bytes,
        self_slot_byte_offset,
        output_allreduce_bytes,
    })
}

/// Compute 1D CUDA `(grid, block)` dimensions for `ep_pack_selected_routes_kernel`.
#[cfg(test)]
#[must_use]
pub fn ep_route_pack_grid(num_tokens: usize, layout: RoutePacketLayout) -> ([u32; 3], [u32; 3]) {
    let total_items = num_tokens.saturating_mul(layout.slots());
    let total_u32 = u32::try_from(total_items).unwrap_or(u32::MAX);
    let blocks = div_ceil(total_u32, EP_ROUTE_PACK_BLOCK_DIM).max(1);
    ([blocks, 1, 1], [EP_ROUTE_PACK_BLOCK_DIM, 1, 1])
}

/// Compute 1D CUDA `(grid, block)` dimensions for `ep_unpack_selected_routes_kernel`.
#[cfg(test)]
#[must_use]
pub fn ep_route_unpack_grid(total_tokens: usize, top_k: usize) -> ([u32; 3], [u32; 3]) {
    let total_items = total_tokens.saturating_mul(top_k);
    let total_u32 = u32::try_from(total_items).unwrap_or(u32::MAX);
    let blocks = div_ceil(total_u32, EP_ROUTE_PACK_BLOCK_DIM).max(1);
    ([blocks, 1, 1], [EP_ROUTE_PACK_BLOCK_DIM, 1, 1])
}

/// Compute 16-byte-aligned scratch offset immediately past `[indices(u32) | weights(f32)]`
/// and validate that the full gathered route buffer fits within `scratch_capacity`.
#[cfg(test)]
pub(crate) fn route_gather_scratch_offset(
    top_k: usize,
    total_route_bytes: usize,
    scratch_capacity: usize,
) -> Result<usize> {
    let raw_end = top_k
        .checked_mul(8)
        .ok_or_else(|| anyhow::anyhow!("EP route scratch header size overflow"))?;
    let aligned_off = raw_end
        .checked_add(15)
        .map(|v| v & !15)
        .ok_or_else(|| anyhow::anyhow!("EP route scratch alignment overflow"))?;
    let required_end = aligned_off
        .checked_add(total_route_bytes)
        .ok_or_else(|| anyhow::anyhow!("EP route scratch end offset overflow"))?;
    anyhow::ensure!(
        required_end <= scratch_capacity,
        "EP route gather requires {required_end} bytes in scratch, but arena has {scratch_capacity}"
    );
    Ok(aligned_off)
}

/// Execute `shared_compute` on `stream` overlapped with an in-flight split-phase all-reduce
/// on `output_ptr` (`ar_bytes` of BF16 elements).
///
/// Guarantees `comm.all_reduce_end(token, stream)` is always invoked once `all_reduce_begin`
/// succeeds, even if `shared_compute` returns an error.
pub(crate) fn run_overlapped_all_reduce_with_shared<F>(
    comm: &dyn CommBackend,
    output_ptr: u64,
    ar_bytes: usize,
    stream: u64,
    shared_compute: F,
) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    let token = comm.all_reduce_begin(output_ptr, ar_bytes, stream)?;
    let shared_res = shared_compute();
    let end_res = comm.all_reduce_end(token, stream);
    shared_res?;
    end_res
}

impl MoeLayer {
    /// GPU-resident EP dispatch entry point (preserves existing public signature).
    ///
    /// Delegates to [`Self::forward_ep_selected_routes_overlap`], eliminating the
    /// legacy CPU D2H synchronization and host routing-table construction.
    pub fn forward_ep_dispatch(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.forward_ep_selected_routes_overlap(input, ctx, stream)
    }

    /// GPU-resident single-token EP overlapped shared-expert forward.
    ///
    /// Pipeline (zero host sync, zero D2H copy):
    /// 1. Gate GEMV + top-k routing on device (`indices_dev`, `weights_dev`) for the replicated token.
    /// 2. Run local routed experts (excluding shared expert) + weighted sum into `moe_output`.
    /// 3. Overlap `comm.all_reduce_begin` on `comm_stream` with shared-expert GEMVs on `stream`,
    ///    then wait via `comm.all_reduce_end` and blend the shared expert once.
    ///
    /// Falls back cleanly to [`Self::forward`] for single-GPU worlds, CUDA graph capture
    /// (where event-based cross-stream overlap is not capturable), FP8 expert layers,
    /// `RoutedScaleLayout::Ts4e128`, or Qwen4 (`ctx.config.is_qwen4_exp()`).
    pub fn forward_ep_selected_routes_overlap(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let Some(comm) = ctx.comm.filter(|c| c.world_size() > 1) else {
            return self.forward(input, ctx, stream);
        };
        if ctx.config.ep_world_size <= 1
            || ctx.graph_capture
            || self.fp8_gate_weight_ptrs.is_some()
            || self.routed_scale_layout == super::RoutedScaleLayout::Ts4e128
            || ctx.config.is_qwen4_exp()
        {
            return self.forward(input, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let output_allreduce_bytes = (h as usize)
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("EP output all-reduce byte count overflow"))?;

        // Step 1: Router projection + top-k selection (100% GPU-resident).
        let router_in = self.router_input(input, 1, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                router_in,
                nvfp4,
                gate_logits,
                num_experts,
                h,
                stream,
            )?;
        } else {
            anyhow::ensure!(
                !self.weights.gate.weight.is_null(),
                "MoE BF16 router was released (ATLAS_QWEN4_DENSE_DEDUP); this route needs the NVFP4 router"
            );
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv,
                router_in,
                &self.weights.gate,
                gate_logits,
                num_experts,
                h,
                stream,
            )?;
        }

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);

        if let Some(bias) = self.correction_bias_dev {
            ops::moe_topk_sigmoid(
                ctx.gpu,
                self.moe_topk_sigmoid_k,
                gate_logits,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                1.0,
                stream,
            )?;
        } else if !super::qwen4_topk_warp::try_launch(
            ctx.gpu,
            gate_logits,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            1,
            stream,
        )? {
            ops::moe_topk_softmax(
                ctx.gpu,
                self.moe_topk,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                stream,
            )?;
        }

        // Step 2: Optional pre-expert norm (Gemma-4) + local routed-only expert FFN.
        // Every EP rank holds the same replicated token and runs identical routing above,
        // so local routed experts consume `indices_dev` directly and Step 3 all-reduces
        // `moe_output` without an unused route-gather collective.
        let expert_input = if let Some(ref norm_w) = self.pre_expert_norm {
            let normed = ctx.buffers.ssm_deinterleaved();
            let eps = ctx.config.rms_norm_eps as f32;
            ops::rms_norm(
                ctx.gpu,
                self.pre_expert_norm_k,
                input,
                norm_w,
                normed,
                1,
                h,
                eps,
                stream,
            )?;
            normed
        } else {
            input
        };

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let shared_out = ctx.buffers.attn_output();
        let null_qw = QuantizedWeight::null();

        // Pass NULL shared weights so routed kernels skip shared compute before all_reduce_begin.
        if self.use_t_layout_for_decode()
            && let (Some(gate_t), Some(up_t), Some(down_t)) =
                (&self.gate_ptrs_t, &self.up_ptrs_t, &self.down_ptrs_t)
        {
            ops::moe_expert_gate_up_shared_t(
                ctx.gpu,
                self.moe_expert_gate_up_shared_t_k,
                expert_input,
                gate_t.packed_ptrs,
                gate_t.scale_ptrs,
                gate_t.scale2_vals,
                expert_gate_out,
                up_t.packed_ptrs,
                up_t.scale_ptrs,
                up_t.scale2_vals,
                expert_up_out,
                indices_dev,
                &null_qw,
                shared_gate_scratch,
                &null_qw,
                shared_up_scratch,
                inter,
                h,
                top_k,
                stream,
            )?;
            ops::moe_expert_silu_down_shared_t(
                ctx.gpu,
                self.moe_expert_silu_down_shared_t_k,
                expert_gate_out,
                expert_up_out,
                down_t.packed_ptrs,
                down_t.scale_ptrs,
                down_t.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                &null_qw,
                shared_out,
                h,
                inter,
                top_k,
                stream,
            )?;
        } else {
            ops::moe_expert_gate_up_shared(
                ctx.gpu,
                self.moe_expert_gate_up_shared,
                expert_input,
                self.gate_ptrs.packed_ptrs,
                self.gate_ptrs.scale_ptrs,
                self.gate_ptrs.scale2_vals,
                expert_gate_out,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                indices_dev,
                &null_qw,
                shared_gate_scratch,
                &null_qw,
                shared_up_scratch,
                inter,
                h,
                top_k,
                stream,
            )?;
            ops::moe_expert_silu_down_shared(
                ctx.gpu,
                self.moe_expert_silu_down_shared,
                expert_gate_out,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                &null_qw,
                shared_out,
                h,
                inter,
                top_k,
                stream,
            )?;
        }

        // Weighted sum of local routed experts only (zeroed shared term prevents EP doubling).
        let output = ctx.buffers.moe_output();
        ctx.gpu
            .memset_async(expert_gate_out, 0, output_allreduce_bytes, stream)?;
        ops::moe_weighted_sum_blend(
            ctx.gpu,
            self.moe_weighted_sum_blend,
            output,
            expert_down_out,
            weights_dev,
            expert_gate_out,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            stream,
        )?;

        // Step 3: Overlap split-phase all-reduce on comm_stream with shared expert on stream.
        let has_shared = shared_inter > 0
            && (!self.weights.shared_expert.gate_proj.is_null()
                || self.shared_gate_t.is_some()
                || self.shared_gate_fp8.is_some());

        run_overlapped_all_reduce_with_shared(
            comm,
            output.0,
            output_allreduce_bytes,
            stream,
            || {
                if has_shared {
                    self.run_shared_expert_decode_overlapped(
                        expert_input,
                        h,
                        shared_inter,
                        shared_gate_scratch,
                        shared_up_scratch,
                        shared_out,
                        ctx,
                        stream,
                    )?;
                }
                Ok(())
            },
        )?;

        // Step 5: Blend shared expert contribution ONCE after all-reduce completes.
        if has_shared {
            if self.weights.shared_expert_gate.weight.0 == 0 {
                ops::residual_add(ctx.gpu, self.residual_add, output, shared_out, h, stream)?;
            } else {
                ops::moe_batched_blend(
                    ctx.gpu,
                    self.moe_batched_blend,
                    output,
                    shared_out,
                    input,
                    self.weights.shared_expert_gate.weight,
                    h,
                    1,
                    stream,
                )?;
            }
        }

        Ok(output)
    }

    /// Execute the single-token shared expert into `(shared_gate_scratch, shared_up_scratch,
    /// shared_out)` while `moe_output` is simultaneously being all-reduced on `comm_stream`.
    fn run_shared_expert_decode_overlapped(
        &self,
        expert_input: DevicePtr,
        h: u32,
        shared_inter: u32,
        shared_gate_scratch: DevicePtr,
        shared_up_scratch: DevicePtr,
        shared_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !self.weights.shared_expert.gate_proj.is_null() {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                expert_input,
                &self.weights.shared_expert.gate_proj,
                shared_gate_scratch,
                shared_inter,
                h,
                stream,
            )?;
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                expert_input,
                &self.weights.shared_expert.up_proj,
                shared_up_scratch,
                shared_inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                shared_gate_scratch,
                shared_up_scratch,
                shared_gate_scratch,
                shared_inter,
                stream,
            )?;
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                shared_gate_scratch,
                &self.weights.shared_expert.down_proj,
                shared_out,
                h,
                shared_inter,
                stream,
            )
        } else if let (Some(sg), Some(su), Some(sd)) =
            (&self.shared_gate_t, &self.shared_up_t, &self.shared_down_t)
        {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t,
                expert_input,
                sg,
                shared_gate_scratch,
                1,
                shared_inter,
                h,
                stream,
            )?;
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t,
                expert_input,
                su,
                shared_up_scratch,
                1,
                shared_inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                shared_gate_scratch,
                shared_up_scratch,
                shared_gate_scratch,
                shared_inter,
                stream,
            )?;
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t,
                shared_gate_scratch,
                sd,
                shared_out,
                1,
                h,
                shared_inter,
                stream,
            )
        } else if let (Some(sg), Some(su), Some(sd)) = (
            self.shared_gate_fp8,
            self.shared_up_fp8,
            self.shared_down_fp8,
        ) {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                expert_input,
                sg,
                shared_gate_scratch,
                1,
                shared_inter,
                h,
                stream,
            )?;
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                expert_input,
                su,
                shared_up_scratch,
                1,
                shared_inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                shared_gate_scratch,
                shared_up_scratch,
                shared_gate_scratch,
                shared_inter,
                stream,
            )?;
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                shared_gate_scratch,
                sd,
                shared_out,
                1,
                h,
                shared_inter,
                stream,
            )
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use spark_comm::CollectiveToken;

    #[test]
    fn test_plan_ep_route_exchange_layouts_and_offsets() {
        for (top_k, expected_layout, expected_pkt_bytes) in [
            (1usize, RoutePacketLayout::TopK6, 48usize),
            (6, RoutePacketLayout::TopK6, 48),
            (7, RoutePacketLayout::TopK8, 64),
            (8, RoutePacketLayout::TopK8, 64),
            (9, RoutePacketLayout::TopK10, 80),
            (10, RoutePacketLayout::TopK10, 80),
        ] {
            for world_size in [2usize, 3, 4] {
                for rank in 0..world_size {
                    let plan = plan_ep_route_exchange(5, top_k, 4096, rank, world_size)
                        .expect("valid EP route plan must succeed");
                    assert_eq!(plan.layout, expected_layout);
                    assert_eq!(plan.rank_route_bytes, 5 * expected_pkt_bytes);
                    assert_eq!(plan.total_route_bytes, 5 * expected_pkt_bytes * world_size);
                    assert_eq!(plan.self_slot_byte_offset, rank * 5 * expected_pkt_bytes);
                    assert_eq!(plan.output_allreduce_bytes, 5 * 4096 * 2);
                }
            }
        }
    }

    #[test]
    fn test_plan_ep_route_exchange_rejects_invalid_inputs() {
        assert!(plan_ep_route_exchange(1, 0, 2560, 0, 2).is_err());
        assert!(plan_ep_route_exchange(1, 11, 2560, 0, 2).is_err());
        assert!(plan_ep_route_exchange(1, 8, 2560, 0, 0).is_err());
        assert!(plan_ep_route_exchange(1, 8, 2560, 2, 2).is_err());
        assert!(plan_ep_route_exchange(usize::MAX, 10, 2560, 0, 2).is_err());
    }

    #[test]
    fn test_ep_route_pack_and_unpack_grid_geometry() {
        assert_eq!(
            ep_route_pack_grid(1, RoutePacketLayout::TopK8),
            ([1, 1, 1], [128, 1, 1])
        );
        assert_eq!(
            ep_route_pack_grid(128, RoutePacketLayout::TopK10),
            ([10, 1, 1], [128, 1, 1])
        );
        assert_eq!(ep_route_unpack_grid(1, 8), ([1, 1, 1], [128, 1, 1]));
        assert_eq!(ep_route_unpack_grid(64, 10), ([5, 1, 1], [128, 1, 1]));
    }

    #[test]
    fn test_route_gather_scratch_offset_alignment_and_bounds() {
        let off = route_gather_scratch_offset(10, 320, 1024).expect("400B fits in 1024B");
        assert_eq!(off, 80); // 10 * 8 = 80 (16-byte aligned)
        let off_odd = route_gather_scratch_offset(7, 256, 1024).expect("320B fits in 1024B");
        assert_eq!(off_odd, 64); // 7 * 8 = 56 -> aligned up to 64
        assert!(route_gather_scratch_offset(10, 1024, 1024).is_err());
    }

    #[test]
    fn test_overlapped_all_reduce_with_shared_ordering_and_error_safety() {
        struct TraceComm {
            log: Mutex<Vec<&'static str>>,
        }
        impl CommBackend for TraceComm {
            fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn all_reduce_begin(&self, _: u64, _: usize, _: u64) -> Result<CollectiveToken> {
                self.log.lock().push("ar_begin");
                Ok(CollectiveToken::from_raw_event(0x42))
            }
            fn all_reduce_end(&self, token: CollectiveToken, _: u64) -> Result<()> {
                assert_eq!(token.raw_event(), 0x42);
                self.log.lock().push("ar_end");
                Ok(())
            }
            fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
                Ok(())
            }
            fn barrier(&self) -> Result<()> {
                Ok(())
            }
            fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
                Ok(())
            }
            fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
                Ok(())
            }
            fn rank(&self) -> usize {
                0
            }
            fn world_size(&self) -> usize {
                2
            }
        }

        let comm = TraceComm {
            log: Mutex::new(Vec::new()),
        };
        run_overlapped_all_reduce_with_shared(&comm, 0x1000, 5120, 1, || {
            comm.log.lock().push("shared_expert");
            Ok(())
        })
        .expect("overlapped execution must succeed");
        assert_eq!(
            *comm.log.lock(),
            vec!["ar_begin", "shared_expert", "ar_end"]
        );

        // Even if shared expert errors, all_reduce_end MUST still be invoked to drain token.
        comm.log.lock().clear();
        let err = run_overlapped_all_reduce_with_shared(&comm, 0x1000, 5120, 1, || {
            comm.log.lock().push("shared_expert_err");
            anyhow::bail!("simulated shared expert failure")
        });
        assert!(err.is_err());
        assert_eq!(
            *comm.log.lock(),
            vec!["ar_begin", "shared_expert_err", "ar_end"]
        );
    }
}
