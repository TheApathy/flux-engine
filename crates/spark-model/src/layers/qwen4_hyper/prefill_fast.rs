// SPDX-License-Identifier: AGPL-3.0-only

//! Full-row numerical hyperconnection preparation (default-off, see
//! `qwen4_fast_proj`). Same kernel order as the exact tile loop but each of
//! the three projections runs once over every row through dequant+cuBLASLt,
//! and the mixed rows are additionally staged into the residual row fronts so
//! every consumer of the exact layout (attention tile packers, SSM/MoE
//! contiguous input) keeps working unchanged.

use anyhow::Result;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Qwen4HyperConnection;
use super::pending::PendingInject;
use crate::layers::qwen4_fast_proj;

impl Qwen4HyperConnection {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_prefill_fast(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        pending: Option<&PendingInject>,
        stage: bool,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        anyhow::ensure!(num_tokens > 0, "empty Qwen4 hyperconnection prefill");
        let r = self.residual_width();
        let down_bytes = num_tokens * self.rank * 2;
        let inject_bytes = num_tokens * self.hc_count * 2;
        anyhow::ensure!(
            down_bytes + inject_bytes <= buffers.sizes().gate_logits,
            "Qwen4 fast hyper scratch exceeds gate-logit arena"
        );
        anyhow::ensure!(
            num_tokens * r * 2 <= buffers.sizes().qkv_output,
            "Qwen4 fast hyper logits exceed QKV arena"
        );
        anyhow::ensure!(
            num_tokens * self.hidden_size * 2 <= buffers.sizes().norm_output,
            "Qwen4 fast mixed rows exceed norm arena"
        );
        let inject = self
            .inject
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("fast HC requires an injecting hyperconnection"))?;

        // A pending injection (ATLAS_QWEN4_PREFILL_MHC_FUSED) is applied by
        // the same pass that normalizes; take_fusable_pending admitted it.
        if let Some(p) = pending {
            self.inject_norm_prefill(p, gpu, eps, stream)?;
        } else {
            KernelLaunch::new(gpu, self.group_norm_k)
                .grid([num_tokens as u32, self.hc_count as u32, 1])
                .block([self.hidden_size.min(1024) as u32, 1, 1])
                .arg_ptr(hyper)
                .arg_ptr(self.norm.weight)
                .arg_ptr(residual)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .arg_f32(eps)
                .launch(stream)?;
        }

        let down_out = buffers.gate_logits();
        qwen4_fast_proj::dequant_gemm(
            gpu, residual, &self.down, down_out, num_tokens, self.rank, r, stream,
        )?;
        KernelLaunch::new(gpu, self.silu_div_k)
            .grid([div_ceil((num_tokens * self.rank) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(down_out)
            .arg_u32((num_tokens * self.rank) as u32)
            .arg_f32(self.hc_count as f32)
            .launch(stream)?;

        let mix_logits = buffers.qkv_output();
        let inject_logits = down_out.offset(down_bytes);
        let mixed = buffers.norm_output();
        if self.upmix_admitted(down_out, residual, mixed)? {
            // ATLAS_QWEN4_PREFILL_MHC_FUSED=upmix: the mix runs in the up
            // projection's epilogue; the 10240-wide logits never reach DRAM.
            qwen4_fast_proj::dequant_gemm(
                gpu,
                residual,
                inject,
                inject_logits,
                num_tokens,
                self.hc_count,
                r,
                stream,
            )?;
            self.upmix_prefill(
                down_out,
                residual,
                inject_logits,
                mixed,
                num_tokens,
                gpu,
                stream,
            )?;
        } else {
            qwen4_fast_proj::dequant_gemm(
                gpu, down_out, &self.up, mix_logits, num_tokens, r, self.rank, stream,
            )?;
            qwen4_fast_proj::dequant_gemm(
                gpu,
                residual,
                inject,
                inject_logits,
                num_tokens,
                self.hc_count,
                r,
                stream,
            )?;
            KernelLaunch::new(gpu, self.mix_k)
                .grid([num_tokens as u32, div_ceil(self.hidden_size as u32, 256), 1])
                .block([256, 1, 1])
                .arg_ptr(residual)
                .arg_ptr(mix_logits)
                .arg_ptr(inject_logits)
                .arg_ptr(mixed)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, self.save_inject_k)
            .grid([div_ceil((num_tokens * self.hc_count) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(inject_logits)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .arg_u32(num_tokens as u32)
            .launch(stream)?;
        // Exact-layout compatibility: mixed rows also live in each residual
        // row's first H elements (the attention tile packer reads them there).
        // Callers whose consumer reads only `mixed` may skip it (`stage`).
        if stage {
            KernelLaunch::new(gpu, self.stage_mixed_k)
                .grid([div_ceil((num_tokens * self.hidden_size) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(mixed)
                .arg_ptr(residual)
                .arg_u32(0)
                .arg_u32(num_tokens as u32)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .launch(stream)?;
        }
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "HC_PREFILL_FAST_ENGAGED selector={} rows={num_tokens} projection=dequant_cublaslt_bf16_non_bit_exact",
                qwen4_fast_proj::SELECTOR
            );
        });
        Ok(mixed)
    }
}
