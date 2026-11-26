// SPDX-License-Identifier: AGPL-3.0-only

//! Phase E: load output projection weight (BF16 + NVFP4 quantized).

use anyhow::Result;

use super::ctx::MistralLayerCtx;
use crate::tp_shard::{TpShardKind, shard_dense_bf16};
use crate::weight_map::{dense, quantize_to_nvfp4};

pub(super) fn load_o_proj(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    let ap = ctx.ap();
    let h = ctx.h;
    let n_heads = ctx.n_heads;
    let v_dim = ctx.v_dim;
    let gpu = ctx.gpu;

    // O projection — RowParallel on the input `n_heads * v_dim` axis under TP,
    // then optionally NVFP4-quantized for decode throughput unless
    // `ATLAS_NVFP4_MLA=0` disables MLA NVFP4.
    let mut o_dense_bf16 = dense(ctx.store, &format!("{ap}.wo.weight"))?;
    let tp_rank = ctx.config.tp_rank;
    let tp_size = ctx.config.tp_world_size.max(1);
    if tp_size > 1 {
        let full_in = n_heads * tp_size * v_dim;
        let (sharded, _, _) = shard_dense_bf16(
            o_dense_bf16.weight,
            h,
            full_in,
            TpShardKind::RowParallel,
            tp_rank,
            tp_size,
            gpu,
        )?;
        o_dense_bf16.weight = sharded;
    }
    let o_nvfp4 = if ctx.disable_nvfp4_mla {
        None
    } else {
        Some(quantize_to_nvfp4(
            &o_dense_bf16,
            h,
            n_heads * v_dim,
            gpu,
            ctx.absmax_k,
            ctx.quantize_k,
            ctx.stream,
        )?)
    };

    ctx.o_dense_bf16 = Some(o_dense_bf16);
    ctx.o_nvfp4 = o_nvfp4;
    Ok(())
}
