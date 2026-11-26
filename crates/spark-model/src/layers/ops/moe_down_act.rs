// SPDX-License-Identifier: AGPL-3.0-only

//! Activation-once dedup MoE down projection (fn-mtp, `ATLAS_QWEN4_MOE_DOWN_ACT=1`).
//!
//! `moe_dedup_act_rows` evaluates SiLU(gate)*up once per routed slot and
//! shared row into an FP32 buffer; `moe_expert_silu_down_shared_dedup_act_*`
//! is the dedup down GEMV reading that buffer instead of re-staging it in
//! every block's shared memory. Same expression, lane partition and warp tree
//! as `moe_expert_silu_down_shared_dedup`, so outputs are byte-identical.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// Outputs per block of an `_act` down kernel with `outp` output pairs per
/// warp (4 warps): `_o1` 8, `_o2` 16, `_o4` 32.
pub const fn moe_down_act_outputs_per_block(outp: u32) -> u32 {
    4 * 2 * outp
}

/// Largest row cap of any `_act` down twin (the r16 tier); the activation
/// buffer is sized for it. A wider tier must raise this constant.
pub const MOE_DOWN_ACT_MAX_ROWS: usize = 16;

/// FP32 elements the activation buffer needs for `rows` verify rows.
pub const fn moe_down_act_elems(rows: usize, top_k: usize, inter: usize) -> usize {
    rows * (top_k + 1) * inter
}

/// `act[slot][k] = silu(g) * u` for every routed slot then every shared row.
#[allow(clippy::too_many_arguments)]
pub fn moe_dedup_act_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    act: DevicePtr,
    k: u32,
    top_k: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k, 128), num_tokens * (top_k + 1), 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(act)
        .arg_u32(k)
        .arg_u32(num_tokens * top_k)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// Dedup down GEMV over the precomputed activation rows. No dynamic shared
/// memory; `rows_max` is the kernel's row cap (8 or 16).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_dedup_act(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    outputs_per_block: u32,
    rows_max: u32,
    act: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        num_tokens <= rows_max,
        "dedup act MoE down: {num_tokens} rows exceed the kernel's {rows_max}-row cap"
    );
    anyhow::ensure!(
        k.is_multiple_of(16),
        "dedup act MoE down: K={k} must be a multiple of 16 (float4 activation reads)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, outputs_per_block), num_tokens * (top_k + 1), 1])
        .block([128, 1, 1])
        .arg_ptr(act)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.weight_scale)
        .arg_f32(sh_down.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(num_tokens)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::{moe_down_act_elems, moe_down_act_outputs_per_block};

    #[test]
    fn activation_buffer_covers_routed_and_shared_rows() {
        // Flash-Next: top_k 10, inter 640, K9 verify -> 9 * 11 * 640 floats.
        assert_eq!(moe_down_act_elems(9, 10, 640), 63_360);
        assert_eq!(moe_down_act_elems(16, 10, 640), 112_640);
    }

    #[test]
    fn outputs_per_block_follow_the_kernel_variant() {
        assert_eq!(moe_down_act_outputs_per_block(1), 8);
        assert_eq!(moe_down_act_outputs_per_block(4), 32);
    }
}
