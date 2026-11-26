// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 5-7 of `Qwen3SsmLayer::decode_batched_inner`: Conv1d + L2 norm +
//! GDN per-token (with intermediate checkpoints). Extracted from
//! `trait_decode_batched.rs` to keep the parent file under 500 LoC.
//! Dispatches one of the fused K=2/3/4/17 paths or the sequential
//! per-token fallback. All buffers + state are owned by the caller; this
//! function only mutates `ssm_state.h_state`, `ssm_state.conv_state`,
//! their intermediates, `conv_out_buf`, and `gdn_out_buf`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::ForwardContext;
use crate::layers::ops;

#[allow(clippy::too_many_arguments)]
pub(super) struct ConvGdnArgs {
    pub num_tokens: usize,
    pub deinterleaved: DevicePtr,
    pub gates_buf: DevicePtr,
    pub conv_out_buf: DevicePtr,
    pub gdn_out_buf: DevicePtr,
    pub h_bytes: usize,
    pub conv_bytes: usize,
    pub qkvz_size: usize,
    pub conv_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub d_conv: usize,
    pub qk_ch: u32,
    pub nk: usize,
    pub nv: usize,
    pub kd: usize,
    pub vd: usize,
    pub bf16: usize,
    pub fp32: usize,
    pub stream: u64,
}

impl Qwen3SsmLayer {
    /// Run conv1d_update_l2norm + GDN over `num_tokens` (multi-token decode
    /// / MTP verify). Picks the K=2/3/4/17 fused WY path if available,
    /// otherwise falls back to the sequential per-token gdn_decode loop.
    pub(super) fn decode_batched_conv_gdn(
        &self,
        ssm_state: &mut SsmLayerState,
        ctx: &ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<()> {
        let ConvGdnArgs {
            num_tokens,
            deinterleaved,
            gates_buf,
            conv_out_buf,
            gdn_out_buf,
            h_bytes,
            conv_bytes,
            qkvz_size,
            conv_dim,
            key_dim,
            value_dim: _,
            d_conv,
            qk_ch,
            nk,
            nv,
            kd,
            vd,
            bf16,
            fp32,
            stream,
        } = *args;

        if num_tokens == 4 {
            // ── K=4 fused path: conv1d+L2norm sequential, GDN WY4 ──
            for t in 0..4u32 {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_out_t,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[t as usize],
                    conv_bytes,
                    stream,
                )?;
            }

            // WY-chunkwise GDN: 2-pass algorithm for 4-token verification.
            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            let wy_chunk_k = self.wy_chunk_kernel(4);
            self.require_wy_f16(4, wy_chunk_k)?;
            if super::ssm_h_fp16_enabled() {
                super::ssm_h_fp16::require_h_f16(ssm_state)?;
                ops::gdn_decode_wy4_f16(
                    ctx.gpu,
                    wy_chunk_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    ssm_state.h_state_intermediates[1],
                    ssm_state.h_state_intermediates[2],
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    0u32,            // state_is_table (contiguous h_state pool)
                    stream,
                )?;
            } else {
                anyhow::ensure!(
                    !ssm_state.h_is_f16,
                    "FP16 h_state encountered in FP32 gdn_decode_wy4 branch"
                );
                ops::gdn_decode_wy4(
                    ctx.gpu,
                    wy_chunk_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    ssm_state.h_state_intermediates[1],
                    ssm_state.h_state_intermediates[2],
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    stream,
                )?;
            }
        } else if num_tokens == 3 {
            // ── K=3 fused path: conv1d+L2norm + intermediates, GDN WY3 ──
            // Single chunk3 kernel covers 3 sequential conv1d updates +
            // L2 norm on Q/K + intermediate-state save (replaces 3 per-
            // token conv1d_l2norm launches + 3 d2d copies = 6 launches
            // → 1). Falls back to the per-token loop if the kernel isn't
            // loaded for the active target.
            if self.conv1d_l2norm_chunk3_k.0 != 0 {
                ops::conv1d_update_l2norm_chunk3(
                    ctx.gpu,
                    self.conv1d_l2norm_chunk3_k,
                    ssm_state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    conv_out_buf,
                    ssm_state.conv_state_intermediates[0],
                    ssm_state.conv_state_intermediates[1],
                    1,
                    conv_dim as u32,
                    d_conv as u32,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    qkvz_size as u32, // input stride (BF16): qkvz_size between tokens
                    conv_dim as u32,  // output stride (BF16): conv_dim between tokens
                    stream,
                )?;
            } else {
                for t in 0..3u32 {
                    let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                    let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                    ops::conv1d_update_l2norm(
                        ctx.gpu,
                        self.conv1d_l2norm_k,
                        ssm_state.conv_state,
                        qkv_t,
                        &self.ssm.conv1d,
                        conv_out_t,
                        conv_dim as u32,
                        d_conv as u32,
                        1,
                        qk_ch,
                        kd as u32,
                        1e-6,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t as usize],
                        conv_bytes,
                        stream,
                    )?;
                }
            }

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            let wy_chunk_k = self.wy_chunk_kernel(3);
            self.require_wy_f16(3, wy_chunk_k)?;
            if super::ssm_h_fp16_enabled() {
                super::ssm_h_fp16::require_h_f16(ssm_state)?;
                ops::gdn_decode_wy3_f16(
                    ctx.gpu,
                    wy_chunk_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    ssm_state.h_state_intermediates[1],
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    0u32,            // state_is_table (contiguous h_state pool)
                    stream,
                )?;
            } else {
                anyhow::ensure!(
                    !ssm_state.h_is_f16,
                    "FP16 h_state encountered in FP32 gdn_decode_wy3 branch"
                );
                ops::gdn_decode_wy3(
                    ctx.gpu,
                    wy_chunk_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    ssm_state.h_state_intermediates[1],
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    stream,
                )?;
            }
        } else if num_tokens == 2 {
            // ── K=2 fused path: conv1d sequential, L2 norm sequential, GDN chunk2 ──
            let qkv_0 = deinterleaved;
            let conv_out_0 = conv_out_buf;
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_k,
                ssm_state.conv_state,
                qkv_0,
                &self.ssm.conv1d,
                conv_out_0,
                conv_dim as u32,
                d_conv as u32,
                1,
                qk_ch,
                kd as u32,
                1e-6,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                ssm_state.conv_state,
                ssm_state.conv_state_intermediates[0],
                conv_bytes,
                stream,
            )?;

            let qkv_1 = deinterleaved.offset(qkvz_size * bf16);
            let conv_out_1 = conv_out_buf.offset(conv_dim * bf16);
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_k,
                ssm_state.conv_state,
                qkv_1,
                &self.ssm.conv1d,
                conv_out_1,
                conv_dim as u32,
                d_conv as u32,
                1,
                qk_ch,
                kd as u32,
                1e-6,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                ssm_state.conv_state,
                ssm_state.conv_state_intermediates[1],
                conv_bytes,
                stream,
            )?;

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            if super::ssm_h_fp16_enabled() {
                let wy_chunk_k = self.wy_chunk_kernel(2);
                self.require_wy_f16(2, wy_chunk_k)?;
                super::ssm_h_fp16::require_h_f16(ssm_state)?;
                ops::gdn_decode_wy2_f16(
                    ctx.gpu,
                    wy_chunk_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    0u32,            // state_is_table (contiguous h_state pool)
                    stream,
                )?;
            } else {
                anyhow::ensure!(
                    !ssm_state.h_is_f16,
                    "FP16 h_state encountered in FP32 gdn_decode_wy2/chunk2 branch"
                );
                // ATLAS_GDN_CHUNK2=1: use the registered-but-unused
                // gated_delta_rule_chunk2 kernel (3-pass H_state vs wy2's 4-pass).
                // Atlas docs: "Reads H_0 once, computes both outputs and H_2 in
                // 3 passes." Identical signature to wy2 — swap is just kernel handle.
                let use_chunk2 = std::env::var("ATLAS_GDN_CHUNK2")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let (k2_kernel, k2_op): (
                    _,
                    fn(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) -> _,
                ) = if use_chunk2 && self.gdn_chunk2_k.0 != 0 {
                    (self.gdn_chunk2_k, ops::gdn_decode_chunk2)
                } else {
                    (self.gdn_wy2_k, ops::gdn_decode_wy2)
                };
                k2_op(
                    ctx.gpu,
                    k2_kernel,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    stream,
                )?;
            }
        } else if let (Some(parent_ids_dev), true) = (
            ctx.ddtree_parent_ids_dev,
            self.gdn_tree_k.0 != 0
                && std::env::var("ATLAS_FORCE_WY17").ok().as_deref() != Some("1"),
        ) {
            anyhow::ensure!(
                !ssm_state.h_is_f16 && !super::ssm_h_fp16_enabled(),
                "ATLAS_SSM_H_FP16 is not supported by the FP32 DDTree GDN verify kernels; unset ATLAS_SSM_H_FP16"
            );
            static DISPATCH_DBG: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let n = DISPATCH_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 3 {
                tracing::info!(
                    "M8A dispatch FIRES #{n}: num_tokens={} parent_ids_dev={:?} gdn_tree_k=non-null",
                    num_tokens,
                    parent_ids_dev
                );
            }
            // ── M8A: DDTree tree-aware GDN verify ──
            // parent_ids_dev is a [num_tokens × i32] device tensor uploaded by
            // verify_d.rs from a.pending_tree_payload before the layer loop.
            // Each token's state load follows parent_ids[i] instead of i-1,
            // letting the verifier walk non-flat tree branches.
            //
            // ATLAS_DDTREE_TREE_CONV_EXACT=1: before each token, re-root
            // conv_state from conv_state_intermediates[parent[t]] when the
            // token is a branch (parent != t-1). This makes the conv1d shift
            // register ancestor-exact so FREE_SLOTS branch commits produce
            // oracle-correct logits. Requires causal_conv1d_tree_reroot compiled.
            let tree_conv_exact = self.conv1d_tree_reroot_k.0 != 0
                && std::env::var("ATLAS_DDTREE_TREE_CONV_EXACT")
                    .ok()
                    .as_deref()
                    == Some("1");
            // Base of this slot's conv intermediates (contiguous: base + t*conv_bytes).
            let conv_inter_base = ssm_state.conv_state_intermediates[0];
            let conv_floats = (conv_bytes / 4) as u32;
            if self.conv1d_l2norm_bf16_tree_sequence_k.0 != 0 {
                let parent_ids_arg = if tree_conv_exact {
                    parent_ids_dev
                } else {
                    spark_runtime::gpu::DevicePtr::NULL
                };
                ops::conv1d_update_l2norm_bf16_tree_sequence(
                    ctx.gpu,
                    self.conv1d_l2norm_bf16_tree_sequence_k,
                    ssm_state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    parent_ids_arg,
                    conv_out_buf,
                    conv_inter_base,
                    num_tokens as u32,
                    conv_dim as u32,
                    d_conv as u32,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    qkvz_size as u32,
                    conv_dim as u32,
                    conv_floats,
                    stream,
                )?;
            } else {
                for t in 0..(num_tokens as u32) {
                    let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                    let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                    if tree_conv_exact {
                        ops::conv1d_tree_reroot(
                            ctx.gpu,
                            self.conv1d_tree_reroot_k,
                            ssm_state.conv_state,
                            conv_inter_base,
                            parent_ids_dev,
                            t,
                            conv_floats,
                            stream,
                        )?;
                    }
                    ops::conv1d_update_l2norm(
                        ctx.gpu,
                        self.conv1d_l2norm_k,
                        ssm_state.conv_state,
                        qkv_t,
                        &self.ssm.conv1d,
                        conv_out_t,
                        conv_dim as u32,
                        d_conv as u32,
                        1,
                        qk_ch,
                        kd as u32,
                        1e-6,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t as usize],
                        conv_bytes,
                        stream,
                    )?;
                }
            }
            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            // Use the FIRST per-token intermediate slot as the base for the
            // tree kernel's contiguous `h_state_inter[T, ...]` output layout.
            // Atlas allocates intermediates[0..num_tokens] contiguously per
            // (b, vh) — the kernel writes inter[t] = base + t * hv directly.
            // wy17 uses inter_stride_floats = h_bytes/4 = nv*kd*vd. Match it
            // so post-verify commit reads from the right per-token slot.
            let inter_stride_floats = (h_bytes / 4) as u32;

            // Prefer M8A v2 (tree-aware WY-fused) when available — bit-
            // equivalent to wy17 on flat-chain payloads. Falls back to the
            // sequential per-token kernel (M8A v1) when WY kernel not loaded.
            if self.gdn_tree_wy_k.0 != 0 {
                ops::gdn_decode_tree_wy(
                    ctx.gpu,
                    self.gdn_tree_wy_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    parent_ids_dev,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    inter_stride_floats,
                    num_tokens as u32,
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    stream,
                )?;
            } else {
                ops::gdn_decode_tree(
                    ctx.gpu,
                    self.gdn_tree_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    parent_ids_dev,
                    gdn_out_buf,
                    ssm_state.h_state_intermediates[0],
                    inter_stride_floats,
                    num_tokens as u32,
                    1, // batch_size
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    stream,
                )?;
            }
        } else if num_tokens == 17
            && self.gdn_wy17_k.0 != 0
            && std::env::var("ATLAS_DISABLE_WY17").ok().as_deref() != Some("1")
        {
            anyhow::ensure!(
                !ssm_state.h_is_f16 && !super::ssm_h_fp16_enabled(),
                "ATLAS_SSM_H_FP16 is not supported by the FP32 WY17 GDN verify kernel; unset ATLAS_SSM_H_FP16 or set ATLAS_DISABLE_WY17=1"
            );
            // ── K=17 (DFlash γ+1): fused WY-Chunkwise path ──
            if self.conv1d_l2norm_bf16_sequence_k.0 != 0 {
                let conv_floats = (conv_bytes / 4) as u32;
                ops::conv1d_update_l2norm_bf16_sequence(
                    ctx.gpu,
                    self.conv1d_l2norm_bf16_sequence_k,
                    ssm_state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    conv_out_buf,
                    ssm_state.conv_state_intermediates[0],
                    num_tokens as u32,
                    conv_dim as u32,
                    d_conv as u32,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    qkvz_size as u32,
                    conv_dim as u32,
                    conv_floats,
                    stream,
                )?;
            } else {
                for t in 0..(num_tokens as u32) {
                    let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                    let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                    ops::conv1d_update_l2norm(
                        ctx.gpu,
                        self.conv1d_l2norm_k,
                        ssm_state.conv_state,
                        qkv_t,
                        &self.ssm.conv1d,
                        conv_out_t,
                        conv_dim as u32,
                        d_conv as u32,
                        1,
                        qk_ch,
                        kd as u32,
                        1e-6,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t as usize],
                        conv_bytes,
                        stream,
                    )?;
                }
            }

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            let inter_stride_floats = (h_bytes / 4) as u32;

            // ── Phase 2 profiling: wall-clock per wy17 call ──
            // Gated by ATLAS_SSM_KERNEL_PROFILE=1. Accumulates total ns spent in
            // wy17 across every SSM layer + every verify step. Sync-before to
            // start clock with empty stream; sync-after to capture GPU finish.
            // Adds ~2 host syncs per layer when active. Disabled by default.
            static SSM_PROFILE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);
            let profile = {
                let v = SSM_PROFILE.load(std::sync::atomic::Ordering::Relaxed);
                if v >= 0 {
                    v == 1
                } else {
                    let enabled =
                        std::env::var("ATLAS_SSM_KERNEL_PROFILE").ok().as_deref() == Some("1");
                    SSM_PROFILE.store(
                        if enabled { 1 } else { 0 },
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    enabled
                }
            };
            // ── Target 2: WY17 occupancy split (ATLAS_WY17_SPLIT=n) ──
            // Route to the v-dim-split kernel when the gate is on and the
            // vsplit PTX is loaded; otherwise the baseline single-CTA/head
            // wy17. Output is bit-identical either way. Encapsulated in a
            // closure so both the profiled and unprofiled branches share it.
            let wy17_split = crate::layers::wy17_split();
            let use_vsplit = wy17_split > 0 && self.gdn_wy17_vsplit_k.0 != 0;
            // Lazy H_i intermediate writes: route to `gdn_wy17_lazy` when the
            // checkpoint stride is > 1 and both the lazy kernel and its replay
            // companion are loaded. The commit path replays skipped intermediate
            // slots on non-checkpoint partial accepts.
            let wy17_lazy_j = crate::layers::wy17_lazy();
            // Shared with the commit path via `TransformerLayer::wy17_lazy_engaged`
            // (implemented in `qwen3_ssm/mod.rs`) so dispatch and commit always
            // agree on whether the lazy kernel produced this verify's intermediates
            // (including vsplit compatibility). Do not re-guard here or dispatch
            // and commit could drift.
            let use_lazy = crate::layer::TransformerLayer::wy17_lazy_engaged(self, num_tokens);
            let eff_lazy_j = if use_lazy { wy17_lazy_j } else { 1 };
            let run_wy17 = |gpu: &dyn spark_runtime::gpu::GpuBackend| -> Result<()> {
                if use_lazy && use_vsplit && self.gdn_wy17_lazy_vsplit_k.0 != 0 {
                    return ops::gdn_decode_wy17_lazy_vsplit(
                        gpu,
                        self.gdn_wy17_lazy_vsplit_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gate_ptr,
                        beta_ptr,
                        gdn_out_buf,
                        ssm_state.h_state_intermediates[0],
                        inter_stride_floats,
                        1, // batch_size
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32, // qk_stride
                        conv_dim as u32, // v_stride
                        (nv * 2) as u32, // gb_stride
                        wy17_split,
                        eff_lazy_j,
                        stream,
                    );
                }
                if use_lazy {
                    return ops::gdn_decode_wy17_lazy(
                        gpu,
                        self.gdn_wy17_lazy_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gate_ptr,
                        beta_ptr,
                        gdn_out_buf,
                        ssm_state.h_state_intermediates[0],
                        inter_stride_floats,
                        1, // batch_size
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32, // qk_stride
                        conv_dim as u32, // v_stride
                        (nv * 2) as u32, // gb_stride
                        eff_lazy_j,
                        stream,
                    );
                }
                if use_vsplit {
                    ops::gdn_decode_wy17_vsplit(
                        gpu,
                        self.gdn_wy17_vsplit_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gate_ptr,
                        beta_ptr,
                        gdn_out_buf,
                        ssm_state.h_state_intermediates[0],
                        inter_stride_floats,
                        1, // batch_size
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32, // qk_stride
                        conv_dim as u32, // v_stride
                        (nv * 2) as u32, // gb_stride
                        wy17_split,
                        stream,
                    )
                } else {
                    ops::gdn_decode_wy17(
                        gpu,
                        self.gdn_wy17_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gate_ptr,
                        beta_ptr,
                        gdn_out_buf,
                        ssm_state.h_state_intermediates[0],
                        inter_stride_floats,
                        1, // batch_size
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32, // qk_stride
                        conv_dim as u32, // v_stride
                        (nv * 2) as u32, // gb_stride
                        stream,
                    )
                }
            };
            if profile {
                ctx.gpu.synchronize(stream)?;
                let t0 = std::time::Instant::now();
                run_wy17(ctx.gpu)?;
                ctx.gpu.synchronize(stream)?;
                let ns = t0.elapsed().as_nanos() as u64;
                crate::layers::qwen3_ssm::ssm_profile_record(ns);
            } else {
                run_wy17(ctx.gpu)?;
            }

            // ── WY17 LAZY-commit retention ──
            // When lazy is active the kernel persisted only checkpoint
            // intermediate slots. A later partial accept that lands on a
            // SKIPPED slot is reconstructed by `gdn_wy17_replay` in the commit
            // path, which needs the SAME per-token k/v/gate/beta inputs this
            // verify used. Those live in the shared forward-context scratch
            // (`conv_out_buf` / `gates_buf`), which the NEXT SSM layer
            // overwrites — so snapshot them into this layer's fixed retention
            // buffers now. Only fires when both the gate is on AND the
            // per-layer retention pool was allocated (buffers non-None). This
            // copy runs on the SAME stream as the verify kernel (ordering
            // guaranteed) and is a plain D2D (CUDA-graph safe, no alloc).
            if let (true, Some(kv_ret), Some(gate_ret)) = (
                use_lazy,
                ssm_state.wy17_kv_retain,
                ssm_state.wy17_gate_retain,
            ) {
                let kv_bytes = num_tokens * conv_dim * bf16;
                let gate_bytes = num_tokens * (nv * 2) * fp32;
                ctx.gpu
                    .copy_d2d_async(conv_out_buf, kv_ret, kv_bytes, stream)?;
                ctx.gpu
                    .copy_d2d_async(gates_buf, gate_ret, gate_bytes, stream)?;
            }
        } else {
            // ── K>=5: chunked path using fused wy4/wy3/wy2 kernels ──
            // K=17 also lands here under the ATLAS_DISABLE_WY17 correctness
            // escape hatch, allowing its persisted intermediates to be
            // compared against the dedicated WY17 kernel.
            //
            // The legacy per-token sequential path (`ops::gdn_decode` in a
            // loop) has a confirmed numerical bug that produces NaN at the
            // last 3 positions for K∈{5..16} (observed empirically at K=16:
            // positions 13–15 = NaN; symptom is target output collapsing to
            // `correct_first_token + !!!!!`). The fused wy4/wy3/wy2 kernels
            // handle the WY-chunkwise update correctly, so we chunk K into
            // sizes of 4 (or smaller for the tail) and run a fused kernel
            // per chunk. h_state flows naturally between chunks because each
            // wy_k kernel reads + writes it in place.
            //
            // wy_k writes K-1 explicit intermediates (slot 0..K-2 of the
            // chunk) plus h_state = state-after-(K-1) implicit. For
            // non-final chunks we must save h_state to the corresponding
            // global intermediate slot so partial-accept rollback can find
            // every per-position state.
            // Chunk-size selector: avoid chunk=1 (chunk=1 falls through to
            // `ops::gdn_decode` which expects FP32 q/k/v but our conv_out_buf
            // is BF16 — reading 2-byte BF16 elements as 4-byte FP32 produces
            // garbage that corrupted output on K=5/9/13/17%4==1. The wy_k
            // fused kernels (wy2/wy3/wy4) all accept BF16 properly, so we
            // strictly use {2, 3, 4}-sized chunks. The split is chosen so
            // each chunk handles a contiguous run of tokens with one
            // wy_k launch.
            fn pick_chunk(remaining: usize) -> usize {
                match remaining {
                    0 => 0,
                    1 => unreachable!(
                        "remaining=1 only happens if prior chunk took too many; handled below"
                    ),
                    2 => 2,
                    3 => 3,
                    4 => 4,
                    5 => 3, // 3 + 2
                    _ => 4, // 4 + ... rest
                }
            }
            let all_inter_base = ssm_state.conv_state_intermediates[0].0;
            let all_inter_contiguous = self.conv1d_l2norm_bf16_sequence_k.0 != 0
                && (1..num_tokens).all(|t| {
                    ssm_state.conv_state_intermediates[t].0
                        == all_inter_base + (t * conv_bytes) as u64
                });
            if all_inter_contiguous {
                let conv_floats = (conv_bytes / 4) as u32;
                ops::conv1d_update_l2norm_bf16_sequence(
                    ctx.gpu,
                    self.conv1d_l2norm_bf16_sequence_k,
                    ssm_state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    conv_out_buf,
                    ssm_state.conv_state_intermediates[0],
                    num_tokens as u32,
                    conv_dim as u32,
                    d_conv as u32,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    qkvz_size as u32,
                    conv_dim as u32,
                    conv_floats,
                    stream,
                )?;
            }
            let mut t_done: usize = 0;
            while t_done < num_tokens {
                let remaining = num_tokens - t_done;
                let chunk = pick_chunk(remaining);

                // ── 1. conv1d_l2norm + save conv_state intermediate per token ──
                if !all_inter_contiguous {
                    for ct in 0..chunk {
                        let t_abs = t_done + ct;
                        let qkv_t = deinterleaved.offset(t_abs * qkvz_size * bf16);
                        let conv_out_t = conv_out_buf.offset(t_abs * conv_dim * bf16);
                        ops::conv1d_update_l2norm(
                            ctx.gpu,
                            self.conv1d_l2norm_k,
                            ssm_state.conv_state,
                            qkv_t,
                            &self.ssm.conv1d,
                            conv_out_t,
                            conv_dim as u32,
                            d_conv as u32,
                            1,
                            qk_ch,
                            kd as u32,
                            1e-6,
                            stream,
                        )?;
                        ctx.gpu.copy_d2d_async(
                            ssm_state.conv_state,
                            ssm_state.conv_state_intermediates[t_abs],
                            conv_bytes,
                            stream,
                        )?;
                    }
                }

                // ── 2. Run gdn fused kernel for the chunk ──
                let q_ptr = conv_out_buf.offset(t_done * conv_dim * bf16);
                let k_ptr = q_ptr.offset(key_dim * bf16);
                let v_ptr = q_ptr.offset(key_dim * 2 * bf16);
                let gate_beta_stride = nv * 2 * fp32;
                let gate_ptr = gates_buf.offset(t_done * gate_beta_stride);
                let beta_ptr = gate_ptr.offset(nv * fp32);
                let gdn_out_t = gdn_out_buf.offset(t_done * args.value_dim * bf16);

                // ATLAS_SSM_H_FP16: one decision point for the chunk width.
                // Under the flag the pool holds FP16, so the FP32 kernel would
                // read half-width data as floats — refuse rather than fall back.
                let wy_chunk_k = self.wy_chunk_kernel(chunk);
                self.require_wy_f16(chunk, wy_chunk_k)?;
                let h_fp16 = super::ssm_h_fp16_enabled();
                // Tripwire for a missed conversion hook. Reading an FP32 pool
                // with an FP16 kernel does not crash — it produces fluent
                // garbage — so assert the invariant instead of trusting it.
                if h_fp16 {
                    super::ssm_h_fp16::require_h_f16(ssm_state)?;
                } else {
                    anyhow::ensure!(
                        !ssm_state.h_is_f16,
                        "FP16 h_state encountered in FP32 chunked WY{chunk} branch"
                    );
                }
                match (chunk, h_fp16) {
                    (4, true) => ops::gdn_decode_wy4_f16(
                        ctx.gpu,
                        wy_chunk_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gate_ptr,
                        beta_ptr,
                        gdn_out_t,
                        ssm_state.h_state_intermediates[t_done],
                        ssm_state.h_state_intermediates[t_done + 1],
                        ssm_state.h_state_intermediates[t_done + 2],
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        (nv * 2) as u32,
                        0u32, // state_is_table (contiguous h_state pool)
                        stream,
                    )?,
                    (4, false) => ops::gdn_decode_wy4(
                        ctx.gpu,
                        wy_chunk_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gate_ptr,
                        beta_ptr,
                        gdn_out_t,
                        ssm_state.h_state_intermediates[t_done],
                        ssm_state.h_state_intermediates[t_done + 1],
                        ssm_state.h_state_intermediates[t_done + 2],
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        (nv * 2) as u32,
                        stream,
                    )?,
                    (3, true) => ops::gdn_decode_wy3_f16(
                        ctx.gpu,
                        wy_chunk_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gate_ptr,
                        beta_ptr,
                        gdn_out_t,
                        ssm_state.h_state_intermediates[t_done],
                        ssm_state.h_state_intermediates[t_done + 1],
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        (nv * 2) as u32,
                        0u32, // state_is_table (contiguous h_state pool)
                        stream,
                    )?,
                    (3, false) => ops::gdn_decode_wy3(
                        ctx.gpu,
                        wy_chunk_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gate_ptr,
                        beta_ptr,
                        gdn_out_t,
                        ssm_state.h_state_intermediates[t_done],
                        ssm_state.h_state_intermediates[t_done + 1],
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        (nv * 2) as u32,
                        stream,
                    )?,
                    (2, true) => ops::gdn_decode_wy2_f16(
                        ctx.gpu,
                        wy_chunk_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gate_ptr,
                        beta_ptr,
                        gdn_out_t,
                        ssm_state.h_state_intermediates[t_done],
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        (nv * 2) as u32,
                        0u32, // state_is_table (contiguous h_state pool)
                        stream,
                    )?,
                    (2, false) => ops::gdn_decode_wy2(
                        ctx.gpu,
                        wy_chunk_k,
                        ssm_state.h_state,
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        gate_ptr,
                        beta_ptr,
                        gdn_out_t,
                        ssm_state.h_state_intermediates[t_done],
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        (nv * 2) as u32,
                        stream,
                    )?,
                    // chunk=1 deliberately not handled here — pick_chunk()
                    // guarantees a size-1 remainder never appears (it
                    // pre-splits e.g. K=5 as 3+2 so every chunk is in
                    // {2, 3, 4}). The ops::gdn_decode single-token kernel
                    // expects FP32 q/k/v but conv_out_buf is BF16, which
                    // caused silent corruption on K=5 / K=9 / K=13 (any
                    // K with K%4==1) — confirmed by `def fibonacci(n):
                    // \n if n <= 0:\n... def/:\n\n\n` output at N=4 K=5.
                    _ => unreachable!("pick_chunk guarantees chunk in {{2, 3, 4}}; got {chunk}"),
                }

                // ── 3. Save h_state to chunk-end intermediate (non-final chunks only) ──
                // wy_k writes K-1 explicit intermediates; the K-th position
                // (= chunk-end) lives in h_state. For the LAST chunk the
                // final h_state IS the canonical post-step state and isn't
                // read via intermediates (full-accept rollback keeps
                // h_state as-is). For non-final chunks we need
                // intermediates[t_done+chunk-1] populated so partial
                // accepts that land mid-K can find the per-position state.
                if t_done + chunk < num_tokens {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.h_state,
                        ssm_state.h_state_intermediates[t_done + chunk - 1],
                        h_bytes,
                        stream,
                    )?;
                }

                t_done += chunk;
            }
        }

        Ok(())
    }
}

impl super::Qwen3SsmLayer {
    /// Select the chunked-path WY kernel for a chunk of `n` tokens.
    ///
    /// Under ATLAS_SSM_H_FP16 the h-state in the pool is FP16, so the FP16 twin
    /// is the ONLY correct kernel — the FP32 one would read half-width data as
    /// floats and emit fluent garbage rather than fail. A zero handle is
    /// returned as zero on purpose; `require_wy_f16` turns that into a hard
    /// error at the call site instead of a silent fallback.
    pub(super) fn wy_chunk_kernel(&self, n: usize) -> spark_runtime::gpu::KernelHandle {
        if super::ssm_h_fp16_enabled() {
            return match n {
                4 => self.gdn_wy4_f16_k,
                3 => self.gdn_wy3_f16_k,
                _ => self.gdn_wy2_f16_k,
            };
        }
        match n {
            4 => self.gdn_wy4_k,
            3 => self.gdn_wy3_k,
            _ => self.gdn_wy2_k,
        }
    }

    /// Refuse to run an FP32 WY kernel over an FP16 pool.
    pub(super) fn require_wy_f16(
        &self,
        n: usize,
        wy_k: spark_runtime::gpu::KernelHandle,
    ) -> anyhow::Result<()> {
        if super::ssm_h_fp16_enabled() && wy_k.0 == 0 {
            anyhow::bail!(
                "ATLAS_SSM_H_FP16: no FP16 h-state twin resolved for the chunk width {n}. \
                 Falling back to the FP32 kernel would read the FP16 pool as floats and \
                 emit fluent garbage, so this refuses instead. Unset ATLAS_SSM_H_FP16."
            );
        }
        Ok(())
    }
}
