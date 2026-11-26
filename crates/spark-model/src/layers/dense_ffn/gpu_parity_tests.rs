// SPDX-License-Identifier: AGPL-3.0-only

use super::{Sm120CoalescedSmallMRoute, select_sm120_coalesced_small_m_route};

#[cfg(feature = "cuda")]
#[test]
fn gpu_parity_l2_strip_m8058_ptx_quantizers_and_fused_residual_norm() -> anyhow::Result<()> {
    use super::{
        CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF, CUBLASLT_DYNAMIC_SCALARS_BYTES,
        CUBLASLT_DYNAMIC_SCALE2_OFF, CUBLASLT_DYNAMIC_STATUS_OFF,
        CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES, swizzle_nvfp4_scales_128x4_host,
    };
    use crate::layers::{ops, try_kernel};
    use crate::weight_map::{DenseWeight, QuantizedWeight, quantize_to_nvfp4};
    use half::bf16;
    use spark_runtime::cublaslt::BlockScaleScalars;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};

    let Some(ptx) = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::all_ptx_sets().into_iter().next())
    else {
        return Ok(());
    };
    let Ok(gpu) = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &ptx.modules) else {
        return Ok(());
    };
    let stream = gpu.default_stream();

    let quant_static_k = try_kernel(
        &gpu,
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4_ptx",
    );
    let quant_static_legacy_k = try_kernel(
        &gpu,
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4",
    );
    let quant_k = if quant_static_k.0 != 0 {
        quant_static_k
    } else {
        quant_static_legacy_k
    };
    let merged_silu_legacy_k = try_kernel(
        &gpu,
        "quantize_nvfp4",
        "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4",
    );
    let merged_silu_ptx_k = try_kernel(
        &gpu,
        "quantize_nvfp4",
        "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4_ptx",
    );
    let from_absmax_legacy_k = try_kernel(
        &gpu,
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax",
    );
    let from_absmax_ptx_k = try_kernel(
        &gpu,
        "quantize_nvfp4",
        "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax_ptx",
    );
    let fused_norm_quant_k = try_kernel(
        &gpu,
        "quantize_nvfp4",
        "residual_add_rms_norm_quantize_nvfp4_128x4",
    );
    let absmax_k = try_kernel(&gpu, "quantize_nvfp4", "nvfp4_global_absmax");
    let quant_w_k = try_kernel(&gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4");
    let resid_norm_k = try_kernel(&gpu, "norm", "residual_add_rms_norm");
    let rms_norm_res_k = try_kernel(&gpu, "norm", "rms_norm_residual");
    let rms_norm_res_absmax_k = try_kernel(&gpu, "norm", "rms_norm_residual_absmax");
    let rms_norm_absmax_k = try_kernel(&gpu, "norm", "rms_norm_absmax");
    let gated_prefill_k = try_kernel(&gpu, "norm", "gated_rms_norm_prefill");
    let gated_prefill_absmax_k = try_kernel(&gpu, "norm", "gated_rms_norm_prefill_absmax");
    let sig_gate_k = try_kernel(&gpu, "residual_add", "sigmoid_gate_mul_batched");
    let sig_gate_absmax_k = try_kernel(&gpu, "residual_add", "sigmoid_gate_mul_batched_absmax");
    let resid_add_k = try_kernel(&gpu, "residual_add", "bf16_residual_add");

    if quant_k.0 == 0
        || merged_silu_legacy_k.0 == 0
        || merged_silu_ptx_k.0 == 0
        || from_absmax_legacy_k.0 == 0
        || from_absmax_ptx_k.0 == 0
        || fused_norm_quant_k.0 == 0
        || absmax_k.0 == 0
        || quant_w_k.0 == 0
        || resid_norm_k.0 == 0
        || rms_norm_res_k.0 == 0
        || rms_norm_res_absmax_k.0 == 0
        || rms_norm_absmax_k.0 == 0
        || gated_prefill_k.0 == 0
        || gated_prefill_absmax_k.0 == 0
        || sig_gate_k.0 == 0
        || sig_gate_absmax_k.0 == 0
        || resid_add_k.0 == 0
    {
        return Ok(());
    }

    // 1. Test M = 8058 (= 7 * 1024 + 890) with H = 512, Inter = 1024:
    //    verifies non-1024 remainder strip + PTX merged SiLU quantizer +
    //    fused `residual_add_rms_norm_quantize_nvfp4_128x4` + `beta = 1.0`
    //    against the unfused reference path!
    let m = 8058u32;
    let strip = 1024u32;
    let h = 512u32;
    let inter = 1024u32;
    let eps = 1e-6f32;
    let scale2_in = 0.05f32;
    let scale2_down = 0.05f32;

    let make_bf16 = |len: usize, seed: usize, amp: f32| -> Vec<u8> {
        let mut out = Vec::with_capacity(len * 2);
        for i in 0..len {
            let x = i
                .wrapping_mul(1315423911)
                .wrapping_add(seed.wrapping_mul(2654435761));
            let v = (((x >> 16) & 0x3FF) as f32 / 512.0 - 1.0) * amp;
            out.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
        }
        out
    };

    let hidden_bytes = make_bf16((m * h) as usize, 1, 0.8);
    let src_bytes = make_bf16((m * h) as usize, 2, 0.4);
    let norm_w_bytes = make_bf16(h as usize, 3, 0.25);
    let gate_up_w_bytes = make_bf16((2 * inter * h) as usize, 4, 0.15);
    let down_w_bytes = make_bf16((h * inter) as usize, 5, 0.15);

    let hidden_ref = gpu.alloc(hidden_bytes.len())?;
    let hidden_strip = gpu.alloc(hidden_bytes.len())?;
    let hidden_fused = gpu.alloc(hidden_bytes.len())?;
    let src_dev = gpu.alloc(src_bytes.len())?;
    let norm_w_dev = gpu.alloc(norm_w_bytes.len())?;
    let resid_dev = gpu.alloc(hidden_bytes.len())?;
    let norm_out_dev = gpu.alloc(hidden_bytes.len())?;
    let ffn_out_ref = gpu.alloc(hidden_bytes.len())?;
    let ffn_out_strip = gpu.alloc(hidden_bytes.len())?;

    gpu.copy_h2d(&hidden_bytes, hidden_ref)?;
    gpu.copy_h2d(&hidden_bytes, hidden_strip)?;
    gpu.copy_h2d(&hidden_bytes, hidden_fused)?;
    gpu.copy_h2d(&src_bytes, src_dev)?;
    gpu.copy_h2d(&norm_w_bytes, norm_w_dev)?;

    let merged_w_bf16 = gpu.alloc(gate_up_w_bytes.len())?;
    let down_w_bf16 = gpu.alloc(down_w_bytes.len())?;
    gpu.copy_h2d(&gate_up_w_bytes, merged_w_bf16)?;
    gpu.copy_h2d(&down_w_bytes, down_w_bf16)?;

    let merged_q = quantize_to_nvfp4(
        &DenseWeight {
            weight: merged_w_bf16,
        },
        (2 * inter) as usize,
        h as usize,
        &gpu,
        absmax_k,
        quant_w_k,
        stream,
    )?;
    let down_q = quantize_to_nvfp4(
        &DenseWeight {
            weight: down_w_bf16,
        },
        h as usize,
        inter as usize,
        &gpu,
        absmax_k,
        quant_w_k,
        stream,
    )?;
    gpu.synchronize(stream)?;

    let swizzle_weight_scales =
        |w: &QuantizedWeight, n: u32, k: u32| -> anyhow::Result<DevicePtr> {
            let mut raw = vec![0u8; (n as usize) * (k as usize / 16)];
            gpu.copy_d2h(w.weight_scale, &mut raw)?;
            let swizzled = swizzle_nvfp4_scales_128x4_host(&raw, n, k)?;
            let dev = gpu.alloc(swizzled.len())?;
            gpu.copy_h2d(&swizzled, dev)?;
            Ok(dev)
        };
    let merged_s128 = swizzle_weight_scales(&merged_q, 2 * inter, h)?;
    let down_s128 = swizzle_weight_scales(&down_q, h, inter)?;

    let full_layout = super::e2m1_scratch_layout(m as usize, inter as usize)?;
    let a_packed = gpu.alloc(full_layout.packed_bytes)?;
    let a_scale = gpu.alloc(full_layout.scale_bytes)?;
    let merged_out_full = gpu.alloc((m as usize) * (2 * inter as usize) * 2)?;
    let merged_out_strip = gpu.alloc((strip as usize) * (2 * inter as usize) * 2)?;

    let norm_w = DenseWeight { weight: norm_w_dev };
    let merged_scalars = BlockScaleScalars::host(scale2_in * merged_q.weight_scale_2, 0.0);
    let down_scalars_b0 = BlockScaleScalars::host(scale2_down * down_q.weight_scale_2, 0.0);
    let down_scalars_b1 = BlockScaleScalars::host(scale2_down * down_q.weight_scale_2, 1.0);

    // (A) Unfused full-M=8058 reference path (using `merged_silu_ptx_k` so
    //     we isolate M_strip=1024 + remainder M=890 parity cleanly; direct
    //     `merged_silu_legacy_k` vs `merged_silu_ptx_k` nibble/scale parity
    //     is checked separately below).
    ops::residual_add_rms_norm(
        &gpu,
        resid_norm_k,
        hidden_ref,
        src_dev,
        &norm_w,
        norm_out_dev,
        resid_dev,
        m,
        h,
        eps,
        stream,
    )?;
    ops::quantize_bf16_to_nvfp4_atlas_128x4(
        &gpu,
        quant_k,
        norm_out_dev,
        a_packed,
        a_scale,
        scale2_in,
        m,
        h,
        stream,
    )?;
    spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
        a_packed.0,
        a_scale.0,
        merged_q.weight.0,
        merged_s128.0,
        merged_out_full.0,
        m,
        2 * inter,
        h,
        2 * inter,
        merged_scalars,
        stream,
    )?;
    ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
        &gpu,
        merged_silu_ptx_k,
        merged_out_full,
        a_packed,
        a_scale,
        scale2_down,
        m,
        inter,
        stream,
    )?;
    spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
        a_packed.0,
        a_scale.0,
        down_q.weight.0,
        down_s128.0,
        ffn_out_ref.0,
        m,
        h,
        inter,
        h,
        down_scalars_b0,
        stream,
    )?;
    ops::residual_add(&gpu, resid_add_k, hidden_ref, ffn_out_ref, m * h, stream)?;

    // (B) L2-stripped M_strip=1024 path using PTX merged SiLU quantizer (reading `norm_out_dev`):
    let mut row0 = 0u32;
    while row0 < m {
        let m_s = (m - row0).min(strip);
        let off = (row0 as usize) * (h as usize) * 2;
        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            &gpu,
            quant_k,
            norm_out_dev.offset(off),
            a_packed,
            a_scale,
            scale2_in,
            m_s,
            h,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            a_packed.0,
            a_scale.0,
            merged_q.weight.0,
            merged_s128.0,
            merged_out_strip.0,
            m_s,
            2 * inter,
            h,
            2 * inter,
            merged_scalars,
            stream,
        )?;
        ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
            &gpu,
            merged_silu_ptx_k,
            merged_out_strip,
            a_packed,
            a_scale,
            scale2_down,
            m_s,
            inter,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            a_packed.0,
            a_scale.0,
            down_q.weight.0,
            down_s128.0,
            ffn_out_strip.offset(off).0,
            m_s,
            h,
            inter,
            h,
            down_scalars_b0,
            stream,
        )?;
        row0 += m_s;
    }
    ops::residual_add(&gpu, resid_add_k, hidden_strip, src_dev, m * h, stream)?;
    ops::residual_add(
        &gpu,
        resid_add_k,
        hidden_strip,
        ffn_out_strip,
        m * h,
        stream,
    )?;

    // (C) Fully fused `residual_add_rms_norm_quantize_nvfp4_128x4` + L2-strip + `beta = 1.0`:
    row0 = 0;
    while row0 < m {
        let m_s = (m - row0).min(strip);
        let off = (row0 as usize) * (h as usize) * 2;
        ops::residual_add_rms_norm_quantize_nvfp4_128x4(
            &gpu,
            fused_norm_quant_k,
            hidden_fused.offset(off),
            src_dev.offset(off),
            &norm_w,
            DevicePtr::NULL,
            a_packed,
            a_scale,
            scale2_in,
            eps,
            m_s,
            h,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            a_packed.0,
            a_scale.0,
            merged_q.weight.0,
            merged_s128.0,
            merged_out_strip.0,
            m_s,
            2 * inter,
            h,
            2 * inter,
            merged_scalars,
            stream,
        )?;
        ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
            &gpu,
            merged_silu_ptx_k,
            merged_out_strip,
            a_packed,
            a_scale,
            scale2_down,
            m_s,
            inter,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            a_packed.0,
            a_scale.0,
            down_q.weight.0,
            down_s128.0,
            hidden_fused.offset(off).0,
            m_s,
            h,
            inter,
            h,
            down_scalars_b1,
            stream,
        )?;
        row0 += m_s;
    }
    gpu.synchronize(stream)?;

    let read_bf16 = |ptr: DevicePtr, bytes: usize| -> anyhow::Result<Vec<f32>> {
        let mut raw = vec![0u8; bytes];
        gpu.copy_d2h(ptr, &mut raw)?;
        Ok(raw
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect())
    };
    let ref_ffn = read_bf16(ffn_out_ref, hidden_bytes.len())?;
    let strip_ffn = read_bf16(ffn_out_strip, hidden_bytes.len())?;
    let ref_h = read_bf16(hidden_ref, hidden_bytes.len())?;
    let fused_h = read_bf16(hidden_fused, hidden_bytes.len())?;

    let mut max_strip_diff = 0.0f32;
    let mut max_fused_diff = 0.0f32;
    let mut max_ref_abs = 0.0f32;
    for i in 0..ref_ffn.len() {
        assert!(strip_ffn[i].is_finite() && fused_h[i].is_finite());
        max_ref_abs = max_ref_abs.max(ref_h[i].abs());
        max_strip_diff = max_strip_diff.max((ref_ffn[i] - strip_ffn[i]).abs());
        max_fused_diff = max_fused_diff.max((ref_h[i] - fused_h[i]).abs());
    }
    // Strip-mined M_strip=1024 (7x1024 + 890) and fused `residual_add_rms_norm_quantize_nvfp4_128x4`
    // + `beta = 1.0` must match unstripped full M=8058 within 2% of peak magnitude (BF16 ULP).
    let tol = (max_ref_abs * 0.02).max(0.02);
    assert!(
        max_strip_diff <= tol,
        "L2 M_strip=1024 at M=8058 diverged: max_strip_diff={max_strip_diff}, tol={tol}"
    );
    assert!(
        max_fused_diff <= tol,
        "Fused residual_add_rms_norm_quantize + beta=1.0 diverged: max_fused_diff={max_fused_diff}, tol={tol}"
    );

    // Direct A/B of `merged_silu_legacy_k` vs `merged_silu_ptx_k` on identical `merged_out_full`:
    // verifies 128x4 scale swizzle + E2M1 nibble packing match (only <= 1-step RNE vs half-away ties).
    let dbg_p_leg = gpu.alloc((strip as usize) * (inter as usize / 2))?;
    let dbg_s_leg = gpu.alloc((strip as usize) * (inter as usize / 16))?;
    let dbg_p_ptx = gpu.alloc((strip as usize) * (inter as usize / 2))?;
    let dbg_s_ptx = gpu.alloc((strip as usize) * (inter as usize / 16))?;
    ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
        &gpu,
        merged_silu_legacy_k,
        merged_out_full,
        dbg_p_leg,
        dbg_s_leg,
        scale2_down,
        strip,
        inter,
        stream,
    )?;
    ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
        &gpu,
        merged_silu_ptx_k,
        merged_out_full,
        dbg_p_ptx,
        dbg_s_ptx,
        scale2_down,
        strip,
        inter,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let mut pl = vec![0u8; (strip as usize) * (inter as usize / 2)];
    let mut pp = vec![0u8; pl.len()];
    let mut sl = vec![0u8; (strip as usize) * (inter as usize / 16)];
    let mut sp = vec![0u8; sl.len()];
    gpu.copy_d2h(dbg_p_leg, &mut pl)?;
    gpu.copy_d2h(dbg_p_ptx, &mut pp)?;
    gpu.copy_d2h(dbg_s_leg, &mut sl)?;
    gpu.copy_d2h(dbg_s_ptx, &mut sp)?;
    let scale_diffs = sl.iter().zip(&sp).filter(|(a, b)| a != b).count();
    let mut nibble_diffs = 0usize;
    for (&a, &b) in pl.iter().zip(&pp) {
        for shift in [0u8, 4u8] {
            let na = (a >> shift) & 0xF;
            let nb = (b >> shift) & 0xF;
            if na != nb {
                nibble_diffs += 1;
                // Adjacent E2M1 code (same sign, magnitude diff <= 1) or +/-0 tie.
                let mag_a = (na & 0x7) as i8;
                let mag_b = (nb & 0x7) as i8;
                let same_sign = (na & 0x8) == (nb & 0x8);
                assert!(
                    (same_sign && (mag_a - mag_b).abs() <= 1) || (mag_a == 0 && mag_b == 0),
                    "merged_silu_ptx nibble diverged by >1 E2M1 step: leg=0x{na:x} ptx=0x{nb:x}"
                );
            }
        }
    }
    assert!(
        scale_diffs * 1000 <= sl.len(),
        "merged_silu_ptx scale_diffs too high: {scale_diffs}/{}",
        sl.len()
    );
    assert!(
        nibble_diffs * 200 <= pl.len() * 2,
        "merged_silu_ptx nibble_diffs too high: {nibble_diffs}/{}",
        pl.len() * 2
    );
    let _ = gpu.free(dbg_p_leg);
    let _ = gpu.free(dbg_s_leg);
    let _ = gpu.free(dbg_p_ptx);
    let _ = gpu.free(dbg_s_ptx);

    // 2. Verify 0-ULP parity of `rms_norm_absmax`, `rms_norm_residual_absmax`,
    //    `gated_rms_norm_prefill_absmax`, `sigmoid_gate_mul_batched_absmax`, and
    //    `quantize_bf16_to_nvfp4_atlas_128x4_from_absmax_ptx`.
    let n_tok = 128u32;
    let dim = 512u32;
    let elem_bytes = (n_tok * dim * 2) as usize;
    let out_a = gpu.alloc(elem_bytes)?;
    let out_b = gpu.alloc(elem_bytes)?;
    let res_a = gpu.alloc(elem_bytes)?;
    let res_b = gpu.alloc(elem_bytes)?;
    let gmax_ref = gpu.alloc(4)?;
    let gmax_fused = gpu.alloc(4)?;

    // 2a. rms_norm_residual vs rms_norm_residual_absmax & rms_norm_absmax
    gpu.memset_async(gmax_ref, 0, 4, stream)?;
    gpu.memset_async(gmax_fused, 0, 4, stream)?;
    ops::rms_norm_residual(
        &gpu,
        rms_norm_res_k,
        src_dev,
        &norm_w,
        out_a,
        res_a,
        n_tok,
        dim,
        eps,
        stream,
    )?;
    ops::nvfp4_global_absmax(&gpu, absmax_k, out_a, gmax_ref, n_tok * dim, stream)?;
    ops::rms_norm_residual_absmax(
        &gpu,
        rms_norm_res_absmax_k,
        src_dev,
        &norm_w,
        out_b,
        res_b,
        gmax_fused,
        n_tok,
        dim,
        eps,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let mut raw_a = vec![0u8; elem_bytes];
    let mut raw_b = vec![0u8; elem_bytes];
    let mut gm_a = [0u8; 4];
    let mut gm_b = [0u8; 4];
    gpu.copy_d2h(out_a, &mut raw_a)?;
    gpu.copy_d2h(out_b, &mut raw_b)?;
    gpu.copy_d2h(gmax_ref, &mut gm_a)?;
    gpu.copy_d2h(gmax_fused, &mut gm_b)?;
    assert_eq!(
        raw_a, raw_b,
        "rms_norm_residual_absmax must be 0-ULP bit-identical"
    );
    assert_eq!(
        gm_a, gm_b,
        "rms_norm_residual_absmax global_max must be bit-identical"
    );

    gpu.memset_async(gmax_fused, 0, 4, stream)?;
    ops::rms_norm_absmax(
        &gpu,
        rms_norm_absmax_k,
        src_dev,
        &norm_w,
        out_b,
        gmax_fused,
        n_tok,
        dim,
        eps,
        stream,
    )?;
    gpu.synchronize(stream)?;
    gpu.copy_d2h(out_b, &mut raw_b)?;
    gpu.copy_d2h(gmax_fused, &mut gm_b)?;
    assert_eq!(raw_a, raw_b, "rms_norm_absmax must be 0-ULP bit-identical");
    assert_eq!(
        gm_a, gm_b,
        "rms_norm_absmax global_max must be bit-identical"
    );

    // 2b. gated_rms_norm_prefill vs gated_rms_norm_prefill_absmax (head_dim=128, 4 heads)
    gpu.memset_async(gmax_ref, 0, 4, stream)?;
    gpu.memset_async(gmax_fused, 0, 4, stream)?;
    ops::gated_rms_norm_prefill(
        &gpu,
        gated_prefill_k,
        src_dev,
        hidden_ref,
        &norm_w,
        out_a,
        4,
        128,
        eps,
        n_tok,
        dim,
        dim,
        stream,
    )?;
    ops::nvfp4_global_absmax(&gpu, absmax_k, out_a, gmax_ref, n_tok * dim, stream)?;
    ops::gated_rms_norm_prefill_absmax(
        &gpu,
        gated_prefill_absmax_k,
        src_dev,
        hidden_ref,
        &norm_w,
        out_b,
        gmax_fused,
        4,
        128,
        eps,
        n_tok,
        dim,
        dim,
        stream,
    )?;
    gpu.synchronize(stream)?;
    gpu.copy_d2h(out_a, &mut raw_a)?;
    gpu.copy_d2h(out_b, &mut raw_b)?;
    gpu.copy_d2h(gmax_ref, &mut gm_a)?;
    gpu.copy_d2h(gmax_fused, &mut gm_b)?;
    assert_eq!(
        raw_a, raw_b,
        "gated_rms_norm_prefill_absmax must be 0-ULP bit-identical"
    );
    assert_eq!(
        gm_a, gm_b,
        "gated_rms_norm_prefill_absmax global_max must be bit-identical"
    );

    // 2c. sigmoid_gate_mul_batched vs sigmoid_gate_mul_batched_absmax
    gpu.memset_async(gmax_ref, 0, 4, stream)?;
    gpu.memset_async(gmax_fused, 0, 4, stream)?;
    ops::sigmoid_gate_mul_batched(
        &gpu, sig_gate_k, src_dev, hidden_ref, out_a, dim, dim, n_tok, stream,
    )?;
    ops::nvfp4_global_absmax(&gpu, absmax_k, out_a, gmax_ref, n_tok * dim, stream)?;
    ops::sigmoid_gate_mul_batched_absmax(
        &gpu,
        sig_gate_absmax_k,
        src_dev,
        hidden_ref,
        out_b,
        gmax_fused,
        dim,
        dim,
        n_tok,
        stream,
    )?;
    gpu.synchronize(stream)?;
    gpu.copy_d2h(out_a, &mut raw_a)?;
    gpu.copy_d2h(out_b, &mut raw_b)?;
    gpu.copy_d2h(gmax_ref, &mut gm_a)?;
    gpu.copy_d2h(gmax_fused, &mut gm_b)?;
    assert_eq!(
        raw_a, raw_b,
        "sigmoid_gate_mul_batched_absmax must be 0-ULP bit-identical"
    );
    assert_eq!(
        gm_a, gm_b,
        "sigmoid_gate_mul_batched_absmax global_max must be bit-identical"
    );

    // 2d. `quantize_bf16_to_nvfp4_atlas_128x4_from_absmax` legacy vs PTX
    let dyn_a = gpu.alloc(CUBLASLT_DYNAMIC_SCALARS_BYTES)?;
    let dyn_b = gpu.alloc(CUBLASLT_DYNAMIC_SCALARS_BYTES)?;
    let packed_b = gpu.alloc(full_layout.packed_bytes)?;
    let scale_b = gpu.alloc(full_layout.scale_bytes)?;
    gpu.memset_async(dyn_a, 0, CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES, stream)?;
    gpu.memset_async(dyn_b, 0, CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES, stream)?;
    ops::nvfp4_global_absmax(
        &gpu,
        absmax_k,
        out_a,
        dyn_a.offset(CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF),
        n_tok * dim,
        stream,
    )?;
    ops::nvfp4_global_absmax(
        &gpu,
        absmax_k,
        out_a,
        dyn_b.offset(CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF),
        n_tok * dim,
        stream,
    )?;
    for (kern, p_out, s_out, dyn_buf) in [
        (from_absmax_legacy_k, a_packed, a_scale, dyn_a),
        (from_absmax_ptx_k, packed_b, scale_b, dyn_b),
    ] {
        spark_runtime::kernel_args::KernelLaunch::new(&gpu, kern)
            .grid([128, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(out_a)
            .arg_ptr(p_out)
            .arg_ptr(s_out)
            .arg_ptr(dyn_buf.offset(CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF))
            .arg_ptr(dyn_buf.offset(CUBLASLT_DYNAMIC_SCALE2_OFF))
            .arg_ptr(dyn_buf.offset(CUBLASLT_DYNAMIC_STATUS_OFF))
            .arg_u32(n_tok)
            .arg_u32(dim)
            .launch(stream)?;
    }
    gpu.synchronize(stream)?;
    let p_bytes = (n_tok as usize) * (dim as usize / 2);
    let s_bytes = (n_tok as usize) * (dim as usize / 16);
    let mut pa = vec![0u8; p_bytes];
    let mut pb = vec![0u8; p_bytes];
    let mut sa = vec![0u8; s_bytes];
    let mut sb = vec![0u8; s_bytes];
    let mut s2a = [0u8; 4];
    let mut s2b = [0u8; 4];
    let mut sta = [0u8; 4];
    let mut stb = [0u8; 4];
    gpu.copy_d2h(a_packed, &mut pa)?;
    gpu.copy_d2h(packed_b, &mut pb)?;
    gpu.copy_d2h(a_scale, &mut sa)?;
    gpu.copy_d2h(scale_b, &mut sb)?;
    gpu.copy_d2h(dyn_a.offset(CUBLASLT_DYNAMIC_SCALE2_OFF), &mut s2a)?;
    gpu.copy_d2h(dyn_b.offset(CUBLASLT_DYNAMIC_SCALE2_OFF), &mut s2b)?;
    gpu.copy_d2h(dyn_a.offset(CUBLASLT_DYNAMIC_STATUS_OFF), &mut sta)?;
    gpu.copy_d2h(dyn_b.offset(CUBLASLT_DYNAMIC_STATUS_OFF), &mut stb)?;
    assert_eq!(s2a, s2b, "from_absmax_ptx scale2_a mismatch");
    assert_eq!(sta, [0, 0, 0, 0], "from_absmax legacy status non-zero");
    assert_eq!(stb, [0, 0, 0, 0], "from_absmax_ptx status non-zero");
    assert_eq!(
        sa, sb,
        "from_absmax_ptx 128x4 scales must match legacy byte-for-byte"
    );
    // Legacy uses round-half-away-from-zero (`atlas_quantize_e2m1`) while PTX uses
    // hardware IEEE RNE (`cvt.rn.satfinite.e2m1x2.f32`) + `rcp.approx.ftz.f32`:
    // every differing nibble must be an adjacent E2M1 midpoint tie (<= 1 step).
    let mut from_absmax_nibble_diffs = 0usize;
    for (&a, &b) in pa.iter().zip(&pb) {
        for shift in [0u8, 4u8] {
            let na = (a >> shift) & 0xF;
            let nb = (b >> shift) & 0xF;
            if na != nb {
                from_absmax_nibble_diffs += 1;
                let mag_a = (na & 0x7) as i8;
                let mag_b = (nb & 0x7) as i8;
                let same_sign = (na & 0x8) == (nb & 0x8);
                assert!(
                    (same_sign && (mag_a - mag_b).abs() <= 1) || (mag_a == 0 && mag_b == 0),
                    "from_absmax_ptx nibble diverged by >1 E2M1 step: leg=0x{na:x} ptx=0x{nb:x}"
                );
            }
        }
    }
    assert!(
        from_absmax_nibble_diffs * 100 <= pa.len() * 2,
        "from_absmax_ptx nibble_diffs too high: {from_absmax_nibble_diffs}/{}",
        pa.len() * 2
    );
    eprintln!(
        "PARITY OK dense_ffn SM120 (M={m}, strip={strip}, H={h}, Inter={inter}): \
         max_strip_diff={max_strip_diff:.6}, max_fused_diff={max_fused_diff:.6}, tol={tol:.6}, \
         merged_silu_scale_diffs={scale_diffs}/{}, merged_silu_nibble_ties={nibble_diffs}/{}, \
         from_absmax_ptx_scales=byte-exact, from_absmax_ptx_nibble_ties={from_absmax_nibble_diffs}/{}, \
         absmax_kernels=0-ULP-exact",
        sl.len(),
        pl.len() * 2,
        pa.len() * 2,
    );

    for ptr in [
        scale_b,
        packed_b,
        dyn_b,
        dyn_a,
        gmax_fused,
        gmax_ref,
        res_b,
        res_a,
        out_b,
        out_a,
        merged_out_strip,
        merged_out_full,
        a_scale,
        a_packed,
        down_s128,
        merged_s128,
        down_q.weight,
        down_q.weight_scale,
        merged_q.weight,
        merged_q.weight_scale,
        down_w_bf16,
        merged_w_bf16,
        ffn_out_strip,
        ffn_out_ref,
        norm_out_dev,
        resid_dev,
        norm_w_dev,
        src_dev,
        hidden_fused,
        hidden_strip,
        hidden_ref,
    ] {
        let _ = gpu.free(ptr);
    }
    Ok(())
}

#[test]
fn sm120_coalesced_small_m_route_gates_gb10_and_selects_medium_n_warp_splitk() {
    // GB10 (<= 64 SMs) is always Disabled.
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 48, 8, 5120, 17408, 16, true),
        Sm120CoalescedSmallMRoute::Disabled
    );
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 64, 8, 5120, 25600, 32, true),
        Sm120CoalescedSmallMRoute::Disabled
    );
    // Explicit opt-out disables even on SM120.
    assert_eq!(
        select_sm120_coalesced_small_m_route(false, true, 188, 8, 5120, 17408, 16, true),
        Sm120CoalescedSmallMRoute::Disabled
    );
    // SM120 N=5120 deep-K projections (fc, down_proj, ssm.out, attn.o) pick MediumNWarpSplitK.
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 188, 8, 5120, 25600, 32, true),
        Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2: true }
    );
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 188, 8, 5120, 17408, 16, true),
        Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2: true }
    );
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 188, 8, 5120, 6144, 16, false),
        Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2: false }
    );
    // Wide deep-K projections (ssm.in_proj_qkvz N=16384, attn.q_proj N=12288) use
    // MediumNWarpSplitK (v2_n16_sk4_m16) when wide_sk4 is enabled, and lm_head
    // (N=248320) uses WideNWarpSplitK (v2_n32_sk4_m16). Narrow (N=1024) delegates.
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 188, 8, 16384, 5120, 16, true),
        Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2: true }
    );
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, false, 188, 8, 16384, 5120, 16, true),
        Sm120CoalescedSmallMRoute::DelegateFastProj
    );
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 188, 8, 1024, 5120, 16, true),
        Sm120CoalescedSmallMRoute::DelegateFastProj
    );
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 188, 7, 248320, 5120, 16, true),
        Sm120CoalescedSmallMRoute::WideNWarpSplitK
    );
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, false, 188, 7, 248320, 5120, 16, true),
        Sm120CoalescedSmallMRoute::DelegateFastProj
    );
    // Out-of-range M or unaligned K fail closed.
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 188, 0, 5120, 17408, 16, true),
        Sm120CoalescedSmallMRoute::Disabled
    );
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 188, 17, 5120, 17408, 16, true),
        Sm120CoalescedSmallMRoute::Disabled
    );
    assert_eq!(
        select_sm120_coalesced_small_m_route(true, true, 188, 8, 5120, 5136, 16, true),
        Sm120CoalescedSmallMRoute::Disabled
    );
}

#[cfg(feature = "cuda")]
#[test]
fn qwen38_27b_ported_coalesced_bf16_kernels_match_reference_and_preserve_outliers()
-> anyhow::Result<()> {
    use crate::weight_map::{DenseWeight, quantize_to_nvfp4};
    use half::bf16;
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::kernel_args::KernelLaunch;

    let Some(ptx) = atlas_kernels::ptx_for_model("qwen3.8-27b") else {
        return Ok(());
    };
    let Ok(gpu) = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &ptx.modules) else {
        return Ok(());
    };
    let stream = gpu.default_stream();

    let absmax_k = crate::layers::try_kernel(&gpu, "quantize_nvfp4", "nvfp4_global_absmax");
    let quantize_k = crate::layers::try_kernel(&gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4");
    let ref_k = crate::layers::try_kernel(&gpu, "w4a16", "w4a16_gemm");
    if absmax_k.0 == 0 || quantize_k.0 == 0 || ref_k.0 == 0 {
        return Ok(());
    }

    const E2M1_LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    fn fp8_e4m3_to_f32(bits: u8) -> f32 {
        let sign = if (bits & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((bits >> 3) & 0x0f) as i32;
        let mant = (bits & 0x07) as f32;
        if exp == 0 {
            sign * (mant / 8.0) * 2.0f32.powi(-6)
        } else if exp == 15 && (bits & 0x07) == 7 {
            f32::NAN
        } else {
            sign * (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
        }
    }

    // Test shapes covering fc / down_proj / ssm.out (N=5120), ssm.qkvz (N=16384),
    // attn.k/v (N=1024), and un-normalized outliers > +-448 (which FP8 E4M3 clips).
    let cases: &[(&str, u32, u32, u32, u32, u32, u32)] = &[
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m16",
            8,
            512,
            1024,
            16,
            16,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n32_sk4_m16",
            8,
            512,
            1024,
            32,
            16,
            512,
        ),
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m32",
            32,
            512,
            1024,
            16,
            32,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m32",
            27,
            512,
            1024,
            16,
            32,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n32_sk4_m32",
            32,
            512,
            1024,
            32,
            32,
            512,
        ),
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m64",
            64,
            512,
            1024,
            16,
            64,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m64",
            53,
            512,
            1024,
            16,
            64,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n32_sk2_m64",
            64,
            512,
            1024,
            32,
            64,
            256,
        ),
        (
            "w4a16_gemm_coalesced_n16_sk4_m16",
            8,
            512,
            1024,
            16,
            16,
            256,
        ),
        ("w4a16_gemm_coalesced_n8_sk8_m16", 8, 512, 1024, 8, 16, 256),
        ("w4a16_gemm_coalesced_v2_n32_m16", 8, 512, 1024, 32, 16, 128),
        ("w4a16_gemm_coalesced_v2_n64_m16", 7, 512, 1024, 64, 16, 256),
    ];

    for &(sym, m, n, k, co_n, tile_m, threads) in cases {
        let kern = crate::layers::try_kernel(&gpu, "w4a16", sym);
        assert_ne!(
            kern.0, 0,
            "ported coalesced kernel `{sym}` missing from qwen3.8-27b PTX bundle"
        );

        // Deterministic BF16 activations including +-1200.0 outliers (> E4M3 max 448).
        let mut a_host = Vec::with_capacity((m * k) as usize);
        for r in 0..m {
            for c in 0..k {
                let v = if c % 97 == 0 {
                    if (r + c) % 2 == 0 {
                        1200.0f32
                    } else {
                        -1200.0f32
                    }
                } else {
                    ((((r * 31 + c * 17) % 101) as f32) - 50.0) * 0.02
                };
                a_host.push(bf16::from_f32(v));
            }
        }
        let mut w_host = Vec::with_capacity((n * k) as usize);
        for r in 0..n {
            for c in 0..k {
                let v = ((((r * 13 + c * 29) % 89) as f32) - 44.0) * 0.008;
                w_host.push(bf16::from_f32(v));
            }
        }

        let a_bytes: Vec<u8> = a_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_bytes: Vec<u8> = w_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out_bytes = (m * n) as usize * 2;

        let a_dev = gpu.alloc(a_bytes.len())?;
        let w_dev = gpu.alloc(w_bytes.len())?;
        let out_ref_dev = gpu.alloc(out_bytes)?;
        let out_test_dev = gpu.alloc(out_bytes)?;
        gpu.copy_h2d(&a_bytes, a_dev)?;
        gpu.copy_h2d(&w_bytes, w_dev)?;

        let qw = quantize_to_nvfp4(
            &DenseWeight { weight: w_dev },
            n as usize,
            k as usize,
            &gpu,
            absmax_k,
            quantize_k,
            stream,
        )?;

        crate::layers::ops::w4a16_gemm(&gpu, ref_k, a_dev, &qw, out_ref_dev, m, n, k, stream)?;
        KernelLaunch::new(&gpu, kern)
            .grid([n.div_ceil(co_n), m.div_ceil(tile_m), 1])
            .block([threads, 1, 1])
            .arg_ptr(a_dev)
            .arg_ptr(qw.weight)
            .arg_ptr(qw.weight_scale)
            .arg_f32(qw.weight_scale_2)
            .arg_ptr(out_test_dev)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)?;
        gpu.synchronize(stream)?;

        let mut ref_raw = vec![0u8; out_bytes];
        let mut test_raw = vec![0u8; out_bytes];
        let mut packed_raw = vec![0u8; (n * (k / 2)) as usize];
        let mut scale_raw = vec![0u8; (n * (k / 16)) as usize];
        gpu.copy_d2h(out_ref_dev, &mut ref_raw)?;
        gpu.copy_d2h(out_test_dev, &mut test_raw)?;
        gpu.copy_d2h(qw.weight, &mut packed_raw)?;
        gpu.copy_d2h(qw.weight_scale, &mut scale_raw)?;

        // Exact FP32 reference matching the coalesced kernel's BF16 weight
        // dequant (`bf16(e2m1 * fp8)`) followed by epilogue `* scale2`.
        let mut max_abs_cpu = 0.0f32;
        let mut max_bf16_ulp = 0u32;
        let mut max_abs_vs_ref = 0.0f32;
        let mut max_abs_clipped = 0.0f32;
        for r in 0..m as usize {
            for col in 0..n as usize {
                let mut sum_exact = 0.0f32;
                let mut sum_clipped = 0.0f32;
                for kk in 0..k as usize {
                    let byte = packed_raw[col * (k as usize / 2) + (kk / 2)];
                    let nib = if (kk & 1) == 1 {
                        (byte >> 4) & 0x0f
                    } else {
                        byte & 0x0f
                    };
                    let sc = fp8_e4m3_to_f32(scale_raw[col * (k as usize / 16) + (kk / 16)]);
                    let w_noscale2 = bf16::from_f32(E2M1_LUT[nib as usize] * sc).to_f32();
                    let a_val = a_host[r * k as usize + kk].to_f32();
                    sum_exact += a_val * w_noscale2;
                    sum_clipped += a_val.clamp(-448.0, 448.0) * w_noscale2;
                }
                let cpu_bf16_val = bf16::from_f32(sum_exact * qw.weight_scale_2);
                let cpu_bf16 = cpu_bf16_val.to_f32();
                let clipped_bf16 = bf16::from_f32(sum_clipped * qw.weight_scale_2).to_f32();
                let idx = r * n as usize + col;
                let ref_v = bf16::from_le_bytes([ref_raw[2 * idx], ref_raw[2 * idx + 1]]).to_f32();
                let test_bf16_val = bf16::from_le_bytes([test_raw[2 * idx], test_raw[2 * idx + 1]]);
                let test_v = test_bf16_val.to_f32();
                assert!(
                    test_v.is_finite(),
                    "{sym} emitted non-finite output at {idx}"
                );
                let ulp = if cpu_bf16_val.to_bits() == test_bf16_val.to_bits()
                    || (cpu_bf16 == 0.0 && test_v == 0.0)
                {
                    0u32
                } else if (cpu_bf16_val.to_bits() & 0x8000) == (test_bf16_val.to_bits() & 0x8000) {
                    u32::from(cpu_bf16_val.to_bits().abs_diff(test_bf16_val.to_bits()))
                } else {
                    u32::from(cpu_bf16_val.to_bits() & 0x7fff)
                        + u32::from(test_bf16_val.to_bits() & 0x7fff)
                };
                max_bf16_ulp = max_bf16_ulp.max(ulp);
                max_abs_cpu = max_abs_cpu.max((cpu_bf16 - test_v).abs());
                max_abs_vs_ref = max_abs_vs_ref.max((ref_v - test_v).abs());
                max_abs_clipped = max_abs_clipped.max((clipped_bf16 - test_v).abs());
            }
        }
        eprintln!(
            "PARITY {sym} (M={m}, N={n}, K={k}): max_bf16_ulp={max_bf16_ulp}, max_abs_vs_exact_cpu={max_abs_cpu:.6}, max_abs_vs_w4a16_gemm={max_abs_vs_ref:.6}, max_diff_from_e4m3_clipped={max_abs_clipped:.3}"
        );
        assert!(
            max_bf16_ulp <= 1,
            "{sym} drifted vs exact BF16xBF16 CPU reference (max_bf16_ulp={max_bf16_ulp}, max_abs_cpu={max_abs_cpu:.6}, max_abs_vs_ref={max_abs_vs_ref:.6})"
        );
        assert!(
            max_abs_clipped > 100.0,
            "{sym} unexpectedly matched E4M3-clipped output (max_diff={max_abs_clipped:.3})"
        );

        for ptr in [
            a_dev,
            w_dev,
            out_ref_dev,
            out_test_dev,
            qw.weight,
            qw.weight_scale,
        ] {
            let _ = gpu.free(ptr);
        }
    }

    // Verify fused BF16xBF16 MMA coalesced warp-split-K gate+up+silu kernels
    // (`m16`, `m32`, `m64`) against exact FP32 CPU reference across full and
    // partial-M tail shapes (including K=128, 1024, 2560).
    let fused_cases: &[(&str, u32, u32, u32, u32, u32, u32)] = &[
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m16_gateup_silu",
            8,
            256,
            1024,
            16,
            16,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n32_sk4_m16_gateup_silu",
            8,
            256,
            1024,
            32,
            16,
            512,
        ),
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m16_gateup_silu",
            7,
            272,
            2560,
            16,
            16,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m16_gateup_silu",
            1,
            128,
            128,
            16,
            16,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m32_gateup_silu",
            32,
            256,
            1024,
            16,
            32,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m32_gateup_silu",
            25,
            272,
            1024,
            16,
            32,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m64_gateup_silu",
            64,
            256,
            1024,
            16,
            64,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n16_sk4_m64_gateup_silu",
            53,
            272,
            1024,
            16,
            64,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n32_sk2_m64_gateup_silu",
            64,
            256,
            1024,
            32,
            64,
            256,
        ),
        (
            "w4a16_gemm_coalesced_v2_n32_sk2_m64_gateup_silu",
            49,
            288,
            1024,
            32,
            64,
            256,
        ),
    ];
    for &(sym, m, n, k, co_n, tile_m, threads) in fused_cases {
        let kern = crate::layers::try_kernel(&gpu, "w4a16", sym);
        assert_ne!(kern.0, 0, "fused coalesced kernel `{sym}` missing");

        let mut a_host = Vec::with_capacity((m * k) as usize);
        for r in 0..m {
            for c in 0..k {
                let v = ((((r * 31 + c * 17) % 101) as f32) - 50.0) * 0.015;
                a_host.push(bf16::from_f32(v));
            }
        }
        let mut wg_host = Vec::with_capacity((n * k) as usize);
        let mut wu_host = Vec::with_capacity((n * k) as usize);
        for r in 0..n {
            for c in 0..k {
                wg_host.push(bf16::from_f32(
                    ((((r * 13 + c * 29) % 89) as f32) - 44.0) * 0.006,
                ));
                wu_host.push(bf16::from_f32(
                    ((((r * 19 + c * 23) % 83) as f32) - 41.0) * 0.006,
                ));
            }
        }

        let a_bytes: Vec<u8> = a_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wg_bytes: Vec<u8> = wg_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wu_bytes: Vec<u8> = wu_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out_bytes = (m * n) as usize * 2;

        let a_dev = gpu.alloc(a_bytes.len())?;
        let wg_dev = gpu.alloc(wg_bytes.len())?;
        let wu_dev = gpu.alloc(wu_bytes.len())?;
        let out_dev = gpu.alloc(out_bytes)?;
        gpu.copy_h2d(&a_bytes, a_dev)?;
        gpu.copy_h2d(&wg_bytes, wg_dev)?;
        gpu.copy_h2d(&wu_bytes, wu_dev)?;

        let qg = quantize_to_nvfp4(
            &DenseWeight { weight: wg_dev },
            n as usize,
            k as usize,
            &gpu,
            absmax_k,
            quantize_k,
            stream,
        )?;
        let qu = quantize_to_nvfp4(
            &DenseWeight { weight: wu_dev },
            n as usize,
            k as usize,
            &gpu,
            absmax_k,
            quantize_k,
            stream,
        )?;

        KernelLaunch::new(&gpu, kern)
            .grid([n.div_ceil(co_n), m.div_ceil(tile_m), 1])
            .block([threads, 1, 1])
            .arg_ptr(a_dev)
            .arg_ptr(qg.weight)
            .arg_ptr(qg.weight_scale)
            .arg_f32(qg.weight_scale_2)
            .arg_ptr(qu.weight)
            .arg_ptr(qu.weight_scale)
            .arg_f32(qu.weight_scale_2)
            .arg_ptr(out_dev)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)?;
        gpu.synchronize(stream)?;

        let mut out_raw = vec![0u8; out_bytes];
        let mut pg_raw = vec![0u8; (n * (k / 2)) as usize];
        let mut sg_raw = vec![0u8; (n * (k / 16)) as usize];
        let mut pu_raw = vec![0u8; (n * (k / 2)) as usize];
        let mut su_raw = vec![0u8; (n * (k / 16)) as usize];
        gpu.copy_d2h(out_dev, &mut out_raw)?;
        gpu.copy_d2h(qg.weight, &mut pg_raw)?;
        gpu.copy_d2h(qg.weight_scale, &mut sg_raw)?;
        gpu.copy_d2h(qu.weight, &mut pu_raw)?;
        gpu.copy_d2h(qu.weight_scale, &mut su_raw)?;

        let mut max_abs = 0.0f32;
        let mut max_rel_den = 1.0f32;
        for r in 0..m as usize {
            for col in 0..n as usize {
                let mut sum_g = 0.0f32;
                let mut sum_u = 0.0f32;
                for kk in 0..k as usize {
                    let a_val = a_host[r * k as usize + kk].to_f32();
                    let bg = pg_raw[col * (k as usize / 2) + (kk / 2)];
                    let ng = if (kk & 1) == 1 {
                        (bg >> 4) & 0x0f
                    } else {
                        bg & 0x0f
                    };
                    let scg = fp8_e4m3_to_f32(sg_raw[col * (k as usize / 16) + (kk / 16)]);
                    let wg_v = bf16::from_f32(E2M1_LUT[ng as usize] * scg).to_f32();
                    sum_g += a_val * wg_v;

                    let bu = pu_raw[col * (k as usize / 2) + (kk / 2)];
                    let nu = if (kk & 1) == 1 {
                        (bu >> 4) & 0x0f
                    } else {
                        bu & 0x0f
                    };
                    let scu = fp8_e4m3_to_f32(su_raw[col * (k as usize / 16) + (kk / 16)]);
                    let wu_v = bf16::from_f32(E2M1_LUT[nu as usize] * scu).to_f32();
                    sum_u += a_val * wu_v;
                }
                let gv = sum_g * qg.weight_scale_2;
                let uv = sum_u * qu.weight_scale_2;
                let cpu_v = bf16::from_f32((gv / (1.0 + (-gv).exp())) * uv).to_f32();
                let idx = r * n as usize + col;
                let got_v = bf16::from_le_bytes([out_raw[2 * idx], out_raw[2 * idx + 1]]).to_f32();
                assert!(
                    got_v.is_finite(),
                    "{sym} emitted non-finite output at {idx}"
                );
                max_abs = max_abs.max((cpu_v - got_v).abs());
                max_rel_den = max_rel_den.max(cpu_v.abs());
            }
        }
        let rel = max_abs / max_rel_den;
        eprintln!("PARITY FUSED {sym} (M={m}, N={n}, K={k}): max_abs={max_abs:.6}, rel={rel:.6}");
        assert!(
            rel <= 0.015 && max_abs <= 0.05,
            "{sym} drifted vs exact CPU SiLU*mul ref (max_abs={max_abs:.6}, rel={rel:.6})"
        );

        for ptr in [
            a_dev,
            wg_dev,
            wu_dev,
            out_dev,
            qg.weight,
            qg.weight_scale,
            qu.weight,
            qu.weight_scale,
        ] {
            let _ = gpu.free(ptr);
        }
    }
    Ok(())
}
