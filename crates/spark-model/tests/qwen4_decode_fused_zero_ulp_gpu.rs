// SPDX-License-Identifier: AGPL-3.0-only

//! Hardware `0 ULP` (byte-for-byte) parity verification on GB10 (`sm_121a`)
//! for the fused 48-CTA Qwen4 GDN decode/verify kernels and fused per-head
//! sigmoid output gates in QSA sparse reduce + BF16 paged decode attention.

#![cfg(all(feature = "cuda", target_os = "linux"))]

use spark_model::layers::ops;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn gen_f32_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            let u = ((xorshift64(&mut s) >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0;
            u * scale
        })
        .collect()
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding = 0x7FFF_u32.wrapping_add(lsb);
    (bits.wrapping_add(rounding) >> 16) as u16
}

fn gen_bf16_bytes(n: usize, seed: u64, scale: f32) -> Vec<u8> {
    gen_f32_vec(n, seed, scale)
        .into_iter()
        .flat_map(|v| f32_to_bf16_bits(v).to_le_bytes())
        .collect()
}

fn f32_slice_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_slice_bytes(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let ptr = gpu.alloc(bytes.len().max(16)).unwrap();
    if !bytes.is_empty() {
        gpu.copy_h2d(bytes, ptr).unwrap();
    }
    ptr
}

fn read_bytes(gpu: &dyn GpuBackend, ptr: DevicePtr, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    gpu.synchronize(gpu.default_stream()).unwrap();
    if len > 0 {
        gpu.copy_d2h(ptr, &mut out).unwrap();
    }
    out
}

#[test]
#[ignore = "needs an idle GB10 and the qwen3.8-flash-next kernel target"]
fn fused_qwen4_gdn_k1_lazyfinal_and_persistent_are_zero_ulp_on_gb10() {
    let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
    let stream = gpu.default_stream();

    let nk = 16usize;
    let kd = 128usize;
    let nv = 48usize;
    let vd = 128usize;
    let d_conv = 4usize;
    let key_dim = nk * kd;
    let value_dim = nv * vd;
    let conv_dim = key_dim * 2 + value_dim; // 10240
    let qkvz_size = conv_dim + value_dim; // 16384
    let qk_ch = (key_dim * 2) as u32;
    let eps = 1e-6f32;
    let h_elems = nv * kd * vd; // 48 * 128 * 128 = 786,432

    let conv_k = gpu
        .kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32")
        .unwrap();
    let conv_seq_k = gpu
        .kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32_sequence")
        .unwrap();
    let gdn_k = gpu
        .kernel("gated_delta_rule", "gated_delta_rule_decode_f32")
        .unwrap();
    let gdn_lazy_k = gpu
        .kernel(
            "gated_delta_rule",
            "gated_delta_rule_decode_f32_sequence_lazyfinal",
        )
        .unwrap();
    let gdn_pers_k = gpu
        .kernel(
            "gated_delta_rule",
            "gated_delta_rule_decode_f32_sequence_persistent",
        )
        .unwrap();
    let norm_k = gpu
        .kernel("qwen4_hyper", "qwen4_gated_rms_norm_sigmoid_f32")
        .unwrap();
    let norm_seq_k = gpu
        .kernel("qwen4_hyper", "qwen4_gated_rms_norm_sigmoid_f32_multi_seq")
        .unwrap();

    let fused_k1 = gpu
        .kernel("qwen4_gdn_verify_fused", "qwen4_gdn_decode_fused_k1")
        .unwrap();
    let fused_lazy = gpu
        .kernel("qwen4_gdn_verify_fused", "qwen4_gdn_verify_fused_lazyfinal")
        .unwrap();
    let fused_pers = gpu
        .kernel(
            "qwen4_gdn_verify_fused",
            "qwen4_gdn_verify_fused_persistent",
        )
        .unwrap();

    let conv_w = DenseWeight {
        weight: upload(&gpu, &gen_bf16_bytes(conv_dim * d_conv, 101, 0.25)),
    };
    let norm_w = DenseWeight {
        weight: upload(&gpu, &gen_bf16_bytes(vd, 202, 1.0)),
    };
    let group_done = gpu.alloc(16 * 4).unwrap();
    gpu.memset(group_done, 0, 16 * 4).unwrap();

    // 1. K=1 decode exact parity (2 consecutive steps to test group_done self-reset)
    let init_conv = f32_slice_bytes(&gen_f32_vec(conv_dim * d_conv, 303, 0.5));
    let init_h = f32_slice_bytes(&gen_f32_vec(h_elems, 404, 0.2));
    let conv_ref = upload(&gpu, &init_conv);
    let conv_fus = upload(&gpu, &init_conv);
    let h_ref = upload(&gpu, &init_h);
    let h_fus = upload(&gpu, &init_h);
    for step in 0..4u64 {
        let qkvz = upload(&gpu, &gen_bf16_bytes(qkvz_size, 505 + step, 0.8));
        let mut gates_host = gen_f32_vec(nv * 2, 606 + step, 1.0);
        for (h_idx, g) in gates_host[..nv].iter_mut().enumerate() {
            *g = match h_idx {
                0 => 0.0,
                1 => 1.0,
                _ => 0.05 + 0.945 * g.abs().clamp(0.0, 1.0),
            };
        }
        for b in &mut gates_host[nv..] {
            *b = b.abs().clamp(0.05, 0.95);
        }
        let gates = upload(&gpu, &f32_slice_bytes(&gates_host));
        let conv_out_ref = gpu.alloc(conv_dim * 4).unwrap();
        let gdn_out_ref = gpu.alloc(value_dim * 4).unwrap();
        let norm_out_ref = gpu.alloc(value_dim * 2).unwrap();
        let norm_out_fus = gpu.alloc(value_dim * 2).unwrap();

        ops::conv1d_update_l2norm(
            &gpu,
            conv_k,
            conv_ref,
            qkvz,
            &conv_w,
            conv_out_ref,
            conv_dim as u32,
            d_conv as u32,
            1,
            qk_ch,
            kd as u32,
            eps,
            stream,
        )
        .unwrap();
        ops::gdn_decode(
            &gpu,
            gdn_k,
            h_ref,
            conv_out_ref,
            conv_out_ref.offset(key_dim * 4),
            conv_out_ref.offset(key_dim * 2 * 4),
            gates,
            gates.offset(nv * 4),
            gdn_out_ref,
            1,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            stream,
        )
        .unwrap();
        ops::gated_rms_norm(
            &gpu,
            norm_k,
            gdn_out_ref,
            qkvz.offset(conv_dim * 2),
            &norm_w,
            norm_out_ref,
            nv as u32,
            vd as u32,
            vd as u32,
            eps,
            vd as u32,
            stream,
        )
        .unwrap();

        ops::qwen4_gdn_decode_fused_k1(
            &gpu,
            fused_k1,
            conv_fus,
            qkvz,
            &conv_w,
            gates,
            gates.offset(nv * 4),
            h_fus,
            &norm_w,
            norm_out_fus,
            group_done,
            nk as u32,
            nv as u32,
            eps,
            eps,
            stream,
        )
        .unwrap();

        assert_eq!(
            read_bytes(&gpu, norm_out_fus, value_dim * 2),
            read_bytes(&gpu, norm_out_ref, value_dim * 2),
            "K=1 normed_out mismatch at step {step}"
        );
        assert_eq!(
            read_bytes(&gpu, conv_fus, conv_dim * d_conv * 4),
            read_bytes(&gpu, conv_ref, conv_dim * d_conv * 4),
            "K=1 conv_state mismatch at step {step}"
        );
        assert_eq!(
            read_bytes(&gpu, h_fus, h_elems * 4),
            read_bytes(&gpu, h_ref, h_elems * 4),
            "K=1 h_state mismatch at step {step}"
        );
        assert_eq!(
            read_bytes(&gpu, group_done, 16 * 4),
            vec![0u8; 16 * 4],
            "K=1 group_done not reset to zero at step {step}"
        );
    }

    // 2. Multi-row verify exact parity: lazyfinal (with and without retain_qkv) & persistent
    for rows in [2usize, 3, 5, 8, 9, 16] {
        for mode in ["lazy_retain", "lazy_null", "persistent"] {
            let init_conv =
                f32_slice_bytes(&gen_f32_vec(conv_dim * d_conv, 700 + rows as u64, 0.5));
            let init_h = f32_slice_bytes(&gen_f32_vec(h_elems, 800 + rows as u64, 0.2));
            let qkvz = upload(
                &gpu,
                &gen_bf16_bytes(rows * qkvz_size, 900 + rows as u64, 0.8),
            );
            let mut gates_host = gen_f32_vec(rows * nv * 2, 1000 + rows as u64, 1.0);
            for r in 0..rows {
                for (h_idx, g) in gates_host[r * nv * 2..r * nv * 2 + nv]
                    .iter_mut()
                    .enumerate()
                {
                    *g = match h_idx {
                        0 => 0.0,
                        1 => 1.0,
                        _ => 0.05 + 0.945 * g.abs().clamp(0.0, 1.0),
                    };
                }
                for b in &mut gates_host[r * nv * 2 + nv..(r + 1) * nv * 2] {
                    *b = b.abs().clamp(0.05, 0.95);
                }
            }
            let gates = upload(&gpu, &f32_slice_bytes(&gates_host));

            let conv_ref = upload(&gpu, &init_conv);
            let conv_fus = upload(&gpu, &init_conv);
            let h_ref = upload(&gpu, &init_h);
            let h_fus = upload(&gpu, &init_h);

            let conv_inter_bytes = rows * conv_dim * d_conv * 4;
            let h_inter_bytes = rows * h_elems * 4;
            let conv_inter_ref = gpu.alloc(conv_inter_bytes).unwrap();
            let conv_inter_fus = gpu.alloc(conv_inter_bytes).unwrap();
            gpu.memset(conv_inter_ref, 0xA5, conv_inter_bytes).unwrap();
            gpu.memset(conv_inter_fus, 0xA5, conv_inter_bytes).unwrap();

            let h_inter_ref = gpu.alloc(h_inter_bytes).unwrap();
            let h_inter_fus = gpu.alloc(h_inter_bytes).unwrap();
            gpu.memset(h_inter_ref, 0x5A, h_inter_bytes).unwrap();
            gpu.memset(h_inter_fus, 0x5A, h_inter_bytes).unwrap();

            let conv_rows_ref = gpu.alloc(rows * qkvz_size * 4).unwrap();
            let gdn_rows_ref = gpu.alloc(rows * qkvz_size * 4).unwrap();
            let retain_fus = gpu.alloc(rows * qkvz_size * 4).unwrap();
            gpu.memset(conv_rows_ref, 0, rows * qkvz_size * 4).unwrap();
            gpu.memset(retain_fus, 0, rows * qkvz_size * 4).unwrap();

            let norm_ref = gpu.alloc(rows * value_dim * 2).unwrap();
            let norm_fus = gpu.alloc(rows * value_dim * 2).unwrap();

            ops::conv1d_update_l2norm_f32_sequence(
                &gpu,
                conv_seq_k,
                conv_ref,
                qkvz,
                &conv_w,
                conv_rows_ref,
                conv_inter_ref,
                rows as u32,
                conv_dim as u32,
                d_conv as u32,
                qk_ch,
                kd as u32,
                eps,
                qkvz_size as u32,
                qkvz_size as u32,
                (conv_dim * d_conv) as u32,
                stream,
            )
            .unwrap();

            let ref_gdn_k = if mode == "persistent" {
                gdn_pers_k
            } else {
                gdn_lazy_k
            };
            ops::gdn_decode_f32_sequence_persistent(
                &gpu,
                ref_gdn_k,
                h_ref,
                conv_rows_ref,
                conv_rows_ref.offset(key_dim * 4),
                conv_rows_ref.offset(key_dim * 2 * 4),
                gates,
                gates.offset(nv * 4),
                gdn_rows_ref,
                h_inter_ref,
                rows as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
                h_elems as u32,
                stream,
            )
            .unwrap();

            ops::gated_rms_norm_f32_multi_seq(
                &gpu,
                norm_seq_k,
                gdn_rows_ref,
                qkvz.offset(conv_dim * 2),
                &norm_w,
                norm_ref,
                nv as u32,
                rows as u32,
                vd as u32,
                eps,
                qkvz_size as u32,
                qkvz_size as u32,
                value_dim as u32,
                stream,
            )
            .unwrap();

            if mode == "persistent" {
                ops::qwen4_gdn_verify_fused_persistent(
                    &gpu,
                    fused_pers,
                    conv_fus,
                    qkvz,
                    &conv_w,
                    gates,
                    gates.offset(nv * 4),
                    h_fus,
                    &norm_w,
                    norm_fus,
                    conv_inter_fus,
                    h_inter_fus,
                    group_done,
                    rows as u32,
                    nk as u32,
                    nv as u32,
                    eps,
                    eps,
                    qkvz_size as u32,
                    (nv * 2) as u32,
                    value_dim as u32,
                    (conv_dim * d_conv) as u32,
                    h_elems as u32,
                    stream,
                )
                .unwrap();
            } else {
                let retain_arg = if mode == "lazy_retain" {
                    retain_fus
                } else {
                    DevicePtr::NULL
                };
                ops::qwen4_gdn_verify_fused_lazyfinal(
                    &gpu,
                    fused_lazy,
                    conv_fus,
                    qkvz,
                    &conv_w,
                    gates,
                    gates.offset(nv * 4),
                    h_fus,
                    &norm_w,
                    norm_fus,
                    retain_arg,
                    conv_inter_fus,
                    h_inter_fus,
                    group_done,
                    rows as u32,
                    nk as u32,
                    nv as u32,
                    eps,
                    eps,
                    qkvz_size as u32,
                    (nv * 2) as u32,
                    value_dim as u32,
                    qkvz_size as u32,
                    (conv_dim * d_conv) as u32,
                    h_elems as u32,
                    stream,
                )
                .unwrap();
            }

            assert_eq!(
                read_bytes(&gpu, norm_fus, rows * value_dim * 2),
                read_bytes(&gpu, norm_ref, rows * value_dim * 2),
                "normed_out mismatch (rows={rows}, mode={mode})"
            );
            assert_eq!(
                read_bytes(&gpu, conv_fus, conv_dim * d_conv * 4),
                read_bytes(&gpu, conv_ref, conv_dim * d_conv * 4),
                "conv_state mismatch (rows={rows}, mode={mode})"
            );
            // Both reference and fused write conv_inter for rows 0..rows
            assert_eq!(
                read_bytes(&gpu, conv_inter_fus, conv_inter_bytes),
                read_bytes(&gpu, conv_inter_ref, conv_inter_bytes),
                "conv_inter mismatch (rows={rows}, mode={mode})"
            );
            assert_eq!(
                read_bytes(&gpu, h_fus, h_elems * 4),
                read_bytes(&gpu, h_ref, h_elems * 4),
                "h_state mismatch (rows={rows}, mode={mode})"
            );
            assert_eq!(
                read_bytes(&gpu, h_inter_fus, h_inter_bytes),
                read_bytes(&gpu, h_inter_ref, h_inter_bytes),
                "h_inter mismatch (rows={rows}, mode={mode})"
            );
            if mode == "lazy_retain" {
                for r in 0..rows {
                    assert_eq!(
                        read_bytes(&gpu, retain_fus.offset(r * qkvz_size * 4), conv_dim * 4),
                        read_bytes(&gpu, conv_rows_ref.offset(r * qkvz_size * 4), conv_dim * 4),
                        "retain_qkv mismatch at row {r} (rows={rows})"
                    );
                }
            }
            assert_eq!(
                read_bytes(&gpu, group_done, 16 * 4),
                vec![0u8; 16 * 4],
                "group_done not reset to zero (rows={rows}, mode={mode})"
            );
        }
    }
}

#[test]
#[ignore = "needs an idle GB10 and the qwen3.8-flash-next kernel target"]
fn fused_qwen4_qsa_sparse_reduce_and_paged_decode_gates_are_zero_ulp_on_gb10() {
    let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
    let stream = gpu.default_stream();

    let nq = 24usize;
    let nkv = 2usize;
    let hd = 256usize;
    let q_dim = nq * hd; // 6144
    let splits = 16usize;

    let sig_k = gpu.kernel("residual_add", "sigmoid_gate_mul").unwrap();
    let sig_batch_k = gpu
        .kernel("residual_add", "sigmoid_gate_mul_batched")
        .unwrap();
    let qsa_red_k = gpu
        .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_bf16_reduce")
        .unwrap();
    let qsa_red_gated_k = gpu
        .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_bf16_reduce_gated")
        .unwrap();
    let qsa_red_rows_k = gpu
        .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_bf16_reduce_rows")
        .unwrap();
    let qsa_red_gated_rows_k = gpu
        .kernel(
            "qwen4_qsa",
            "qwen4_qsa_sparse_attention_bf16_reduce_gated_rows",
        )
        .unwrap();
    let paged_k = gpu.kernel("paged_decode", "paged_decode_attn").unwrap();
    let paged_gated_k = gpu
        .kernel("paged_decode", "paged_decode_attn_gated")
        .unwrap();

    // 1. QSA sparse reduce single-row + multi-row gated 0-ULP parity
    for rows in [1usize, 3, 5, 9] {
        let p_out = upload(
            &gpu,
            &f32_slice_bytes(&gen_f32_vec(
                rows * nq * splits * hd,
                2100 + rows as u64,
                1.2,
            )),
        );
        let p_max = upload(
            &gpu,
            &f32_slice_bytes(&gen_f32_vec(rows * nq * splits, 2200 + rows as u64, 2.0)),
        );
        let mut sums = gen_f32_vec(rows * nq * splits, 2300 + rows as u64, 1.0);
        for s in &mut sums {
            *s = s.abs() + 0.1;
        }
        let p_sum = upload(&gpu, &f32_slice_bytes(&sums));
        // Interleaved Q+Gate layout: stride = 2 * q_dim, gate starts at +q_dim
        let qg_stride = 2 * q_dim;
        let qg_buf = upload(
            &gpu,
            &gen_bf16_bytes(rows * qg_stride, 2400 + rows as u64, 1.5),
        );
        let gate_ptr = qg_buf.offset(q_dim * 2);

        let out_ref = gpu.alloc(rows * q_dim * 2).unwrap();
        let out_fus = gpu.alloc(rows * q_dim * 2).unwrap();

        if rows == 1 {
            KernelLaunch::new(&gpu, qsa_red_k)
                .grid([nq as u32, 1, 1])
                .block([hd as u32, 1, 1])
                .arg_ptr(p_out)
                .arg_ptr(p_max)
                .arg_ptr(p_sum)
                .arg_ptr(out_ref)
                .arg_u32(nq as u32)
                .arg_u32(hd as u32)
                .arg_u32(splits as u32)
                .launch(stream)
                .unwrap();
            ops::sigmoid_gate_mul(
                &gpu,
                sig_k,
                out_ref,
                gate_ptr,
                out_ref,
                q_dim as u32,
                stream,
            )
            .unwrap();

            KernelLaunch::new(&gpu, qsa_red_gated_k)
                .grid([nq as u32, 1, 1])
                .block([hd as u32, 1, 1])
                .pdl(true)
                .arg_ptr(p_out)
                .arg_ptr(p_max)
                .arg_ptr(p_sum)
                .arg_ptr(gate_ptr)
                .arg_ptr(out_fus)
                .arg_u32(nq as u32)
                .arg_u32(hd as u32)
                .arg_u32(splits as u32)
                .launch(stream)
                .unwrap();

            assert_eq!(
                read_bytes(&gpu, out_fus, q_dim * 2),
                read_bytes(&gpu, out_ref, q_dim * 2),
                "single-row QSA gated reduce mismatch"
            );
        }

        KernelLaunch::new(&gpu, qsa_red_rows_k)
            .grid([nq as u32, rows as u32, 1])
            .block([hd as u32, 1, 1])
            .arg_ptr(p_out)
            .arg_ptr(p_max)
            .arg_ptr(p_sum)
            .arg_ptr(out_ref)
            .arg_u32(nq as u32)
            .arg_u32(hd as u32)
            .arg_u32(splits as u32)
            .launch(stream)
            .unwrap();
        ops::sigmoid_gate_mul_batched(
            &gpu,
            sig_batch_k,
            out_ref,
            gate_ptr,
            out_ref,
            q_dim as u32,
            qg_stride as u32,
            rows as u32,
            stream,
        )
        .unwrap();

        KernelLaunch::new(&gpu, qsa_red_gated_rows_k)
            .grid([nq as u32, rows as u32, 1])
            .block([hd as u32, 1, 1])
            .pdl(true)
            .arg_ptr(p_out)
            .arg_ptr(p_max)
            .arg_ptr(p_sum)
            .arg_ptr(gate_ptr)
            .arg_ptr(out_fus)
            .arg_u32(nq as u32)
            .arg_u32(hd as u32)
            .arg_u32(splits as u32)
            .arg_u32(qg_stride as u32)
            .launch(stream)
            .unwrap();

        assert_eq!(
            read_bytes(&gpu, out_fus, rows * q_dim * 2),
            read_bytes(&gpu, out_ref, rows * q_dim * 2),
            "multi-row QSA gated reduce mismatch (rows={rows})"
        );
    }

    // 2. BF16 paged decode gated 0-ULP parity across rows = 1, 3, 5, 9
    let block_size = 16usize;
    let num_blocks = 8usize;
    let kv_elems = num_blocks * block_size * nkv * hd;
    let k_cache = upload(&gpu, &gen_bf16_bytes(kv_elems, 3100, 0.7));
    let v_cache = upload(&gpu, &gen_bf16_bytes(kv_elems, 3200, 0.7));
    let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();

    for rows in [1usize, 3, 5, 9] {
        let max_blocks_per_seq = num_blocks;
        let mut bt = Vec::with_capacity(rows * max_blocks_per_seq);
        let mut lens = Vec::with_capacity(rows);
        for r in 0..rows {
            for b in 0..max_blocks_per_seq {
                bt.push(b as u32);
            }
            lens.push((64 + r) as u32);
        }
        let bt_dev = upload(&gpu, &u32_slice_bytes(&bt));
        let lens_dev = upload(&gpu, &u32_slice_bytes(&lens));

        let qg_stride = 2 * q_dim;
        let qg_buf = upload(
            &gpu,
            &gen_bf16_bytes(rows * qg_stride, 3300 + rows as u64, 0.9),
        );
        let gate_ptr = qg_buf.offset(q_dim * 2);

        let out_ref = gpu.alloc(rows * q_dim * 2).unwrap();
        let out_fus = gpu.alloc(rows * q_dim * 2).unwrap();

        ops::paged_decode_attn_bf16(
            &gpu,
            paged_k,
            qg_buf,
            k_cache,
            v_cache,
            out_ref,
            bt_dev,
            lens_dev,
            max_blocks_per_seq as u32,
            rows as u32,
            nq as u32,
            nkv as u32,
            hd as u32,
            block_size as u32,
            inv_sqrt_d,
            qg_stride as u32,
            0,
            stream,
        )
        .unwrap();
        ops::sigmoid_gate_mul_batched(
            &gpu,
            sig_batch_k,
            out_ref,
            gate_ptr,
            out_ref,
            q_dim as u32,
            qg_stride as u32,
            rows as u32,
            stream,
        )
        .unwrap();

        ops::paged_decode_attn_bf16_gated(
            &gpu,
            paged_gated_k,
            qg_buf,
            k_cache,
            v_cache,
            gate_ptr,
            out_fus,
            bt_dev,
            lens_dev,
            max_blocks_per_seq as u32,
            rows as u32,
            nq as u32,
            nkv as u32,
            hd as u32,
            block_size as u32,
            inv_sqrt_d,
            qg_stride as u32,
            qg_stride as u32,
            0,
            stream,
        )
        .unwrap();

        assert_eq!(
            read_bytes(&gpu, out_fus, rows * q_dim * 2),
            read_bytes(&gpu, out_ref, rows * q_dim * 2),
            "paged_decode_attn_gated mismatch (rows={rows})"
        );
    }
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bf16_ordered(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -((bits & 0x7fff) as i32)
    } else {
        bits as i32
    }
}

fn max_abs_diff_f32_bytes(lhs: &[u8], rhs: &[u8]) -> (f32, usize) {
    let a = bytes_to_f32_vec(lhs);
    let b = bytes_to_f32_vec(rhs);
    assert_eq!(a.len(), b.len());
    let mut max_err = 0.0f32;
    let mut worst_idx = 0usize;
    for (idx, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.is_finite() && y.is_finite(),
            "non-finite f32 at idx {idx}: lhs={x}, rhs={y}"
        );
        let diff = (x - y).abs();
        if diff > max_err {
            max_err = diff;
            worst_idx = idx;
        }
    }
    (max_err, worst_idx)
}

fn max_bf16_ulp_and_abs_diff(lhs: &[u8], rhs: &[u8]) -> (u16, f32) {
    assert_eq!(lhs.len(), rhs.len());
    let mut max_ulp = 0u16;
    let mut max_abs = 0.0f32;
    for (idx, (ca, cb)) in lhs.chunks_exact(2).zip(rhs.chunks_exact(2)).enumerate() {
        let a = u16::from_le_bytes([ca[0], ca[1]]);
        let b = u16::from_le_bytes([cb[0], cb[1]]);
        let fa = bf16_bits_to_f32(a);
        let fb = bf16_bits_to_f32(b);
        assert!(
            fa.is_finite() && fb.is_finite(),
            "non-finite bf16 at idx {idx}: lhs=0x{a:04x} ({fa}), rhs=0x{b:04x} ({fb})"
        );
        max_abs = max_abs.max((fa - fb).abs());
        let ulp = (bf16_ordered(a) - bf16_ordered(b))
            .unsigned_abs()
            .min(u32::from(u16::MAX)) as u16;
        max_ulp = max_ulp.max(ulp);
    }
    (max_ulp, max_abs)
}

#[test]
#[ignore = "needs an idle GB10 and the qwen3.8-flash-next kernel target"]
fn qwen4_gdn_splitv4_parity_on_gb10() {
    let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
    let stream = gpu.default_stream();

    let nk = 16usize;
    let kd = 128usize;
    let nv = 48usize;
    let vd = 128usize;
    let d_conv = 4usize;
    let key_dim = nk * kd;
    let value_dim = nv * vd;
    let conv_dim = key_dim * 2 + value_dim; // 10240
    let qkvz_size = conv_dim + value_dim; // 16384
    let eps = 1e-6f32;
    let h_elems = nv * kd * vd;

    let base_k1 = gpu
        .kernel("qwen4_gdn_verify_fused", "qwen4_gdn_decode_fused_k1")
        .unwrap();
    let base_lazy = gpu
        .kernel("qwen4_gdn_verify_fused", "qwen4_gdn_verify_fused_lazyfinal")
        .unwrap();
    let base_pers = gpu
        .kernel(
            "qwen4_gdn_verify_fused",
            "qwen4_gdn_verify_fused_persistent",
        )
        .unwrap();
    let base_multi_lazy = gpu
        .kernel(
            "qwen4_gdn_verify_fused",
            "qwen4_gdn_verify_fused_lazyfinal_multi_seq",
        )
        .unwrap();

    let sv4_k1 = gpu
        .kernel(
            "qwen4_gdn_verify_fused",
            "qwen4_gdn_decode_fused_k1_splitv4",
        )
        .unwrap();
    let sv4_lazy = gpu
        .kernel(
            "qwen4_gdn_verify_fused",
            "qwen4_gdn_verify_fused_lazyfinal_splitv4",
        )
        .unwrap();
    let sv4_pers = gpu
        .kernel(
            "qwen4_gdn_verify_fused",
            "qwen4_gdn_verify_fused_persistent_splitv4",
        )
        .unwrap();
    let sv4_multi_lazy = gpu
        .kernel(
            "qwen4_gdn_verify_fused",
            "qwen4_gdn_verify_fused_lazyfinal_multi_seq_splitv4",
        )
        .unwrap();

    let conv_w = DenseWeight {
        weight: upload(&gpu, &gen_bf16_bytes(conv_dim * d_conv, 1101, 0.25)),
    };
    let norm_w = DenseWeight {
        weight: upload(&gpu, &gen_bf16_bytes(vd, 1202, 1.0)),
    };
    let group_done_base = gpu.alloc(32 * 4).unwrap();
    let group_done_sv4 = gpu.alloc(32 * 4).unwrap();
    let split_done = gpu.alloc(nv * 4).unwrap();
    gpu.memset(group_done_base, 0, 32 * 4).unwrap();
    gpu.memset(group_done_sv4, 0, 32 * 4).unwrap();
    gpu.memset(split_done, 0, nv * 4).unwrap();

    let max_rows = 8usize;
    let scratch_layout = ops::gdn_split_v_scratch_layout(max_rows, nv).unwrap();
    let scratch_gdn = gpu.alloc(scratch_layout.total_bytes).unwrap();
    let scratch_ss = scratch_gdn.offset(scratch_layout.ss_offset_bytes);

    let make_gates = |rows: usize, seed: u64| -> Vec<f32> {
        let mut gates_host = gen_f32_vec(rows * nv * 2, seed, 1.0);
        for r in 0..rows {
            for (h_idx, g) in gates_host[r * nv * 2..r * nv * 2 + nv]
                .iter_mut()
                .enumerate()
            {
                *g = match h_idx {
                    0 => 0.0,
                    1 => 1.0,
                    _ => 0.05 + 0.945 * g.abs().clamp(0.0, 1.0),
                };
            }
            for b in &mut gates_host[r * nv * 2 + nv..(r + 1) * nv * 2] {
                *b = b.abs().clamp(0.05, 0.95);
            }
        }
        gates_host
    };

    // 1. K=1 Split-V vs baseline over 4 consecutive steps + 0-ULP self-consistency vs persistent_splitv4(K=4)
    let init_conv = f32_slice_bytes(&gen_f32_vec(conv_dim * d_conv, 1303, 0.5));
    let init_h = f32_slice_bytes(&gen_f32_vec(h_elems, 1404, 0.2));
    let conv_base = upload(&gpu, &init_conv);
    let conv_sv4 = upload(&gpu, &init_conv);
    let h_base = upload(&gpu, &init_h);
    let h_sv4 = upload(&gpu, &init_h);
    let mut seq_qkvz_bytes = Vec::with_capacity(4 * qkvz_size * 2);
    let mut seq_gates_vals = Vec::with_capacity(4 * nv * 2);
    let mut k1_sv4_normed_all = Vec::with_capacity(4 * value_dim * 2);

    for step in 0..4usize {
        let qkvz_host = gen_bf16_bytes(qkvz_size, 1505 + step as u64, 0.8);
        let gates_host = make_gates(1, 1606 + step as u64);
        seq_qkvz_bytes.extend_from_slice(&qkvz_host);
        seq_gates_vals.extend_from_slice(&gates_host);
        let qkvz = upload(&gpu, &qkvz_host);
        let gates = upload(&gpu, &f32_slice_bytes(&gates_host));
        let norm_base = gpu.alloc(value_dim * 2).unwrap();
        let norm_sv4 = gpu.alloc(value_dim * 2).unwrap();

        ops::qwen4_gdn_decode_fused_k1(
            &gpu,
            base_k1,
            conv_base,
            qkvz,
            &conv_w,
            gates,
            gates.offset(nv * 4),
            h_base,
            &norm_w,
            norm_base,
            group_done_base,
            nk as u32,
            nv as u32,
            eps,
            eps,
            stream,
        )
        .unwrap();
        ops::qwen4_gdn_decode_fused_k1_splitv4(
            &gpu,
            sv4_k1,
            conv_sv4,
            qkvz,
            &conv_w,
            gates,
            gates.offset(nv * 4),
            h_sv4,
            &norm_w,
            norm_sv4,
            group_done_sv4,
            nk as u32,
            nv as u32,
            eps,
            eps,
            scratch_gdn,
            scratch_ss,
            split_done,
            stream,
        )
        .unwrap();

        let norm_base_b = read_bytes(&gpu, norm_base, value_dim * 2);
        let norm_sv4_b = read_bytes(&gpu, norm_sv4, value_dim * 2);
        k1_sv4_normed_all.extend_from_slice(&norm_sv4_b);
        let (max_ulp, max_norm_abs) = max_bf16_ulp_and_abs_diff(&norm_sv4_b, &norm_base_b);
        let h_sv4_b = read_bytes(&gpu, h_sv4, h_elems * 4);
        let h_base_b = read_bytes(&gpu, h_base, h_elems * 4);
        let (h_err, h_idx) = max_abs_diff_f32_bytes(&h_sv4_b, &h_base_b);
        eprintln!(
            "Split-V K=1 step={step}: h_max_abs={h_err:.3e} (idx={h_idx}, vh={}, k={}, v={}), norm_bf16_max_ulp={max_ulp}, norm_max_abs={max_norm_abs:.3e}",
            h_idx / (kd * vd),
            (h_idx % (kd * vd)) / vd,
            h_idx % vd,
        );
        if h_err > 1e-6 {
            let ha = bytes_to_f32_vec(&h_sv4_b);
            let hb = bytes_to_f32_vec(&h_base_b);
            let mut count = 0usize;
            for (i, (&x, &y)) in ha.iter().zip(hb.iter()).enumerate() {
                if (x - y).abs() > 1e-6 {
                    if count < 16 {
                        eprintln!(
                            "  diff idx={i} (vh={}, k={}, v={}): sv4={x:.7} base={y:.7} diff={:.3e}",
                            i / (kd * vd),
                            (i % (kd * vd)) / vd,
                            i % vd,
                            (x - y).abs()
                        );
                    }
                    count += 1;
                }
            }
            eprintln!("  total elements with diff > 1e-6: {count}");
        }
        assert_eq!(
            read_bytes(&gpu, conv_sv4, conv_dim * d_conv * 4),
            read_bytes(&gpu, conv_base, conv_dim * d_conv * 4),
            "K=1 Split-V conv_state must be 0-ULP identical at step {step}"
        );
        assert!(h_err <= 1e-6, "K=1 Split-V h_state err {h_err} > 1e-6");
        assert!(
            max_ulp <= 1 && max_norm_abs <= 1e-3,
            "K=1 Split-V normed_out BF16 ULP {max_ulp} > 1 or norm_abs {max_norm_abs} > 1e-3"
        );
        assert_eq!(
            read_bytes(&gpu, group_done_sv4, 32 * 4),
            vec![0u8; 32 * 4],
            "K=1 Split-V group_done not reset at step {step}"
        );
        assert_eq!(
            read_bytes(&gpu, split_done, nv * 4),
            vec![0u8; nv * 4],
            "K=1 Split-V split_done not reset at step {step}"
        );
    }

    // Verify 0-ULP identity between 4x K=1 Split-V and 1x persistent_splitv4(K=4)
    {
        let rows = 4usize;
        let qkvz_seq = upload(&gpu, &seq_qkvz_bytes);
        let gates_seq = upload(&gpu, &f32_slice_bytes(&seq_gates_vals));
        let conv_p = upload(&gpu, &init_conv);
        let h_p = upload(&gpu, &init_h);
        let conv_inter_p = gpu.alloc(rows * conv_dim * d_conv * 4).unwrap();
        let h_inter_p = gpu.alloc(rows * h_elems * 4).unwrap();
        let norm_p = gpu.alloc(rows * value_dim * 2).unwrap();
        ops::qwen4_gdn_verify_fused_persistent_splitv4(
            &gpu,
            sv4_pers,
            conv_p,
            qkvz_seq,
            &conv_w,
            gates_seq,
            gates_seq.offset(nv * 4),
            h_p,
            &norm_w,
            norm_p,
            conv_inter_p,
            h_inter_p,
            group_done_sv4,
            rows as u32,
            nk as u32,
            nv as u32,
            eps,
            eps,
            qkvz_size as u32,
            (nv * 2) as u32,
            value_dim as u32,
            (conv_dim * d_conv) as u32,
            h_elems as u32,
            scratch_gdn,
            scratch_ss,
            split_done,
            stream,
        )
        .unwrap();
        assert_eq!(
            read_bytes(&gpu, norm_p, rows * value_dim * 2),
            k1_sv4_normed_all,
            "persistent_splitv4(K=4) must be 0-ULP identical to 4x k1_splitv4"
        );
        assert_eq!(
            read_bytes(&gpu, h_p, h_elems * 4),
            read_bytes(&gpu, h_sv4, h_elems * 4),
            "persistent_splitv4(K=4) final H must be 0-ULP identical to 4x k1_splitv4"
        );
        assert_eq!(
            read_bytes(&gpu, conv_p, conv_dim * d_conv * 4),
            read_bytes(&gpu, conv_sv4, conv_dim * d_conv * 4),
            "persistent_splitv4(K=4) conv_state must be 0-ULP identical to 4x k1_splitv4"
        );
        assert_eq!(read_bytes(&gpu, group_done_sv4, 32 * 4), vec![0u8; 32 * 4]);
        assert_eq!(read_bytes(&gpu, split_done, nv * 4), vec![0u8; nv * 4]);
    }

    // 2. K=8 lazyfinal_splitv4 & persistent_splitv4 vs baseline
    for mode in ["lazy_retain", "persistent"] {
        let rows = 8usize;
        let qkvz = upload(&gpu, &gen_bf16_bytes(rows * qkvz_size, 2100, 0.8));
        let gates = upload(&gpu, &f32_slice_bytes(&make_gates(rows, 2200)));
        let conv_b = upload(&gpu, &init_conv);
        let conv_s = upload(&gpu, &init_conv);
        let h_b = upload(&gpu, &init_h);
        let h_s = upload(&gpu, &init_h);
        let conv_inter_bytes = rows * conv_dim * d_conv * 4;
        let h_inter_bytes = rows * h_elems * 4;
        let conv_inter_b = gpu.alloc(conv_inter_bytes).unwrap();
        let conv_inter_s = gpu.alloc(conv_inter_bytes).unwrap();
        let h_inter_b = gpu.alloc(h_inter_bytes).unwrap();
        let h_inter_s = gpu.alloc(h_inter_bytes).unwrap();
        gpu.memset(h_inter_b, 0x5A, h_inter_bytes).unwrap();
        gpu.memset(h_inter_s, 0x5A, h_inter_bytes).unwrap();
        let retain_b = gpu.alloc(rows * qkvz_size * 4).unwrap();
        let retain_s = gpu.alloc(rows * qkvz_size * 4).unwrap();
        gpu.memset(retain_b, 0, rows * qkvz_size * 4).unwrap();
        gpu.memset(retain_s, 0, rows * qkvz_size * 4).unwrap();
        let norm_b = gpu.alloc(rows * value_dim * 2).unwrap();
        let norm_s = gpu.alloc(rows * value_dim * 2).unwrap();

        if mode == "persistent" {
            ops::qwen4_gdn_verify_fused_persistent(
                &gpu,
                base_pers,
                conv_b,
                qkvz,
                &conv_w,
                gates,
                gates.offset(nv * 4),
                h_b,
                &norm_w,
                norm_b,
                conv_inter_b,
                h_inter_b,
                group_done_base,
                rows as u32,
                nk as u32,
                nv as u32,
                eps,
                eps,
                qkvz_size as u32,
                (nv * 2) as u32,
                value_dim as u32,
                (conv_dim * d_conv) as u32,
                h_elems as u32,
                stream,
            )
            .unwrap();
            ops::qwen4_gdn_verify_fused_persistent_splitv4(
                &gpu,
                sv4_pers,
                conv_s,
                qkvz,
                &conv_w,
                gates,
                gates.offset(nv * 4),
                h_s,
                &norm_w,
                norm_s,
                conv_inter_s,
                h_inter_s,
                group_done_sv4,
                rows as u32,
                nk as u32,
                nv as u32,
                eps,
                eps,
                qkvz_size as u32,
                (nv * 2) as u32,
                value_dim as u32,
                (conv_dim * d_conv) as u32,
                h_elems as u32,
                scratch_gdn,
                scratch_ss,
                split_done,
                stream,
            )
            .unwrap();
        } else {
            ops::qwen4_gdn_verify_fused_lazyfinal(
                &gpu,
                base_lazy,
                conv_b,
                qkvz,
                &conv_w,
                gates,
                gates.offset(nv * 4),
                h_b,
                &norm_w,
                norm_b,
                retain_b,
                conv_inter_b,
                h_inter_b,
                group_done_base,
                rows as u32,
                nk as u32,
                nv as u32,
                eps,
                eps,
                qkvz_size as u32,
                (nv * 2) as u32,
                value_dim as u32,
                qkvz_size as u32,
                (conv_dim * d_conv) as u32,
                h_elems as u32,
                stream,
            )
            .unwrap();
            ops::qwen4_gdn_verify_fused_lazyfinal_splitv4(
                &gpu,
                sv4_lazy,
                conv_s,
                qkvz,
                &conv_w,
                gates,
                gates.offset(nv * 4),
                h_s,
                &norm_w,
                norm_s,
                retain_s,
                conv_inter_s,
                h_inter_s,
                group_done_sv4,
                rows as u32,
                nk as u32,
                nv as u32,
                eps,
                eps,
                qkvz_size as u32,
                (nv * 2) as u32,
                value_dim as u32,
                qkvz_size as u32,
                (conv_dim * d_conv) as u32,
                h_elems as u32,
                scratch_gdn,
                scratch_ss,
                split_done,
                stream,
            )
            .unwrap();
        }

        assert_eq!(
            read_bytes(&gpu, conv_s, conv_dim * d_conv * 4),
            read_bytes(&gpu, conv_b, conv_dim * d_conv * 4),
            "K=8 Split-V conv_state mismatch ({mode})"
        );
        assert_eq!(
            read_bytes(&gpu, conv_inter_s, conv_inter_bytes),
            read_bytes(&gpu, conv_inter_b, conv_inter_bytes),
            "K=8 Split-V conv_inter mismatch ({mode})"
        );
        if mode == "lazy_retain" {
            assert_eq!(
                read_bytes(&gpu, retain_s, rows * qkvz_size * 4),
                read_bytes(&gpu, retain_b, rows * qkvz_size * 4),
                "K=8 Split-V retain_qkv must be 0-ULP identical"
            );
            assert_eq!(
                read_bytes(&gpu, h_s, h_elems * 4),
                init_h,
                "K=8 lazyfinal Split-V must not mutate h_state"
            );
        }
        let (h_err, _) = max_abs_diff_f32_bytes(
            &read_bytes(&gpu, h_s, h_elems * 4),
            &read_bytes(&gpu, h_b, h_elems * 4),
        );
        let (hi_err, _) = max_abs_diff_f32_bytes(
            &read_bytes(&gpu, h_inter_s, h_inter_bytes),
            &read_bytes(&gpu, h_inter_b, h_inter_bytes),
        );
        let (max_ulp, max_norm_abs) = max_bf16_ulp_and_abs_diff(
            &read_bytes(&gpu, norm_s, rows * value_dim * 2),
            &read_bytes(&gpu, norm_b, rows * value_dim * 2),
        );
        eprintln!(
            "Split-V K=8 mode={mode}: h_err={h_err:.3e}, h_inter_err={hi_err:.3e}, norm_bf16_ulp={max_ulp}, norm_abs={max_norm_abs:.3e}"
        );
        assert!(h_err <= 1e-6, "K=8 {mode} h_err {h_err} > 1e-6");
        assert!(hi_err <= 1e-6, "K=8 {mode} h_inter_err {hi_err} > 1e-6");
        assert!(
            max_norm_abs <= 1e-3,
            "K=8 {mode} norm_abs {max_norm_abs} > 1e-3 (max_ulp={max_ulp})"
        );
        assert_eq!(read_bytes(&gpu, group_done_sv4, 32 * 4), vec![0u8; 32 * 4]);
        assert_eq!(read_bytes(&gpu, split_done, nv * 4), vec![0u8; nv * 4]);
    }

    // 3. C=1 KEEP_LATE circular ring multi_seq_splitv4: step 1 (flush=0) + step 2 (pending=5, flush=1)
    {
        let rows = 8usize;
        let ring_cap = 32usize;
        let ring_row_bytes = ((conv_dim * 2 + nv * 8) + 15) & !15;
        let ring_bytes = ring_cap * ring_row_bytes;
        let ring_b = gpu.alloc(ring_bytes).unwrap();
        let ring_s = gpu.alloc(ring_bytes).unwrap();
        gpu.memset(ring_b, 0, ring_bytes).unwrap();
        gpu.memset(ring_s, 0, ring_bytes).unwrap();

        let conv_b = upload(&gpu, &init_conv);
        let conv_s = upload(&gpu, &init_conv);
        let h_b = upload(&gpu, &init_h);
        let h_s = upload(&gpu, &init_h);

        let ctrl_b_dev = gpu.alloc(ops::GdnKeepLateCtrl::SIZE_BYTES).unwrap();
        let ctrl_s_dev = gpu.alloc(ops::GdnKeepLateCtrl::SIZE_BYTES).unwrap();
        let desc_b_dev = gpu.alloc(ops::GdnFusedMultiSeqDesc::SIZE_BYTES).unwrap();
        let desc_s_dev = gpu.alloc(ops::GdnFusedMultiSeqDesc::SIZE_BYTES).unwrap();
        let flags = ops::GDN_FUSED_FLAG_KEEP_LATE | ops::GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR;

        let upload_ctrl_and_desc = |ctrl_dev: DevicePtr,
                                    desc_dev: DevicePtr,
                                    ring: DevicePtr,
                                    conv: DevicePtr,
                                    h: DevicePtr,
                                    gdone: DevicePtr,
                                    pending_base: u32,
                                    pending_rows: u32,
                                    write_base: u32,
                                    flush_now: u32| {
            let ctrl = ops::GdnKeepLateCtrl {
                ring_base: ring.0,
                pending_base,
                pending_rows,
                write_base,
                executed_flag: 0,
                ring_cap: ring_cap as u32,
                flush_now,
            };
            let mut cbytes = [0u8; ops::GdnKeepLateCtrl::SIZE_BYTES];
            ctrl.write_le_bytes(&mut cbytes);
            gpu.copy_h2d(&cbytes, ctrl_dev).unwrap();

            let desc = ops::GdnFusedMultiSeqDesc {
                conv_state: conv.0,
                h_state: h.0,
                retain_qkv: ring.0,
                conv_inter: 0,
                h_state_inter: ctrl_dev.0,
                group_done: gdone.0,
                row_base: 0,
                num_tokens: rows as u32,
                flags,
                _pad: 0,
            };
            let mut dbytes = [0u8; ops::GdnFusedMultiSeqDesc::SIZE_BYTES];
            desc.write_le_bytes(&mut dbytes);
            gpu.copy_h2d(&dbytes, desc_dev).unwrap();
        };

        for (step, (pbase, prows, wbase, flush)) in
            [(0u32, 0u32, 0u32, 0u32), (0u32, 5u32, 5u32, 1u32)]
                .into_iter()
                .enumerate()
        {
            let qkvz = upload(
                &gpu,
                &gen_bf16_bytes(rows * qkvz_size, 3100 + step as u64, 0.8),
            );
            let gates = upload(
                &gpu,
                &f32_slice_bytes(&make_gates(rows, 3200 + step as u64)),
            );
            let norm_b = gpu.alloc(rows * value_dim * 2).unwrap();
            let norm_s = gpu.alloc(rows * value_dim * 2).unwrap();

            upload_ctrl_and_desc(
                ctrl_b_dev,
                desc_b_dev,
                ring_b,
                conv_b,
                h_b,
                group_done_base,
                pbase,
                prows,
                wbase,
                flush,
            );
            upload_ctrl_and_desc(
                ctrl_s_dev,
                desc_s_dev,
                ring_s,
                conv_s,
                h_s,
                group_done_sv4,
                pbase,
                prows,
                wbase,
                flush,
            );

            ops::qwen4_gdn_verify_fused_lazyfinal_multi_seq(
                &gpu,
                base_multi_lazy,
                desc_b_dev,
                qkvz,
                &conv_w,
                gates,
                gates.offset(nv * 4),
                &norm_w,
                norm_b,
                1,
                nk as u32,
                nv as u32,
                eps,
                eps,
                qkvz_size as u32,
                (nv * 2) as u32,
                value_dim as u32,
                qkvz_size as u32,
                0,
                0,
                stream,
            )
            .unwrap();
            ops::qwen4_gdn_verify_fused_lazyfinal_multi_seq_splitv4(
                &gpu,
                sv4_multi_lazy,
                desc_s_dev,
                qkvz,
                &conv_w,
                gates,
                gates.offset(nv * 4),
                &norm_w,
                norm_s,
                1,
                nk as u32,
                nv as u32,
                eps,
                eps,
                qkvz_size as u32,
                (nv * 2) as u32,
                value_dim as u32,
                qkvz_size as u32,
                0,
                0,
                scratch_gdn,
                scratch_ss,
                split_done,
                stream,
            )
            .unwrap();

            assert_eq!(
                read_bytes(&gpu, conv_s, conv_dim * d_conv * 4),
                read_bytes(&gpu, conv_b, conv_dim * d_conv * 4),
                "KEEP_LATE step={step} conv_state mismatch"
            );
            assert_eq!(
                read_bytes(&gpu, ring_s, ring_bytes),
                read_bytes(&gpu, ring_b, ring_bytes),
                "KEEP_LATE step={step} ring bytes mismatch"
            );
            let (h_err, _) = max_abs_diff_f32_bytes(
                &read_bytes(&gpu, h_s, h_elems * 4),
                &read_bytes(&gpu, h_b, h_elems * 4),
            );
            let (max_ulp, max_norm_abs) = max_bf16_ulp_and_abs_diff(
                &read_bytes(&gpu, norm_s, rows * value_dim * 2),
                &read_bytes(&gpu, norm_b, rows * value_dim * 2),
            );
            let ctrl_after = read_bytes(&gpu, ctrl_s_dev, ops::GdnKeepLateCtrl::SIZE_BYTES);
            let executed_flag = u32::from_le_bytes(ctrl_after[20..24].try_into().unwrap());
            eprintln!(
                "Split-V KEEP_LATE step={step} (pending={prows}, flush={flush}): h_err={h_err:.3e}, norm_ulp={max_ulp}, norm_abs={max_norm_abs:.3e}, executed={executed_flag}"
            );
            assert_eq!(executed_flag, 1, "KEEP_LATE executed_flag must be set to 1");
            assert!(h_err <= 1e-6, "KEEP_LATE step={step} h_err {h_err} > 1e-6");
            assert!(
                max_norm_abs <= 1e-3,
                "KEEP_LATE step={step} norm_abs {max_norm_abs} > 1e-3 (max_ulp={max_ulp})"
            );
            assert_eq!(read_bytes(&gpu, group_done_sv4, 32 * 4), vec![0u8; 32 * 4]);
            assert_eq!(read_bytes(&gpu, split_done, nv * 4), vec![0u8; nv * 4]);
        }
    }
}

#[test]
#[ignore = "needs an idle GB10 and the qwen3.8-27b kernel target"]
fn qwen38_gdn_sequence_splitv4_and_gateup_n32_zero_ulp_on_gb10() {
    let ptx = atlas_kernels::ptx_for_model("qwen3.8-27b").expect("qwen3.8-27b PTX target");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
    let stream = gpu.default_stream();

    // Part A: gated_delta_rule_decode_f32_sequence_{lazyfinal,persistent}_splitv4 (0 ULP exact)
    {
        let nk = 16usize;
        let kd = 128usize;
        let nv = 48usize;
        let vd = 128usize;
        let rows = 8usize;
        let key_dim = nk * kd;
        let qkvz_size = 2 * key_dim + 2 * nv * vd;
        let h_elems = nv * kd * vd;

        let base_lazy = gpu
            .kernel(
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_lazyfinal",
            )
            .unwrap();
        let sv4_lazy = gpu
            .kernel(
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_lazyfinal_splitv4",
            )
            .unwrap();
        let base_pers = gpu
            .kernel(
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_persistent",
            )
            .unwrap();
        let sv4_pers = gpu
            .kernel(
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_persistent_splitv4",
            )
            .unwrap();

        let init_h = f32_slice_bytes(&gen_f32_vec(h_elems, 4101, 0.2));
        let conv_f32 = upload(
            &gpu,
            &f32_slice_bytes(&gen_f32_vec(rows * qkvz_size, 4202, 0.4)),
        );
        let mut gates_host = gen_f32_vec(rows * nv * 2, 4303, 1.0);
        for r in 0..rows {
            for g in &mut gates_host[r * nv * 2..r * nv * 2 + nv] {
                *g = g.abs().clamp(0.05, 0.99);
            }
            for b in &mut gates_host[r * nv * 2 + nv..(r + 1) * nv * 2] {
                *b = b.abs().clamp(0.05, 0.95);
            }
        }
        let gates = upload(&gpu, &f32_slice_bytes(&gates_host));

        for (label, bk, sk) in [
            ("lazyfinal", base_lazy, sv4_lazy),
            ("persistent", base_pers, sv4_pers),
        ] {
            let h_b = upload(&gpu, &init_h);
            let h_s = upload(&gpu, &init_h);
            let out_b = gpu.alloc(rows * qkvz_size * 4).unwrap();
            let out_s = gpu.alloc(rows * qkvz_size * 4).unwrap();
            gpu.memset(out_b, 0, rows * qkvz_size * 4).unwrap();
            gpu.memset(out_s, 0, rows * qkvz_size * 4).unwrap();
            let hi_b = gpu.alloc(rows * h_elems * 4).unwrap();
            let hi_s = gpu.alloc(rows * h_elems * 4).unwrap();
            gpu.memset(hi_b, 0x33, rows * h_elems * 4).unwrap();
            gpu.memset(hi_s, 0x33, rows * h_elems * 4).unwrap();

            ops::gdn_decode_f32_sequence_persistent(
                &gpu,
                bk,
                h_b,
                conv_f32,
                conv_f32.offset(key_dim * 4),
                conv_f32.offset(key_dim * 2 * 4),
                gates,
                gates.offset(nv * 4),
                out_b,
                hi_b,
                rows as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
                h_elems as u32,
                stream,
            )
            .unwrap();
            ops::gdn_decode_f32_sequence_persistent_splitv4(
                &gpu,
                sk,
                h_s,
                conv_f32,
                conv_f32.offset(key_dim * 4),
                conv_f32.offset(key_dim * 2 * 4),
                gates,
                gates.offset(nv * 4),
                out_s,
                hi_s,
                rows as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
                h_elems as u32,
                stream,
            )
            .unwrap();

            assert_eq!(
                read_bytes(&gpu, out_s, rows * qkvz_size * 4),
                read_bytes(&gpu, out_b, rows * qkvz_size * 4),
                "gdn_f32_sequence_{label}_splitv4 output must be 0-ULP identical"
            );
            assert_eq!(
                read_bytes(&gpu, h_s, h_elems * 4),
                read_bytes(&gpu, h_b, h_elems * 4),
                "gdn_f32_sequence_{label}_splitv4 h_state must be 0-ULP identical"
            );
            assert_eq!(
                read_bytes(&gpu, hi_s, rows * h_elems * 4),
                read_bytes(&gpu, hi_b, rows * h_elems * 4),
                "gdn_f32_sequence_{label}_splitv4 h_inter must be 0-ULP identical"
            );
        }
        eprintln!("Qwen3.8-27B gdn_f32_sequence_{{lazyfinal,persistent}}_splitv4: 0-ULP PASS");
    }

    // Part B: w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64 & n64_pipe_k64 vs staged baseline (0 ULP exact)
    {
        let m = 8u32;
        let n = 17_408u32;
        let k = 5_120u32;
        let base_k = gpu
            .kernel("w4a16", "w4a16_gemm_t_m32_n64_gateup_silu")
            .unwrap();
        let k64_k = gpu
            .kernel("w4a16", "w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64")
            .unwrap();
        let n32_k = gpu
            .kernel("w4a16", "w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64")
            .unwrap();

        let input = upload(&gpu, &gen_bf16_bytes((m * k) as usize, 5101, 0.5));
        let gen_bytes = |len: usize, seed: u64, mask: u8| -> Vec<u8> {
            let mut s = seed | 1;
            (0..len)
                .map(|_| ((xorshift64(&mut s) >> 32) as u8) & mask)
                .collect()
        };
        let make_weight = |seed: u64| spark_model::weight_map::QuantizedWeight {
            weight: upload(&gpu, &gen_bytes((k / 2 * n) as usize, seed, 0xFF)),
            // Valid finite positive FP8 E4M3 scales in [0x28..=0x2F] U [0x38..=0x3F]
            // (0.25..=0.46875 and 1.0..=1.875).
            weight_scale: upload(
                &gpu,
                &gen_bytes((k / 16 * n) as usize, seed + 1, 0x1F)
                    .into_iter()
                    .map(|b| 0x28u8 + (b & 0x17))
                    .collect::<Vec<_>>(),
            ),
            weight_scale_2: 0.015625,
            input_scale: DevicePtr::NULL,
        };
        let gate_w = make_weight(5202);
        let up_w = make_weight(5303);

        let out_bytes = (m * n * 2) as usize;
        let out_base = gpu.alloc(out_bytes).unwrap();
        let out_k64 = gpu.alloc(out_bytes).unwrap();
        let out_n32 = gpu.alloc(out_bytes).unwrap();

        ops::w4a16_gemm_n64_m32_gateup_silu(
            &gpu, base_k, input, &gate_w, &up_w, out_base, m, n, k, stream,
        )
        .unwrap();
        ops::w4a16_gemm_n64_m32_gateup_silu(
            &gpu, k64_k, input, &gate_w, &up_w, out_k64, m, n, k, stream,
        )
        .unwrap();
        ops::w4a16_gemm_n32_m32_gateup_silu(
            &gpu, n32_k, input, &gate_w, &up_w, out_n32, m, n, k, stream,
        )
        .unwrap();

        let base_bytes = read_bytes(&gpu, out_base, out_bytes);
        let k64_bytes = read_bytes(&gpu, out_k64, out_bytes);
        let n32_bytes = read_bytes(&gpu, out_n32, out_bytes);
        assert_eq!(
            k64_bytes, base_bytes,
            "w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64 must be 0-ULP byte-exact vs baseline"
        );
        assert_eq!(
            n32_bytes, base_bytes,
            "w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64 must be 0-ULP byte-exact vs baseline"
        );
        eprintln!(
            "Qwen3.8-27B gateup_silu_{{n64,n32}}_pipe_k64 (M=8, N=17408, K=5120): 0-ULP PASS"
        );
    }
}
