// SPDX-License-Identifier: AGPL-3.0-only

//! Live SM121a GB10 parity tests for 1M-context QSA v3 selection, multi-row NVFP4 QSA
//! sparse attention, fused NVFP4 + YaRN attention prep, and TP=2 (8, 24) FlashInfer GDN.
//! Ignored by default because they require an exclusive GB10 GPU lock.

use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

// SAFETY: `atlas_gdn_fi_sm121a_prefill` is compiled and linked with the C ABI
// declared here whenever `cfg(atlas_fi_gdn)` is active by `build.rs`.
#[cfg(atlas_fi_gdn)]
unsafe extern "C" {
    fn atlas_gdn_fi_sm121a_prefill(
        qkv: *mut core::ffi::c_void,
        gate_beta: *mut core::ffi::c_void,
        output: *mut core::ffi::c_void,
        state: *mut core::ffi::c_void,
        total_len: i32,
        conv_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let ptr = gpu.alloc(bytes.len().max(16)).unwrap();
    if !bytes.is_empty() {
        gpu.copy_h2d(bytes, ptr).unwrap();
    }
    ptr
}

fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, len: usize) -> Vec<u8> {
    gpu.synchronize(gpu.default_stream()).unwrap();
    let mut out = vec![0u8; len];
    gpu.copy_d2h(ptr, &mut out).unwrap();
    out
}

#[test]
#[ignore = "needs an idle GB10 and the qwen3.8-flash-next kernel target"]
fn qsa_select_v3_and_rows_v3_match_reference_at_1m_tokens_on_gb10() {
    let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
    let stream = gpu.default_stream();

    // 1,048,572..1,048,576 tokens => 262,143..262,144 compressed groups (8 slabs of 33,792).
    let max_groups = 262_144usize;
    let num_rows = 5usize;
    let base_seq_len = 1_048_572u32;
    let base_pos = base_seq_len - 1;
    let out_width = 2_051usize;

    let mut host_logits = vec![0.0f32; num_rows * max_groups];
    for r in 0..num_rows {
        for g in 0..max_groups {
            // Include periodic ties across slab boundaries to verify lowest-index tie breaking.
            let bucket = ((g * 7919 + r * 104_729) % 4096) as f32;
            let slab_bonus = if g > 200_000 && g % 511 == r {
                5000.0
            } else {
                0.0
            };
            host_logits[r * max_groups + g] = bucket * 0.0625 + slab_bonus;
        }
    }
    let logits_bytes: Vec<u8> = host_logits.iter().flat_map(|x| x.to_le_bytes()).collect();
    let d_logits = upload(&gpu, &logits_bytes);
    let d_ref = gpu.alloc(out_width * 4).unwrap();
    let d_v3 = gpu.alloc(out_width * 4).unwrap();
    let d_rows_v3 = gpu.alloc(num_rows * out_width * 4).unwrap();

    let k_ref = gpu.kernel("qwen4_qsa", "qwen4_qsa_select_expand").unwrap();
    let k_v3 = gpu
        .kernel("qwen4_qsa", "qwen4_qsa_select_expand_par_v3")
        .unwrap();
    let k_rows_v3 = {
        let h = crate::layers::try_kernel(&gpu, "qwen4_qsa", "qwen4_qsa_select_expand_rows_v3");
        if h.0 != 0 {
            h
        } else {
            gpu.kernel(
                "qwen4_qsa_prefill_attn_fp8",
                "qwen4_qsa_select_expand_rows_v3",
            )
            .unwrap()
        }
    };

    KernelLaunch::new(&gpu, k_rows_v3)
        .grid([1, num_rows as u32, 1])
        .block([1024, 1, 1])
        .arg_ptr(d_logits)
        .arg_u64(max_groups as u64)
        .arg_ptr(d_rows_v3)
        .arg_u32(base_pos)
        .arg_u32(base_seq_len)
        .launch(stream)
        .unwrap();
    let rows_v3_bytes = download(&gpu, d_rows_v3, num_rows * out_width * 4);

    for r in 0..num_rows {
        let seq_len = base_seq_len + r as u32;
        let pos = base_pos + r as u32;
        let visible_groups = seq_len / 4;
        let row_logits = d_logits.offset(r * max_groups * 4);

        KernelLaunch::new(&gpu, k_ref)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(row_logits)
            .arg_ptr(d_ref)
            .arg_u32(visible_groups)
            .arg_u32(pos)
            .arg_u32(seq_len)
            .launch(stream)
            .unwrap();
        KernelLaunch::new(&gpu, k_v3)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(row_logits)
            .arg_ptr(d_v3)
            .arg_u32(visible_groups)
            .arg_u32(pos)
            .arg_u32(seq_len)
            .launch(stream)
            .unwrap();

        let ref_bytes = download(&gpu, d_ref, out_width * 4);
        let v3_bytes = download(&gpu, d_v3, out_width * 4);
        let rows_slice = &rows_v3_bytes[r * out_width * 4..(r + 1) * out_width * 4];
        assert_eq!(v3_bytes, ref_bytes, "1M select_v3 mismatch at row {r}");
        assert_eq!(
            rows_slice, ref_bytes,
            "1M select_rows_v3 mismatch at row {r}"
        );
    }
}

#[test]
#[ignore = "needs an idle GB10 and the qwen3.8-flash-next kernel target"]
fn qsa_nvfp4_multi_row_verify_and_fused_attn_prep_match_reference_on_gb10() {
    let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
    let stream = gpu.default_stream();

    // Part A: Fused NVFP4 + YaRN prep byte parity against reshape_and_cache_flash_nvfp4
    // for both TP=1 (nq=24, nkv=2) and TP=2 (nq=12, nkv=1).
    let head_dim = 256usize;
    let rotary_dim = 64u32;
    let block_size = 16usize;
    let rows = 6usize;
    let eps = 1e-6f32;
    let attn_factor = 1.138_629_4f32;

    let k_freq = gpu
        .kernel("qwen4_attn_prep_fused", "qwen4_rope_freq_table")
        .unwrap();
    let k_prep_nvfp4 = gpu
        .kernel("qwen4_attn_prep_fused", "qwen4_attn_prep_fused_nvfp4")
        .unwrap();
    let k_prep_yarn = gpu
        .kernel("qwen4_attn_prep_fused", "qwen4_attn_prep_fused_yarn")
        .unwrap();
    let k_reshape = gpu
        .kernel("reshape_and_cache", "reshape_and_cache_flash_nvfp4")
        .unwrap();

    let d_freq = gpu.alloc(128 * 4).unwrap();
    KernelLaunch::new(&gpu, k_freq)
        .grid([1, 1, 1])
        .block([rotary_dim / 2, 1, 1])
        .arg_ptr(d_freq)
        .arg_u32(rotary_dim)
        .arg_f32(10_000_000.0)
        .launch(stream)
        .unwrap();

    for (nq, nkv) in [(24usize, 2usize), (12usize, 1usize)] {
        let total_heads = nq + 2 * nkv;
        let row_stride = total_heads * head_dim;
        let k_offset = nq * head_dim;
        let v_offset = (nq + nkv) * head_dim;
        let data_bytes = block_size * nkv * head_dim / 2;
        let scale_bytes = block_size * nkv * (head_dim / 16);
        let block_stride_bytes = data_bytes + scale_bytes;

        let qkv_bf16: Vec<u16> = (0..rows * row_stride)
            .map(|i| bf16::from_f32(((i % 97) as f32 - 48.0) * 0.03125).to_bits())
            .collect();
        let q_norm: Vec<u16> = (0..head_dim)
            .map(|i| bf16::from_f32(0.85 + (i % 13) as f32 * 0.02).to_bits())
            .collect();
        let k_norm: Vec<u16> = (0..head_dim)
            .map(|i| bf16::from_f32(0.90 + (i % 11) as f32 * 0.02).to_bits())
            .collect();
        let pos: Vec<u32> = (0..rows as u32).map(|r| 262_144 + r).collect();
        let slots: Vec<i64> = (0..rows as i64).collect();

        let to_u8 = |s: &[u16]| -> Vec<u8> { s.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let d_qkv_a = upload(&gpu, &to_u8(&qkv_bf16));
        let d_qkv_b = upload(&gpu, &to_u8(&qkv_bf16));
        let d_qn = upload(&gpu, &to_u8(&q_norm));
        let d_kn = upload(&gpu, &to_u8(&k_norm));
        let d_pos = upload(
            &gpu,
            &pos.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        );
        let d_slots = upload(
            &gpu,
            &slots
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        );

        let d_k_bf16 = gpu.alloc(block_size * nkv * head_dim * 2).unwrap();
        let d_v_bf16 = gpu.alloc(block_size * nkv * head_dim * 2).unwrap();
        let d_k_fused = gpu.alloc(block_stride_bytes).unwrap();
        let d_v_fused = gpu.alloc(block_stride_bytes).unwrap();
        let d_k_ref = gpu.alloc(block_stride_bytes).unwrap();
        let d_v_ref = gpu.alloc(block_stride_bytes).unwrap();
        for (p, sz) in [
            (d_k_fused, block_stride_bytes),
            (d_v_fused, block_stride_bytes),
            (d_k_ref, block_stride_bytes),
            (d_v_ref, block_stride_bytes),
        ] {
            gpu.memset_async(p, 0, sz, stream).unwrap();
        }

        // Run BF16 YaRN prep to rotate Q and K in-place on d_qkv_a, then quantize via reshape_and_cache_flash_nvfp4.
        KernelLaunch::new(&gpu, k_prep_yarn)
            .grid([(rows * (nq + nkv)) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_qkv_a)
            .arg_ptr(d_qn)
            .arg_ptr(d_kn)
            .arg_ptr(d_pos)
            .arg_ptr(d_pos)
            .arg_ptr(d_pos)
            .arg_ptr(d_freq)
            .arg_ptr(d_k_bf16)
            .arg_ptr(d_v_bf16)
            .arg_ptr(d_slots)
            .arg_u32(nq as u32)
            .arg_u32(nkv as u32)
            .arg_u32(row_stride as u32)
            .arg_u32(k_offset as u32)
            .arg_u32(v_offset as u32)
            .arg_u32(rotary_dim)
            .arg_u32(block_size as u32)
            .arg_f32(eps)
            .arg_f32(attn_factor)
            .launch(stream)
            .unwrap();
        KernelLaunch::new(&gpu, k_reshape)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_qkv_a.offset(k_offset * 2))
            .arg_ptr(d_qkv_a.offset(v_offset * 2))
            .arg_ptr(d_k_ref)
            .arg_ptr(d_v_ref)
            .arg_ptr(d_slots)
            .arg_u32(nkv as u32)
            .arg_u32(head_dim as u32)
            .arg_u32(block_size as u32)
            .arg_u32(row_stride as u32)
            .arg_u32(row_stride as u32)
            .arg_u64(block_stride_bytes as u64)
            .arg_u64(data_bytes as u64)
            .launch(stream)
            .unwrap();

        // Run fused NVFP4 + YaRN prep on d_qkv_b.
        KernelLaunch::new(&gpu, k_prep_nvfp4)
            .grid([(rows * (nq + nkv)) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_qkv_b)
            .arg_ptr(d_qn)
            .arg_ptr(d_kn)
            .arg_ptr(d_pos)
            .arg_ptr(d_pos)
            .arg_ptr(d_pos)
            .arg_ptr(d_freq)
            .arg_ptr(d_k_fused)
            .arg_ptr(d_v_fused)
            .arg_ptr(d_slots)
            .arg_u32(nq as u32)
            .arg_u32(nkv as u32)
            .arg_u32(row_stride as u32)
            .arg_u32(k_offset as u32)
            .arg_u32(v_offset as u32)
            .arg_u32(rotary_dim)
            .arg_u32(block_size as u32)
            .arg_u64(block_stride_bytes as u64)
            .arg_u64(data_bytes as u64)
            .arg_f32(eps)
            .arg_f32(attn_factor)
            .launch(stream)
            .unwrap();

        assert_eq!(
            download(&gpu, d_qkv_b, rows * row_stride * 2),
            download(&gpu, d_qkv_a, rows * row_stride * 2),
            "fused NVFP4 Q/K in-place rotation mismatch for nq={nq}, nkv={nkv}"
        );
        assert_eq!(
            download(&gpu, d_k_fused, block_stride_bytes),
            download(&gpu, d_k_ref, block_stride_bytes),
            "fused NVFP4 K cache byte mismatch for nq={nq}, nkv={nkv}"
        );
        assert_eq!(
            download(&gpu, d_v_fused, block_stride_bytes),
            download(&gpu, d_v_ref, block_stride_bytes),
            "fused NVFP4 V cache byte mismatch for nq={nq}, nkv={nkv}"
        );
    }

    // Part B: Multi-row (K=5) NVFP4 QSA sparse attention legacy (0 ULP) and v2 (<= 1 BF16 ULP).
    let nq = 24u32;
    let nkv = 2u32;
    let splits = 8u32;
    let verify_rows = 5usize;
    let out_width = 2_051usize;
    let num_blocks = 8usize;
    let data_bytes = block_size * (nkv as usize) * head_dim / 2;
    let scale_bytes = block_size * (nkv as usize) * (head_dim / 16);
    let block_stride_bytes = data_bytes + scale_bytes;

    let mut kv_bytes = vec![0u8; num_blocks * block_stride_bytes];
    for b in 0..num_blocks {
        for i in 0..data_bytes {
            kv_bytes[b * block_stride_bytes + i] = ((b * 37 + i * 13) & 0x77) as u8;
        }
        for s in 0..scale_bytes {
            // Valid non-NaN FP8 E4M3 scale (~0.25..1.0).
            kv_bytes[b * block_stride_bytes + data_bytes + s] = 0x30 + ((b + s) % 8) as u8;
        }
    }
    let d_k_pool = upload(&gpu, &kv_bytes);
    let d_v_pool = upload(&gpu, &kv_bytes.iter().rev().copied().collect::<Vec<u8>>());
    let q_bf16: Vec<u16> = (0..verify_rows * (nq as usize) * head_dim)
        .map(|i| bf16::from_f32(((i % 61) as f32 - 30.0) * 0.05).to_bits())
        .collect();
    let d_q = upload(
        &gpu,
        &q_bf16
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );

    let mut indices = vec![-1i32; verify_rows * out_width];
    for r in 0..verify_rows {
        for j in 0..96 {
            indices[r * out_width + j] = ((j + r) % (num_blocks * block_size)) as i32;
        }
    }
    let d_indices = upload(
        &gpu,
        &indices
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let bt_single: Vec<i32> = (0..num_blocks as i32).collect();
    let bt_rows: Vec<i32> = (0..verify_rows)
        .flat_map(|_| bt_single.iter().copied())
        .collect();
    let d_bt_rows = upload(
        &gpu,
        &bt_rows
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );

    let part_elems = verify_rows * (nq as usize) * (splits as usize) * head_dim;
    let stat_elems = verify_rows * (nq as usize) * (splits as usize);
    let d_pout = gpu.alloc(part_elems * 4).unwrap();
    let d_pmax = gpu.alloc(stat_elems * 4).unwrap();
    let d_psum = gpu.alloc(stat_elems * 4).unwrap();
    let d_out_ref = gpu
        .alloc(verify_rows * (nq as usize) * head_dim * 2)
        .unwrap();
    let d_out_rows = gpu
        .alloc(verify_rows * (nq as usize) * head_dim * 2)
        .unwrap();
    let d_out_rows_v2 = gpu
        .alloc(verify_rows * (nq as usize) * head_dim * 2)
        .unwrap();

    let k_single = gpu
        .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_nvfp4_partial")
        .unwrap();
    let k_reduce_single = gpu
        .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_bf16_reduce")
        .unwrap();
    let k_rows = gpu
        .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_nvfp4_partial_rows")
        .unwrap();
    let k_rows_v2 = gpu
        .kernel(
            "qwen4_qsa",
            "qwen4_qsa_sparse_attention_nvfp4_partial_rows_v2",
        )
        .unwrap();
    let k_reduce_rows = gpu
        .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_bf16_reduce_rows")
        .unwrap();

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    for r in 0..verify_rows {
        KernelLaunch::new(&gpu, k_single)
            .grid([nq, splits, 1])
            .block([head_dim as u32, 1, 1])
            .arg_ptr(d_q.offset(r * (nq as usize) * head_dim * 2))
            .arg_ptr(d_k_pool)
            .arg_ptr(d_v_pool)
            .arg_ptr(d_indices.offset(r * out_width * 4))
            .arg_ptr(d_bt_rows.offset(r * num_blocks * 4))
            .arg_ptr(d_pout)
            .arg_ptr(d_pmax)
            .arg_ptr(d_psum)
            .arg_u64(block_stride_bytes as u64)
            .arg_u64(block_stride_bytes as u64)
            .arg_u64(data_bytes as u64)
            .arg_u32(block_size as u32)
            .arg_u32(nq)
            .arg_u32(nkv)
            .arg_u32(head_dim as u32)
            .arg_f32(scale)
            .arg_u32(splits)
            .launch(stream)
            .unwrap();
        KernelLaunch::new(&gpu, k_reduce_single)
            .grid([nq, 1, 1])
            .block([head_dim as u32, 1, 1])
            .arg_ptr(d_pout)
            .arg_ptr(d_pmax)
            .arg_ptr(d_psum)
            .arg_ptr(d_out_ref.offset(r * (nq as usize) * head_dim * 2))
            .arg_u32(nq)
            .arg_u32(head_dim as u32)
            .arg_u32(splits)
            .launch(stream)
            .unwrap();
    }

    for (kernel, out_ptr) in [(k_rows, d_out_rows), (k_rows_v2, d_out_rows_v2)] {
        KernelLaunch::new(&gpu, kernel)
            .grid([nq, splits, verify_rows as u32])
            .block([head_dim as u32, 1, 1])
            .arg_ptr(d_q)
            .arg_ptr(d_k_pool)
            .arg_ptr(d_v_pool)
            .arg_ptr(d_indices)
            .arg_ptr(d_bt_rows)
            .arg_ptr(d_pout)
            .arg_ptr(d_pmax)
            .arg_ptr(d_psum)
            .arg_u64(block_stride_bytes as u64)
            .arg_u64(block_stride_bytes as u64)
            .arg_u64(data_bytes as u64)
            .arg_u32(block_size as u32)
            .arg_u32(nq)
            .arg_u32(nkv)
            .arg_u32(head_dim as u32)
            .arg_f32(scale)
            .arg_u32(splits)
            .arg_u32(nq * head_dim as u32)
            .arg_u32(num_blocks as u32)
            .launch(stream)
            .unwrap();
        KernelLaunch::new(&gpu, k_reduce_rows)
            .grid([nq, verify_rows as u32, 1])
            .block([head_dim as u32, 1, 1])
            .arg_ptr(d_pout)
            .arg_ptr(d_pmax)
            .arg_ptr(d_psum)
            .arg_ptr(out_ptr)
            .arg_u32(nq)
            .arg_u32(head_dim as u32)
            .arg_u32(splits)
            .launch(stream)
            .unwrap();
    }

    let ref_out = download(&gpu, d_out_ref, verify_rows * (nq as usize) * head_dim * 2);
    let rows_out = download(&gpu, d_out_rows, verify_rows * (nq as usize) * head_dim * 2);
    let rows_v2_out = download(
        &gpu,
        d_out_rows_v2,
        verify_rows * (nq as usize) * head_dim * 2,
    );
    assert_eq!(
        rows_out, ref_out,
        "NVFP4 partial_rows must be 0-ULP identical to per-row partial"
    );
    for (idx, (a, b)) in ref_out
        .chunks_exact(2)
        .zip(rows_v2_out.chunks_exact(2))
        .enumerate()
    {
        let fa = bf16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
        let fb = bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        assert!(
            (fa - fb).abs() <= 2e-2,
            "NVFP4 partial_rows_v2 diverged at {idx}: ref={fa} v2={fb}"
        );
    }
}

#[cfg(atlas_fi_gdn)]
#[test]
#[ignore = "needs an idle GB10 and the qwen3.8-flash-next kernel target"]
fn tp2_sharded_flashinfer_gdn_bridge_matches_tp1_full_heads_on_gb10() {
    let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
    let stream = gpu.default_stream();

    let t = 64usize;
    let d = 128usize;
    let (hq_full, hv_full, conv_full) = (16usize, 48usize, 10_240usize);
    let (hq_tp2, hv_tp2, conv_tp2) = (8usize, 24usize, 5_120usize);

    let qkv_full: Vec<u16> = (0..t * conv_full)
        .map(|i| bf16::from_f32(((i % 83) as f32 - 41.0) * 0.015625).to_bits())
        .collect();
    let gb_full: Vec<f32> = (0..t * 2 * hv_full)
        .map(|i| {
            let col = i % (2 * hv_full);
            if col < hv_full {
                // Multiplicative decay gate in (0.85, 0.99).
                0.90 + (i % 7) as f32 * 0.01
            } else {
                // Beta in (0.05, 0.35).
                0.10 + (i % 5) as f32 * 0.04
            }
        })
        .collect();
    let state_init_full: Vec<f32> = (0..hv_full * d * d)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.002)
        .collect();

    // Slice into TP=2 rank 0 and rank 1 planar [Q | K | V] shards!
    let mut qkv_shards = [vec![0u16; t * conv_tp2], vec![0u16; t * conv_tp2]];
    let mut gb_shards = [vec![0.0f32; t * 2 * hv_tp2], vec![0.0f32; t * 2 * hv_tp2]];
    for row in 0..t {
        let src = &qkv_full[row * conv_full..(row + 1) * conv_full];
        let q_src = &src[0..hq_full * d];
        let k_src = &src[hq_full * d..2 * hq_full * d];
        let v_src = &src[2 * hq_full * d..conv_full];
        for r in 0..2 {
            let dst = &mut qkv_shards[r][row * conv_tp2..(row + 1) * conv_tp2];
            dst[0..hq_tp2 * d].copy_from_slice(&q_src[r * hq_tp2 * d..(r + 1) * hq_tp2 * d]);
            dst[hq_tp2 * d..2 * hq_tp2 * d]
                .copy_from_slice(&k_src[r * hq_tp2 * d..(r + 1) * hq_tp2 * d]);
            dst[2 * hq_tp2 * d..conv_tp2]
                .copy_from_slice(&v_src[r * hv_tp2 * d..(r + 1) * hv_tp2 * d]);

            let gb_row = &gb_full[row * 2 * hv_full..(row + 1) * 2 * hv_full];
            let gb_dst = &mut gb_shards[r][row * 2 * hv_tp2..(row + 1) * 2 * hv_tp2];
            gb_dst[0..hv_tp2].copy_from_slice(&gb_row[r * hv_tp2..(r + 1) * hv_tp2]);
            gb_dst[hv_tp2..2 * hv_tp2]
                .copy_from_slice(&gb_row[hv_full + r * hv_tp2..hv_full + (r + 1) * hv_tp2]);
        }
    }

    let u16_bytes = |s: &[u16]| -> Vec<u8> { s.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let f32_bytes = |s: &[f32]| -> Vec<u8> { s.iter().flat_map(|x| x.to_le_bytes()).collect() };

    let d_qkv_full = upload(&gpu, &u16_bytes(&qkv_full));
    let d_gb_full = upload(&gpu, &f32_bytes(&gb_full));
    let d_state_full = upload(&gpu, &f32_bytes(&state_init_full));
    let d_out_full = gpu.alloc(t * hv_full * d * 2).unwrap();

    let ptr = |p: DevicePtr| p.0 as usize as *mut core::ffi::c_void;
    // SAFETY: `d_qkv_full`, `d_gb_full`, `d_out_full`, and `d_state_full` are
    // valid, distinct, non-null CUDA device allocations on `gpu`'s active
    // context with byte capacities matching `(t=64, hv_full=48, d=128,
    // conv_full=10_240)`. `t` and `conv_full` are positive and fit in `i32`,
    // `stream` is `gpu`'s valid default CUDA stream, and the buffers outlive
    // the synchronous `download(...)` calls that follow.
    let rc_full = unsafe {
        atlas_gdn_fi_sm121a_prefill(
            ptr(d_qkv_full),
            ptr(d_gb_full),
            ptr(d_out_full),
            ptr(d_state_full),
            t as i32,
            conv_full as i32,
            stream as usize as *mut core::ffi::c_void,
        )
    };
    assert_eq!(rc_full, 0, "TP=1 (16, 48) FlashInfer GDN bridge failed");

    let out_full_bytes = download(&gpu, d_out_full, t * hv_full * d * 2);
    let state_full_bytes = download(&gpu, d_state_full, hv_full * d * d * 4);

    let mut reassembled_out = vec![0u8; t * hv_full * d * 2];
    let mut reassembled_state = Vec::with_capacity(hv_full * d * d * 4);

    for r in 0..2 {
        let d_qkv = upload(&gpu, &u16_bytes(&qkv_shards[r]));
        let d_gb = upload(&gpu, &f32_bytes(&gb_shards[r]));
        let state_slice = &state_init_full[r * hv_tp2 * d * d..(r + 1) * hv_tp2 * d * d];
        let d_state = upload(&gpu, &f32_bytes(state_slice));
        let d_out = gpu.alloc(t * hv_tp2 * d * 2).unwrap();

        // SAFETY: `d_qkv`, `d_gb`, `d_out`, and `d_state` are valid, distinct,
        // non-null CUDA device allocations on `gpu`'s active context with byte
        // capacities matching `(t=64, hv_tp2=24, d=128, conv_tp2=5_120)`. `t`
        // and `conv_tp2` are positive and fit in `i32`, `stream` is `gpu`'s
        // valid default CUDA stream, and the buffers outlive the synchronous
        // `download(...)` calls that follow.
        let rc = unsafe {
            atlas_gdn_fi_sm121a_prefill(
                ptr(d_qkv),
                ptr(d_gb),
                ptr(d_out),
                ptr(d_state),
                t as i32,
                conv_tp2 as i32,
                stream as usize as *mut core::ffi::c_void,
            )
        };
        assert_eq!(rc, 0, "TP=2 rank {r} (8, 24) FlashInfer GDN bridge failed");

        let shard_out = download(&gpu, d_out, t * hv_tp2 * d * 2);
        let shard_state = download(&gpu, d_state, hv_tp2 * d * d * 4);
        let shard_row_bytes = hv_tp2 * d * 2;
        let full_row_bytes = hv_full * d * 2;
        for row in 0..t {
            let dst_off = row * full_row_bytes + r * shard_row_bytes;
            reassembled_out[dst_off..dst_off + shard_row_bytes]
                .copy_from_slice(&shard_out[row * shard_row_bytes..(row + 1) * shard_row_bytes]);
        }
        reassembled_state.extend_from_slice(&shard_state);
    }

    assert_eq!(
        reassembled_out, out_full_bytes,
        "TP=2 (8, 24) FlashInfer GDN output must be byte-identical to TP=1 (16, 48)"
    );
    assert_eq!(
        reassembled_state, state_full_bytes,
        "TP=2 (8, 24) FlashInfer GDN final recurrent state must be byte-identical to TP=1 (16, 48)"
    );
}
