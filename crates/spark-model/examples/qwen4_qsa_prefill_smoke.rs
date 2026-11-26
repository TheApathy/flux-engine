// SPDX-License-Identifier: AGPL-3.0-only

//! In-engine smoke test of the QSA-sparse prefill core (`prefill_sparse_attention`).
//!
//! Exercises the exact Rust launch path the server uses (kernel loading, shared-
//! memory attributes, scratch layout, argument lists, paged K/V and side-cache
//! addressing) on one attention layer, with the real layer-3 indexer weights:
//!   1. a one-layer BF16 `PagedKvCache` behind a shuffled block table, filled
//!      with random K/V;
//!   2. the QSA side cache filled through the public `update_prefill_index`
//!      (one 4-row group at a time, as decode-order prefill would);
//!   3. `prefill_sparse_attention` on one chunk, ungated, with the dump hook;
//!   4. the same call with a gate, checked on the host against
//!      bf16(ungated) * sigmoid(gate).
//!
//! `QSA_SMOKE_WEIGHTS` must point to a directory containing the raw little-endian
//! layer-3 QSA indexer weight blobs:
//!   - `index_qk_proj.bin`: BF16 weights for the QSA Q/K indexer projection;
//!   - `q_layernorm.bin`: BF16/F32 weight vector for the Q indexer RMS/LayerNorm;
//!   - `k_layernorm.bin`: BF16/F32 weight vector for the K indexer RMS/LayerNorm.
//!
//! ```text
//! ATLAS_QWEN4_PREFILL_QSA_DUMP=<dump> ATLAS_QWEN4_PREFILL_QSA_DUMP_LAYERS=0 \
//!   QSA_SMOKE_WEIGHTS=<dir> QSA_SMOKE_START=8192 QSA_SMOKE_ROWS=2048 \
//!   ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p spark-model --example qwen4_qsa_prefill_smoke
//! ```

use anyhow::{Context, Result, ensure};
use spark_model::layer::AttnMetadataDev;
use spark_model::layers::Qwen4QsaIndexer;
use spark_model::layers::qwen4_qsa::prefill_sparse::SparseChunk;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

const HIDDEN: usize = 2560;
const NQ: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
const PAGE: usize = 16;

/// Deterministic xorshift normal-ish values (sum of uniforms), as bf16 bits.
fn bf16_noise(n: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            let mut acc = 0.0f32;
            for _ in 0..4 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                acc += (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            }
            ((acc * scale).to_bits() >> 16) as u16
        })
        .collect()
}

/// Marker for padding-free primitive types whose bit representation is always initialized.
///
/// # Safety
/// Every byte of any valid value of `Self` must be initialized (no padding bytes).
unsafe trait Pod: Copy {}
// SAFETY: Primitive integer types have no uninitialized padding bytes.
unsafe impl Pod for u8 {}
// SAFETY: Primitive integer types have no uninitialized padding bytes.
unsafe impl Pod for u16 {}
// SAFETY: Primitive integer types have no uninitialized padding bytes.
unsafe impl Pod for u32 {}
// SAFETY: Primitive integer types have no uninitialized padding bytes.
unsafe impl Pod for i32 {}
// SAFETY: Primitive integer types have no uninitialized padding bytes.
unsafe impl Pod for i64 {}

fn upload<T: Pod>(gpu: &dyn GpuBackend, data: &[T]) -> Result<DevicePtr> {
    let bytes = std::mem::size_of_val(data);
    // SAFETY: `T: Pod` guarantees every byte of `data` is initialized with no padding,
    // `data.as_ptr()` is valid for `bytes` readable bytes, and the slice does not outlive `data`.
    let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
    let ptr = gpu.alloc(bytes.max(16))?;
    gpu.copy_h2d(raw, ptr)?;
    Ok(ptr)
}

fn read_bin(dir: &str, name: &str) -> Result<Vec<u8>> {
    std::fs::read(format!("{dir}/{name}")).with_context(|| format!("read {dir}/{name}"))
}

fn env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is required"))?
        .parse()
        .with_context(|| name.to_string())
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn main() -> Result<()> {
    let weights = std::env::var("QSA_SMOKE_WEIGHTS").context("QSA_SMOKE_WEIGHTS is required")?;
    let start = env_usize("QSA_SMOKE_START")?;
    let rows = env_usize("QSA_SMOKE_ROWS")?;
    ensure!(
        start.is_multiple_of(4) && rows.is_multiple_of(4),
        "start and rows must be 4-aligned"
    );
    let total = start + rows;

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = HIDDEN;
    config.head_dim = HD;
    config.num_attention_heads = NQ;
    config.num_key_value_heads = NKV;
    config.partial_rotary_factor = 0.25;
    config.rope_theta = 10_000_000.0;
    config.yarn_factor = 0.0;
    let eps = 1.0e-6f32;
    let rotary = config.rotary_dim() as u32;
    ensure!(rotary == 64, "rotary_dim {rotary}");

    let w = |n: &str| -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: upload(gpu, &read_bin(&weights, n)?)?,
        })
    };
    let qsa = Qwen4QsaIndexer::new(
        w("index_qk_proj.bin")?,
        w("q_layernorm.bin")?,
        w("k_layernorm.bin")?,
        gpu,
        &config,
    )?;

    // One BF16 attention layer, pages handed out in a scrambled order.
    let pages = total.div_ceil(PAGE);
    let mut kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: PAGE,
            num_kv_heads: NKV,
            head_dim: HD,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: Vec::new(),
            layer_dims: Vec::new(),
            cache_blocks_per_seq: None,
        },
        pages * 2 + 7,
        gpu,
    )?;
    let mut table = Vec::with_capacity(pages);
    for i in 0..pages * 2 + 7 {
        let b = kv.alloc_block()?;
        if i % 2 == 1 && table.len() < pages {
            table.push(b as i32);
        }
    }
    ensure!(table.len() == pages, "page allocation");
    let bt = upload(gpu, &table)?;
    let k_pool = kv.k_pool_ptr(0);
    let v_pool = kv.v_pool_ptr(0);
    let page_bytes = kv.k_block_stride_bytes_for_layer(0);
    let k_all = bf16_noise(total * NKV * HD, 11, 1.0);
    let v_all = bf16_noise(total * NKV * HD, 12, 1.0);
    let row = NKV * HD;
    for (p, &phys) in table.iter().enumerate() {
        let lo = p * PAGE * row;
        let hi = ((p + 1) * PAGE).min(total) * row;
        let to_bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        gpu.copy_h2d(
            &to_bytes(&k_all[lo..hi]),
            k_pool.offset(phys as usize * page_bytes),
        )?;
        gpu.copy_h2d(
            &to_bytes(&v_all[lo..hi]),
            v_pool.offset(phys as usize * page_bytes),
        )?;
    }

    // Per-token metadata for the whole sequence: positions, physical slots.
    let positions: Vec<u32> = (0..total as u32).collect();
    let slots: Vec<i64> = (0..total)
        .map(|t| table[t / PAGE] as i64 * PAGE as i64 + (t % PAGE) as i64)
        .collect();
    let pos_dev = upload(gpu, &positions)?;
    let slot_dev = upload(gpu, &slots)?;
    let seq_len = upload(gpu, &[total as i32])?;
    let meta_at = |t: usize| AttnMetadataDev {
        qwen4_qsa_required: true,
        positions: pos_dev.offset(t * 4),
        positions_h: pos_dev.offset(t * 4),
        positions_w: pos_dev.offset(t * 4),
        slot: slot_dev.offset(t * 8),
        seq_len,
        block_table: bt,
        max_blocks_per_seq: pages as u32,
        num_seqs: 1,
    };

    // QSA side cache through the public decode-order index update.
    let hidden = upload(gpu, &bf16_noise(total * HIDDEN, 13, 0.5))?;
    let raw = gpu.alloc(4 * 128 * 2)?;
    let pooled = gpu.alloc(4 * 128 * 2)?;
    let first = gpu.alloc(64)?;
    for g in (0..total).step_by(4) {
        qsa.update_prefill_index(
            hidden.offset(g * HIDDEN * 2),
            4,
            g,
            raw,
            pooled,
            first,
            &kv,
            meta_at(g),
            HIDDEN as u32,
            eps,
            config.rope_theta as f32,
            rotary,
            gpu,
            stream,
        )?;
    }

    // The chunk: random post-RoPE queries; ungated then gated.
    // Fused-QKV row layout as the engine builds it: [q | gate | k | v], so Q
    // and the gate are both read in place at the QKV row stride.
    let qkv_stride = 2 * NQ * HD + 2 * NKV * HD;
    let mut qkv_host = bf16_noise(rows * qkv_stride, 14, 2.0);
    for (i, g) in bf16_noise(rows * NQ * HD, 15, 3.0).into_iter().enumerate() {
        qkv_host[(i / (NQ * HD)) * qkv_stride + NQ * HD + i % (NQ * HD)] = g;
    }
    let q = upload(gpu, &qkv_host)?;
    let gate = q.offset(NQ * HD * 2);
    let out = gpu.alloc(rows * NQ * HD * 2)?;
    let mut chunk = SparseChunk {
        hidden: hidden.offset(start * HIDDEN * 2),
        q,
        q_stride: qkv_stride as u32,
        out,
        gate: DevicePtr::NULL,
        gate_stride: 0,
        rows,
        start,
        attn_layer_idx: 0,
        hidden_size: HIDDEN as u32,
        eps,
        rope_theta: config.rope_theta as f32,
        rotary_dim: rotary,
        softmax_scale: 1.0 / (HD as f32).sqrt(),
        e4m3: false,
        num_query_heads: NQ,
    };
    qsa.prefill_sparse_attention(&chunk, &kv, meta_at(start), gpu, stream)?;
    gpu.synchronize(stream)?;
    let mut plain = vec![0u8; rows * NQ * HD * 2];
    gpu.copy_d2h(out, &mut plain)?;

    chunk.gate = gate;
    chunk.gate_stride = qkv_stride as u32;
    qsa.prefill_sparse_attention(&chunk, &kv, meta_at(start), gpu, stream)?;
    gpu.synchronize(stream)?;
    let mut gated = vec![0u8; rows * NQ * HD * 2];
    gpu.copy_d2h(out, &mut gated)?;

    let bf16_ordered = |bits: u16| -> i32 {
        if bits & 0x8000 != 0 {
            -((bits & 0x7fff) as i32)
        } else {
            bits as i32
        }
    };
    let (mut mismatches, mut nonfinite) = (0usize, 0usize);
    for i in 0..rows * NQ * HD {
        let o = bf16_to_f32(u16::from_le_bytes([plain[2 * i], plain[2 * i + 1]]));
        let gv = bf16_to_f32(qkv_host[(i / (NQ * HD)) * qkv_stride + NQ * HD + i % (NQ * HD)]);
        let want_f32 = o * (1.0 / (1.0 + (-gv).exp()));
        let want = half::bf16::from_f32(want_f32).to_bits();
        let got = u16::from_le_bytes([gated[2 * i], gated[2 * i + 1]]);
        let got_f32 = bf16_to_f32(got);
        if !o.is_finite() || !want_f32.is_finite() || !got_f32.is_finite() {
            nonfinite += 1;
        } else if (bf16_ordered(want) - bf16_ordered(got)).abs() > 1 {
            // One-ulp slack in sign-magnitude ordered space: host expf vs device __expf.
            mismatches += 1;
        }
    }
    println!(
        "qsa_prefill_smoke start={start} rows={rows}: non-finite={nonfinite}; gated vs bf16(o)*sigmoid(g) >1-ulp mismatches={mismatches}"
    );
    ensure!(nonfinite == 0 && mismatches == 0, "smoke FAILED");
    println!("smoke PASS");
    Ok(())
}
