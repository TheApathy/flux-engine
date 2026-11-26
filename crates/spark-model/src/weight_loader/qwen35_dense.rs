// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::{ModelWeightLoader, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::{Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::tp_shard::{
    TpAttentionDims, TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_quantized_nvfp4,
};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, Nvfp4Variant, QuantizeCtx, SsmWeights, dense,
    dense_auto, detect_nvfp4_variant, gpu_concat_rows, interleave_ba, load_kv_scales,
    load_ssm_qwen35, quantize_to_nvfp4, quantized_any,
};

#[path = "qwen35_dense/ffn.rs"]
mod ffn;

#[path = "qwen35_dense/memlean_release.rs"]
mod memlean_release;

#[cfg(test)]
#[path = "qwen35_dense/w3_integration_tests.rs"]
mod w3_integration_tests;

pub struct Qwen35DenseWeightLoader;

/// Release a replaced NVFP4 buffer after TP sharding: tombstone and free via
/// [`WeightStore::release_consumed_alias`] if `ptr` aliases a store entry, or
/// free directly via `gpu.free` if it was freshly allocated (e.g. from the
/// `quantized_any` FP8/BF16 fallback path).
fn release_ptr_or_store_alias(
    store: &WeightStore,
    candidate_keys: &[String],
    ptr: DevicePtr,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    if ptr.is_null() {
        return Ok(());
    }
    for key in candidate_keys {
        if let Ok(tensor) = store.get(key)
            && tensor.ptr == ptr
        {
            let dtype = tensor.dtype;
            store.release_consumed_alias(key, ptr, dtype, gpu)?;
            return Ok(());
        }
    }
    gpu.free(ptr)
}

/// Logical geometry of a temporary BF16 projection buffer paired with the full
/// pre-shard dimensions stored in [`WeightStore`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DequantSourceGeometry {
    local_rows: usize,
    local_cols: usize,
    store_rows: usize,
    store_cols: usize,
}

impl DequantSourceGeometry {
    fn unsharded(rows: usize, cols: usize) -> Self {
        Self {
            local_rows: rows,
            local_cols: cols,
            store_rows: rows,
            store_cols: cols,
        }
    }

    fn sharded(local_rows: usize, local_cols: usize, store_rows: usize, store_cols: usize) -> Self {
        Self {
            local_rows,
            local_cols,
            store_rows,
            store_cols,
        }
    }

    fn is_tp_sharded(self) -> bool {
        self.local_rows != self.store_rows || self.local_cols != self.store_cols
    }
}

/// Release a temporary BF16 expansion once a durable replacement exists.
///
/// Unsharded raw BF16 projections alias the immutable [`WeightStore`] and
/// remain owned by it. ModelOpt/CompressedTensors FP8 and NVFP4 projections —
/// as well as any TP-sharded BF16 projection (`tp_size > 1`) — are placed into
/// separately owned BF16 buffers by `load_ssm_qwen35` / `dense_auto`; those
/// construction-only buffers must be freed after concat/requantization.
fn release_owned_dequant_source(
    store: &WeightStore,
    prefix: &str,
    dense: DenseWeight,
    geom: DequantSourceGeometry,
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    ensure!(
        !dense.weight.is_null(),
        "replacement source {prefix} has a null BF16 pointer"
    );
    ensure!(
        geom.local_rows > 0 && geom.local_cols > 0 && geom.store_rows > 0 && geom.store_cols > 0,
        "replacement source {prefix} has empty logical geometry local=[{},{}] store=[{},{}]",
        geom.local_rows,
        geom.local_cols,
        geom.store_rows,
        geom.store_cols
    );

    // CompressedTensors uses `.weight_packed`; ModelOpt FP8/NVFP4 and raw
    // BF16 use `.weight`. This matches `load_ssm_qwen35`'s packed-first
    // detection when a compressed checkpoint carries both spellings.
    let weight_name = format!("{prefix}.weight");
    let packed_name = format!("{prefix}.weight_packed");
    let source_name = if store.contains(&packed_name) {
        &packed_name
    } else {
        &weight_name
    };
    let source = store.get(source_name)?;

    if source.ptr == dense.weight {
        ensure!(
            !geom.is_tp_sharded()
                && source.dtype == WeightDtype::BF16
                && source.shape.as_slice() == [geom.store_rows, geom.store_cols],
            "aliased replacement source {source_name} must be unsharded BF16 [{},{}], got {:?} {:?}",
            geom.store_rows,
            geom.store_cols,
            source.dtype,
            source.shape
        );
        return Ok(0);
    }

    match source.dtype {
        WeightDtype::BF16 if geom.is_tp_sharded() => ensure!(
            source.shape.as_slice() == [geom.store_rows, geom.store_cols],
            "TP-sharded BF16 replacement source {source_name} must be [{},{}], got {:?}",
            geom.store_rows,
            geom.store_cols,
            source.shape
        ),
        WeightDtype::FP8E4M3 => ensure!(
            source.shape.as_slice() == [geom.store_rows, geom.store_cols],
            "FP8 replacement source {source_name} must be [{},{}], got {:?}",
            geom.store_rows,
            geom.store_cols,
            source.shape
        ),
        WeightDtype::UInt8 => {
            ensure!(
                geom.store_cols.is_multiple_of(2),
                "packed replacement source {source_name} requires even logical K, got {}",
                geom.store_cols
            );
            ensure!(
                source.shape.as_slice() == [geom.store_rows, geom.store_cols / 2],
                "packed replacement source {source_name} must be [{},{}], got {:?}",
                geom.store_rows,
                geom.store_cols / 2,
                source.shape
            );
        }
        other => anyhow::bail!(
            "owned replacement source {prefix} must originate from {}FP8E4M3 or packed UInt8, got {other:?}",
            if geom.is_tp_sharded() {
                "TP-sharded BF16, "
            } else {
                ""
            }
        ),
    }

    let logical_bytes = geom
        .local_rows
        .checked_mul(geom.local_cols)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("replacement source {prefix} BF16 size overflows usize"))?;
    gpu.free(dense.weight)?;
    Ok(logical_bytes)
}

impl ModelWeightLoader for Qwen35DenseWeightLoader {
    fn supports_tp(&self) -> bool {
        // Both FullAttention layers (NVFP4-from-disk and BF16 → NVFP4 paths)
        // and LinearAttention (GDN SSM) layers (HeadParallel sharding in
        // `weight_map::load_ssm_qwen35`) are TP-sharded across ranks.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types = if config.layer_types.is_empty() {
            (0..config.num_hidden_layers)
                .map(|i| config.layer_type(i))
                .collect::<Vec<_>>()
        } else {
            config.layer_types.clone()
        };

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu
            .kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_weight_rne")
            .or_else(|_| gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4"))?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;

        let variant = detect_nvfp4_variant(store, config);
        let weight_format = WeightFormat::detect(store, config);
        let flashinfer_ffn_requested = crate::layers::prefill_ffn_flashinfer_enabled()?;
        let flashinfer_attn_proj_requested = crate::layers::prefill_proj_flashinfer_enabled()?;
        let flashinfer_ssm_proj_requested = crate::layers::prefill_ssm_flashinfer_enabled()?;
        let cublaslt_ssm_proj_requested = crate::layers::ssm_proj_cublaslt_enabled();
        ensure!(
            !flashinfer_ffn_requested || variant == Nvfp4Variant::Standard,
            "ATLAS_PREFILL_FFN_FLASHINFER=1 requires a Standard ModelOpt NVFP4 checkpoint"
        );
        ensure!(
            !flashinfer_attn_proj_requested || variant == Nvfp4Variant::Standard,
            "ATLAS_PREFILL_PROJ_FLASHINFER=1 requires a Standard ModelOpt NVFP4 checkpoint"
        );
        ensure!(
            !flashinfer_ssm_proj_requested || variant == Nvfp4Variant::Standard,
            "ATLAS_PREFILL_SSM_FLASHINFER=1 requires a Standard ModelOpt NVFP4 checkpoint"
        );
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        ensure!(
            !flashinfer_attn_proj_requested,
            "ATLAS_PREFILL_PROJ_FLASHINFER=1 requires a Linux CUDA build"
        );
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        ensure!(
            !flashinfer_ssm_proj_requested,
            "ATLAS_PREFILL_SSM_FLASHINFER=1 requires a Linux CUDA build"
        );
        tracing::info!(
            "Weight format: {:?}, NVFP4 variant: {:?}",
            weight_format,
            variant
        );

        let mut lean = memlean_release::DenseLean::from_env()?;
        let shared_splitk = lean.plan().shared_scratch.then(|| {
            crate::layers::shared_splitk::SharedSplitkWorkspace::for_max_n(
                h.max(config.intermediate_size)
                    .max(config.ssm_qkvz_size())
                    .max(config.num_key_value_heads * config.head_dim),
            )
        });
        let mut w3_session = ffn::prepare_w3_session(config, gpu)?;
        let ffn_context = ffn::DenseFfnLoadContext::new(
            store,
            config,
            gpu,
            variant,
            absmax_k,
            quantize_k,
            stream,
            flashinfer_ffn_requested,
            lean.plan(),
            shared_splitk.as_ref(),
        );

        // Fast engine recovery: serve the per-layer transposed NVFP4 copies
        // from disk instead of rebuilding them. No-op unless
        // ATLAS_WEIGHT_CACHE=1. `finish` runs only on the success path, so a
        // load that bails leaves the blob unpublished.
        super::transform_cache::init(
            store,
            config,
            gpu,
            &format!("qwen35_dense/{weight_format:?}/{variant:?}"),
        );

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            let (ffn, released_gate_up) = ffn_context.load(i, &lp, &mut w3_session)?;
            let mut released = crate::memlean::ReleasedBytes {
                ffn_gate_up: released_gate_up,
                ..crate::memlean::ReleasedBytes::default()
            };

            match lt {
                LayerType::FullAttention => {
                    // Allocation-receipt pass, summed over layers and logged
                    // with the target-layers ledger phase.
                    let _attention_pass = spark_runtime::alloc_receipt::pass("attention");
                    let p = format!("{lp}.self_attn");
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let kv_tp_rank = config.kv_tp_rank();
                    let kv_tp_size = config.kv_tp_size();
                    let (attn, q_nvfp4, k_nvfp4, v_nvfp4) = match variant {
                        // NVFP4 already packed in the checkpoint
                        // (compressed-tensors `.weight_packed` or
                        // modelopt uint8 `.weight` + `weight_scale_2`).
                        // `quantized_auto` reads either schema; we shard
                        // packed bytes directly. Treating modelopt as
                        // BF16-then-quantize previously crashed at the
                        // first full_attention layer because dense_auto
                        // returned a uint8 ptr aliased as BF16, and the
                        // absmax kernel read 2× the allocation.
                        Nvfp4Variant::CompressedTensors | Nvfp4Variant::Standard => {
                            // NVFP4-from-disk path: column-parallel Q/K/V, row-parallel O.
                            let group_size = 16usize;
                            let load_nvfp4 = |name: &str,
                                              full_n: usize,
                                              full_k: usize,
                                              kind: TpShardKind|
                             -> Result<crate::weight_map::QuantizedWeight> {
                                let prefix = format!("{p}.{name}");
                                let src = quantized_any(
                                    store,
                                    &prefix,
                                    full_n,
                                    full_k,
                                    gpu,
                                    variant,
                                    QuantizeCtx {
                                        absmax_k,
                                        quantize_k,
                                        stream,
                                    },
                                )?;
                                let (eff_rank, eff_size) = if matches!(name, "k_proj" | "v_proj") {
                                    (kv_tp_rank, kv_tp_size)
                                } else {
                                    (tp_rank, tp_size)
                                };
                                if eff_size == 1 {
                                    return Ok(src);
                                }
                                let sharded = shard_quantized_nvfp4(
                                    &src, full_n, full_k, kind, eff_rank, eff_size, group_size, gpu,
                                )?;
                                if sharded.weight != src.weight {
                                    release_ptr_or_store_alias(
                                        store,
                                        &[
                                            format!("{prefix}.weight_packed"),
                                            format!("{prefix}.weight"),
                                        ],
                                        src.weight,
                                        gpu,
                                    )?;
                                }
                                if sharded.weight_scale != src.weight_scale {
                                    release_ptr_or_store_alias(
                                        store,
                                        &[format!("{prefix}.weight_scale")],
                                        src.weight_scale,
                                        gpu,
                                    )?;
                                }
                                Ok(sharded)
                            };
                            let [q, k, v, o] = load_qkvo_tp(config, load_nvfp4)?;
                            let dummy = DenseWeight {
                                weight: DevicePtr::NULL,
                            };
                            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
                            let attn = AttentionWeights {
                                q_proj: dummy,
                                k_proj: dummy,
                                v_proj: dummy,
                                o_proj: o,
                                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                                q_norm_full: None,
                                k_norm_full: None,
                                k_scale,
                                v_scale,
                            };
                            (attn, Some(q), Some(k), Some(v))
                        }
                        Nvfp4Variant::Fp8Dequanted | Nvfp4Variant::Bf16Raw => {
                            // BF16 → NVFP4 path: shard BF16 then quantize per-rank.
                            let load_bf16_then_nvfp4 = |name: &str,
                                                        full_n: usize,
                                                        full_k: usize,
                                                        kind: TpShardKind|
                             -> Result<(
                                DenseWeight,
                                crate::weight_map::QuantizedWeight,
                            )> {
                                let src = dense_auto(store, &format!("{p}.{name}.weight"), gpu)?;
                                let (eff_rank, eff_size) = if matches!(name, "k_proj" | "v_proj") {
                                    (kv_tp_rank, kv_tp_size)
                                } else {
                                    (tp_rank, tp_size)
                                };
                                let (sharded_ptr, local_n, local_k) = shard_dense_bf16(
                                    src.weight, full_n, full_k, kind, eff_rank, eff_size, gpu,
                                )?;
                                let sharded = DenseWeight {
                                    weight: sharded_ptr,
                                };
                                let q = quantize_to_nvfp4(
                                    &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
                                )?;
                                if sharded_ptr != src.weight {
                                    gpu.free(sharded_ptr)?;
                                }
                                Ok((src, q))
                            };
                            let [
                                (q_dense, q_nvfp4),
                                (k_dense, k_nvfp4),
                                (v_dense, v_nvfp4),
                                (o_dense, o_nvfp4),
                            ] = load_qkvo_tp(config, load_bf16_then_nvfp4)?;
                            // `o_proj` executes via `o_nvfp4`; release its
                            // unsharded BF16 dequant buffer (if freshly
                            // allocated from FP8) instead of leaking it.
                            let attn_dims = TpAttentionDims::from_config(config);
                            release_owned_dequant_source(
                                store,
                                &format!("{p}.o_proj"),
                                o_dense,
                                DequantSourceGeometry::unsharded(attn_dims.h, attn_dims.full_o_in),
                                gpu,
                            )?;

                            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

                            let attn = AttentionWeights {
                                q_proj: q_dense,
                                k_proj: k_dense,
                                v_proj: v_dense,
                                o_proj: o_nvfp4,
                                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                                q_norm_full: None,
                                k_norm_full: None,
                                k_scale,
                                v_scale,
                            };
                            (attn, Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4))
                        }
                    };

                    let mut layer = Qwen3AttentionLayer::new(
                        input_norm,
                        attn,
                        post_attn_norm,
                        ffn,
                        attn_idx,
                        q_nvfp4,
                        k_nvfp4,
                        v_nvfp4,
                        gpu,
                        layer_kv_dtypes[attn_idx],
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;

                    // Transpose NVFP4 weights for the small-M prefill /
                    // K=γ verify GEMM path (`w4a16_gemm_t_m16`). Without
                    // these, the multi_seq decode path falls back to
                    // per-token GEMV at γ=16 (3 × 17 launches/layer).
                    let num_heads = config.num_attention_heads;
                    let num_kv_heads = config.num_key_value_heads;
                    let head_dim = config.head_dim;
                    let gated = config.attn_gated;
                    let q_proj_n = if gated {
                        num_heads * head_dim * 2
                    } else {
                        num_heads * head_dim
                    };
                    if let Some(ref qw) = q_nvfp4 {
                        let qt = qw.transpose_for_gemm_cached(
                            gpu,
                            q_proj_n,
                            h,
                            &format!("L{i}.self_attn.q_proj.t"),
                        )?;
                        let kt = k_nvfp4.as_ref().unwrap().transpose_for_gemm_cached(
                            gpu,
                            num_kv_heads * head_dim,
                            h,
                            &format!("L{i}.self_attn.k_proj.t"),
                        )?;
                        let vt = v_nvfp4.as_ref().unwrap().transpose_for_gemm_cached(
                            gpu,
                            num_kv_heads * head_dim,
                            h,
                            &format!("L{i}.self_attn.v_proj.t"),
                        )?;
                        let ot = layer.attn.o_proj.transpose_for_gemm_cached(
                            gpu,
                            h,
                            num_heads * head_dim,
                            &format!("L{i}.self_attn.o_proj.t"),
                        )?;
                        layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
                        // Eagerly allocate the QKV/O split-K FP32 workspace
                        // (illegal during CUDA graph capture). Sized for
                        // `max(kv_dim, h)` so `o_proj` (`N = h`) can share it
                        // cleanly alongside `k_proj`/`v_proj`. No-op when
                        // `attn_qkv_splitk()` resolves to 0 (explicit 0/1, or
                        // unset on <=64-SM GPUs) or split-K kernels are missing.
                        layer.alloc_qkv_splitk_workspace(
                            gpu,
                            (num_kv_heads * head_dim).max(h) as u32,
                            shared_splitk.as_ref(),
                        )?;
                    }

                    #[cfg(all(feature = "cuda", target_os = "linux"))]
                    if flashinfer_attn_proj_requested {
                        layer.prepare_flashinfer_projection_prefill(gpu, i, variant, config)?;
                    }
                    lean.finish_attention_layer(store, &p, gpu, stream, &mut released)?;

                    layers.push(Box::new(layer));
                    attn_idx += 1;
                }
                LayerType::LinearAttention => {
                    let convert_pass = spark_runtime::alloc_receipt::pass("ssm/convert");
                    let nv = config.linear_num_value_heads;
                    let nk = config.linear_num_key_heads;
                    let qkv_rows = config.ssm_qkv_size();
                    let z_rows = config.ssm_z_size();
                    let value_dim = nv * config.linear_value_head_dim;

                    // load_ssm_qwen35 returns BF16 dense regardless of source
                    // quantization (modelopt uint8 .weight, compressed-tensors
                    // .weight_packed, or raw BF16). Including in_proj_a / b /
                    // out_proj which previous versions of this loader assumed
                    // were always BF16 — that assumption breaks on Sehyo /
                    // Huihui-style NVFP4 checkpoints which pack all five SSM
                    // projections.
                    let ssm35 = load_ssm_qwen35(store, &lp, gpu, variant, config)?;
                    let qkv_dense = ssm35.in_proj_qkv;
                    let z_dense = ssm35.in_proj_z;
                    let out_proj_dense = ssm35.out_proj;
                    let in_proj_a = ssm35.in_proj_a;
                    let in_proj_b = ssm35.in_proj_b;
                    let conv1d = ssm35.conv1d;
                    let a_log = ssm35.a_log;
                    let dt_bias = ssm35.dt_bias;
                    let norm = ssm35.norm;

                    let qkvz_dense =
                        gpu_concat_rows(&qkv_dense, qkv_rows, &z_dense, z_rows, h, gpu)?;

                    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;

                    // QKVZ and BA are durable copies. Release only separately
                    // allocated BF16 dequant/shard buffers; unsharded raw BF16
                    // store aliases remain untouched. Without this, the
                    // official mixed ModelOpt checkpoint accumulates roughly
                    // 10 GiB of dead qkv/z/out expansions over its 48
                    // linear-attention layers.
                    let tp_size = config.tp_world_size.max(1);
                    let mut released_dequant_bytes = 0usize;
                    released_dequant_bytes += release_owned_dequant_source(
                        store,
                        &format!("{lp}.linear_attn.in_proj_qkv"),
                        qkv_dense,
                        DequantSourceGeometry::sharded(qkv_rows, h, qkv_rows * tp_size, h),
                        gpu,
                    )?;
                    released_dequant_bytes += release_owned_dequant_source(
                        store,
                        &format!("{lp}.linear_attn.in_proj_z"),
                        z_dense,
                        DequantSourceGeometry::sharded(z_rows, h, z_rows * tp_size, h),
                        gpu,
                    )?;
                    released_dequant_bytes += release_owned_dequant_source(
                        store,
                        &format!("{lp}.linear_attn.in_proj_a"),
                        in_proj_a,
                        DequantSourceGeometry::sharded(nv, h, nv * tp_size, h),
                        gpu,
                    )?;
                    released_dequant_bytes += release_owned_dequant_source(
                        store,
                        &format!("{lp}.linear_attn.in_proj_b"),
                        in_proj_b,
                        DequantSourceGeometry::sharded(nv, h, nv * tp_size, h),
                        gpu,
                    )?;

                    let qkvz_size = config.ssm_qkvz_size();
                    let qkvz_nvfp4 = quantize_to_nvfp4(
                        &qkvz_dense,
                        qkvz_size,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;

                    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm_cached(
                        gpu,
                        qkvz_size,
                        h,
                        &format!("L{i}.linear_attn.qkvz.t"),
                    )?;

                    let out_proj_nvfp4 = quantize_to_nvfp4(
                        &out_proj_dense,
                        h,
                        value_dim,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;

                    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm_cached(
                        gpu,
                        h,
                        value_dim,
                        &format!("L{i}.linear_attn.out_proj.t"),
                    )?;

                    let ssm = SsmWeights {
                        in_proj_qkvz: qkvz_dense,
                        in_proj_ba: ba_dense,
                        conv1d,
                        a_log,
                        dt_bias,
                        norm,
                        out_proj: out_proj_nvfp4,
                    };

                    let mut layer = Qwen3SsmLayer::new_sequential_with_splitk(
                        input_norm,
                        ssm,
                        post_attn_norm,
                        ffn,
                        Some(qkvz_nvfp4),
                        Some(qkvz_nvfp4_t),
                        Some(out_proj_nvfp4_t),
                        config,
                        gpu,
                        shared_splitk.as_ref(),
                    )?;
                    drop(convert_pass);
                    if cublaslt_ssm_proj_requested {
                        let _e4m3_pass = spark_runtime::alloc_receipt::pass("ssm/qkvz-e4m3");
                        layer
                            .prepare_qkvz_e4m3_cache(gpu, config, stream)
                            .with_context(|| {
                                format!("prepare eager SSM QKVZ e4m3 cache for layer {i}")
                            })?;
                    }
                    lean.adopt_ssm_scratch(&mut layer)?;
                    let _prefill_pass =
                        spark_runtime::alloc_receipt::pass("ssm/prefill-fp8+release");
                    layer.predequant_for_prefill(gpu, config, stream)?;
                    #[cfg(all(feature = "cuda", target_os = "linux"))]
                    if flashinfer_ssm_proj_requested {
                        layer.prepare_flashinfer_ssm_prefill(gpu, i, config)?;
                    }
                    released_dequant_bytes += release_owned_dequant_source(
                        store,
                        &format!("{lp}.linear_attn.out_proj"),
                        out_proj_dense,
                        DequantSourceGeometry::sharded(h, value_dim, h, value_dim * tp_size),
                        gpu,
                    )?;
                    if released_dequant_bytes != 0 {
                        tracing::debug!(
                            "SSM[{lp}] released {:.2} MiB of construction-only BF16 dequants",
                            released_dequant_bytes as f64 / (1024.0 * 1024.0)
                        );
                    }
                    lean.finish_ssm_layer(
                        store,
                        &lp,
                        &mut layer,
                        qkvz_size * h * 2,
                        gpu,
                        stream,
                        &mut released,
                    )?;
                    layers.push(Box::new(layer));
                }
                LayerType::Moe => unreachable!("Qwen3.5 dense has no standalone MoE layers"),
            }
            lean.record(i, released);

            if (i + 1) % 10 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
                spark_runtime::progress::layer(i + 1, config.num_hidden_layers);
            }
        }

        tracing::info!(
            "Qwen3.5 dense weight loader: {} layers ({} attention, {} SSM, dense FFN)",
            layers.len(),
            attn_idx,
            layers.len() - attn_idx,
        );

        lean.log_total(layers.len());
        ffn::finish_w3_session(w3_session)?;
        super::transform_cache::finish();

        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.norm.weight"))
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        super::qwen35_mixed_precision::load_lm_head(store, config, gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // Dense models don't have MoE-shaped MTP — see `load_mtp_dense_weights`.
        Ok(None)
    }

    fn load_mtp_dense_weights(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::weight_map::MtpDenseWeights>> {
        if !store.contains("mtp.fc.weight") {
            return Ok(None);
        }
        tracing::info!("Loading dense MTP weights (Qwen3.5/3.6 27B-class)...");
        let mtp = crate::weight_map::load_mtp_dense(store)?;
        if mtp.is_some() {
            tracing::info!(
                "Dense MTP weights loaded: 1 attention layer + dense MLP (no expert routing)"
            );
        }
        Ok(mtp)
    }

    /// Load the ViT tower for dense Qwen3.5/3.6 checkpoints that ship one.
    ///
    /// The dense weight loader handles the text-only Qwen3.6-27B-FP8 sibling
    /// (no vision) AND vision-capable dense checkpoints like AEON-Q36-27B,
    /// which carry the full 333-tensor `model.visual.*` tower in the same
    /// `qwen3_5` + `num_experts==0` config that routes here (see
    /// `factory::loader_for_config`). The default trait impl returns `None`,
    /// which silently dropped the vision tower for the latter — the image
    /// `<|image_pad|>` token was then embedded as plain text, producing
    /// fluent-but-wrong "vision" output. The ViT load logic is identical to
    /// the MoE-VL sibling (same `model.visual.*` / nested layout, same FP8→
    /// BF16 auto-dequant), so delegate to it. It returns `Ok(None)` when
    /// `config.vision` is `None` (the text-only dense path), keeping that
    /// path byte-for-byte unchanged.
    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionEncoder>> {
        super::Qwen35WeightLoader.load_vision_encoder(store, config, gpu)
    }
}

#[cfg(test)]
mod dequant_release_tests {
    use std::collections::HashMap;

    use spark_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::{DenseWeight, DequantSourceGeometry, release_owned_dequant_source};

    fn store(name: &str, ptr: DevicePtr, shape: Vec<usize>, dtype: WeightDtype) -> WeightStore {
        WeightStore::from_map(HashMap::from([(
            name.to_string(),
            WeightTensor { ptr, shape, dtype },
        )]))
    }

    #[test]
    fn raw_bf16_alias_remains_store_owned() {
        let gpu = MockGpuBackend::new();
        let ptr = gpu.alloc(16).unwrap();
        let store = store("p.weight", ptr, vec![2, 4], WeightDtype::BF16);

        let released = release_owned_dequant_source(
            &store,
            "p",
            DenseWeight { weight: ptr },
            DequantSourceGeometry::unsharded(2, 4),
            &gpu,
        )
        .unwrap();

        assert_eq!(released, 0);
        assert_eq!(gpu.alloc_count(), 1);
        assert!(store.contains("p.weight"));
    }

    #[test]
    fn owned_fp8_and_packed_expansions_are_freed() {
        for (name, shape, dtype) in [
            ("p.weight", vec![2, 4], WeightDtype::FP8E4M3),
            ("p.weight_packed", vec![2, 2], WeightDtype::UInt8),
        ] {
            let gpu = MockGpuBackend::new();
            let source = gpu.alloc(8).unwrap();
            let expanded = gpu.alloc(16).unwrap();
            let store = store(name, source, shape, dtype);

            assert_eq!(
                release_owned_dequant_source(
                    &store,
                    "p",
                    DenseWeight { weight: expanded },
                    DequantSourceGeometry::unsharded(2, 4),
                    &gpu,
                )
                .unwrap(),
                16
            );
            assert_eq!(gpu.alloc_count(), 1);
            assert!(store.contains(name));
        }
    }

    #[test]
    fn tp_sharded_bf16_fp8_and_packed_owned_shards_are_freed() {
        // Column-parallel TP=2: local [2, 4] from full store [4, 4] (packed [4, 2]).
        for (name, shape, dtype, store_bytes) in [
            ("p.weight", vec![4, 4], WeightDtype::BF16, 32),
            ("p.weight", vec![4, 4], WeightDtype::FP8E4M3, 16),
            ("p.weight_packed", vec![4, 2], WeightDtype::UInt8, 8),
        ] {
            let gpu = MockGpuBackend::new();
            let source = gpu.alloc(store_bytes).unwrap();
            let sharded = gpu.alloc(16).unwrap();
            let store = store(name, source, shape, dtype);

            assert_eq!(
                release_owned_dequant_source(
                    &store,
                    "p",
                    DenseWeight { weight: sharded },
                    DequantSourceGeometry::sharded(2, 4, 4, 4),
                    &gpu,
                )
                .unwrap(),
                16
            );
            assert_eq!(gpu.alloc_count(), 1);
            assert!(store.contains(name));
        }

        // Row-parallel TP=2: local [4, 2] from full store [4, 4] (packed [4, 2]).
        let gpu = MockGpuBackend::new();
        let source = gpu.alloc(32).unwrap();
        let sharded = gpu.alloc(16).unwrap();
        let store = store("out.weight", source, vec![4, 4], WeightDtype::BF16);
        assert_eq!(
            release_owned_dequant_source(
                &store,
                "out",
                DenseWeight { weight: sharded },
                DequantSourceGeometry::sharded(4, 2, 4, 4),
                &gpu,
            )
            .unwrap(),
            16
        );
        assert_eq!(gpu.alloc_count(), 1);
    }

    #[test]
    fn hostile_ownership_and_geometry_fail_before_free() {
        let gpu = MockGpuBackend::new();
        let fp8 = gpu.alloc(8).unwrap();
        let fp8_store = store("p.weight", fp8, vec![2, 4], WeightDtype::FP8E4M3);
        assert!(
            release_owned_dequant_source(
                &fp8_store,
                "p",
                DenseWeight { weight: fp8 },
                DequantSourceGeometry::unsharded(2, 4),
                &gpu,
            )
            .is_err()
        );

        let bf16 = gpu.alloc(32).unwrap();
        let bf16_store = store("s.weight", bf16, vec![4, 4], WeightDtype::BF16);
        assert!(
            release_owned_dequant_source(
                &bf16_store,
                "s",
                DenseWeight { weight: bf16 },
                DequantSourceGeometry::sharded(2, 4, 4, 4),
                &gpu,
            )
            .is_err()
        );

        let malformed = gpu.alloc(8).unwrap();
        let expanded = gpu.alloc(16).unwrap();
        let malformed_store = store("q.weight", malformed, vec![2, 3], WeightDtype::FP8E4M3);
        assert!(
            release_owned_dequant_source(
                &malformed_store,
                "q",
                DenseWeight { weight: expanded },
                DequantSourceGeometry::unsharded(2, 4),
                &gpu,
            )
            .is_err()
        );
        assert!(
            release_owned_dequant_source(
                &malformed_store,
                "q",
                DenseWeight {
                    weight: DevicePtr::NULL,
                },
                DequantSourceGeometry::unsharded(2, 4),
                &gpu,
            )
            .is_err()
        );
        assert_eq!(gpu.alloc_count(), 4);
    }
}
