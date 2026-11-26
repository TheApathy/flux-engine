// SPDX-License-Identifier: AGPL-3.0-only

mod attention_arms;
mod attn_sliding_window;
mod dense_dedup;
mod dense_fp8;
mod linear_attn_arms;
mod mxfp8_release;
mod tq_plus_weight_rotation;

use anyhow::{Context, Result};
use atlas_core::config::{LayerType, ModelConfig, ModelOptWeightFormat};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::super::{ModelWeightLoader, QuantFormat, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::moe::qwen4_router_bf16::{
    QWEN4_ROUTER_BF16_ENV, RouterPrecision, StoredRouter, layer_router_precision,
    selected as router_bf16_selected, validate_stored_bf16_router,
};
use crate::layers::{FfnComponent, MoeLayer, Qwen3AttentionLayer, Qwen4HyperConnection};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_fp8_block_scaled};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Nvfp4Variant, QuantizedWeight, dense_auto, dense_f32_safe,
    detect_nvfp4_variant, load_fp8_block_scaled_as_fp8weight, load_kv_scales, load_moe_qwen35,
    load_moe_qwen35_fp8_experts, quantize_to_nvfp4,
};

thread_local! {
    /// Set while a caller that runs the phased unified-layout MoE transpose
    /// itself loads its layers (the Qwen4 MTP layer under
    /// `ATLAS_QWEN4_MTP_UNIFIED_T=1`). The load-time `transpose_for_prefill`,
    /// which keeps the untransposed experts, is then skipped.
    static DEFER_MOE_TRANSPOSE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Scoped deferral of the load-time MoE transpose (see `DEFER_MOE_TRANSPOSE`).
pub(super) struct DeferMoeTranspose {
    previous: bool,
}

impl DeferMoeTranspose {
    pub(super) fn enter() -> Self {
        Self {
            previous: DEFER_MOE_TRANSPOSE.with(|flag| flag.replace(true)),
        }
    }
}

impl Drop for DeferMoeTranspose {
    fn drop(&mut self) {
        let previous = self.previous;
        DEFER_MOE_TRANSPOSE.with(|flag| flag.set(previous));
    }
}

pub(super) fn load_layers(
    loader: &dyn ModelWeightLoader,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    ensure_supported_weight_formats(config)?;
    let dense_dedup_layers = config.is_qwen4_exp() && dense_dedup::selected();
    // ATLAS_QWEN4_RELEASE_MXFP8_SOURCES (default on): release the checkpoint
    // MXFP8 store originals once their one-time dequant/requantize to NVFP4
    // has run (see `mxfp8_release` module doc). Independent of
    // `ATLAS_QWEN4_DENSE_DEDUP` above. A tensor a native-dense reader pins is
    // skipped per name (`source_pins`, MXFP8_RELEASE_SKIPPED).
    let release_mxfp8_sources = config.is_qwen4_exp() && mxfp8_release::selected()?;

    let layer_types = if config.layer_types.is_empty() {
        (0..config.num_hidden_layers)
            .map(|i| config.layer_type(i))
            .collect::<Vec<_>>()
    } else {
        config.layer_types.clone()
    };

    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    let mut attn_idx = 0usize;
    let router_precision = layer_router_precision(
        router_bf16_selected()?,
        config.is_qwen4_exp(),
        config.weight_prefix == "mtp",
    )?;
    let shared_precision = crate::layers::moe::qwen4_shared_bf16::layer_shared_precision(
        crate::layers::moe::qwen4_shared_bf16::selected()?,
        config.is_qwen4_exp(),
        config.weight_prefix == "mtp",
    )?;
    // ATLAS_QWEN4_DENSE_FP8=1 (default off): extra native MXFP8 decode operands.
    let dense_fp8 = dense_fp8::engaged(config)?;
    if dense_fp8 {
        dense_fp8::reset_ledger();
    }

    // Per-(layer, role) precision schedule hook. Per-site overrides are not
    // yet wired into Qwen35WeightLoader; warn loudly if a loader supplies any
    // non-Inherit override so operators are not misled into thinking it took
    // effect.
    let precision = loader.precision_schedule(config);
    if precision.has_any_override() {
        tracing::warn!(
            "Precision schedule {:?} specifies overrides, but per-site precision \
             overrides are not yet wired in Qwen35WeightLoader; using checkpoint dtypes",
            precision,
        );
    }

    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu
        .kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_weight_rne")
        .or_else(|_| gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4"))?;
    let stream = gpu.default_stream();
    let h = config.hidden_size;

    // Detect weight format and quantization strategy.
    let variant = detect_nvfp4_variant(store, config);
    let weight_format = WeightFormat::detect(store, config);

    // Resolve runtime quantization format from the detected on-disk
    // variant. This determines which kernels are used for
    // decode/prefill/verify.
    let quant_format = if variant == Nvfp4Variant::Fp8Dequanted {
        QuantFormat::Fp8
    } else {
        QuantFormat::Nvfp4
    };
    let native_fp8 = quant_format == QuantFormat::Fp8;
    tracing::info!(
        "Weight format: {:?}, NVFP4 variant: {:?}, quant_format: {:?}",
        weight_format,
        variant,
        quant_format,
    );

    // Fast engine recovery — see `weight_loader::transform_cache`. No-op
    // unless ATLAS_WEIGHT_CACHE=1.
    crate::weight_loader::transform_cache::init(
        store,
        config,
        gpu,
        &format!("qwen35_moe/{weight_format:?}/{variant:?}/{quant_format:?}"),
    );

    // Estimate MoE transpose memory: 3 projections × local_experts × (packed + scale) per layer.
    // Skip transposition if GPU memory is insufficient — fallback grouped GEMM is used instead.
    // A caller that runs the phased unified transpose itself defers it (no budget check).
    let defer_to_m2_setup = config.is_qwen4_exp()
        || (config.model_type == "minimax_m2"
            && std::env::var("ATLAS_UNIFIED_MOE_LAYOUT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false));
    let skip_moe_transpose = DEFER_MOE_TRANSPOSE.with(|flag| flag.get()) || defer_to_m2_setup || {
        let inter = config.moe_intermediate_size;
        let group_size = 16usize;
        // gate/up: [inter, h/2] packed + [inter, h/group] scale
        let gu_bytes = inter * h / 2 + inter * h / group_size;
        // down:    [h, inter/2] packed + [h, inter/group] scale
        let d_bytes = h * inter / 2 + h * inter / group_size;
        let local_experts = (0..config.num_experts)
            .filter(|&e| config.is_local_expert(e))
            .count();
        let per_layer = local_experts * (2 * gu_bytes + d_bytes);
        let total = per_layer * layer_types.len();
        let available = gpu.free_memory().unwrap_or(0);
        let headroom = 2 * 1024 * 1024 * 1024; // 2 GiB for KV cache + buffers
        let skip = total > available.saturating_sub(headroom);
        if skip {
            tracing::warn!(
                "Skipping MoE weight transposition ({:.1} GiB needed, {:.1} GiB available). \
                     Prefill will use fallback grouped GEMM.",
                total as f64 / (1024.0 * 1024.0 * 1024.0),
                available as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        skip
    };

    for (i, lt) in layer_types.iter().enumerate() {
        let lp = config.layer_prefix(i);
        let (input_norm, post_attn_norm) = if config.is_qwen4_exp() {
            // These fields are unused once the hyperconnection path is
            // installed, but the shared Qwen3 core constructors require a
            // DenseWeight. Alias a real offset-RMS tensor rather than inventing
            // an allocation.
            let norm = dense_f32_safe(
                store,
                &format!("{lp}.attn_hyper_connection.hc_norm.weight"),
                gpu,
            )?;
            (norm, norm)
        } else {
            (
                dense_f32_safe(store, &format!("{lp}.input_layernorm.weight"), gpu)?,
                dense_f32_safe(store, &format!("{lp}.post_attention_layernorm.weight"), gpu)?,
            )
        };

        // When native_fp8, skip NVFP4 routed experts — FP8 fused batch1/2/3
        // kernels handle all MoE dispatch including MTP verify.
        // Saves ~33 GB on 122B EP=2, enabling FP8+MTP within memory budget.
        //
        // Diagnostic env: ATLAS_FORCE_NVFP4_MOE=1 forces the NVFP4 path even
        // for FP8 models — used to localize FP8 grouped-GEMM amplification
        // bug (L0 moe_out 3.3x too large vs HF). Keeps NVFP4 experts loaded
        // AND skips set_fp8_experts so forward dispatch falls through to the
        // NVFP4 path.
        let force_nvfp4_moe = std::env::var("ATLAS_FORCE_NVFP4_MOE").ok().as_deref() == Some("1");
        let skip_nvfp4_experts = native_fp8 && !force_nvfp4_moe;
        if skip_nvfp4_experts {
            tracing::info!(
                "FP8: skipping NVFP4 routed experts (FP8 fused MoE batch1/2/3 handles all dispatch)"
            );
        } else if native_fp8 && force_nvfp4_moe {
            tracing::warn!(
                "ATLAS_FORCE_NVFP4_MOE=1: routing MoE through NVFP4 path (diagnostic — slower)"
            );
        }
        // Allocation-receipt passes (`STARTUP_ALLOC_RECEIPT ... pass=`): summed
        // over layers, logged with the enclosing ledger phase.
        let moe_pass = spark_runtime::alloc_receipt::pass("moe");
        let moe_weights = load_moe_qwen35(
            store,
            &lp,
            config.num_experts,
            gpu,
            config,
            variant,
            absmax_k,
            quantize_k,
            stream,
            skip_nvfp4_experts,
            shared_precision == crate::layers::moe::qwen4_shared_bf16::SharedPrecision::Bf16,
        )?;
        // ATLAS_QWEN4_PREFILL_NATIVE=1 family shared: exact-MXFP8 BF16 prefill copy.
        crate::layers::qwen4_prefill_native::shared::register_layer(
            store,
            gpu,
            config,
            &lp,
            moe_weights.shared_expert_gate.weight,
            // ATLAS_QWEN4_SHARED_BF16=1 already holds the same exact BF16
            // tensors: alias them (0 new bytes) instead of a second copy.
            moe_weights
                .shared_bf16
                .as_ref()
                .map(|s| [s.gate.weight, s.up.weight, s.down.weight]),
        )?;
        let gate_nvfp4 = if router_precision == RouterPrecision::Bf16 {
            let key = format!("{lp}.mlp.gate.weight");
            let stored = store.get(&key)?;
            validate_stored_bf16_router(
                &key,
                StoredRouter {
                    dtype: stored.dtype,
                    shape: &stored.shape,
                    stored_ptr: stored.ptr.0,
                    loaded_ptr: moe_weights.gate.weight.0,
                },
                config.num_experts,
                h,
            )?;
            if i == 0 {
                tracing::info!(
                    "QWEN4_ROUTER_BF16_ENGAGED selector={QWEN4_ROUTER_BF16_ENV} layers={} \
                     experts={} hidden={h} stored={:?} decode=dense_gemv_bf16 prefill=dense_gemm_bf16",
                    layer_types.len(),
                    config.num_experts,
                    stored.dtype
                );
            }
            None
        } else if config.is_qwen4_exp() {
            Some(crate::weight_map::quantize_to_nvfp4_cached(
                &moe_weights.gate,
                config.num_experts,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
                &format!("{lp}.moe_gate.nvfp4"),
            )?)
        } else {
            Some(quantize_to_nvfp4(
                &moe_weights.gate,
                config.num_experts,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?)
        };
        if gate_nvfp4.is_some() && config.is_qwen4_exp() {
            // ATLAS_QWEN4_PREFILL_NATIVE=1: prefill routes on the resident BF16 gate.
            let retained = crate::layers::qwen4_prefill_native::store_retains_bf16(
                store,
                &format!("{lp}.mlp.gate.weight"),
                moe_weights.gate.weight,
            );
            crate::layers::qwen4_prefill_native::register_router(
                config,
                moe_weights.gate.weight,
                config.num_experts,
                h,
                retained,
            )?;
        }
        let mut moe_layer =
            MoeLayer::new(moe_weights, config.num_experts, gate_nvfp4, gpu, config)?;
        if let Some(groups) = store.routed_original_groups(&lp) {
            moe_layer.attach_routed_original_groups(groups)?;
        }
        // ATLAS_QWEN4_DENSE_DEDUP=1: the NVFP4 router and shared expert are
        // built; their BF16 router alias and MXFP8 store originals are unread.
        if dense_dedup_layers && dense_dedup::is_target_layer_prefix(&lp) {
            gpu.synchronize(stream)?;
            let has_nvfp4_router = moe_layer.has_nvfp4_router();
            let bytes = dense_dedup::DedupBytes {
                store_originals: dense_dedup::release_router_bf16(
                    store,
                    &lp,
                    &mut moe_layer.weights.gate,
                    has_nvfp4_router,
                    gpu,
                )? + dense_dedup::release_shared_expert_sources(store, &lp, gpu)?,
                bf16_copies: 0,
            };
            dense_dedup::log_layer(&lp, "moe-router-shared", bytes);
        }
        if release_mxfp8_sources && dense_dedup::is_target_layer_prefix(&lp) {
            gpu.synchronize(stream)?;
            let bytes = mxfp8_release::release_shared_expert(store, &lp, gpu)?;
            mxfp8_release::log_layer(&lp, "shared-expert", bytes);
        }
        // Phase 2.7 Tier C: flag DFlash capture layers so the MoE forward
        // can dispatch the Frankenstein kernel route (env-var-gated). The
        // capture-layer indices are already offset-adjusted in factory.rs
        // before being placed on `config.dflash_capture_layers`.
        moe_layer.is_dflash_capture_layer = config.dflash_capture_layers.contains(&i);
        // With native FP8, the FP8 fused MoE kernel handles both prefill and decode.
        // Skip transposition and predequant (saves ~30 GB + CPU time for 122B EP=2).
        // ATLAS_FORCE_NVFP4_MOE=1 inverts: do the prep so NVFP4 path is usable.
        if (!native_fp8 || force_nvfp4_moe) && !skip_moe_transpose {
            moe_layer.transpose_for_prefill(gpu, config)?;
        }
        if !native_fp8 || force_nvfp4_moe {
            moe_layer.predequant_for_prefill(gpu, config, stream)?;
        }

        // Native FP8 MoE: load FP8 expert weights for decode/prefill.
        // Because `skip_nvfp4_experts` skipped loading NVFP4 routed experts,
        // no NVFP4 fallback exists and every failure here must fail closed.
        if skip_nvfp4_experts {
            let fp8_experts =
                load_moe_qwen35_fp8_experts(store, &lp, config.num_experts, gpu, config)
                    .with_context(|| format!("Layer {i}: failed to load native FP8 MoE experts"))?;
            let sp = format!("{lp}.mlp.shared_expert");
            use crate::weight_map::Fp8ExpertWeight as FEW;
            let shared_fp8 = FEW {
                gate_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{sp}.gate_proj"),
                    gpu,
                )
                .with_context(|| {
                    format!("Layer {i}: failed to load shared expert FP8 gate_proj")
                })?,
                up_proj: load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.up_proj"), gpu)
                    .with_context(|| {
                    format!("Layer {i}: failed to load shared expert FP8 up_proj")
                })?,
                down_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{sp}.down_proj"),
                    gpu,
                )
                .with_context(|| {
                    format!("Layer {i}: failed to load shared expert FP8 down_proj")
                })?,
            };
            moe_layer
                .set_fp8_experts(&fp8_experts, shared_fp8, gpu)
                .with_context(|| format!("Layer {i}: failed to build FP8 expert pointer tables"))?;
            tracing::info!("Layer {i}: MoE experts loaded as native FP8");
        }

        if dense_fp8 {
            anyhow::ensure!(
                !native_fp8,
                "ATLAS_QWEN4_DENSE_FP8=1 refused: layer {i} uses the native-FP8 MoE route"
            );
            moe_layer.set_qwen4_shared_fp8(dense_fp8::load_shared(store, &lp, config, gpu)?);
        }
        let ffn = FfnComponent::Moe(moe_layer);
        drop(moe_pass);

        match lt {
            LayerType::FullAttention if native_fp8 => {
                let _attention_pass = spark_runtime::alloc_receipt::pass("attention");
                // ── Native FP8 path: FP8 for both decode AND prefill ──
                // NO NVFP4 dequant — saves ~30 GB peak memory on 122B EP=2.
                // Decode uses w8a16_gemv, prefill uses w8a16_gemm (both with
                // E4M3 LUT + BF16 2D block scales from checkpoint).
                let p = format!("{lp}.self_attn");
                tracing::info!("Layer {i}: loading attention FP8 native (zero-copy)");

                // FP8 block-scaled QKVO: column-parallel Q/K/V, row-parallel O.
                // Block size is 128 for Qwen3.5 native FP8 checkpoints.
                let tp_rank = config.tp_rank;
                let tp_size = config.tp_world_size.max(1);
                let kv_tp_rank = config.kv_tp_rank();
                let kv_tp_size = config.kv_tp_size();
                let block_size = 128usize;
                let load_fp8_proj = |name: &str,
                                     _full_n: usize,
                                     _full_k: usize,
                                     kind: TpShardKind|
                 -> Result<crate::weight_map::Fp8Weight> {
                    let prefix = format!("{p}.{name}");
                    let src = load_fp8_block_scaled_as_fp8weight(store, &prefix, gpu)?;
                    let (eff_rank, eff_size) = if matches!(name, "k_proj" | "v_proj") {
                        (kv_tp_rank, kv_tp_size)
                    } else {
                        (tp_rank, tp_size)
                    };
                    if eff_size == 1 {
                        return Ok(src);
                    }
                    let weight_key = format!("{prefix}.weight");
                    let scale_key = format!("{prefix}.weight_scale_inv");
                    let weight_dtype = store.get(&weight_key)?.dtype;
                    let scale_dtype = store.get(&scale_key)?.dtype;
                    let sharded =
                        shard_fp8_block_scaled(&src, kind, eff_rank, eff_size, block_size, gpu)?;
                    store.release_consumed_alias(&weight_key, src.weight, weight_dtype, gpu)?;
                    store.release_consumed_alias(&scale_key, src.row_scale, scale_dtype, gpu)?;
                    Ok(sharded)
                };
                let [q_fp8, k_fp8, v_fp8, o_fp8] = load_qkvo_tp(config, load_fp8_proj)?;
                tracing::info!(
                    "Layer {i}: FP8 Q/K/V/O loaded, {:.1} GB free",
                    gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0)
                );

                // O proj needs a QuantizedWeight placeholder for the AttentionWeights struct.
                // Use a dummy — the actual O proj uses o_fp8w via w8a16_gemv/gemm.
                let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
                let dummy = DenseWeight {
                    weight: spark_runtime::gpu::DevicePtr::NULL,
                };
                let dummy_qw = QuantizedWeight::null();
                let attn = AttentionWeights {
                    q_proj: dummy,
                    k_proj: dummy,
                    v_proj: dummy,
                    o_proj: dummy_qw,
                    q_norm: dense_f32_safe(store, &format!("{p}.q_norm.weight"), gpu)?,
                    k_norm: dense_f32_safe(store, &format!("{p}.k_norm.weight"), gpu)?,
                    q_norm_full: None,
                    k_norm_full: None,
                    k_scale,
                    v_scale,
                };

                let layer_kv_dtype = layer_kv_dtypes[attn_idx];
                let mut layer = Qwen3AttentionLayer::new(
                    input_norm,
                    attn,
                    post_attn_norm,
                    ffn,
                    attn_idx,
                    None,
                    None,
                    None, // No NVFP4 — w8a16_gemm handles prefill
                    gpu,
                    layer_kv_dtype,
                    config.fp8_kv_calibration_tokens,
                    config,
                )?;

                // Set checkpoint FP8 weights for decode (w8a16_gemv) and prefill fallback (w8a16_gemm).
                layer.set_fp8_weights(Some(q_fp8), Some(k_fp8), Some(v_fp8), Some(o_fp8));

                // Transpose FP8 weights for fast prefill (w8a16_gemm_t: coalesced reads).
                // This allocates N*K bytes per projection but gives ~14x prefill speedup.
                if let Err(e) = layer.transpose_fp8_for_prefill(gpu, stream) {
                    tracing::warn!(
                        "Layer {i}: FP8 transpose failed, using non-transposed prefill: {e}"
                    );
                } else {
                    tracing::info!("Layer {i}: FP8 weights transposed for fast prefill");
                }

                // Lever 5: opt-in sliding-window cap on full-attention layers
                // (ATLAS_ATTN_SLIDING_WINDOW). Default None = global attention.
                if let Some(w) = attn_sliding_window::sliding_window_override() {
                    layer.set_sliding_window(Some(w));
                }

                layers.push(Box::new(layer));
                attn_idx += 1;
            }
            LayerType::FullAttention => {
                let _attention_pass = spark_runtime::alloc_receipt::pass("attention");
                let layer = attention_arms::build_full_attention_nvfp4(
                    i,
                    store,
                    &lp,
                    gpu,
                    variant,
                    config,
                    h,
                    absmax_k,
                    quantize_k,
                    stream,
                    layer_kv_dtypes[attn_idx],
                    attn_idx,
                    input_norm,
                    post_attn_norm,
                    ffn,
                    release_mxfp8_sources,
                )?;
                layers.push(layer);
                attn_idx += 1;
            }
            // LinearAttention always builds via the NVFP4 path (with native
            // FP8 prefill GEMM installed on FP8-on-disk checkpoints).
            LayerType::LinearAttention => {
                let _ssm_pass = spark_runtime::alloc_receipt::pass("ssm");
                let layer = linear_attn_arms::build_linear_attention_nvfp4(
                    store,
                    &lp,
                    gpu,
                    variant,
                    config,
                    h,
                    absmax_k,
                    quantize_k,
                    stream,
                    input_norm,
                    post_attn_norm,
                    ffn,
                    release_mxfp8_sources,
                )?;
                layers.push(layer);
            }
            LayerType::Moe => unreachable!("Qwen3.5 has no standalone MoE layers"),
        }

        if config.is_qwen4_exp() {
            let _hyper_pass = spark_runtime::alloc_receipt::pass("hyper");
            let mut hyper_sources = Vec::new();
            let attn_hyper = load_qwen4_hyper(
                store,
                &format!("{lp}.attn_hyper_connection"),
                true,
                config,
                gpu,
                absmax_k,
                quantize_k,
                stream,
                &mut hyper_sources,
            )?;
            let mlp_hyper = load_qwen4_hyper(
                store,
                &format!("{lp}.mlp_hyper_connection"),
                true,
                config,
                gpu,
                absmax_k,
                quantize_k,
                stream,
                &mut hyper_sources,
            )?;
            // ATLAS_QWEN4_DENSE_DEDUP=1: release the BF16 sources unless pinned
            // by a reader (`ATLAS_QWEN4_HC_BF16`, `ATLAS_QWEN4_PREFILL_NATIVE`)
            // via `source_pins`.
            if dense_dedup_layers && dense_dedup::is_target_layer_prefix(&lp) {
                gpu.synchronize(stream)?;
                let bytes = dense_dedup::DedupBytes {
                    store_originals: dense_dedup::release_bf16_sources(store, &hyper_sources, gpu)?,
                    bf16_copies: 0,
                };
                dense_dedup::log_layer(&lp, "hyper", bytes);
            }
            layers
                .last_mut()
                .expect("layer was just pushed")
                .set_qwen4_hyperconnections(attn_hyper, mlp_hyper)?;
        }

        if (i + 1) % 10 == 0 || i < 5 {
            let free_gb = gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!("Loaded layers 0..{} — {free_gb:.1} GB free", i + 1);
            spark_runtime::progress::layer(i + 1, config.num_hidden_layers);
        }
    }

    tracing::info!(
        "Qwen3.5 weight loader: {} layers ({} attention, {} linear_attn)",
        layers.len(),
        attn_idx,
        layers.len() - attn_idx,
    );

    // Emitted once, after every layer is loaded, not from inside the
    // per-layer loop: `shared_precision` is a single per-call decision (see
    // `layer_shared_precision` above), so this must not depend on whether
    // layer 0 in particular happened to be an MoE layer — a receipt that
    // can silently never fire is the one thing a receipt must not do.
    // `layer_types.len()` is this call's own layer count, which excludes
    // the MTP layer: the MTP layer is loaded through a separate call to
    // `load_layers` with `config.weight_prefix == "mtp"`, which
    // `layer_shared_precision` refuses outright, so `shared_precision` can
    // only be `Bf16` here on a call that has no MTP layer in it.
    if shared_precision == crate::layers::moe::qwen4_shared_bf16::SharedPrecision::Bf16 {
        // Matches `load_moe_qwen35`'s own `inter`, which also loads the
        // shared expert from `config.moe_intermediate_size` — not
        // `shared_expert_intermediate_size`, despite the name.
        let inter = config.moe_intermediate_size;
        let extra_bytes = crate::layers::moe::qwen4_shared_bf16::extra_resident_bytes(
            h,
            inter,
            layer_types.len(),
        );
        tracing::info!(
            "QWEN4_SHARED_BF16 engaged scope=decode-only extra_resident_bytes={extra_bytes} \
             layers={} hidden={h} shared_intermediate={inter}",
            layer_types.len(),
        );
    }
    if dense_fp8 {
        dense_fp8::emit_receipt()?;
    }
    if release_mxfp8_sources {
        let total = mxfp8_release::take_total();
        tracing::info!(
            "MXFP8_SOURCES_RELEASED_TOTAL selector={} bytes={total} gib={:.2}",
            mxfp8_release::SELECTOR,
            total as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        spark_runtime::startup_ledger::record(
            "mxfp8-sources-released",
            format!("bytes={total}"),
            None,
            Some(-i64::try_from(total).unwrap_or(i64::MAX)),
        );
    }
    // MXFP8_RELEASE_SKIPPED_TOTAL prints at the end of the factory build,
    // after the final-mixer and LM-head releases (`source_pins::emit_totals`).

    // Qwen4 still has model-level hyperconnection and PLE projections after
    // the layer vector is built. Keep the writer open so those transforms
    // participate in the same provenance-locked artifact.
    if !config.is_qwen4_exp() {
        crate::weight_loader::transform_cache::finish();
    }

    Ok(layers)
}

fn ensure_supported_weight_formats(config: &ModelConfig) -> Result<()> {
    let Some(quant) = config.quantization_config.as_ref() else {
        return Ok(());
    };
    let is_mixed = quant.quant_algo.eq_ignore_ascii_case("MIXED_PRECISION")
        || quant
            .config_groups
            .iter()
            .any(|group| group.weight_format == ModelOptWeightFormat::Mxfp8);
    if !is_mixed {
        return Ok(());
    }
    anyhow::ensure!(
        !quant.config_groups.is_empty(),
        "Qwen Flash-Next ModelOpt MIXED_PRECISION requires explicit per-target config_groups"
    );
    for group in &quant.config_groups {
        for target in &group.targets {
            anyhow::ensure!(
                mixed_target_has_consumer(target, group.weight_format),
                "Qwen Flash-Next mixed target {target:?} ({:?}) has no fail-closed Flux consumer",
                group.weight_format
            );
        }
    }
    Ok(())
}

fn mixed_target_has_consumer(target: &str, format: ModelOptWeightFormat) -> bool {
    let fp8_layer_suffixes = &[
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "self_attn.indexer.index_qk_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
        "mlp.shared_expert.down_proj",
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
    ];
    let nvfp4_layer_suffixes = &[
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.experts",
        "ple.ple_embedding.ngram_embedding",
    ];
    match format {
        ModelOptWeightFormat::Fp8 | ModelOptWeightFormat::Mxfp8 => {
            numbered_target_has_suffix(target, "model.language_model.layers.", fp8_layer_suffixes)
                || numbered_target_has_suffix(target, "model.layers.", fp8_layer_suffixes)
                || numbered_target_has_suffix(
                    target,
                    "model.visual.blocks.",
                    &["attn.proj", "attn.qkv", "mlp.linear_fc1"],
                )
                || matches!(
                    target,
                    "model.visual.merger.linear_fc1" | "model.visual.merger.linear_fc2"
                )
        }
        ModelOptWeightFormat::Nvfp4 => {
            target == "lm_head"
                || numbered_target_has_suffix(
                    target,
                    "model.language_model.layers.",
                    nvfp4_layer_suffixes,
                )
                || numbered_target_has_suffix(target, "model.layers.", nvfp4_layer_suffixes)
                || numbered_target_has_suffix(target, "mtp.layers.", &["mlp.experts"])
                || numbered_target_has_suffix(target, "model.visual.blocks.", &["mlp.linear_fc2"])
        }
    }
}

fn numbered_target_has_suffix(target: &str, prefix: &str, suffixes: &[&str]) -> bool {
    let Some(rest) = target.strip_prefix(prefix) else {
        return false;
    };
    let Some((index, suffix)) = rest.split_once('.') else {
        return false;
    };
    !index.is_empty()
        && index.bytes().all(|byte| byte.is_ascii_digit())
        && suffixes.contains(&suffix)
}

/// `ATLAS_QWEN4_PREFILL_NATIVE=1`: pair each NVFP4 hyper-connection copy with
/// the store-resident BF16 tensor it was quantized from (no new allocation).
fn register_native_hc(
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
    down: &crate::weight_map::QuantizedWeight,
    up: &crate::weight_map::QuantizedWeight,
    inject: Option<&crate::weight_map::QuantizedWeight>,
    [down_bf16, up_bf16, inject_bf16]: [spark_runtime::gpu::DevicePtr; 3],
) -> Result<()> {
    use crate::layers::qwen4_prefill_native::{Module, register_bf16, store_retains_bf16};
    let (r, rank) = (config.residual_width(), config.hc_lowrank);
    let kept = |name: &str, ptr| store_retains_bf16(store, &format!("{prefix}.{name}.weight"), ptr);
    register_bf16(
        config,
        Module::HcDown,
        down,
        down_bf16,
        rank,
        r,
        kept("input_mix_weight_down", down_bf16),
    )?;
    register_bf16(
        config,
        Module::HcUp,
        up,
        up_bf16,
        r,
        rank,
        kept("input_mix_weight_up", up_bf16),
    )?;
    if let Some(inject) = inject {
        let ok = kept("block_inject_weight", inject_bf16);
        register_bf16(
            config,
            Module::HcInject,
            inject,
            inject_bf16,
            config.hc_count,
            r,
            ok,
        )?;
    }
    Ok(())
}

/// `ATLAS_QWEN4_DENSE_DEDUP=1`: release the model-level hyper-connection
/// mixer's BF16 sources once its NVFP4 copies exist, unless pinned by a reader
/// (`ATLAS_QWEN4_HC_BF16`, `ATLAS_QWEN4_PREFILL_NATIVE`) via `source_pins`.
pub(super) fn release_final_mixer_sources(
    store: &WeightStore,
    prefix: &str,
    down: DenseWeight,
    up: DenseWeight,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    // Target mixer only: `mtp.hyper_connection_mixer` is read by the MTP layer.
    if !dense_dedup::selected() || !dense_dedup::is_target_layer_prefix(prefix) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let sources = [
        (format!("{prefix}.input_mix_weight_down"), down),
        (format!("{prefix}.input_mix_weight_up"), up),
    ];
    let bytes = dense_dedup::DedupBytes {
        store_originals: dense_dedup::release_bf16_sources(store, &sources, gpu)?,
        bf16_copies: 0,
    };
    dense_dedup::log_layer(prefix, "hyper", bytes);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_qwen4_hyper(
    store: &WeightStore,
    prefix: &str,
    with_injection: bool,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    sources: &mut Vec<(String, DenseWeight)>,
) -> Result<Qwen4HyperConnection> {
    let r = config.residual_width();
    let rank = config.hc_lowrank;
    let norm = dense_f32_safe(store, &format!("{prefix}.hc_norm.weight"), gpu)?;
    let down_key = format!("{prefix}.input_mix_weight_down.weight");
    let up_key = format!("{prefix}.input_mix_weight_up.weight");
    let inject_key = format!("{prefix}.block_inject_weight.weight");
    // ATLAS_QWEN4_HC_BF16=1 (default off): the control is only meaningful
    // for a tensor that is BF16 *in the checkpoint*; refuse before loading
    // anything if that is not true (`validate_stored_bf16_tensor`).
    let keep_bf16 = crate::layers::qwen4_hc_bf16::selected()?;
    if keep_bf16 {
        let check = |key: &str, expected_shape: &[usize]| -> Result<()> {
            let stored = store.get(key)?;
            crate::layers::qwen4_hc_bf16::validate_stored_bf16_tensor(
                key,
                crate::layers::qwen4_hc_bf16::StoredHcTensor {
                    dtype: stored.dtype,
                    shape: &stored.shape,
                },
                expected_shape,
            )
        };
        check(&down_key, &[rank, r])?;
        check(&up_key, &[r, rank])?;
        if with_injection {
            check(&inject_key, &[config.hc_count, r])?;
        }
    }
    let down_dense = dense_auto(store, &down_key, gpu)?;
    let up_dense = dense_auto(store, &up_key, gpu)?;
    let down = crate::weight_map::quantize_to_nvfp4_cached(
        &down_dense,
        rank,
        r,
        gpu,
        absmax_k,
        quantize_k,
        stream,
        &format!("{prefix}.down.nvfp4"),
    )?;
    let up = crate::weight_map::quantize_to_nvfp4_cached(
        &up_dense,
        r,
        rank,
        gpu,
        absmax_k,
        quantize_k,
        stream,
        &format!("{prefix}.up.nvfp4"),
    )?;
    sources.push((format!("{prefix}.input_mix_weight_down"), down_dense));
    sources.push((format!("{prefix}.input_mix_weight_up"), up_dense));
    let inject_dense = if with_injection {
        Some(dense_auto(store, &inject_key, gpu)?)
    } else {
        None
    };
    if let Some(dense) = inject_dense {
        sources.push((format!("{prefix}.block_inject_weight"), dense));
    }
    let inject = match inject_dense {
        Some(ref dense) => Some(crate::weight_map::quantize_to_nvfp4_cached(
            dense,
            config.hc_count,
            r,
            gpu,
            absmax_k,
            quantize_k,
            stream,
            &format!("{prefix}.inject.nvfp4"),
        )?),
        None => None,
    };
    let inject_dense_ptr = inject_dense
        .as_ref()
        .map_or(spark_runtime::gpu::DevicePtr::NULL, |d| d.weight);
    // ATLAS_QWEN4_PREFILL_NATIVE=1 registers from the BF16 pointers before
    // ATLAS_QWEN4_HC_BF16 decides whether those DenseWeights stay resident.
    register_native_hc(
        store,
        prefix,
        config,
        &down,
        &up,
        inject.as_ref(),
        [down_dense.weight, up_dense.weight, inject_dense_ptr],
    )?;
    // Keep the BF16 dense copies the loader already holds instead of
    // letting them drop once quantized above (`keep_bf16` was already
    // validated/computed before any tensor above was loaded).
    let (down_bf16, up_bf16, inject_bf16) = crate::layers::qwen4_hc_bf16::resident_copies(
        keep_bf16,
        down_dense,
        up_dense,
        inject_dense,
    );
    Qwen4HyperConnection::new_with_bf16(
        norm,
        down,
        up,
        inject,
        down_bf16,
        up_bf16,
        inject_bf16,
        config.hidden_size,
        config.hc_count,
        rank,
        gpu,
    )
}

#[cfg(test)]
#[path = "load_layers/source_pins_tests.rs"]
mod source_pins_tests;

#[cfg(test)]
mod tests {
    use atlas_core::config::{
        ModelConfig, ModelOptQuantizationGroup, ModelOptWeightFormat, QuantizationConfig,
    };

    use super::{ensure_supported_weight_formats, mixed_target_has_consumer};

    #[test]
    fn pinned_qwen38_mixed_target_classes_have_exact_consumers() {
        for target in [
            "model.language_model.layers.0.linear_attn.in_proj_a",
            "model.language_model.layers.3.self_attn.q_proj",
            "model.language_model.layers.3.self_attn.indexer.index_qk_proj",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj",
            "model.visual.blocks.0.attn.qkv",
            "model.visual.blocks.0.mlp.linear_fc1",
            "model.visual.merger.linear_fc2",
        ] {
            assert!(
                mixed_target_has_consumer(target, ModelOptWeightFormat::Mxfp8),
                "missing MXFP8 consumer for {target}"
            );
        }
        for target in [
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            "model.language_model.layers.3.self_attn.q_proj",
        ] {
            assert!(
                mixed_target_has_consumer(target, ModelOptWeightFormat::Fp8),
                "missing per-tensor FP8 consumer for {target}"
            );
        }
        for target in [
            "model.language_model.layers.0.mlp.experts",
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.0.mlp.up_proj",
            "model.language_model.layers.0.mlp.down_proj",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding",
            "mtp.layers.0.mlp.experts",
            "mtp.layers.48.mlp.experts",
            "model.visual.blocks.0.mlp.linear_fc2",
            "lm_head",
        ] {
            assert!(
                mixed_target_has_consumer(target, ModelOptWeightFormat::Nvfp4),
                "missing NVFP4 consumer for {target}"
            );
        }
        assert!(!mixed_target_has_consumer(
            "model.visual.blocks.0.mlp.linear_fc2",
            ModelOptWeightFormat::Mxfp8
        ));
        assert!(!mixed_target_has_consumer(
            "model.visual.blocks.0.mlp.linear_fc1",
            ModelOptWeightFormat::Nvfp4
        ));
        for target in [
            "model.language_model.layers.x.linear_attn.in_proj_a",
            "model.language_model.layers.0.linear_attn.unknown_proj",
            "model.language_model.layers.0.linear_attn.in_proj_a.trailing",
            "model.language_model.layers.0x.linear_attn.in_proj_a",
            "model.visual.blocks.-1.attn.qkv",
            "model.visual.blocks.0.attn.unknown",
            "mtp.layers.0.mlp.experts.trailing",
        ] {
            assert!(
                !mixed_target_has_consumer(target, ModelOptWeightFormat::Mxfp8)
                    && !mixed_target_has_consumer(target, ModelOptWeightFormat::Fp8)
                    && !mixed_target_has_consumer(target, ModelOptWeightFormat::Nvfp4),
                "hostile or unknown target must fail closed: {target}"
            );
        }
    }

    #[test]
    fn known_mixed_mxfp8_target_is_admitted() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".into();
        config.quantization_config = Some(QuantizationConfig {
            quant_method: "modelopt".into(),
            quant_algo: "MIXED_PRECISION".into(),
            format: String::new(),
            ignore_modules: Vec::new(),
            config_groups: vec![ModelOptQuantizationGroup {
                name: "attention".into(),
                weight_format: ModelOptWeightFormat::Mxfp8,
                group_size: 32,
                targets: vec!["model.language_model.layers.3.self_attn.q_proj".into()],
            }],
        });

        ensure_supported_weight_formats(&config)
            .expect("wired mixed target class must be admitted before GPU access");
    }

    #[test]
    fn unknown_mixed_target_fails_before_loader_touches_the_gpu() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".into();
        config.quantization_config = Some(QuantizationConfig {
            quant_method: "modelopt".into(),
            quant_algo: "MIXED_PRECISION".into(),
            format: String::new(),
            ignore_modules: Vec::new(),
            config_groups: vec![ModelOptQuantizationGroup {
                name: "unknown".into(),
                weight_format: ModelOptWeightFormat::Mxfp8,
                group_size: 32,
                targets: vec!["model.unknown.projection".into()],
            }],
        });

        let err = ensure_supported_weight_formats(&config)
            .expect_err("unknown mixed target must fail closed");
        assert!(err.to_string().contains("no fail-closed Flux consumer"));
    }

    #[test]
    fn uniform_nvfp4_remains_admitted() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        ensure_supported_weight_formats(&config)
            .expect("existing NVFP4 path must remain unchanged");
    }
}
