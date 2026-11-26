// SPDX-License-Identifier: AGPL-3.0-only
//
// Helper functions for the LinearAttention arm of `load_layers` (NVFP4 decode
// with optional native FP8 prefill GEMM on FP8-on-disk checkpoints).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, Qwen3SsmLayer};
use crate::weight_map::{
    DenseWeight, Nvfp4Variant, SsmWeights, gpu_concat_rows, interleave_ba, load_ssm_qwen35,
    quantize_to_nvfp4,
};

fn retain_bf16_out_proj(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

/// Per-layer MiB of legacy prefill copies skipped when
/// `ATLAS_QWEN4_RELEASE_BF16_QKVZ=1` is active on Qwen4:
///   - SSM: `qkvz_nvfp4_t` (9/16 B/elem) + `out_proj_nvfp4_t` (9/16 B/elem)
///     + `out_proj_fp8` predequant (1 B/elem).
///   - Attention: `q/k/v/o` transposed NVFP4 (9/16 B/elem) + FP8 predequant
///     (1 B/elem), including the 2× gated-Q factor when `config.attn_gated`.
fn legacy_prefill_skipped_mib(config: &ModelConfig) -> (f64, f64) {
    let h = config.hidden_size;
    let qkvz_size = config.ssm_qkvz_size();
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let ssm_sixteenths = qkvz_size * h * 9 + h * value_dim * 25;

    let q_dim = config.num_attention_heads * config.head_dim;
    let q_proj_n = if config.attn_gated { q_dim * 2 } else { q_dim };
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let attn_sixteenths = (q_proj_n + 2 * kv_dim + q_dim) * h * 25;

    let denom = 16.0 * 1024.0 * 1024.0;
    (
        ssm_sixteenths as f64 / denom,
        attn_sixteenths as f64 / denom,
    )
}

pub(super) fn release_replaced_bf16_source(
    store: &WeightStore,
    name: &str,
    dense: DenseWeight,
    logical_bytes: usize,
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    // CompressedTensors checkpoints store packed NVFP4 SSM projections under
    // `{prefix}.weight_packed` without `{prefix}.weight` (matching
    // `load_ssm_qwen35`'s packed-first detection).
    let packed_name = name
        .strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.weight_packed"));
    let resolved_name = match packed_name.as_deref() {
        Some(packed) if store.contains(packed) => packed,
        _ => name,
    };
    let source = store.get(resolved_name)?;
    if source.ptr == dense.weight {
        anyhow::ensure!(
            source.dtype == WeightDtype::BF16,
            "replacement source alias {resolved_name} must be BF16, got {:?}",
            source.dtype
        );
        // A store alias a reader pins stays resident (MXFP8_RELEASE_SKIPPED).
        if crate::source_pins::skip_if_pinned(
            resolved_name,
            logical_bytes,
            crate::source_pins::PinKind::Bf16Alias,
        )?
        .is_some()
        {
            return Ok(0);
        }
        return store.release_consumed_alias(resolved_name, dense.weight, WeightDtype::BF16, gpu);
    }

    anyhow::ensure!(
        !dense.weight.is_null(),
        "replacement source buffer {resolved_name} must not be null"
    );
    anyhow::ensure!(
        matches!(
            source.dtype,
            WeightDtype::BF16 | WeightDtype::FP8E4M3 | WeightDtype::UInt8
        ),
        "replacement source {resolved_name} must originate from BF16, FP8, or packed NVFP4, got {:?}",
        source.dtype
    );

    // Mixed ModelOpt FP8 and packed NVFP4 projections are expanded into owned
    // BF16 buffers at load, then copied/quantized into the durable
    // QKVZ/BA/output operands. They are not WeightStore aliases, so
    // release_consumed_alias is a no-op and would retain roughly 10 GiB across
    // 48 dense Qwen3.8 GDN layers.
    gpu.free(dense.weight)?;
    Ok(logical_bytes)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_linear_attention_nvfp4(
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    release_mxfp8_sources: bool,
) -> Result<Box<dyn TransformerLayer>> {
    let ssm35 = load_ssm_qwen35(store, lp, gpu, variant, config)?;
    // ATLAS_QWEN4_PREFILL_NATIVE: checked before the sources are tombstoned below.
    let native_src = |name: &str| {
        crate::layers::qwen4_prefill_native::ships_mxfp8(store, &format!("{lp}.linear_attn.{name}"))
    };
    let (qkvz_mxfp8, out_mxfp8) = (
        native_src("in_proj_qkv") && native_src("in_proj_z"),
        native_src("out_proj"),
    );

    let qkv_rows = config.ssm_qkv_size();
    let z_rows = config.ssm_z_size();
    let qkvz_dense = gpu_concat_rows(
        &ssm35.in_proj_qkv,
        qkv_rows,
        &ssm35.in_proj_z,
        z_rows,
        h,
        gpu,
    )?;

    let nv = config.linear_num_value_heads;
    let nk = config.linear_num_key_heads;
    let ba_dense = interleave_ba(
        &DenseWeight {
            weight: ssm35.in_proj_a.weight,
        },
        &DenseWeight {
            weight: ssm35.in_proj_b.weight,
        },
        nv,
        nk,
        h,
        gpu,
    )?;

    // The combined QKVZ and interleaved BA buffers above are durable copies.
    // On BF16 checkpoints, retaining their four source allocations in the
    // immutable WeightStore needlessly raises construction peak by gigabytes.
    // Store aliases are tombstoned; owned FP8-dequant buffers are freed after
    // the durable copy. This distinction prevents construction-only buffers
    // from accumulating across all linear-attention layers.
    let mut reclaimed_source_bytes = 0usize;
    reclaimed_source_bytes += release_replaced_bf16_source(
        store,
        &format!("{lp}.linear_attn.in_proj_qkv.weight"),
        ssm35.in_proj_qkv,
        qkv_rows * h * 2,
        gpu,
    )?;
    reclaimed_source_bytes += release_replaced_bf16_source(
        store,
        &format!("{lp}.linear_attn.in_proj_z.weight"),
        ssm35.in_proj_z,
        z_rows * h * 2,
        gpu,
    )?;
    reclaimed_source_bytes += release_replaced_bf16_source(
        store,
        &format!("{lp}.linear_attn.in_proj_a.weight"),
        ssm35.in_proj_a,
        nv * h * 2,
        gpu,
    )?;
    reclaimed_source_bytes += release_replaced_bf16_source(
        store,
        &format!("{lp}.linear_attn.in_proj_b.weight"),
        ssm35.in_proj_b,
        nv * h * 2,
        gpu,
    )?;

    let retain_out_proj =
        retain_bf16_out_proj(std::env::var("ATLAS_GDN_BF16_WEIGHTS").ok().as_deref());
    let release_bf16_qkvz = !retain_out_proj
        && retain_bf16_out_proj(
            std::env::var("ATLAS_QWEN4_RELEASE_BF16_QKVZ")
                .ok()
                .as_deref(),
        );
    let skip_qwen4_legacy_prefill = config.is_qwen4_exp() && release_bf16_qkvz;
    if skip_qwen4_legacy_prefill {
        let (ssm_mib, attn_mib) = legacy_prefill_skipped_mib(config);
        static LOG_ONCE: std::sync::Once = std::sync::Once::new();
        LOG_ONCE.call_once(|| {
            tracing::info!(
                "QWEN4_RELEASE_BF16_QKVZ_LEGACY_PREFILL_SKIPPED selector=ATLAS_QWEN4_RELEASE_BF16_QKVZ ssm_mib_per_layer={ssm_mib} attn_mib_per_layer={attn_mib}"
            );
        });
    }

    let qkvz_size = config.ssm_qkvz_size();
    let qkvz_nvfp4 =
        quantize_to_nvfp4(&qkvz_dense, qkvz_size, h, gpu, absmax_k, quantize_k, stream)?;

    let qkvz_nvfp4_t = if skip_qwen4_legacy_prefill {
        None
    } else {
        Some(qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?)
    };

    let value_dim = nv * config.linear_value_head_dim;
    let out_proj_nvfp4 = quantize_to_nvfp4(
        &ssm35.out_proj,
        h,
        value_dim,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;

    // ATLAS_QWEN4_PREFILL_NATIVE=1: prefill E4M3 from the exact MXFP8 BF16.
    crate::layers::qwen4_prefill_native::register_mxfp8(
        gpu,
        config,
        crate::layers::qwen4_prefill_native::Module::GdnQkvz,
        &qkvz_nvfp4,
        qkvz_dense.weight,
        qkvz_size,
        h,
        qkvz_mxfp8,
        stream,
    )?;
    crate::layers::qwen4_prefill_native::register_mxfp8(
        gpu,
        config,
        crate::layers::qwen4_prefill_native::Module::GdnOut,
        &out_proj_nvfp4,
        ssm35.out_proj.weight,
        h,
        value_dim,
        out_mxfp8,
        stream,
    )?;

    let out_proj_nvfp4_t = if skip_qwen4_legacy_prefill {
        None
    } else {
        Some(out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?)
    };

    // Native FP8 SSM prefill GEMM for FP8-on-disk checkpoints. Because GDN
    // conv weights have strongly asymmetric channel scales (k-segment ~18×
    // smaller than v-segment), passing them through the FP8→BF16→NVFP4→BF16
    // chain attenuates direction in the k-channel. Bypass the NVFP4
    // intermediate by installing a single-scale FP8 copy of `qkvz_dense` and
    // `ssm35.out_proj` and dispatching prefill through `fp8_gemm_n128`.
    let (qkvz_fp8_prefill, out_proj_fp8_prefill) = if matches!(variant, Nvfp4Variant::Fp8Dequanted)
    {
        tracing::info!(
            "SSM[{lp}] in_proj_qkvz + out_proj via native FP8 prefill GEMM \
                 (BF16 act × FP8 weight via fp8_gemm_n128)"
        );
        let b2f_k = gpu.kernel("w4a16", "bf16_to_fp8")?;
        let qkvz_total = (qkvz_size * h) as u32;
        let qkvz_fp8 = gpu.alloc(qkvz_size * h)?;
        crate::layers::ops::bf16_to_fp8(
            gpu,
            b2f_k,
            qkvz_dense.weight,
            qkvz_fp8,
            qkvz_total,
            stream,
        )?;
        let out_total = (h * value_dim) as u32;
        let out_fp8 = gpu.alloc(h * value_dim)?;
        crate::layers::ops::bf16_to_fp8(
            gpu,
            b2f_k,
            ssm35.out_proj.weight,
            out_fp8,
            out_total,
            stream,
        )?;
        gpu.synchronize(stream)?;
        (Some(qkvz_fp8), Some(out_fp8))
    } else {
        (None, None)
    };

    let ssm = SsmWeights {
        in_proj_qkvz: qkvz_dense,
        in_proj_ba: ba_dense,
        conv1d: ssm35.conv1d,
        a_log: ssm35.a_log,
        dt_bias: ssm35.dt_bias,
        norm: ssm35.norm,
        out_proj: out_proj_nvfp4,
    };

    // `ATLAS_QWEN4_GDN_BF16=1` (default off, decode-only, refused at launch
    // for anything but C1/batch=1 qwen4_exp — see `qwen4_decode_bf16_arms`).
    //
    // CORRECTNESS NOTE (review fix, was a bug in the first cut of this
    // control): `qkvz_nvfp4` must stay `Some(qkvz_nvfp4)` unconditionally.
    // It is NOT decode-only — `qwen4_prefill_exact.rs` ("exact SSM
    // requires NVFP4 QKVZ") and the FlashInfer SSM prefill route
    // (`init.rs`'s FlashInfer-SSM admission) both assert `qkvz_nvfp4` is
    // `Some` directly, independent of `qkvz_nvfp4_t`. And `qkvz_nvfp4_t`
    // itself is NOT unconditionally built: every production recipe that
    // sets `ATLAS_QWEN4_RELEASE_BF16_QKVZ=1` makes `skip_qwen4_legacy_prefill`
    // true, which makes `qkvz_nvfp4_t` `None` too (see above). Nulling
    // `qkvz_nvfp4` under that combination broke exact prefill. Instead,
    // keep the exact MXFP8->BF16 QKVZ concat alive and route decode
    // through a dedicated decode-only field, exactly like out_proj below.
    let gdn_bf16_engaged = if config.is_qwen4_exp() {
        crate::layers::qwen4_decode_bf16_arms::gdn_bf16_selected()?
    } else {
        false
    };
    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        Some(qkvz_nvfp4),
        qkvz_nvfp4_t,
        out_proj_nvfp4_t,
        config,
        gpu,
    )?;
    if gdn_bf16_engaged {
        // `ATLAS_QWEN4_GDN_BF16=1`: decode-only BF16 QKVZ. `qkvz_nvfp4`
        // above is untouched, so every prefill/verify consumer that
        // asserts it directly keeps working unchanged.
        layer.qkvz_dense_decode_only = Some(qkvz_dense);
        static LOG_QKVZ: std::sync::Once = std::sync::Once::new();
        LOG_QKVZ.call_once(|| {
            tracing::info!(
                "QWEN4_GDN_BF16 qkvz: decode-only BF16 dense_gemv, prefill unchanged (extra_resident_bytes_per_layer={})",
                qkvz_size * h * 2
            );
        });
    }
    if super::dense_fp8::engaged(config)? {
        layer.set_qwen4_dense_fp8(super::dense_fp8::load_gdn(store, lp, config, gpu)?)?;
    }
    if !skip_qwen4_legacy_prefill {
        layer.predequant_for_prefill(gpu, config, stream)?;
    }
    // Install native FP8 prefill weights AFTER `predequant_for_prefill`
    // (which sets `out_proj_fp8` from NVFP4 + scale2). The FP8 path
    // overrides both pointers when active, routing prefill through
    // `fp8_gemm_n128` instead of `w4a16_gemm_t`. Decode batch paths
    // retain their NVFP4 fallback via the `qkvz_nvfp4*` fields above.
    if qkvz_fp8_prefill.is_some() || out_proj_fp8_prefill.is_some() {
        layer.set_fp8_prefill_only_weights(qkvz_fp8_prefill, out_proj_fp8_prefill);
    }
    // ATLAS_GDN_BF16_WEIGHTS=1 extension: also install BF16 out_proj so
    // the prefill dispatcher takes the dense_gemm BF16 path (highest
    // dispatch priority). Eliminates FP8/NVFP4 quant noise on out_proj
    // — the noise was previously amplified by post_attn_norm's RMSNorm
    // into wildly different gate inputs at the MoE block (cos=0.42 vs
    // HF). Test fix for long-context drift root cause (commit 1db7572
    // and onward investigation). ssm35.out_proj is the BF16 weight
    // (loaded via dense_auto with FP8→BF16 dequant).
    // ATLAS_QWEN4_RELEASE_BF16_QKVZ=1 (default off): free the BF16 QKVZ concat
    // (~84 MB per layer, ~3 GB on Flash-Next) once its NVFP4 copy exists.
    // ATLAS_QWEN4_GDN_BF16=1 aliases this same concat as
    // `qkvz_dense_decode_only` (`DenseWeight` is a non-owning handle), so
    // freeing it here would leave decode reading freed memory. Keep it.
    if release_bf16_qkvz && !gdn_bf16_engaged {
        gpu.synchronize(stream)?;
        if let Some(ptr) = layer.take_bf16_qkvz() {
            gpu.free(ptr)?;
            reclaimed_source_bytes += qkvz_size * h * 2;
        }
    }
    if retain_out_proj {
        // ssm35.out_proj weight is BF16 on GPU (from load_ssm_qwen35 →
        // dense_auto on Fp8Dequanted variant). It's a separate buffer
        // from out_proj_nvfp4 / out_proj_fp8_prefill. Set as dense path.
        layer.out_proj_dense = Some(ssm35.out_proj);
        tracing::info!(
            "SSM[{lp}] ATLAS_GDN_BF16_WEIGHTS: out_proj routed through BF16 dense_gemm (overrides FP8/NVFP4)"
        );
    } else if gdn_bf16_engaged {
        // `ATLAS_QWEN4_GDN_BF16=1`: keep the BF16 out_proj alive and route
        // it through the decode-only field (`out_proj_dense_decode_only`)
        // rather than `out_proj_dense`, so prefill's FP8/NVFP4 out_proj
        // dispatch in `trait_prefill_phase3.rs` is untouched.
        layer.out_proj_dense_decode_only = Some(ssm35.out_proj);
        static LOG_OUT_PROJ: std::sync::Once = std::sync::Once::new();
        LOG_OUT_PROJ.call_once(|| {
            tracing::info!(
                "QWEN4_GDN_BF16 out_proj: decode-only BF16 dense_gemv, prefill unchanged (extra_resident_bytes_per_layer={})",
                h * value_dim * 2
            );
        });
    } else {
        reclaimed_source_bytes += release_replaced_bf16_source(
            store,
            &format!("{lp}.linear_attn.out_proj.weight"),
            ssm35.out_proj,
            h * value_dim * 2,
            gpu,
        )?;
    }
    if reclaimed_source_bytes != 0 {
        tracing::debug!(
            "SSM[{lp}] released {:.2} MiB of replaced BF16 source projections",
            reclaimed_source_bytes as f64 / (1024.0 * 1024.0)
        );
    }
    // ATLAS_QWEN4_DENSE_DEDUP=1: the durable QKVZ/BA/out_proj operands above
    // are built, and their BF16 dequants are already gone, so the FP8 store
    // originals have no reader left.
    if skip_qwen4_legacy_prefill
        && !retain_out_proj
        && super::dense_dedup::selected()
        && super::dense_dedup::is_target_layer_prefix(lp)
    {
        gpu.synchronize(stream)?;
        let mut bytes = super::dense_dedup::DedupBytes::default();
        for proj in [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
        ] {
            bytes.store_originals += super::dense_dedup::release_fp8_store_source(
                store,
                &format!("{lp}.linear_attn.{proj}"),
                gpu,
            )?;
        }
        super::dense_dedup::log_layer(lp, "linear_attn", bytes);
    }
    // ATLAS_QWEN4_RELEASE_MXFP8_SOURCES: release the MXFP8 store originals of
    // in_proj_qkv/in_proj_z/out_proj only here, after every reader above has
    // run. `release_replaced_bf16_source` looks those same store names up
    // (`store.get`) to decide alias vs owned copy, so releasing right after
    // `load_ssm_qwen35` tombstoned them first and failed the load with
    // "was consumed during model construction". Target layers only, matching
    // dense dedup: an MTP layer's tensors may be read after this loader.
    if release_mxfp8_sources && super::dense_dedup::is_target_layer_prefix(lp) {
        gpu.synchronize(stream)?;
        let linear_attn_prefix = format!("{lp}.linear_attn");
        let bytes = super::mxfp8_release::release_linear_attn(store, &linear_attn_prefix, gpu)?;
        super::mxfp8_release::log_layer(lp, "linear_attn", bytes);
    }
    Ok(Box::new(layer))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
    use spark_runtime::weights::WeightTensor;

    use super::{
        DenseWeight, ModelConfig, WeightDtype, WeightStore, legacy_prefill_skipped_mib,
        release_replaced_bf16_source, retain_bf16_out_proj,
    };

    fn store_with(
        gpu: &MockGpuBackend,
        name: &str,
        dtype: WeightDtype,
        bytes: usize,
    ) -> WeightStore {
        WeightStore::from_map(HashMap::from([(
            name.to_string(),
            WeightTensor {
                ptr: gpu.alloc(bytes).expect("allocate store tensor"),
                shape: vec![bytes / dtype.byte_size()],
                dtype,
            },
        )]))
    }

    #[test]
    fn bf16_out_proj_retention_selector_is_exact() {
        assert!(retain_bf16_out_proj(Some("1")));
        for rejected in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
            assert!(!retain_bf16_out_proj(rejected));
        }
    }

    #[test]
    fn bf16_store_alias_is_consumed_once_and_tombstoned() {
        let gpu = MockGpuBackend::new();
        let name = "model.layers.0.linear_attn.in_proj_qkv.weight";
        let store = store_with(&gpu, name, WeightDtype::BF16, 1024);
        let alias = DenseWeight {
            weight: store.get(name).unwrap().ptr,
        };

        assert_eq!(
            release_replaced_bf16_source(&store, name, alias, 1024, &gpu).unwrap(),
            1024
        );
        assert_eq!(gpu.alloc_count(), 0, "the consumed alias is freed once");
        assert!(
            !store.contains(name),
            "the stale store pointer is tombstoned"
        );
        assert!(store.get(name).is_err(), "later lookup must fail closed");
        assert!(
            release_replaced_bf16_source(&store, name, alias, 1024, &gpu).is_err(),
            "a second loader release must not reach device free"
        );
    }

    #[test]
    fn converted_loader_allocation_is_freed_without_consuming_fp8_source() {
        let gpu = MockGpuBackend::new();
        let name = "model.layers.0.linear_attn.in_proj_z.weight";
        let store = store_with(&gpu, name, WeightDtype::FP8E4M3, 512);
        let copy = DenseWeight {
            weight: gpu.alloc(1024).expect("allocate converted BF16 copy"),
        };

        assert_eq!(gpu.alloc_count(), 2);
        assert_eq!(
            release_replaced_bf16_source(&store, name, copy, 1024, &gpu).unwrap(),
            1024
        );
        assert_eq!(gpu.alloc_count(), 1, "only the converted copy is freed");
        assert!(store.contains(name), "the FP8 source remains store-owned");
        assert_eq!(store.get(name).unwrap().dtype, WeightDtype::FP8E4M3);
    }

    #[test]
    fn unknown_source_name_fails_closed_without_freeing_the_buffer() {
        let gpu = MockGpuBackend::new();
        let store = WeightStore::from_map(HashMap::new());
        let copy = DenseWeight {
            weight: gpu.alloc(1024).expect("allocate loader buffer"),
        };

        assert!(release_replaced_bf16_source(&store, "absent.weight", copy, 1024, &gpu).is_err());
        assert_eq!(
            gpu.alloc_count(),
            1,
            "an ownership lookup failure must not guess and free"
        );
    }

    #[test]
    fn compressed_tensors_weight_packed_source_frees_owned_bf16_expansion() {
        let gpu = MockGpuBackend::new();
        let packed = "model.layers.0.linear_attn.in_proj_qkv.weight_packed";
        let store = store_with(&gpu, packed, WeightDtype::UInt8, 256);
        let copy = DenseWeight {
            weight: gpu.alloc(1024).expect("allocate dequantized BF16 copy"),
        };

        assert_eq!(gpu.alloc_count(), 2);
        assert_eq!(
            release_replaced_bf16_source(
                &store,
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                copy,
                1024,
                &gpu,
            )
            .unwrap(),
            1024
        );
        assert_eq!(gpu.alloc_count(), 1, "only the owned BF16 copy is freed");
        assert!(store.contains(packed), "packed source remains store-owned");
    }

    #[test]
    fn legacy_prefill_skipped_mib_accounts_for_gated_attention() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.hidden_size = 2560;
        config.num_attention_heads = 24;
        config.num_key_value_heads = 2;
        config.head_dim = 256;
        config.attn_gated = true;
        config.linear_num_key_heads = 16;
        config.linear_num_value_heads = 48;
        config.linear_key_head_dim = 128;
        config.linear_value_head_dim = 128;

        let (ssm_mib, attn_mib) = legacy_prefill_skipped_mib(&config);
        assert!((ssm_mib - 45.9375).abs() < 1e-6, "ssm={ssm_mib}");
        assert!((attn_mib - 74.21875).abs() < 1e-6, "attn={attn_mib}");

        config.attn_gated = false;
        let (_, ungated_attn_mib) = legacy_prefill_skipped_mib(&config);
        assert!(
            (ungated_attn_mib - 50.78125).abs() < 1e-6,
            "ungated_attn={ungated_attn_mib}"
        );
    }
}
