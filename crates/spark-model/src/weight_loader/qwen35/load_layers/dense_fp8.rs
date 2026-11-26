// SPDX-License-Identifier: AGPL-3.0-only

//! Loader half of `ATLAS_QWEN4_DENSE_FP8=1` (see `layers/qwen4_dense_fp8`).
//!
//! Each selected matrix is served from the checkpoint tensors themselves:
//! the E4M3 `.weight` and the UE8M0 `.weight_scale` stay resident in the
//! weight store for the process lifetime (pinned against
//! `ATLAS_QWEN4_RELEASE_MXFP8_SOURCES` and `ATLAS_QWEN4_DENSE_DEDUP` by
//! `crate::source_pins`), so the decode arm aliases them. No copy, no
//! requantization, zero extra resident bytes; the NVFP4 operands the
//! existing arms build are untouched and still serve prefill and every
//! multi-row route.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::qwen4_dense_fp8::mxfp8::{MXFP8_BLOCK, validate_scales};
use crate::layers::qwen4_dense_fp8::proj::{AttnFp8, Fp8Kernels, GdnFp8, MxFp8Proj, SharedFp8};
use crate::layers::qwen4_dense_fp8::{self as rule, Module, QWEN4_DENSE_FP8_ENV};
use crate::weight_map::{ModeloptMixedRoute, modelopt_mixed_route};

/// Whether this loader pass builds FP8 copies (selector + per-layer rule).
pub(super) fn engaged(config: &ModelConfig) -> Result<bool> {
    rule::layer_engaged(
        rule::selected()?,
        config.is_qwen4_exp(),
        config.weight_prefix == "mtp",
    )
}

/// `model.language_model.layers.17` -> 17.
fn layer_index(lp: &str) -> Result<usize> {
    lp.rsplit('.')
        .next()
        .and_then(|tail| tail.parse().ok())
        .with_context(|| format!("{QWEN4_DENSE_FP8_ENV}: cannot read a layer index from {lp}"))
}

/// Fail-closed check that a checkpoint tensor pair is exactly what
/// `dense_gemv_mxfp8` reads: F8_E4M3 `.weight [n,k]` (equal to `expect` when
/// given) and U8 UE8M0 `.weight_scale [n, k/32]`. Pure, so it is unit-tested.
fn validate_mxfp8_pair(
    prefix: &str,
    weight: (WeightDtype, &[usize]),
    scale: (WeightDtype, &[usize]),
    expect: Option<[usize; 2]>,
) -> Result<[usize; 2]> {
    let (w_dtype, w_shape) = weight;
    let (s_dtype, s_shape) = scale;
    let &[n, k] = w_shape else {
        anyhow::bail!(
            "{QWEN4_DENSE_FP8_ENV}=1 refused: {prefix}.weight shape {w_shape:?} is not [n,k]"
        );
    };
    ensure!(
        w_dtype == WeightDtype::FP8E4M3,
        "{QWEN4_DENSE_FP8_ENV}=1 refused: {prefix}.weight is {w_dtype:?}, not F8_E4M3"
    );
    if let Some(expect) = expect {
        ensure!(
            [n, k] == expect,
            "{QWEN4_DENSE_FP8_ENV}=1 refused: {prefix}.weight is [{n},{k}], expected {expect:?}"
        );
    }
    ensure!(
        n > 0 && k > 0 && k % MXFP8_BLOCK == 0,
        "{QWEN4_DENSE_FP8_ENV}=1 refused: {prefix}.weight [{n},{k}] is not block-32 MXFP8"
    );
    ensure!(
        s_dtype == WeightDtype::UInt8 && s_shape == [n, k / MXFP8_BLOCK],
        "{QWEN4_DENSE_FP8_ENV}=1 refused: {prefix}.weight_scale is {s_dtype:?} {s_shape:?}, \
         not U8 [{n},{}]",
        k / MXFP8_BLOCK
    );
    Ok([n, k])
}

/// The validated checkpoint shape of `{prefix}`: declared MXFP8 in
/// hf_quant_config and a well-formed weight/scale pair in the store.
fn mxfp8_shape(
    store: &WeightStore,
    config: &ModelConfig,
    prefix: &str,
    expect: Option<[usize; 2]>,
) -> Result<[usize; 2]> {
    ensure!(
        modelopt_mixed_route(config, prefix) == ModeloptMixedRoute::Mxfp8ViaBf16,
        "{QWEN4_DENSE_FP8_ENV}=1 refused: {prefix} is not declared MXFP8 in hf_quant_config"
    );
    let weight = store.get(&format!("{prefix}.weight"))?;
    let scale = store.get(&format!("{prefix}.weight_scale"))?;
    validate_mxfp8_pair(
        prefix,
        (weight.dtype, &weight.shape),
        (scale.dtype, &scale.shape),
        expect,
    )
}

/// Alias one checkpoint MXFP8 tensor pair after the full fail-closed check
/// (declared MXFP8, dtypes, shapes, no reserved 255 exponent, alignment).
fn mxfp8_proj(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    prefix: &str,
    expect: [usize; 2],
) -> Result<MxFp8Proj> {
    let [n, k] = mxfp8_shape(store, config, prefix, Some(expect))?;
    let weight = store.get(&format!("{prefix}.weight"))?;
    let scale = store.get(&format!("{prefix}.weight_scale"))?;
    let mut exponents = vec![0u8; scale.byte_size()];
    gpu.copy_d2h(scale.ptr, &mut exponents)?;
    validate_scales(prefix, &exponents)?;
    MxFp8Proj::alias(weight.ptr, scale.ptr, n, k).with_context(|| format!("aliasing {prefix}"))
}

fn require_single_rank(config: &ModelConfig) -> Result<()> {
    ensure!(
        config.tp_world_size <= 1 && config.ep_world_size <= 1,
        "{QWEN4_DENSE_FP8_ENV}=1 refused: the FP8 copies are not sharded; run TP1/EP1"
    );
    Ok(())
}

/// GDN `in_proj_qkv`, `in_proj_z` (decoded into adjacent slices, the
/// `gpu_concat_rows` order) and `out_proj`.
pub(super) fn load_gdn(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<GdnFp8> {
    require_single_rank(config)?;
    let h = config.hidden_size;
    let p = format!("{lp}.linear_attn");
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let projs = [
        mxfp8_proj(
            store,
            config,
            gpu,
            &format!("{p}.in_proj_qkv"),
            [config.ssm_qkv_size(), h],
        )?,
        mxfp8_proj(
            store,
            config,
            gpu,
            &format!("{p}.in_proj_z"),
            [config.ssm_z_size(), h],
        )?,
        mxfp8_proj(store, config, gpu, &format!("{p}.out_proj"), [h, value_dim])?,
    ];
    let gdn = GdnFp8::new(layer_index(lp)?, projs, Fp8Kernels::resolve(gpu)?)?;
    LEDGER.account(Module::Gdn, gdn.checkpoint_bytes());
    Ok(gdn)
}

/// Attention q (checkpoint layout: per-head [q | gate] when gated), k, v, o.
pub(super) fn load_attention(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<AttnFp8> {
    require_single_rank(config)?;
    ensure!(
        !super::tq_plus_weight_rotation::weight_rotation_enabled(),
        "{QWEN4_DENSE_FP8_ENV}=1 refused: TQ+ weight rotation changes the q/k/v operands"
    );
    let (h, hd) = (config.hidden_size, config.head_dim);
    let q_dim = config.num_attention_heads * hd;
    let q_rows = if config.attn_gated { q_dim * 2 } else { q_dim };
    let kv_rows = config.num_key_value_heads * hd;
    let p = format!("{lp}.self_attn");
    let proj = |name: &str, shape| mxfp8_proj(store, config, gpu, &format!("{p}.{name}"), shape);
    let projs = [
        proj("q_proj", [q_rows, h])?,
        proj("k_proj", [kv_rows, h])?,
        proj("v_proj", [kv_rows, h])?,
        proj("o_proj", [h, q_dim])?,
    ];
    let attn = AttnFp8::new(layer_index(lp)?, projs, Fp8Kernels::resolve(gpu)?);
    LEDGER.account(Module::Attn, attn.checkpoint_bytes());
    Ok(attn)
}

/// Shared expert gate/up `[inter, h]` and down `[h, inter]`.
pub(super) fn load_shared(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<SharedFp8> {
    require_single_rank(config)?;
    let h = config.hidden_size;
    let p = format!("{lp}.mlp.shared_expert");
    let [inter, gate_k] = mxfp8_shape(store, config, &format!("{p}.gate_proj"), None)?;
    ensure!(
        gate_k == h,
        "{p}.gate_proj is [{inter},{gate_k}], expected k={h}"
    );
    let proj = |name: &str, shape| mxfp8_proj(store, config, gpu, &format!("{p}.{name}"), shape);
    let projs = [
        proj("gate_proj", [inter, h])?,
        proj("up_proj", [inter, h])?,
        proj("down_proj", [h, inter])?,
    ];
    let shared = SharedFp8::new(layer_index(lp)?, projs, Fp8Kernels::resolve(gpu)?)?;
    LEDGER.account(Module::Shared, shared.checkpoint_bytes());
    Ok(shared)
}

/// Aliased checkpoint bytes and layer counts per module for the load receipt.
struct Ledger {
    bytes: [AtomicU64; 3],
    layers: [AtomicU64; 3],
}

static LEDGER: Ledger = Ledger::new();

impl Ledger {
    const fn new() -> Self {
        Self {
            bytes: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            layers: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        }
    }

    fn account(&self, module: Module, bytes: u64) {
        let i = module as usize;
        self.bytes[i].fetch_add(bytes, Ordering::Relaxed);
        self.layers[i].fetch_add(1, Ordering::Relaxed);
    }

    fn reset(&self) {
        for (bytes, layers) in self.bytes.iter().zip(self.layers.iter()) {
            bytes.store(0, Ordering::Relaxed);
            layers.store(0, Ordering::Relaxed);
        }
    }

    fn take(&self) -> [(Module, u64, u64); 3] {
        Module::ALL.map(|m| {
            let i = m as usize;
            (
                m,
                self.layers[i].swap(0, Ordering::Relaxed),
                self.bytes[i].swap(0, Ordering::Relaxed),
            )
        })
    }
}

/// Reset the process-wide ledger at the start of a `load_layers` pass so
/// partial failures from an earlier load cannot contaminate the next receipt.
pub(super) fn reset_ledger() {
    LEDGER.reset();
}

/// The load-time receipt line (pure, so its format is testable).
fn receipt_line(entries: &[(Module, u64, u64)]) -> String {
    let modules: Vec<&str> = entries
        .iter()
        .filter(|e| e.1 > 0)
        .map(|e| e.0.name())
        .collect();
    let total: u64 = entries.iter().map(|e| e.2).sum();
    let per_module: Vec<String> = entries
        .iter()
        .map(|(m, layers, bytes)| {
            format!("{}_layers={layers} {}_bytes={bytes}", m.name(), m.name())
        })
        .collect();
    format!(
        "QWEN4_DENSE_FP8 engaged scope=decode-only modules={} extra_resident_bytes=0 \
         aliased_checkpoint_bytes={total} {} kernel=dense_gemv_mxfp8 weights=native_mxfp8_block32 \
         requant=none prefill=nvfp4_unchanged",
        modules.join(","),
        per_module.join(" ")
    )
}

/// The receipt for an engaged pass; refuses one where a module built no
/// layer (a hook that never ran must not read as engaged).
fn complete_receipt(entries: &[(Module, u64, u64)]) -> Result<String> {
    ensure!(
        entries.iter().all(|e| e.1 > 0),
        "{QWEN4_DENSE_FP8_ENV}=1 engaged but a module built no FP8 layers: {entries:?}"
    );
    Ok(receipt_line(entries))
}

/// Emit the INFO receipt once all layers are built, draining the ledger so a
/// subsequent load in the same process starts from zero.
pub(super) fn emit_receipt() -> Result<()> {
    tracing::info!("{}", complete_receipt(&LEDGER.take())?);
    Ok(())
}

#[cfg(test)]
#[path = "dense_fp8_tests.rs"]
mod tests;
