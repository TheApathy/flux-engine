// SPDX-License-Identifier: AGPL-3.0-only
//
// `ATLAS_QWEN4_RELEASE_MXFP8_SOURCES` (default ON for Flash-Next / qwen4_exp
// loads): release each layer's checkpoint ModelOpt MXFP8 (E4M3 weight + E8M0
// per-32 scale) source tensors once `quantized_modelopt_mixed` /
// `dense_modelopt_mixed_or_fp8_or_bf16` (`weight_map/quant_helpers.rs`) has
// dequantized them to an owned BF16 buffer and quantized that to the NVFP4
// copy every prefill/decode/verify route reads:
//
// * GDN (`linear_attn`) `in_proj_qkv`, `in_proj_z`, `out_proj` — read exactly
//   once by `load_ssm_qwen35` (`weight_map/ssm_qwen35.rs`), which returns
//   owned BF16 buffers independent of the store;
// * attention `q_proj`/`k_proj`/`v_proj`/`o_proj` — read exactly once by the
//   `quantized_modelopt_mixed` call in `load_layers/attention_arms.rs`'s
//   `Standard`/`CompressedTensors` arm (the arm Flash-Next's MXFP8 checkpoint
//   actually takes; q/k/v/o never alias a BF16 store tensor there);
// * shared-expert `gate_proj`/`up_proj`/`down_proj` — read exactly once by
//   `load_moe_qwen35`'s shared-expert arm, same conversion.
//
// This is a narrower, always-reachable sibling of `ATLAS_QWEN4_DENSE_DEDUP`
// (`dense_dedup.rs`), whose own module doc already carries the consumer
// census for these exact tensor families ("None of these bytes is read
// after construction"). `ATLAS_QWEN4_DENSE_DEDUP`'s attention/GDN release
// arms are additionally gated behind `ATLAS_QWEN4_RELEASE_BF16_QKVZ=1` (and,
// for attention, only fire on the `Fp8Dequanted`/`Bf16Raw` arm that
// Flash-Next's MXFP8 checkpoint does not take), so they do not cover the
// MXFP8 sources on a default build. This module releases the MXFP8 store
// originals unconditionally once their one-time read has happened,
// independent of those other legacy-prefill selectors — the release
// primitive (`dense_dedup::release_fp8_store_source`) is shared and is a
// no-op on anything that is not an FP8E4M3 (MXFP8's on-GPU weight dtype)
// `.weight`, so re-running it under both selectors is safe and idempotent.
//
// Output is byte-identical by construction: every tensor this module frees
// has already had its one and only read; the kernels that run afterward
// read the already-built NVFP4 (or FP8-via-BF16) copies, never the
// checkpoint MXFP8 originals.
//
// A tensor a native-dense reader pins (`ATLAS_QWEN4_DENSE_FP8`,
// `ATLAS_QWEN4_PREFILL_NATIVE` and the BF16 controls, umbrella-aware) is
// skipped per name by `dense_dedup::release_fp8_store_source` through
// `source_pins` and reported as `MXFP8_RELEASE_SKIPPED`. The earlier
// whole-selector skip read the raw env and missed `ATLAS_QWEN4_DENSE_NATIVE=1`.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, bail};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::dense_dedup::release_fp8_store_source;

/// Running total of bytes released this load, drained by `take_total` into
/// the one end-of-load `MXFP8_SOURCES_RELEASED_TOTAL` line and startup
/// ledger row. Per-process, so concurrent loads (not a supported
/// configuration today) would interleave; a single load's layers are built
/// serially on one thread.
static RELEASED_TOTAL: AtomicUsize = AtomicUsize::new(0);

pub(super) const SELECTOR: &str = "ATLAS_QWEN4_RELEASE_MXFP8_SOURCES";

/// Strict parse, default ON: unset is on, `"1"` is on, `"0"` is off, any
/// other value is a load error (consistent with `crate::memlean::parse`).
pub(super) fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => bail!("{SELECTOR} must be exactly 0 or 1, got {other:?}"),
    }
}

/// Read the selector from the process environment.
pub(super) fn selected() -> Result<bool> {
    match std::env::var(SELECTOR) {
        Ok(value) => parse(Some(value.as_str())),
        Err(std::env::VarError::NotPresent) => parse(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("{SELECTOR} must be valid UTF-8 and exactly 0 or 1")
        }
    }
}

/// Release a shared expert's MXFP8 gate/up/down store originals. Delegates to
/// the same primitive `dense_dedup::release_shared_expert_sources` uses, so
/// running both selectors is a harmless double call (the second is a no-op:
/// `release_fp8_store_source` tombstones on first release and returns 0
/// afterward).
pub(super) fn release_shared_expert(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    let mut released = 0usize;
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        released += release_fp8_store_source(
            store,
            &format!("{layer_prefix}.mlp.shared_expert.{proj}"),
            gpu,
        )?;
    }
    Ok(released)
}

/// Release one GDN layer's MXFP8 `in_proj_qkv`/`in_proj_z`/`out_proj` store
/// originals. Called at the tail of `build_linear_attention_nvfp4`, strictly
/// after every other store reader (`ships_mxfp8`,
/// `release_replaced_bf16_source`, `dense_fp8::load_gdn`, `dense_dedup`) has
/// finished; `load_ssm_qwen35` also reads MXFP8 `in_proj_a`/`in_proj_b`,
/// which are left to `dense_dedup`.
pub(super) fn release_linear_attn(
    store: &WeightStore,
    linear_attn_prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    let mut released = 0usize;
    for proj in ["in_proj_qkv", "in_proj_z", "out_proj"] {
        released += release_fp8_store_source(store, &format!("{linear_attn_prefix}.{proj}"), gpu)?;
    }
    Ok(released)
}

/// Release one attention projection's MXFP8 store original
/// (`{attn_prefix}.{proj}`, one of `q_proj`/`k_proj`/`v_proj`/`o_proj`).
/// Called at the tail of `build_full_attention_nvfp4` for the
/// `Standard`/`CompressedTensors` attention arm, strictly after all other
/// store readers (`quantized_modelopt_mixed`, `modelopt_mixed_bf16_retained`,
/// `dense_fp8::load_attention`, `dense_dedup`) have finished.
pub(super) fn release_attention_proj(
    store: &WeightStore,
    attn_prefix: &str,
    proj: &str,
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    release_fp8_store_source(store, &format!("{attn_prefix}.{proj}"), gpu)
}

/// Log one layer's releases, if any, under the family name (`shared-expert`,
/// `linear_attn`, `attention`).
pub(super) fn log_layer(layer: &str, family: &str, bytes: usize) {
    crate::source_pins::flush(layer, family);
    if bytes == 0 {
        return;
    }
    RELEASED_TOTAL.fetch_add(bytes, Ordering::Relaxed);
    tracing::info!(
        "MXFP8_SOURCES_RELEASED selector={SELECTOR} layer={layer} family={family} bytes={bytes} mib={:.2}",
        bytes as f64 / (1024.0 * 1024.0),
    );
}

/// Drain the running total for the one end-of-load ledger record.
pub(super) fn take_total() -> usize {
    RELEASED_TOTAL.swap(0, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::*;

    fn store_of(gpu: &MockGpuBackend, tensors: &[(&str, WeightDtype, usize)]) -> WeightStore {
        WeightStore::from_map(
            tensors
                .iter()
                .map(|&(name, dtype, elements)| {
                    let ptr = gpu.alloc(elements * dtype.byte_size()).unwrap();
                    (
                        name.to_string(),
                        WeightTensor {
                            ptr,
                            shape: vec![elements],
                            dtype,
                        },
                    )
                })
                .collect::<HashMap<_, _>>(),
        )
    }

    // -- selector/skip precedence (pure rule tests) --------------------

    #[test]
    fn selector_defaults_on() {
        assert!(parse(None).unwrap());
    }

    #[test]
    fn selector_is_strict_about_other_values() {
        assert!(parse(Some("1")).unwrap());
        assert!(!parse(Some("0")).unwrap());
        for bad in ["", "true", "on", " 1", "2"] {
            let error = parse(Some(bad)).unwrap_err().to_string();
            assert!(
                error.contains("ATLAS_QWEN4_RELEASE_MXFP8_SOURCES must be exactly 0 or 1"),
                "{bad:?}: {error}"
            );
        }
    }

    // -- loader tests: releases free and tombstone, nothing else is touched --

    #[test]
    fn shared_expert_release_frees_and_tombstones_only_its_three_tensors() {
        let gpu = MockGpuBackend::new();
        let p = "model.language_model.layers.0";
        let names = ["gate_proj", "up_proj", "down_proj"]
            .map(|n| format!("{p}.mlp.shared_expert.{n}.weight"));
        let mut tensors: Vec<(&str, WeightDtype, usize)> = names
            .iter()
            .map(|name| (name.as_str(), WeightDtype::FP8E4M3, 64))
            .collect();
        let router = format!("{p}.mlp.gate.weight");
        tensors.push((router.as_str(), WeightDtype::BF16, 8));
        let store = store_of(&gpu, &tensors);

        let released = release_shared_expert(&store, p, &gpu).unwrap();
        assert_eq!(released, 3 * 64);
        for name in &names {
            let error = store
                .get(name)
                .err()
                .expect("released lookup fails")
                .to_string();
            assert!(
                error.contains("consumed during model construction"),
                "{error}"
            );
        }
        assert!(store.contains(&router));
        // Second call is a no-op, not a double free.
        assert_eq!(release_shared_expert(&store, p, &gpu).unwrap(), 0);
    }

    #[test]
    fn linear_attn_release_covers_qkv_z_out_proj_and_nothing_else() {
        let gpu = MockGpuBackend::new();
        let p = "model.language_model.layers.3.linear_attn";
        let names = ["in_proj_qkv", "in_proj_z", "out_proj"].map(|n| format!("{p}.{n}.weight"));
        let mut tensors: Vec<(&str, WeightDtype, usize)> = names
            .iter()
            .map(|name| (name.as_str(), WeightDtype::FP8E4M3, 32))
            .collect();
        let untouched = format!("{p}.in_proj_a.weight");
        tensors.push((untouched.as_str(), WeightDtype::FP8E4M3, 32));
        let store = store_of(&gpu, &tensors);

        let released = release_linear_attn(&store, p, &gpu).unwrap();
        assert_eq!(released, 3 * 32);
        for name in &names {
            assert!(!store.contains(name));
        }
        assert!(store.contains(&untouched));
    }

    #[test]
    fn attention_release_is_per_projection_and_tombstones_that_name() {
        let gpu = MockGpuBackend::new();
        let p = "model.language_model.layers.1.self_attn";
        let q = format!("{p}.q_proj.weight");
        let k = format!("{p}.k_proj.weight");
        let store = store_of(
            &gpu,
            &[
                (q.as_str(), WeightDtype::FP8E4M3, 128),
                (k.as_str(), WeightDtype::FP8E4M3, 128),
            ],
        );

        assert_eq!(
            release_attention_proj(&store, p, "q_proj", &gpu).unwrap(),
            128
        );
        let error = store
            .get(&q)
            .err()
            .expect("released lookup fails")
            .to_string();
        assert!(
            error.contains("consumed during model construction"),
            "{error}"
        );
        assert!(store.contains(&k));
    }

    #[test]
    fn log_layer_accumulates_and_take_total_drains_and_resets() {
        // Reset first: other tests in this file never call `log_layer`, but
        // guard against future additions sharing the process-wide counter.
        let _ = take_total();
        log_layer("layer.0", "shared-expert", 100);
        log_layer("layer.1", "attention", 0);
        log_layer("layer.1", "linear_attn", 23);
        assert_eq!(take_total(), 123);
        assert_eq!(take_total(), 0, "a second drain sees nothing left");
    }

    #[test]
    fn non_fp8_tensors_are_left_alone() {
        let gpu = MockGpuBackend::new();
        let p = "model.language_model.layers.2";
        let name = format!("{p}.mlp.shared_expert.gate_proj.weight");
        let store = store_of(&gpu, &[(name.as_str(), WeightDtype::BF16, 64)]);
        assert_eq!(release_shared_expert(&store, p, &gpu).unwrap(), 0);
        assert!(store.contains(&name));
    }

    // -- builder-tail ordering (regression for the 51e77acfa gate abort) --

    /// Replays the tail of `build_linear_attention_nvfp4` with
    /// ATLAS_QWEN4_DENSE_DEDUP=1, ATLAS_QWEN4_RELEASE_BF16_QKVZ=1 and this
    /// selector on: the replaced-source reclaim (`store.get` on each name),
    /// then dense dedup, then the MXFP8 release. Returns the first error.
    fn linear_attn_tail(
        store: &WeightStore,
        gpu: &MockGpuBackend,
        lp: &str,
        mxfp8_first: bool,
    ) -> Result<usize> {
        let la = format!("{lp}.linear_attn");
        if mxfp8_first {
            release_linear_attn(store, &la, gpu)?;
        }
        for proj in [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
        ] {
            // Owned BF16 dequant of the FP8 source, as load_ssm_qwen35 returns.
            let owned = crate::weight_map::DenseWeight {
                weight: gpu.alloc(64)?,
            };
            super::super::linear_attn_arms::release_replaced_bf16_source(
                store,
                &format!("{la}.{proj}.weight"),
                owned,
                64,
                gpu,
            )?;
        }
        if super::super::dense_dedup::is_target_layer_prefix(lp) {
            for proj in [
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_a",
                "in_proj_b",
                "out_proj",
            ] {
                release_fp8_store_source(store, &format!("{la}.{proj}"), gpu)?;
            }
        }
        let mut released = 0;
        if super::super::dense_dedup::is_target_layer_prefix(lp) {
            released = release_linear_attn(store, &la, gpu)?;
        }
        Ok(released)
    }

    fn gdn_store(gpu: &MockGpuBackend, lp: &str) -> (WeightStore, Vec<String>) {
        let names: Vec<String> = [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
        ]
        .iter()
        .map(|n| format!("{lp}.linear_attn.{n}.weight"))
        .collect();
        let tensors: Vec<(&str, WeightDtype, usize)> = names
            .iter()
            .map(|n| (n.as_str(), WeightDtype::FP8E4M3, 32))
            .collect();
        (store_of(gpu, &tensors), names)
    }

    #[test]
    fn three_knobs_release_after_every_reader_has_no_consumed_error() {
        let gpu = MockGpuBackend::new();
        let lp = "model.language_model.layers.0";
        let (store, names) = gdn_store(&gpu, lp);
        let released = linear_attn_tail(&store, &gpu, lp, false)
            .expect("release ordered after all readers must not hit a consumed weight");
        // Dense dedup already released all five; the MXFP8 pass is a no-op.
        assert_eq!(released, 0);
        for name in &names {
            assert!(!store.contains(name), "{name} must end released");
        }
    }

    #[test]
    fn releasing_before_the_reclaim_reproduces_the_gate_abort() {
        let gpu = MockGpuBackend::new();
        let lp = "model.language_model.layers.0";
        let (store, _) = gdn_store(&gpu, lp);
        let error = linear_attn_tail(&store, &gpu, lp, true)
            .unwrap_err()
            .to_string();
        assert!(
            error
                .contains("linear_attn.in_proj_qkv.weight' was consumed during model construction"),
            "{error}"
        );
    }

    #[test]
    fn mtp_layer_sources_are_never_released() {
        let gpu = MockGpuBackend::new();
        let lp = "mtp.layers.0";
        let (store, names) = gdn_store(&gpu, lp);
        assert_eq!(linear_attn_tail(&store, &gpu, lp, false).unwrap(), 0);
        for name in &names {
            assert!(store.contains(name), "{name} must stay readable for MTP");
        }
    }
}
