// SPDX-License-Identifier: AGPL-3.0-only

//! MEMLEAN releases for the dense hybrid loader (selectors in
//! `crate::memlean`). Every release runs in the loader's per-layer
//! construction window after the construction stream is drained, the same
//! quiescent pattern as `release_owned_dequant_source` (BUG #29), and every
//! released store name is tombstoned, so a later lookup fails closed.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::Qwen3SsmLayer;
use crate::layers::qwen3_ssm::SharedActE4m3Scratch;
use crate::memlean::{DenseLoadPlan, ReleasedBytes};

/// SSM checkpoint projections this loader converts: FP8 `.weight`
/// dequantized to BF16 (`load_ssm_qwen35`), then requantized to the NVFP4
/// QKVZ / out_proj copies every runtime route reads.
const SSM_CONVERTED: [&str; 3] = ["in_proj_qkv", "in_proj_z", "out_proj"];
/// Attention projections converted by `quantized_any` (FP8 -> BF16 -> NVFP4)
/// or `dense_auto` (FP8 -> owned BF16).
const ATTENTION_CONVERTED: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

/// Release `{prefix}.weight` when it is an FP8 E4M3 checkpoint tensor the
/// loader has already converted. Packed NVFP4 and BF16 store tensors are read
/// in place by runtime attention routes and are left alone. Returns the bytes
/// released.
pub(super) fn release_converted_fp8_weight(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    let name = format!("{prefix}.weight");
    if !store.contains(&name) {
        return Ok(0);
    }
    let (ptr, dtype) = {
        let weight = store.get(&name)?;
        (weight.ptr, weight.dtype)
    };
    if dtype != WeightDtype::FP8E4M3 {
        return Ok(0);
    }
    store.release_consumed_alias(&name, ptr, dtype, gpu)
}

/// Release dead original SSM store tensors after `qwen35_dense` has dequantized
/// and requantized `in_proj_qkv`, `in_proj_z`, and `out_proj` into fresh
/// `qkvz_nvfp4` / `out_proj_nvfp4` allocations. Unlike attention layers (which
/// read packed `WeightDtype::UInt8` checkpoint weights in place), dense SSM
/// layers never touch the original checkpoint `FP8E4M3` or `UInt8` store
/// tensors (`.weight` / `.weight_packed` and paired `FP8E4M3` `.weight_scale`)
/// again after construction.
pub(super) fn release_converted_ssm_weight(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    let mut released = 0usize;
    let mut released_packed_nvfp4 = false;

    for (suffix, allow_fp8) in [("weight", true), ("weight_packed", false)] {
        let name = format!("{prefix}.{suffix}");
        if !store.contains(&name) {
            continue;
        }
        let (ptr, dtype) = {
            let weight = store.get(&name)?;
            (weight.ptr, weight.dtype)
        };
        let eligible = dtype == WeightDtype::UInt8 || (allow_fp8 && dtype == WeightDtype::FP8E4M3);
        if !eligible {
            continue;
        }
        let bytes = store.release_consumed_alias(&name, ptr, dtype, gpu)?;
        released += bytes;
        if bytes != 0 && dtype == WeightDtype::UInt8 {
            released_packed_nvfp4 = true;
        }
    }

    if released_packed_nvfp4 {
        let scale_name = format!("{prefix}.weight_scale");
        if store.contains(&scale_name) {
            let (ptr, dtype) = {
                let weight = store.get(&scale_name)?;
                (weight.ptr, weight.dtype)
            };
            if dtype == WeightDtype::FP8E4M3 {
                released += store.release_consumed_alias(&scale_name, ptr, dtype, gpu)?;
            }
        }
    }

    Ok(released)
}

/// One load's MEMLEAN decisions and running totals.
pub(super) struct DenseLean {
    plan: DenseLoadPlan,
    /// `ATLAS_GDN_BF16_WEIGHTS=1` keeps the BF16 QKVZ concat its forced path
    /// reads.
    gdn_bf16_forced: bool,
    shared_act: Option<SharedActE4m3Scratch>,
    total: ReleasedBytes,
}

impl DenseLean {
    pub(super) fn from_env() -> Result<Self> {
        Self::new(
            DenseLoadPlan::from_env()?,
            std::env::var("ATLAS_GDN_BF16_WEIGHTS").ok().as_deref() == Some("1"),
        )
    }

    fn new(plan: DenseLoadPlan, gdn_bf16_forced: bool) -> Result<Self> {
        if plan != DenseLoadPlan::default() {
            tracing::info!(
                ?plan,
                gdn_bf16_forced,
                "MEMLEAN dense loader plan (see crate::memlean for each selector)"
            );
        }
        if plan.ssm_bf16_qkvz && gdn_bf16_forced {
            tracing::warn!(
                "{}=1 ignored: ATLAS_GDN_BF16_WEIGHTS=1 reads the BF16 QKVZ concat",
                crate::memlean::SSM_BF16_QKVZ
            );
        }
        Ok(Self {
            plan,
            gdn_bf16_forced,
            shared_act: plan.shared_scratch.then(SharedActE4m3Scratch::default),
            total: ReleasedBytes::default(),
        })
    }

    pub(super) fn plan(&self) -> DenseLoadPlan {
        self.plan
    }

    /// Hand a newly built SSM layer the shared cuBLASLt e4m3 activation arena.
    pub(super) fn adopt_ssm_scratch(&self, layer: &mut Qwen3SsmLayer) -> Result<()> {
        match &self.shared_act {
            Some(shared) => layer.adopt_shared_act_e4m3_scratch(shared),
            None => Ok(()),
        }
    }

    /// Release what a fully built SSM layer no longer reads.
    pub(super) fn finish_ssm_layer(
        &self,
        store: &WeightStore,
        layer_prefix: &str,
        layer: &mut Qwen3SsmLayer,
        qkvz_bf16_bytes: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
        released: &mut ReleasedBytes,
    ) -> Result<()> {
        let release_qkvz = self.plan.ssm_bf16_qkvz && !self.gdn_bf16_forced;
        if !release_qkvz && !self.plan.fp8_sources {
            return Ok(());
        }
        gpu.synchronize(stream)?;
        if release_qkvz && let Some(ptr) = layer.take_bf16_qkvz() {
            gpu.free(ptr)?;
            released.bf16_qkvz += qkvz_bf16_bytes;
        }
        if self.plan.fp8_sources {
            for name in SSM_CONVERTED {
                released.fp8_sources += release_converted_ssm_weight(
                    store,
                    &format!("{layer_prefix}.linear_attn.{name}"),
                    gpu,
                )?;
            }
        }
        Ok(())
    }

    /// Release the converted FP8 projections of a fully built attention layer.
    pub(super) fn finish_attention_layer(
        &self,
        store: &WeightStore,
        attn_prefix: &str,
        gpu: &dyn GpuBackend,
        stream: u64,
        released: &mut ReleasedBytes,
    ) -> Result<()> {
        if !self.plan.fp8_sources {
            return Ok(());
        }
        gpu.synchronize(stream)?;
        for name in ATTENTION_CONVERTED {
            released.fp8_sources +=
                release_converted_fp8_weight(store, &format!("{attn_prefix}.{name}"), gpu)?;
        }
        Ok(())
    }

    /// Log one layer's releases and add them to the load total.
    pub(super) fn record(&mut self, layer: usize, released: ReleasedBytes) {
        released.log(layer);
        self.total.bf16_qkvz += released.bf16_qkvz;
        self.total.fp8_sources += released.fp8_sources;
        self.total.ffn_gate_up += released.ffn_gate_up;
    }

    pub(super) fn log_total(&self, layers: usize) {
        if self.total.total() == 0 && self.shared_act.is_none() {
            return;
        }
        let gib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        tracing::info!(
            "MEMLEAN_RELEASED_TOTAL layers={layers} bf16_qkvz_gib={:.2} fp8_sources_gib={:.2} ffn_gate_up_gib={:.2} total_gib={:.2} shared_scratch={}",
            gib(self.total.bf16_qkvz),
            gib(self.total.fp8_sources),
            gib(self.total.ffn_gate_up),
            gib(self.total.total()),
            self.shared_act.is_some(),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::{DenseLean, release_converted_fp8_weight, release_converted_ssm_weight};
    use crate::memlean::{DenseLoadPlan, ReleasedBytes};

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

    #[test]
    fn only_converted_fp8_weights_are_released_and_tombstoned() {
        let gpu = MockGpuBackend::new();
        let store = store_of(
            &gpu,
            &[
                ("a.weight", WeightDtype::FP8E4M3, 64),
                ("a.weight_scale", WeightDtype::FP32, 1),
                ("b.weight", WeightDtype::UInt8, 64),
                ("c.weight", WeightDtype::BF16, 32),
            ],
        );
        assert_eq!(release_converted_fp8_weight(&store, "a", &gpu).unwrap(), 64);
        assert_eq!(release_converted_fp8_weight(&store, "b", &gpu).unwrap(), 0);
        assert_eq!(release_converted_fp8_weight(&store, "c", &gpu).unwrap(), 0);
        assert_eq!(
            release_converted_fp8_weight(&store, "missing", &gpu).unwrap(),
            0
        );
        assert_eq!(gpu.alloc_count(), 3);
        let error = store
            .get("a.weight")
            .err()
            .expect("a released weight's lookup fails")
            .to_string();
        assert!(
            error.contains("consumed during model construction"),
            "{error}"
        );
        assert!(store.contains("a.weight_scale"));
        // A second release of the same name is a no-op, not a double free.
        assert_eq!(release_converted_fp8_weight(&store, "a", &gpu).unwrap(), 0);
    }

    #[test]
    fn attention_release_covers_q_k_v_o_and_nothing_else() {
        let gpu = MockGpuBackend::new();
        let p = "model.layers.3.self_attn";
        let names = ["q_proj", "k_proj", "v_proj", "o_proj"].map(|n| format!("{p}.{n}.weight"));
        let mut tensors: Vec<(&str, WeightDtype, usize)> = names
            .iter()
            .map(|name| (name.as_str(), WeightDtype::FP8E4M3, 128))
            .collect();
        let norm = format!("{p}.q_norm.weight");
        tensors.push((norm.as_str(), WeightDtype::BF16, 8));
        let store = store_of(&gpu, &tensors);
        let lean = DenseLean::new(
            DenseLoadPlan {
                fp8_sources: true,
                ..DenseLoadPlan::default()
            },
            false,
        )
        .unwrap();
        let mut released = ReleasedBytes::default();
        lean.finish_attention_layer(&store, p, &gpu, 0, &mut released)
            .unwrap();
        assert_eq!(released.fp8_sources, 4 * 128);
        assert_eq!(gpu.alloc_count(), 1);
        assert!(store.contains(&norm));
    }

    #[test]
    fn nothing_is_released_with_every_selector_off() {
        let gpu = MockGpuBackend::new();
        let p = "model.layers.3.self_attn";
        let q = format!("{p}.q_proj.weight");
        let store = store_of(&gpu, &[(q.as_str(), WeightDtype::FP8E4M3, 128)]);
        let lean = DenseLean::new(DenseLoadPlan::default(), false).unwrap();
        let mut released = ReleasedBytes::default();
        lean.finish_attention_layer(&store, p, &gpu, 0, &mut released)
            .unwrap();
        assert_eq!(released, ReleasedBytes::default());
        assert!(store.contains(&q));
    }

    #[test]
    fn totals_accumulate_every_family() {
        let mut lean = DenseLean::new(DenseLoadPlan::default(), false).unwrap();
        lean.record(
            0,
            ReleasedBytes {
                bf16_qkvz: 1,
                fp8_sources: 2,
                ffn_gate_up: 3,
            },
        );
        lean.record(
            1,
            ReleasedBytes {
                bf16_qkvz: 10,
                fp8_sources: 20,
                ffn_gate_up: 30,
            },
        );
        assert_eq!(lean.total.total(), 66);
        assert_eq!(lean.total.ffn_gate_up, 33);
    }

    #[test]
    fn ssm_release_frees_uint8_packed_weights_and_fp8_scales_but_keeps_bf16() {
        let gpu = MockGpuBackend::new();
        let store = store_of(
            &gpu,
            &[
                // ModelOpt NVFP4 packed SSM tensor + FP8 block scales
                ("ssm_opt.weight", WeightDtype::UInt8, 256),
                ("ssm_opt.weight_scale", WeightDtype::FP8E4M3, 32),
                ("ssm_opt.weight_scale_2", WeightDtype::FP32, 1),
                // Compressed-tensors NVFP4 packed SSM tensor + FP8 block scales
                ("ssm_ct.weight_packed", WeightDtype::UInt8, 128),
                ("ssm_ct.weight_scale", WeightDtype::FP8E4M3, 16),
                // FP8 SSM tensor (keeps FP32 weight_scale)
                ("ssm_fp8.weight", WeightDtype::FP8E4M3, 64),
                ("ssm_fp8.weight_scale", WeightDtype::FP32, 1),
                // BF16 SSM tensor must not be released
                ("ssm_bf16.weight", WeightDtype::BF16, 32),
            ],
        );
        assert_eq!(
            release_converted_ssm_weight(&store, "ssm_opt", &gpu).unwrap(),
            256 + 32
        );
        assert_eq!(
            release_converted_ssm_weight(&store, "ssm_ct", &gpu).unwrap(),
            128 + 16
        );
        assert_eq!(
            release_converted_ssm_weight(&store, "ssm_fp8", &gpu).unwrap(),
            64
        );
        assert_eq!(
            release_converted_ssm_weight(&store, "ssm_bf16", &gpu).unwrap(),
            0
        );
        // Tombstoned packed weights and FP8 block scales fail closed; FP32 scales and BF16 remain.
        assert!(store.get("ssm_opt.weight").is_err());
        assert!(store.get("ssm_opt.weight_scale").is_err());
        assert!(store.contains("ssm_opt.weight_scale_2"));
        assert!(store.get("ssm_ct.weight_packed").is_err());
        assert!(store.get("ssm_ct.weight_scale").is_err());
        assert!(store.get("ssm_fp8.weight").is_err());
        assert!(store.contains("ssm_fp8.weight_scale"));
        assert!(store.contains("ssm_bf16.weight"));
        assert_eq!(gpu.alloc_count(), 3);
        // Repeat release is a no-op.
        assert_eq!(
            release_converted_ssm_weight(&store, "ssm_opt", &gpu).unwrap(),
            0
        );
    }
}
