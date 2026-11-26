// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{
    is_target_layer_prefix, parse_selected, release_bf16_sources, release_fp8_store_source,
    release_router_bf16, release_shared_expert_sources,
};
use crate::weight_map::DenseWeight;

fn tensor(gpu: &MockGpuBackend, dtype: WeightDtype, elements: usize) -> WeightTensor {
    WeightTensor {
        ptr: gpu
            .alloc(elements * dtype.byte_size())
            .expect("allocate store tensor"),
        shape: vec![elements],
        dtype,
    }
}

#[test]
fn selector_is_exact_and_default_off() {
    assert!(parse_selected(Some("1")));
    for rejected in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some(" 1"),
        Some("1 "),
    ] {
        assert!(!parse_selected(rejected), "{rejected:?}");
    }
}

#[test]
fn mtp_layer_is_never_deduplicated() {
    assert!(is_target_layer_prefix("model.language_model.layers.3"));
    assert!(!is_target_layer_prefix("mtp.layers.0"));
}

#[test]
fn fp8_original_and_its_scale_are_released_and_tombstoned() {
    let gpu = MockGpuBackend::new();
    let prefix = "model.layers.3.self_attn.q_proj";
    let store = WeightStore::from_map(HashMap::from([
        (
            format!("{prefix}.weight"),
            tensor(&gpu, WeightDtype::FP8E4M3, 12288 * 64),
        ),
        (
            format!("{prefix}.weight_scale"),
            tensor(&gpu, WeightDtype::UInt8, 12288 * 2),
        ),
        (
            "model.layers.3.self_attn.q_norm.weight".to_string(),
            tensor(&gpu, WeightDtype::BF16, 256),
        ),
    ]));
    assert_eq!(gpu.alloc_count(), 3);

    let released = release_fp8_store_source(&store, prefix, &gpu).unwrap();
    assert_eq!(released, 12288 * 64 + 12288 * 2);
    assert_eq!(gpu.alloc_count(), 1, "weight and scale freed, norm kept");
    assert!(
        store.get(&format!("{prefix}.weight")).is_err(),
        "fails closed"
    );
    assert!(store.get(&format!("{prefix}.weight_scale")).is_err());
    assert!(store.contains("model.layers.3.self_attn.q_norm.weight"));

    // A second pass sees the tombstone and frees nothing.
    assert_eq!(release_fp8_store_source(&store, prefix, &gpu).unwrap(), 0);
    assert_eq!(gpu.alloc_count(), 1);
}

#[test]
fn non_fp8_sources_are_left_alone() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::from([
        // A BF16 alias the runtime may still hold.
        (
            "a.weight".to_string(),
            tensor(&gpu, WeightDtype::BF16, 1024),
        ),
        // Packed NVFP4 read in place.
        (
            "b.weight".to_string(),
            tensor(&gpu, WeightDtype::UInt8, 512),
        ),
        (
            "b.weight_scale".to_string(),
            tensor(&gpu, WeightDtype::FP8E4M3, 64),
        ),
    ]));
    assert_eq!(release_fp8_store_source(&store, "a", &gpu).unwrap(), 0);
    assert_eq!(release_fp8_store_source(&store, "b", &gpu).unwrap(), 0);
    assert_eq!(gpu.alloc_count(), 3);
    assert!(store.contains("a.weight") && store.contains("b.weight"));
    assert!(release_fp8_store_source(&store, "absent", &gpu).unwrap() == 0);
}

/// Release-budget accounting on the Flash-Next geometry (hidden 2560, 24 query
/// heads x 256 gated, 2 KV heads, GDN 16 key / 48 value heads x 128, MXFP8
/// with one E8M0 byte per 32 weights).
///
/// Distinguishes two paths:
/// * On Flash-Next MXFP8 (`Standard`/`CompressedTensors` attention arm),
///   `attention_bf16` is never allocated and `o_bf16_source` is `None`:
///   `mxfp8_release` frees the 12 layers of `attention_store` (~0.57 GiB) and
///   GDN `in_proj_qkv`/`in_proj_z`/`out_proj`, while `dense_dedup` frees all
///   36 layers of `ssm_store` (~2.00 GiB, incl. `in_proj_a`/`in_proj_b`) plus
///   mixers/routers/shared experts (~1.54 GiB, tested below).
/// * On the `Fp8Dequanted`/`Bf16Raw` attention arm, `dense_dedup` also frees
///   `attention_store` and `attention_bf16` (~1.69 GiB across 12 layers),
///   reaching ~3.69 GiB combined with `ssm_store`.
#[test]
fn flash_next_release_budget() {
    let mxfp8 = |n: usize, k: usize| n * k + n * (k / 32);
    let bf16 = |n: usize, k: usize| n * k * 2;
    let h = 2560;
    let q = 24 * 256 * 2;
    let kv = 2 * 256;
    let o_in = 24 * 256;
    let attention_store = mxfp8(q, h) + 2 * mxfp8(kv, h) + mxfp8(h, o_in);
    let attention_bf16 = bf16(q, h) + 2 * bf16(kv, h) + bf16(h, o_in);
    let qkv = 2 * 16 * 128 + 48 * 128;
    let z = 48 * 128;
    let ssm_store = mxfp8(qkv, h) + mxfp8(z, h) + 2 * mxfp8(48, h) + mxfp8(h, 48 * 128);
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);

    let indexer = mxfp8(640, h);
    assert!((mib(attention_store + indexer) - 607.0 / 12.0).abs() < 0.05);
    assert!((mib(attention_bf16) - 95.0).abs() < 0.1);
    assert!((mib(ssm_store) * 36.0 - 2048.0).abs() < 10.0);

    // Flash-Next MXFP8: `dense_dedup` frees `ssm_store` (~2.00 GiB) and
    // `mxfp8_release` frees `attention_store` (~0.57 GiB), no `attention_bf16`.
    let mxfp8_dense_dedup_ssm_gib = gib(36 * ssm_store);
    assert!(
        (1.95..2.05).contains(&mxfp8_dense_dedup_ssm_gib),
        "{mxfp8_dense_dedup_ssm_gib:.3} GiB"
    );
    let mxfp8_attn_plus_ssm_gib = gib(12 * attention_store + 36 * ssm_store);
    assert!(
        (2.5..2.65).contains(&mxfp8_attn_plus_ssm_gib),
        "{mxfp8_attn_plus_ssm_gib:.3} GiB"
    );

    // `Fp8Dequanted`/`Bf16Raw` attention-dedup arm also drops `attention_bf16`.
    let fp8_dequanted_arm_gib = gib(12 * (attention_store + attention_bf16) + 36 * ssm_store);
    assert!(
        (3.6..3.8).contains(&fp8_dequanted_arm_gib),
        "{fp8_dequanted_arm_gib:.3} GiB"
    );
}

/// Every BF16 q/k/v fallback dispatch in the attention layer is guarded, so a
/// route that reached a released copy errors instead of reading freed memory.
#[test]
fn every_attention_bf16_fallback_is_null_guarded() {
    let sources = [
        include_str!("../../../layers/qwen3_attention/prefill/paged_qkv.rs"),
        include_str!("../../../layers/qwen3_attention/prefill/cache_skip_qkv.rs"),
        include_str!("../../../layers/qwen3_attention/decode/attention_forward.rs"),
        include_str!("../../../layers/qwen3_attention/decode/attention_forward_kv.rs"),
        include_str!("../../../layers/qwen3_attention/trait_impl/multi_seq/qkv.rs"),
    ];
    let mut calls = 0;
    for source in sources {
        let lines: Vec<&str> = source.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed == "ops::dense_gemv(" || trimmed == "ops::dense_gemm_attn_prefill(" {
                calls += 1;
                assert!(
                    i >= 4 && lines[i - 4].trim() == "anyhow::ensure!(",
                    "unguarded BF16 fallback: {trimmed} at line {}",
                    i + 1
                );
                assert!(lines[i - 3].contains(".weight.is_null()"));
            }
        }
    }
    // 14 since the int/v0.9.15.7 merge (9d00431c3): the four
    // ATLAS_QWEN4_ATTN_BF16 decode-only copies (q gated/ungated, k, v) are
    // guarded the same way; source_pins keeps them when ATTN_BF16 is on.
    assert_eq!(calls, 14, "the fallback census changed; re-audit dedup");
}

#[test]
fn hyper_connection_bf16_aliases_are_released_and_tombstoned() {
    let gpu = MockGpuBackend::new();
    let p = "model.language_model.layers.0.attn_hyper_connection";
    let down = tensor(&gpu, WeightDtype::BF16, 320 * 10240);
    let up = tensor(&gpu, WeightDtype::BF16, 10240 * 320);
    let inject = tensor(&gpu, WeightDtype::BF16, 4 * 10240);
    let sources = vec![
        (
            format!("{p}.input_mix_weight_down"),
            DenseWeight { weight: down.ptr },
        ),
        (
            format!("{p}.input_mix_weight_up"),
            DenseWeight { weight: up.ptr },
        ),
        (
            format!("{p}.block_inject_weight"),
            DenseWeight { weight: inject.ptr },
        ),
    ];
    let store = WeightStore::from_map(HashMap::from([
        (format!("{p}.input_mix_weight_down.weight"), down),
        (format!("{p}.input_mix_weight_up.weight"), up),
        (format!("{p}.block_inject_weight.weight"), inject),
        (
            format!("{p}.hc_norm.weight"),
            tensor(&gpu, WeightDtype::FP32, 10240),
        ),
    ]));
    assert_eq!(gpu.alloc_count(), 4);

    let released = release_bf16_sources(&store, &sources, &gpu).unwrap();
    assert_eq!(released, 2 * (2 * 320 * 10240) + 2 * 4 * 10240);
    assert_eq!(gpu.alloc_count(), 1, "the read hc_norm stays");
    assert!(
        store
            .get(&format!("{p}.input_mix_weight_down.weight"))
            .is_err()
    );
    assert!(store.contains(&format!("{p}.hc_norm.weight")));
    // A second pass sees the tombstones and frees nothing.
    assert_eq!(release_bf16_sources(&store, &sources, &gpu).unwrap(), 0);
}

#[test]
fn an_owned_hyper_connection_dequant_is_freed_with_its_fp8_original() {
    let gpu = MockGpuBackend::new();
    let p = "model.language_model.hyper_connection_mixer.input_mix_weight_down";
    let original = tensor(&gpu, WeightDtype::FP8E4M3, 320 * 10240);
    let owned = DenseWeight {
        weight: gpu.alloc(320 * 10240 * 2).unwrap(),
    };
    let store = WeightStore::from_map(HashMap::from([
        (format!("{p}.weight"), original),
        (
            format!("{p}.weight_scale"),
            tensor(&gpu, WeightDtype::UInt8, 320 * 320),
        ),
    ]));
    assert_eq!(gpu.alloc_count(), 3);
    let released = release_bf16_sources(&store, &[(p.to_string(), owned)], &gpu).unwrap();
    assert_eq!(released, 320 * 10240 * 2 + 320 * 10240 + 320 * 320);
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn router_bf16_is_released_only_beside_an_nvfp4_router() {
    let gpu = MockGpuBackend::new();
    let lp = "model.language_model.layers.5";
    let router = tensor(&gpu, WeightDtype::BF16, 512 * 2560);
    let mut gate = DenseWeight { weight: router.ptr };
    let store = WeightStore::from_map(HashMap::from([(format!("{lp}.mlp.gate.weight"), router)]));

    // Without an NVFP4 router the BF16 one is the router: kept.
    assert_eq!(
        release_router_bf16(&store, lp, &mut gate, false, &gpu).unwrap(),
        0
    );
    assert!(!gate.weight.is_null());
    assert_eq!(gpu.alloc_count(), 1);

    assert_eq!(
        release_router_bf16(&store, lp, &mut gate, true, &gpu).unwrap(),
        512 * 2560 * 2
    );
    assert!(
        gate.weight.is_null(),
        "the BF16 router arms now fail closed"
    );
    assert_eq!(gpu.alloc_count(), 0);
    assert!(store.get(&format!("{lp}.mlp.gate.weight")).is_err());
}

#[test]
fn router_that_is_not_the_store_alias_is_left_alone() {
    let gpu = MockGpuBackend::new();
    let lp = "model.language_model.layers.5";
    let router = tensor(&gpu, WeightDtype::BF16, 512 * 2560);
    let other = gpu.alloc(512 * 2560 * 2).unwrap();
    let mut gate = DenseWeight { weight: other };
    let store = WeightStore::from_map(HashMap::from([(format!("{lp}.mlp.gate.weight"), router)]));
    assert_eq!(
        release_router_bf16(&store, lp, &mut gate, true, &gpu).unwrap(),
        0
    );
    assert_eq!(gate.weight, other);
    assert_eq!(gpu.alloc_count(), 2);
}

#[test]
fn shared_expert_mxfp8_originals_are_released() {
    let gpu = MockGpuBackend::new();
    let lp = "model.language_model.layers.7";
    let mut map = HashMap::new();
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let p = format!("{lp}.mlp.shared_expert.{proj}");
        map.insert(
            format!("{p}.weight"),
            tensor(&gpu, WeightDtype::FP8E4M3, 640 * 2560),
        );
        map.insert(
            format!("{p}.weight_scale"),
            tensor(&gpu, WeightDtype::UInt8, 640 * 80),
        );
    }
    map.insert(
        format!("{lp}.mlp.shared_expert_gate.weight"),
        tensor(&gpu, WeightDtype::BF16, 2560),
    );
    let store = WeightStore::from_map(map);
    assert_eq!(gpu.alloc_count(), 7);
    let released = release_shared_expert_sources(&store, lp, &gpu).unwrap();
    assert_eq!(released, 3 * (640 * 2560 + 640 * 80));
    assert_eq!(gpu.alloc_count(), 1, "the read shared_expert_gate stays");
    assert!(store.contains(&format!("{lp}.mlp.shared_expert_gate.weight")));
}

/// The sources the mixer/router/shared-expert releases free on the Flash-Next
/// geometry: the load-log ledger's `weights/other` BF16 (1220 MiB),
/// `weights/dense-ffn` (120 MiB, the routers) and `weights/shared-expert`
/// (232 MiB) rows.
#[test]
fn flash_next_mixer_router_shared_budget() {
    let (h, r, rank, hc, layers) = (2560usize, 10240usize, 320usize, 4usize, 48usize);
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    let hyper = 2 * (rank * r * 2) + hc * r * 2;
    let final_mixer = 2 * (rank * r * 2);
    let mixers = layers * 2 * hyper + final_mixer;
    assert!((mib(mixers) - 1220.1).abs() < 0.1, "{:.1}", mib(mixers));
    let routers = layers * 512 * h * 2;
    assert!((mib(routers) - 120.0).abs() < 0.01);
    let mxfp8 = |n: usize, k: usize| n * k + n * (k / 32);
    let shared = layers * (2 * mxfp8(640, h) + mxfp8(h, 640));
    assert!((mib(shared) - 232.0).abs() < 0.5, "{:.1}", mib(shared));
    let gib = (mixers + routers + shared) as f64 / (1024.0 * 1024.0 * 1024.0);
    assert!((1.5..1.56).contains(&gib), "{gib:.3} GiB");
}

/// Every BF16 router arm of the MoE layer is guarded, so a route that reached
/// a released router errors instead of reading freed memory.
#[test]
fn every_moe_bf16_router_arm_is_null_guarded() {
    let sources = [
        include_str!("../../../layers/moe/forward.rs"),
        include_str!("../../../layers/moe/forward_batched.rs"),
        include_str!("../../../layers/moe/forward_ep.rs"),
        include_str!("../../../layers/moe/forward_k2.rs"),
        include_str!("../../../layers/moe/forward_k3.rs"),
        include_str!("../../../layers/moe/forward_prefill.rs"),
        include_str!("../../../layers/moe/forward_prefill_fp8.rs"),
    ];
    let mut arms = 0;
    for source in sources {
        let lines: Vec<&str> = source.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.trim() != "&self.weights.gate," {
                continue;
            }
            arms += 1;
            let call = (i.saturating_sub(6)..i)
                .rev()
                .find(|&j| {
                    let t = lines[j].trim();
                    // The PREFILL_NATIVE BF16 router arm (int/v0.9.15.7).
                    t.starts_with("ops::dense_gem") || t.ends_with("::router_dense_gemm(")
                })
                .unwrap_or_else(|| panic!("no dense call above line {}", i + 1));
            let guard = (call.saturating_sub(4)..call)
                .any(|j| lines[j].contains("self.weights.gate.weight.is_null()"));
            assert!(guard, "unguarded BF16 router arm above line {}", i + 1);
        }
    }
    // 8 since the int/v0.9.15.7 merge (9d00431c3): forward_prefill's
    // ATLAS_QWEN4_PREFILL_NATIVE router_dense_gemm arm, guarded the same way.
    assert_eq!(arms, 8, "the router arm census changed; re-audit dedup");
}
