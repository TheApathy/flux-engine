// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::{DenseFfnLayer, DenseFfnWeights, SharedE2m1Scratch, merged_gate_up_views};
use crate::weight_map::QuantizedWeight;

fn layer(gpu: &MockGpuBackend) -> DenseFfnLayer {
    let weights = DenseFfnWeights {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
    };
    DenseFfnLayer::new(weights, gpu).expect("mock dense FFN layer")
}

#[test]
fn merged_views_read_exactly_the_separate_gate_and_up_bytes() {
    let gpu = MockGpuBackend::new();
    let gate_bytes: Vec<u8> = (0..512).map(|i| (i * 7 + 3) as u8).collect();
    let up_bytes: Vec<u8> = (0..512).map(|i| (i * 13 + 11) as u8).collect();
    let merged = gpu.alloc(1024).unwrap();
    let mut gate_then_up = gate_bytes.clone();
    gate_then_up.extend_from_slice(&up_bytes);
    gpu.copy_h2d(&gate_then_up, merged).unwrap();

    let (gate, up) = merged_gate_up_views(merged, gate_bytes.len()).unwrap();
    assert_eq!(gate, merged);
    let mut read = vec![0_u8; 512];
    gpu.copy_d2h(gate, &mut read).unwrap();
    assert_eq!(read, gate_bytes);
    gpu.copy_d2h(up, &mut read).unwrap();
    assert_eq!(read, up_bytes);
}

#[test]
fn merged_views_reject_null_and_unaligned_extents() {
    let gpu = MockGpuBackend::new();
    let merged = gpu.alloc(1024).unwrap();
    assert!(merged_gate_up_views(spark_runtime::gpu::DevicePtr::NULL, 512).is_err());
    assert!(merged_gate_up_views(merged, 0).is_err());
    assert!(merged_gate_up_views(merged, 100).is_err());
    // Qwen3.8-27B: 17408 x 5120 / 2 packed gate bytes.
    assert!(merged_gate_up_views(merged, 17_408 * 5_120 / 2).is_ok());
}

#[test]
fn adopting_layers_share_one_arena_and_one_stream_binding() {
    let gpu = MockGpuBackend::new();
    let shared = SharedE2m1Scratch::default();
    let mut first = layer(&gpu);
    let mut second = layer(&gpu);
    first.adopt_shared_e2m1_scratch(&shared).unwrap();
    second.adopt_shared_e2m1_scratch(&shared).unwrap();

    let before = gpu.alloc_count();
    let maximum = first.ensure_e2m1_scratch(&gpu, 8_192, 17_408).unwrap();
    let reused = second.ensure_e2m1_scratch(&gpu, 2_079, 5_120).unwrap();
    assert_eq!(maximum, reused);
    assert_eq!(gpu.alloc_count() - before, 3, "packed, scales, absmax once");

    drop(first.lock_e2m1_use(7).unwrap());
    assert!(second.lock_e2m1_use(9).is_err());
    drop(second.lock_e2m1_use(7).unwrap());
}

#[test]
fn private_arenas_stay_per_layer() {
    let gpu = MockGpuBackend::new();
    let first = layer(&gpu);
    let second = layer(&gpu);
    let before = gpu.alloc_count();
    let a = first.ensure_e2m1_scratch(&gpu, 8_192, 17_408).unwrap();
    let b = second.ensure_e2m1_scratch(&gpu, 8_192, 17_408).unwrap();
    assert_ne!(a.0, b.0);
    assert_eq!(gpu.alloc_count() - before, 6);
    drop(first.lock_e2m1_use(7).unwrap());
    drop(second.lock_e2m1_use(9).unwrap());
}

#[test]
fn a_shared_arena_grows_in_place_for_every_adopter() {
    let gpu = MockGpuBackend::new();
    let shared = SharedE2m1Scratch::default();
    let mut first = layer(&gpu);
    let mut second = layer(&gpu);
    first.adopt_shared_e2m1_scratch(&shared).unwrap();
    second.adopt_shared_e2m1_scratch(&shared).unwrap();
    let before = gpu.alloc_count();
    let small = first.ensure_e2m1_scratch(&gpu, 128, 5_120).unwrap();
    let grown = second.ensure_e2m1_scratch(&gpu, 4_096, 17_408).unwrap();
    assert_ne!(small.0, grown.0);
    assert_eq!(
        first.ensure_e2m1_scratch(&gpu, 4_096, 17_408).unwrap(),
        grown
    );
    assert_eq!(
        gpu.alloc_count() - before,
        3,
        "the outgrown arena was freed"
    );
}

#[test]
fn adoption_after_a_private_allocation_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut owned = layer(&gpu);
    owned.ensure_e2m1_scratch(&gpu, 128, 5_120).unwrap();
    let error = owned
        .adopt_shared_e2m1_scratch(&SharedE2m1Scratch::default())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("before the layer allocates its own"),
        "{error}"
    );
}
