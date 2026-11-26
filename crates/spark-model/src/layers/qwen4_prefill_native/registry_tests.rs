// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::mock::MockGpuBackend;

use super::super::Family;
use super::super::e4m3::e4m3_decode;
use super::*;

fn nvfp4_stub(gpu: &MockGpuBackend, n: usize, k: usize) -> QuantizedWeight {
    QuantizedWeight {
        weight: gpu.alloc(n * k / 2).unwrap(),
        weight_scale: gpu.alloc(n * k / 16).unwrap(),
        weight_scale_2: 1.0e-3,
        input_scale: DevicePtr::NULL,
    }
}

/// BF16 values `(i % 7 - 3) * 2^-9`: one block scale, so exactly representable.
fn upload_bf16(gpu: &MockGpuBackend, n: usize, k: usize) -> (DevicePtr, Vec<f32>) {
    let values: Vec<f32> = (0..n * k)
        .map(|i| ((i % 7) as f32 - 3.0) * 2f32.powi(-9))
        .collect();
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let ptr = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(&bytes, ptr).unwrap();
    (ptr, values)
}

fn read_native(gpu: &MockGpuBackend, e: &NativeFp8, len: usize) -> (Vec<f32>, f32) {
    let mut rows = vec![0u8; len];
    gpu.copy_d2h(e.w8, &mut rows).unwrap();
    let mut scale = [0u8; 4];
    gpu.copy_d2h(e.scale, &mut scale).unwrap();
    let s = f32::from_le_bytes(scale);
    (rows.iter().map(|&c| e4m3_decode(c) * s).collect(), s)
}

#[test]
fn mxfp8_entry_is_resident_exact_and_keyed_by_the_nvfp4_copy() {
    let gpu = MockGpuBackend::new();
    let (n, k) = (32usize, 64usize);
    let nvfp4 = nvfp4_stub(&gpu, n, k);
    let (bf16, values) = upload_bf16(&gpu, n, k);
    let mut reg = Registry::default();
    let entry = reg
        .insert_mxfp8(&gpu, Module::GdnQkvz, &nvfp4, bf16, n, k, None, 0)
        .unwrap();
    let (back, s) = read_native(&gpu, &entry, n * k);
    assert_eq!(s, entry.scale_value);
    assert_eq!(
        back, values,
        "single-scale MXFP8 must round-trip bit-exactly"
    );
    assert_eq!(reg.stats.inexact, 0);
    assert_eq!(reg.fp8_bytes, n * k + 256);
    assert_eq!(reg.fp8(&nvfp4, n, k, None).unwrap(), Some(entry));
    assert_eq!(
        reg.fp8(&nvfp4, n, k + 16, None).unwrap(),
        None,
        "shape is part of the key"
    );
    let reused = QuantizedWeight {
        weight_scale_2: 2.0e-3,
        ..nvfp4
    };
    assert_eq!(
        reg.fp8(&reused, n, k, None).unwrap(),
        None,
        "a reused address with another scale2 misses"
    );
    assert!(reg.has_family(Family::Ssm) && !reg.has_family(Family::Attn));
    assert!(
        reg.insert_mxfp8(&gpu, Module::GdnQkvz, &nvfp4, bf16, n, k, None, 0)
            .is_err(),
        "double registration"
    );
}

#[test]
fn gated_q_entry_refuses_a_different_permutation() {
    let gpu = MockGpuBackend::new();
    let (heads, hd, k) = (2usize, 8usize, 32usize);
    let n = 2 * heads * hd;
    let nvfp4 = nvfp4_stub(&gpu, n, k);
    let (bf16, _) = upload_bf16(&gpu, n, k);
    let mut reg = Registry::default();
    reg.insert_mxfp8(
        &gpu,
        Module::AttnQ,
        &nvfp4,
        bf16,
        n,
        k,
        Some((heads, hd)),
        0,
    )
    .unwrap();
    assert!(reg.fp8(&nvfp4, n, k, Some((heads, hd))).unwrap().is_some());
    assert!(reg.fp8(&nvfp4, n, k, None).is_err());
    assert!(reg.fp8(&nvfp4, n, k, Some((heads * 2, hd / 2))).is_err());
}

#[test]
fn router_qg_sentinel_means_no_permutation() {
    let gpu = MockGpuBackend::new();
    let (n, k) = (16usize, 32usize);
    let nvfp4 = nvfp4_stub(&gpu, n, k);
    let (bf16, _) = upload_bf16(&gpu, n, k);
    let mut reg = Registry::default();
    reg.insert_mxfp8(&gpu, Module::AttnK, &nvfp4, bf16, n, k, None, 0)
        .unwrap();
    // `Some((0, 0))` is the router-cache sentinel of the FP8 path, not a permutation.
    assert!(reg.fp8(&nvfp4, n, k, Some((0, 0))).unwrap().is_some());
}

#[test]
fn bf16_and_router_entries_alias_without_allocating() {
    let gpu = MockGpuBackend::new();
    let (n, k) = (320usize, 10240usize);
    let nvfp4 = nvfp4_stub(&gpu, n, k);
    let bf16 = gpu.alloc(n * k * 2).unwrap();
    let mut reg = Registry::default();
    reg.insert_bf16(Module::HcDown, &nvfp4, bf16, n, k).unwrap();
    assert_eq!(reg.bf16(&nvfp4, n, k), Some((bf16, Module::HcDown)));
    assert_eq!(reg.bf16(&nvfp4, k, n), None);
    assert!(
        reg.insert_bf16(Module::GdnOut, &nvfp4, bf16, n, k).is_err(),
        "MXFP8 module through the BF16 table"
    );
    let gate = gpu.alloc(512 * 2560 * 2).unwrap();
    reg.insert_router(gate, 512, 2560).unwrap();
    assert!(reg.router(gate) && !reg.router(bf16));
    assert!(reg.insert_router(gate, 512, 2560).is_err());
    assert_eq!(reg.fp8_bytes, 0);
    assert_eq!(reg.bf16_alias_bytes, n * k * 2 + 512 * 2560 * 2);
    assert_eq!(reg.registered().get("router"), Some(&1));
    assert!(reg.has_family(Family::Hc) && reg.has_family(Family::Router));
}

#[test]
fn rejects_unaligned_shapes_and_wrong_tables() {
    let gpu = MockGpuBackend::new();
    let nvfp4 = nvfp4_stub(&gpu, 16, 32);
    let (bf16, _) = upload_bf16(&gpu, 16, 32);
    let mut reg = Registry::default();
    assert!(
        reg.insert_mxfp8(&gpu, Module::GdnOut, &nvfp4, bf16, 15, 32, None, 0)
            .is_err()
    );
    assert!(
        reg.insert_mxfp8(&gpu, Module::HcUp, &nvfp4, bf16, 16, 32, None, 0)
            .is_err()
    );
    assert!(
        reg.insert_mxfp8(
            &gpu,
            Module::GdnOut,
            &nvfp4,
            DevicePtr::NULL,
            16,
            32,
            None,
            0
        )
        .is_err()
    );
}
