// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::DecodeScope;
use super::super::gemv_v2::{TpoPolicy, V2Kernels};
use super::{AttnFp8, Fp8Kernels, GdnFp8, MxFp8Proj, SharedFp8, SharedScratch};

/// Reference variant: one `dense_gemv_mxfp8` launch per matrix.
fn kernels() -> Fp8Kernels {
    Fp8Kernels {
        gemv: KernelHandle(0x51),
        silu_mul: KernelHandle(0x52),
        v2: None,
    }
}

/// `fused_v2` variant.
fn kernels_v2() -> Fp8Kernels {
    Fp8Kernels {
        v2: Some(V2Kernels {
            seg_t32: KernelHandle(0x61),
            seg_t64: KernelHandle(0x62),
            gus_t32: KernelHandle(0x63),
            gus_t64: KernelHandle(0x64),
            tpo: TpoPolicy::Auto,
        }),
        ..kernels()
    }
}

fn buf(gpu: &MockGpuBackend) -> DevicePtr {
    gpu.alloc(64 * 1024).unwrap()
}

/// A checkpoint-like `[n, k]` MXFP8 tensor pair, aliased (no copy).
fn proj(gpu: &MockGpuBackend, n: usize, k: usize) -> MxFp8Proj {
    let codes = gpu.alloc(n * k).unwrap();
    let scales = gpu.alloc(n * k / 32).unwrap();
    MxFp8Proj::alias(codes, scales, n, k).unwrap()
}

#[test]
fn alias_wraps_checkpoint_buffers_without_allocating() {
    let gpu = MockGpuBackend::new();
    let codes = gpu.alloc(8 * 64).unwrap();
    let scales = gpu.alloc(8 * 2).unwrap();
    let before = gpu.alloc_count();
    let p = MxFp8Proj::alias(codes, scales, 8, 64).unwrap();
    assert_eq!(gpu.alloc_count(), before, "aliasing allocates nothing");
    assert_eq!(p.n(), 8);
    assert_eq!(p.checkpoint_bytes(), 8 * 64 + 8 * 2);
}

#[test]
fn alias_rejects_geometry_and_alignment_the_kernel_cannot_read() {
    let gpu = MockGpuBackend::new();
    let codes = gpu.alloc(4096).unwrap();
    let scales = gpu.alloc(128).unwrap();
    assert!(MxFp8Proj::alias(codes, scales, 2, 48).is_err(), "k % 32");
    assert!(MxFp8Proj::alias(codes, scales, 0, 64).is_err());
    assert!(
        MxFp8Proj::alias(codes.offset(8), scales, 2, 64).is_err(),
        "16-byte alignment"
    );
    assert!(MxFp8Proj::alias(DevicePtr::NULL, scales, 2, 64).is_err());
}

#[test]
fn gdn_arm_launches_qkv_and_z_only_inside_the_decode_scope() {
    let gpu = MockGpuBackend::new();
    let gdn = GdnFp8::new(
        3,
        [proj(&gpu, 8, 64), proj(&gpu, 4, 64), proj(&gpu, 64, 32)],
        kernels(),
    )
    .unwrap();
    let (x, y) = (buf(&gpu), buf(&gpu));
    let launches = gpu.launch_count();
    assert!(
        gdn.qkvz(&gpu, x, y, 12, 64, 0).is_none(),
        "prefill keeps NVFP4"
    );
    assert!(gdn.out_proj(&gpu, x, y, 64, 32, 0).is_none());
    assert_eq!(
        (gpu.launch_count() - launches, gdn.receipt().dispatches()),
        (0, 0)
    );

    let _scope = DecodeScope::enter();
    gdn.qkvz(&gpu, x, y, 12, 64, 0).unwrap().unwrap();
    gdn.out_proj(&gpu, x, y, 64, 32, 0).unwrap().unwrap();
    assert_eq!(gpu.launch_count() - launches, 3, "qkv, z, out_proj");
    assert_eq!(gdn.receipt().dispatches(), 2);
    assert_eq!(
        gdn.checkpoint_bytes(),
        (8 * 64 + 16) + (4 * 64 + 8) + (64 * 32 + 64)
    );
}

#[test]
fn geometry_mismatch_fails_without_launch_or_receipt() {
    let gpu = MockGpuBackend::new();
    let gdn = GdnFp8::new(
        0,
        [proj(&gpu, 8, 64), proj(&gpu, 4, 64), proj(&gpu, 64, 32)],
        kernels(),
    )
    .unwrap();
    let (x, y) = (buf(&gpu), buf(&gpu));
    let _scope = DecodeScope::enter();
    assert!(gdn.qkvz(&gpu, x, y, 16, 64, 0).unwrap().is_err());
    assert!(gdn.out_proj(&gpu, x, y, 64, 64, 0).unwrap().is_err());
    assert_eq!((gpu.launch_count(), gdn.receipt().dispatches()), (0, 0));
    assert!(
        GdnFp8::new(
            0,
            [proj(&gpu, 8, 64), proj(&gpu, 4, 32), proj(&gpu, 64, 32)],
            kernels()
        )
        .is_err()
    );
}

#[test]
fn attention_arm_projects_q_kv_and_o() {
    let gpu = MockGpuBackend::new();
    let attn = AttnFp8::new(
        1,
        [
            proj(&gpu, 16, 64),
            proj(&gpu, 4, 64),
            proj(&gpu, 4, 64),
            proj(&gpu, 64, 32),
        ],
        kernels(),
    );
    let (x, a, b) = (buf(&gpu), buf(&gpu), buf(&gpu));
    assert!(attn.kv(&gpu, x, a, b, 4, 64, 0).is_none());
    let _scope = DecodeScope::enter();
    attn.q(&gpu, x, a, 16, 64, 0).unwrap().unwrap();
    attn.kv(&gpu, x, a, b, 4, 64, 0).unwrap().unwrap();
    attn.o(&gpu, x, a, 64, 32, 0).unwrap().unwrap();
    assert_eq!(gpu.launch_count(), 4, "q, k, v, o");
    assert_eq!(attn.receipt().dispatches(), 3);
}

#[test]
fn shared_arm_chains_gate_up_silu_down() {
    let gpu = MockGpuBackend::new();
    let shared = SharedFp8::new(
        2,
        [proj(&gpu, 32, 64), proj(&gpu, 32, 64), proj(&gpu, 64, 32)],
        kernels(),
    )
    .unwrap();
    let scratch = SharedScratch {
        input: buf(&gpu),
        gate: buf(&gpu),
        up: buf(&gpu),
        out: buf(&gpu),
    };
    assert!(
        shared.forward(&gpu, scratch, 96, 0).is_err(),
        "wrong hidden width"
    );
    assert_eq!(gpu.launch_count(), 0);
    shared.forward(&gpu, scratch, 64, 0).unwrap();
    assert_eq!(gpu.launch_count(), 4, "gate, up, silu_mul, down");
    assert_eq!(shared.receipt().dispatches(), 1);
    assert!(!shared.active());
    let _scope = DecodeScope::enter();
    assert!(shared.active());
}

#[test]
fn shared_arm_refuses_shapes_that_do_not_chain() {
    let gpu = MockGpuBackend::new();
    let bad_up = [proj(&gpu, 32, 64), proj(&gpu, 64, 64), proj(&gpu, 64, 32)];
    assert!(SharedFp8::new(0, bad_up, kernels()).is_err());
    let bad_down = [proj(&gpu, 32, 64), proj(&gpu, 32, 64), proj(&gpu, 64, 64)];
    assert!(SharedFp8::new(0, bad_down, kernels()).is_err());
}

#[test]
fn kernel_resolution_returns_live_handles() {
    let gpu = MockGpuBackend::new();
    let resolved = Fp8Kernels::resolve(&gpu).unwrap();
    assert_ne!(resolved.gemv.0, 0);
    assert_ne!(resolved.silu_mul.0, 0);
    let v2 = resolved.v2.expect("fused_v2 is the default variant");
    for h in [v2.seg_t32, v2.seg_t64, v2.gus_t32, v2.gus_t64] {
        assert_ne!(h.0, 0);
    }
}

#[test]
fn fused_v2_gdn_launches_qkvz_once_and_out_proj_once() {
    let gpu = MockGpuBackend::new();
    let gdn = GdnFp8::new(
        3,
        [proj(&gpu, 8, 64), proj(&gpu, 4, 64), proj(&gpu, 64, 32)],
        kernels_v2(),
    )
    .unwrap();
    let (x, y) = (buf(&gpu), buf(&gpu));
    assert!(
        gdn.qkvz(&gpu, x, y, 12, 64, 0).is_none(),
        "prefill keeps NVFP4"
    );
    let _scope = DecodeScope::enter();
    assert!(gdn.qkvz(&gpu, x, y, 16, 64, 0).unwrap().is_err());
    assert!(gdn.out_proj(&gpu, x, y, 64, 64, 0).unwrap().is_err());
    assert_eq!((gpu.launch_count(), gdn.receipt().dispatches()), (0, 0));
    gdn.qkvz(&gpu, x, y, 12, 64, 0).unwrap().unwrap();
    gdn.out_proj(&gpu, x, y, 64, 32, 0).unwrap().unwrap();
    assert_eq!(gpu.launch_count(), 2, "qkv|z fused, out_proj");
    assert_eq!(gdn.receipt().dispatches(), 2);
}

#[test]
fn fused_v2_attention_fuses_k_and_v() {
    let gpu = MockGpuBackend::new();
    let attn = AttnFp8::new(
        1,
        [
            proj(&gpu, 16, 64),
            proj(&gpu, 4, 64),
            proj(&gpu, 4, 64),
            proj(&gpu, 64, 32),
        ],
        kernels_v2(),
    );
    let (x, a, b) = (buf(&gpu), buf(&gpu), buf(&gpu));
    let _scope = DecodeScope::enter();
    assert!(attn.kv(&gpu, x, a, b, 8, 64, 0).unwrap().is_err());
    assert!(attn.q(&gpu, x, a, 16, 32, 0).unwrap().is_err());
    assert_eq!(gpu.launch_count(), 0);
    attn.q(&gpu, x, a, 16, 64, 0).unwrap().unwrap();
    attn.kv(&gpu, x, a, b, 4, 64, 0).unwrap().unwrap();
    attn.o(&gpu, x, a, 64, 32, 0).unwrap().unwrap();
    assert_eq!(gpu.launch_count(), 3, "q, k|v fused, o");
    assert_eq!(attn.receipt().dispatches(), 3);
}

#[test]
fn fused_v2_shared_is_gate_up_silu_then_down() {
    let gpu = MockGpuBackend::new();
    let shared = SharedFp8::new(
        2,
        [proj(&gpu, 32, 64), proj(&gpu, 32, 64), proj(&gpu, 64, 32)],
        kernels_v2(),
    )
    .unwrap();
    let scratch = SharedScratch {
        input: buf(&gpu),
        gate: buf(&gpu),
        up: buf(&gpu),
        out: buf(&gpu),
    };
    assert!(shared.forward(&gpu, scratch, 96, 0).is_err());
    assert_eq!(gpu.launch_count(), 0);
    shared.forward(&gpu, scratch, 64, 0).unwrap();
    assert_eq!(gpu.launch_count(), 2, "gate|up|silu_mul fused, down");
    assert_eq!(shared.receipt().dispatches(), 1);
}
