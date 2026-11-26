// SPDX-License-Identifier: AGPL-3.0-only

//! GB10 parity check for `dense_gemv_mxfp8`. Ignored by default: it needs
//! an idle GPU and a build with the Flash-Next kernel target. Run with
//! `cargo test -p spark-model --lib qwen4_dense_fp8::mxfp8_gpu -- --ignored`.

use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::DecodeScope;
use super::mxfp8::tests::synthetic;
use super::mxfp8::{bf16_to_f32, dequant_bf16, kernel_gemv};
use super::proj::{AttnFp8, Fp8Kernels, MxFp8Proj};
use crate::layers::ops;
use crate::weight_map::DenseWeight;

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let ptr = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(bytes, ptr).unwrap();
    ptr
}

fn read_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Vec<u16> {
    let mut out = vec![0u8; n * 2];
    gpu.synchronize(gpu.default_stream()).unwrap();
    gpu.copy_d2h(ptr, &mut out).unwrap();
    out.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

#[test]
#[ignore = "needs an idle GB10 and the qwen3.8-flash-next kernel target"]
fn native_mxfp8_gemv_matches_host_model_and_dequant_gemv_on_gb10() {
    let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
    let stream = gpu.default_stream();
    let (n, k) = (1024usize, 2560usize);
    let (a, codes, scales) = synthetic(n, k, 7);
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (a_dev, w_dev, s_dev) = (
        upload(&gpu, &a_bytes),
        upload(&gpu, &codes),
        upload(&gpu, &scales),
    );
    let y_dev = gpu.alloc(n * 2).unwrap();

    let proj = MxFp8Proj::alias(w_dev, s_dev, n, k).unwrap();
    let attn = AttnFp8::new(0, [proj; 4], Fp8Kernels::resolve(&gpu).unwrap());
    {
        let _scope = DecodeScope::enter();
        attn.o(&gpu, a_dev, y_dev, n as u32, k as u32, stream)
            .unwrap()
            .unwrap();
    }
    let y = read_bf16(&gpu, y_dev, n);
    assert_eq!(
        y,
        kernel_gemv(&a, &codes, &scales, n, k).unwrap(),
        "GPU == host model"
    );

    // Reference: exact MXFP8 -> BF16 dequant, then the BF16 dense GEMV.
    let w_bf16 = gpu.alloc(n * k * 2).unwrap();
    KernelLaunch::new(
        &gpu,
        gpu.kernel("dequant_mxfp8_bf16", "dequant_mxfp8_bf16")
            .unwrap(),
    )
    .grid([div_ceil(k as u32, 64), div_ceil(n as u32, 4), 1])
    .block([64, 4, 1])
    .arg_ptr(w_dev)
    .arg_ptr(s_dev)
    .arg_ptr(w_bf16)
    .arg_u32(n as u32)
    .arg_u32(k as u32)
    .arg_u32((k / 32) as u32)
    .launch(stream)
    .unwrap();
    let y_ref_dev = gpu.alloc(n * 2).unwrap();
    let dense_k = gpu.kernel("gemv", "dense_gemv_bf16").unwrap();
    ops::dense_gemv(
        &gpu,
        dense_k,
        a_dev,
        &DenseWeight { weight: w_bf16 },
        y_ref_dev,
        n as u32,
        k as u32,
        stream,
    )
    .unwrap();
    let y_ref = read_bf16(&gpu, y_ref_dev, n);
    let w_host = dequant_bf16(&codes, &scales, n, k);
    let mut worst = 0f64;
    for r in 0..n {
        let mag: f64 = w_host[r * k..][..k]
            .iter()
            .zip(&a)
            .map(|(&w, &x)| f64::from(bf16_to_f32(w) * bf16_to_f32(x)).abs())
            .sum();
        let (g, d) = (
            f64::from(bf16_to_f32(y[r])),
            f64::from(bf16_to_f32(y_ref[r])),
        );
        // Both outputs are BF16-rounded; the rest is FP32 accumulation order.
        let bound = d.abs() * 2f64.powi(-7) + mag * 2f64.powi(-14);
        assert!((g - d).abs() <= bound, "row {r}: mxfp8 {g} vs dequant {d}");
        worst = worst.max((g - d).abs() / mag.max(f64::MIN_POSITIVE));
    }
    eprintln!(
        "dense_gemv_mxfp8 vs dequant+dense_gemv_bf16 [{n}x{k}]: worst |dy|/sum|a*w| {worst:.3e}"
    );
}

// ---------------------------------------------------------------------------
// fused_v2 byte equality against the reference kernel.
// ---------------------------------------------------------------------------

use super::gemv_v2::tests::adversarial;
use super::gemv_v2::{TpoPolicy, V2Kernels};

fn upload_case(
    gpu: &dyn GpuBackend,
    n: usize,
    k: usize,
    seed: u64,
    adv: bool,
) -> (DevicePtr, MxFp8Proj) {
    let (a, codes, scales) = if adv {
        adversarial(n, k, seed)
    } else {
        synthetic(n, k, seed)
    };
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let proj = MxFp8Proj::alias(upload(gpu, &codes), upload(gpu, &scales), n, k).unwrap();
    (upload(gpu, &a_bytes), proj)
}

fn ref_gemv(
    gpu: &dyn GpuBackend,
    kernel: spark_runtime::gpu::KernelHandle,
    a: DevicePtr,
    p: &MxFp8Proj,
    out: DevicePtr,
) {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(p.n(), 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(p.weight())
        .arg_ptr(p.scale())
        .arg_ptr(out)
        .arg_u32(p.n())
        .arg_u32(p.k())
        .arg_u32(p.k() / 32)
        .launch(gpu.default_stream())
        .unwrap();
}

/// Every (K, N) the three modules launch on Flash-Next (hidden 2560, GDN
/// qkv 10240 + z 6144, value width 6144, q 12288 gated / 6144, k=v 512,
/// shared inter 640), random and adversarial data (subnormal and NaN E4M3
/// codes, every UE8M0 exponent incl. 0/9/10/246/247/254/255, subnormal and
/// signed-zero activations), both thread mappings, fused segments and the
/// fused gate/up/silu_mul, all compared byte for byte with the reference
/// `dense_gemv_mxfp8` (+ `silu_mul_separate`).
#[test]
#[ignore = "needs an idle GB10 and the qwen3.8-flash-next kernel target"]
fn fused_v2_is_byte_identical_to_reference_dense_gemv_mxfp8_on_gb10() {
    let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
    let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
    let stream = gpu.default_stream();
    let reference = Fp8Kernels::resolve(&gpu).unwrap();
    let mut compared = 0usize;
    for tpo in [32u32, 64] {
        let v2 = V2Kernels::resolve(&gpu, TpoPolicy::Fixed(tpo)).unwrap();
        for adv in [false, true] {
            // Two-segment launches: (n0, n1, k).
            for (n0, n1, k) in [
                (10240usize, 6144usize, 2560usize), // GDN qkv | z
                (512, 512, 2560),                   // attn k | v
                (12288, 0, 2560),                   // attn q (gated)
                (6144, 0, 2560),                    // attn q (ungated)
                (2560, 0, 6144),                    // GDN out_proj, attn o
                (2560, 0, 640),                     // shared down
                (37, 5, 96),                        // ragged tails
            ] {
                let seed = (n0 * 31 + n1 * 7 + k) as u64 ^ u64::from(tpo);
                let (a, p0) = upload_case(&gpu, n0, k, seed, adv);
                let p1 = (n1 > 0).then(|| {
                    let (_, p) = upload_case(&gpu, n1, k, seed ^ 0x77, adv);
                    p
                });
                let (r0, r1) = (
                    gpu.alloc(n0 * 2).unwrap(),
                    gpu.alloc(n1.max(1) * 2).unwrap(),
                );
                let (f0, f1) = (
                    gpu.alloc(n0 * 2).unwrap(),
                    gpu.alloc(n1.max(1) * 2).unwrap(),
                );
                ref_gemv(&gpu, reference.gemv, a, &p0, r0);
                if let Some(p1) = &p1 {
                    ref_gemv(&gpu, reference.gemv, a, p1, r1);
                }
                v2.seg(&gpu, a, (&p0, f0), p1.as_ref().map(|p| (p, f1)), stream)
                    .unwrap();
                assert_eq!(
                    read_bf16(&gpu, f0, n0),
                    read_bf16(&gpu, r0, n0),
                    "seg0 [{n0},{k}] tpo {tpo} adv {adv}"
                );
                if n1 > 0 {
                    assert_eq!(
                        read_bf16(&gpu, f1, n1),
                        read_bf16(&gpu, r1, n1),
                        "seg1 [{n1},{k}] tpo {tpo} adv {adv}"
                    );
                }
                compared += n0 + n1;
            }
            // Shared gate|up|silu_mul against gate GEMV + up GEMV + silu_mul_separate.
            for (n, k) in [(640usize, 2560usize), (37, 96)] {
                let seed = (n + k) as u64 ^ 0xC0FFEE ^ u64::from(tpo);
                let (a, gate) = upload_case(&gpu, n, k, seed, adv);
                let (_, up) = upload_case(&gpu, n, k, seed ^ 0x1234, adv);
                let (rg, ru) = (gpu.alloc(n * 2).unwrap(), gpu.alloc(n * 2).unwrap());
                ref_gemv(&gpu, reference.gemv, a, &gate, rg);
                ref_gemv(&gpu, reference.gemv, a, &up, ru);
                ops::silu_mul(&gpu, reference.silu_mul, rg, ru, rg, n as u32, stream).unwrap();
                let (fa, fu) = (gpu.alloc(n * 2).unwrap(), gpu.alloc(n * 2).unwrap());
                v2.gate_up_silu(&gpu, a, &gate, &up, fa, fu, stream)
                    .unwrap();
                assert_eq!(
                    read_bf16(&gpu, fa, n),
                    read_bf16(&gpu, rg, n),
                    "act [{n},{k}] tpo {tpo} adv {adv}"
                );
                assert_eq!(
                    read_bf16(&gpu, fu, n),
                    read_bf16(&gpu, ru, n),
                    "up [{n},{k}] tpo {tpo} adv {adv}"
                );
                compared += 2 * n;
            }
        }
    }
    eprintln!("fused_v2 == dense_gemv_mxfp8: {compared} BF16 outputs byte-identical");
}
