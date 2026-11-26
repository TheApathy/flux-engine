// SPDX-License-Identifier: AGPL-3.0-only

//! Byte-parity oracle: `w4a16_gemm` (baseline M_TILE=64 prefill FFN kernel)
//! vs `w4a16_gemm_pipe` (cp.async double-buffered byte-exact shadow).
//!
//! Both kernels must produce byte-identical outputs on identical NVFP4
//! inputs — same packed nibbles, same E4M3 scales, same scale2, same
//! BF16 activations — at prefill shapes (small M, boundary M=64, masked
//! tail rows). Also prints per-kernel timing at the real M=18 prefill
//! shape to confirm the pipe kernel removes the latency-bound fixed cost.
//!
//! ```text
//! cargo run --release -p spark-model --example w4a16_gemm_pipe_parity
//! ```

use std::time::Instant;

use anyhow::{Result, bail, ensure};
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

// xorshift64* — deterministic, no rand dep.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 40) as u8
    }
    fn finite_bf16(&mut self) -> u16 {
        let unit = ((self.next() >> 40) as f32) / ((1u32 << 24) as f32);
        bf16::from_f32((unit - 0.5) * 12.0).to_bits()
    }
    fn finite_e4m3_scale(&mut self) -> u8 {
        // Finite positive E4M3 scales in 0x30..=0x40 (0.5 .. 2.0).
        0x30 + (self.byte() % 17)
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn launch_gemm(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernel: KernelHandle,
    a: DevicePtr,
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed)
        .arg_ptr(scale)
        .arg_f32(scale2)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

struct Case {
    m: u32,
    n: u32,
    k: u32,
}

fn run_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    base: KernelHandle,
    pipe: KernelHandle,
    case: &Case,
    rng: &mut Rng,
) -> Result<()> {
    let m = case.m as usize;
    let n = case.n as usize;
    let k = case.k as usize;

    let mut a_bytes = vec![0u8; m * k * 2];
    for b in a_bytes.chunks_mut(2) {
        b.copy_from_slice(&rng.finite_bf16().to_le_bytes());
    }
    let mut packed = vec![0u8; n * (k / 2)];
    for b in packed.iter_mut() {
        *b = rng.byte();
    }
    let mut scale = vec![0u8; n * (k / 16)];
    for b in scale.iter_mut() {
        *b = rng.finite_e4m3_scale();
    }
    let scale2 = 1.0f32;

    let a_d = upload(gpu, &a_bytes)?;
    let p_d = upload(gpu, &packed)?;
    let s_d = upload(gpu, &scale)?;
    let c_base = gpu.alloc(m * n * 2)?;
    let c_pipe = gpu.alloc(m * n * 2)?;
    let result = (|| -> Result<()> {
        gpu.synchronize(stream)?;

        // Warm both kernels once (module load / ptx jit), then measure.
        launch_gemm(
            gpu, stream, base, a_d, p_d, s_d, scale2, c_base, case.m, case.n, case.k,
        )?;
        launch_gemm(
            gpu, stream, pipe, a_d, p_d, s_d, scale2, c_pipe, case.m, case.n, case.k,
        )?;
        gpu.synchronize(stream)?;

        // Parity.
        let mut base_out = vec![0u8; m * n * 2];
        let mut pipe_out = vec![0u8; m * n * 2];
        gpu.copy_d2h(c_base, &mut base_out)?;
        gpu.copy_d2h(c_pipe, &mut pipe_out)?;

        let mut any_nonzero = false;
        for (idx, chunk) in base_out.chunks_exact(2).enumerate() {
            let word = u16::from_le_bytes([chunk[0], chunk[1]]);
            ensure!(
                (word & 0x7f80) != 0x7f80,
                "case M={} N={} K={}: non-finite BF16 output 0x{word:04x} at element {idx}",
                case.m,
                case.n,
                case.k
            );
            if (word & 0x7fff) != 0 {
                any_nonzero = true;
            }
        }
        ensure!(
            any_nonzero,
            "case M={} N={} K={}: output is all zeros",
            case.m,
            case.n,
            case.k
        );

        let mut mismatches = 0usize;
        for (i, (b0, b1)) in base_out.iter().zip(pipe_out.iter()).enumerate() {
            if b0 != b1 {
                let row = i / (n * 2);
                let col = (i % (n * 2)) / 2;
                let lo = i & !1;
                if mismatches < 5 {
                    let base_word = u16::from_le_bytes([base_out[lo], base_out[lo + 1]]);
                    let pipe_word = u16::from_le_bytes([pipe_out[lo], pipe_out[lo + 1]]);
                    eprintln!(
                        "  MISMATCH (row={row}, col={col}): base={base_word:04x} pipe={pipe_word:04x}"
                    );
                }
                mismatches += 1;
            }
        }
        if mismatches > 0 {
            bail!(
                "case M={} N={} K={}: {} byte mismatches — pipe kernel NOT bit-exact",
                case.m,
                case.n,
                case.k,
                mismatches
            );
        }
        println!(
            "PASS M={:3} N={:5} K={:5}: byte-identical finite outputs ({})",
            case.m,
            case.n,
            case.k,
            m * n * 2
        );

        // Timing at this shape: 20 alternating launches each, report median.
        let iters = 20;
        let mut base_samples = Vec::with_capacity(iters);
        let mut pipe_samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            launch_gemm(
                gpu, stream, base, a_d, p_d, s_d, scale2, c_base, case.m, case.n, case.k,
            )?;
            gpu.synchronize(stream)?;
            base_samples.push(t0.elapsed().as_secs_f64() * 1000.0);

            let t1 = Instant::now();
            launch_gemm(
                gpu, stream, pipe, a_d, p_d, s_d, scale2, c_pipe, case.m, case.n, case.k,
            )?;
            gpu.synchronize(stream)?;
            pipe_samples.push(t1.elapsed().as_secs_f64() * 1000.0);
        }
        base_samples.sort_by(f64::total_cmp);
        pipe_samples.sort_by(f64::total_cmp);
        let base_ms = (base_samples[iters / 2 - 1] + base_samples[iters / 2]) / 2.0;
        let pipe_ms = (pipe_samples[iters / 2 - 1] + pipe_samples[iters / 2]) / 2.0;
        println!(
            "  time (median): base={base_ms:.3}ms pipe={pipe_ms:.3}ms speedup={:.2}x",
            base_ms / pipe_ms
        );
        Ok(())
    })();
    for ptr in [a_d, p_d, s_d, c_base, c_pipe] {
        let _ = gpu.free(ptr);
    }
    result
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let base = gpu.kernel("w4a16", "w4a16_gemm")?;
    let pipe = gpu.kernel("w4a16", "w4a16_gemm_pipe")?;
    println!(
        "kernels resolved: w4a16_gemm={} w4a16_gemm_pipe={}",
        base.0, pipe.0
    );

    let mut rng = Rng(0x9E3779B97F4A7C15);

    // gate/up shape: N=17408, K=5120 across M values.
    let gate_up = Case {
        m: 130,
        n: 17408,
        k: 5120,
    };
    for m in [3u32, 18, 64, 130] {
        run_case(gpu, stream, base, pipe, &Case { m, ..gate_up }, &mut rng)?;
    }
    // down shape: N=5120, K=17408.
    let down = Case {
        m: 130,
        n: 5120,
        k: 17408,
    };
    for m in [3u32, 18, 64, 130] {
        run_case(gpu, stream, base, pipe, &Case { m, ..down }, &mut rng)?;
    }
    println!("ALL PASS: w4a16_gemm_pipe is byte-identical to w4a16_gemm");
    Ok(())
}
