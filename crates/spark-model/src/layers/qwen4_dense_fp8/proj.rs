// SPDX-License-Identifier: AGPL-3.0-only

//! Native MXFP8 projections (checkpoint E4M3 codes + UE8M0 block-32 scales,
//! aliased from the weight store) and the per-layer bundles the decode arms
//! call. Legacy launches go through [`MxFp8Proj::gemv`] and default fused-v2
//! launches go through [`V2Kernels`], checking call-site geometry against the
//! stored matrices; the layer receipt is bumped only after the kernels were
//! enqueued.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::gemv_v2::{self, GemvVariant, V2Kernels};
use super::mxfp8::{MXFP8_BLOCK, validate_geometry};
use super::{LayerReceipt, Module, in_decode_scope};
use crate::layers::ops;

/// Module and function of the block-32 UE8M0 GEMV
/// (`kernels/gb10/common/dense_gemv_mxfp8.cu`, module = file stem).
pub(crate) const GEMV_MODULE: &str = "dense_gemv_mxfp8";
pub(crate) const GEMV_FUNC: &str = "dense_gemv_mxfp8";

/// One `[n, k]` MXFP8 matrix as the checkpoint ships it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MxFp8Proj {
    weight: DevicePtr,
    scale: DevicePtr,
    n: u32,
    k: u32,
}

impl MxFp8Proj {
    /// Wrap resident checkpoint buffers. The kernel reads 16 codes per
    /// `uint4`, so the code buffer must be 16-byte aligned (rows then are,
    /// since k is a multiple of 32).
    pub(crate) fn alias(weight: DevicePtr, scale: DevicePtr, n: usize, k: usize) -> Result<Self> {
        validate_geometry(n, k, k / MXFP8_BLOCK)?;
        ensure!(
            !weight.is_null() && !scale.is_null() && weight.0.is_multiple_of(16),
            "MXFP8 projection buffers must be non-null and the codes 16-byte aligned"
        );
        Ok(Self {
            weight,
            scale,
            n: u32::try_from(n).context("MXFP8 projection n exceeds u32")?,
            k: u32::try_from(k).context("MXFP8 projection k exceeds u32")?,
        })
    }

    pub(crate) fn n(&self) -> u32 {
        self.n
    }

    pub(crate) fn k(&self) -> u32 {
        self.k
    }

    pub(crate) fn weight(&self) -> DevicePtr {
        self.weight
    }

    pub(crate) fn scale(&self) -> DevicePtr {
        self.scale
    }

    /// The call-site geometry must match the stored matrix.
    fn check(&self, n: u32, k: u32) -> Result<()> {
        ensure!(
            n == self.n && k == self.k,
            "MXFP8 projection is [{},{}] but the decode call site asks for [{n},{k}]",
            self.n,
            self.k
        );
        Ok(())
    }

    /// Checkpoint bytes this projection reads (codes + exponents).
    pub(crate) fn checkpoint_bytes(&self) -> u64 {
        let elements = u64::from(self.n) * u64::from(self.k);
        elements + elements / MXFP8_BLOCK as u64
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv(
        &self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        input: DevicePtr,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        self.check(n, k)?;
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(self.weight)
            .arg_ptr(self.scale)
            .arg_ptr(output)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(k / MXFP8_BLOCK as u32)
            .launch(stream)
    }
}

/// Kernel handles every bundle needs, resolved once at load.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fp8Kernels {
    /// Reference `dense_gemv_mxfp8` (one launch per matrix).
    pub(crate) gemv: KernelHandle,
    pub(crate) silu_mul: KernelHandle,
    /// `Some` = `fused_v2` launches (bit-identical; see `gemv_v2.rs`).
    pub(crate) v2: Option<V2Kernels>,
}

impl Fp8Kernels {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let gemv = gpu
            .kernel(GEMV_MODULE, GEMV_FUNC)
            .context("ATLAS_QWEN4_DENSE_FP8=1 needs dense_gemv_mxfp8 in this target's PTX")?;
        let silu_mul = gpu
            .kernel("residual_add", "silu_mul_separate")
            .context("ATLAS_QWEN4_DENSE_FP8=1 needs silu_mul_separate in this target's PTX")?;
        ensure!(
            gemv.0 != 0 && silu_mul.0 != 0,
            "ATLAS_QWEN4_DENSE_FP8=1 resolved a null kernel handle"
        );
        let (variant, tpo) = gemv_v2::selected()?;
        let v2 = match variant {
            GemvVariant::FusedV2 => Some(V2Kernels::resolve(gpu, tpo)?),
            GemvVariant::Reference => None,
        };
        Ok(Self { gemv, silu_mul, v2 })
    }
}

/// Run `f` only inside the decode scope; record the receipt on success.
fn in_scope(receipt: &LayerReceipt, f: impl FnOnce() -> Result<()>) -> Option<Result<()>> {
    in_decode_scope().then(|| f().map(|()| receipt.record()))
}

/// GDN `in_proj_qkv` and `in_proj_z` (two GEMVs into adjacent slices of the
/// sequential `[Q|K|V|Z]` output) and `out_proj` of one linear-attention layer.
#[derive(Debug)]
pub(crate) struct GdnFp8 {
    qkv: MxFp8Proj,
    z: MxFp8Proj,
    out_proj: MxFp8Proj,
    kernels: Fp8Kernels,
    receipt: LayerReceipt,
}

impl GdnFp8 {
    pub(crate) fn new(
        layer: usize,
        [qkv, z, out_proj]: [MxFp8Proj; 3],
        kernels: Fp8Kernels,
    ) -> Result<Self> {
        ensure!(
            qkv.k == z.k && out_proj.n == qkv.k,
            "GDN MXFP8 shapes qkv[{},{}] z[{},{}] out[{},{}] do not chain",
            qkv.n,
            qkv.k,
            z.n,
            z.k,
            out_proj.n,
            out_proj.k
        );
        gemv_v2::emit_receipt(kernels.v2.as_ref(), &[qkv.k, out_proj.k]);
        Ok(Self {
            qkv,
            z,
            out_proj,
            kernels,
            receipt: LayerReceipt::new(Module::Gdn, layer),
        })
    }

    pub(crate) fn checkpoint_bytes(&self) -> u64 {
        [&self.qkv, &self.z, &self.out_proj]
            .iter()
            .map(|p| p.checkpoint_bytes())
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn receipt(&self) -> &LayerReceipt {
        &self.receipt
    }

    /// `None` outside the decode scope: the caller keeps its NVFP4 chain.
    /// `n` is the full qkvz width the call site expects.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qkvz(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        in_scope(&self.receipt, || {
            ensure!(
                n == self.qkv.n + self.z.n,
                "GDN qkvz call site asks for {n} rows, MXFP8 holds {} + {}",
                self.qkv.n,
                self.z.n
            );
            let z_out = output.offset(self.qkv.n as usize * 2);
            if let Some(v2) = &self.kernels.v2 {
                self.qkv.check(self.qkv.n, k)?;
                self.z.check(self.z.n, k)?;
                return v2.seg(
                    gpu,
                    input,
                    (&self.qkv, output),
                    Some((&self.z, z_out)),
                    stream,
                );
            }
            self.qkv
                .gemv(gpu, self.kernels.gemv, input, output, self.qkv.n, k, stream)?;
            self.z
                .gemv(gpu, self.kernels.gemv, input, z_out, self.z.n, k, stream)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn out_proj(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        in_scope(&self.receipt, || {
            if let Some(v2) = &self.kernels.v2 {
                self.out_proj.check(n, k)?;
                return v2.seg(gpu, input, (&self.out_proj, output), None, stream);
            }
            self.out_proj
                .gemv(gpu, self.kernels.gemv, input, output, n, k, stream)
        })
    }
}

/// Attention q (checkpoint layout, gate interleaved when gated), k, v, o.
#[derive(Debug)]
pub(crate) struct AttnFp8 {
    q: MxFp8Proj,
    k: MxFp8Proj,
    v: MxFp8Proj,
    o: MxFp8Proj,
    kernels: Fp8Kernels,
    receipt: LayerReceipt,
}

impl AttnFp8 {
    pub(crate) fn new(layer: usize, [q, k, v, o]: [MxFp8Proj; 4], kernels: Fp8Kernels) -> Self {
        gemv_v2::emit_receipt(kernels.v2.as_ref(), &[q.k, k.k, v.k, o.k]);
        Self {
            q,
            k,
            v,
            o,
            kernels,
            receipt: LayerReceipt::new(Module::Attn, layer),
        }
    }

    pub(crate) fn checkpoint_bytes(&self) -> u64 {
        [&self.q, &self.k, &self.v, &self.o]
            .iter()
            .map(|p| p.checkpoint_bytes())
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn receipt(&self) -> &LayerReceipt {
        &self.receipt
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn q(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        in_scope(&self.receipt, || {
            if let Some(v2) = &self.kernels.v2 {
                self.q.check(n, k)?;
                return v2.seg(gpu, input, (&self.q, output), None, stream);
            }
            self.q
                .gemv(gpu, self.kernels.gemv, input, output, n, k, stream)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn kv(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        in_scope(&self.receipt, || {
            if let Some(v2) = &self.kernels.v2 {
                self.k.check(n, k)?;
                self.v.check(n, k)?;
                return v2.seg(gpu, input, (&self.k, k_out), Some((&self.v, v_out)), stream);
            }
            self.k
                .gemv(gpu, self.kernels.gemv, input, k_out, n, k, stream)?;
            self.v
                .gemv(gpu, self.kernels.gemv, input, v_out, n, k, stream)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn o(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        in_scope(&self.receipt, || {
            if let Some(v2) = &self.kernels.v2 {
                self.o.check(n, k)?;
                return v2.seg(gpu, input, (&self.o, output), None, stream);
            }
            self.o
                .gemv(gpu, self.kernels.gemv, input, output, n, k, stream)
        })
    }
}

/// Scratch the shared-expert arm writes; all `[n]` BF16 device buffers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SharedScratch {
    pub(crate) input: DevicePtr,
    pub(crate) gate: DevicePtr,
    pub(crate) up: DevicePtr,
    pub(crate) out: DevicePtr,
}

/// Shared expert gate/up/down: `down(silu(gate(x)) * up(x))`.
#[derive(Debug)]
pub(crate) struct SharedFp8 {
    gate: MxFp8Proj,
    up: MxFp8Proj,
    down: MxFp8Proj,
    kernels: Fp8Kernels,
    receipt: LayerReceipt,
}

impl SharedFp8 {
    pub(crate) fn new(
        layer: usize,
        [gate, up, down]: [MxFp8Proj; 3],
        kernels: Fp8Kernels,
    ) -> Result<Self> {
        ensure!(
            gate.n == up.n && gate.k == up.k && down.k == gate.n && down.n == gate.k,
            "shared expert MXFP8 shapes gate[{},{}] up[{},{}] down[{},{}] do not chain",
            gate.n,
            gate.k,
            up.n,
            up.k,
            down.n,
            down.k
        );
        gemv_v2::emit_receipt(kernels.v2.as_ref(), &[gate.k, down.k]);
        Ok(Self {
            gate,
            up,
            down,
            kernels,
            receipt: LayerReceipt::new(Module::Shared, layer),
        })
    }

    pub(crate) fn checkpoint_bytes(&self) -> u64 {
        [&self.gate, &self.up, &self.down]
            .iter()
            .map(|p| p.checkpoint_bytes())
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn receipt(&self) -> &LayerReceipt {
        &self.receipt
    }

    /// True when this call must replace the fused NVFP4 shared expert.
    pub(crate) fn active(&self) -> bool {
        in_decode_scope()
    }

    /// Writes `scratch.out = down(silu(gate(x)) * up(x))`; `scratch.gate`
    /// is reused in place for the activation. `hidden` is the model width.
    pub(crate) fn forward(
        &self,
        gpu: &dyn GpuBackend,
        scratch: SharedScratch,
        hidden: u32,
        stream: u64,
    ) -> Result<()> {
        let (inter, h) = (self.gate.n, self.gate.k);
        ensure!(
            hidden == h,
            "shared expert MXFP8 is [{inter},{h}] but the layer width is {hidden}"
        );
        if let Some(v2) = &self.kernels.v2 {
            // Same bytes in scratch.gate (activation), scratch.up and
            // scratch.out as the four-launch chain below.
            v2.gate_up_silu(
                gpu,
                scratch.input,
                &self.gate,
                &self.up,
                scratch.gate,
                scratch.up,
                stream,
            )?;
            v2.seg(gpu, scratch.gate, (&self.down, scratch.out), None, stream)?;
            self.receipt.record();
            return Ok(());
        }
        let gemv = self.kernels.gemv;
        self.gate
            .gemv(gpu, gemv, scratch.input, scratch.gate, inter, h, stream)?;
        self.up
            .gemv(gpu, gemv, scratch.input, scratch.up, inter, h, stream)?;
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            scratch.gate,
            scratch.up,
            scratch.gate,
            inter,
            stream,
        )?;
        self.down
            .gemv(gpu, gemv, scratch.gate, scratch.out, h, inter, stream)?;
        self.receipt.record();
        Ok(())
    }
}

#[cfg(test)]
#[path = "proj_tests.rs"]
mod tests;
