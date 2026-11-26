// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::error::{AtlasError, Result};

/// A loaded CUDA kernel module with cached function handles.
///
/// Wraps cudarc's CudaModule to provide a clean interface for loading
/// PTX source (compiled at build time by build.rs) and launching kernels.
pub struct KernelModule {
    module: Arc<CudaModule>,
}

impl KernelModule {
    /// Load a PTX module from source string into the given CUDA context.
    ///
    /// The PTX source should be compiled at build time via nvcc --ptx
    /// and embedded with include_str!.
    pub fn from_ptx_src(ctx: &Arc<CudaContext>, ptx_src: &str) -> Result<Self> {
        let ptx = Ptx::from_src(ptx_src);
        let module = ctx
            .load_module(ptx)
            .map_err(|e| AtlasError::ModuleLoad(format!("PTX load failed: {e}")))?;
        Ok(Self { module })
    }

    /// Get a kernel function handle by name.
    pub fn get_function(&self, name: &str) -> Result<CudaFunction> {
        self.module
            .load_function(name)
            .map_err(|e| AtlasError::ModuleLoad(format!("Function '{name}' not found: {e}")))
    }
}

/// Launch configuration helper for SM121.
pub fn launch_config(n: u32, block_size: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(block_size), 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch vector_add kernel: `C[i] = A[i] + B[i]`.
///
/// Uses cudarc's `launch_builder` API with `u64` device pointers (`u64`
/// implements `DeviceRepr`, so no FFI conversion is needed).
///
/// # Safety
///
/// - `func` must refer to a kernel whose signature matches `(f32*, f32*, f32*, u32)`.
/// - `a_ptr`, `b_ptr`, and `c_ptr` must be valid CUDA device pointers to `f32` arrays
///   of length `>= n` (`c_ptr` writable and non-aliasing with `a_ptr`/`b_ptr` unless
///   the kernel supports in-place updates) and remain live until `stream` completes.
pub unsafe fn launch_vector_add(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    n: u32,
) -> Result<()> {
    let cfg = launch_config(n, 256);
    // SAFETY: Caller guarantees `func` matches `(f32*, f32*, f32*, u32)` and all
    // device pointers are valid for `n` elements through `stream` completion.
    unsafe {
        stream
            .launch_builder(func)
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&c_ptr)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| AtlasError::KernelLaunch(format!("vector_add launch failed: {e}")))?;
    }
    Ok(())
}
