// SPDX-License-Identifier: AGPL-3.0-only

#![deny(clippy::all)]

pub mod alloc_receipt;
pub mod autoematic;
pub mod buffers;
pub mod byte_resume;
#[cfg(feature = "cuda")]
pub mod cublaslt;
// Metal/no-cuda builds get unreachable stubs so spark-model's unconditional
// references to these cuda-only entry points still resolve (compile-only).
#[cfg(not(feature = "cuda"))]
#[path = "cublaslt_metal_stub.rs"]
pub mod cublaslt;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(unix)]
pub mod fast_weights;
pub mod gpu;
#[allow(clippy::missing_const_for_thread_local)]
pub mod gpu_capture_region;
pub mod kernel_args;
pub mod kernel_audit;
pub mod kv_cache;
pub mod kv_dequant;
pub mod kv_spill;
#[cfg(feature = "metal")]
pub mod metal_backend;
pub mod prefix_cache;
pub mod progress;
pub mod radix_tree;
pub mod sampler;
pub mod scratch_init;
pub mod startup_ledger;
pub mod weights;

#[cfg(test)]
#[path = "weights/gguf_device_test_sha256.rs"]
pub(crate) mod gguf_device_test_sha256;
