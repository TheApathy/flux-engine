# FlashInfer/CUTLASS SM121 native boundary

This directory builds an out-of-tree, Python/Torch/TVM-free C ABI shared
library (`libatlas_fi_fp4_sm121.so`) around FlashInfer 0.6.6's SM120/SM121
CUTLASS NVFP4 GEMM. It is not statically linked into Cargo's default build;
at runtime on `(aarch64, SM121)`, `spark-server` dynamically loads it via
`ATLAS_FLASHINFER_SM121_LIB` with fail-closed SHA-256 preflight
(`crates/spark-server/src/main_modules/flashinfer_artifact.rs`) and dispatches
admitted prefill GEMMs through `crates/spark-model/src/layers/ops/flashinfer_sm121.rs`
(`ATLAS_PREFILL_{FFN,PROJ,SSM}_FLASHINFER`).

## Build and static verification

The default dependency paths select the locally installed FlashInfer 0.6.6
source bundle. Override `FLASHINFER_DATA_ROOT`, `FLASHINFER_LICENSE_FILE`,
`CUDA_HOME`, or `BUILD_DIR` when reproducing elsewhere. Dependency verification
is fail-closed, `ATLAS_FI_SKIP_DEP_VERIFY=1` is diagnostic only and must not be
used for a qualified artifact.

```sh
bash native/flashinfer-sm121/build.sh
bash native/flashinfer-sm121/test-static.sh
```

The build emits `build/libatlas_fi_fp4_sm121.so`, compiled only for
`compute_121a,sm_121a`. It contains BF16 output instantiations for three tiles,
each with data-parallel and StreamK schedules. The version script exports only:

- `atlas_fi_nvfp4_sm121_workspace_size`
- `atlas_fi_nvfp4_sm121_bf16`
- `atlas_fi_nvfp4_sm121_last_error`

Both operational functions catch all C++ exceptions and return integer status
codes. The last-error message is thread-local. The CUDA stream is an opaque
pointer in the public header so Rust does not need CUDA headers.

## ABI and current link seam

Inputs are group-16 NVFP4: packed E2M1 A row-major and B column-major, UE4M3
128x4-interleaved scale factors, a device FP32 global-scale pointer, and BF16
row-major output. Tactic IDs 0 through 5 are the three CTA shapes in source
order, each data-parallel then StreamK. Workspace is shape- and tactic-specific.
query it once, retain a fixed device allocation, and freeze the tactic before
CUDA graph capture.

Runtime Rust integration uses dynamic loading (`libloading` / `dlopen`) rather
than static Cargo `build.rs` linking: `spark-server` validates the SONAME path
and SHA-256 preflight (`crates/spark-server/src/main_modules/flashinfer_artifact.rs`),
binds the three exported C ABI symbols in `crates/spark-model/src/layers/ops/flashinfer_sm121.rs`,
owns the workspace on the active CUDA stream, and fails closed on architecture,
artifact hash, layout, workspace, or status errors. ModelOpt weights are routed
only when the 128x4 scale-layout admission checks pass.

No runtime or GPU qualification is implied by the static build. Required gates
remain workspace query, all-tactic `can_implement`, canaries/immutability,
finite/numerical quality, checkpoint layout, graph replay, and balanced timing
at the exact Qwen3.8 projection shapes.
