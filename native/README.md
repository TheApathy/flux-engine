<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Native C++/CUDA Sidecars &amp; CUTLASS/FlashInfer Extensions (<code>native/</code>)</b></p>

# Native C++/CUDA Prototypes (`native/`)

Isolated, standalone C++/CUDA `sm_121a` shared-library prototypes (not linked into the default Cargo/`spark-server` build). Each subdirectory builds and validates its own C ABI boundary independently.

---

## Modules

| Directory | Purpose |
|---|---|
| [`flashinfer-sm121/`](flashinfer-sm121/README.md) | Isolated Python/Torch/TVM-free C ABI prototype wrapping FlashInfer / CUTLASS `sm_121a` NVFP4 (`E2M1` + `UE4M3`) GEMM (`libatlas_fi_fp4_sm121.so`). |
| [`qwen38-ssm-residual-sm121/`](qwen38-ssm-residual-sm121/) | Isolated `sm_121a` C ABI prototype for NVFP4 quantization residual recovery (`BF16(A - dequant(Q(A)))`) and two-branch BF16 output addition (`libatlas_qwen38_ssm_residual.so`). |
