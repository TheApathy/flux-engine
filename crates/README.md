<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Rust Workspace Domain Crates (<code>crates/</code>)</b></p>

# Flux Engine Rust Workspace Crates (`crates/`)

**Flux Engine** is structured as a **16-crate** pure-Rust workspace with zero Python/PyTorch runtime dependencies. Eleven crates provide the active production runtime, kernels, model implementations, storage, collective communication, and HTTP server; five crates (`atlas-quant`, `atlas-norm`, `atlas-activation`, `atlas-embed`, `atlas-reduce`) are thin trait-interface stubs reserved for future modular decomposition while their active CUDA/PTX implementations live in `atlas-kernels`, `spark-runtime`, and `spark-model`.

---

## Active Runtime & Kernel Crates (11)

| Layer | Crate | Role & Key Abstractions |
|---|---|---|
| **HTTP, Scheduler & CLI Surface** | [`spark-server/`](spark-server/) | Binary crate (`flux-server`, `flux-cli`, `spark`): OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`) & Anthropic (`/v1/messages`) HTTP server, continuous-batching scheduler (`scheduler/`), speculative decode orchestration (`K=5` MTP / `gamma=15` DFlash), SSE streaming, tool-call parsers, TUI, and CLI entrypoint. |
| **GPU Runtime & KV Cache** | [`spark-runtime/`](spark-runtime/) | `GpuBackend` trait (`AtlasCudaBackend`, `MetalBackend`, `MockGpuBackend`), CUDA/Metal stream & graph management, cuBLASLt wrappers, UMA headroom/carveout guards, paged KV-cache block allocator, radix prefix cache, TurboQuant+ / FP8 KV dequantization, and host sampler. |
| **Model Execution** | [`spark-model/`](spark-model/) | `Model`, `ModelWeightLoader`, `TransformerLayer`, chunked prefill (`w16384` / `w4096`), and speculative verify implementations (`Qwen3.8-Flash-Next`, `Qwen3.8-27B`, `DeepSeek-V4.1`, `GLM-5.3`, `Gemma-4`, `Qwen3.6/3.5`, `MiniMax-M2.7`, `Nemotron-3`). |
| **CUDA / Metal Kernel Build** | [`atlas-kernels/`](atlas-kernels/) | Build-time (`build.rs`) `(hardware × model × quant)` kernel discovery, parallel `nvcc` / Metal compilation, embedded PTX/CUBIN/metallib lookup tables, and `MODEL.toml` behavior/sampling metadata (`NVFP4`, `MXFP8`, `EXL3`, `TurboQuant+`, `RMSNorm`, `mHC`, `SwiGLU`, `MRoPE`, `PLE`). |
| **Core Primitives** | [`atlas-core/`](atlas-core/) | Hardware capability detection (`capabilities.rs`), device tensor/dtype descriptors, CUDA/Metal stream & kernel descriptors, and model config parsers. |
| **Collective Comm** | [`spark-comm/`](spark-comm/) | Multi-node `NCCL` / 200GbE `RoCEv2` and TCP Tensor-Parallel / Expert-Parallel (`TP=2/4`, `EP=2/4`) communication backends. |
| **Storage & GDS** | [`spark-storage/`](spark-storage/) · [`cufile-sys/`](cufile-sys/) · [`rest-store/`](rest-store/) | High-speed `io_uring` NVMe KV-cache swap, 4-ring sector-aligned PLE DMA, GPUDirect Storage (`cuFile`) FFI bindings, and retrieval-based speculative drafting store (`rest-store`). |
| **Structured Decoding** | [`xgrammar/`](xgrammar/) | Safe pure-Rust implementation of pushdown-automaton JSON schema / EBNF grammar-constrained token masking. |
| **Microbenchmarks** | [`atlas-spark-bench/`](atlas-spark-bench/) | Standalone kernel-level microbenchmark harness for isolated GEMM, GDN, QSA, and MoE timing on Blackwell GPUs. |

---

## Reserved Interface Stub Crates (5)

| Crate | Status | Notes |
|---|---|---|
| [`atlas-quant/`](atlas-quant/) | **Interface stub** | Reserved trait seam for standalone quantization extraction; active `NVFP4`/`MXFP8`/`EXL3`/`TurboQuant+` kernels currently reside in `atlas-kernels/`, `spark-runtime/`, and `kernels/`. |
| [`atlas-norm/`](atlas-norm/) | **Interface stub** | Reserved trait seam for normalization extraction; active `RMSNorm` and 4-stream `mHC` kernels currently reside in `atlas-kernels/` and `spark-model/`. |
| [`atlas-activation/`](atlas-activation/) | **Interface stub** | Reserved trait seam for activation extraction; active `SwiGLU`/`GeGLU` kernels currently reside in `atlas-kernels/` and `spark-model/`. |
| [`atlas-embed/`](atlas-embed/) | **Interface stub** | Reserved trait seam for embedding/RoPE extraction; active `MRoPE`/`YaRN` and 4-ring `PLE` loaders currently reside in `spark-model/` and `spark-storage/`. |
| [`atlas-reduce/`](atlas-reduce/) | **Interface stub** | Reserved trait seam for reduction extraction; active argmax/top-`k`/softmax reductions currently reside in `atlas-kernels/` and `spark-runtime/`. |
