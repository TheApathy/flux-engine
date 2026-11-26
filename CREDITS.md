# Credits & Upstream Attribution

> Flux Engine is an independently maintained Rust inference project developed from a fork of Atlas. We recognize the original Atlas authors and the wider open-source and research communities whose work contributed to its foundation. Flux's ongoing development, architecture, and release roadmap are maintained independently. See [Credits](CREDITS.md), [Third-Party Notices](THIRD_PARTY_NOTICES.md), [Source and Licenses](SOURCE_AND_LICENSES.md), and [License](LICENSE) for project lineage, contributor recognition, and required notices.

---

## 1. Foundational Upstream Attribution: Atlas

**Flux Engine** (`TheApathy/flux-engine`) is built directly upon the **Atlas** pure-Rust LLM inference engine and CUDA kernel stack. We extend our highest recognition and gratitude to the creators, architects, and contributors of the upstream Atlas repositories:

1. **Monumental Systems: [`MonumentalSystems/atlas`](https://github.com/MonumentalSystems/atlas)**
   - **Contribution:** Original inception and foundational architecture of the Atlas pure-Rust LLM inference runtime, workspace crate hierarchy (`atlas-core`, `atlas-kernels`, `spark-runtime`, `spark-model`, `spark-server`, `spark-storage`), scheduler, HTTP/OpenAI-compatible server, and hardware-targeted CUDA PTX compilation bridge.
   - **License:** GNU Affero General Public License v3.0 only (`AGPL-3.0-only`).

2. **Avarok Cybersecurity: [`Avarok-Cybersecurity/atlas`](https://github.com/Avarok-Cybersecurity/atlas)**
   - **Contribution:** Upstream stewardship, Blackwell (`sm_120` / `sm_121a`) runtime integration, pure-Rust `xgrammar` structured-generation port, Ratatui terminal interface (`spark-server/src/tui`), and multi-model inference foundation.
   - **License:** GNU Affero General Public License v3.0 only (`AGPL-3.0-only`).

3. **Fork Lineage Baseline: [`TheApathy/apathy-atlas`](https://github.com/TheApathy/apathy-atlas)**
   - **Contribution:** Direct upstream fork baseline preserving the full commit history, `Qwen3.8-27B` DFlash speculative decoding work, and attribution bridge between `MonumentalSystems/atlas` / `Avarok-Cybersecurity/atlas` and `TheApathy/flux-engine`.

---

## 2. Individual Contributors

We gratefully recognize the engineers and open-source contributors whose code, fixes, and validation directly strengthened Atlas and Flux Engine (listed by public GitHub handle):

- **[`AzeezIsh`](https://github.com/AzeezIsh)** — Upstream Atlas maintainer and core runtime, scheduler, server, and Blackwell engineering contributions.
- **[`nero-`](https://github.com/nero-)** — Kernel optimization, engineering contributions, and DGX Spark (`GB10`) model and benchmark validation.
- **[`maplepaladin73`](https://github.com/maplepaladin73)** — SSM slot-compaction reliability fixes, PLE n-gram trellis decode, and `qwen4_exp` multi-file weight/kernel testing and fixes.
- **`TheApathy` & Upstream Atlas Contributors** — Ongoing Blackwell (`sm_121a` / `sm_120`) kernel engineering, model bring-up, testing, and documentation.

---

## 3. GPU Kernel, Quantization, Research & Ecosystem Credits

Flux Engine's Blackwell execution pipeline builds upon open-source GPU kernel research, model ecosystems, and vendor libraries:

| Project / Organization | Repository / URL | Area of Contribution |
|---|---|---|
| **Atlas (`Monumental Systems` & `Avarok Cybersecurity`)** | [`MonumentalSystems/atlas`](https://github.com/MonumentalSystems/atlas) · [`Avarok-Cybersecurity/atlas`](https://github.com/Avarok-Cybersecurity/atlas) | Core pure-Rust LLM inference runtime, workspace crate architecture, scheduler, `sm_121a` Blackwell kernel stack, and `qwen4_exp` model target. |
| **Mia (`MiaAI-Lab`)** | [`MiaAI-Lab`](https://github.com/MiaAI-Lab) | `Qwen3.8-Flash-Next` mixed-precision `NVFP4`/`MXFP8` (`QAD5500-Hybrid` & PLE Offload) checkpoint quantization, single-DGX-Spark (`GB10`) inference research, and long-context evaluation benchmarks. |
| **Local Inference Lab (`LIL`)** | [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) | Standardized multi-context prefill/decode, multi-stream concurrency, and `--test-profile estonia` 136.5K long-context verification benchmark harness (`MIT`). |
| **Qwen Team (Alibaba Cloud)** | [`QwenLM`](https://github.com/QwenLM) | Qwen3.5 / Qwen3.8 hybrid Gated DeltaNet (GDN) + QSA + MoE + mHC + PLE architectures, Inferact Multi-Token Prediction (MTP) design, and Qwen chat templates (`Apache-2.0`). |
| **DeepSeek-AI Research** | [`deepseek-ai`](https://github.com/deepseek-ai) | Research and reference implementations for Manifold-Constrained Hyper-Connections (`mHC`, [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)), Lightning Indexer sparse attention (`DSA`), Engram n-gram tables, DSML tool encoding, and Multi-Token Prediction (`MTP`). |
| **FlashInfer Contributors** | [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) | Chunked Gated DeltaNet (GDN) WY state recurrence primitives and FlashAttention-2 paged/ragged attention kernels ([arXiv:2501.01005](https://arxiv.org/abs/2501.01005), `Apache-2.0`). |
| **Dao-AILab (FlashAttention)** | [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) | FlashAttention-2 & FlashAttention-4 ([arXiv:2307.08691](https://arxiv.org/abs/2307.08691), [arXiv:2603.05451](https://arxiv.org/abs/2603.05451)) online softmax rescaling, Flash-Decoding split-K reductions, and polynomial `sw_exp`. |
| **NVIDIA CUTLASS, ModelOpt & CUDA Toolkit** | [`NVIDIA/cutlass`](https://github.com/NVIDIA/cutlass) · [`NVIDIA/TensorRT-Model-Optimizer`](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | Block-scaled Tensor Core MMA (`MXFP8` / `NVFP4` `E2M1` + `UE8M0`), `ldmatrix` / `cp.async` collective load patterns, ModelOpt mixed-precision quantization formats (`QAD`), and `cublasLtMatmul` device-scaled FP8 heuristics (`BSD-3-Clause` for CUTLASS). |
| **MLC-AI XGrammar** | [`mlc-ai/xgrammar`](https://github.com/mlc-ai/xgrammar) | Original C++ pushdown-automaton grammar-constrained decoding design (`v0.1.32`, `Apache-2.0`), ported line-by-line to 100% safe Rust in `crates/xgrammar`. |
| **`ggml` & `llama.cpp` Contributors** | [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) · [`ggml-org/ggml`](https://github.com/ggml-org/ggml) | MMQ and block-dequantization CUDA headers (`q4k_vendor`, `MIT`) and adaptive speculative draft-depth controller (`llama.cpp` PR #27210). |
| **ExLlamaV3 (`Turboderp`)** | [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3) | `EXL3` quantization format, Hadamard transforms, and MoE kernel foundations (`MIT`, Copyright (c) 2025 Turboderp). |
| **TurboQuant & TurboQuant+ (`TheTom`)** | [`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus) · [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) | Canonical Randomized Hadamard Transform (`WHT`) with two-sided Rademacher sign masks, $N(0, 1/d)$ Lloyd-Max codebooks, sparse-V dequantization, and asymmetric KV cache compression ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), `Apache-2.0` / `MIT`). |
| **TensorFold Contributors** | [`ashhart/TensorFold`](https://github.com/ashhart/TensorFold) | Token-serial FP32 GDN prefill `chain_kernel` design (`v0.6.6`, `Apache-2.0`). |
| **Flash Linear Attention (`fla`) & SGLang** | [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) · [`sgl-project/sglang`](https://github.com/sgl-project/sglang) | Chunked Gated DeltaNet algorithmic references and serving-system design insights (`MIT` / `Apache-2.0`). |
| **SpecForge & DFlash Contributors** | [`z-lab/dflash`](https://github.com/z-lab/dflash) | Block-diffusion speculative drafting (`DFlash`) and tree-attention verification architecture. |
| **Open-Weight Model & Benchmark Communities** | [`zai-org`](https://huggingface.co/zai-org) · [`mistralai`](https://huggingface.co/mistralai) · [`google`](https://huggingface.co/google) · [`nvidia`](https://huggingface.co/nvidia) · [`MiniMaxAI`](https://huggingface.co/MiniMaxAI) · [`openai/grade-school-math`](https://github.com/openai/grade-school-math) · [`TIGER-AI-Lab/MMLU-Pro`](https://github.com/TIGER-AI-Lab/MMLU-Pro) | Upstream model architectures, chat template formats (`jinja-templates/`), and open evaluation datasets (`GSM8K`, `MMLU-Pro`). |

---

## 4. Rust Ecosystem & Crate Authors

Flux Engine relies on the work of the Rust open-source community (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §3 for the complete table of direct dependencies and licenses):

- **CUDA & Low-Level FFI:** `cudarc`, `libc`, `libloading`, `io-uring` (Tokio contributors), `memmap2`, `objc2` / `dispatch2`.
- **Async Runtime & HTTP Serving:** `tokio`, `tokio-stream`, `axum`, `tower-http`, `http-body`, `futures`, `ureq`.
- **Tokenization, Templates & Serialization:** `tokenizers` (Hugging Face), `safetensors` (Hugging Face), `minijinja` (Armin Ronacher), `serde`, `serde_json` (David Tolnay & contributors), `toml`, `aho-corasick` & `walkdir` (Andrew Gallant).
- **Concurrency, Allocators & Data Structures:** `parking_lot` (Amanieu d'Antras), `rayon` (Niko Matsakis & Josh Stone), `dashmap`, `bitvec`, `ahash`, `mimalloc`, `half`, `rand`.
- **Observability, CLI & Terminal UI:** `tracing`, `tracing-subscriber`, `log`, `prometheus`, `clap`, `ratatui`, `crossterm`, `tui-textarea`, `criterion`.
- **Cryptography & Integrity:** `ring` (Brian Smith; `Apache-2.0 AND ISC`), `base64`, `thiserror`, `anyhow`.

---

## 5. `TheApathy/` Engineering Contributions

Key engineering additions in `TheApathy/flux-engine` and `TheApathy/apathy-atlas`:

- **`v28q` & `v17q` Compact Transposed W4A8 MoE Grouped GEMMs:** 16-warp B-weight transcoding, dual-occupancy (`2 CTAs/SM`) down-projection, zero-`sE` register-quad shuffle MXFP8 epilogues, and zero-allocation shared-expert pretranscoding.
- **4-Stream mHC Hyper-Connection Roofline Pipeline:** `PIN_M=4096` cuBLASLt heuristic admission, split `qwen4_hc_prep_up` L2 residency, `qwen4_hc_upmix_roof_t128`, and zero-spill warp-per-row `qwen4_hc_combine_moe_gemm_mx_gatepre`.
- **QSA Sparse Attention Pipeline:** 5-CTA/SM `ldmatrix.x4` block scoring (`96` regs, `0` spills), bimodal `topk_v2` / `topk_fused_c3_v2` selection with `32 MiB` scratch expansion, and `cp.async` double-buffered `qkpv_amax_pipe` with 16-token half-tile skipping.
- **GDN Linear Attention Fusions:** 64-row fused `qwen4_fp8_act_and_ba_gates_mma`, shared-memory staged `qwen4_conv1d_l2norm_f16_prefill_v3_commit`, native CUDA GDN prefill fallback with optional source-built FlashInfer bridge (`Mode::FiStatic` / `qwen4_gdn_norm_fp8`), and `WY17` lazy state commit (`ATLAS_SSM_GDN_LAZY=1`).
- **Zero-Overhead NVFP4 PLE Streaming:** Pre-deinterleaved `Arc<PackedPleBlock>` RAM cache (`0` TLB shootdowns) and 4-ring sector-aligned `PleRowsPlan::Pipelined`.
- **Lossless `K=5` Inferact MTP & `gamma=15` DFlash Speculative Decode:** Device-chained 4-draft MTP proposer, batched 5-row mHC & QSA verification (`pool_endpoint` fix + 0-ULP multi-row Flash-Decoding), unaligned tail-chunk (`rows >= 2`) wide prefill support, and DFlash block-diffusion verification.
