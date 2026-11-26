# Source Code Availability, OCI Layer Mapping & License Disclosures

- **Kit Version:** `Flux Reproducibility Kit`
- **Engine Release:** `ghcr.io/theapathy/flux-engine:v1.0.2`
- **Primary License:** **GNU Affero General Public License v3.0 only (`SPDX-License-Identifier: AGPL-3.0-only`)**
- **Sanitization Policy:** `FLUX-SEC-0001`

---

## 1. Upstream Lineage & AGPL-3.0 Corresponding Source Disclosure

**Flux Engine** (`TheApathy/flux-engine`) is an independently maintained Rust and CUDA inference runtime developed from a fork of **Atlas**:

- **Original Upstream Repositories:**
  - [`MonumentalSystems/atlas`](https://github.com/MonumentalSystems/atlas). Copyright (c) Monumental Systems and Atlas Contributors (`AGPL-3.0-only`).
  - [`Avarok-Cybersecurity/atlas`](https://github.com/Avarok-Cybersecurity/atlas). Copyright (c) Avarok Cybersecurity and Atlas Contributors (`AGPL-3.0-only`).
- **Flux Engine Modifications & SM121a/SM120 Kernel Stack:**
  - [`TheApathy/flux-engine`](https://github.com/TheApathy/flux-engine). Copyright (c) 2026 TheApathy Contributors (`AGPL-3.0-only`).

### Complete Corresponding Source (AGPL-3.0 §1, §6, §13)

In compliance with Sections 1, 6, and 13 of the GNU Affero General Public License v3.0, the **Complete Corresponding Source (CCS)** for binaries distributed inside `ghcr.io/theapathy/flux-engine` corresponds to the exact git revisions pinned in [`ENGINE_STABLE_PIN.json`](ENGINE_STABLE_PIN.json) and [`CONTAINER_BUILD_RECEIPT.json`](CONTAINER_BUILD_RECEIPT.json):

The Complete Corresponding Source includes:
1. **Rust Workspace Crates:** `crates/atlas-core`, `crates/atlas-kernels`, `crates/spark-runtime`, `crates/spark-comm`, `crates/spark-model`, `crates/spark-storage`, `crates/spark-server`, `crates/cufile-sys`, `crates/xgrammar`, `crates/rest-store`, `Cargo.toml`, and `Cargo.lock`.
2. **CUDA Kernel Translation Units (`.cu` / `.cuh`):** All CUDA translation units under `kernels/gb10/` and `kernels/rtx6000/`, storage kernels under `crates/spark-storage/kernels/`, and C++/CUDA sources under `crates/spark-model/csrc/` (including the FlashInfer CuTe-DSL export scripts for `csrc/fi_gdn/`).
3. **Target & Codegen Descriptors:** All `HARDWARE.toml`, `MODEL.toml`, `KERNEL.toml`, `build-support/`, and `build.rs` code generators required to produce the embedded PTX/cubin modules and link `/usr/local/bin/flux-server`.

---

## 2. OCI Container Layer Mapping (`ghcr.io/theapathy/flux-engine`)

| Container Path | Source Origin in Repository | License | Description |
| :--- | :--- | :--- | :--- |
| `/opt/flux/bin/flux-server-real` | `TheApathy/flux-engine` (`crates/spark-server`) | `AGPL-3.0-only` | Stripped AArch64/x86_64 inference server binary with embedded SM121a/SM120 PTX/cubin kernels (symlinked at `/usr/local/bin/flux-server`) |
| `/usr/local/bin/flux` | `repro-kit/flux-autodetect` | `AGPL-3.0-only` | Hardware auto-detection, diagnostics, and serving CLI (symlinked at `/usr/local/bin/flux-autodetect`) |
| `/usr/local/bin/flux-bench` | `repro-kit/flux-bench` | `AGPL-3.0-only` | Zero-dependency Python 3 stdlib benchmark & hardware inspection CLI |
| `/usr/local/bin/flux-lens` | `repro-kit/flux_lens.py` | `AGPL-3.0-only` | Exact-token context/concurrency trial runner |
| `/usr/local/bin/lil-bench` | `repro-kit/lil-bench` | `AGPL-3.0-only` | Turnkey `llm-inference-bench` launcher and card generator |
| `/opt/flux/llm-inference-bench` | `repro-kit/llm-inference-bench/` | `MIT` | Vendored `local-inference-lab/llm-inference-bench` (`v0.7.9`, locally patched for Flux Engine; no precompiled ELFs) |
| `/opt/flux/repro-kit/benchmark-config.yaml` | `repro-kit/benchmark-config.yaml` | `AGPL-3.0-only` | Declarative benchmark protocol, hardware targets, and pass/fail gates |
| `/opt/flux/repro-kit/model-manifest.json` | `repro-kit/model-manifest.json` | `AGPL-3.0-only` | Checkpoint size & SHA-256 verification manifest |
| `/opt/flux/repro-kit/recipes/*.env` | `repro-kit/recipes/*.env` | `AGPL-3.0-only` | Sanitized runtime operating recipes (`max-prefill`, `interactive-w4096`, `4x-spark`, etc.) |
| `/opt/flux/repro-kit/METHODOLOGY.md` | `repro-kit/METHODOLOGY.md` | `AGPL-3.0-only` | Complete reviewer Q&A and measurement methodology |
| `/opt/flux/repro-kit/SOURCE_AND_LICENSES.md` | `repro-kit/SOURCE_AND_LICENSES.md` | `AGPL-3.0-only` | Complete Corresponding Source & third-party license notices |

---

## 3. Third-Party Open-Source Notices

| Component | Upstream Project | SPDX License | Usage in Flux Engine |
| :--- | :--- | :--- | :--- |
| **Atlas Inference Engine** | [`MonumentalSystems/atlas`](https://github.com/MonumentalSystems/atlas), [`Avarok-Cybersecurity/atlas`](https://github.com/Avarok-Cybersecurity/atlas) | `AGPL-3.0-only` | Core Rust workspace and runtime foundation |
| **FlashInfer** | [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) | `Apache-2.0` | SM121a CuTe-DSL Gated DeltaNet chunked recurrence (`csrc/fi_gdn`) & FlashAttention-2 (`csrc/fi_attn`) |
| **NVIDIA CUTLASS / CuTe** | [`NVIDIA/cutlass`](https://github.com/NVIDIA/cutlass) | `BSD-3-Clause` | Block-scaled NVFP4 / FP8 Tensor Core GEMM templates |
| **XGrammar (Rust Port)** | [`mlc-ai/xgrammar`](https://github.com/mlc-ai/xgrammar) (`v0.1.32`) | `Apache-2.0` | Pure-Rust port of structured decoding engine (`crates/xgrammar`) |
| **ExLlamaV3** | [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3) | `MIT` | Optional `EXL3` quantization kernels for `glm5.3-flash` |
| **GGML / llama.cpp** | [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) | `MIT` | Vendored quantized GEMM/embedding headers (`q4k_vendor`) |
| **TurboQuant+** | [`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus) & [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) (`arXiv:2504.19874`) | `Apache-2.0` / `MIT` | Walsh-Hadamard transform & KV cache compression primitives |
| **llm-inference-bench** | [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) (`v0.7.9`) | `MIT` | Vendored benchmark & accuracy harness (`repro-kit/llm-inference-bench/`, with local Flux Engine patches; see [`llm-inference-bench/LICENSE`](llm-inference-bench/LICENSE)) |
| **GSM8K Dataset** | [`openai/grade-school-math`](https://github.com/openai/grade-school-math) | `MIT` | Bundled grade-school math evaluation split (`repro-kit/llm-inference-bench/data/gsm8k_test.jsonl`) |
| **MMLU-Pro Dataset** | [`TIGER-AI-Lab/MMLU-Pro`](https://github.com/TIGER-AI-Lab/MMLU-Pro) | `Apache-2.0` | Bundled 1,000-item MMLU-Pro evaluation split (`repro-kit/llm-inference-bench/data/mmlu_pro_1000.jsonl`) |
| **NVIDIA CUDA Runtime** | `nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04` | NVIDIA CUDA EULA | Standard target-platform dynamic system libraries (`libcuda.so`, `libcudart.so`, `libcublasLt.so`) under AGPL-3.0 §1 System Libraries exception |
