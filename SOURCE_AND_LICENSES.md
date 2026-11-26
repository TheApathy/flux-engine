# Complete Corresponding Source & License Compliance (`SOURCE_AND_LICENSES.md`)

- **Primary License:** GNU Affero General Public License v3.0 only (`AGPL-3.0-only`) — see [`LICENSE`](LICENSE)
- **Upstream Lineage:** [`MonumentalSystems/atlas`](https://github.com/MonumentalSystems/atlas) & [`Avarok-Cybersecurity/atlas`](https://github.com/Avarok-Cybersecurity/atlas) → [`TheApathy/apathy-atlas`](https://github.com/TheApathy/apathy-atlas) → **[`TheApathy/flux-engine`](https://github.com/TheApathy/flux-engine)**

---

## 1. Origin & Upstream Attribution Notice

Flux Engine is an independently maintained Rust and CUDA/PTX inference runtime developed from a fork of **Atlas**:

- **Monumental Systems ([`MonumentalSystems/atlas`](https://github.com/MonumentalSystems/atlas)):** Original creators and architects of the Atlas pure-Rust LLM inference runtime, workspace crate structure, scheduler, HTTP server, and PTX build system (`AGPL-3.0-only`).
- **Avarok Cybersecurity ([`Avarok-Cybersecurity/atlas`](https://github.com/Avarok-Cybersecurity/atlas)):** Upstream stewards, Blackwell (`sm_120` / `sm_121a`) integration, pure-Rust `xgrammar` port, and Ratatui terminal interface (`AGPL-3.0-only`).
- **TheApathy ([`TheApathy/flux-engine`](https://github.com/TheApathy/flux-engine) & [`TheApathy/apathy-atlas`](https://github.com/TheApathy/apathy-atlas)):** Blackwell (`sm_121a` GB10 & `sm_120` RTX PRO 6000) kernel and runtime stack for `Qwen3.8-Flash-Next-NVFP4` (`v28q`/`v17q` compact W4A8 MoE, 4-stream mHC `t128` roof + `gatepre`, QSA `qkpv_amax_pipe` + `ldmatrix.x4` block scoring, GDN `conv_v3` + `norm_fp8_v2`, 4-ring NVMe PLE DMA, and lossless `K=5` Inferact MTP speculative decoding), `Qwen3.8-27B-NVFP4` (`gamma=15` DFlash block-diffusion speculative decoding), `GLM-5.3-Flash`, and `DeepSeek-V4.1`.

See [`CREDITS.md`](CREDITS.md), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and [`LICENSES/`](LICENSES/) for complete contributor recognition, third-party attributions, and verbatim license texts.

---

## 2. Complete Corresponding Source (`AGPL-3.0 §1, §6, §13`)

Every binary (`spark`, `flux-server`, `flux`) and container image built from `TheApathy/flux-engine` is covered by `AGPL-3.0-only`. The Complete Corresponding Source (CCS) contained in this repository includes:

1. **All 16 Rust Workspace Crates (`crates/*`):**
   - `atlas-core`, `atlas-quant`, `atlas-norm`, `atlas-activation`, `atlas-embed`, `atlas-reduce`, `atlas-kernels`
   - `spark-runtime`, `spark-comm`, `spark-model`, `spark-server`, `spark-storage`, `atlas-spark-bench`
   - `cufile-sys`, `xgrammar`, `rest-store`
2. **All Hardware & Model Kernel Sources (`kernels/**`, `crates/spark-model/csrc/**`, `crates/spark-storage/kernels/**`, `native/flashinfer-sm121/`):**
   - Every `.cu`, `.cuh`, `.cpp`, `.h`, and `.metal` kernel source, plus `HARDWARE.toml`, `MODEL.toml`, and `KERNEL.toml` descriptors compiled into embedded PTX/cubin modules and linked into the runtime binary.
   - **No Precompiled Object Files Shipped:** Prebuilt FlashInfer/CuTe-DSL `.o` object files are intentionally excluded from version control. Default builds compile 100% from source using the native CUDA GDN prefill fallback (`cfg(not(atlas_fi_gdn))`) on both `gb10` (`sm_121a`) and `rtx6000` (`sm_120`); see [`crates/spark-model/csrc/fi_gdn/NOTICE`](crates/spark-model/csrc/fi_gdn/NOTICE).
3. **Build System, Recipes, Chat Templates & Reproducibility Tooling:**
   - Root `Cargo.toml`, `Cargo.lock`, `deny.toml`, `build-support/`, per-crate `build.rs` generators, `docker/` Dockerfiles, `recipes/` operating profiles (`FLUX_*` / `ATLAS_*`), `jinja-templates/`, and `bench/` / `repro-kit/` harnesses.
4. **Remote Network Interaction & Container Distribution (`AGPL-3.0 §13`):**
   - Network-facing deployments serve an interactive Corresponding Source link in the built-in Web UI (`crates/spark-server/src/api/flux_ui.rs`) pointing to `https://github.com/TheApathy/flux-engine`.
   - Production Docker images bundle `LICENSE`, `THIRD_PARTY_NOTICES.md`, `SOURCE_AND_LICENSES.md`, and `CREDITS.md`.
5. **System Libraries Boundary (`AGPL-3.0 §1`):**
   - Target-system NVIDIA CUDA runtime and driver libraries (`libcuda.so`, `libcudart.so`, `libcublasLt.so`, `libnccl.so`, `libcufile.so.0`) are dynamically linked or loaded via `dlopen` from the host/container runtime under the AGPL-3.0 §1 System Libraries exception; no proprietary NVIDIA headers or vendor binaries are checked into this repository.

---

## 3. Third-Party Components Included in or Adapted by the Combined Work

All third-party code included in or adapted by Flux Engine uses licenses compatible with `AGPL-3.0-only`, with full attribution details in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and verbatim license texts in [`LICENSES/`](LICENSES/).

| Component | Upstream Project | License | Location in Tree |
|---|---|---|---|
| **MLC-AI XGrammar (Pure-Rust Port)** | [`mlc-ai/xgrammar`](https://github.com/mlc-ai/xgrammar) (`v0.1.32`) | `Apache-2.0` | `crates/xgrammar/` ([`NOTICE`](crates/xgrammar/NOTICE)) |
| **FlashInfer SM121a/SM120 Attention & Optional GDN Bridge** | [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) | `Apache-2.0` | `crates/spark-model/csrc/fi_gdn/` ([`NOTICE`](crates/spark-model/csrc/fi_gdn/NOTICE)), `crates/spark-model/csrc/fi_attn/`, `native/flashinfer-sm121/` |
| **NVIDIA CUTLASS / CuTe** | [`NVIDIA/cutlass`](https://github.com/NVIDIA/cutlass) | `BSD-3-Clause` | `LICENSES/BSD-3-Clause.txt`, `native/flashinfer-sm121/LICENSES/BSD-3-Clause.txt`, `kernels/gb10/` |
| **`ggml` / `llama.cpp` MMQ & Adaptive Router** | [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) · [`ggml-org/ggml`](https://github.com/ggml-org/ggml) | `MIT` | `kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/`, `crates/spark-model/src/layers/dflash_head/throughput_router.rs` |
| **ExLlamaV3 (`EXL3`) Kernels** | [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3) (`e648f1a1`) | `MIT` (`Copyright (c) 2025 Turboderp`) | `kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_private*`, `crates/atlas-kernels/build.rs` |
| **TurboQuant+ Primitives** | [`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus) | `Apache-2.0` | `kernels/gb10/common/tq_plus_*`, `crates/spark-model/src/weight_loader/qwen35/load_layers/tq_plus_weight_rotation.rs` |
| **`llama-cpp-turboquant` Ports** | [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) (`de44bfe60`) | `MIT` | `kernels/gb10/common/tq_plus_signs.cuh`, `paged_decode_attn_*turbo*.cu`, `reshape_and_cache_turbo.cu`, `wht_bf16.cu` |
| **TensorFold GDN Chain Kernel** | [`ashhart/TensorFold`](https://github.com/ashhart/TensorFold) (`v0.6.6`) | `Apache-2.0` | `kernels/gb10/qwen3.8-flash-next/nvfp4/gdn_prefill_tf_chain.cu` |
| **Flash Linear Attention (`fla`) & SGLang** | [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) · [`sgl-project/sglang`](https://github.com/sgl-project/sglang) | `MIT` / `Apache-2.0` | `kernels/gb10/` (chunked GDN algorithmic references) |
| **DeepSeek-V4.1 & DSpark Reference Ports** | [`deepseek-ai`](https://github.com/deepseek-ai) | `MIT` | `crates/spark-server/src/dsv41/`, `crates/spark-model/src/layers/deepseek_v41_engram/`, `crates/spark-model/src/layers/dflash_head/markov.rs`, `bench/deepseek-v4/dspark_probe/model.py` |
| **`llm-inference-bench` Harness** | [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) | `MIT` (locally modified; prebuilt binaries removed) | `repro-kit/llm-inference-bench/` |
| **GSM8K & MMLU-Pro Datasets** | [`openai/grade-school-math`](https://github.com/openai/grade-school-math) · [`TIGER-AI-Lab/MMLU-Pro`](https://github.com/TIGER-AI-Lab/MMLU-Pro) | `MIT` / `Apache-2.0` | `repro-kit/llm-inference-bench/data/gsm8k_test.jsonl`, `mmlu_pro_1000.jsonl` |
| **Model Chat Templates** | Qwen (`Apache-2.0`), GLM (`MIT`), Mistral (`Apache-2.0`), Gemma (Gemma Terms of Use), Nemotron (NVIDIA Open Model License) | Per upstream model family | `jinja-templates/` (see [`jinja-templates/README.md`](jinja-templates/README.md)) |
| **Rust Crates (`crates.io`)** | 51 direct / 339 external packages (355 total workspace + external entries in `Cargo.lock`) | `MIT`, `Apache-2.0`, `BSD`, `ISC`, `Zlib`, `Unlicense`, `CDLA-Permissive-2.0` | `Cargo.toml`, `Cargo.lock`, [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §3 |
