# Third-Party Notices & License Disclosures

> Flux Engine is an independently maintained Rust inference project developed from a fork of Atlas. We recognize the original Atlas authors and the wider open-source and research communities whose work contributed to its foundation. Flux's ongoing development, architecture, and release roadmap are maintained independently. See [Credits](CREDITS.md), [Source and Licenses](SOURCE_AND_LICENSES.md), and [License](LICENSE) for project lineage, contributor recognition, and required notices.

---

## 1. Primary Upstream License (`AGPL-3.0-only`)

**Flux Engine** (`TheApathy/flux-engine`) is a derivative work of **Atlas**:

- **Original Upstream Repositories:**
  - [`MonumentalSystems/atlas`](https://github.com/MonumentalSystems/atlas) — Copyright (c) Monumental Systems and Atlas Contributors.
  - [`Avarok-Cybersecurity/atlas`](https://github.com/Avarok-Cybersecurity/atlas) — Copyright (c) Avarok Cybersecurity and Atlas Contributors.
- **Fork Baseline Repository:**
  - [`TheApathy/apathy-atlas`](https://github.com/TheApathy/apathy-atlas)
- **Flux Engine Derivative Modifications:**
  - Copyright (c) 2026 TheApathy Contributors.
- **Governing License:** **GNU Affero General Public License v3.0 only (`SPDX-License-Identifier: AGPL-3.0-only`)** — full text in [`LICENSE`](LICENSE).

Under the terms of the GNU Affero General Public License v3.0:
1. All modified files preserve required copyright, attribution, and modification notices.
2. All source distributions, container/binary distributions, and network-facing deployments of the covered work preserve the `AGPL-3.0-only` license and Corresponding Source obligations under Sections 6 and 13 of the AGPL-3.0.

---

## 2. Embedded, Ported & Adapted Third-Party Components

Full verbatim copies of the standard permissive licenses referenced below are provided in [`LICENSES/`](LICENSES/).

### 2.1 MLC-AI XGrammar (Pure-Rust Derivative Port in `crates/xgrammar`)
- **Upstream Project:** [`mlc-ai/xgrammar`](https://github.com/mlc-ai/xgrammar) (`v0.1.32`)
- **Upstream Copyright:** Copyright (c) 2024 MLC-AI Contributors
- **Upstream License:** Apache License, Version 2.0 (`Apache-2.0`, [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt))
- **Adaptation & Modification Notice (Apache-2.0 §4):** `crates/xgrammar` is a zero-`unsafe` (`#![forbid(unsafe_code)]`) pure-Rust line-by-line derivative port of the upstream C++ core of `mlc-ai/xgrammar` `v0.1.32` (see `crates/xgrammar/src/lib.rs` and [`crates/xgrammar/NOTICE`](crates/xgrammar/NOTICE)), extended with Rust-native support types and additional XML tool-calling grammar converters. It is distributed in this workspace under `AGPL-3.0-only` in accordance with Apache-2.0 one-way compatibility with GPLv3/AGPLv3 and Apache-2.0 §4 attribution requirements.

### 2.2 FlashInfer Kernels & Optional Gated DeltaNet Bridge
- **Upstream Project:** [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer)
- **Upstream Copyright:** Copyright (c) 2023–2026 FlashInfer Contributors
- **Upstream License:** Apache License, Version 2.0 (`Apache-2.0`, [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt))
- **Usage:** Source-level SM121a/SM120 attention integration (`crates/spark-model/csrc/fi_attn/`, `native/flashinfer-sm121/`) and optional Gated DeltaNet bridge sources (`crates/spark-model/csrc/fi_gdn/`). Precompiled CuTe-DSL `.o` object files are intentionally not distributed in this repository; default builds use the native CUDA GDN prefill fallback (see [`crates/spark-model/csrc/fi_gdn/NOTICE`](crates/spark-model/csrc/fi_gdn/NOTICE)).

### 2.3 NVIDIA CUTLASS & Dynamic CUDA System Libraries
- **Upstream Project:** [`NVIDIA/cutlass`](https://github.com/NVIDIA/cutlass) & NVIDIA CUDA Toolkit
- **Upstream Copyright:** Copyright (c) 2017–2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- **Upstream License:** BSD 3-Clause License (`BSD-3-Clause`, [`LICENSES/BSD-3-Clause.txt`](LICENSES/BSD-3-Clause.txt)) for CUTLASS; NVIDIA Software License Agreement / EULA for target-system CUDA runtime libraries.
- **System Libraries Boundary (AGPL-3.0 §1):** Zero NVIDIA proprietary CUDA Toolkit headers or compiled vendor binaries are committed to version control. Flux Engine dynamically links or `dlopen`s system-installed CUDA libraries (`libcuda.so`, `libcudart.so`, `libcublasLt.so`, `libnccl.so`, `libcufile.so.0`) on the target platform under the AGPL-3.0 §1 System Libraries exception.

### 2.4 `ggml` & `llama.cpp` Quantization Headers & Adaptive Speculative Router
- **Upstream Project:** [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) & [`ggml-org/ggml`](https://github.com/ggml-org/ggml)
- **Upstream Copyright:** Copyright (c) 2023–2025 The ggml authors (Georgi Gerganov and contributors)
- **Upstream License:** MIT License (`MIT`, [`LICENSES/MIT.txt`](LICENSES/MIT.txt))
- **Usage:**
  - Vendored MMQ and block-dequantization CUDA headers under `kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/` (14 files, preserving MIT license headers and `LICENSE`), `#include`d by `kernels/gb10/glm5.3-flash/iq3/`.
  - Adaptive draft-depth hysteresis controller (`ClimbDropRouter`) ported from `llama.cpp` PR #27210 (`common/speculative-adaptive.h`) in `crates/spark-model/src/layers/dflash_head/throughput_router.rs`.

### 2.5 ExLlamaV3 (`EXL3`) Quantization & Hadamard Kernels
- **Upstream Project:** [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3) (commit `e648f1a131365aae15920073e761a3fa5a527654`)
- **Upstream Copyright:** Copyright (c) 2025 Turboderp
- **Upstream License:** MIT License (`MIT`, [`LICENSES/MIT.txt`](LICENSES/MIT.txt))
- **Usage:** Derived EXL3 MoE and Hadamard CUDA kernel files (`kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_private.cu` and `kernels/gb10/glm5.3-flash/exl3/glm53_exl3_moe_private_hadamard.cuh`, retaining `SPDX-License-Identifier: MIT` and upstream copyright) plus optional pinned build-time integration in `crates/atlas-kernels/build.rs`.

### 2.6 TurboQuant+ & `llama-cpp-turboquant`
- **Upstream Projects:**
  - [`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus) — Apache License, Version 2.0 (`Apache-2.0`, [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt))
  - [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) (including commit `de44bfe60`) — MIT License (`MIT`, [`LICENSES/MIT.txt`](LICENSES/MIT.txt))
- **Upstream Copyright:** Copyright (c) 2025–2026 TheTom and contributors
- **Usage:** Randomized Walsh-Hadamard Transform (`WHT`) sign masks (`kernels/gb10/common/tq_plus_signs.cuh`), inner quantization and weight rotation (`kernels/gb10/common/tq_plus_innerq*`, `crates/spark-model/src/weight_loader/qwen35/load_layers/tq_plus_weight_rotation.rs`), and sparse-V dequantization in `kernels/gb10/common/paged_decode_attn_*turbo*.cu`, `reshape_and_cache_turbo.cu`, and `wht_bf16.cu`.

### 2.7 TensorFold Gated DeltaNet Prefill Kernel
- **Upstream Project:** [`ashhart/TensorFold`](https://github.com/ashhart/TensorFold) (`v0.6.6`)
- **Upstream Copyright:** Copyright (c) 2026 TensorFold contributors
- **Upstream License:** Apache License, Version 2.0 (`Apache-2.0`, [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt))
- **Usage:** Adapted FP32 token-serial GDN prefill recurrence (`chain_kernel` adaptation) in `kernels/gb10/qwen3.8-flash-next/nvfp4/gdn_prefill_tf_chain.cu`.

### 2.8 Flash Linear Attention (`fla`) & SGLang
- **Upstream Projects:**
  - [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) — MIT License (`MIT`, [`LICENSES/MIT.txt`](LICENSES/MIT.txt)); Copyright (c) 2023–2025, Songlin Yang, Yu Zhang
  - [`sgl-project/sglang`](https://github.com/sgl-project/sglang) — Apache License, Version 2.0 (`Apache-2.0`, [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)); Copyright (c) SGLang Contributors
- **Usage:** Chunked Gated DeltaNet algorithmic references and CUDA kernel adaptations in `kernels/gb10/`.

### 2.9 DeepSeek-V4.1 & DSpark Reference Ports
- **Upstream Project:** [`deepseek-ai`](https://github.com/deepseek-ai) (DeepSeek open-weight reference inference and encoding implementations)
- **Upstream Copyright:** Copyright (c) DeepSeek-AI
- **Upstream License:** MIT License (`MIT`, [`LICENSES/MIT.txt`](LICENSES/MIT.txt))
- **Usage:**
  - DeepSeek-V4.1 chat/DSML prompt encoding, parsing, and grammar ports (`crates/spark-server/src/dsv41/`).
  - DeepSeek-V4.1 Engram n-gram hashing and token mapping (`crates/spark-model/src/layers/deepseek_v41_engram/`).
  - DSpark Markov bigram draft-head scoring logic (`crates/spark-model/src/layers/dflash_head/markov.rs`).
  - Offline DeepSeek-V4 / DSpark reference probe (`bench/deepseek-v4/dspark_probe/model.py`).

### 2.10 `llm-inference-bench` & Bundled Evaluation Datasets
- **`llm-inference-bench`:**
  - **Upstream Project:** [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench)
  - **Upstream Copyright:** Copyright (c) Local Inference Lab contributors
  - **Upstream License:** MIT License (`MIT`, [`LICENSES/MIT.txt`](LICENSES/MIT.txt); see also `repro-kit/llm-inference-bench/LICENSE`)
  - **Local Modifications:** Vendored under `repro-kit/llm-inference-bench/` with precompiled x86-64 ELF helper binaries (`llm_amd_fabric`, `llm_p2pmark`) and the embedded base64/zlib binary payload removed; helper tools are built from their included C++/CUDA sources via `Makefile`.
- **GSM8K Evaluation Dataset:**
  - **Upstream Project:** [`openai/grade-school-math`](https://github.com/openai/grade-school-math)
  - **Upstream Copyright:** Copyright (c) 2021 OpenAI
  - **Upstream License:** MIT License (`MIT`, [`LICENSES/MIT.txt`](LICENSES/MIT.txt))
  - **Location:** `repro-kit/llm-inference-bench/data/gsm8k_test.jsonl`
- **MMLU-Pro Evaluation Dataset Subset:**
  - **Upstream Project:** [`TIGER-AI-Lab/MMLU-Pro`](https://github.com/TIGER-AI-Lab/MMLU-Pro)
  - **Upstream Copyright:** Copyright (c) TIGER-Lab
  - **Upstream License:** Apache License, Version 2.0 (`Apache-2.0`, [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt))
  - **Location:** `repro-kit/llm-inference-bench/data/mmlu_pro_1000.jsonl`

### 2.11 External Model Chat Templates (`jinja-templates/`) & Built-In Tool Formats
The override templates under [`jinja-templates/`](jinja-templates/) and built-in chat/tool-call format handlers implement the upstream chat-template specifications published with their respective model families:

| Model Family | Upstream Organization | Files in Repository | Upstream License / Model Terms |
|---|---|---|---|
| **Qwen 3.5 / 3.8** | Alibaba Cloud Qwen Team ([`Qwen`](https://huggingface.co/Qwen)) | `jinja-templates/qwen3_5.jinja`, `jinja-templates/qwen3_5_moe.jinja`, `jinja-templates/openai/qwen3_5_moe.jinja` | `Apache-2.0` ([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)) |
| **GLM-5 / GLM-4.7** | Zhipu AI ([`zai-org`](https://huggingface.co/zai-org)) | `jinja-templates/glm5_next.jinja` | `MIT` ([`LICENSES/MIT.txt`](LICENSES/MIT.txt)) |
| **Mistral** | Mistral AI ([`mistralai`](https://huggingface.co/mistralai)) | `jinja-templates/mistral.jinja` | `Apache-2.0` ([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)) |
| **Gemma 4** | Google ([`google`](https://huggingface.co/google)) | `jinja-templates/gemma4.jinja` | Google Gemma Terms of Use (applies when serving Gemma models) |
| **Nemotron-H** | NVIDIA ([`nvidia`](https://huggingface.co/nvidia)) | `jinja-templates/nemotron_h.jinja` | NVIDIA Open Model License (applies when serving Nemotron models) |
| **DeepSeek** | DeepSeek-AI ([`deepseek-ai`](https://huggingface.co/deepseek-ai)) | Built-in Rust DSML encoder/parser (`crates/spark-server/src/dsv41/`, `crates/xgrammar/src/schema/`) | `MIT` ([`LICENSES/MIT.txt`](LICENSES/MIT.txt)) |
| **MiniMax** | MiniMax ([`MiniMaxAI`](https://huggingface.co/MiniMaxAI)) | Built-in Rust XML tool-call parser/grammar (`crates/spark-server/src/tool_parser/minimax_xml.rs`, `crates/xgrammar/src/schema/`) | Upstream model-card format compatibility |

---

## 3. Rust Workspace Crate Dependencies & Licenses

All 51 direct external Rust crates declared across the workspace (`Cargo.toml` and `crates/*/Cargo.toml`) — as well as all transitive dependencies resolved from `crates.io` in `Cargo.lock` — are distributed under open-source licenses compatible with `AGPL-3.0-only` (enforced in CI via [`deny.toml`](deny.toml)). Exact resolved patch versions for all direct and transitive packages are pinned in `Cargo.lock`.

| Crate Name | Version(s) | SPDX License Identifier | Upstream Repository |
|---|---|---|---|
| `ahash` | `0.8.12` | `MIT OR Apache-2.0` | `tkaitchuck/ahash` |
| `aho-corasick` | `1.1.4` | `Unlicense OR MIT` | `BurntSushi/aho-corasick` |
| `anyhow` | `1.0.105` | `MIT OR Apache-2.0` | `dtolnay/anyhow` |
| `axum` | `0.8.8` | `MIT` | `tokio-rs/axum` |
| `base64` | `0.13.1, 0.22.1` | `MIT OR Apache-2.0` | `marshallpierce/rust-base64` |
| `bitvec` | `1.0.1` | `MIT` | `bitvecto-rs/bitvec` |
| `clap` | `4.5.60` | `MIT OR Apache-2.0` | `clap-rs/clap` |
| `criterion` | `0.5.1` (dev) | `Apache-2.0 OR MIT` | `bheisler/criterion.rs` |
| `crossterm` | `0.28.1` | `MIT` | `crossterm-rs/crossterm` |
| `cudarc` | `0.19.2` | `MIT OR Apache-2.0` | `chelsea0x3b/cudarc` |
| `dashmap` | `6.2.1` | `MIT` | `xacrimon/dashmap` |
| `dispatch2` | `0.3.1` | `Zlib OR Apache-2.0 OR MIT` | `madsmtm/objc2` |
| `futures` | `0.3.32` | `MIT OR Apache-2.0` | `rust-lang/futures-rs` |
| `half` | `2.7.1` | `MIT OR Apache-2.0` | `VoidStarKat/half-rs` |
| `http-body` | `1.0.1` | `MIT` | `hyperium/http-body` |
| `image` | `0.25.9` | `MIT OR Apache-2.0` | `image-rs/image` |
| `io-uring` | `0.7.12` | `MIT OR Apache-2.0` | `tokio-rs/io-uring` |
| `lazy_static` | `1.5.0` | `MIT OR Apache-2.0` | `rust-lang-nursery/lazy-static.rs` |
| `libc` | `0.2.191` | `MIT OR Apache-2.0` | `rust-lang/libc` |
| `libloading` | `0.8.9, 0.9.0` | `ISC` | `nagisa/rust_libloading` |
| `log` | `0.4.29` | `MIT OR Apache-2.0` | `rust-lang/log` |
| `memmap2` | `0.9.12` | `MIT OR Apache-2.0` | `RazrFalcon/memmap2-rs` |
| `mimalloc` | `0.1.52` | `MIT` | `purpleprotocol/mimalloc_rust` (`microsoft/mimalloc`) |
| `minijinja` | `2.21.0` | `Apache-2.0` | `mitsuhiko/minijinja` |
| `objc2` | `0.6.4` | `MIT` | `madsmtm/objc2` |
| `objc2-foundation` | `0.3.2` | `MIT` | `madsmtm/objc2` |
| `objc2-metal` | `0.3.2` | `Zlib OR Apache-2.0 OR MIT` | `madsmtm/objc2` |
| `parking_lot` | `0.12.5` | `MIT OR Apache-2.0` | `Amanieu/parking_lot` |
| `prometheus` | `0.13.4` | `Apache-2.0` | `tikv/rust-prometheus` |
| `rand` | `0.8.6, 0.9.5` | `MIT OR Apache-2.0` | `rust-random/rand` |
| `rand_chacha` | `0.3.1, 0.9.0` | `MIT OR Apache-2.0` | `rust-random/rand` |
| `rand_distr` | `0.4.3` | `MIT OR Apache-2.0` | `rust-random/rand` |
| `ratatui` | `0.29.0` | `MIT` | `ratatui/ratatui` |
| `rayon` | `1.12.0` | `MIT OR Apache-2.0` | `rayon-rs/rayon` |
| `ring` | `0.17.14` | `Apache-2.0 AND ISC` | `briansmith/ring` |
| `safetensors` | `0.7.0` | `Apache-2.0` | `huggingface/safetensors` |
| `serde` | `1.0.228` | `MIT OR Apache-2.0` | `serde-rs/serde` |
| `serde_json` | `1.0.149` | `MIT OR Apache-2.0` | `serde-rs/json` |
| `tempfile` | `3.26.0` (dev) | `MIT OR Apache-2.0` | `Stebalien/tempfile` |
| `thiserror` | `1.0.69, 2.0.18` | `MIT OR Apache-2.0` | `dtolnay/thiserror` |
| `tokenizers` | `0.23.1` | `Apache-2.0` | `huggingface/tokenizers` |
| `tokio` | `1.54.1` | `MIT` | `tokio-rs/tokio` |
| `tokio-stream` | `0.1.18` | `MIT` | `tokio-rs/tokio` |
| `toml` | `0.8.23` | `MIT OR Apache-2.0` | `toml-rs/toml` |
| `tower-http` | `0.6.8` | `MIT` | `tower-rs/tower-http` |
| `tracing` | `0.1.44` | `MIT` | `tokio-rs/tracing` |
| `tracing-subscriber` | `0.3.22` | `MIT` | `tokio-rs/tracing` |
| `tui-textarea` | `0.7.0` | `MIT` | `rhysd/tui-textarea` |
| `unicode-width` | `0.1.14, 0.2.0` | `MIT OR Apache-2.0` | `unicode-rs/unicode-width` |
| `ureq` | `3.2.0` | `MIT OR Apache-2.0` | `algesten/ureq` |
| `walkdir` | `2.5.0` | `Unlicense OR MIT` | `BurntSushi/walkdir` |

**Note on `ring 0.17.14` and transitive crates:** `ring 0.17.14` is licensed under `Apache-2.0 AND ISC` (verified against `ring-0.17.14/Cargo.toml`, `LICENSE`, `LICENSE-BoringSSL`, and `LICENSE-other-bits`; it carries no legacy 4-clause OpenSSL advertising clause and is fully compatible with `AGPL-3.0-only`). Notable transitive crates in `Cargo.lock` include `webpki-roots` (`CDLA-Permissive-2.0`), `ryu` (`Apache-2.0 OR BSL-1.0`, selectable under `Apache-2.0`), and `r-efi` (`MIT OR Apache-2.0 OR LGPL-2.1-or-later`, selectable under `MIT` or `Apache-2.0`).

---

## 4. Verbatim License Texts

- [`LICENSE`](LICENSE) — GNU Affero General Public License, Version 3 (`AGPL-3.0-only`)
- [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) — Apache License, Version 2.0
- [`LICENSES/MIT.txt`](LICENSES/MIT.txt) — MIT License
- [`LICENSES/BSD-3-Clause.txt`](LICENSES/BSD-3-Clause.txt) — BSD 3-Clause License
- [`LICENSES/ISC.txt`](LICENSES/ISC.txt) — ISC License

---

## 5. License Compliance Verification

Every release of `TheApathy/flux-engine` is checked to verify:
1. `SPDX-License-Identifier` headers are enforced across first-party source files via `.licenserc.yaml`, while preserving upstream permissive copyright and license notices on vendored/derived third-party components.
2. Third-party Cargo dependency licenses, sources, and security advisories are audited via `cargo-deny` (`deny.toml`).
3. [`LICENSE`](LICENSE), [`CREDITS.md`](CREDITS.md), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`SOURCE_AND_LICENSES.md`](SOURCE_AND_LICENSES.md), and [`LICENSES/`](LICENSES/) are included in source releases and container distributions.
4. Zero proprietary model weights, proprietary prompt captures, or unredistributable prebuilt vendor binaries are committed to version control.
