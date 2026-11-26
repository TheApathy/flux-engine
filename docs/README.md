<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Documentation Index (<code>docs/</code>)</b></p>

# Documentation Index (`docs/`)

Architecture guides, operator runbooks, kernel profiles, and per-model performance recipes for **Flux Engine** on NVIDIA Blackwell (`GB10` / `sm_121a`, `SM120` / `sm_120a`) and Apple Metal.

---

## 1. Operator Guides, Verification & Quick Reference Cards

| Document | Purpose |
|---|---|
| [`verification/`](verification/README.md) | **Benchmark Methodology & Verification**: [`BENCHMARK_VERIFICATION.md`](verification/BENCHMARK_VERIFICATION.md) — roofline physics, prefix-cache controls, speculative-decode token accounting, and quality-gate proofs. |
| [`API.md`](API.md) | Verified HTTP API & compatibility matrix (`OpenAI` `/v1/chat/completions` & `/v1/responses`, `Anthropic` `/v1/messages`, `Ollama` discovery, auth, CORS, `xgrammar`, telemetry). |
| [`CLAUDE_CODE.md`](CLAUDE_CODE.md) | Connecting Claude Code, OpenCode, Cursor, Continue, Cline, and OpenAI/Anthropic SDKs directly to Flux Engine. |
| [`OPERATOR-GUIDE.md`](OPERATOR-GUIDE.md) | One-command serving (container & host), checkpoint layout, build commands (`-p spark-server`), OpenAI query examples, `<think>` controls, and GB10 UMA rules. |
| [`CONTAINER_GUIDE.md`](CONTAINER_GUIDE.md) | Production container guide (`ghcr.io/theapathy/flux-engine:v1.0.3`), building from source, `flux serve / inspect / list-models`, and AGPL-3.0 provenance. |
| [`../ENGINE-CARD.md`](ENGINE-CARD.md) | Engine identity, supported model architectures, auto-detection table, and headline GB10 benchmarks. |
| [`../MODEL-CARD.md`](MODEL-CARD.md) | `Qwen3.8-Flash-Next-NVFP4-Offload` and `Qwen3.8-27B-NVFP4` layer topology, quantization map, and unified memory footprint. |
| [`KERNEL-PROFILE.md`](KERNEL-PROFILE.md) | Per-stage `sm_121a` CUDA kernel timing (`1,348.0 ms / 8K` prefill and `K=5` / `K=16` speculative verify). |
| [`../FLAGS.md`](FLAGS.md) | Complete reference of `FLUX_*` and `ATLAS_*` environment variables and CLI flags. |

---

## 2. Performance Recipes & Measurement Records

| Document | Purpose |
|---|---|
| [`QWEN38_FLASH_NEXT_PERFORMANCE_RECIPE.md`](QWEN38_FLASH_NEXT_PERFORMANCE_RECIPE.md) | How `Qwen3.8-Flash-Next-NVFP4` reaches `6,082.2 tok/s` prefill at 16K, `5,694.4 tok/s` at 128K, `57.4 tok/s` lossless `K=5` MTP decode, and `5/5` Estonia 136K on one GB10, including levers that measured null or regressed. |
| [`QWEN38_PERFORMANCE_RECIPE.md`](QWEN38_PERFORMANCE_RECIPE.md) | `Qwen3.8-27B-NVFP4` DFlash (`gamma=15`) measurement record (`63.9–64.0 tok/s`), roofline bandwidth analysis, and refutation ledger. |
| [`QWEN38_UNIFIED_PROFILES.md`](QWEN38_UNIFIED_PROFILES.md) | Built-in launch profile contracts and dual-target build profiles. |
| [`CUSTOM-FLUX-PLAYBOOK.md`](CUSTOM-FLUX-PLAYBOOK.md) | Playbook of custom `sm_121a` optimizations, memory guards, and speculative verification fixes. |
| [`APATHY_TUI_BENCHMARKS.md`](APATHY_TUI_BENCHMARKS.md) | Interactive Ratatui TUI telemetry and benchmark dashboard documentation. |

---

## 3. Architecture, Hardware, Design & Governance

| Document | Purpose |
|---|---|
| [`../ARCHITECTURE.md`](ARCHITECTURE.md) | Deep walkthrough of the 16-crate Rust workspace and `sm_121a` CUDA/PTX kernel subsystems. |
| [`FLUX_KERNELS.md`](FLUX_KERNELS.md) | Build-time `(hardware × model × quant)` kernel discovery, `nvcc` compilation, and zero-spill verification. |
| [`FLUX_SPARK_JOURNEY.md`](FLUX_SPARK_JOURNEY.md) | Architectural evolution from baseline W4A16 to `6,082.2 tok/s` prefill and lossless `K=5` MTP on DGX Spark. |
| [`SM120_SM103_BRINGUP.md`](SM120_SM103_BRINGUP.md) & [`METAL_BACKEND.md`](METAL_BACKEND.md) | Discrete Blackwell (`sm_120a` / `sm_103a`) bring-up guide and experimental Apple Metal backend reference. |
| [`HARDWARE.md`](HARDWARE.md) & [`DEPLOYMENT.md`](DEPLOYMENT.md) | Adding hardware targets/model families, unified memory sizing, and single/multi-node deployment. |
| [`turboquant-plus.md`](turboquant-plus.md) & [`weight-cache.md`](weight-cache.md) | KV-cache quantization (`TurboQuant+`) and persistent post-transform weight caching. |
| [`DEBUGGING_METHODOLOGY.md`](DEBUGGING_METHODOLOGY.md) & [`GOVERNANCE.md`](GOVERNANCE.md) | Kernel numerical debugging methodology and verification gates. |
| [`adr/`](adr/README.md) & [`design/`](design/) | Architecture Decision Records and speculative-verification design notes. |
| [`CITATIONS.md`](CITATIONS.md) & [`compatibility/`](compatibility/README.md) | Academic citations, prior-art bibliography, and hardware/model compatibility matrices. |
| [`reproduction/`](reproduction/README.md) & [`env-inventory/`](env-inventory/env_inventory_summary.md) | Reproducible benchmark guides and environment variable inventory. |
| [`../SIGNING.md`](../SIGNING.md) · [`../CLAUDE.md`](../CLAUDE.md) · [`../llms.txt`](../llms.txt) | Release signing & SLSA L3 verification, public contributor/agent guide, and LLM documentation index. |
