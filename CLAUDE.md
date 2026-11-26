<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Flux Engine — Contributor & Coding Agent Guide (`CLAUDE.md`)

**Flux Engine** (`TheApathy/flux-engine`) is a pure-Rust + custom CUDA/PTX
large language model inference engine licensed under **`AGPL-3.0-only`**. It
targets NVIDIA Grace Blackwell (`GB10`, `sm_121a` / `sm_121f`), discrete
Blackwell (`SM120` / `sm_120a` and `SM103` / `sm_103a`), and experimental Apple
Silicon (`Metal`).

---

## 1. Repository & Workspace Map

Flux Engine is a **16-crate Rust workspace** (`Cargo.toml`, Edition `2024`,
MSRV `1.88`, version `1.0.3`):

### Active Crates (11)

| Crate | Responsibility |
|---|---|
| `crates/spark-server` | Binaries (`flux-server`, `flux-cli`, `spark`), HTTP router (`OpenAI`, `Anthropic`, `Ollama` discovery, disaggregated routes), continuous-batching scheduler, speculative decode orchestration, tool-call parsers, Ratatui TUI, Web UI. |
| `crates/spark-model` | Model implementations (`Qwen3.8-Flash-Next`, `Qwen3.8-27B`, `DeepSeek-V4.1`, `GLM-5.3-Flash`, `Gemma-4`, `Qwen3.6`/`Qwen3.5`, `Mistral-Small-4`, `MiniMax-M2.7`, `Nemotron-3`), weight loaders, chunked prefill, speculative verify, and optional `csrc/fi_attn` / `csrc/fi_gdn` bridges. |
| `crates/spark-runtime` | `GpuBackend` trait (`AtlasCudaBackend`, `MetalBackend`, `MockGpuBackend`), CUDA/Metal streams & graphs, cuBLASLt wrappers, UMA headroom guards, paged KV cache, radix prefix cache, TurboQuant+ KV dequantization, sampler. |
| `crates/atlas-kernels` | Build-time (`build.rs`) `(hardware × model × quant)` kernel discovery, parallel `nvcc`/Metal compilation, embedded PTX/CUBIN/metallib tables, and `MODEL.toml` metadata. |
| `crates/atlas-core` | Hardware capability detection (`capabilities.rs`), tensor/dtype descriptors, stream/kernel descriptors, and model config parsers. |
| `crates/spark-comm` | Multi-GPU / multi-node `NCCL` (`RoCEv2` / NVLink) and TCP `TP`/`EP` collective backends. |
| `crates/spark-storage` | `io_uring` / `pread64` / `mmap` NVMe KV-cache swap and 4-ring sector-aligned PLE DMA. |
| `crates/cufile-sys` | GPUDirect Storage (`cuFile`) FFI bindings. |
| `crates/rest-store` | Retrieval-based speculative drafting store (`--dflash-lookup retrieval`). |
| `crates/xgrammar` | Safe pure-Rust pushdown-automaton JSON Schema / EBNF structured-output grammar engine (Apache-2.0 port; see `crates/xgrammar/NOTICE`). |
| `crates/atlas-spark-bench` | Standalone microbenchmark harness for isolated GEMM, GDN, QSA, and MoE kernels. |

*(The remaining 5 crates — `atlas-quant`, `atlas-norm`, `atlas-activation`,
`atlas-embed`, `atlas-reduce` — are reserved trait-interface stubs.)*

### Key Top-Level Directories

- `kernels/`: Per-hardware CUDA (`kernels/gb10/`, `kernels/rtx6000/`, `kernels/gb300/`) and Metal (`kernels/metal/common/`) kernel sources + `HARDWARE.toml` / `MODEL.toml` manifests.
- `recipes/` & `repro-kit/recipes/`: Mirrored operating recipes (`.env`, `.toml`, `.argv`). Keep `recipes/` and `repro-kit/recipes/` byte-identical and respect SHA-256 pins in `repro-kit/ENGINE_STABLE_PIN.json`.
- `repro-kit/`: Container entrypoint (`flux-autodetect`), benchmark harnesses (`flux-bench`, `lil-bench`), formal PTX/invariant verifiers (`repro-kit/formal/`), and Python contract tests (`repro-kit/tests/`).
- `docker/`: Production Dockerfiles (`docker/sm120/Dockerfile`, `docker/gb10/Dockerfile`, per-model images).
- `docs/`: Technical guides (`docs/API.md`, `docs/CLAUDE_CODE.md`, `docs/SM120_SM103_BRINGUP.md`, `docs/KERNEL-PROFILE.md`, `docs/compatibility/`, `docs/verification/`).

---

## 2. Build, Lint & Test Commands

### No-GPU CI / Local Static Verification

```bash
# Clippy (all workspace crates + tests, warnings denied):
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
  cargo clippy --workspace --all-targets -- -D warnings

# Rust formatting check (Edition 2024):
cargo fmt --all -- --check

# CPU unit & static-source integration tests:
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
  cargo test --workspace --tests
```

### Real CUDA Builds (Requires CUDA 13.0+ `nvcc` on `PATH`)

```bash
export PATH=/usr/local/cuda/bin:$PATH

# NVIDIA DGX Spark / GB10 (sm_121a / sm_121f):
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release --bin flux-server

# Discrete Blackwell SM120 (RTX PRO 6000 / RTX 5090):
ATLAS_TARGET_HW=rtx6000 ATLAS_TARGET_MODEL="qwen3.8-flash-next,qwen3.8-27b" ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release --bin flux-server
```

### Python, Kernel ISA, Privacy & Hygiene Checks

```bash
# Python contract & harness tests:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest repro-kit/tests tests

# Formal PTX ISA & zero-spill contract check (requires nvcc, no physical GPU):
python3 -B repro-kit/formal/verify_ptx_isa_contracts.py

# Typo & leak checks:
typos --config _typos.toml
python3 -B scripts/verify_no_leaks.py --root .
```

---

## 3. Mandatory Engineering Invariants

1. **License & SPDX Headers:** Every first-party source, kernel, and script file
   must start with `// SPDX-License-Identifier: AGPL-3.0-only` (or `# SPDX-License-Identifier: AGPL-3.0-only`).
   Never strip third-party MIT/Apache-2.0/BSD notices or commit precompiled
   binary blobs (`.o`, `.so`, `.a`, ELFs).
2. **`unsafe` & FFI Soundness:** Every `unsafe` block, `unsafe fn`, and FFI
   boundary requires an explicit, truthful `// SAFETY:` comment explaining every
   checked invariant (pointer validity, alignment, bounds, lifetime, stream
   synchronization, and thread-safety).
3. **Zero Register Spills on Critical CUDA Kernels:** Compile modified `.cu`
   files with `-Xptxas -v -fmad=false` and verify `0 bytes stack frame, 0 spill
   stores, 0 spill loads` on production hot-path kernels. Never introduce
   thread-divergent early returns before `__syncthreads()` or warp shuffles.
4. **Fail-Closed Admission & Secure Defaults:** Default bind address is
   `127.0.0.1` (`--bind 0.0.0.0` requires explicit opt-in and `--api-key` /
   `--auth-tokens-file` for non-loopback exposure). CORS defaults to loopback
   origins only. Never silently fall back across numerical precision contracts
   or bypass OOM / UMA headroom admission guards.
5. **Privacy, Zero PII & Peer-Respectful Tone:** Never commit personal paths
   (`/home/<user>`, `/Users/<user>`), private hostnames, API keys, raw system
   logs, or internal scratch notes. Keep all code comments, docs, and benchmark
   comparisons factual and respectful of upstream projects and peer engines.
6. **Verify Before Claiming:** Never invent cryptographic hashes, benchmark
   numbers, or CLI flags. Cross-check flags against `crates/spark-server/src/cli.rs`
   and routes against `crates/spark-server/src/main_modules/serve_router.rs`.
