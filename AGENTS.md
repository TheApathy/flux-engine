<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Flux Engine — Production Engineering & Coding Agent Contract (`AGENTS.md`)

This document is the authoritative operating contract for **all** coding agents
(Claude Code, OpenAI Codex, Cursor, OpenCode, GitHub Copilot, and custom agents)
and human contributors working on **Flux Engine** (`TheApathy/flux-engine`).

> **Private notes rule:** Keep personal or machine-local scratch notes in
> `AGENTS.local.md` or `CLAUDE.local.md` (both strictly gitignored). **Never**
> commit personal paths, private hostnames, LAN IPs, raw logs, or secrets.

---

## 1. Architecture & Crate Boundaries

Flux Engine is a **16-crate Rust workspace** (`Cargo.toml`, Edition `2024`,
MSRV `1.88`, `AGPL-3.0-only`) paired with custom Blackwell CUDA/PTX kernels and
an experimental Apple Silicon Metal backend:

| Layer / Crate | Ownership & Hard Boundaries |
|---|---|
| `crates/spark-server` | Binaries (`flux-server`, `flux-cli`, `spark`), OpenAI/Anthropic/Ollama HTTP & SSE router, continuous-batching scheduler, speculative decode orchestration, tool-call parsers, Ratatui TUI, Web UI. **Must never panic on untrusted HTTP/JSON/SSE inputs.** |
| `crates/spark-model` | Model architectures (`Qwen3.8-Flash-Next`, `Qwen3.8-27B`, `DeepSeek-V4.1`, `GLM-5.3-Flash`, `Gemma-4`, `Qwen3.6`/`Qwen3.5`, `Mistral-Small-4`, `MiniMax-M2.7`, `Nemotron-3`), weight loaders, chunked prefill, speculative verification. |
| `crates/spark-runtime` | `GpuBackend` trait (`AtlasCudaBackend`, `MetalBackend`, `MockGpuBackend`), CUDA/Metal streams & CUDA graphs, cuBLASLt wrappers, UMA headroom guards, paged KV cache, radix prefix cache, TurboQuant+ KV dequant, sampler. |
| `crates/atlas-kernels` | Build-time (`build.rs`) `(hardware × model × quant)` discovery, parallel `nvcc`/`xcrun` compilation, embedded PTX/CUBIN/metallib tables, `HARDWARE.toml`/`MODEL.toml` metadata. |
| `crates/atlas-core` | Hardware capability detection, tensor/dtype descriptors, stream/kernel descriptors, config parsers. |
| `crates/spark-comm` | Multi-GPU / multi-node NCCL (`RoCEv2` / NVLink) & TCP `TP`/`EP` collectives. |
| `crates/spark-storage` | `io_uring` / `pread64` / `mmap` NVMe KV-cache swap and sector-aligned PLE DMA. |
| `crates/cufile-sys` | GPUDirect Storage (`cuFile`) FFI bindings. |
| `crates/rest-store` | Retrieval-based speculative drafting store (`--dflash-lookup retrieval`). |
| `crates/xgrammar` | Pure-Rust pushdown-automaton JSON Schema / EBNF structured-output grammar engine (Apache-2.0 port; preserve `crates/xgrammar/NOTICE`). |
| `crates/atlas-spark-bench` | Standalone microbenchmark harness for GEMM, GDN, QSA, and MoE kernels. |

*(The remaining 5 crates — `atlas-quant`, `atlas-norm`, `atlas-activation`,
`atlas-embed`, `atlas-reduce` — are reserved trait-interface stubs.)*

### Hardware & Mirroring Contracts
- **Supported CUDA targets:** `gb10` (`sm_121a`/`sm_121f`), `rtx6000` (`sm_120a`), `gb300` (`sm_103a`); **Experimental:** `metal` (`aarch64` macOS).
- **Mirrored directories:** Keep `recipes/` and `repro-kit/recipes/` byte-identical and honor SHA-256 pins in `repro-kit/ENGINE_STABLE_PIN.json`.
- **No-GPU compilation invariant:** Every Rust change must continue compiling and passing tests with `ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000`.

---

## 2. Formal Verification & Proof-First Engineering

Flux Engine is a production inference engine — **unit tests alone are not enough**
for scheduling protocols, bit-packing, memory arenas, RNG coupling, or hot-path
CUDA kernels. Contributions touching these subsystems **must** extend and pass
the corresponding formal verifier before a PR will be considered:

| Subsystem Changed | Mandatory Formal Gate | Proof Standard |
|---|---|---|
| Bit-packing, SplitMix64 RNG / Gumbel coupling, Leviathan maximal coupling, FP8/NVFP4 block-scale math, SMEM layout, carveout arenas, QSA/prefix-cache safety, ragged PLE descriptors | `python3 -B repro-kit/formal/verify_nextgen_invariants.py` | **Z3 SMT (`UNSAT`)** — every invariant must be mathematically proved across the full domain (currently 19 UNSAT theorems). |
| Continuous batching, speculative decode coupling, multi-rank prefix agreement, slot lifecycle, stream drain, fence runway, Q-row splicing | `python3 -B repro-kit/formal/verify_scheduler_tla_model.py` | **TLA+ Explicit-State BFS** — 0 deadlocks, 0 invariant violations across full reachable state space (currently 99,596 states / 431,088 transitions / 8 invariants). |
| Production CUDA kernels (`.cu` / `.cuh` under `kernels/`) | `python3 -B repro-kit/formal/verify_ptx_isa_contracts.py` | **`nvcc -Xptxas -v -fmad=false`** on `sm_121a` & `sm_120a`: **`0 bytes stack frame, 0 spill stores, 0 spill loads`** plus exact MMA/TMA PTX instruction contracts. |

> **Rule:** If you change an invariant, **update the formal model first**, prove
> it passes (UNSAT / 0 deadlocks / 0 spills), and include the proof summary in
> your PR description. Never weaken or delete a formal check to make a patch pass.

---

## 3. Mandatory Pre-PR Verification Checklist

Enable the repository's opt-in pre-push hook once per clone:

```bash
git config core.hooksPath .githooks
```

Before opening **any** Pull Request, run and verify every applicable gate:

```bash
# 1. Fast hygiene gate (PII/leak check, Markdown links, .gitignore guard, syntax, rustfmt):
./scripts/ci_local.sh --fast

# 2. Rust lint & CPU test suite (no GPU required):
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --all-targets -- -D warnings
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test --workspace --tests

# 3. Formal verification (Z3 SMT + TLA+ BFS scheduler model):
python3 -B repro-kit/formal/verify_nextgen_invariants.py
python3 -B repro-kit/formal/verify_scheduler_tla_model.py

# 4. Offline Python & contract test suite:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest -q repro-kit/tests tests

# 5. If CUDA kernels (.cu/.cuh) were touched (requires nvcc, no physical GPU):
python3 -B repro-kit/formal/verify_ptx_isa_contracts.py
```

---

## 4. Mandatory 4-Part PR Description (No Exceptions)

Every PR description must give a reviewer full context without forcing them to
reverse-engineer the diff. Answer these **four questions in order**:

1. **Why:** The concrete defect, performance bottleneck, safety hazard, or missing capability and its root cause.
2. **What changed:** Exact crates/files/symbols modified and the before/after behavior or invariant.
3. **Claimed result / benefit:** Strictly bounded to what was measured or proved (state *designed* vs *implemented & qualified*).
4. **Verification evidence:** Exact commands run, hardware/software context, and observed outputs (including Z3 / TLA+ / PTX zero-spill results where applicable).

---

## 5. Anti-Slop & Hard Rejection Criteria

Flux Engine welcomes focused, rigorously verified human and AI-assisted
contributions — and **rejects low-effort or unverified AI slop**. PRs exhibiting
any of the following will be closed or returned without merge:

- **Unverified AI output:** Disclose AI assistance honestly in the PR body. You must personally read, compile, test, and own every line you submit. Never claim "tests pass" without running them.
- **Drive-by churn:** Cosmetic reformatting, mass comment rewrites, renaming identifiers without functional need, or speculative "cleanup/refactor" PRs touching unrelated files.
- **Unsafe or panicking Rust:** Any `unsafe` block/fn or FFI call without a truthful, verifiable `// SAFETY:` invariant; any `.unwrap()`/`.expect()` reachable from untrusted HTTP/JSON/SSE inputs in `spark-server`; unchecked integer arithmetic on KV/tensor byte sizes.
- **Silent numerical or safety regressions:** Silently falling back across quantization/precision contracts, loosening UMA/OOM admission guards, or changing safe loopback defaults (`127.0.0.1`, explicit `--api-key` / `--auth-tokens-file` for non-loopback exposure).
- **Hallucinated claims or peer disparagement:** Invented CLI flags (verify against `crates/spark-server/src/cli.rs`), invented HTTP routes (verify against `crates/spark-server/src/main_modules/serve_router.rs`), fabricated benchmark numbers, marketing superlatives, or disrespectful remarks toward upstream projects or peer engines.
- **Licensing & binary hygiene violations:** Missing `SPDX-License-Identifier: AGPL-3.0-only` header on first-party source/kernel/script files, stripped MIT/Apache-2.0/BSD third-party notices, or committed binary blobs (`.o`, `.a`, `.so`, `.cubin`, ELFs).
- **Privacy, PII, or secret leaks:** Personal usernames, local home-directory paths, private hostnames/LAN IPs, API keys, debug logs, or scratch journals (`scripts/verify_no_leaks.py` and `gitleaks` must report `0` violations).
- **CI, `.gitignore`, or supply-chain regressions:** Unpinned GitHub Actions tags (every action must be pinned to a 40-char commit SHA except `slsa-framework/slsa-github-generator@v2.1.0`), `pull_request_target`, missing `persist-credentials: false`, weakening `step-security/harden-runner`, or `.gitignore` rules that shadow tracked source files.
- **Monolithic file bloat:** Keep new Rust source modules concise and modular (≤500 LoC soft guideline enforced as an advisory by `.github/workflows/file-size-cap.yml`).

---

## 6. Security, Signing & Governance

- **Branch & tag integrity:** `main` and `v*` release tags enforce verified SSH commit/tag signatures against [`.github/allowed_signers`](.github/allowed_signers) (see [`SIGNING.md`](SIGNING.md)).
- **Vulnerability reporting:** Never open public issues for security vulnerabilities; follow [`.github/SECURITY.md`](.github/SECURITY.md) (GitHub Private Vulnerability Reporting).
- **Companion docs:** See [`CLAUDE.md`](CLAUDE.md) for quick-reference build recipes, [`CONTRIBUTING.md`](CONTRIBUTING.md) for contributor workflow, and [`docs/verification/`](docs/verification/) for hardware qualification methodology.

---

## 7. Continuous Self-Improvement Loop

Flux Engine operates on an automated weekly quality ratchet:

- **Automated signals (weekly):** [`.github/workflows/weekly-health.yml`](.github/workflows/weekly-health.yml) (`scripts/self_improvement_audit.py`), OpenSSF Scorecard (`scorecard.yml`), ClusterFuzzLite (`cifuzz.yml`), CodeQL SAST (`codeql.yml`), `cargo-deny` (`security.yml`), and grouped Dependabot updates (`.github/dependabot.yml`).
- **Ratchet rule:** Whenever a bug, fuzz crash, CI flake, or Scorecard deduction is fixed, **encode the invariant** (add a Z3/TLA+ assertion, a fuzz corpus seed, a Python contract test, or a `ci_local.sh` check) so the same class of defect can never regress.
