<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Operator &amp; Build Scripts (<code>scripts/</code>)</b></p>

# Utility & Build Scripts (`scripts/`)

Core environment translation, checkpoint preparation, and build verification scripts for **Flux Engine**.

> For model-specific serving launchers and benchmark harnesses, see [`bench/`](../bench/) and [`repro-kit/`](../repro-kit/).

| Script | Purpose |
|---|---|
| [`flux_env_bridge.sh`](flux_env_bridge.sh) | Dual-namespace (`FLUX_*` <-> `ATLAS_*`) environment translator and operating mode loader (`--list-modes` lists all 14 wired profiles). |
| [`repack_qwen38_flash_next_ple_nvfp4.py`](repack_qwen38_flash_next_ple_nvfp4.py) | Streams the Per-Layer Embedding (PLE) table of `Qwen3.8-Flash-Next` into 128 sector-aligned NVFP4 shards (`ple-offload/`) with bounded host RAM. |
| [`build-qwen38-dual-profiled.sh`](build-qwen38-dual-profiled.sh) | Hermetic dual-target release builder compiling `gb10/qwen3.8-27b/nvfp4` and `gb10/qwen3.8-flash-next/nvfp4` with a cryptographic `build-receipt.json`. |
| [`check.sh`](check.sh) | Fast Rust-only workspace check/clippy/test wrapper (`ATLAS_SKIP_BUILD=1`, `CUDARC_CUDA_VERSION=13000`). |
| [`check-kernels-include-cache.sh`](check-kernels-include-cache.sh) | Verifies `atlas-kernels` PTX cache invalidation when quoted CUDA `#include` headers change or vanish. |
| [`check-kernels-shared-target.sh`](check-kernels-shared-target.sh) | Verifies shared `CARGO_TARGET_DIR` isolation across git worktrees via `ATLAS_BUILD_SOURCE_ROOT`. |
| [`check-qwen-bf16-tree-abi.py`](check-qwen-bf16-tree-abi.py) | Static ABI contract verifier for Qwen BF16 attention kernel launch arguments. |
| [`start-ep2.sh`](start-ep2.sh) | Two-node NVIDIA DGX Spark (`2× GB10`) RoCEv2 `EP=2` launch helper. |
| [`verify_no_leaks.py`](verify_no_leaks.py) | Repository security & sanitization scanner verifying zero credentials, local paths, or host identifiers. |
| [`generate_cargo_spdx_sbom.py`](generate_cargo_spdx_sbom.py) | Zero-dependency deterministic SPDX 2.3 JSON SBOM generator reading `Cargo.toml` + `Cargo.lock`. |
| [`verify_release.sh`](verify_release.sh) | One-command release bundle verifier (SHA-256 checksums, SSH signed tag via `.github/allowed_signers`, SLSA L3 provenance, Cosign keyless bundle, and GitHub Artifact Attestations). |
| [`dsv41_parity/`](dsv41_parity/) & [`fixtures/`](fixtures/) | DeepSeek-V4.1 parity fixture generator and minimal AGPL-licensed synthetic prompt/tool fixtures. |
