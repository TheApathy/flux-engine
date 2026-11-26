# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| latest `main` | ✅ |
| older commits | ❌ |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Instead, please report vulnerabilities privately via [GitHub Private Vulnerability Reporting](https://github.com/TheApathy/flux-engine/security/advisories/new) with:

1. **Description**: What the vulnerability is and its potential impact
2. **Reproduction steps**: Minimal steps to reproduce the issue
3. **Environment**: OS, CUDA version, GPU model, Rust version
4. **Affected component**: Which crate or kernel is affected

We will acknowledge receipt within **48 hours** and provide an initial assessment within **7 days**.

## Scope

Flux Engine is an inference server that runs locally with GPU access. The primary threat surface includes:

- **CUDA kernel safety**: Out-of-bounds memory access and buffer overflows in GPU kernels
- **HTTP API**: Input validation on the OpenAI- and Anthropic-compatible endpoints (`spark-server`)
- **Weight loading**: Malicious safetensors/GGUF files and path traversal during model loading
- **Unsafe Rust**: Flux Engine uses `unsafe` blocks for CUDA/NCCL/io_uring FFI, making those modules high-priority review targets

## Automated Auditing & Release Verification

Flux Engine runs automated security and supply-chain checks in CI:

- **`cargo deny`**: Audits dependencies for known advisories, license compliance, and banned crates (weekly + on PRs touching `Cargo.toml`, `Cargo.lock`, or `deny.toml`)
- **`cargo clippy`**: Static analysis across the Rust workspace and tests
- **Signed Releases & Provenance**: Tagged releases require signed Git tags verified against [`.github/allowed_signers`](allowed_signers) and ship SLSA Build Level 3 provenance, Sigstore Cosign keyless bundles, GitHub Artifact Attestations, and deterministic SPDX 2.3 SBOMs. See [`SIGNING.md`](../SIGNING.md) for verification instructions.

## Disclosure Policy

We follow coordinated disclosure. Once a fix is available, we will:

1. Merge the fix to `main`
2. Tag a release
3. Credit the reporter (unless anonymity is requested)
