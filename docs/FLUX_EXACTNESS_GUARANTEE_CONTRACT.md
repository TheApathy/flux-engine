<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Flux Engine Exactness Guarantee Contract

This document defines the six non-negotiable engineering exactness guarantees enforced across Flux Engine's Blackwell (`sm_121a`/`sm_121f`/`sm_120a`/`sm_103a`) CUDA kernels, Rust runtime/scheduler control plane, and offline quantization & serving certification harnesses.

Every performance optimization introduced across Flux v1.2, v1.3, and v1.4 MUST preserve these six guarantees simultaneously—fast paths are never permitted to trade silent numerical drift, cross-request leakage, or non-deterministic speculative divergence for throughput.

---

## 1. The Six Joint Exactness Guarantees

### Guarantee 1 — Zero Silent FMA Boundary Drift (`cvt.rn.bf16.f32`)

- **Problem:** CUDA `nvcc`/`ptxas` default contraction (`fp_contract`) can fuse FP32 multiply-adds across nominal BF16 staging boundaries, producing subtle prefill-vs-verify divergence in recurrent layers (GDN) and deep residual stacks.
- **Invariant:**
  - Critical kernels compile with `--fmad=false` in ISA verification and emit **explicit inline PTX `cvt.rn.bf16.f32`** wherever a BF16 intermediate rounding boundary must match prefill or host reference arithmetic bit-for-bit:
    - GDN 1D correction-record verify & commit (`gdn_correction_replay_sm120.cu`: `beta = sigmoid(b)` and `o = S q` prior to FP32 RMSNorm).
    - Scalar Tensor-FP8 quantizers (`tensor_fp8_quant_sm120.cu`) where aligning to BF16 layer boundaries.
    - 48-SM roofline W4A4 decode GEMM deterministic reduction epilogue (`w4a4_decode_roofline_sm120.cu`).
- **Verification:** CPU bit-exact Rust unit tests verifying 0-ULP prefill vs verify vs 1D-replay vs 2D-snapshot parity, plus `tools/flux_exactness_cert.py` cold-vs-warm replay determinism.

### Guarantee 2 — Batch & Verify-Width Invariance (`B=1 == B=4/8`, `W=1 == W=8 == W=16`)

- **Problem:** Cross-CTA `atomicAdd` reductions introduce thread-scheduling non-determinism across runs and concurrency levels; varying `split_k` or ignoring prefix-local penalties across speculative draft columns breaks `W=1 == W=8 == W=16` equivalence.
- **Invariant:**
  - **Zero cross-CTA `atomicAdd` in exact decode/verify reductions:** Multi-split GEMMs (e.g., `w4a4_gemm_m16_sk3_sm120` + `reduce_w4a4_sk3_bf16_sm120`) accumulate into an ordered FP32 workspace and reduce in a fixed deterministic order.
  - **Consistent `split_k` across `M <= 16`:** Both `W=8` and `W=16` fit inside a single `m16` Blackwell MMA tile height using the exact same reduction order as `M=1` (`B=1`, `W=1`).
  - **Deterministic counter-based speculative acceptance (`exact_spec_accept_sm120.cu`):**
    - Bit-exact `Philox4_32_10` counter RNG keyed strictly by `(seed, abs_pos, 0, 0)`.
    - Exact IEEE `__fdiv_rn(logit, temp)` temperature scaling.
    - Prefix-aware presence and frequency penalty adjustments accumulated strictly over `0..col-1`, guaranteeing bit-identical acceptance/sampling outcomes across `W=1`, `W=8`, and `W=16`.
- **Verification:** Live multi-concurrency `B=1` vs `B=4/8` and multi-width `W=1` vs `W=8` vs `W=16` certification in `tools/flux_exactness_cert.py` (`trajectory_exact_rate == 1.0`, `first_divergence == null`, `top1_flip_rate == 0.0`, `D_KL < 1e-4`).

### Guarantee 3 — FP4 / FP8 Numerical Hygiene

- **E2M1 FP4 (`-0.0` Canonicalization & RaZeR-KV SWAR):**
  - Hardware `cvt.rn.satfinite.e2m1x2.f32` can emit negative zero (`0x8`). Every Flux E2M1 producer (`rms_norm_nvfp4_sm120.cu`, `prefill_paged_bitfp4_sm120.cu`, `paged_decode_attn_bitfp4_sm120.cu`) runs the branchless zero-carry SWAR canonicalizer:
    ```text
    neg0 = r & ~((r & 0x7777_7777) + 0x7777_7777) & 0x8888_8888
    r    = r ^ neg0
    ```
    mapping every `0x8` nibble to `+0.0` (`0x0`) without perturbing any of the 15 non-zero E2M1 encodings.
  - Under `ATLAS_BITFP4_RAZER_KV=1`, code `0x8` is repurposed as `±5.0` (sign stored in UE4M3 scale bit 7 `neg_sv`) and reconstructed in 6 branchless SWAR ops (`razer_swar_split_u32`: `x_main + x_comp`), achieving strictly lower bias-free MSE than standard NVFP4 on both WHT-256 rotated Keys and channel-transposed `V^T` block-16 slabs.
- **FP8 E4M3fn Saturation & Address Safety:**
  - Strict `[-448.0, 448.0]` clamping guarantees **zero** `0x7F` / `0xFF` FP8 NaN bytes on normal finite inputs.
  - All row byte offsets use 64-bit signed/unsigned arithmetic (`int64_t` / `checked_mul`) to eliminate 32-bit index overflow at `128K–1M` contexts.
  - Explicit non-aliasing pointer guards prevent silent in-place overwrite corruption.
  - `TensorFp8EpilogueDeferral` defers partition scales to the FP32 GEMM epilogue **only** when scales lie inside `[0.125, 8.0]` and satisfy `per_block_k_uniform`.
- **LM-Head Precision Tiering (`BF16` vs `DualPlaneNvfp4 8.5 bpw`):**
  - `DualPlaneNvfp4` (`W_hat = (UE4M3(B_scale) * scale2) * (E2M1(W_hi) + 0.125 * E2M1(W_lo))`) achieves `~38.5 dB` SQNR (`+6.7 dB` over block-128 FP8 E4M3 `~31.8 dB`, `+18 dB` over single-plane NVFP4 `~20.5 dB`), `Top-1 flip < 0.1%`, `D_KL < 1e-4 @ T=1.0`, and exact lowest-vocab-index tie-breaking with `NaN` masking (`lm_head_nvfp4_argmax_sm120.cu`).
  - Because `DualPlaneNvfp4`'s information-theoretic Gaussian ceiling is `~38.5 dB`, `tools/certify_quant_exactness.py` explicitly distinguishes the hardware-calibrated `8.5 bpw` tier (`>= 37.5 dB`) from the strict `>= 45.0 dB` BF16-equivalence tier (met by BF16 `lm_head` at `~55.6 dB`, which remains the default unless `ATLAS_DUAL_PLANE_NVFP4=1` is explicitly opted in).

### Guarantee 4 — Recurrent State & Slot Isolation

- **Tail-Row Zeroing:** Every verify/commit step explicitly zeros inactive rows `[live_tokens, max_tokens)` (`gdn_zero_tail_rows_sm120`) so stale activations never bleed across variable-width batches.
- **Non-Mutating Verify + Atomic Cross-Layer Commit:** GDN 1D correction-record verify (`gdn_verify_record_1d_sm120`) emits `(delta[t], k[t], decay[t])` without modifying `h_state`; only accepted prefixes `0..d_num_accepted-1` are replayed in-place by `gdn_commit_replay_1d_sm120` with zero host sync.
- **Duplicate-Slot & Generation Guards:** Both GDN commit and exact GPU speculative acceptance validate `Generation` and `sharing <= 1` fail-closed descriptors before mutating device state.
- **Cross-Slot Canary Isolation:** Mixed-length concurrent batches (`B=4`, `B=8`) MUST exhibit **zero** cross-slot canary leakage across all `(slot_i, slot_j)` pairs (`i != j`).

### Guarantee 5 — Default OFF & Fail-Closed Control Plane

- **Bit-Identical Default Baseline:** Every opt-in optimization (`ATLAS_GDN_1D_REPLAY`, `ATLAS_TENSOR_FP8_PREFILL`, `ATLAS_GDN_L2_WINDOW_MIB`, `ATLAS_GB10_FAST_PREFILL`, `ATLAS_SPEC_EXACT_GPU_ACCEPT`, `ATLAS_W4A4_DECODE_SK3`, `ATLAS_BITFP4_KV`, `ATLAS_BITFP4_RAZER_KV`, `ATLAS_FUSED_RMSNORM_NVFP4`, `ATLAS_LM_HEAD_FUSED_ARGMAX`, `ATLAS_DUAL_PLANE_NVFP4`, `ATLAS_DCOCO_SIDECAR*`, `ATLAS_DFLASH_DEFER_EMIT`, `ATLAS_DFLASH_OPT_BRANCH0`, etc.) defaults **OFF** when unset or empty, executing zero extra allocations and preserving the baseline code path.
- **Strict Environment Parsing:**
  - Every new control reads via `std::env::var_os` (never silently dropping `VarError::NotUnicode`) and fails closed on invalid tokens or partial multi-variable configurations.
  - Every env var is registered in `repro-kit/tools/env_inventory_overrides.tsv`, synced into `crates/spark-server/src/recipe_contract/env_catalog.tsv`, and (for `ATLAS_DFLASH_*`) classified in `cli_dflash_environment_rules.rs` + `cli_dflash_environment_census_tests.rs`.
- **Probability Normalization & Sidecar Authentication:**
  - `exact_spec_accept_sparse_sm120` checks `|sum(p) - 1.0f| <= 1e-4f` on both target `q` and sparse draft support `p_sparse`, writing an explicit device status bitmask on violation, and guarantees a valid target-support fallback (never `-1`/OOB) when residual mass `sum_x max(q(x) - p_sparse(x), 0) == 0`.
  - Binary sidecars (`ATLDC001`, `AutoEMatic`, `ATLGTS01`) enforce Linux `O_NOFOLLOW`, canonical absolute paths, `nlink == 1`, pre/open/post `FileIdentity` anti-TOCTOU checks, whole-file + embedded SHA-256 verification, weight finiteness scans rejecting `NaN`/`±Inf`, and transactional `DeviceAllocations` rollback on partial GPU upload failure.

### Guarantee 6 — Binary-Level PTX/SASS ISA Contracts & Clean-Room Boundaries

- **Compiled-ISA Verification (`repro-kit/formal/verify_ptx_isa_contracts.py`):**
  - Mechanically proves directly on `sm_121a` PTX + disassembled SASS:
    1. **Zero stack frame & zero register spills** (`0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`, zero `STL`/`LDL`).
    2. **Zero illegal odd 64-bit uniform descriptors (`desc[UR<odd>]`)** across every SASS instruction (mitigating CUDA 13.0 `ptxas` `LDGSTS` register-allocator hazard on GB10).
    3. **Native block-scaled MMA enforcement** (`kind::mxf8f6f4` / `kind::mxf4nvf4` `OMMA.SF.16864` with zero unintended half-rate plain FP8 MMAs).
    4. **Positive L2 discard emission** (`discard.global.L2` / `CCTL.E.RML2`) where contracted.
  - Multi-architecture compilation across `sm_121a`, `sm_121f`, `sm_120a`, and `sm_103a` (with clean non-trapping or guarded fallback on `sm_103a`) must produce zero spills and zero odd `desc[UR]`.
- **Clean-Room & Repository Hygiene:**
  - 100% clean-room `SPDX-License-Identifier: AGPL-3.0-only` source files.
  - Zero production `.unwrap()` calls in Rust libraries/binaries; `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` must exit 0 with 0 warnings.
  - `python3 scripts/verify_no_leaks.py` must report **0 leaks** across the entire repository (zero local usernames, home paths, hostnames, tokens, or credential files).
  - Governance files (`KERNEL.toml`, `AGENTS.md`) remain immutable outside dedicated governance commits.

---

## 2. Quantitative Certification Matrix

| Certification Check | Metric | Pass Gate | Owner Tool |
|---|---|---|---|
| Exact arithmetic | Exact integer/mod output | `100%` exact (`4/4`) | `tools/flux_exactness_cert.py` |
| Cold needle passkey (`4K/8K/16K/32K`) | Verbatim recall incl. `FLUX-V12-KV-14336-9472` | `100%` exact | `tools/flux_exactness_cert.py` |
| Verbatim `code_rust` | Canonical SHA-256 of emitted Rust function | Exact SHA-256 match | `tools/flux_exactness_cert.py` |
| Cold-vs-warm prefix/GDN replay | Token trajectory & max logprob diff | `100%` exact, `<= 1e-5` logprob diff, `D_KL < 1e-4` | `tools/flux_exactness_cert.py` |
| Mixed batch invariance (`B=1` vs `B=4/8`) | Target trajectory & cross-slot canary check | `100%` exact, `0` canary leaks | `tools/flux_exactness_cert.py` |
| Verify-width invariance (`W=1` vs `W=8` vs `W=16`) | Trajectory exactness, first divergence, `D_KL` | `100%` exact, `first_div = null`, `top1_flip = 0`, `D_KL < 1e-4` | `tools/flux_exactness_cert.py` |
| `DualPlaneNvfp4` (`8.5 bpw`) `lm_head` | Weight/logit SQNR, Top-1 flip, `D_KL@T=1`, tie-break | `SQNR >= 37.5 dB`, `+6 dB` vs FP8, `flip < 0.1%`, `D_KL < 1e-4`, exact tie-break | `tools/certify_quant_exactness.py` |
| Strict `BF16` `lm_head` equivalence | Logit SQNR vs FP32 | `SQNR >= 45.0 dB` (`~55.6 dB` observed) | `tools/certify_quant_exactness.py` |
| `BitFP4 + 6-op SWAR RaZeR-KV` | `-0.0` SWAR, 6-op split, MSE reduction, attn cosine | `0` SWAR errors, `MSE_razer < MSE_nvfp4`, `logit_cos >= 0.995`, `out_cos >= 0.990` | `tools/certify_quant_exactness.py` |
| PTX/SASS ISA formal contracts | Stack/spill bytes, odd `desc[UR]`, MMA/L2 ISA | `0 B` stack/spills, `0` odd `desc[UR]`, `100% PROVED` | `repro-kit/formal/verify_ptx_isa_contracts.py` |
| Env catalog & leak hygiene | Uncataloged env vars / Tier-0 leaks | `0` uncataloged, `0` leaks | `env_inventory.py --check`, `verify_no_leaks.py` |

---

## 3. Certification Commands

### Offline Self-Tests (CPU-Only, Safe with `CUDA_VISIBLE_DEVICES=""`)

```bash
# 1. OpenAI-API exactness harness self-test (in-process mock server + negative gates)
CUDA_VISIBLE_DEVICES="" python3 tools/flux_exactness_cert.py --self-test

# 2. Offline quantization exactness certifier (DualPlaneNvfp4 + BitFP4/RaZeR-KV)
CUDA_VISIBLE_DEVICES="" python3 tools/certify_quant_exactness.py --self-test

# 3. Repository hygiene & env catalog gates
python3 repro-kit/tools/env_inventory.py --check . /tmp/env_check_exactness && rm -rf /tmp/env_check_exactness
python3 scripts/verify_no_leaks.py
```

### Live Server Certification & Cross-Config Comparison

```bash
# Capture baseline reference (e.g., W=1 / default flags)
python3 tools/flux_exactness_cert.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen3.8-27B \
  --needle-lengths 4096,8192,16384,32768 \
  --batch-sizes 4,8 \
  --json-out /tmp/cert_baseline_w1.json

# Capture candidate run (e.g., W=8 or W=16 with v1.4 opt-in flags enabled)
python3 tools/flux_exactness_cert.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen3.8-27B \
  --needle-lengths 4096,8192,16384,32768 \
  --batch-sizes 4,8 \
  --json-out /tmp/cert_candidate_w16.json

# Certify exactness diff (trajectory exactness, first divergence, top-1 flip, D_KL < 1e-4)
python3 tools/flux_exactness_cert.py \
  --compare /tmp/cert_baseline_w1.json /tmp/cert_candidate_w16.json \
  --max-kl 1e-4 --max-top1-flip-rate 0.0

# Or certify W=1 == W=8 == W=16 directly across live servers
python3 tools/flux_exactness_cert.py \
  --width-urls w1=http://127.0.0.1:8081,w8=http://127.0.0.1:8088,w16=http://127.0.0.1:8096 \
  --model Qwen3.8-27B
```

### Offline Checkpoint Weight Certification & PTX/SASS Verification

```bash
# Certify an exported lm_head weight matrix (.npy or .npz) under DualPlaneNvfp4 8.5 bpw
CUDA_VISIBLE_DEVICES="" python3 tools/certify_quant_exactness.py \
  --weights /path/to/lm_head_weight.npy --min-sqnr-db 37.5 --max-kl 1e-4

# Enforce strict >= 45 dB BF16-equivalence gate (fails closed on 8.5 bpw DualPlaneNvfp4)
CUDA_VISIBLE_DEVICES="" python3 tools/certify_quant_exactness.py \
  --weights /path/to/lm_head_weight.npy --min-sqnr-db 45.0 --max-kl 1e-4

# Mechanical PTX + SASS contract verification on sm_121a (serialized under build lock)
flock /tmp/atlas_v14_build.lock python3 repro-kit/formal/verify_ptx_isa_contracts.py
```
