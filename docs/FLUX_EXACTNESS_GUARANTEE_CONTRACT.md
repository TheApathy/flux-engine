<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Flux Engine Exactness Guarantee Contract

This document defines the seven non-negotiable engineering exactness guarantees enforced across Flux Engine's Blackwell (`sm_121a`/`sm_121f`/`sm_120a`/`sm_103a`) CUDA kernels, Rust runtime/scheduler control plane, and offline quantization & serving certification harnesses.

Every performance optimization introduced across Flux v1.2, v1.3, v1.4, v1.5, and v1.6 MUST preserve these seven guarantees simultaneously—fast paths are never permitted to trade silent numerical drift, cross-request leakage, or non-deterministic speculative divergence for throughput.

---

## 1. The Seven Joint Exactness Guarantees

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
  - **Cross-`M` Deterministic Batch Dispatch (`ATLAS_DETERMINISTIC_BATCH=1`, default OFF):**
    - Dynamic arrival batch sizes (`M = 1..128`) MUST NOT cross heterogeneous kernel/GEMM dispatch thresholds (`m4`/`m8`/`m17`/`m32`, `batched_verify_tc_dense_active`, cuBLASLt vs custom GEMV, `qwen4_hyper`, `qwen4_fast_proj`, `moe` decode/verify, `qwen3_attention/decode`, `qwen3_ssm/trait_decode_batched`, or `lm_head` batched verify selectors) when strict `B=1 == B=2 == B=4 == B=8 == B=16` invariance is requested.
    - Enabling `ATLAS_DETERMINISTIC_BATCH=1` pins every decode/verify projection across `M = 1..128` to a single `M`-invariant, zero-atomic, fixed-order backend/tile, driving `batch_b1_vs_b8_exact` from `1/4` to `4/4`.
- **Verification:** Live multi-concurrency `B=1` vs `B=4/8` (`4/4` canonical trajectories via `--require-batch-invariance`) and multi-width `W=1` vs `W=8` vs `W=16` certification in `tools/flux_exactness_cert.py` (`trajectory_exact_rate == 1.0`, `first_divergence == null`, `top1_flip_rate == 0.0`, `D_KL < 1e-4`).

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
- **LM-Head Precision Pareto Ladder (`BF16` vs `INT8 per-32 BF16sc` vs `DualPlaneNvfp4` vs `MXFP8`):**
  - **`INT8 per-32 BF16sc` (`8.50 bpw`, `1.88x` smaller than BF16) Pareto-dominates both `DualPlaneNvfp4` and `MXFP8`:** Signed `int8` (`-127..=127`) spends all 7 magnitude bits linearly per 32-element Gaussian block and converts **0-ULP lossless** into BF16 (`7` mantissa + `1` implicit bit), achieving **`~45.4 dB` SQNR**, `mean D_KL < 1e-6 @ T=1.0`, and `0%` top-1 flips—outperforming `DualPlaneNvfp4` (`8.50 bpw`, `~38.5 dB` SQNR; `9.0 bpw`, `~41.3 dB` SQNR) by `+6.5+ dB` and `MXFP8` (`8.25 bpw`, `~31.5 dB` SQNR) by `+13.5+ dB` at identical `8.50 bpw` footprint.
  - **`DualPlaneNvfp4` (`8.50 bpw`):** `W_hat = (UE4M3(B_scale) * scale2) * (E2M1(W_hi) + 0.125 * E2M1(W_lo))` achieves `~38.5 dB` SQNR (`+6.7 dB` over block-128 FP8 E4M3 `~31.8 dB`, `+18 dB` over single-plane NVFP4 `~20.5 dB`), `Top-1 flip < 0.1%`, `D_KL < 1e-4 @ T=1.0`, and exact lowest-vocab-index tie-breaking with `NaN` masking (`lm_head_nvfp4_argmax_sm120.cu`).
  - `tools/certify_quant_exactness.py` certifies all four schemes side-by-side (`BF16` `~55.6 dB`, `INT8 per-32 BF16sc` `~45.4 dB >= 44.0 dB` & `D_KL < 1e-6`, `DualPlaneNvfp4` `~38.5 dB >= 37.5 dB`, `MXFP8` `~31.5 dB`) and enforces strict fail-closed rejection when `DualPlaneNvfp4` is tested against a `>= 45.0 dB` threshold.

### Guarantee 4 — Recurrent State & Slot Isolation

- **Tail-Row Zeroing:** Every verify/commit step explicitly zeros inactive rows `[live_tokens, max_tokens)` (`gdn_zero_tail_rows_sm120`) so stale activations never bleed across variable-width batches.
- **Non-Mutating Verify + Atomic Cross-Layer Commit:** GDN 1D correction-record verify (`gdn_verify_record_1d_sm120`) emits `(delta[t], k[t], decay[t])` without modifying `h_state`; only accepted prefixes `0..d_num_accepted-1` are replayed in-place by `gdn_commit_replay_1d_sm120` with zero host sync.
- **Duplicate-Slot & Generation Guards:** Both GDN commit and exact GPU speculative acceptance validate `Generation` and `sharing <= 1` fail-closed descriptors before mutating device state.
- **Cross-Slot Canary Isolation:** Mixed-length concurrent batches (`B=4`, `B=8`) MUST exhibit **zero** cross-slot canary leakage across all `(slot_i, slot_j)` pairs (`i != j`).

### Guarantee 5 — Default OFF & Fail-Closed Control Plane

- **Bit-Identical Default Baseline:** Every opt-in optimization (`ATLAS_DETERMINISTIC_BATCH`, `ATLAS_COMPLETIONS_RAW_NO_THINK_PREFIX`, `ATLAS_LM_HEAD_INT8_BLOCK32`, `ATLAS_GDN_1D_REPLAY`, `ATLAS_TENSOR_FP8_PREFILL`, `ATLAS_GDN_L2_WINDOW_MIB`, `ATLAS_GB10_FAST_PREFILL`, `ATLAS_SPEC_EXACT_GPU_ACCEPT`, `ATLAS_W4A4_DECODE_SK3`, `ATLAS_BITFP4_KV`, `ATLAS_BITFP4_RAZER_KV`, `ATLAS_FUSED_RMSNORM_NVFP4`, `ATLAS_LM_HEAD_FUSED_ARGMAX`, `ATLAS_DUAL_PLANE_NVFP4`, `ATLAS_DCOCO_SIDECAR*`, `ATLAS_DFLASH_DEFER_EMIT`, `ATLAS_DFLASH_OPT_BRANCH0`, etc.) defaults **OFF** when unset or empty, executing zero extra allocations and preserving the baseline code path.
- **Raw `/v1/completions` vs `/v1/chat/completions` Cross-Engine Tokenization Contract:**
  - By default, `crates/spark-server/src/api/completions.rs` prepends `<think></think>\n\n` (`completion_thinking_prefix`) to raw string `prompt` requests on thinking-capable checkpoints so interactive raw completions do not silently enter open-ended chain-of-thought.
  - Because cross-engine OpenAI-API exactness benchmarks feeding raw text to `/v1/completions` diverge at token `0` if one engine prepends `<think></think>\n\n` and another does not, cross-engine exactness certification MUST use one of three sanctioned exact-tokenization paths:
    1. **Pre-tokenized `/v1/completions` via `"prompt_token_ids"`** (`tools/flux_exactness_cert.py --cross-engine-parity --endpoint /v1/completions --prompt-token-ids <path.json>`): bypasses both BPE tokenization and `<think></think>\n\n` injection verbatim.
    2. **Chat endpoint with thinking explicitly disabled** (`tools/flux_exactness_cert.py --cross-engine-parity --endpoint /v1/chat/completions --enable-thinking false`): sends both `"enable_thinking": false` and `"chat_template_kwargs": {"enable_thinking": false}` so engines render the exact same non-thinking chat template.
    3. **Opt-in raw-text server toggle (`ATLAS_COMPLETIONS_RAW_NO_THINK_PREFIX=1`, `ZERO_OR_ONE`, default OFF):** disables both the automatic `<think></think>\n\n` input prepend and `strip_thinking_tags` output leading-whitespace trimming on raw `/v1/completions` text prompts when explicitly armed by the operator (preserving stop-sequence stripping).
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

### Guarantee 7 — Cold-Prefill vs Warm-Prefix-Resume Precision Parity

- **Problem:** Prefix-cache hits resume generation by restoring cached KV/GDN state up to the aligned prefix boundary and evaluating only the uncached suffix tokens (`1..block_size` rows). If cold prefill executes boundary tokens under one MoE/GEMM precision mode (e.g., `W4A4` batched prefill GEMM) while warm prefix-cache resume evaluates those same boundary tokens under a different precision mode (e.g., `W4A16` / `W4A8` small-`M` decode GEMM), even a 1-ULP logit perturbation on near-tie greedy steps causes deterministic trajectory divergence between cold and warm runs—breaking prefix-cache transparency!
- **Invariant:**
  - **Matched boundary precision across cold & warm execution:** Cold prefill and warm prefix-cache resume MUST NEVER evaluate overlapping or boundary tokens under mismatched MoE/GEMM precision modes (such as `W4A4` cold prefill vs `W4A16` warm resume).
  - Any fast-prefill precision policy (including `ATLAS_GB10_FAST_PREFILL=1`, `ATLAS_TENSOR_FP8_PREFILL=1`, and `ATLAS_FUSED_RMSNORM_NVFP4=1`) MUST either keep identical precision routing across cold-prefill and warm-resume boundary tokens OR partition prefix-cache keys by active precision mode so mismatched cached states can never be replayed across precision boundaries.
- **Verification:** `tools/flux_exactness_cert.py` Suite 3 (`cold_vs_warm_replay`: exact token trajectory match, `<= 1e-5` max abs logprob diff, `D_KL < 1e-4` with positive warm `cached_tokens`).

---

## 2. Quantitative Certification Matrix

| Certification Check | Metric | Pass Gate | Owner Tool |
|---|---|---|---|
| Exact arithmetic | Exact integer/mod output | `100%` exact (`4/4`) | `tools/flux_exactness_cert.py` |
| Cold needle passkey (`4K/8K/16K/32K`) | Verbatim recall incl. `FLUX-V12-KV-14336-9472` | `100%` exact | `tools/flux_exactness_cert.py` |
| Verbatim `code_rust` | Canonical SHA-256 of emitted Rust function | Exact SHA-256 match | `tools/flux_exactness_cert.py` |
| Cold-vs-warm prefix/GDN replay (Guarantee 7) | Matched cold/warm precision, trajectory & logprob diff | `100%` exact, `<= 1e-5` logprob diff, `D_KL < 1e-4` | `tools/flux_exactness_cert.py` |
| Mixed batch invariance (`B=1` vs `B=4/8`) | `4/4` canonical trajectories & cross-slot canary check | `4/4` (`100%`) exact (`--require-batch-invariance`), `0` canary leaks | `tools/flux_exactness_cert.py` |
| Cross-engine prompt-token parity | Pre-tokenized `prompt_token_ids` or chat `enable_thinking=false` | `100%` exact (`--cross-engine-parity`), `D_KL < 1e-4` | `tools/flux_exactness_cert.py` |
| Verify-width invariance (`W=1` vs `W=8` vs `W=16`) | Trajectory exactness, first divergence, `D_KL` | `100%` exact, `first_div = null`, `top1_flip = 0`, `D_KL < 1e-4` | `tools/flux_exactness_cert.py` |
| `INT8 per-32 BF16sc` (`8.50 bpw`) `lm_head` | 0-ULP `int8->BF16`, SQNR, `D_KL@T=1`, Pareto dominance | `SQNR >= 44.0 dB` (`~45.4 dB`), `D_KL < 1e-6`, `0%` flips, beats DualPlane & MXFP8 | `tools/certify_quant_exactness.py` |
| `DualPlaneNvfp4` (`8.50 bpw`) `lm_head` | Weight/logit SQNR, Top-1 flip, `D_KL@T=1`, tie-break | `SQNR >= 37.5 dB`, `+6 dB` vs FP8, `flip < 0.1%`, `D_KL < 1e-4`, exact tie-break | `tools/certify_quant_exactness.py` |
| Strict `BF16` `lm_head` equivalence | Logit SQNR vs FP32 | `SQNR >= 45.0 dB` (`~55.6 dB` observed) | `tools/certify_quant_exactness.py` |
| `BitFP4 + 6-op SWAR RaZeR-KV` | `-0.0` SWAR, 6-op split, MSE reduction, attn cosine | `0` SWAR errors, `MSE_razer < MSE_nvfp4`, `logit_cos >= 0.995`, `out_cos >= 0.990` | `tools/certify_quant_exactness.py` |
| PTX/SASS ISA formal contracts | Stack/spill bytes, odd `desc[UR]`, MMA/L2 ISA | `0 B` stack/spills, `0` odd `desc[UR]`, `100% PROVED` | `repro-kit/formal/verify_ptx_isa_contracts.py` |
| Env catalog & leak hygiene | Uncataloged env vars / Tier-0 leaks | `0` uncataloged, `0` leaks | `env_inventory.py --check`, `verify_no_leaks.py` |

---

## 3. Live GB10 Flux v1.4 Exactness Certification Receipts

Measured directly on **NVIDIA GB10** (`SM 12.1`, `48 SMs`, `24.0 MiB` L2) at verified Flux v1.4 commit `d5df69cedb` (`33` kernels / `421` entries `PROVED (sm_121a PTX+SASS)`, `0 B` stack/spills, `0` odd `desc[UR]`).

### 3.1 Live GB10 GPU Kernel `0-ULP` Exactness Proofs

| Kernel / Track | Live GB10 Shape | Exactness Proof (`0-ULP`) | Observed GB10 Telemetry |
|---|---|---|---|
| `lm_head_int8_block32` (`V14-T5` / `V15-T2` / `V16-T3`) | `Qwen3.8-Flash-Next` (`V=248320, K=2560`, `675.4 MB` vs `1,271.4 MB` BF16) | `M1==M8==M16==M32==M64 diff=0/248320`, `gemm_argmax == fused_argmax == 12345` | `M=1..16`: `3.23–3.29 ms` (`206.9 GB/s`, `I2F=0`); `M=32`: `3.45 ms` (`192.1 GB/s`, `1.89x`); `M=64`: `5.54 ms` (v1.5 `128t`) / `256t` bank-conflict-free `80-reg` in v1.6 |
| `lm_head_int8_block32` (`V14-T5` / `V15-T2` / `V16-T3`) | `Qwen3.8-27B` (`V=248320, K=5120`, `1,350.9 MB` vs `2,542.8 MB` BF16) | `M1==M8==M16==M32==M64 diff=0/248320`, `gemm_argmax == fused_argmax == 12345` | `M=1..16`: `6.81–7.11 ms` (`196.7 GB/s`, `I2F=0`); `M=32`: `7.36 ms` (`177.2 GB/s`) |
| `w4a4_decode_roofline_sm120` v2 (`V14-T4A` / `V15-T3`) | `Qwen3.8-27B` `down_proj` (`5120×17408`, `split_k=3`, `50.14 MB`, `0 B` SMEM, `80 regs`) | `M1==M8 diff=0/5120`, `M1==M16 diff=0/5120` | `M=1..16`: `254.7–277.2 us` (`196.9 GB/s`, `1.63x` vs v1.4 `420 us`) |
| `w4a4_decode_roofline_sm120` v2 (`V14-T4A` / `V15-T3`) | `Qwen3.8-27B` `gate_proj` (`17408×5120`, `split_k=3`, `50.14 MB`, `0 B` SMEM, `80 regs`) | `M1==M8 diff=0/17408`, `M1==M16 diff=0/17408` | `M=1..16`: `249.6–291.5 us` (`200.8 GB/s`, ~97% GB10 LPDDR5X roofline) |
| `gdn_correction_replay_sm120` (`V14-T1`) | `Qwen3.8-27B` (`48L`, `nk=16, nv=48, D=128`) | `status=0`, `515.0 KB/L` 1D record vs `48.0 MB` 2D snapshot | `verify_1d`: `33.1 us/L` (`W=8`) / `40.6 us/L` (`W=16`); 48L `commit_1d`: `1.402 ms` (`215.4 GB/s R+W`) |

### 3.2 Live GB10 End-to-End Serving Exactness (`Qwen3.8-Flash-Next`)

| Serving Exactness Gate | Flux v1.4 Live GB10 Receipt (`arm1` Base & `arm2` Opt-In) | Status / v1.5 Action |
|---|---|---|
| Cold-vs-warm prefix/GDN replay (`prefix_repeat_exact`) | **`4/4`** (`traj_code_ringbuf`, `traj_prose_nav`, `traj_code_dijkstra`, `traj_arith_seq`; 256-tok SHA-256 identical, `first_diff = null`) | **PASS (`4/4`)** |
| Cold long-context needle recall | **`4/4`** (`needle_4k` 9,240 ptoks, `needle_8k` 18,467 ptoks, `needle_16k` 37,793 ptoks, plus `16,647`-ptok `FLUX-V12-KV-14336-9472` verbatim recall) | **PASS (`4/4`)** |
| Cross-request canary isolation (`B=8`) | **`0` canary leaks** across every concurrent `B=8` batch | **PASS (`0` leaks)** |
| Baseline `arm1` vs v1.4 opt-in `arm2` (`--compare`) | **`4/4` trajectory exact** (`diff = None`; matching SHA-256 `cb5977fdbf0df266`, `99e1b8bfb0a36534`, `c7bd29da45937832`, `352f3ac613aea27c`) | **PASS (`4/4`)** |
| Mixed batch invariance (`batch_b1_vs_b8_exact`) | `1/4` under default dynamic-`M` routing (`arith_seq` exact; `diff=124/32/258` on `ringbuf/prose_nav/dijkstra`) | **Remediated in v1.5** via `ATLAS_DETERMINISTIC_BATCH=1` (target `4/4` under `--require-batch-invariance`) |
| Raw `/v1/completions` cross-engine tokenization | Auto-prepended `<think></think>\n\n` shifted token `0` on raw string prompts | **Remediated in v1.5** via `--cross-engine-parity` (`--prompt-token-ids` / chat `enable_thinking=false`) & `ATLAS_COMPLETIONS_RAW_NO_THINK_PREFIX=1` |

---

## 4. Certification Commands

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
# Capture baseline reference with strict cross-engine parity + 4/4 B=1==B=4/8 gate
python3 tools/flux_exactness_cert.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen3.8-27B \
  --endpoint /v1/chat/completions \
  --enable-thinking false \
  --cross-engine-parity \
  --require-batch-invariance \
  --needle-lengths 4096,8192,16384,32768 \
  --batch-sizes 4,8 \
  --json-out /tmp/cert_baseline_w1.json

# Or run exact pre-tokenized /v1/completions probes (zero tokenizer / think-prefix drift)
python3 tools/flux_exactness_cert.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen3.8-27B \
  --endpoint /v1/completions \
  --cross-engine-parity \
  --prompt-token-ids /path/to/prompts_token_ids.json \
  --require-batch-invariance \
  --batch-sizes 4,8 \
  --json-out /tmp/cert_tids_ref.json

# Capture candidate run (e.g., W=8 or W=16 with v1.5 opt-in flags such as ATLAS_DETERMINISTIC_BATCH=1)
python3 tools/flux_exactness_cert.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen3.8-27B \
  --endpoint /v1/chat/completions \
  --enable-thinking false \
  --cross-engine-parity \
  --require-batch-invariance \
  --needle-lengths 4096,8192,16384,32768 \
  --batch-sizes 4,8 \
  --json-out /tmp/cert_candidate_w16.json

# Certify exactness diff (trajectory exactness, first divergence, top-1 flip, D_KL < 1e-4, 4/4 batch gate)
python3 tools/flux_exactness_cert.py \
  --compare /tmp/cert_baseline_w1.json /tmp/cert_candidate_w16.json \
  --require-batch-invariance \
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

# Mechanical PTX + SASS contract verification on sm_121a
python3 repro-kit/formal/verify_ptx_isa_contracts.py
```
