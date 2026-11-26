# Changelog — Flux Engine

> Flux Engine is an independently maintained Rust inference project developed from a fork of Atlas. We recognize the original Atlas authors and the wider open-source and research communities whose work contributed to its foundation. See [Credits](CREDITS.md), [Third-Party Notices](THIRD_PARTY_NOTICES.md), and [License](LICENSE) for project lineage, contributor recognition, and required notices.

---

## [v1.0.3] — 2026-11-26 (Open-Source Readiness, Discrete Blackwell `SM120` Certification & Supply-Chain Provenance)

- **Certified Discrete Blackwell (`SM120` — `1× RTX PRO 6000 Blackwell Workstation Edition`, `600 W`, `96 GB GDDR7`):**
  - **`Qwen3.8-Flash-Next-NVFP4` (`BF16` KV, `mmap` PLE, `DFlash2` `W=8`):** `C=1` decode up to **`661.1 tok/s`** arith / **`507.0 tok/s`** code-edit / **`370.1 tok/s`** code / **`330.9 tok/s`** reasoning at **`11.66 ms/round`** (default FP8 `lm_head`), or **`678.3 tok/s`** arith / **`550.5 tok/s`** code-edit at **`11.29–11.40 ms/round`** (opt-in NVFP4 `lm_head`); `16K`/`32K` cold prefill **`30,869` / `30,394 tok/s`** exact `W4A8` (**`34,308 tok/s`** opt-in `W4A4`, `5/5` needles); concurrency scaling to **`3,687.1 tok/s`** (`C=16` arith wall) and **`3,405.7 tok/s`** steady (`C=64`).
  - **`Qwen3.8-27B-NVFP4` (`BF16` KV):** `8K`/`16K` cold prefill **`16,248` / `15,012 tok/s`** (`3/3` needles); hoisted cross-sequence batched `DFlash2` **`438.4 tok/s`** (`C=1` arith) / **`590.1`** (`C=2` wall) / **`600.1`** (`C=4` arith) / **`1,073.8`** (`C=8` arith wall, `~2,327` steady); `C=64` plain **`1,974.8 tok/s`** (`97.3%` GDDR7 roofline).
  - **`1M` YaRN4 Context & `1×`–`8×` Pooled NVFP4 KV:** **`3/3` `1,011,019`-token passkey recall PASS** on `1× 96 GB` (`82.07–82.18 s` TTFT, `12,303–12,319 tok/s`); resident KV pool `1.46M` (`hp=auto`) to `2.30M` (`hp=0`, `2.0M` in `13.50 GiB`) on `1× 96 GB`, auto-pooling `6.3M → 8.4M → 16.8M` across `2×/4×/8×` discrete GPUs via `flux-autodetect`.
  - Added hardened `docker/sm120/Dockerfile` (`FLUX_ENGINE_VERSION=1.0.3`, `ATLAS_QWEN4_PREFILL_ATTN_FI="0"`, universal/GLM-5.3 binary symlinks) and complete bring-up receipts in [`docs/SM120_SM103_BRINGUP.md`](docs/SM120_SM103_BRINGUP.md).
- **Supply-Chain Provenance, Hardened CI & Open-Source Packaging:**
  - Upgraded release automation to **SLSA Build Level 3** (`slsa-github-generator`), **Sigstore Cosign keyless** Rekor signing, deterministic **SPDX 2.3 SBOM** (`scripts/generate_cargo_spdx_sbom.py`), GitHub Artifact Attestations, fail-closed `.github/allowed_signers` SSH signed-tag enforcement, signed multi-arch GHCR container releases (`container-release.yml` + Trivy SARIF), OpenSSF Scorecard, CodeQL SAST, GPU-less `nvcc` PTX/SASS zero-spill CI (`kernel-compile.yml`), and prompt-injection-hardened `claude.yml` / `claude-code-review.yml`.
  - Consolidated AGPL-3.0-only licensing, third-party notices, cryptographic signing docs (`SIGNING.md`), `CITATION.cff`, `TRADEMARKS.md`, public contributor `CLAUDE.md`, `llms.txt`, `docs/API.md`, and `docs/CLAUDE_CODE.md`, and aligned CLI/recipe defaults for safe loopback (`127.0.0.1`) binding.

---

## [v1.0.0] — 2026-11-23 (GA Production Release: v1.0.0 / v1.0 / 1.0.0 / 1.0 / gb10)

### Headline Results on NVIDIA DGX Spark (`1× GB10` / `sm_121a`, `ghcr.io/theapathy/flux-engine:v1.0.0`)
- **Out-of-the-Box Zero-Flag Container Parity & 33s Cold Start:**
  - AOT `sm_121f` CUBIN compilation + GPU-batched MoE weight transpose reduces cold container readiness to **`32.55s`** (base `C=1`), **`33.18s`** (`C=16` + DFlash2), and **`34.01s`** (`C=1` + DFlash2).
  - `FLUX_MODE=auto` routes directly to `flux-nextgen-max-prefill-decode.env` and auto-attaches a mounted companion `Qwen3.8-Flash-Next-DFlash2` drafter (`NVFP4` `W=8`, `noise_only=1`, `noise_graph=1`, `rejection` verify with `temp_scale=1.15`, `kv_pool_tokens=262144`) across `C=1..16` with standard Docker runtime flags and zero `--entrypoint` / `--env-file` / `-e ATLAS_*` overrides.
- **Zero-Flag `C=1..16` Throughput & `0-ULP` Exactness (`Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`):**
  - `C=1` Base-Only: **`77.96 tok/s`** (`greedy_code_256`, `EXACT_MATCH`), **`78.94 tok/s`** (`greedy_math_256`, `EXACT_MATCH`), **`6,380.8 tok/s`** (`6,469.9 tok/s` peak) `16K` cold prefill.
  - `C=1` Base + DFlash2: **`82.39 tok/s`** (`greedy_code_256`, `EXACT_MATCH`), **`81.03 tok/s`** (`greedy_math_256`, `EXACT_MATCH`), **`6,230.8 tok/s`** (`6,304.5 tok/s` peak) `16K` cold prefill, **`120/120`** FP8 projection weights cached (`complete=true`), **`0`** graph recaptures.
  - `C=1..16` Non-Greedy Prose (`T=1.0`, `top_p=0.95`, `top_k=20`, thinking ON, `256` tok/stream): **`66.9`** (`C=1`), **`104.3`** (`C=2`), **`140.6`** (`C=4`), **`213.1`** (`C=8`), and **`298.8 tok/s`** (`C=16`); Greedy `C=16` = **`298.7 tok/s`** (`0-ULP` bit-identical to `K=1`).
  - Long-context decode: **`52.0 tok/s`** at `128K C=1` (`50.0 tok/s` stock) and **`77.2 tok/s`** at `16K C=2`; Marconi warm prefix reuse **`5.93× / 7.34×`** speedup; HashMap `C=16` **`413.2 tok/s`**.
- **Correctness, Long-Context, Tool-Calling & Frontier Science Certification:**
  - **`0-ULP` Bit-Exactness:** `8/8` greedy SHA-256 `0-ULP` exact parity vs `K=1` (`QSA` verify graph `pack_key` bit-10 fix + exact `TV=0` prefix-only `s_w[0]` rejection verify + ring-wrapped `4K..262K` DFlash2 noise graph).
  - **`L3 Tool-Eval Hard Mode` (`92` scenarios):** **`92/92 (100.0% completion)`**, `0` HTTP `500` (tool-call reconcile fix), `0` HTTP `503`.
  - **`L4 Long-Context & Consistency`:** `Estonia 136.6K` **`5/5 (100%) @ 63.0 tok/s`**, `Needle-Checksum` **`64/64 (100%)`**, `Hotel-Lights` **`15/16 (93.8%)`**, `3/3` NIAH.
  - **`L5a GPQA-Diamond` (`198` graduate science Qs):** **`152/169 (89.94%)`** on non-KV-evicted items (`124/132 = 93.9%` normal, `28/37 = 75.7%` thinking-budget capped) and `160/198 (80.8%)` prior to the `v1.0.0` `ATLAS_KV_ADMIT_WATERMARK=122880` (`120K`) admission bound that prevents `262K` shared-pool `KV_POOL_EXHAUSTED` overcommit on `131K` reasoning chains while preserving full `C=16` throughput on `<=16K` workloads.
- **UMA Headroom Hardening & Clean Packaging:**
  - Bounded UMA-headroom-aware LRU eviction for `verify_kgamma_graphs` (`12`/tier, `3,072 MiB` pressure threshold) and `batch_decode_graphs` (`48` steady / `16` under pressure), plus `3,584 MiB` `C>=8` FP8 cache floor margin and `NOISE_GRAPH_REGISTRY` cleanup on `Drop`.
  - Three audited `sm_121a` binaries (`flux-server-real`, `flux-server-universal`, `flux-server-glm53`) packaged for `ghcr.io/theapathy/flux-engine:v1.0.0` (`:v1.0`, `:gb10`, `:latest`).

---

## [v0.9.16] — 2026-11-19 (Next-Gen Unified Release: v0.9.16 / 9.16.0 / gb10 / latest)

### Headline Results on NVIDIA DGX Spark (`1× GB10` / `sm_121a`, `ghcr.io/theapathy/flux-engine:v0.9.16`)
- **Lossless `0-ULP` Speculative Decode & Multi-Step Reasoning (`K=5` Native MTP + `K=7` Drafting, `100%` SHA-256 Greedy Match vs `K=1`):**
  - Single-stream (`C=1`, `T=0`): **`87.82 tok/s`** (`Short 256`) and **`81.01 tok/s`** (`Code 1024`) on `NVFP4-Offload`; **`74.79 tok/s`** (`Short 256`), **`73.76 tok/s`** (`Code 1024`), and **`65.17 tok/s`** (`Algo 1024`) on `QAD5500-Hybrid` (`+73.7%` to `+76.0%` over `41.85–43.06 tok/s` `K=1`, `0-ULP` identical SHA-256); **`61.23 tok/s`** unbiased stochastic `Gumbel-Max` (`T=0.7`).
  - Certified `3 / 3` multi-step `enable_thinking=true` reasoning chains (`aime_mod_arith` `2,572` tok @ **`65.00 tok/s`** -> `31` with `100%` SHA-256 token identity vs `K=1` across the `2,048` QSA sparse boundary; `aime_diophantine` `1,902` tok @ **`61.66 tok/s`** -> `45`; `long_qsa_boundary_reasoning` `1,774` tok @ **`58.36 tok/s`** -> `432`).
  - Shipped `1×1×1` `48`-CTA `0-ULP` fused GDN verify (`ATLAS_QWEN4_GDN_VERIFY_FUSED=1`), PLE segmented CUDA-graph suffix (`Qwen4NativeMtp`), fused QSA/paged sigmoid gating (`ATLAS_QWEN4_QSA_FUSED_GATE=1`), `rows <= 64` CUDA Programmatic Dependent Launch (`ATLAS_CUDA_PDL=1`), ARMv9.2 `WFET` stream synchronization (`ATLAS_CUDA_WFET_SYNC=1`), and M17 cross-barrier GEMV prefetch (`209 / 209` `sm_121a` modules with `0` stack bytes and `0` register spills).
- **Sustained `SM121a` Cold & Warm Prefill (`6,055–6,432 tok/s` Cold, `12,056 eff. tok/s` Marconi Warm):**
  - Cold `16K`: **`6,055.26 tok/s`** (`QAD5500-Hybrid`) and **`6,380.0 tok/s`** HTTP / **`6,432.4 tok/s`** TTFT (`NVFP4-Offload`); Cold `8K`: **`6,007.89 tok/s`** (`QAD5500-Hybrid`) / **`6,236.8 tok/s`** (`NVFP4-Offload`); unaligned `7,515`-token prompt: **`5,842.45 tok/s`** via per-output `128`-row FP8 TMA padding (`ATLAS_QWEN4_FP8_PAD128=1`); warm `8K` Marconi prefix hit: **`682.7 ms`** (`12,056 eff. tok/s`).
  - Added strip-mined L2-resident GDN-FI prefill (`ATLAS_QWEN4_PREFILL_GDN_FI=2`), W4A8 shared-expert pretranscode, `v28q=2` zero-spill compact MoE (`#pragma unroll 1` SASS I-cache fix), 9-attribute persistent cuBLASLt plan cache (`ATLAS_CUBLASLT_CACHE=1`), and batched Marconi suppressed continuation prefill.
- **`C=1..128` High-Concurrency Cycles, Display-Carveout (`0 MiB` UMA KV Cost) & `SLO`-Budgeted Interleave:**
  - Expanded `QWEN4_NATIVE_MAX_ROWS` from `32` to **`128`** with compact lazy-`H` SSM pool (`-42.7 GiB` UMA at `C=128`), `12:1` GQA `SK=32` Warp-0 ragged paged decode (`ATLAS_PAGED_DECODE_RAGGED_GQA=1`), `64`-seq batched MTP proposer, and `2`-frame balanced `K=2` speculative verify (`C=65..128`, up to `256` rows/cycle): **`91.50 agg tok/s`** (`Spec C=32`), **`108.49 agg tok/s`** (`Spec C=64`), **`107.12 agg tok/s`** (`Spec C=72`), and **`140.33 agg tok/s`** (`Native C=128`) on `display-carveout` (`4,090 MiB` framebuffer reclaim).
  - `--decode-gap-target-ms 150` (`WIDE=2048` prepared interleave) cuts max decode stall during concurrent `8K` prefill **`89×`** (`30,086 ms -> 337.6 ms` at **`36.28 tok/s`** active decode) while accelerating concurrent `8K` TTFT **`20.4×`** (`137.8s -> 6.74s`).
- **Single-Spark `1,048,576`-Token (`1M`) `4.0×` YaRN + `NVFP4`-KV + `QSA v3` & Multi-Spark (`TP=2`, `PP=3`, `1P+1D/2D`, disaggregated routing):**
  - Unlocked full `1,048,576`-token single-Spark context (`--mode 1m-nvfp4`, `~29 GiB` 4-bit `E2M1 + UE4M3` `NVFP4` KV pool vs `~103 GiB` BF16) with unit `attention_factor = 1.0` frequency-only `4.0×` YaRN synced across target/QSA/MTP heads and slabbed `32`-pass radix `QSA v3` (`>131K..1M` tokens).
  - Shipped `2×` Spark `(8, 24)` asymmetric GDN `TP=2` (`--mode 2x-spark`), `3×` Spark `20+20+20` `PP=3` pipeline ring & `1P+1D/2D` disaggregated prefill/decode (`--mode 3x-spark`), and `NonCausalSpan` prefix-cache isolation.

---

## [v0.9.15] — 2026-10-25 (Unified Release: v0.9.15 / 9.15 / gb10)

### Headline Results on NVIDIA DGX Spark (`1× GB10` / `sm_121a`, `ghcr.io/theapathy/flux-engine:v0.9.15`)
- **Single-Stream `C=1` Lossless Native-MTP `K=5` Speculative Decode (`8/8 id=OK`, `0` ULP greedy token-identical vs `K=1`):**
  - Short-context (`C=1`): **`91.26 tok/s`** (`91.47 tok/s` window) on `tf_streams_code_256`, **`87.72 tok/s`** (`87.94 tok/s` window) on `sparkdash_code_400`, **`86.86 tok/s`** on `rep0` (`+17.7%` vs `73.81` base), **`66.20 tok/s`** on `sparkdash_prose_400`, **`65.37 tok/s`** on `tf_streams_prose_256`, **`63.98 tok/s`** on `chat0` (`+13.5%` vs `56.39` base), **`62.59 tok/s`** on `code0` (`+18.8%` vs `52.67` base), and **`50.58 tok/s`** on `prose0` (`+12.0%` vs `45.16` base).
  - Long-context (`C=1`): **`55.27 tok/s`** (`66.17 tok/s` at `c=0.35`, `+66.8%` vs `33.14` base) at `8K` (`prose8k`), **`53.72 tok/s`** (`+19.1%` vs `45.11` base) at `32K` (`legal32k`), **`53.03 tok/s`** (`+28.2%` vs `41.36` base) at `64K` (`doc64k`), **`46.1 tok/s`** / **`43.79 tok/s`** (`mix128k`) at `128K`, and **`41.8 tok/s`** at `255K / 262K`.
- **Multi-Stream Concurrent Native-MTP `K=5` (`100%` `parity=True`, `0` serial fallbacks):**
  - Added cross-sequence batched `K=5` verify & 4-step MTP proposer (`ATLAS_QWEN4_BATCHED_K5=1`), `DEDUP_MAX_SLOTS = 255` unbounded concurrency, bounded-transpose MoE (`16 MiB` cap), and multi-row BF16/W4A16 verify kernels (`dense_gemv_bf16_batchn_tf`, `dense_gemv_ba_gates_batchn`, 8-warp `w4a16_gemv_batch_logits_exact_sw_m25_o1w4`, and 3-CTA/SM `pf_dual_m25`):
    - `C=2`: **`173.55 tok/s`** (`tf_streams_code_256`), **`121.94 tok/s`** (`tf_streams_prose_256`), **`105.22 tok/s`** (`sparkdash_code_400`), **`82.35 tok/s`** (`sparkdash_prose_400`)
    - `C=3`: **`258.22 tok/s`** (`tf_streams_code_256`), **`181.90 tok/s`** (`tf_streams_prose_256`), **`121.12 tok/s`** (`sparkdash_code_400`), **`95.52 tok/s`** (`sparkdash_prose_400`)
    - `C=4`: **`340.45 tok/s`** (`tf_streams_code_256`), **`241.71 tok/s`** (`tf_streams_prose_256`), **`128.66 tok/s`** (`sparkdash_code_400`), **`97.45 tok/s`** (`sparkdash_prose_400`)
    - `C=5`: **`420.28 tok/s`** (`tf_streams_code_256`), **`295.73 tok/s`** (`tf_streams_prose_256`), **`137.00 tok/s`** (`sparkdash_code_400`), **`103.46 tok/s`** (`sparkdash_prose_400`)
    - `C=8`: **`580.66 tok/s`** (`conc_harness.py`, `4.12x` scaling over `C=1`)
    - Plain `K=1` (`C=8` native batched QSA on GSM8K/MMLU-Pro): **`85.5 tok/s`** (vs `35.2 tok/s` at `C=1`).
- **Sustained Prefill (`20/20` bit-identical):**
  - **High-Prefill (`W16384`, `--mode max-prefill`):** **`6,028.5 tok/s`** median (`6,050.1 tok/s` peak) at `8K`, **`6,291.0 tok/s`** median (`6,299.3 tok/s` peak; `6,318.8 tok/s` stage run; `6,420.3 tok/s` chunk kernel peak) at `16K`, **`6,168.5 tok/s`** median (`6,169.0 tok/s` peak; `6,233.5 tok/s` stage run) at `32K`, **`6,006.5 tok/s`** median (`6,035.1 tok/s` peak) at `64K`, **`5,905.6 tok/s`** median (`5,925.5 tok/s` peak) at `128K`, and **`5,524.7 tok/s`** at `255K / 262K` (`MemAvailable >= 7,429 MiB`).
  - **Interactive (`W4096`, `--mode interactive-w4096`, `5/5 >= 5,000 tok/s`):** **`5,447.7 tok/s`** (`5,449.8 tok/s` peak) at `8K`, **`5,326.8 tok/s`** (`5,340.9 tok/s` peak) at `16K`, **`5,294.8 tok/s`** at `32K`, **`5,158.2 tok/s`** (`5,174.5 tok/s` peak) at `64K`, **`5,063.0 tok/s`** median (`5,068.6 tok/s` peak) at `128K`, and **`4,990.7 tok/s`** at `255K / 262K` (`MemAvailable >= 9,890 MiB`).
- **Native 262K Context (`262,144` tokens, `--mode 256k`):**
  - Full `262,144`-token KV allocation (`MemAvailable >= 11,452 MiB`), **`5/5` exact needle retrieval** from `1%` (`2,621`) to `99%` (`259,522`) depth, exact boundary enforcement (`262,144` accepted, `262,145` HTTP `400`), and post-cancellation slot recovery.
- **Greedy Bit-Exact Determinism (`b5fa66be2`) & Full GSM8K Accuracy:**
  - Fixed `ssm_Slot0` recurrent state leak across requests (`3x256` bit-identical raw logits); scored **`1,319 / 1,319` = `97.27%` (`1,283` correct, `0` interventions)** on full GSM8K.
- **Fast Direct-IO Loading & Container Provenance:**
  - Cold-start container readiness in **`138.0s` median (`<= 150s`)**, `88/88` provenance/recipe/closure tests PASS, published as `ghcr.io/theapathy/flux-engine:v0.9.15` (`9.15`, `0.9.15`, `gb10`; rollback baseline `v0.9.12-6k-stable` preserved).

---

## [v0.9.12-6k-stable] — 2026-10-17

### Unified Production Container & Multi-Checkpoint Compatibility (`ghcr.io/theapathy/flux-engine:9.12` / `:v0.9.12-6k-stable`)
- **Dual-Mode PLE Repacker (`scripts/repack_qwen38_flash_next_ple_nvfp4.py`, `flux repack-ple`):**
  - Added **Mode A (Native NVFP4 PLE passthrough)**: auto-detects checkpoints that already store pre-quantized NVFP4 PLE weights (`qwen3_5_moe` / `qwen3_5_moe_text`, `embed_tokens_per_layer.{weight,weight_scale,weight_scale_2}`) and copies them byte-for-byte into `ple/*.ple.bin` (`0.0` quantization error) while stripping PLE tensors from GPU safetensors shards (`~56 GiB -> ~30.6 GiB`).
  - Preserved **Mode B (BF16/FP8 -> Group-16 NVFP4 quantization)** with PyTorch ModelOpt (`max_e4m3 = 448.0`) and standalone NumPy bit-exact fallback (`--self-test`).
  - Added automatic `CYBER-FROST` Jinja2 `chat_template` sanitizer (`enable_thinking is defined and enable_thinking is false`) during metadata copy.
- **QAD5500 Hybrid & Built-In Weight-Only NVFP4 MTP Support:**
  - Extended `flux-server-real` (`6a4616cf7-qad5500`) to recognize nested `quantization_config.quant_algo == "MIXED_PRECISION"` and auto-detect in-checkpoint weight-only NVFP4 MTP (`mtp.layers.0.mlp.experts.*.weight_scale` without `input_scale`, `NumberedNvfp4WeightOnly`) with `--mtp-vocab 0` reuse of `lm_head`.
  - Prioritized in-checkpoint `mtp.*` weights over external sibling drafter directories in `flux-autodetect`.
- **`Qwen3.8-27B-NVFP4` (64-Layer Dense Hybrid DeltaNet):**
  - Bundled fused residual + RMSNorm + quantization paths and `k32_v17q` GEMM kernels (`--gpu-memory-utilization 0.55` default on GB10 UMA).
- **Container Runtime & Orchestrator Hardening (`flux` / `flux-autodetect`):**
  - Stripped `/usr/local/cuda/compat` from `LD_LIBRARY_PATH` whenever a real host `libcuda.so.1` is mounted (`using_compat_stub: false`, `driver_ok: true`).
  - Added `list-recipes`, `4x-spark-cluster.env`, `FLUX_DRAFT_MODEL=none` (`--draft-model none`), `FLUX_GAMMA`, `FLUX_THINKING`, `FLUX_THINKING_BUDGET`, `FLUX_CPU_AFFINITY`, and enriched `flux diagnose` JSON schema (`cuda_runtime`, `memory`, `selected_model`, `readiness`).

---

## [v0.9.11-6k-stable] — 2026-10-17

### Headline Results on NVIDIA DGX Spark (`1× GB10` / `sm_121a`)
- **16K-Chunk Prefill (`--mode max-prefill`, `w16384`):** Reaches **`6,082.2 tok/s` peak** (`6,076.7 tok/s` median at `16K`), **`5,830.2 tok/s`** at `8K`, **`5,969.6 tok/s`** at `32K`, **`5,828.3 tok/s`** at `64K`, and **`5,694.4 tok/s` peak** (`5,684.0 tok/s` median) at `128K` with `20/20` bit-identical token hashes across repetitions and `7,072 – 8,450 MiB` free unified memory floor.
- **4K-Chunk Interactive Prefill (`--mode interactive-w4096`, `w4096`):** Reaches **`5,276.3 tok/s` peak** (`5,266.8 tok/s` median) at `8K` and sustains **`5,016.1 – 5,266.8 tok/s`** across `8K..64K` (`4,899.8 tok/s` at `128K`) while preserving **`11,907 – 12,695 MiB`** free unified memory.
- **Lossless `K=5` Inferact MTP Speculative Decode:** Reaches **`57.4 tok/s` peak** (`56.3 tok/s` median on coding workloads, mean accept length `3.02–3.15`) and matches `--no-speculative` greedy decode **10/10 (`2,560 / 2,560` tokens bit-for-bit identical)** across `0..64K` contexts.
- **136.5K Estonia Long-Context 6-Hop Reasoning (`--mode long-context-136k`):** Scores **`5/5` (`100.0% PASS`)** across two independent campaigns (`10/10` total) on the 136,576-token 6-hop chain past the planted `Latvia` decoy (`5,400.4 tok/s` prefill, `25.29 s` TTFT, `48.6 tok/s` server median decode, `8,412+ MiB` free memory floor).
- **Universal Local-Model Container Appliance (`ghcr.io/theapathy/flux-engine:gb10`):** Added `/usr/local/bin/flux` smart entrypoint and `flux-autodetect` to inspect any mounted local checkpoint (`/models/<model>`) and configure `--kernel-target`, operating recipe (`FLUX_MODE`), PLE DMA rings, and speculative decoding (`gamma`) automatically.

### `sm_121a` CUDA/PTX Kernels & Rust Runtime
- **Compact Transposed W4A8 MoE (`v28q` & `v17q`):**
  - Added zero-`sE` register-quad shuffle MXFP8 epilogues (`v28_mxfp8_reg_quad` & `v17_mxfp8_reg_quad`), 16-warp prologue and in-loop B-weight transcoding (`tid < 512`), and hoisted `sfa[2]` scales (`0` register spills).
  - Routed `down_proj` through dual-occupancy `qwen4_moe_compact_t_gemm_k32_v17q` (`256` threads = `2 CTAs/SM`, `122` regs, `0` spills).
  - Fused MoE gate activation quantization into `qwen4_mxfp8_quant_rows_v16q4_gate` and added chunk-aware router GEMM dispatch (`fp8_gemm_ex` with `ROUTER_WEIGHT_CACHE` at `w4096` = `8.6 ms/8K`, and single-pass `ops::fp8_gemm_n128` at `w16384`).
  - Redirected `SHARED_W4A8_PRETRANSCODE` scratch (`5.07 MiB`) into idle `ctx.buffers.ssm_deinterleaved()` (`0 bytes` persistent VRAM overhead, saving `+243.4 MiB`) and gated unused `helpers_c.rs` FP8 buffers (`+285.0 MiB` saved).
- **4-Stream Multi-Stream Hyper-Connections (`mHC`):**
  - Fixed `gemm_order_verified` in `prefill_inject_gemm_admit.rs` by clamping `query_m` to `PIN_M = 4096` for `ref_m >= 545`, unlocking fused `injgemm`, `combine`, and `postpre` at `M = 16,384` (`-228 ms/16K`).
  - Split `qwen4_hc_prep_up` immediately ahead of `qwen4_hc_upmix_roof_t128` for 100% L2 residency of `s.up`.
  - Implemented warp-per-row period-6 unrolled `qwen4_hc_combine_moe_gemm_mx_gatepre` (`128` regs, `0` spills, `0` `token_to_perm` D2D copies).
- **Query-Sparse Attention (`QSA`):**
  - Upgraded `qwen4_qsa_prefill_score` and `qwen4_qsa_prefill_score_gmax` in-place with `ldmatrix.sync.aligned.m8n8.x4.shared.b16` (`4×` fewer shared loads), shift/mask paging, and streaming score stores at **5 CTAs/SM (`96` regs, `0` spills)**.
  - Added bimodal block selection: `qwen4_qsa_prefill_topk_v2` for `<= 8,192` blocks and `qwen4_qsa_prefill_topk_fused_c3_v2` (`32 MiB` score scratch) for `> 8,192` blocks.
  - Added per-half (`16`-token) `do_h0 / do_h1` HMMA skipping in `qwen4_qsa_prefill_attn_fp8_qkpv_amax_pipe` while preserving 100% PTX identity on reference entry points.
  - Skipped `index_qk_proj` and `iq_norm_rope_fused` on the 2,048 dense-window rows of Chunk 0.
- **Gated DeltaNet (`GDN`):**
  - Added 64-row tile fused `qwen4_fp8_act_and_ba_gates_mma` (`72` regs, `0` spills) and `COMMIT=true` `qwen4_conv1d_l2norm_f16_prefill_v3_commit` (`TOKENS=64`, shared-memory filter staging).
  - Removed `710 MiB` of unused legacy `gdn_buf_qkv/gb/out/z` buffers in `impl_a1.rs`.
- **NVFP4 Per-Layer Embeddings (`PLE`):**
  - Implemented zero-thread pre-deinterleaved `Arc<PackedPleBlock>` RAM cache (`PLE_CHUNK_CACHE`) and chunk-adaptive `PleRowsPlan::Pipelined`, eliminating `munmap` TLB shootdowns and cutting pure PLE overhead to `14.3 ms/8K`.
- **Lossless `K=5` MTP Speculative Decode & Unaligned Tail Chunks:**
  - Fixed QSA verify pool endpoint calculation (`pool_endpoint = seq_len + rows - 1`, now in `crates/spark-model/src/layers/qwen4_qsa.rs` & `qwen4_qsa/device.rs`) and `qwen4_qsa_required` guard so long-context speculative decode (`>= 2,048` generated tokens) is crash-free and `0-ULP` bit-exact.
  - Added 0-ULP multi-row `partial_rows_v2`/`reduce_rows` QSA verification and `ATLAS_QWEN4_K5_BATCH_HYPER=1`.
  - Relaxed `.is_multiple_of(16)` prefill guards to `rows >= 2` across 9 files with sub-4-token `stage_prefill_raw` handling.
