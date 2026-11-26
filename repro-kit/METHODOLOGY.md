# Flux Reproducibility Kit — Benchmark Methodology & Reviewer Q&A

- **Current Engine Release:** `v1.0.2+` (`v1.0.3` source tree; see `repro-kit/benchmark-config.yaml`)
- **Live Reproduction Tooling:**
  - `flux-bench` (`repro-kit/flux-bench`): End-to-end HTTP prefill sweep, manifest pin verification, and certified reference verification.
  - `lil-bench` (`repro-kit/lil-bench`): Local-Inference-Lab (`llm-inference-bench v0.7.9`) presets for `matrix`, `concurrency`, `estonia`, `coding`, `prefill`, `accuracy`, `suite`, and SVG/PNG/Markdown card generation.
  - `flux lens` (`repro-kit/flux_lens.py`): Standard-library-only exact-token context/concurrency trial runner and paired output comparator.
  - `r15-gates/` & `r15-container-suite.py` (`repro-kit/R15_EXACT_IMAGE_RELEASE.md`): Exact-image release qualification suite covering startup, tool API, GSM8K/MMLU-Pro, long-context MTP, native 262K, concurrency, prefill/decode profiles, memory floor, cancellation/slot reuse, and complete raw-log safety.
- **Historical Certified Baseline Dataset (`v0.9.12-6k-stable`):**
  The bundled reference measurements in `repro-kit/results.json` and `repro-kit/environment.json` record the historical `v0.9.12-6k-stable` GB10 baseline campaign (`6,063.5–6,082.2 tok/s` at 16K prefill, `5,684.0–5,694.4 tok/s` at 128K prefill, `57.4 tok/s` lossless K=5 coding decode, and `5/5` Estonia 136.6K needle retrieval). Those sealed reference numbers are preserved verbatim below for reproducibility comparison; live runs on `v1.0.2+` are invoked via `flux-bench`, `lil-bench`, `flux lens`, and `r15-gates/`.

---

## Direct Answers to External Reviewer Questions

### 1. Is that 6,000 prefill tokens/second per request or aggregate?

**Per single request (`concurrency = 1`, `batch_size = 1`, `--max-batch-size 1 --max-num-seqs 1`).**

- Every GB10 prefill figure reported in our headline tables is **single-request end-to-end Time-To-First-Token (TTFT) throughput** *(discrete SM120 tables in `docs/SM120_SM103_BRINGUP.md` report both server-measured TTFT and client wall-clock TTFT side by side)*, defined strictly as:
  ```text
  Prefill Throughput (tok/s) = prompt_tokens / TTFT_seconds
  ```
  where `TTFT_seconds` is the full HTTP client wall-clock duration from sending `POST /v1/chat/completions` (`max_tokens=1, temperature=0.0`) over loopback to receiving the response containing the first generated token.
- At `16,384` prompt tokens (`16K`), a single request completes in **`2.694–2.702 seconds`**, yielding **`6,063.5–6,082.2 tok/s`** for that single stream (`~1,348 ms` per 8K chunk) in the historical `v0.9.12-6k-stable` baseline.
- **Stage-Timer Breakdown (`ms/8K` at 16K context, `1,348.0 ms` per 8,192 tokens across 48 layers):**
  | Stage | Time per 8K Tokens (`ms`) | Share of Prefill (`%`) | Key SM121a Kernel / Path |
  | :--- | ---: | ---: | :--- |
  | **MoE Routed + Shared Experts (512 routed top-10 + 1 shared)** | `684.5 ms` | `50.8%` | `qwen4_moe_w4a8_v28` (`mma.sync.aligned.m16n8k64...e2m1.block_scale`), inter-expert worklist planner, fused slot quant |
  | **Gated DeltaNet Linear Attention (36 layers)** | `312.8 ms` | `23.2%` | `atlas_gdn_fi_sm121a_prefill_f16` (FlashInfer CuTe-DSL WY chunk scan), `gdn_conv_v3`, `gdn_norm_fp8_v2` |
  | **QSA Sparse/Full Attention + Projections (12 layers)** | `198.4 ms` | `14.7%` | `qsa_sparse_e4m3_qkpv` (`v2`), fused group-max top-k, `o_proj` absmax epilogue fold |
  | **4-Stream mHC Hyper-Connections (48 layers)** | `94.2 ms` | `7.0%` | `qwen4_hc_upmix_roof_t128_amax`, fused hyper-connection path |
  | **4-Ring NVFP4 PLE N-Gram Offload + Host/Stream Glue** | `58.1 ms` | `4.3%` | 4-ring sector-aligned DMA (`ATLAS_PLE_READ_RINGS=4`, `ATLAS_PLE_READ_AFFINITY=big`), `qwen4_ple_prefill_fused` |
  | **Total per 8,192 Tokens** | **`1,348.0 ms`** | **`100.0%`** | **`6,076.7 – 6,082.2 tok/s` single-request (`C=1`)** |

- **Multi-Request Concurrency & Multi-Spark (`2x–4x GB10`) Scaling:**
  - On a **single DGX Spark (`1x GB10`)**, the 512-expert MoE grouped GEMM is already compute-saturated at `C=1` (`w16384` routes `16,384 * 10 = 163,840` token-expert pairs across 512 experts = `320` rows/expert per chunk). Running concurrent requests in `--mode interactive-w4096` (`4,096`-token chunks, `11,907 MiB` free UMA floor) maintains `5,164–5,266 tok/s` aggregate prefill while interleaving decode steps with bounded tail latency.
  - Across a **4x DGX Spark cluster (`4x GB10` connected via 200GbE QSFP56 ConnectX-7 RoCEv2)** *(projected scaling estimates — not yet live-benchmarked on 4x hardware)*:
    - In **4-way Data-Parallel / Disaggregated Prefill** mode (`C=4`), projected aggregate cluster prefill scales linearly to **`~23,800–24,200 tok/s`**.
    - In **TP=2, EP=4 Single-Request** mode, projected cross-node single-request prefill targets **`~18,400+ tok/s`** across **512 GB** of aggregate cluster memory (`4x 128 GB` per-node UMA linked over 200GbE RoCEv2).

---

### 2. Which model and quantization?

- **Model ID:** `Hotshot2000/Qwen3.8-Flash-Next-NVFP4` (repacked NVFP4 PLE offload `Qwen3.8-Flash-Next-NVFP4-Offload`, kernel target `qwen3.8-flash-next`, checkpoint manifest SHA-256 `ed8aa879996dc322e3720eb9848a4193c4cfc1d7c05e0421d85108429bd15869`, 548 files).
- **Architecture (`qwen4_exp_text`):**
  - **48 hidden layers** in a 3:1 hybrid pattern (`full_attention_interval = 4`):
    - **36 Gated DeltaNet (GDN)** linear-attention layers (`linear_num_key_heads = 16`, `linear_num_value_heads = 48`, `head_dim = 128`, `conv_kernel_dim = 4`, FP32 recurrent state).
    - **12 Query-Sparse Attention (QSA)** full-attention layers (`num_attention_heads = 24`, `num_key_value_heads = 2`, `head_dim = 256`, `indexer_budget = 2048`, `indexer_compress_ratio = 4`).
  - **Sparse MoE per layer:** **512 routed experts** (`moe_intermediate_size = 640`) + **1 shared expert** (`shared_expert_intermediate_size = 640`) with **top-10 expert routing** (`num_experts_per_tok = 10`).
  - **Multi-Stream Hyper-Connections (mHC):** 4-stream residual hyper-connections (`hc_count = 4`, `hc_lowrank = 320`, `hidden_size = 2560`).
  - **Per-Layer Embeddings (PLE):** 20,000,000-entry 3-gram PLE table (`ple_embed_dim = 2560`, split across 128 sector-aligned NVFP4 shards).
  - **Speculative Draft Head (`Qwen3.8-Flash-Next-Inferact-MTP`):** 1-layer full-attention MoE Multi-Token Prediction (MTP) head (`num_drafts = 4`, verification batch `K = 5`, `mtp_vocab = 96000`).
- **Quantization:**
  - **Routed MoE Weights:** **NVFP4** (`FP4 E2M1` 4-bit float weights with hardware `UE4M3` block-16 scale factors, calibrated via NVIDIA ModelOpt `0.46.0` on 128 real CNN/DailyMail sequences).
  - **Fast Projections (GDN / QSA Linear Layers):** Cached **FP8 E4M3** tensor-scaled weights (`120` projection matrices — `72` GDN + `48` QSA — cached via `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=1`).
  - **Activations:** Dynamic **MXFP8 / FP8-E4M3** (slot-quantized MoE inputs and fused `o_proj` / QSA `qkpv` FP8 E4M3 activations).
  - **KV Cache & Residual Stream:** **BF16** KV cache (`--kv-cache-dtype bf16`) and BF16 residual stream, paired with **FP8-E4M3** QSA sparse block-selection/attention core (`qsa_sparse_e4m3_qkpv`).
  - **PLE Offload:** Direct sector-aligned **NVFP4** (`qwen38-flash-next-ple-nvfp4-direct-v1`, 4-ring asynchronous DMA).

---

### 3. Which GPU and number of GPUs?

- **Primary Certified Hardware:** **1x NVIDIA GB10** (Grace Blackwell Superchip inside a single **NVIDIA DGX Spark** node).
  - **GPU Architecture:** Blackwell `sm_121a` (Compute Capability `12.1`), **48 Streaming Multiprocessors (SMs)**, **32 MiB L2 cache**, native register-backed block-scaled FP4/FP8 Tensor Core instructions (`mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32.Kind::f8f6f4.block_scale.scale_vec::2X`).
  - **Memory:** **128 GB physical (`119.7 GiB` OS-visible) coherent LPDDR5X Unified Memory Architecture (UMA)** shared between Grace CPU and Blackwell GPU (`301 GB/s` theoretical peak, `273 GB/s` sustained STREAM triad).
  - **CPU:** 20-core ARMv9.2 Grace CPU (`10x Cortex-X925` performance cores at indices `5-9,15-19` + `10x Cortex-A725` efficiency cores at `0-4,10-14`). The server process and PLE DMA rings are pinned to the 10 `Cortex-X925` performance cores (`taskset -c 5-9,15-19`, `ATLAS_PLE_READ_AFFINITY=big`).
  - **Software Stack:** Ubuntu 24.04 LTS (`aarch64`), Linux kernel `6.17.0-1014-nvidia`, NVIDIA Driver `580.126.09`, CUDA `13.0` (`V13.0.88`), cuBLAS `13.1.0.3`.
- **Supported Multi-Node & Discrete GPU Targets:**
  - **2x / 3x / 4x NVIDIA DGX Spark (`GB10`):** Connected via dual-port NVIDIA ConnectX-7 200GbE QSFP56 RoCEv2 (`recipes/2x-spark-tp2-disagg.env`, `recipes/3x-spark-pp3-disagg.env`, `recipes/4x-spark-cluster.env`).
  - **NVIDIA RTX PRO 6000 Blackwell (`SM120`):** 188 SMs, 96 MiB L2 cache, 96 GB GDDR7 (`1,792 GB/s`), sharing the register-backed block-scaled FP4/FP8 Tensor Core ISA with `SM121a`.
  - **NVIDIA B300/GB300 (`SM103a`, Phase-A builds cleanly), B200 (`SM100a`), & H100/H200 (`SM90a`):** Roadmap / launcher capability stubs (not yet live-benchmarked; see `docs/SM120_SM103_BRINGUP.md`).

---

### 4. What prompt length and batch size?

- **Batch Size & Concurrency:** `batch_size = 1`, `concurrency = 1` (`--max-batch-size 1 --max-num-seqs 1`).
- **Prefill Sweep Prompt Lengths (`max_tokens = 1`, `temperature = 0.0`):**
  - **`8,192`** tokens (`8K`)
  - **`16,384`** tokens (`16K`)
  - **`32,768`** tokens (`32K`)
  - **`65,536`** tokens (`64K`)
  - **`131,072`** tokens (`128K`, `--max-seq-len 135168`)
- **Long-Context Needle Prompt Length (Estonia Benchmark, `lil-bench estonia`):**
  - **`136,576`** prompt tokens (`707,468` characters, **`136,584` active KV tokens** during `K=5` speculative verification, `--max-seq-len 147456`).
- **Decode Generation Lengths:**
  - **`256` tokens** per prompt across the 10-prompt greedy lossless verification suite (`0`, `8K`, `32K`, `64K` prompt depths = `2,560` generated tokens per arm).
  - **`512–2,048` tokens** in `lil-bench coding` and `lil-bench matrix` (`llm-inference-bench v0.7.9`).

---

### 5. How is prefill time measured?

1. **Primary Metric: End-to-End HTTP Client Wall-Clock TTFT (`flux-bench` / `lil-bench` / `flux lens`):**
   - Measured in the external client process using monotonic wall-clock timers (`time.perf_counter()`) wrapping the full HTTP `POST /v1/chat/completions` request (`max_tokens=1, temperature=0.0`) from socket write to complete receipt of the first generated token.
   - Run live against a serving endpoint with `flux-bench run --endpoint http://127.0.0.1:8899 --output /results/results.json` or `lil-bench prefill`.
2. **Zero Prefix-Cache / State-Cache Reuse:**
   - Server is launched with `--enable-prefix-caching false` and `--ssm-cache-slots 0`.
   - Every request zeroes its allocated KV and Gated DeltaNet recurrent state slots (`KV_ZERO_COALESCE_ENGAGED`, `ALLOC_SEQ_ASYNC_ENGAGED`) and processes 100% of prompt tokens through all 48 layers.
3. **Cross-Checked Against Server CUDA Event Timers:**
   - The server logs internal CUDA event timing and per-stage census receipts (`ms/8K`). Client wall-clock TTFT includes full HTTP serialization, token scheduling, PLE prefetch, all 48 transformer/GDN layers, final norm, `lm_head`, and greedy argmax sampling.

---

### 6. Does the result reproduce over repeated runs?

**Yes: both across repeated requests within a run (`N=3` timed repetitions per length) and across independent builds.**

1. **Within-Run Repetition Protocol (`flux-bench run`):**
   - At each context length (`8K, 16K, 32K, 64K, 128K`), `flux-bench run` executes **1 uncounted transition request (`rep 0`) followed by `N = 3` consecutive timed repetitions (`rep 1, 2, 3`)** using the same prompt with unique `cache_salt` values, and checks identical output SHA-256 hashes across all counted repetitions.
   - **Historical `v0.9.12-6k-stable` Baseline Variance (`w16384`, archived in `results.json`):**
     | Context | Tokens | Rep 1 (tok/s) | Rep 2 (tok/s) | Rep 3 (tok/s) | Median (tok/s) | Min–Max Spread | Output SHA-256 (All 4 Reps) |
     | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
     | **8K** | `8,192` | `5,815.7` | `5,787.6` | `5,816.7` | **`5,815.7`** | `0.50%` | `dbba921ce205` (`4/4` identical) |
     | **16K** | `16,384` | `6,061.8` | `6,071.6` | `6,063.5` | **`6,063.5`** | `0.16%` | `0695b563acde` (`4/4` identical) |
     | **32K** | `32,768` | `5,969.6` | `5,945.0` | `5,943.0` | **`5,945.0`** | `0.45%` | `e47125968b3b` (`4/4` identical) |
     | **64K** | `65,536` | `5,801.0` | `5,803.2` | `5,807.9` | **`5,803.2`** | `0.12%` | `0695b563acde` (`4/4` identical) |
     | **128K** | `131,072` | `5,684.0` | `5,679.2` | `5,694.4` | **`5,684.0`** | `0.27%` | `c51cdffb6e38` (`4/4` identical) |
2. **Cross-Campaign Stability (`v0.9.12-6k-stable` Baseline Archive):**
   - Across four consecutive independent `w16384` baseline verification runs, 16K prefill median remained within `6,038.4–6,076.7 tok/s` (`6,082.2 tok/s` certified burst peak) and 128K median reached `5,684.0 tok/s` (`5,694.4 tok/s` max).
   - Every run passed the **`Same-prompt identity: 5 groups, 20 requests: all identical`** gate with identical output hashes (`dbba921ce205`, `0695b563acde`, `e47125968b3b`, `0695b563acde`, `c51cdffb6e38`).

---

### 7. Does output correctness remain comparable?

**Yes: verified via four independent correctness and quality gates (reproducible with `flux-bench`, `lil-bench`, `flux lens`, and `r15-gates/`):**

1. **Bit-Identical Greedy Speculative Decode (`r15-gates/plain-k5-long-context` & `flux lens`):**
   - Compares `K=5` speculative decoding (`--speculative --num-drafts 4`) against `K=1` standard non-speculative decoding (`ATLAS_DISABLE_SPECULATION=1`) under greedy sampling (`temperature = 0.0`) across short and long context depths.
   - **Historical Baseline Result (`v0.9.12-6k-stable`):** **`LOSSLESS: 10 identical, 0 different`** (`2,560 / 2,560` generated tokens match bit-for-bit across `0`, `8K`, `32K`, and `64K` prompts, proving Gated DeltaNet recurrent state rollback and QSA KV cache management are mathematically lossless).
2. **Same-Prompt Multi-Repetition Determinism (`flux-bench run`):**
   - Across all 5 context lengths (`8K, 16K, 32K, 64K, 128K`) and repeated trials per length, `flux-bench run` hashes outputs and fails (`status = "FAIL"`, exit code `1`) on any divergence. All certified `w16384` and `w4096` baseline runs scored **`20/20` bit-identical**.
3. **136,584-Token Estonia Long-Context Factual Retrieval (`lil-bench estonia`):**
   - Evaluates needle-in-a-haystack factual retrieval at **`136,576` prompt tokens** (`136,584` active KV tokens across 9 chunked prefill passes) with the fast prefill + `K=5` speculative decode stack active.
   - **Historical Baseline Result:** **`5/5 (100.0% PASS)`** on Run 1 (`25.29s` avg TTFT = `5,400.4 tok/s`, `43.8 tok/s` decode) and **`5/5 (100.0% PASS)`** on Run 2 (`10/10` total across both independent campaigns).
4. **Accuracy & Exact-Image Release Qualification (`lil-bench accuracy` & `r15-gates/`):**
   - Run `lil-bench accuracy gsm8k` / `lil-bench accuracy mmlu-pro` for standalone accuracy evaluation, or run the full exact-image release qualification suite (`r15_exact_image_release.py` + `r15-gates/`, see `repro-kit/R15_EXACT_IMAGE_RELEASE.md`) for matched target-only vs. `K=5` release gating.
   - In the historical checkpoint qualification recorded in the model manifest (`model-manifest.json`), AIME26 evaluation (30 problems × 8 repeats, `max_tokens = 130,000`, `temperature = 1.0`) scored **`98.75%` pass@1 (`237/240`, SEM `0.61pp`), `100%` majority@8, `100%` pass@8**, inside the BF16 reference band (`>= 236/240`).
