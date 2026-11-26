# Model Card — Qwen3.8-Flash-Next & Qwen3.8-27B (NVFP4 on GB10)

Architecture, quantization layout, memory footprint, and quality verification for the two primary checkpoints served by Flux Engine on **NVIDIA DGX Spark (GB10 / `sm_121a`)**.

---

## 1. `Qwen3.8-Flash-Next-NVFP4-Offload` + `Inferact-MTP` (`K=5`)

### 1.1 Architecture (`qwen4_exp`, `125B+51B PLE`, `6B` Active)

| Parameter | Value | Notes |
|---|---|---|
| **Layers** | `48` (`36` GDN + `12` QSA) | `3:1` hybrid pattern (`full_attention_interval = 4`) |
| **Hidden dimension** | `2,560` | 4-stream Multi-Stream Hyper-Connections (`hc_count = 4`, `hc_lowrank = 320`) |
| **Gated DeltaNet (GDN)** | `36` layers | `16` key heads, `48` value heads, `head_dim = 128`, `conv_kernel_dim = 4`, FP32 recurrent state |
| **Query-Sparse Attention (QSA)** | `12` layers | `24` query heads, `2` KV heads (`GQA 12:1`), `head_dim = 256`, `indexer_budget = 2,048`, `compress_ratio = 4` |
| **Routed + Shared MoE** | `512` routed + `1` shared | `top-10` routing (`num_experts_per_tok = 10`), `moe_intermediate_size = 640` |
| **Per-Layer Embeddings (PLE)** | `20,000,000` 3-gram entries | `ple_embed_dim = 2,560`, stored across `128` sector-aligned NVFP4 shards (`ple/`) |
| **Speculative Drafter** | `Qwen3.8-Flash-Next-Inferact-MTP` | Native 1-layer full-attention MoE MTP head (`num_drafts = 4`, verify batch `K = 5`, `mtp_vocab = 96,000`) |
| **Vocabulary size** | `248,320` | Identical tokenizer across target and MTP head |
| **Checkpoint manifest** | `548` files | Manifest SHA-256: `ed8aa879996dc322e3720eb9848a4193c4cfc1d7c05e0421d85108429bd15869` |

### 1.2 Quantization & Precision Map

| Subsystem | Weight Precision | Activation / State Precision | Runtime Kernel Path (`sm_121a`) |
|---|---|---|---|
| **512 Routed MoE Experts** | **NVFP4** (`E2M1` + `UE4M3` block-16 scales, ModelOpt `0.46.0`) | Dynamic **MXFP8** (`v16q4` slot-quantized) | `qwen4_moe_w4a8_v28q` (`gate_up`) + `qwen4_moe_compact_t_gemm_k32_v17q` (`down`, `2 CTAs/SM`) |
| **1 Shared MoE Expert** | **NVFP4** (`SHARED_W4A8_PRETRANSCODE=1` in-place) | Dynamic **MXFP8** | Fused shared-expert W4A8 path (`0` persistent scratch overhead) |
| **GDN / QSA Fast Projections** | **FP8-E4M3** (`120` matrices — `72` GDN + `48` QSA — cached via `FAST_PROJ_FP8_WEIGHT_CACHE=1`) | **FP8-E4M3** (`qwen4_fp8_act_and_ba_gates_mma`) | cuBLASLt tensor-scaled FP8 GEMM (`_RESERVE_GB=6`, `7`, or `10`) |
| **QSA Attention Core & KV Cache** | — | **BF16** KV cache + **FP8-E4M3** `qkpv` attention core | `qwen4_qsa_prefill_score` (`ldmatrix.x4`, `5 CTAs/SM`) + `qwen4_qsa_prefill_attn_fp8_qkpv_amax_pipe` |
| **GDN Recurrent State** | — | **FP32** state (`[48, 128, 128]` per layer: `48` value heads × `128×128`) + **FP16** WY scan | `atlas_gdn_fi_sm121a_prefill_f16` + `qwen4_conv1d_l2norm_f16_prefill_v3_commit` |
| **4-Stream mHC Stream Mixing** | **BF16 / FP32** | **BF16** streams + fused **FP8** `postpre` amax | `qwen4_hc_upmix_roof_t128_amax` + `qwen4_hc_combine_moe_gemm_mx_gatepre` |
| **20M-Entry PLE Table** | **NVFP4** (`128` shards on NVMe) | **BF16** combined embedding stream | 4-ring sector-aligned `O_DIRECT` DMA (`ATLAS_PLE_READ_RINGS=4`) + `PLE_CHUNK_CACHE` |

### 1.3 Unified Memory Footprint (`128 GB` GB10 UMA)

| Component | Resident Footprint (`w16384` `max-prefill`) | Resident Footprint (`w4096` `interactive`) |
|---|---:|---:|
| **Target + Inferact MTP Weights (NVFP4/BF16)** | `~74.2 GiB` | `~74.2 GiB` |
| **FP8 Fast-Projection Cache (`~2.5 GiB`) + CUDA/Runtime/OS Working Set (`~12.3 GiB`)** | `~14.8 GiB` | `~14.8 GiB` |
| **KV Cache (`135,168` max tokens, 12 QSA layers, BF16) + GDN State Pool** | `~11.4 GiB` | `~11.4 GiB` |
| **Prefill Chunk Arena (`WIDE=16384` vs `WIDE=4096`) + Runtime Scratch** | `~11.8 GiB` | `~7.4 GiB` |
| **Measured Free Memory Floor (`MemAvailable` during 8K–128K sweep)** | **`7,072 – 8,450 MiB`** | **`11,907 – 12,695 MiB`** |

### 1.4 Quality & Equivalence Gates

- **`LOSSLESS 10/10` Greedy Speculative Decode Equivalence:** Across 10 diverse greedy prompts (`256` tokens each = `2,560` tokens across `0`, `8K`, `32K`, and `64K` prompt depths), `K=5` speculative decoding (`57.4 tok/s` peak) produces **2,560 / 2,560 bit-for-bit identical tokens** to non-speculative `K=1` decoding (`flux` launcher `--no-speculative` / raw server `ATLAS_DISABLE_SPECULATION=1`, `33.4 tok/s`).
- **`20/20` Multi-Repetition Prefill Determinism:** Across all 5 context lengths (`8K, 16K, 32K, 64K, 128K`) x 4 repetitions, output token hashes are **100% bit-identical** (`same_prompt_identity = PASS`).
- **Estonia 136,584-Token 6-Hop Needle Retrieval:** **`5/5 (100.0% PASS)`** across two independent campaigns (`10/10` total) at `136,576` prompt tokens (`136,584` active KV tokens), resolving the 6-hop chain past the planted `Latvia` decoy in `25.29 s` TTFT (`5,400.4 tok/s` prefill, `48.6 tok/s` server decode).
- **AIME26 Mathematical Reasoning:** **`98.75%` pass@1 (`237/240`, SEM `0.61pp`)**, **`100%` majority@8**, and **`100%` pass@8** at `max_tokens = 130,000, temperature = 1.0`.

---

## 2. `Qwen3.8-27B-NVFP4` + `DFlash` Block-Diffusion Drafter (`gamma=15`)

### 2.1 Architecture (`qwen3_5_text` Dense 27B Hybrid)

| Parameter | Value | Notes |
|---|---|---|
| **Layers** | `64` (`48` Gated DeltaNet + `16` GQA full attention) | `3:1` hybrid pattern |
| **Hidden dimension** | `5,120` | Dense FFN (`intermediate_size = 17,408`) |
| **Target checkpoint** | `nvidia/Qwen3.8-27B-NVFP4` (historical 2026-08 archive runs used `unsloth/Qwen3.8-27B-NVFP4`, `21.8 GiB`) | Compressed-tensors NVFP4 (`E2M1` + FP8-E4M3 islands) |
| **Speculative drafters** | `z-lab/Qwen3.8-27B-DFlash2` (`5`-layer, `block_size = 8`) · legacy `z-lab/Qwen3.6-27B-DFlash` (`8`-layer, `block_size = 16`) | Bidirectional block-diffusion drafters conditioned on 5 target hidden layers (`[1, 16, 31, 46, 61]`); `block_size = 16` proposes `gamma = 15` tokens in one pass (`block_size = 8` proposes `gamma = 7`) |

### 2.2 Performance & Numerical Reference (`GB10`)

- **Throughput:** **`63.91 – 64.05 tok/s` median** (`~5.0×` over the `12.90 tok/s` no-speculation floor) on the 400-token deterministic MinHeap coding probe (`5/5` bit-identical completion SHA-256 `f3b4b2cedd789f2c7f339d7df741957254e6914fa3bf3ce62fc387a502612b9a`).
- **Verify Roofline:** A single `K=16` (`1` committed + `15` draft) verify step sweeps `16.33 GB` of weights and state (`verify(k) = 75.53 + 1.890·k ms` vs `70.37 ms` LPDDR5X bandwidth floor at `232 GB/s`). On the 400-token coding probe, mean emission is **`8.33` tokens/cycle** (`7.33 / 15` drafts + `1` bonus) in **`129.8 ms`** (`8.33 / 0.1298 = 64.18 tok/s`); on the 800-token coding probe, mean emission is **`6.81` tokens/cycle** (`5.81 / 15` drafts + `1` bonus) in **`131.25 ms`** (`6.81 / 0.13125 = 51.89 tok/s`).
- **Launcher vs binary flags:** `--no-speculative`, `--ple-ram`, `--ple-ram-dir`, and `--gamma` are `flux` container/launcher flags; the raw `flux-server` binary uses `--speculative` / `--dflash` / `--num-drafts` and `ATLAS_DISABLE_SPECULATION=1` (see [`FLAGS.md`](FLAGS.md)).
