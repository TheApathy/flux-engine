# Kernel Profile — `sm_121a` Prefill & Speculative Verify Census

Per-stage CUDA kernel timing and register/spill characteristics on **1× NVIDIA GB10 (`sm_121a`, 48 SMs, 32 MiB L2, 128 GB LPDDR5X)**, compiled with `nvcc -arch=sm_121a --cubin -Xptxas -v -fmad=false`.

Every kernel below compiles with **`0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`**.

---

## 1. `Qwen3.8-Flash-Next-NVFP4` Prefill Stage Census (`1,348.0 ms / 8K` at 16K Context)

At `16,384` prompt tokens (`FLUX_MODE=max-prefill`, `WIDE=16384`, `concurrency=1`), a single request completes end-to-end prefill in **`2.694 – 2.702 s`** (**`6,076.7 tok/s` median, `6,082.2 tok/s` peak**), corresponding to **`1,348.0 ms` per 8,192 tokens** across all 48 layers:

| Subsystem Stage (48 Layers) | Time / 8K (`ms`) | Share (`%`) | Primary `sm_121a` Kernels | Register / Occupancy / Hardware Path |
|---|---:|---:|---|---|
| **1. Routed + Shared MoE (`512` routed top-10 + `1` shared)** | `684.5 ms` | `50.8%` | `qwen4_moe_w4a8_v28q`<br/>`qwen4_moe_compact_t_gemm_k32_v17q`<br/>`qwen4_mxfp8_quant_rows_v16q4_gate` | Block-scaled `mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32.Kind::f8f6f4.block_scale.scale_vec::2X`, 16-warp B-weight transcoding, zero-`sE` register-quad shuffle MXFP8 epilogue, `v17q` down-proj at `256` threads = **`2 CTAs/SM` (`122` regs, `0` spills)**, inter-expert planner (`MOE_PLAN_ORDER=inter`). |
| **2. Gated DeltaNet Linear Attention (`36` layers)** | `312.8 ms` | `23.2%` | `atlas_gdn_fi_sm121a_prefill_f16`<br/>`qwen4_conv1d_l2norm_f16_prefill_v3_commit`<br/>`qwen4_fp8_act_and_ba_gates_mma`<br/>`qwen4_gdn_norm_fp8` (`v2`) | FlashInfer CuTe-DSL WY chunkwise recurrence, shared-memory filter-staged causal conv1d + L2 norm (`TOKENS=64`), 64-row tile fused FP8 activation + `beta/alpha` gate MMA (`72` regs, `0` spills), static-scale E4M3 output norm. |
| **3. Query-Sparse Attention (`12` layers)** | `198.4 ms` | `14.7%` | `qwen4_qsa_prefill_score` / `_gmax`<br/>`qwen4_qsa_prefill_topk_v2` / `_fused_c3_v2`<br/>`qwen4_qsa_prefill_attn_fp8_qkpv_amax_pipe` | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` block scoring at **`5 CTAs/SM` (`96` regs, `0` spills)**, bimodal top-`k` block selection (`<= 8,192` vs `> 8,192` blocks), `cp.async` double-buffered FP8 `E4M3` `Q·K` and `P·V` pipeline with 16-token half-tile `do_h0/do_h1` skipping and fused `o_proj` absmax fold. |
| **4. 4-Stream mHC Hyper-Connections (`48` layers)** | `94.2 ms` | `7.0%` | `qwen4_hc_prep_up`<br/>`qwen4_hc_upmix_roof_t128_amax`<br/>`qwen4_hc_combine_moe_gemm_mx_gatepre` | `PIN_M=4096` cuBLASLt heuristic admission for `M=16,384`, split `prep_up` for 100% L2 residency of `s.up` (`32 MiB` L2), warp-per-row period-6 unrolled `combine_moe_gemm_mx_gatepre` (`128` regs, `0` spills, `0` `token_to_perm` D2D copies). |
| **5. 4-Ring NVFP4 PLE Offload & Host/Stream Glue** | `58.1 ms` | `4.3%` | `qwen4_ple_prefill_fused`<br/>`PleRowsPlan::Pipelined` + `PLE_CHUNK_CACHE` | 4-ring sector-aligned `O_DIRECT` NVMe DMA pinned to Cortex-X925 cores (`ATLAS_PLE_READ_RINGS=4`, `ATLAS_PLE_READ_AFFINITY=big`), pre-deinterleaved `Arc<PackedPleBlock>` RAM cache (`0` `munmap` TLB shootdowns). |
| **Total per 8,192 Tokens (`w16384`)** | **`1,348.0 ms`** | **`100.0%`** | **All 16 runtime engagement receipts active** | **`6,076.7 tok/s` median / `6,082.2 tok/s` peak (`C=1`)** |

---

## 2. `Qwen3.8-Flash-Next-NVFP4` Lossless `K=5` Speculative Verify Profile (`57.4 tok/s` Peak)

During decode (`--speculative --num-drafts 4`), the 1-layer Inferact MTP proposer chains 4 draft steps on device (`ATLAS_QWEN4_MTP_CHAIN=1`, `ATLAS_QWEN4_MTP_CONF=0.6`), and the 48-layer target model verifies all `K=5` rows in a single batched pass:

| Verify Subsystem (`K=5` Rows) | Kernel / Mechanism | Why It Matters on GB10 (`sm_121a`) |
|---|---|---|
| **5-Row Routed + Shared MoE** | `ATLAS_QWEN4_K5_NATIVE_MOE=1`, `ATLAS_MOE_T_LANES_DOWN_T8=1` | Coalesces weight reads across all 5 candidate tokens (`1` committed + `4` drafts) so expert weights are streamed once per verify cycle from LPDDR5X. |
| **5-Row Gated DeltaNet (`36` layers)** | `ATLAS_QWEN4_K5_BATCH_GDN=1`, `ATLAS_QWEN4_K5_BATCH_CONV=1`, `ATLAS_QWEN4_GDN_COMMIT_NO_MIRROR=1` | Advances all 5 candidate recurrent states in one launch and commits the accepted prefix slot directly without redundant host/device state mirrors (`pool_endpoint = seq_len + rows - 1`). |
| **QSA Verify (`12` layers)** | `ATLAS_QWEN4_QSA_VERIFY_ROWS=1` (`partial_rows_v2` + `reduce_rows`) | The specialized five-row kernel is admitted only when the complete verify frame ends at or before token `2,048`. Frames crossing or above that endpoint retain native K=5 but evaluate QSA row by row. Historical `LOSSLESS 10/10` results through 64K are model-level token-identity evidence and must not be attributed to the specialized multi-row kernel. |
| **5-Row mHC Stream Mixing** | `ATLAS_QWEN4_K5_BATCH_HYPER=1`, `ATLAS_QWEN4_HC_DUAL_ROWS=1` | Fuses the 4-stream hyper-connection projections across all 5 verify rows, eliminating per-row kernel launch overhead. |
| **5-Row PLE Verify Fetch** | `ATLAS_QWEN4_PLE_VERIFY_BATCH=1` | Deduplicates and batches 3-gram PLE table reads across all 5 candidate tokens in one DMA transaction. |

---

## 3. `Qwen3.8-27B-NVFP4` DFlash `K=16` (`gamma=15`) Verify Profile (`64.0 tok/s` Median)

On the dense 27B hybrid model (`48` GDN + `16` GQA layers), a single `K=16` verify step sweeps **`16.33 GB`** of weights and state (`32 FLOP/byte` vs `238–1190` balance point: strictly memory-bandwidth-bound on LPDDR5X):

| Cycle Component (`gamma=15`, `K=16`) | Measured Time (`ms`) | Key `sm_121a` Kernels & Selectors |
|---|---:|---|
| **Target Verify (`48` GDN + `16` GQA + Dense FFN)** | `105.57 ms`<br/>(`75.53 + 1.890·k ms`) | `w4a16_gemm_t_m32_n64` (`ATLAS_FFN_TC=1`), `gated_delta_rule_wy17` (`ATLAS_SSM_GDN_LAZY=1`: skips 15/16 intermediate DRAM state writes for `+5.16 tok/s`, `ATLAS_SSM_GDN_SEQ_PERSISTENT=1` for `+1.70 tok/s`), `paged_decode_attn_kgamma_nvfp4`, `ATLAS_ATTN_QKV_FUSED=1`. |
| **DFlash Drafter Propose (`8`-layer v1 B16 / `5`-layer DFlash2 B8)** | `22.16 ms` *(8L B16)* | Bidirectional block-diffusion pass conditioned on target hidden captures `[1, 16, 31, 46, 61]` with `ATLAS_DFLASH_DRAFT_SPLITK=8` (`+1.1 tok/s`). |
| **Host Accept / Tree Commit / Glue** | `3.52 ms` | Greedy exact-match verify walk + lazy GDN replay from root state on partial accept. Matched cycle math (`emitted / cycle_s`):<br/>• **800-token probe:** `5.81` mean accepted drafts + `1` bonus = **`6.81` tokens / `131.25 ms` cycle = `51.89 tok/s`**.<br/>• **400-token coding probe (`gamma=15`, BF16 KV):** `7.33` mean accepted drafts + `1` bonus = **`8.33` tokens / `129.8 ms` cycle = `64.18 tok/s`** (`63.9–64.0 tok/s` median across repeated runs). |
