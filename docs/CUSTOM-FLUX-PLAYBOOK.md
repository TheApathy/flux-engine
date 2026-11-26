# Custom-Flux Playbook — the GB10 `sm_121a` delta over upstream Atlas

This playbook summarizes the architectural anatomy of `TheApathy/flux-engine`
over upstream Atlas (`MonumentalSystems/atlas` and `Avarok-Cybersecurity/atlas`
via `TheApathy/apathy-atlas`) on NVIDIA Grace Blackwell (`GB10`, `sm_121a`,
128 GB unified LPDDR5x @ 273 GB/s). Every kernel, runtime flag, and recipe
described here is shipped in this repository.

---

## 1. Where the delta lives (by subsystem)

| Subsystem | Path | What changed |
|---|---|---|
| `Qwen3.8-Flash-Next` `sm_121a` kernels | `kernels/gb10/qwen3.8-flash-next/nvfp4/` | `cutlass_5222` (`v28q`/`v17q`) W4A8 compact MoE, `qwen4_qsa_prefill_select` (`ldmatrix.x4` score + bimodal `topk_v2`/`c3_v2` + `qkpv_amax_pipe`), `conv1d_v3_commit`, `fp8_act_and_ba_gates_mma`, 4-stream `mHC` (`t128` roof + `gatepre`), and fused NVFP4 PLE kernels. |
| Optional FlashInfer SM121a GDN & Attention bridge | `crates/spark-model/csrc/{fi_gdn,fi_attn}/` | Native CUDA GDN prefill by default, plus optional source-built CuTe-DSL chunked WY recurrence (`atlas_gdn_fi_sm121a_prefill_f16` / `gdn_umma_prefill`) and FP16/NF8 bridges (see `csrc/fi_gdn/NOTICE`). |
| Shared GB10 `sm_121a` kernels | `kernels/gb10/common/` | TurboQuant+ (`turbo2`/`3`/`4`/`8`, asymmetric `fp8k_turbo4v`/`bf16k_turbo4v`/`turbo4k_turbo3v`, `wht_bf16.cu`, `tq_plus_innerq.cu`), `WY17` lazy GDN recurrence, transposed split-K `w4a16_gemm`, and MLA prefill/decode. |
| Model & layer orchestration | `crates/spark-model/src/layers/{moe,qwen3_ssm,qwen3_attention,qwen4_hyper,dflash_head}/` | Chunk-adaptive wide prefill (`w16384` / `w4096`), `PIN_M=4096` cuBLASLt admission, `PackedPleBlock` 4-ring NVFP4 streaming, lossless `K=5` Inferact MTP verify, and `gamma=15` DFlash verify. |
| Scheduler & CLI surface | `crates/spark-server/src/` | `flux-server` / `flux-cli` / `spark` binaries, startup `sync_flux_env_aliases()` (`FLUX_*` -> `ATLAS_*`), and batched speculative verify scheduling. |
| Reproducibility & benchmarks | `bench/`, `recipes/`, `repro/`, `repro-kit/` | Self-verifying launchers (`serve.sh`), per-flag references (`FLAGS.md`), 136.5K Estonia & lossless decode harnesses, and OCI Reproducibility Kit v0.1 (`flux-bench`). |

---

## 2. The six tuning families

The `FLUX_*` / `ATLAS_*` optimization surface clusters into six orthogonal families:

1. **Compact Transposed W4A8 / W4A16 MoE & Split-K GEMMs:**
   - On `Qwen3.8-Flash-Next` (512 routed top-10 + 1 shared expert), MoE is 50.8%
     of prefill (`684.5 ms / 8K`). Routing `gate_up` through 16-warp B-transcoding
     `v28` and `down_proj` through 2-CTA/SM `v17q` with zero-`sE` register-quad
     MXFP8 epilogues, fused `v16q4_gate` slot quantization, and `MOE_PLAN_ORDER=inter`
     eliminates register spills, epilogue barriers, and tail-wave SM starvation.
   - On dense `Qwen3.8-27B`, routing verify projections (`ffn_down`, `ssm_out`,
     `ssm_qkvz`, `attn_qkv`) through lossless FP32 split-K NVFP4 kernels
     saturates all 48 SMs at small batch sizes (`M=16..17`).
2. **Gated DeltaNet (`GDN`) Linear Attention Fusions:**
   - Fusing 64-row FP8 input quantization with `beta/alpha` gate MMA
     (`qwen4_fp8_act_and_ba_gates_mma`), staging `TOKENS=64` causal `conv1d` +
     `L2` norm in shared memory (`conv1d_v3_commit`), dispatching the WY
     recurrence through `atlas_gdn_fi_sm121a_prefill_f16`, and fusing output
     `rmsnorm + mxfp8_quant` (`GDN_NORM_FP8_V2=1`) cuts 36-layer GDN prefill to
     `312.8 ms / 8K`.
   - During decode verification, `WY17` lazy state commit (`ATLAS_WY17_LAZY=1`)
     and GDN undo rollback journaling avoid redundant per-token state writes.
3. **Query-Sparse Attention (`QSA`) & Multi-Head Latent Attention (`MLA`):**
   - `ldmatrix.x4` block scoring at 5 CTAs/SM (`96` registers, `0` spills),
     bimodal `topk_v2` (`<= 8,192` blocks) / `topk_fused_c3_v2` (`> 8,192`
     blocks) selection, and `cp.async` double-buffered `qkpv_amax_pipe` with
     16-token half-tile skipping bring 12-layer QSA prefill down to `198.4 ms / 8K`.
4. **4-Stream Multi-Stream Hyper-Connections (`mHC`):**
   - Clamping `query_m` to `PIN_M=4096` for cuBLASLt admission (`-228 ms / 16K`),
     splitting `qwen4_hc_prep_up` ahead of `qwen4_hc_upmix_roof_t128` for 100% L2
     residency inside GB10's 32 MiB L2 cache, and unrolling period-6 warp-per-row
     `qwen4_hc_combine_moe_gemm_mx_gatepre` reduce 48-layer mHC overhead to
     `94.2 ms / 8K`.
5. **Sector-Aligned NVFP4 PLE Streaming & Unified Memory Reuse:**
   - Pre-deinterleaved `Arc<PackedPleBlock>` host caching with 4 asynchronous DMA
     rings pinned to the 10 Grace `Cortex-X925` cores (`ATLAS_PLE_READ_AFFINITY=big`)
     streams the 51B-parameter PLE table in `14.3 ms / 8K` with zero `munmap` TLB
     shootdowns.
   - Reusing idle `ssm_deinterleaved` scratch for `SHARED_W4A8_PRETRANSCODE`
     (`+243.4 MiB` saved) and enforcing `_RESERVE_GB=7` (`w16384`) / `_RESERVE_GB=10`
     (`136.5K`) keep `MemAvailable` safely above the `6,144 MiB` GB10 watchdog.
6. **Lossless `K=5` Inferact MTP & `gamma=15` DFlash Speculative Decode:**
   - Chaining all 4 Inferact MTP draft steps on device (`ATLAS_QWEN4_MTP_CHAIN=1`)
     and verifying all 5 rows in a single batched pass across GDN, QSA, mHC, and
     MoE reaches **56.5 tok/s** median (**57.4 tok/s** peak) while remaining
     **100% lossless (`2,560 / 2,560` greedy tokens bit-exact vs `K=1`)**.

---

## 3. How to reproduce and extend this tuning (the method)

1. **Start from a pinned recipe** in [`recipes/`](../recipes/) (`flux-v9-max-prefill.env`
   or `flux-v9-w4096-interactive.env`) and build with explicit target variables
   (`FLUX_TARGET_MODEL=qwen3.8-flash-next FLUX_TARGET_QUANT=nvfp4`).
2. **Verify zero register spills at compile time:** compile modified `.cu` files
   with `nvcc -arch=sm_121a --cubin -Xptxas -v -fmad=false` and reject any change
   that introduces a stack frame or spill loads/stores.
3. **Gate on 0-ULP kernel parity first:** compare new fused kernels against the
   `flux-exact-reference.env` oracle before running full-model benchmarks.
4. **Interleave A/B runs and enforce token-identity + memory-floor gates:**
   require `same_prompt_identity = PASS` across `8K..128K`, `LOSSLESS 10/10`
   (`2,560 / 2,560` greedy tokens exact), `5/5 PASS` on 136.5K Estonia, and
   `MemAvailable >= 6,656 MiB` before promoting any change to default recipes.
