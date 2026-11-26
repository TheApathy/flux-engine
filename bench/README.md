<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Benchmark &amp; Verification Harnesses (<code>bench/</code>)</b></p>

# Benchmark & Verification Harnesses (`bench/`)

Self-verifying benchmark launchers, flag references, and deterministic reproduction clients for Flux Engine on **NVIDIA DGX Spark (GB10 / `sm_121a`)**.

---

## Directory Map

| Harness | Target Model / Suite | Headline Measured Performance (`1× GB10`) | Start Here |
|---|---|---|---|
| **[`llm-inference-bench/`](llm-inference-bench/)** | Turnkey integration for [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) (`v0.7.9`) across all models | `5/5 (100% PASS)` Estonia `136,576`-token 6-hop (`5,400.4 tok/s` prefill, `48.6 tok/s` decode), `0..128K` context + decode matrices (`v9-w16384` & `v9-w4096`) | [`llm-inference-bench/README.md`](llm-inference-bench/README.md) · [`run-lil-bench.sh`](llm-inference-bench/run-lil-bench.sh) |
| **[`qwen38-flash-next-gb10/`](qwen38-flash-next-gb10/)** | `Qwen3.8-Flash-Next-NVFP4-Offload` + `Inferact-MTP` (`K=5`) | **`6,082.2 tok/s` peak** (`6,076.7` median) @ 16K prefill, **`5,694.4 tok/s`** @ 128K, **`57.4 tok/s` peak** (`56.3` median) `LOSSLESS 10/10` `K=5` decode | [`qwen38-flash-next-gb10/README.md`](qwen38-flash-next-gb10/README.md) · [`FLAGS.md`](qwen38-flash-next-gb10/FLAGS.md) |
| **[`qwen38-gb10/`](qwen38-gb10/)** | `Qwen3.8-27B-NVFP4` + `DFlash` block-diffusion drafter (`gamma=15`) | **`63.91 – 64.05 tok/s` median** (`4.8×` vs no-speculation) on 400-tok MinHeap probe (`58.01 tok/s` on public `incoai/Qwen3.5-27B-DFlash2`) | [`qwen38-gb10/README.md`](qwen38-gb10/README.md) · [`FLAGS.md`](qwen38-gb10/FLAGS.md) · [`repro/qwen38-coding/`](../repro/qwen38-coding/) |
| **[`qwen/`](qwen/)** | `Qwen3.6-27B-NVFP4` + `DFlash` (`gamma=16 -> K=17`) | Dense 27B `K=17` WY-chunked GDN speculative decode, drafter A/B comparisons, and language evaluation matrix | [`qwen/README.md`](qwen/README.md) |
| **[`laguna/`](laguna/)** | `poolside/Laguna-S-2.1-NVFP4` *(archived historical harness)* | Historical Laguna-S-2.1 DFlash speculative decode, cuBLASLt prefill, and 69-case tool-calling eval (`laguna-s-2.1` target is not compiled in this tree) | [`laguna/README.md`](laguna/README.md) |
| **[`deepseek-v4/`](deepseek-v4/)** | `DeepSeek-V4-Flash` / `DeepSeek-V4.1` (`dsv41`) | Absorbed-latent MLA (`HDIM=320`), Engram memory, and `MROW=2` deduplicated MoE verify | [`deepseek-v4/README.md`](deepseek-v4/README.md) |

---

## Quick Commands

```bash
# 1. Run local-inference-lab/llm-inference-bench (matrix, prefill, estonia, coding, gsm8k, mmlu-pro, gpqa)
./bench/llm-inference-bench/run-lil-bench.sh matrix
./bench/llm-inference-bench/run-lil-bench.sh estonia

# 2. Launch Qwen3.8-Flash-Next (w16384 max-prefill + lossless K=5 MTP decode)
MODEL_DIR=/path/to/models/Qwen3.8-Flash-Next-NVFP4 ./bench/qwen38-flash-next-gb10/serve.sh

# 3. Launch Qwen3.8-27B + DFlash v2 (gamma=15, 64.0 tok/s decode)
MODEL_DIR=/path/to/models/Qwen3.8-27B-NVFP4 \
DRAFT=/path/to/models/Qwen3.8-27B-DFlash-v2 \
  ./bench/qwen38-gb10/serve.sh
```
