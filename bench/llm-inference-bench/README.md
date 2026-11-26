<p align="left"><a href="../../README.md"><b>← Flux Engine</b></a> · <a href="../README.md"><b>Benchmarks (<code>bench/</code>)</b></a> · <b><code>llm-inference-bench</code></b></p>

# Running `local-inference-lab/llm-inference-bench` Against Flux Engine

First-class integration and measured reference receipts for running [**`local-inference-lab/llm-inference-bench`**](https://github.com/local-inference-lab/llm-inference-bench) (`v0.7.9`) against Flux Engine on **NVIDIA DGX Spark (GB10 / `sm_121a`)** or any discrete Blackwell/Hopper GPU.

---

## 1. One-Command Turnkey Runner (`run-lil-bench.sh`)

[`run-lil-bench.sh`](run-lil-bench.sh) automatically locates or clones `local-inference-lab/llm-inference-bench`, checks `/health` and `/v1/models` on the running Flux Engine server, auto-detects `--port` and `--model`, validates `--max-seq-len` requirements, and runs any built-in `llm-inference-bench` profile or custom flag combination:

```bash
# 1. Full 0..128K prefill + C=1 sustained decode matrix (30s per cell)
./bench/llm-inference-bench/run-lil-bench.sh matrix

# 2. Standalone cold-prefill sweep across 8K, 16K, 32K, 64K, and 128K
./bench/llm-inference-bench/run-lil-bench.sh prefill

# 3. 136,576-token 6-hop Estonia needle-in-a-haystack benchmark (5 runs, C=1)
./bench/llm-inference-bench/run-lil-bench.sh estonia

# 4. C=1 burst coding decode benchmark
./bench/llm-inference-bench/run-lil-bench.sh coding

# 5. Accuracy & stability suites (GSM8K, MMLU-Pro, GPQA Diamond, Needle-Checksum, Tool-Eval)
./bench/llm-inference-bench/run-lil-bench.sh gsm8k
./bench/llm-inference-bench/run-lil-bench.sh mmlu-pro
./bench/llm-inference-bench/run-lil-bench.sh gpqa
./bench/llm-inference-bench/run-lil-bench.sh repetition
./bench/llm-inference-bench/run-lil-bench.sh tool-eval

# 6. Or pass any raw llm_decode_bench.py flags directly:
./bench/llm-inference-bench/run-lil-bench.sh \
  --contexts 0,16384,65536 --concurrency 1 --duration 30 --display-mode screen
```

---

## 2. Measured Reference Results (`1× NVIDIA GB10`, `Qwen3.8-Flash-Next-NVFP4`)

### 2.1 `--test-profile estonia` (`136,576` Prompt Tokens, 6-Hop Chain)

Full report: [`results/LIL-ESTONIA-136K-v9.md`](results/LIL-ESTONIA-136K-v9.md) · Raw JSON: [`results/lil-estonia-136k-v9.json`](results/lil-estonia-136k-v9.json)

| Metric | `v9` Baseline Receipt | `v0.9.12-6k-stable` (`flux-estonia-136k.env`) |
|---|---:|---:|
| **Prompt Size** | `707,468` chars / `136,576` tokens | `707,468` chars / `136,576` tokens (`136,584` active KV) |
| **Prefill Scout (`max_tokens=1`)** | **`5,239.6 tok/s`** (`26.07 s` TTFT) | **`5,400.4 tok/s`** median (`5,428.3` peak, `25.29 s` TTFT) |
| **Accuracy (`5` resampled runs, `C=1`)** | **`5/5 (100.0% PASS)`** (`Estonia`, `0` decoy) | **`5/5 (100.0% PASS)`** (`10/10` across both campaigns) |
| **Client Generation Throughput** | **`43.6 tok/s`** aggregate (`43.4` mean) | **`48.2 tok/s`** peak (`43.8` mean) |
| **Server `Done:` Decode Throughput** | **`43.3 tok/s`** median (`2.85` accept len) | **`48.6 tok/s`** median (`2.88` accept len) |
| **Free Unified Memory Floor (`MemAvailable`)** | **`8,847 MiB`** (`0` watchdog kills) | **`8,431 MiB`** (`0` watchdog kills) |

---

### 2.2 `0..128K` Context & Sustained Decode Matrix (`v9-w4096` vs `v9-w16384`)

Full reports: [`results/LIL-DECODE-BENCH-v9-w4096.md`](results/LIL-DECODE-BENCH-v9-w4096.md) · [`results/LIL-DECODE-BENCH-v9-w16384.md`](results/LIL-DECODE-BENCH-v9-w16384.md) · Raw JSON: [`results/lil-decode-bench-v9-w16384.json`](results/lil-decode-bench-v9-w16384.json)

| Context (`tokens`) | `v9-w16384` LIL Prefill (`tok/s`) | `v9-w16384` Server Decode (`tok/s`) | `v9-w4096` LIL Prefill (`tok/s`) | `v9-w4096` LIL Client Decode (`tok/s`) | `v9-w4096` Server Decode (`tok/s`) | `v9-w4096` Accept Len |
|---:|---:|---:|---:|---:|---:|---:|
| **`0`** | — | `42.2` | — | **`42.8`** | **`43.0`** | `2.34` |
| **`8,192`** | **`5,342`** | `42.6` | `4,641` | **`42.7`** | **`42.7`** | `2.26` |
| **`16,384`** | **`5,754`** | `40.8` | `4,676` | **`43.5`** | **`43.1`** | `2.41` |
| **`32,768`** | **`5,722`** | `39.2` | `4,663` | **`43.2`** | **`43.1`** | `2.31` |
| **`65,536`** | **`5,627`** | `40.3` | `4,634` | **`42.1`** | **`41.6`** | `2.33` |
| **`131,072`** | **`5,508`** | `40.1` | `4,501` | **`40.7`** | **`41.9`** | `2.16` |
| **Coding Peak (`C=1`)** | — | **`52.1` med / `58.7` max** (`3.15` accept) | — | **`56.3` med / `57.4` max** | **`56.3` med / `57.4` max** | **`3.02`** |

---

## 3. Three Things to Know So `llm-inference-bench` Runs Without Issues

1. **Running `--test-profile estonia` (`136,576` prompt tokens) requires `--max-seq-len 147456` (`FLUX_MODE=long-context-136k`):**
   - Standard `128K` server launches set `--max-seq-len 135168`. Because `estonia` sends a `136,576`-token prompt plus `--max-tokens 5120` (`141,696` total tokens), a `135,168`-limit server will reject the request with an HTTP `400` context-length error.
   - Start the server with [`recipes/flux-estonia-136k.env`](../../recipes/flux-estonia-136k.env) (`FLUX_MODE=long-context-136k`), which raises `_RESERVE_GB=10` for 144K KV-cache headroom and sets `--max-seq-len 147456 --request-timeout 1800`.
2. **Understanding 30-second sustained decode windows (`re-prefill windows`):**
   - In LIL's default sustained-decode matrix (`--duration 30`), if a generated response hits natural `EOS` before the 30-second timer expires, LIL immediately starts a second request at that same context length (e.g., `65,536` tokens).
   - That second request's cold prefill (~11 s at 64K) executes inside the 30-second decode window, lowering LIL's client-side window `tok/s` while the server's `Done:` decode `tok/s` reflects pure token generation. To avoid mid-window re-prefills, either pass `--max-tokens 4096` or use `--request-count` burst mode.
3. **Do not pass `--reasoning-effort` against Qwen3.8 checkpoints:**
   - Qwen3.8's Jinja chat template explicitly raises a template exception if `reasoning_effort` is present in the request JSON. Leave `--reasoning-effort` unset (the default in `run-lil-bench.sh`) and toggle reasoning via `FLUX_THINKING=on|off` or `--default-chat-template-kwargs`.
