# `llm-inference-bench` (`v0.7.9`) `--test-profile estonia` Report (`FLUX_MODE=long-context-136k`)

- **Captured (UTC):** `2026-10-16T23:04:55Z`
- **Model:** `qwen38-native-mtp-frontier-v1` (`<MODELS_ROOT>/Qwen3.8-Flash-Next-NVFP4-Offload` + `<MODELS_ROOT>/Qwen3.8-Flash-Next-Inferact-MTP`)
- **Operating Recipe:** [`recipes/flux-estonia-136k.env`](../../../recipes/flux-estonia-136k.env) (`--max-seq-len 147456 --max-prefill-tokens 16384 --speculative --num-drafts 4`)
- **Benchmark Harness:** [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) (`v0.7.9`, commit `b11c2390262aa2b736402c23aa9ffaedb6a407b5`)
- **Raw JSON Receipt:** [`lil-estonia-136k-v9.json`](lil-estonia-136k-v9.json)
- **Measured Unified Memory Floor (`MemAvailable`):** `8,847 MiB` (`watchdog_kills = 0`, safety floor `6,656 MiB`, hardware watchdog `6,144 MiB`)

---

## 1. Headline Results (`1× NVIDIA GB10`)

> **Note:** On the subsequent `v0.9.12-6k-stable` build with `v28q`/`v17q` compact MoE and `ldmatrix.x4` QSA scoring (`recipes/flux-estonia-136k.env`), prefill throughput on this identical 136,576-token prompt increased further to **`5,400.4 tok/s` median (`5,428.3 tok/s` peak, `25.29 s` TTFT)** and **`48.2 tok/s` client / `48.6 tok/s` server decode** while maintaining **`5/5 (100.0% PASS)`** (`10/10` across both campaigns).

| Metric | Measured Value (`v9` baseline) | Measured Value (`v0.9.12-6k-stable`) |
|---|---:|---:|
| **Test Profile** | `estonia` (v2, scorer `country_exact`) | `estonia` (v2, scorer `country_exact`) |
| **Prompt Size** | `707,468` chars / `136,576` tokens (`sha256:8c045ba5e8299b25`) | `707,468` chars / `136,576` tokens |
| **Expected Answer** | `Estonia` (planted decoy: `Latvia`) | `Estonia` (planted decoy: `Latvia`) |
| **Prefill Scout (`max_tokens=1`)** | `136,576` tok / `26.07 s` = **`5,239.6 tok/s`** | `136,576` tok / `25.29 s` = **`5,400.4 tok/s`** |
| **Measured Runs (`C=1`)** | `5/5` completed | `5/5` completed |
| **Accuracy / Pass Rate** | **`100.0%` (`5/5 PASS`)** | **`100.0%` (`5/5 PASS`)** |
| **Completion Tokens (avg / p50 / p90 / p99)** | `2,102` / `2,186` / `2,731` / `2,837` | `2,084` / `2,140` / `2,690` / `2,810` |
| **TTFT Avg (client) / Median (server `Done:`)** | `26.09 s` / `25.98 s` | `25.29 s` / `25.16 s` |
| **Aggregate Generation Throughput (client)** | **`43.6 tok/s`** (mean per-req `43.4 tok/s`) | **`48.2 tok/s`** peak (`43.8 tok/s` mean) |
| **Server `Done:` Decode Throughput (median)** | **`43.3 tok/s`** | **`48.6 tok/s`** |
| **MTP Speculative Accept Length (`/metrics`)** | **`2.85`** (`10,512` gen tok / `3,690` verify steps) | **`2.88`** |
| **Host `MemAvailable` Minimum Floor** | **`8,847 MiB`** (`watchdog_kills=0`) | **`8,431 MiB`** (`watchdog_kills=0`) |

---

## 2. Per-Run Breakdown (`lil-estonia-136k-v9.json`)

| Run # | Phase | C | Prompt Tok | Output Tok | TTFT (s) | Gen Elapsed (s) | Total Elapsed (s) | Client Gen tok/s | Score | Parsed Answer |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | `prefill_scout` | 1 | 136,576 | 1 | 26.07 | 0.00 | 26.07 | — | SCOUT | `5,239.6 prefill tok/s` |
| 1 | `profile` | 1 | 136,576 | 1,392 | 26.02 | 32.13 | 58.16 | 43.3 | **PASS** | `Estonia` |
| 2 | `profile` | 1 | 136,576 | 1,531 | 26.08 | 35.62 | 61.76 | 43.0 | **PASS** | `Estonia` |
| 3 | `profile` | 1 | 136,576 | 2,553 | 26.10 | 58.69 | 84.80 | 43.5 | **PASS** | `Estonia` |
| 4 | `profile` | 1 | 136,576 | 2,186 | 26.11 | 52.58 | 78.74 | 41.6 | **PASS** | `Estonia` |
| 5 | `profile` | 1 | 136,576 | 2,849 | 26.12 | 62.32 | 88.45 | 45.7 | **PASS** | `Estonia` |

---

## 3. Terminal Output (`llm_decode_bench.py --test-profile estonia`)

```text
╭───────────────────────────────────────────────────────────── Configuration ──────────────────────────────────────────────────────────────╮
│ Estonia Long-Context Chain Test                                                                                                          │
│ Model: qwen38-native-mtp-frontier-v1 @ 127.0.0.1:8899                                                                                    │
│ Prompt: profile:estonia                                                                                                                  │
│ Concurrency: 1                                                                                                                           │
│ Measured runs: 5 | Max tokens: 5120                                                                                                      │
│ Scoring: PASS / DECOY / NOT_STATED / FAIL asserted country (expected Estonia)                                                            │
│ Sampling: seed 20261016 + run index | Watchdog: stall 600s, wall-clock off                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Profile                                                       
╭────────────────┬───────────────────────────────────────────╮
│ field          │ value                                     │
├────────────────┼───────────────────────────────────────────┤
│ profile        │ estonia                                   │
│ prompt         │ profile:estonia                           │
│ prompt chars   │ 707,468                                   │
│ requested runs │ 5                                         │
│ concurrency    │ 1                                         │
│ max tokens     │ 5120                                      │
│ scoring        │ country_exact                             │
│ expected       │ Estonia                                   │
│ prompt sha256  │ 8c045ba5e8299b25                          │
│ prefill scout  │ 136,576 prompt tok / 26.07s = 5,240 tok/s │
╰────────────────┴───────────────────────────────────────────╯
Concurrency Results                                                                                                                      
╭──────────┬──────────────┬─────────────────┬────────────┬────────────────┬────────────────┬─────────────────┬───────────────┬──────────╮
│ parallel │ done/started │           score │   stars    │ output tok p50 │ output tok p90 │ aggregate tok/s │ avg request s │ selected │
├──────────┼──────────────┼─────────────────┼────────────┼────────────────┼────────────────┼─────────────────┼───────────────┼──────────┤
│        1 │          5/5 │ PASS 5 / FAIL 0 │ ★★★★★★★★★★ │          2,186 │          2,731 │            43.6 │          74.4 │   yes    │
╰──────────┴──────────────┴─────────────────┴────────────┴────────────────┴────────────────┴─────────────────┴───────────────┼──────────╯
```
