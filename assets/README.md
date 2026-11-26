<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Visual Telemetry &amp; SVG Architecture Assets (<code>assets/</code>)</b></p>

# Flux Engine Visual Telemetry & Architecture Assets (`assets/`)

Dark/light-compatible vector graphics (`SVG`) and rendered benchmark summary cards (`PNG`) used in **Flux Engine** documentation.

---

## Asset Index

| Asset File | Dimensions / Format | Purpose |
|---|---|---|
| [`logo.svg`](logo.svg) / [`flux-engine-hero.svg`](flux-engine-hero.svg) | `1160×248` SVG | **Flux Engine Hero Banner**: Blackwell `sm_121a` Tensor Core emblem, `v1.0.3 · RELEASE` badge, and hardware telemetry highlights. |
| [`social-preview.svg`](social-preview.svg) / [`.png`](social-preview.png) | `1280×640` SVG / PNG | **GitHub Social Preview Card (`2:1`)**: OpenGraph preview banner featuring Flux Engine, Blackwell `SM121a`/`SM120a` native kernels, Pure Rust runtime, DFlash2/MTP speculation, drop-in APIs, and SLSA Level 3 signing highlights. |
| [`prefill-curve.svg`](prefill-curve.svg) | `1160×540` SVG | **Single-GB10 Prefill Throughput Curve (`8K` → `255K`, `v1.0.3`)**: Max-Prefill (`w16384`, `6,501 tok/s` @ 16K peak, `5,937` @ 128K, `5,525` @ 255K) vs `w4096` interactive (`5,448 → 4,991 tok/s`) vs `K=1` reference vs untuned baseline. |
| [`decode-comparison.svg`](decode-comparison.svg) | `1160×500` SVG | **Single-Stream GB10 Decode Roofline Comparison (`v1.0.3`)**: `Qwen3.8-Flash-Next` `MTP K=5` (`91.3 tok/s`) & `DFlash2 W=8` (`89.1 tok/s`) and `Qwen3.8-27B` `DFlash γ=15` (`72.4` FP8-KV / `64.0` BF16-KV) & `DFlash2 γ=7` (`58.0 tok/s`) vs `273 GB/s` LPDDR5x ceiling. |
| [`gb10-benchmarks.svg`](gb10-benchmarks.svg) | `1160×552` SVG | **NVIDIA GB10 / DGX Spark Benchmark Summary (`sm_121a`, `128 GB` UMA, `v1.0.3`)**: Cold & warm prefill (`12,056` warm 8K, `6,501` 16K peak, `5,525` 255K, `5,400` 136.5K Estonia `5/5 PASS`, `1,773`/`1,636` 27B) and lossless speculative decode/concurrency scaling (`91.3` `K=5` / `113.4` `DFlash2` `C=1` → `580.7` `C=8` Flash-Next; `81.1` `C=1` → `172.6` `C=8` 27B). |
| [`sm120-benchmarks.svg`](sm120-benchmarks.svg) | `1160×552` SVG | **Discrete Blackwell SM120 Benchmark Summary (`1× RTX PRO 6000 WS`, `600 W`, `96 GB GDDR7`, `v1.0.3`)**: Cold prefill (`30,869` W4A8 / `34,308` W4A4 @ 16K, `16,248` 27B @ 8K, `1M` YaRN4 `3/3 PASS` in `82.1 s`) and decode/concurrency scaling (`678` `C=1` → `3,406` `C=64` Flash-Next; `497` `C=1` → `1,975` `C=64` 27B at `97.3%` GDDR7 roofline). |
| [`sm121a-architecture.svg`](sm121a-architecture.svg) | `1280×480` SVG | **SM121a Layer Execution Pipeline & 8K Stage Census (`v1.0.3`)**: 7-stage zero-spill execution flow across `NVMe PLE Ring4`, `4-stream mHC`, `36× GDN / 12× QSA`, `Router + v8 Planner`, `W4A8 MoE`, and `MTP K=5 Verify`. |
| [`flux-v0915-prefill-plain-k5-card.svg`](flux-v0915-prefill-plain-k5-card.svg) / [`.png`](flux-v0915-prefill-plain-k5-card.png) | `1240×770` SVG / PNG | **Prefill Throughput Card (`8K` → `262K`)**: Plain `K=1` and Native-MTP `K=5` across `W16384` (`6,420.3 tok/s` 16K chunk peak) and `W4096` profiles. |
| [`flux-v0915-decode-plain-k5-card.svg`](flux-v0915-decode-plain-k5-card.svg) / [`.png`](flux-v0915-decode-plain-k5-card.png) | `1240×810` SVG / PNG | **Decode Throughput Card**: Single-stream short/long-context (`8K` → `262K`, incl. Estonia `136.5K` `5/5 PASS`) and multi-stream `C=1..8` concurrent decode. |
| [`flux-v0915-k5-card.svg`](flux-v0915-k5-card.svg) / [`.png`](flux-v0915-k5-card.png) | `1240×820` SVG / PNG | **Native-MTP `K=5` Qualification Summary Card**: Prefill, lossless `0-ULP` decode parity, and `C=1..8` batched concurrency verification. |
| [`flux-v0915-plain-card.svg`](flux-v0915-plain-card.svg) / [`.png`](flux-v0915-plain-card.png) | `1240×790` SVG / PNG | **Plain `K=1` Reference Summary Card**: Non-speculative baseline throughput (`8K` → `262K`), GSM8K (`97.27%`), and needle-in-a-haystack verification. |
| [`favicon.svg`](favicon.svg) | `512×512` SVG | Geometric Blackwell `sm_121a` vector icon. |
