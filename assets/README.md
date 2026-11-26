<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Visual Telemetry &amp; SVG Architecture Assets (<code>assets/</code>)</b></p>

# Flux Engine Visual Telemetry & Architecture Assets (`assets/`)

Dark/light-compatible vector graphics (`SVG`) and rendered benchmark summary cards (`PNG`) used in **Flux Engine** documentation.

---

## Asset Index

| Asset File | Dimensions / Format | Purpose |
|---|---|---|
| [`logo.svg`](logo.svg) / [`flux-engine-hero.svg`](flux-engine-hero.svg) | `1160×248` SVG | **Flux Engine Hero Banner**: Blackwell `sm_121a` Tensor Core emblem, `v1.0.3 · RELEASE` badge, and hardware telemetry highlights. |
| [`social-preview.svg`](social-preview.svg) / [`.png`](social-preview.png) | `1280×640` SVG / PNG | **GitHub Social Preview Card (`2:1`, `200 KB`)**: OpenGraph preview banner featuring Flux Engine `v1.0.3`, GB10 + Discrete SM120 (`600 W`) certified telemetry, `1M` YaRN4 context, and SLSA Level 3 signing highlights. |
| [`prefill-curve.svg`](prefill-curve.svg) | `1160×540` SVG | **Single-GB10 Prefill Throughput Curve (`4K` → `136.5K`)**: Max-Prefill (`w16384`, `6,082.2 tok/s` peak) vs `w4096` interactive (`5,276.3 tok/s`) vs `K=1` reference vs untuned baseline. |
| [`decode-comparison.svg`](decode-comparison.svg) | `1160×500` SVG | **Single-Stream GB10 Decode Roofline Comparison**: `Qwen3.8-27B DFlash γ=15` (`63.9 tok/s`) and `Qwen3.8-Flash-Next K=5 MTP` (`56.5 tok/s` med / `57.4` peak) vs `273 GB/s` LPDDR5x ceiling. |
| [`sm121a-architecture.svg`](sm121a-architecture.svg) | `1280×480` SVG | **SM121a Layer Execution Pipeline & 8K Stage Census**: 7-stage execution flow (`1,348 ms` per 8K chunk) across `NVMe PLE Ring4` (`[128, 2560]`), `4-stream mHC` (`[T, 4, 2560]`), `36× GDN / 12× QSA`, `Router + v8 Planner`, `W4A8 MoE`, and `MTP K=5 Verify`. |
| [`flux-v0915-prefill-plain-k5-card.svg`](flux-v0915-prefill-plain-k5-card.svg) / [`.png`](flux-v0915-prefill-plain-k5-card.png) | `1240×770` SVG / PNG | **Prefill Throughput Card (`8K` → `262K`)**: Plain `K=1` and Native-MTP `K=5` across `W16384` (`6,420.3 tok/s` 16K chunk peak) and `W4096` profiles. |
| [`flux-v0915-decode-plain-k5-card.svg`](flux-v0915-decode-plain-k5-card.svg) / [`.png`](flux-v0915-decode-plain-k5-card.png) | `1240×810` SVG / PNG | **Decode Throughput Card**: Single-stream short/long-context (`8K` → `262K`, incl. Estonia `136.5K` `5/5 PASS`) and multi-stream `C=1..8` concurrent decode. |
| [`flux-v0915-k5-card.svg`](flux-v0915-k5-card.svg) / [`.png`](flux-v0915-k5-card.png) | `1240×820` SVG / PNG | **Native-MTP `K=5` Qualification Summary Card**: Prefill, lossless `0-ULP` decode parity, and `C=1..8` batched concurrency verification. |
| [`flux-v0915-plain-card.svg`](flux-v0915-plain-card.svg) / [`.png`](flux-v0915-plain-card.png) | `1240×790` SVG / PNG | **Plain `K=1` Reference Summary Card**: Non-speculative baseline throughput (`8K` → `262K`), GSM8K (`97.27%`), and needle-in-a-haystack verification. |
| [`favicon.svg`](favicon.svg) | `512×512` SVG | Geometric Blackwell `sm_121a` vector icon. |
