<p align="left"><a href="../../README.md"><b>← Flux Engine</b></a> · <a href="../README.md"><b>Documentation (<code>docs/</code>)</b></a> · <b>Reproduction Playbooks (<code>docs/reproduction/</code>)</b></p>

# Model Reproduction Playbooks (`docs/reproduction/`)

Step-by-step operator reproduction guides for verifying deterministic outputs, single-stream prefill, and speculative decode throughput on NVIDIA GB10 (`sm_121a`).

---

## Playbooks

| Document | Model Target | Summary |
|---|---|---|
| [`qwen38.md`](qwen38.md) | `Qwen3.8-27B-NVFP4` & `Qwen3.8-Flash-Next-NVFP4` | Running `repro/qwen38-coding/run.py`, evidence sanitization policy, and `verify_no_leaks.py` checks on DGX Spark (`GB10`). |

For full host and container reproduction steps, see [`REPRODUCING.md`](../../REPRODUCING.md) and [`repro-kit/README.md`](../../repro-kit/README.md).
