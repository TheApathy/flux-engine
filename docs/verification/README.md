<p align="left"><a href="../../README.md"><b>← Flux Engine</b></a> · <a href="../README.md"><b>Documentation (<code>docs/</code>)</b></a> · <b>Benchmark Methodology &amp; Verification (<code>docs/verification/</code>)</b></p>

# Benchmark Methodology & Verification (`docs/verification/`)

Public verification documentation covering byte-traffic roofline calculations, prefix-cache controls, speculative-decode token accounting, and quality-gate proofs backing every performance and accuracy metric in **Flux Engine**.

---

## Verification Index

| Document | Scope & Questions Answered |
|---|---|
| [`BENCHMARK_VERIFICATION.md`](BENCHMARK_VERIFICATION.md) | **Roofline Physics & Verification Analysis**:<br/>• Why `6,082.2 tok/s` prefill and `56.5 / 63.9 tok/s` decode obey GB10's `273 GB/s` LPDDR5x bandwidth ceiling.<br/>• Proof that prefix caching (`--enable-prefix-caching false`) and SSM caching (`--ssm-cache-slots 0`) are disabled during cold-prefill benchmarks.<br/>• `2,560 / 2,560` greedy bit-exact token identity proof (`K=5` vs `K=1`) and `5/5` (`100%`) Estonia `136.5K` 6-hop needle proof.<br/>• Transparent disclosure of the quarantined `72.25 tok/s` 13-arg/11-param PTX ABI bug. |

For per-stage kernel latency profiles and reproduction workflows, see [`docs/KERNEL-PROFILE.md`](../KERNEL-PROFILE.md), [`REPRODUCING.md`](../../REPRODUCING.md), and [`repro-kit/METHODOLOGY.md`](../../repro-kit/METHODOLOGY.md).
