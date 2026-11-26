# Reproducing the measurements in this repository

`TheApathy/flux-engine` is a standalone native Rust and `sm_121a` / `sm_120a` CUDA/PTX
inference repository developed from a fork of
[Atlas](https://github.com/Avarok-Cybersecurity/atlas) and
[MonumentalSystems/atlas](https://github.com/MonumentalSystems/atlas) via
[TheApathy/apathy-atlas](https://github.com/TheApathy/apathy-atlas). Where
`apathy-atlas` kept measured model stacks on separate branches (`perf/qwen38-gb10-dflash`,
`qwen`, `laguna`, `ds4-flash`), `flux-engine` consolidates the `sm_121a` / `sm_120a` kernel
targets, self-verifying launchers, operating recipes, and OCI reproducibility
kit onto `main`.

| Harness / Directory | Model Target | What it measures |
|---|---|---|
| [`bench/qwen38-flash-next-gb10/`](../bench/qwen38-flash-next-gb10/) | `Qwen/Qwen3.8-Flash-Next` (`Hotshot2000/Qwen3.8-Flash-Next-NVFP4` + repacked NVFP4 PLE + Inferact MTP) | Single-stream 16K-chunk prefill (`v1.0.3` headline `6,501 tok/s` @ 32K, `v0.9.12` `w16384` certified baseline `5,830.2` @ 8K, `6,082.2` @ 16K, `5,969.6` @ 32K, `5,694.4` @ 128K), `w4096` interactive prefill (`5,046–5,276 tok/s`), lossless `K=5` MTP decode (`56.5 tok/s` med / `57.4` peak, `2,560/2,560` greedy exact), and 136,576-token 6-hop Estonia needle retrieval (`5,400.4 tok/s` prefill, `48.2 tok/s` cli / `48.6 tok/s` srv decode, `5/5 PASS`). |
| [`bench/qwen38-gb10/`](../bench/qwen38-gb10/) | `nvidia/Qwen3.8-27B-NVFP4` (or compatible NVFP4 checkpoint) + `z-lab/Qwen3.8-27B-DFlash2` / `v1` | Single-stream `gamma=15` DFlash speculative decode (`63.9 tok/s` on qualified `BF16-KV`, `58.012 tok/s` on corrected-ABI `v3` control) on the deterministic MinHeap coding probe. |
| [`repro-kit/`](../repro-kit/) | `Qwen3.8-Flash-Next-NVFP4-Offload` (`v1.0.3` / `v1.0.x`) | One-command container (`flux-bench run`) and host (`benchmark.sh`) reproduction with SHA-256 manifest verification (`model-manifest.json`, `ENGINE_STABLE_PIN.json`). |
| [`repro/qwen38-coding/`](../repro/qwen38-coding/) | `Qwen3.8-27B` / `Qwen3.8-Flash-Next` | Deterministic coding endpoint verifier (`run.py`) writing output-sanitized SHA-256 fingerprints and server TPS telemetry. |

## Where every commit came from

`flux-engine` preserves full attribution to both upstream Atlas repositories and
the `apathy-atlas` model branches:

| Upstream / Lineage Stage | Repository | What it contributed |
|---|---|---|
| Foundational pure-Rust engine & Laguna-S-2.1 enablement | [`MonumentalSystems/atlas`](https://github.com/MonumentalSystems/atlas) | Workspace crate architecture (`atlas-core`, `atlas-kernels`, `spark-runtime`, `spark-model`, `spark-server`, `spark-storage`), scheduler, HTTP server, and 36 load-bearing Laguna-S-2.1 commits. |
| Multi-model Blackwell runtime & structured generation | [`Avarok-Cybersecurity/atlas`](https://github.com/Avarok-Cybersecurity/atlas) | Upstream stewardship, SM120/SM121 integration, pure-Rust `xgrammar` port, Ratatui TUI, and 12-model baseline matrix. |
| GB10 DFlash & TurboQuant+ fork baseline | [`TheApathy/apathy-atlas`](https://github.com/TheApathy/apathy-atlas) | `Qwen3.8-27B` / `Qwen3.6-27B` DFlash speculative decode, split-K W4A16 GEMMs, `WY17` lazy GDN recurrence, and TurboQuant+ KV compression. |
| `Qwen3.8-Flash-Next` & multi-model `sm_121a` stack | [`TheApathy/flux-engine`](https://github.com/TheApathy/flux-engine) | `cutlass_5222` (`v28q`/`v17q`) W4A8 compact MoE, `qwen4_qsa_prefill_select` + `qkpv_amax_pipe`, `gdn_umma_prefill` + `conv1d_v3_commit` + `rmsnorm + mxfp8_quant`, 4-stream `mHC` `t128` roof + `PIN_M=4096`, 4-ring NVFP4 PLE streaming, lossless `K=5` Inferact MTP, and `FLUX_*` operating recipes. |

All harnesses target GB10-class hardware (NVIDIA Grace Blackwell, `sm_121a`,
128 GB unified LPDDR5x). Nothing in any harness is hardcoded to a particular
machine: paths, ports, and checkpoint locations come from environment variables,
and the launchers refuse to guess rather than silently pick a default that is
wrong for your setup.

```bash
cat bench/qwen38-flash-next-gb10/README.md
cat bench/qwen38-gb10/README.md
cat repro-kit/README.md
```

## The two Qwen3.8 harnesses are siblings, not variants

[`bench/qwen38-flash-next-gb10/`](../bench/qwen38-flash-next-gb10/) (`125B+51B PLE`,
`6B` active MoE + 1-layer `K=5` Inferact MTP head) and
[`bench/qwen38-gb10/`](../bench/qwen38-gb10/) (27B dense + `5`-layer DFlash2 `B8` /
`8`-layer DFlash v1 `B16` drafter) are deliberately the same shape: one self-verifying `serve.sh`,
one complete `FLAGS.md` reference, one `ARTIFACTS.sha256` manifest, and
deterministic Python probes that verify output hashes alongside throughput.

They are **not** re-parameterizations of each other, and settings do not
transfer between them:
- `Qwen3.8-Flash-Next` compiles `--kernel-target qwen3.8-flash-next`, uses
  block-scaled W4A8 (`NVFP4` x `MXFP8`) grouped MoE (`v28q`/`v17q`),
  12-layer QSA sparse attention (`--qwen4-qsa`), 4-ring NVFP4 PLE streaming, and
  a 4-draft (`K=5`) Inferact MTP head that is **100% lossless (`2,560 / 2,560`
  greedy tokens bit-exact vs `K=1`)**.
- `Qwen3.8-27B` compiles `--kernel-target qwen3.8-27b`, uses dense W4A16
  split-K FFN/SSM projections, and runs a bidirectional DFlash drafter (`5`-layer
  DFlash2 `B8` qualified default or `8`-layer DFlash v1 `B16` at `gamma=15`,
  `ATLAS_DFLASH_DRAFT_CAP=15`, `ATLAS_DDTREE_MAX_NODES=16`).

Read the `README.md` and `FLAGS.md` in the directory for the model you are
actually serving.

## What the asserts are for

Every launcher here refuses to report a healthy serve unless its invariants
hold, and each assert exists because its absence once produced a clean-looking
but wrong measurement. The recurring failure mode in inference benchmarking is
not a crash: it is a server that comes up healthy while running a configuration
nobody asked for, and then produces plausible numbers about the wrong thing.

Four of those checks generalize across the whole repository:

1. **A gate the binary cannot read is a silent no-op.** Setting a `FLUX_*` or
   `ATLAS_*` environment variable proves nothing about whether the binary honors
   it. `bench/qwen38-flash-next-gb10/serve.sh` verifies that both the kernel
   target (`qwen3.8-flash-next`) and the required `ATLAS_QWEN4_*` gate strings
   actually exist inside the binary before exec'ing it.
2. **Prefix caching must be explicitly off when measuring prefill.** A warm
   radix cache or SSM state slot makes a second repetition look 10× faster
   without running a single prefill kernel. All prefill benchmarks run with
   `--enable-prefix-caching false --ssm-cache-slots 0` and verify identical
   output hashes across uncached repetitions.
3. **Speculative speedup without a token-identity check can hide silent
   corruption.** For `Qwen3.8-Flash-Next`, `K=5` MTP decode is gated on
   `LOSSLESS 10/10` (`2,560 / 2,560` tokens bit-for-bit identical to
   non-speculative `K=1` decode via the `flux` launcher `--no-speculative` flag
   or raw `flux-server` `ATLAS_DISABLE_SPECULATION=1` across `0..64K` context)
   plus `5/5 PASS` on the 136,576-token 6-hop Estonia needle benchmark.
4. **Unified memory floor (`MemAvailable >= 6,656 MiB`) is a correctness gate on
   GB10.** Because CPU and GPU share 128 GB of LPDDR5x without a PCIe boundary,
   dropping below `6,144 MiB` triggers the hardware watchdog. Every recipe pins
   `_RESERVE_GB` (`7` GiB for `w16384`, `10` GiB for `136.5K`) so the free
   memory floor stays above `7,072–12,695 MiB`.

## Measurements are content-mix dependent

Speculative decoding throughput varies with how predictable the next tokens are:
structured code and algorithmic implementations accept more drafted tokens per
cycle (`3.02–3.15` of 4 drafts on `Qwen3.8-Flash-Next` `K=5` MTP, and `7.33` of
15 drafts — `8.33` emitted tokens/cycle — on the 400-token `Qwen3.8-27B` DFlash
MinHeap probe, or `5.81` of 15 on the 800-token probe), whereas open-ended creative prose
accepts fewer. Always report the probe name, token count, and output hash
alongside any decode tok/s figure.
