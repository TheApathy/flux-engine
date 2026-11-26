# ADR-0007: Composing tensor + expert parallelism

**Status:** Accepted
**Date:** 2026-04-30

## Context

Several models we ship don't fit on one GB10 (122B-A10B, MiniMax-M2 /
M2.7-NVFP4, Mistral-Small-4, Nemotron-Super-120B). On two GB10s connected
by RoCEv2, we have three plausible parallelism shapes:

1. **Pure TP=2**: every layer's GEMMs sharded across both GPUs.
   all-reduce after every linear. Cleanest mental model, worst
   communication-to-compute ratio for MoE models (you all-reduce the
   *expert outputs* every layer).
2. **Pure EP=2**: experts sharded across GPUs. Routed dispatch + combine
   via NCCL all-to-all per MoE layer. Attention runs replicated on
   both ranks. Saves MoE weight memory. Communication is on the
   token-routing path, not the per-layer linear.
3. **TP + EP overlap**: attention sharded TP=2, MoE experts sharded
   EP=2. Two overlapping NCCL groups share one comm. Best memory
   utilization. Most complex.

We initially shipped pure EP=2 only. The fail-mode that taught us
the alternatives matter: prefill TTFT was dominated by EP all-to-all
communication, while decode was fast because EP collectives during
decode are tiny. TP=2 attention helps prefill but hurts decode. Some
models have only-MoE experts to shard (good fit for EP), others have
big dense FFNs (better fit for TP).

## Decision

Atlas supports **all three modes via a single `--tp-size N --ep-size M`
flag pair**:

| Mode | `--tp-size` | `--ep-size` | Use when |
|---|---|---|---|
| Single GPU | 1 | 1 | Default, model fits on one rank. |
| Pure EP=2 | 1 | 2 | MoE expert sharding only (122B, MiniMax). |
| Pure TP=2 | 2 | 1 | Dense / attention sharding only (rare today). |
| TP + EP overlap | 2 | 2 | Both sharded. NCCL groups share one comm. |

Implementation:

- `crates/spark-comm/src/nccl_backend.rs` exposes the collective ops
  (`all_reduce`, `broadcast`, `all_gather`, expert-routed all-to-all).
- TP layers (`qwen3_attention/`, `dense_ffn/`) are aware of `world_size`
  and shard their weight columns/rows accordingly. SSM weights remain
  full-replica per rank: wastes memory, but SSM-state sharding is a
  separate problem.
- EP layers (`moe/`) shard the routed experts across ranks. The shared
  expert is replicated.
- The launcher (`scripts/start-ep2.sh`) sets the NCCL env identically on
  both ranks. Mismatched flags between rank 0 and rank 1 (e.g. one rank
  enables `--speculative` and the other doesn't) produce SSM
  intermediate-buffer errors at runtime (see [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md)).

The EP+TP overlap case (TP=2, EP=2) shipped
with the unified-layout MoE rewrite, gated behind
`ATLAS_UNIFIED_MOE_LAYOUT=1` until decode regression is closed.

## Consequences

**Better:**
- One launcher. One mental model. The user picks `--tp-size` and
  `--ep-size`. The runtime composes.
- Prefill TTFT and decode tok/s can be separately optimized by choosing
  the right shape per model (we picked TP=2,EP=2 for MiniMax M2.7
  precisely because pure EP=2 prefill was bottlenecked).
- All weight loaders are TP-aware,
  so adding a TP shape for a new model is loader work, not infra work.

**Worse:**
- The MTP / DFlash flag-symmetry requirement is a
  recurring footgun. Asymmetric flags don't fail at startup. They fail
  mid-run with confusing buffer errors.
- TP+EP overlap requires a shared NCCL communicator: debugging
  collective hangs ("which group is stuck?") is harder than in pure
  modes.
- SSM full-replica is wasteful: at TP=2, every rank holds the full SSM
  conv1d / hidden state. Acceptable today (SSM weights are small).
  future SSM-heavy models may force a redesign.

**New problems we created:**
- The `--tp-size N` UX implies arbitrary N, but we have only ever
  tested N=1 and N=2. TP>2 paths are theoretically wired but unverified.
- Cross-node EP=2 has hardware-specific NCCL env (RoCEv2 NIC names,
  GDR-disabled flags). The launcher hardcodes GB10's `enp1s0f0np0`. See
  ADR-0009 (kernel target tuples) for the related per-hardware
  configuration story. The launcher needs the same treatment.
