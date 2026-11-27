// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle, PinnedHostBuffer};
use spark_runtime::kv_cache::PagedKvCache;

use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen4K5DeviceAttentionTopology;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Architecture-agnostic transformer model.
///
/// Composes `Vec<Box<dyn TransformerLayer>>` into a full forward pass.
/// Adding a new model only requires implementing [`TransformerLayer`]
/// for each layer type — the model loop stays unchanged.
#[allow(dead_code)]
pub struct TransformerModel {
    pub(super) config: ModelConfig,
    /// M8A: per-step DDTree parent-ids channel. verify_d.rs uploads a
    /// payload's parent_indices tensor and stashes the device pointer
    /// here BEFORE the K=γ layer loop; the SSM dispatch reads it to
    /// decide flat (gdn_wy17_k) vs tree-aware (gdn_tree_k) kernel.
    /// `None` outside DDTree verify or when payload is flat-chain.
    pub ddtree_parent_ids_dev: Mutex<Option<spark_runtime::gpu::DevicePtr>>,
    /// Task #34: TRUE when the K=γ verify that just ran took the graph-safe
    /// FLAT-CHAIN tree_wy injection (verify_d.rs substitutes the persistent
    /// linear-chain parent_ids at `k == ddtree_parent_ids_capacity` when
    /// `ATLAS_DISABLE_TREE_WY` is unset and CUDA graphs are on). The injected
    /// route runs `gated_delta_rule_tree_wy`, which leaves the live `h_state`
    /// UNTOUCHED — the commit must read this flag (in addition to the
    /// scheduler-set `ddtree_parent_ids_dev` stash) to route a FULL accept
    /// through the `h_intermediate[K-1]` copy path instead of the stale
    /// `h_state` fast path. Written by every `decode_verify_graphed_kgamma`
    /// call; consumed (swap-to-false) by `commit_verify_state_async`.
    pub dflash_flat_tree_route: std::sync::atomic::AtomicBool,
    pub(super) verify_transactions: Mutex<crate::traits::verify_owner::Transactions>,
    pub(super) verify_batch_transactions:
        Mutex<crate::traits::verify_owner::batch::BatchTransactions>,
    /// Was the LAST K=γ verify's attention ancestor-exact for EVERY tree
    /// node? `true` for flat/chain payloads (prefix reads ARE the exact
    /// conditioning) and for the per-row KV indirection path
    /// (`ATLAS_TREE_AWARE_ATTN` + `ATLAS_DDTREE_TREE_AWARE_VERIFY`). `false`
    /// when a NON-flat tree payload ran with prefix metadata only (chain
    /// mode or `ATLAS_DDTREE_DFS_REORDER=1` depth-prefix reads): DFS prefix
    /// reads are ancestor-exact only for the leftmost spine, so branch rows
    /// read a spine sibling and miss their own key (the 2026-07-08
    /// deep-branch-tail commit corruption root cause). The scheduler reads
    /// this via `Model::dflash_tree_ancestor_attn_exact` to gate the FULL
    /// tree-commit walker; when false it must degrade to the flat-safe
    /// walker, which only consumes correctly-conditioned spine rows.
    /// Written by every `decode_verify_graphed_kgamma` call.
    pub dflash_tree_ancestor_attn: std::sync::atomic::AtomicBool,
    /// Length of the parent_ids tensor above (= num_tree_tokens = K verify).
    pub ddtree_num_tree_tokens: Mutex<usize>,
    /// CUDA-graph-safe persistent parent_ids buffer for the K=γ verify
    /// path. Allocated ONCE at init, sized for `dflash_kgamma * 4` bytes
    /// (= 17 i32 slots on the 27b target). Holds two state-overlays:
    ///   * Default (linear-chain): `[-1, 0, 1, ..., K-2]` — bit-equivalent
    ///     to the wy17 flat path when consumed by `gated_delta_rule_tree_wy`.
    ///     Re-stamped after every tree-mode verify by `clear_ddtree_parent_ids`.
    ///   * Tree-mode override: `set_ddtree_parent_ids` overwrites in place
    ///     with the kernel-frame mapping of the current payload.
    ///
    /// The DEVICE POINTER never changes — that's the invariant captured
    /// CUDA graphs rely on. `DevicePtr::NULL` when DFlash is disabled.
    pub ddtree_parent_ids_persistent: spark_runtime::gpu::DevicePtr,
    /// Capacity of `ddtree_parent_ids_persistent` in i32 slots (= dflash_kgamma).
    /// Zero when DFlash is disabled. Used to stamp linear-chain defaults.
    pub ddtree_parent_ids_capacity: usize,
    /// Host-side mirror of the kernel-frame parent_ids tensor. Populated by
    /// `set_ddtree_parent_ids` so the K=γ verify path can derive per-token
    /// tree depths (for depth-based RoPE positions and seq_lens) without
    /// a D2H copy. Empty when no tree payload is active (chain mode).
    ///
    /// Layout matches `ddtree_parent_ids_persistent`: index 0 is the bonus
    /// (parent=-1), index i+1 corresponds to draft i, with each entry the
    /// kernel-frame parent (`-1` = pre-tree state / bonus's parent, `k≥0`
    /// = parent kernel slot).
    pub ddtree_parent_ids_host: Mutex<Vec<i32>>,
    /// Inverse DFS permutation from the last K=γ verify (option C reorder).
    /// `dfs_inv_perm[orig_compact_idx] = dfs_slot_idx`. Empty when DFS
    /// reorder is inactive (env var off, or payload was trivially flat).
    ///
    /// Populated by `verify_d.rs` BEFORE the SSM kernel runs (because the
    /// kernel writes h_state_inter in DFS slot order). Read by
    /// `commit_verify_state_async_with_slot` to translate the original-
    /// compact `last_inter_slot` from greedy_sample_ddtree into the DFS
    /// slot where the SSM canonical state actually lives.
    pub ddtree_dfs_inv_perm: Mutex<Vec<usize>>,
    /// ATLAS_TREE_AWARE_ATTN: persistent per-row KV indirection buffer.
    /// Layout: `[dflash_kgamma rows × dflash_kgamma cols]` i32, row-major.
    /// `verify_d.rs` writes the ancestor-chain mapping for the current
    /// K=γ tree (single-shot upload) so the paged-decode attention kernel
    /// can remap tree-window positions to true ancestors.
    /// `DevicePtr::NULL` when DFlash is disabled.
    pub tree_kv_indir_persistent: spark_runtime::gpu::DevicePtr,
    /// Row stride of `tree_kv_indir_persistent`, in i32 slots
    /// (= dflash_kgamma). Zero when DFlash is disabled.
    pub tree_kv_indir_stride: usize,
    /// ATLAS_TREE_AWARE_ATTN CUDA graph fix: persistent 1×i32 device buffer
    /// holding the current tree-window base position (= `seq.seq_len`).
    /// The paged-decode-attn + tree-kv-scatter kernels read this via
    /// pointer arg so a captured graph sees the fresh value on each replay
    /// — the prior scalar arg was baked into the kernel-launch node at
    /// capture time and went stale as `seq.seq_len` advanced.
    /// `verify_d.rs` writes the current `seq.seq_len` here before each K=γ
    /// verify step. `DevicePtr::NULL` when DFlash is disabled.
    pub tree_kv_indir_base_persistent: spark_runtime::gpu::DevicePtr,
    /// Pinned-host shadow of `tree_kv_indir_base_persistent` (1×i32).
    /// `verify_d.rs` writes `seq.seq_len` into this pinned buffer then
    /// fires `cuMemcpyHtoDAsync` from it on the verify stream. Pinned
    /// host memory establishes a proper stream-ordered dependency for
    /// the H2D copy (pageable Vec sources do not, leading to a race
    /// where captured graph kernels dereference the device buffer
    /// before the upload completes → stale `kv_indir_base` → `1, 2, 2, 3`
    /// style token-duplication artifacts).
    /// `null` when DFlash is disabled.
    pub tree_kv_indir_base_host_pinned: Mutex<Option<PinnedHostBuffer>>,
    /// ATLAS_TREE_KV_PACK: per-attention-layer packed-KV scratch pool.
    /// One entry per FullAttention layer; each is the K-pool base pointer
    /// for `[num_seqs blocks × stride tokens]` of the layer's KV dtype.
    /// `verify_d.rs` triggers a small scatter kernel that materializes the
    /// ancestor chain for each row into this scratch, and the existing
    /// `paged_decode_attn_*` kernels run over it with NULL indirection
    /// (fast BC=4 batched path). Empty when DFlash is disabled or
    /// `ATLAS_TREE_KV_PACK` is unset.
    pub tree_kv_pack_scratch_k: Vec<spark_runtime::gpu::DevicePtr>,
    /// Same layout as `tree_kv_pack_scratch_k` for V pool.
    pub tree_kv_pack_scratch_v: Vec<spark_runtime::gpu::DevicePtr>,
    /// Identity block table for the packed-KV path: `[num_seqs × 1]` i32
    /// with `bt[seq] = seq`. Allocated once.
    pub tree_kv_pack_block_table: spark_runtime::gpu::DevicePtr,
    /// `seq_lens` buffer for the packed-KV path: `[num_seqs]` i32 holding
    /// `depth[t] + 1` (= chain length for row t). Uploaded per-step by
    /// `verify_d.rs` when the packed-KV path is active.
    pub tree_kv_pack_seq_lens: spark_runtime::gpu::DevicePtr,
    /// Bytes per scratch block (one block per seq). Equals
    /// `stride * num_kv_heads * head_dim * elem_bytes` for FP8 (and
    /// `nvfp4_data_bytes + nvfp4_scale_bytes` for NVFP4). Stored in u64
    /// because the consumer kernel takes `block_stride_bytes` as u64.
    pub tree_kv_pack_block_stride_bytes: u64,
    /// NVFP4-only: data-section size in bytes per block (scales follow
    /// immediately after). Zero for FP8.
    pub tree_kv_pack_data_section_bytes: u64,
    /// FP8 scatter kernel handle (`tree_kv_scatter_fp8`). `KernelHandle(0)`
    /// when not loaded.
    pub tree_kv_pack_scatter_fp8_kernel: spark_runtime::gpu::KernelHandle,
    /// NVFP4 scatter kernel handle (`tree_kv_scatter_nvfp4`).
    pub tree_kv_pack_scatter_nvfp4_kernel: spark_runtime::gpu::KernelHandle,
    /// True when `ATLAS_TREE_KV_PACK=1` AND scratch is allocated AND
    /// at least one scatter kernel is loaded.
    pub tree_kv_pack_active: bool,
    pub(super) embed_tokens: DenseWeight,
    pub(super) final_norm: DenseWeight,
    /// True when the explicit terminal mixer already performed the model-level
    /// final normalization and the LM head must consume that H-wide row as-is.
    pub(super) final_norm_identity: bool,
    pub(super) qwen4_final_mixer: Option<crate::layers::Qwen4HyperConnection>,
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(super) qwen4_ple: Option<crate::layers::Qwen4PleLayer>,
    /// Cross-chunk PLE next-chunk prefetch cache (engages at `stream_level >= 2`,
    /// including final tails), keyed by the sequence slot that owns each entry.
    /// Exact token identity is revalidated before consumption.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(super) qwen4_ple_prefetch:
        Mutex<HashMap<usize, crate::layers::qwen4_ple::PleChunkPrefetch>>,
    pub(super) lm_head_weight: DenseWeight,
    pub(super) lm_head_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub(super) lm_head_nvfp4: Option<QuantizedWeight>,
    /// Transposed NVFP4 lm_head shared with the DFlash drafter propose path.
    /// Target verification uses the exact row-major family instead. Built at
    /// load when ATLAS_LM_HEAD_T=1 (~0.63 GB on qwen3.6-27b).
    pub(super) lm_head_nvfp4_t: Option<QuantizedWeight>,
    pub(super) layers: Vec<Box<dyn TransformerLayer>>,
    pub(super) buffers: BufferArena,
    pub(super) kv_cache: Mutex<PagedKvCache>,
    /// Page-locked host staging for batched metadata H2D transfers.
    /// Allocated once at init and released by its RAII owner before `gpu`.
    ///
    /// Protected by `parking_lot::Mutex` held only during short host-pack /
    /// enqueue scopes (never across NCCL collectives or layer loops) so
    /// `TransformerModel` is soundly `Send + Sync` without `UnsafeCell`.
    pub(super) pinned_staging: Mutex<PinnedMetaStaging>,
    pub(super) gpu: std::sync::Arc<dyn GpuBackend>,
    pub(super) rms_norm_kernel: KernelHandle,
    /// Byte-exact strided row copy (`strided_copy_rows_16`); 0 when absent.
    /// Read by the DFlash prefill capture in `impl_b3.rs` under
    /// `ATLAS_DFLASH_CAPTURE_STRIDED=1`: one copy per ring span instead of one
    /// D2D copy per row.
    pub(super) strided_copy_rows_kernel: KernelHandle,
    /// `qwen4_hyper::qwen4_hc_mean_rows` kernel for `DflashCaptureMode::Qwen4HcMean`.
    /// `KernelHandle(0)` when unused.
    pub(super) qwen4_hc_mean_rows_kernel: KernelHandle,
    pub(super) bf16_to_f32_kernel: KernelHandle,
    pub(super) dense_gemv_kernel: KernelHandle,
    /// FP32-output variant of dense_gemv_bf16. Used by the LM head when
    /// `use_fp32_logits` is true, so the FP32 accumulator is preserved across
    /// the BF16-storage rounding boundary that flips greedy argmax tiebreaks
    /// on Gemma-4-31B (top-1 vs top-2 = 0.125 logit gap = exact BF16 step at
    /// value 16-32 → BF16 store snaps the wrong way and starts a stop-word
    /// loop). Loaded once at model init.
    pub(super) dense_gemv_fp32out_kernel: KernelHandle,
    pub(super) w4a16_gemv_kernel: KernelHandle,
    pub(super) w4a16_gemv_sw_kernel: KernelHandle,
    pub(super) gemv_sw: bool,
    pub(super) w4a16_gemv_logits_kernel: KernelHandle, // FP32 output for LM head
    /// Exact K1-order multi-row NVFP4 LM-head kernels. Individual handles may
    /// be absent; production then fails closed to independent row-major K1
    /// GEMV launches for that tier.
    pub(super) w4a16_exact_lm_head_kernels: ops::W4a16ExactLmHeadKernels,
    pub(super) w4a16_gemm_kernel: KernelHandle,
    /// `w4a16_gemm_t_m32_n64` used only to qualify/build the optional
    /// transposed copy shared with the drafter. 0 when absent.
    pub(super) w4a16_gemm_t_m32_n64_kernel: KernelHandle,
    /// Legacy handle retained for ABI compatibility; target LM-head verify
    /// must not route through its different accumulation order.
    pub(super) w4a16_gemv_batch2_kernel: KernelHandle,
    /// Legacy M=3 LM-head handle retained for ABI compatibility. Production
    /// M=3 verification uses the exact m4 tier or serial K1 fallback.
    pub(super) w4a16_gemv_batch3_logits_kernel: KernelHandle,
    pub(super) dense_gemm_kernel: KernelHandle,
    pub(super) argmax_kernel: KernelHandle,
    /// Row-parallel `argmax_bf16_rows` for batched greedy sampling and verify;
    /// zero where the module does not carry it, which selects scalar fallback.
    pub(super) argmax_rows_kernel: KernelHandle,
    pub(super) argmax_logits_kernel: KernelHandle, // FP32 argmax for logits
    /// `topk_candidates_bf16` (ATLAS_SAMPLER_DEVICE_TOPK=1); zero when the
    /// route is off. `topk_candidates_buf` holds its ids / values / count
    /// (`device_topk_buf_bytes`), allocated only when the route is on.
    pub(super) topk_candidates_kernel: KernelHandle,
    /// `edited_sum_exp_bf16` (certified F2); zero when the route is off.
    pub(super) f2_sum_kernel: KernelHandle,
    pub(super) topk_candidates_buf: spark_runtime::gpu::DevicePtr,
    /// Optional batched GPU target Gumbel-Max verify kernel
    /// (`sampler_topk_candidates::target_gumbel_verify_rows_bf16`). Zero on
    /// targets whose kernel bundle does not provide it.
    pub(super) target_gumbel_verify_kernel: KernelHandle,
    /// Reusable 8 KiB staging buffer for up to `GUMBEL_VERIFY_MAX_ROWS` (256)
    /// rows of batched GPU target Gumbel-Max verify metadata + output tokens.
    /// `DevicePtr::NULL` when `target_gumbel_verify_kernel` is absent.
    pub(super) target_gumbel_verify_buf: spark_runtime::gpu::DevicePtr,
    /// Optional per-row Gumbel-Max verify specs staged immediately before a
    /// verify forward so `verify_d` / `verify_qwen4_batched` can fold
    /// `target_gumbel_verify_rows_bf16` directly into the verify tail / CUDA
    /// graph instead of launching a discarded `argmax_bf16_rows` + second drain.
    pub(super) staged_verify_gumbel_rows:
        Mutex<Option<Vec<spark_runtime::sampler::device_topk::GumbelVerifyRowSpec>>>,
    /// Tokens produced in-graph by the immediately preceding verify forward
    /// when `staged_verify_gumbel_rows` was consumed in-graph.
    pub(super) in_graph_gumbel_verify_tokens: Mutex<Option<Vec<u32>>>,
    /// Optional batched GPU maximal-coupling rejection verify kernel
    /// (`sampler_topk_candidates::target_rejection_verify_rows_bf16`). Zero on
    /// targets whose kernel bundle does not provide it.
    pub(crate) target_rejection_verify_kernel: KernelHandle,
    /// Reusable 13 KiB staging buffer for up to `REJECTION_MAX_ROWS` (256)
    /// rows of batched GPU maximal-coupling rejection verify metadata + packed
    /// output words. `DevicePtr::NULL` when `target_rejection_verify_kernel` is
    /// absent.
    pub(crate) target_rejection_verify_buf: DevicePtr,
    /// Shared 80 KiB GPU draft-distribution scratch buffer
    /// (`q_cand_ids: [512, 20] u32` at offset `0`, `q_probs: [512, 20] f32` at
    /// offset `40_960`) written by probabilistic DFlash2 propose and read by
    /// `target_rejection_verify_rows_bf16`.
    pub(crate) target_rejection_q_buf: DevicePtr,
    /// Optional per-row maximal-coupling rejection verify specs staged
    /// immediately before a verify forward so `verify_d` /
    /// `verify_qwen4_batched` can fold `target_rejection_verify_rows_bf16`
    /// directly into the verify tail / CUDA graph.
    pub(crate) staged_verify_rejection_rows:
        Mutex<Option<Vec<crate::model::trait_impl::device_topk::RejectionVerifyRowSpec>>>,
    /// Packed words produced in-graph by the immediately preceding verify
    /// forward when `staged_verify_rejection_rows` was consumed in-graph.
    pub(crate) in_graph_rejection_verify_words:
        Mutex<Option<Vec<crate::model::trait_impl::device_topk::PackedRejectionWord>>>,
    pub(super) batched_embed_kernel: KernelHandle,
    pub(super) fill_slots_kernel: KernelHandle,
    /// Cached CUDA graph for single-sequence decode (layer loop + norm + LM head).
    /// CUDA graph cache for n=1 decode, keyed by `seq.slot_idx`. The captured
    /// graph has SSM h_state/conv_state pointers baked in as kernel arguments,
    /// so a graph captured for slot S can ONLY be replayed for slot S — replay
    /// for any other slot reads/writes the wrong sequence's recurrent state
    /// and produces gibberish for both sequences. With concurrent users we may
    /// alternate between slots in n=1 decode (e.g. via the per-seq fresh-decode
    /// fix in scheduler::step_decode_only), so we keep one graph per slot.
    pub(super) decode_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Per-slot QSA sparse-regime (`seq_len >= 2048`) recorded when `decode_graph`
    /// was captured, so multi-token speculative steps jumping past `2048` (`2045 -> 2050`)
    /// still invalidate any pre-2048 graph on that slot.
    pub(super) decode_graph_sparse_regime: Mutex<std::collections::HashMap<usize, bool>>,
    /// Cached CUDA graphs for batched decode.
    ///
    /// Two cache schemes coexist:
    /// 1. **padded_n key** (default): keyed by `(vec![], padded_n)`. Used when
    ///    `ATLAS_SSM_MULTI_SEQ_GRAPH` is OFF — graphs are NOT replayed across
    ///    batches because SSM h_state/conv_state pointers are slot-specific.
    /// 2. **slot-set key** (graph mode): keyed by `(sorted_slot_ids, padded_n)`.
    ///    Used when `ATLAS_SSM_MULTI_SEQ_GRAPH=1`. The graph bakes the per-slot
    ///    SSM pool pointers in; replay is safe as long as the same slot set is
    ///    active. When a slot is freed, all entries containing it are dropped
    ///    (`free_sequence`).
    pub(super) batch_decode_graphs: Mutex<HashMap<(Vec<usize>, usize), GraphHandle>>,
    /// Cached CUDA graphs for the **piecewise** multi-seq decode path
    /// (`ATLAS_MULTISEQ_GRAPHS=1`, see `decode_batch_dispatch_piecewise`).
    ///
    /// Keyed by `(padded_n, segment_id)` ONLY — no slot tuple. Each entry
    /// is a captured run of consecutive address-stable layers (SSM/FFN) or
    /// the final-norm+LM-head tail. Attention layers are NOT represented
    /// here; they run eagerly between segment replays. Because every
    /// captured segment reads only fixed device addresses (SSM ptr table,
    /// attn metadata base, model buffers — all re-uploaded before replay),
    /// a segment graph replays for any active slot set of the same padded
    /// size, so entries are NEVER invalidated on `free_sequence`.
    pub(super) piecewise_decode_graphs: Mutex<HashMap<(usize, usize), GraphHandle>>,
    /// Pre-allocated SSM state pool for stable GPU addresses across graph replays.
    pub(super) ssm_pool: SsmStatePool,
    /// SSM state snapshot pool for Marconi prefix caching.
    pub(super) ssm_snapshots: SsmSnapshotPool,
    /// Fixed max blocks per sequence (max_seq_len / block_size + 1).
    /// Used as constant stride in attention metadata for CUDA graph compatibility.
    pub(super) max_blocks_per_seq: u32,
    /// Permanent KV cache block for padding sequences in batched decode.
    pub(super) dummy_kv_block: u32,
    /// Profile mode: skip graphs, sync+time each layer. Set ATLAS_PROFILE=1.
    pub(super) profile: bool,
    /// One-shot profile flag for the next prefill request only. Set
    /// ATLAS_PROFILE_FIRST=1 to capture per-step timing on the first prefill
    /// after startup without disabling CUDA graphs for subsequent decodes.
    /// Consumed (atomically swapped to false) by `prefill_chunk` / `prefill`.
    pub(super) profile_first_pending: std::sync::atomic::AtomicBool,
    /// When true, decode and verify paths skip CUDA graph capture/replay.
    /// Driven by FP8 KV calibration, `ATLAS_DIAG_GEMMA4`, `ATLAS_DUMP_HIDDEN`,
    /// and scoped per-sequence serial batch decode (`decode_serial_batch.rs`).
    pub(super) suppress_graphs: std::sync::atomic::AtomicBool,
    /// Draft proposer for arm 0 — the PRIMARY arm, whatever the build
    /// installed last. With `--dflash` this is the `BlockDiffusionDraftHead`;
    /// with plain `--speculative` it is the in-checkpoint `MtpHead`.
    ///
    /// Read through [`TransformerModel::active_proposer`], never directly,
    /// so the speculation gate's arm selection is honoured. The two capability
    /// predicates (`has_proposer`, `mtp_lastk` gating) deliberately ask
    /// [`TransformerModel::any_proposer`] instead: "is speculation possible at
    /// all" is not the same question as "which arm is live right now".
    pub(super) proposer: Option<Arc<dyn DraftProposer>>,
    /// Draft proposer for arm 1 — the ALTERNATE arm. Populated only when the
    /// build produced two proposers: `set_dflash_proposer` demotes the native
    /// MTP head here instead of dropping it, so `--dflash --speculative`
    /// yields DFlash on arm 0 and MTP on arm 1 and the gate can pick per step.
    ///
    /// `None` on every single-proposer build, which is every build that does
    /// not pass both flags — so the default configuration is bit-identical to
    /// before this field existed.
    pub(super) proposer_alt: Option<Arc<dyn DraftProposer>>,
    /// Which proposer arm is live. Read on the decode hot path by
    /// [`TransformerModel::active_proposer`], written only by the scheduler's
    /// speculation gate between steps.
    ///
    /// An atomic rather than a `&mut` field because the scheduler holds the
    /// model as `&dyn Model` for the whole run and every step function takes a
    /// shared reference; and a plain atomic rather than a lock because this is
    /// read once per propose on the critical path.
    pub(super) active_proposer_arm: std::sync::atomic::AtomicU8,
    /// Dedicated buffer for saving hidden state before MTP head runs.
    /// Size: hidden_size * 4 bytes (one FP32 vector). MTP overwrites shared
    /// buffers (norm_output etc.), so the target hidden must be saved here first.
    pub(super) mtp_hidden_save: DevicePtr,
    /// Sequence-bound native-MTP retained prefixes, laid out [slot, K, 4H]
    /// in BF16. This does not isolate target/proposer execution scratch.
    pub(super) qwen4_mtp_hidden_frames: Option<(
        DevicePtr,
        Mutex<super::qwen4_mtp_hidden::NativeMtpHiddenFrames>,
    )>,
    /// Owner of the one-per-layer QSA verify journals while a frame has them
    /// armed. Consulted only under `ATLAS_QWEN4_K5_CMULTI` (serial K5 C2-C8).
    pub(super) qwen4_qsa_journal_owner: Mutex<super::qwen4_k5_cmulti::QsaJournalOwnership>,
    /// Verifier-owned row staging for the cross-sequence native-MTP verify
    /// (`ATLAS_QWEN4_BATCHED_K5`). Present exactly when per-slot verify lanes
    /// were configured; `None` keeps every batched path disarmed.
    pub(super) qwen4_cross_seq_staging: Option<crate::layer::Qwen4CrossSeqStaging>,
    /// Dedicated disjoint device buffer for cross-sequence decode verify
    /// attention metadata (`positions | slots | seq_lens | block_tables`),
    /// lazily allocated so decode verify metadata never aliases
    /// `[0 .. 2 * (P + D) * top_k * 4)` of `buffers.scratch()` written by
    /// `ffn.forward_prefill`.
    pub(super) qwen4_verify_attn_meta: Mutex<Option<(DevicePtr, usize)>>,
    /// Batched GDN verify-commit kernel (`gdn_batch_commit::gdn_batch_commit_copy_f32`).
    /// `KernelHandle(0)` when absent from the loaded kernel bundle.
    pub(super) gdn_batch_commit_kernel: KernelHandle,
    /// Commit-path environment flags resolved once at model construction.
    pub(super) commit_runtime_flags: super::trait_impl::commit_cache::CommitRuntimeFlags,
    /// Lazy per-slot device pointer table cache for batched verify commits.
    pub(super) slot_commit_table: Mutex<super::trait_impl::commit_cache::SlotCommitTable>,
    /// This model's claim on the process verify-lane count; dropping the
    /// model resets it unless a newer model registered since.
    pub(super) qwen4_verify_lanes: crate::layers::qwen4_verify_lanes::LanesRegistration,
    /// Last-K prompt-tail target hidden capture buffer for MTP prefill.
    ///
    /// Allocated when `ATLAS_MTP_LASTK_PREFILL=N` (N>0). Layout:
    /// `[K × hidden_size × fp_size]` (BF16 or FP32 depending on residual mode)
    /// — the last K target-side hidden states from the final prefill chunk are
    /// copied here so the MTP head can replay them through `forward_one` to
    /// populate its own KV cache before the first decode. Without this, MTP's
    /// self-attention sees zero prompt context at long ctx, dropping draft
    /// accept from 1.83 → 0.92 (target_seq=5085 vs mtp_seq=423 at 4K-prompt
    /// + 1K decode).
    pub(super) mtp_lastk_buf: Option<DevicePtr>,
    /// Capacity (K) of `mtp_lastk_buf`. Read once from
    /// `ATLAS_MTP_LASTK_PREFILL` at model init; 0 disables the feature.
    pub(super) mtp_lastk_capacity: usize,
    /// Immutable, strictly parsed loaded-serial qualification control.
    ///
    /// The proposer and `mtp_lastk_buf` remain allocated when this is true so
    /// loaded-serial controls preserve the candidate's memory footprint, but
    /// capture/replay return before host-ring, rotary, proposer, or GPU work.
    pub(super) speculation_forced_disabled: bool,
    /// `ATLAS_MTP_LASTK_DEFER` (K8 item A): the MTP last-K replay `finalize_last`
    /// recorded instead of running; `finish_deferred_prefill_tail` runs it after
    /// the first token. At most one (the decision requires a single live sequence).
    pub(super) k8a_lastk_deferred: Mutex<Option<super::host_bucket::DeferredLastK>>,
    /// K9 item X: the (in, out) events that order the deferred replay against the
    /// default stream (`host_bucket::replay_order_plan`); created on first use, `None`
    /// when the backend has no events.
    pub(super) k9x_lastk_events: std::sync::OnceLock<Option<(u64, u64)>>,
    /// Native Qwen4 frame capability per width (index = rows,
    /// `0..=QWEN4_NATIVE_MAX_ROWS`, C0-C256), computed once from fixed process
    /// facts (`decode_qwen4_capability.rs`).
    pub(super) qwen4_native_capability_cache:
        Box<[std::sync::OnceLock<std::result::Result<(), crate::qwen4_batch_policy::NativeGap>>]>,
    /// DFlash 5-layer hidden-state stack. Allocated only when a
    /// `BlockDiffusionDraftHead` proposer is built. Layout:
    /// `[dflash_hidden_save_rows × n_capture × dflash_capture_width × bf16]`
    /// shallow-to-deep at the layer indices declared by `dflash_capture_layers`.
    /// Holds the most-recently-decoded / verified rows' intermediate hiddens;
    /// the drafter consumes them via its `fc` projection on the next propose()
    /// call or batched capture retirement. None for non-DFlash runs.
    pub(super) dflash_hidden_save: Option<DevicePtr>,
    /// Number of token rows allocated in `dflash_hidden_save` (`0` when `None`).
    /// Sized to `k_max.max(QWEN4_CROSS_SEQ_MAX_ROWS)` when cross-sequence verify
    /// lanes are armed so batched verify and K=1 batched bootstrap capture
    /// without aliasing.
    pub(crate) dflash_hidden_save_rows: usize,
    /// CROSS-SEQ BATCHED DFLASH VERIFY (#39): lazily-allocated `[c*K, H]` BF16
    /// scratch that collects every sequence's post-mixer FFN input per layer so
    /// the batched verify can run ONE FFN GEMM across all sequences' rows.
    /// Allocated on first use (sized `max_batch_tokens × H × bf16`), `None`
    /// until the batched verify path fires.
    pub(super) dflash_batched_ffn_input: parking_lot::Mutex<Option<DevicePtr>>,
    /// Layer indices to capture for DFlash. Empty when DFlash is disabled.
    /// Sourced from drafter's `dflash_config.target_layer_ids` at model build.
    pub(super) dflash_capture_layers: Vec<usize>,
    /// Opt-in (`ATLAS_DFLASH_EARLY_STAGE1=1`) CUDA event recorded inside target
    /// verify right after the final DFlash capture layer (e.g. Layer 61 on 27B
    /// or Layer 45 on Flash-Next) so Stage-1 `fc` + `hidden_norm` can overlap
    /// with the trailing target layers + LM head + verify D2H. `0` when OFF.
    pub(super) dflash_capture_done_event: u64,
    /// Width and source offset for each captured DFlash slice. Ordinary
    /// targets use the entire residual row; compatibility bridges may expose
    /// a narrower explicit slice.
    pub(super) dflash_capture_width: usize,
    pub(super) dflash_capture_offset: usize,
    /// Target-side transform selected after validating the drafter pairing.
    pub(super) dflash_capture_mode: atlas_core::config::DflashCaptureMode,
    /// Cached CUDA graphs for K=2 verification, **keyed by `seq.slot_idx`**.
    /// Same rationale as `decode_graph`: the captured graph has SSM
    /// h_state/conv_state pointers baked in as kernel arguments, so replay for
    /// a different slot writes to the wrong sequence's recurrent state. With
    /// concurrent users alternating through MTP verify, a single
    /// `Option<GraphHandle>` would corrupt both slots' SSM state.
    pub(super) verify2_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for K=3 verification, keyed by `seq.slot_idx`.
    pub(super) verify3_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for K=4 verification, keyed by `seq.slot_idx`.
    pub(super) verify4_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for K=γ verification, keyed by
    /// `(slot_idx, k, pack_key, qwen4_k5_attention_topology)` (see
    /// [`VerifyKgammaGraphKey`]). `pack_key` carries 11 mode bits (`0..=10`:
    /// tree KV pack, tree indirection, layer-0 prefix graph, QSA regime,
    /// sampling/TC modes, etc.) so incompatible kernel-arg bindings never alias
    /// across captured graphs.
    pub(super) verify_kgamma_graph: Mutex<
        std::collections::HashMap<
            (usize, usize, u32, Option<Qwen4K5DeviceAttentionTopology>),
            GraphHandle,
        >,
    >,
    /// Monotonic LRU tick tracker paired with `verify_kgamma_graph` for bounded
    /// per-slot QSA-regime and global LRU eviction without changing
    /// `verify_kgamma_graph`'s map type.
    pub(super) verify_kgamma_lru: Mutex<VerifyKgammaGraphLru>,
    /// Prefix cache for KV block reuse across requests.
    pub(super) prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
    /// Secondary CUDA stream for pipelining checkpoint D2D with MTP propose.
    pub(super) secondary_stream: u64,
    /// CUDA event for GPU-side inter-stream synchronization (avoids CPU-blocking sync).
    pub(super) secondary_event: u64,
    /// Communication backend for expert parallelism (EP) all-reduce.
    /// None for single-GPU (no distributed communication needed).
    pub(super) comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
    /// Small GPU buffer for EP token broadcast (4 bytes).
    pub(super) ep_cmd_buf: DevicePtr,
    /// Additional sequence slots lazily allocated on multi-rank workers (`C > 1`).
    pub(super) worker_extra_slots: Mutex<HashMap<usize, crate::traits::SequenceState>>,
    /// Head-rank (`comm.rank() == 0 && world_size > 1`) command wire state so
    /// per-step active slot metadata stays in 1:1 lockstep with `ep_worker_step_impl`.
    pub(super) ep_wire_state: Mutex<EpHeadWireState>,
    /// Sequence slots freed on head rank awaiting `0xFFFFFFF1` worker notification.
    pub(super) pending_freed_slots: Mutex<std::collections::VecDeque<usize>>,
    /// Self-speculative decoding mode: draft via layer-skipping (no MTP weights needed).
    pub(super) self_speculative: bool,
    /// Last token index passed to save_hidden_for_mtp (for EP broadcast to rank 1).
    pub(super) last_mtp_hidden_idx: std::sync::atomic::AtomicUsize,
    /// Optional vision encoder for VL models (Qwen3-VL).
    pub(super) vision_encoder: Option<crate::layers::VisionEncoder>,
    /// C1-only owned final-merger aggregate, coherent grids and exact input cache key.
    /// Encoder scratch and DeepStack rows are never consumed as image tokens.
    pub(super) vision_embeddings: Mutex<super::vision_embeddings::VisionEmbeddingState>,
    /// Save SSM snapshots every N blocks during chunked prefill.
    /// 0 = disabled (leaf-only). When > 0, intermediate checkpoints are saved
    /// at block boundaries, enabling partial prefix SSM restore.
    pub(super) ssm_checkpoint_interval: usize,
    /// Kernel handle for fused SSM state normalization (prevents state explosion
    /// during long chunked prefill — the SSM forgetting bug).
    pub(super) ssm_state_norm_kernel: KernelHandle,
    /// GPU buffer for ssm_state_clamp_norm_fused's pointer table `[num_ssm_layers]`.
    pub(super) ssm_norm_ptrs_buf: DevicePtr,

    /// One-shot FP32 -> FP16 h-state converter (`ATLAS_SSM_H_FP16`, stage 2).
    pub(super) ssm_h_f32_to_f16_kernel: KernelHandle,
    /// Staging buffer for it, one layer wide (`h_bytes / 2`).
    ///
    /// The conversion is a narrowing COMPACTION and cannot be done in place:
    /// thread `2i`'s write lands inside thread `i`'s read with nothing ordering
    /// them. Allocated lazily on first use, so a serve without the flag pays
    /// nothing.
    pub(super) ssm_h_f16_scratch: std::sync::OnceLock<DevicePtr>,
    /// `ATLAS_ALLOC_SEQ_ASYNC` (K10 item A): recorded on the default stream after
    /// a new slot's zeroing memsets; the caller's prefill stream waits on it.
    /// Created on first use.
    pub(super) slot_zero_event: std::sync::OnceLock<u64>,
    /// One-shot FP16 -> FP32 h-state widener (`ATLAS_SSM_H_FP16`, stage 2).
    ///
    /// Used by the full-sequence disk-swap path so the swap FILE is always
    /// FP32 regardless of what the slot held. See `ssm_h_f32_scratch`.
    pub(super) ssm_h_f16_to_f32_kernel: KernelHandle,
    /// Staging buffer for the widener, one layer wide (`h_bytes`).
    ///
    /// Widening is an EXPANSION and likewise cannot be done in place.
    pub(super) ssm_h_f32_scratch: std::sync::OnceLock<DevicePtr>,
    // ── Two-phase SSM prefill buffers ──
    // These hold GDN inputs/outputs for the full sequence, allowing the GDN
    // recurrence to run in a single kernel launch while GEMM projections are
    // processed in smaller chunks (memory-bounded).
    //
    // Allocated at model init for max_seq_len tokens. Reused across layers
    // (only one layer runs at a time) and across sequences.
    /// Packed QKV for two-phase SSM prefill: [max_seq_len, conv_dim] BF16.
    /// Layout per token: [Q(key_dim) | K(key_dim) | V(value_dim)].
    pub(super) gdn_buf_qkv: DevicePtr,
    /// Interleaved gate/beta for two-phase SSM prefill: [max_seq_len, 2*num_v_heads] FP32.
    /// Layout per token: [gate(nv) | beta(nv)].
    pub(super) gdn_buf_gate_beta: DevicePtr,
    /// Full-sequence GDN output: [max_seq_len, value_dim] BF16
    pub(super) gdn_buf_out: DevicePtr,
    /// Full-sequence Z gate (for gated RMS norm in phase 3): [max_seq_len, value_dim] BF16
    pub(super) gdn_buf_z: DevicePtr,
    /// Max sequence length these buffers were allocated for.
    pub(super) gdn_buf_max_len: usize,

    /// Logit softcapping kernel: logits = cap * tanh(logits / cap).
    /// KernelHandle(0) = disabled (no softcapping for this model).
    pub(super) logit_softcap_kernel: KernelHandle,
    /// FP32 variant of logit softcap. KernelHandle(0) when not loaded.
    /// Used when `use_fp32_logits` is true.
    pub(super) logit_softcap_fp32_kernel: KernelHandle,
    /// Whether the single-token decode LM head produces FP32 logits (rather
    /// than BF16). True when `config.use_fp32_residual()` AND the LM head is
    /// a dense BF16 weight (no NVFP4 quant). Drives:
    ///   - dense_gemv_bf16_fp32out kernel writes to `logits_fp32_buf`
    ///   - logit_softcap_fp32 kernel applied in place on the FP32 buffer
    ///   - sampler reads FP32 directly, skipping BF16→FP32 expansion
    ///
    /// Gated by config.model_type=="gemma4" via use_fp32_residual() — other
    /// models keep the BF16 path. Single-token decode and single-stream
    /// prefill (`prefill_a`/`b`/`c` via `self.lm_head()`) write FP32 to
    /// `logits_fp32_buf`; multi-row batched decode/verify paths (`lm_head_batched`)
    /// write BF16 to `buffers.logits()`.
    pub(super) use_fp32_logits: bool,
    /// FP32 logits scratch [vocab_size × 4 bytes]. NULL when `use_fp32_logits`
    /// is false (no allocation).
    pub(super) logits_fp32_buf: DevicePtr,
    /// Embedding scale kernel: embeddings *= sqrt(hidden_size).
    /// KernelHandle(0) = disabled (no scaling for this model).
    pub(super) embed_scale_kernel: KernelHandle,
}

impl TransformerModel {
    /// Stable per-instance identity key derived from the non-empty heap
    /// allocation backing `qwen4_native_capability_cache` (invariant across
    /// moves and unique per live `TransformerModel`, cleaned up in `Drop`).
    #[inline]
    pub(crate) fn instance_key(&self) -> usize {
        self.qwen4_native_capability_cache.as_ptr() as usize
    }
}

/// Cache key for [`TransformerModel::verify_kgamma_graph`]:
/// `(slot_idx, k, pack_key, qwen4_k5_attention_topology)`.
pub(super) type VerifyKgammaGraphKey = (usize, usize, u32, Option<Qwen4K5DeviceAttentionTopology>);

/// Monotonic LRU tick tracker paired with [`TransformerModel::verify_kgamma_graph`].
#[derive(Debug, Default)]
pub(super) struct VerifyKgammaGraphLru {
    pub(super) tick: u64,
    pub(super) last_used: HashMap<VerifyKgammaGraphKey, u64>,
}

impl VerifyKgammaGraphLru {
    pub(super) fn touch(&mut self, key: VerifyKgammaGraphKey) {
        self.tick = self.tick.saturating_add(1);
        self.last_used.insert(key, self.tick);
    }

    pub(super) fn remove(&mut self, key: &VerifyKgammaGraphKey) {
        self.last_used.remove(key);
    }

    pub(super) fn retain_live(&mut self, live: &HashMap<VerifyKgammaGraphKey, GraphHandle>) {
        self.last_used.retain(|key, _| live.contains_key(key));
    }

    #[must_use]
    pub(super) fn last_used_tick(&self, key: &VerifyKgammaGraphKey) -> u64 {
        self.last_used.get(key).copied().unwrap_or(0)
    }
}

/// Pinned host memory staging buffer with reusable metadata Vecs.
pub(crate) struct PinnedMetaStaging {
    /// Backend-owned page-locked host buffer.
    pub(super) buffer: PinnedHostBuffer,
    /// Start offset of the fast-H2D ring region in `buffer` (disjoint from `[0..h2d_ring_base)`).
    pub(super) h2d_ring_base: usize,
    /// Current write cursor within `[h2d_ring_base..buffer.len())`.
    pub(super) h2d_ring_cursor: usize,
    /// Reusable `Vec<u32>` for positions (avoids per-chunk heap allocation).
    pub(super) positions: Vec<u32>,
    pub(super) positions_h: Vec<u32>,
    pub(super) positions_w: Vec<u32>,
    /// Reusable `Vec<i64>` for slot mappings (avoids per-chunk heap allocation).
    pub(super) slots: Vec<i64>,
}

impl PinnedMetaStaging {
    /// Carve an 8-byte-aligned slice of `bytes` from the dedicated fast-H2D
    /// ring tail `[h2d_ring_base..buffer.len())`. Returns `None` if `bytes`
    /// exceeds the ring capacity.
    #[inline]
    pub(super) fn reserve_h2d_range(&mut self, bytes: usize) -> Option<usize> {
        let base = self.h2d_ring_base;
        let cap = self.buffer.len();
        let ring_len = cap.checked_sub(base)?;
        if bytes > ring_len {
            return None;
        }
        let aligned_cursor = (self.h2d_ring_cursor + 7) & !7;
        let start = if aligned_cursor.checked_add(bytes)? <= cap {
            aligned_cursor
        } else {
            (base + 7) & !7
        };
        let end = start.checked_add(bytes)?;
        if end > cap {
            return None;
        }
        self.h2d_ring_cursor = end;
        Some(start)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum EpHeadWireState {
    #[default]
    Idle,
    PrefillArgs {
        remaining_u32: u8,
    },
    PrefillTokensPending,
    PrefillReady,
    VerifyFixedArgs {
        remaining_u32: u8,
    },
    VerifyFixedReady,
    VerifyKgammaLen,
    VerifyKgammaTokens,
    VerifyKgammaReady,
    VerifyResultPending,
    DecodeReady,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TransformerModel>();
};
