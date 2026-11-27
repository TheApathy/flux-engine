// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{DflashCaptureMode, LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_pool_geometry::{DDTREE_KERNEL_KMAX, checked_ssm_speculative_geometry};
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub fn new(
        config: ModelConfig,
        embed_tokens: DenseWeight,
        final_norm: DenseWeight,
        qwen4_final_mixer: Option<crate::layers::Qwen4HyperConnection>,
        #[cfg(all(feature = "cuda", target_os = "linux"))] qwen4_ple: Option<
            crate::layers::Qwen4PleLayer,
        >,
        lm_head_weight: DenseWeight,
        lm_head_nvfp4: Option<QuantizedWeight>,
        layers: Vec<Box<dyn TransformerLayer>>,
        buffers: BufferArena,
        kv_cache: PagedKvCache,
        mtp_weights: Vec<MtpWeights>,
        mtp_dense_weights: Option<crate::weight_map::MtpDenseWeights>,
        qwen4_mtp_proposer: Option<Arc<dyn DraftProposer>>,
        gpu: std::sync::Arc<dyn GpuBackend>,
        max_seq_len: usize,
        max_batch_size: usize,
        mtp_quant: crate::layers::MtpQuantization,
        use_native_mtp: bool,
        prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
        mtp_vocab_size: u32,
        comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
        self_speculative: bool,
        num_drafts: usize,
        vision_encoder: Option<crate::layers::VisionEncoder>,
        ssm_cache_slots: usize,
        ssm_checkpoint_interval: usize,
    ) -> Result<Self> {
        let final_norm_identity = config.final_norm_is_identity();
        super::final_norm::check_final_norm_identity(&config, qwen4_final_mixer.as_ref())?;
        let speculation_forced_disabled = super::speculation_control::forced_disabled_from_env()?;
        let fp32_residual = config.use_fp32_residual();
        let rms_norm_kernel = if fp32_residual {
            gpu.kernel("norm", "rms_norm_f32")
                .or_else(|_| gpu.kernel("norm", "rms_norm"))?
        } else {
            gpu.kernel("norm", "rms_norm")?
        };
        let bf16_to_f32_kernel = if fp32_residual {
            gpu.kernel("residual_add", "bf16_to_f32")
                .unwrap_or(KernelHandle(0))
        } else {
            KernelHandle(0) // BF16 models don't need conversion
        };
        let dense_gemv_kernel = gpu.kernel("gemv", "dense_gemv_bf16")?;
        // FP32-output dense GEMV — only loaded when LM head needs FP32 logits.
        // For models that don't use FP32 residual, this stays KernelHandle(0)
        // and the BF16 path is taken. The kernel lives in the same `gemv`
        // module as `dense_gemv_bf16` so this lookup is cheap.
        let dense_gemv_fp32out_kernel = if fp32_residual {
            gpu.kernel("gemv", "dense_gemv_bf16_fp32out")
                .unwrap_or(KernelHandle(0))
        } else {
            KernelHandle(0)
        };
        let w4a16_gemv_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;
        let w4a16_gemv_sw_kernel =
            crate::layers::try_kernel(gpu.as_ref(), "w4a16_gemv", "w4a16_gemv_sw");
        let gemv_sw = ops::gemv_sw_enabled();
        let w4a16_gemv_logits_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_logits")?;
        let w4a16_exact_lm_head_kernels = ops::W4a16ExactLmHeadKernels::new(
            crate::layers::try_kernel(
                gpu.as_ref(),
                "w4a16_gemv",
                ops::ExactLmHeadTier::M4.symbol(),
            ),
            crate::layers::try_kernel(
                gpu.as_ref(),
                "w4a16_gemv",
                ops::ExactLmHeadTier::M8.symbol(),
            ),
            crate::layers::try_kernel(
                gpu.as_ref(),
                "w4a16_gemv",
                ops::ExactLmHeadTier::M17.symbol(),
            ),
            crate::layers::try_kernel(
                gpu.as_ref(),
                "w4a16_gemv",
                ops::ExactLmHeadTier::M32.symbol(),
            ),
        )
        .with_rt2(
            crate::layers::try_kernel(
                gpu.as_ref(),
                "w4a16_gemv_rt",
                ops::ExactLmHeadTier::M4.symbol_rt2(),
            ),
            crate::layers::try_kernel(
                gpu.as_ref(),
                "w4a16_gemv_rt",
                ops::ExactLmHeadTier::M8.symbol_rt2(),
            ),
            crate::layers::try_kernel(
                gpu.as_ref(),
                "w4a16_gemv_rt",
                ops::ExactLmHeadTier::M17.symbol_rt2(),
            ),
            crate::layers::try_kernel(
                gpu.as_ref(),
                "w4a16_gemv_rt",
                ops::ExactLmHeadTier::M32.symbol_rt2(),
            ),
        )
        .with_wide_n_sw4(crate::layers::try_kernel(
            gpu.as_ref(),
            "w4a16_gemv",
            "w4a16_gemv_batch_logits_exact_sw_m8_o4w4",
        ))
        .with_wide_n_sw4_m10(crate::layers::try_kernel(
            gpu.as_ref(),
            "w4a16_gemv",
            "w4a16_gemv_batch_logits_exact_sw_m10_o4w4",
        ))
        .with_wide_n_sw4_m15_m25(
            crate::layers::try_kernel(
                gpu.as_ref(),
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_sw_m15_o2w4",
            ),
            crate::layers::try_kernel(
                gpu.as_ref(),
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_sw_m25_o1w4",
            ),
        );
        tracing::info!(
            target: "atlas::lm_head",
            policy = "exact_dynamic_m_or_serial_k1",
            m4 = w4a16_exact_lm_head_kernels.is_present(ops::ExactLmHeadTier::M4),
            m8 = w4a16_exact_lm_head_kernels.is_present(ops::ExactLmHeadTier::M8),
            m17 = w4a16_exact_lm_head_kernels.is_present(ops::ExactLmHeadTier::M17),
            m32 = w4a16_exact_lm_head_kernels.is_present(ops::ExactLmHeadTier::M32),
            rt2 = ops::w4a16_gemv_rt2_enabled(),
            "LM_HEAD_EXACT_STARTUP"
        );
        let w4a16_gemm_kernel = gpu.kernel("w4a16", "w4a16_gemm")?;
        let w4a16_gemm_t_m32_n64_kernel =
            crate::layers::try_kernel(gpu.as_ref(), "w4a16", "w4a16_gemm_t_m32_n64");
        let w4a16_gemv_batch2_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?;
        let w4a16_gemv_batch3_logits_kernel =
            gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3_logits")?;
        let dense_gemm_kernel = gpu.kernel("gemm", "dense_gemm_bf16")?;
        let argmax_kernel = gpu.kernel("argmax", "argmax_bf16")?;
        let argmax_rows_kernel =
            crate::layers::try_kernel(gpu.as_ref(), "argmax", "argmax_bf16_rows");
        let argmax_logits_kernel = gpu.kernel("argmax", "argmax_fp32")?;
        let (topk_candidates_kernel, f2_sum_kernel, topk_candidates_buf) =
            crate::model::trait_impl::device_topk::load_device_topk(gpu.as_ref())?;
        let (target_gumbel_verify_kernel, target_gumbel_verify_buf) =
            crate::model::trait_impl::device_topk::load_batched_gumbel_verify(gpu.as_ref())?;
        let (target_rejection_verify_kernel, target_rejection_verify_buf, target_rejection_q_buf) =
            crate::model::trait_impl::device_topk::load_batched_rejection_verify(gpu.as_ref())?;
        let batched_embed_kernel = if fp32_residual {
            gpu.kernel("embed_from_argmax", "batched_embed_f32")
                .or_else(|_| gpu.kernel("embed_from_argmax", "batched_embed"))?
        } else {
            gpu.kernel("embed_from_argmax", "batched_embed")?
        };
        let fill_slots_kernel = gpu.kernel("metadata_fill", "fill_slots_from_block_table")?;
        let profile = std::env::var("ATLAS_PROFILE").is_ok();
        let profile_first = std::env::var("ATLAS_PROFILE_FIRST").is_ok();

        tracing::info!(
            "TransformerModel: {} layers, vocab={}, hidden={}{}{}",
            layers.len(),
            config.vocab_size,
            config.hidden_size,
            if profile { " [PROFILE MODE]" } else { "" },
            if profile_first {
                " [PROFILE_FIRST]"
            } else {
                ""
            },
        );

        // Build SSM state pool (with MTP intermediate/checkpoint pools only if speculative decoding enabled)
        // num_intermediates = K (per-token SSM h/conv state snapshots).
        // For MTP K=2/3/4 verify: K = num_drafts + 1.
        // For DFlash K=γ verify: K = γ + 1 (drafter's γ drafts + 1 verified bonus slot).
        // Pool size = max of both so DFlash and MTP can coexist on the same model.
        let dflash_enabled = !config.dflash_capture_layers.is_empty();
        // ── DDTree wide-tree verify capacity (ATLAS_DDTREE_MAX_NODES) ──
        // The drafter emits γ draft positions, so the FLAT verify width is
        // `dflash_kgamma` (= γ+1). But a DDTree branch tree can hold MORE
        // nodes than γ (top-k siblings expanded per depth, up to a budget).
        // `ddtree_cap` sizes every verify-side PERSISTENT buffer (SSM
        // intermediates, parent_ids, tree-KV indirection, hidden-save) to
        // admit a wider tree. It is clamped to the tree-WY kernel's
        // compile-time K_MAX=32. Default = dflash_kgamma ⇒ NO behavior change
        // unless the operator opts in; the flat/counting path keeps verifying
        // exactly `dflash_kgamma` tokens. When widened, flat verify (k <
        // ddtree_cap) routes through the proven wy_k path (the persistent
        // parent injection at `k == capacity` no longer matches at the flat
        // width — validated bit-identical to the tree_wy-linear-chain path).
        // DFlash needs the SSM verify pools regardless of MTP weight presence
        // or lm_head quantization — its K=γ verify path checkpoints SSM state
        // for partial-accept rollback. Force `has_mtp` on whenever DFlash is
        // active so the checkpoint pools exist.
        let multi_rank_worker_mtp = (use_native_mtp || dflash_enabled)
            && comm
                .as_ref()
                .is_some_and(|c| c.world_size() > 1 && c.rank() != 0);
        let has_mtp = self_speculative
            || (use_native_mtp && !mtp_weights.is_empty() && lm_head_nvfp4.is_some())
            || (use_native_mtp && mtp_dense_weights.is_some() && lm_head_nvfp4.is_some())
            || qwen4_mtp_proposer.is_some()
            || multi_rank_worker_mtp
            || dflash_enabled;
        let requested_ddtree_capacity = std::env::var("ATLAS_DDTREE_MAX_NODES")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok());
        // Opt-in `ATLAS_MTP_DLOOP=1` on dense native-MTP targets widens Stage-2
        // verify width to K=9 (`8` drafts + 1 anchor). Strictly zero-cost / no-op
        // when OFF or on MoE targets (refused fail-closed).
        let dloop_required_k = crate::layers::mtp_dloop::required_verify_k_for_model(
            crate::layers::mtp_dloop::mtp_dloop_enabled(),
            has_mtp,
            dflash_enabled,
            crate::layers::mtp_dloop::TargetFfnTopology::from_num_experts(config.num_experts),
        );
        let geometry_num_drafts = num_drafts.max(dloop_required_k.saturating_sub(1));
        let speculative_geometry = checked_ssm_speculative_geometry(
            has_mtp,
            dflash_enabled,
            geometry_num_drafts,
            requested_ddtree_capacity,
        )?;
        let dflash_kgamma = speculative_geometry.dflash_verify_width;
        let ddtree_cap = speculative_geometry.ddtree_capacity;
        let num_intermediates = speculative_geometry.num_intermediates;
        if ddtree_cap > dflash_kgamma {
            tracing::info!(
                "ATLAS_DDTREE_MAX_NODES: wide-tree verify capacity = {ddtree_cap} \
                 (flat width dflash_kgamma = {dflash_kgamma}, kernel K_MAX = {DDTREE_KERNEL_KMAX})"
            );
        }
        let compact_qwen4_dflash = config.is_qwen4_exp()
            && dflash_enabled
            && !self_speculative
            && num_intermediates <= 10
            && std::env::var("ATLAS_QWEN4_HYBRID_ALL_K").ok().as_deref() == Some("1")
            && std::env::var("ATLAS_DDTREE_TREE_AWARE_VERIFY")
                .ok()
                .as_deref()
                != Some("1");
        let native_qwen4_mtp =
            (qwen4_mtp_proposer.is_some() && !dflash_enabled && !self_speculative)
                || compact_qwen4_dflash;
        let ssm_pool = SsmStatePool::new_with_mode(
            &config,
            max_batch_size,
            has_mtp,
            native_qwen4_mtp,
            num_intermediates,
            gpu.as_ref(),
        )?;

        // DDTree graph-safe persistent `parent_ids` buffer.
        //
        // The K=γ verify CUDA graph reads parent_ids from a FIXED device
        // pointer baked into the captured kernel arguments. Allocating a
        // fresh scratch buffer per verify (the pre-fix behavior in
        // set_ddtree_parent_ids) made the captured pointer stale on
        // replay → kernel read garbage. We allocate ONCE here and let
        // both default (linear-chain) and tree-mode payloads share the
        // same address, mutating only the contents.
        //
        // Capacity = ddtree_cap (at least dflash_kgamma). Tree-mode upload
        // never exceeds that configured cap. Pre-stamped with the linear chain
        // `[-1, 0, 1, ..., K-2]` so flat-payload verify reuses tree_wy
        // without an upload (zero-copy bit-equivalence path).
        let (parent_ids_persistent, parent_ids_capacity) = if dflash_kgamma > 0 {
            let cap = ddtree_cap;
            let bytes = cap * std::mem::size_of::<i32>();
            let buf = gpu.alloc(bytes)?;
            // Stamp the linear-chain default: parents[0] = -1, parents[i] = i-1.
            let mut chain = Vec::<i32>::with_capacity(cap);
            chain.push(-1);
            for i in 1..cap {
                chain.push((i - 1) as i32);
            }
            // SAFETY: `chain` is a live initialized `Vec<i32>` of length `cap`
            // (`size_of_val` bytes, POD); `u8` has alignment 1 and the slice
            // borrows `chain` immutably only across `copy_h2d_group_on_stream`.
            let byte_view = unsafe {
                std::slice::from_raw_parts(
                    chain.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(chain.as_slice()),
                )
            };
            gpu.copy_h2d_group_on_stream(
                &[HostToDeviceCopy::new(byte_view, buf)],
                gpu.default_stream(),
            )?;
            tracing::info!(
                "DDTree: allocated graph-safe parent_ids buffer ({cap} slots, ptr={:?})",
                buf
            );
            (buf, cap)
        } else {
            (DevicePtr::NULL, 0usize)
        };

        // ATLAS_TREE_AWARE_ATTN persistent indirection buffer.
        //
        // For K=γ verify, when tree-mode is active and the user opted in via
        // `ATLAS_TREE_AWARE_ATTN=1`, `trait_impl/verify_d.rs` builds a per-row
        // indirection table that maps each query's compact tree slot back to
        // its ancestor chain. Layout: `[num_rows=ddtree_cap, stride=ddtree_cap]`
        // i32, row-major. The kernel reads `indir[seq_idx*stride + (pos - base)]`
        // to remap positions in the tree window.
        //
        // Pre-filled with identity (`indir[t][j] = j`) so a stale-but-graph-
        // captured pointer always behaves like chain-mode if someone forgets
        // to update it before launch.
        let (tree_kv_indir_persistent, tree_kv_indir_stride) = if dflash_kgamma > 0 {
            let stride = ddtree_cap;
            let cells = stride * stride;
            let bytes = cells * std::mem::size_of::<i32>();
            let buf = gpu.alloc(bytes)?;
            let mut identity = Vec::<i32>::with_capacity(cells);
            for _row in 0..stride {
                for j in 0..stride {
                    identity.push(j as i32);
                }
            }
            // SAFETY: `identity` is a live initialized `Vec<i32>` of length
            // `cells` (`size_of_val` bytes, POD); `u8` has alignment 1 and the
            // slice borrows `identity` immutably only across
            // `copy_h2d_group_on_stream`.
            let byte_view = unsafe {
                std::slice::from_raw_parts(
                    identity.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(identity.as_slice()),
                )
            };
            gpu.copy_h2d_group_on_stream(
                &[HostToDeviceCopy::new(byte_view, buf)],
                gpu.default_stream(),
            )?;
            tracing::info!(
                "ATLAS_TREE_AWARE_ATTN: allocated indirection buffer ({}x{} i32, ptr={:?})",
                stride,
                stride,
                buf
            );
            (buf, stride)
        } else {
            (DevicePtr::NULL, 0usize)
        };

        // ATLAS_TREE_AWARE_ATTN CUDA graph fix: persistent 1×i32 device buffer
        // holding the current tree-window base position. The kernel reads this
        // via pointer arg so a captured graph sees the fresh value on each
        // replay. Pre-stamped to 0 (matches the chain-mode default); the host
        // writes the current `seq.seq_len` here before each K=γ verify step.
        //
        // A pinned-host shadow buffer is also allocated so the per-step H2D
        // upload establishes a proper stream-ordered dependency for the
        // captured graph kernels. A pageable-source (Vec-backed) async copy
        // can race ahead of the kernel launch on graph replay paths.
        let (tree_kv_indir_base_persistent, tree_kv_indir_base_host_pinned) = if dflash_kgamma > 0 {
            let buf = gpu.alloc(std::mem::size_of::<i32>())?;
            gpu.memset_async(buf, 0, std::mem::size_of::<i32>(), gpu.default_stream())?;
            gpu.synchronize(gpu.default_stream())?;
            let mut host = gpu.alloc_host_pinned(std::mem::size_of::<i32>())?;
            // Pre-zero the pinned shadow so any pre-tree verify (rare) is
            // consistent with the device buffer's zero-init.
            host.as_mut_slice().fill(0);
            tracing::info!(
                "ATLAS_TREE_AWARE_ATTN: allocated kv_indir_base buffer (1xi32, dev={:?}, host_pinned={:?})",
                buf,
                host.as_ptr()
            );
            (buf, Some(host))
        } else {
            (DevicePtr::NULL, None)
        };

        // ATLAS_TREE_KV_PACK: per-attention-layer packed-KV scratch pool.
        //
        // When `ATLAS_TREE_KV_PACK=1` + `ATLAS_TREE_AWARE_ATTN=1`, the K=γ
        // verify path scatters ancestor KV from the paged cache into a
        // contiguous `[num_seqs × stride]` block per layer, then re-runs
        // paged_decode_attn against the scratch with NULL indirection so
        // the fast BC=4 batched path fires (vs the 3.6x-slower
        // single-position fallback inside the indirected kernel).
        //
        // Sizing (qwen3.6-27b, FP8, stride=17, num_seqs=17, nkv=4, hd=128):
        //   per-layer K bytes = num_seqs * stride * nkv * hd * elem_bytes
        //                     = 17 * 17 * 4 * 128 * 1 = 147,968 B (~145 KB)
        //   per-layer (K+V)   = ~290 KB
        //   16 attn layers    = ~4.6 MB total (negligible).
        //
        // NVFP4: per-block = data + scale section, similar order of magnitude.
        let tree_kv_pack_active_env =
            std::env::var("ATLAS_TREE_KV_PACK").ok().as_deref() == Some("1");
        let (
            tree_kv_pack_scratch_k,
            tree_kv_pack_scratch_v,
            tree_kv_pack_block_table,
            tree_kv_pack_seq_lens,
            tree_kv_pack_block_stride_bytes,
            tree_kv_pack_data_section_bytes,
            tree_kv_pack_scatter_fp8_kernel,
            tree_kv_pack_scatter_nvfp4_kernel,
            tree_kv_pack_active,
        ) = if tree_kv_pack_active_env && dflash_kgamma > 0 {
            let num_attn = config.num_attention_layers();
            // num_seqs sized by k_max (= γ+1). Conservative; matches the K=γ
            // verify batch dimension. The scatter kernel only writes the rows
            // actually used per step.
            let num_seqs = ddtree_cap;
            let stride = ddtree_cap; // = max_chain_len (wide-tree capacity)
            let kv_dtype = kv_cache.dtype();
            // Compute per-block bytes following the same layout the
            // paged_decode_attn kernels expect for the given dtype.
            let nkv = config.num_key_value_heads;
            let hd = config.head_dim;
            let (block_bytes, data_section_bytes) = match kv_dtype {
                spark_runtime::kv_cache::KvCacheDtype::Nvfp4 => {
                    // data + per-group FP8 scales (group size 16)
                    let elems = stride * nkv * hd;
                    let data = elems / 2;
                    let scales = elems / 16;
                    ((data + scales) as u64, data as u64)
                }
                spark_runtime::kv_cache::KvCacheDtype::Fp8 => {
                    let elems = stride * nkv * hd;
                    (elems as u64, 0u64)
                }
                _ => {
                    // Other dtypes (BF16, Turbo*) not yet wired for pack.
                    (0u64, 0u64)
                }
            };
            if block_bytes == 0 {
                tracing::warn!(
                    "ATLAS_TREE_KV_PACK requested but KV dtype {:?} not supported — disabling",
                    kv_dtype
                );
                (
                    Vec::new(),
                    Vec::new(),
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    0u64,
                    0u64,
                    KernelHandle(0),
                    KernelHandle(0),
                    false,
                )
            } else {
                let pool_bytes_per_layer = block_bytes as usize * num_seqs;
                let mut scratch_k = Vec::with_capacity(num_attn);
                let mut scratch_v = Vec::with_capacity(num_attn);
                for _ in 0..num_attn {
                    scratch_k.push(gpu.alloc(pool_bytes_per_layer)?);
                    scratch_v.push(gpu.alloc(pool_bytes_per_layer)?);
                }
                // Identity block table: `bt[seq] = seq` (`num_seqs` i32 entries,
                // `max_blocks_per_seq = 1`, one synthetic block per sequence).
                // The consumer kernel passes `1` instead of the real
                // `max_blocks_per_seq`.
                let bt_entries = num_seqs;
                let bt_bytes = bt_entries * std::mem::size_of::<i32>();
                let bt_buf = gpu.alloc(bt_bytes)?;
                let identity: Vec<i32> = (0..num_seqs as i32).collect();
                // SAFETY: `identity` is a live initialized `Vec<i32>` of length
                // `num_seqs` (`size_of_val` bytes, POD); `u8` has alignment 1
                // and the slice borrows `identity` immutably only across
                // `copy_h2d_group_on_stream`.
                let bt_view = unsafe {
                    std::slice::from_raw_parts(
                        identity.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(identity.as_slice()),
                    )
                };
                gpu.copy_h2d_group_on_stream(
                    &[HostToDeviceCopy::new(bt_view, bt_buf)],
                    gpu.default_stream(),
                )?;

                let seq_lens_bytes = num_seqs * std::mem::size_of::<i32>();
                let seq_lens_buf = gpu.alloc(seq_lens_bytes)?;
                gpu.memset_async(seq_lens_buf, 0, seq_lens_bytes, gpu.default_stream())?;
                gpu.synchronize(gpu.default_stream())?;

                let scatter_fp8 = gpu
                    .kernel("tree_kv_scatter", "tree_kv_scatter_fp8")
                    .unwrap_or_else(|e| {
                        tracing::warn!("tree_kv_scatter_fp8 not loadable: {e}");
                        KernelHandle(0)
                    });
                let scatter_nvfp4 = gpu
                    .kernel("tree_kv_scatter", "tree_kv_scatter_nvfp4")
                    .unwrap_or_else(|e| {
                        tracing::warn!("tree_kv_scatter_nvfp4 not loadable: {e}");
                        KernelHandle(0)
                    });
                let active = match kv_dtype {
                    spark_runtime::kv_cache::KvCacheDtype::Fp8 => scatter_fp8.0 != 0,
                    spark_runtime::kv_cache::KvCacheDtype::Nvfp4 => scatter_nvfp4.0 != 0,
                    _ => false,
                };
                tracing::info!(
                    "ATLAS_TREE_KV_PACK: allocated {} attn-layer scratch pools ({} bytes each K/V), \
                     stride={}, num_seqs={}, dtype={:?}, active={}",
                    num_attn,
                    pool_bytes_per_layer,
                    stride,
                    num_seqs,
                    kv_dtype,
                    active
                );
                (
                    scratch_k,
                    scratch_v,
                    bt_buf,
                    seq_lens_buf,
                    block_bytes,
                    data_section_bytes,
                    scatter_fp8,
                    scatter_nvfp4,
                    active,
                )
            }
        } else {
            (
                Vec::new(),
                Vec::new(),
                DevicePtr::NULL,
                DevicePtr::NULL,
                0u64,
                0u64,
                KernelHandle(0),
                KernelHandle(0),
                false,
            )
        };

        // Marconi prefix-cache snapshots plus the Phase-C decode-rollback
        // ring. Keep one boundary snapshot beyond the per-sequence rollback
        // cap; startup preflight reserves this exact ring for every batch slot.
        let _qwen4_snapshot_scope =
            super::ssm_pool_geometry::Qwen4SnapshotScope::enter(config.is_qwen4_exp());
        #[allow(unused_mut)]
        let mut ssm_snapshots = SsmSnapshotPool::new(
            ssm_cache_slots,
            ssm_pool.h_bytes,
            ssm_pool.conv_bytes,
            ssm_pool.num_ssm_layers,
            (atlas_kernels::ROLLBACK_RESTEER_CAP as usize) + 1,
            max_batch_size,
            gpu.as_ref(),
        )?;
        // Carry Qwen4 PLE conv history with every Marconi and decode-ring
        // snapshot: via `SlotHistory` when `ATLAS_QWEN4_SEQSTATE=1` is active,
        // or via `ple_conv_snapshots` by default (avoiding duplicate slabs).
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if crate::model::seqstate::enabled() && config.is_qwen4_exp() {
            if let Some(ple) = qwen4_ple.as_ref()
                && (ssm_snapshots.is_enabled() || ssm_snapshots.decode_rollback_enabled())
            {
                let [(live, stride), (checkpoint, _)] = ple.seqstate_slot_regions(0)?;
                ssm_snapshots.attach_slot_history(
                    gpu.as_ref(),
                    live,
                    checkpoint,
                    stride,
                    ple.seqstate_slot_count(),
                )?;
            }
        } else if let Some(ple) = &qwen4_ple {
            ssm_snapshots.attach_ple_conv_layers(
                vec![super::ssm_snapshot::PleConvLayerBinding {
                    conv_state: ple.conv_state_base_ptr(),
                    conv_checkpoint: ple.conv_checkpoint_base_ptr(),
                    max_slots: ple.max_batch_slots(),
                }],
                ple.conv_state_stride_bytes(),
                (atlas_kernels::ROLLBACK_RESTEER_CAP as usize) + 1,
                max_batch_size,
                ssm_cache_slots,
                gpu.as_ref(),
            )?;
        }
        drop(_qwen4_snapshot_scope);
        // ATLAS_SSM_H_FP16 x DFlash: REFUSE. Measured 2026-08-20.
        //
        // Only wy2/wy3/wy4 have f16 twins. The GDN h-state readers and writers
        // the DFlash arm actually runs — gated_delta_rule_tree,
        // gated_delta_rule_tree_wy and the wy17 family — are FP32 with no f16
        // variant anywhere in this tree or upstream. An FP32 kernel reading an
        // FP16 pool does not fault; it emits fluent garbage
        // (kernels/gb10/common/README-f16-ssm-h.md).
        //
        // Measured on the serving config, gamma=7, MinHeap probe:
        //     FP32   44.42 tok/s   accepted 4.78   content 49caed9e...
        //     FP16    9.18 tok/s   accepted 0.20   content 7d0c2014...
        // Different output hash: the flag changes what the model SAYS. Verify
        // time was unchanged (100.87 -> 100.95 ms), so it was not even reducing
        // state traffic — it was only corrupting it.
        //
        // Refuse rather than warn: the failure is silent, fluent and would be
        // attributed to the drafter, not to a dtype flag.
        if crate::layers::qwen3_ssm::ssm_h_fp16_enabled() && dflash_enabled {
            anyhow::bail!(
                "ATLAS_SSM_H_FP16=1 is incompatible with DFlash speculative decoding: \
                 the tree/wy17 GDN kernels this path runs are FP32-only and would read \
                 the FP16 pool as FP32, producing fluent but WRONG output (measured \
                 9.18 vs 44.42 tok/s and a different content hash). Unset ATLAS_SSM_H_FP16, \
                 or port gated_delta_rule_decode_f16_norm and the tree/wy17 f16 twins first."
            );
        }

        // ATLAS_SSM_H_FP16 x Marconi.
        //
        // Under the flag a DECODING slot holds FP16, but every snapshot on disk
        // and in the pool is FP32: `ssm_snapshots.save()` takes `h_is_f16` and
        // widens per layer via ssm_h_state_f16_to_f32. That keeps restore,
        // spill, fault-in, tier-fingerprint and swap all dtype-agnostic, so the
        // two features compose. The widener is resolved at pool construction;
        // `save()` bails if the flag is live and the kernel is missing rather
        // than writing half-width bytes into an FP32 snapshot.
        if crate::layers::qwen3_ssm::ssm_h_fp16_enabled() && ssm_cache_slots > 0 {
            tracing::info!(
                "ATLAS_SSM_H_FP16=1 with {ssm_cache_slots} Marconi slots: \
                 snapshots widen FP16->FP32 on save"
            );
        }

        if ssm_checkpoint_interval > 0 && ssm_cache_slots > 0 {
            tracing::info!(
                "Marconi intermediate checkpoints: every {} blocks ({} tokens at block_size={})",
                ssm_checkpoint_interval,
                ssm_checkpoint_interval * kv_cache.block_size(),
                kv_cache.block_size(),
            );
        }

        // Fixed metadata stride for CUDA graph compatibility
        let max_blocks_per_seq = (max_seq_len / kv_cache.block_size() + 1) as u32;

        // Permanent dummy KV block for padding sequences. Must be explicitly
        // zeroed: `gpu.alloc()` returns uninitialized memory, and any kernel
        // OOB-read (now routed here via the sentinel block_table_flat default
        // fill in upload_batch_metadata_*) would otherwise dequant random
        // bytes and inject garbage into attention scores.
        let mut kv_cache = kv_cache;
        let dummy_kv_block = kv_cache.alloc_block()?;
        kv_cache.zero_block(dummy_kv_block, gpu.as_ref(), gpu.default_stream())?;
        gpu.synchronize(gpu.default_stream())?;

        // ATLAS_LM_HEAD_T=1: transposed NVFP4 lm_head copy retained for
        // post-construction sharing with the DFlash drafter's propose head
        // (`dflash_lm_head_t`). Target verification no longer reads this
        // layout: `lm_head_batched` routes M=2..=32 through the exact
        // row-major family or independent K1 GEMVs. ~0.63 GB extra memory.
        let lm_head_nvfp4_t = if std::env::var("ATLAS_LM_HEAD_T").ok().as_deref() == Some("1")
            && w4a16_gemm_t_m32_n64_kernel.0 != 0
        {
            match lm_head_nvfp4.as_ref() {
                Some(w) => {
                    match w.transpose_for_gemm(gpu.as_ref(), config.vocab_size, config.hidden_size)
                    {
                        Ok(t) => {
                            tracing::info!(
                                "lm_head NVFP4-T built for DFlash drafter sharing (vocab={})",
                                config.vocab_size
                            );
                            Some(t)
                        }
                        Err(e) => {
                            tracing::warn!(
                                "lm_head transpose failed ({e:#}); DFlash drafter sharing and ATLAS_LM_HEAD_TC=1 disabled (verify stays on exact row-major LM head)"
                            );
                            None
                        }
                    }
                }
                None => None,
            }
        } else {
            None
        };

        // Build MTP proposer (extracted to keep `new` under the file cap).
        let proposer: Option<Arc<dyn DraftProposer>> = qwen4_mtp_proposer.or_else(|| {
            super::impl_a1_init::build_mtp_proposer(
                use_native_mtp,
                mtp_weights,
                mtp_dense_weights,
                embed_tokens,
                lm_head_nvfp4,
                &config,
                gpu.as_ref(),
                mtp_quant,
                mtp_vocab_size,
                max_seq_len,
            )
        });

        if self_speculative {
            let num_ssm = config.num_ssm_layers();
            let num_attn = config.num_attention_layers();
            tracing::info!(
                "Self-speculative decoding: ENABLED (skipping {} SSM layers, keeping {} attention layers)",
                num_ssm,
                num_attn,
            );
        }

        // Qwen4 MTP consumes the pre-final four-stream row in BF16; legacy
        // MTP consumes one hidden stream and may use FP32 residuals.
        let mtp_hidden_bytes = if config.is_qwen4_exp() {
            config.residual_width() * 2
        } else {
            config.hidden_size * 4
        };
        let mtp_hidden_save = gpu.alloc(mtp_hidden_bytes)?;
        let qwen4_mtp_hidden_frames = if config.is_qwen4_exp()
            && proposer.as_ref().is_some_and(|proposer| {
                proposer.kind() == crate::speculative::DraftProposerKind::Qwen4NativeMtp
            }) {
            let frames = super::qwen4_mtp_hidden::NativeMtpHiddenFrames::new(
                max_batch_size,
                num_intermediates,
                mtp_hidden_bytes,
            )
            .map_err(anyhow::Error::msg)?;
            let storage = gpu.alloc(frames.allocation_bytes())?;
            Some((storage, Mutex::new(frames)))
        } else {
            None
        };
        // ATLAS_QWEN4_BATCHED_K5 / ATLAS_DFLASH_BATCHED_VERIFY: allocate one
        // per-slot GDN-retain (and Qwen4 QSA-journal) verify lane per physical
        // slot. Fixed before any lazily allocated transaction buffer exists so
        // concurrent batched verifies never clobber lane 0.
        let qwen4_worker_batched_verify = config.is_qwen4_exp() && multi_rank_worker_mtp;
        let qwen4_batched_k5_armed = config.is_qwen4_exp()
            && (qwen4_mtp_hidden_frames.is_some()
                || qwen4_worker_batched_verify
                || !config.dflash_capture_layers.is_empty())
            && crate::layers::qwen4_verify_lanes::batched_k5_selected()?;
        let dflash_batched_verify_lanes =
            std::env::var("ATLAS_DFLASH_BATCHED_VERIFY").ok().as_deref() == Some("1");
        let batched_k5_lanes = resolve_verify_lane_count(
            qwen4_batched_k5_armed,
            dflash_batched_verify_lanes,
            max_batch_size,
        )?;
        // Every model registers its own lane count (1 when batched verify is
        // off), so a hot swap never inherits the previous model's lanes.
        let qwen4_verify_lanes =
            crate::layers::qwen4_verify_lanes::configure_lanes(batched_k5_lanes)?;
        let qwen4_cross_seq_staging = if qwen4_batched_k5_armed && batched_k5_lanes > 1 {
            Some(crate::layer::Qwen4CrossSeqStaging::allocate(
                gpu.as_ref(),
                &config,
            )?)
        } else {
            None
        };

        // Last-K prompt-tail target hidden capture buffer for MTP prefill.
        // Gated by ATLAS_MTP_LASTK_PREFILL=N (N>0 enables; defaults to 64 on Qwen4, 0 disabled elsewhere).
        // Sized at BF16 `residual_width() * 2` on Qwen4 and FP32 `hidden_size * 4` elsewhere.
        let mtp_lastk_capacity: usize = std::env::var("ATLAS_MTP_LASTK_PREFILL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(if config.is_qwen4_exp() { 64 } else { 0 });
        let mtp_recondition_rows = if config.is_qwen4_exp() && proposer.is_some() {
            crate::layers::qwen4_mtp_rows::MAX_STAGED_ROWS
        } else {
            0
        };
        let mtp_lastk_buf =
            if (mtp_lastk_capacity > 0 || mtp_recondition_rows > 0) && proposer.is_some() {
                let row_bytes = if config.is_qwen4_exp() {
                    config.residual_width() * 2
                } else {
                    config.hidden_size * 4
                };
                if mtp_lastk_capacity > 0 {
                    tracing::info!(
                        "MTP last-K prefill: ENABLED (K={mtp_lastk_capacity} tokens, \
                     {} KiB hidden capture buffer)",
                        mtp_lastk_capacity * row_bytes / 1024,
                    );
                }
                // Tail: one FP32-wide logits row (the prompt replay parks target
                // logits and k8a scratch save here), followed on Qwen4 by a
                // dedicated 9-row staging slice (`mtp_recondition_staging_buf`)
                // for K7 committed-row rewrite (`ATLAS_QWEN4_MTP_RECONDITION`) so
                // it never overlaps `ATLAS_MTP_LASTK_DEFER=1`.
                let lastk_bytes = if mtp_lastk_capacity > 0 {
                    mtp_lastk_capacity * row_bytes + config.vocab_size * 4
                } else {
                    0
                };
                Some(gpu.alloc(lastk_bytes + mtp_recondition_rows * row_bytes)?)
            } else {
                None
            };
        crate::layers::qwen4_mtp_rows::set_staged_rows_capacity(if mtp_lastk_buf.is_some() {
            mtp_recondition_rows
        } else {
            0
        });
        if config.is_qwen4_exp() && proposer.is_some() {
            let lastk_allocated = mtp_lastk_capacity > 0 && mtp_lastk_buf.is_some();
            tracing::info!(
                "QWEN4_MTP_LASTK_RUNTIME_RECEIPT schema=1 forced_disabled={} \
                 proposer_loaded=true buffer_allocated={} capacity={} replay_enabled={}",
                speculation_forced_disabled,
                lastk_allocated,
                mtp_lastk_capacity,
                lastk_allocated && !speculation_forced_disabled,
            );
        }

        // DFlash 5-layer hidden-state stack. Allocated when a
        // BlockDiffusionDraftHead is the active proposer (`config.dflash_capture_layers`
        // populated by the loader from the drafter's `dflash_config.target_layer_ids`).
        // Size: k_max × N_capture × hidden_size × bf16 (17 × 5 × 2048 × 2 = 348 KB).
        // k_max = γ+1 = 17 covers all verify paths (K=2..4 and DFlash γ=16).
        //
        // When ATLAS_DUMP_HIDDEN is set we still force a capture even if the
        // DFlash proposer isn't active, so training data can be collected
        // off the production K=3 MTP path. Layer-selection precedence:
        //
        //   1. ATLAS_DUMP_LAYERS env var (comma-separated, eg "1,16,31,46,61")
        //      — explicit override, use when retraining a drafter with a
        //      different layer spec than the canonical formula.
        //
        //   2. Derived from `num_hidden_layers` via the canonical 5-layer
        //      DFlash spacing: `[1, 1+s, 1+2s, 1+3s, 1+4s]` where
        //      `s = (N - 4) / 4`. This matches the z-lab DFlash drafter
        //      family's `target_layer_ids` across observed targets:
        //      - 40-layer Qwen3.6-35B-A3B: s=9  → [1, 10, 19, 28, 37]
        //      - 64-layer Qwen3.6-27B:     s=15 → [1, 16, 31, 46, 61]
        //
        // Deriving indices from `num_hidden_layers` ensures hidden dumps
        // match the target model depth rather than hardcoding one depth.
        let mut dflash_capture_layers: Vec<usize> = config.dflash_capture_layers.clone();
        if dflash_capture_layers.is_empty() && std::env::var("ATLAS_DUMP_HIDDEN").is_ok() {
            let (layers, source) = if let Ok(raw) = std::env::var("ATLAS_DUMP_LAYERS") {
                let parsed: Vec<usize> = raw
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if parsed.is_empty() {
                    return Err(anyhow::anyhow!(
                        "ATLAS_DUMP_LAYERS set but parsed to empty list: {:?}",
                        raw
                    ));
                }
                (parsed, "ATLAS_DUMP_LAYERS")
            } else {
                let n = config.num_hidden_layers;
                if n < 5 {
                    return Err(anyhow::anyhow!(
                        "ATLAS_DUMP_HIDDEN: cannot derive 5 capture layers from \
                         num_hidden_layers={} (need ≥ 5). Set ATLAS_DUMP_LAYERS \
                         explicitly.",
                        n
                    ));
                }
                // `s = floor((N - 4) / 4)`. Range check below catches N = 5/6/7
                // where `s == 0` (indices would collide); N >= 8 yields `s >= 1`.
                let s = (n.saturating_sub(4)) / 4;
                if s == 0 || 1 + 4 * s >= n {
                    return Err(anyhow::anyhow!(
                        "ATLAS_DUMP_HIDDEN: derived spacing s={} from N={} is \
                         degenerate. Set ATLAS_DUMP_LAYERS explicitly.",
                        s,
                        n
                    ));
                }
                let derived = vec![1, 1 + s, 1 + 2 * s, 1 + 3 * s, 1 + 4 * s];
                (derived, "derived from num_hidden_layers")
            };
            // Sanity: all indices must be in-range for this model.
            for &li in &layers {
                if li >= config.num_hidden_layers {
                    return Err(anyhow::anyhow!(
                        "ATLAS_DUMP_HIDDEN: capture layer {} out of range \
                         (num_hidden_layers={})",
                        li,
                        config.num_hidden_layers
                    ));
                }
            }
            tracing::info!(
                "ATLAS_DUMP_HIDDEN set: dflash_capture_layers = {:?} \
                 (source: {}, num_hidden_layers={})",
                layers,
                source,
                config.num_hidden_layers
            );
            dflash_capture_layers = layers;
        }
        let dflash_capture_width = if config.dflash_capture_width == 0 {
            config.residual_width()
        } else {
            config.dflash_capture_width
        };
        let dflash_capture_offset = config.dflash_capture_offset;
        anyhow::ensure!(
            dflash_capture_offset.saturating_add(dflash_capture_width) <= config.residual_width(),
            "DFlash capture slice offset={} width={} exceeds target residual width {}",
            dflash_capture_offset,
            dflash_capture_width,
            config.residual_width(),
        );
        let dflash_capture_mode = config.dflash_capture_mode;
        match dflash_capture_mode {
            DflashCaptureMode::ResidualSlice => {}
            DflashCaptureMode::Qwen4HyperProjected => {
                anyhow::ensure!(
                    config.is_qwen4_exp()
                        && dflash_capture_width == config.hidden_size
                        && dflash_capture_offset == 0,
                    "projected DFlash capture requires a Qwen4 target with H-wide, zero-offset output"
                );
                anyhow::ensure!(
                    qwen4_final_mixer.is_some(),
                    "projected DFlash capture requires the target Qwen4 terminal hyperconnection mixer"
                );
            }
            DflashCaptureMode::Qwen4LayerAttnHc => {
                anyhow::ensure!(
                    config.is_qwen4_exp()
                        && dflash_capture_width == config.hidden_size
                        && dflash_capture_offset == 0,
                    "Qwen4LayerAttnHc DFlash capture requires a Qwen4 target with H-wide, zero-offset output"
                );
                for &tap in &dflash_capture_layers {
                    let next = tap.checked_add(1).ok_or_else(|| {
                        anyhow::anyhow!("Qwen4LayerAttnHc tap layer {tap} index overflow")
                    })?;
                    anyhow::ensure!(
                        next < layers.len() && layers[next].qwen4_attn_hyper().is_some(),
                        "Qwen4LayerAttnHc DFlash capture tap {tap} requires layers[{next}].qwen4_attn_hyper()"
                    );
                }
            }
            DflashCaptureMode::Qwen4HcMean => {
                anyhow::ensure!(
                    config.is_qwen4_exp()
                        && dflash_capture_width == config.hidden_size
                        && dflash_capture_offset == 0,
                    "Qwen4HcMean DFlash capture requires a Qwen4 target with H-wide, zero-offset output"
                );
            }
        }
        let qwen4_hc_mean_rows_kernel = if dflash_capture_mode == DflashCaptureMode::Qwen4HcMean {
            gpu.kernel("qwen4_hyper", "qwen4_hc_mean_rows")?
        } else {
            KernelHandle(0)
        };
        let (dflash_hidden_save, dflash_hidden_save_rows) = if dflash_capture_layers.is_empty() {
            (None, 0usize)
        } else {
            let n = dflash_capture_layers.len();
            // Max verify width T = γ+1 (= dflash_kgamma): the K=γ verify
            // captures hidden state for ALL T input rows via
            // `try_dflash_capture(token_idx=0..T)`. Track the real flat/tree
            // capacities instead of hardcoding a historical width, and expand
            // to the full 128-row cross-sequence frame when batched verify
            // lanes are armed so batched verify and K=1 bootstrap capture
            // without aliasing.
            let k_max = dflash_kgamma.max(ddtree_cap);
            let dflash_batched_verify =
                std::env::var("ATLAS_DFLASH_BATCHED_VERIFY").ok().as_deref() == Some("1");
            let save_rows = k_max.max(if batched_k5_lanes > 1 || dflash_batched_verify {
                crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS
            } else {
                0
            });
            (
                Some(gpu.alloc(save_rows * n * dflash_capture_width * 2)?),
                save_rows,
            )
        };

        // EP command buffer for token broadcast (4 bytes, u32)
        let ep_cmd_buf = gpu.alloc(4)?;

        // Secondary stream + event for pipelining checkpoint D2D with MTP propose.
        let secondary_stream = gpu.create_stream()?;
        let secondary_event = gpu.create_event()?;
        let dflash_capture_done_event = if !dflash_capture_layers.is_empty()
            && (crate::layers::dflash_head::async_propose::dflash_early_stage1_enabled()
                || crate::layers::dflash_head::async_propose::dflash_opt_branch0_enabled())
        {
            match gpu.create_event() {
                Ok(ev) => ev,
                Err(error) => {
                    tracing::warn!(
                        "ATLAS_DFLASH_EARLY_STAGE1: create_event failed ({error:#}); disabling early Stage-1"
                    );
                    0
                }
            }
        } else {
            0
        };

        // EP: register moe_output buffer with NCCL and provide bf16_add kernel.
        if let Some(ref comm) = comm
            && comm.world_size() == 2
        {
            let moe_ptr = buffers.moe_output().0;
            let moe_bytes = buffers.sizes().moe_output;
            match comm.register_buffer(moe_ptr, moe_bytes) {
                Ok(_) => tracing::info!("Registered moe_output ({moe_bytes} B) with NCCL"),
                Err(e) => tracing::warn!("ncclCommRegister moe_output failed (non-fatal): {e}"),
            }
            match gpu.kernel("bf16_add", "bf16_add_inplace") {
                Ok(k) => comm.set_add_kernel(k.0),
                Err(e) => {
                    tracing::warn!("bf16_add_inplace kernel not found (send/recv disabled): {e}")
                }
            }
        }

        // Allocate pinned host staging buffer for batched metadata H2D
        // plus an 8 MiB fast-H2D ring tail so stream-ordered async H2D uploads
        // never alias `[0..meta_staging_bytes)` or in-flight sibling uploads.
        let meta_staging_bytes = buffers.sizes().scratch.max(64 * 1024);
        let pinned_bytes = meta_staging_bytes + 8 * 1024 * 1024;
        let pinned_buffer = gpu.alloc_host_pinned(pinned_bytes)?;
        tracing::info!("Pinned metadata staging: {} KB", pinned_bytes / 1024);
        let max_batch_tokens = buffers.max_batch_tokens();
        let pinned_staging = Mutex::new(PinnedMetaStaging {
            buffer: pinned_buffer,
            h2d_ring_base: meta_staging_bytes,
            h2d_ring_cursor: meta_staging_bytes,
            positions: Vec::with_capacity(max_batch_tokens),
            positions_h: Vec::with_capacity(max_batch_tokens),
            positions_w: Vec::with_capacity(max_batch_tokens),
            slots: Vec::with_capacity(max_batch_tokens),
        });

        // SSM state normalization kernel + pointer buffer (for chunked prefill).
        let ssm_norm_k = gpu
            .kernel("ssm_state_norm", "ssm_state_clamp_norm_fused")
            .unwrap_or(KernelHandle(0));
        // ATLAS_SSM_H_FP16 stage 2: one-shot FP32 -> FP16 h-state converter.
        // try_kernel, not kernel: absent on targets that never built
        // ssm_h_dtype, and its absence must not break an ordinary FP32 serve.
        // A zero handle here makes ssm_h_to_f16_dispatch refuse loudly rather
        // than run FP16 kernels over an unconverted pool.
        let ssm_h_f32_to_f16_k =
            crate::layers::try_kernel(gpu.as_ref(), "ssm_h_dtype", "ssm_h_state_f32_to_f16");
        let ssm_h_f16_to_f32_k =
            crate::layers::try_kernel(gpu.as_ref(), "ssm_h_dtype", "ssm_h_state_f16_to_f32");
        let gdn_batch_commit_kernel = crate::layers::try_kernel(
            gpu.as_ref(),
            "gdn_batch_commit",
            "gdn_batch_commit_copy_f32",
        );
        Self::warm_gdn_lazyfinal_splitv4_kernel(gpu.as_ref());
        if atlas_core::device::active_sm_count() > 64 {
            crate::layers::dense_ffn::warm_coalesced_nvfp4_sm120_kernels(gpu.as_ref());
        }
        let commit_runtime_flags =
            super::trait_impl::commit_cache::CommitRuntimeFlags::from_env(config.is_qwen4_exp());
        let slot_commit_table = Mutex::new(super::trait_impl::commit_cache::SlotCommitTable::new(
            ssm_pool.max_slots,
            ssm_pool.num_ssm_layers,
        ));

        // Logit softcapping (Gemma-4: cap=30.0). Only load if model uses it.
        let logit_softcap_kernel = if config.final_logit_softcapping > 0.0 {
            gpu.kernel("logit_softcap", "logit_softcap_bf16")
                .unwrap_or_else(|e| {
                    tracing::warn!("logit_softcap kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        // FP32 softcap variant — only loaded when both softcap and FP32
        // residual are active (i.e. Gemma-4 dense). Other models keep the
        // BF16 softcap (or no softcap at all).
        let logit_softcap_fp32_kernel = if config.final_logit_softcapping > 0.0 && fp32_residual {
            gpu.kernel("logit_softcap", "logit_softcap_fp32")
                .unwrap_or_else(|e| {
                    tracing::warn!("logit_softcap_fp32 kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        // Opt-in FP32 logits gate (`ATLAS_GEMMA4_FP32_LMHEAD=1`, default OFF).
        // Keeping BF16 logits by default avoids ~1 MB/token D2H host sampling on
        // 262K vocabularies while allowing FP32 lm_head + softcap for bisection.
        let env_override = std::env::var("ATLAS_GEMMA4_FP32_LMHEAD").ok();
        let fp32_requested = matches!(env_override.as_deref(), Some("1") | Some("true"));
        let use_fp32_logits = fp32_requested
            && fp32_residual
            && ((lm_head_nvfp4.is_none() && dense_gemv_fp32out_kernel.0 != 0)
                || (lm_head_nvfp4.is_some() && w4a16_gemv_logits_kernel.0 != 0));
        // Dedicated FP32 logits scratch — only the single-token decode path
        // uses it. Prefill and batched-decode lm_head still write BF16 to the
        // shared `buffers.logits()`. Sized for one row of `vocab_size` FP32.
        let logits_fp32_buf = if use_fp32_logits {
            let bytes = config.vocab_size * 4;
            let p = gpu.alloc(bytes)?;
            tracing::info!(
                "FP32 LM head + softcap active (model_type={}, vocab={}). \
                 Decode logits scratch: {} bytes.",
                config.model_type,
                config.vocab_size,
                bytes,
            );
            p
        } else {
            DevicePtr::NULL
        };

        // Embedding scale (Gemma-4: sqrt(hidden_size)). Only load if model uses it.
        let embed_scale_kernel = if config.embed_scale > 0.0 {
            gpu.kernel("embed_scale", "bf16_scale_inplace")
                .unwrap_or_else(|e| {
                    tracing::warn!("embed_scale kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        if config.embed_scale > 0.0 {
            tracing::info!(
                "Embedding scale: {:.4} (sqrt({}))",
                config.embed_scale,
                config.hidden_size
            );
        }
        let ssm_norm_ptrs = if ssm_pool.num_ssm_layers > 0 {
            gpu.alloc(ssm_pool.num_ssm_layers * 8)
                .unwrap_or(DevicePtr::NULL)
        } else {
            DevicePtr::NULL
        };

        // GDN prefill buffers: sized for max_batch_tokens (the prefill chunk size),
        // NOT max_seq_len. For prompts longer than this, prefill_twophase falls back
        // to standard chunked prefill which carries h_state/conv_state between chunks.
        // The GDN recurrence is sequential anyway, so chunking is mathematically identical.
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        // GDN buffers only needed when GDN linear attention layers exist
        // (conv_dim > 0). Mamba-2 models (Nemotron) have conv_dim=0 — skip alloc
        // to avoid cuMemAlloc(0) error.
        let skip_qwen4_gdn_bufs = config.is_qwen4_exp()
            && std::env::var("ATLAS_QWEN4_RELEASE_BF16_QKVZ")
                .ok()
                .as_deref()
                == Some("1");
        let gdn_buf_len = if skip_qwen4_gdn_bufs {
            0
        } else {
            max_batch_tokens.min(max_seq_len)
        };
        let (gdn_qkv, gdn_gate_beta, gdn_out, gdn_z) = if conv_dim > 0 && gdn_buf_len > 0 {
            let qkv = gpu.alloc(gdn_buf_len * conv_dim * 2)?;
            let gb = gpu.alloc(gdn_buf_len * nv * 2 * 4)?;
            let o = gpu.alloc(gdn_buf_len * value_dim * 2)?;
            let z = gpu.alloc(gdn_buf_len * value_dim * 2)?;
            let total_mb =
                (gdn_buf_len * (conv_dim * 2 + nv * 2 * 4 + value_dim * 2 * 2)) / (1024 * 1024);
            tracing::info!(
                "GDN prefill buffers: {total_mb} MB for {gdn_buf_len} tokens (chunked SSM prefill)"
            );
            (qkv, gb, o, z)
        } else {
            (
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
            )
        };

        // FP8 calibration only runs when the cache is actually FP8 — the
        // `observe()` call in `layers/attention/decode.rs` sits inside the FP8
        // cache branch. For BF16 or NVFP4 caches the MODEL.toml
        // `fp8_kv_calibration_tokens` value is unused and must not suppress
        // CUDA graphs.
        let has_fp8_calibration = config.fp8_kv_calibration_tokens > 0
            && kv_cache.dtype() == spark_runtime::kv_cache::KvCacheDtype::Fp8;
        Ok(Self {
            config,
            ddtree_parent_ids_dev: Mutex::new(None),
            dflash_flat_tree_route: std::sync::atomic::AtomicBool::new(false),
            verify_transactions: Mutex::new(Default::default()),
            verify_batch_transactions: Mutex::new(Default::default()),
            // Ancestor-exact until a non-flat tree verify without per-row
            // indirection proves otherwise (flat/chain verifies are exact).
            dflash_tree_ancestor_attn: std::sync::atomic::AtomicBool::new(true),
            ddtree_num_tree_tokens: Mutex::new(0),
            ddtree_parent_ids_persistent: parent_ids_persistent,
            ddtree_parent_ids_capacity: parent_ids_capacity,
            ddtree_parent_ids_host: Mutex::new(Vec::new()),
            ddtree_dfs_inv_perm: Mutex::new(Vec::new()),
            tree_kv_indir_persistent,
            tree_kv_indir_stride,
            tree_kv_indir_base_persistent,
            tree_kv_indir_base_host_pinned: Mutex::new(tree_kv_indir_base_host_pinned),
            tree_kv_pack_scratch_k,
            tree_kv_pack_scratch_v,
            tree_kv_pack_block_table,
            tree_kv_pack_seq_lens,
            tree_kv_pack_block_stride_bytes,
            tree_kv_pack_data_section_bytes,
            tree_kv_pack_scatter_fp8_kernel,
            tree_kv_pack_scatter_nvfp4_kernel,
            tree_kv_pack_active,
            embed_tokens,
            final_norm,
            final_norm_identity,
            qwen4_final_mixer,
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            qwen4_ple,
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            qwen4_ple_prefetch: Mutex::new(HashMap::new()),
            lm_head_weight,
            lm_head_fp8: None,
            lm_head_int8_block32: None,
            lm_head_nvfp4,
            lm_head_nvfp4_t,
            layers,
            buffers,
            kv_cache: Mutex::new(kv_cache),
            strided_copy_rows_kernel: crate::layers::try_kernel(
                gpu.as_ref(),
                "strided_copy_rows",
                "strided_copy_rows_16",
            ),
            qwen4_hc_mean_rows_kernel,
            gpu,
            rms_norm_kernel,
            bf16_to_f32_kernel,
            dense_gemv_kernel,
            dense_gemv_fp32out_kernel,
            w4a16_gemv_kernel,
            w4a16_gemv_sw_kernel,
            gemv_sw,
            w4a16_gemv_logits_kernel,
            w4a16_exact_lm_head_kernels,
            w4a16_gemm_kernel,
            w4a16_gemm_t_m32_n64_kernel,
            w4a16_gemv_batch2_kernel,
            w4a16_gemv_batch3_logits_kernel,
            dense_gemm_kernel,
            argmax_kernel,
            argmax_rows_kernel,
            argmax_logits_kernel,
            topk_candidates_kernel,
            f2_sum_kernel,
            topk_candidates_buf,
            target_gumbel_verify_kernel,
            target_gumbel_verify_buf,
            staged_verify_gumbel_rows: Mutex::new(None),
            in_graph_gumbel_verify_tokens: Mutex::new(None),
            target_rejection_verify_kernel,
            target_rejection_verify_buf,
            target_rejection_q_buf,
            staged_verify_rejection_rows: Mutex::new(None),
            in_graph_rejection_verify_words: Mutex::new(None),
            batched_embed_kernel,
            fill_slots_kernel,
            decode_graph: Mutex::new(std::collections::HashMap::new()),
            decode_graph_sparse_regime: Mutex::new(std::collections::HashMap::new()),
            batch_decode_graphs: Mutex::new(HashMap::new()),
            piecewise_decode_graphs: Mutex::new(HashMap::new()),
            // Suppress CUDA graphs during FP8 KV calibration (lifted once
            // calibration completes), when Gemma-4 diagnostics are enabled
            // (`ATLAS_DIAG_GEMMA4=1|true`), or when hidden-state dumping is
            // active (`ATLAS_DUMP_HIDDEN`). MLA runs graphed once eager-only
            // syncs are skipped; any per-step metadata zeroing runs before
            // `CaptureGuard::begin`.
            suppress_graphs: std::sync::atomic::AtomicBool::new(
                has_fp8_calibration
                    || std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true")
                    || std::env::var("ATLAS_DUMP_HIDDEN").is_ok(),
            ),
            ssm_pool,
            ssm_snapshots,
            max_blocks_per_seq,
            dummy_kv_block,
            profile,
            profile_first_pending: std::sync::atomic::AtomicBool::new(profile_first),
            proposer,
            // Arm 1 stays empty here. `set_dflash_proposer` (impl_b3.rs) is
            // the only thing that fills it, by demoting the MTP head this
            // constructor just built when a DFlash drafter is installed on
            // top. A build with one proposer keeps `None` and behaves exactly
            // as it did before the arm split existed.
            proposer_alt: None,
            active_proposer_arm: std::sync::atomic::AtomicU8::new(0),
            mtp_hidden_save,
            qwen4_mtp_hidden_frames,
            qwen4_qsa_journal_owner: Mutex::new(Default::default()),
            qwen4_cross_seq_staging,
            qwen4_verify_attn_meta: Mutex::new(None),
            gdn_batch_commit_kernel,
            commit_runtime_flags,
            slot_commit_table,
            qwen4_verify_lanes,
            mtp_lastk_buf,
            mtp_lastk_capacity,
            speculation_forced_disabled,
            k8a_lastk_deferred: Mutex::new(None),
            k9x_lastk_events: std::sync::OnceLock::new(),
            qwen4_native_capability_cache: (0..=crate::layers::QWEN4_NATIVE_MAX_ROWS)
                .map(|_| std::sync::OnceLock::new())
                .collect(),
            dflash_hidden_save,
            dflash_hidden_save_rows,
            dflash_batched_ffn_input: parking_lot::Mutex::new(None),
            dflash_capture_layers,
            dflash_capture_done_event,
            dflash_capture_width,
            dflash_capture_offset,
            dflash_capture_mode,
            verify2_graph: Mutex::new(std::collections::HashMap::new()),
            verify3_graph: Mutex::new(std::collections::HashMap::new()),
            verify4_graph: Mutex::new(std::collections::HashMap::new()),
            verify_kgamma_graph: Mutex::new(std::collections::HashMap::new()),
            verify_kgamma_lru: Mutex::new(super::types::VerifyKgammaGraphLru::default()),
            prefix_cache,
            secondary_stream,
            secondary_event,
            comm,
            ep_cmd_buf,
            worker_extra_slots: Mutex::new(std::collections::HashMap::new()),
            ep_wire_state: Mutex::new(super::types::EpHeadWireState::Idle),
            pending_freed_slots: Mutex::new(std::collections::VecDeque::new()),
            self_speculative,
            last_mtp_hidden_idx: std::sync::atomic::AtomicUsize::new(0),
            vision_encoder,
            vision_embeddings: Mutex::new(super::vision_embeddings::VisionEmbeddingState::new(
                max_seq_len,
                max_batch_size,
            )),
            pinned_staging,
            ssm_checkpoint_interval,
            ssm_state_norm_kernel: ssm_norm_k,
            ssm_norm_ptrs_buf: ssm_norm_ptrs,
            ssm_h_f32_to_f16_kernel: ssm_h_f32_to_f16_k,
            ssm_h_f16_scratch: std::sync::OnceLock::new(),
            slot_zero_event: std::sync::OnceLock::new(),
            ssm_h_f16_to_f32_kernel: ssm_h_f16_to_f32_k,
            ssm_h_f32_scratch: std::sync::OnceLock::new(),
            gdn_buf_qkv: gdn_qkv,
            gdn_buf_gate_beta: gdn_gate_beta,
            gdn_buf_out: gdn_out,
            gdn_buf_z: gdn_z,
            gdn_buf_max_len: gdn_buf_len,
            logit_softcap_kernel,
            logit_softcap_fp32_kernel,
            use_fp32_logits,
            logits_fp32_buf,
            embed_scale_kernel,
        })
    }

    /// Attach the per-row scaled FP8 E4M3 target `lm_head` weight built by the
    /// model factory.
    pub(crate) fn set_lm_head_fp8(&mut self, weight: crate::weight_map::Fp8DenseWeight) {
        self.lm_head_fp8 = Some(weight);
    }

    /// Attach the owned `INT8 Block-32 BF16-Scale (8.50 bpw)` target `lm_head`
    /// weight eagerly quantized at model load time.
    pub(crate) fn set_lm_head_int8_block32(&mut self, weight: ops::Int8Block32DeviceWeight) {
        if let Some(prev) = self.lm_head_int8_block32.replace(weight) {
            ops::free_lm_head_int8_block32_weight(self.gpu.as_ref(), prev);
        }
    }
}

/// Resolve the number of per-slot `gdn_lazy_retain` verify lanes to register
/// for a model build.
///
/// Allocates `max_batch_size` lanes whenever either Qwen4 cross-sequence K5
/// verification OR flat batched DFlash verification
/// (`ATLAS_DFLASH_BATCHED_VERIFY=1`, including non-Qwen4 models such as
/// Qwen3.8-27B) is armed.
fn resolve_verify_lane_count(
    qwen4_batched_k5_armed: bool,
    dflash_batched_verify_enabled: bool,
    max_batch_size: usize,
) -> Result<usize> {
    crate::layers::qwen4_verify_lanes::lanes_for(
        qwen4_batched_k5_armed || dflash_batched_verify_enabled,
        max_batch_size,
    )
}

#[cfg(test)]
mod dflash_verify_width_tests {
    use super::{checked_ssm_speculative_geometry, resolve_verify_lane_count};

    #[test]
    fn verify_width_is_anchor_plus_actual_drafts() {
        for (drafts, expected) in [(15, 16), (6, 7), (1, 2)] {
            assert_eq!(
                checked_ssm_speculative_geometry(true, true, drafts, None)
                    .unwrap()
                    .dflash_verify_width,
                expected
            );
        }
        assert_eq!(
            checked_ssm_speculative_geometry(false, false, 15, None)
                .unwrap()
                .dflash_verify_width,
            0
        );
    }

    #[test]
    fn dflash_batched_verify_allocates_per_slot_retain_lanes_for_non_qwen4() {
        // Non-Qwen4 model (`qwen4_batched_k5_armed == false`) with
        // `ATLAS_DFLASH_BATCHED_VERIFY=1` must allocate one lane per slot so
        // C>=2 sequences do not clobber lane 0 of `gdn_lazy_retain`.
        assert_eq!(resolve_verify_lane_count(false, true, 8).unwrap(), 8);
        assert_eq!(resolve_verify_lane_count(false, true, 16).unwrap(), 16);
        assert_eq!(resolve_verify_lane_count(false, true, 1).unwrap(), 1);
        assert_eq!(resolve_verify_lane_count(false, false, 8).unwrap(), 1);
        assert_eq!(resolve_verify_lane_count(true, false, 8).unwrap(), 8);
    }
}
