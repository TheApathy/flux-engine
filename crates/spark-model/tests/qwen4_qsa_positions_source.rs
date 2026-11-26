// SPDX-License-Identifier: AGPL-3.0-only

const CUDA: &str = include_str!("../../../kernels/gb10/common/qwen4_qsa.cu");
const HOST: &str = include_str!("../src/layers/qwen4_qsa.rs");
const DEVICE: &str = include_str!("../src/layers/qwen4_qsa/device.rs");
const PREFILL: &str = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_inner.rs");
const F8: &str = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_only.rs");
const K5_DECODE: &str = include_str!("../src/layers/qwen3_attention/trait_impl/decode_inner.rs");
const COMMIT: &str = include_str!("../src/model/trait_impl/async_chkpt.rs");
const VERIFY: &str = include_str!("../src/model/trait_impl/verify_d.rs");
const MODEL_TYPES: &str = include_str!("../src/model/types.rs");

fn kernel(name: &str) -> &str {
    CUDA.split(&format!("void {name}("))
        .nth(1)
        .expect("production kernel")
        .split("extern \"C\"")
        .next()
        .unwrap()
}

fn ordered(source: &str, markers: &[&str]) {
    let mut rest = source;
    for marker in markers {
        let position = rest
            .find(marker)
            .unwrap_or_else(|| panic!("missing/order: {marker}"));
        rest = &rest[position + marker.len()..];
    }
}

#[test]
fn serial_and_batched_staging_use_physical_slots() {
    assert!(CUDA.contains("#include \"qwen4_qsa_positions.cuh\""));
    for name in [
        "qwen4_qsa_stage_pool",
        "qwen4_qsa_store_compressed",
        "qwen4_qsa_stage_prefill_raw",
    ] {
        let body = kernel(name);
        assert!(
            body.contains("qwen4_qsa_positions::physical_slot("),
            "{name}"
        );
        assert!(body.contains("physical.ring_row"), "{name}");
        assert!(!body.contains("position & (COMPRESS_RATIO - 1)"), "{name}");
        assert!(!body.contains("positions[token] &"), "{name}");
    }
}

#[test]
fn device_selection_uses_sequence_length_not_rotary_position() {
    let body = kernel("qwen4_qsa_select_expand_device_meta");
    assert!(body.contains("qwen4_qsa_positions::physical_query(sequence_length, position)"));
    assert!(!body.contains("position = *position_ptr"));
    // Keep the parameter layout stable; removal requires an explicit ABI change.
    assert!(body.contains("const unsigned int* __restrict__ position_ptr,"));
    // Re-pinned at eca7c08ef: the launch now picks the `_par` sibling (same ABI) or
    // the serial kernel through `select_kernel`; both derive position from seq_len.
    let par = CUDA
        .split("__launch_bounds__(1024) qwen4_qsa_select_expand_device_meta_par(")
        .nth(1)
        .expect("parallel select kernel")
        .split("extern \"C\"")
        .next()
        .unwrap();
    assert!(par.contains("qwen4_qsa_positions::physical_query(sequence_length, position)"));
    assert!(!par.contains("position = *position_ptr"));
    assert!(par.contains("const unsigned int* __restrict__ position_ptr,"));
    ordered(
        DEVICE,
        &[
            "(self.select_expand_device, 1)",
            "KernelLaunch::new(gpu, select_kernel)",
            ".arg_ptr(meta.positions)",
            ".arg_ptr(meta.seq_len)",
        ],
    );
}

#[test]
fn host_requires_current_physical_query_before_effects() {
    let body = HOST.split("pub fn update_and_select(").nth(1).unwrap();
    ordered(
        body,
        &[
            "valid_physical_query(position, sequence_length)",
            "self.cache(",
            "ops::dense_gemv(",
        ],
    );
}

#[test]
fn optional_batch_fails_before_affected_layer_and_index_effects() {
    ordered(
        PREFILL,
        &[
            "ATLAS_QWEN4_ATTN_PREFILL_BATCH",
            "ATLAS_QWEN4_QSA_PREFILL_GEMM",
            "validate_prefill_index_batch(num_tokens, seq_len_start)",
            "attn_hyper.prepare_prefill_exact(",
        ],
    );
    let body = HOST.split("pub fn update_prefill_index(").nth(1).unwrap();
    ordered(
        body,
        &[
            "validate_prefill_index_batch(num_tokens, seq_len_start)",
            "self.cache(",
            "ops::dense_gemm(",
        ],
    );
    assert!(HOST.contains("prefill_index_batch_is_safe(num_tokens, seq_len_start)"));
    assert!(HOST.contains("ATLAS_QWEN4_QSA_PREFILL_GEMM"));
}

#[test]
fn rotary_streams_and_f8_shipping_sequence_are_retained() {
    assert!(HOST.contains("mod position_contract;"));
    assert!(HOST.contains(
        "self.apply_rope(\n            cache.projected_qk,\n            meta.positions,"
    ));
    ordered(
        F8,
        &[
            "seq_len_start + row + 1",
            "positions: metadata.positions.offset(row * 4)",
            "positions_h: metadata.positions_h.offset(row * 4)",
            "positions_w: metadata.positions_w.offset(row * 4)",
            "slot: metadata.slot.offset(row * 8)",
            "self.attention_forward(",
            "seq_len_start + row",
        ],
    );
    assert!(!HOST.contains("ATLAS_QWEN4_QSA_PREFILL_GEMM_FALLBACK"));
}

#[test]
fn cache_and_stage_launch_abis_remain_unchanged() {
    ordered(
        HOST,
        &[
            "KernelLaunch::new(gpu, self.stage_pool)",
            ".arg_ptr(cache.projected_qk)",
            ".arg_ptr(cache.raw_ring)",
            ".arg_ptr(cache.pooled_key)",
            ".arg_ptr(cache.first_position)",
            ".arg_ptr(meta.slot)",
            ".arg_ptr(meta.positions)",
            ".arg_u32(cache.block_size as u32)",
        ],
    );
    ordered(
        DEVICE,
        &[
            "KernelLaunch::new(gpu, self.stage_pool)",
            ".arg_ptr(cache.projected_qk)",
            ".arg_ptr(cache.raw_ring)",
            ".arg_ptr(cache.pooled_key)",
            ".arg_ptr(cache.first_position)",
            ".arg_ptr(meta.slot)",
            ".arg_ptr(meta.positions)",
            ".arg_u32(cache.block_size as u32)",
        ],
    );
}

#[test]
fn k5_qsa_transaction_snapshots_before_effect_and_commits_before_early_return() {
    ordered(
        K5_DECODE,
        &[
            // Re-pinned at eca7c08ef: rustfmt split the snapshot call across lines.
            ".snapshot_verify_k5_row(\n",
            "kv_cache,\n",
            "token_metadata.slot,\n",
            "row,\n",
            "ctx.gpu,\n",
            "stream,\n",
            "let attn_input =",
            "attention_forward_preprojected",
        ],
    );
    ordered(
        COMMIT,
        &[
            "if !crate::layers::qwen4_qsa::verify_journal_rows(k)",
            "layer.commit_qwen4_qsa_rows",
            "if num_accepted == 0",
            "self.commit_qwen4_qsa_k5_only_dispatch(num_accepted, k)",
        ],
    );
}

#[test]
fn k5_qsa_transaction_is_graph_safe_reverse_and_fail_closed() {
    let snapshot = kernel("qwen4_qsa_snapshot_verify_k5_row");
    ordered(
        snapshot,
        &[
            "row == 0 && d == 0",
            "*valid_mask = 0",
            "undo_raw_offsets[row] = raw_offset",
            "atomicOr(valid_mask, 1u << row)",
        ],
    );
    let commit = kernel("qwen4_qsa_commit_verify_k5_prefix");
    assert!(commit.contains("if (valid == 0) return;"));
    assert!(commit.contains("valid != VERIFY_K5_COMPLETE_MASK"));
    assert!(commit.contains("asm volatile(\"trap;\")"));
    assert!(
        commit.contains(
            "for (int row = VERIFY_K5_ROWS - 1; row >= static_cast<int>(accepted); --row)"
        )
    );
    assert!(commit.contains("undo_raw_offsets[row]"));
    assert!(commit.contains("undo_compressed_offsets[row]"));
    assert!(commit.contains("*valid_mask = 0"));
}

#[test]
fn k5_device_attention_topology_owns_the_graph_cache_key() {
    assert!(VERIFY.contains("let qwen4_k5_attention_topology ="));
    assert!(VERIFY.contains(".map(|plan| plan.topology())"));
    assert!(
        VERIFY.contains("let cache_key = (seq.slot_idx, k, pack_key, qwen4_k5_attention_topology)")
    );
    assert!(MODEL_TYPES.contains("Option<Qwen4K5DeviceAttentionTopology>"));
    assert!(!K5_DECODE.contains("let _topology"));
}
