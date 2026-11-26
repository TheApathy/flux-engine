# SPDX-License-Identifier: AGPL-3.0-only
"""Host/static contract for the flat-K5 QSA transaction."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "crates/spark-model/src"
SERVER = ROOT / "crates/spark-server/src/scheduler/verify_dflash_step.rs"


class Qwen4QsaK5TransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cuda = (ROOT / "kernels/gb10/common/qwen4_qsa.cu").read_text()
        cls.target_cuda = (
            ROOT
            / "kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_qsa_prefill_attn_fp8.cu"
        ).read_text()
        cls.qsa = (MODEL / "layers/qwen4_qsa.rs").read_text()
        cls.decode = (
            MODEL / "layers/qwen3_attention/trait_impl/decode_inner.rs"
        ).read_text()
        cls.commit = (MODEL / "model/trait_impl/async_chkpt.rs").read_text()
        cls.verify = (MODEL / "model/trait_impl/verify_d.rs").read_text()
        cls.types = (MODEL / "model/types.rs").read_text()
        cls.server = SERVER.read_text()
        cls.attention = (
            MODEL / "layers/qwen3_attention/decode/attention_forward.rs"
        ).read_text()
        cls.device_attention = (
            MODEL / "layers/qwen3_attention/qwen4_device_attention.rs"
        ).read_text()
        cls.qsa_route = (
            MODEL / "layers/qwen3_attention/decode/qsa_route.rs"
        ).read_text()
        cls.sampled = (
            ROOT / "crates/spark-server/src/scheduler/sampled_dflash.rs"
        ).read_text()
        cls.self_spec = (
            ROOT / "crates/spark-server/src/scheduler/spec_step.rs"
        ).read_text()
        cls.ep = (MODEL / "model/impl_a2.rs").read_text()
        cls.mtp_step = (
            ROOT / "crates/spark-server/src/scheduler/mtp_step.rs"
        ).read_text()
        cls.batched = (
            MODEL / "model/trait_impl/verify_dflash_batched.rs"
        ).read_text()

    def test_snapshot_precedes_attention_and_row_zero_clears_in_graph(self):
        snapshot = self.decode.index("snapshot_verify_k5_row(")
        attention = self.decode.index("let attn_input =", snapshot)
        self.assertLess(snapshot, attention)
        kernel = self.cuda.split("void qwen4_qsa_snapshot_verify_k5_row(", 1)[1]
        kernel = kernel.split('extern "C"', 1)[0]
        clear = kernel.index("row == 0 && d == 0")
        save = kernel.index("undo_raw_offsets[row] = raw_offset")
        arm = kernel.index("atomicOr(valid_mask, 1u << row)")
        self.assertLess(clear, save)
        self.assertLess(save, arm)

    def test_commit_is_reverse_fail_closed_and_consumes_full_accept(self):
        kernel = self.cuda.split("void qwen4_qsa_commit_verify_k5_prefix(", 1)[1]
        kernel = kernel.split('extern "C"', 1)[0]
        self.assertIn("if (valid == 0) return;", kernel)
        self.assertIn("accepted != 0 && valid != VERIFY_K5_COMPLETE_MASK", kernel)
        self.assertIn("(valid & (1u << row)) == 0", kernel)
        self.assertIn('asm volatile("trap;")', kernel)
        self.assertIn(
            "row = VERIFY_K5_ROWS - 1; row >= static_cast<int>(accepted); --row",
            kernel,
        )
        self.assertIn("*valid_mask = 0", kernel)
        body = self.commit.split("commit_verify_state_async_dispatch(", 1)[1]
        tail = body.rsplit("self.commit_qwen4_qsa_k5_only_dispatch", 1)[0]
        self.assertIn("ple.rollback_and_checkpoint", tail)

    def test_reverse_rollback_matches_every_accepted_prefix_and_page_phase(self):
        block_size = 16

        def offsets(slot):
            block, in_block = divmod(slot, block_size)
            ring = in_block % 4
            group = in_block // 4
            return block * 4 + ring, (
                block * (block_size // 4) + group if ring == 3 else None
            )

        def apply(raw, compressed, slot, value):
            raw_offset, compressed_offset = offsets(slot)
            raw[raw_offset] = value
            if compressed_offset is not None:
                base = (slot // block_size) * 4
                compressed[compressed_offset] = sum(raw[base : base + 4])

        for start in range(2 * block_size + 1):
            for accepted in range(6):
                raw = [1000 + i for i in range(16)]
                compressed = [2000 + i for i in range(16)]
                expected_raw = raw.copy()
                expected_compressed = compressed.copy()
                undo = []
                for row in range(5):
                    raw_offset, compressed_offset = offsets(start + row)
                    undo.append(
                        (
                            raw_offset,
                            raw[raw_offset],
                            compressed_offset,
                            compressed[compressed_offset]
                            if compressed_offset is not None
                            else None,
                        )
                    )
                    apply(raw, compressed, start + row, 10000 + row)
                for row in range(4, accepted - 1, -1):
                    raw_offset, raw_value, compressed_offset, compressed_value = undo[row]
                    raw[raw_offset] = raw_value
                    if compressed_offset is not None:
                        compressed[compressed_offset] = compressed_value
                for row in range(accepted):
                    apply(
                        expected_raw,
                        expected_compressed,
                        start + row,
                        10000 + row,
                    )
                self.assertEqual(raw, expected_raw, (start, accepted))
                self.assertEqual(compressed, expected_compressed, (start, accepted))

    def test_partial_journal_zero_accept_restores_every_saved_alias(self):
        block_size = 16

        def offsets(slot):
            block, in_block = divmod(slot, block_size)
            ring = in_block % 4
            group = in_block // 4
            return block * 4 + ring, (
                block * (block_size // 4) + group if ring == 3 else None
            )

        for start in range(2 * block_size + 1):
            for valid_mask in range(1, 1 << 5):
                raw = [1000 + i for i in range(16)]
                compressed = [2000 + i for i in range(16)]
                expected_raw = raw.copy()
                expected_compressed = compressed.copy()
                undo = [None] * 5
                for row in range(5):
                    if not (valid_mask & (1 << row)):
                        continue
                    raw_offset, compressed_offset = offsets(start + row)
                    undo[row] = (
                        raw_offset,
                        raw[raw_offset],
                        compressed_offset,
                        compressed[compressed_offset]
                        if compressed_offset is not None
                        else None,
                    )
                    raw[raw_offset] = 10000 + row
                    if compressed_offset is not None:
                        compressed[compressed_offset] = 20000 + row
                for row in range(4, -1, -1):
                    if undo[row] is None:
                        continue
                    raw_offset, raw_value, compressed_offset, compressed_value = undo[row]
                    raw[raw_offset] = raw_value
                    if compressed_offset is not None:
                        compressed[compressed_offset] = compressed_value
                self.assertEqual(raw, expected_raw, (start, valid_mask))
                self.assertEqual(compressed, expected_compressed, (start, valid_mask))

    def test_topology_is_part_of_graph_key(self):
        self.assertIn("let qwen4_k5_attention_topology =", self.verify)
        self.assertIn(".map(|plan| plan.topology())", self.verify)
        self.assertIn(
            "let cache_key = (seq.slot_idx, k, pack_key, qwen4_k5_attention_topology)",
            self.verify,
        )
        self.assertIn("Option<Qwen4K5DeviceAttentionTopology>", self.types)
        self.assertNotIn("let _topology", self.decode)

    def test_graph_admission_is_fail_closed_to_topology_safe_qsa_routes(self):
        self.assertIn("checked_next_power_of_two()", self.device_attention)
        self.assertIn("if num_tokens != 5", self.device_attention)
        self.assertIn("|| !metadata.qwen4_qsa_required", self.device_attention)
        graph_guard = self.verify.split("let use_graphs =", 1)[1].split(";", 1)[0]
        self.assertIn("qwen4_qsa_graph_route_safe(", graph_guard)
        self.assertIn("seq.seq_len", graph_guard)
        self.assertIn("qwen4_k5_attention_topology.is_some()", graph_guard)
        self.assertIn(
            "crate::layers::qwen4_qsa::graph_safe_enabled()",
            graph_guard,
        )
        helper = self.verify.split("fn qwen4_qsa_graph_route_safe(", 1)[1]
        helper = helper.split("\n}", 1)[0]
        self.assertIn("!qsa_required", helper)
        self.assertNotIn("frame_k", helper)

        exclusions = self.device_attention.split(
            'ensure!(\n            exact_zero_env("ATLAS_QWEN4_K5_BATCH_HYPER")?', 1
        )[1].split(");", 1)[0]
        self.assertIn('!exact_env("ATLAS_PAGED_DECODE_SPLITK")?', exclusions)

    def test_graph_safe_preprojected_route_retains_k5_transaction(self):
        snapshot = self.decode.index("snapshot_verify_k5_row(")
        routed = self.decode.index("attention_forward_preprojected_device(", snapshot)
        self.assertLess(snapshot, routed)
        routed_body = self.decode[routed : self.decode.index("block_table,", routed)]
        self.assertIn("plan.row(row)?", routed_body)
        device_helper = self.attention.split(
            "fn attention_forward_preprojected_device(", 1
        )[1].split("/// Execute one serial verifier row", 1)[0]
        self.assertIn("Some(device_row)", device_helper)
        self.assertIn("true,", device_helper)

        graph_safe = self.qsa_route.split(
            "else if crate::layers::qwen4_qsa::graph_safe_enabled()", 1
        )[1].split("} else {", 1)[0]
        self.assertIn("update_and_select_graph_safe(", graph_safe)
        self.assertNotIn("host_seq_len.expect", graph_safe)

    def test_specialized_rows_v2_partitions_every_sparse_rank_once(self):
        body = self.cuda.split(
            "qwen4_qsa_sparse_attention_bf16_partial_rows_v2(", 1
        )[1]
        body = body.split("qwen4_qsa_sparse_attention_bf16_reduce_rows(", 1)[0]
        self.assertIn("split + warp * num_splits", body)
        self.assertNotIn("split * 32u + warp", body)

        output_width = 2051
        num_splits = 8
        spa_warps = 8
        visits = [0] * output_width
        for split in range(num_splits):
            for warp in range(spa_warps):
                for slot in range(
                    split + warp * num_splits,
                    output_width,
                    num_splits * spa_warps,
                ):
                    visits[slot] += 1
        self.assertEqual(set(visits), {1})

        old_visits = [0] * output_width
        for split in range(num_splits):
            for warp in range(spa_warps):
                for slot in range(
                    split * 32 + warp,
                    output_width,
                    num_splits * spa_warps,
                ):
                    old_visits[slot] += 1
        self.assertEqual(sum(count == 0 for count in old_visits), 1536)

    def test_multi_row_qsa_is_quarantined_at_sparse_boundary(self):
        self.assertIn("VERIFY_ROWS_DENSE_ENDPOINT: usize = 2_048", self.qsa)
        admission = self.qsa.split(
            "pub(crate) fn update_and_select_verify_rows(", 1
        )[1].split("let use_device_meta", 1)[0]
        self.assertIn("!verify_rows_context_safe(seq_len, num_tokens)", admission)

    def test_route_telemetry_records_executed_dispatches_not_geometry(self):
        journal_prepare = self.decode.split(
            ".prepare_verify_k5_transaction(kv_cache, ctx.gpu)?;", 1
        )[0].rsplit("let qsa_journal =", 1)[1]
        self.assertIn("if qsa_journal {", journal_prepare)
        self.assertNotIn("verify_route_telemetry_enabled", journal_prepare)
        self.assertIn(
            'const VERIFY_ROUTE_TELEMETRY_ENV: &str = "ATLAS_QWEN4_QSA_ROUTE_TELEMETRY"',
            self.qsa,
        )
        self.assertIn("QSA_VERIFY_ROUTE_RECEIPT route={route_label}", self.qsa)
        self.assertIn("frame_id={frame_id}", self.qsa)
        self.assertIn("VERIFY_ROUTE_FRAME_ID.fetch_add(1", self.qsa)
        self.assertIn("if layer_idx == 0", self.qsa)
        for route in ("DensePaged", "SparseQsa", "HighSpeedSwap", "Mla"):
            self.assertIn(route, self.qsa)
        for field in (
            "dense_paged_rows",
            "sparse_qsa_rows",
            "high_speed_swap_rows",
            "mla_rows",
            "sparse_rows_v2_frame_total",
        ):
            self.assertIn(field, self.qsa)
        call = self.decode.index("crate::layers::qwen4_qsa::record_verify_route(")
        admission = self.decode.index("let qsa_rows_precomputed =")
        dispatch = self.decode.index("let attention_result =", admission)
        completed = self.decode.index("attention_result?", dispatch)
        stage_mark = self.decode.index(
            'spark_runtime::cublaslt::stage_timer::mark("v_attn"', completed
        )
        self.assertLess(admission, call)
        self.assertLess(dispatch, completed)
        self.assertLess(completed, call)
        self.assertLess(call, stage_mark)
        body = self.decode[call:stage_mark]
        self.assertIn("attention_route,", body)
        self.assertNotIn("qsa_rows_precomputed", body)
        route_selection = self.decode[dispatch:completed]
        self.assertNotIn("let mut handled_sparse_rows = false", route_selection)
        self.assertIn("attention_forward_verify_host_row_with_route(", route_selection)
        self.assertIn("row_routes.record(row_route)", route_selection)
        self.assertIn("VerifyAttentionRoute::RowSerial(row_routes)", route_selection)

        dispatch_site = self.attention.split("let dispatch_route =", 1)[1].split(
            'stage_timer::mark("core"', 1
        )[0]
        self.assertIn("VerifyRowAttentionRoute::HighSpeedSwap", dispatch_site)
        self.assertIn("VerifyRowAttentionRoute::SparseQsa", dispatch_site)
        self.assertIn("VerifyRowAttentionRoute::DensePaged", dispatch_site)
        self.assertIn("if matches!(", dispatch_site)
        self.assertIn("Ok((output, dispatch_route))", self.attention)

        sparse_launch = self.qsa_route.split(
            "let sparse_rows_kernel = qsa.sparse_attention_bf16_rows(", 1
        )[1]
        self.assertIn("SparseRowsKernel::Legacy", sparse_launch)
        self.assertIn("SparseRowsKernel::V2", sparse_launch)

    def test_sparse_multirow_route_admits_pure_sparse_frames_and_rejects_straddling(self):
        admission = self.qsa.split(
            "pub(crate) fn update_and_select_verify_rows(", 1
        )[1].split("let use_device_meta", 1)[0]
        self.assertIn("!verify_rows_context_safe(seq_len, num_tokens)", admission)
        sparse_dispatch = self.qsa_route.split("let all_sparse_qsa =", 1)[1].split(
            "if !all_sparse_qsa", 1
        )[0]
        self.assertIn("seq_len >= 2048", sparse_dispatch)

    def test_straddling_verify_rows_are_rejected_and_post_2048_graph_requires_graph_safe_qsa(self):
        graph_guard = self.verify.split("let use_graphs =", 1)[1].split(";", 1)[0]
        self.assertIn("qwen4_qsa_graph_route_safe(", graph_guard)
        self.assertIn("seq.seq_len", graph_guard)
        dense_helper = self.verify.split(
            "fn qwen4_qsa_dense_prefix_graph_eligible(", 1
        )[1].split("\n}", 1)[0]
        self.assertIn("VERIFY_ROWS_DENSE_ENDPOINT", dense_helper)
        self.assertIn("host_seq_len", dense_helper)
        self.assertIn(".checked_add(rows)", dense_helper)
        helper = self.verify.split("fn qwen4_qsa_graph_route_safe(", 1)[1]
        helper = helper.split("\n}", 1)[0]
        self.assertIn("qwen4_qsa_dense_prefix_graph_eligible(", helper)
        self.assertIn("graph_safe_qsa && host_seq_len >= 2_048", helper)

    def test_sparse_rows_error_clears_precomputed_cursor(self):
        sparse = self.decode.split("let attention_result =", 1)[1]
        body, tail = sparse.split("attention_result?", 1)
        self.assertIn("self.try_qsa_sparse_attention_rows(", body)
        self.assertIn("if qsa_rows_precomputed", body)
        self.assertIn("return Ok(route_telemetry.then_some(route))", body)
        self.assertIn("qsa.clear_precomputed_verify_rows()", body)
        self.assertIn("spark_runtime::cublaslt::stage_timer::mark", tail)

    def test_qsa_k5_route_and_overlap_fail_closed_before_effects(self):
        admission = self.verify.index("Qwen4 QSA K5 requires the qualified")
        first_effect = self.verify.index("self.ssm_h_to_f16_dispatch(seq)?")
        self.assertLess(admission, first_effect)
        self.assertIn("&& !transactional_qsa", self.attention)
        journal = self.decode.split("let qsa_journal =", 1)[1].split(";", 1)[0]
        self.assertIn("metadata.qwen4_qsa_required", journal)
        self.assertIn("verify_journal_rows(num_tokens)", journal)

    def test_unsupported_cross_sequence_qsa_batch_is_rejected_twice(self):
        self.assertIn("&& !a.seq.qwen4_qsa_required", self.mtp_step)
        self.assertIn("!seq.qwen4_qsa_required", self.batched)
        self.assertIn("does not implement per-sequence QSA transactions", self.batched)

    def test_error_shadow_and_terminal_paths_use_qsa_only_rollback(self):
        prod = self.server.split("#[cfg(test)]", 1)[0]
        self.assertGreaterEqual(prod.count("commit_qwen4_qsa_for_sequence("), 3)
        self.assertIn("transactional K1 replay is not implemented", self.server)
        helper = self.server.split("pub(super) fn rollback_failed_verify_frame", 1)[1]
        helper = helper.split("\n}", 1)[0]
        self.assertIn("model.abort_verify_state(seq, k_verify)", helper)
        terminal = self.server.split("match plan.last_inter_slot {", 1)[1]
        none = terminal.split("Some(last_inter_slot) if tree_mode", 1)[0]
        self.assertIn("rollback_failed_verify_frame(model, &mut a.seq, k_verify)", none)

    def test_all_scheduler_abort_families_restore_qsa_and_recurrent_state(self):
        self.assertGreaterEqual(self.server.count("rollback_failed_verify_frame("), 9)
        self.assertIn("sampled DFlash emitted-prefix reconciliation", self.sampled)
        self.assertIn("rollback_failed_verify_frame(", self.sampled)
        self.assertGreaterEqual(self.self_spec.count("abort_self_spec_verify("), 7)
        self.assertIn("<Self as Model>::abort_verify_state(self, seq, k)", self.ep)
        self.assertIn("if let Err(error) = self.decode_verify_dflash", self.ep)

    def test_serial_oracle_qsa_failure_reaches_common_fail_closed_abort(self):
        commit = self.server.split("let commit_res = if k5_commit_parity", 1)[1]
        commit = commit.split("let t_commit_us =", 1)[0]
        qsa_result = "let mut result = model.commit_qwen4_qsa_for_sequence(&a.seq, 0, k_verify);"
        self.assertIn(qsa_result, commit)
        fallback = commit.split(qsa_result, 1)[1]
        self.assertNotIn("commit_qwen4_qsa_for_sequence(&a.seq, 0, k_verify)?", fallback)
        guard = fallback.index("if result.is_ok()")
        restore = fallback.index("result = model.pre_verify_copy_async(&mut a.seq)")
        replay = fallback.index("for &token in tokens.iter().take(total_accepted)")
        self.assertLess(guard, restore)
        self.assertLess(restore, replay)

        failed = self.server.split("if let Err(e) = commit_res", 1)[1]
        failed = failed.split("if let Some(fixture)", 1)[0]
        self.assertIn("a.seq.seq_len = pre_verify_len", failed)
        self.assertIn("a.seq.tokens.truncate(pre_verify_len)", failed)
        self.assertIn("rollback_failed_verify_frame(model, &mut a.seq, k_verify)", failed)
        self.assertIn("model.clear_ddtree_parent_ids()", failed)
        self.assertIn('fail_speculative_step(a, "verify commit", &e)', failed)
        bootstrap_failure = (
            ROOT / "crates/spark-server/src/scheduler/mtp_bootstrap_failure.rs"
        ).read_text()
        fail_helper = bootstrap_failure.split("fn fail_speculative_step(", 1)[1]
        fail_helper = fail_helper.split("\n}", 1)[0]
        self.assertIn("a.finished = true", fail_helper)

    def test_abort_drains_secondary_before_default_stream_restore(self):
        abort = self.commit.split("pub(in crate::model) fn abort_verify_state_dispatch", 1)[1]
        abort = abort.split("pub(in crate::model) fn commit_verify_state_async_dispatch", 1)[0]
        drain = abort.index("synchronize(self.secondary_stream)")
        qsa = abort.index("commit_qwen4_qsa_k5_only_dispatch(0, k)")
        recurrent = abort.index("pre_verify_copy_async_dispatch(seq)")
        clear_tree = abort.index(".swap(false, std::sync::atomic::Ordering::Acquire)")
        self.assertLess(clear_tree, drain)
        self.assertLess(drain, qsa)
        self.assertLess(qsa, recurrent)

    def test_commit_retains_flat_tree_route_until_final_qsa_boundary(self):
        commit = self.commit.split("pub(in crate::model) fn commit_verify_state_async_dispatch", 1)[1]
        commit = commit.split("\n}\n\n/// `ATLAS_SPEC_SKIP_SSM_RESTORE_CONTROL", 1)[0]
        self.assertIn(".load(std::sync::atomic::Ordering::Acquire)", commit)
        self.assertNotIn(".swap(false", commit)

        reject = commit.split("if num_accepted == 0 {", 1)[1]
        reject = reject.split("// M8A:", 1)[0]
        reject_event = reject.index("record_event(")
        reject_clear = reject.index(".store(false, std::sync::atomic::Ordering::Release)")
        reject_qsa = reject.index("commit_qwen4_qsa_k5_only_dispatch")
        self.assertLess(reject_event, reject_clear)
        self.assertLess(reject_clear, reject_qsa)

        accepted = commit.split("if let Some(ple) = &self.qwen4_ple", 1)[1]
        accepted_ple = accepted.index("ple.rollback_and_checkpoint")
        accepted_event = accepted.index("record_event(")
        accepted_clear = accepted.index(
            ".store(false, std::sync::atomic::Ordering::Release)"
        )
        accepted_qsa = accepted.index("commit_qwen4_qsa_k5_only_dispatch")
        self.assertLess(accepted_ple, accepted_event)
        self.assertLess(accepted_event, accepted_clear)
        self.assertLess(accepted_clear, accepted_qsa)


if __name__ == "__main__":
    unittest.main()
