# SPDX-License-Identifier: AGPL-3.0-only
"""Offline source contracts and requested-byte model for phased routed slabs.

No Rust execution, compiler, GPU, model reads, or physical-memory claims.
The inline Rust mock tests remain uncompiled and unexecuted by this suite.
"""

import hashlib
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
HELPERS = "crates/spark-model/src/layers/moe/helpers_a.rs"
MODEL = "crates/spark-model/src/layers/moe/mod.rs"
INIT = "crates/spark-model/src/layers/moe/init.rs"
BUDGET = "crates/spark-model/src/factory/m2_setup.rs"
PREFLIGHT = "crates/spark-server/src/main_modules/serve_phases/preflight.rs"


def source(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def integer_constant(text, name):
    match = re.search(r"const " + name + r": usize = ([0-9_]+);", text)
    if match is None:
        raise AssertionError("missing literal constant " + name)
    return int(match.group(1).replace("_", ""))


def inner(text):
    return text.split("pub(super) fn transpose_for_prefill_unified_inner(", 1)[1].split("#[cfg(test)]", 1)[0]


def free_block(text, phase):
    position = 0 if phase == "B" else text.index("Phase C: transpose down routed experts")
    start = text.index("if !keep_originals {", position)
    opening = text.index("{", start)
    depth = 0
    for end in range(opening, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end + 1]
    raise AssertionError("unterminated free block")


def aligned(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def projection(experts, rows, columns, cursor=256):
    packed = aligned(rows, 64) * (columns // 2)
    scales = aligned(rows, 64) * (columns // 16)
    first = aligned(cursor, 256)
    scale_offset = aligned(packed, 256)
    stride = aligned(scale_offset + scales, 256)
    return {
        "first": first,
        "packed": packed,
        "scales": scales,
        "scale_offset": scale_offset,
        "stride": stride,
        "end": first + experts * stride,
    }


def requested_routed_peak(layers, gate_up_payload, down_payload, split):
    """Only routed requested-byte lifetimes, excluding every other pool."""
    current = layers * (gate_up_payload + down_payload)
    peak = current
    for _ in range(layers):
        current += gate_up_payload + 256 if split else gate_up_payload + down_payload + 256
        peak = max(peak, current)
        current -= gate_up_payload
        if split:
            current += down_payload + 256
            peak = max(peak, current)
        current -= down_payload
    return peak, current


class Qwen4TwoPhaseTransposeSourceTests(unittest.TestCase):
    def test_canonical_selector_remains_unified_target_only(self):
        selector = source(HELPERS).split("fn canonical_qwen4_routed_slab_selected(", 1)[1]
        selector = selector.split("fn validate_canonical_qwen4_routed_experts(", 1)[0]
        for condition in [
            "!keep_originals", "config.is_qwen4_exp()",
            "config.num_hidden_layers == QWEN4_ROUTED_SLAB_LAYERS",
            "config.num_experts == QWEN4_ROUTED_SLAB_EXPERTS",
            "config.hidden_size == QWEN4_ROUTED_SLAB_HIDDEN",
            "config.moe_intermediate_size == QWEN4_ROUTED_SLAB_INTER",
            "config.ep_world_size == 1", "config.ep_rank == 0",
        ]:
            self.assertIn(condition, selector)
        self.assertNotIn("mtp", selector)

    def test_distinct_backend_owned_bases_are_initialized_and_registered_before_use(self):
        fields = source(MODEL)
        init = source(INIT)
        for name in ["routed_gate_up_transpose_slab", "routed_down_transpose_slab"]:
            self.assertIn(name + ": Option<DevicePtr>", fields)
            self.assertIn(name + ": None", init)
        allocator = source(HELPERS).split("fn allocate_routed_transpose_slab(", 1)[1]
        allocator = allocator.split("fn canonical_qwen4_routed_slab_selected(", 1)[0]
        self.assertLess(allocator.index("owner.is_none()"), allocator.index("let slab = allocate(allocation_bytes)?"))
        self.assertLess(allocator.index("!slab.is_null()"), allocator.index("*owner = Some(slab)"))
        self.assertLess(allocator.index("*owner = Some(slab)"), allocator.index("Ok(slab)"))
        self.assertNotIn("free(", allocator)

    def test_down_allocation_waits_until_full_existing_phase_b_and_propagates_error(self):
        text = inner(source(HELPERS))
        gate_up = text.index("let routed_gate_up_slab =")
        phase_a = text.index("Phase A: transpose gate+up routed experts")
        phase_b = text.index("Phase B: free gate+up untransposed")
        shared_free = text.index("self.weights.shared_expert.up_proj.weight_scale = DevicePtr::NULL")
        down = text.index("let routed_down_slab =")
        down_transpose = text.index("let down_t =")
        phase_d = text.index("Phase D: free down untransposed")
        self.assertLess(gate_up, phase_a)
        self.assertIn("layout.gate_up_allocation_bytes", text[gate_up:phase_a])
        self.assertNotIn("layout.allocation_bytes", text[gate_up:phase_a])
        self.assertLess(phase_a, phase_b)
        self.assertLess(phase_b, shared_free)
        self.assertLess(shared_free, down)
        self.assertLess(down, down_transpose)
        self.assertLess(down_transpose, phase_d)
        allocation = text[down:down_transpose]
        self.assertIn("&mut self.routed_down_transpose_slab", allocation)
        self.assertIn("layout.down_allocation_bytes", allocation)
        self.assertIn("|bytes| gpu.alloc(bytes)", allocation)
        self.assertIn(")?)", allocation)
        for fallback in ["unwrap", "Err(", "alloc_managed", "or_else", "transpose_for_gemm"]:
            self.assertNotIn(fallback, allocation)
        self.assertNotIn("gpu.alloc(layout.allocation_bytes)", text)

    def test_existing_original_free_blocks_are_byte_identical(self):
        text = inner(source(HELPERS))
        # Re-pinned: Phase B/D synchronize `gpu.default_stream()` once when
        # GPU slab transpose or owned routed backing groups are active; the
        # original per-expert frees remain unchanged in the `else` branch.
        for phase, digest in [
            ("B", "bdfdacbe8642ca767a24bcd19710f51492d018928d92f803fdbd1883f4e2de05"),
            ("D", "4db1d2f4478218b87ffa2e30be67d2cfb2128ed691312ce1f20f3d2b48c89dd9"),
        ]:
            self.assertEqual(hashlib.sha256(free_block(text, phase).encode()).hexdigest(), digest)

    def test_each_projection_uses_its_correct_slab_and_checked_layout(self):
        text = source(HELPERS)
        layout = text.split("impl RoutedExpertSlabLayout {", 1)[1].split("fn allocate_routed_transpose_slab(", 1)[0]
        self.assertIn("let down = RoutedProjectionSlabLayout::build(\n            ROUTED_TRANSPOSE_SLAB_ALIGNMENT,", layout)
        self.assertIn("routed_slab_align_up(up.end_offset)?", layout)
        self.assertIn("routed_slab_align_up(down.end_offset)?", layout)
        self.assertIn(".checked_add(down_allocation_bytes)", layout)
        body = inner(text)
        self.assertIn("(routed_slab_layout, routed_gate_up_slab)", body)
        self.assertIn("(routed_slab_layout, routed_down_slab)", body)
        self.assertIn("layout.gate,", body)
        self.assertIn("layout.up,", body)
        self.assertIn("layout.down,", body)

    def test_requested_layout_geometry_and_per_base_ranges(self):
        text = source(HELPERS)
        experts = integer_constant(text, "QWEN4_ROUTED_SLAB_EXPERTS")
        hidden = integer_constant(text, "QWEN4_ROUTED_SLAB_HIDDEN")
        intermediate = integer_constant(text, "QWEN4_ROUTED_SLAB_INTER")
        gate = projection(experts, intermediate, hidden)
        up = projection(experts, intermediate, hidden, gate["end"])
        down = projection(experts, hidden, intermediate)
        self.assertEqual(up["end"], 943_718_656)
        self.assertEqual(down["end"], 471_859_456)
        self.assertEqual(up["end"] + down["end"], 1_415_578_112)
        self.assertEqual(integer_constant(text, "QWEN4_ROUTED_SLAB_PAYLOAD"), 1_415_577_600)
        count = 0
        for projections, size in [([gate, up], up["end"]), ([down], down["end"])]:
            ranges = []
            for layout in projections:
                for expert in range(experts):
                    packed = layout["first"] + expert * layout["stride"]
                    scales = packed + layout["scale_offset"]
                    ranges.extend([(packed, packed + layout["packed"]), (scales, scales + layout["scales"])])
            ranges.sort()
            count += len(ranges)
            self.assertTrue(all(256 <= start < end <= size and start % 256 == 0 for start, end in ranges))
            self.assertTrue(all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])))
        self.assertEqual(count, 3072)
        self.assertIn("QWEN4_ROUTED_SLAB_PAYLOAD + 2 * ROUTED_TRANSPOSE_SLAB_ALIGNMENT", text)

    def test_requested_lifetime_model_not_physical_or_whole_startup_peak(self):
        text = source(HELPERS)
        layers = integer_constant(text, "QWEN4_ROUTED_SLAB_LAYERS")
        gate_up = integer_constant(text, "QWEN4_PHASE_B_REQUESTED_BYTES")
        down = integer_constant(text, "QWEN4_PHASE_D_REQUESTED_BYTES")
        old_one_layer = requested_routed_peak(1, gate_up, down, False)
        split_one_layer = requested_routed_peak(1, gate_up, down, True)
        self.assertEqual(old_one_layer[0] - split_one_layer[0], 450 * 1024 * 1024)
        old = requested_routed_peak(layers, gate_up, down, False)
        split = requested_routed_peak(layers, gate_up, down, True)
        self.assertEqual(split[1] - old[1], 12_288)
        self.assertEqual(old[0] - split[0], 471_847_168)
        self.assertEqual(split[1], 67_947_749_376)

    def test_all_layout_admission_and_projection_keep_full_conservative_sum(self):
        budget = source(BUDGET)
        self.assertIn("unified_layer_bytes: checked_transpose_sum(&[full_layer, host_transient, 512])?", budget)
        self.assertIn("unified_layer_bytes: 21_552", budget)
        budget_compact = re.sub(r",(?=[)\]])", "", re.sub(r"\s+", "", budget))
        for contract in [
            'ensure_retained_pass_above_floor(gpu,"hybrid",retained_budget.full_pass_bytes,safety)?',
            'ensure_retained_pass_above_floor(gpu,"full",retained_budget.full_pass_bytes,safety)?',
            'ensure_retained_pass_above_floor(gpu,"gate+up",retained_budget.gate_up_pass_bytes,safety)?',
        ]:
            self.assertIn(contract, budget_compact)
        projection = source(PREFLIGHT)
        self.assertIn("const ROUTED_TRANSPOSED_SLAB_BYTES: usize = 1_415_578_112;", projection)
        self.assertIn("this estimate takes no reclaim credit", projection)
        self.assertIn("assert_eq!(plain, 1_415_578_112 + lm_head)", source("crates/spark-server/src/main_modules/serve_phases/preflight_projection_tests.rs"))

    def test_receipt_reports_two_bases_and_allocation_order_without_memory_claim(self):
        text = inner(source(HELPERS))
        for contract in [
            'schema = "atlas.moe-transposed-expert-slab.v2"', "slab_count = 2",
            "gate_up_allocation_bytes = layout.gate_up_allocation_bytes",
            "down_allocation_bytes = layout.down_allocation_bytes",
            "down_allocation_after_phase_b = true", "base_unexposed = true",
        ]:
            self.assertIn(contract, text)

    def test_mock_rust_definitions_cover_padding_tables_owner_and_allocation_failure(self):
        text = source(HELPERS)
        for name in [
            "phase_slab_owners_fail_closed_without_overwriting_prior_owner",
            "small_phase_slabs_preserve_padded_bytes_tables_and_original_lifetimes",
            "phase_slab_sizing_rejects_overflow",
        ]:
            self.assertIn("fn " + name + "()", text)
        self.assertIn("RoutedExpertSlabLayout::build(2, 64, 32)?", text)
        self.assertIn("injected down allocation failure", text)
        self.assertIn("assert_eq!(gate_up_owner, Some(first))", text)
        self.assertIn("assert_transposed_table(&gpu, &down)?", text)


if __name__ == "__main__":
    unittest.main()
