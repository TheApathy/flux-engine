# SPDX-License-Identifier: AGPL-3.0-only
"""Hostile source contract for the default-off K5 device-attention seam."""

import re
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
LAYERS = ROOT / "crates/spark-model/src/layers"
ATTN = LAYERS / "qwen3_attention"
class Qwen4K5DeviceAttentionStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = (ATTN / "qwen4_device_attention.rs").read_text()
        cls.mod_rs = (ATTN / "mod.rs").read_text()
        cls.decode = (ATTN / "trait_impl/decode_inner.rs").read_text()
        cls.forward = (ATTN / "decode/attention_forward.rs").read_text()
        cls.qsa_route = (ATTN / "decode/qsa_route.rs").read_text()
        cls.qsa = (LAYERS / "qwen4_qsa.rs").read_text()
        cls.qsa_device = (LAYERS / "qwen4_qsa/device.rs").read_text()
        cls.cuda = (ROOT / "kernels/gb10/common/qwen4_qsa.cu").read_text()
        cls.loader = (
            ROOT / "crates/spark-model/src/weight_loader/qwen35/load_layers.rs"
        ).read_text()
        cls.attention_arms = (
            ROOT
            / "crates/spark-model/src/weight_loader/qwen35/load_layers/attention_arms.rs"
        ).read_text()
        cls.attention_init = (ATTN / "init.rs").read_text()
        cls.verify = (
            ROOT / "crates/spark-model/src/model/trait_impl/verify_d.rs"
        ).read_text()

    def test_selector_is_strict_default_off_and_admits_k5_or_hybrid_all_k(self):
        self.assertIn('"ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH"', self.module)
        self.assertRegex(
            self.module,
            r'None\s*\|\s*Some\("0"\)\s*=>\s*Ok\(false\)',
        )
        self.assertRegex(self.module, r'Some\("1"\)\s*=>\s*Ok\(true\)')
        self.assertIn("qwen4_hybrid_all_k()", self.module)
        adaptive = self.module.split(
            "if num_tokens != 5 && !allow_all_k", 1
        )[1].split("}", 1)[0]
        self.assertIn("return Ok(None)", adaptive)
        self.assertIn("ATLAS_QWEN4_K5_HYBRID", self.module)
        self.assertIn("qwen4_qsa_required", self.module)
    def test_exact_layer_and_topology_contract(self):
        self.assertIn(
            "[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]",
            self.module,
        )
        self.assertIn("position >= 2_048", self.module)
        self.assertIn("visible_groups.div_ceil(8)", self.module)
        self.assertIn("sequence_length.is_multiple_of(4)", self.module)
        self.assertIn("DensePaged { num_splits: 2 }", self.module)
        self.assertIn("fn topology(self)", self.module)
        self.assertIn("pool_endpoint_mask", self.module)
        self.assertIn("score_grids", self.module)
    def test_loader_global_layers_map_to_attention_ordinals(self):
        def exact_increment_contract(source):
            fp8 = source.split("LayerType::FullAttention if native_fp8 =>", 1)[1]
            fp8 = fp8.split("LayerType::FullAttention =>", 1)[0]
            nvfp4 = source.split("LayerType::FullAttention =>", 1)[1]
            nvfp4 = nvfp4.split("LayerType::LinearAttention =>", 1)[0]
            arms = (fp8, nvfp4)
            return source.count("attn_idx += 1;") == 2 and all(
                arm.count("attn_idx += 1;") == 1
                and arm.index("attn_idx,") < arm.index("layers.push")
                < arm.index("attn_idx += 1;")
                for arm in arms
            )

        self.assertTrue(exact_increment_contract(self.loader))
        double = self.loader.replace("attn_idx += 1;", "attn_idx += 2;", 1)
        self.assertFalse(exact_increment_contract(double))
        constructor = self.attention_arms.split("Qwen3AttentionLayer::new(", 1)[1]
        self.assertRegex(constructor, r"ffn,\s+attn_idx,\s+q_nvfp4")
        public_new = self.attention_init.split("pub fn new(", 1)[1].split("pub fn new_ungated", 1)[0]
        self.assertRegex(public_new, r"ffn,\s+attn_layer_idx,\s+q_nvfp4")
        stored = self.attention_init.split("Ok(Self {", 1)[1]
        storage = "qwen4_qsa: None,\n            attn_layer_idx,\n            gated,"
        self.assertIn(storage, stored)
        self.assertNotIn(storage, stored.replace("attn_layer_idx,", "attn_layer_idx: 0,"))
        global_layers = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
        mapping = [(layer, ordinal) for ordinal, layer in enumerate(global_layers)]
        self.assertEqual(mapping[0], (3, 0))
        self.assertEqual(mapping[-1], (47, 11))
        self.assertTrue(all(ordinal < len(global_layers) for _, ordinal in mapping))
        self.assertFalse(all(ordinal in global_layers for _, ordinal in mapping))
        self.assertIn(
            "self.attn_layer_idx < FULL_ATTENTION_LAYERS.len()", self.module
        )
        self.assertNotIn(
            "FULL_ATTENTION_LAYERS.contains(&self.attn_layer_idx)", self.module
        )
    def test_plan_precedes_first_candidate_effect(self):
        plan = self.decode.index("Qwen4K5DeviceAttentionPlan::from_host")
        first_effect = min(
            self.decode.index("attn_hyper.prepare_decode"),
            self.decode.index("attn_hyper.prepare_batched"),
        )
        self.assertLess(plan, first_effect)
        self.assertIn("attention_forward_preprojected_device", self.decode)
    def test_candidate_entry_has_no_host_sequence_scalar(self):
        marker = "fn attention_forward_preprojected_device("
        self.assertEqual(self.forward.count(marker), 1)
        signature = self.forward.split(marker, 1)[1].split(") ->", 1)[0]
        self.assertNotIn("seq_len", signature)
        self.assertIn("Qwen4DeviceAttentionRow", signature)
        self.assertIn("update_and_select_device", self.qsa_route)
        self.assertIn("run_paged_decode", self.forward)
    def test_device_qsa_uses_stable_position_and_length_pointers(self):
        self.assertIn("mod device;", self.qsa)
        self.assertIn("score_device: KernelHandle", self.qsa)
        self.assertIn("select_expand_device: KernelHandle", self.qsa)
        self.assertIn("fn update_and_select_device", self.qsa_device)
        self.assertIn(".arg_ptr(meta.seq_len)", self.qsa_device)
        self.assertIn(".arg_ptr(meta.positions)", self.qsa_device)
        self.assertNotRegex(
            self.qsa_device,
            r"\.arg_u32\((visible_groups|position|sequence_length)",
        )
    def test_device_qsa_preserves_incumbent_launch_order(self):
        body = self.qsa_device.split("fn update_and_select_device", 1)[1]
        markers = (
            "ops::dense_gemv(",
            "ops::rms_norm(",
            "self.stage_pool",
            "self.apply_rope(",
            "self.store_compressed",
            "self.score_device",
            "self.select_expand_device",
        )
        offsets = [body.index(marker) for marker in markers]
        self.assertEqual(offsets, sorted(offsets))

    def test_cuda_variants_retain_score_and_selector_math(self):
        score = self.cuda.split("qwen4_qsa_score_device_meta", 1)[1]
        score = score.split('extern "C"', 1)[0]
        for marker in (
            "group / groups_per_page",
            "query[h * INDEX_DIM + d]",
            "warp_sum(partial)",
            "fmaxf(partial, 0.0f)",
            "0.08838834764831845f",
        ):
            self.assertIn(marker, score)
        select = self.cuda.split("qwen4_qsa_select_expand_device_meta", 1)[1]
        select = select.split('extern "C"', 1)[0]
        for marker in (
            "heap_sift_down",
            "less_pair",
            "heap_indices[j] > key",
            "group * COMPRESS_RATIO + offset",
            "tail_start",
        ):
            self.assertIn(marker, select)

    def test_cuda_variants_load_dynamic_values_only_from_device(self):
        for symbol in (
            "qwen4_qsa_score_device_meta",
            "qwen4_qsa_select_expand_device_meta",
        ):
            definitions = re.findall(rf'void\s+{symbol}\s*\(', self.cuda)
            self.assertEqual(len(definitions), 1)
        score = self.cuda.split("qwen4_qsa_score_device_meta", 1)[1]
        score = score.split("}", 1)[0]
        self.assertIn("const unsigned int* __restrict__ sequence_length_ptr", score)
        select = self.cuda.split("qwen4_qsa_select_expand_device_meta", 1)[1]
        select = select.split("}", 1)[0]
        self.assertIn("const unsigned int* __restrict__ position_ptr", select)
        self.assertIn("const unsigned int* __restrict__ sequence_length_ptr", select)

    def test_module_is_wired_and_verify_graph_is_explicitly_gated(self):
        self.assertIn("mod qwen4_device_attention;", self.mod_rs)
        self.assertIn(
            "(!self.config.is_qwen4_exp() || qwen4_verify_graph_enabled())",
            self.verify,
        )
        self.assertIn("qwen4_qsa_graph_route_safe(", self.verify)
        self.assertNotIn("ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH", self.verify)

    def test_selected_route_is_poison_not_fallback(self):
        self.assertIn("exact_qkv.ok_or_else", self.decode)
        self.assertIn("lost exact K5 QKV admission", self.decode)
        self.assertIn("device_row.is_none() || !use_orchestrator", self.forward)
        self.assertIn("requires QSA on every attention layer", self.module)

    def test_new_source_files_are_bounded_and_spdx(self):
        budgets = {
            ATTN / "qwen4_device_attention.rs": 260,
            LAYERS / "qwen4_qsa/device.rs": 307,
            Path(__file__): 250,
        }
        for path, budget in budgets.items():
            lines = path.read_text().splitlines()
            self.assertIn("SPDX-License-Identifier: AGPL-3.0-only", lines[0])
            self.assertLessEqual(len(lines), budget)


if __name__ == "__main__":
    unittest.main()
