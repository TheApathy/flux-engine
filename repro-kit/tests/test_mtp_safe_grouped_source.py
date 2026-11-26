"""Native-MTP-safe routed grouping: source contracts and references only.

These checks read source text; they do not compile or execute Rust, CUDA or
a model. They pin the trace that admits plain decode, native MTP (--speculative,
in-checkpoint or --mtp-from-path), and DFlash (--dflash) with
ATLAS_FAST_LOAD_ROUTED_GROUPS=1 while self-speculative and n-gram drafters
stay refused.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
TARGET_PREFIX = "model.language_model.layers."


def read(path):
    return (ROOT / path).read_text()


def layer_prefix(weight_prefix, layer):
    """Reference of ModelConfig::layer_prefix (atlas-core config/methods.rs)."""
    return f"model.layers.{layer}" if not weight_prefix else f"{weight_prefix}.layers.{layer}"


def grouping_admitted(speculative=False, dflash=False, self_speculative=False, ngram=False):
    """Reference of the serve-side grouping guard's drafter terms."""
    return not self_speculative and not ngram


class MtpSafeGroupedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read("crates/spark-runtime/src/fast_weights/routed_groups.rs")
        cls.plan_tests = read("crates/spark-runtime/src/fast_weights/routed_groups_tests.rs")
        cls.owner_tests = read("crates/spark-runtime/src/weights/routed_backing_tests.rs")
        cls.store = read("crates/spark-runtime/src/weights.rs")
        cls.serve = read("crates/spark-server/src/main_modules/serve_phases/weights.rs")
        cls.serve_load = read("crates/spark-server/src/main_modules/serve_load.rs")
        cls.build = read("crates/spark-model/src/factory/build.rs")
        cls.qwen35 = read("crates/spark-model/src/weight_loader/qwen35.rs")
        cls.attach = read("crates/spark-model/src/weight_loader/qwen35/load_layers.rs")
        cls.methods = read("crates/atlas-core/src/config/methods.rs")
        cls.mtp_loader = read("crates/spark-model/src/weight_loader/qwen4_mtp.rs")
        cls.mtp_head = read("crates/spark-model/src/layers/qwen4_mtp.rs")

    def test_mtp_tensor_is_never_planned_into_a_group(self):
        self.assertIn(f'const PREFIX: &str = "{TARGET_PREFIX}";', self.plan)
        member = self.plan.split("fn parse_member(", 1)[1].split("\nimpl RoutedBackingLoad", 1)[0]
        self.assertIn("meta.name.strip_prefix(PREFIX) else { return Ok(None);", " ".join(member.split()))
        self.assertFalse("mtp.layers.0.mlp.experts.0.gate_proj.weight".startswith(TARGET_PREFIX))
        inventories = self.plan.split("fn from_inventories(", 1)[1].split("let requested_bytes", 1)[0]
        inventories_ns = "".join(inventories.split())
        self.assertLess(inventories_ns.index("groupingrequirescompletecanonicaltargetroutedNVFP4inventory"),
                        inventories_ns.index('ensure!(!by_name.keys().any(|name|name.starts_with("mtp.")),'))
        self.assertIn("fn native_mtp_experts_beside_the_target_inventory_are_never_planned_or_given_a_destination",
                      self.plan_tests)

    def test_mtp_layer_prefix_cannot_attach_a_target_group(self):
        mtp_config = self.qwen35.split("pub(crate) fn qwen4_mtp_config(", 1)[1].split("\n}\n", 1)[0]
        self.assertIn('config.weight_prefix = "mtp".to_string();', mtp_config)
        self.assertIn("config.pp_layer_offset = 0;", mtp_config)
        self.assertIn('format!("{}.layers.{global_idx}", self.weight_prefix)', self.methods)
        self.assertEqual(layer_prefix("mtp", 0), "mtp.layers.0")
        self.assertFalse(layer_prefix("mtp", 0).startswith(TARGET_PREFIX))
        self.assertIn("let lp = config.layer_prefix(i);", self.attach)
        self.assertIn("if let Some(groups) = store.routed_original_groups(&lp)", self.attach)
        self.assertIn('format!("{PREFIX}{layer}")', self.plan)
        self.assertIn('store.routed_original_groups("mtp.layers.0").is_none()', self.owner_tests)

    def test_k5_build_order_releases_target_groups_before_any_mtp_read(self):
        body = self.build.split("pub fn build_model(", 1)[1]
        order = [
            "|| loader.load_layers(store, &config, gpu.as_ref(), &attn_layer_dtypes)",
            "|| maybe_run_minimax_m2_moe_transpose(&config, gpu.as_ref(), &mut layers)",
            "let mtp_weights = loader.load_mtp_weights_multi(store, &config, gpu.as_ref())?;",
            "crate::weight_loader::qwen4_mtp::classify_qwen4_mtp_store(store, &config)?",
            "crate::weight_loader::load_qwen4_mtp_layer(",
            "crate::layers::Qwen4MtpHead::new(",
            "let release_mode = select_qwen4_release_mode(",
        ]
        positions = [body.index(fragment) for fragment in order]
        self.assertEqual(positions, sorted(positions))

    def test_mtp_readers_use_only_mtp_names_and_released_targets_fail_closed(self):
        for source in (self.mtp_loader, self.mtp_head):
            self.assertNotIn(TARGET_PREFIX, source)
        classify = self.mtp_loader.split("pub(crate) fn classify_qwen4_mtp_metadata", 1)[1]
        self.assertIn('if name.starts_with("mtp.") {', classify.split("if mtp_count == 0", 1)[0])
        self.assertIn("fn target_group_release_refuses_released_names_and_leaves_native_mtp_tensors_readable",
                      self.owner_tests)
        self.assertIn("released: routed_groups.released.clone()", self.store)

    def test_mtp_sidecar_stays_individual_and_cannot_carry_groups(self):
        # Bundled MTP sidecar resolution resolves the sidecar dir first and
        # checks --speculative in the sidecar plan; the region starts there.
        sidecar = self.serve_load.split("let resolved_mtp_dir: Option<std::path::PathBuf> =", 1)[1].split(
            "store.merge_disjoint(mtp_store)?;", 1)[0]
        self.assertIn('anyhow::ensure!(args.speculative, "--mtp-from-path requires --speculative");', sidecar)
        self.assertIn("spark_runtime::fast_weights::FastSafetensorsLoader::new()", sidecar)
        self.assertNotIn("grouped_routed_originals", sidecar)
        self.assertIn('ensure!(other_groups.is_empty(),"cannotmergeagroupedsidecarstore");', "".join(self.store.split()))

    def test_guard_admits_native_mtp_and_dflash_while_refusing_self_and_ngram(self):
        gate = self.serve.split("let grouped_routed_originals =", 1)[1].split("let store =", 1)[0]
        condition = gate.split("anyhow::ensure!(", 1)[1].split(',\n', 1)[0]
        self.assertNotIn("args.speculative", condition)
        self.assertNotIn("args.dflash", condition)
        for fragment in ("!args.self_speculative", "!args.ngram_speculative",
                         "config.tp_world_size == 1", "ep_size == 1", "use_fast_load"):
            self.assertIn(fragment, condition)
        self.assertNotIn("||", condition)
        self.assertTrue(grouping_admitted())
        self.assertTrue(grouping_admitted(speculative=True))
        self.assertTrue(grouping_admitted(dflash=True))
        for mask in range(1, 16):
            flags = dict(zip(("speculative", "dflash", "self_speculative", "ngram"),
                             (bool(mask & (1 << bit)) for bit in range(4))))
            with self.subTest(**flags):
                self.assertEqual(grouping_admitted(**flags), (mask & ~0b0011) == 0)


if __name__ == "__main__":
    unittest.main()
