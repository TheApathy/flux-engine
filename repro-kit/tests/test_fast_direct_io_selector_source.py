"""Source contracts only, not execution of the Rust parser or fast loader."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class FastDirectIoSelectorSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "crates/spark-runtime/src/fast_weights/mod.rs").read_text()
        cls.production, cls.native = cls.source.split("#[cfg(test)]", 1)
        cls.parser = cls.production.split("fn effective_direct_io_tensor_cap(", 1)[1].split(
            "/// Diagnostic model", 1
        )[0]
        cls.load = cls.production.split("impl WeightLoader for FastSafetensorsLoader", 1)[1].split(
            "fn load_trace_enabled()", 1
        )[0]
        cls.shard = cls.production.split("fn load_shard_fast(", 1)[1]

    def test_default_is_unchanged_and_unset_retains_configured_value(self):
        self.assertIn("pub const DEFAULT_DIRECT_IO_TENSOR_CAP: usize = 5000;", self.production)
        self.assertEqual(
            self.production.count("direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP"), 2
        )
        self.assertIn("return Ok(configured_cap);", self.parser)

    def test_parser_rejects_invalid_bytes_before_positive_usize_conversion(self):
        for guard in (
            "value.to_str().ok_or_else(invalid)?",
            "value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit())",
            "value.parse::<usize>().map_err(|_| invalid())?",
            "if cap == 0",
        ):
            self.assertIn(guard, self.parser)
        self.assertLess(self.parser.index("value.is_empty()"), self.parser.index("parse::<usize>"))
        self.assertNotIn("trim(", self.parser)

    def test_environment_is_read_once_before_checkpoint_and_gpu_operations(self):
        read = "std::env::var_os(DIRECT_IO_TENSOR_CAP_ENV)"
        self.assertEqual(self.production.count(read), 1)
        self.assertLess(self.load.index(read), self.load.index("resolve_shards(model_dir)?"))
        self.assertLess(self.load.index("effective_direct_io_tensor_cap("), self.load.index("gpu."))
        self.assertIn("direct_io_override.as_deref()", self.load)
        self.assertNotIn("set_var", self.source)

    def test_override_log_is_diagnostic_and_not_a_success_claim(self):
        self.assertIn("if direct_io_override.is_some()", self.load)
        self.assertIn("diagnostic {DIRECT_IO_TENSOR_CAP_ENV}={direct_io_tensor_cap}", self.load)
        self.assertIn("selection only", self.load)
        self.assertIn("O_DIRECT open may still fall back to buffered reads", self.load)

    def test_effective_cap_reaches_both_normal_and_extra_shards(self):
        compact = " ".join(self.load.split())
        self.assertIn(
            "load_shard_fast( shard_path, tensor_filter.as_deref(), gpu, &skip_fn, "
            "self.try_direct_io, direct_io_tensor_cap,", compact
        )
        self.assertIn(
            "load_shard_fast( &extra, None, gpu, &no_skip, self.try_direct_io, "
            "direct_io_tensor_cap,", compact
        )

    def test_filtering_explicit_disable_and_open_fallback_are_preserved(self):
        decision = "let wants_direct = try_direct_io && tensors.len() <= direct_io_tensor_cap;"
        self.assertIn(decision, self.shard)
        self.assertLess(self.shard.index("tensors.retain(|t| allow_set"), self.shard.index(decision))
        self.assertLess(self.shard.index("tensors.retain(|t| !skip_fn"), self.shard.index(decision))
        self.assertIn(".then(|| direct_io::open_direct(shard_path))", self.shard)
        self.assertIn("falling back to buffered reads", self.shard)
        self.assertIn("let data_fd = direct_file.as_ref().unwrap_or(&buffered_file)", self.shard)

    def test_memory_guards_scalar_pool_and_owned_reader_are_preserved(self):
        for existing in (
            "check_oom_guard(", "estimate_load_bytes(", "estimate_has_fp8(",
            "build_expert_scalar_slab_layout(&tensors)?", "let reader_file = data_fd",
            'context("failed to duplicate fast-loader reader descriptor")?',
            "let raw_fd = reader_file.as_raw_fd()", "bounce.drain()?",
        ):
            self.assertIn(existing, self.production)

    def test_native_parser_regressions_are_defined_not_executed(self):
        for name in (
            "direct_io_cap_unset_preserves_configured_values",
            "direct_io_cap_accepts_only_positive_ascii_decimal_usize",
            "direct_io_cap_rejects_zero_whitespace_signs_unicode_and_overflow",
            "direct_io_cap_rejects_non_unicode_environment_bytes",
        ):
            self.assertIn("fn " + name, self.native)
        self.assertIn('format!("{}0", usize::MAX)', self.native)
        self.assertIn("OsStr::from_bytes(&[b'2', 0xff, b'0'])", self.native)


if __name__ == "__main__":
    unittest.main()
