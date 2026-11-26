"""Bounded source contracts only, not Rust, native or runtime validation."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "crates/spark-server/src/main.rs"
KV = ROOT / "crates/spark-server/src/main_modules/serve_phases/kv_cache.rs"


def compact(source):
    return re.sub(r"\s+", "", source)


def function(source, name):
    return source.split("fn " + name + "(", 1)[1].split("\n}\n", 1)[0]


def validate_capacity_guard(source):
    # 90ce7ffa2 moved the resolution into resolve_prefill_budget_with_explicit_overrides.
    body = compact(function(source, "resolve_prefill_budget_with_explicit_overrides"))
    parse = "parse_max_batch_tokens_override(override_value,default_max_batch_tokens)?;"
    capacity = compact(
        """let max_batch_tokens_capacity = args
            .max_seq_len
            .checked_mul(args.max_batch_size)
            .context("--max-seq-len * --max-batch-size exceeds usize capacity")?
            .max(default_max_batch_tokens);"""
    )
    check = "anyhow::ensure!(max_batch_tokens<=max_batch_tokens_capacity,"
    for required in (parse, capacity, check):
        if required not in body:
            raise ValueError("missing strict shared capacity admission")
    if not body.index(parse) < body.index(capacity) < body.index(check):
        raise ValueError("capacity admission must follow strict numeric parsing")
    if body.index(check) > body.index("ifoverride_value.is_some()"):
        raise ValueError("oversized override must fail before acceptance logging")


def validate_alias_capture(source):
    body = compact(function(source, "sync_flux_env_aliases"))
    for required in (
        "let vars: Vec<(OsString, OsString)> = std::env::vars_os().collect();",
        "flux_env_alias_if_absent(&key, value, |alias| std::env::var_os(alias).is_some())",
        "std::env::set_var(&atlas_key, &value);",
    ):
        if compact(required) not in body:
            raise ValueError("alias capture must preserve OS values and ATLAS precedence")
    for forbidden in ("std::env::vars()", "to_string_lossy", ".to_str()"):
        if forbidden in body:
            raise ValueError("alias capture must not decode environment values")
    mapping = compact(function(source, "flux_env_alias_key"))
    if "letkey=key.to_str()?;" not in mapping or "std::env::" in mapping:
        raise ValueError("key mapping must be pure and ignore non-Unicode keys")
    selection = compact(function(source, "flux_env_alias_if_absent"))
    for required in (
        "value:OsString,",
        "letatlas_key=flux_env_alias_key(key)?;",
        "ifatlas_key_exists(&atlas_key){returnNone;}",
        "Some((atlas_key,value))",
    ):
        if required not in selection:
            raise ValueError("alias selection must preserve raw values without overriding ATLAS")


def validate_runtime_order(source):
    main = compact(function(source, "main"))
    # Synchronous `main` synchronizes aliases and pre-runtime CLI env setup
    # before invoking `run_application(entry)` which constructs Tokio.
    expected_prefix = ")->Result<()>{sync_flux_env_aliases();sync_dflash_lookup_controls();"
    if not main.startswith(expected_prefix) or not main.endswith("run_application(entry)"):
        raise ValueError("synchronous main must synchronize aliases before the runtime")
    if main.count("run_application(") != 1:
        raise ValueError("synchronous main must invoke run_application exactly once")
    if "#[tokio::main]\nasync fn run_application(entry: EntryCli) -> Result<()> {" not in source:
        raise ValueError("only the separate application entrypoint should construct Tokio")
    if source.count("#[tokio::main]") != 1:
        raise ValueError("unexpected additional Tokio entrypoint")
    application = function(source, "run_application")
    if "sync_flux_env_aliases" in application or "sync_dflash_lookup_controls" in application:
        raise ValueError("application runtime must not mutate aliases")


class StartupOverflowEnvSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = MAIN.read_text(encoding="utf-8")
        cls.kv = KV.read_text(encoding="utf-8")

    def test_numeric_override_is_bounded_after_parsing_in_the_shared_resolver(self):
        validate_capacity_guard(self.kv)

    def test_missing_checked_product_floor_or_bound_mutants_are_rejected(self):
        for old, new in (
            (".checked_mul(args.max_batch_size)", ".saturating_mul(args.max_batch_size)"),
            (".max(default_max_batch_tokens);", ";"),
            ("max_batch_tokens <= max_batch_tokens_capacity", "max_batch_tokens >= max_batch_tokens_capacity"),
        ):
            with self.subTest(mutant=old):
                self.assertEqual(self.kv.count(old), 1)
                with self.assertRaises(ValueError):
                    validate_capacity_guard(self.kv.replace(old, new, 1))

    def test_alias_capture_retains_raw_os_values_and_existing_atlas_precedence(self):
        validate_alias_capture(self.main)

    def test_lossy_or_unicode_only_capture_mutants_are_rejected(self):
        for old, new in (
            ("std::env::vars_os().collect()", "std::env::vars().collect()"),
            ("std::env::var_os(alias).is_some()", "false"),
            ("    Some((atlas_key, value))\n", "    Some((atlas_key, value.to_string_lossy().into_owned().into()))\n"),
        ):
            with self.subTest(mutant=old):
                self.assertEqual(self.main.count(old), 1)
                with self.assertRaises(ValueError):
                    validate_alias_capture(self.main.replace(old, new, 1))

    def test_meta_exclusions_and_special_mapping_names_are_preserved(self):
        mapping = function(self.main, "flux_env_alias_key")
        for key in (
            "FLUX_MODE", "FLUX_QUIET", "FLUX_MODELS_ROOT", "FLUX_ATLAS_ROOT",
            "FLUX_ACTIVE_ARGV_FILE", "FLUX_OPT_OUT", "FLUX_THINKING",
        ):
            self.assertIn('"' + key + '"', mapping)
        self.assertIn("META_KEYS.contains(&key) || suffix.is_empty()", mapping)
        for fragment in (
            '"QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB" =>',
            '"ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB".to_string()',
            '"PREFILL_WIDE" => "ATLAS_QWEN4_PREFILL_WIDE".to_string()',
            '_ => format!("ATLAS_{suffix}")',
        ):
            self.assertIn(fragment, mapping)

    def test_synchronous_main_precedes_runtime_construction(self):
        validate_runtime_order(self.main)

    def test_runtime_before_alias_and_extra_entrypoint_mutants_are_rejected(self):
        for old, new in (
            (
                "    sync_flux_env_aliases();\n    sync_dflash_lookup_controls();",
                "    run_application(entry);\n    sync_flux_env_aliases();\n    sync_dflash_lookup_controls();",
            ),
            (
                "    sync_flux_env_aliases();\n    sync_dflash_lookup_controls();",
                "    sync_flux_env_aliases();\n    run_application(entry);\n    sync_dflash_lookup_controls();",
            ),
            (
                "    sync_flux_env_aliases();\n    sync_dflash_lookup_controls();",
                "    sync_dflash_lookup_controls();\n    sync_flux_env_aliases();",
            ),
            ("fn main() -> Result<()> {", "#[tokio::main]\nasync fn main() -> Result<()> {"),
        ):
            self.assertEqual(self.main.count(old), 1)
            with self.assertRaises(ValueError):
                validate_runtime_order(self.main.replace(old, new, 1))

    def test_pure_alias_and_non_unicode_rust_regressions_are_defined(self):
        for name in (
            "meta_empty_and_unrelated_keys_do_not_alias",
            "ordinary_and_special_alias_names_are_preserved",
            "existing_atlas_values_win_even_when_empty",
            "non_unicode_keys_are_ignored_and_alias_values_are_preserved_raw",
        ):
            self.assertIn("fn " + name + "()", self.main)
        tests = self.main.split("mod flux_env_alias_tests {", 1)[1]
        self.assertNotIn("std::env::", tests)
        self.assertIn(r'OsStr::from_bytes(b"FLUX_\xff")', tests)
        self.assertIn(r'OsString::from_vec(b"\xff".to_vec())', tests)

    def test_exact_capacity_overflow_and_small_geometry_rust_regressions_are_defined(self):
        for name in (
            "batch_override_accepts_exact_sequence_capacity_but_not_larger_values",
            "batch_capacity_multiplication_overflow_fails_without_an_override",
            "speculative_rows_remain_part_of_the_override_minimum",
            "non_rung_decode_widths_reserve_their_physical_padding",
        ):
            self.assertIn("fn " + name + "()", self.kv)
        self.assertIn("[(capacity + 1).to_string(), usize::MAX.to_string()]", self.kv)
        self.assertIn('for override_value in [None, Some("8192")]', self.kv)
        self.assertIn('Some("6")', self.kv)
        self.assertIn('Some("7")', self.kv)
        self.assertIn("Some(&padded.to_string())", self.kv)
        self.assertIn("Some(&(padded + 1).to_string())", self.kv)


if __name__ == "__main__":
    unittest.main()
