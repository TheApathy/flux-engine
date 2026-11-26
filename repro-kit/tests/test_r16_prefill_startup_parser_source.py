"""Bounded source wiring checks only. These do not execute Rust or load a model."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASES = ROOT / "crates/spark-server/src/main_modules/serve_phases"


def compact(source):
    return re.sub(r"\s+", "", source)


def function(source, name):
    pattern = r"fn " + re.escape(name) + r"(?:<[^>]*>)?\("
    return re.split(pattern, source, maxsplit=1)[1].split("\n}\n", 1)[0]


def validate_override_reader(source):
    body = compact(function(source, "resolve_prefill_budget"))
    for required in (
        "let override_value=std::env::var_os(MAX_BATCH_TOKENS_SELECTOR);",
        "let override_value=selector_text(MAX_BATCH_TOKENS_SELECTOR,override_value.as_deref())?;",
        "resolve_prefill_budget_with_override(args,ssm_prefill_chunk,override_value)",
    ):
        if compact(required) not in body:
            raise ValueError("override reader is not bound to strict optional Unicode parsing")
    if ".ok()" in body or "unwrap_or" in body:
        raise ValueError("environment errors must not become absent overrides")


def validate_override_parser(source):
    body = compact(function(source, "parse_max_batch_tokens_override"))
    for required in (
        "let Some(value)=value else {return Ok(minimum);};",
        "!value.is_empty()&&value.bytes().all(|byte|byte.is_ascii_digit())",
        'value.parse::<usize>().with_context(||format!("{MAX_BATCH_TOKENS_SELECTOR}exceedsusizecapacity"))?;',
        "tokens>=minimum",
        "Ok(tokens)",
    ):
        if compact(required) not in body:
            raise ValueError("missing strict override parser control")
    if ".ok()" in body or "unwrap_or" in body or "tracing::warn!" in body:
        raise ValueError("invalid overrides must not use a fallback")
    # 90ce7ffa2 (wide override) moved the resolution into
    # resolve_prefill_budget_with_explicit_overrides; the wrapper must delegate.
    wrapper = compact(function(source, "resolve_prefill_budget_with_override"))
    if "resolve_prefill_budget_with_explicit_overrides(args,ssm_prefill_chunk,override_value,wide_override,)" not in wrapper:
        raise ValueError("override wrapper does not delegate to the explicit resolution")
    resolution = compact(function(source, "resolve_prefill_budget_with_explicit_overrides"))
    binding = "parse_max_batch_tokens_override(override_value,default_max_batch_tokens)?;"
    if binding not in resolution:
        raise ValueError("resolved additive/speculative/padded minimum is not enforced")


def validate_early_wide(preflight):
    body = compact(function(preflight, "preflight_reserve"))
    resolve = "letresolved_prefill=super::resolve_prefill_budget(args,ssm_prefill_chunk)?;"
    check = "super::validate_qwen4_prefill_geometry(config,resolved_prefill.prefill_budget)?;"
    if resolve + check not in body:
        raise ValueError("WIDE must fail immediately after shared budget resolution")
    if body.index(check) > body.index("spark_runtime::buffers::BufferSizes::from_config("):
        raise ValueError("WIDE validation follows arena geometry")


class PrefillStartupParserSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kv = (PHASES / "kv_cache.rs").read_text()
        cls.preflight = (PHASES / "preflight.rs").read_text()

    def test_exact_selectors_and_strict_optional_unicode_reader(self):
        for constant, selector in (
            ("QWEN4_WIDE_SELECTOR", "ATLAS_QWEN4_PREFILL_WIDE"),
            ("QWEN4_MULTI_CHUNK_SELECTOR", "ATLAS_QWEN4_PREFILL_MULTI_CHUNK"),
            ("MAX_BATCH_TOKENS_SELECTOR", "ATLAS_MAX_BATCH_TOKENS"),
        ):
            self.assertIn(f'const {constant}: &str = "{selector}";', self.kv)
        reader = compact(function(self.kv, "selector_text"))
        self.assertIn("value.to_str().with_context(", reader)
        self.assertIn(".transpose()", reader)
        self.assertNotIn(".ok()", reader)
        validate_override_reader(self.kv)

    def test_override_uses_strict_parser_and_resolved_minimum(self):
        validate_override_parser(self.kv)
        source = compact(function(self.kv, "resolve_prefill_budget_with_explicit_overrides"))
        self.assertIn(
            "resolve_default_max_batch_tokens(prefill_budget,args.max_batch_size,spec_tokens,qwen4_batched_verify_rows,)?;",
            source,
        )
        default_min = compact(function(self.kv, "resolve_default_max_batch_tokens"))
        self.assertIn(".checked_add(decode_headroom)", default_min)
        self.assertIn(".max(spec_tokens)", default_min)
        self.assertIn("spark_runtime::buffers::padded_decode_rows(max_batch_size)", default_min)

    def test_wide_preserves_model_scope_and_disabled_multi_chunk_inertia(self):
        wrapper = compact(function(self.kv, "validate_qwen4_prefill_geometry"))
        for fragment in (
            "std::env::var_os(QWEN4_WIDE_SELECTOR)",
            "std::env::var_os(QWEN4_MULTI_CHUNK_SELECTOR)",
            "validate_qwen4_prefill_geometry_os_values(config.is_qwen4_exp(),wide.as_deref(),multi_chunk.as_deref(),prefill_budget,)",
        ):
            self.assertIn(fragment, wrapper)
        body = compact(function(self.kv, "validate_qwen4_prefill_geometry_os_values"))
        scoped = "if!is_qwen4{returnOk(());}"
        disabled = 'ifmatches!(wide,None|Some("0")){returnOk(());}'
        wide = "selector_text(QWEN4_WIDE_SELECTOR,wide)?"
        multi = "selector_text(QWEN4_MULTI_CHUNK_SELECTOR,multi_chunk)?"
        self.assertLess(body.index(scoped), body.index(wide))
        self.assertLess(body.index(disabled), body.index(multi))
        self.assertIn(
            "validate_qwen4_prefill_geometry_values(true,wide,multi_chunk,prefill_budget)", body
        )

    def test_server_and_model_wide_width_tables_match(self):
        server = function(self.kv, "validate_qwen4_prefill_geometry_values")
        model = (ROOT / "crates/spark-model/src/layers/qwen4_prefill_moe.rs").read_text()
        native = function(model, "parse_wide")
        extract = lambda body: set(re.findall(r'Some\("(\d+)"\)\s*=>', body)) - {"0"}
        self.assertEqual(extract(server), {"2048", "4096", "8192", "16384", "32768"})
        self.assertEqual(extract(server), extract(native))
        for body in (server, native):
            self.assertIn('None | Some("0")', body)
        self.assertIn('multi_chunk == Some("1")', server)
        self.assertIn("wide == prefill_budget", server)

    def test_early_preflight_is_before_weight_loading_and_late_guard_remains(self):
        validate_early_wide(self.preflight)
        serve = (PHASES.parent / "serve_load.rs").read_text()
        self.assertLess(serve.index("serve_phases::preflight_reserve("), serve.index("serve_phases::load_weight_store("))
        self.assertIn("serve_phases::validate_qwen4_prefill_geometry(&config, prefill_budget)?;", serve)

    def test_lossy_environment_reader_mutant_is_rejected(self):
        mutant = self.kv.replace(
            "std::env::var_os(MAX_BATCH_TOKENS_SELECTOR)",
            "std::env::var(MAX_BATCH_TOKENS_SELECTOR).ok()",
            1,
        )
        with self.assertRaises(ValueError):
            validate_override_reader(mutant)

    def test_smaller_override_and_missing_error_propagation_mutants_are_rejected(self):
        for old, new in (
            ("tokens >= minimum", "tokens > 0"),
            ("override_value, default_max_batch_tokens)?", "override_value, default_max_batch_tokens).unwrap_or(default_max_batch_tokens)"),
            ("byte.is_ascii_digit()", "byte.is_ascii()"),
        ):
            self.assertIn(old, self.kv)
            with self.assertRaises(ValueError):
                validate_override_parser(self.kv.replace(old, new, 1))

    def test_missing_wrong_scope_and_late_wide_guard_mutants_are_rejected(self):
        guard = "    super::validate_qwen4_prefill_geometry(config, resolved_prefill.prefill_budget)?;\n"
        self.assertEqual(self.preflight.count(guard), 1)
        for mutant in (
            self.preflight.replace(guard, "", 1),
            self.preflight.replace("config, resolved_prefill.prefill_budget", "config, 4096", 1),
            self.preflight.replace(guard, "", 1).replace("    let gdn_two_phase_bytes:", guard + "    let gdn_two_phase_bytes:", 1),
        ):
            self.assertNotEqual(mutant, self.preflight)
            with self.assertRaises(ValueError):
                validate_early_wide(mutant)

    def test_native_regression_definitions_are_present(self):
        for name in (
            "every_supported_wide_width_requires_its_exact_scheduler_budget",
            "malformed_wide_values_do_not_select_the_disabled_path",
            "scoped_non_unicode_selectors_fail_but_disabled_and_unrelated_modes_stay_inert",
            "batch_override_preserves_unset_equal_and_larger_decimal_values",
            "batch_override_rejects_non_decimal_spellings",
            "batch_override_rejects_values_below_the_resolved_minimum",
            "batch_override_checks_usize_overflow_without_an_arbitrary_upper_cap",
            "batch_override_non_unicode_is_not_treated_as_absent",
            "prefix_checkpoint_clamp_remains_part_of_the_override_minimum",
            "speculative_rows_remain_part_of_the_override_minimum",
        ):
            self.assertIn("fn " + name + "()", self.kv)


if __name__ == "__main__":
    unittest.main()
