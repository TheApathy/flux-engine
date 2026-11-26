"""Bounded source/mutation contracts, not Rust execution or runtime proof."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "crates/spark-runtime/src"
PHASES = ROOT / "crates/spark-server/src/main_modules/serve_phases"


def compact(source):
    return re.sub(r",(?=[)\]])", "", re.sub(r"\s+", "", source))


def code_only(source):
    pattern = r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"'
    return re.sub(pattern, lambda match: re.sub(r"[^\n]", " ", match[0]), source, flags=re.S)


def function(source, name):
    code = code_only(source)
    match = re.search(r"\bfn\s+" + re.escape(name) + r"\s*\(", code)
    if match is None:
        raise ValueError("missing function " + name)
    start = code.index("{", match.end())
    depth = 1
    for end in range(start + 1, len(code)):
        depth += (code[end] == "{") - (code[end] == "}")
        if depth == 0:
            return source[match.start():end + 1]
    raise ValueError("unclosed function " + name)


def require(source, fragments, label):
    source = compact(source)
    for fragment in fragments:
        if compact(fragment) not in source:
            raise ValueError("missing " + label + ": " + fragment)


def validate_math(source):
    for name, identity, operation in (("product", 1, "mul"), ("sum", 0, "add")):
        body = function(source, name)
        require(body, (
            f"values.iter().try_fold({identity}usize,",
            f".checked_{operation}(value).with_context(",
        ), "checked fold")
        if re.search(r"\.(?:unwrap|expect|wrapping_\w+|saturating_\w+)\(", body):
            raise ValueError("checked fold must return an error, not panic or wrap")
    align = function(source, "align_up")
    require(align, (
        "anyhow::ensure!(alignment.is_power_of_two(),",
        "Ok(sum(&[value, alignment - 1], label)? & !(alignment - 1))",
    ), "checked alignment")
    if align.index("ensure!") > align.index("alignment - 1"):
        raise ValueError("alignment must be validated before subtraction")
    widths = function(source, "checked")
    require(widths, (
        'product(&[config.linear_num_key_heads, config.linear_key_head_dim], "linear key width")?',
        'product(&[config.linear_num_value_heads, config.linear_value_head_dim], "linear value width")?',
        'product(&[config.mamba_num_heads, config.mamba_head_dim], "Mamba inner width")?',
        'product(&[2, config.n_groups, config.ssm_state_size], "Mamba BC width")?',
        'residual: product(&[config.hidden_size, config.hc_count.max(1)], "residual width")?',
        'ssm_qkvz: product(&[sum(&[key, value], "SSM KV width")?, 2], "SSM QKVZ width")?',
        'ssm_ba: product(&[config.linear_num_value_heads, 2], "SSM BA width")?',
        'sum(&[mamba_inner, mamba_xbc, config.mamba_num_heads], "Mamba input projection width")?',
    ), "local checked width")
    if re.search(r"[+*]|\.(?:saturating|wrapping)_", code_only(widths)):
        raise ValueError("local derived widths must not use unchecked arithmetic")


def validate_sizes(source):
    require(source, (
        "product(&[$($value),+], stringify!($($value),+))?",
        "sum(&[$($value),+], stringify!($($value),+))?",
    ), "fallible sizing macro")
    body = function(source, "from_config")
    require(body, (
        ") -> Result<Self>",
        "anyhow::ensure!(kv_block_size > 0,",
        "let widths = ArenaWidths::checked(config)?;",
        "let max_blocks = add!(max_seq_len / kv_block_size, 1);",
        "decode_meta.checked_meta_bytes(max_blocks)?",
        'align_up(pos_bytes, 8, "prefill slots alignment")?',
        'align_up(slot_end, 4, "prefill block table alignment")?',
        'align_up(bt_end, 4, "prefill sequence lengths alignment")?',
        "sizes.total_bytes()?; Ok(sizes)",
    ), "checked arena admission")
    if body.index("kv_block_size > 0") > body.index("max_seq_len / kv_block_size"):
        raise ValueError("block size must be validated before division")
    for getter in (
        "residual_width", "ssm_qkvz_size", "ssm_ba_size",
        "mamba2_d_inner", "mamba2_d_xbc", "mamba2_in_proj_size",
    ):
        if "config." + getter + "(" in compact(body):
            raise ValueError("unchecked model getter in sizing")
    production = code_only(body).replace("64 * 1024", "65536")
    if re.search(r"[+*]|\.(?:expect|unwrap)\(|\.(?:saturating|wrapping)_", production):
        raise ValueError("unchecked arithmetic or panic in arena sizing")
    total = function(source, "total_bytes")
    require(total, ('-> Result<usize>', 'sum(&[', '], "total bytes")'), "checked total")
    declared = re.findall(r"pub (\w+): usize", source.split("impl BufferSizes", 1)[0])
    summed = re.findall(r"self\.(\w+)", total)
    # 22 since fix/test-triage added moe_gate_scalars (gatepre, preallocated).
    if len(declared) != 22 or sorted(declared) != sorted(summed):
        raise ValueError("total must sum all 22 buffers exactly once")
    for helper in (
        "splitk_workspace_bytes_for_sm", "nvfp4_worklist_capacity_items",
        "qwen38_flashinfer_merged_output_bytes",
    ):
        helper_body = function(source, helper)
        require(helper_body, ("-> Result<usize>", "mul!("), "checked sizing helper")
        if re.search(r"[+*]|\.(?:expect|unwrap)\(", code_only(helper_body)):
            raise ValueError("unchecked helper arithmetic")
    require(
        function(source, "splitk_workspace_bytes"),
        ("-> Result<usize>", "splitk_workspace_bytes_for_sm("),
        "checked splitk wrapper",
    )
    worklist = function(source, "nvfp4_worklist_capacity_items")
    require(worklist, ("anyhow::ensure!(m_tile > 0,",), "worklist divisor admission")
    if worklist.index("m_tile > 0") > worklist.index("div_ceil(m_tile)"):
        raise ValueError("tile must be validated before division")


def validate_callers(arena, build, preflight):
    constructor = function(arena, "new")
    require(constructor, (
        "max_batch_size, )?;", "let total_bytes = sizes.total_bytes()?;",
    ), "fallible allocation admission")
    if constructor.index("sizes.total_bytes()?") > constructor.index("gpu.alloc("):
        raise ValueError("checked total must precede the first allocation")
    for caller in (build, preflight):
        require(caller, ("max_batch_size, )? .total_bytes()?;",), "fallible live estimate")
        if caller.count("BufferSizes::from_config(") != 1:
            raise ValueError("unexpected arena call-site count")


def validate_metadata(source):
    body = function(source, "checked_meta_bytes")
    require(body, (
        "-> anyhow::Result<usize>",
        'product(&[24, self.rows], "decode metadata prefix")?',
        'product(&[self.rows, max_blocks, 4], "decode metadata block table")?',
        'sum(&[prefix, table], "decode metadata total")',
    ), "checked decode metadata")
    if re.search(r"[+*]|\.block_table_off\(", code_only(body)):
        raise ValueError("checked metadata must not call legacy unchecked offsets")


def validate_gdn(source):
    body = function(source, "checked_gdn_two_phase_bytes")
    require(body, (
        "if config.num_ssm_layers() == 0 { return Ok(0); }",
        'checked_preflight_mul(config.linear_num_key_heads, config.linear_key_head_dim, "GDN key width")?',
        'checked_preflight_mul(config.linear_num_value_heads, config.linear_value_head_dim, "GDN value width")?',
        'checked_preflight_mul(key_dim, 2, "GDN QK width")?',
        'value_dim, "GDN conv width")?',
        "if conv_dim == 0 { return Ok(0); }",
        'checked_preflight_mul(conv_dim, 2, "GDN conv row bytes")?',
        'checked_preflight_mul(config.linear_num_value_heads, 8, "GDN gates row bytes")?',
        'checked_preflight_mul(value_dim, 2, "GDN value row bytes")?',
        'checked_preflight_mul(value, 2, "GDN two value buffers")?',
        'checked_preflight_mul(rows, row_bytes, "GDN two-phase total bytes")',
    ), "pure checked GDN estimate")
    if re.search(r"[+*]|std::env|gpu\.", code_only(body)):
        raise ValueError("GDN estimate must remain pure and checked")
    live = function(source, "preflight_reserve")
    require(live, (
        "let gdn_two_phase_bytes: usize = checked_gdn_two_phase_bytes(config, max_batch_tokens)?;",
        "let per_tok_bytes = checked_gdn_two_phase_bytes(config, 1)?;",
    ), "checked live GDN estimate")
    if "let key_dim =" in live or "let conv_dim =" in live:
        raise ValueError("legacy inline GDN estimate remains")


class CheckedArenaSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.math = (RUNTIME / "buffers/checked_math.rs").read_text()
        cls.sizes = (RUNTIME / "buffers/sizes.rs").read_text()
        cls.arena = (RUNTIME / "buffers.rs").read_text()
        cls.metadata = (RUNTIME / "buffers/decode_meta.rs").read_text()
        cls.build = (ROOT / "crates/spark-model/src/factory/build.rs").read_text()
        cls.preflight = (PHASES / "preflight.rs").read_text()

    def assert_mutants_rejected(self, source, validator, mutations):
        for old, new in mutations:
            with self.subTest(mutant=old):
                self.assertIn(old, source)
                with self.assertRaises(ValueError):
                    validator(source.replace(old, new, 1))

    def test_math_and_local_derived_widths_are_fallible(self):
        validate_math(self.math)

    def test_wrapping_missing_alignment_guard_and_width_mutants_are_rejected(self):
        self.assert_mutants_rejected(self.math, validate_math, (
            (".checked_mul(value)", ".wrapping_mul(value)"),
            (".checked_add(value)", ".saturating_add(value)"),
            ("alignment.is_power_of_two()", "alignment > 0"),
            ('sum(&[value, alignment - 1], label)?', 'sum(&[value, alignment - 1], label).unwrap()'),
            ('config.hidden_size, config.hc_count.max(1)', 'config.hidden_size, config.hc_count'),
            ('"SSM KV width")?', '"SSM KV width").unwrap()'),
        ))

    def test_arena_sizing_and_all_22_total_fields_are_checked(self):
        validate_sizes(self.sizes)

    def test_macro_alignment_metadata_and_total_mutants_are_rejected(self):
        self.assert_mutants_rejected(self.sizes, validate_sizes, (
            ('stringify!($($value),+))?', 'stringify!($($value),+)).unwrap()'),
            ('kv_block_size > 0', 'kv_block_size >= 0'),
            ('let max_blocks = add!(max_seq_len / kv_block_size, 1);', 'let max_blocks = max_seq_len / kv_block_size + 1;'),
            ('decode_meta.checked_meta_bytes(max_blocks)?', 'decode_meta.meta_bytes(max_blocks)'),
            ('"prefill slots alignment")?', '"prefill slots alignment").unwrap()'),
            ('sizes.total_bytes()?;', 'sizes.total_bytes();'),
            ('self.splitk_workspace,', ''),
            ('self.moe_gate_scalars,', ''),
            ('m_tile > 0', 'm_tile >= 0'),
        ))

    def test_all_live_callers_propagate_errors_before_allocation(self):
        validate_callers(self.arena, self.build, self.preflight)

    def test_missing_result_and_late_total_mutants_are_rejected(self):
        self.assert_mutants_rejected(
            self.arena,
            lambda value: validate_callers(value, self.build, self.preflight),
            (
                ('max_batch_size,\n        )?;', 'max_batch_size,\n        );'),
                ('let total_bytes = sizes.total_bytes()?;', 'let total_bytes = sizes.total_bytes();'),
                ('let total_bytes = sizes.total_bytes()?;\n\n        let hidden_states = gpu.alloc(sizes.hidden_states)?;', 'let hidden_states = gpu.alloc(sizes.hidden_states)?;\n        let total_bytes = sizes.total_bytes()?;'),
            ),
        )
        for caller in ("build", "preflight"):
            source = getattr(self, caller)
            mutant = source.replace(")?\n    .total_bytes()?;", ")\n    .total_bytes()?;", 1)
            self.assertNotEqual(mutant, source)
            values = dict(arena=self.arena, build=self.build, preflight=self.preflight)
            values[caller] = mutant
            with self.subTest(caller=caller), self.assertRaises(ValueError):
                validate_callers(**values)

    def test_checked_metadata_preserves_the_legacy_helper(self):
        validate_metadata(self.metadata)
        self.assertIn("self.block_table_off() + self.rows * max_blocks * 4", function(self.metadata, "meta_bytes"))

    def test_unchecked_metadata_prefix_table_and_sum_mutants_are_rejected(self):
        self.assert_mutants_rejected(self.metadata, validate_metadata, (
            ('product(&[24, self.rows], "decode metadata prefix")?', 'product(&[24, self.rows], "decode metadata prefix").unwrap()'),
            ('&[self.rows, max_blocks, 4]', '&[self.rows, max_blocks]'),
            ('sum(&[prefix, table], "decode metadata total")', 'Ok(prefix + table)'),
        ))

    def test_live_gdn_preflight_is_pure_and_checked(self):
        validate_gdn(self.preflight)

    def test_missing_gdn_guard_factors_and_live_propagation_mutants_are_rejected(self):
        self.assert_mutants_rejected(self.preflight, validate_gdn, (
            ('if config.num_ssm_layers() == 0 {', 'if false {'),
            ('config.linear_value_head_dim,\n        "GDN value width",', '1,\n        "GDN value width",'),
            ('value, 2, "GDN two value buffers"', 'value, 1, "GDN two value buffers"'),
            ('rows, row_bytes, "GDN two-phase total bytes"', '1, row_bytes, "GDN two-phase total bytes"'),
            ('checked_gdn_two_phase_bytes(config, max_batch_tokens)?;', 'checked_gdn_two_phase_bytes(config, max_batch_tokens).unwrap();'),
        ))

    def test_adversarial_native_rust_regressions_are_defined_not_executed(self):
        tests = (RUNTIME / "buffers/tests.rs").read_text()
        for name in (
            "arena_rejects_oversized_rows_before_any_mock_allocation",
            "arena_rejects_block_table_and_decode_metadata_overflow",
            "zero_kv_block_size_is_an_error_before_any_mock_allocation",
            "arena_rejects_malformed_derived_widths_before_any_mock_allocation",
            "arena_rejects_aggregate_overflow_even_when_each_field_fits",
            "arena_total_bytes_rejects_overflow_after_public_field_changes",
            "checked_arena_metadata_preserves_c128_padding_and_mrope_capacity",
        ):
            self.assertIn("fn " + name + "()", tests)
        self.assertIn("assert_eq!(gpu.alloc_count(), 0);", tests)
        self.assertIn("usize::MAX / 16", tests)
        self.assertIn("fn splitk_and_worklist_helpers_fail_closed_on_overflow()", self.sizes)
        self.assertIn("fn derived_widths_reject_sum_gate_and_mamba_projection_overflow()", self.math)
        gdn_tests = (PHASES / "preflight_arena_tests.rs").read_text()
        for name in (
            "checked_gdn_estimate_preserves_ordinary_geometry",
            "checked_gdn_estimate_rejects_row_and_dimension_overflow",
            "inactive_gdn_estimate_stays_zero",
        ):
            self.assertIn("fn " + name + "()", gdn_tests)


if __name__ == "__main__":
    unittest.main()
