# SPDX-License-Identifier: AGPL-3.0-only
"""K2 source/mutation checks only, not Rust execution or QSA parity."""

import hashlib
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
IMPL = ROOT / "crates/spark-model/src/model/trait_impl"


def compact(source):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", source))


def body(source, marker):
    source, marker = compact(source), compact(marker)
    start = source.index("{", source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError("unclosed source body")


DELEGATION = compact('''
    if self.config.is_qwen4_exp() {
        let verified = self.decode_verify_graphed_kgamma_dispatch(tokens, seq, _stream)?;
        return verified.try_into().map_err(|rows: Vec<u32>| {
            anyhow::anyhow!("Qwen4 K2 verifier returned {} rows, expected 2", rows.len())
        });
    }
''')
GUARD = compact('''
    qwen4_legacy_k2_admission(
        self.config.is_qwen4_exp(), k,
        std::env::var("ATLAS_QWEN4_VERIFY_LEGACY_K2").ok().as_deref() == Some("1"),
    ).map_err(anyhow::Error::msg)?;
''')


def validate_fixed(source):
    fixed = body(source, "fn decode_verify_graphed_dispatch(")
    if not fixed.startswith(DELEGATION) or fixed.count(DELEGATION) != 1:
        raise AssertionError("Qwen4 K2 must delegate and propagate exact-width errors before any work")
    suffix = fixed[len(DELEGATION):]
    if not suffix.startswith("letpositions=seq.rotary_positions.verify_tail(seq.seq_len,&[0,1])?;"):
        raise AssertionError("non-Qwen4 entry changed")


def validate_generic(source):
    helper = body(source, "fn qwen4_legacy_k2_admission(")
    expected = compact('''
        if target_is_qwen4 && rows == 2 && legacy_selected {
            return Err("ATLAS_QWEN4_VERIFY_LEGACY_K2 cannot re-enter the Qwen4 fixed-K2 wrapper");
        }
        Ok(())
    ''')
    if helper != expected:
        raise AssertionError("legacy rejection must be pure and limited to Qwen4 K2")
    entry = body(source, "fn decode_verify_graphed_kgamma_dispatch(")
    if entry.count(GUARD) != 1:
        raise AssertionError("generic verifier must fail closed on the actual legacy selector")
    if "decode_verify_graphed_dispatch(" in entry:
        raise AssertionError("recursive fixed-K2 back-edge remains")
    for effect in ("self.ssm_h_to_f16_dispatch(seq)?;", "self.pre_verify_copy_async(seq)?;",
                   "self.embed(", "ensure_blocks_through_decode("):
        if entry.index(GUARD) >= entry.index(effect):
            raise AssertionError("legacy guard follows a verifier effect")


class Qwen4K2DelegationSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed = (IMPL / "verify_b.rs").read_text(encoding="utf-8")
        cls.generic = (IMPL / "verify_d.rs").read_text(encoding="utf-8")

    def test_fixed_k2_delegates_before_effects_with_exact_array_conversion(self):
        validate_fixed(self.fixed)

    def test_generic_cannot_reenter_fixed_wrapper_even_with_legacy_optin(self):
        validate_generic(self.generic)

    def test_nonqwen_fixed_verifier_prefix_and_suffix_match_pinned_digests(self):
        anchor = "        let positions = seq.rotary_positions.verify_tail(seq.seq_len, &[0, 1])?;"
        suffix = self.fixed[self.fixed.index(anchor):]
        # begin/end_capture is wrapped in CaptureGuard (same success-path order, Drop closes on error)
        # and the qwen4-only final-hidden call uses qwen4_final_hidden_nvfp4; non-qwen dispatch is pinned.
        self.assertEqual(hashlib.sha256(suffix.encode()).hexdigest(),
                         "06193b2dceaa7191a746c0905a226717e8fadeb908a41448eab6c20d78e0862d")
        prefix = self.fixed.split("    pub(super) fn decode_verify_graphed_dispatch(", 1)[0]
        self.assertEqual(hashlib.sha256(prefix.encode()).hexdigest(),
                         "c0cda51b573f789f906484d54cd76612341abc0fef7646120081832783cfd463")

    def test_model_trait_fixed_k2_reaches_the_guarded_wrapper(self):
        source = (IMPL / "mod.rs").read_text(encoding="utf-8")
        wrapper = body(source, "fn decode_verify_graphed(")
        # Owned native K5 batching checks a latched verify failure and opens
        # the verify frame before the same dispatch.
        self.assertEqual(
            wrapper,
            "crate::model::verify_failure::check()?;self.ep_head_sync_for_verify(seq)?;"
            "self.begin_verify_frame(seq,tokens.len())?;"
            "self.decode_verify_graphed_dispatch(tokens,seq,_stream)",
        )
        for required in ("self.config.residual_width()", "self.qwen4_verify_ple(",
                         # Verify rows route through the NVFP4-forcing final-hidden helper.
                         "self.qwen4_final_hidden_nvfp4(", "self.qwen4_verify_layer("):
            self.assertIn(required, self.generic)

    def test_removed_wrong_target_swallowed_error_and_wrong_width_rejected(self):
        source = compact(self.fixed)
        for before, after in (
            (DELEGATION, ""),
            (DELEGATION, DELEGATION.replace("self.config.is_qwen4_exp()", "true")),
            (DELEGATION, DELEGATION.replace("self.config.is_qwen4_exp()", "false")),
            (DELEGATION, DELEGATION.replace("_stream)?;", "_stream).unwrap_or_default();")),
            (DELEGATION, DELEGATION.replace("returnverified.try_into()", "verified.try_into()")),
            (DELEGATION, DELEGATION.replace("expected2", "expected3")),
            (DELEGATION, DELEGATION.replace("tokens,seq,_stream", "tokens,seq,0")),
        ):
            with self.subTest(change=after[:60]), self.assertRaises(AssertionError):
                validate_fixed(source.replace(before, after, 1))

    def test_delegation_after_snapshot_or_embedding_rejected(self):
        source = compact(self.fixed)
        for mutation in ("self.pre_verify_copy_async(seq)?;", "self.embed(tokens[0],hidden,stream)?;"):
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                validate_fixed(source.replace(DELEGATION, mutation + DELEGATION, 1))

    def test_legacy_selector_target_width_and_error_propagation_mutations_rejected(self):
        source = compact(self.generic)
        for before, after in (
            (GUARD, ""),
            (GUARD, GUARD.replace("self.config.is_qwen4_exp()", "false")),
            (GUARD, GUARD.replace(",k,", ",3,")),
            (GUARD, GUARD.replace('Some("1")', 'Some("0")')),
            (GUARD, GUARD.replace(".map_err(anyhow::Error::msg)?;", ".ok();")),
            ("target_is_qwen4&&rows==2&&legacy_selected", "target_is_qwen4&&rows==3&&legacy_selected"),
            ("target_is_qwen4&&rows==2&&legacy_selected", "rows==2&&legacy_selected"),
            ("target_is_qwen4&&rows==2&&legacy_selected", "false"),
        ):
            with self.subTest(mutation=before), self.assertRaises(AssertionError):
                validate_generic(source.replace(before, after, 1))

    def test_recursive_backedge_and_post_effect_guard_rejected(self):
        source = compact(self.generic)
        with self.assertRaises(AssertionError):
            validate_generic(source.replace(GUARD, GUARD + "self.decode_verify_graphed_dispatch(&pair,seq,_stream)?;", 1))
        changed = source.replace(GUARD, "", 1).replace("self.pre_verify_copy_async(seq)?;",
                                                     "self.pre_verify_copy_async(seq)?;" + GUARD, 1)
        with self.assertRaises(AssertionError):
            validate_generic(changed)

    def test_pure_native_recursion_and_nonqwen_matrix_definitions_exist(self):
        tests = body(self.generic, "mod qwen4_verify_route_tests")
        for name in ("qwen4_fixed_k2_refuses_the_recursive_legacy_selector",
                     "fixed_k2_legacy_selector_leaves_other_targets_and_widths_unchanged"):
            native = body(tests, "fn " + name + "(")
            self.assertIn("qwen4_legacy_k2_admission(", native)
            self.assertNotIn("set_var(", native)
            self.assertNotIn("remove_var(", native)


if __name__ == "__main__":
    unittest.main()
