"""Qwen4 shortened-frame source wiring, not Rust execution or accuracy proof."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPL = ROOT / "crates/spark-model/src/model/trait_impl"


def compact(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", text))


def validate_route(source, width):
    source = compact(source)
    entry = f") -> Result<[u32; {width}]> {{"
    body = source.split(compact(entry), 1)[1]
    route = (
        "ifself.config.is_qwen4_exp(){"
        "letverified=self.decode_verify_graphed_kgamma_dispatch(tokens,seq,_stream)?;"
        "returnverified.try_into().map_err(|rows:Vec<u32>|{"
    )
    if route not in body:
        raise ValueError("missing model-qualified, propagating Qwen4 dispatch")
    # Comments may precede the branch; no code may mutate a frame before it.
    prefix = body.split("ifself.config.is_qwen4_exp(){", 1)[0]
    if any(word in prefix for word in ("let", "self.", "seq.", "return", "unsafe")):
        raise ValueError("Qwen4 dispatch must precede generic work")
    error = f'"Qwen4K{width}verifierreturned{{}}rows,expected{width}",rows.len()'
    if error not in body.split("letpositions=", 1)[0]:
        raise ValueError("wrong-width results must fail closed")


class Qwen4ShortVerifySourceTests(unittest.TestCase):
    def source(self, width):
        name = "verify_c.rs" if width == 3 else "verify_c2.rs"
        source = (IMPL / name).read_text()
        return re.sub(r"//[^\n]*", "", source)

    def test_k3_routes_before_generic_work(self):
        validate_route(self.source(3), 3)

    def test_k4_routes_before_generic_work(self):
        validate_route(self.source(4), 4)

    def test_removed_or_wrong_target_branch_is_rejected(self):
        for width in (3, 4):
            for target in ("false", "!self.config.is_qwen4_exp()"):
                with self.subTest(width=width, target=target), self.assertRaises(ValueError):
                    validate_route(self.source(width).replace("self.config.is_qwen4_exp()", target), width)

    def test_errors_cannot_fall_back_into_generic_verifier(self):
        for width in (3, 4):
            source = self.source(width)
            for old, new in (
                ("_dispatch(tokens, seq, _stream)?", "_dispatch(tokens, seq, _stream).unwrap_or_default()"),
                ("return verified.try_into()", "verified.try_into()"),
            ):
                with self.subTest(width=width, old=old), self.assertRaises(ValueError):
                    validate_route(source.replace(old, new), width)

    def test_dispatch_after_mutation_is_rejected(self):
        for width in (3, 4):
            source = self.source(width).replace(
                "if self.config.is_qwen4_exp() {",
                "self.pre_verify_copy_async(seq)?; if self.config.is_qwen4_exp() {",
            )
            with self.subTest(width=width), self.assertRaises(ValueError):
                validate_route(source, width)

    def test_wrong_width_results_are_rejected(self):
        for width in (3, 4):
            source = self.source(width).replace("expected " + str(width), "expected 99")
            with self.subTest(width=width), self.assertRaises(ValueError):
                validate_route(source, width)

    def test_generic_verifier_owns_qwen4_geometry_and_state(self):
        source = (IMPL / "verify_d.rs").read_text()
        self.assertIn("self.config.residual_width()", source)
        self.assertIn("self.qwen4_verify_ple(", source)
        # Verify rows route through the NVFP4-forcing final-hidden helper.
        self.assertIn("self.qwen4_final_hidden_nvfp4(", source)
        self.assertIn("self.qwen4_verify_layer(", source)
        self.assertNotIn("decode_verify_graphed_k3_dispatch(", source)
        self.assertNotIn("decode_verify_graphed_k4_dispatch(", source)


if __name__ == "__main__":
    unittest.main()
