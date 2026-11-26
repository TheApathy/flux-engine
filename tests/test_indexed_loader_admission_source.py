# SPDX-License-Identifier: AGPL-3.0-only
"""Lightweight source/arithmetic checks, not execution of the Rust regressions."""

from fractions import Fraction
import math
from pathlib import Path
import random
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
LOADER = (ROOT / "crates/spark-runtime/src/weights/loader.rs").read_text()
LOAD_FNS = (ROOT / "crates/spark-runtime/src/weights/loader/load_fns.rs").read_text()
NATIVE = (ROOT / "crates/spark-runtime/src/weights/loader/tests.rs").read_text()
MAX_BYTES = (1 << (struct.calcsize("P") * 8)) - 1


def combined_model(weights, multiplier, construction, reserve):
    """Model the checked binary-significand arithmetic for comparison to Fraction."""
    if not math.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("invalid multiplier")
    bits = struct.unpack(">Q", struct.pack(">d", multiplier))[0]
    significand = (bits & ((1 << 52) - 1)) | (1 << 52)
    exponent = ((bits >> 52) & 0x7FF) - 1075
    numerator = weights * significand
    if not numerator:
        scaled = 0
    elif exponent < 0:
        denominator = 1 << -exponent
        scaled = (numerator + denominator - 1) // denominator
    else:
        if exponent >= 128 or numerator * (1 << exponent) > (1 << 128) - 1:
            raise OverflowError("scaled weight peak")
        scaled = numerator * (1 << exponent)
    peak = scaled + construction
    required = peak + reserve
    if scaled > MAX_BYTES or peak > MAX_BYTES or required > MAX_BYTES:
        raise OverflowError("combined byte count")
    return peak, required


class IndexedAdmissionSourceTests(unittest.TestCase):
    def test_index_resolves_before_shared_preflight(self):
        self.assertLess(LOADER.index(".map(read_index_shards)"), LOADER.index("estimate_load_bytes("))
        self.assertNotIn("shard_files = vec![]", LOADER)
        self.assertIn("shards.keys().map(|name| model_dir.join(name))", LOADER)

    def test_same_resolved_plan_is_loaded(self):
        self.assertEqual(LOADER.count("if let Some(shards) = &indexed_shards"), 2)
        self.assertIn("shards,", LOADER[LOADER.index("load_sharded("):])
        child = LOAD_FNS.split("pub(super) fn load_sharded", 1)[1]
        self.assertNotIn("read_to_string", child)
        self.assertNotIn("estimate_load_bytes", child)
        self.assertNotIn("peak_multiplier", child)

    def test_both_index_names_use_the_same_resolution_path(self):
        prefix = LOADER.split("// Resolve once", 1)[0]
        self.assertIn("Some(index_path)", prefix)
        self.assertIn("Some(consolidated_index)", prefix)

    def test_combined_accounting_is_before_any_load(self):
        check = LOADER.index("combined_admission_bytes(")
        first_load = min(LOADER.index("load_sharded("), LOADER.index("load_single("))
        self.assertLess(check, first_load)
        self.assertIn("let construction = self.construction_overhead_bytes", LOADER)
        self.assertIn("if required > free", LOADER)
        self.assertNotIn("peak_estimated + oom_reserve_bytes > free", LOADER)

    def test_reserve_and_default_multipliers_are_preserved(self):
        self.assertIn("unwrap_or(if has_fp8 { 1.5 } else { 1.3 })", LOADER)
        self.assertIn(".checked_add(reserve_bytes)", LOADER)
        self.assertIn("if !offload_logged", LOAD_FNS)
        self.assertIn("check_oom_guard(", LOAD_FNS)
        self.assertIn("oom_reserve_bytes,", LOAD_FNS)

    def test_float_to_integer_saturation_and_shape_overflow_cannot_admit(self):
        self.assertIn("multiplier.to_bits()", LOADER)
        self.assertIn("numerator.div_ceil(", LOADER)
        self.assertIn("usize::try_from(scaled)", LOADER)
        self.assertIn(".checked_add(construction_bytes)", LOADER)
        self.assertIn(".checked_mul(dim)", LOADER)
        self.assertIn(".checked_mul(elem_bytes)", LOADER)
        self.assertNotIn("shape.iter().product()", LOADER)
        self.assertEqual(LOADER.count("estimated_tensor_bytes(&shape, dtype)?"), 2)

    def test_native_regression_definitions_are_retained(self):
        names = [
            "indexed_combined_peak_rejects_the_old_split_admission_gap",
            "indexed_exact_fit_counts_shared_shards_and_construction_once",
            "indexed_combined_admission_preserves_the_full_reserve",
            "indexed_explicit_multiplier_is_not_reduced_or_applied_to_construction",
            "indexed_fp8_default_multiplier_uses_actual_shard_headers",
            "indexed_filter_is_shared_by_admission_and_upload",
            "indexed_multi_shard_peak_is_combined_before_any_allocation",
            "missing_indexed_shard_fails_before_any_allocation",
            "single_and_unindexed_shards_keep_combined_admission",
            "combined_accounting_rejects_invalid_multipliers_and_overflow",
            "combined_accounting_does_not_round_large_integer_byte_counts_down",
            "header_estimation_rejects_shape_and_byte_overflow",
        ]
        for name in names:
            self.assertIn("fn " + name + "()", NATIVE)
        self.assertGreaterEqual(NATIVE.count("alloc_count(), 0"), 8)

    def test_original_gap_and_exact_fit_arithmetic(self):
        self.assertEqual(combined_model(88, 1.0, 13, 0), (101, 101))
        self.assertEqual(combined_model(88, 1.0, 12, 0), (100, 100))
        self.assertEqual(combined_model(64, 1.25, 20, 0), (100, 100))
        self.assertEqual(combined_model(3, 1.5, 7, 11), (12, 23))
        self.assertEqual(combined_model(MAX_BYTES, 1.0, 0, 0), (MAX_BYTES, MAX_BYTES))
        if MAX_BYTES > 1 << 53:
            value = (1 << 53) + 1
            self.assertEqual(combined_model(value, 1.0, 0, 0), (value, value))

    def test_arithmetic_model_matches_exact_fraction_reference(self):
        rng = random.Random(98237)
        for _ in range(2500):
            weights = rng.randrange(MAX_BYTES + 1)
            multiplier = rng.choice([1.0, 1.2, 1.25, 1.3, 1.5, 2.0, 5.0])
            construction = rng.randrange(1 << 20)
            reserve = rng.randrange(1 << 20)
            exact = math.ceil(weights * Fraction.from_float(multiplier))
            required = exact + construction + reserve
            if required > MAX_BYTES:
                with self.assertRaises(OverflowError):
                    combined_model(weights, multiplier, construction, reserve)
            else:
                self.assertEqual(
                    combined_model(weights, multiplier, construction, reserve),
                    (exact + construction, required),
                )

    def test_invalid_and_overflow_inputs_fail_closed_in_arithmetic_model(self):
        for multiplier in [math.nan, math.inf, -math.inf, -1.0, 0.0, 0.5]:
            with self.assertRaises(ValueError):
                combined_model(1, multiplier, 0, 0)
        for args in [
            (MAX_BYTES, 2.0, 0, 0),
            (1, 1.0, MAX_BYTES, 0),
            (1, 1.0, 0, MAX_BYTES),
            (1, float.fromhex("0x1.fffffffffffffp+1023"), 0, 0),
        ]:
            with self.assertRaises(OverflowError):
                combined_model(*args)


if __name__ == "__main__":
    unittest.main()
