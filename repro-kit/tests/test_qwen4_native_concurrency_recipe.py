# SPDX-License-Identifier: AGPL-3.0-only
"""Pure tests for the native-concurrency qualification recipe.

The recipe must differ only in the wide prefill geometry from its pinned
historical `flux-v9-max-prefill` snapshot fixture (not the live recipe), must
start under the gate's pinned launch flags, and must not assign any key the gate
owns. The Rust startup rule and fast-prefill window are mirrored here and bound
to their source text, so a source change that would invalidate the mirror fails
these tests.
"""

import ast
import hashlib
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT.parent
RECIPES = ROOT / "recipes"
RECIPE = RECIPES / "flux-qwen4-native-concurrency-c8.env"
# C8 is a frozen qualification artifact: it derives from the pinned historical
# max-prefill snapshot fixture rather than the live recipe.
FROZEN_BASE_COMMIT = "3c95f36c8"
SOURCE_RECIPE = ROOT / "tests" / "fixtures" / f"flux-v9-max-prefill.{FROZEN_BASE_COMMIT}.recipe"
SOURCE_RECIPE_SHA256 = "79ad682f9b5022561af58e0c8d666f4fc5784a4f42ff65b2952dfd2a219dd33f"
DRIVER = ROOT / "r15-gates" / "native-concurrency" / "native-concurrency-c2-c8"
PROBE = ROOT / "qwen4_native_concurrency_probe.py"
KV_CACHE_RS = TREE / "crates/spark-server/src/main_modules/serve_phases/kv_cache.rs"
MAIN_RS = TREE / "crates/spark-server/src/main.rs"
PREFILL_MOE_RS = TREE / "crates/spark-model/src/layers/qwen4_prefill_moe.rs"
WIDE, MULTI = "ATLAS_QWEN4_PREFILL_WIDE", "ATLAS_QWEN4_PREFILL_MULTI_CHUNK"
WIDE_VALUES = {"2048": 2048, "4096": 4096, "8192": 8192, "16384": 16384, "32768": 32768}
LEGACY_ROWS = 2048


def assignments(path):
    """Ordered KEY=VALUE pairs; every key assigned once (the sealed parser's rule)."""
    pairs, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # The launcher, suite and broker parsers all accept a leading `export`.
        match = re.fullmatch(r"(?:export )?([A-Z][A-Z0-9_]*)=(\S*)", line)
        assert match, f"{path.name}: malformed line {line!r}"
        assert match[1] not in seen, f"{path.name}: {match[1]} assigned twice"
        seen.add(match[1])
        pairs.append((match[1], match[2]))
    return pairs


def module_literal(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [name]:
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def effective_wide(env):
    """main.rs aliasing: an ATLAS_ value wins; FLUX_PREFILL_WIDE / FLUX_QWEN4_PREFILL_WIDE fill it."""
    if WIDE in env:
        return env[WIDE]
    return env.get("FLUX_PREFILL_WIDE", env.get("FLUX_QWEN4_PREFILL_WIDE"))


def startup_geometry_ok(env, prefill_budget):
    """Mirror of validate_qwen4_prefill_geometry_values for a qwen4 config."""
    wide = effective_wide(env)
    if wide in (None, "0"):
        return True
    if wide not in WIDE_VALUES:
        return False
    multi = env.get(MULTI, env.get("FLUX_QWEN4_PREFILL_MULTI_CHUNK"))
    return multi == "1" and WIDE_VALUES[wide] == prefill_budget


def chunk_window(rows, start, rows_cap):
    """Mirror of chunk_window_for_geometry's first-chunk arm."""
    return rows > 0 and start + rows <= rows_cap


def fixed_flags():
    return dict(module_literal(DRIVER, "FIXED_LAUNCH_FLAGS"))


class RecipeTests(unittest.TestCase):
    def test_frozen_base_is_the_pinned_max_prefill_snapshot(self):
        digest = hashlib.sha256(SOURCE_RECIPE.read_bytes()).hexdigest()
        self.assertEqual(digest, SOURCE_RECIPE_SHA256)
        self.assertIn(f"frozen at {FROZEN_BASE_COMMIT}", RECIPE.read_text())

    def test_only_the_wide_geometry_differs_from_max_prefill(self):
        source, recipe = assignments(SOURCE_RECIPE), assignments(RECIPE)
        expected = [(k, "0" if k == WIDE else v) for k, v in source if k != "FLUX_QWEN4_PREFILL_WIDE"]
        self.assertEqual(recipe, expected)
        self.assertEqual(dict(source)[WIDE], "16384")

    def test_gate_owned_keys_are_not_assigned(self):
        optimization = module_literal(PROBE, "OPTIMIZATION_SELECTORS")
        diagnostics = module_literal(DRIVER, "DIAGNOSTIC_ENVIRONMENT")
        limits = module_literal(DRIVER, "DIAGNOSTIC_LIMIT_KEYS")
        reserved = {"ATLAS_QWEN4_NATIVE_BATCH", *optimization, *diagnostics, *limits}
        self.assertEqual(len(reserved), 11)  # six selectors, three diagnostics, two limits
        self.assertFalse(reserved & set(dict(assignments(RECIPE))))

    def test_native_route_prerequisites_are_kept(self):
        env = dict(assignments(RECIPE))
        self.assertTrue(env["RUST_LOG"].startswith("info"))
        self.assertEqual(env["ATLAS_MOE_T_LANES"], "1")
        self.assertEqual(env["ATLAS_UNIFIED_MOE_LAYOUT"], "1")
        for refused in ("ATLAS_LM_HEAD_TC", "ATLAS_SSM_H_FP16", "ATLAS_QWEN4_LMHEAD_BF16",
                        "ATLAS_TARGET_LMHEAD_VOCAB", "ATLAS_DISABLE_SPECULATION"):
            self.assertNotIn(refused, env)

    def test_recipe_starts_under_the_gate_flags_and_serving_recipes_do_not(self):
        flags = fixed_flags()
        self.assertEqual((flags["--max-prefill-tokens"], flags["--max-seq-len"]), ("1024", "2048"))
        budget = int(flags["--max-prefill-tokens"])
        self.assertTrue(startup_geometry_ok(dict(assignments(RECIPE)), budget))
        for path in sorted(RECIPES.glob("*.env")):
            env = dict(assignments(path))
            if path != RECIPE and effective_wide(env) not in (None, "0"):
                self.assertFalse(startup_geometry_ok(env, budget), path.name)

    def test_native_prompts_stay_inside_the_fast_family_window(self):
        # Every prompt up to the budget is one chunk at offset 0 inside the legacy window.
        budget = int(fixed_flags()["--max-prefill-tokens"])
        self.assertLessEqual(budget, LEGACY_ROWS)
        for prompt in range(1, budget + 1):
            self.assertTrue(chunk_window(prompt, 0, LEGACY_ROWS))

    def test_mirrors_are_bound_to_the_rust_source(self):
        kv = KV_CACHE_RS.read_text()
        body = kv.split("fn validate_qwen4_prefill_geometry_values(", 1)[1].split("\n}\n", 1)[0]
        rust_wide_values = set(re.findall(r'Some\("(\d+)"\)\s*=>', body)) - {"0"}
        self.assertEqual(rust_wide_values, set(WIDE_VALUES))
        for line in ('None | Some("0") => return Ok(()),', 'multi_chunk == Some("1"),',
                     "wide == prefill_budget,"):
            self.assertIn(line, body)
        main = MAIN_RS.read_text()
        self.assertIn('"PREFILL_WIDE" => "ATLAS_QWEN4_PREFILL_WIDE".to_string(),', main)
        self.assertIn('_ => format!("ATLAS_{suffix}"),', main)
        moe = PREFILL_MOE_RS.read_text()
        self.assertIn("pub(crate) const MULTI_CHUNK_ROWS: usize = 2048;", moe)
        self.assertIn("rows: MULTI_CHUNK_ROWS,", moe)
        self.assertIn("rows > 0 && (end <= g.rows || (multi_chunk && multi_window(rows, start, end, g)))", moe)
        self.assertIn('None | Some("0") => Ok(None),', moe)


if __name__ == "__main__":
    unittest.main()
