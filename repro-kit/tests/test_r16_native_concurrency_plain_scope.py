# SPDX-License-Identifier: AGPL-3.0-only
"""Pure tests for the native-concurrency driver's opt-in plain-only scope.

The driver imports only the standard library, so its scope helpers and its
validator builder run here without Docker, servers or pinned modules. The
full-scope receipt must stay key-for-key identical, and a plain receipt must
name a gate the release wrapper never accepts as native-concurrency-c2-c8.
"""

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "r15-gates" / "native-concurrency" / "native-concurrency-c2-c8"


def module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    loader.exec_module(value)
    return value


D = module(DRIVER_PATH, "r16_native_concurrency_driver_plain_scope")
SOURCE = DRIVER_PATH.read_text()


def required_gates(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["REQUIRED_GATES"]:
            return ast.literal_eval(node.value)
    raise AssertionError(path)


def observations(modes):
    return [
        {"width": width, "mode": mode, "full_frames": 8, "memory_samples_mib": [7000, 6900, 7100]}
        for mode in modes for _profile in range(3) for width in D.LEVELS
    ]


def build(scope, modes):
    suite = types.SimpleNamespace(
        canonical_bytes=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )
    written = {}

    def write_artifact(_suite, _runtime, locator, payload, mode=0o400):
        written[locator] = json.loads(payload)
        return hashlib.sha256(payload).hexdigest()

    gate, _required = D.scope_contract(scope, modes)
    args = types.SimpleNamespace(scope=scope, gate=gate, modes=modes, suite=suite, max_tokens=64,
                                 cancel_token=20, container_id="c" * 64, run_id="run-1")
    context = types.SimpleNamespace(proofs=[], timing_rows=[], diagnostics=[], model="m",
                                    provenance=types.SimpleNamespace(public={}), base="concurrency/x",
                                    runtime=None)
    with mock.patch.object(D, "write_artifact", write_artifact):
        evidence = D.build_validator(args, context, observations(modes), {"matrix": 1}, [])
    return evidence, written["concurrency/x/validator.json"]


class PlainScopeTests(unittest.TestCase):
    def test_full_scope_is_the_default_and_unchanged(self):
        self.assertIn('parser.add_argument("--scope", default="full")', SOURCE)
        self.assertEqual(D.GATE, "native-concurrency-c2-c8")
        self.assertEqual(D.scope_contract("full", D.MODES), (D.GATE, D.MODES))
        self.assertEqual(D.scope_coverage("full"), {})
        evidence, validator = build("full", D.MODES)
        self.assertEqual(evidence["modes"], ["none", "native-mtp-k5"])
        self.assertEqual(evidence["fresh_processes"], 18)
        self.assertEqual(set(validator), {"schema", "gate", "status", "release_qualified",
                                          "container_id", "run_id", "raw"})
        self.assertEqual(validator["gate"], D.GATE)

    def test_full_scope_keeps_the_historical_completeness_message(self):
        D.require_complete("full", D.MODES, observations(D.MODES))
        with self.assertRaisesRegex(ValueError, "both modes and all 18 arms are required"):
            D.require_complete("full", ("none",), observations(("none",)))

    def test_plain_scope_is_labelled_and_excludes_k5(self):
        gate, required = D.scope_contract("plain", ("none",))
        self.assertEqual((gate, required), ("native-concurrency-c2-c8-plain", ("none",)))
        D.require_complete("plain", ("none",), observations(("none",)))
        evidence, validator = build("plain", ("none",))
        self.assertEqual(evidence["modes"], ["none"])
        self.assertEqual(evidence["fresh_processes"], 9)
        self.assertEqual(validator["gate"], "native-concurrency-c2-c8-plain")
        self.assertEqual(validator["native_k5"], "NOT_COVERED")
        self.assertEqual(validator["not_covered_modes"], ["native-mtp-k5"])
        self.assertEqual(validator["covered_modes"], ["none"])

    def test_plain_scope_refuses_other_modes_and_short_runs(self):
        for modes in (D.MODES, ("native-mtp-k5",), ("native-mtp-k5", "none")):
            with self.assertRaisesRegex(ValueError, "--scope plain requires exactly --modes none"):
                D.scope_contract("plain", modes)
        with self.assertRaisesRegex(ValueError, "must be full or plain"):
            D.scope_contract("k5", D.MODES)
        with self.assertRaisesRegex(ValueError, "plain scope requires mode none and all 9 arms"):
            D.require_complete("plain", ("none",), observations(("none",))[:8])

    def test_release_wrapper_never_accepts_the_plain_gate(self):
        for path in (ROOT / "r15_exact_image_release.py", ROOT / "r15-container-suite.py"):
            gates = required_gates(path)
            self.assertIn(D.GATE, gates)
            self.assertNotIn(D.SCOPES["plain"][0], gates)

    def test_receipt_and_failure_use_the_scoped_gate(self):
        self.assertIn('"gate": args.gate,', SOURCE)
        self.assertIn('"gate": getattr(args, "gate", GATE),', SOURCE)
        self.assertEqual(SOURCE.count("**scope_coverage(args.scope),"), 2)
        parse = SOURCE.index("args.gate, _required = scope_contract(args.scope, args.modes)")
        self.assertLess(parse, SOURCE.index("verify_parent_suite(args.suite_sha256)"))
        self.assertIn("require_complete(args.scope, args.modes, observations)", SOURCE)


if __name__ == "__main__":
    unittest.main()
