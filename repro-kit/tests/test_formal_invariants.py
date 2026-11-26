# SPDX-License-Identifier: AGPL-3.0-only
"""Pytest gate executing the Flux formal verification suites (Z3, Kani MIR, TLA+, PTX/SASS ISA)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
FORMAL_SCRIPT = ROOT / "repro-kit" / "formal" / "verify_nextgen_invariants.py"
PTX_ISA_SCRIPT = ROOT / "repro-kit" / "formal" / "verify_ptx_isa_contracts.py"
TLA_MODEL_SCRIPT = ROOT / "repro-kit" / "formal" / "verify_scheduler_tla_model.py"
KANI_SCRIPT = ROOT / "repro-kit" / "formal" / "run_kani.sh"
KANI_HOME = Path(os.environ.get("KANI_HOME") or (Path.home() / ".kani" / "kani-0.68.0"))
NVCC_BIN = Path(shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FormalInvariantSuiteTests(unittest.TestCase):
    def test_all_z3_formal_theorems_hold(self) -> None:
        # Execute in a subprocess so libz3's internal C++ thread pool does not
        # persist in the pytest process and trip single-thread containment tests.
        proc = subprocess.run(
            [sys.executable, "-B", str(FORMAL_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"Z3 formal verification failed:\n{proc.stdout}\n{proc.stderr}")
        payload = json.loads(proc.stdout)
        results = payload["proofs"]
        self.assertGreaterEqual(len(results), 19)
        for r in results:
            self.assertTrue(r["status"].startswith("PROVED"), f"Theorem {r['name']} failed: {r}")

    @unittest.skipUnless(KANI_HOME.is_dir(), "AWS Kani 0.68.0 standalone bundle not installed")
    def test_all_five_kani_rust_mir_harnesses_verify(self) -> None:
        proc = subprocess.run(
            [str(KANI_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"Kani failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn(
            "Complete - 5 successfully verified harnesses, 0 failures, 5 total.",
            proc.stdout,
        )

    def test_tla_scheduler_explicit_state_model_checker_holds(self) -> None:
        mod = _load_module("verify_scheduler_tla_model", TLA_MODEL_SCRIPT)
        summary = mod.verify_tla_state_space()
        self.assertEqual(summary["deadlocks"], 0)
        self.assertGreater(int(summary["reachable_states"]), 100)

    @unittest.skipUnless(NVCC_BIN.is_file(), "CUDA nvcc not available on host")
    def test_sm121a_compiled_ptx_and_sass_isa_contracts_hold(self) -> None:
        mod = _load_module("verify_ptx_isa_contracts", PTX_ISA_SCRIPT)
        proofs = mod.verify_all_ptx_isa()
        self.assertGreaterEqual(len(proofs), 21)
        for p in proofs:
            self.assertEqual(p.max_stack_bytes, 0)
            self.assertEqual(p.max_spill_stores, 0)
            self.assertEqual(p.max_spill_loads, 0)


if __name__ == "__main__":
    unittest.main()
