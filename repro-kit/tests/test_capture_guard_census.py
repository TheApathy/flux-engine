#!/usr/bin/env python3
"""Source-only census: every `begin_capture`/`cudaStreamBeginCapture` call
site under `crates/` must either go through `trait_impl::CaptureGuard` or be
on an explicit, individually-justified allowlist of hand-rolled patterns that
have been audited and verified safe (they close the capture on every exit path
by a different mechanism than `CaptureGuard`).

This is the regression guard for the bug class `CaptureGuard` exists to
close: a bare `gpu.begin_capture(stream)?; <fallible body>; gpu.end_capture
(stream)?;` skips `end_capture` on any `Err`/panic in the body, leaving the
CUDA stream permanently reported as capturing. If a new capture region is
added without `CaptureGuard` (or without updating this allowlist with a
reviewed justification), this test fails and names the file:line.
"""

import re
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[2]
CRATES = ROOT / "crates"

# Audited call sites confirmed to close the capture WITHOUT CaptureGuard:
#   - target_model_exl3.rs / dflash2_runtime.rs: hand-rolled — body runs
#     without `?`, `end_capture().and_then(destroy_graph)` cleanup on `Err`
#     (error-path cleanup; does not catch panics).
#   - ffn_graph_runtime.rs: explicit panic-safe state machine (catch_unwind
#     around begin/dispatch/end/destroy).
#   - deepseek_v41/forward.rs: `end_capture` always runs before the body's
#     `Result` propagates, and a failed body destroys any nonzero partial-graph
#     handle before re-raising (error-path cleanup; does not catch panics).
# A NEW bare `begin_capture` call site anywhere else in crates/ is not on
# this list and must use CaptureGuard instead of adding itself here.
ALLOWLISTED_HAND_ROLLED_CAPTURES = {
    "crates/spark-model/src/model/glm53/target_model_exl3.rs",
    "crates/spark-model/src/model/glm53/dflash2_runtime.rs",
    "crates/spark-model/src/model/glm53/ffn_graph_runtime.rs",
    "crates/spark-model/src/weight_loader/deepseek_v41/forward.rs",
}

# The CaptureGuard implementation itself legitimately calls begin_capture.
CAPTURE_GUARD_IMPL = "crates/spark-model/src/model/trait_impl/capture_guard.rs"

BEGIN_CAPTURE_CALL = re.compile(r"\bbegin_capture\s*\(")
CUDA_BEGIN_CAPTURE_CALL = re.compile(
    r"\b(?:cudaStreamBeginCapture|cuStreamBeginCapture(?:_v2)?)\s*\("
)

EXCLUDE_DIR_PARTS = {"tests", "target"}
CFG_TEST_ATTR = re.compile(r"^\s*#\[cfg\(test\)\]\s*$")
MOD_DECL = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+[A-Za-z_][A-Za-z0-9_]*\b")
LINE_STR_OR_COMMENT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'|//.*$')


def _non_cfg_test_lines(lines):
    """Yield `(lineno, line)` pairs outside inline `#[cfg(test)] mod ... { ... }` blocks."""
    saw_cfg_test = False
    in_cfg_test_mod = False
    brace_depth = 0
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if in_cfg_test_mod:
            code = LINE_STR_OR_COMMENT.sub("", line)
            brace_depth += code.count("{") - code.count("}")
            if brace_depth <= 0:
                in_cfg_test_mod = False
                brace_depth = 0
            continue
        if CFG_TEST_ATTR.match(line):
            saw_cfg_test = True
            continue
        if saw_cfg_test:
            if not stripped or stripped.startswith("#[") or stripped.startswith("//"):
                continue
            if MOD_DECL.match(line):
                code = LINE_STR_OR_COMMENT.sub("", line)
                if "{" in code:
                    brace_depth = code.count("{") - code.count("}")
                    in_cfg_test_mod = brace_depth > 0
                saw_cfg_test = False
                continue
            saw_cfg_test = False
        yield lineno, line


def _rust_files():
    for path in CRATES.rglob("*.rs"):
        if any(part in EXCLUDE_DIR_PARTS for part in path.parts):
            continue
        if path.name.endswith("_tests.rs"):
            continue
        yield path


class CaptureGuardCensusTests(TestCase):
    def test_every_begin_capture_call_site_is_guarded_or_allowlisted(self):
        unexplained = []
        for path in _rust_files():
            rel = path.relative_to(ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            for lineno, line in _non_cfg_test_lines(lines):
                if not BEGIN_CAPTURE_CALL.search(line):
                    continue
                # Skip the trait default/decl (`fn begin_capture(`), doc
                # comments, and the backend's own definition of the method.
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("///"):
                    continue
                if re.search(r"\bfn\s+begin_capture\s*\(", line):
                    continue
                if rel == CAPTURE_GUARD_IMPL:
                    continue
                if rel in ALLOWLISTED_HAND_ROLLED_CAPTURES:
                    continue
                unexplained.append(f"{rel}:{lineno}: {stripped}")

        self.assertEqual(
            unexplained,
            [],
            "Unguarded begin_capture call site(s) found outside CaptureGuard "
            "and the reviewed hand-rolled allowlist — wrap with "
            "trait_impl::capture_guard::CaptureGuard or add a reviewed, "
            "justified allowlist entry in this test:\n"
            + "\n".join(unexplained),
        )

    def test_non_cfg_test_lines_skips_inline_test_mods_but_keeps_prod_code(self):
        sample = [
            "fn before() { gpu.begin_capture(s)?; }",
            "#[cfg(test)]",
            "mod unit_tests {",
            "    fn t() { gpu.begin_capture(s)?; }",
            "}",
            "fn after() { gpu.begin_capture(s)?; }",
        ]
        kept = [line for _, line in _non_cfg_test_lines(sample)]
        self.assertEqual(
            kept,
            [
                "fn before() { gpu.begin_capture(s)?; }",
                "fn after() { gpu.begin_capture(s)?; }",
            ],
        )

    def test_cuda_begin_capture_driver_call_is_confined_to_the_backend(self):
        # The raw driver call (`cuStreamBeginCapture`) must only ever be
        # invoked from inside the one backend method that implements
        # `GpuBackend::begin_capture` — never directly from model code,
        # which must always go through the trait (and, from there,
        # CaptureGuard or an allowlisted hand-rolled region).
        allowed = {
            # The actual call, inside the one `GpuBackend::begin_capture` impl.
            "crates/spark-runtime/src/cuda_backend/gpu_impl.rs",
            # The `extern "C"` FFI declaration — not a call site.
            "crates/spark-runtime/src/cuda_backend.rs",
            # Standalone low-level CapturedStep primitive in spark-storage.
            "crates/spark-storage/src/cuda_graph.rs",
        }
        unexplained = []
        for path in _rust_files():
            rel = path.relative_to(ROOT).as_posix()
            if rel in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if CUDA_BEGIN_CAPTURE_CALL.search(line) and not line.strip().startswith("//"):
                    unexplained.append(f"{rel}:{lineno}: {line.strip()}")
        self.assertEqual(unexplained, [])

    def test_allowlist_entries_still_exist_and_still_call_begin_capture(self):
        # If one of these files is deleted or refactored to no longer call
        # begin_capture directly, the allowlist entry is stale and should be
        # removed — this catches that drift instead of silently widening
        # the exemption forever.
        for rel in ALLOWLISTED_HAND_ROLLED_CAPTURES:
            path = ROOT / rel
            self.assertTrue(path.exists(), f"allowlisted file missing: {rel}")
            code_lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if not line.strip().startswith("//")
            ]
            self.assertTrue(
                any(BEGIN_CAPTURE_CALL.search(line) for line in code_lines),
                f"{rel} is allowlisted as a hand-rolled capture region but no "
                "longer calls begin_capture — remove the stale allowlist entry",
            )
