"""Source wiring only. These checks do not execute Rust or establish GPU parity."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPL = ROOT / "crates/spark-model/src/model/trait_impl"
CALL = "super::decode_qwen4_graph_policy::plain_qsa_graph_capture_allowed("
ARGS = (
    "self.config.is_qwen4_exp(),"
    "seq.qwen4_qsa_required,seq.seq_len,"
    "crate::layers::qwen4_qsa::graph_safe_enabled(),"
)


def compact(text):
    return re.sub(r"\s+", "", text)


def validate_dispatch(source):
    source = compact(source)
    binding = "letqsa_graph_capture_allowed=" + CALL + ARGS + ");"
    if binding not in source:
        raise ValueError("missing source-bound QSA policy")
    for name, end in (
        ("use_graphs", "letuse_qwen4_ple_suffix_graph"),
        ("use_qwen4_ple_suffix_graph", "letctx="),
    ):
        expression = source.split("let" + name + "=", 1)[1].split(end, 1)[0]
        if "&&qsa_graph_capture_allowed" not in expression:
            raise ValueError("missing graph admission: " + name)
    if source.index(binding) > source.index("letuse_graphs="):
        raise ValueError("policy evaluated after graph decision")


def validate_suffix(source):
    source = compact(source)
    guard = "anyhow::ensure!(" + CALL + ARGS + "),"
    if guard not in source:
        raise ValueError("missing defensive QSA policy")
    for operation in (
        "stage_timer::record_raw(stream)",
        "self.layers[0].decode(",
        "ple.forward_token(",
        "self.gpu.launch_graph(",
        "CaptureGuard::begin(",  # 13c712cdb: the capture is opened through the RAII guard
    ):
        if source.index(guard) > source.index(operation):
            raise ValueError("QSA check follows GPU work")


class PlainQsaGraphAdmissionSourceTests(unittest.TestCase):
    def test_policy_stays_pure_and_conservative(self):
        source = (IMPL / "decode_qwen4_graph_policy.rs").read_text()
        policy = compact(source.split("#[cfg(test)]", 1)[0])
        self.assertIn(
            "!target_is_qwen4||!qsa_required||(graph_safe_qsa&&host_seq_len>=2_048)",
            policy,
        )
        self.assertNotIn("std::env", policy)
        self.assertNotIn("unsafe", policy)

    def test_policy_module_is_compiled(self):
        self.assertIn(
            "mod decode_qwen4_graph_policy;", (IMPL / "mod.rs").read_text()
        )

    def test_both_plain_graph_decisions_are_guarded(self):
        validate_dispatch((IMPL / "decode_a.rs").read_text())

    def test_suffix_refuses_before_gpu_work(self):
        validate_suffix((IMPL / "decode_qwen4_ple_graph.rs").read_text())

    def test_each_unguarded_graph_path_is_rejected(self):
        source = (IMPL / "decode_a.rs").read_text()
        for occurrence in (0, 1):
            parts = source.split("&& qsa_graph_capture_allowed")
            self.assertEqual(len(parts), 3)
            parts[occurrence] += parts.pop(occurrence + 1)
            with self.assertRaises(ValueError):
                validate_dispatch("&& qsa_graph_capture_allowed".join(parts))

    def test_wrong_sequence_metadata_is_rejected(self):
        source = (IMPL / "decode_a.rs").read_text()
        for old, new in (
            ("self.config.is_qwen4_exp(),", "true,"),
            ("seq.qwen4_qsa_required,", "false,"),
            ("                seq.seq_len,", "                2_048,"),
            ("crate::layers::qwen4_qsa::graph_safe_enabled(),", "true,"),
        ):
            self.assertIn(old, source)
            with self.assertRaises(ValueError):
                validate_dispatch(source.replace(old, new))

    def test_late_suffix_guard_is_rejected(self):
        source = (IMPL / "decode_qwen4_ple_graph.rs").read_text()
        start = source.index("        anyhow::ensure!(")
        end = source.index("        anyhow::ensure!(", start + 1)
        guard = source[start:end]
        source = source[:start] + source[end:]
        insertion = source.index("        ple.forward_token(")
        with self.assertRaises(ValueError):
            validate_suffix(source[:insertion] + guard + source[insertion:])

    def test_native_policy_regressions_are_present(self):
        source = (IMPL / "decode_qwen4_graph_policy.rs").read_text()
        for name in (
            "non_qwen4_scheduler_qsa_hint_retains_graphs",
            "non_qsa_decode_retains_graphs",
            "host_qsa_metadata_is_never_captured",
            "graph_safe_indexer_does_not_make_dense_topology_dynamic",
            "graph_safe_sparse_topology_can_be_captured",
            "first_sparse_step_is_the_admission_boundary",
        ):
            self.assertIn("fn " + name + "()", source)


if __name__ == "__main__":
    unittest.main()
