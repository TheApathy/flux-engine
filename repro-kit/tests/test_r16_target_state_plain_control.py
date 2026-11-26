# SPDX-License-Identifier: AGPL-3.0-only
"""Source/proof-helper checks, not Rust execution or a GPU state receipt."""

import importlib.util
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "crates/spark-model/src/model/qwen4_target_state_receipt.rs"


class TargetStatePlainControlSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.classifier = cls.source.split("fn classify_capture_mode(", 1)[1].split(
            "\n#[cfg(", 1
        )[0]
        cls.capture = cls.source.split("fn capture_enabled(", 1)[1].split(
            "\n#[cfg(", 1
        )[0]

    def test_production_capture_uses_immutable_source_classifier(self):
        self.assertIn("validate_target(model, seq)?;", self.capture)
        self.assertIn(
            "let force_serial = model.speculation_forced_disabled;", self.capture
        )
        self.assertIn(
            "let proposer_kind = model.active_proposer().map(|proposer| proposer.kind());",
            self.capture,
        )
        self.assertIn(
            "classify_capture_mode(proposer_kind, force_serial)?;", self.capture
        )
        self.assertNotIn("std::env", self.classifier)

    def test_no_proposer_is_unloaded_with_either_kill_switch_setting(self):
        self.assertIn(
            '(None, _) => ("unloaded-target-only", "ordinary-target-only", false)',
            self.classifier,
        )
        self.assertNotIn("disabled control without a proposer", self.source)

    def test_loaded_serial_engaged_and_foreign_proposers_stay_distinct(self):
        compact = re.sub(r"\s+", " ", self.classifier)
        self.assertIn(
            '(Some(DraftProposerKind::Qwen4NativeMtp), true) => { '
            '("loaded-serial", "ordinary-target-only", true) }',
            compact,
        )
        self.assertIn(
            '(Some(DraftProposerKind::Qwen4NativeMtp), false) => { '
            '("engaged-native-mtp", "native-mtp", true) }',
            compact,
        )
        self.assertIn(
            '(Some(kind), _) => bail!("target-state receipt rejects proposer kind {kind:?}")',
            compact,
        )

    def test_secondary_wait_remains_exclusive_to_engaged_native_mode(self):
        self.assertRegex(
            self.capture,
            r'if mode == "engaged-native-mtp" \{\s*model\s*\.sync_secondary\(\)',
        )
        self.assertEqual(self.capture.count(".sync_secondary()"), 1)
        self.assertLess(
            self.capture.index(".sync_secondary()"),
            self.capture.index(".synchronize(stream)"),
        )

    def test_schema_hash_domains_and_state_coverage_are_not_relaxed(self):
        for marker in (
            "const SCHEMA: u32 = 1;",
            'b"atlas-qwen4-fed-tokens-v1"',
            'b"atlas-qwen4-target-state-v1"',
            "seq.seq_len > 0 && seq.tokens.len() == seq.seq_len",
            "seq.slot_idx < model.ssm_pool.max_slots",
            "gdn_layers == QWEN4_GDN_LAYERS",
            "!state.h_is_f16",
            "ple_live.len() == ple_stride",
            "kv_cache.config().cache_blocks_per_seq.is_none() && seq.disk_block_ids.is_empty()",
            "seq.block_table.len() >= required_blocks",
            "kv_cache.num_layers() == QWEN4_ATTN_LAYERS",
            "kv_cache.dtype_for_layer(layer_idx) == KvCacheDtype::Bf16",
            "let limit = capture_limit()?;",
            "if CAPTURES.load(Ordering::Acquire) >= limit {",
        ):
            self.assertIn(marker, self.source)

    def test_plain_process_proof_still_requires_the_real_kill_switch(self):
        path = ROOT / "repro-kit/r15-container-suite.py"
        spec = importlib.util.spec_from_file_location("plain_control_proof_test", path)
        suite = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(suite)
        plain_argv = [b"/sealed/server", b"serve", b"--no-tui"]
        suite._verify_live_mode(
            plain_argv, {b"ATLAS_DISABLE_SPECULATION": b"1"}, "none"
        )
        suite._verify_live_mode(
            plain_argv + [b"--speculative", b"--num-drafts", b"4"],
            {b"ATLAS_DISABLE_SPECULATION": b"1"},
            "none",
        )
        for environment in ({}, {b"ATLAS_DISABLE_SPECULATION": b"0"}):
            with self.subTest(environment=environment), self.assertRaisesRegex(
                ValueError, "requires effective ATLAS_DISABLE_SPECULATION=1"
            ):
                suite._verify_live_mode(plain_argv, environment, "none")


if __name__ == "__main__":
    unittest.main()
