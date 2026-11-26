# SPDX-License-Identifier: AGPL-3.0-only
"""Source wiring checks only, not an HTTP or scheduler execution receipt."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RawCompletionAuditWiring(unittest.TestCase):
    def test_reporting_reaches_both_scheduler_request_paths(self):
        source = (ROOT / "crates/spark-server/src/api/completions.rs").read_text()
        blocking = source.split("pub async fn completions(", 1)[1].split(
            "pub(super) async fn completions_stream(", 1
        )[0]
        streaming = source.split("pub(super) async fn completions_stream(", 1)[1].split(
            "#[cfg(test)]", 1
        )[0]
        for implementation in (blocking, streaming):
            self.assertIn(
                "intervention_ledger: intervention_ledger.clone()", implementation
            )
            self.assertIn("snapshot_shared_interventions(", implementation)
            self.assertNotIn("intervention_ledger: None", implementation)
        self.assertIn('"completion_thinking_prefix"', source)
        self.assertIn('"completion_bos_prefix"', source)
        self.assertIn('"completion_output_cleanup"', source)


if __name__ == "__main__":
    unittest.main()
