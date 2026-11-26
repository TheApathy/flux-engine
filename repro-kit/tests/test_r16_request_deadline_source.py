# SPDX-License-Identifier: AGPL-3.0-only
"""Source wiring only. Rust deadline behavior requires native CPU tests."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "crates/spark-server/src/api"
SAMPLING = API_DIR / "chat/sampling_setup.rs"
CHAT_MOD = API_DIR / "chat/mod.rs"
CHAT_BLOCKING = API_DIR / "chat_blocking.rs"
CHAT_STREAM = API_DIR / "chat_stream_dispatch.rs"


class RequestDeadlineSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SAMPLING.read_text(encoding="utf-8")
        cls.chat_mod = CHAT_MOD.read_text(encoding="utf-8")
        cls.chat_blocking = CHAT_BLOCKING.read_text(encoding="utf-8")
        cls.chat_stream = CHAT_STREAM.read_text(encoding="utf-8")

    def test_checked_conversion_rejects_invalid_inputs_and_preserves_zero(self):
        helper = self.source.split("fn checked_request_deadline(", 1)[1].split(
            "pub(super) fn contract_decode_deadline(", 1
        )[0]
        self.assertIn("!timeout_secs.is_finite() || timeout_secs < 0.0", helper)
        self.assertIn("if timeout_secs == 0.0 {\n        return Ok(None);", helper)
        self.assertIn("Duration::try_from_secs_f32(timeout_secs)", helper)
        self.assertIn("now.checked_add(duration)", helper)
        self.assertNotIn("unwrap(", helper)
        self.assertNotIn("Instant::now() +", helper)

    def test_both_dispatch_paths_share_the_validated_request_deadline(self):
        build = self.source.split("pub(super) fn build_sampling(", 1)[1].split(
            "#[cfg(test)]", 1
        )[0]
        self.assertIn("req.timeout.unwrap_or(if dsv41 {\n        0.0", build)
        self.assertIn("state.request_timeout as f32", build)
        self.assertIn("checked_request_deadline(timeout_secs, std::time::Instant::now())", build)
        self.assertIn("StatusCode::BAD_REQUEST", build)
        self.assertIn('Some("timeout")', build)
        self.assertIn('Some("invalid_timeout")', build)
        self.assertNotIn("Duration::from_secs_f32(", build)
        self.assertNotIn("Instant::now() +", build)

        # Verify chat/mod.rs resolves timeout_at via sampling_setup::build_sampling and
        # forwards it to both streaming and blocking dispatchers without raw deadline math.
        self.assertIn("sampling_setup::build_sampling(", self.chat_mod)
        stream_call = self.chat_mod.split(
            "super::chat_stream_dispatch::dispatch_streaming(", 1
        )[1].split(".await;", 1)[0]
        blocking_call = self.chat_mod.split(
            "super::chat_blocking::run_blocking_path(", 1
        )[1].split(".await;", 1)[0]
        self.assertIn("timeout_at,", stream_call)
        self.assertIn("timeout_at,", blocking_call)
        self.assertIn("timeout_at: Option<std::time::Instant>,", self.chat_stream)
        self.assertIn("pub timeout_at: Option<std::time::Instant>,", self.chat_blocking)
        for name, text in (
            ("chat/mod.rs", self.chat_mod),
            ("chat_stream_dispatch.rs", self.chat_stream),
            ("chat_blocking.rs", self.chat_blocking),
        ):
            self.assertNotIn("Duration::from_secs_f32(", text, name)
            self.assertNotIn("Instant::now() +", text, name)

    def test_rust_regressions_cover_both_overflow_boundaries(self):
        for name in [
            "zero_still_disables_the_deadline",
            "positive_integral_and_fractional_timeouts_keep_their_deadlines",
            "negative_and_nonfinite_timeouts_are_errors_not_disabled_deadlines",
            "huge_finite_timeout_returns_an_error_without_panicking",
            "representable_duration_respects_the_platform_clock_range",
        ]:
            self.assertIn("fn " + name + "()", self.source)


if __name__ == "__main__":
    unittest.main()
