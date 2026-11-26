# SPDX-License-Identifier: AGPL-3.0-only
"""FLUX_DEBUG, `flux bundle` and its scrubbing, and launcher console output."""

import importlib.machinery
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import TestCase, mock

SCRIPT = Path(__file__).resolve().parents[1] / "flux-autodetect"


def _load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


fa = _load(SCRIPT, "flux_autodetect_debug_bundle")

# Planted values that must never survive scrubbing.
PLANTED_FIXTURES = {
    "bearer": "Bearer abcDEF123456ghiJKL",
    "hf": "hf_PlantedHuggingFace0123",
    "sk": "sk-ant-planted-0123456789",
    "ghp": "ghp_PlantedGitHub012345",
    "kv": "password=hunter2planted",
    "env": "envvalue-planted-777",
}
PROMPT = "Write a haiku about my secret project Bluebird"
ENVIRON = {
    "USER": "alice",
    "HOSTNAME": "dgx-alice-07",
    "MY_SERVICE_TOKEN": PLANTED_FIXTURES["env"],
    "WANDB_API_KEY": "wandbplanted99",
    "FLUX_MODE": "auto",
}
LOG = f"""2026-10-29T13:33:09.565672Z  INFO spark::model_resolver: Model path: /home/alice/models/fn (local directory)
2026-10-29T13:33:09.565700Z  INFO spark::main_modules::serve_router: Listening on 192.168.1.20:8899 (peer 10.0.0.3, fd12:3456::1) started by alice
2026-10-29T13:33:09.565800Z  WARN spark::auth: header Authorization: {PLANTED_FIXTURES['bearer']} from alice@example.com on dgx-alice-07
2026-10-29T13:33:09.565900Z  INFO spark::download: token {PLANTED_FIXTURES['hf']} {PLANTED_FIXTURES['sk']} {PLANTED_FIXTURES['ghp']} {PLANTED_FIXTURES['kv']} svc={PLANTED_FIXTURES['env']} wandbplanted99
2026-10-29T13:33:09.566000Z  INFO spark::api::chat_stream::tool_handlers: Tool call: write_file({{"path": "/tmp/bluebird.txt", "content": "{PROMPT}"}})
2026-10-29T13:33:09.566100Z  INFO spark::api::chat::msg_entry: Dropped content-free client system message (would bias the model toward terse output) dropped={PROMPT}
2026-10-29T13:33:09.566200Z  INFO atlas::tool_debug: raw pre-parse output (tools_active, choice 0): "{PROMPT}"
2026-10-29T13:33:09.566300Z DEBUG spark::scheduler::decode_logits_step: DECODE: n=4 step=15.0ms (66.7 tok/s) tokens=[151644, 872, 198, 4521]
2026-10-29T13:33:09.566400Z  INFO spark::api: request body {{"messages": [{{"role": "user", "content": "{PROMPT}"}}]}}
2026-10-29T13:33:09.566500Z  INFO spark::scheduler::prefill_b_step: Prefilling: 12 prompt tokens, max_tokens=256 engine 0.9.15.5 on 127.0.0.1
"""


def _scrub(text, include_prompts=False):
    counts = {}
    return fa.scrub_text(text, include_prompts=include_prompts, environ=ENVIRON, counts=counts), counts


class ScrubTextTest(TestCase):
    def test_planted_secrets_paths_ips_and_prompts_are_removed(self):
        out, counts = _scrub(LOG)
        for value in list(PLANTED_FIXTURES.values()) + ["hunter2planted", "wandbplanted99"]:
            self.assertNotIn(value.split("=")[-1].split()[-1], out, value)
        for ident in ("/home/alice", "alice@example.com", "dgx-alice-07", "192.168.1.20", "10.0.0.3", "fd12:3456::1"):
            self.assertNotIn(ident, out)
        self.assertNotIn("alice", out)
        self.assertNotIn("Bluebird", out)
        self.assertNotIn("151644", out)
        for cat in ("auth_token", "api_key", "key_value", "env_secret", "home_path", "email", "ip_address",
                    "hostname", "username", "prompt_text", "token_ids"):
            self.assertGreater(counts.get(cat, 0), 0, cat)

    def test_operational_facts_survive(self):
        out, _ = _scrub(LOG)
        for keep in ("Prefilling: 12 prompt tokens, max_tokens=256", "0.9.15.5", "127.0.0.1",
                     "spark::model_resolver", "DECODE: n=4 step=15.0ms", "Tool call: write_file("):
            self.assertIn(keep, out)

    def test_prompt_opt_in_keeps_text_but_never_secrets(self):
        out, counts = _scrub(LOG, include_prompts=True)
        self.assertIn("Bluebird", out)
        self.assertIn("151644", out)
        self.assertNotIn(PLANTED_FIXTURES["hf"], out)
        self.assertNotIn("/home/alice", out)
        self.assertNotIn("prompt_text", counts)

    def test_env_snapshot_keeps_names_and_redacts_secret_values(self):
        snap = fa._scrub_env_snapshot({"HF_TOKEN": "hf_x" * 4, "FLUX_DEBUG": "1", "ATLAS_API_KEY": "k" * 8, "HOME": "/home/a"})
        self.assertEqual(snap, {"ATLAS_API_KEY": "<REDACTED>", "FLUX_DEBUG": "1", "HF_TOKEN": "<REDACTED>"})


class DebugLogEnvTest(TestCase):
    def test_off_by_default(self):
        self.assertEqual(fa.resolve_debug_log_env({}, results_writable=True), {})
        self.assertEqual(fa.resolve_debug_log_env({"FLUX_DEBUG": "0"}, results_writable=True), {})

    def test_on_defaults_the_file_under_results(self):
        self.assertEqual(
            fa.resolve_debug_log_env({"FLUX_DEBUG": "1"}, results_writable=True),
            {"FLUX_DEBUG": "1", "FLUX_DEBUG_FILE": "/results/flux-debug.log"},
        )

    def test_explicit_file_is_kept(self):
        env = {"FLUX_DEBUG": "on", "FLUX_DEBUG_FILE": "/tmp/d.log"}
        self.assertEqual(fa.resolve_debug_log_env(env, results_writable=False), {"FLUX_DEBUG": "1"})

    def test_no_results_warns_and_turns_it_off_without_blocking_the_start(self):
        err = io.StringIO()
        with redirect_stderr(err):
            out = fa.resolve_debug_log_env({"FLUX_DEBUG": "1"}, results_writable=False)
        self.assertEqual(out, {"FLUX_DEBUG": "0"})
        self.assertIn("debug log is off", err.getvalue())

    def test_bad_values_fail_fast(self):
        with self.assertRaises(ValueError):
            fa.resolve_debug_log_env({"FLUX_DEBUG": "verbose"}, results_writable=True)
        with self.assertRaises(ValueError):
            fa.resolve_debug_log_env({"FLUX_DEBUG": "1", "FLUX_DEBUG_MAX_MB": "0"}, results_writable=True)

    def test_keys_are_not_aliased_to_atlas(self):
        for k in ("FLUX_DEBUG", "FLUX_DEBUG_FILE", "FLUX_DEBUG_MAX_MB", "FLUX_DEBUG_FILTER"):
            self.assertIn(k, fa._FLUX_META_ENV_KEYS)

    def test_help_documents_switch_and_export(self):
        for argv in (["flux", "env"], ["flux", "--help"]):
            buf = io.StringIO()
            with mock.patch.object(fa.sys, "argv", argv), redirect_stdout(buf):
                fa.main()
            self.assertIn("FLUX_DEBUG", buf.getvalue())
        self.assertIn("flux-debug-<version>-<UTC>.tar.gz", buf.getvalue())


class _BundleBase(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.results = Path(self.tmp.name)

    def _bundle(self, include_prompts=False):
        env = {**ENVIRON, "FLUX_PORT": "1", "PATH": os.environ.get("PATH", "")}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(fa.shutil, "which", return_value=None), \
                mock.patch.object(fa.socket, "gethostname", return_value="dgx-alice-07"), \
                mock.patch.object(fa, "export_submission_bundle", side_effect=OSError("no models")):
            return fa.export_debug_bundle(self.results, {"host": True, "logs": True}, include_prompts=include_prompts)

    def _members(self, archive):
        with tarfile.open(archive) as tar:
            return {m.name.split("/", 1)[1]: tar.extractfile(m).read().decode() for m in tar.getmembers()}


class BundleTest(_BundleBase):

    def test_bundle_after_a_failed_load_has_the_expected_scrubbed_files(self):
        failure = ("[flux] ===== model load failed: load-phase debug log, last 1 of 1 lines =====\n"
                   "DEBUG load /home/alice/m\n[flux] ===== end of load-phase debug log =====\n"
                   "Error: Failed to parse config.json\n\nCaused by:\n    0: EOF\n")
        (self.results / "flux-server.log").write_text(LOG + failure)
        (self.results / "flux-debug.log").write_text(LOG)
        (self.results / "flux-debug.log.1").write_text("older " + PLANTED_FIXTURES["sk"] + "\n")
        (self.results / "flux-load-debug.log").write_text("DEBUG killed mid-load at /home/alice/m\n")
        (self.results / "flux-launch.json").write_text(json.dumps({"argv": ["/opt/flux/bin/flux-server-real", "--model-from-path", "/home/alice/m"]}))
        res = self._bundle()

        archive = Path(res["archive"])
        self.assertEqual(archive.parent, self.results)
        self.assertRegex(archive.name, r"^flux-debug-[A-Za-z0-9._-]+-\d{8}T\d{6}Z\.tar\.gz$")
        self.assertEqual([p.name for p in self.results.glob("*.tar.gz")], [archive.name])
        members = self._members(archive)
        for name in ("MANIFEST.json", "logs/flux-server.log", "logs/flux-debug.log", "logs/flux-debug.log.1",
                     "logs/flux-load-debug.log", "load-failure/from-flux-server-log.txt",
                     "load-failure/killed-load-tail.txt", "launch/flux-launch.json", "launch/environment.json",
                     "system/facts.json", "system/endpoint.json"):
            self.assertIn(name, members)
        blob = "\n".join(members.values())
        for value in list(PLANTED_FIXTURES.values()) + ["/home/alice", "dgx-alice-07", "192.168.1.20", "Bluebird", "151644"]:
            self.assertNotIn(value, blob)
        self.assertIn("Caused by:", members["load-failure/from-flux-server-log.txt"])
        manifest = json.loads(members["MANIFEST.json"])
        self.assertEqual(manifest["prompt_content"], "redacted")
        listed = {f["file"]: f for f in manifest["files"]}
        self.assertEqual(set(listed), set(members) - {"MANIFEST.json"})
        self.assertGreater(listed["logs/flux-server.log"]["redactions"]["content_record"], 0)
        self.assertIn("submission bundle skipped", manifest["submission_bundle"]["error"])
        endpoint = json.loads(members["system/endpoint.json"])
        self.assertIsNone(endpoint["/v1/models"]["status"])

    def test_stopped_container_with_only_a_server_log(self):
        (self.results / "flux-server.log").write_text("INFO Listening on 0.0.0.0:8899\n")
        members = self._members(self._bundle()["archive"])
        self.assertIn("logs/flux-server.log", members)
        self.assertNotIn("logs/flux-debug.log", members)
        self.assertNotIn("load-failure/killed-load-tail.txt", members)

    def test_prompt_opt_in(self):
        (self.results / "flux-server.log").write_text(LOG)
        members = self._members(self._bundle(include_prompts=True)["archive"])
        self.assertIn("Bluebird", members["logs/flux-server.log"])
        self.assertNotIn(PLANTED_FIXTURES["hf"], members["logs/flux-server.log"])

    def test_unwritable_results_is_a_clear_error(self):
        err = io.StringIO()
        with mock.patch.object(fa.sys, "argv", ["flux", "bundle", "--output-dir", str(self.results / "missing")]), \
                redirect_stderr(err), redirect_stdout(io.StringIO()):
            self.assertEqual(fa.main(), 1)
        self.assertIn("cannot write the debug bundle", err.getvalue())


# --- Console output snapshot across exec, tee, failure, and FLUX_DEBUG -------

FAKE_SERVER = """import sys
print("2026-10-29T13:00:00.000001Z  INFO spark::main_modules::serve: Atlas Spark starting...", flush=True)
print("2026-10-29T13:00:00.000002Z  WARN spark::main_modules::serve_router: Atlas is listening on 0.0.0.0:8899", flush=True)
print("server stderr line", file=sys.stderr, flush=True)
print("2026-10-29T13:00:00.000003Z  INFO spark::main_modules::serve_router: Listening on 0.0.0.0:8899", flush=True)
sys.exit(int(sys.argv[1]))
"""

DRIVER = """import importlib.machinery, importlib.util, os, sys
from unittest import mock
launcher, fake, rc, mode = sys.argv[1:5]
loader = importlib.machinery.SourceFileLoader("launcher_under_test", launcher)
spec = importlib.util.spec_from_loader(loader.name, loader)
fa = importlib.util.module_from_spec(spec)
loader.exec_module(fa)
info = {"kernel_target": "qwen3.8-27b", "selected_binary": "/opt/flux/bin/flux-server-universal",
        "path": "/models/qwen3.8-27b", "name": "qwen3.8-27b", "family": "qwen", "quantization": "nvfp4",
        "ple_dir": None, "speculative": {"mode": "none"}, "max_position_embeddings": 262144,
        "recommended_tool_call_parser": "qwen3_coder", "num_hidden_layers": 64,
        "hidden_size": 5120, "status": "ready"}
orig_plan = fa.build_launch_plan
def plan(*a, **k):
    p = orig_plan(*a, **k)
    p["command"] = [sys.executable, fake, rc]
    return p
orig_is_dir, orig_access = fa._safe_is_dir, os.access
tee = mode == "tee"
patches = [
    mock.patch.object(fa, "resolve_model_query", return_value=info),
    mock.patch.object(fa, "build_launch_plan", side_effect=plan),
    mock.patch.object(fa, "detect_hardware_topology",
                      return_value={"hardware_recipe": "flux-v9-max-prefill.env", "unified_memory_architecture": True}),
    mock.patch.object(fa, "collect_diagnostics", return_value={"warnings": []}),
    mock.patch.object(fa, "_safe_is_dir", side_effect=lambda p: (tee and str(p) == "/results") or orig_is_dir(p)),
    mock.patch.object(fa.os, "access", side_effect=lambda p, m: (tee and str(p) == "/results") or orig_access(p, m)),
]
for p in patches:
    p.start()
sys.argv = ["flux", "serve"]
sys.exit(fa.main())
"""

TS = re.compile(r"\d{4}-\d\d-\d\dT[\d:.]+Z")

EXPECTED_CONSOLE_STDOUT = [
    "",
    "\u2500\u2500 A P A T H Y   F L U X   E N G I N E   v1.0.2  \u00b7  Qwen3.8-27B launch profile",
    "   \u2022 Checkpoint     : qwen3.8-27b (nvfp4, 64L)",
    "   \u2022 Kernel target  : qwen3.8-27b (flux-server-universal)",
    "   \u2022 Recipe         : flux-qwen38-27b-dflash2.env",
    "   \u2022 Speculation    : disabled (target-only)",
    "   \u2022 Geometry       : concurrency=C1 \u00b7 max_seq_len=262,144 \u00b7 prefill_chunk=8192 \u00b7 kv=nvfp4",
    "   \u2022 Modes          : FLUX_MODE recipe modes are Qwen3.8-Flash-Next only; this model runs its own recipe",
    "",
    "[flux] Auto-detected Model : qwen3.8-27b (qwen, nvfp4)",
    "[flux] Kernel Target       : qwen3.8-27b (/opt/flux/bin/flux-server-universal)",
    "[flux] Execution Mode      : auto (Qwen3.8-27B launch profile (flux-qwen38-27b-dflash2.env), target-only decode; the FLUX_MODE recipe modes apply to Qwen3.8-Flash-Next only)",
    "[flux] Active Recipe File  : flux-qwen38-27b-dflash2.env",
    "[flux] Concurrency / Seq   : concurrency=1, max_seq_len=262144, prefill_chunk=8192, kv_dtype=nvfp4",
    "[flux] Thinking Budget Cap : 262144 (model_profile)",
    "TS  INFO spark::main_modules::serve: Atlas Spark starting...",
    "TS  WARN spark::main_modules::serve_router: Atlas is listening on 0.0.0.0:8899",
    "TS  INFO spark::main_modules::serve_router: Listening on 0.0.0.0:8899",
]


class ConsoleIdentitySnapshotTest(TestCase):
    """Verify stable launcher console output across exec, tee, failure, and FLUX_DEBUG paths."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.fake = d / "fake_server.py"
        cls.fake.write_text(FAKE_SERVER)
        cls.driver = d / "driver.py"
        cls.driver.write_text(DRIVER)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, rc, mode, extra_env=None):
        log = Path(self.tmp.name) / f"server-{mode}-{rc}.log"
        log.unlink(missing_ok=True)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": self.tmp.name, "FLUX_LOG_FILE": str(log),
               "FLUX_LOAD_DEBUG_FILE": str(Path(self.tmp.name) / "none-yet.log"), "PYTHONDONTWRITEBYTECODE": "1",
               "FLUX_ENGINE_VERSION": "1.0.2",
               **(extra_env or {})}
        p = subprocess.run([sys.executable, "-B", str(self.driver), str(SCRIPT), str(self.fake), str(rc), mode],
                           env=env, capture_output=True, text=True, timeout=20)
        tee = log.read_text() if log.exists() else ""
        norm = lambda t: TS.sub("TS", t).splitlines()
        err_lines = [line for line in norm(p.stderr) if not line.startswith("[flux] WARN: /opt/flux/lib/")]
        return p.returncode, norm(p.stdout), err_lines, norm(tee)

    def _assert_matches_snapshot(self, rc, mode, extra_env=None):
        got_rc, got_out, got_err, got_tee = self._run(rc, mode, extra_env)
        self.assertEqual(got_rc, rc)
        self.assertIn("Listening on 0.0.0.0:8899", "\n".join(got_out + got_tee))
        self.assertEqual(got_out, EXPECTED_CONSOLE_STDOUT)
        self.assertEqual(got_err, ["server stderr line"])
        self.assertEqual(got_tee, [])
        return got_rc, got_out, got_err, got_tee

    def test_successful_start_exec_path(self):
        self._assert_matches_snapshot(0, "exec")

    def test_successful_start_tee_path_and_log_file(self):
        self.assertEqual(self._assert_matches_snapshot(0, "tee"), self._run(0, "exec"))

    def test_failed_start_tee_path(self):
        self._assert_matches_snapshot(1, "tee")

    def test_debug_on_leaves_the_console_unchanged(self):
        debug_env = {"FLUX_DEBUG": "1", "FLUX_DEBUG_FILE": str(Path(self.tmp.name) / "flux-debug.log")}
        self.assertEqual(self._assert_matches_snapshot(0, "tee", debug_env), self._run(0, "exec"))


# --- Redaction hardening tests -----------------------------------------------

CONTENT_LOG = """2026-10-29T13:00:00.1Z DEBUG spark::scheduler::mtp_step: MTP bootstrap: tok=5 → drafts=[7, 8]
2026-10-29T13:00:00.2Z  INFO spark::api::chat_stream::tool_handlers: Tool call: write_file({"path": "a",
  "content": "second line of the Bluebird prompt"
  …
2026-10-29T13:00:00.3Z  INFO spark::api::chat_stream::tool_handlers: Tool call: run(Bluebird truncated pre…)
2026-10-29T13:00:00.4Z  INFO atlas::tool_debug: raw pre-parse output (tools_active, choice 0): "Bluebird"
2026-10-29T13:00:00.5Z DEBUG spark::scheduler::decode_logits_step: DECODE: n=1 step=15.0ms (66.7 tok/s) tokens=[12345]
2026-10-29T13:00:00.6Z  INFO spark::scheduler::verify_dflash_step: DFLASH K=γ verify: γ=3 verify_input=[1, 2] verified=[1]
2026-10-29T13:00:00.7Z DEBUG spark::api: Message { role: "user", content: "Bluebird" }
2026-10-29T13:00:00.8Z DEBUG spark::api: body {\\"content\\": \\"Bluebird\\"}
2026-10-29T13:00:00.9Z  INFO spark::scheduler: Prefilling: 12 prompt tokens, max_tokens=256
"""


class RedactionHardeningTest(TestCase):
    def test_secret_flag_values_are_redacted_in_both_forms(self):
        argv = ["/opt/flux/bin/flux-server-real", "serve", "--auth-token", "tok-SPACE-123", "--hf-token=tok-EQ-456",
                "--require-auth", "--auth-tokens-file", "/secrets/t", "--max-prefill-tokens", "4096"]
        out = fa.redact_argv(argv)
        self.assertEqual(out[3], "<REDACTED>")
        self.assertEqual(out[4], "--hf-token=<REDACTED>")
        self.assertEqual(out[5], "--require-auth")
        self.assertEqual(out[7], "<REDACTED>")
        self.assertEqual(out[8:], ["--max-prefill-tokens", "4096"])
        text, counts = _scrub(json.dumps(argv) + "\ncmd: " + " ".join(argv))
        self.assertNotIn("tok-SPACE-123", text)
        self.assertNotIn("tok-EQ-456", text)
        self.assertIn("4096", text)
        self.assertGreater(counts["secret_flag"], 0)

    def test_launch_record_is_redacted_and_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            plan = {"command": ["/bin/s", "serve", "--auth-token", "tok-SPACE-123"],
                    "env": {"HF_TOKEN": "hf_PlantedHuggingFace0123", "FLUX_MODE": "auto"}, "mode": "auto"}
            with mock.patch.object(fa.sys, "argv", ["flux", "serve", "--api-key", "argvsecret99"]):
                fa.write_launch_record(Path(d), plan, {"name": "m"})
            path = Path(d) / "flux-launch.json"
            body = path.read_text()
            for secret in ("tok-SPACE-123", "hf_PlantedHuggingFace0123", "argvsecret99"):
                self.assertNotIn(secret, body)
            self.assertIn('"FLUX_MODE": "auto"', body)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_auth_headers_and_url_credentials(self):
        text, counts = _scrub("Authorization: Basic dXNlcjpwYXNzd29yZA== fetch https://bob:pw-Planted@hub.example/x")
        self.assertNotIn("dXNlcjpwYXNzd29yZA", text)
        self.assertNotIn("pw-Planted", text)
        self.assertIn("https://<REDACTED>@hub.example/x", text)
        self.assertGreater(counts["url_credentials"], 0)

    def test_content_records_are_dropped_whole(self):
        counts = {}
        out = fa.scrub_log_text(CONTENT_LOG, counts=counts, environ={})
        self.assertNotIn("Bluebird", out)
        self.assertNotIn("second line", out)
        for ids in ("[7, 8]", "12345", "[1, 2]"):
            self.assertNotIn(ids, out)
        self.assertEqual(counts["content_record"], 8)
        self.assertIn("Prefilling: 12 prompt tokens, max_tokens=256", out)
        self.assertEqual(len(out.splitlines()), 9, out)

    def test_content_records_kept_on_opt_in(self):
        out = fa.scrub_log_text(CONTENT_LOG, include_prompts=True, environ={})
        self.assertIn("Bluebird", out)
        self.assertIn("[7, 8]", out)

    def test_token_lists_of_any_length_are_redacted_inline(self):
        text, _ = _scrub("tok drafts=[7, 8] tokens=[12345] pending_drafts=[9]")
        for ids in ("7, 8", "12345", "[9]"):
            self.assertNotIn(ids, text)

    def test_public_submission_bundle_drops_prompt_and_content_lines(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "flux-server.log").write_text(
                "INFO Prefilling: 12 prompt tokens\nINFO content filter Bluebird\nINFO Listening on 0.0.0.0:8899\n")
            with mock.patch.object(fa, "generate_standalone_model_card", return_value={}):
                fa.export_submission_bundle(Path(d), {"host": True, "logs": True})
            body = (Path(d) / "submission-bundle" / "flux-server.sanitized.log").read_text()
        self.assertEqual(body.strip(), "INFO Listening on 0.0.0.0:8899")


class BundleFileSafetyTest(_BundleBase):
    def test_archive_is_owner_only(self):
        (self.results / "flux-server.log").write_text("INFO ok\n")
        archive = Path(self._bundle()["archive"])
        self.assertEqual(archive.stat().st_mode & 0o777, 0o600)

    def test_partial_file_is_removed_on_failure(self):
        (self.results / "flux-server.log").write_text("INFO ok\n")
        with mock.patch("tarfile.open", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self._bundle()
        self.assertEqual(list(self.results.glob(".*partial")), [])
        self.assertEqual(list(self.results.glob("*.tar.gz")), [])

    def test_symlinks_out_of_results_are_skipped(self):
        outside = tempfile.NamedTemporaryFile("w", delete=False)
        outside.write("SECRET-OUTSIDE-FILE\n")
        outside.close()
        self.addCleanup(os.unlink, outside.name)
        (self.results / "flux-server.log").symlink_to(outside.name)
        members = self._members(self._bundle()["archive"])
        self.assertNotIn("logs/flux-server.log", members)
        self.assertNotIn("SECRET-OUTSIDE-FILE", "\n".join(members.values()))
        manifest = json.loads(members["MANIFEST.json"])
        self.assertEqual(manifest["skipped"][0]["file"], "flux-server.log")

    def test_bundle_uses_record_dropping_for_logs(self):
        (self.results / "flux-debug.log").write_text(CONTENT_LOG)
        members = self._members(self._bundle()["archive"])
        self.assertNotIn("Bluebird", members["logs/flux-debug.log"])
        self.assertIn("<REDACTED_RECORD: request content>", members["logs/flux-debug.log"])

    def test_help_states_exec_path_and_privacy(self):
        buf = io.StringIO()
        with mock.patch.object(fa.sys, "argv", ["flux", "env"]), redirect_stdout(buf):
            fa.main()
        text = buf.getvalue()
        self.assertIn("execs into the", text)
        self.assertIn("0600", text)
        self.assertIn("FLUX_OPT_OUT=all", text)


# --- Token-ID redaction and private file ownership tests ---------------------

TOKEN_ID_LOG = """2026-10-29T13:00:01.1Z DEBUG spark::scheduler::decode_step: CONC_DIAG n=2: [0: slot=0 seq_len=40 bt=3/3 last=12345 out_n=3] [1: slot=1 seq_len=9 bt=7/1 last=678 out_n=1]
2026-10-29T13:00:01.2Z  INFO spark::scheduler::decode_logits_step: raw_argmax=4242 sampled=4242 think_ended=true SERIAL_POLICY_DECODE
2026-10-29T13:00:01.3Z  INFO spark::scheduler::mtp_step: sampled=31337 SPEC_RAW_BOOTSTRAP
2026-10-29T13:00:01.4Z  INFO spark::scheduler::run_standard: Prefill first token: 27182
2026-10-29T13:00:01.5Z  INFO spark::scheduler::verify_k2_step: K2 REJECT: ep=1μs seq_len=50 last_tok=16180 prev_draft=11 v0_verified=12 new_draft=13
2026-10-29T13:00:01.6Z ERROR spark::scheduler::spec_policy_accept: spec-policy verify: raw argmax is outside vocabulary; finishing seq row=0 token=99999 vocab=151936
2026-10-29T13:00:01.7Z  INFO spark::main_modules::serve_phases::tokenizer_runtime: Thinking end token: 151668 (</think>)
"""


class TokenIdAndOwnershipTest(TestCase):
    def test_per_sequence_token_ids_never_survive(self):
        counts = {}
        out = fa.scrub_log_text(TOKEN_ID_LOG, counts=counts, environ={})
        for tid in ("12345", "678", "4242", "31337", "27182", "16180", "99999"):
            self.assertNotIn(tid, out)
        # The ERROR line keeps its diagnosis, only the id goes; model constants stay.
        self.assertIn("raw argmax is outside vocabulary", out)
        self.assertIn("vocab=151936", out)
        self.assertIn("Thinking end token: 151668", out)
        self.assertGreaterEqual(counts["content_record"], 5)

    def test_inline_token_id_fields_without_record_drop(self):
        text, counts = _scrub("x last=12345 tok=7 token_id=8 next=9 argmax=10 id=11 raw_top1_id=Some(12) seq_len=40")
        for tid in ("12345", "=7", "=8", "=9", "=10", "=11", "12)"):
            self.assertNotIn(tid, text)
        self.assertIn("seq_len=40", text)

    def test_private_files_take_the_directory_owner(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "f.json"
            fa._write_private(path, "{}")
            st, dst = path.stat(), Path(d).stat()
            self.assertEqual((st.st_uid, st.st_gid), (dst.st_uid, dst.st_gid))
            self.assertEqual(st.st_mode & 0o777, 0o600)

    def test_ownership_failure_falls_back_to_0640_with_one_warning(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with mock.patch.object(fa.os, "fchown", side_effect=PermissionError("EPERM")), \
                    mock.patch.object(fa, "_OWNERSHIP_WARNED", False), redirect_stderr(err):
                fa._write_private(Path(d) / "a", "x")
                fa._write_private(Path(d) / "b", "y")
            self.assertEqual((Path(d) / "a").stat().st_mode & 0o777, 0o640)
            self.assertEqual(err.getvalue().count("WARN"), 1)

    def test_archive_takes_the_results_owner(self):
        with tempfile.TemporaryDirectory() as d:
            env = {**ENVIRON, "FLUX_PORT": "1", "PATH": os.environ.get("PATH", "")}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(fa.shutil, "which", return_value=None), \
                    mock.patch.object(fa, "export_submission_bundle", side_effect=OSError("no models")):
                archive = Path(fa.export_debug_bundle(Path(d), {"host": True, "logs": True})["archive"])
            self.assertEqual(archive.stat().st_uid, Path(d).stat().st_uid)
            self.assertEqual(archive.stat().st_mode & 0o777, 0o600)

    def test_unrelated_atlas_debug_does_not_become_flux_debug(self):
        env = fa._synchronize_flux_atlas_env({"ATLAS_DEBUG": "2", "ATLAS_DEBUG_FILE": "/x", "ATLAS_LOAD_DEBUG": "z",
                                              "ATLAS_MAX_BATCH_TOKENS": "8"})
        for k in ("FLUX_DEBUG", "FLUX_DEBUG_FILE", "FLUX_LOAD_DEBUG"):
            self.assertNotIn(k, env)
        self.assertEqual(env["FLUX_MAX_BATCH_TOKENS"], "8")

    def test_secret_value_starting_with_a_dash(self):
        self.assertEqual(fa.redact_argv(["--auth-token", "-dashSecret1", "--port", "1"]),
                         ["--auth-token", "<REDACTED>", "--port", "1"])
        self.assertEqual(fa.redact_argv(["--require-auth", "--port", "1"]), ["--require-auth", "--port", "1"])
        text, _ = _scrub("cmd --auth-token -dashSecret1 --port 1")
        self.assertNotIn("dashSecret1", text)

    def test_help_describes_when_the_launcher_stays_parent(self):
        buf = io.StringIO()
        with mock.patch.object(fa.sys, "argv", ["flux", "env"]), redirect_stdout(buf):
            fa.main()
        text = " ".join(buf.getvalue().split())
        self.assertIn("Under the default FLUX_OPT_OUT=all it execs into the server even without a TTY", text)


class SubmissionOwnershipTest(TestCase):
    """The pre-existing card/submission outputs take the /results owner too."""

    def _calls(self, fn):
        calls = []
        with mock.patch.object(fa.os, "chown", side_effect=lambda p, u, g, follow_symlinks=True: calls.append(
                (Path(p).name, u, g, follow_symlinks))):
            fn()
        return calls

    def test_card_and_submission_bundle_are_chowned_to_the_results_owner(self):
        with tempfile.TemporaryDirectory() as d:
            results = Path(d)
            (results / "flux-server.log").write_text("INFO Listening on 0.0.0.0:8899\n")
            st = results.stat()
            with mock.patch.object(fa.shutil, "which", return_value=None):
                calls = self._calls(lambda: fa.export_submission_bundle(results, {"host": True, "logs": True}))
            names = {c[0] for c in calls}
            for expected in ("MODEL_CARD.md", "flux-diagnostics.json", "submission-bundle",
                             "SUBMISSION_MANIFEST.json", "flux-server.sanitized.log"):
                self.assertIn(expected, names)
            self.assertTrue(all((u, g, f) == (st.st_uid, st.st_gid, False) for _, u, g, f in calls))

    def test_refused_chown_warns_once_and_keeps_going(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("a", "b"):
                (Path(d) / n).write_text("x")
            before_mode = (Path(d) / "b").stat().st_mode & 0o777
            err = io.StringIO()
            with mock.patch.object(fa.os, "chown", side_effect=PermissionError("EPERM")), \
                    mock.patch.object(fa, "_OWNERSHIP_WARNED", False), redirect_stderr(err):
                fa.adopt_results_owner([Path(d) / "a", Path(d) / "b"], Path(d))
            self.assertEqual(err.getvalue().count("WARN"), 1)
            # Modes are left as they were; only ownership is attempted.
            self.assertEqual((Path(d) / "b").stat().st_mode & 0o777, before_mode)
            self.assertEqual((Path(d) / "b").read_text(), "x")

    def test_real_chown_to_own_directory_owner_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "submission-bundle"
            sub.mkdir()
            (sub / "x.json").write_text("{}")
            err = io.StringIO()
            with redirect_stderr(err):
                fa.adopt_results_owner([sub], Path(d))
            self.assertEqual(err.getvalue(), "")
            self.assertEqual((sub / "x.json").stat().st_uid, Path(d).stat().st_uid)


class BundleImageLabelsTest(_BundleBase):
    """`flux bundle` carries the build-time copy of the image labels when the image has one."""

    def _labels_bundle(self, path):
        with mock.patch.object(fa, "IMAGE_LABELS_PATH", path):
            members = self._members(self._bundle()["archive"])
        manifest = json.loads(members["MANIFEST.json"])
        listed = {f["file"]: f for f in manifest["files"]}
        self.assertIn("system/image-labels.json", listed)
        return json.loads(members["system/image-labels.json"]), members, listed

    def test_labels_file_is_bundled(self):
        labels = {"org.opencontainers.image.version": "9.15.6-rc1",
                  "com.theapathy.flux.binary-sha256": "a" * 64,
                  "org.opencontainers.image.source": "https://github.com/TheApathy/flux-engine"}
        path = self.results / "IMAGE_LABELS.json"
        path.write_text(json.dumps({"schema": "flux-image-labels-v1", "labels": labels}))
        rec, members, listed = self._labels_bundle(path)
        self.assertTrue(rec["present"])
        self.assertEqual(rec["labels"], labels)
        self.assertEqual(rec["file_schema"], "flux-image-labels-v1")
        self.assertEqual(listed["system/image-labels.json"]["source"], str(path))
        self.assertIn("IMAGE_LABELS.json", json.loads(members["system/facts.json"])["image_labels_note"])

    def test_older_image_without_the_file_records_absence(self):
        rec, _, _ = self._labels_bundle(self.results / "missing" / "IMAGE_LABELS.json")
        self.assertFalse(rec["present"])
        self.assertNotIn("labels", rec)
        self.assertIn("docker inspect", rec["reason"])

    def test_bad_file_is_reported_not_trusted(self):
        path = self.results / "IMAGE_LABELS.json"
        for body in ("{not json", json.dumps({"labels": {"a": 1}}), json.dumps(["x"]), "\xff"):
            path.write_text(body, encoding="latin-1")
            rec = fa._image_labels_record(path)
            self.assertTrue(rec["present"])
            self.assertIn("unreadable", rec["error"])
            self.assertNotIn("labels", rec)
        path.write_text(" " * (fa.IMAGE_LABELS_MAX_BYTES + 1))
        self.assertIn("larger than", fa._image_labels_record(path)["error"])

    def test_labels_still_pass_through_the_scrubber(self):
        path = self.results / "IMAGE_LABELS.json"
        path.write_text(json.dumps({"labels": {"com.example.note": f"built on dgx-alice-07 {PLANTED_FIXTURES['hf']}"}}))
        rec, members, listed = self._labels_bundle(path)
        self.assertNotIn(PLANTED_FIXTURES["hf"], members["system/image-labels.json"])
        self.assertNotIn("dgx-alice-07", members["system/image-labels.json"])
        self.assertGreater(listed["system/image-labels.json"]["redactions"].get("api_key", 0), 0)
