# SPDX-License-Identifier: AGPL-3.0-only
"""`flux settings` / `flux set`: the launcher client of /v1/flux/settings."""

import http.server
import importlib.machinery
import importlib.util
import io
import json
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import TestCase, mock

SCRIPT = Path(__file__).resolve().parents[1] / "flux-autodetect"


def _load_flux_autodetect():
    loader = importlib.machinery.SourceFileLoader("flux_autodetect_settings", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


fa = _load_flux_autodetect()

DOC = {
    "object": "flux.settings",
    "model": "fixture",
    "settings": {"thinking.default": {"value": False, "source": "default"}},
    "restart_only": ["max_seq_len"],
    "persist_file": None,
}


class _Handler(http.server.BaseHTTPRequestHandler):
    seen = []
    tokens = []

    def log_message(self, *args):
        pass

    def _reply(self, status, doc):
        body = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _Handler.seen.append(("GET", self.path, self.headers.get("Authorization"), None))
        _Handler.tokens.append(self.headers.get("X-Flux-Admin-Token"))
        self._reply(200, DOC)

    def do_PATCH(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append(("PATCH", self.path, self.headers.get("Authorization"), body))
        _Handler.tokens.append(self.headers.get("X-Flux-Admin-Token"))
        if "max_seq_len" in body:
            self._reply(409, {"error": {"code": "restart_required", "message": "needs a restart"}})
            return
        changed = [{"key": k, "old": None, "new": v} for k, v in body.items()]
        self._reply(200, {**DOC, "changed": changed})


class ParseSetArgsTest(TestCase):
    def test_values_are_json_with_string_fallback(self):
        body = fa.parse_set_args([
            "thinking.default=true",
            "sampling.tools.temperature=0.6",
            "max_tokens.default=null",
            "label=hello world",
        ])
        self.assertEqual(body, {
            "thinking.default": True,
            "sampling.tools.temperature": 0.6,
            "max_tokens.default": None,
            "label": "hello world",
        })

    def test_reset_and_flags(self):
        body = fa.parse_set_args(["--reset", "thinking.max_budget", "--api-key", "k", "--url", "u", "a=1"])
        self.assertEqual(body, {"thinking.max_budget": None, "a": 1})

    def test_bad_arguments_are_errors(self):
        for bad in ([], ["novalue"], ["=1"], ["--reset"]):
            with self.assertRaises(ValueError, msg=bad):
                fa.parse_set_args(bad)


class LiveServerTest(TestCase):
    def setUp(self):
        _Handler.seen = []
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.env = {"FLUX_PORT": str(self.server.server_address[1])}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)

    def _run(self, fn, args, env=None):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict("os.environ", {**self.env, **(env or {})}, clear=True), \
                redirect_stdout(out), redirect_stderr(err):
            rc = fn(args)
        return rc, out.getvalue(), err.getvalue()

    def test_settings_shows_values_and_sources(self):
        rc, out, _ = self._run(fa.cmd_settings, [])
        self.assertEqual(rc, 0)
        self.assertIn("thinking.default", out)
        self.assertIn("[default]", out)
        self.assertEqual(_Handler.seen[0][:2], ("GET", "/v1/flux/settings"))

    def test_set_sends_one_patch_with_the_bearer_key(self):
        rc, out, _ = self._run(fa.cmd_set, ["thinking.default=true"], {"FLUX_API_KEY": "s3cret"})
        self.assertEqual(rc, 0)
        method, path, auth, body = _Handler.seen[0]
        self.assertEqual((method, path, auth, body), ("PATCH", "/v1/flux/settings", "Bearer s3cret", {"thinking.default": True}))
        self.assertIn("thinking.default: null -> true", out)
        self.assertNotIn("s3cret", out)

    def test_a_refusal_is_reported_and_fails(self):
        rc, _, err = self._run(fa.cmd_set, ["max_seq_len=4"])
        self.assertEqual(rc, 1)
        self.assertIn("restart_required", err)


class ClientErrorTest(TestCase):
    def _run(self, fn, args, env):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict("os.environ", env, clear=True), redirect_stdout(out), redirect_stderr(err):
            rc = fn(args)
        return rc, out.getvalue(), err.getvalue()

    def test_unreachable_server_is_one_clean_error_line(self):
        # Port 9 (discard) on loopback is closed in the test environment.
        for fn, args in ((fa.cmd_settings, []), (fa.cmd_set, ["a=1"])):
            rc, out, err = self._run(fn, args + ["--url", "http://127.0.0.1:9"], {})
            self.assertEqual(rc, 1)
            self.assertTrue(err.startswith("[flux] ERROR: cannot reach"), err)
            self.assertNotIn("Traceback", err)

    def test_a_non_json_answer_is_one_clean_error_line(self):
        class Html(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = b"<html>502 Bad Gateway</html>"
                self.send_response(502)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = http.server.HTTPServer(("127.0.0.1", 0), Html)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            rc, _, err = self._run(fa.cmd_settings, [], {"FLUX_PORT": str(server.server_address[1])})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5.0)
        self.assertEqual(rc, 1)
        self.assertIn("[flux] ERROR:", err)
        self.assertIn("without JSON", err)


class UrlAndKeyTest(TestCase):
    def test_url_flag_and_env_both_name_the_server_base(self):
        with mock.patch.dict("os.environ", {"FLUX_SETTINGS_URL": "http://h:1/"}, clear=True):
            self.assertEqual(fa._settings_url([]), "http://h:1/v1/flux/settings")
            self.assertEqual(fa._settings_url(["--url", "http://g:2"]), "http://g:2/v1/flux/settings")
            self.assertEqual(
                fa._settings_url(["--url", "http://g:2/v1/flux/settings/"]),
                "http://g:2/v1/flux/settings",
            )

    def test_an_api_key_on_argv_warns(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(fa._settings_api_key(["--api-key", "s3cret"]), "s3cret")
        self.assertIn("visible in `ps`", err.getvalue())
        self.assertNotIn("s3cret", err.getvalue())


class AdminTokenTest(TestCase):
    def setUp(self):
        _Handler.seen = []
        _Handler.tokens = []
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)

    def test_set_sends_the_local_admin_token_and_never_prints_it(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            token_file = Path(d) / "admin.token"
            token_file.write_text("feedface" * 8 + "\n")
            out, err = io.StringIO(), io.StringIO()
            env = {"FLUX_PORT": self.port, "FLUX_ADMIN_TOKEN_FILE": str(token_file)}
            with mock.patch.dict("os.environ", env, clear=True), redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(fa.cmd_set(["thinking.default=true"]), 0)
                self.assertEqual(fa.cmd_settings([]), 0)
        self.assertEqual(_Handler.tokens[0], "feedface" * 8, "PATCH carries the token")
        self.assertIsNone(_Handler.tokens[1], "GET does not")
        self.assertNotIn("feedface", out.getvalue() + err.getvalue())

    def test_get_does_not_send_the_admin_token(self):
        # GET needs no secret; the token only travels with a change.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            token_file = Path(d) / "admin.token"
            token_file.write_text("cafe" * 16)
            env = {"FLUX_PORT": self.port, "FLUX_ADMIN_TOKEN_FILE": str(token_file)}
            with mock.patch.dict("os.environ", env, clear=True), redirect_stdout(io.StringIO()):
                fa.cmd_settings([])
        self.assertEqual(_Handler.seen[0][0], "GET")
        self.assertIsNone(_Handler.tokens[0])

    def test_help_lists_keys_and_the_spontaneous_budget_caveat(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(fa.cmd_set(["--help"]), 0)
        text = out.getvalue()
        for needle in ("thinking.default", "sampling.<preset>.<field>", "SPONTANEOUS-thinking", "docker exec"):
            self.assertIn(needle, text)


class AdminTokenScopeTest(TestCase):
    def test_the_token_is_read_only_for_a_loopback_url_and_per_port(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            flux = Path(d) / "flux"
            flux.mkdir()
            (flux / "admin.8899.token").write_text("a" * 64)
            (flux / "admin.9000.token").write_text("b" * 64)
            env = {"XDG_RUNTIME_DIR": d, "FLUX_ADMIN_TOKEN_FILE": ""}
            orig_paths = fa._admin_token_paths
            with mock.patch.dict("os.environ", env, clear=True), mock.patch.object(
                fa,
                "_admin_token_paths",
                side_effect=lambda port: [
                    p for p in orig_paths(port) if not str(p).startswith("/run/flux")
                ],
            ):
                self.assertEqual(fa.read_admin_token("http://127.0.0.1:8899/v1/flux/settings"), "a" * 64)
                self.assertEqual(fa.read_admin_token("http://localhost:9000/v1/flux/settings"), "b" * 64)
                self.assertEqual(fa.read_admin_token("http://[::1]:8899/v1/flux/settings"), "a" * 64)
                # Never sent off this machine, whatever the port.
                for remote in ("http://10.0.0.5:8899/v1/flux/settings",
                               "http://gpu-box:8899/v1/flux/settings",
                               "http://127.0.0.1.attacker.example:8899/v1/flux/settings"):
                    self.assertIsNone(fa.read_admin_token(remote), remote)
                # No file for that port: nothing.
                self.assertIsNone(fa.read_admin_token("http://127.0.0.1:7000/v1/flux/settings"))

    def test_a_remote_url_never_carries_the_token(self):
        _Handler.seen = []
        _Handler.tokens = []
        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        import tempfile
        try:
            with tempfile.TemporaryDirectory() as d:
                token_file = Path(d) / "t"
                token_file.write_text("c" * 64)
                env = {"FLUX_ADMIN_TOKEN_FILE": str(token_file)}
                with mock.patch.dict("os.environ", env, clear=True), redirect_stdout(io.StringIO()):
                    # Reach the same test server under a non-loopback name.
                    fa.cmd_set(["a=1", "--url", f"http://127.0.0.1:{port}"])
                    with mock.patch.object(fa, "_url_host_port", return_value=("10.0.0.5", port)):
                        fa.cmd_set(["a=1", "--url", f"http://127.0.0.1:{port}"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5.0)
        self.assertEqual(_Handler.tokens, ["c" * 64, None])
