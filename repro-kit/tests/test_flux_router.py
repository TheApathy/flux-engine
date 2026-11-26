# SPDX-License-Identifier: AGPL-3.0-only
"""flux-router: multi-Spark DP router for N single-Spark flux-servers (TRACKER K2)."""

import contextlib
import csv
import http.client
import http.server
import importlib.machinery
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest import TestCase, mock, skipUnless

SCRIPT = Path(__file__).resolve().parents[1] / "router" / "flux-router"


def _load():
    loader = importlib.machinery.SourceFileLoader("flux_router", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


fr = _load()
B = fr.Backend


def _docker_haproxy_ok():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", "haproxy:2.8"], capture_output=True).returncode == 0


HAVE_HAPROXY = _docker_haproxy_ok()


def _write_router_dir(d: Path, cfg: str) -> None:
    """haproxy.cfg + flux_state.lua + 503.http, readable by the image's non-root haproxy user."""
    d.chmod(0o755)
    (d / "haproxy.cfg").write_text(cfg)
    shutil.copyfile(fr.LUA_FILE, d / "flux_state.lua")
    (d / "503.http").write_bytes(fr.render_errorfile_503())
    for f in d.iterdir():
        f.chmod(0o644)


class LuaScanner(TestCase):
    @skipUnless(HAVE_HAPROXY, "needs docker image haproxy:2.8")
    def test_flux_state_cases_under_haproxy_lua53(self):
        src = (SCRIPT.parents[1] / "tests/lua/test_flux_state.lua").read_text()
        src = (src.replace('dofile("router/flux_state.lua")', 'dofile("/w/flux_state.lua")')
                  .replace("os.exit(1)", 'error("scanner case failed under haproxy lua")'))
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            shutil.copyfile(fr.LUA_FILE, dp / "flux_state.lua")
            (dp / "t.lua").write_text(src.replace('print(("flux_state.lua', 'core.Info(("flux_state.lua'))
            (dp / "c.cfg").write_text("global\n  lua-load /w/t.lua\ndefaults\n  mode http\n")
            dp.chmod(0o755)
            for f in dp.iterdir():
                f.chmod(0o644)
            r = subprocess.run(["docker", "run", "--rm", "-v", f"{d}:/w:ro", "haproxy:2.8",
                                "haproxy", "-c", "-f", "/w/c.cfg"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("cases OK", r.stdout + r.stderr)

    @skipUnless(shutil.which("lua5.4") or shutil.which("lua5.3"), "needs a lua interpreter")
    def test_flux_state_cases(self):
        lua = shutil.which("lua5.3") or shutil.which("lua5.4")
        r = subprocess.run([lua, "tests/lua/test_flux_state.lua"], cwd=SCRIPT.parents[1],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class ParseBackends(TestCase):
    def test_parses_host_port_and_cap(self):
        self.assertEqual(fr.parse_backends("spark-a:8888=32, 10.0.0.2:9000=16"),
                         [B("spark-a", 8888, 32), B("10.0.0.2", 9000, 16)])

    def test_cap_may_be_omitted_for_discovery(self):
        self.assertEqual(fr.parse_backends("h:1"), [B("h", 1, None)])

    def test_bracketed_ipv6(self):
        got = fr.parse_backends("[::1]:9000=8")
        self.assertEqual(got, [B("::1", 9000, 8)])
        self.assertEqual(got[0].addr, "[::1]:9000")

    def test_rejects_bad_entries(self):
        for bad in ("", "h", "h:x", "h:1=0", "h:1=-3", "h:0", "h:70000", "h:1=129", "h:1=+4", "::1:9000",
                    "[zz]:1", "-h:1", "h_x:1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                fr.parse_backends(bad)

    def test_duplicates_are_canonical(self):
        for dup in ("SPARK-A:1,spark-a:1", "[::1]:9,[0:0::1]:9", "10.0.0.1:5,10.0.0.1:5=4"):
            with self.subTest(dup=dup), self.assertRaises(ValueError):
                fr.parse_backends(dup)

    def test_config_injection_is_refused(self):
        for evil in ("spark-a\n:8888=8", "spark-a\r:8888", "h:1 disabled\nglobal\n  lua-load /etc/passwd\n#:2", "h\n  stats enable:1",
                     "h 1:2", "h\t:1", "h;x:1", "h'x:1"):
            with self.subTest(evil=evil), self.assertRaises(ValueError):
                fr.parse_backends(evil)


class RenderConfig(TestCase):
    def setUp(self):
        self.cfg = fr.render_config([B("a", 8888, 32), B("::1", 8889, 16)], bind_port=9999)

    def test_leastconn_with_per_backend_maxconn(self):
        self.assertIn("balance leastconn", self.cfg)
        self.assertRegex(self.cfg, r"server b0 a:8888 .*maxconn 32")
        self.assertRegex(self.cfg, r"server b1 \[::1\]:8889 .*maxconn 16")

    def test_timeouts(self):
        for t in ("timeout check 3s", "timeout http-request 30s", "timeout http-keep-alive 10s",
                  "timeout queue 120s"):
            self.assertIn(t, self.cfg)

    def test_readiness_check_uses_health(self):
        self.assertIn("option httpchk GET /health", self.cfg)
        self.assertIn("http-check expect status 200", self.cfg)

    def test_stateful_routes_return_501_but_stateless_responses_pass(self):
        self.assertIn("status 501", self.cfg)
        for prefix in fr.STATEFUL_PREFIXES:
            self.assertIn(prefix, self.cfg)
        import re
        rx = re.compile(fr.STATEFUL_BY_ID)
        self.assertTrue(rx.match("/v1/responses/resp_1/cancel"))
        self.assertTrue(rx.match("/v1/chat/completions/chatcmpl-1"))
        self.assertFalse(rx.match("/v1/responses"))
        self.assertFalse(rx.match("/v1/chat/completions"))

    def test_only_connection_failures_are_retried(self):
        self.assertIn("retry-on conn-failure", self.cfg)
        self.assertNotIn("all-retryable-errors", self.cfg)

    def test_needs_caps(self):
        with self.assertRaises(ValueError):
            fr.render_config([B("a", 1, None)])

    def test_cors_options_excluded_and_responses_body_check_post_only(self):
        self.assertIn("acl is_get method GET", self.cfg)
        self.assertIn("acl is_post method POST", self.cfg)
        self.assertIn("acl is_options method OPTIONS", self.cfg)
        self.assertIn("if !is_options stateful or !is_options by_id", self.cfg)
        self.assertIn("if is_post responses responses_too_big", self.cfg)
        self.assertIn("lua.flux_state_ref if is_post responses", self.cfg)
        self.assertIn("if is_post responses body_bad", self.cfg)
        self.assertIn("if is_post responses state_ref", self.cfg)

    def test_settings_mutations_and_disagg_blocked_in_config(self):
        self.assertIn("acl disagg path /v1/disagg", self.cfg)
        self.assertIn("acl disagg path_beg /v1/disagg/", self.cfg)
        self.assertIn("acl flux_settings path /v1/flux/settings", self.cfg)
        self.assertIn("acl flux_settings path_beg /v1/flux/settings/", self.cfg)
        self.assertIn("if !is_options disagg", self.cfg)
        self.assertIn("if !is_options !is_get flux_settings", self.cfg)

    def test_local_health_monitor_uri(self):
        self.assertIn("monitor-uri /health", self.cfg)
        self.assertIn("monitor fail if { nbsrv(sparks) lt 1 }", self.cfg)

    def test_errorfile_503_format_and_wiring(self):
        self.assertIn("errorfile 503 /etc/flux-router/503.http", self.cfg)
        raw = fr.render_errorfile_503()
        head, sep, body = raw.partition(b"\r\n\r\n")
        self.assertEqual(sep, b"\r\n\r\n")
        lines = head.decode("ascii").split("\r\n")
        self.assertEqual(lines[0], "HTTP/1.1 503 Service Unavailable")
        hdrs = {k.lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
        self.assertEqual(hdrs.get("content-type"), "application/json")
        self.assertEqual(hdrs.get("retry-after"), "5")
        self.assertEqual(int(hdrs.get("content-length", -1)), len(body))
        payload = json.loads(body.decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "server_error")
        self.assertTrue(payload["error"]["message"])
        for evil in ("/tmp/503.http\nstats enable", "/tmp/a b.http", "relative/503.http"):
            with self.subTest(evil=evil), self.assertRaises(ValueError):
                fr.render_config([B("a", 1, 4)], errorfile_503_path=evil)

    def test_stats_port_validation(self):
        for bad in (0, -1, 65536, 70000, True, False):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                fr.render_config([B("a", 1, 4)], stats_port=bad)
        for good in (1, 8404, 65535):
            cfg = fr.render_config([B("a", 1, 4)], stats_port=good)
            self.assertIn(f"bind 127.0.0.1:{good}", cfg)

    @skipUnless(HAVE_HAPROXY, "needs docker image haproxy:2.8")
    def test_haproxy_accepts_rendered_config(self):
        with tempfile.TemporaryDirectory() as d:
            _write_router_dir(Path(d), fr.render_config([B("127.0.0.1", 9000, 32), B("::1", 9001, 8)],
                                                         stats_port=9002))
            r = subprocess.run(["docker", "run", "--rm", "-v", f"{d}:/etc/flux-router:ro", "haproxy:2.8",
                                "haproxy", "-c", "-f", "/etc/flux-router/haproxy.cfg"],
                               capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class RuntimeAndCli(TestCase):
    def test_cli_rejects_invalid_stats_port(self):
        for bad in ("0", "-1", "65536"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                fr.main(["--backends", "a:8888=4", "--skip-backend-checks", "--render-only",
                         "--stats-port", bad])

    def test_cli_accepts_valid_stats_port(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = fr.main(["--backends", "a:8888=4", "--skip-backend-checks", "--render-only",
                          "--stats-port", "8404"])
        self.assertEqual(rc, 0)
        self.assertIn("bind 127.0.0.1:8404", buf.getvalue())

    def test_runtime_dir_prefers_default_then_xdg_then_mkdtemp(self):
        with tempfile.TemporaryDirectory() as d_run, tempfile.TemporaryDirectory() as d_xdg:
            p_run, p_xdg = Path(d_run), Path(d_xdg)
            with mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(p_xdg)}, clear=False):
                # 1. Writable /run/flux-router wins over XDG_RUNTIME_DIR
                self.assertEqual(fr._resolve_run_dir(8888, create=True, default_dir=p_run), p_run)
                # 2. Missing /run/flux-router falls back to $XDG_RUNTIME_DIR/flux-router (mode 0700)
                missing = p_run / "does-not-exist"
                got_xdg = fr._resolve_run_dir(8888, create=True, default_dir=missing)
                self.assertEqual(got_xdg, p_xdg / "flux-router")
                self.assertTrue(got_xdg.is_dir())
                self.assertEqual(got_xdg.stat().st_mode & 0o777, 0o700)
            # 3. Neither default_dir nor XDG_RUNTIME_DIR -> unique private mkdtemp dir
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("XDG_RUNTIME_DIR", None)
                d1 = fr._resolve_run_dir(8899, create=True, default_dir=p_run / "missing")
                d2 = fr._resolve_run_dir(8899, create=True, default_dir=p_run / "missing")
                try:
                    self.assertNotEqual(d1, d2)
                    for d in (d1, d2):
                        self.assertTrue(d.name.startswith("flux-router-8899-"))
                        self.assertNotEqual(d, Path(f"/tmp/flux-router-{os.getuid()}"))
                        self.assertEqual(d.stat().st_mode & 0o777, 0o700)
                finally:
                    shutil.rmtree(d1, ignore_errors=True)
                    shutil.rmtree(d2, ignore_errors=True)

    def test_per_port_files_avoid_clobbering_across_instances(self):
        with tempfile.TemporaryDirectory() as d_run:
            p_run = Path(d_run)
            exec_calls = []

            def fake_execv(exe, args):
                exec_calls.append((exe, args))

            with mock.patch.object(fr, "DEFAULT_RUN_DIR", p_run), \
                    mock.patch.object(fr.shutil, "which", return_value="/usr/sbin/haproxy"), \
                    mock.patch.object(fr.os, "execv", side_effect=fake_execv):
                for port in (8888, 8889):
                    with contextlib.redirect_stderr(io.StringIO()):
                        fr.main(["--backends", "a:9000=4", "--skip-backend-checks",
                                 "--port", str(port)])
            self.assertEqual(len(exec_calls), 2)
            for port in (8888, 8889):
                lua_f = p_run / f"flux_state-{port}.lua"
                err_f = p_run / f"503-{port}.http"
                cfg_f = p_run / f"flux-router-{port}.cfg"
                self.assertTrue(lua_f.is_file())
                self.assertTrue(err_f.is_file())
                self.assertTrue(cfg_f.is_file())
                cfg_txt = cfg_f.read_text()
                self.assertIn(f"lua-load {lua_f}", cfg_txt)
                self.assertIn(f"errorfile 503 {err_f}", cfg_txt)
                self.assertNotIn(f"/tmp/flux-router-{os.getuid()}", cfg_txt)
                self.assertEqual(err_f.read_bytes(), fr.render_errorfile_503())

    def test_render_only_uses_per_port_filenames_without_leaking_mkdtemp(self):
        with tempfile.TemporaryDirectory() as empty_tmp:
            buf = io.StringIO()
            with mock.patch.object(fr, "DEFAULT_RUN_DIR", Path(empty_tmp) / "missing"), \
                    mock.patch.dict(os.environ, {"TMPDIR": empty_tmp}, clear=False), \
                    contextlib.redirect_stdout(buf):
                os.environ.pop("XDG_RUNTIME_DIR", None)
                rc = fr.main(["--backends", "a:9000=4", "--skip-backend-checks",
                              "--port", "8891", "--render-only"])
            self.assertEqual(rc, 0)
            self.assertIn("flux_state-8891.lua", buf.getvalue())
            self.assertIn("503-8891.http", buf.getvalue())
            self.assertEqual(list(Path(empty_tmp).iterdir()), [])

    def test_dockerfile_runs_as_non_root_haproxy_with_owned_run_dir(self):
        dockerfile = (SCRIPT.parent / "Dockerfile").read_text()
        self.assertIn("chown -R haproxy:haproxy /run/flux-router", dockerfile)
        idx_root = dockerfile.find("USER root")
        idx_haproxy = dockerfile.find("USER haproxy")
        idx_entry = dockerfile.find("ENTRYPOINT")
        self.assertNotEqual(idx_root, -1)
        self.assertNotEqual(idx_haproxy, -1)
        self.assertNotEqual(idx_entry, -1)
        self.assertLess(idx_root, idx_haproxy)
        self.assertLess(idx_haproxy, idx_entry)


class DiscoveryAuth(TestCase):
    def _capture(self, env):
        seen = {}

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            seen["auth"] = req.get_header("Authorization")
            return Resp(b'{"data": [{"id": "m"}]}')
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(fr.urllib.request, "urlopen", fake_urlopen):
            for k in ("FLUX_ROUTER_DISCOVERY_KEY", "FLUX_ROUTER_DISCOVERY_KEY_FILE"):
                if k not in env:
                    os.environ.pop(k, None)
            got = fr._get_json(B("h", 1, 4), "/v1/models")
        return got, seen.get("auth")

    def test_key_from_env_is_sent_on_discovery(self):
        got, auth = self._capture({"FLUX_ROUTER_DISCOVERY_KEY": "sk-test"})
        self.assertEqual(got["data"][0]["id"], "m")
        self.assertEqual(auth, "Bearer sk-test")

    def test_key_from_file(self):
        with tempfile.NamedTemporaryFile("w") as f:
            f.write("sk-file\n")
            f.flush()
            _, auth = self._capture({"FLUX_ROUTER_DISCOVERY_KEY_FILE": f.name})
        self.assertEqual(auth, "Bearer sk-file")

    def test_no_key_no_header(self):
        self.assertIsNone(self._capture({})[1])

    def test_key_never_rendered(self):
        with mock.patch.dict(os.environ, {"FLUX_ROUTER_DISCOVERY_KEY": "sk-secret"}):
            self.assertNotIn("sk-secret", fr.render_config([B("a", 1, 4)]))


class StartupGates(TestCase):
    def test_same_model_passes(self):
        fr.check_same_model({"a": "qwen38", "b": "qwen38"})

    def test_mismatch_and_unreachable_are_refused(self):
        for ids in ({"a": "qwen38", "b": "glm53"}, {"a": "qwen38", "b": None}):
            with self.subTest(ids=ids), self.assertRaises(SystemExit):
                fr.check_same_model(ids)

    def test_malformed_published_capacity_is_refused(self):
        # Regression: JSON true passed as an int cap; malformed values were ignored beside an explicit cap
        for serving in ({"concurrency": True}, {"concurrency": "32"}, {"concurrency": [32]}, {"concurrency": 0},
                        {"concurrency": 129}, {"concurrency": {"n": 8}}):
            for cap in (None, 8):
                with self.subTest(serving=serving, cap=cap), self.assertRaises(SystemExit):
                    fr.resolve_caps([B("a", 1, cap)], fetch=lambda b, p, s=serving: {"serving": s})
        for bad in ([], "x", {"serving": []}):
            with self.subTest(modes=bad), self.assertRaises(SystemExit):
                fr.resolve_caps([B("a", 1, 8)], fetch=lambda b, p, m=bad: m)

    def test_absent_metadata_keeps_explicit_cap(self):
        self.assertEqual(fr.resolve_caps([B("a", 1, 8)], fetch=lambda b, p: None), [B("a", 1, 8)])
        self.assertEqual(fr.resolve_caps([B("a", 1, 8)], fetch=lambda b, p: {"modes": []}), [B("a", 1, 8)])

    def test_concurrency_preferred_over_ingress_queue(self):
        got = fr.resolve_caps([B("a", 1, None)],
                              fetch=lambda b, p: {"serving": {"concurrency": 16, "ingress_queue": 32}})
        self.assertEqual(got, [B("a", 1, 16)])

    def test_ingress_queue_alone_is_not_capacity(self):
        for serving in ({"ingress_queue": 32}, {"max_num_seqs": 32}):
            with self.subTest(serving=serving), self.assertRaises(SystemExit):
                fr.resolve_caps([B("a", 1, None)], fetch=lambda b, p, s=serving: {"serving": s})
        self.assertEqual(fr.resolve_caps([B("a", 1, 8)], fetch=lambda b, p: {"serving": {"ingress_queue": 32}}),
                         [B("a", 1, 8)])

    def test_cap_discovered_from_flux_modes(self):
        got = fr.resolve_caps([B("a", 1, None)], fetch=lambda b, p: {"serving": {"concurrency": 16}})
        self.assertEqual(got, [B("a", 1, 16)])

    def test_missing_serving_block_needs_explicit_cap(self):
        with self.assertRaises(SystemExit):
            fr.resolve_caps([B("a", 1, None)], fetch=lambda b, p: {"modes": []})
        self.assertEqual(fr.resolve_caps([B("a", 1, 8)], fetch=lambda b, p: {}), [B("a", 1, 8)])

    def test_explicit_cap_must_match_published(self):
        with self.assertRaises(SystemExit):
            fr.resolve_caps([B("a", 1, 32)], fetch=lambda b, p: {"serving": {"concurrency": 16}})

    def test_published_cap_out_of_range_refused(self):
        with self.assertRaises(SystemExit):
            fr.resolve_caps([B("a", 1, None)], fetch=lambda b, p: {"serving": {"concurrency": 129}})


# ---------------------------------------------------------------- integration

def _free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockFlux:
    """flux-server stand-in. Strict: 429 the moment inflight would exceed cap
    (flux-server itself has max(4C, 32) of slack, so this is harsher than reality)."""

    def __init__(self, name, cap, delay=0.4):
        self.name, self.cap, self.delay = name, cap, delay
        self.inflight = self.peak = self.served = self.rejected = self.die_hits = self.mutations = 0
        self.hung = threading.Event()
        self._stop = threading.Event()
        self._workers = []
        self._sockets = []
        self.lock = threading.Lock()
        m = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_OPTIONS(self):
                if m.hung.is_set():
                    m._stop.wait(3600)
                    self.close_connection = True
                    return
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _mutate(self):
                if m.hung.is_set():
                    m._stop.wait(3600)
                    self.close_connection = True
                    return
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                with m.lock:
                    m.mutations += 1
                return self._json(200, {"mutated": m.name})

            do_PATCH = _mutate
            do_PUT = _mutate
            do_DELETE = _mutate

            def do_GET(self):
                if m.hung.is_set():
                    m._stop.wait(3600)
                    self.close_connection = True
                    return
                if self.path == "/health":
                    return self._json(200, {"status": "ok"})
                if self.path == "/v1/models":
                    return self._json(200, {"data": [{"id": "mock-model"}]})
                if self.path == "/v1/flux/settings":
                    return self._json(200, {"settings": {"backend": m.name}})
                if self.path == "/v1/disagg" or self.path.startswith("/v1/disagg/"):
                    return self._json(200, {"role": "unified"})
                return self._json(404, {})

            def _admit(self):
                with m.lock:
                    if m.inflight >= m.cap:
                        m.rejected += 1
                        return False
                    m.inflight += 1
                    m.peak = max(m.peak, m.inflight)
                    return True

            def _release(self):
                with m.lock:
                    m.inflight -= 1
                    m.served += 1

            def do_POST(self):
                if m.hung.is_set():
                    m._stop.wait(3600)
                    self.close_connection = True
                    return
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path == "/die":
                    with m.lock:
                        m.die_hits += 1
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(b"7\r\ndata: 0\r\n")
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                if not self._admit():
                    return self._json(429, {"error": "full"})
                if self.path.startswith("/stream"):
                    n = int(self.path.rsplit("/", 1)[-1]) if self.path.count("/") == 2 else 5
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Transfer-Encoding", "chunked")
                        self.end_headers()
                        for i in range(n):
                            chunk = f"data: {i}\n\n".encode()
                            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                            self.wfile.flush()
                            if m._stop.wait(0.5):
                                self.close_connection = True
                                return
                        self.wfile.write(b"0\r\n\r\n")
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        self.close_connection = True  # client went away: free the slot now
                    finally:
                        self._release()
                    return
                if m._stop.wait(m.delay):
                    self._release()
                    self.close_connection = True
                    return
                self._release()  # free the slot before replying: no slack for the router
                self._json(200, {"backend": m.name})

        class _Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def process_request_thread(srv, request, client_address):
                with m.lock:
                    m._sockets.append(request)
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    with m.lock:
                        try:
                            m._sockets.remove(request)
                        except ValueError:
                            pass

            def process_request(srv, request, client_address):
                t = threading.Thread(
                    target=srv.process_request_thread,
                    args=(request, client_address),
                    daemon=True,
                )
                with m.lock:
                    m._workers.append(t)
                t.start()

        self.port = _free_port()
        self.httpd = _Server(("127.0.0.1", self.port), H)
        self._serve_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._serve_thread.start()

    def close(self):
        self._stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()  # a dead server refuses connections
        with self.lock:
            sockets = list(self._sockets)
            workers = list(self._workers)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self._serve_thread.join(timeout=5.0)
        for t in workers:
            t.join(timeout=5.0)


@skipUnless(HAVE_HAPROXY, "needs docker image haproxy:2.8")
class RouterIntegration(TestCase):
    CAP = 4
    QUEUE_TIMEOUT = 120
    DELAY = 0.4

    def setUp(self):
        self.a, self.b = MockFlux("a", self.CAP, self.DELAY), MockFlux("b", self.CAP, self.DELAY)
        self.port, self.stats = _free_port(), _free_port()
        cfg = fr.render_config([B("127.0.0.1", m.port, self.CAP) for m in (self.a, self.b)],
                               bind_port=self.port, queue_timeout_s=self.QUEUE_TIMEOUT, stats_port=self.stats)
        self.rdir = tempfile.TemporaryDirectory()
        _write_router_dir(Path(self.rdir.name), cfg)
        self.proc = subprocess.Popen(
            ["docker", "run", "--rm", "--network", "host", "-v", f"{self.rdir.name}:/etc/flux-router:ro",
             "haproxy:2.8", "haproxy", "-f", "/etc/flux-router/haproxy.cfg"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(self._teardown)
        deadline = time.time() + 30
        while self._status("b0") != "UP" or self._status("b1") != "UP":
            if self.proc.poll() is not None:
                self.fail("haproxy exited: " + self.proc.stderr.read().decode(errors="replace"))
            if time.time() > deadline:
                self.fail("backends never came UP")
            time.sleep(0.2)

    def _teardown(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.rdir.cleanup()
        for m in (self.a, self.b):
            with contextlib.suppress(Exception):
                m.close()

    def _status(self, server):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.stats}/stats;csv", timeout=2) as r:
                for row in csv.DictReader(io.StringIO(r.read().decode().lstrip("# "))):
                    if row["pxname"] == "sparks" and row["svname"] == server:
                        return row["status"]
        except Exception:
            return None
        return None

    def _wait_status(self, server, want, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._status(server) == want:
                return
            time.sleep(0.2)
        self.fail(f"{server} never became {want} (now {self._status(server)})")

    def _post(self, path="/v1/chat/completions", timeout=60, body=b"{}"):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", body,
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                r.read()
                return r.status, r.headers.get("X-Flux-Backend")
        except urllib.error.HTTPError as e:
            return e.code, None
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            return type(e).__name__, None

    def _req(self, method, path, body=None, headers=None, timeout=30):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            r = conn.getresponse()
            data = r.read()
            return r.status, {k.lower(): v for k, v in r.getheaders()}, data
        finally:
            conn.close()

    def _burst(self, n, path="/v1/chat/completions"):
        results, lock = [], threading.Lock()

        def one():
            r = self._post(path)
            with lock:
                results.append(r)
        threads = [threading.Thread(target=one) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results


class RouterBehaviour(RouterIntegration):
    def test_overload_queues_at_router_and_holds_backend_cap_exactly(self):
        n = 3 * 2 * self.CAP
        results = self._burst(n)
        self.assertEqual([s for s, _ in results].count(200), n, results)
        self.assertEqual(self.a.rejected + self.b.rejected, 0, "a backend returned 429")
        self.assertEqual(self.a.peak, self.CAP)  # filled, never exceeded
        self.assertEqual(self.b.peak, self.CAP)
        self.assertEqual({b for _, b in results}, {"b0", "b1"})

    def test_keepalive_counts_requests_not_connections(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        for _ in range(6):
            conn.request("POST", "/v1/chat/completions", b"{}", {"Content-Type": "application/json"})
            r = conn.getresponse()
            r.read()
            self.assertEqual(r.status, 200)
        conn.close()
        self.assertEqual(self.a.served + self.b.served, 6)
        self.assertGreater(min(self.a.served, self.b.served), 0, "keep-alive pinned one backend")

    def test_stateful_routes_501_and_stateless_responses_routed(self):
        self.assertEqual(self._post("/v1/files")[0], 501)
        self.assertEqual(self._post("/v1/conversations")[0], 501)
        self.assertEqual(self._post("/v1/responses/resp_abc/cancel")[0], 501)
        self.assertEqual(self._post("/v1/responses")[0], 200)

    def test_responses_with_backend_state_references_are_501(self):
        for body in ({"input": "x", "previous_response_id": "resp_1"},
                     {"input": "x", "conversation": "conv_1"},
                     {"input": "x", "conversation": {"id": "conv_1"}}):
            with self.subTest(body=body):
                self.assertEqual(self._post("/v1/responses", body=json.dumps(body).encode())[0], 501)
        raw = b'{"input":"x",\n "previous_response_id"\t:\n"resp_1"}'
        self.assertEqual(self._post("/v1/responses", body=raw)[0], 501)
        for raw in (b'{"model":"m","input":"x","previous\\u005fresponse_id":"resp_1"}',
                    b'{"convers\\u0061tion":"conv_1"}'):
            with self.subTest(raw=raw):
                self.assertEqual(self._post("/v1/responses", body=raw)[0], 501)

    def test_nested_metadata_is_not_state(self):
        body = {"model": "m", "input": "hi", "metadata": {"conversation": "book club"}}
        self.assertEqual(self._post("/v1/responses", body=json.dumps(body).encode())[0], 200)

    def test_unparsable_responses_body_is_400(self):
        for raw in (b'{"input":"x"', b'not json', b'{"input":"x"} {"previous_response_id":"r"}'):
            with self.subTest(raw=raw):
                self.assertEqual(self._post("/v1/responses", body=raw)[0], 400)

    def test_chunked_responses_body_is_inspected(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        parts = [b'{"input":"x",', b'"previous_response_id":', b'"resp_1"}']
        conn.putrequest("POST", "/v1/responses")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.putheader("Content-Type", "application/json")
        conn.endheaders()
        for p in parts:
            conn.send(f"{len(p):x}\r\n".encode() + p + b"\r\n")
            time.sleep(0.1)
        conn.send(b"0\r\n\r\n")
        r = conn.getresponse()
        r.read()
        conn.close()
        self.assertEqual(r.status, 501)

    def test_body_exactly_at_limit_is_inspected_not_413(self):
        pad = fr.RESPONSES_BODY_MAX - len(b'{"input":""}')
        body = b'{"input":"' + b"x" * pad + b'"}'
        self.assertEqual(len(body), fr.RESPONSES_BODY_MAX)
        self.assertEqual(self._post("/v1/responses", body=body)[0], 200)

    def test_stateless_responses_pass(self):
        for body in ({"input": "x"}, {"input": "x", "previous_response_id": None},
                     {"input": "x", "conversation": None, "store": False}):
            with self.subTest(body=body):
                self.assertEqual(self._post("/v1/responses", body=json.dumps(body).encode())[0], 200)
        # the same key text inside a chat prompt is not state
        chat = json.dumps({"messages": [{"role": "user", "content": '"previous_response_id": "resp_1"'}]})
        self.assertEqual(self._post("/v1/chat/completions", body=chat.encode())[0], 200)

    def test_oversized_responses_body_is_413_but_chat_is_not(self):
        big = json.dumps({"input": "x" * (fr.RESPONSES_BODY_MAX + 10)}).encode()
        self.assertEqual(self._post("/v1/responses", body=big)[0], 413)
        self.assertEqual(self._post("/v1/chat/completions", body=big)[0], 200)

    def test_sse_is_not_buffered(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/stream", b"{}",
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as r:
            first = r.readline()
            t_first = time.perf_counter() - t0
            rest = r.read()
        self.assertEqual(first, b"data: 0\n")
        self.assertIn(b"data: 4", rest)
        self.assertLess(t_first, 1.5, "first event held back: buffered stream takes about 2.5 s")
        self.assertGreater(time.perf_counter() - t0, 2.0)

    def test_client_cancel_mid_stream_frees_the_slot(self):
        socks = []
        for _ in range(2 * self.CAP):  # fill every slot with a 20 s stream
            s = socket.create_connection(("127.0.0.1", self.port))
            s.sendall(b"POST /stream/40 HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}")
            s.recv(64)
            socks.append(s)
        self.assertEqual(self.a.inflight + self.b.inflight, 2 * self.CAP)
        for s in socks:
            s.close()
        # the cancellation must reach the backends (it frees their decode slots);
        # flux-server notices at its next token, the mock at its next 0.5 s chunk
        deadline = time.time() + 2.0
        while self.a.inflight + self.b.inflight and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.a.inflight + self.b.inflight, 0, "cancel never reached the backend")
        t0 = time.perf_counter()
        code, _ = self._post(timeout=10)
        self.assertEqual(code, 200)
        self.assertLess(time.perf_counter() - t0, 3.0, "cancelled streams kept their slots")

    def test_no_retry_after_response_bytes(self):
        code, _ = self._post("/die", timeout=10)
        self.assertNotEqual(code, 200)
        time.sleep(0.5)
        self.assertEqual(self.a.die_hits + self.b.die_hits, 1, "request was replayed after bytes were sent")

    def test_dead_backend_leaves_rotation(self):
        self.a.close()
        self._wait_status("b0", "DOWN")
        self.assertEqual([self._post()[0] for _ in range(6)], [200] * 6)

    def test_hung_backend_leaves_rotation(self):
        self.b.hung.set()
        self._wait_status("b1", "DOWN")
        before = self.a.served
        self.assertEqual([self._post(timeout=10)[0] for _ in range(6)], [200] * 6)
        self.assertEqual(self.a.served - before, 6)

    def test_cors_preflight_options_passes_through_on_all_routes(self):
        routes = (
            ("/v1/responses", "POST"),
            ("/v1/conversations", "POST"),
            ("/v1/files", "POST"),
            ("/v1/batches", "POST"),
            ("/v1/responses/resp_1", "GET"),
            ("/v1/responses/resp_1/cancel", "POST"),
            ("/v1/chat/completions/chatcmpl-1", "GET"),
            ("/v1/flux/settings", "PATCH"),
            ("/v1/disagg/status", "GET"),
            ("/v1/disagg/prefill", "POST"),
        )
        for path, req_method in routes:
            with self.subTest(path=path, req_method=req_method):
                status, hdrs, _ = self._req(
                    "OPTIONS", path,
                    headers={
                        "Origin": "https://app.example",
                        "Access-Control-Request-Method": req_method,
                    },
                )
                self.assertEqual(status, 204, (path, status, hdrs))
                self.assertEqual(hdrs.get("access-control-allow-origin"), "*")
                self.assertIn(hdrs.get("x-flux-backend"), ("b0", "b1"))

    def test_flux_settings_allows_get_and_blocks_mutations_with_501(self):
        status, hdrs, data = self._req("GET", "/v1/flux/settings")
        self.assertEqual(status, 200)
        self.assertIn(hdrs.get("x-flux-backend"), ("b0", "b1"))
        self.assertIn("settings", json.loads(data))

        for method, path in (
            ("PATCH", "/v1/flux/settings"),
            ("POST", "/v1/flux/settings"),
            ("PUT", "/v1/flux/settings"),
            ("DELETE", "/v1/flux/settings"),
            ("PATCH", "/v1/flux/settings/"),
        ):
            with self.subTest(method=method, path=path):
                status, hdrs, data = self._req(
                    method, path, body=b'{"temperature": 0.5}',
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(status, 501)
                self.assertEqual(hdrs.get("content-type"), "application/json")
                err = json.loads(data)["error"]
                self.assertEqual(err["type"], "not_supported")
                self.assertTrue(err["message"])
        self.assertEqual(self.a.mutations + self.b.mutations, 0)

    def test_disagg_routes_return_501(self):
        for method, path in (
            ("GET", "/v1/disagg"),
            ("GET", "/v1/disagg/status"),
            ("POST", "/v1/disagg/prefill"),
            ("POST", "/v1/disagg/decode"),
            ("POST", "/v1/disagg/generate"),
        ):
            with self.subTest(method=method, path=path):
                body = b"{}" if method == "POST" else None
                hdr_in = {"Content-Type": "application/json"} if body else {}
                status, hdrs, data = self._req(method, path, body=body, headers=hdr_in)
                self.assertEqual(status, 501)
                self.assertEqual(hdrs.get("content-type"), "application/json")
                err = json.loads(data)["error"]
                self.assertEqual(err["type"], "not_supported")

    def test_health_answers_locally_when_saturated_and_json_503_when_all_down(self):
        socks = []
        queued = []
        try:
            # Fill every backend slot with long-running streams
            for _ in range(2 * self.CAP):
                s = socket.create_connection(("127.0.0.1", self.port))
                s.sendall(b"POST /stream/40 HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}")
                s.recv(64)
                socks.append(s)
            self.assertEqual(self.a.inflight + self.b.inflight, 2 * self.CAP)
            # Also push extra requests into HAProxy's backend queue
            for _ in range(2):
                qs = socket.create_connection(("127.0.0.1", self.port))
                qs.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}")
                queued.append(qs)
            time.sleep(0.1)
            # GET /health must respond locally and immediately despite 100% GPU saturation
            t0 = time.perf_counter()
            status, _, _ = self._req("GET", "/health", timeout=5)
            elapsed = time.perf_counter() - t0
            self.assertEqual(status, 200)
            self.assertLess(elapsed, 1.0, f"/health queued behind saturated slots ({elapsed:.2f}s)")
        finally:
            for s in queued + socks:
                with contextlib.suppress(Exception):
                    s.close()

        # Take all backends down -> /health and normal routes both return JSON 503 + Retry-After: 5
        self.a.close()
        self.b.close()
        self._wait_status("b0", "DOWN")
        self._wait_status("b1", "DOWN")
        for method, path, body in (
            ("GET", "/health", None),
            ("POST", "/v1/chat/completions", b"{}"),
        ):
            with self.subTest(method=method, path=path):
                hdr_in = {"Content-Type": "application/json"} if body else {}
                status, hdrs, data = self._req(method, path, body=body, headers=hdr_in, timeout=10)
                self.assertEqual(status, 503)
                self.assertEqual(hdrs.get("content-type"), "application/json")
                self.assertEqual(hdrs.get("retry-after"), "5")
                err = json.loads(data)["error"]
                self.assertEqual(err["type"], "server_error")
                self.assertTrue(err["message"])


class RouterQueueTimeout(RouterIntegration):
    QUEUE_TIMEOUT = 1
    DELAY = 4.0

    def test_queue_timeout_returns_503_without_backend_429(self):
        results = self._burst(2 * self.CAP + 2)  # two more than total capacity
        codes = sorted(str(s) for s, _ in results)
        self.assertEqual(codes.count("200"), 2 * self.CAP, codes)
        self.assertEqual(codes.count("503"), 2, codes)
        self.assertEqual(self.a.rejected + self.b.rejected, 0)

    def test_queue_timeout_503_is_json_with_retry_after(self):
        results, lock = [], threading.Lock()

        def one():
            r = self._req("POST", "/v1/chat/completions", body=b"{}",
                          headers={"Content-Type": "application/json"}, timeout=30)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=one) for _ in range(2 * self.CAP + 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        timed_out = [r for r in results if r[0] == 503]
        self.assertEqual(len(timed_out), 2, [r[0] for r in results])
        for _, hdrs, data in timed_out:
            self.assertEqual(hdrs.get("content-type"), "application/json")
            self.assertEqual(hdrs.get("retry-after"), "5")
            err = json.loads(data)["error"]
            self.assertEqual(err["type"], "server_error")
            self.assertTrue(err["message"])
