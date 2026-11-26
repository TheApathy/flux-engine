# SPDX-License-Identifier: AGPL-3.0-only
"""Offline tests for ts_identity.py (GLM-5.3 P2). No GPU; a mock flux-server stands in."""
import importlib.util
import json
import os
import re
import socket
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from unittest import TestCase, mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ts_identity", HERE.parent / "ts_identity.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)
MOCK = str(HERE / "mock_flux_server.py")
CLI = ts.REPO / "crates/spark-server/src"


def free_port():
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def args(**kw):
    a = ts.argparse.Namespace(server=MOCK, model="/m", draft_model="/d", admitted=False, port=free_port(),
                              trials=1, req_dir=HERE.parent / "req", load_timeout=20, min_mem_gb=0,
                              watchdog_gb=0, control_arms=set(), pass_env=[])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class Plan(TestCase):
    def test_t_has_no_dflash_and_s_has_escape_and_timing(self):
        env_t, argv_t = ts.arm_plan("T", args(), 1)
        env_s, argv_s = ts.arm_plan("S", args(), 1)
        self.assertNotIn("--dflash", argv_t)
        self.assertIn("--dflash", argv_s)
        self.assertEqual(argv_s[argv_s.index("--draft-model") + 1], "/d")
        self.assertEqual(env_s["ATLAS_GLM53_UNVALIDATED_BRINGUP"], "1")
        self.assertEqual(env_s["ATLAS_GLM53_PHASE_TIMING"], "1")
        self.assertNotIn("ATLAS_GLM53_UNVALIDATED_BRINGUP", env_t)

    def test_admitted_drops_the_escape(self):
        env_s, _ = ts.arm_plan("S", args(admitted=True), 1)
        self.assertNotIn("ATLAS_GLM53_UNVALIDATED_BRINGUP", env_s)

    def test_s_env_is_t_env_plus_recipe_deltas(self):
        env_t, _ = ts.arm_plan("T", args(), 1)
        env_s, _ = ts.arm_plan("S", args(), 1)
        shared = set(env_t) & set(env_s) - {"ATLAS_GLM53_PHASE_TIMING"}
        diff = {k for k in shared if env_t[k] != env_s[k]}
        self.assertEqual(diff, set(), "S recipe changes target-only env values")

    def test_every_flag_used_exists_in_the_serve_cli(self):
        src = "".join(p.read_text() for p in CLI.glob("cli*.rs"))
        _, argv = ts.arm_plan("S", args(), 1)
        for f in [x for x in argv if x.startswith("--")]:
            name = f[2:]
            with self.subTest(flag=f):
                self.assertTrue(re.search(rf"pub {name.replace('-', '_')}\b|\"{name}\"", src), f)

    def test_unmappable_recipe_default_is_refused(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            f.write("defaults:\n  max_model_len: 2048\n  new_knob: 1\n")
            f.flush()
            with self.assertRaises(SystemExit):
                ts.load_recipe(Path(f.name))


class EndToEnd(TestCase):
    def run_ts(self, extra_env=None, arms="T1,S1,T2,S2"):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, extra_env or {}):
            rc = ts.main(["--server", MOCK, "--model", "/m", "--draft-model", "/d", "--out", d,
                          "--arms", arms, "--trials", "1", "--port", str(free_port()),
                          "--min-mem-gb", "0", "--watchdog-gb", "0", "--load-timeout", "20",
                          *sum((["--pass-env", k] for k in (extra_env or {})), [])])
            status = (Path(d) / "status.txt").read_text()
            summary = json.loads((Path(d) / "summary.json").read_text())
        return rc, status, summary

    def test_identical_outputs_pass(self):
        rc, status, summary = self.run_ts()
        self.assertEqual(rc, 0, status)
        self.assertTrue(status.startswith("TS_PASS"))
        self.assertIn("binding server_sha256=", status)
        self.assertEqual(set(summary["median_decode_tok_s"]), {"T", "S"})

    def test_s_divergence_fails(self):
        first = json.loads((HERE.parent / "req/request-code2.json").read_text())["prompt"][-4]
        rc, status, _ = self.run_ts({"MOCK_S_DIVERGE": str(first)})
        self.assertEqual(rc, 1)
        self.assertIn("trial-1-code2: S output differs", status)

    def test_control_that_cannot_differ_fails(self):
        rc, status, _ = self.run_ts({"MOCK_CONTROL_SAME": "1"})
        self.assertEqual(rc, 1)
        self.assertIn("gate cannot fail", status)

    def test_server_that_never_starts_fails(self):
        rc, status, _ = self.run_ts({"MOCK_FAIL_START": "1"})
        self.assertEqual(rc, 1)
        self.assertIn("before ready", status)

    def test_admitted_run_without_admission_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            rc = ts.main(["--server", MOCK, "--model", "/m", "--draft-model", "/d", "--out", d, "--arms", "T1,S1",
                          "--trials", "1", "--port", str(free_port()), "--min-mem-gb", "0", "--watchdog-gb", "0",
                          "--load-timeout", "20", "--admitted"])
            self.assertEqual(rc, 1)
            self.assertIn("S1: server exited", (Path(d) / "status.txt").read_text())


p3spec = importlib.util.spec_from_file_location("p3_state", HERE.parent / "p3_state.py")
p3 = importlib.util.module_from_spec(p3spec)
p3spec.loader.exec_module(p3)
RUST = (ts.REPO / "crates/spark-model/src/model/glm53/speculative_admission.rs").read_text()


def rust_rules(ev):
    """The checks of Glm53SpeculativeEvidence::validate(), mirrored (MIN_* read from the source)."""
    mins = {k: int(v) for k, v in re.findall(r"const (MIN_\w+): (?:u32|usize) = (\d+);", RUST)}
    return (ev["engine"] == "flux-engine" and ev["scope"] == "greedy"
            and len(ev["identity_prompts"]) >= mins["MIN_PROMPTS"]
            and ev["identity_trials_equal"] >= mins["MIN_IDENTITY_TRIALS"] and ev["identity_trials_differ"] == 0
            and ev["state_gate_passed"] and ev["state_prefix_positions_equal"] >= mins["MIN_PREFIX_POSITIONS"]
            and ev["state_full_positions_equal"] >= mins["MIN_FULL_POSITIONS"]
            and ev["perturbed_prompt_control_differs"] and ev["kernel_harness_bit_identical"]
            and all(ev[k] for k in ev if k.startswith("control_"))
            and len(ev["identity_binary_sha256"]) == 64 and len(ev["state_binary_sha256"]) == 64)


fe_spec = importlib.util.spec_from_file_location("flux_evidence", HERE.parent / "flux_evidence.py")
fe = importlib.util.module_from_spec(fe_spec)
fe_spec.loader.exec_module(fe)


def write_receipt(path, **override):
    """A harness/run.sh receipt as an executed, passing GB10 run on this tree would write it."""
    import hashlib
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    r = {"schema": "atlas.glm53.rowbatch_harness.v1",
         "harness_source_sha256": sha(HERE.parent / "harness/rowbatch_equiv.cu"),
         "kernel_source": fe.HARNESS_KERNEL, "kernel_source_sha256": sha(ts.REPO / fe.HARNESS_KERNEL),
         "spec_source_fnv1a64": ts.spec_source_fnv1a64(), "exllamav3_rev": fe.EXLLAMAV3_REV,
         "nvcc": ["Build cuda_13.0"], "binary_sha256": "a" * 64, "build_rc": 0, "executed": True, "run_rc": 0,
         "result": "BIT-IDENTICAL", "failing_cases": 0, "result_lines": 1, "exllamav3_dirty": False,
         "gpu": "NVIDIA GB10", **override}
    Path(path).write_text(json.dumps(r))


class P3AndEvidence(TestCase):
    @classmethod
    def common(cls, d, extra=()):
        return ["--server", MOCK, "--model", "/m", "--draft-model", "/d", "--out", d, "--port", str(free_port()),
                "--min-mem-gb", "0", "--watchdog-gb", "0", "--load-timeout", "20", *extra]

    @classmethod
    def runs(cls, trials="3", fnv=None, extra_env=None):
        root = Path(tempfile.mkdtemp())
        env = {"MOCK_SPEC_FNV": fnv or ts.spec_source_fnv1a64(), **(extra_env or {})}
        pe = sum((["--pass-env", k] for k in env), [])
        with mock.patch.dict(os.environ, env):
            rc2 = ts.main(cls.common(str(root / "p2"), ["--trials", trials, *pe]))
            rc3 = p3.main(cls.common(str(root / "p3"), ["--trials", "2", *pe]))
        write_receipt(root / "harness.json")
        return rc2, rc3, root

    def evidence(self, root):
        out = root / "evidence.json"
        out.unlink(missing_ok=True)
        r = __import__("subprocess").run([sys.executable, str(HERE.parent / "make_evidence.py"), "--flux",
                                          str(root / "p2"), str(root / "p3"), str(root / "harness.json"), str(out)],
                                         capture_output=True, text=True)
        return r.returncode, (json.loads(out.read_text()) if out.exists() else None), r.stderr

    @classmethod
    def setUpClass(cls):
        cls.good_rc2, cls.good_rc3, cls.good = cls.runs()

    def copy_good(self):
        dst = Path(tempfile.mkdtemp()) / "run"
        __import__("shutil").copytree(self.good, dst)
        return dst

    def refused(self, root, why):
        rc, ev, err = self.evidence(root)
        self.assertEqual(rc, 2, err)
        self.assertIsNone(ev)
        self.assertIn(why, err)

    def test_p3_passes_and_evidence_admits(self):
        self.assertEqual((self.good_rc2, self.good_rc3), (0, 0), (self.good / "p3/status.txt").read_text())
        rc, ev, err = self.evidence(self.copy_good())
        self.assertEqual(rc, 0, err)
        self.assertTrue(rust_rules(ev), json.dumps(ev, indent=1))

    def test_generated_evidence_has_exactly_the_rust_fields(self):
        _, ev, _ = self.evidence(self.copy_good())
        body = RUST[RUST.index("pub(crate) struct Glm53SpeculativeEvidence {"):]
        body = body[:body.index("\n}\n")]
        self.assertEqual(set(ev), set(re.findall(r"pub (\w+):", body)))

    def test_logged_source_hash_is_recorded_per_s_arm(self):
        rec = json.loads((self.good / "p2/S1/arm.json").read_text())
        self.assertEqual(rec["spec_source_fnv1a64"], ts.spec_source_fnv1a64())
        self.assertIsNone(json.loads((self.good / "p2/T1/arm.json").read_text())["spec_source_fnv1a64"])

    def test_python_reads_the_rust_manifest(self):
        self.assertGreaterEqual(len(ts.SPEC_SOURCES), 20)
        self.assertIn("crates/spark-server/src/scheduler/glm53_policy_driver.rs", ts.SPEC_SOURCES)

    def test_binary_built_from_other_sources_is_refused(self):
        rc2, rc3, root = self.runs(fnv="0123456789abcdef")
        self.assertEqual((rc2, rc3), (0, 0))  # the runs themselves passed...
        self.refused(root, "binary logged sources")  # ...but they do not certify this tree

    def test_failed_kernel_harness_is_refused(self):
        for field, value, why in (("result", "DIFFER", "not bit-identical"), ("executed", False, "not built and executed"),
                                  ("run_rc", 1, "not built and executed"), ("harness_source_sha256", "0" * 64, "changed since it ran"),
                                  ("exllamav3_rev", "0" * 40, "pinned revision"), ("exllamav3_dirty", True, "clean checkout"),
                                  ("result_lines", 2, "exactly one RESULT"), ("gpu", None, "GB10")):
            root = self.copy_good()
            write_receipt(root / "harness.json", **{field: value})
            with self.subTest(field=field):
                self.refused(root, why)

    def test_plain_log_is_not_a_receipt(self):
        root = self.copy_good()
        (root / "harness.json").unlink()
        (root / "harness.log").write_text("RESULT: BIT-IDENTICAL (0 failing cases)\n")
        rc, ev, err = self.evidence(root)
        self.assertEqual(rc, 2)

    def test_failed_p2_is_refused(self):
        first = json.loads((HERE.parent / "req/request-code2.json").read_text())["prompt"][-4]
        _, _, root = self.runs(extra_env={"MOCK_S_DIVERGE": str(first)})
        self.refused(root, "status is not TS_PASS")

    def test_missing_state_hash_file_is_refused(self):
        root = self.copy_good()
        (root / "p3/Cext.hash").unlink()
        self.refused(root, "Cext.hash missing")

    def test_missing_control_responses_are_not_a_failed_control(self):
        root = self.copy_good()
        rec = json.loads((root / "p3/Cleg/arm.json").read_text())
        rec["responses"] = {}
        (root / "p3/Cleg/arm.json").write_text(json.dumps(rec))
        self.refused(root, "Cleg")

    def test_missing_p3_arm_is_refused(self):
        root = self.copy_good()
        __import__("shutil").rmtree(root / "p3/S2h")
        self.refused(root, "P3 arms")

    def test_tampered_raw_response_is_refused(self):
        root = self.copy_good()
        f = root / "p2/S1/trial-1-short.response.json"
        body = json.loads(f.read_text())
        body["choices"][0]["text"] += "x"
        f.write_text(json.dumps(body))
        self.refused(root, "does not match its arm record")

    def test_foreign_status_binding_is_refused(self):
        root = self.copy_good()
        st = (root / "p3/status.txt").read_text().splitlines()
        (root / "p3/status.txt").write_text(st[0] + "\nbinding server_sha256=" + "f" * 64 + "\n")
        self.refused(root, "binary differs from the status binding")

    def rewrite_hash(self, root, arm, fn):
        f = root / f"p3/{arm}.hash"
        f.write_text("".join(fn(line) + "\n" for line in f.read_text().splitlines()))

    def test_hash_without_state_fields_is_refused(self):
        root = self.copy_good()
        for arm in ("Th", "S3h", "S2h", "Cleg", "Cext"):
            self.rewrite_hash(root, arm, lambda l: " ".join(l.split()[:3]))
        self.refused(root, "missing KDA/DSA state fields")

    def test_bogus_field_cannot_fake_a_failed_control(self):
        root = self.copy_good()
        th = (root / "p3/Th.hash").read_text()
        (root / "p3/Cleg.hash").write_text(th.replace("src=walk", "src=prefix").replace("\n", " bogus=1\n"))
        self.refused(root, "unknown state fields")

    def test_control_identical_in_state_fields_is_refused(self):
        root = self.copy_good()
        lines = (root / "p3/Th.hash").read_text().splitlines()[:-16]  # drop Th's trailing control request (16 positions)
        (root / "p3/Cext.hash").write_text("".join(l.replace("src=walk", "src=full") + "\n" for l in lines))
        self.refused(root, "Cext: negative control did not fail the state gate")

    def test_request_ids_must_not_go_backwards(self):
        root = self.copy_good()
        lines = (root / "p3/S3h.hash").read_text().splitlines()
        (root / "p3/S3h.hash").write_text("".join(l + "\n" for l in lines[16:32] + lines[:16] + lines[32:]))
        self.refused(root, "request id went backwards")

    def test_duplicate_state_field_is_refused(self):
        root = self.copy_good()
        self.rewrite_hash(root, "S3h", lambda l: l + " o0.conv=" + "0" * 16)
        self.refused(root, "duplicate state field")

    def test_truncated_final_position_is_refused(self):
        root = self.copy_good()
        lines = (root / "p3/S3h.hash").read_text().splitlines()
        (root / "p3/S3h.hash").write_text("".join(l + "\n" for l in lines[:-1]))
        self.refused(root, "its completion count implies")

    def test_uniform_truncation_across_every_arm_is_refused(self):
        root = self.copy_good()
        for arm in ("Th", "S3h", "S2h", "Cleg", "Cext"):
            f = root / f"p3/{arm}.hash"
            kept, last = [], {}
            lines = f.read_text().splitlines()
            # drop each request's final committed position, identically in every arm
            for i, l in enumerate(lines):
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                if nxt.split(" ", 1)[0] == l.split(" ", 1)[0]:
                    kept.append(l)
            f.write_text("".join(l + "\n" for l in kept))
        self.refused(root, "its completion count implies")

    def test_spec_trace_may_commit_its_terminal_draft(self):
        # verify_policy_transaction.rs: a matching terminal draft is committed, so the S trace can
        # end one position past the walk; that is legitimate and must not refuse
        root = self.copy_good()
        f = root / "p3/S3h.hash"
        lines = f.read_text().splitlines()
        last = lines[15]  # the warm-up request's final line
        req, pos = last.split()[0], int(last.split()[1].split("=")[1])
        extra = last.replace(f"pos={pos}", f"pos={pos + 1}")
        f.write_text("".join(l + "\n" for l in lines[:16] + [extra] + lines[16:]))
        rc, ev, err = self.evidence(root)
        self.assertEqual(rc, 0, err)

    def test_walk_trace_one_past_its_endpoint_is_refused(self):
        root = self.copy_good()
        f = root / "p3/Th.hash"
        lines = f.read_text().splitlines()
        last = lines[15]
        pos = int(last.split()[1].split("=")[1])
        f.write_text("".join(l + "\n" for l in lines[:16] + [last.replace(f"pos={pos}", f"pos={pos + 1}")] + lines[16:]))
        self.refused(root, "its completion count implies")

    def test_bad_warmup_is_refused(self):
        for warm, why in (({"completion_tokens": 0, "finish": "length"}, "positive int"),
                          ({"completion_tokens": 160, "finish": "error"}, "warm-up finish_reason")):
            root = self.copy_good()
            rec = json.loads((root / "p3/S2h/arm.json").read_text())
            rec["warmup"] = warm
            (root / "p3/S2h/arm.json").write_text(json.dumps(rec))
            with self.subTest(warm=warm):
                self.refused(root, why)

    def test_short_control_is_malformed_not_a_state_failure(self):
        root = self.copy_good()
        f = root / "p3/Cleg.hash"
        lines = f.read_text().splitlines()
        f.write_text("".join(l + "\n" for l in lines[16:]))  # the warm-up trace is missing
        self.refused(root, "malformed state trace")

    def test_missing_warmup_count_is_refused(self):
        root = self.copy_good()
        rec = json.loads((root / "p3/Th/arm.json").read_text())
        rec.pop("warmup")
        (root / "p3/Th/arm.json").write_text(json.dumps(rec))
        self.refused(root, "warm-up completion count")

    def test_p3_score_only_rescores_saved_runs(self):
        root = self.copy_good()
        rc = p3.main(self.common(str(root / "p3"), ["--score-only"]))
        self.assertEqual(rc, 0, (root / "p3/status.txt").read_text())
        self.assertIn("rescored", (root / "p3/status.txt").read_text())

    def test_missing_or_out_of_range_ahead_is_refused(self):
        for fn in (lambda l: re.sub(r" d\d+ahead=\d+", "", l), lambda l: l.replace("d3ahead=0", "d3ahead=999").replace("d3ahead=1", "d3ahead=999")):
            root = self.copy_good()
            self.rewrite_hash(root, "S2h", fn)
            with self.subTest():
                self.refused(root, "ahead byte")

    def test_request_dropped_from_every_arm_is_refused(self):
        root = self.copy_good()
        for arm in ("Th", "S3h", "S2h", "Cleg", "Cext"):
            lines = (root / f"p3/{arm}.hash").read_text().splitlines()
            (root / f"p3/{arm}.hash").write_text("".join(l + "\n" for l in lines[16:]))  # drop the warm-up trace
        self.refused(root, "malformed state trace")

    def test_error_finish_and_boolean_token_count_are_refused(self):
        for field, value, why in (("finish_reason", "error", "finish_reason"), ("completion_tokens", True, "completion_tokens")):
            root = self.copy_good()
            f = root / "p2/S1/trial-1-long.response.json"
            body = json.loads(f.read_text())
            if field == "finish_reason":
                body["choices"][0]["finish_reason"] = value
            else:
                body["usage"]["completion_tokens"] = value
            f.write_text(json.dumps(body))
            with self.subTest(field=field):
                self.refused(root, why)

    def test_failed_build_removes_a_stale_record(self):
        root = self.copy_good()
        (root / "evidence.json").write_text('{"stale": true}')
        write_receipt(root / "harness.json", result="DIFFER")
        out = root / "evidence.json"
        r = __import__("subprocess").run([sys.executable, str(HERE.parent / "make_evidence.py"), "--flux",
                                          str(root / "p2"), str(root / "p3"), str(root / "harness.json"), str(out)],
                                         capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertFalse(out.exists(), "a stale evidence file survived a failed build")

    def test_compiled_pending_record_matches_evidence_json(self):
        rec = json.loads((HERE.parent / "evidence.json").read_text())
        self.assertEqual(rec["engine"], "flux-engine")
        self.assertEqual(rec["spec_source_fnv1a64"], ts.spec_source_fnv1a64())
        self.assertFalse(rust_rules(rec), "the checked-in record must stay pending until P3 runs on flux")


class HarnessRunner(TestCase):
    """harness/run.sh lifecycle with a fake nvcc (no CUDA): the receipt describes only this invocation."""
    RUN = HERE.parent / "harness/run.sh"

    def fake_nvcc(self, d: Path, mode: str) -> Path:
        # mode: fail (compile error) | pass (binary prints BIT-IDENTICAL) | differ (binary fails)
        body = {"fail": "exit 2",
                "twice": 'o=$(echo "$@" | sed -E "s/.*-o ([^ ]+).*/\\1/"); printf "#!/bin/sh\\necho \\"RESULT: BIT-IDENTICAL (0 failing cases)\\"\\necho \\"RESULT: DIFFER (2 cases)\\"\\n" > "$o"; chmod +x "$o"',
                "noout": "exit 0",
                "pass": 'o=$(echo "$@" | sed -E "s/.*-o ([^ ]+).*/\\1/"); printf "#!/bin/sh\\necho \\"RESULT: BIT-IDENTICAL (0 failing cases)\\"\\n" > "$o"; chmod +x "$o"',
                "differ": 'o=$(echo "$@" | sed -E "s/.*-o ([^ ]+).*/\\1/"); printf "#!/bin/sh\\necho \\"RESULT: DIFFER (3 cases)\\"\\nexit 1\\n" > "$o"; chmod +x "$o"'}[mode]
        f = d / f"nvcc-{mode}"
        f.write_text("#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'Cuda compilation tools, fake'; exit 0; fi\n" + body + "\n")
        f.chmod(0o755)
        return f

    def run_harness(self, mode, stale=False, path=None, exl=None):
        import subprocess
        d = Path(tempfile.mkdtemp())
        out = d / "out"
        out.mkdir()
        if stale:
            (out / "run.log").write_text("RESULT: BIT-IDENTICAL (0 failing cases)\n")
            (out / "rowbatch_equiv").write_text("old binary")
            (out / "receipt.json").write_text('{"stale": true}')
        env = {**os.environ, "NVCC": str(self.fake_nvcc(d, mode))}
        if path is not None:
            env["PATH"] = path
        if exl is not None:
            env["ATLAS_EXLLAMAV3_SOURCE"] = str(exl)
        r = subprocess.run(["bash", str(self.RUN), str(out)], env=env, capture_output=True, text=True)
        return r.returncode, json.loads((out / "receipt.json").read_text()), out

    def test_failed_build_never_reports_a_stale_pass(self):
        rc, rec, out = self.run_harness("fail", stale=True)
        self.assertNotEqual(rc, 0)
        self.assertEqual((rec["build_rc"] != 0, rec["executed"], rec["result"], rec["binary_sha256"]), (True, False, None, None))
        self.assertFalse((out / "run.log").exists())

    def test_executed_failure_is_recorded_and_exits_nonzero(self):
        rc, rec, _ = self.run_harness("differ")
        self.assertNotEqual(rc, 0)
        self.assertEqual((rec["executed"], rec["run_rc"], rec["result"], rec["failing_cases"]), (True, 1, "DIFFER", 3))

    def test_executed_pass_is_bound_to_this_tree(self):
        rc, rec, _ = self.run_harness("pass")
        self.assertEqual(rc, 0)
        self.assertEqual((rec["executed"], rec["run_rc"], rec["result"], rec["failing_cases"]), (True, 0, "BIT-IDENTICAL", 0))
        self.assertEqual(rec["spec_source_fnv1a64"], ts.spec_source_fnv1a64())
        self.assertEqual(len(rec["binary_sha256"]), 64)

    def test_last_differ_after_a_pass_is_not_a_pass(self):
        rc, rec, _ = self.run_harness("twice")
        self.assertEqual((rec["result"], rec["result_lines"]), (None, 2))

    def test_compiler_that_writes_nothing_cannot_reuse_an_old_binary(self):
        rc, rec, out = self.run_harness("noout", stale=True)
        self.assertNotEqual(rc, 0)
        self.assertIsNone(rec["result"])
        self.assertFalse((out / "run.log").exists() and "BIT-IDENTICAL" in (out / "run.log").read_text())

    def test_missing_nvidia_smi_still_writes_a_fresh_receipt(self):
        import shutil
        d = Path(tempfile.mkdtemp())
        for tool in ("python3", "git", "sed", "rm", "mkdir", "dirname", "chmod", "printf", "echo", "sh", "bash"):
            if shutil.which(tool):
                (d / tool).symlink_to(shutil.which(tool))
        rc, rec, _ = self.run_harness("pass", stale=True, path=str(d))
        self.assertIsNone(rec["gpu"])
        self.assertEqual(rec["result"], "BIT-IDENTICAL")  # recorded, but evidence refuses: not on a GB10

    def test_dirty_exllamav3_checkout_is_recorded(self):
        import subprocess
        repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "h.cuh").write_text("x")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"], check=True)
        (repo / "h.cuh").write_text("edited")
        _, rec, _ = self.run_harness("pass", exl=repo)
        self.assertTrue(rec["exllamav3_dirty"])
