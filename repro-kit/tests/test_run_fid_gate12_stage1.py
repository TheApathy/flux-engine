# SPDX-License-Identifier: AGPL-3.0-only
"""Offline mocked CPU unit tests for `run_fid_gate12_stage1.py`.

Covers every orchestration control with zero real Docker,
GPU, or server activity:
1. Positive 3-arm orchestration (`ref_R`, `ctl_D`, `cand_C`, 36 requests, descriptive
   comparison only): verifies exact SHA-256 pin checks, immutable Docker image ID verification
   and launch by `EXPECTED_IMAGE_ID` (`sha256:6b2de5f9...`), preallocated `create_intent.json`
   and `--cidfile` / `--label io.atlas.fidgate12.owner=<token>`, sequential container ID
   recording, ID-only `stop`/`logs`/`rm`/`inspect`, and descriptive comparisons
   (`evidence_valid=True`, `numerical_acceptance="not_evaluated"`, `gate_passed=None`).
2. Nonblocking lock & hold-file admission refusal (`exit 2`).
3. Reliable `/proc` & GPU/port admission detection (`exit 2`, never kills external jobs).
4. Output directory & container-name collision refusal (`exit 2`).
5. Startup failure & collection failure with guaranteed ID-only cleanup.
6. Whole-run deadline expiry with reserved cleanup budget.
7. Hung cleanup operation (`docker stop` / `docker rm` timeout or lingering container).
8. Uncertain-create reconciliation (Control 1):
   - Subcase A: `docker run` CLI times out (`cid` initially `None`), but Docker actually
     created our container (recorded via `--cidfile` / label). `finally:` reconciles it,
     verifies task ownership (`name`, `label`, `image` match `create_intent.json`), adopts
     its 64-hex `cid`, and stops/logs/removes it cleanly.
   - Subcase B: `docker run` fails (`cid` is `None`) and an unrelated container appears with
     the same name (`fidgate12-<run_id>-ref_R`) but a different/missing owner label:
     reconciliation rejects the unowned container and NEVER signals, stops, or removes it.
   - Subcase C: `docker run` times out (`cid` is `None`) AND Docker is unresponsive to
     reconciliation queries (`docker ps` / `docker inspect` time out): retains honest
     `uncertain_container_state=True` and `uncertain_server_work=True` (`server_cancellation_guaranteed=False`).
9. Strict whole-task lifecycle deadline with zero post-deadline cleanup minima (Control 2):
   - Covers near-deadline startup/cleanup and signal (`SIGTERM`) paths where remaining
     whole-run budget is exhausted (`<= 0s`): verifies no command is ever invoked with a
     fresh 1s/2s/5s/10s floor after `run_deadline` expires, and exhausted cleanup records
     `uncertain_container_state=True` and `uncertain_server_work=True`.
10. Immutable image ID verification & launch (Control 3):
    - Verifies `docker image inspect` fails closed (`Stage1AdmissionError`) on digest mismatch
      and that `docker run` launches `EXPECTED_IMAGE_ID` directly rather than a mutable tag.
"""

from __future__ import annotations

import fcntl
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FID_DIR = ROOT / "fid_gate12"
ORCH_MOD_PATH = FID_DIR / "run_fid_gate12_stage1.py"
FID_MOD_PATH = FID_DIR / "fid_gate12.py"
MANIFEST_PATH = FID_DIR / "fid_gate12_manifest.jsonl"


def load_module(name: str, path: Path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


F = load_module("fid_gate12_mod_for_orch", FID_MOD_PATH)
O = load_module("run_fid_gate12_stage1_mod", ORCH_MOD_PATH)


def write_valid_mock_panel(out_jsonl: Path, manifest_rows: list[dict]) -> None:
    """Use `fid_gate12.collect_panel` with an in-memory mock HTTP callback to produce a valid 12-row panel."""
    by_user = {r["user"]: r for r in manifest_rows}

    def _mock_post(url: str, body: dict, timeout_s: float) -> dict:
        mrow = by_user[body["messages"][1]["content"]]
        content_lp = []
        tids = []
        pieces = []
        for pos in range(64):
            tok0 = f"tok_{pos}"
            tid0 = 1000 + pos
            lp0 = -0.10
            lp1 = -1.60
            top_cands = [
                {"token": tok0, "logprob": lp0, "bytes": None},
                {"token": f"cand1_{pos}", "logprob": lp1, "bytes": None},
                {"token": f"cand2_{pos}", "logprob": lp1 - 0.5, "bytes": None},
                {"token": f"cand3_{pos}", "logprob": lp1 - 1.0, "bytes": None},
                {"token": f"cand4_{pos}", "logprob": lp1 - 1.5, "bytes": None},
            ]
            content_lp.append(
                {"token": tok0, "logprob": lp0, "bytes": None, "top_logprobs": top_cands}
            )
            tids.append(tid0)
            pieces.append(tok0)
        return {
            "id": f"chatcmpl-mock-{mrow['index']}",
            "object": "chat.completion",
            "model": "default",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(pieces)},
                    "finish_reason": "length",
                    "logprobs": {"content": content_lp},
                    "token_ids": tids,
                }
            ],
            "usage": {
                "prompt_tokens": mrow["expected_prompt_tokens_nothink"],
                "completion_tokens": 64,
                "total_tokens": mrow["expected_prompt_tokens_nothink"] + 64,
            },
        }

    F.collect_panel(
        url="http://127.0.0.1:18899/v1/chat/completions",
        out_path=out_jsonl,
        manifest_path=MANIFEST_PATH,
        max_tokens=64,
        http_post=_mock_post,
    )


class FakeCommandEnv:
    """Stateful fake command runner tracking Docker container lifecycle, labels, cidfiles, and timeouts."""

    def __init__(self, manifest_rows: list[dict], clock_list: list[float]) -> None:
        self.manifest_rows = manifest_rows
        self.clock_list = clock_list
        self.containers: dict[str, dict] = {}
        self.next_cid_seq = 1
        self.commands_log: list[list[str]] = []
        self.timeouts_log: list[tuple[list[str], float, float]] = []
        self.nvidia_smi_output = ""
        self.image_id_to_return = O.EXPECTED_IMAGE_ID
        self.fail_collect_on_arm: str | None = None
        self.timeout_collect_on_arm: str | None = None
        self.stop_times_out = False
        self.rm_times_out = False
        self.advance_per_collect_s = 10.0
        self.advance_on_stop_s = 0.0
        self.create_succeeds_but_cli_times_out = False
        self.create_fails_and_spawns_impostor = False
        self.docker_unresponsive_after_create = False
        self.inspect_daemon_error_after_create = False
        self.raise_sigterm_during_collect = False

    def add_preexisting_container(
        self,
        cid: str,
        name: str,
        status: str = "Up 5 minutes",
        image: str = "other:latest",
        owner_label: str = "external-owner",
    ) -> None:
        self.containers[cid] = {
            "id": cid,
            "name": name,
            "status": status,
            "running": status.lower().startswith("up"),
            "image": image,
            "owner_label": owner_label,
            "external": True,
        }

    def __call__(self, argv: list[str], timeout_s: float) -> O.CmdResult:
        self.commands_log.append(list(argv))
        self.timeouts_log.append((list(argv), float(timeout_s), float(self.clock_list[0])))

        if argv[:3] == ["docker", "image", "inspect"]:
            target = argv[-1]
            if target in (O.DEFAULT_IMAGE, self.image_id_to_return):
                return O.CmdResult(0, self.image_id_to_return + "\n", "")
            return O.CmdResult(1, "", f"Error: No such image: {target}")

        if argv[:2] == ["docker", "ps"]:
            if self.docker_unresponsive_after_create and "--filter" in argv:
                return O.CmdResult(124, "", "docker ps timed out", timed_out=True)

            # Check if this is a filtered reconciliation query (`--format {{.ID}}`)
            if "--filter" in argv:
                flt_idx = argv.index("--filter") + 1
                flt = argv[flt_idx]
                matched_ids: list[str] = []
                for c in self.containers.values():
                    if flt.startswith(f"label={O.OWNER_LABEL_KEY}="):
                        want_lbl = flt.split("=", 2)[2]
                        if c.get("owner_label") == want_lbl:
                            matched_ids.append(c["id"])
                    elif flt.startswith("name=^/") and flt.endswith("$"):
                        want_name = flt[len("name=^/") : -1]
                        if c.get("name") == want_name:
                            matched_ids.append(c["id"])
                return O.CmdResult(0, "\n".join(matched_ids) + ("\n" if matched_ids else ""), "")

            lines = []
            for c in self.containers.values():
                lines.append(f"{c['id']}\t{c['name']}\t{c['status']}\t{c['image']}")
            return O.CmdResult(0, "\n".join(lines) + ("\n" if lines else ""), "")

        if argv[0] == "nvidia-smi":
            return O.CmdResult(0, self.nvidia_smi_output, "")

        if argv[:2] == ["docker", "run"]:
            name_idx = argv.index("--name") + 1
            cname = argv[name_idx]
            cidfile_path: Path | None = None
            if "--cidfile" in argv:
                cidfile_path = Path(argv[argv.index("--cidfile") + 1])
            owner_lbl = ""
            if "--label" in argv:
                lbl_arg = argv[argv.index("--label") + 1]
                if lbl_arg.startswith(f"{O.OWNER_LABEL_KEY}="):
                    owner_lbl = lbl_arg.split("=", 1)[1]
            serve_idx = argv.index("flux")
            image_arg = argv[serve_idx - 1]

            if self.create_fails_and_spawns_impostor:
                # Simulate `docker run` failing while an external/unowned container with the same name appears
                imp_cid = "b" * 64
                self.add_preexisting_container(
                    imp_cid,
                    cname,
                    status="Up 1 second",
                    image=image_arg,
                    owner_label="unrelated-external-task",
                )
                return O.CmdResult(1, "", "docker: Error response from daemon: conflict")

            cid = f"{self.next_cid_seq:064x}"
            self.next_cid_seq += 1
            self.containers[cid] = {
                "id": cid,
                "name": cname,
                "status": "Up 1 second",
                "running": True,
                "image": image_arg,
                "owner_label": owner_lbl,
                "external": False,
            }
            if cidfile_path is not None:
                cidfile_path.write_text(cid, encoding="utf-8")

            if self.create_succeeds_but_cli_times_out:
                # Container was created in daemon & cidfile, but CLI `docker run` timed out with empty stdout!
                return O.CmdResult(124, "", "[TimeoutExpired during docker run]", timed_out=True)

            return O.CmdResult(0, cid + "\n", "")

        if argv[:2] == ["docker", "inspect"]:
            if self.docker_unresponsive_after_create:
                return O.CmdResult(124, "", "docker inspect timed out", timed_out=True)
            if self.inspect_daemon_error_after_create:
                return O.CmdResult(
                    1,
                    "",
                    "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
                    timed_out=False,
                )
            cid = argv[-1]
            if cid not in self.containers:
                return O.CmdResult(1, "", f"Error: No such object: {cid}")
            c = self.containers[cid]
            fmt = argv[3] if len(argv) > 3 else ""
            if O.OWNER_LABEL_KEY in fmt:
                line = f"{c['id']}\t/{c['name']}\t{c.get('owner_label', '')}\t{c['image']}\n"
                return O.CmdResult(0, line, "")
            if "Running" in fmt:
                return O.CmdResult(0, "true\n" if c["running"] else "false\n", "")
            return O.CmdResult(0, "running\n" if c["running"] else "exited\n", "")

        if argv[:2] == ["docker", "exec"]:
            cid = argv[2]
            assert cid in self.containers and not self.containers[cid]["external"]
            return O.CmdResult(
                0,
                "/opt/flux/bin/flux-server-real serve --model Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid --max-seq-len 16384\n",
                "FLUX_MAX_SEQ_LEN=16384\nFLUX_DISABLE_SPECULATION=1\n",
            )

        if argv[:2] == ["docker", "stop"]:
            cid = argv[-1]
            assert cid in self.containers and not self.containers[cid]["external"], (
                f"Attempted to stop unowned/external container: {cid}"
            )
            self.clock_list[0] += self.advance_on_stop_s
            if self.stop_times_out:
                return O.CmdResult(124, "", "stop timed out", timed_out=True)
            self.containers[cid]["running"] = False
            self.containers[cid]["status"] = "Exited (0) 1 second ago"
            return O.CmdResult(0, cid + "\n", "")

        if argv[:2] == ["docker", "logs"]:
            cid = argv[-1]
            assert cid in self.containers and not self.containers[cid]["external"]
            return O.CmdResult(0, f"server logs for {cid[:12]}\n", "")

        if argv[:2] == ["docker", "rm"]:
            cid = argv[-1]
            assert cid in self.containers and not self.containers[cid]["external"], (
                f"Attempted to rm unowned/external container: {cid}"
            )
            if self.rm_times_out:
                return O.CmdResult(124, "", "rm timed out", timed_out=True)
            del self.containers[cid]
            return O.CmdResult(0, cid + "\n", "")

        if argv[:2] == ["docker", "kill"]:
            cid = argv[-1]
            assert cid in self.containers and not self.containers[cid]["external"], (
                f"Attempted to kill unowned/external container: {cid}"
            )
            self.containers[cid]["running"] = False
            return O.CmdResult(0, cid + "\n", "")

        # Subprocess calls to `fid_gate12.py collect` or `fid_gate12.py compare`
        if len(argv) >= 3 and argv[1].endswith("fid_gate12.py"):
            subcmd = argv[2]
            if subcmd == "collect":
                out_jsonl = Path(argv[4])
                arm_name = out_jsonl.parent.name
                self.clock_list[0] += self.advance_per_collect_s
                if self.raise_sigterm_during_collect:
                    os.kill(os.getpid(), signal.SIGTERM)
                if self.timeout_collect_on_arm == arm_name:
                    return O.CmdResult(
                        124,
                        "",
                        "FID_GATE_ERROR: client_wall_deadline_exceeded",
                        timed_out=True,
                    )
                if self.fail_collect_on_arm == arm_name:
                    return O.CmdResult(2, "", "FID_GATE_ERROR: simulated_validation_failure")
                write_valid_mock_panel(out_jsonl, self.manifest_rows)
                return O.CmdResult(0, "COLLECT_OK prompts=12\n", "")
            if subcmd == "compare":
                rc = F.main(argv[2:])
                return O.CmdResult(rc, "compare completed\n", "")

        raise AssertionError(f"Unexpected command in FakeCommandEnv: {argv}")


class TestRunFidGate12Stage1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_rows = F.load_and_validate_manifest(MANIFEST_PATH)
        cls._fixture_td = tempfile.TemporaryDirectory()
        root = Path(cls._fixture_td.name)
        cls.fake_binary = root / "flux-server-fixture"
        cls.fake_binary.write_bytes(b"offline-fixture-binary-bytes\n")
        cls.fake_model_dir = root / "model-fixture"
        cls.fake_model_dir.mkdir()
        cls._orig_bin_sha = O.EXPECTED_BINARY_SHA256
        cls._orig_model_shas = dict(O.EXPECTED_MODEL_FILE_SHA256)
        O.EXPECTED_BINARY_SHA256 = O.sha256_file(cls.fake_binary)
        patched_model_shas = {}
        for rel_name in cls._orig_model_shas:
            fpath = cls.fake_model_dir / rel_name
            fpath.write_text(f'{{"fixture": "{rel_name}"}}\n', encoding="utf-8")
            patched_model_shas[rel_name] = O.sha256_file(fpath)
        O.EXPECTED_MODEL_FILE_SHA256 = patched_model_shas

    @classmethod
    def tearDownClass(cls):
        O.EXPECTED_BINARY_SHA256 = cls._orig_bin_sha
        O.EXPECTED_MODEL_FILE_SHA256 = cls._orig_model_shas
        cls._fixture_td.cleanup()

    def _make_orchestrator(
        self,
        td_path: Path,
        fake_env: FakeCommandEnv,
        clock_list: list[float],
        *,
        out_root: Path | None = None,
        lock_path: Path | None = None,
        hold_path: Path | None = None,
        proc_scanner=lambda: [],
        mem_reader=lambda: 112000,
        port_busy_probe=lambda h, p: False,
        health_probe=lambda u, t: True,
        total_run_cap_s: float = 1650.0,
        startup_cap_s: float = 180.0,
        cleanup_cap_s: float = 60.0,
        verify_pins: bool = True,
    ) -> O.Stage1Orchestrator:
        return O.Stage1Orchestrator(
            out_root=out_root or (td_path / "runs" / "stage1-test"),
            run_id="20261101T230000Z",
            lock_path=lock_path or (td_path / ".flux-build.lock"),
            hold_path=hold_path or (td_path / ".flux-build.hold"),
            binary_path=self.fake_binary,
            model_dir=self.fake_model_dir,
            harness_path=FID_MOD_PATH,
            manifest_path=MANIFEST_PATH,
            total_run_cap_s=total_run_cap_s,
            startup_cap_s=startup_cap_s,
            cleanup_cap_s=cleanup_cap_s,
            verify_pins=verify_pins,
            cmd_runner=fake_env,
            clock=lambda: clock_list[0],
            sleep_fn=lambda s: clock_list.__setitem__(0, clock_list[0] + s),
            proc_scanner=proc_scanner,
            mem_reader=mem_reader,
            port_busy_probe=port_busy_probe,
            health_probe=health_probe,
        )

    def test_01_positive_3_arm_stage1_orchestration_and_pin_verification(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            clock_list = [100.0]
            fake_env = FakeCommandEnv(self.manifest_rows, clock_list)
            orch = self._make_orchestrator(td_path, fake_env, clock_list, verify_pins=True)

            summary = orch.run()
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(summary["uncertain_container_state"])
            self.assertFalse(summary["uncertain_server_work"])
            self.assertFalse(summary["server_cancellation_guaranteed"])
            self.assertEqual(len(summary["owned_container_ids"]), 3)
            self.assertEqual([a["arm"] for a in summary["arms"]], ["ref_R", "ctl_D", "cand_C"])
            self.assertEqual(len(fake_env.containers), 0)

            # Verify immutable image ID was verified and recorded in `pins` and launched in `docker run`
            self.assertEqual(summary["pins"]["image_id"], O.EXPECTED_IMAGE_ID)
            run_cmds = [cmd for cmd in fake_env.commands_log if cmd[:2] == ["docker", "run"]]
            self.assertEqual(len(run_cmds), 3)
            for rcmd in run_cmds:
                self.assertIn(O.EXPECTED_IMAGE_ID, rcmd)
                self.assertNotIn(O.DEFAULT_IMAGE, rcmd)
                self.assertIn("--cidfile", rcmd)
                self.assertIn("--label", rcmd)

            # Verify all 3 descriptive comparisons ran with `enforce_gates=False` (`gate_passed=None`)
            self.assertEqual(len(summary["comparisons"]), 3)
            for cmp_entry in summary["comparisons"]:
                self.assertTrue(cmp_entry["evidence_valid"])
                self.assertEqual(cmp_entry["numerical_acceptance"], "not_evaluated")
                self.assertIsNone(cmp_entry["gate_passed"])
                self.assertEqual(cmp_entry["first_token_top1_agree"], 12)
                self.assertEqual(cmp_entry["continuations_identical"], 12)

    def test_02_refuses_when_lock_busy_or_hold_file_present(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            clock_list = [0.0]
            fake_env = FakeCommandEnv(self.manifest_rows, clock_list)
            lock_path = td_path / ".flux-build.lock"
            hold_path = td_path / ".flux-build.hold"

            hold_path.write_text("hold\n", encoding="utf-8")
            orch1 = self._make_orchestrator(
                td_path, fake_env, clock_list, lock_path=lock_path, hold_path=hold_path
            )
            with self.assertRaisesRegex(O.Stage1AdmissionError, "build_hold_file_present"):
                orch1.run()
            hold_path.unlink()

            ext_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o664)
            fcntl.flock(ext_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                orch2 = self._make_orchestrator(
                    td_path, fake_env, clock_list, lock_path=lock_path, hold_path=hold_path
                )
                with self.assertRaisesRegex(
                    O.Stage1AdmissionError, "lock_busy_held_by_another_owner"
                ):
                    orch2.run()
            finally:
                fcntl.flock(ext_fd, fcntl.LOCK_UN)
                os.close(ext_fd)

    def test_03_proc_scanner_detects_truncated_flux_server_real_and_compilers(self):
        with tempfile.TemporaryDirectory() as td:
            fake_proc = Path(td) / "proc"
            fake_proc.mkdir()

            p1 = fake_proc / "900001"
            p1.mkdir()
            (p1 / "comm").write_text("flux-server-rea\n", encoding="utf-8")
            (p1 / "cmdline").write_bytes(b"/opt/flux/bin/flux-server-real\x00serve\x00")

            conflicts = O.scan_conflicting_processes(proc_root=fake_proc)
            self.assertEqual(len(conflicts), 1)
            self.assertIn("pid=900001(comm=flux-server-rea)", conflicts[0])

            p2 = fake_proc / "900002"
            p2.mkdir()
            (p2 / "comm").write_text("rustc\n", encoding="utf-8")
            (p2 / "cmdline").write_bytes(b"/usr/bin/rustc\x00--crate-name\x00spark\x00")
            conflicts2 = O.scan_conflicting_processes(proc_root=fake_proc)
            self.assertEqual(len(conflicts2), 2)

            clock_list = [0.0]
            fake_env = FakeCommandEnv(self.manifest_rows, clock_list)
            orch = self._make_orchestrator(
                Path(td),
                fake_env,
                clock_list,
                proc_scanner=lambda: O.scan_conflicting_processes(proc_root=fake_proc),
            )
            with self.assertRaisesRegex(
                O.Stage1AdmissionError, "conflicting_host_processes_active.*flux-server-rea"
            ):
                orch.run()
            # Only `docker image inspect` ran during `_verify_all_pins`; zero `docker run` calls
            self.assertFalse(any(cmd[:2] == ["docker", "run"] for cmd in fake_env.commands_log))

    def test_04_refuses_on_output_dir_or_container_name_collision_without_touching_other_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            clock_list = [0.0]
            fake_env = FakeCommandEnv(self.manifest_rows, clock_list)

            existing_out = td_path / "existing_out"
            existing_out.mkdir()
            orch_dir = self._make_orchestrator(
                td_path, fake_env, clock_list, out_root=existing_out
            )
            with self.assertRaisesRegex(
                O.Stage1AdmissionError, "output_directory_already_exists_refusing_overwrite"
            ):
                orch_dir.run()

            ext_cid = "f" * 64
            fake_env.add_preexisting_container(
                ext_cid,
                "fidgate12-20261101T230000Z-ref_R",
                status="Exited (0) 2 minutes ago",
            )
            orch_col = self._make_orchestrator(
                td_path, fake_env, clock_list, out_root=td_path / "fresh_out1"
            )
            with self.assertRaisesRegex(O.Stage1AdmissionError, "container_name_collision"):
                orch_col.run()
            self.assertIn(ext_cid, fake_env.containers)

            fake_env.containers.clear()
            other_cid = "e" * 64
            fake_env.add_preexisting_container(
                other_cid,
                "other-owner-job",
                status="Up 3 minutes",
                image="flux-engine:v0.9.15.6-rc1",
            )
            orch_other = self._make_orchestrator(
                td_path, fake_env, clock_list, out_root=td_path / "fresh_out2"
            )
            with self.assertRaisesRegex(O.Stage1AdmissionError, "conflicting_running_container"):
                orch_other.run()
            self.assertTrue(fake_env.containers[other_cid]["running"])

    def test_05_startup_or_collection_failure_cleans_up_owned_container_by_id(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            clock_a = [0.0]
            env_a = FakeCommandEnv(self.manifest_rows, clock_a)
            orch_a = self._make_orchestrator(
                td_path,
                env_a,
                clock_a,
                out_root=td_path / "startup_fail",
                health_probe=lambda u, t: False,
            )
            with self.assertRaisesRegex(O.Stage1ExecutionError, "startup_health_timeout"):
                orch_a.run()
            self.assertEqual(len(env_a.containers), 0)
            run_sum_a = json.loads((td_path / "startup_fail" / "run_summary.json").read_text())
            self.assertEqual(run_sum_a["status"], "failed")
            self.assertTrue(run_sum_a["uncertain_server_work"])
            self.assertFalse(run_sum_a["uncertain_container_state"])
            self.assertEqual(len(run_sum_a["arms"]), 1)
            self.assertTrue(run_sum_a["arms"][0]["cleanup"]["container_removed"])

            clock_b = [0.0]
            env_b = FakeCommandEnv(self.manifest_rows, clock_b)
            env_b.fail_collect_on_arm = "ctl_D"
            orch_b = self._make_orchestrator(
                td_path,
                env_b,
                clock_b,
                out_root=td_path / "collect_fail",
            )
            with self.assertRaisesRegex(O.Stage1ExecutionError, r"\[ctl_D\] collect_failed"):
                orch_b.run()
            self.assertEqual(len(env_b.containers), 0)
            run_sum_b = json.loads((td_path / "collect_fail" / "run_summary.json").read_text())
            self.assertEqual(run_sum_b["status"], "failed")
            self.assertEqual([a["arm"] for a in run_sum_b["arms"]], ["ref_R", "ctl_D"])
            self.assertFalse((td_path / "collect_fail" / "cand_C").exists())

    def test_06_whole_run_deadline_expiry_preserves_cleanup_reserve(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            clock_list = [0.0]
            fake_env = FakeCommandEnv(self.manifest_rows, clock_list)
            fake_env.advance_per_collect_s = 200.0
            orch = self._make_orchestrator(
                td_path,
                fake_env,
                clock_list,
                out_root=td_path / "deadline_run",
                total_run_cap_s=450.0,
            )
            with self.assertRaisesRegex(
                O.Stage1ExecutionError, r"\[cand_C\] run_deadline_exceeded_before_arm_start"
            ):
                orch.run()
            self.assertEqual(len(fake_env.containers), 0)

    def test_07_hung_cleanup_does_not_skip_rm_and_reports_uncertain_container_state(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            clock_1 = [0.0]
            env_1 = FakeCommandEnv(self.manifest_rows, clock_1)
            env_1.stop_times_out = True
            orch_1 = self._make_orchestrator(
                td_path, env_1, clock_1, out_root=td_path / "stop_timeout_ok_rm"
            )
            sum_1 = orch_1.run()
            self.assertEqual(sum_1["status"], "completed")
            self.assertFalse(sum_1["uncertain_container_state"])
            self.assertTrue(sum_1["arms"][0]["cleanup"]["stop_timed_out"])
            self.assertTrue(sum_1["arms"][0]["cleanup"]["container_removed"])
            self.assertEqual(len(env_1.containers), 0)

            clock_2 = [0.0]
            env_2 = FakeCommandEnv(self.manifest_rows, clock_2)
            env_2.stop_times_out = True
            env_2.rm_times_out = True
            orch_2 = self._make_orchestrator(
                td_path, env_2, clock_2, out_root=td_path / "rm_timeout_uncertain"
            )
            with self.assertRaisesRegex(
                O.Stage1ExecutionError, "cleanup_left_uncertain_container_state"
            ):
                orch_2.run()

            sum_2 = json.loads(
                (td_path / "rm_timeout_uncertain" / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sum_2["status"], "failed")
            self.assertTrue(sum_2["uncertain_container_state"])
            self.assertTrue(sum_2["uncertain_server_work"])
            self.assertFalse(sum_2["server_cancellation_guaranteed"])
            self.assertFalse(sum_2["arms"][0]["cleanup"]["container_removed"])

    def test_08_uncertain_create_reconciliation_and_ownership_verification(self):
        """Verifies Control 1: create-succeeds-but-CLI-times-out is reconciled and cleaned up; impostor is never touched; unresponsive Docker retains uncertainty."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # Subcase A: `docker run` creates the container (and writes `--cidfile`/label) but CLI times out (`cid` is None).
            # `finally:` reconciles via `--cidfile` + `docker ps --filter label=...` + `docker inspect`, verifies ownership,
            # adopts `cid`, and stops/logs/removes the container before aborting!
            clock_a = [0.0]
            env_a = FakeCommandEnv(self.manifest_rows, clock_a)
            env_a.create_succeeds_but_cli_times_out = True
            orch_a = self._make_orchestrator(
                td_path, env_a, clock_a, out_root=td_path / "create_timeout_reconciled"
            )
            with self.assertRaisesRegex(O.Stage1ExecutionError, "docker_run_failed"):
                orch_a.run()

            # Container was reconciled, verified as ours, and removed!
            self.assertEqual(len(env_a.containers), 0)
            sum_a = json.loads(
                (td_path / "create_timeout_reconciled" / "run_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(sum_a["status"], "failed")
            self.assertEqual(len(sum_a["owned_container_ids"]), 1)
            arm_a = sum_a["arms"][0]
            self.assertIsNotNone(arm_a["reconciliation"]["verified_cid"])
            self.assertTrue(arm_a["cleanup"]["container_removed"])
            self.assertFalse(sum_a["uncertain_container_state"])
            self.assertTrue((td_path / "create_timeout_reconciled" / "ref_R" / "create_intent.json").is_file())

            # Subcase B: `docker run` fails while an unrelated container with the same name (`fidgate12-...-ref_R`)
            # but a different owner label exists -> reconciliation rejects it and NEVER stops/removes it!
            clock_b = [0.0]
            env_b = FakeCommandEnv(self.manifest_rows, clock_b)
            env_b.create_fails_and_spawns_impostor = True
            orch_b = self._make_orchestrator(
                td_path, env_b, clock_b, out_root=td_path / "impostor_rejected"
            )
            with self.assertRaisesRegex(O.Stage1ExecutionError, "docker_run_failed"):
                orch_b.run()

            # Impostor container (`"b"*64`) is still running untouched, and `owned_container_ids` is empty!
            self.assertIn("b" * 64, env_b.containers)
            self.assertTrue(env_b.containers["b" * 64]["running"])
            sum_b = json.loads(
                (td_path / "impostor_rejected" / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sum_b["owned_container_ids"], [])
            self.assertIsNone(sum_b["arms"][0]["reconciliation"]["verified_cid"])
            self.assertEqual(len(sum_b["arms"][0]["reconciliation"]["rejected_candidates"]), 1)

            # Subcase C: `docker run` times out AND Docker is unresponsive to reconciliation queries ->
            # retains honest `uncertain_container_state=True` and `uncertain_server_work=True`!
            clock_c = [0.0]
            env_c = FakeCommandEnv(self.manifest_rows, clock_c)
            env_c.create_succeeds_but_cli_times_out = True
            env_c.docker_unresponsive_after_create = True
            orch_c = self._make_orchestrator(
                td_path, env_c, clock_c, out_root=td_path / "create_timeout_unresponsive"
            )
            with self.assertRaisesRegex(O.Stage1ExecutionError, "docker_run_failed"):
                orch_c.run()

            sum_c = json.loads(
                (td_path / "create_timeout_unresponsive" / "run_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(sum_c["uncertain_container_state"])
            self.assertTrue(sum_c["uncertain_server_work"])
            self.assertFalse(sum_c["server_cancellation_guaranteed"])

            # Subcase D: `docker run` times out, `docker ps` succeeds, and `docker inspect` returns
            # a nonzero daemon/communication error (NOT `No such object`) -> does NOT treat failed inspect as
            # absence, skips unverified cleanup, and preserves `uncertain_container_state=True` and `uncertain_server_work=True`!
            clock_d = [0.0]
            env_d = FakeCommandEnv(self.manifest_rows, clock_d)
            env_d.create_succeeds_but_cli_times_out = True
            env_d.inspect_daemon_error_after_create = True
            orch_d = self._make_orchestrator(
                td_path, env_d, clock_d, out_root=td_path / "create_timeout_ps_ok_inspect_err"
            )
            with self.assertRaisesRegex(O.Stage1ExecutionError, "docker_run_failed"):
                orch_d.run()

            sum_d = json.loads(
                (td_path / "create_timeout_ps_ok_inspect_err" / "run_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(sum_d["status"], "failed")
            self.assertEqual(sum_d["owned_container_ids"], [])
            self.assertTrue(sum_d["uncertain_container_state"])
            self.assertTrue(sum_d["uncertain_server_work"])
            self.assertFalse(sum_d["server_cancellation_guaranteed"])
            reconcile_d = sum_d["arms"][0]["reconciliation"]
            self.assertIsNone(reconcile_d["verified_cid"])
            self.assertFalse(reconcile_d["absence_proven"])
            self.assertTrue(reconcile_d["reconcile_uncertain"])
            self.assertGreaterEqual(len(reconcile_d["inspect_failed_candidates"]), 1)

    def test_09_strict_whole_run_deadline_zero_post_deadline_minima_and_signal_cleanup(self):
        """Verifies Control 2: no fresh 1s/2s/5s/10s floor is granted after `run_deadline` expires, and signal paths respect the same deadline."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # Subcase A: Near-deadline cleanup where `docker stop` consumes the remaining whole-run budget (`run_deadline`).
            # Once `clock >= run_deadline`, `docker logs`, `docker rm -f`, and `docker inspect` MUST short-circuit
            # immediately without invoking `cmd_runner` with any fresh positive timeout!
            clock_a = [0.0]
            env_a = FakeCommandEnv(self.manifest_rows, clock_a)
            env_a.stop_times_out = True
            env_a.advance_on_stop_s = 260.0  # pushes clock past `total_run_cap_s=260.0` during `docker stop`!
            env_a.fail_collect_on_arm = "ref_R"
            orch_a = self._make_orchestrator(
                td_path,
                env_a,
                clock_a,
                out_root=td_path / "near_deadline_cleanup",
                total_run_cap_s=260.0,
            )
            with self.assertRaisesRegex(O.Stage1ExecutionError, "collect_failed"):
                orch_a.run()

            # Verify `docker stop` actually exhausted `run_deadline`, and that no
            # `docker logs`, `docker rm`, or post-`stop` `docker inspect` ran afterward.
            self.assertGreaterEqual(clock_a[0], orch_a.run_deadline)
            stop_indices = [
                idx for idx, argv in enumerate(env_a.commands_log) if argv[:2] == ["docker", "stop"]
            ]
            self.assertEqual(len(stop_indices), 1)
            post_stop_cmds = [argv[:2] for argv in env_a.commands_log[stop_indices[0] + 1 :]]
            for forbidden in (["docker", "logs"], ["docker", "rm"], ["docker", "inspect"]):
                self.assertNotIn(
                    forbidden,
                    post_stop_cmds,
                    f"Command {forbidden} ran after run_deadline expired during docker stop: {post_stop_cmds}",
                )

            # Verify no command was EVER passed to `cmd_runner` after `clock_a[0] > orch_a.run_deadline`,
            # and every call to `cmd_runner` had `timeout_s <= orch_a.run_deadline - call_clock`!
            for argv, timeout_s, call_clock in env_a.timeouts_log:
                self.assertGreater(
                    timeout_s,
                    0.0,
                    f"Command invoked with non-positive timeout: {argv}",
                )
                self.assertLessEqual(
                    call_clock + timeout_s,
                    orch_a.run_deadline + 1e-6,
                    f"Command {argv} granted timeout {timeout_s}s exceeding run_deadline {orch_a.run_deadline} at t={call_clock}",
                )

            sum_a = json.loads(
                (td_path / "near_deadline_cleanup" / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(sum_a["uncertain_container_state"])
            self.assertTrue(sum_a["uncertain_server_work"])
            self.assertTrue(sum_a["arms"][0]["cleanup"]["rm_timed_out"])

            # Subcase B: Abnormal signal (`SIGTERM`) during collection triggers bounded ID-only cleanup
            clock_b = [0.0]
            env_b = FakeCommandEnv(self.manifest_rows, clock_b)
            env_b.raise_sigterm_during_collect = True
            orch_b = self._make_orchestrator(
                td_path,
                env_b,
                clock_b,
                out_root=td_path / "sigterm_cleanup",
                total_run_cap_s=1650.0,
            )
            with self.assertRaisesRegex(O.Stage1ExecutionError, "terminated_by_signal_SIGTERM"):
                orch_b.run()
            self.assertEqual(len(env_b.containers), 0)
            sum_b = json.loads(
                (td_path / "sigterm_cleanup" / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sum_b["status"], "failed")
            self.assertTrue(sum_b["arms"][0]["cleanup"]["container_removed"])

    def test_10_immutable_image_id_verification_fails_closed_on_mismatch(self):
        """Verifies Control 3: `docker image inspect` fails closed (`Stage1AdmissionError`) if the image digest does not match `EXPECTED_IMAGE_ID`."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            clock_list = [0.0]
            fake_env = FakeCommandEnv(self.manifest_rows, clock_list)
            fake_env.image_id_to_return = (
                "sha256:0000000000000000000000000000000000000000000000000000000000000000"
            )
            orch = self._make_orchestrator(
                td_path, fake_env, clock_list, out_root=td_path / "bad_image_out", verify_pins=True
            )
            with self.assertRaisesRegex(O.Stage1AdmissionError, "image_id_mismatch"):
                orch.run()
            self.assertFalse((td_path / "bad_image_out").exists())
            self.assertFalse(any(cmd[:2] == ["docker", "run"] for cmd in fake_env.commands_log))


if __name__ == "__main__":
    unittest.main()
