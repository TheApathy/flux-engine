# SPDX-License-Identifier: AGPL-3.0-only
"""Hostile offline reproductions for the exact-image review holds."""

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import test_r15_container_suite as suite_fixture
import test_r15_exact_image_release as outer_fixture
import test_r12_container_release_runner as runner_fixture

S = suite_fixture.S
Q = outer_fixture.Q


class ReleaseHoldRegressions(unittest.TestCase):
    def test_every_marker_at_every_read_boundary_split(self):
        markers = (
            b"CUDA fatal",
            b"CUDA out of memory",
            b"illegal memory access",
            b"NVRM",
            b"Xid",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "raw.log"
            for label, marker in zip(S.RAW_LOG_PATTERNS, markers):
                for split in range(1, len(marker)):
                    with self.subTest(label=label, split=split):
                        with path.open("wb") as stream:
                            stream.seek(4 * 1024 * 1024 - split)
                            stream.write(marker + b"\n")
                        descriptor = os.open(path, os.O_RDONLY)
                        try:
                            first = S.scan_raw_log(descriptor)
                        finally:
                            os.close(descriptor)
                        second = Q._scan_stable_raw_log(root, "raw.log")
                        for census in (first, second):
                            self.assertEqual(census["match_count"], 1)
                            self.assertEqual(
                                census["per_pattern_match_counts"][label], 1
                            )

    def test_scans_preserve_word_boundaries_and_unbounded_separators(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.log"
            path.write_bytes(b"AXidB aNVRMb\nCuDa" + b"-" * 10000 + b"FaTaL\n")
            descriptor = os.open(path, os.O_RDONLY)
            try:
                first = S.scan_raw_log(descriptor)
            finally:
                os.close(descriptor)
            second = Q._scan_stable_raw_log(Path(temp), "raw.log")
            self.assertEqual(
                first["per_pattern_match_counts"], second["per_pattern_match_counts"]
            )
            self.assertEqual(first["match_count"], 1)

    def test_source_root_and_every_driver_require_proven_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            _path, manifest = outer_fixture.make_fixture(temp)
            provenance = manifest["provenance"]
            source = Path(provenance["source_root_path"])
            plan = json.loads(Path(manifest["suite"]["plan_path"]).read_bytes())
            foreign = copy.deepcopy(plan)
            foreign["source_root_path"] = str(Path(temp) / "unproven")
            Path(foreign["source_root_path"]).mkdir()
            with self.assertRaisesRegex(ValueError, "source root differs"):
                Q._validate_program_closure(
                    provenance, manifest["suite"], foreign, source
                )
            driver = source / "target" / "untracked-driver"
            driver.parent.mkdir()
            driver.write_bytes(b"#!/bin/sh\nexit 0\n")
            driver.chmod(0o755)
            foreign = copy.deepcopy(plan)
            foreign["gate_jobs"][Q.REQUIRED_GATES[0]] = {
                "program_path": str(driver),
                "program_sha256": Q.sha256_file(driver),
            }
            with self.assertRaisesRegex(ValueError, "proven tracked executable"):
                Q._validate_program_closure(
                    provenance, manifest["suite"], foreign, source
                )
            Q._validate_program_closure(provenance, manifest["suite"], plan, source)

    def test_empty_retrievals_and_per_item_correctness_disagreement_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            _path, manifest = outer_fixture.make_fixture(temp)
            manifest["_runtime_dir"] = manifest["runtime_dir"]
            for gate in ("native-262k", "mmlu-pro", "gsm8k"):
                evidence = outer_fixture.gate_evidence(gate, manifest, b"clean")
                validator = (
                    Path(manifest["runtime_dir"]) / evidence["validator_artifact"]
                )
                document = json.loads(validator.read_bytes())
                if gate == "native-262k":
                    for row in document["raw"]["retrievals"]:
                        row["answers"] = row["expected"] = []
                else:
                    document["raw"]["items"][0]["k5_correct"] = False
                    document["raw"]["items"][-1]["k5_correct"] = True
                validator.write_bytes(Q.canonical_bytes(document))
                evidence["validator_receipt_sha256"] = Q.sha256_file(validator)
                with self.subTest(gate=gate), self.assertRaises(ValueError):
                    Q._validate_gate_evidence(
                        gate, evidence, manifest, outer_fixture.CONTAINER_ID
                    )

    def test_bind_alias_and_nested_mounts_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint, runtime = Path(temp) / "checkpoint", Path(temp) / "runtime"
            checkpoint.mkdir()
            runtime.mkdir()
            info = checkpoint.stat()
            device = f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}"
            base = f"1 0 {device} / / rw - ext4 fixture rw\n"
            alias = (
                base
                + f"2 1 {device} {checkpoint}/child {runtime} rw - ext4 fixture rw\n"
            )
            with mock.patch.object(Path, "read_text", return_value=alias):
                with self.assertRaisesRegex(ValueError, "bind aliases"):
                    Q._require_disjoint_directories(
                        checkpoint, runtime, "checkpoint", "runtime", verify=True
                    )
            nested = base + f"2 1 {device} / {checkpoint}/nested rw - ext4 fixture rw\n"
            with mock.patch.object(Path, "read_text", return_value=nested):
                with self.assertRaisesRegex(ValueError, "nested mounts"):
                    Q._require_disjoint_directories(
                        checkpoint, runtime, "checkpoint", "runtime", verify=True
                    )

    def test_preflight_parent_swap_cannot_create_external_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            _path, manifest = outer_fixture.make_fixture(temp)
            manifest["_runtime_dir"] = manifest["runtime_dir"]
            runtime = Path(manifest["runtime_dir"])
            nested, outside = runtime / "nested", Path(temp) / "outside"
            nested.mkdir()
            outside.mkdir()
            manifest["launch"]["lockfile"] = str(nested / "release.lock")
            Q._prepare_release_launch_paths(manifest)
            nested.rename(runtime / "original")
            nested.symlink_to(outside, target_is_directory=True)
            runner = self.make_runner(manifest)
            with self.assertRaises(OSError):
                runner.run(lambda *_: None)
            self.assertFalse((outside / "release.lock").exists())

    def make_runner(self, manifest, backend=None):
        launch = manifest["launch"]
        return Q.r12.R12ContainerReleaseRunner(
            image_id=manifest["image"]["id"],
            name=launch["name"],
            run_id=launch["run_id"],
            cidfile=Path(launch["cidfile"]),
            lockfile=Path(launch["lockfile"]),
            diagnostics_dir=Path(launch["diagnostics_dir"]),
            extra_run_args=launch["extra_run_args"],
            container_command=launch["container_command"],
            backend=backend,
            retry_interval_s=0,
            sleeper=lambda _delay: None,
        )

    def test_cidfile_and_capture_remain_anchored_after_parent_swap(self):
        with tempfile.TemporaryDirectory() as temp:
            _path, manifest = outer_fixture.make_fixture(temp)
            manifest["_runtime_dir"] = manifest["runtime_dir"]
            runtime = Path(manifest["runtime_dir"])
            cid_parent, original, outside = (
                runtime / "nested",
                runtime / "original",
                Path(temp) / "outside",
            )
            cid_parent.mkdir()
            outside.mkdir()
            manifest["launch"]["cidfile"] = str(cid_parent / "container.cid")

            class SwappingBackend(runner_fixture.FakeDockerBackend):
                def popen(self, argv):
                    cid_parent.rename(original)
                    cid_parent.symlink_to(outside, target_is_directory=True)
                    return super().popen(argv)

            runner = self.make_runner(manifest, SwappingBackend())
            _result, cleanup = runner.run(lambda *_: "offline")
            self.assertTrue(cleanup.lock_safe)
            self.assertTrue((original / "container.cid").is_file())
            self.assertFalse((outside / "container.cid").exists())
            capture = Q._release_diagnostic_capture(
                manifest, outer_fixture._CaptureBackend()
            )
            try:
                diagnostics = Path(manifest["launch"]["diagnostics_dir"])
                diagnostics.rename(runtime / "original-diagnostics")
                diagnostics.symlink_to(outside, target_is_directory=True)
                record = Q.lifecycle.ContainerRecord(
                    container_id=outer_fixture.CONTAINER_ID,
                    name="fixture",
                    image_id=outer_fixture.IMAGE_ID,
                    labels={},
                    running=False,
                )
                capture(record)
                self.assertTrue(
                    (
                        runtime
                        / "original-diagnostics"
                        / (record.container_id + ".inspect.json")
                    ).is_file()
                )
                self.assertFalse(
                    (outside / (record.container_id + ".inspect.json")).exists()
                )
            finally:
                capture.close()

    def test_detached_double_fork_is_rejected_killed_and_reaped(self):
        source = b"""#!/usr/bin/env python3
import os, subprocess
p = subprocess.Popen(["/usr/bin/python3", "-c", "import time; time.sleep(30)"], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(p.pid, flush=True)
"""
        descriptor = suite_fixture.seal_bytes("detached-driver", source, 0o500)
        try:
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as receipt:
                with self.assertRaisesRegex(RuntimeError, "process descendant"):
                    S.run_gate_program(
                        ["fixture"],
                        executable_fd=descriptor,
                        receipt_fd=receipt.fileno(),
                        raw_log_fd=receipt.fileno(),
                        stdout_fd=output.fileno(),
                        stderr_fd=output.fileno(),
                        timeout_seconds=3,
                    )
                pid = int(os.pread(output.fileno(), 100, 0).strip())
                self.assertFalse(Path(f"/proc/{pid}").exists())
                self.assertEqual(S._direct_children(), [])
        finally:
            os.close(descriptor)

    def test_live_broker_requires_two_ordered_distinct_acknowledged_processes(self):
        source = b"""#!/usr/bin/env python3
import json, socket, sys
sock=socket.socket(fileno=int(sys.argv[1]))
for index, mode in enumerate(("none", "native-mtp-k5")):
    sock.send(json.dumps({"schema":"flux-r15-live-process-request/1","mode":mode,"host_pid":index+100}).encode())
    assert json.loads(sock.recv(16384))["status"] == "PASS"
"""
        descriptor = suite_fixture.seal_bytes("proof-driver", source, 0o500)
        plan = {"in_container_binary_sha256": "1" * 64, "recipe_sha256": "2" * 64}

        def verifier(request, _plan, _identities):
            return suite_fixture.fixture_proof(
                request["mode"], request["host_pid"], plan
            )

        try:
            with tempfile.TemporaryFile() as output, mock.patch.object(
                S, "verify_live_server", side_effect=verifier
            ) as verify:
                proofs = S.run_gate_program(
                    ["fixture", S.PROCESS_PROOF_TOKEN],
                    executable_fd=descriptor,
                    receipt_fd=output.fileno(),
                    raw_log_fd=output.fileno(),
                    stdout_fd=output.fileno(),
                    stderr_fd=output.fileno(),
                    timeout_seconds=3,
                    process_proof=(plan, {}),
                )
            self.assertEqual(
                [proof["mode"] for proof in proofs], ["none", "native-mtp-k5"]
            )
            self.assertEqual(verify.call_count, 2)
        finally:
            os.close(descriptor)

    def test_outer_timeout_drains_detached_gate_process(self):
        source = b"""#!/usr/bin/env python3
import subprocess, time
p=subprocess.Popen(["/bin/sleep", "30"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(p.pid, flush=True)
time.sleep(30)
"""
        descriptor = suite_fixture.seal_bytes("timed-out-suite", source, 0o500)
        try:
            with tempfile.TemporaryFile() as output:
                with self.assertRaisesRegex(RuntimeError, "process descendant"):
                    Q.SubprocessSuiteBackend().run(
                        ["fixture"],
                        timeout_s=1,
                        stdout_fd=output.fileno(),
                        stderr_fd=output.fileno(),
                        env={"PATH": "/usr/bin:/bin"},
                        executable_fd=descriptor,
                        plan_fd=output.fileno(),
                        receipt_fd=output.fileno(),
                    )
                pid = int(os.pread(output.fileno(), 100, 0).strip())
                self.assertFalse(Path(f"/proc/{pid}").exists())
                self.assertEqual(S._direct_children(), [])
        finally:
            os.close(descriptor)

    def fake_process(self, root):
        proc_root, container_root = root / "proc", root / "container-root"
        process, init = proc_root / "123", proc_root / "456"
        for directory in (
            process / "ns",
            process / "fd",
            process / "net",
            init / "ns",
            proc_root / "self/ns",
            container_root / "opt/flux/bin",
            container_root / "opt/flux/repro-kit/recipes",
            container_root / "models",
        ):
            directory.mkdir(parents=True)
        binary = container_root / "opt/flux/bin/server"
        binary.write_bytes(b"fixture executable bytes")
        recipe = container_root / "opt/flux/repro-kit/recipes/release.env"
        recipe.write_bytes(b"FLUX_MODE=interactive\n")
        (process / "root").symlink_to(container_root, target_is_directory=True)
        (process / "exe").symlink_to(binary)
        (process / "ns/mnt").write_bytes(b"namespace")
        os.link(process / "ns/mnt", init / "ns/mnt")
        (process / "ns/net").write_bytes(b"network namespace")
        os.link(process / "ns/net", init / "ns/net")
        os.link(process / "ns/net", proc_root / "self/ns/net")
        (process / "fd/7").symlink_to("socket:[9001]")
        header = "  sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
        (process / "net/tcp").write_text(
            header + "0: 0100007F:22B8 00000000:0000 0A 0:0 00:0 0 1000 0 9001 1\n"
        )
        (process / "net/tcp6").write_text(header)
        cgroup = b"0::/docker/" + suite_fixture.CONTAINER_ID.encode() + b"\n"
        for path in (process / "cgroup", init / "cgroup"):
            path.write_bytes(cgroup)
        (process / "stat").write_bytes(b"123 (server) S " + b"0 " * 18 + b"1000\n")
        (process / "mountinfo").write_text(
            "1 0 0:1 / / ro - overlay overlay ro\n"
            "2 1 0:2 / /models ro - ext4 /dev/fixture ro\n"
        )
        (process / "environ").write_bytes(b"FLUX_MODE=interactive\0")
        (process / "cmdline").write_bytes(
            b"server\0serve\0--no-tui\0--model-from-path\0/models\0--speculative\0--num-drafts\0"
            + b"4\0"
        )
        plan = {
            "in_container_binary_path": "/opt/flux/bin/server",
            "in_container_recipe_path": "/opt/flux/repro-kit/recipes/release.env",
            "in_container_binary_sha256": S.sha256_bytes(binary.read_bytes()),
            "recipe_sha256": S.sha256_bytes(recipe.read_bytes()),
        }
        identities = {
            "container_id": suite_fixture.CONTAINER_ID,
            "image_id": suite_fixture.IMAGE_ID,
            "run_id": suite_fixture.RUN_ID,
            "base_url": "http://127.0.0.1:8888",
        }
        document = {
            "Id": identities["container_id"],
            "Image": identities["image_id"],
            "State": {"Running": True, "Pid": 456},
            "HostConfig": {"NetworkMode": "host"},
            "Mounts": [
                {
                    "Type": "bind",
                    "Destination": "/models",
                    "Source": str(container_root / "models"),
                    "RW": False,
                }
            ],
        }
        return proc_root, process, plan, identities, document

    def test_launch_failure_closes_both_proof_channels(self):
        supervisor, driver = mock.Mock(), mock.Mock()
        driver.fileno.return_value = 123
        with mock.patch.object(
            S.socket, "socketpair", return_value=(supervisor, driver)
        ), mock.patch.object(
            S.subprocess, "Popen", side_effect=OSError("launch failed")
        ):
            with self.assertRaisesRegex(OSError, "launch failed"):
                S._run_gate_program_owned(
                    ["fixture", S.PROCESS_PROOF_TOKEN],
                    executable_fd=1,
                    receipt_fd=2,
                    raw_log_fd=3,
                    stdout_fd=4,
                    stderr_fd=5,
                    timeout_seconds=1,
                    process_proof=({}, {}),
                )
        supervisor.close.assert_called_once_with()
        driver.close.assert_called_once_with()

    def test_live_verifier_checks_actual_bytes_namespace_cgroup_recipe_and_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            proc_root, process, plan, identities, document = self.fake_process(
                Path(temp)
            )
            read_fd, write_fd = os.pipe()
            try:
                with mock.patch.object(S, "PROC_ROOT", proc_root), mock.patch.object(
                    S.os, "pidfd_open", side_effect=lambda _pid: os.dup(read_fd)
                ), mock.patch.object(
                    S.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        [], 0, json.dumps(document).encode(), b""
                    ),
                ):
                    request = {
                        "schema": "flux-r15-live-process-request/1",
                        "mode": "native-mtp-k5",
                        "host_pid": 123,
                    }
                    proof = S.verify_live_server(request, plan, identities)
                    self.assertEqual(proof["status"], "PASS")
                    for path in (
                        Path(temp) / "container-root/opt/flux/bin/server",
                        Path(temp)
                        / "container-root/opt/flux/repro-kit/recipes/release.env",
                    ):
                        original = path.read_bytes()
                        path.write_bytes(b"tampered bytes")
                        with self.assertRaisesRegex(ValueError, "hash differs"):
                            S.verify_live_server(request, plan, identities)
                        path.write_bytes(original)
                    for name, replacement, error in (
                        ("environ", b"FLUX_MODE=wrong\0", "environment"),
                        ("cgroup", b"0::/wrong\n", "cgroup"),
                        (
                            "cmdline",
                            b"server\0--no-tui\0--model-from-path\0/models\0--speculative\0--num-drafts\x003\0",
                            "native K5",
                        ),
                        (
                            "cmdline",
                            b"server\0--no-tui\0--model-from-path\0/models\0--speculative\0--num-drafts\x004\0--draft-model=other\0",
                            "native K5",
                        ),
                        (
                            "cmdline",
                            b"server\0--no-tui\0--model-from-path\0/models\0--speculative\0--num-drafts\x004\0--dflash\0",
                            "native K5",
                        ),
                        (
                            "environ",
                            b"FLUX_MODE=interactive\0FLUX_DRAFT_MODEL=other\0",
                            "native K5",
                        ),
                        (
                            "environ",
                            b"FLUX_MODE=interactive\0FLUX_MTP_PATH=other\0",
                            "native K5",
                        ),
                        (
                            "mountinfo",
                            b"1 0 0:1 / / rw - overlay overlay rw\n",
                            "writable mount",
                        ),
                    ):
                        path = process / name
                        original = path.read_bytes()
                        path.write_bytes(replacement)
                        with self.subTest(name=name), self.assertRaisesRegex(
                            ValueError, error
                        ):
                            S.verify_live_server(request, plan, identities)
                        path.write_bytes(original)
                    (process / "ns/mnt").unlink()
                    (process / "ns/mnt").write_bytes(b"other namespace")
                    with self.assertRaisesRegex(ValueError, "namespace"):
                        S.verify_live_server(request, plan, identities)
            finally:
                os.close(read_fd)
                os.close(write_fd)

    def test_live_path_walk_rejects_symlink_parent_into_writable_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "opt/flux").mkdir(parents=True)
            (root / "results").mkdir()
            (root / "results/server").write_bytes(b"untrusted")
            (root / "opt/flux/bin").symlink_to(
                root / "results", target_is_directory=True
            )
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with self.assertRaises(OSError):
                    S._open_process_file(descriptor, "/opt/flux/bin/server")
            finally:
                os.close(descriptor)

    def test_recipe_language_rejects_expansion_duplicates_and_commands(self):
        self.assertEqual(
            S._recipe_environment(b"# note\nexport A='literal value'\n"),
            {b"A": b"literal value"},
        )
        for payload in (
            b"A=$HOME\n",
            b"A=x\nA=y\n",
            b"source elsewhere\n",
            b"A=`command`\n",
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                S._recipe_environment(payload)


if __name__ == "__main__":
    unittest.main()
