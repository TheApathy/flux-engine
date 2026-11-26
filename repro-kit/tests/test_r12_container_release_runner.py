#!/usr/bin/env python3
"""Docker-free fault injection for the concrete R12 container runner."""

from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import unittest
from unittest import mock

REPRO_KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_KIT))

import container_lifecycle_v3 as lifecycle  # noqa: E402
import r12_container_release_runner as release  # noqa: E402

CONTAINER_ID = "b" * 64
IMAGE_ID = "sha256:" + "2" * 64


class ToxicStrError(BaseException):
    def __str__(self):
        raise RuntimeError("toxic-str")


class ToxicTracebackError(BaseException):
    def __getattribute__(self, name):
        if name == "__traceback__":
            raise ToxicTracebackError()
        return super().__getattribute__(name)


class ToxicBool:
    def __bool__(self):
        raise RuntimeError("toxic-bool")


class FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = (CONTAINER_ID + "\n").encode(),
        stderr: bytes = b"",
        pending_polls: int = 0,
    ) -> None:
        self.returncode: int | None = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.pending_polls = pending_polls

    def poll(self):
        if self.pending_polls:
            self.pending_polls -= 1
            return None
        return self.returncode

    def communicate(self, timeout=None):
        del timeout
        return self.stdout, self.stderr


class FakeDockerBackend:
    """Small argv-level Docker model; no subprocess or daemon is used."""

    def __init__(self) -> None:
        self.containers: dict[str, dict] = {}
        self.commands: list[tuple[str, ...]] = []
        self.discovery_delay = 0
        self.discovery_calls = 0
        self.daemon_failure = False
        self.inspect_failure = False
        self.stop_rc = 0
        self.kill_rc = 0
        self.diagnostic_failure = False
        self.launch_process = FakeProcess()
        self.raise_once: BaseException | None = None

    def _visible_ids(self) -> list[str]:
        self.discovery_calls += 1
        if self.discovery_calls <= self.discovery_delay:
            return []
        return sorted(self.containers)

    def popen(self, argv):
        command = tuple(argv)
        self.commands.append(command)
        self.assert_launch_argv(command)
        name = command[command.index("--name") + 1]
        cidfile = Path(command[command.index("--cidfile") + 1])
        labels: dict[str, str] = {}
        for index, value in enumerate(command):
            if value == "--label":
                key, label_value = command[index + 1].split("=", 1)
                labels[key] = label_value
        image_id = next(
            value for value in command if lifecycle.IMAGE_ID_RE.fullmatch(value)
        )
        self.containers[CONTAINER_ID] = {
            "Id": CONTAINER_ID,
            "Name": f"/{name}",
            "Image": image_id,
            "Config": {"Labels": labels},
            "State": {"Running": True},
        }
        cidfile.write_text(CONTAINER_ID + "\n", encoding="ascii")
        return self.launch_process

    def assert_launch_argv(self, argv):
        if argv[:3] != ("docker", "run", "--detach"):
            raise AssertionError(f"unexpected launch argv: {argv!r}")

    def run(self, argv, *, timeout_s):
        del timeout_s
        command = tuple(argv)
        self.commands.append(command)
        if self.raise_once is not None:
            error = self.raise_once
            self.raise_once = None
            raise error
        if self.daemon_failure:
            return release.CommandResult(125, "", "synthetic daemon unavailable")
        if command[:3] == ("docker", "container", "ls"):
            ids = self._visible_ids()
            if "--quiet" in command:
                filters = [
                    command[index + 1].removeprefix("label=")
                    for index, value in enumerate(command)
                    if value == "--filter"
                ]
                matched = []
                for container_id in ids:
                    labels = self.containers[container_id]["Config"]["Labels"]
                    if all(
                        labels.get(item.split("=", 1)[0]) == item.split("=", 1)[1]
                        for item in filters
                    ):
                        matched.append(container_id)
                return release.CommandResult(
                    0, "\n".join(matched) + ("\n" if matched else "")
                )
            rows = [
                f"{container_id}\t{self.containers[container_id]['Name'].removeprefix('/')}"
                for container_id in ids
            ]
            return release.CommandResult(0, "\n".join(rows) + ("\n" if rows else ""))
        if command[:3] == ("docker", "container", "inspect"):
            if self.inspect_failure:
                return release.CommandResult(1, "", "synthetic inspect failure")
            container_id = command[-1]
            document = self.containers.get(container_id)
            if document is None:
                return release.CommandResult(1, "", "not found")
            return release.CommandResult(0, json.dumps(document))
        if command[:3] == ("docker", "container", "stop"):
            if self.stop_rc == 0:
                self.containers[command[-1]]["State"]["Running"] = False
            return release.CommandResult(self.stop_rc, "", "synthetic stop failure")
        if command[:3] == ("docker", "container", "kill"):
            if self.kill_rc == 0:
                self.containers[command[-1]]["State"]["Running"] = False
            return release.CommandResult(self.kill_rc, "", "synthetic kill failure")
        if command[:3] == ("docker", "container", "logs"):
            return release.CommandResult(
                1 if self.diagnostic_failure else 0,
                "server log\n",
                "synthetic log failure",
            )
        raise AssertionError(f"unexpected offline Docker command: {command!r}")


def make_identity(root: Path) -> lifecycle.LaunchIdentity:
    return lifecycle.LaunchIdentity(
        name="flux-r12-owned",
        image_id=IMAGE_ID,
        run_id="offline-r12",
        owner_token="offline-owner",
        cleanup_version=release.R12_CLEANUP_VERSION,
        cidfile=str(root / "launch.cid"),
    )


class DockerLifecycleAdapterTests(unittest.TestCase):
    def test_os_error_detail_does_not_evaluate_hostile_strerror_truthiness(self):
        error = OSError(5, "synthetic")
        error.strerror = ToxicBool()
        detail = release._os_error_detail(error)
        self.assertIn("errno=5", detail)
        self.assertIn("OSError", detail)

    def test_cidfile_missing_malformed_and_symlink_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = make_identity(root)
            adapter = release.DockerLifecycleAdapter(
                identity,
                backend=FakeDockerBackend(),
                diagnostics_dir=root / "diagnostics",
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(adapter.read_cidfile(identity.cidfile).status, "missing")
            Path(identity.cidfile).write_text("short\n", encoding="ascii")
            self.assertEqual(adapter.read_cidfile(identity.cidfile).status, "error")
            Path(identity.cidfile).unlink()
            target = root / "target.cid"
            target.write_text(CONTAINER_ID + "\n", encoding="ascii")
            Path(identity.cidfile).symlink_to(target)
            self.assertEqual(adapter.read_cidfile(identity.cidfile).status, "error")

    def test_daemon_and_inspect_failures_are_never_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = make_identity(root)
            backend = FakeDockerBackend()
            adapter = release.DockerLifecycleAdapter(
                identity,
                backend=backend,
                diagnostics_dir=root / "diagnostics",
                sleeper=lambda _seconds: None,
            )
            backend.daemon_failure = True
            self.assertEqual(adapter.inspect(identity.name).status, "error")
            backend.daemon_failure = False
            backend.containers[CONTAINER_ID] = {
                "Id": CONTAINER_ID,
                "Name": f"/{identity.name}",
                "Image": identity.image_id,
                "Config": {"Labels": identity.ownership_labels},
                "State": {"Running": True},
            }
            backend.inspect_failure = True
            self.assertEqual(adapter.inspect(identity.name).status, "error")

    def test_foreign_exact_name_collision_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = make_identity(root)
            backend = FakeDockerBackend()
            labels = dict(identity.ownership_labels)
            labels[lifecycle.OWNER_LABEL] = "foreign-owner"
            backend.containers[CONTAINER_ID] = {
                "Id": CONTAINER_ID,
                "Name": f"/{identity.name}",
                "Image": identity.image_id,
                "Config": {"Labels": labels},
                "State": {"Running": True},
            }
            adapter = release.DockerLifecycleAdapter(
                identity,
                backend=backend,
                diagnostics_dir=root / "diagnostics",
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )
            receipt = lifecycle.CleanupReceipt()
            with self.assertRaises(lifecycle.UnsafeToUnlock):
                lifecycle.reconcile_before_unlock(
                    adapter,
                    identity,
                    lambda: self.fail("foreign container authorized unlock"),
                    stable_sweeps=2,
                    max_cycles=2,
                    receipt=receipt,
                )
            self.assertTrue(any("identity conflict" in item for item in receipt.errors))
            self.assertFalse(
                any(
                    command[:3] == ("docker", "container", "stop")
                    for command in backend.commands
                )
            )

    def test_stop_falls_back_to_kill_and_double_failure_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = make_identity(root)
            backend = FakeDockerBackend()
            backend.containers[CONTAINER_ID] = {
                "Id": CONTAINER_ID,
                "Name": f"/{identity.name}",
                "Image": identity.image_id,
                "Config": {"Labels": identity.ownership_labels},
                "State": {"Running": True},
            }
            adapter = release.DockerLifecycleAdapter(
                identity,
                backend=backend,
                diagnostics_dir=root / "diagnostics",
                sleeper=lambda _seconds: None,
            )
            backend.stop_rc = 1
            adapter.stop_owned(CONTAINER_ID)
            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])
            backend.containers[CONTAINER_ID]["State"]["Running"] = True
            backend.kill_rc = 1
            with self.assertRaisesRegex(RuntimeError, "stop failed.*kill failed"):
                adapter.stop_owned(CONTAINER_ID)

    def test_stop_and_kill_failure_remains_bounded_test_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = make_identity(root)
            backend = FakeDockerBackend()
            backend.stop_rc = 1
            backend.kill_rc = 1
            backend.containers[CONTAINER_ID] = {
                "Id": CONTAINER_ID,
                "Name": f"/{identity.name}",
                "Image": identity.image_id,
                "Config": {"Labels": identity.ownership_labels},
                "State": {"Running": True},
            }
            adapter = release.DockerLifecycleAdapter(
                identity,
                backend=backend,
                diagnostics_dir=root / "diagnostics",
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )
            receipt = lifecycle.CleanupReceipt()
            released: list[bool] = []
            with self.assertRaises(lifecycle.UnsafeToUnlock):
                lifecycle.reconcile_before_unlock(
                    adapter,
                    identity,
                    lambda: released.append(True),
                    stable_sweeps=2,
                    max_cycles=2,
                    receipt=receipt,
                )
            self.assertEqual(released, [])
            self.assertTrue(any("stop failed" in item for item in receipt.errors))
            self.assertTrue(backend.containers[CONTAINER_ID]["State"]["Running"])

    def test_diagnostic_command_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = make_identity(root)
            backend = FakeDockerBackend()
            backend.diagnostic_failure = True
            adapter = release.DockerLifecycleAdapter(
                identity,
                backend=backend,
                diagnostics_dir=root / "diagnostics",
                sleeper=lambda _seconds: None,
            )
            record = lifecycle.ContainerRecord(
                CONTAINER_ID,
                f"/{identity.name}",
                identity.image_id,
                identity.ownership_labels,
                False,
            )
            # Inspect succeeds first; log capture is then injected to fail.
            backend.containers[CONTAINER_ID] = {
                "Id": CONTAINER_ID,
                "Name": record.name,
                "Image": record.image_id,
                "Config": {"Labels": dict(record.labels)},
                "State": {"Running": False},
            }
            with self.assertRaisesRegex(RuntimeError, "diagnostic server.log"):
                adapter.capture_after_stop(record)

    def test_interrupted_launch_client_is_retained_until_settled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = make_identity(root)
            backend = FakeDockerBackend()
            process = FakeProcess(pending_polls=1)
            adapter = release.DockerLifecycleAdapter(
                identity,
                backend=backend,
                diagnostics_dir=root / "diagnostics",
                sleeper=lambda _seconds: None,
            )
            adapter.launch_process = process
            self.assertFalse(adapter.settle_launch_client())
            self.assertTrue(adapter.settle_launch_client())
            self.assertIsNone(adapter.launch_process)


class R12ContainerReleaseRunnerTests(unittest.TestCase):
    def test_fallback_backoff_baseexception_cannot_escape_unresolved_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            lockfile = root / "gpu.lock"
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=lockfile,
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )
            real_reconcile = lifecycle.reconcile_before_unlock
            reconcile_calls = 0
            sleep_calls = 0

            def escaped_wrapper(body, *_args, **_kwargs):
                body()
                raise RuntimeError("synthetic wrapper escape")

            def flaky_reconcile(*args, **kwargs):
                nonlocal reconcile_calls
                reconcile_calls += 1
                if reconcile_calls == 1:
                    raise RuntimeError("synthetic fallback defect")
                return real_reconcile(*args, **kwargs)

            def toxic_backoff(_seconds):
                nonlocal sleep_calls
                sleep_calls += 1
                if sleep_calls == 1:
                    raise ToxicStrError()

            with (
                mock.patch.object(
                    release.lifecycle, "run_with_cleanup_v3", escaped_wrapper
                ),
                mock.patch.object(
                    release.lifecycle, "reconcile_before_unlock", flaky_reconcile
                ),
                mock.patch.object(release.time, "sleep", toxic_backoff),
                self.assertRaisesRegex(RuntimeError, "synthetic wrapper escape"),
            ):
                runner.run(lambda _identity, container_id: container_id)

            self.assertGreaterEqual(sleep_calls, 1)
            self.assertGreaterEqual(reconcile_calls, 2)
            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])
            descriptor = os.open(lockfile, os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def test_signal_after_spawn_before_popen_return_retains_launch_authority(self):
        class InterruptingPopenBackend(FakeDockerBackend):
            def popen(self, argv):
                process = super().popen(argv)
                Path(argv[argv.index("--cidfile") + 1]).unlink()
                self.discovery_delay = 1000
                signal.raise_signal(signal.SIGTERM)
                return process

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = InterruptingPopenBackend()
            lockfile = root / "gpu.lock"
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=lockfile,
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )

            with self.assertRaises(lifecycle.DeferredTermination) as raised:
                runner.run(lambda _identity, container_id: container_id)

            self.assertEqual(raised.exception.signals, (signal.SIGTERM,))
            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])
            self.assertIn(CONTAINER_ID, runner.adapter.authoritative_launch_ids)
            self.assertTrue(
                any(
                    command[:3] == ("docker", "container", "stop")
                    for command in backend.commands
                )
            )

    def test_signal_before_launch_id_journal_still_stops_exact_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            backend.discovery_delay = 1000
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=root / "gpu.lock",
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )
            original_remember = runner.adapter._remember_successful_launch
            injected = False

            def interrupt_before_remember(result):
                nonlocal injected
                if not injected:
                    injected = True
                    Path(runner.identity.cidfile).unlink()
                    signal.raise_signal(signal.SIGINT)
                return original_remember(result)

            runner.adapter._remember_successful_launch = interrupt_before_remember
            with self.assertRaises(lifecycle.DeferredTermination) as raised:
                runner.run(lambda _identity, container_id: container_id)

            self.assertEqual(raised.exception.signals, (signal.SIGINT,))
            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])
            self.assertIn(CONTAINER_ID, runner.adapter.authoritative_launch_ids)

    def test_list_inspect_contradiction_cannot_prove_absence_or_unlock_live(self):
        class ContradictingBackend(FakeDockerBackend):
            def __init__(self):
                super().__init__()
                self.contradictory_inspects = 0

            def run(self, argv, *, timeout_s):
                command = tuple(argv)
                if (
                    command[:3] == ("docker", "container", "inspect")
                    and self.contradictory_inspects > 0
                ):
                    self.commands.append(command)
                    self.contradictory_inspects -= 1
                    return release.CommandResult(
                        1, "", f"Error: No such object: {command[-1]}"
                    )
                return super().run(argv, timeout_s=timeout_s)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = ContradictingBackend()
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=root / "gpu.lock",
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )

            def qualify(identity, container_id):
                del container_id
                Path(identity.cidfile).unlink()
                backend.contradictory_inspects = 4
                return "qualified"

            result, receipt = runner.run(qualify)

            self.assertEqual(result, "qualified")
            self.assertTrue(receipt.lock_released)
            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])
            self.assertTrue(
                any("contradicted positive" in error for error in receipt.errors),
                receipt.errors,
            )
            self.assertTrue(
                any(
                    command[:3] == ("docker", "container", "stop")
                    for command in backend.commands
                )
            )

    def test_runner_refuses_worker_thread_before_lock_or_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            lockfile = root / "gpu.lock"
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=lockfile,
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                owner_token="offline-owner",
            )
            failures: list[BaseException] = []

            def target() -> None:
                try:
                    runner.run(lambda _identity, container_id: container_id)
                except BaseException as exc:
                    failures.append(exc)

            worker = threading.Thread(target=target)
            worker.start()
            worker.join(1.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(backend.commands, [])
            self.assertFalse(lockfile.exists())
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], RuntimeError)
            self.assertIn("main thread", str(failures[0]))

    def test_fallback_merges_inner_and_cleanup_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=root / "gpu.lock",
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )
            real_reconcile = lifecycle.reconcile_before_unlock
            injected = False

            def escaped_wrapper(body, *_args, **_kwargs):
                body()
                raise lifecycle.DeferredTermination((signal.SIGINT,))

            def reconcile_with_cleanup_signal(*args, **kwargs):
                nonlocal injected
                if not injected:
                    injected = True
                    signal.raise_signal(signal.SIGTERM)
                return real_reconcile(*args, **kwargs)

            with (
                mock.patch.object(
                    release.lifecycle, "run_with_cleanup_v3", escaped_wrapper
                ),
                mock.patch.object(
                    release.lifecycle,
                    "reconcile_before_unlock",
                    reconcile_with_cleanup_signal,
                ),
                self.assertRaises(lifecycle.DeferredTermination) as raised,
            ):
                runner.run(lambda _identity, container_id: container_id)

            self.assertEqual(raised.exception.signals, (signal.SIGINT, signal.SIGTERM))
            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])

    def test_toxic_traceback_wrapper_error_cannot_escape_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            lockfile = root / "gpu.lock"
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=lockfile,
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )

            def escaped_wrapper(body, *_args, **_kwargs):
                body()
                raise ToxicTracebackError()

            with (
                mock.patch.object(
                    release.lifecycle, "run_with_cleanup_v3", escaped_wrapper
                ),
                self.assertRaises(ToxicTracebackError),
            ):
                runner.run(lambda _identity, container_id: container_id)

            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])
            descriptor = os.open(lockfile, os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def test_signal_during_outer_gate_exit_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            lockfile = root / "gpu.lock"
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=lockfile,
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )
            original_gate = lifecycle._TerminationGate

            class ExitInterruptingGate(original_gate):
                injected = False

                def __exit__(self, exc_type, exc, traceback):
                    if exc is None and not self.injected:
                        self.injected = True
                        signal.raise_signal(signal.SIGINT)
                    return super().__exit__(exc_type, exc, traceback)

            with (
                mock.patch.object(
                    release.lifecycle, "_TerminationGate", ExitInterruptingGate
                ),
                self.assertRaises(lifecycle.DeferredTermination) as raised,
            ):
                runner.run(lambda _identity, container_id: container_id)

            self.assertEqual(raised.exception.signals, (signal.SIGINT,))
            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])
            descriptor = os.open(lockfile, os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def test_signal_before_fallback_cleanup_phase_still_reconciles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            lockfile = root / "gpu.lock"
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=lockfile,
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )

            def escaped_wrapper(body, *_args, **_kwargs):
                body()
                raise RuntimeError("synthetic wrapper escape")

            original_gate = lifecycle._TerminationGate

            class InterruptingGate(original_gate):
                injected = False

                def begin_cleanup(self):
                    if not self.injected:
                        self.injected = True
                        signal.raise_signal(signal.SIGINT)
                    super().begin_cleanup()

            with (
                mock.patch.object(
                    release.lifecycle, "run_with_cleanup_v3", escaped_wrapper
                ),
                mock.patch.object(
                    release.lifecycle, "_TerminationGate", InterruptingGate
                ),
                self.assertRaisesRegex(RuntimeError, "synthetic wrapper escape"),
            ):
                runner.run(lambda _identity, container_id: container_id)

            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])
            descriptor = os.open(lockfile, os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def test_fallback_defect_and_signal_during_backoff_keep_cleanup_gated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            lockfile = root / "gpu.lock"
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=lockfile,
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )
            real_reconcile = lifecycle.reconcile_before_unlock
            reconcile_calls = 0
            sleep_calls = 0

            def escaped_wrapper(body, *_args, **_kwargs):
                body()
                raise RuntimeError("synthetic wrapper escape")

            def flaky_reconcile(*args, **kwargs):
                nonlocal reconcile_calls
                reconcile_calls += 1
                if reconcile_calls == 1:
                    raise RuntimeError("synthetic fallback defect")
                return real_reconcile(*args, **kwargs)

            def interrupted_backoff(_seconds):
                nonlocal sleep_calls
                sleep_calls += 1
                signal.raise_signal(signal.SIGINT)

            with (
                mock.patch.object(
                    release.lifecycle, "run_with_cleanup_v3", escaped_wrapper
                ),
                mock.patch.object(
                    release.lifecycle, "reconcile_before_unlock", flaky_reconcile
                ),
                mock.patch.object(release.time, "sleep", interrupted_backoff),
                self.assertRaisesRegex(RuntimeError, "synthetic wrapper escape"),
            ):
                runner.run(lambda _identity, container_id: container_id)

            self.assertGreaterEqual(sleep_calls, 1)
            self.assertGreaterEqual(reconcile_calls, 2)
            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])

            descriptor = os.open(lockfile, os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def test_successful_launch_id_bypasses_delayed_list_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            backend.discovery_delay = 1000
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=root / "gpu.lock",
                diagnostics_dir=root / "diagnostics",
                extra_run_args=("--gpus", "all"),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )

            result, receipt = runner.run(lambda _identity, container_id: container_id)

            self.assertEqual(result, CONTAINER_ID)
            self.assertTrue(receipt.lock_released)
            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])
            self.assertTrue(
                any(
                    command[:3] == ("docker", "container", "stop")
                    for command in backend.commands
                )
            )

    def test_toxic_exception_text_cannot_escape_post_launch_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=root / "gpu.lock",
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=("serve",),
                backend=backend,
                diagnostic_capture=lambda _record: None,
                owner_token="offline-owner",
                stable_sweeps=2,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )

            def qualify(_identity, container_id):
                backend.raise_once = ToxicStrError()
                return container_id

            result, receipt = runner.run(qualify)

            self.assertEqual(result, CONTAINER_ID)
            self.assertTrue(receipt.lock_released)
            self.assertFalse(backend.containers[CONTAINER_ID]["State"]["Running"])
            self.assertTrue(
                any("ToxicStrError" in error for error in receipt.errors),
                receipt.errors,
            )

    def test_delayed_creation_end_to_end_stops_before_diagnostics_and_unlocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeDockerBackend()
            backend.discovery_delay = 2
            diagnostics: list[str] = []
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=root / "launch.cid",
                lockfile=root / "gpu.lock",
                diagnostics_dir=root / "diagnostics",
                extra_run_args=("--gpus", "all"),
                container_command=("serve", "/models/checkpoint"),
                backend=backend,
                diagnostic_capture=lambda record: diagnostics.append(
                    record.container_id
                ),
                owner_token="offline-owner",
                stable_sweeps=3,
                retry_interval_s=0,
                sleeper=lambda _seconds: None,
            )
            qualified: list[str] = []

            result, receipt = runner.run(
                lambda identity, container_id: (
                    qualified.append(identity.run_id),
                    container_id,
                )[1]
            )

            self.assertEqual(result, CONTAINER_ID)
            self.assertEqual(qualified, ["offline-r12"])
            self.assertTrue(receipt.lock_safe)
            self.assertTrue(receipt.lock_released)
            self.assertEqual(diagnostics, [CONTAINER_ID])
            stop_index = next(
                index
                for index, command in enumerate(backend.commands)
                if command[:3] == ("docker", "container", "stop")
            )
            self.assertGreater(len(backend.commands), stop_index + 1)
            launch = backend.commands[0]
            self.assertIn(IMAGE_ID, launch)
            self.assertNotIn("--rm", launch)
            self.assertEqual(launch.count("--restart=no"), 1)

    def test_preexisting_cidfile_refuses_before_docker_or_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cidfile = root / "launch.cid"
            cidfile.write_text(CONTAINER_ID + "\n", encoding="ascii")
            backend = FakeDockerBackend()
            runner = release.R12ContainerReleaseRunner(
                image_id=IMAGE_ID,
                name="flux-r12-owned",
                run_id="offline-r12",
                cidfile=cidfile,
                lockfile=root / "gpu.lock",
                diagnostics_dir=root / "diagnostics",
                extra_run_args=(),
                container_command=(),
                backend=backend,
                owner_token="offline-owner",
            )
            with self.assertRaisesRegex(RuntimeError, "pre-existing"):
                runner.run(lambda _identity, _container_id: None)
            self.assertEqual(backend.commands, [])

    def test_launch_rejects_runner_owned_docker_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = make_identity(root)
            adapter = release.DockerLifecycleAdapter(
                identity,
                backend=FakeDockerBackend(),
                diagnostics_dir=root / "diagnostics",
            )
            for bad in (
                "--rm",
                "--name=foreign",
                "--cidfile",
                "--label=x=y",
                "--restart",
                "--restart=always",
                "-d",
                "ubuntu:latest",
                "--privileged",
            ):
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        adapter.launch((bad,), ())

    def test_read_only_rootfs_is_an_exact_value_less_flag(self):
        self.assertEqual(
            release.DockerLifecycleAdapter._validate_extra_run_args(
                ("--read-only", "--gpus", "all")
            ),
            ("--read-only", "--gpus", "all"),
        )
        for bad in ("--read-only=true", "--read-only=false"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                release.DockerLifecycleAdapter._validate_extra_run_args((bad,))


if __name__ == "__main__":
    unittest.main()
