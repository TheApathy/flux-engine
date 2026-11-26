#!/usr/bin/env python3
"""Adversarial, Docker-free tests for container lifecycle-v3."""

from __future__ import annotations

import sys
import signal
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

REPRO_KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_KIT))

import container_lifecycle_v3 as lifecycle  # noqa: E402

CONTAINER_ID = "a" * 64
IMAGE_ID = "sha256:" + "1" * 64


class FatalCleanup(BaseException):
    pass


class ToxicStrError(BaseException):
    def __str__(self):
        raise RuntimeError("toxic-str")


class ToxicTracebackError(BaseException):
    def __getattribute__(self, name):
        if name == "__traceback__":
            raise ToxicTracebackError()
        return super().__getattribute__(name)


class ToxicTypeNameMeta(type):
    def __getattribute__(cls, name):
        if name == "__name__":
            raise RuntimeError("toxic type name")
        return super().__getattribute__(name)


class ToxicTypeNameError(BaseException, metaclass=ToxicTypeNameMeta):
    pass


class ToxicRenderedName:
    def __format__(self, _format_spec):
        raise RuntimeError("toxic formatted type name")


class ToxicNonStringTypeNameMeta(type):
    def __getattribute__(cls, name):
        if name == "__name__":
            return ToxicRenderedName()
        return super().__getattribute__(name)


class ToxicNonStringTypeNameError(
    BaseException, metaclass=ToxicNonStringTypeNameMeta
):
    pass


def launch_identity() -> lifecycle.LaunchIdentity:
    return lifecycle.LaunchIdentity(
        name="flux-release-owned",
        image_id=IMAGE_ID,
        run_id="offline-run",
        owner_token="offline-owner-token",
        cleanup_version="release-cleanup-v3",
        cidfile="/offline/launch.cid",
    )


class FakeAdapter:
    """Cycle-driven fake that never invokes Docker or the network."""

    def __init__(self, identity: lifecycle.LaunchIdentity) -> None:
        self.identity = identity
        self.cycle = 0
        self.create_at_cycle: int | None = None
        self.running = False
        self.stopped = False
        self.cidfile = lifecycle.CidfileObservation.missing()
        self.inspect_error = False
        self.raise_inspect_once: BaseException | None = None
        self.raise_stop: BaseException | None = None
        self.raise_diagnostics: BaseException | None = None
        self.launch_settled = True
        self.events: list[str] = []
        self.sweep_labels: list[dict[str, str]] = []
        self.wait_seen = threading.Event()

    def settle_launch_client(self) -> bool:
        self.events.append("settle")
        return self.launch_settled

    def read_cidfile(self, path: str) -> lifecycle.CidfileObservation:
        self.events.append(f"cidfile:{path}")
        return self.cidfile

    def _record(self) -> lifecycle.ContainerRecord | None:
        if self.create_at_cycle is None or self.cycle < self.create_at_cycle:
            return None
        return lifecycle.ContainerRecord(
            container_id=CONTAINER_ID,
            name=self.identity.name,
            image_id=self.identity.image_id,
            labels=self.identity.ownership_labels,
            running=self.running,
        )

    def inspect(self, ref: str) -> lifecycle.InspectObservation:
        self.events.append(f"inspect:{ref}")
        if self.raise_inspect_once is not None:
            error = self.raise_inspect_once
            self.raise_inspect_once = None
            raise error
        if self.inspect_error:
            return lifecycle.InspectObservation.failed("synthetic daemon failure")
        record = self._record()
        if record is not None and ref in (record.container_id, record.name):
            return lifecycle.InspectObservation.present(record)
        return lifecycle.InspectObservation.absent()

    def sweep_by_labels(self, labels):
        self.events.append("label-sweep")
        self.sweep_labels.append(dict(labels))
        if self.inspect_error:
            return [lifecycle.InspectObservation.failed("synthetic sweep failure")]
        record = self._record()
        if record is None:
            return []
        return [lifecycle.InspectObservation.present(record)]

    def stop_owned(self, container_id: str) -> None:
        self.events.append(f"stop:{container_id}")
        if self.raise_stop is not None:
            raise self.raise_stop
        self.running = False
        self.stopped = True

    def capture_after_stop(self, record: lifecycle.ContainerRecord) -> None:
        self.events.append(f"diagnostics:{record.container_id}")
        if self.raise_diagnostics is not None:
            raise self.raise_diagnostics
        if record.running is not False:
            raise AssertionError("diagnostics ran before stopped-state confirmation")

    def wait(self) -> None:
        self.events.append("wait")
        self.cycle += 1
        self.wait_seen.set()
        time.sleep(0.001)


class CleanupV3Tests(unittest.TestCase):
    def test_toxic_traceback_body_error_cleans_before_propagation(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        adapter.create_at_cycle = 0
        adapter.running = True
        released: list[str] = []

        def body():
            raise ToxicTracebackError()

        with self.assertRaises(ToxicTracebackError):
            lifecycle.run_with_cleanup_v3(
                body,
                adapter,
                identity,
                lambda: released.append("released"),
                stable_sweeps=2,
            )
        self.assertEqual(released, ["released"])
        self.assertTrue(adapter.stopped)

    def test_gate_enter_rolls_back_both_handlers_after_install_signal(self):
        previous = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }
        real_signal = signal.signal
        injected = False

        def install_then_interrupt(signum, handler):
            nonlocal injected
            result = real_signal(signum, handler)
            if signum == signal.SIGINT and not injected:
                injected = True
                signal.raise_signal(signal.SIGINT)
            return result

        with (
            mock.patch.object(lifecycle.signal, "signal", install_then_interrupt),
            self.assertRaises(lifecycle.DeferredTermination),
        ):
            with lifecycle._TerminationGate():
                self.fail("a pending install signal must prevent body entry")

        self.assertIs(signal.getsignal(signal.SIGINT), previous[signal.SIGINT])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous[signal.SIGTERM])

    def test_gate_exit_restores_both_handlers_before_pending_signal_fires(self):
        previous = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }
        gate = lifecycle._TerminationGate()
        gate.__enter__()
        gate.begin_cleanup()
        real_signal = signal.signal
        injected = False

        def restore_then_interrupt(signum, handler):
            nonlocal injected
            result = real_signal(signum, handler)
            if signum == signal.SIGINT and not injected:
                injected = True
                signal.raise_signal(signal.SIGINT)
            return result

        with (
            mock.patch.object(lifecycle.signal, "signal", restore_then_interrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            gate.__exit__(None, None, None)

        self.assertIs(signal.getsignal(signal.SIGINT), previous[signal.SIGINT])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous[signal.SIGTERM])

    def test_signal_during_gate_exit_is_propagated_after_handler_restore(self):
        original_gate = lifecycle._TerminationGate

        class ExitInterruptingGate(original_gate):
            injected = False

            def __exit__(self, exc_type, exc, traceback):
                if exc is None and not self.injected:
                    self.injected = True
                    signal.raise_signal(signal.SIGINT)
                return super().__exit__(exc_type, exc, traceback)

        identity = launch_identity()
        adapter = FakeAdapter(identity)
        released: list[str] = []
        with (
            mock.patch.object(lifecycle, "_TerminationGate", ExitInterruptingGate),
            self.assertRaises(lifecycle.DeferredTermination) as raised,
        ):
            lifecycle.run_with_cleanup_v3(
                lambda: "body-complete",
                adapter,
                identity,
                lambda: released.append("released"),
                stable_sweeps=2,
            )
        self.assertEqual(raised.exception.signals, (signal.SIGINT,))
        self.assertEqual(released, ["released"])

    def test_nested_gate_exit_preserves_body_error_and_all_signals(self):
        previous = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }
        outer = lifecycle._TerminationGate()
        real_signal = signal.signal
        injected = False

        def restore_parent_then_interrupt(signum, handler):
            nonlocal injected
            result = real_signal(signum, handler)
            if (
                signum == signal.SIGINT
                and getattr(handler, "__self__", None) is outer
                and not injected
            ):
                injected = True
                # The inner exit has both signals blocked.  This becomes
                # pending until the complete outer handler set is restored.
                signal.raise_signal(signal.SIGINT)
            return result

        with self.assertRaisesRegex(RuntimeError, "body failure"):
            with outer:
                with mock.patch.object(
                    lifecycle.signal, "signal", restore_parent_then_interrupt
                ):
                    with lifecycle._TerminationGate() as inner:
                        inner.begin_cleanup()
                        signal.raise_signal(signal.SIGTERM)
                        raise RuntimeError("body failure")

        self.assertEqual(outer.received, [signal.SIGTERM, signal.SIGINT])
        self.assertIs(signal.getsignal(signal.SIGINT), previous[signal.SIGINT])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous[signal.SIGTERM])

    def test_public_identity_hashes_secret_and_host_cidfile_path(self):
        identity = launch_identity()
        public = identity.public()
        self.assertNotIn(identity.owner_token, repr(public))
        self.assertNotIn(identity.cidfile, repr(public))
        self.assertIn("owner_token_sha256", public)
        self.assertIn("cidfile_sha256", public)

    def test_launch_identity_rejects_path_like_names(self):
        for name in ("/owned", "nested/owned", ".", ".."):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "non-path Docker name"
            ):
                lifecycle.LaunchIdentity(
                    name=name,
                    image_id=IMAGE_ID,
                    run_id="offline",
                    owner_token="token",
                    cleanup_version="v3",
                    cidfile="/offline/cid",
                )

    def test_name_accepts_only_exact_plain_or_single_slash_form(self):
        identity = launch_identity()
        for accepted in (identity.name, f"/{identity.name}"):
            with self.subTest(accepted=accepted):
                record = lifecycle.ContainerRecord(
                    CONTAINER_ID,
                    accepted,
                    identity.image_id,
                    identity.ownership_labels,
                    False,
                )
                self.assertTrue(lifecycle._ownership(identity, record)[0])
        for rejected in (
            f"//{identity.name}",
            f"///{identity.name}",
            f"/{identity.name}/",
            f"prefix-{identity.name}",
        ):
            with self.subTest(rejected=rejected):
                record = lifecycle.ContainerRecord(
                    CONTAINER_ID,
                    rejected,
                    identity.image_id,
                    identity.ownership_labels,
                    False,
                )
                self.assertFalse(lifecycle._ownership(identity, record)[0])

    def test_delayed_create_after_first_absence_is_stopped_before_diagnostics(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        adapter.create_at_cycle = 1
        adapter.running = True
        released: list[bool] = []

        receipt = lifecycle.reconcile_before_unlock(
            adapter,
            identity,
            lambda: (adapter.events.append("release"), released.append(True)),
            stable_sweeps=2,
        )

        self.assertTrue(receipt.lock_safe)
        self.assertTrue(receipt.lock_released)
        self.assertEqual(released, [True])
        self.assertTrue(adapter.stopped)
        self.assertIn("absent-sweep:1", receipt.events)
        self.assertLess(
            adapter.events.index(f"stop:{CONTAINER_ID}"),
            adapter.events.index(f"diagnostics:{CONTAINER_ID}"),
        )
        self.assertLess(
            adapter.events.index(f"diagnostics:{CONTAINER_ID}"),
            adapter.events.index("release"),
        )
        self.assertEqual(adapter.sweep_labels[-1], identity.ownership_labels)

    def test_unreadable_cidfile_is_fail_closed(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        adapter.cidfile = lifecycle.CidfileObservation.failed("permission denied")
        released: list[bool] = []

        with self.assertRaises(lifecycle.UnsafeToUnlock):
            lifecycle.reconcile_before_unlock(
                adapter,
                identity,
                lambda: released.append(True),
                stable_sweeps=2,
                max_cycles=3,
            )

        self.assertEqual(released, [])

    def test_cleanup_baseexception_is_quarantined_then_owned_container_stops(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        adapter.create_at_cycle = 0
        adapter.running = True
        adapter.raise_inspect_once = FatalCleanup("synthetic cleanup defect")
        released: list[bool] = []

        receipt = lifecycle.reconcile_before_unlock(
            adapter,
            identity,
            lambda: released.append(True),
            stable_sweeps=2,
        )

        self.assertEqual(released, [True])
        self.assertTrue(adapter.stopped)
        self.assertIn("quarantine-baseexception", receipt.events)
        self.assertTrue(
            any("FatalCleanup" in error for error in receipt.errors), receipt.errors
        )

    def test_unresolved_reconcile_keeps_worker_alive_and_lock_held(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        adapter.inspect_error = True
        released: list[bool] = []
        completed: list[str] = []
        failures: list[BaseException] = []

        def target() -> None:
            try:
                lifecycle.reconcile_before_unlock(
                    adapter,
                    identity,
                    lambda: released.append(True),
                    stable_sweeps=2,
                )
                completed.append("returned")
            except BaseException as exc:  # pragma: no cover - diagnostic guard
                failures.append(exc)

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        self.assertTrue(adapter.wait_seen.wait(1.0))
        time.sleep(0.02)
        self.assertTrue(worker.is_alive())
        self.assertEqual(released, [])
        self.assertEqual(completed, [])

        adapter.inspect_error = False
        worker.join(2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(released, [True])
        self.assertEqual(completed, ["returned"])
        self.assertEqual(failures, [])

    def test_cleanup_wrapper_refuses_worker_thread_before_body(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        body_calls: list[bool] = []
        failures: list[BaseException] = []

        def target() -> None:
            try:
                lifecycle.run_with_cleanup_v3(
                    lambda: body_calls.append(True),
                    adapter,
                    identity,
                    lambda: None,
                    stable_sweeps=2,
                )
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=target)
        worker.start()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(body_calls, [])
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RuntimeError)
        self.assertIn("main thread", str(failures[0]))

    def test_launch_client_must_settle_before_absence_counts(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        settle_calls = 0

        def settle() -> bool:
            nonlocal settle_calls
            settle_calls += 1
            return settle_calls >= 3

        adapter.settle_launch_client = settle
        receipt = lifecycle.reconcile_before_unlock(
            adapter, identity, lambda: None, stable_sweeps=2
        )
        self.assertGreaterEqual(settle_calls, 3)
        self.assertEqual(receipt.stable_absence_sweeps, 2)

    def test_body_baseexception_is_re_raised_only_after_safe_unlock(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        released: list[str] = []

        def body():
            raise FatalCleanup("body interrupted")

        with self.assertRaisesRegex(FatalCleanup, "body interrupted"):
            lifecycle.run_with_cleanup_v3(
                body,
                adapter,
                identity,
                lambda: released.append("released"),
                stable_sweeps=2,
            )
        self.assertEqual(released, ["released"])

    def test_signal_during_body_reconciles_before_propagation(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        released: list[str] = []

        def body():
            signal.raise_signal(signal.SIGTERM)
            self.fail("signal handler did not unwind the body")

        with self.assertRaises(lifecycle.DeferredTermination) as raised:
            lifecycle.run_with_cleanup_v3(
                body,
                adapter,
                identity,
                lambda: released.append("released"),
                stable_sweeps=2,
            )
        self.assertEqual(raised.exception.signals, (signal.SIGTERM,))
        self.assertEqual(released, ["released"])

    def test_signal_in_body_return_transition_cannot_escape_cleanup(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        released: list[str] = []
        original_gate = lifecycle._TerminationGate

        class TransitionSignalGate(original_gate):
            injected = False

            def begin_cleanup(self):
                if not self.injected:
                    self.injected = True
                    self._handle(signal.SIGTERM, None)
                super().begin_cleanup()

        lifecycle._TerminationGate = TransitionSignalGate
        try:
            with self.assertRaises(lifecycle.DeferredTermination):
                lifecycle.run_with_cleanup_v3(
                    lambda: "body-complete",
                    adapter,
                    identity,
                    lambda: released.append("released"),
                    stable_sweeps=2,
                )
        finally:
            lifecycle._TerminationGate = original_gate
        self.assertEqual(released, ["released"])

    def test_repeated_cleanup_and_unlock_signals_are_deferred_and_deduplicated(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        released: list[str] = []
        original_wait = adapter.wait

        def wait():
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGTERM)
            original_wait()

        def release():
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
            released.append("released")

        adapter.wait = wait
        with self.assertRaises(lifecycle.DeferredTermination) as raised:
            lifecycle.run_with_cleanup_v3(
                lambda: "body-complete",
                adapter,
                identity,
                release,
                stable_sweeps=2,
            )
        self.assertEqual(raised.exception.signals, (signal.SIGTERM, signal.SIGINT))
        self.assertEqual(released, ["released"])

    def test_quarantine_receipt_is_deduplicated_and_retry_has_minimum_backoff(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        adapter.inspect_error = True
        receipt = lifecycle.CleanupReceipt()
        started = time.monotonic()
        with self.assertRaises(lifecycle.UnsafeToUnlock):
            lifecycle.reconcile_before_unlock(
                adapter,
                identity,
                lambda: self.fail("unresolved quarantine released the lock"),
                stable_sweeps=2,
                max_cycles=3,
                receipt=receipt,
            )
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, lifecycle.MIN_RETRY_BACKOFF_S * 1.8)
        self.assertLessEqual(len(receipt.errors), 2)
        self.assertLessEqual(len(receipt.events), 1)
        self.assertGreater(receipt.suppressed_errors, 0)
        self.assertGreater(receipt.suppressed_events, 0)

    def test_receipt_caps_unique_error_and_event_storage(self):
        receipt = lifecycle.CleanupReceipt()
        for index in range(lifecycle.MAX_RECEIPT_ERRORS + 20):
            lifecycle._error(receipt, f"error-{index}-" + "x" * 1000)
        for index in range(lifecycle.MAX_RECEIPT_EVENTS + 20):
            lifecycle._event(receipt, f"event-{index}")
        self.assertEqual(len(receipt.errors), lifecycle.MAX_RECEIPT_ERRORS)
        self.assertEqual(len(receipt.events), lifecycle.MAX_RECEIPT_EVENTS)
        self.assertTrue(
            all(
                len(value) <= lifecycle.MAX_RECEIPT_TEXT_CHARS
                for value in receipt.errors
            )
        )
        self.assertEqual(receipt.suppressed_errors, 20)
        self.assertEqual(receipt.suppressed_events, 20)

    def test_diagnostic_failure_is_recorded_but_stopped_container_unlocks(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        adapter.create_at_cycle = 0
        adapter.running = False
        adapter.raise_diagnostics = FatalCleanup("synthetic diagnostic failure")
        released: list[bool] = []
        receipt = lifecycle.reconcile_before_unlock(
            adapter,
            identity,
            lambda: released.append(True),
            stable_sweeps=2,
        )
        self.assertEqual(released, [True])
        self.assertTrue(receipt.lock_safe)
        self.assertTrue(any("diagnostic failure" in item for item in receipt.errors))

    def test_exact_name_with_wrong_token_never_gets_stopped_or_unlocked(self):
        identity = launch_identity()
        adapter = FakeAdapter(identity)
        adapter.create_at_cycle = 0
        adapter.running = True

        def wrong_record():
            labels = dict(identity.ownership_labels)
            labels[lifecycle.OWNER_LABEL] = "somebody-else"
            return lifecycle.ContainerRecord(
                CONTAINER_ID,
                identity.name,
                identity.image_id,
                labels,
                True,
            )

        adapter._record = wrong_record
        released: list[bool] = []
        with self.assertRaises(lifecycle.UnsafeToUnlock):
            lifecycle.reconcile_before_unlock(
                adapter,
                identity,
                lambda: released.append(True),
                stable_sweeps=2,
                max_cycles=3,
            )
        self.assertFalse(adapter.stopped)
        self.assertEqual(released, [])


class MemoryFloorTests(unittest.TestCase):
    def test_non_string_fatal_sampler_type_name_is_memfloor_error(self):
        fatal_seen = threading.Event()

        def sample() -> float:
            fatal_seen.set()
            raise ToxicNonStringTypeNameError()

        monitor = lifecycle.MemoryFloorMonitor(
            sample, floor_mib=6656.0, interval_s=0.001
        )
        started = monitor.start()
        self.assertTrue(fatal_seen.wait(1.0))
        assert monitor._thread is not None
        monitor._thread.join(1.0)
        receipt = monitor.finish(
            arm_started_s=started,
            arm_finished_s=time.monotonic(),
            arm_completed=True,
        )
        self.assertEqual(receipt["status"], "ERROR")
        self.assertTrue(
            any("fatal memory sampler" in item for item in receipt["errors"]),
            receipt["errors"],
        )

    def test_toxic_fatal_sampler_type_name_is_memfloor_error(self):
        fatal_seen = threading.Event()

        def sample() -> float:
            fatal_seen.set()
            raise ToxicTypeNameError()

        monitor = lifecycle.MemoryFloorMonitor(
            sample, floor_mib=6656.0, interval_s=0.001
        )
        started = monitor.start()
        self.assertTrue(fatal_seen.wait(1.0))
        assert monitor._thread is not None
        monitor._thread.join(1.0)
        receipt = monitor.finish(
            arm_started_s=started,
            arm_finished_s=time.monotonic(),
            arm_completed=True,
        )
        self.assertEqual(receipt["status"], "ERROR")
        self.assertTrue(
            any("fatal memory sampler" in item for item in receipt["errors"]),
            receipt["errors"],
        )

    def test_toxic_fatal_sampler_text_is_memfloor_error(self):
        fatal_seen = threading.Event()

        def sample() -> float:
            fatal_seen.set()
            raise ToxicStrError()

        monitor = lifecycle.MemoryFloorMonitor(
            sample, floor_mib=6656.0, interval_s=0.001
        )
        started = monitor.start()
        self.assertTrue(fatal_seen.wait(1.0))
        assert monitor._thread is not None
        monitor._thread.join(1.0)
        receipt = monitor.finish(
            arm_started_s=started,
            arm_finished_s=time.monotonic(),
            arm_completed=True,
        )
        self.assertEqual(receipt["status"], "ERROR")
        self.assertTrue(
            any("ToxicStrError" in item for item in receipt["errors"]),
            receipt["errors"],
        )

    def test_fatal_sampler_is_memfloor_error(self):
        fatal_seen = threading.Event()
        calls = 0

        def sample() -> float:
            nonlocal calls
            calls += 1
            if calls == 1:
                return 7000.0
            fatal_seen.set()
            raise FatalCleanup("synthetic sampler death")

        monitor = lifecycle.MemoryFloorMonitor(
            sample, floor_mib=6656.0, interval_s=0.001
        )
        started = monitor.start()
        self.assertTrue(fatal_seen.wait(1.0))
        receipt = monitor.finish(
            arm_started_s=started,
            arm_finished_s=time.monotonic(),
            arm_completed=True,
        )
        self.assertEqual(receipt["status"], "ERROR")
        self.assertTrue(
            any("fatal memory sampler" in item for item in receipt["errors"])
        )

    def test_clean_premature_sampler_death_is_memfloor_error(self):
        class PrematureMonitor(lifecycle.MemoryFloorMonitor):
            def _worker(self) -> None:
                self._premature_exit = True

        monitor = PrematureMonitor(lambda: 7000.0, floor_mib=6656.0)
        started = monitor.start()
        assert monitor._thread is not None
        monitor._thread.join(1.0)
        receipt = monitor.finish(
            arm_started_s=started,
            arm_finished_s=time.monotonic(),
            arm_completed=True,
        )
        self.assertEqual(receipt["status"], "ERROR")
        self.assertTrue(any("terminated before" in item for item in receipt["errors"]))

    def test_hung_sampler_is_memfloor_error(self):
        release = threading.Event()
        entered = threading.Event()

        def sample() -> float:
            entered.set()
            release.wait()
            return 7000.0

        monitor = lifecycle.MemoryFloorMonitor(
            sample, floor_mib=6656.0, interval_s=0.001
        )
        started = monitor.start()
        self.assertTrue(entered.wait(1.0))
        receipt = monitor.finish(
            arm_started_s=started,
            arm_finished_s=time.monotonic(),
            arm_completed=True,
            join_timeout_s=0.01,
        )
        self.assertEqual(receipt["status"], "ERROR")
        self.assertTrue(any("join timeout" in item for item in receipt["errors"]))
        release.set()
        assert monitor._thread is not None
        monitor._thread.join(1.0)

    def test_low_memory_is_fail_not_sampler_error(self):
        monitor = lifecycle.MemoryFloorMonitor(
            lambda: 6500.0, floor_mib=6656.0, interval_s=0.001
        )
        started = monitor.start()
        deadline = time.monotonic() + 1.0
        while not monitor.samples and time.monotonic() < deadline:
            time.sleep(0.001)
        receipt = monitor.finish(
            arm_started_s=started,
            arm_finished_s=time.monotonic(),
            arm_completed=True,
        )
        self.assertEqual(receipt["status"], "FAIL")
        self.assertEqual(receipt["errors"], [])

    def test_incomplete_sampler_time_coverage_is_error(self):
        monitor = lifecycle.MemoryFloorMonitor(
            lambda: 7000.0,
            floor_mib=6656.0,
            interval_s=0.001,
            coverage_slack_s=0.01,
        )
        started = monitor.start()
        deadline = time.monotonic() + 1.0
        while not monitor.samples and time.monotonic() < deadline:
            time.sleep(0.001)
        receipt = monitor.finish(
            arm_started_s=started - 1.0,
            arm_finished_s=time.monotonic(),
            arm_completed=True,
        )
        self.assertEqual(receipt["status"], "ERROR")
        self.assertTrue(any("coverage started" in item for item in receipt["errors"]))

    def test_internal_sampler_stall_is_error_even_if_endpoints_are_covered(self):
        second_sample_started = threading.Event()
        release_second_sample = threading.Event()
        calls = 0

        def sample() -> float:
            nonlocal calls
            calls += 1
            if calls == 2:
                second_sample_started.set()
                self.assertTrue(release_second_sample.wait(1.0))
            return 7000.0

        monitor = lifecycle.MemoryFloorMonitor(
            sample,
            floor_mib=6656.0,
            interval_s=0.001,
            coverage_slack_s=0.02,
        )
        started = monitor.start()
        self.assertTrue(second_sample_started.wait(1.0))
        time.sleep(0.06)
        release_second_sample.set()
        deadline = time.monotonic() + 1.0
        while len(monitor.samples) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        receipt = monitor.finish(
            arm_started_s=started,
            arm_finished_s=time.monotonic(),
            arm_completed=True,
        )
        self.assertEqual(receipt["status"], "ERROR")
        self.assertGreater(receipt["max_sample_gap_s"], 0.02)
        self.assertTrue(
            any("internal gap" in item for item in receipt["errors"]),
            receipt["errors"],
        )

    def test_non_finite_sample_is_error(self):
        fatal_seen = threading.Event()

        def sample() -> float:
            fatal_seen.set()
            return float("nan")

        monitor = lifecycle.MemoryFloorMonitor(
            sample, floor_mib=6656.0, interval_s=0.001
        )
        started = monitor.start()
        self.assertTrue(fatal_seen.wait(1.0))
        assert monitor._thread is not None
        monitor._thread.join(1.0)
        receipt = monitor.finish(
            arm_started_s=started,
            arm_finished_s=time.monotonic(),
            arm_completed=True,
        )
        self.assertEqual(receipt["status"], "ERROR")
        self.assertTrue(any("non-finite" in item for item in receipt["errors"]))


if __name__ == "__main__":
    unittest.main()
