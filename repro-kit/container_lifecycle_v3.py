#!/usr/bin/env python3
"""Fail-closed container cleanup and memory-floor evidence primitives.

This module contains no Docker invocation.  A release runner supplies a small
adapter around its already-audited command runner.  The lifecycle owns the
ordering and unlock decision:

* settle the exact detached launch client before accepting any absence proof;
* inspect the cidfile, exact name, remembered IDs, and exact-label sweep on
  every reconciliation cycle;
* stop an exactly owned running container before collecting diagnostics;
* require repeated clean stopped/absent sweeps before releasing the lock;
* quarantine every ``BaseException`` while cleanup is unresolved; and
* turn sampler death, fatal exceptions, or incomplete time coverage into a
  ``MEMFLOOR`` ``ERROR`` rather than a pass.

Production callers must use ``run_with_cleanup_v3`` (or preserve its exact
contract).  ``max_cycles`` is an offline-test escape hatch and is rejected by
that wrapper so a production path cannot accidentally turn a bounded probe
into permission to release its flock.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import hashlib
import math
import re
import signal
import threading
import time
from types import FrameType, TracebackType
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

OWNER_LABEL = "com.theapathy.flux.e2e.cleanup-token"
RUN_LABEL = "com.theapathy.flux.e2e-sealed.run"
VERSION_LABEL = "com.theapathy.flux.e2e.cleanup-version"
CID_RE = re.compile(r"[0-9a-f]{64}")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
DEFAULT_STABLE_SWEEPS = 4
MIN_RETRY_BACKOFF_S = 0.01
MAX_RECEIPT_ERRORS = 128
MAX_RECEIPT_EVENTS = 256
MAX_RECEIPT_TEXT_CHARS = 512


def _safe_type_name(value: object) -> str:
    """Return a diagnostic type name without trusting a custom metaclass."""
    try:
        name = type(value).__name__
    except BaseException:
        return "object"
    return name[:128] if type(name) is str else "object"


def _safe_text(value: object, *, limit: int = MAX_RECEIPT_TEXT_CHARS) -> str:
    """Render arbitrary adapter data without allowing ``__str__`` to escape.

    Cleanup runs after launch authority has been exercised, so even diagnostic
    formatting is part of the fail-closed boundary.  A hostile or simply
    broken exception object must not be able to unwind the reconciliation loop
    and release the process-owned flock while a container may still be live.
    """
    try:
        text = str(value)
    except BaseException:
        text = f"<unprintable {_safe_type_name(value)}>"
    try:
        text = text.replace("\x00", "?")
        if len(text) > limit:
            text = text[: limit - 3] + "..."
        return text
    except BaseException:  # pragma: no cover - exact-str operations are stable
        return "<unprintable text>"


def _safe_exception(exc: BaseException) -> str:
    return f"{_safe_type_name(exc)}: {_safe_text(exc)}"


def _safe_traceback(exc: BaseException) -> TracebackType | None:
    """Read an exception traceback without invoking a hostile override."""
    try:
        value = BaseException.__getattribute__(exc, "__traceback__")
    except BaseException:
        return None
    return value if value is None or isinstance(value, TracebackType) else None


@dataclass(frozen=True)
class LaunchIdentity:
    """Exact identity written before the detached container launch begins."""

    name: str
    image_id: str
    run_id: str
    owner_token: str
    cleanup_version: str
    cidfile: str
    remembered_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or "/" in self.name
            or self.name in (".", "..")
        ):
            raise ValueError("launch name must be one non-path Docker name")
        if not self.run_id or not self.owner_token:
            raise ValueError("launch name, run ID, and owner token must be non-empty")
        if IMAGE_ID_RE.fullmatch(self.image_id) is None:
            raise ValueError("image_id must be an immutable sha256 image ID")
        if not self.cleanup_version or not self.cidfile:
            raise ValueError("cleanup version and cidfile must be non-empty")
        if any(CID_RE.fullmatch(value) is None for value in self.remembered_ids):
            raise ValueError("remembered container IDs must be full 64-hex IDs")

    @property
    def ownership_labels(self) -> dict[str, str]:
        return {
            OWNER_LABEL: self.owner_token,
            RUN_LABEL: self.run_id,
            VERSION_LABEL: self.cleanup_version,
        }

    def public(self) -> dict[str, object]:
        """Return receipt-safe identity without disclosing the random token."""
        return {
            "name": self.name,
            "image_id": self.image_id,
            "run_id": self.run_id,
            "owner_token_sha256": hashlib.sha256(
                self.owner_token.encode("utf-8")
            ).hexdigest(),
            "cleanup_version": self.cleanup_version,
            "cidfile_sha256": hashlib.sha256(self.cidfile.encode("utf-8")).hexdigest(),
            "remembered_ids": list(self.remembered_ids),
        }


@dataclass(frozen=True)
class ContainerRecord:
    """Minimal ``docker inspect`` projection needed for a stop decision."""

    container_id: str
    name: str
    image_id: str
    labels: Mapping[str, object]
    running: object


@dataclass(frozen=True)
class InspectObservation:
    """Tri-state inspect result; errors are never treated as absence."""

    status: str
    record: ContainerRecord | None = None
    error: str | None = None

    @classmethod
    def absent(cls) -> "InspectObservation":
        return cls("absent")

    @classmethod
    def present(cls, record: ContainerRecord) -> "InspectObservation":
        return cls("present", record=record)

    @classmethod
    def failed(cls, error: str) -> "InspectObservation":
        return cls("error", error=error)


@dataclass(frozen=True)
class CidfileObservation:
    """A missing cidfile is ordinary absence; malformed/unreadable is error."""

    status: str
    container_id: str | None = None
    error: str | None = None

    @classmethod
    def missing(cls) -> "CidfileObservation":
        return cls("missing")

    @classmethod
    def present(cls, container_id: str) -> "CidfileObservation":
        return cls("present", container_id=container_id)

    @classmethod
    def failed(cls, error: str) -> "CidfileObservation":
        return cls("error", error=error)


class LifecycleAdapter(Protocol):
    """Release-runner operations used by the fail-closed reconciler."""

    def settle_launch_client(self) -> bool: ...

    def read_cidfile(self, path: str) -> CidfileObservation: ...

    def inspect(self, ref: str) -> InspectObservation: ...

    def sweep_by_labels(
        self, labels: Mapping[str, str]
    ) -> Sequence[InspectObservation]: ...

    def stop_owned(self, container_id: str) -> None: ...

    def capture_after_stop(self, record: ContainerRecord) -> None: ...

    def wait(self) -> None: ...


@dataclass
class CleanupReceipt:
    schema: str = "atlas.container.cleanup.v3"
    lock_safe: bool = False
    lock_released: bool = False
    launch_client_settled: bool = False
    stable_absence_sweeps: int = 0
    stable_stopped_sweeps: int = 0
    cycles: int = 0
    stopped_ids: list[str] = field(default_factory=list)
    diagnostics_attempted_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    deferred_signals: list[int] = field(default_factory=list)
    suppressed_errors: int = 0
    suppressed_events: int = 0


class UnsafeToUnlock(BaseException):
    """Offline-only bounded reconciliation ended without a safety proof."""


class DeferredTermination(BaseException):
    """A termination signal was deferred until after cleanup and unlock."""

    def __init__(self, signals: Sequence[int]) -> None:
        self.signals = tuple(signals)
        super().__init__(f"termination deferred until cleanup: {self.signals}")


class _TerminationGate(AbstractContextManager["_TerminationGate"]):
    """Turn the first body signal into an unwind, then defer cleanup signals."""

    def __init__(self) -> None:
        self._phase = "body"
        self.received: list[int] = []
        self._previous: dict[int, Any] = {}
        self._parent: _TerminationGate | None = None
        self._enabled = threading.current_thread() is threading.main_thread()

    def __enter__(self) -> "_TerminationGate":
        if not self._enabled:
            raise RuntimeError("container cleanup signal gate requires the main thread")
        if self._enabled:
            guarded = {signal.SIGINT, signal.SIGTERM}
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, guarded)
            try:
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous = signal.getsignal(signum)
                    self._previous[signum] = previous
                    owner = getattr(previous, "__self__", None)
                    if isinstance(owner, _TerminationGate):
                        if self._parent is not None and self._parent is not owner:
                            raise RuntimeError(
                                "termination signals are owned by different parent gates"
                            )
                        self._parent = owner
                    signal.signal(signum, self._handle)
                if self._parent is not None:
                    # Nested production/fallback gates form one signal
                    # transaction.  Sharing the journal prevents a signal
                    # observed by the inner gate from disappearing when its
                    # handlers are restored to the outer gate.
                    self.received = self._parent.received
            except BaseException:
                # Context manager __exit__ is not called when __enter__ fails.
                # Roll back every handler installed so far while both signals
                # remain blocked, then restore the caller's mask.
                for signum, handler in self._previous.items():
                    signal.signal(signum, handler)
                self._previous.clear()
                self._parent = None
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                raise
            try:
                # A signal raised during installation is pending while blocked.
                # It is delivered here only after both handlers are installed.
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException:
                # A body-phase gate signal can raise while the mask is being
                # restored. Since __enter__ has not returned, transactionally
                # restore the prior handlers before propagating it.
                rollback_mask = signal.pthread_sigmask(signal.SIG_BLOCK, guarded)
                try:
                    for signum, handler in self._previous.items():
                        signal.signal(signum, handler)
                    self._previous.clear()
                    self._parent = None
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, rollback_mask)
                raise
        return self

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        if signum not in self.received:
            self.received.append(signum)
        if self._phase == "body":
            # Change phase before raising.  In particular, this closes the
            # body-return/begin-cleanup race: a signal in that bytecode gap
            # turns the gap itself into the cleanup transition, so every
            # subsequent signal is deferred while the body unwinds.
            self._phase = "cleanup"
            raise DeferredTermination(self.received)

    def begin_cleanup(self) -> None:
        self._phase = "cleanup"

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        # Block both signals while restoring both handlers. Pending signals are
        # delivered only after the complete handler set is back in place, so a
        # signal cannot interrupt restoration and leave one stale gate behind.
        guarded = {signal.SIGINT, signal.SIGTERM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, guarded)
        if self._parent is not None:
            # Once a child gate starts restoring the parent handlers, every
            # launched-body path is already unwinding into cleanup.  Advance
            # the parent before unmasking so a pending signal is journaled
            # instead of replacing the child's original exception.
            self._parent.begin_cleanup()
        restore_error: BaseException | None = None
        for signum, handler in self._previous.items():
            try:
                signal.signal(signum, handler)
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
        self._previous.clear()
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        self._parent = None
        if restore_error is not None:
            raise restore_error
        if exc is None and self.received:
            raise DeferredTermination(self.received)
        return None


def _ownership(
    identity: LaunchIdentity, record: ContainerRecord
) -> tuple[bool, dict[str, object]]:
    observed = {
        "id": record.container_id,
        "name": record.name,
        "image_id": record.image_id,
        "run_label": record.labels.get(RUN_LABEL),
        "owner_token_match": record.labels.get(OWNER_LABEL) == identity.owner_token,
        "version_label": record.labels.get(VERSION_LABEL),
        "running": record.running,
    }
    owned = (
        CID_RE.fullmatch(record.container_id) is not None
        and record.name in (identity.name, f"/{identity.name}")
        and observed["image_id"] == identity.image_id
        and observed["run_label"] == identity.run_id
        and observed["owner_token_match"] is True
        and observed["version_label"] == identity.cleanup_version
        and type(record.running) is bool
    )
    return owned, observed


def _bounded_append(
    values: list[str], value: object, *, limit: int, receipt: CleanupReceipt, kind: str
) -> None:
    text = _safe_text(value)
    if text in values:
        if kind == "error":
            receipt.suppressed_errors += 1
        else:
            receipt.suppressed_events += 1
        return
    if len(values) >= limit:
        if kind == "error":
            receipt.suppressed_errors += 1
        else:
            receipt.suppressed_events += 1
        return
    values.append(text)


def _error(receipt: CleanupReceipt, value: object) -> None:
    _bounded_append(
        receipt.errors,
        value,
        limit=MAX_RECEIPT_ERRORS,
        receipt=receipt,
        kind="error",
    )


def _event(receipt: CleanupReceipt, value: object) -> None:
    _bounded_append(
        receipt.events,
        value,
        limit=MAX_RECEIPT_EVENTS,
        receipt=receipt,
        kind="event",
    )


def _fail_closed_backoff(seconds: float) -> None:
    """Best-effort retry delay that can never unwind unresolved cleanup."""
    if seconds <= 0:
        return
    try:
        time.sleep(seconds)
        return
    except BaseException:
        pass
    # Keep an independent standard-library fallback so an injected/broken
    # sleep implementation does not turn the quarantine into a hot loop.  If
    # both timing primitives fail, retaining the lock and retrying is still the
    # only safe outcome.
    try:
        threading.Event().wait(seconds)
    except BaseException:
        pass


def _quarantine_wait(adapter: LifecycleAdapter, receipt: CleanupReceipt) -> None:
    try:
        started = time.monotonic()
    except BaseException:
        started = None
    try:
        adapter.wait()
    except BaseException as exc:
        _error(receipt, f"wait {_safe_exception(exc)}")
    # The lifecycle, not an adapter implementation, owns the minimum retry
    # cadence.  A broken/no-op/raising adapter wait therefore cannot turn an
    # unresolved production quarantine into an unbounded CPU and receipt DoS.
    try:
        elapsed = time.monotonic() - started if started is not None else 0.0
    except BaseException:
        elapsed = 0.0
    remaining = MIN_RETRY_BACKOFF_S - elapsed
    if remaining > 0:
        _fail_closed_backoff(remaining)


def _collect_cycle(
    adapter: LifecycleAdapter,
    identity: LaunchIdentity,
    known_ids: set[str],
    receipt: CleanupReceipt,
) -> tuple[dict[str, ContainerRecord], bool]:
    """Return exact-owned records and whether the whole sweep was error-free."""
    clean = True
    refs = [*sorted(known_ids), *identity.remembered_ids]
    cidfile = adapter.read_cidfile(identity.cidfile)
    if cidfile.status == "present":
        if (
            cidfile.container_id is None
            or CID_RE.fullmatch(cidfile.container_id) is None
        ):
            _error(receipt, "cidfile present without a full 64-hex container ID")
            clean = False
        else:
            refs.append(cidfile.container_id)
    elif cidfile.status == "error":
        _error(receipt, f"cidfile error: {cidfile.error or 'unspecified'}")
        clean = False
    elif cidfile.status != "missing":
        _error(receipt, f"invalid cidfile status: {cidfile.status!r}")
        clean = False
    refs.append(identity.name)

    observations: list[tuple[str, InspectObservation]] = []
    for ref in dict.fromkeys(refs):
        observations.append((f"ref:{ref}", adapter.inspect(ref)))
    for index, observation in enumerate(
        adapter.sweep_by_labels(identity.ownership_labels)
    ):
        observations.append((f"label-sweep:{index}", observation))

    owned_by_id: dict[str, ContainerRecord] = {}
    representation_by_id: dict[str, ContainerRecord] = {}
    for source, observation in observations:
        if observation.status == "absent":
            continue
        if observation.status == "error":
            _error(
                receipt, f"{source} inspect error: {observation.error or 'unspecified'}"
            )
            clean = False
            continue
        if observation.status != "present" or observation.record is None:
            _error(receipt, f"{source} returned malformed inspect observation")
            clean = False
            continue
        record = observation.record
        owned, observed = _ownership(identity, record)
        prior = representation_by_id.get(record.container_id)
        if prior is not None and prior != record:
            _error(
                receipt,
                f"{source} returned conflicting inspect data for {record.container_id}",
            )
            clean = False
            continue
        representation_by_id[record.container_id] = record
        if not owned:
            _error(receipt, f"{source} identity conflict: {observed!r}")
            clean = False
            continue
        known_ids.add(record.container_id)
        owned_by_id[record.container_id] = record
    if len(owned_by_id) > 1:
        _error(
            receipt,
            "multiple containers matched one exact cleanup identity: "
            + ",".join(sorted(owned_by_id)),
        )
        clean = False
    return owned_by_id, clean


def reconcile_before_unlock(
    adapter: LifecycleAdapter,
    identity: LaunchIdentity,
    release_lock: Callable[[], None],
    *,
    stable_sweeps: int = DEFAULT_STABLE_SWEEPS,
    max_cycles: int | None = None,
    receipt: CleanupReceipt | None = None,
) -> CleanupReceipt:
    """Reconcile until safe, then and only then call ``release_lock``.

    With the production default ``max_cycles=None``, all failures remain in an
    in-process retry loop.  The bounded mode exists for deterministic offline
    tests; it raises ``UnsafeToUnlock`` without calling ``release_lock``.
    """
    if type(stable_sweeps) is not int or stable_sweeps < 2:
        raise ValueError("stable_sweeps must be a plain integer >= 2")
    if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
        raise ValueError("max_cycles must be a positive plain integer")

    result = receipt or CleanupReceipt()
    known_ids = set(identity.remembered_ids)
    stopped_ids: set[str] = set(result.stopped_ids)
    diagnostics_attempted: set[str] = set(result.diagnostics_attempted_ids)
    absent = 0
    stopped = 0
    safe = False

    while not safe:
        result.cycles += 1
        if max_cycles is not None and result.cycles > max_cycles:
            raise UnsafeToUnlock(
                "bounded offline cleanup ended without an unlock safety proof"
            )
        try:
            if not adapter.settle_launch_client():
                _event(result, "launch-client-pending")
                absent = 0
                stopped = 0
                _quarantine_wait(adapter, result)
                continue
            result.launch_client_settled = True
            owned_by_id, clean = _collect_cycle(adapter, identity, known_ids, result)
            if not clean or len(owned_by_id) > 1:
                absent = 0
                stopped = 0
                _event(result, "quarantine-unresolved-sweep")
                _quarantine_wait(adapter, result)
                continue

            if owned_by_id:
                record = next(iter(owned_by_id.values()))
                absent = 0
                if record.running is True:
                    stopped = 0
                    _event(result, f"stop:{record.container_id}")
                    adapter.stop_owned(record.container_id)
                    if record.container_id not in stopped_ids:
                        stopped_ids.add(record.container_id)
                        result.stopped_ids.append(record.container_id)
                    # Stop confirmation and diagnostics occur on a subsequent
                    # complete sweep, never before this stop operation.
                    _quarantine_wait(adapter, result)
                    continue
                if record.running is not False:
                    _error(
                        result,
                        f"owned {record.container_id} has non-boolean running state",
                    )
                    stopped = 0
                    _event(result, "quarantine-invalid-running-state")
                    _quarantine_wait(adapter, result)
                    continue
                if record.container_id not in diagnostics_attempted:
                    _event(result, f"diagnostics:{record.container_id}")
                    diagnostics_attempted.add(record.container_id)
                    result.diagnostics_attempted_ids.append(record.container_id)
                    try:
                        adapter.capture_after_stop(record)
                    except BaseException as exc:
                        # Diagnostics are evidence, not a reason to leave a
                        # stopped container holding the GPU lock forever.
                        _error(result, f"post-stop diagnostics {_safe_exception(exc)}")
                stopped += 1
                result.stable_stopped_sweeps = stopped
                result.stable_absence_sweeps = 0
                _event(result, f"stopped-sweep:{stopped}")
                if stopped < stable_sweeps:
                    _quarantine_wait(adapter, result)
                    continue
            else:
                stopped = 0
                absent += 1
                result.stable_absence_sweeps = absent
                result.stable_stopped_sweeps = 0
                _event(result, f"absent-sweep:{absent}")
                if absent < stable_sweeps:
                    _quarantine_wait(adapter, result)
                    continue

            result.lock_safe = True
            safe = True
        except UnsafeToUnlock:
            raise
        except BaseException as exc:
            # Includes KeyboardInterrupt/SystemExit and adapter defects.  No
            # cleanup exception is allowed to unwind past an unresolved flock.
            absent = 0
            stopped = 0
            _error(result, f"cleanup {_safe_exception(exc)}")
            _event(result, "quarantine-baseexception")
            _quarantine_wait(adapter, result)

    # Lock release is deliberately outside the cleanup exception quarantine.
    # Once here, stopped/absence safety has already been proved.  Retrying a
    # callback that partially released and then raised could otherwise run a
    # second reconciliation loop without the flock it claims to retain.
    _event(result, "unlock-authorized")
    release_lock()
    result.lock_released = True
    _event(result, "lock-released")
    return result


T = TypeVar("T")


def run_with_cleanup_v3(
    body: Callable[[], T],
    adapter: LifecycleAdapter,
    identity: LaunchIdentity,
    release_lock: Callable[[], None],
    *,
    stable_sweeps: int = DEFAULT_STABLE_SWEEPS,
) -> tuple[T, CleanupReceipt]:
    """Run an owned-container body and retain its lock until cleanup is safe."""
    result: T | None = None
    original: tuple[BaseException, TracebackType | None] | None = None
    gate = _TerminationGate()
    with gate:
        try:
            try:
                result = body()
            except BaseException as exc:
                original = (exc, _safe_traceback(exc))
            finally:
                gate.begin_cleanup()
        except BaseException as exc:
            # A body-return signal can interrupt the finally suite immediately
            # before begin_cleanup().  The handler has already advanced its
            # own phase, and this outer catch guarantees reconciliation still
            # runs instead of escaping the wrapper.
            if original is None:
                original = (exc, _safe_traceback(exc))
            gate.begin_cleanup()
        cleanup = CleanupReceipt()
        cleanup.deferred_signals = gate.received
        cleanup = reconcile_before_unlock(
            adapter,
            identity,
            release_lock,
            stable_sweeps=stable_sweeps,
            max_cycles=None,
            receipt=cleanup,
        )
        if original is not None:
            exc, traceback = original
            if isinstance(exc, DeferredTermination):
                raise DeferredTermination(gate.received)
            raise BaseException.with_traceback(exc, traceback)
        if gate.received:
            raise DeferredTermination(gate.received)
        return result, cleanup  # type: ignore[return-value]


@dataclass(frozen=True)
class MemorySample:
    monotonic_s: float
    available_mib: float


class MemoryFloorMonitor:
    """Threaded sampler whose infrastructure failures are receipt ``ERROR``s."""

    def __init__(
        self,
        sample: Callable[[], float],
        *,
        floor_mib: float,
        interval_s: float = 1.0,
        coverage_slack_s: float | None = None,
    ) -> None:
        if (
            not math.isfinite(float(interval_s))
            or not math.isfinite(float(floor_mib))
            or interval_s <= 0
            or floor_mib < 0
        ):
            raise ValueError("interval must be positive and floor non-negative")
        self._sample = sample
        self.floor_mib = float(floor_mib)
        self.interval_s = float(interval_s)
        self.coverage_slack_s = (
            max(1.0, 4.0 * interval_s)
            if coverage_slack_s is None
            else float(coverage_slack_s)
        )
        if not math.isfinite(self.coverage_slack_s) or self.coverage_slack_s < 0:
            raise ValueError("coverage slack must be non-negative")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._fatal: BaseException | None = None
        self._premature_exit = False
        self._finished = False
        self.samples: list[MemorySample] = []

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                value = self._sample()
                if type(value) not in (int, float) or isinstance(value, bool):
                    raise TypeError("memory sampler must return a plain number")
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        "memory sampler returned a non-finite or negative value"
                    )
                with self._lock:
                    self.samples.append(MemorySample(time.monotonic(), value))
                self._stop.wait(self.interval_s)
        except BaseException as exc:
            self._fatal = exc
        finally:
            if not self._stop.is_set():
                self._premature_exit = True

    def start(self) -> float:
        if self._thread is not None:
            raise RuntimeError("memory sampler already started")
        started = time.monotonic()
        self._thread = threading.Thread(
            target=self._worker, name="container-memfloor-v3", daemon=True
        )
        self._thread.start()
        return started

    def finish(
        self,
        *,
        arm_started_s: float,
        arm_finished_s: float,
        arm_completed: bool,
        join_timeout_s: float = 5.0,
    ) -> dict[str, object]:
        if self._thread is None:
            raise RuntimeError("memory sampler was not started")
        if self._finished:
            raise RuntimeError("memory sampler was already finished")
        if not math.isfinite(arm_started_s) or not math.isfinite(arm_finished_s):
            raise ValueError("arm timestamps must be finite")
        if type(arm_completed) is not bool:
            raise TypeError("arm_completed must be an exact boolean")
        if not math.isfinite(join_timeout_s) or join_timeout_s < 0:
            raise ValueError("join timeout must be finite and non-negative")
        self._finished = True
        if arm_finished_s < arm_started_s:
            raise ValueError("arm finish precedes arm start")
        died_before_stop = not self._thread.is_alive() and not self._stop.is_set()
        self._stop.set()
        self._thread.join(join_timeout_s)

        errors: list[str] = []
        if not arm_completed:
            errors.append("arm did not complete")
        if died_before_stop or self._premature_exit:
            errors.append("memory sampler terminated before arm completion")
        if self._thread.is_alive():
            errors.append("memory sampler did not stop before join timeout")
        if self._fatal is not None:
            errors.append(f"fatal memory sampler {_safe_exception(self._fatal)}")
        with self._lock:
            samples = list(self.samples)
        if not samples:
            errors.append("memory sampler produced no coverage")
            max_sample_gap_s: float | None = None
        else:
            start_gap = samples[0].monotonic_s - arm_started_s
            end_gap = arm_finished_s - samples[-1].monotonic_s
            max_sample_gap_s = max(
                (
                    later.monotonic_s - earlier.monotonic_s
                    for earlier, later in zip(samples, samples[1:])
                ),
                default=0.0,
            )
            if start_gap > self.coverage_slack_s:
                errors.append(
                    f"memory coverage started {start_gap:.3f}s after arm start"
                )
            if end_gap > self.coverage_slack_s:
                errors.append(
                    f"memory coverage ended {end_gap:.3f}s before arm completion"
                )
            if max_sample_gap_s > self.coverage_slack_s:
                errors.append(
                    "memory coverage had an internal gap of "
                    f"{max_sample_gap_s:.3f}s"
                )

        minimum = min((item.available_mib for item in samples), default=None)
        if errors:
            status = "ERROR"
        elif minimum is not None and minimum < self.floor_mib:
            status = "FAIL"
        else:
            status = "PASS"
        return {
            "id": "MEMFLOOR",
            "status": status,
            "floor_mib": self.floor_mib,
            "minimum_mib": minimum,
            "sample_count": len(samples),
            "arm_completed": arm_completed,
            "coverage_slack_s": self.coverage_slack_s,
            "max_sample_gap_s": max_sample_gap_s,
            "errors": errors,
            "samples": [
                {
                    "monotonic_s": item.monotonic_s,
                    "available_mib": item.available_mib,
                }
                for item in samples
            ],
        }
