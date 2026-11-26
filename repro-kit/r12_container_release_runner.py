#!/usr/bin/env python3
"""Concrete R12 detached-container runner backed by lifecycle-v3.

The R8 qualification program was the last live release/container runner, but
it carried a private copy of its cleanup state machine.  R12 uses this module
as the concrete Docker adapter and calls the reviewed lifecycle primitive
directly.  Docker execution is injected, so the complete failure matrix can be
tested without a daemon, GPU, server, network, or image.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import threading
import time
from types import TracebackType
from typing import Callable, Generic, Iterator, Mapping, Protocol, Sequence, TypeVar

import container_lifecycle_v3 as lifecycle

R12_CLEANUP_VERSION = "release-cleanup-v4-r12"
DEFAULT_COMMAND_TIMEOUT_S = 45.0
DEFAULT_STOP_GRACE_S = 30
MAX_COMMAND_DETAIL_CHARS = 400

# Release qualification intentionally permits only the Docker options used by
# the sealed recipe.  Every accepted option consumes exactly one value, which
# makes it impossible for a positional image to be smuggled ahead of the
# immutable image ID owned by this runner.
ALLOWED_VALUE_RUN_OPTIONS = frozenset(
    {
        "--device",
        "--env",
        "--gpus",
        "--ipc",
        "--mount",
        "--network",
        "--publish",
        "--security-opt",
        "--shm-size",
        "--ulimit",
        "--volume",
        "-e",
        "-p",
        "-v",
    }
)
ALLOWED_FLAG_RUN_OPTIONS = frozenset({"--read-only"})


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class LaunchProcess(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]: ...


class CommandBackend(Protocol):
    def run(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult: ...

    def popen(self, argv: Sequence[str]) -> LaunchProcess: ...


class SubprocessBackend:
    """Real no-shell subprocess backend used by the R12 runner."""

    def run(self, argv: Sequence[str], *, timeout_s: float) -> CommandResult:
        completed = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
            text=True,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def popen(self, argv: Sequence[str]) -> LaunchProcess:
        return subprocess.Popen(
            list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )


def _detail(value: object) -> str:
    return lifecycle._safe_text(value, limit=MAX_COMMAND_DETAIL_CHARS)


def _os_error_detail(exc: OSError) -> str:
    """Describe an OS error without copying a host path into receipts/logs."""
    try:
        errno = OSError.__getattribute__(exc, "errno")
    except BaseException:
        errno = None
    try:
        strerror = OSError.__getattribute__(exc, "strerror")
    except BaseException:
        strerror = None
    message = (
        strerror
        if type(strerror) is str and strerror
        else lifecycle._safe_type_name(exc)
    )
    return f"errno={_detail(errno)} {_detail(message)}"


def _capture_once(parent_fd: int, name: str, payload: bytes) -> None:
    """Reuse an identical immutable capture across independent cleanup sweeps."""
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o400,
            dir_fd=parent_fd,
        )
    except FileExistsError:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or before.st_uid != os.geteuid()
                    or stat.S_IMODE(before.st_mode) != 0o400
                    or before.st_size != len(payload)):
                raise RuntimeError("existing diagnostic capture identity differs")
            offset = 0
            while offset < len(payload):
                chunk = os.read(descriptor, min(1024 * 1024, len(payload) - offset))
                if not chunk or chunk != payload[offset:offset + len(chunk)]:
                    raise RuntimeError("existing diagnostic capture bytes differ")
                offset += len(chunk)
            identity = lambda item: (item.st_dev, item.st_ino, item.st_mode,
                                     item.st_uid, item.st_nlink, item.st_size,
                                     item.st_mtime_ns, item.st_ctime_ns)
            if (identity(before) != identity(os.fstat(descriptor))
                    or identity(before) != identity(os.stat(
                        name, dir_fd=parent_fd, follow_symlinks=False))):
                raise RuntimeError("existing diagnostic capture changed during comparison")
        finally:
            os.close(descriptor)
        return
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write during diagnostics capture")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _open_nofollow_parent(path: str, *, create: bool = False) -> tuple[int, str]:
    """Open the parent directory of ``path`` walking components no-follow.

    Returns ``(parent_fd, final_name)`` so reads and writes stay anchored to
    the real directory inode; a symlink at any component fails closed.
    """

    parts = [part for part in path.split("/") if part]
    if not path.startswith("/") or not parts:
        raise OSError(errno.EINVAL, "path must be absolute and non-root", path)
    for part in parts:
        if part in (".", ".."):
            raise OSError(errno.EINVAL, "unnormalized path component", part)
    parent_fd = os.open(
        "/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, dir_fd=parent_fd)
                except FileExistsError:
                    pass
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = child
        return parent_fd, parts[-1]
    except BaseException:
        os.close(parent_fd)
        raise


@contextmanager
def owned_child_processes() -> Iterator[None]:
    """Contain the outer suite too, including a supervisor timeout/crash."""

    def children() -> list[int]:
        return [
            int(value)
            for value in Path(f"/proc/self/task/{os.getpid()}/children")
            .read_text("ascii")
            .split()
        ]

    if (
        threading.active_count() != 1
        or len(os.listdir("/proc/self/task")) != 1
        or children()
    ):
        raise RuntimeError(
            "suite containment requires an idle single-thread supervisor"
        )
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("suite containment requires Linux pidfds")
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if (
        libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0
        or libc.prctl(36, 1, 0, 0, 0) != 0
    ):
        raise RuntimeError("suite containment requires Linux child subreaper support")
    try:
        yield
    finally:
        leaked = bool(children())
        deadline = time.monotonic() + 5.0
        while children():
            for pid in children():
                try:
                    descriptor = os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(descriptor)
            while True:
                try:
                    reaped, _status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not reaped:
                    break
            if time.monotonic() >= deadline:
                raise RuntimeError("suite descendants could not be drained")
            time.sleep(0.01)
        if libc.prctl(36, previous.value, 0, 0, 0) != 0:
            raise RuntimeError("cannot restore suite child subreaper state")
        if leaked:
            raise RuntimeError("suite left a process descendant")


class DockerLifecycleAdapter:
    """Exact-identity Docker adapter for ``container_lifecycle_v3``."""

    def __init__(
        self,
        identity: lifecycle.LaunchIdentity,
        *,
        backend: CommandBackend,
        diagnostics_dir: Path,
        diagnostic_capture: Callable[[lifecycle.ContainerRecord], None] | None = None,
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
        stop_grace_s: int = DEFAULT_STOP_GRACE_S,
        retry_interval_s: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if command_timeout_s <= 0 or retry_interval_s < 0:
            raise ValueError(
                "command timeout must be positive and retry interval non-negative"
            )
        if type(stop_grace_s) is not int or stop_grace_s < 0:
            raise ValueError("stop grace must be a non-negative plain integer")
        self.identity = identity
        self.backend = backend
        self.diagnostics_dir = diagnostics_dir
        self.diagnostic_capture = diagnostic_capture
        self.command_timeout_s = float(command_timeout_s)
        self.stop_grace_s = stop_grace_s
        self.retry_interval_s = float(retry_interval_s)
        self.sleeper = sleeper
        self.launch_process: LaunchProcess | None = None
        self.launch_argv: tuple[str, ...] | None = None
        self.launch_result: CommandResult | None = None
        self.authoritative_launch_ids: list[str] = []
        self.cidfile_parent_fd: int | None = None
        self.cidfile_name: str | None = None
        self.diagnostics_parent_fd: int | None = None

    @staticmethod
    def _validate_extra_run_args(values: Sequence[str]) -> tuple[str, ...]:
        forbidden = (
            "--name",
            "--cidfile",
            "--label",
            "--label-file",
            "--detach",
            "--rm",
            "--restart",
        )
        raw = tuple(values)
        result: list[str] = []
        index = 0
        while index < len(raw):
            value = raw[index]
            if type(value) is not str or not value:
                raise ValueError("extra Docker run arguments must be non-empty strings")
            option = value.split("=", 1)[0]
            if option in forbidden or value == "-d" or value.startswith("-d="):
                raise ValueError(f"R12 runner owns Docker option {option!r}")
            if option in ALLOWED_FLAG_RUN_OPTIONS:
                if value != option:
                    raise ValueError(f"Docker flag does not take a value: {option!r}")
                result.append(value)
                index += 1
                continue
            if option not in ALLOWED_VALUE_RUN_OPTIONS:
                raise ValueError(f"Docker run option is not allowlisted: {option!r}")
            if "=" in value:
                if not option.startswith("--") or not value.split("=", 1)[1]:
                    raise ValueError(f"Docker run option requires a value: {option!r}")
                result.append(value)
                index += 1
                continue
            if index + 1 >= len(raw):
                raise ValueError(f"Docker run option requires a value: {option!r}")
            option_value = raw[index + 1]
            if type(option_value) is not str or not option_value:
                raise ValueError(
                    f"Docker run option requires a string value: {option!r}"
                )
            result.extend((value, option_value))
            index += 2
        return tuple(result)

    def _remember_successful_launch(self, result: CommandResult) -> str | None:
        if result.returncode != 0:
            return None
        container_id = result.stdout.strip()
        if lifecycle.CID_RE.fullmatch(container_id) is None:
            return None
        if container_id not in self.authoritative_launch_ids:
            self.authoritative_launch_ids.append(container_id)
        return container_id

    def launch(
        self,
        extra_run_args: Sequence[str],
        container_command: Sequence[str],
        *,
        entrypoint: str | None = None,
    ) -> str:
        """Launch one detached exact-image container and return its full ID."""
        if self.launch_process is not None:
            raise RuntimeError("refusing overlapping detached Docker clients")
        extra = self._validate_extra_run_args(extra_run_args)
        command = tuple(container_command)
        if any(type(value) is not str or not value for value in command):
            raise ValueError("container command entries must be non-empty strings")
        if entrypoint is not None and (
            type(entrypoint) is not str
            or not entrypoint.startswith("/")
            or "\x00" in entrypoint
        ):
            raise ValueError("container entrypoint must be an absolute NUL-free path")
        argv = [
            "docker",
            "run",
            "--detach",
            "--restart=no",
            "--name",
            self.identity.name,
            "--cidfile",
            (
                self.identity.cidfile
                if self.cidfile_parent_fd is None
                else f"/proc/{os.getpid()}/fd/{self.cidfile_parent_fd}/{self.cidfile_name}"
            ),
        ]
        if entrypoint is not None:
            argv.extend(("--entrypoint", entrypoint))
        for key, value in sorted(self.identity.ownership_labels.items()):
            argv.extend(("--label", f"{key}={value}"))
        argv.extend(extra)
        argv.append(self.identity.image_id)
        argv.extend(command)
        self.launch_argv = tuple(argv)
        # A signal between child creation and retaining the Popen object would
        # otherwise leave a detached container with no authoritative client
        # handle.  Keep only this tiny creation/store window atomic; the
        # potentially long communicate call remains interruptible.
        guarded = {signal.SIGINT, signal.SIGTERM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, guarded)
        try:
            self.launch_process = self.backend.popen(argv)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        stdout, stderr = self.launch_process.communicate(timeout=self.command_timeout_s)
        # Retain a successful launch ID before dropping the process handle.
        # If a signal becomes pending here, it is delivered only after either
        # the process or the exact result is discoverable by reconciliation.
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, guarded)
        try:
            returncode = self.launch_process.returncode
            if type(returncode) is not int:
                raise RuntimeError(
                    "detached Docker client settled without an exit code"
                )
            self.launch_result = CommandResult(
                returncode,
                stdout.decode("utf-8", "replace"),
                stderr.decode("utf-8", "replace"),
            )
            self._remember_successful_launch(self.launch_result)
            self.launch_process = None
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if returncode != 0:
            raise RuntimeError(
                f"detached Docker launch failed rc={returncode}: {_detail(self.launch_result.stderr)}"
            )
        container_id = self._remember_successful_launch(self.launch_result)
        if container_id is None:
            raise RuntimeError(
                "detached Docker launch did not return one full container ID"
            )
        return container_id

    def settle_launch_client(self) -> bool:
        process = self.launch_process
        if process is None:
            if self.launch_result is not None:
                self._remember_successful_launch(self.launch_result)
            return True
        if process.poll() is None:
            return False
        stdout, stderr = process.communicate(timeout=0.0)
        returncode = process.returncode
        if type(returncode) is not int:
            raise RuntimeError("settled Docker client has no exact exit code")
        guarded = {signal.SIGINT, signal.SIGTERM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, guarded)
        try:
            self.launch_result = CommandResult(
                returncode,
                stdout.decode("utf-8", "replace"),
                stderr.decode("utf-8", "replace"),
            )
            self._remember_successful_launch(self.launch_result)
            self.launch_process = None
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        return True

    def read_cidfile(self, path: str) -> lifecycle.CidfileObservation:
        try:
            if self.cidfile_parent_fd is None:
                parent_fd, name = _open_nofollow_parent(path)
            else:
                parent_fd, name = os.dup(self.cidfile_parent_fd), self.cidfile_name
        except FileNotFoundError:
            return lifecycle.CidfileObservation.missing()
        except NotADirectoryError:
            return lifecycle.CidfileObservation.failed(
                "cidfile parent is not a directory"
            )
        except OSError as exc:
            return lifecycle.CidfileObservation.failed(
                f"walk {lifecycle._safe_type_name(exc)}: {_os_error_detail(exc)}"
            )
        try:
            try:
                metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return lifecycle.CidfileObservation.missing()
            except OSError as exc:
                return lifecycle.CidfileObservation.failed(
                    f"lstat {lifecycle._safe_type_name(exc)}: {_os_error_detail(exc)}"
                )
            if not stat.S_ISREG(metadata.st_mode):
                return lifecycle.CidfileObservation.failed(
                    "cidfile is not a regular file"
                )
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                return lifecycle.CidfileObservation.failed(
                    f"open {lifecycle._safe_type_name(exc)}: {_os_error_detail(exc)}"
                )
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                    return lifecycle.CidfileObservation.failed(
                        "cidfile changed during read"
                    )
                raw = os.read(descriptor, 66)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_fd)
        try:
            value = raw.decode("ascii")
        except UnicodeDecodeError:
            return lifecycle.CidfileObservation.failed("cidfile is not ASCII")
        if value.endswith("\n"):
            value = value[:-1]
        if lifecycle.CID_RE.fullmatch(value) is None:
            return lifecycle.CidfileObservation.failed(
                "cidfile does not contain one full container ID"
            )
        return lifecycle.CidfileObservation.present(value)

    def _run(self, argv: Sequence[str]) -> CommandResult:
        return self.backend.run(argv, timeout_s=self.command_timeout_s)

    def _container_index(self) -> tuple[dict[str, str], str | None]:
        result = self._run(
            (
                "docker",
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--format",
                "{{.ID}}\t{{.Names}}",
            )
        )
        if result.returncode != 0:
            return (
                {},
                f"docker container ls rc={result.returncode}: {_detail(result.stderr)}",
            )
        index: dict[str, str] = {}
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) != 2 or lifecycle.CID_RE.fullmatch(fields[0]) is None:
                return {}, "docker container ls returned malformed identity data"
            container_id, name = fields
            if not name or name in index.values():
                return {}, "docker container ls returned duplicate or empty name data"
            index[container_id] = name
        return index, None

    def _inspect_id(
        self, container_id: str, *, positively_listed: bool = False
    ) -> lifecycle.InspectObservation:
        result = self._run(
            (
                "docker",
                "container",
                "inspect",
                "--format",
                "{{json .}}",
                container_id,
            )
        )
        if result.returncode != 0:
            normalized = result.stderr.strip().casefold()
            absent_messages = {
                f"error: no such object: {container_id}".casefold(),
                f"error response from daemon: no such object: {container_id}".casefold(),
                f"error: no such container: {container_id}".casefold(),
                f"error response from daemon: no such container: {container_id}".casefold(),
            }
            if normalized in absent_messages:
                if positively_listed:
                    return lifecycle.InspectObservation.failed(
                        "docker inspect contradicted positive container-list evidence"
                    )
                return lifecycle.InspectObservation.absent()
            return lifecycle.InspectObservation.failed(
                f"docker inspect rc={result.returncode}: {_detail(result.stderr)}"
            )
        try:
            document = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            return lifecycle.InspectObservation.failed(
                f"docker inspect JSON {lifecycle._safe_exception(exc)}"
            )
        if type(document) is not dict:
            return lifecycle.InspectObservation.failed(
                "docker inspect did not return an object"
            )
        config = document.get("Config")
        state_doc = document.get("State")
        labels = config.get("Labels") if type(config) is dict else None
        record = lifecycle.ContainerRecord(
            container_id=document.get("Id"),
            name=document.get("Name"),
            image_id=document.get("Image"),
            labels=labels if type(labels) is dict else {},
            running=(state_doc.get("Running") if type(state_doc) is dict else None),
        )
        return lifecycle.InspectObservation.present(record)

    def inspect(self, ref: str) -> lifecycle.InspectObservation:
        if lifecycle.CID_RE.fullmatch(ref) is not None:
            # A full ID returned by a successful detached launch is
            # authoritative.  Do not gate direct inspection through the
            # eventually-consistent ``docker container ls`` index.
            return self._inspect_id(ref)
        index, error = self._container_index()
        if error is not None:
            return lifecycle.InspectObservation.failed(error)
        matches = [container_id for container_id, name in index.items() if name == ref]
        if len(matches) > 1:
            return lifecycle.InspectObservation.failed(
                "multiple exact-name matches in Docker container index"
            )
        container_id = matches[0] if matches else None
        if container_id is None:
            return lifecycle.InspectObservation.absent()
        return self._inspect_id(container_id, positively_listed=True)

    def sweep_by_labels(
        self, labels: Mapping[str, str]
    ) -> Sequence[lifecycle.InspectObservation]:
        argv = ["docker", "container", "ls", "--all", "--no-trunc", "--quiet"]
        for key, value in sorted(labels.items()):
            argv.extend(("--filter", f"label={key}={value}"))
        result = self._run(argv)
        if result.returncode != 0:
            return [
                lifecycle.InspectObservation.failed(
                    f"docker label sweep rc={result.returncode}: {_detail(result.stderr)}"
                )
            ]
        ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(ids) != len(set(ids)) or any(
            lifecycle.CID_RE.fullmatch(value) is None for value in ids
        ):
            return [
                lifecycle.InspectObservation.failed(
                    "docker label sweep returned malformed or duplicate IDs"
                )
            ]
        # The successful launch result is stronger than list visibility.  Add
        # it to every ownership sweep and inspect it directly.
        all_ids = list(dict.fromkeys([*self.authoritative_launch_ids, *ids]))
        listed = set(ids)
        return [
            self._inspect_id(container_id, positively_listed=container_id in listed)
            for container_id in all_ids
        ]

    def stop_owned(self, container_id: str) -> None:
        if lifecycle.CID_RE.fullmatch(container_id) is None:
            raise ValueError("stop target must be one full container ID")
        try:
            stopped = self._run(
                (
                    "docker",
                    "container",
                    "stop",
                    "--time",
                    str(self.stop_grace_s),
                    container_id,
                )
            )
        except BaseException as exc:
            stop_error = lifecycle._safe_exception(exc)
        else:
            if stopped.returncode == 0:
                return
            stop_error = f"rc={stopped.returncode}: {_detail(stopped.stderr)}"
        try:
            killed = self._run(("docker", "container", "kill", container_id))
        except BaseException as exc:
            kill_error = lifecycle._safe_exception(exc)
        else:
            if killed.returncode == 0:
                return
            kill_error = f"rc={killed.returncode}: {_detail(killed.stderr)}"
        raise RuntimeError(
            f"docker stop failed ({stop_error}); kill failed ({kill_error})"
        )

    def capture_after_stop(self, record: lifecycle.ContainerRecord) -> None:
        if self.diagnostic_capture is not None:
            self.diagnostic_capture(record)
            return
        # The diagnostics directory may itself have a poisoned component; walk
        # it no-follow and create every capture file exclusively.
        try:
            diagnostics_path = str(Path(self.diagnostics_dir) / "x")
            if self.diagnostics_parent_fd is None:
                parent_diagnostics_fd, _name = _open_nofollow_parent(
                    diagnostics_path, create=True
                )
            else:
                parent_diagnostics_fd = os.dup(self.diagnostics_parent_fd)
        except FileNotFoundError:
            raise
        try:
            for suffix, argv in (
                (
                    "inspect.json",
                    ("docker", "container", "inspect", record.container_id),
                ),
                (
                    "server.log",
                    (
                        "docker",
                        "container",
                        "logs",
                        "--timestamps",
                        record.container_id,
                    ),
                ),
            ):
                result = self._run(argv)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"diagnostic {suffix} rc={result.returncode}: {_detail(result.stderr)}"
                    )
                name = f"{record.container_id}.{suffix}"
                _capture_once(parent_diagnostics_fd, name, result.stdout.encode("utf-8"))
        finally:
            if parent_diagnostics_fd is not None:
                os.close(parent_diagnostics_fd)

    def wait(self) -> None:
        self.sleeper(self.retry_interval_s)


T = TypeVar("T")


class R12ContainerReleaseRunner(Generic[T]):
    """Acquire the release flock, launch one exact image, qualify, and clean."""

    def __init__(
        self,
        *,
        image_id: str,
        name: str,
        run_id: str,
        cidfile: Path,
        lockfile: Path,
        diagnostics_dir: Path,
        extra_run_args: Sequence[str],
        entrypoint: str | None = None,
        container_command: Sequence[str],
        backend: CommandBackend | None = None,
        diagnostic_capture: Callable[[lifecycle.ContainerRecord], None] | None = None,
        owner_token: str | None = None,
        stable_sweeps: int = lifecycle.DEFAULT_STABLE_SWEEPS,
        retry_interval_s: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.identity = lifecycle.LaunchIdentity(
            name=name,
            image_id=image_id,
            run_id=run_id,
            owner_token=owner_token or secrets.token_hex(32),
            cleanup_version=R12_CLEANUP_VERSION,
            cidfile=str(cidfile),
        )
        self.cidfile = cidfile
        self.lockfile = lockfile
        self.extra_run_args = tuple(extra_run_args)
        self.entrypoint = entrypoint
        self.container_command = tuple(container_command)
        if type(stable_sweeps) is not int or stable_sweeps < 2:
            raise ValueError("stable_sweeps must be a plain integer >= 2")
        self.stable_sweeps = stable_sweeps
        self.adapter = DockerLifecycleAdapter(
            self.identity,
            backend=backend or SubprocessBackend(),
            diagnostics_dir=diagnostics_dir,
            diagnostic_capture=diagnostic_capture,
            retry_interval_s=retry_interval_s,
            sleeper=sleeper,
        )

    def run(
        self,
        qualify: Callable[[lifecycle.LaunchIdentity, str], T],
    ) -> tuple[T, lifecycle.CleanupReceipt]:
        cid_parent, cid_name = _open_nofollow_parent(str(self.cidfile))
        lock_parent: int | None = None
        diagnostics_parent: int | None = None
        try:
            lock_parent, lock_name = _open_nofollow_parent(
                str(self.lockfile), create=True
            )
            diagnostics_parent, _name = _open_nofollow_parent(
                str(self.adapter.diagnostics_dir / "capture"), create=True
            )
            self.adapter.cidfile_parent_fd = cid_parent
            self.adapter.cidfile_name = cid_name
            self.adapter.diagnostics_parent_fd = diagnostics_parent
            return self._run_with_paths(
                qualify, cid_parent, cid_name, lock_parent, lock_name
            )
        finally:
            self.adapter.cidfile_parent_fd = None
            self.adapter.cidfile_name = None
            self.adapter.diagnostics_parent_fd = None
            for descriptor in (diagnostics_parent, lock_parent, cid_parent):
                if descriptor is not None:
                    os.close(descriptor)

    def _run_with_paths(
        self,
        qualify: Callable[[lifecycle.LaunchIdentity, str], T],
        cid_parent: int,
        cid_name: str,
        lock_parent: int,
        lock_name: str,
    ) -> tuple[T, lifecycle.CleanupReceipt]:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("R12 container release requires the main thread")
        try:
            os.stat(cid_name, dir_fd=cid_parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("refusing a pre-existing R12 cidfile")
        descriptor = os.open(
            lock_name,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=lock_parent,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("R12 lockfile is not a regular file")
        except BaseException:
            os.close(descriptor)
            raise
        released = False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release_lock() -> None:
                nonlocal released
                if released:
                    return
                # Reconciliation has already proved the container stopped or
                # absent.  Make this final descriptor release no-throw so an
                # incidental close error cannot re-enter post-launch cleanup.
                released = True
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except BaseException:
                    pass
                try:
                    os.close(descriptor)
                except BaseException:
                    pass

            def body() -> T:
                container_id = self.adapter.launch(
                    self.extra_run_args,
                    self.container_command,
                    entrypoint=self.entrypoint,
                )
                return qualify(self.identity, container_id)

            # Keep a second gate installed around the production wrapper.  The
            # inner lifecycle gate owns normal body-signal semantics; this
            # outer gate closes the otherwise unprotected transition from an
            # unexpectedly escaped wrapper into emergency reconciliation.
            fallback_gate = lifecycle._TerminationGate()
            primary_result: tuple[T, lifecycle.CleanupReceipt] | None = None
            primary_error: tuple[BaseException, TracebackType | None] | None = None
            with fallback_gate:
                try:
                    try:
                        primary_result = lifecycle.run_with_cleanup_v3(
                            body,
                            self.adapter,
                            self.identity,
                            release_lock,
                            stable_sweeps=self.stable_sweeps,
                        )
                    except BaseException as exc:
                        primary_error = (exc, lifecycle._safe_traceback(exc))
                    finally:
                        fallback_gate.begin_cleanup()
                except BaseException as exc:
                    # A signal between the primary wrapper restoring its
                    # handlers and begin_cleanup() advances the outer gate to
                    # cleanup before raising.  Preserve an earlier wrapper
                    # failure, otherwise retain this interruption.
                    if primary_error is None:
                        primary_error = (exc, lifecycle._safe_traceback(exc))
                    fallback_gate.begin_cleanup()

                # The production wrapper normally returns only after releasing
                # the flock.  If it escapes or returns incorrectly after a
                # launch attempt, reconcile again without ever restoring the
                # process signal handlers while ownership remains unresolved.
                if not released and self.adapter.launch_argv is not None:
                    while not released:
                        try:
                            lifecycle.reconcile_before_unlock(
                                self.adapter,
                                self.identity,
                                release_lock,
                                stable_sweeps=self.stable_sweeps,
                            )
                        except BaseException:
                            if not released:
                                lifecycle._fail_closed_backoff(
                                    lifecycle.MIN_RETRY_BACKOFF_S
                                )

                if primary_error is not None:
                    exc, traceback = primary_error
                    if isinstance(exc, lifecycle.DeferredTermination):
                        signals: list[int] = []
                        for signum in (*exc.signals, *fallback_gate.received):
                            if signum not in signals:
                                signals.append(signum)
                        raise lifecycle.DeferredTermination(signals)
                    raise BaseException.with_traceback(exc, traceback)
                if fallback_gate.received:
                    raise lifecycle.DeferredTermination(fallback_gate.received)
                if primary_result is None:
                    raise RuntimeError("cleanup wrapper returned no result")
                return primary_result
        except BaseException:
            # Setup failures before a launch attempt can release immediately.
            if not released:
                if self.adapter.launch_argv is not None:
                    # A post-launch path is required to resolve above and must
                    # never reach an unconditional unlock here.
                    raise RuntimeError(
                        "post-launch cleanup escaped without an unlock safety proof"
                    )
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
                    released = True
            raise
