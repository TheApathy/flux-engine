#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Pinned Stage 1 ONLY Orchestration Script (`run_fid_gate12_stage1.py`).

Implements all fail-closed admission, isolation, container-ID-only lifecycle,
uncertain-create reconciliation, immutable image-ID verification/launch, and
strict whole-run deadline controls for the
bounded Stage 1 12-prompt fidelity diagnostic (`3` sequential precision arms x
`12` prompts = `36` requests, `max_tokens=64`, descriptive comparison only):

1. Non-Blocking Lock & Hold-File Admission:
   - Refuses immediately (`exit 2`) if the configured build hold file (`FLUX_HOLD_FILE`) exists.
   - Acquires the configured build lock file (`FLUX_LOCK_FILE`) via `fcntl.flock(fd, LOCK_EX | LOCK_NB)`.
     If held by another process, refuses immediately (`exit 2`) without waiting or killing.
2. Reliable Process, Compiler, GPU, Container, Port, and Memory Admission:
   - Scans `/proc/[0-9]*` (`comm`, `exe` symlink basename, and `cmdline` argv[0] basename)
     excluding self and ancestor PIDs to detect any running `flux-server*` (including
     `flux-server-real`, whose 15-char `/proc/<pid>/comm` is truncated to `flux-server-rea`),
     `rustc`, `cargo`, `nvcc`, `ptxas`, or `fake_flux.py`.
   - Checks active Docker containers (`docker ps -a --format ...`) for any conflicting
     `flux`/`memgate`/`fidgate` container or any preexisting container whose name collides
     with this run's planned container names (`fidgate12-<run_id>-<arm>`).
   - Checks `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` for
     any active GPU compute process.
   - Checks TCP port `18899` (`127.0.0.1:18899`) to ensure no listener is bound.
   - Checks host `/proc/meminfo` `MemAvailable >= min_mem_available_mib` (default `106,500 MiB`).
   - Any admission failure refuses cleanly (`exit 2`) and NEVER kills, stops, or interrupts
     another owner's process, container, or job.
3. Exclusive Fresh Output Directory & Cryptographic Pin Verification (Including Immutable Image ID):
   - Creates `out_root` with `mkdir(parents=True, exist_ok=False)` — fails closed if the
     output directory already exists.
   - Verifies exact SHA-256 pins of `fid_gate12.py`, `fid_gate12_manifest.jsonl`, candidate
     binary `fn-dce98957acaf`, checkpoint metadata/tokenizer/template files, and verifies
     via `docker image inspect --format '{{.Id}}'` that both the configured image reference
    * `EXPECTED_IMAGE_ID` (`sha256:5ded06475ebed77b8025e974ad491e15fa7a07a2de36319f58df5d196e2ab633`,
      from saved `v0.9.15.6/local-val/summary.txt` and `memlean/gate/runs/*/run.txt`)
     resolve to the exact expected image digest. Launches `docker run` using the immutable
     `EXPECTED_IMAGE_ID` directly rather than a mutable tag.
4. Preallocated Task Ownership Token, `--cidfile`, Uncertain-Create Reconciliation & ID-Only Teardown:
   - Before invoking `docker run` for an arm, preallocates a unique per-arm ownership token
     (`fidgate12:<run_id>:<arm>:<nonce>`), writes `<arm>/create_intent.json` (with
     `container_name`, `owner_token`, `cidfile_path`, and `expected_image_id`), and passes
     `--cidfile <arm>/container.cid` and `--label io.atlas.fidgate12.owner=<owner_token>`
     to `docker run`.
   - If `docker run` succeeds, times out, or exits non-zero, `finally:` reconciles any
     container that Docker may have created before aborting: it inspects candidates from
     stdout, `container.cid`, and `docker ps -a --filter label=io.atlas.fidgate12.owner=<owner_token>`
     / `--filter name=^/<container_name>$`, and verifies via `docker inspect` that the
     container's `{{.Id}}`, `{{.Name}}`, `{{index .Config.Labels "io.atlas.fidgate12.owner"}}`,
     and `{{.Config.Image}}` match our preallocated `create_intent.json` before adopting its
     64-hex `cid` into `self.owned_container_ids`. Unverified or external containers are NEVER
     signaled or removed. If Docker cannot answer during reconciliation, records
     `uncertain_container_state=True` and `uncertain_server_work=True`.
   - Teardown executes `stop`, `logs`, `rm -f`, and post-`rm` `inspect` as independent bounded
     steps on verified `cid` only.
5. Strict Whole-Run Lifecycle Deadline (Zero Post-Deadline Minima, Signal-Safe):
   - Enforces a single monotonic `run_deadline = t_run_start + total_run_cap_s` (default
     `1,650.0 s`) across all work, reserved cleanup, and abnormal/signal (`SIGINT`/`SIGTERM`)
     paths.
   - Every operation's timeout is strictly bounded by `remaining_total_s()` (`run_deadline - clock()`)
     with ZERO fresh 1s/2s/5s/10s floors granted after the deadline expires. When remaining
     time reaches `<= 0`, commands immediately short-circuit as timed out (`returncode=124`,
     `timed_out=True`) and cleanup records `uncertain_container_state=True` and
     `uncertain_server_work=True` (`server_cancellation_guaranteed=False`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable

# Exact Pinned Hashes
EXPECTED_HARNESS_SHA256 = "fe381d52490cf7f0db87758640bc648b9e9b3b2def7e2c4076624e5c45c2df25"
EXPECTED_MANIFEST_SHA256 = "34f27ce71072c56a9d3fd34216500b040429197775f9b2beb7148ce0aee4de65"
EXPECTED_BINARY_SHA256 = "663c5259a3ca5363a5138f4598aa0b6ab0848dbfb5027e67c5968dcfea6ea487"
EXPECTED_IMAGE_ID = "sha256:5ded06475ebed77b8025e974ad491e15fa7a07a2de36319f58df5d196e2ab633"
OWNER_LABEL_KEY = "io.atlas.fidgate12.owner"

EXPECTED_MODEL_FILE_SHA256 = {
    "export-manifest.json": "e661248072f16123c4b53ff494a395f64fe9c7877c9a55697142d859c0237880",
    "HYBRID.json": "c0ebf65ab2466773485b7b5a7a90f7cad2c6130d03cda21d9ab6ddb4f168a45f",
    "hf_quant_config.json": "369953bc72fbe6d5af5558fae1a4492b61bad9429110d471163eaedf55644673",
    "config.json": "2cb54e1f854b60eb62268cef738252957a2af4c457e55966fa442e1406b66418",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "158ec34979f1cb4dab73e28d4a0127517bba1d614a5d824041000cddb7659380",
    "chat_template.jinja": "650a598c634ad303dd4ecbd3c0b320116fa9a930d9eaaf04b25bddeef0f249e1",
    "model.safetensors.index.json": "e65210a65667dae8d83edde5ee7f833f68a79110f54264445e421f6c26ad7ccb",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK_PATH = Path(os.environ.get("FLUX_LOCK_FILE", str(Path("/tmp") / "flux-build.lock")))
DEFAULT_HOLD_PATH = Path(os.environ.get("FLUX_HOLD_FILE", str(Path("/tmp") / "flux-build.hold")))
DEFAULT_BINARY_PATH = Path(
    os.environ.get("FLUX_BINARY_PATH", str(REPO_ROOT / "target" / "release" / "flux-server"))
)
DEFAULT_MODEL_DIR = Path(
    os.environ.get("FLUX_MODEL_DIR", str(Path.home() / "models" / "Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"))
)
DEFAULT_IMAGE = "flux-engine:v0.9.15.6-rc1"
DEFAULT_PORT = 18899

HERE = Path(__file__).resolve().parent
DEFAULT_HARNESS_PATH = HERE / "fid_gate12.py"
DEFAULT_MANIFEST_PATH = HERE / "fid_gate12_manifest.jsonl"

# Hard Budget Caps (Seconds)
DEFAULT_TOTAL_RUN_CAP_S = 1650.0
DEFAULT_STARTUP_CAP_S = 180.0
DEFAULT_COLLECT_PANEL_CAP_S = 300.0
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_CLEANUP_CAP_S = 60.0
DEFAULT_COMPARE_CAP_S = 30.0

DEFAULT_MIN_MEM_AVAILABLE_MIB = 106500
DEFAULT_FLOOR_MIB = 12288


@dataclass(frozen=True)
class ArmSpec:
    name: str
    description: str
    extra_sets: tuple[str, ...]


STAGE1_ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        name="ref_R",
        description="Same-engine all-BF16 dense precision reference (DENSE_NATIVE=1 + memlean coexistence)",
        extra_sets=(
            "ATLAS_QWEN4_DENSE_NATIVE=1",
            "ATLAS_QWEN4_DENSE_DEDUP=1",
            "ATLAS_QWEN4_RELEASE_MXFP8_SOURCES=1",
            "ATLAS_QWEN4_RELEASE_BF16_QKVZ=1",
        ),
    ),
    ArmSpec(
        name="ctl_D",
        description="Wave 5 default NVFP4 requant control",
        extra_sets=(),
    ),
    ArmSpec(
        name="cand_C",
        description="Candidate decode-only FP8 dense + BF16 router/HC/head",
        extra_sets=(
            "ATLAS_QWEN4_DENSE_FP8=1",
            "ATLAS_QWEN4_ROUTER_BF16=1",
            "ATLAS_QWEN4_HC_BF16=1",
            "ATLAS_QWEN4_LMHEAD_BF16=1",
        ),
    ),
)

SHARED_SETS: tuple[str, ...] = (
    "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=0",
    "ATLAS_QWEN4_PLE_SEGMENTED_GRAPHS=0",
)

SHARED_ENVS: tuple[str, ...] = (
    "FLUX_MODE=auto",
    "FLUX_DISABLE_SPECULATION=1",
    "FLUX_MAX_SEQ_LEN=16384",
    "FLUX_LOG_FILE=/results/server.log",
    "PYTHONDONTWRITEBYTECODE=1",
)


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    elapsed_s: float = 0.0


class Stage1AdmissionError(Exception):
    """Raised when pre-flight admission or pin verification refuses to start (`exit 2`)."""


class Stage1ExecutionError(Exception):
    """Raised when startup, collection, deadline, signal, or cleanup fails during execution (`exit 1`)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with tmp.open("w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def is_valid_container_id(cid: str) -> bool:
    s = cid.strip()
    return len(s) >= 12 and len(s) <= 64 and all(c in "0123456789abcdef" for c in s)


def is_docker_not_found_error(res: CmdResult) -> bool:
    """Return True ONLY when `docker inspect` explicitly reports that the object does not exist.

    Any other nonzero returncode (daemon error, socket error, permission error, or timeout)
    is NOT proof of absence.
    """
    if res.timed_out or res.returncode == 0:
        return False
    msg = (res.stderr + "\n" + res.stdout).lower()
    return "no such object:" in msg or "no such container:" in msg


def default_cmd_runner(argv: list[str], timeout_s: float) -> CmdResult:
    t0 = time.monotonic()
    if timeout_s <= 0:
        return CmdResult(
            returncode=124,
            stdout="",
            stderr=f"command_deadline_expired_before_start ({timeout_s:.3f}s)",
            timed_out=True,
            elapsed_s=0.0,
        )
    try:
        cp = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return CmdResult(
            returncode=cp.returncode,
            stdout=cp.stdout or "",
            stderr=cp.stderr or "",
            timed_out=False,
            elapsed_s=time.monotonic() - t0,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CmdResult(
            returncode=124,
            stdout=out,
            stderr=err + f"\n[TimeoutExpired after {timeout_s:.2f}s]",
            timed_out=True,
            elapsed_s=time.monotonic() - t0,
        )


def _ancestor_pids(start_pid: int | None = None, proc_root: Path = Path("/proc")) -> set[int]:
    pids: set[int] = set()
    cur = os.getpid() if start_pid is None else start_pid
    while cur > 1 and cur not in pids:
        pids.add(cur)
        stat_path = proc_root / str(cur) / "stat"
        try:
            raw = stat_path.read_text(encoding="utf-8", errors="replace")
            rparen = raw.rfind(")")
            if rparen == -1:
                break
            fields = raw[rparen + 2 :].split()
            ppid = int(fields[1])
            cur = ppid
        except Exception:  # noqa: BLE001
            break
    return pids


def scan_conflicting_processes(proc_root: Path = Path("/proc")) -> list[str]:
    """Scan `/proc` for any running `flux-server*` (including `flux-server-real`), compiler, or fake server."""
    excluded = _ancestor_pids(proc_root=proc_root)
    conflicts: list[str] = []
    compiler_names = frozenset({"rustc", "cargo", "nvcc", "ptxas"})

    if not proc_root.is_dir():
        return conflicts

    for entry in sorted(proc_root.iterdir(), key=lambda p: p.name):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue

        comm = ""
        exe_base = ""
        argv: list[str] = []
        try:
            comm = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except Exception:  # noqa: BLE001
            pass
        try:
            exe_target = os.readlink(entry / "exe")
            exe_base = os.path.basename(exe_target.split(" (deleted)")[0])
        except Exception:  # noqa: BLE001
            pass
        try:
            raw_cmd = (entry / "cmdline").read_bytes()
            argv = [
                part.decode("utf-8", errors="replace")
                for part in raw_cmd.split(b"\x00")
                if part
            ]
        except Exception:  # noqa: BLE001
            pass

        argv0_base = os.path.basename(argv[0]) if argv else ""
        cmdline_joined = " ".join(argv)

        matched_reason: str | None = None
        for label, cand in (("comm", comm), ("exe", exe_base), ("argv0", argv0_base)):
            if not cand:
                continue
            if cand.startswith("flux-server"):
                matched_reason = f"{label}={cand}"
                break
            if cand in compiler_names:
                matched_reason = f"{label}={cand}"
                break

        if matched_reason is None and "fake_flux.py" in cmdline_joined and "--port" in cmdline_joined:
            matched_reason = "cmdline=fake_flux.py"

        if matched_reason is not None:
            conflicts.append(f"pid={pid}({matched_reason})")

    return conflicts


def read_mem_available_mib(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    raw = meminfo_path.read_text(encoding="utf-8", errors="replace")
    for line in raw.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            return int(parts[1]) // 1024
    raise Stage1AdmissionError(f"MemAvailable not found in {meminfo_path}")


def default_port_probe(host: str, port: int) -> bool:
    """Return True if a TCP listener is already accepting connections on `(host, port)`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def default_health_probe(url: str, timeout_s: float) -> bool:
    import urllib.request

    if timeout_s <= 0:
        return False
    try:
        with urllib.request.urlopen(url, timeout=min(2.0, timeout_s)) as resp:
            return getattr(resp, "status", 200) == 200
    except Exception:  # noqa: BLE001
        return False


class Stage1Orchestrator:
    """Fail-closed Stage 1 orchestrator with injectable system dependencies for offline CPU testing."""

    def __init__(
        self,
        *,
        out_root: Path,
        run_id: str,
        lock_path: Path = DEFAULT_LOCK_PATH,
        hold_path: Path = DEFAULT_HOLD_PATH,
        binary_path: Path = DEFAULT_BINARY_PATH,
        model_dir: Path = DEFAULT_MODEL_DIR,
        harness_path: Path = DEFAULT_HARNESS_PATH,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        image: str = DEFAULT_IMAGE,
        expected_image_id: str = EXPECTED_IMAGE_ID,
        port: int = DEFAULT_PORT,
        total_run_cap_s: float = DEFAULT_TOTAL_RUN_CAP_S,
        startup_cap_s: float = DEFAULT_STARTUP_CAP_S,
        collect_panel_cap_s: float = DEFAULT_COLLECT_PANEL_CAP_S,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        cleanup_cap_s: float = DEFAULT_CLEANUP_CAP_S,
        compare_cap_s: float = DEFAULT_COMPARE_CAP_S,
        min_mem_available_mib: int = DEFAULT_MIN_MEM_AVAILABLE_MIB,
        floor_mib: int = DEFAULT_FLOOR_MIB,
        verify_pins: bool = True,
        install_signal_handlers: bool = True,
        cmd_runner: Callable[[list[str], float], CmdResult] = default_cmd_runner,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        proc_scanner: Callable[[], list[str]] = scan_conflicting_processes,
        mem_reader: Callable[[], int] = read_mem_available_mib,
        port_busy_probe: Callable[[str, int], bool] = default_port_probe,
        health_probe: Callable[[str, float], bool] = default_health_probe,
        nonce_factory: Callable[[], str] = lambda: secrets.token_hex(8),
    ) -> None:
        self.out_root = out_root.resolve()
        self.run_id = run_id
        self.lock_path = lock_path
        self.hold_path = hold_path
        self.binary_path = binary_path
        self.model_dir = model_dir
        self.harness_path = harness_path
        self.manifest_path = manifest_path
        self.image = image
        self.expected_image_id = expected_image_id
        self.resolved_image_id: str | None = None
        self.port = port

        self.total_run_cap_s = float(total_run_cap_s)
        self.startup_cap_s = float(startup_cap_s)
        self.collect_panel_cap_s = float(collect_panel_cap_s)
        self.request_timeout_s = float(request_timeout_s)
        self.cleanup_cap_s = float(cleanup_cap_s)
        self.compare_cap_s = float(compare_cap_s)
        self.min_mem_available_mib = int(min_mem_available_mib)
        self.floor_mib = int(floor_mib)
        self.verify_pins = bool(verify_pins)
        self.install_signal_handlers = bool(install_signal_handlers)

        self.cmd_runner = cmd_runner
        self.clock = clock
        self.sleep_fn = sleep_fn
        self.proc_scanner = proc_scanner
        self.mem_reader = mem_reader
        self.port_busy_probe = port_busy_probe
        self.health_probe = health_probe
        self.nonce_factory = nonce_factory

        self.t_start = self.clock()
        self.run_deadline = self.t_start + self.total_run_cap_s
        self.owned_container_ids: set[str] = set()
        self._lock_fd: int | None = None

    def elapsed_s(self) -> float:
        return self.clock() - self.t_start

    def remaining_total_s(self) -> float:
        """Return exact remaining whole-task lifecycle budget (`<= 0.0` once `run_deadline` is reached)."""
        return max(0.0, self.run_deadline - self.clock())

    def remaining_for_work_s(self, arms_remaining_to_clean: int) -> float:
        """Return remaining wall time after reserving cleanup budget for `arms_remaining_to_clean` plus compare."""
        reserved = (arms_remaining_to_clean * self.cleanup_cap_s) + self.compare_cap_s
        return (self.run_deadline - self.clock()) - reserved

    def _run_bounded(self, argv: list[str], cap_s: float, ceiling_deadline: float | None = None) -> CmdResult:
        """Execute `argv` bounded strictly by `min(cap_s, remaining_total_s(), ceiling_deadline - clock())`.

        Never grants any fresh positive timeout minimum after `run_deadline` or `ceiling_deadline`
        has expired: if remaining budget is `<= 0`, returns `CmdResult(124, ..., timed_out=True)`
        immediately without spawning a child process.
        """
        now = self.clock()
        rem = self.run_deadline - now
        if ceiling_deadline is not None:
            rem = min(rem, ceiling_deadline - now)
        budget = min(float(cap_s), rem)
        if budget <= 0.0:
            return CmdResult(
                returncode=124,
                stdout="",
                stderr=f"deadline_expired_before_command ({argv[0] if argv else 'cmd'}, remaining={budget:.3f}s)",
                timed_out=True,
                elapsed_s=0.0,
            )
        return self.cmd_runner(argv, budget)

    def _acquire_nonblocking_lock(self) -> None:
        if self.hold_path.exists():
            raise Stage1AdmissionError(f"build_hold_file_present: {self.hold_path}")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise Stage1AdmissionError(
                f"lock_busy_held_by_another_owner: {self.lock_path}"
            ) from exc
        self._lock_fd = fd

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                pass
            try:
                os.close(self._lock_fd)
            except Exception:  # noqa: BLE001
                pass
            self._lock_fd = None

    def _verify_all_pins(self) -> dict[str, str]:
        pins: dict[str, str] = {}
        for label, path, exp in (
            ("harness", self.harness_path, EXPECTED_HARNESS_SHA256),
            ("manifest", self.manifest_path, EXPECTED_MANIFEST_SHA256),
            ("binary", self.binary_path, EXPECTED_BINARY_SHA256),
        ):
            if not path.is_file():
                raise Stage1AdmissionError(f"missing_pinned_{label}: {path}")
            actual = sha256_file(path)
            if self.verify_pins and actual != exp:
                raise Stage1AdmissionError(
                    f"{label}_sha256_mismatch: expected {exp}, got {actual} ({path})"
                )
            pins[f"{label}_sha256"] = actual

        if not self.model_dir.is_dir():
            raise Stage1AdmissionError(f"missing_model_dir: {self.model_dir}")
        for rel_name, exp_sha in EXPECTED_MODEL_FILE_SHA256.items():
            fpath = self.model_dir / rel_name
            if not fpath.is_file():
                raise Stage1AdmissionError(f"missing_model_file: {fpath}")
            if self.verify_pins:
                actual_sha = sha256_file(fpath)
                if actual_sha != exp_sha:
                    raise Stage1AdmissionError(
                        f"model_file_sha256_mismatch ({rel_name}): expected {exp_sha}, got {actual_sha}"
                    )
                pins[f"model:{rel_name}"] = actual_sha

        # Verify immutable Docker image ID (Control 3) and pin it for `docker run`
        img_res = self._run_bounded(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
            10.0,
        )
        if img_res.timed_out or img_res.returncode != 0:
            raise Stage1AdmissionError(
                f"docker_image_inspect_failed ({self.image}, rc={img_res.returncode}, "
                f"timed_out={img_res.timed_out}): {img_res.stderr.strip()}"
            )
        actual_image_id = img_res.stdout.strip().splitlines()[0].strip() if img_res.stdout.strip() else ""
        if self.verify_pins and actual_image_id != self.expected_image_id:
            raise Stage1AdmissionError(
                f"image_id_mismatch ({self.image}): expected {self.expected_image_id}, got {actual_image_id}"
            )
        # Also verify `docker image inspect` on the immutable ID itself so `docker run <expected_image_id>` is guaranteed valid
        id_res = self._run_bounded(
            ["docker", "image", "inspect", "--format", "{{.Id}}", actual_image_id],
            10.0,
        )
        if id_res.timed_out or id_res.returncode != 0 or id_res.stdout.strip() != actual_image_id:
            raise Stage1AdmissionError(
                f"immutable_image_id_inspect_failed ({actual_image_id}, rc={id_res.returncode}, "
                f"timed_out={id_res.timed_out}): {id_res.stderr.strip()}"
            )
        self.resolved_image_id = actual_image_id
        pins["image_ref"] = self.image
        pins["image_id"] = actual_image_id
        return pins

    def _check_live_admission(self, planned_container_names: set[str]) -> int:
        # 1. Host processes (/proc comm, exe, cmdline)
        proc_conflicts = self.proc_scanner()
        if proc_conflicts:
            raise Stage1AdmissionError(
                f"conflicting_host_processes_active: {', '.join(proc_conflicts)}"
            )

        # 2. TCP port check
        if self.port_busy_probe("127.0.0.1", self.port):
            raise Stage1AdmissionError(f"port_busy: 127.0.0.1:{self.port}")

        # 3. Active Docker containers & name collisions (`docker ps -a`)
        ps_res = self._run_bounded(
            ["docker", "ps", "-a", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"],
            10.0,
        )
        if ps_res.timed_out or ps_res.returncode != 0:
            raise Stage1AdmissionError(
                f"docker_ps_failed (rc={ps_res.returncode}, timed_out={ps_res.timed_out}): {ps_res.stderr.strip()}"
            )
        for line in ps_res.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            cid = parts[0].strip() if len(parts) > 0 else ""
            cname = parts[1].strip() if len(parts) > 1 else ""
            cstatus = parts[2].strip() if len(parts) > 2 else ""
            cimage = parts[3].strip() if len(parts) > 3 else ""
            if cname in planned_container_names:
                raise Stage1AdmissionError(
                    f"container_name_collision: '{cname}' already exists (id={cid}, status={cstatus})"
                )
            is_running = cstatus.lower().startswith("up")
            if is_running and (
                any(tok in cname.lower() for tok in ("flux", "memgate", "fidgate"))
                or "flux-engine" in cimage.lower()
                or (self.expected_image_id and self.expected_image_id[:19] in cimage.lower())
            ):
                raise Stage1AdmissionError(
                    f"conflicting_running_container: name='{cname}' id={cid} image='{cimage}'"
                )

        # 4. Active GPU compute apps (`nvidia-smi`)
        smi_res = self._run_bounded(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            10.0,
        )
        if smi_res.timed_out or smi_res.returncode != 0:
            raise Stage1AdmissionError(
                f"nvidia_smi_check_failed (rc={smi_res.returncode}, timed_out={smi_res.timed_out}): {smi_res.stderr.strip()}"
            )
        active_gpu_apps = [ln.strip() for ln in smi_res.stdout.splitlines() if ln.strip()]
        if active_gpu_apps:
            raise Stage1AdmissionError(
                f"conflicting_gpu_compute_processes: {'; '.join(active_gpu_apps)}"
            )

        # 5. Host MemAvailable admission
        mem_mib = self.mem_reader()
        if mem_mib < self.min_mem_available_mib:
            raise Stage1AdmissionError(
                f"insufficient_mem_available: {mem_mib} MiB < required {self.min_mem_available_mib} MiB"
            )
        return mem_mib

    def _reconcile_intended_container(
        self,
        *,
        arm_dir: Path,
        container_name: str,
        owner_token: str,
        cidfile_path: Path,
        expected_image_ref: str,
        stdout_cid_hint: str,
        ceiling_deadline: float,
    ) -> dict[str, Any]:
        """Reconcile whether `docker run` created a task-owned container even if `docker run` timed out or failed (Control 1).

        Candidates are gathered from:
        1) `stdout_cid_hint` (if `docker run` emitted a container ID before exiting/timing out),
        2) `cidfile_path` (`--cidfile <arm>/container.cid`),
        3) `docker ps -a --filter label=io.atlas.fidgate12.owner=<owner_token>`,
        4) `docker ps -a --filter name=^/<container_name>$`.

        For every candidate, we run `docker inspect` to verify its full 64-hex `{{.Id}}`,
        `{{.Name}}`, `{{index .Config.Labels "io.atlas.fidgate12.owner"}}`, and `{{.Config.Image}}`
        against our preallocated `create_intent.json`.
        - A container is adopted into `self.owned_container_ids` ONLY if its verified label
          equals `owner_token` AND its name equals `container_name` AND its image matches
          `expected_image_ref`.
        - An unrelated container (mismatched label/owner) is NEVER adopted or signaled.
        - If any lookup/inspect command times out or fails while reconciling an uncertain create,
          `reconcile_uncertain=True` is returned so the caller records `uncertain_container_state=True`.
        """
        candidates: list[str] = []

        def _add_cand(raw: str) -> None:
            s = raw.strip()
            if s and s not in candidates:
                candidates.append(s)

        if stdout_cid_hint and is_valid_container_id(stdout_cid_hint):
            _add_cand(stdout_cid_hint)

        if cidfile_path.is_file():
            try:
                cid_from_file = cidfile_path.read_text(encoding="utf-8", errors="replace").strip()
                if is_valid_container_id(cid_from_file):
                    _add_cand(cid_from_file)
            except Exception:  # noqa: BLE001
                pass

        reconcile_uncertain = False
        ps_queries = [
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"label={OWNER_LABEL_KEY}={owner_token}",
                "--format",
                "{{.ID}}",
            ],
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name=^/{container_name}$",
                "--format",
                "{{.ID}}",
            ],
        ]
        for q_argv in ps_queries:
            q_res = self._run_bounded(q_argv, 5.0, ceiling_deadline=ceiling_deadline)
            if q_res.timed_out or q_res.returncode != 0:
                reconcile_uncertain = True
            else:
                for ln in q_res.stdout.splitlines():
                    if is_valid_container_id(ln.strip()):
                        _add_cand(ln.strip())

        verified_cid: str | None = None
        rejected_candidates: list[dict[str, str]] = []
        not_found_candidates: list[str] = []
        inspect_failed_candidates: list[dict[str, Any]] = []
        inspect_format = (
            f"{{{{.Id}}}}\t{{{{.Name}}}}\t{{{{index .Config.Labels \"{OWNER_LABEL_KEY}\"}}}}\t{{{{.Config.Image}}}}"
        )

        for cand in candidates:
            insp = self._run_bounded(
                ["docker", "inspect", "--format", inspect_format, cand],
                5.0,
                ceiling_deadline=ceiling_deadline,
            )
            if insp.timed_out:
                reconcile_uncertain = True
                inspect_failed_candidates.append(
                    {"candidate": cand, "rc": insp.returncode, "timed_out": True, "stderr": insp.stderr.strip()}
                )
                continue
            if insp.returncode != 0:
                if is_docker_not_found_error(insp):
                    not_found_candidates.append(cand)
                else:
                    # Nonzero inspect without explicit not-found proof is NOT absence
                    reconcile_uncertain = True
                    inspect_failed_candidates.append(
                        {"candidate": cand, "rc": insp.returncode, "timed_out": False, "stderr": insp.stderr.strip()}
                    )
                continue
            first_line = insp.stdout.splitlines()[0] if insp.stdout.splitlines() else ""
            parts = first_line.split("\t")
            full_id = parts[0].strip() if len(parts) > 0 else ""
            raw_name = parts[1].strip().lstrip("/") if len(parts) > 1 else ""
            lbl_owner = parts[2].strip() if len(parts) > 2 else ""
            cfg_image = parts[3].strip() if len(parts) > 3 else ""

            if (
                is_valid_container_id(full_id)
                and raw_name == container_name
                and lbl_owner == owner_token
                and cfg_image == expected_image_ref
            ):
                verified_cid = full_id
                self.owned_container_ids.add(full_id)
                try:
                    (arm_dir / "container.id").write_text(full_id + "\n", encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
                break
            else:
                rejected_candidates.append(
                    {
                        "candidate": cand,
                        "full_id": full_id,
                        "name": raw_name,
                        "owner_label": lbl_owner,
                        "image": cfg_image,
                    }
                )

        # Absence is proven ONLY when all ps queries succeeded (`not reconcile_uncertain`),
        # `verified_cid is None`, and every candidate was either explicitly `No such object`
        # or verified as belonging to a different owner (`rejected_candidates`), with zero
        # `inspect_failed_candidates`.
        absence_proven = (
            verified_cid is None
            and not reconcile_uncertain
            and len(inspect_failed_candidates) == 0
        )

        return {
            "verified_cid": verified_cid,
            "candidates_checked": candidates,
            "not_found_candidates": not_found_candidates,
            "inspect_failed_candidates": inspect_failed_candidates,
            "rejected_candidates": rejected_candidates,
            "absence_proven": absence_proven,
            "reconcile_uncertain": reconcile_uncertain,
        }

    def _cleanup_owned_container(
        self,
        cid: str,
        arm_dir: Path,
        cleanup_deadline: float,
    ) -> dict[str, Any]:
        """Stop, capture logs, and remove a task-owned container by exact container ID (`cid`).

        Strictly bounded by `min(cleanup_deadline, self.run_deadline)` with ZERO fresh
        post-deadline timeout minima (Control 2). Never chains `stop && logs && rm`.
        """
        if cid not in self.owned_container_ids:
            raise Stage1ExecutionError(
                f"refusing_to_manage_unowned_container_id: {cid}"
            )
        t0 = self.clock()
        effective_deadline = min(cleanup_deadline, self.run_deadline)

        def _rem() -> float:
            return max(0.0, effective_deadline - self.clock())

        # Step 1: docker stop <cid>
        rem1 = _rem()
        stop_timeout_arg = max(1, int(min(30.0, rem1 * 0.5))) if rem1 > 0.0 else 1
        stop_res = self._run_bounded(
            ["docker", "stop", "-t", str(stop_timeout_arg), cid],
            min(stop_timeout_arg + 5.0, rem1),
            ceiling_deadline=effective_deadline,
        )

        # Step 2: docker logs <cid> (always attempted if budget > 0, even if `stop` failed/timed out)
        logs_res = self._run_bounded(
            ["docker", "logs", cid],
            10.0,
            ceiling_deadline=effective_deadline,
        )
        try:
            (arm_dir / "docker.log").write_text(
                logs_res.stdout + ("\n--- STDERR ---\n" + logs_res.stderr if logs_res.stderr else ""),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

        # Step 3: docker rm -f <cid> (ALWAYS attempted if budget > 0, regardless of stop/logs outcome)
        rm_res = self._run_bounded(
            ["docker", "rm", "-f", cid],
            15.0,
            ceiling_deadline=effective_deadline,
        )

        # Step 4: Verify container is actually gone via `docker inspect`
        insp_res = self._run_bounded(
            ["docker", "inspect", "--format", "{{.State.Status}}", cid],
            5.0,
            ceiling_deadline=effective_deadline,
        )
        # Absence after `rm -f` is proven ONLY when `docker inspect` explicitly reports `No such object`/`No such container`
        container_removed = (
            not rm_res.timed_out
            and rm_res.returncode == 0
            and is_docker_not_found_error(insp_res)
        )
        uncertain_container_state = not container_removed

        return {
            "container_id": cid,
            "stop_rc": stop_res.returncode,
            "stop_timed_out": stop_res.timed_out,
            "logs_rc": logs_res.returncode,
            "logs_timed_out": logs_res.timed_out,
            "rm_rc": rm_res.returncode,
            "rm_timed_out": rm_res.timed_out,
            "inspect_after_rm_rc": insp_res.returncode,
            "inspect_after_rm_timed_out": insp_res.timed_out,
            "container_removed": container_removed,
            "uncertain_container_state": uncertain_container_state,
            "cleanup_elapsed_s": round(self.clock() - t0, 6),
        }

    def _run_single_arm(
        self,
        arm: ArmSpec,
        arm_idx: int,
        total_arms: int,
        planned_names: set[str],
    ) -> dict[str, Any]:
        arms_remaining_to_clean = total_arms - arm_idx
        arm_dir = self.out_root / arm.name
        arm_dir.mkdir(parents=True, exist_ok=False)

        container_name = f"fidgate12-{self.run_id}-{arm.name}"
        owner_token = f"fidgate12:{self.run_id}:{arm.name}:{self.nonce_factory()}"
        cidfile_path = arm_dir / "container.cid"
        launch_image_ref = self.resolved_image_id or self.expected_image_id

        pre_mem_mib = self._check_live_admission(planned_names)

        work_rem_s = self.remaining_for_work_s(arms_remaining_to_clean)
        if work_rem_s <= 0:
            raise Stage1ExecutionError(
                f"[{arm.name}] run_deadline_exceeded_before_arm_start: remaining_for_work_s={work_rem_s:.2f}s"
            )

        # Record intended ownership BEFORE invoking `docker run` (Control 1)
        create_intent = {
            "arm": arm.name,
            "container_name": container_name,
            "owner_label_key": OWNER_LABEL_KEY,
            "owner_token": owner_token,
            "cidfile_path": str(cidfile_path),
            "expected_image_id": launch_image_ref,
            "recorded_at_elapsed_s": round(self.elapsed_s(), 6),
        }
        write_json_atomic(arm_dir / "create_intent.json", create_intent)

        cid: str | None = None
        create_attempted = False
        create_outcome_uncertain = False
        stdout_cid_hint = ""
        floor_breached = False
        floor_breach_msg: str | None = None
        stop_watch_event = threading.Event()
        watch_thread: threading.Thread | None = None
        mem_samples: list[int] = [pre_mem_mib]

        arm_meta: dict[str, Any] = {
            "arm": arm.name,
            "description": arm.description,
            "container_name": container_name,
            "owner_token": owner_token,
            "launched_image_id": launch_image_ref,
            "container_id": None,
            "status": "starting",
            "pre_mem_available_mib": pre_mem_mib,
            "min_mem_available_mib": pre_mem_mib,
            "extra_sets": list(arm.extra_sets),
            "shared_sets": list(SHARED_SETS),
            "shared_envs": list(SHARED_ENVS),
            "startup_elapsed_s": None,
            "collect_elapsed_s": None,
            "floor_breached": False,
            "uncertain_server_work": False,
            "uncertain_container_state": False,
            "server_cancellation_guaranteed": False,
            "reconciliation": None,
            "cleanup": None,
            "error": None,
        }
        write_json_atomic(arm_dir / "arm_meta.json", arm_meta)

        try:
            model_name = self.model_dir.name
            docker_cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--cidfile",
                str(cidfile_path),
                "--label",
                f"{OWNER_LABEL_KEY}={owner_token}",
                "--gpus",
                "all",
                "--ipc=host",
                "--ulimit",
                "memlock=8388608",
                "-p",
                f"127.0.0.1:{self.port}:8899",
                "-v",
                f"{self.model_dir}:/models/{model_name}:ro",
                "-v",
                f"{self.binary_path}:/opt/flux/bin/flux-server-real:ro",
                "-v",
                f"{arm_dir}:/results",
            ]
            for env_kv in SHARED_ENVS:
                docker_cmd.extend(["-e", env_kv])
            # Launch by verified immutable `launch_image_ref` (`sha256:6b2de5f9...`), NOT mutable tag (Control 3)
            docker_cmd.extend([launch_image_ref, "flux", "serve", "--model", model_name])
            for s_kv in SHARED_SETS + arm.extra_sets:
                docker_cmd.extend(["--set", s_kv])

            run_timeout = min(30.0, max(0.0, self.remaining_for_work_s(arms_remaining_to_clean)))
            create_attempted = True
            run_res = self._run_bounded(docker_cmd, run_timeout)
            stdout_cid_hint = (
                run_res.stdout.strip().splitlines()[0].strip() if run_res.stdout.strip() else ""
            )

            if run_res.timed_out or run_res.returncode != 0:
                create_outcome_uncertain = True
                arm_meta["uncertain_container_state"] = True
                if run_res.timed_out:
                    arm_meta["uncertain_server_work"] = True
                raise Stage1ExecutionError(
                    f"[{arm.name}] docker_run_failed (rc={run_res.returncode}, timed_out={run_res.timed_out}): {run_res.stderr.strip()}"
                )

            if not is_valid_container_id(stdout_cid_hint):
                create_outcome_uncertain = True
                arm_meta["uncertain_container_state"] = True
                raise Stage1ExecutionError(
                    f"[{arm.name}] invalid_container_id_from_docker_run: {stdout_cid_hint!r}"
                )

            # Verify ownership of the created container before recording it
            reconcile_info = self._reconcile_intended_container(
                arm_dir=arm_dir,
                container_name=container_name,
                owner_token=owner_token,
                cidfile_path=cidfile_path,
                expected_image_ref=launch_image_ref,
                stdout_cid_hint=stdout_cid_hint,
                ceiling_deadline=self.run_deadline,
            )
            arm_meta["reconciliation"] = reconcile_info
            if reconcile_info["verified_cid"] is None:
                create_outcome_uncertain = True
                arm_meta["uncertain_container_state"] = True
                raise Stage1ExecutionError(
                    f"[{arm.name}] created_container_failed_ownership_verification: {reconcile_info}"
                )
            cid = reconcile_info["verified_cid"]
            arm_meta["container_id"] = cid
            arm_meta["status"] = "waiting_for_health"
            write_json_atomic(arm_dir / "arm_meta.json", arm_meta)

            # Start 1 Hz memory floor watchdog strictly targeting verified `cid`
            def _watchdog() -> None:
                nonlocal floor_breached, floor_breach_msg
                while not stop_watch_event.is_set():
                    try:
                        m = self.mem_reader()
                        mem_samples.append(m)
                        with (arm_dir / "memtrace.tsv").open("a", encoding="utf-8") as mf:
                            mf.write(f"{self.elapsed_s():.3f}\t{m}\n")
                        if m < self.floor_mib:
                            floor_breached = True
                            floor_breach_msg = (
                                f"FLOOR_BREACH: MemAvailable {m} MiB < floor {self.floor_mib} MiB"
                            )
                            (arm_dir / "ABORT").write_text(floor_breach_msg + "\n", encoding="utf-8")
                            # Strictly kill ONLY our verified `cid`, bounded by remaining total budget
                            self._run_bounded(["docker", "kill", cid], 10.0)
                            return
                    except Exception:  # noqa: BLE001
                        pass
                    stop_watch_event.wait(1.0)

            watch_thread = threading.Thread(target=_watchdog, daemon=True)
            watch_thread.start()

            # Wait for `/health` within `min(startup_cap_s, remaining_for_work_s)`
            t_health0 = self.clock()
            startup_budget = min(
                self.startup_cap_s,
                self.remaining_for_work_s(arms_remaining_to_clean),
            )
            if startup_budget <= 0:
                arm_meta["uncertain_server_work"] = True
                raise Stage1ExecutionError(
                    f"[{arm.name}] startup_deadline_expired_before_health_poll"
                )
            startup_deadline = min(self.run_deadline, t_health0 + startup_budget)
            healthy = False
            health_url = f"http://127.0.0.1:{self.port}/health"

            while self.clock() < startup_deadline:
                if floor_breached:
                    raise Stage1ExecutionError(f"[{arm.name}] {floor_breach_msg}")
                rem_startup = startup_deadline - self.clock()
                if rem_startup <= 0:
                    break
                if self.health_probe(health_url, min(2.0, rem_startup)):
                    healthy = True
                    break
                # Verify container is still running by ID, strictly bounded by `startup_deadline`
                insp = self._run_bounded(
                    ["docker", "inspect", "-f", "{{.State.Running}}", cid],
                    5.0,
                    ceiling_deadline=startup_deadline,
                )
                if insp.timed_out:
                    arm_meta["uncertain_server_work"] = True
                    break
                if insp.returncode != 0 or insp.stdout.strip() != "true":
                    raise Stage1ExecutionError(
                        f"[{arm.name}] container_exited_during_startup (cid={cid[:12]})"
                    )
                sleep_slice = min(1.0, max(0.0, startup_deadline - self.clock()))
                if sleep_slice > 0:
                    self.sleep_fn(sleep_slice)

            arm_meta["startup_elapsed_s"] = round(self.clock() - t_health0, 6)
            if not healthy:
                arm_meta["uncertain_server_work"] = True
                raise Stage1ExecutionError(
                    f"[{arm.name}] startup_health_timeout_after_{arm_meta['startup_elapsed_s']:.2f}s "
                    f"(budget={startup_budget:.2f}s)"
                )

            # Capture resolved server cmdline & env inside container `cid`
            rem_work_for_exec = self.remaining_for_work_s(arms_remaining_to_clean)
            if rem_work_for_exec <= 0:
                arm_meta["uncertain_server_work"] = True
                raise Stage1ExecutionError(
                    f"[{arm.name}] run_deadline_exceeded_before_proc_dump (rem_work={rem_work_for_exec:.2f}s)"
                )
            proc_dump = self._run_bounded(
                [
                    "docker",
                    "exec",
                    cid,
                    "sh",
                    "-c",
                    'for p in /proc/[0-9]*; do c=$(tr "\\0" " " < $p/cmdline 2>/dev/null); '
                    'case "$c" in */flux-server*" serve "*) echo "$c"; '
                    'tr "\\0" "\\n" < $p/environ | grep -E "^(ATLAS|FLUX)_" | sort > /dev/stderr; break;; esac; done',
                ],
                min(10.0, rem_work_for_exec),
            )
            (arm_dir / "cmdline.txt").write_text(proc_dump.stdout, encoding="utf-8")
            (arm_dir / "env.txt").write_text(proc_dump.stderr, encoding="utf-8")

            # Run 12-prompt collection bounded strictly by remaining work budget (Control 2)
            rem_work = self.remaining_for_work_s(arms_remaining_to_clean)
            if rem_work <= 1.0:
                arm_meta["uncertain_server_work"] = True
                raise Stage1ExecutionError(
                    f"[{arm.name}] run_deadline_exceeded_before_collection (rem_work={rem_work:.2f}s)"
                )
            # Reserve up to 2.0s inside `rem_work` so `fid_gate12.py`'s internal `--total-wall-cap-s`
            # triggers and writes `fid.meta.json` before the parent subprocess timeout hits `rem_work`.
            subproc_cap = min(self.collect_panel_cap_s, rem_work)
            effective_panel_cap = max(0.5, subproc_cap - 2.0) if subproc_cap > 2.5 else subproc_cap
            effective_req_timeout = min(self.request_timeout_s, effective_panel_cap)

            arm_meta["status"] = "collecting"
            write_json_atomic(arm_dir / "arm_meta.json", arm_meta)

            t_col0 = self.clock()
            col_cmd = [
                sys.executable,
                str(self.harness_path),
                "collect",
                f"http://127.0.0.1:{self.port}/v1/chat/completions",
                str(arm_dir / "fid.jsonl"),
                "--manifest",
                str(self.manifest_path),
                "--manifest-sha256",
                EXPECTED_MANIFEST_SHA256,
                "--max-tokens",
                "64",
                "--top-logprobs",
                "5",
                "--seed",
                "1234",
                "--request-timeout-s",
                f"{effective_req_timeout:.3f}",
                "--total-wall-cap-s",
                f"{effective_panel_cap:.3f}",
            ]
            col_res = self._run_bounded(col_cmd, subproc_cap)
            arm_meta["collect_elapsed_s"] = round(self.clock() - t_col0, 6)
            (arm_dir / "collect_stdout.log").write_text(col_res.stdout, encoding="utf-8")
            (arm_dir / "collect_stderr.log").write_text(col_res.stderr, encoding="utf-8")

            if floor_breached:
                arm_meta["floor_breached"] = True
                raise Stage1ExecutionError(f"[{arm.name}] {floor_breach_msg}")

            # Inspect `fid.meta.json` if written by `fid_gate12.py`
            fid_meta_path = arm_dir / "fid.meta.json"
            if fid_meta_path.is_file():
                try:
                    fmeta = json.loads(fid_meta_path.read_text(encoding="utf-8"))
                    if fmeta.get("uncertain_server_work"):
                        arm_meta["uncertain_server_work"] = True
                except Exception:  # noqa: BLE001
                    pass

            if col_res.timed_out or col_res.returncode != 0:
                if (
                    col_res.timed_out
                    or "deadline_exceeded" in col_res.stderr
                    or "timeout" in col_res.stderr.lower()
                ):
                    arm_meta["uncertain_server_work"] = True
                raise Stage1ExecutionError(
                    f"[{arm.name}] collect_failed (rc={col_res.returncode}, timed_out={col_res.timed_out}): "
                    f"{col_res.stderr.strip()}"
                )

            arm_meta["status"] = "completed"
        except BaseException as exc:
            arm_meta["status"] = "failed"
            arm_meta["error"] = str(exc)
            raise
        finally:
            stop_watch_event.set()
            if watch_thread is not None:
                watch_thread.join(timeout=1.0)
            arm_meta["min_mem_available_mib"] = min(mem_samples) if mem_samples else pre_mem_mib
            arm_meta["floor_breached"] = floor_breached

            # Strictly clamp cleanup deadline to `min(now + cleanup_cap_s, run_deadline)` — NEVER exceed `run_deadline`!
            cleanup_deadline = min(self.clock() + self.cleanup_cap_s, self.run_deadline)

            # If `docker run` was attempted and `cid` is still `None`, reconcile uncertain create before cleanup (Control 1)
            if create_attempted and cid is None:
                reconcile_info = self._reconcile_intended_container(
                    arm_dir=arm_dir,
                    container_name=container_name,
                    owner_token=owner_token,
                    cidfile_path=cidfile_path,
                    expected_image_ref=launch_image_ref,
                    stdout_cid_hint=stdout_cid_hint,
                    ceiling_deadline=cleanup_deadline,
                )
                arm_meta["reconciliation"] = reconcile_info
                if reconcile_info["verified_cid"] is not None:
                    cid = reconcile_info["verified_cid"]
                    arm_meta["container_id"] = cid
                elif reconcile_info["absence_proven"]:
                    # Explicit absence proven (all ps queries succeeded, and every candidate was either
                    # explicitly `No such object` or verified as belonging to another owner)
                    arm_meta["uncertain_container_state"] = False
                else:
                    # Preserve uncertain container/server state whenever ownership/absence is not proven
                    arm_meta["uncertain_container_state"] = True
                    arm_meta["uncertain_server_work"] = True

            if cid is not None:
                cleanup_info = self._cleanup_owned_container(cid, arm_dir, cleanup_deadline)
                arm_meta["cleanup"] = cleanup_info
                if cleanup_info["container_removed"]:
                    arm_meta["uncertain_container_state"] = False
                else:
                    arm_meta["uncertain_container_state"] = True
                    arm_meta["uncertain_server_work"] = True

            write_json_atomic(arm_dir / "arm_meta.json", arm_meta)

        if arm_meta["uncertain_container_state"]:
            raise Stage1ExecutionError(
                f"[{arm.name}] cleanup_left_uncertain_container_state for cid={cid}"
            )
        return arm_meta

    def _run_descriptive_comparisons(self) -> list[dict[str, Any]]:
        compare_dir = self.out_root / "compare"
        compare_dir.mkdir(parents=True, exist_ok=False)
        pairs = (
            ("cand_C_vs_ref_R", "ref_R", "cand_C"),
            ("ctl_D_vs_ref_R", "ref_R", "ctl_D"),
            ("cand_C_vs_ctl_D", "ctl_D", "cand_C"),
        )
        results: list[dict[str, Any]] = []
        for label, ref_arm, alt_arm in pairs:
            rem = self.remaining_total_s()
            if rem <= 0:
                raise Stage1ExecutionError(
                    f"run_deadline_exceeded_before_compare ({label}): remaining={rem:.2f}s"
                )
            json_out = compare_dir / f"{label}.descriptive.json"
            txt_out = compare_dir / f"{label}.descriptive.txt"
            # Note: NO `--enforce-gates` flag -> descriptive default (`numerical_acceptance="not_evaluated"`, `gate_passed=None`)
            cmp_cmd = [
                sys.executable,
                str(self.harness_path),
                "compare",
                str(self.out_root / ref_arm / "fid.jsonl"),
                str(self.out_root / alt_arm / "fid.jsonl"),
                "--manifest",
                str(self.manifest_path),
                "--manifest-sha256",
                EXPECTED_MANIFEST_SHA256,
                "--max-tokens",
                "64",
                "--top-logprobs",
                "5",
                "--require-token-ids",
                "--json-out",
                str(json_out),
            ]
            res = self._run_bounded(cmp_cmd, self.compare_cap_s)
            txt_out.write_text(res.stdout + (res.stderr if res.stderr else ""), encoding="utf-8")
            if res.timed_out or res.returncode != 0:
                raise Stage1ExecutionError(
                    f"compare_failed ({label}, rc={res.returncode}, timed_out={res.timed_out}): {res.stderr.strip()}"
                )
            summary_obj = json.loads(json_out.read_text(encoding="utf-8"))
            if (
                summary_obj.get("evidence_valid") is not True
                or summary_obj.get("numerical_acceptance") != "not_evaluated"
                or summary_obj.get("gate_passed") is not None
            ):
                raise Stage1ExecutionError(
                    f"compare_contract_violation ({label}): expected descriptive mode "
                    "(evidence_valid=True, numerical_acceptance='not_evaluated', gate_passed=None)"
                )
            results.append(
                {
                    "pair": label,
                    "ref_arm": ref_arm,
                    "alt_arm": alt_arm,
                    "evidence_valid": summary_obj["evidence_valid"],
                    "numerical_acceptance": summary_obj["numerical_acceptance"],
                    "gate_passed": summary_obj["gate_passed"],
                    "first_token_top1_agree": summary_obj["first_token_top1_agree"],
                    "continuations_identical": summary_obj["continuations_identical"],
                    "pos0_shared_top5_median_max_abs_dlp": summary_obj[
                        "pos0_shared_top5_median_max_abs_dlp"
                    ],
                    "pos0_shared_top5_max_abs_dlp": summary_obj[
                        "pos0_shared_top5_max_abs_dlp"
                    ],
                    "json_path": str(json_out),
                    "json_sha256": sha256_file(json_out),
                }
            )
        return results

    def run(self) -> dict[str, Any]:
        # 1. Exclusive output directory creation (`exist_ok=False`) before anything else
        if self.out_root.exists():
            raise Stage1AdmissionError(
                f"output_directory_already_exists_refusing_overwrite: {self.out_root}"
            )

        self._acquire_nonblocking_lock()
        arm_summaries: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        status = "in_progress"
        error_msg: str | None = None
        pins: dict[str, str] = {}

        prev_sigint = None
        prev_sigterm = None
        can_install_signals = (
            self.install_signal_handlers
            and threading.current_thread() is threading.main_thread()
        )

        def _handle_termination_signal(signum: int, _frame: Any) -> None:
            sig_name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)
            raise Stage1ExecutionError(f"terminated_by_signal_{sig_name} (signum={signum})")

        if can_install_signals:
            prev_sigint = signal.getsignal(signal.SIGINT)
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, _handle_termination_signal)
            signal.signal(signal.SIGTERM, _handle_termination_signal)

        try:
            pins = self._verify_all_pins()
            planned_names = {f"fidgate12-{self.run_id}-{arm.name}" for arm in STAGE1_ARMS}
            initial_mem_mib = self._check_live_admission(planned_names)

            # Create fresh output root only after lock + pin + admission checks succeed
            self.out_root.mkdir(parents=True, exist_ok=False)

            write_json_atomic(
                self.out_root / "run_summary.json",
                {
                    "schema_version": "fid_gate12_stage1_run.v1",
                    "run_id": self.run_id,
                    "status": "in_progress",
                    "out_root": str(self.out_root),
                    "initial_mem_available_mib": initial_mem_mib,
                    "pins": pins,
                    "total_run_cap_s": self.total_run_cap_s,
                    "owned_container_ids": [],
                    "arms": [],
                    "comparisons": [],
                },
            )

            for idx, arm in enumerate(STAGE1_ARMS):
                arm_meta = self._run_single_arm(
                    arm=arm,
                    arm_idx=idx,
                    total_arms=len(STAGE1_ARMS),
                    planned_names=planned_names,
                )
                arm_summaries.append(arm_meta)

            comparisons = self._run_descriptive_comparisons()
            status = "completed"
        except BaseException as exc:
            status = "failed"
            error_msg = str(exc)
            raise
        finally:
            if can_install_signals:
                if prev_sigint is not None:
                    signal.signal(signal.SIGINT, prev_sigint)
                if prev_sigterm is not None:
                    signal.signal(signal.SIGTERM, prev_sigterm)

            if self.out_root.is_dir():
                any_uncertain_container = any(
                    a.get("uncertain_container_state", False) for a in arm_summaries
                )
                any_uncertain_server = any(
                    a.get("uncertain_server_work", False) for a in arm_summaries
                )
                for arm in STAGE1_ARMS:
                    am_path = self.out_root / arm.name / "arm_meta.json"
                    if am_path.is_file() and not any(a.get("arm") == arm.name for a in arm_summaries):
                        try:
                            am_obj = json.loads(am_path.read_text(encoding="utf-8"))
                            arm_summaries.append(am_obj)
                            if am_obj.get("uncertain_container_state"):
                                any_uncertain_container = True
                            if am_obj.get("uncertain_server_work"):
                                any_uncertain_server = True
                        except Exception:  # noqa: BLE001
                            pass

                summary = {
                    "schema_version": "fid_gate12_stage1_run.v1",
                    "run_id": self.run_id,
                    "status": status,
                    "error": error_msg,
                    "out_root": str(self.out_root),
                    "total_elapsed_s": round(self.elapsed_s(), 6),
                    "total_run_cap_s": self.total_run_cap_s,
                    "owned_container_ids": sorted(self.owned_container_ids),
                    "uncertain_container_state": any_uncertain_container,
                    "uncertain_server_work": any_uncertain_server,
                    "server_cancellation_guaranteed": False,
                    "pins": pins,
                    "arms": arm_summaries,
                    "comparisons": comparisons,
                    "scientific_qualifications": {
                        "workload": "12 pinned prompts x 3 sequential precision arms = 36 requests total (max_tokens=64, T=0.0, top_logprobs=5, return_token_ids=True, zero retries, Stage 2 LCB cases unbundled)",
                        "token_id_vs_top_logprobs": "choice.token_ids provides exact sampled continuation token IDs; top_logprobs exposes post-transform host-path decoded strings (bytes=None)",
                        "reference_arm_and_prefill_confound": "ref_R (ATLAS_QWEN4_DENSE_NATIVE=1) is a same-engine dense-precision reference (with QAD NVFP4 routed experts and PREFILL_NATIVE=1 enabled), not a truth oracle or decode-only BF16 isolator",
                        "descriptive_comparison_only": "Comparisons run with enforce_gates=False (evidence_valid=True, numerical_acceptance='not_evaluated', gate_passed=None); no numerical-accuracy-pass or NVIDIA parity claim is made.",
                    },
                }
                write_json_atomic(self.out_root / "run_summary.json", summary)
            self._release_lock()

        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pinned Stage 1 ONLY Orchestrator for 12-prompt fidelity diagnostic (36 requests total)"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Fresh output directory (must not exist; default: .../runs/fid-gate12-stage1-<UTC>)",
    )
    parser.add_argument("--run-id", default=None, help="Unique run ID token (default: UTC timestamp)")
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--hold-path", type=Path, default=DEFAULT_HOLD_PATH)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--expected-image-id", default=EXPECTED_IMAGE_ID)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--total-run-cap-s", type=float, default=DEFAULT_TOTAL_RUN_CAP_S)
    parser.add_argument("--startup-cap-s", type=float, default=DEFAULT_STARTUP_CAP_S)
    parser.add_argument("--collect-panel-cap-s", type=float, default=DEFAULT_COLLECT_PANEL_CAP_S)
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--cleanup-cap-s", type=float, default=DEFAULT_CLEANUP_CAP_S)
    parser.add_argument("--compare-cap-s", type=float, default=DEFAULT_COMPARE_CAP_S)
    parser.add_argument("--min-mem-available-mib", type=int, default=DEFAULT_MIN_MEM_AVAILABLE_MIB)
    parser.add_argument("--floor-mib", type=int, default=DEFAULT_FLOOR_MIB)

    args = parser.parse_args(argv)
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_root = args.out_root or Path(
        os.environ.get("FLUX_OUT_ROOT", str(REPO_ROOT / "results" / f"fid-gate12-stage1-{run_id}"))
    )

    orch = Stage1Orchestrator(
        out_root=out_root,
        run_id=run_id,
        lock_path=args.lock_path,
        hold_path=args.hold_path,
        binary_path=args.binary,
        model_dir=args.model_dir,
        harness_path=args.harness,
        manifest_path=args.manifest,
        image=args.image,
        expected_image_id=args.expected_image_id,
        port=args.port,
        total_run_cap_s=args.total_run_cap_s,
        startup_cap_s=args.startup_cap_s,
        collect_panel_cap_s=args.collect_panel_cap_s,
        request_timeout_s=args.request_timeout_s,
        cleanup_cap_s=args.cleanup_cap_s,
        compare_cap_s=args.compare_cap_s,
        min_mem_available_mib=args.min_mem_available_mib,
        floor_mib=args.floor_mib,
    )
    try:
        summary = orch.run()
        print(
            f"STAGE1_OK run_id={summary['run_id']} elapsed_s={summary['total_elapsed_s']:.2f} "
            f"out_root={summary['out_root']}",
            flush=True,
        )
        return 0
    except Stage1AdmissionError as exc:
        print(f"STAGE1_ADMISSION_REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2
    except Stage1ExecutionError as exc:
        print(f"STAGE1_EXECUTION_FAILED: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
