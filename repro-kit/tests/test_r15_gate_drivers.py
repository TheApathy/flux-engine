#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline tests for the S1 r15-gates/raw-log-safety gate driver.

The harness runs the driver the way the tracked suite runs it: one fresh
process with inherited raw-log and receipt descriptors, identity argv
tokens, and a validators directory inside a synthetic runtime tree.
Census rejection tests independently require every planted forbidden marker
to fail without a PASS receipt. Clean receipts are replayed through the real
wrapper recomputation in ``r15_exact_image_release.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
DRIVER = KIT / "r15-gates" / "raw-log-safety"
SUITE_PATH = KIT / "r15-container-suite.py"
WRAPPER_PATH = KIT / "r15_exact_image_release.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _module("r15_container_suite", SUITE_PATH)
Q = _module("r15_exact_image_release", WRAPPER_PATH)


CONTAINER_ID = "c" * 64
IMAGE_ID = "sha256:" + "a" * 64
IMAGE_DIGEST = "sha256:" + "b" * 64
RUN_ID = "8" * 40


def _run_driver(
    runtime: Path,
    raw_log_bytes: bytes,
    *,
    identities: dict[str, str] | None = None,
    validators_dir: Path | None = None,
    extra_flags: list[str] | None = None,
):
    """Run the driver exactly as the suite would launch it.

    Returns (return_code, receipt_bytes, stderr_text).  Every descriptor the
    driver receives is the same kind the suite passes: a regular read-only
    raw log and a writable receipt sink bounded by the suite later.
    """

    effective = {
        "container-id": CONTAINER_ID,
        "image-id": IMAGE_ID,
        "image-digest": IMAGE_DIGEST,
        "base-url": "http://127.0.0.1:0",
        "run-id": RUN_ID,
    }
    for name, value in (identities or {}).items():
        effective[name] = value
    if validators_dir is not None:
        validators_dir = Path(validators_dir)
    else:
        validators_dir = runtime / "validators"
    argv = [sys.executable, str(DRIVER)]
    for flag in ("--container-id", "--image-id", "--image-digest"):
        argv.append(flag)
        argv.append(effective[flag[2:]])
    argv.extend(("--base-url", effective["base-url"]))
    argv.extend(("--run-id", effective["run-id"]))
    argv.extend(("--validators-dir", str(validators_dir)))
    argv.extend(extra_flags or ())
    raw_log = runtime / "complete-raw.log"
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_log, "wb") as handle:
        handle.write(raw_log_bytes)
    raw_log_fd = os.open(raw_log, os.O_RDONLY)
    receipt_path = runtime / "driver-receipt.json"
    if receipt_path.exists():
        receipt_path.unlink()
    receipt_fd = os.open(receipt_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        argv.extend(("--receipt-fd", str(receipt_fd), "--raw-log-fd", str(raw_log_fd)))
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            pass_fds=(receipt_fd, raw_log_fd),
        )
        _stdout, stderr_text = process.communicate(timeout=60)
        code = process.returncode
        if stderr_text is None:
            error = ""
        else:
            error = stderr_text.decode("utf-8", "replace")
        receipt_bytes = receipt_path.read_bytes() if receipt_path.exists() else None
        return code, receipt_bytes, error
    finally:
        os.close(receipt_fd)
        os.close(raw_log_fd)


class RawLogSafetyDriverTests(unittest.TestCase):
    def test_clean_log_passes_and_writes_readable_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            code, receipt_bytes, error = _run_driver(
                runtime, b"complete fixture server log without fatal markers\n"
            )
            self.assertEqual(error, "")
            self.assertEqual(code, 0, error)
            self.assertIsNotNone(receipt_bytes)
            receipt = json.loads(receipt_bytes)
            self.assertEqual(
                set(receipt),
                {
                    "schema",
                    "gate",
                    "status",
                    "release_qualified",
                    "execution_scope",
                    "image_id",
                    "image_digest",
                    "container_id",
                    "run_id",
                    "evidence",
                },
            )
            self.assertEqual(receipt["schema"], Q.GATE_SCHEMA)
            self.assertEqual(receipt["gate"], "raw-log-safety")
            self.assertEqual(receipt["status"], "PASS")
            self.assertTrue(receipt["release_qualified"])
            self.assertEqual(receipt["execution_scope"], "immutable-container")
            self.assertEqual(receipt["container_id"], CONTAINER_ID)
            self.assertEqual(receipt["image_id"], IMAGE_ID)
            self.assertEqual(receipt["image_digest"], IMAGE_DIGEST)
            self.assertEqual(receipt["run_id"], RUN_ID)
            evidence = receipt["evidence"]
            self.assertEqual(
                set(evidence),
                {
                    "scan_scope",
                    "patterns",
                    "match_count",
                    "raw_log_bytes",
                    "raw_log_sha256",
                    "all_processes_included",
                    "validator_artifact",
                    "validator_receipt_sha256",
                },
            )
            self.assertEqual(evidence["scan_scope"], "complete-uncapped")
            self.assertEqual(evidence["patterns"], list(Q.RAW_LOG_PATTERNS))
            self.assertEqual(evidence["match_count"], 0)
            self.assertNotIn("per_pattern_match_counts", evidence)
            self.assertEqual(
                evidence["raw_log_bytes"],
                len(b"complete fixture server log without fatal markers\n"),
            )
            self.assertEqual(
                evidence["raw_log_sha256"],
                hashlib.sha256(
                    b"complete fixture server log without fatal markers\n"
                ).hexdigest(),
            )
            self.assertTrue(evidence["all_processes_included"])
            self.assertEqual(
                evidence["validator_artifact"], "validators/raw-log-safety.json"
            )

    def test_validator_artifact_is_hard_and_wrapper_recomputes_it(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            code, receipt_bytes, _error = _run_driver(runtime, b"clean seeds only\n")
            self.assertEqual(code, 0)
            receipt = json.loads(receipt_bytes)
            evidence = receipt["evidence"]
            validator_path = runtime / "validators" / "raw-log-safety.json"
            mode = stat.S_IMODE(validator_path.stat().st_mode)
            self.assertEqual(oct(mode), "0o400")
            self.assertEqual(validator_path.stat().st_nlink, 1)
            validator_raw = validator_path.read_bytes()
            document = json.loads(validator_raw)
            self.assertEqual(
                set(document),
                {
                    "schema",
                    "gate",
                    "status",
                    "release_qualified",
                    "container_id",
                    "run_id",
                    "raw",
                },
            )
            self.assertEqual(document["schema"], Q.GATE_VALIDATOR_SCHEMA)
            self.assertEqual(document["gate"], "raw-log-safety")
            self.assertEqual(document["container_id"], CONTAINER_ID)
            self.assertEqual(document["run_id"], RUN_ID)
            raw = document["raw"]
            self.assertEqual(
                set(raw),
                {
                    "scan_scope",
                    "per_pattern_match_counts",
                    "raw_log_bytes",
                    "all_processes_included",
                },
            )
            self.assertEqual(
                raw["per_pattern_match_counts"],
                {label: 0 for label in Q.RAW_LOG_PATTERNS},
            )
            self.assertEqual(
                evidence["validator_receipt_sha256"],
                hashlib.sha256(validator_raw).hexdigest(),
            )
            manifest = {"_runtime_dir": str(runtime), "launch": {"run_id": RUN_ID}}
            replayed = Q._validate_raw_log_evidence(evidence, manifest, CONTAINER_ID)
            self.assertEqual(replayed, evidence)

    def test_forbidden_marker_fails_closed_without_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            code, receipt_bytes, error = _run_driver(
                runtime, b"journal reports XID 79 and cuda Fatal\n"
            )
            self.assertEqual(code, 2, error)
            self.assertFalse(receipt_bytes)
            self.assertIn("Xid", error)
            validator_path = runtime / "validators" / "raw-log-safety.json"
            self.assertFalse(validator_path.exists())

    def test_lowercase_marker_still_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            raw_log = runtime / "complete-raw.log"
            for payload in (
                b"xid 79\n",
                b"the supply line: nvrm power limit\n",
                b"CUDA out of memory expected\n",
                b"illegal memory access expected\n",
            ):
                self.assertFalse(raw_log.exists())
                code, receipt_bytes, error = _run_driver(runtime, payload)
                self.assertEqual(code, 2, error)
                self.assertFalse(receipt_bytes)
                validator_path = runtime / "validators" / "raw-log-safety.json"
                self.assertFalse(validator_path.exists())
                # The suite census must infer the same result from the bytes.
                descriptor = os.open(raw_log, os.O_RDONLY)
                try:
                    suite_scan = S.scan_raw_log(descriptor)
                finally:
                    os.close(descriptor)
                self.assertEqual(
                    suite_scan["match_count"], 1, suite_scan["per_pattern_match_counts"]
                )
                raw_log.unlink()

    def test_straddling_forbidden_marker_fails_without_receipt(self):
        needle = b"cuda--fatal"
        boundary = 4 * 1024 * 1024
        start = boundary - len(needle) + 3
        payload = bytearray(b"\x00" * (boundary * 2))
        payload[start : start + len(needle)] = needle
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            code, receipt_bytes, error = _run_driver(runtime, bytes(payload))
            self.assertEqual(code, 2, error)
            self.assertFalse(receipt_bytes)
            self.assertIn("CUDA fatal", error)
            self.assertFalse((runtime / "validators/raw-log-safety.json").exists())

    def test_every_forbidden_label_straddles_the_old_read_boundary(self):
        boundary = 4 * 1024 * 1024
        for label, marker in (
            ("CUDA fatal", b"CUDA fatal"),
            ("CUDA out of memory", b"cuda-_out:of memory"),
            ("illegal memory access", b"illegal-memory_access"),
            ("NVRM", b"NVRM"),
            ("Xid", b"Xid"),
        ):
            for split in range(1, len(marker)):
                with self.subTest(
                    label=label, split=split
                ), tempfile.TemporaryDirectory() as temp:
                    runtime = Path(temp) / "runtime"
                    payload = b"\0" * (boundary - split) + marker + b"\n"
                    code, receipt, error = _run_driver(runtime, payload)
                    self.assertEqual(code, 2, error)
                    self.assertFalse(receipt)
                    self.assertIn(label, error)
                    self.assertFalse(
                        (runtime / "validators/raw-log-safety.json").exists()
                    )

    def test_word_boundaries_do_not_turn_clean_suffixes_into_markers(self):
        boundary = 4 * 1024 * 1024
        for marker, word in (
            (b"Xid", b"XidSuffix"),
            (b"NVRM", b"NVRMSuffix"),
            (b"Xid", b"prefixXid"),
            (b"NVRM", b"prefixNVRM"),
        ):
            for split in (0, 1, len(marker)):
                with self.subTest(
                    word=word, split=split
                ), tempfile.TemporaryDirectory() as temp:
                    runtime = Path(temp) / "runtime"
                    payload = b"\0" * (boundary - len(marker) + split) + word + b"\n"
                    code, receipt, error = _run_driver(runtime, payload)
                    self.assertEqual(code, 0, error)
                    self.assertEqual(json.loads(receipt)["evidence"]["match_count"], 0)

    def test_unbounded_separators_cannot_hide_forbidden_labels(self):
        for label, first, last in (
            ("CUDA fatal", b"cuda", b"fatal"),
            ("CUDA out of memory", b"cuda", b"out:of-memory"),
            ("illegal memory access", b"illegal", b"memory_access"),
        ):
            for length in (4096, 4 * 1024 * 1024 + 64):
                with self.subTest(
                    label=label, length=length
                ), tempfile.TemporaryDirectory() as temp:
                    runtime = Path(temp) / "runtime"
                    payload = b"\0" * 128 + first + b"-" * length + last + b"\n"
                    code, receipt, error = _run_driver(runtime, payload)
                    self.assertEqual(code, 2, error)
                    self.assertFalse(receipt)
                    self.assertIn(label, error)
                    self.assertFalse(
                        (runtime / "validators/raw-log-safety.json").exists()
                    )

    def test_match_census_limit_fails_without_a_pass_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            code, receipt, error = _run_driver(runtime, b"Xid\n" * 100_001)
            self.assertEqual(code, 1, error)
            self.assertFalse(receipt)
            self.assertIn("census exceeded its bound", error)
            self.assertFalse((runtime / "validators/raw-log-safety.json").exists())

    def test_planted_validator_symlink_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            validators = runtime / "validators"
            validators.mkdir(parents=True)
            plant = runtime / "plant.json"
            plant.write_bytes(b"planted\n")
            (validators / "raw-log-safety.json").symlink_to(plant)
            code, receipt_bytes, _error = _run_driver(runtime, b"clean\n")
            self.assertEqual(code, 1)
            self.assertFalse(receipt_bytes)
            self.assertEqual(plant.read_bytes(), b"planted\n")
            self.assertFalse(
                (validators / "raw-log-safety.json").is_file()
                and not (validators / "raw-log-safety.json").is_symlink()
            )
            self.assertTrue((validators / "raw-log-safety.json").is_symlink())

    def test_missing_validators_directory_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            code, receipt_bytes, _error = _run_driver(
                runtime, b"clean\n", validators_dir=runtime / "validators" / "deeper"
            )
            self.assertEqual(code, 1)
            self.assertFalse(receipt_bytes)

    def test_identity_tokens_are_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            code, _receipt, _error = _run_driver(
                runtime, b"clean\n", identities={"run-id": "unsafe/run"}
            )
            self.assertEqual(code, 1)

    def test_run_ids_match_suite_and_outer_safe_identifier_contract(self):
        for run_id in (
            "run-r16-container-qualification",
            "A",
            "9",
            "Run_16.alpha-1",
            "a" * 96,
        ):
            with self.subTest(run_id=run_id), tempfile.TemporaryDirectory() as temp:
                runtime = Path(temp) / "runtime"
                self.assertIsNotNone(S.SAFE_RUN_ID.fullmatch(run_id))
                self.assertIsNotNone(Q.SAFE_RUN_ID.fullmatch(run_id))
                code, receipt, error = _run_driver(
                    runtime, b"clean\n", identities={"run-id": run_id}
                )
                self.assertEqual(code, 0, error)
                self.assertEqual(json.loads(receipt)["run_id"], run_id)

    def test_run_ids_reject_unsafe_characters_and_invalid_boundaries(self):
        for run_id in (
            "",
            "a" * 97,
            "-run",
            "_run",
            ".run",
            "run/name",
            "run name",
            "run\nname",
            "run:name",
            "run\tname",
            "run\N{LATIN SMALL LETTER E WITH ACUTE}",
        ):
            with self.subTest(run_id=run_id), tempfile.TemporaryDirectory() as temp:
                runtime = Path(temp) / "runtime"
                self.assertIsNone(S.SAFE_RUN_ID.fullmatch(run_id))
                self.assertIsNone(Q.SAFE_RUN_ID.fullmatch(run_id))
                code, receipt, _error = _run_driver(
                    runtime, b"clean\n", identities={"run-id": run_id}
                )
                self.assertEqual(code, 1)
                self.assertFalse(receipt)
                self.assertFalse((runtime / "validators/raw-log-safety.json").exists())

    def test_duplicate_arguments_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            code, _receipt, _error = _run_driver(
                runtime,
                b"clean\n",
                extra_flags=["--container-id", CONTAINER_ID],
            )
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
