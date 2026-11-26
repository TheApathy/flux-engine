# SPDX-License-Identifier: AGPL-3.0-only

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SUITE_PATH = Path(__file__).resolve().parents[1] / "r15-container-suite.py"
SPEC = importlib.util.spec_from_file_location("r15_container_suite", SUITE_PATH)
assert SPEC and SPEC.loader
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)

IMAGE_ID = "sha256:" + "a" * 64
IMAGE_DIGEST = "sha256:" + "b" * 64
CONTAINER_ID = "c" * 64
RUN_ID = "r15-suite-test"

DUMMY_DRIVER = r"""#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess

PATTERNS = (
    "CUDA fatal",
    "CUDA out of memory",
    "illegal memory access",
    "NVRM",
    "Xid",
)

parser = argparse.ArgumentParser()
parser.add_argument("--gate", required=True)
parser.add_argument("--container-id", required=True)
parser.add_argument("--image-id", required=True)
parser.add_argument("--image-digest", required=True)
parser.add_argument("--base-url", required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--receipt-fd", required=True, type=int)
parser.add_argument("--raw-log-fd", required=True, type=int)
parser.add_argument("--process-proof-fd", type=int)
parser.add_argument("--emit-fatal", action="store_true")
parser.add_argument("--emit-lowercase", action="store_true")
parser.add_argument("--emit-raw-output", action="store_true")
parser.add_argument("--leave-descendant", action="store_true")
args = parser.parse_args()

if args.emit_fatal:
    print("NVRM synthetic hostile marker")
elif args.emit_lowercase:
    print("xid 73 reaches the journal: cuda fatal")
elif args.gate != "raw-log-safety":
    print(f"gate {args.gate} complete")
elif args.emit_raw_output:
    print("raw safety must remain silent")

evidence = {"fixture": args.gate}
if args.gate == "raw-log-safety":
    info = os.fstat(args.raw_log_fd)
    raw = os.pread(args.raw_log_fd, info.st_size, 0)
    counts = {}
    for pattern in PATTERNS:
        counts[pattern] = raw.lower().count(pattern.lower().encode())
    evidence = {
        "scan_scope": "complete-uncapped",
        "patterns": list(PATTERNS),
        "match_count": sum(counts.values()),
        "raw_log_bytes": len(raw),
        "raw_log_sha256": hashlib.sha256(raw).hexdigest(),
        "all_processes_included": True,
        "validator_artifact": "raw-validator.json",
        "validator_receipt_sha256": hashlib.sha256(b"raw-validator").hexdigest(),
    }

receipt = {
    "schema": "flux-r15-container-gate-receipt/1",
    "gate": args.gate,
    "status": "PASS",
    "release_qualified": True,
    "execution_scope": "immutable-container",
    "image_id": args.image_id,
    "image_digest": args.image_digest,
    "container_id": args.container_id,
    "run_id": args.run_id,
    "evidence": evidence,
}
payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
os.write(args.receipt_fd, payload)
os.fsync(args.receipt_fd)
if args.leave_descendant:
    subprocess.Popen(
        ["/bin/sleep", "30"],
        pass_fds=(args.receipt_fd,),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
"""


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_bytes(name, payload, mode=0o400):
    descriptor = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    os.write(descriptor, payload)
    os.fchmod(descriptor, mode)
    seals = (
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL
    )
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
    return descriptor


def make_plan(
    root,
    *,
    fatal_gate=None,
    noisy_raw=False,
    descendant_gate=None,
    lowercase_gate=None,
):
    root = Path(root)
    source = root / "source"
    runtime = root / "runtime"
    source.mkdir()
    runtime.mkdir()
    for directory in ("gates", "stdout", "stderr"):
        (runtime / directory).mkdir()
    driver = source / "gate-driver"
    driver.write_text(DUMMY_DRIVER, encoding="utf-8")
    driver.chmod(0o755)
    jobs = {}
    for index, gate in enumerate(S.REQUIRED_GATES):
        argv = [
            str(driver),
            "--gate",
            gate,
            "--container-id",
            "@CONTAINER_ID@",
            "--image-id",
            "@IMAGE_ID@",
            "--image-digest",
            "@IMAGE_DIGEST@",
            "--base-url",
            "@BASE_URL@",
            "--run-id",
            "@RUN_ID@",
            "--receipt-fd",
            "@GATE_RECEIPT_FD@",
            "--raw-log-fd",
            "@RAW_LOG_FD@",
        ]
        if gate == "startup-readiness":
            argv.extend(("--process-proof-fd", S.PROCESS_PROOF_TOKEN))
        if gate == fatal_gate:
            argv.append("--emit-fatal")
        if gate == lowercase_gate:
            argv.append("--emit-lowercase")
        if gate == "raw-log-safety" and noisy_raw:
            argv.append("--emit-raw-output")
        if gate == descendant_gate:
            argv.append("--leave-descendant")
        jobs[gate] = {
            "program_path": str(driver),
            "program_sha256": sha256_file(driver),
            "argv": argv,
            "timeout_seconds": 30,
            "receipt_relpath": f"gates/{index:02d}-{gate}.json",
            "stdout_relpath": f"stdout/{index:02d}-{gate}.log",
            "stderr_relpath": f"stderr/{index:02d}-{gate}.log",
        }
    example = S.strict_json_loads(
        (SUITE_PATH.parent / "examples" / "r15-container-suite-plan.example.json").read_bytes(),
        "checked-in plan example",
    )
    return {
        # The suite loads plan /4 only since 911ec7cef; its native-context
        # contract is the reviewed one in the checked-in example.
        "schema": S.PLAN_SCHEMA_V4,
        "native_context_contract": example["native_context_contract"],
        "release_version": S.RELEASE_VERSION,
        "runtime_dir": str(runtime),
        "source_root_path": str(source),
        "source_revision": "d" * 40,
        "source_tree": "e" * 40,
        "source_bundle_manifest_sha256": "f" * 64,
        "in_container_binary_sha256": "1" * 64,
        "in_container_binary_path": "/opt/flux/bin/flux-server-real",
        "in_container_recipe_path": "/opt/flux/repro-kit/recipes/release.env",
        "recipe_sha256": "2" * 64,
        "checkpoint_id": "example/checkpoint",
        "checkpoint_revision": "3" * 40,
        "checkpoint_manifest_sha256": "4" * 64,
        # Plan /4 pairs with profile contract /2; the reviewed one is in the example.
        "performance_profile_contract": example["performance_profile_contract"],
        "gate_jobs": jobs,
        "raw_log_relpath": "complete-raw.log",
    }


def load_plan(plan):
    descriptor = seal_bytes("test-plan", S.canonical_bytes(plan))
    try:
        return S.load_plan(descriptor)
    finally:
        os.close(descriptor)


def fixture_proof(mode, pid, plan):
    return {
        "schema": "flux-r15-live-process-proof/1",
        "status": "PASS",
        "mode": mode,
        "container_id": CONTAINER_ID,
        "run_id": RUN_ID,
        "host_pid": pid,
        "start_time_ticks": pid,
        "executable_sha256": plan["in_container_binary_sha256"],
        "recipe_sha256": plan["recipe_sha256"],
        "recipe_environment_verified": True,
        "mount_namespace_match": True,
        "cgroup_match": True,
        "paths_read_only": True,
    }


_FIXTURE_SUITE_SUBPROCESS = r"""
import importlib.machinery, importlib.util, json, sys
from unittest import mock

payload = json.loads(sys.stdin.read())
suite_path = payload["suite_path"]
plan = payload["plan"]
identities = payload["identities"]

loader = importlib.machinery.SourceFileLoader("r15_container_suite_sub", suite_path)
spec = importlib.util.spec_from_loader(loader.name, loader)
S = importlib.util.module_from_spec(spec)
loader.exec_module(S)

def fixture_proof(mode, pid):
    return {
        "schema": "flux-r15-live-process-proof/1",
        "status": "PASS",
        "mode": mode,
        "container_id": identities["container_id"],
        "run_id": identities["run_id"],
        "host_pid": pid,
        "start_time_ticks": pid,
        "executable_sha256": plan["in_container_binary_sha256"],
        "recipe_sha256": plan["recipe_sha256"],
        "recipe_environment_verified": True,
        "mount_namespace_match": True,
        "cgroup_match": True,
        "paths_read_only": True,
    }

real_run = S.run_gate_program

def run(argv, **kwargs):
    startup = kwargs.pop("process_proof") is not None
    argv = [
        item
        for item in argv
        if item not in ("--process-proof-fd", S.PROCESS_PROOF_TOKEN)
    ]
    real_run(argv, **kwargs)
    return (
        [fixture_proof(mode, index + 1) for index, mode in enumerate(("none", "native-mtp-k5"))]
        if startup
        else []
    )

try:
    with mock.patch.object(S, "run_gate_program", side_effect=run):
        result = S.execute_suite(plan, identities)
    sys.stdout.write(json.dumps({"ok": True, "result": result}))
except Exception as exc:
    sys.stdout.write(json.dumps({"ok": False, "error_type": type(exc).__name__, "error_msg": str(exc)}))
"""


def execute_fixture_suite(plan, identities):
    # Run in a fresh single-threaded subprocess so `r15-container-suite.py`'s
    # real `/proc/self/task` single-thread containment invariant holds even when
    # `pytest` autoloads multi-threaded plugins in the parent process.
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _FIXTURE_SUITE_SUBPROCESS],
        input=json.dumps(
            {
                "suite_path": str(SUITE_PATH),
                "plan": plan,
                "identities": identities,
            }
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"fixture suite subprocess exited {completed.returncode}: {completed.stderr}"
        )
    outcome = json.loads(completed.stdout)
    if outcome["ok"]:
        return outcome["result"]
    error_cls = {
        "ValueError": ValueError,
        "RuntimeError": RuntimeError,
        "OSError": OSError,
    }.get(outcome["error_type"], RuntimeError)
    raise error_cls(outcome["error_msg"])


class R15ContainerSuiteTests(unittest.TestCase):
    def test_plan_profile_contract_is_required_strict_and_source_named(self):
        with tempfile.TemporaryDirectory() as temp:
            original = make_plan(temp)
            for change in (
                lambda plan: plan.update(schema="flux-r15-container-suite-plan/1"),
                lambda plan: plan.pop("performance_profile_contract"),
                lambda plan: plan["performance_profile_contract"].update(
                    schema="legacy"
                ),
                lambda plan: plan["performance_profile_contract"]["profiles"][
                    "interactive"
                ].update(recipe_relpath="unbound.env"),
                lambda plan: plan["performance_profile_contract"]["profiles"][
                    "interactive"
                ].update(recipe_sha256="0" * 64),
                lambda plan: plan["performance_profile_contract"]["profiles"][
                    "high-prefill"
                ].update(
                    argv_overlay=[
                        "--max-seq-len",
                        "262144",
                        "--max-prefill-tokens",
                        "4096",
                    ]
                ),
            ):
                with self.subTest(change=change):
                    candidate = S.strict_json_loads(
                        S.canonical_bytes(original), "synthetic plan"
                    )
                    change(candidate)
                    with self.assertRaises(ValueError):
                        load_plan(candidate)

    def test_checked_in_plan_example_is_strict_json_with_every_gate(self):
        example = (
            SUITE_PATH.parent / "examples" / "r15-container-suite-plan.example.json"
        )
        value = S.strict_json_loads(example.read_bytes(), "checked-in plan example")
        self.assertEqual(value["schema"], S.PLAN_SCHEMA_V4)
        self.assertEqual(set(value["gate_jobs"]), set(S.REQUIRED_GATES))

    def test_sealed_standalone_cannot_qualify_without_live_process_proofs(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = make_plan(temp)
            plan_fd = seal_bytes("suite-plan", S.canonical_bytes(plan))
            suite_fd = seal_bytes("suite-program", SUITE_PATH.read_bytes(), 0o500)
            receipt_path = Path(temp) / "suite-receipt.json"
            receipt_fd = os.open(
                receipt_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o400
            )
            try:
                completed = subprocess.run(
                    [
                        str(SUITE_PATH),
                        "--container-id",
                        CONTAINER_ID,
                        "--image-id",
                        IMAGE_ID,
                        "--image-digest",
                        IMAGE_DIGEST,
                        "--base-url",
                        "http://127.0.0.1:18898",
                        "--run-id",
                        RUN_ID,
                        "--plan-fd",
                        str(plan_fd),
                        "--receipt-fd",
                        str(receipt_fd),
                    ],
                    executable=f"/proc/self/fd/{suite_fd}",
                    pass_fds=(suite_fd, plan_fd, receipt_fd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=30,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("omitted live process proofs", completed.stderr.decode())
                raw = os.pread(receipt_fd, os.fstat(receipt_fd).st_size, 0)
            finally:
                os.close(receipt_fd)
                os.close(suite_fd)
                os.close(plan_fd)
            self.assertEqual(raw, b"")

    def test_plan_rejects_placeholder_artifact_and_source_bypasses(self):
        with tempfile.TemporaryDirectory() as temp:
            source = make_plan(temp)
            candidates = []

            missing_placeholder = copy.deepcopy(source)
            missing_placeholder["gate_jobs"][S.REQUIRED_GATES[0]]["argv"].remove(
                "@RUN_ID@"
            )
            candidates.append(missing_placeholder)

            duplicate_artifact = copy.deepcopy(source)
            duplicate_artifact["gate_jobs"][S.REQUIRED_GATES[1]]["receipt_relpath"] = (
                duplicate_artifact["gate_jobs"][S.REQUIRED_GATES[0]]["receipt_relpath"]
            )
            candidates.append(duplicate_artifact)

            traversal = copy.deepcopy(source)
            traversal["gate_jobs"][S.REQUIRED_GATES[0]]["receipt_relpath"] = (
                "../escape.json"
            )
            candidates.append(traversal)

            outside = Path(temp) / "outside-driver"
            outside.write_text(DUMMY_DRIVER, encoding="utf-8")
            outside_program = copy.deepcopy(source)
            outside_program["gate_jobs"][S.REQUIRED_GATES[0]]["program_path"] = str(
                outside
            )
            outside_program["gate_jobs"][S.REQUIRED_GATES[0]]["argv"][0] = str(outside)
            candidates.append(outside_program)

            private_model_path = copy.deepcopy(source)
            private_model_path["checkpoint_id"] = "home/private/checkpoint"
            candidates.append(private_model_path)

            parent_component = copy.deepcopy(source)
            parent_component["source_root_path"] = str(
                Path(temp) / "source" / ".." / "source"
            )
            candidates.append(parent_component)

            for index, candidate in enumerate(candidates):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_plan(candidate)

    def test_suite_rejects_fatal_log_and_noisy_final_scanner(self):
        for mutation in ("fatal", "noisy"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                source = make_plan(
                    temp,
                    fatal_gate=("startup-readiness" if mutation == "fatal" else None),
                    noisy_raw=(mutation == "noisy"),
                )
                plan = load_plan(source)
                identities = {
                    "container_id": CONTAINER_ID,
                    "image_id": IMAGE_ID,
                    "image_digest": IMAGE_DIGEST,
                    "base_url": "http://127.0.0.1:18898",
                    "run_id": RUN_ID,
                }
                with self.assertRaises(ValueError):
                    execute_fixture_suite(plan, identities)

    def test_suite_census_matches_marker_case_insensitively(self):
        with tempfile.TemporaryDirectory() as temp:
            # Lowercase spellings ("xid", "cuda fatal") of the fatal markers
            # must never pass the suite-level census.
            source = make_plan(temp, lowercase_gate="tool-api-streaming-blocking")
            plan = load_plan(source)
            identities = {
                "container_id": CONTAINER_ID,
                "image_id": IMAGE_ID,
                "image_digest": IMAGE_DIGEST,
                "base_url": "http://127.0.0.1:18898",
                "run_id": RUN_ID,
            }
            with self.assertRaisesRegex(ValueError, "forbidden marker"):
                execute_fixture_suite(plan, identities)

    def test_raw_census_counts_per_label_case_insensitively(self):
        payload = (
            b"nvrm gpu fault reported\nCUDA fatal: flipped bits\n"
            b"Xid 73 logged twice xid 79\nCUDA OUT OF MEMORY in worker\n"
            b"little_chunks_count = severe; illegal-MEMORY-access seen\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            fixture = Path(temp) / "census.log"
            fixture.write_bytes(payload)
            descriptor = os.open(fixture, os.O_RDONLY)
            try:
                census = S.scan_raw_log(descriptor)
            finally:
                os.close(descriptor)
        counts = census["per_pattern_match_counts"]
        self.assertEqual(
            counts,
            {
                "CUDA fatal": 1,
                "CUDA out of memory": 1,
                "illegal memory access": 1,
                "NVRM": 1,
                "Xid": 2,
            },
        )
        self.assertEqual(census["match_count"], 6)
        self.assertEqual(census["raw_log_bytes"], len(payload))
        self.assertEqual(census["raw_log_sha256"], hashlib.sha256(payload).hexdigest())

    def test_suite_terminates_and_rejects_descriptor_holding_descendants(self):
        with tempfile.TemporaryDirectory() as temp:
            source = make_plan(temp, descendant_gate="startup-readiness")
            plan = load_plan(source)
            identities = {
                "container_id": CONTAINER_ID,
                "image_id": IMAGE_ID,
                "image_digest": IMAGE_DIGEST,
                "base_url": "http://127.0.0.1:18898",
                "run_id": RUN_ID,
            }
            with self.assertRaisesRegex(RuntimeError, "process descendant"):
                execute_fixture_suite(plan, identities)

    def test_suite_rejects_runtime_directory_path_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            source = make_plan(temp)
            plan = load_plan(source)
            runtime = Path(source["runtime_dir"])
            runtime.rename(Path(temp) / "original-runtime")
            runtime.mkdir()
            for directory in ("gates", "stdout", "stderr"):
                (runtime / directory).mkdir()
            identities = {
                "container_id": CONTAINER_ID,
                "image_id": IMAGE_ID,
                "image_digest": IMAGE_DIGEST,
                "base_url": "http://127.0.0.1:18898",
                "run_id": RUN_ID,
            }
            with self.assertRaisesRegex(ValueError, "changed before execution"):
                execute_fixture_suite(plan, identities)


if __name__ == "__main__":
    unittest.main()
