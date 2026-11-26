#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline Git fixtures for the narrow committed-source presence gate."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ignored_source_gate", PROJECT / "repro-kit/check_ignored_sources.py"
)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class RepositoryFixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="source-closure-test-")
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.env = os.environ.copy()
        for name in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_COMMON_DIR",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_CONFIG_PARAMETERS",
            "GIT_CONFIG_COUNT",
        ):
            self.env.pop(name, None)
        self.env["GIT_CONFIG_NOSYSTEM"] = "1"
        self.env["GIT_CONFIG_GLOBAL"] = os.devnull
        self.git("init", "-q")
        self.policy = (
            "*token*\n*secret*\n*.credentials*\n*credential*\n.env\n*.key\n"
            + "*TOKEN*\n*SECRET*\n*CREDENTIAL*\n*API_KEY*\n*api_key*\n"
            + "service-account*.json\nsa-key*.json\n**/.aws/\nkubeconfig\n**/.kube/\n"
            + "".join(f"!/{path}\n" for path in GATE.REQUIRED_SOURCES)
        )
        self.write(".gitignore", self.policy)
        for path in GATE.REQUIRED_SOURCES:
            self.write(path, "// Synthetic source-presence fixture.\n")
        self.git("add", "--", ".gitignore", *GATE.REQUIRED_SOURCES)
        self.git("commit", "-qm", "source presence fixture")

    def close(self):
        self.temporary.cleanup()

    def git(self, *args):
        return subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Source Closure Test",
                "-c",
                "user.email=source-gate@example.invalid",
                *args,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            check=True,
        )

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def commit_policy(self, text):
        self.write(".gitignore", text)
        self.git("add", "--", ".gitignore")
        self.git("commit", "-qm", "update fixture policy")

    def nested_repository(self, path):
        self.write(f"{path}/src/new.rs", "// Synthetic nested source.\n")
        self.git("init", "-q", path)
        nested = str(self.root / path)
        self.git("-C", nested, "add", "--", "src/new.rs")
        self.git("-C", nested, "commit", "-qm", "nested source fixture")


class IgnoredSourceClosureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RepositoryFixture()
        self.addCleanup(self.fixture.close)

    def check(self):
        return GATE.check_repository(self.fixture.root)

    def test_clean_committed_sources_and_narrow_exceptions_pass(self):
        self.assertEqual(
            self.check(), self.fixture.git("rev-parse", "HEAD").stdout.decode().strip()
        )

    def test_source_absent_from_committed_tree_fails(self):
        self.fixture.git("rm", "--", GATE.REQUIRED_SOURCES[0])
        self.fixture.git("commit", "-qm", "delete required fixture source")
        with self.assertRaisesRegex(GATE.GateError, "absent from the committed tree"):
            self.check()

    def test_working_tree_source_deletion_fails(self):
        (self.fixture.root / GATE.REQUIRED_SOURCES[1]).unlink()
        with self.assertRaisesRegex(GATE.GateError, "working file is unavailable"):
            self.check()

    def test_untracked_required_source_cannot_replace_committed_source(self):
        self.fixture.git("rm", "--cached", "--", GATE.REQUIRED_SOURCES[0])
        self.fixture.git("commit", "-qm", "untrack required fixture source")
        self.assertTrue((self.fixture.root / GATE.REQUIRED_SOURCES[0]).is_file())
        with self.assertRaisesRegex(GATE.GateError, "absent from the committed tree"):
            self.check()

    def test_staged_source_deletion_fails_before_commit(self):
        self.fixture.git("rm", "--cached", "--", GATE.REQUIRED_SOURCES[0])
        with self.assertRaisesRegex(GATE.GateError, "index differs"):
            self.check()

    def test_working_source_modification_fails(self):
        self.fixture.write(GATE.REQUIRED_SOURCES[0], "// Different source.\n")
        with self.assertRaisesRegex(GATE.GateError, "working file differs"):
            self.check()

    def test_symlink_cannot_replace_required_regular_source(self):
        source = self.fixture.root / GATE.REQUIRED_SOURCES[0]
        source.unlink()
        source.symlink_to("unavailable.rs")
        with self.assertRaisesRegex(GATE.GateError, "working file is not regular"):
            self.check()

    def test_committed_symlink_source_is_rejected(self):
        source = self.fixture.root / GATE.REQUIRED_SOURCES[0]
        source.unlink()
        source.symlink_to("unavailable.rs")
        self.fixture.git("add", "--", GATE.REQUIRED_SOURCES[0])
        self.fixture.git("commit", "-qm", "replace fixture source with symlink")
        with self.assertRaisesRegex(GATE.GateError, "not a regular committed source"):
            self.check()

    def test_required_source_ignored_after_exact_exception_fails(self):
        self.fixture.commit_policy(
            self.fixture.policy + GATE.REQUIRED_SOURCES[0] + "\n"
        )
        with self.assertRaisesRegex(GATE.GateError, "required source is ignored"):
            self.check()

    def test_uncommitted_policy_cannot_override_committed_ignore_policy(self):
        self.fixture.write(".gitignore", self.fixture.policy + "!api_token\n")
        with self.assertRaisesRegex(GATE.GateError, "working file differs"):
            self.check()

    def test_credential_basename_unignore_is_rejected(self):
        self.fixture.commit_policy(self.fixture.policy + "!api_token\n")
        with self.assertRaisesRegex(
            GATE.GateError, "credential filename is no longer ignored"
        ):
            self.check()

    def test_broad_source_extension_unignore_is_rejected(self):
        self.fixture.commit_policy(self.fixture.policy + "!*.rs\n")
        with self.assertRaisesRegex(
            GATE.GateError, "credential filename is no longer ignored"
        ):
            self.check()

    def test_local_exclude_cannot_replace_committed_secret_policy(self):
        self.fixture.commit_policy(self.fixture.policy.replace("*secret*\n", ""))
        self.fixture.write(".git/info/exclude", "*secret*\n")
        with self.assertRaisesRegex(GATE.GateError, "not committed root policy"):
            self.check()

    def test_local_exclude_cannot_replace_committed_env_policy(self):
        self.fixture.commit_policy(self.fixture.policy.replace(".env\n", ""))
        self.fixture.write(".git/info/exclude", ".env\n")
        with self.assertRaisesRegex(GATE.GateError, "not committed root policy"):
            self.check()

    def test_redundant_local_excludes_do_not_displace_committed_policy(self):
        self.fixture.write(".git/info/exclude", "*secret*\n.env\n*token*\n")
        self.check()

    def test_committed_crlf_ignore_policy_is_supported(self):
        self.fixture.commit_policy(self.fixture.policy.replace("\n", "\r\n"))
        self.check()

    def test_ambiguous_normalized_rule_spelling_fails_closed(self):
        self.fixture.commit_policy(
            self.fixture.policy.replace("*secret*\n", "*secret*  \n")
        )
        with self.assertRaisesRegex(GATE.GateError, "not committed root policy"):
            self.check()

    def test_malformed_or_uncommitted_credential_provenance_is_rejected(self):
        actual_run = subprocess.run
        probe = os.fsencode(GATE.CREDENTIAL_PROBES[0])
        records = (
            (b".gitignore\0" + b"1\0*token*\0", "invalid credential-rule"),
            (
                b".gitignore\0" + b"1\0*token*\0" + probe + b"\0extra\0",
                "invalid credential-rule",
            ),
            (
                b".gitignore\0" + b"1\0*token*\0other\0",
                "invalid credential-rule",
            ),
            (
                b".gitignore\0" + b"1\0*token*\0" + probe,
                "unterminated filename inventory",
            ),
            (
                b".git/info/exclude\0" + b"1\0*token*\0" + probe + b"\0",
                "not committed root policy",
            ),
            (
                b"crates/x/.gitignore\0" + b"1\0*token*\0" + probe + b"\0",
                "not committed root policy",
            ),
            (
                b".gitignore\0" + b"0\0*token*\0" + probe + b"\0",
                "not committed root policy",
            ),
            (
                b".gitignore\0" + b"9999\0*token*\0" + probe + b"\0",
                "not committed root policy",
            ),
            (
                b".gitignore\0" + b"invalid\0*token*\0" + probe + b"\0",
                "not committed root policy",
            ),
            (
                b".gitignore\0" + b"1" * 5000 + b"\0*token*\0" + probe + b"\0",
                "not committed root policy",
            ),
            (
                b".gitignore\0" + b"1\0*secret*\0" + probe + b"\0",
                "not committed root policy",
            ),
            (
                b".gitignore\0" + b"1\0\0" + probe + b"\0",
                "not committed root policy",
            ),
            (
                b".gitignore\0" + b"1\0!api_token\0" + probe + b"\0",
                "credential filename is no longer ignored",
            ),
        )
        for record, message in records:
            with self.subTest(record=record):

                def corrupt_provenance(command, **kwargs):
                    if "check-ignore" in command and "--verbose" in command:
                        return subprocess.CompletedProcess(command, 0, record, b"")
                    return actual_run(command, **kwargs)

                with mock.patch.object(
                    GATE.subprocess, "run", side_effect=corrupt_provenance
                ):
                    with self.assertRaisesRegex(GATE.GateError, message):
                        self.check()

    def test_ignored_non_source_credential_filename_is_not_read_as_source(self):
        self.fixture.write(
            "crates/x/local_credentials.json", "synthetic test fixture\n"
        )
        self.fixture.write("api_token", "synthetic test fixture\n")
        self.check()

    def test_ignored_source_inventory_is_filename_safe(self):
        path = "crates/x/src/token [*]\nextra.rs"
        self.fixture.write(path, "// Synthetic ignored source.\n")
        with self.assertRaisesRegex(
            GATE.GateError, "ignored source files require review"
        ) as caught:
            self.check()
        self.assertIn("token [*]\\nextra.rs", str(caught.exception))
        self.assertNotIn("\n", str(caught.exception))

    def test_untracked_source_inventory_is_filename_safe(self):
        path = "crates/x/src/weird [*]\nspace name.rs"
        self.fixture.write(path, "// Synthetic untracked source.\n")
        with self.assertRaisesRegex(
            GATE.GateError, "untracked source files require review"
        ) as caught:
            self.check()
        self.assertIn("weird [*]\\nspace name.rs", str(caught.exception))
        self.assertNotIn("\n", str(caught.exception))

    def test_case_folding_cannot_hide_untracked_source(self):
        path = "crates/atlas-embed/src/Token.rs"
        self.fixture.write(path, "// Synthetic case-colliding source.\n")
        for ignore_case in ("false", "true", "false"):
            with self.subTest(ignore_case=ignore_case):
                self.fixture.git("config", "core.ignoreCase", ignore_case)
                inventory = self.fixture.git(
                    "ls-files", "--others", "--exclude-standard", "-z", "--", path
                ).stdout
                self.assertEqual(
                    inventory, b"" if ignore_case == "true" else os.fsencode(path) + b"\0"
                )
                with self.assertRaisesRegex(
                    GATE.GateError, "untracked source files require review"
                ) as caught:
                    self.check()
                self.assertIn(path, str(caught.exception))
                self.assertEqual(
                    self.fixture.git("config", "--get", "core.ignoreCase").stdout,
                    os.fsencode(ignore_case) + b"\n",
                )

    def test_untracked_nested_repository_is_an_opaque_inventory_failure(self):
        path = "crates/untracked_nested"
        self.fixture.nested_repository(path)
        inventory = self.fixture.git(
            "ls-files", "--others", "--exclude-standard", "-z", "--", path
        ).stdout
        self.assertEqual(inventory, os.fsencode(path) + b"/\0")
        with self.assertRaisesRegex(
            GATE.GateError, "untracked opaque directories require review"
        ):
            self.check()

    def test_opaque_directory_inventory_is_filename_safe(self):
        path = "crates/untracked [*]\nspace name"
        self.fixture.nested_repository(path)
        with self.assertRaisesRegex(
            GATE.GateError, "untracked opaque directories require review"
        ) as caught:
            self.check()
        self.assertIn("untracked [*]\\nspace name/", str(caught.exception))
        self.assertNotIn("\n", str(caught.exception))

    def test_ignored_nested_repository_is_an_opaque_inventory_failure(self):
        path = "kernels/token_nested [*]\nspace name"
        self.fixture.nested_repository(path)
        inventory = self.fixture.git(
            "ls-files",
            "--others",
            "--exclude-standard",
            "--ignored",
            "-z",
            "--",
            path,
        ).stdout
        self.assertEqual(inventory, os.fsencode(path) + b"/\0")
        with self.assertRaisesRegex(
            GATE.GateError, "ignored opaque directories require review"
        ) as caught:
            self.check()
        self.assertIn("token_nested [*]\\nspace name/", str(caught.exception))
        self.assertNotIn("\n", str(caught.exception))

    def test_ordinary_non_source_assets_are_outside_inventory_scope(self):
        self.fixture.write("crates/x/assets/example.json", "{}\n")
        self.fixture.write("kernels/x/README.txt", "Synthetic fixture notes.\n")
        self.check()

    def test_generated_target_sources_are_not_inventory_offenders(self):
        self.fixture.write(
            "crates/x/target/build/generated.rs", "// Generated fixture.\n"
        )
        self.fixture.write(
            "crates/x/compiler-target/build/generated.cu", "// Generated fixture.\n"
        )
        self.check()

    def test_git_start_failure_is_fatal(self):
        with mock.patch.object(GATE.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(GATE.GateError, "could not start"):
                self.check()

    def test_inventory_git_error_is_fatal_without_echoing_stderr(self):
        actual_run = subprocess.run

        def fail_inventory(command, **kwargs):
            if "ls-files" in command and "--others" in command:
                return subprocess.CompletedProcess(
                    command, 128, b"", b"private-stderr-marker"
                )
            return actual_run(command, **kwargs)

        stderr = io.StringIO()
        with mock.patch.object(GATE.subprocess, "run", side_effect=fail_inventory):
            with redirect_stderr(stderr):
                self.assertEqual(GATE.main(["--repo", str(self.fixture.root)]), 1)
        self.assertIn("ls-files failed with status 128", stderr.getvalue())
        self.assertNotIn("private-stderr-marker", stderr.getvalue())

    def test_check_ignore_status_two_is_fatal_not_not_ignored(self):
        actual_run = subprocess.run

        def fail_check_ignore(command, **kwargs):
            if "check-ignore" in command:
                return subprocess.CompletedProcess(
                    command, 2, b"", b"inspection failed"
                )
            return actual_run(command, **kwargs)

        with mock.patch.object(GATE.subprocess, "run", side_effect=fail_check_ignore):
            with self.assertRaisesRegex(
                GATE.GateError, "check-ignore failed with status 2"
            ):
                self.check()

    def test_ignored_inventory_git_error_is_fatal(self):
        actual_run = subprocess.run

        def fail_ignored_inventory(command, **kwargs):
            if "ls-files" in command and "--ignored" in command:
                return subprocess.CompletedProcess(
                    command, 128, b"", b"inspection failed"
                )
            return actual_run(command, **kwargs)

        with mock.patch.object(
            GATE.subprocess, "run", side_effect=fail_ignored_inventory
        ):
            with self.assertRaisesRegex(
                GATE.GateError, "ls-files failed with status 128"
            ):
                self.check()

    def test_head_movement_during_inspection_is_fatal(self):
        actual_run = subprocess.run
        head_reads = 0

        def move_head(command, **kwargs):
            nonlocal head_reads
            if "HEAD^{commit}" in command:
                head_reads += 1
                if head_reads == 2:
                    return subprocess.CompletedProcess(
                        command, 0, b"0" * 40 + b"\n", b""
                    )
            return actual_run(command, **kwargs)

        with mock.patch.object(GATE.subprocess, "run", side_effect=move_head):
            with self.assertRaisesRegex(GATE.GateError, "committed tree changed"):
                self.check()

    def test_no_repository_cannot_report_ok(self):
        outside = Path(self.fixture.temporary.name) / "outside"
        outside.mkdir()
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(GATE.main(["--repo", str(outside)]), 1)
        self.assertNotIn("OK", stdout.getvalue())
        self.assertIn("FAIL", stderr.getvalue())

    def test_shell_wrapper_propagates_gate_failure(self):
        outside = Path(self.fixture.temporary.name) / "outside"
        outside.mkdir()
        result = subprocess.run(
            [
                "bash",
                str(PROJECT / "repro-kit/check_ignored_sources.sh"),
                "--repo",
                str(outside),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertNotIn(b"OK", result.stdout)
        self.assertIn(b"FAIL", result.stderr)

    def test_repository_without_head_is_rejected(self):
        with mock.patch.object(GATE.Git, "run", autospec=True) as run:
            run.side_effect = [
                subprocess.CompletedProcess(
                    [], 0, os.fsencode(self.fixture.root) + b"\n", b""
                ),
                GATE.GateError("Git inspection rev-parse failed with status 128"),
            ]
            with self.assertRaisesRegex(GATE.GateError, "rev-parse failed"):
                self.check()

    def test_repository_redirection_environment_is_removed(self):
        with mock.patch.dict(
            os.environ, {"GIT_DIR": "/unrelated", "GIT_WORK_TREE": "/unrelated"}
        ):
            self.check()

    def test_unterminated_inventory_is_fatal(self):
        with self.assertRaisesRegex(GATE.GateError, "unterminated filename inventory"):
            GATE.nul_records(b"crates/x/src/source.rs")

    def test_real_policy_preserves_guards_and_exact_exceptions(self):
        lines = (PROJECT / ".gitignore").read_text(encoding="utf-8").splitlines()
        for pattern in ("*token*", "*secret*", "*.credentials*", "*credential*"):
            self.assertIn(pattern, lines)
        for path in GATE.REQUIRED_SOURCES:
            self.assertIn(f"!/{path}", lines)


if __name__ == "__main__":
    unittest.main()
