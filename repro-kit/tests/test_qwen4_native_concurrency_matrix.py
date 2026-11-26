#!/usr/bin/env python3

import atexit
import copy
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "repro-kit"
sys.path.insert(0, str(REPRO))
MODULE_PATH = REPRO / "qwen4_native_concurrency_matrix.py"
_TEMP_SNAPSHOT: tempfile.TemporaryDirectory | None = None


def _load_matrix_module():
    """Load the reviewed matrix module, falling back to a committed temp snapshot if the worktree is dirty."""
    global _TEMP_SNAPSHOT

    def _exec(path: Path):
        spec = importlib.util.spec_from_file_location("qwen4_native_concurrency_matrix", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        return _exec(MODULE_PATH)
    except Exception as exc:
        if "differs from its captured Git blob" not in str(exc):
            raise
    # When one of the three reviewed scripts has uncommitted working-tree edits
    # (e.g., before branch commit), test the exact working-tree bytes inside an
    # isolated temp Git repo so production's strict Git-blob preflight runs unmodified.
    tmp = tempfile.TemporaryDirectory(prefix="flux-matrix-snapshot-")
    _TEMP_SNAPSHOT = tmp
    atexit.register(tmp.cleanup)
    repo = Path(tmp.name)
    for rel in (
        Path("repro-kit/qwen4_native_concurrency_matrix.py"),
        Path("repro-kit/qwen4_native_concurrency_probe.py"),
        Path("repro-kit/r9_source_to_binary.py"),
    ):
        dest = repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    for cmd in (
        ["git", "-C", str(repo), "init", "-q"],
        ["git", "-C", str(repo), "add", "repro-kit"],
        ["git", "-C", str(repo), "commit", "-q", "-m", "snapshot"],
    ):
        subprocess.run(cmd, env=git_env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return _exec(repo / "repro-kit/qwen4_native_concurrency_matrix.py")


matrix = _load_matrix_module()


class NativeConcurrencyMatrixTests(unittest.TestCase):
    MAX_TOKENS = 16
    MINIMUM_FULL_FRAMES = 8
    MEMORY_FLOOR_MIB = 64

    @staticmethod
    def sha(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def git(
        repository: Path, *arguments: str, input_bytes: bytes | None = None
    ) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return completed.stdout

    @staticmethod
    def reviewed_probe_sha256() -> str:
        return matrix.reviewed_probe_source().sha256

    @classmethod
    def ordinal(cls, label: str) -> int:
        return int(label.removeprefix("row-"))

    @classmethod
    def session(cls, label: str) -> str:
        return f"{cls.ordinal(label) + 1:016x}"

    @classmethod
    def logits(cls, label: str, output_tokens: int | None = None) -> list[dict]:
        count = cls.MAX_TOKENS if output_tokens is None else output_tokens
        return [
            {
                "index": index,
                "fnv": cls.sha(f"fnv:{label}:{index}")[:16],
                "sha256": cls.sha(f"logit:{label}:{index}"),
            }
            for index in range(1, count)
        ]

    @classmethod
    def request(cls, label: str) -> dict:
        prompt_tokens = 32 + cls.ordinal(label)
        token_ids = [cls.ordinal(label) + 7] * cls.MAX_TOKENS
        completion_tokens = len(token_ids)
        return {
            "label": label,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usage_identity": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "token_ids_sha256": matrix.canonical_hash(token_ids),
            "token_ids": token_ids,
            "finish_reason": "length",
            "elapsed_s": 0.25,
        }

    @classmethod
    def state(
        cls,
        label: str,
        output_tokens: int | None = None,
        *,
        computed_unemitted: bool = False,
    ) -> dict:
        output = cls.MAX_TOKENS if output_tokens is None else output_tokens
        seq_len = cls.request(label)["prompt_tokens"] + output - 1 + computed_unemitted
        return {
            "schema": 1,
            "mode": "unloaded-target-only",
            "scheduler_path": "ordinary-target-only",
            "proposer_loaded": False,
            "session": cls.session(label),
            "seq_len": seq_len,
            "token_count": seq_len,
            "output_tokens": output,
            "gdn_layers": 36,
            "attention_layers": 12,
            "logical_kv_blocks": 12,
            "kv_token_positions": seq_len * 12,
            "compared_buffers": 49,
            "compared_bytes": 4096,
            "fed_tokens_sha256": cls.sha(f"fed:{label}:{output}"),
            "target_state_sha256": cls.sha(f"state:{label}:{output}"),
        }

    @classmethod
    def runtime_state(
        cls,
        label: str,
        line_number: int,
        output_tokens: int | None = None,
        *,
        computed_unemitted: bool = False,
    ) -> dict:
        return {
            **cls.state(label, output_tokens, computed_unemitted=computed_unemitted),
            "line_number": line_number,
        }

    @classmethod
    def baseline(cls, label: str) -> dict:
        logits = cls.logits(label)
        return {
            **cls.request(label),
            "session": cls.session(label),
            "logits_rows": len(logits),
            "logits": logits,
            "logits_sha256_chain": matrix.canonical_hash(logits),
            "target_state": cls.state(label),
            "log_sha256": cls.sha(f"baseline-log:{label}"),
        }

    @classmethod
    def frame(
        cls,
        labels: list[str],
        indices: list[int],
        slots: list[int],
        line_number: int,
        route: str,
    ) -> dict:
        return {
            "line_number": line_number,
            "route": route,
            "rows": len(labels),
            "sessions": [cls.session(label) for label in labels],
            "slots": slots,
            "pre_step_seq_lens": [
                cls.request(label)["prompt_tokens"] + index - 1
                for label, index in zip(labels, indices)
            ],
            "logits": [
                {
                    "session": cls.session(label),
                    **cls.logits(label)[index - 1],
                }
                for label, index in zip(labels, indices)
            ],
        }

    @classmethod
    def wave(cls, name: str, labels: list[str], route: str) -> dict:
        ordered = labels if name == "forward" else list(reversed(labels))
        slots = list(range(len(ordered)))
        frames = [
            cls.frame(
                ordered,
                [index] * len(ordered),
                slots,
                index * 10,
                route,
            )
            for index in range(1, cls.MAX_TOKENS)
        ]
        terminal_states = {
            label: cls.runtime_state(label, 200 + index)
            for index, label in enumerate(ordered)
        }
        return {
            "name": name,
            "route": route,
            "request_order": ordered,
            "full_width_frames": len(frames),
            "first_full_frame_order": ordered,
            "slot_sets": [frame["slots"] for frame in frames],
            "session_slots": {
                cls.session(label): slot for label, slot in zip(ordered, slots)
            },
            "logits": {label: cls.logits(label) for label in ordered},
            "terminal_states": terminal_states,
            "frames": frames,
            "state_sha256": {
                label: terminal_states[label]["target_state_sha256"]
                for label in ordered
            },
            "log_sha256": cls.sha(f"wave-log:{route}:{name}:{len(labels)}"),
            "requests": [cls.request(label) for label in ordered],
        }

    @classmethod
    def churn(cls, route: str) -> tuple[dict, dict]:
        originals = [f"row-{index:02d}" for index in range(8)]
        target = originals[-1]
        replacement = "row-08"
        survivors = originals[:-1]
        frames = [
            cls.frame(originals, [index] * 8, list(range(8)), index * 10, route)
            for index in range(1, 9)
        ]
        frames.extend(
            cls.frame(
                [*survivors, replacement],
                [index + 8] * 7 + [index],
                list(range(8)),
                90 + index * 10,
                route,
            )
            for index in range(1, 8)
        )
        cancelled_state = cls.runtime_state(
            target, 90, output_tokens=8, computed_unemitted=True
        )
        terminal_states = {
            target: cancelled_state,
            **{
                label: cls.runtime_state(label, 200 + index)
                for index, label in enumerate([*survivors, replacement])
            },
        }
        cancelled_logits = cls.logits(target, output_tokens=9)
        all_labels = [*originals, replacement]
        churn = {
            "route": route,
            "cancelled": {
                "label": target,
                "content_chunks_before_close": 8,
                "closed_unix": 100.0,
                "session": cls.session(target),
                "state_output_tokens": 8,
                "terminal_geometry": "computed-unemitted",
                "retired_slot": 7,
                "logits": cancelled_logits,
                "target_state": cancelled_state,
            },
            "replacement": cls.request(replacement),
            "replacement_started_unix": 101.0,
            "replacement_reused_retired_slot": True,
            "survivors": [cls.request(label) for label in survivors],
            "session_slots": {
                **{cls.session(label): index for index, label in enumerate(originals)},
                cls.session(replacement): 7,
            },
            "logits": {
                label: (cancelled_logits if label == target else cls.logits(label))
                for label in all_labels
            },
            "terminal_states": terminal_states,
            "frames": frames,
            "log_sha256": cls.sha(f"churn-log:{route}"),
        }
        post_logits = cls.logits(replacement)
        post_drain = {
            "request": cls.request(replacement),
            "session": cls.session(replacement),
            "logits": post_logits,
            "logits_sha256_chain": matrix.canonical_hash(post_logits),
            "target_state": cls.runtime_state(replacement, 10),
            "frames": [],
            "log_sha256": cls.sha(f"post-drain-log:{route}"),
        }
        return churn, post_drain

    @classmethod
    def arm(cls, profile: str, width: int) -> dict:
        route = "serial" if profile == "serial-control" else "native"
        profile_index = matrix.PROFILES.index(profile)
        identity = profile_index * 10 + width
        manifest_sha256 = cls.sha(f"manifest:{profile}:{width}")
        cases = [
            {
                "label": f"row-{index:02d}",
                "ordinal": index,
                "prompt_sha256": cls.sha(f"prompt:{width}:{index}"),
            }
            for index in range(width + (1 if width == 8 else 0))
        ]
        labels = [case["label"] for case in cases[:width]]
        churn, post_drain = cls.churn(route) if width == 8 else (None, None)
        arm = {
            "schema": matrix.probe.SCHEMA,
            "evidence_tier": "full",
            "selector_profile": profile,
            "width": width,
            "expected_route": route,
            "model": "qwen38",
            "max_tokens": cls.MAX_TOKENS,
            "minimum_full_frames": cls.MINIMUM_FULL_FRAMES,
            "memory_floor_mib": cls.MEMORY_FLOOR_MIB,
            "probe_name": "qwen4_native_concurrency_probe.py",
            "probe_sha256": cls.reviewed_probe_sha256(),
            "expected_binary_sha256": cls.sha("binary"),
            "expected_manifest_sha256": manifest_sha256,
            "cases": cases,
            "plan_sha256": "1" * 64,
            "provenance": {
                "manifest_schema": matrix.probe.LAUNCH_SCHEMA,
                "manifest_name": f"{profile}-c{width}-launch.json",
                "manifest_sha256": manifest_sha256,
                "binary_name": "flux-server",
                "binary_sha256": cls.sha("binary"),
                "release_version": matrix.probe.RELEASE_VERSION,
                "source_revision": "3" * 40,
                "source_tree": "4" * 40,
                "source_bundle_manifest_name": "r15-source-manifest.json",
                "source_bundle_manifest_sha256": cls.sha("source-manifest"),
                "source_bundle_name": "r15-source.tar.zst",
                "source_bundle_sha256": cls.sha("source-bundle"),
                "build_receipt_name": "r15-build.json",
                "build_receipt_sha256": cls.sha("build-receipt"),
                "source_to_binary_receipt_name": "r15-source-to-binary.json",
                "source_to_binary_receipt_sha256": cls.sha("source-to-binary"),
                "checkpoint_id": "hf://JMNI-Labs/Qwen3.8-Flash-Next-NVFP4",
                "checkpoint_revision": "5" * 40,
                "checkpoint_manifest_sha256": cls.sha("checkpoint-manifest"),
                "checkpoint_files_verified": 12,
                "checkpoint_bytes_verified": 999_999,
                "selector_profile": profile,
                "configured_max_batch_size": width,
                "configured_max_num_seqs": width,
                "server_process": {
                    "pid": 1000 + identity,
                    "boot_id": "12345678-1234-1234-1234-123456789abc",
                    "start_ticks": 10_000 + identity,
                    "log_device": 1,
                    "log_inode": 20_000 + identity,
                },
            },
            "result": "PASS",
            "arm_qualified": True,
            "release_qualified": False,
            "started_unix": 1000.0,
            "finished_unix": 1010.0,
            "server_pid": 1000 + identity,
            "server_log_name": f"{profile}-c{width}.log",
            "server_log_qualified_start_offset": 0,
            "server_log_qualified_end_offset": 4096,
            "server_log_sha256": cls.sha(f"server-log:{profile}:{width}"),
            "safety_log_marker_counts": {
                marker: 0 for marker in matrix.probe.FORBIDDEN_LOG_MARKERS
            },
            "memory": {
                "samples": 10,
                "mem_available_floor_kib": cls.MEMORY_FLOOR_MIB * 1024,
                "server_vm_rss_peak_kib": 4096,
                "server_vm_hwm_peak_kib": 8192,
            },
            "baseline_c1": [cls.baseline(case["label"]) for case in cases],
            "waves": [
                cls.wave("forward", labels, route),
                cls.wave("reverse", labels, route),
            ],
            "reference_receipt": (
                None
                if profile == "native-reference"
                else {"name": "reference.json", "sha256": cls.sha("reference")}
            ),
            "api_churn": churn,
            "post_drain_reuse": post_drain,
            "native_engagement_widths": [] if route == "serial" else [width],
            "optimization_selector_engagement_counts": {
                selector: 1 if profile == "native-optimized" else 0
                for selector in matrix.probe.OPTIMIZATION_SELECTORS
            },
            "limitations": list(matrix.FULL_LIMITATIONS),
        }
        arm["plan_sha256"] = matrix.canonical_hash(
            {key: arm[key] for key in matrix.PLAN_KEYS}
        )
        return arm

    @classmethod
    def valid_matrix(cls) -> tuple[dict, dict]:
        arms = {}
        public = {}
        for profile in matrix.PROFILES:
            for width in matrix.WIDTHS:
                arm = cls.arm(profile, width)
                arms[(profile, width)] = arm
                public[(profile, width)] = {
                    "name": f"{profile}-c{width}.json",
                    "sha256": cls.sha(f"arm:{profile}:{width}"),
                }
        for width in matrix.WIDTHS:
            reference = public[("native-reference", width)]
            for profile in ("native-optimized", "serial-control"):
                arms[(profile, width)]["reference_receipt"] = dict(reference)
        return arms, public

    def assert_arm_rejected(self, arm: dict, pattern: str) -> None:
        with self.assertRaisesRegex(matrix.probe.ProbeError, pattern):
            matrix.validate_arm(arm, "fixture")

    def test_valid_nine_arm_matrix_is_release_qualified(self) -> None:
        arms, public = self.valid_matrix()
        result = matrix.validate_matrix(arms, public)
        self.assertTrue(result["release_qualified"])
        self.assertEqual(result["schema"], matrix.SCHEMA)
        self.assertEqual(len(result["arms"]), 9)
        self.assertEqual(result["probe_name"], "qwen4_native_concurrency_probe.py")
        self.assertEqual(result["probe_sha256"], self.reviewed_probe_sha256())
        self.assertEqual(
            len({row["server_process"]["pid"] for row in result["arms"]}), 9
        )

    def test_self_attested_probe_digest_fails_closed(self) -> None:
        arm = self.arm("native-reference", 2)
        arm["probe_sha256"] = self.sha("attacker-controlled-probe")
        arm["plan_sha256"] = matrix.canonical_hash(
            {key: arm[key] for key in matrix.PLAN_KEYS}
        )
        self.assert_arm_rejected(arm, "reviewed qualifier source")

    def test_alternate_same_named_probe_source_fails_closed(self) -> None:
        arm = self.arm("native-reference", 2)
        with tempfile.TemporaryDirectory() as directory:
            alternate = Path(directory) / "qwen4_native_concurrency_probe.py"
            alternate.write_bytes(Path(matrix.probe.__file__).read_bytes())
            with mock.patch.object(matrix.probe, "__file__", str(alternate)):
                self.assert_arm_rejected(arm, "not the reviewed sibling")

    def _isolated_matrix_repo(self, directory: str) -> tuple[Path, object]:
        repository = Path(directory) / "repo"
        repository.mkdir()
        self.git(repository, "init", "--quiet")
        for rel in matrix.SOURCE_RELATIVE_PATHS:
            dest = repository / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((matrix.REPOSITORY_ROOT / rel).read_bytes())
        self.git(repository, "add", *[rel.as_posix() for rel in matrix.SOURCE_RELATIVE_PATHS])
        self.git(
            repository,
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        )
        closure = matrix.capture_source_closure(repository_root=repository)
        return repository, closure

    def test_same_path_probe_replacement_fails_against_tracked_git_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, closure = self._isolated_matrix_repo(directory)
            probe_path = repository / matrix.PROBE_RELATIVE_PATH
            original = probe_path.read_bytes()
            with (
                mock.patch.object(matrix, "REPOSITORY_ROOT", repository),
                mock.patch.object(matrix, "PREIMPORT_SOURCE_CLOSURE", closure),
                mock.patch.object(matrix.probe, "__file__", str(probe_path)),
            ):
                probe_path.write_bytes(original + b"\n# hostile same-path replacement\n")
                with self.assertRaisesRegex(
                    matrix.probe.ProbeError, "differs from its captured Git blob"
                ):
                    matrix.reviewed_probe_source()

    def test_clean_git_provenance_preflight_passes(self) -> None:
        seal, payload = matrix.preflight_probe_import()
        self.assertEqual(seal, matrix.PREIMPORT_PROBE_SEAL)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), seal[3])

    def test_git_provenance_disables_replace_refs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            self.git(repository, "init", "--quiet")
            tracked = repository / "tracked.txt"
            original = b"reviewed tracked bytes\n"
            replacement = b"hostile replacement bytes\n"
            tracked.write_bytes(original)
            self.git(repository, "add", "tracked.txt")
            self.git(
                repository,
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            )
            original_oid = (
                self.git(repository, "rev-parse", "HEAD:tracked.txt")
                .decode("ascii")
                .strip()
            )
            replacement_oid = (
                self.git(
                    repository,
                    "hash-object",
                    "-w",
                    "--stdin",
                    input_bytes=replacement,
                )
                .decode("ascii")
                .strip()
            )
            self.git(
                repository,
                "update-ref",
                f"refs/replace/{original_oid}",
                replacement_oid,
            )
            self.assertEqual(
                self.git(repository, "cat-file", "blob", original_oid), replacement
            )
            self.assertEqual(
                matrix.preflight_git_output(
                    ["cat-file", "blob", original_oid],
                    "replace-ref fixture",
                    repository_root=repository,
                ),
                original,
            )

    def test_git_provenance_ignores_foreign_repository_and_config_environment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            foreign = Path(directory) / "foreign"
            foreign.mkdir()
            self.git(foreign, "init", "--quiet")
            redirected = os.environ.copy()
            redirected.update(
                {
                    "GIT_DIR": str(foreign / ".git"),
                    "GIT_WORK_TREE": str(foreign),
                }
            )
            raw = subprocess.run(
                [
                    "git",
                    "-C",
                    str(matrix.REPOSITORY_ROOT),
                    "rev-parse",
                    "--show-toplevel",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                env=redirected,
            ).stdout
            self.assertEqual(Path(raw.decode().strip()).resolve(), foreign.resolve())

            hostile = {
                "GIT_DIR": str(foreign / ".git"),
                "GIT_WORK_TREE": str(foreign),
                "GIT_COMMON_DIR": str(foreign / ".git"),
                "GIT_OBJECT_DIRECTORY": str(foreign / ".git" / "objects"),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(foreign / ".git" / "objects"),
                "GIT_NAMESPACE": "hostile",
                "GIT_INDEX_FILE": str(foreign / ".git" / "index"),
                "GIT_CONFIG": str(foreign / "hostile.config"),
                "GIT_CONFIG_SYSTEM": str(foreign / "system.config"),
                "GIT_CONFIG_GLOBAL": str(foreign / "global.config"),
                "GIT_CONFIG_NOSYSTEM": "0",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.repositoryformatversion",
                "GIT_CONFIG_VALUE_0": "999",
            }
            with mock.patch.dict(os.environ, hostile, clear=False):
                safe_environment = matrix.sanitized_git_environment()
                self.assertEqual(
                    safe_environment,
                    {
                        "PATH": os.defpath,
                        "LC_ALL": "C",
                        "LANG": "C",
                        "GIT_NO_LAZY_FETCH": "1",
                    },
                )
                self.assertEqual(
                    [key for key in safe_environment if key.startswith("GIT_")],
                    ["GIT_NO_LAZY_FETCH"],
                )
                observed = matrix.preflight_git_output(
                    ["rev-parse", "--show-toplevel"],
                    "sanitized repository root",
                )
            self.assertEqual(
                Path(observed.decode().strip()).resolve(), matrix.REPOSITORY_ROOT
            )

    def test_source_closure_derives_every_object_from_one_captured_commit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            self.git(repository, "init", "--quiet")
            tracked = repository / "tracked.py"
            tracked.write_text("reviewed = 'first'\n", encoding="utf-8")
            self.git(repository, "add", "tracked.py")
            self.git(
                repository,
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "first",
            )
            first = self.git(repository, "rev-parse", "HEAD").decode().strip()
            tracked.write_text("reviewed = 'second'\n", encoding="utf-8")
            self.git(repository, "add", "tracked.py")
            self.git(
                repository,
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "second",
            )
            second = self.git(repository, "rev-parse", "HEAD").decode().strip()
            self.git(repository, "reset", "--hard", "--quiet", first)

            original = matrix.preflight_git_output
            calls: list[tuple[str, ...]] = []

            def race(arguments, label, *, repository_root=matrix.REPOSITORY_ROOT):
                result = original(arguments, label, repository_root=repository_root)
                calls.append(tuple(arguments))
                if arguments == ["rev-parse", "--verify", "HEAD^{commit}"]:
                    if len([call for call in calls if call == tuple(arguments)]) == 1:
                        self.git(repository, "reset", "--hard", "--quiet", second)
                return result

            with mock.patch.object(
                matrix, "preflight_git_output", side_effect=race
            ), self.assertRaisesRegex(
                matrix.SourcePreflightError,
                "captured Git blob|HEAD changed",
            ):
                matrix.capture_source_closure(
                    repository_root=repository,
                    relative_paths=(Path("tracked.py"),),
                )

            flattened = [argument for call in calls for argument in call]
            self.assertIn(f"{first}^{{tree}}", flattened)
            self.assertIn(f"{first}:tracked.py", flattened)
            self.assertNotIn("HEAD^{tree}", flattened)
            self.assertNotIn("HEAD:tracked.py", flattened)

    def test_git_preflight_cannot_materialize_a_promised_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin = root / "origin"
            partial = root / "partial"
            origin.mkdir()
            self.git(origin, "init", "--quiet")
            tracked = origin / "tracked.txt"
            tracked.write_bytes(b"promised payload\n")
            self.git(origin, "add", "tracked.txt")
            self.git(
                origin,
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            )
            self.git(origin, "config", "uploadpack.allowFilter", "true")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-checkout",
                    "--filter=blob:none",
                    f"file://{origin}",
                    str(partial),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            blob = self.git(partial, "rev-parse", "HEAD:tracked.txt").decode().strip()
            object_root = partial / ".git/objects"
            before = {
                path.relative_to(object_root): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in object_root.rglob("*")
                if path.is_file()
            }
            with self.assertRaisesRegex(
                matrix.SourcePreflightError, "cannot resolve promised blob"
            ):
                matrix.preflight_git_output(
                    ["cat-file", "blob", blob],
                    "promised blob",
                    repository_root=partial,
                )
            after = {
                path.relative_to(object_root): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in object_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_reviewed_modules_use_private_captured_source_identities(self) -> None:
        self.assertTrue(matrix.PROBE_MODULE_NAME.startswith("_atlas_reviewed_"))
        self.assertTrue(matrix.R9_MODULE_NAME.startswith("_atlas_reviewed_"))
        self.assertEqual(matrix.probe.__name__, matrix.PROBE_MODULE_NAME)
        self.assertEqual(matrix.r9_provenance.__name__, matrix.R9_MODULE_NAME)
        self.assertIs(matrix.probe.r9_provenance, matrix.r9_provenance)
        self.assertEqual(
            Path(matrix.probe.__file__).resolve(),
            matrix.REPOSITORY_ROOT / matrix.PROBE_RELATIVE_PATH,
        )

    def test_matrix_and_r9_helper_same_path_replacements_fail_closed(self) -> None:
        for relative, label in (
            (matrix.MATRIX_RELATIVE_PATH, "matrix sealer"),
            (matrix.R9_HELPER_RELATIVE_PATH, "R9 provenance helper"),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                repository, closure = self._isolated_matrix_repo(directory)
                path = repository / relative
                original = path.read_bytes()
                with (
                    mock.patch.object(matrix, "REPOSITORY_ROOT", repository),
                    mock.patch.object(matrix, "PREIMPORT_SOURCE_CLOSURE", closure),
                    mock.patch.object(
                        matrix.probe,
                        "__file__",
                        str(repository / matrix.PROBE_RELATIVE_PATH),
                    ),
                    mock.patch.object(
                        matrix.r9_provenance,
                        "__file__",
                        str(repository / matrix.R9_HELPER_RELATIVE_PATH),
                    ),
                ):
                    path.write_bytes(original + b"\n# hostile replacement\n")
                    with self.assertRaisesRegex(
                        matrix.probe.ProbeError,
                        "differs from its captured Git blob",
                    ):
                        matrix.reviewed_source_closure()

    def test_same_bytes_with_changed_source_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, closure = self._isolated_matrix_repo(directory)
            path = repository / matrix.MATRIX_RELATIVE_PATH
            original_stat = path.stat()
            with (
                mock.patch.object(matrix, "REPOSITORY_ROOT", repository),
                mock.patch.object(matrix, "PREIMPORT_SOURCE_CLOSURE", closure),
                mock.patch.object(
                    matrix.probe,
                    "__file__",
                    str(repository / matrix.PROBE_RELATIVE_PATH),
                ),
                mock.patch.object(
                    matrix.r9_provenance,
                    "__file__",
                    str(repository / matrix.R9_HELPER_RELATIVE_PATH),
                ),
            ):
                os.utime(
                    path,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1),
                )
                with self.assertRaisesRegex(
                    matrix.probe.ProbeError,
                    "source identity changed after preflight",
                ):
                    matrix.reviewed_source_closure()

    def test_cli_preimport_failure_has_no_traceback_or_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self._isolated_matrix_repo(directory)
            probe_path = repository / matrix.PROBE_RELATIVE_PATH
            original = probe_path.read_bytes()
            probe_path.write_bytes(original + b"\n# hostile replacement\n")
            completed = subprocess.run(
                [sys.executable, "-B", str(repository / matrix.MATRIX_RELATIVE_PATH)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("FAIL: reviewed", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertNotIn(str(repository), completed.stderr)

    def test_cli_existing_output_and_success_print_only_basename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private-parent"
            root.mkdir()
            existing = root / "already.json"
            existing.write_text("occupied\n", encoding="utf-8")
            existing_args = mock.Mock(receipt=existing, arm=[])
            with mock.patch.object(matrix, "parse_args", return_value=existing_args):
                with self.assertRaises(FileExistsError) as raised:
                    matrix.main()
            self.assertIn(existing.name, str(raised.exception))
            self.assertNotIn(str(root), str(raised.exception))

            arms, _ = self.valid_matrix()
            paths = {key: root / f"{key[0]}-c{key[1]}.json" for key in arms}
            for key, arm in arms.items():
                paths[key].write_bytes(matrix.probe.json_canonical(arm))
            reference_digests = {
                width: hashlib.sha256(
                    paths[("native-reference", width)].read_bytes()
                ).hexdigest()
                for width in matrix.WIDTHS
            }
            for profile in ("native-optimized", "serial-control"):
                for width in matrix.WIDTHS:
                    arm = arms[(profile, width)]
                    arm["reference_receipt"] = {
                        "name": paths[("native-reference", width)].name,
                        "sha256": reference_digests[width],
                    }
                    paths[(profile, width)].write_bytes(
                        matrix.probe.json_canonical(arm)
                    )
            specs = [
                [str(paths[key]), hashlib.sha256(paths[key].read_bytes()).hexdigest()]
                for key in arms
            ]
            receipt = root / "sealed.json"
            success_args = mock.Mock(receipt=receipt, arm=specs)
            stdout = io.StringIO()
            with mock.patch.object(
                matrix, "parse_args", return_value=success_args
            ), mock.patch("sys.stdout", stdout):
                self.assertEqual(matrix.main(), 0)
            self.assertTrue(receipt.is_file())
            self.assertIn(receipt.name, stdout.getvalue())
            self.assertNotIn(str(root), stdout.getvalue())

    def test_arm_hash_and_parse_share_one_capture_during_swap_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arms, _ = self.valid_matrix()
            paths = {key: root / f"{key[0]}-c{key[1]}.json" for key in arms}
            for key, arm in arms.items():
                paths[key].write_bytes(matrix.probe.json_canonical(arm))
            reference_digests = {
                width: hashlib.sha256(
                    paths[("native-reference", width)].read_bytes()
                ).hexdigest()
                for width in matrix.WIDTHS
            }
            for profile in ("native-optimized", "serial-control"):
                for width in matrix.WIDTHS:
                    arm = arms[(profile, width)]
                    arm["reference_receipt"] = {
                        "name": paths[("native-reference", width)].name,
                        "sha256": reference_digests[width],
                    }
                    paths[(profile, width)].write_bytes(
                        matrix.probe.json_canonical(arm)
                    )
            specs = [
                [str(paths[key]), hashlib.sha256(paths[key].read_bytes()).hexdigest()]
                for key in arms
            ]
            target = paths[("native-reference", 2)]
            malicious = matrix.probe.json_canonical(
                {**arms[("native-reference", 2)], "model": "/private/model"}
            )
            capture = matrix.probe.read_stable_file_bytes
            attack_exercised = False

            def capture_during_swap(path: Path, label: str):
                nonlocal attack_exercised
                result = capture(path, label)
                if path == target:
                    original = path.read_bytes()
                    path.write_bytes(malicious)
                    try:
                        self.assertEqual(path.read_bytes(), malicious)
                        attack_exercised = True
                    finally:
                        path.write_bytes(original)
                return result

            with mock.patch.object(
                matrix.probe,
                "read_stable_file_bytes",
                side_effect=capture_during_swap,
            ):
                loaded, _, _ = matrix.load_arms(specs)
            self.assertTrue(attack_exercised)
            self.assertEqual(loaded[("native-reference", 2)]["model"], "qwen38")

    def test_shallow_v2_style_fixture_fails_closed(self) -> None:
        arm = self.arm("native-reference", 2)
        arm["waves"][0] = {
            "name": "forward",
            "route": "native",
            "full_width_frames": 2,
        }
        self.assert_arm_rejected(arm, "object keys differ")

        arm = self.arm("native-reference", 2)
        arm["minimum_full_frames"] = 7
        self.assert_arm_rejected(arm, "exact built-in integer >= 8")

    def test_exact_built_in_integer_types_are_required(self) -> None:
        for field, hostile in (("samples", True), ("samples", 10.0)):
            with self.subTest(hostile=hostile):
                arm = self.arm("native-reference", 2)
                arm["memory"][field] = hostile
                self.assert_arm_rejected(arm, "exact built-in integer")
        arm = self.arm("native-reference", 2)
        arm["baseline_c1"][0]["token_ids"][0] = True
        self.assert_arm_rejected(arm, "exact built-in integer")
        arm = self.arm("native-reference", 2)
        arm["waves"][0]["slot_sets"][0][0] = False
        self.assert_arm_rejected(arm, "exact built-in integer")
        arm = self.arm("native-reference", 2)
        arm["native_engagement_widths"] = [True]
        self.assert_arm_rejected(arm, "exact built-in integer")

    def test_nonfinite_numbers_fail_closed(self) -> None:
        arm = self.arm("native-reference", 2)
        arm["started_unix"] = float("inf")
        self.assert_arm_rejected(arm, "finite built-in number")
        arm = self.arm("native-reference", 2)
        arm["waves"][0]["requests"][0]["elapsed_s"] = float("nan")
        self.assert_arm_rejected(arm, "finite built-in number")

    def test_json_exponent_overflow_fails_before_arm_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "overflow.json"
            payload = b'{"started_unix":1e400}'
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(matrix.probe.ProbeError, "non-finite"):
                matrix.load_arms([[str(path), digest]] * 9)

    def test_baseline_logits_chain_and_state_are_replayed(self) -> None:
        arm = self.arm("native-reference", 2)
        arm["baseline_c1"][0]["logits"][0]["sha256"] = self.sha("mutated")
        self.assert_arm_rejected(arm, "logits SHA-256 chain differs")
        arm = self.arm("native-reference", 2)
        arm["baseline_c1"][0]["target_state"]["kv_token_positions"] += 1
        self.assert_arm_rejected(arm, "KV position census differs")

    def test_frame_logits_sequence_and_session_slot_binding_are_replayed(self) -> None:
        arm = self.arm("native-reference", 4)
        arm["waves"][0]["frames"][0]["logits"][0]["sha256"] = self.sha("mutated")
        self.assert_arm_rejected(arm, "frame logits do not bind")
        arm = self.arm("native-reference", 4)
        arm["waves"][0]["frames"][1]["pre_step_seq_lens"][0] += 1
        self.assert_arm_rejected(arm, "sequence length differs|not stepwise")
        arm = self.arm("native-reference", 4)
        arm["waves"][0]["frames"][1]["slots"][0] = 3
        self.assert_arm_rejected(arm, "aliases a physical slot|moved physical slots")

    def test_normal_terminal_and_frame_geometry_reject_both_offsets(self) -> None:
        for offset in (-1, 1):
            with self.subTest(terminal_offset=offset):
                arm = self.arm("native-reference", 2)
                state = arm["baseline_c1"][0]["target_state"]
                for key in ("seq_len", "token_count"):
                    state[key] += offset
                state["kv_token_positions"] = state["seq_len"] * 12
                self.assert_arm_rejected(arm, "token census differs")
            with self.subTest(frame_offset=offset):
                arm = self.arm("native-reference", 2)
                for frame in arm["waves"][0]["frames"]:
                    frame["pre_step_seq_lens"] = [
                        value + offset for value in frame["pre_step_seq_lens"]
                    ]
                self.assert_arm_rejected(arm, "sequence length differs")

    def test_cancelled_pending_output_has_a_distinct_exact_raw_row_extent(self) -> None:
        arm = self.arm("native-reference", 8)
        churn = arm["api_churn"]
        target = churn["cancelled"]["label"]
        cancelled = churn["cancelled"]
        cancelled["terminal_geometry"] = "pending-output"
        cancelled["logits"].pop()
        for key in ("seq_len", "token_count"):
            cancelled["target_state"][key] -= 1
        cancelled["target_state"]["kv_token_positions"] -= 12
        last_target = churn["frames"][7]
        last_target["rows"] -= 1
        for key in ("sessions", "slots", "pre_step_seq_lens", "logits"):
            last_target[key].pop()
        matrix.validate_arm(arm, "fixture")
        self.assertEqual(churn["logits"][target][-1]["index"], 7)

    def test_computed_unemitted_row_cannot_be_dropped_invented_or_mislabeled(
        self,
    ) -> None:
        arm = self.arm("native-reference", 8)
        arm["api_churn"]["cancelled"]["logits"].pop()
        self.assert_arm_rejected(arm, "logits list is incomplete")
        arm = self.arm("native-reference", 8)
        arm["api_churn"]["cancelled"]["terminal_geometry"] = "pending-output"
        self.assert_arm_rejected(arm, "logits list is incomplete")
        arm = self.arm("native-reference", 8)
        arm["api_churn"]["cancelled"]["logits"].append(self.logits("row-07")[8])
        self.assert_arm_rejected(arm, "logits list is incomplete")
        arm = self.arm("native-reference", 8)
        cancelled = arm["api_churn"]["cancelled"]
        cancelled["logits"][-1]["sha256"] = self.sha("wrong-extra-row")
        self.assert_arm_rejected(arm, "exact C1 prefix")

    def test_cancelled_receipt_requires_final_computed_row_in_its_last_frame(
        self,
    ) -> None:
        arm = self.arm("native-reference", 8)
        last_target = arm["api_churn"]["frames"][7]
        last_target["rows"] -= 1
        for key in ("sessions", "slots", "pre_step_seq_lens", "logits"):
            last_target[key].pop()
        self.assert_arm_rejected(arm, "final frame does not bind target-fed extent")

    def test_terminal_state_must_follow_final_frame(self) -> None:
        arm = self.arm("native-reference", 2)
        label = arm["waves"][0]["request_order"][0]
        arm["waves"][0]["terminal_states"][label]["line_number"] = 20
        self.assert_arm_rejected(arm, "retired session re-entered|terminal state")

    def test_forward_and_reverse_orders_are_exact_at_every_width(self) -> None:
        for width in matrix.WIDTHS:
            with self.subTest(width=width):
                arm = self.arm("native-reference", width)
                arm["waves"][1]["request_order"] = arm["waves"][0]["request_order"]
                self.assert_arm_rejected(arm, "exact required order")

    def test_first_full_frame_and_summaries_are_independently_recomputed(self) -> None:
        arm = self.arm("native-reference", 4)
        arm["waves"][0]["first_full_frame_order"] = list(
            reversed(arm["waves"][0]["first_full_frame_order"])
        )
        self.assert_arm_rejected(arm, "first full frame")
        arm = self.arm("native-reference", 4)
        arm["waves"][0]["slot_sets"][0] = [3, 2, 1, 0]
        self.assert_arm_rejected(arm, "slot_sets")
        arm = self.arm("native-reference", 4)
        label = arm["waves"][0]["request_order"][0]
        arm["waves"][0]["state_sha256"][label] = self.sha("wrong")
        self.assert_arm_rejected(arm, "state SHA-256 summary")

    def test_every_full_frame_preserves_request_and_slot_order(self) -> None:
        arm = self.arm("native-reference", 4)
        frame = arm["waves"][0]["frames"][3]
        for field in ("sessions", "slots", "pre_step_seq_lens", "logits"):
            frame[field] = list(reversed(frame[field]))
        arm["waves"][0]["slot_sets"][3] = list(frame["slots"])
        self.assert_arm_rejected(arm, "exact request order")

    def test_c8_retirement_reuse_and_restoration_are_ordered(self) -> None:
        arm = self.arm("native-reference", 8)
        arm["api_churn"]["cancelled"]["retired_slot"] = 6
        self.assert_arm_rejected(arm, "exact retired slot")
        arm = self.arm("native-reference", 8)
        target = arm["api_churn"]["cancelled"]["label"]
        arm["api_churn"]["cancelled"]["target_state"]["line_number"] = 45
        arm["api_churn"]["terminal_states"][target]["line_number"] = 45
        self.assert_arm_rejected(
            arm,
            "retired session re-entered|changed live owner|before cancelled-state retirement",
        )
        arm = self.arm("native-reference", 8)
        for frame in arm["api_churn"]["frames"]:
            if self.session("row-08") in frame["sessions"]:
                frame["rows"] = 7
        self.assert_arm_rejected(arm, "cardinality differs|restored a full C8")

    def test_c8_requires_final_admission_to_own_highest_live_slot(self) -> None:
        arm = self.arm("native-reference", 8)
        churn = arm["api_churn"]
        target_session = self.session("row-07")
        higher_session = self.session("row-00")
        replacement_session = self.session("row-08")
        for frame in churn["frames"]:
            frame["slots"] = [
                (
                    0
                    if session in (target_session, replacement_session)
                    else 7 if session == higher_session else slot
                )
                for session, slot in zip(frame["sessions"], frame["slots"])
            ]
        churn["session_slots"][target_session] = 0
        churn["session_slots"][higher_session] = 7
        churn["session_slots"][replacement_session] = 0
        churn["cancelled"]["retired_slot"] = 0
        self.assert_arm_rejected(arm, "highest live slot")

    def test_c8_cancelled_prefix_and_handoff_timestamps_are_exact(self) -> None:
        arm = self.arm("native-reference", 8)
        arm["api_churn"]["cancelled"]["logits"][0]["sha256"] = self.sha("wrong")
        self.assert_arm_rejected(arm, "exact C1 prefix")
        arm = self.arm("native-reference", 8)
        arm["api_churn"]["replacement_started_unix"] = 99.0
        self.assert_arm_rejected(arm, "before cancellation closed")

    def test_post_drain_must_be_isolated_and_full(self) -> None:
        arm = self.arm("native-reference", 8)
        arm["post_drain_reuse"]["frames"] = [
            copy.deepcopy(arm["api_churn"]["frames"][0])
        ]
        self.assert_arm_rejected(arm, "post-drain isolated request")
        arm = self.arm("native-reference", 8)
        arm["post_drain_reuse"]["logits_sha256_chain"] = self.sha("wrong")
        self.assert_arm_rejected(arm, "post-drain logits chain")

    def test_memory_floor_and_exact_memory_schema_fail_closed(self) -> None:
        arm = self.arm("native-reference", 2)
        arm["memory"]["mem_available_floor_kib"] -= 1
        self.assert_arm_rejected(arm, "below the arm contract")
        arm = self.arm("native-reference", 2)
        arm["memory"]["peak_mib"] = 1
        self.assert_arm_rejected(arm, "object keys differ")

    def test_full_r15_public_provenance_is_required(self) -> None:
        arm = self.arm("native-reference", 2)
        del arm["provenance"]["source_bundle_sha256"]
        self.assert_arm_rejected(arm, "object keys differ")
        arm = self.arm("native-reference", 2)
        arm["provenance"]["build_receipt_name"] = "/host/private/build.json"
        self.assert_arm_rejected(arm, "basename without host path")
        arm = self.arm("native-reference", 2)
        arm["provenance"]["release_version"] = "v0.9.13-rc.r17.20260230"
        self.assert_arm_rejected(arm, "release version is invalid")

    def test_checkpoint_id_must_be_a_path_free_public_model_id(self) -> None:
        hostile = (
            "/home/private/checkpoint",
            "home/private",
            "home/user/private/checkpoint",
            "../private",
            "owner/model/extra",
            "owner\\model",
            "owner model/name",
        )
        for checkpoint_id in hostile:
            with self.subTest(checkpoint_id=checkpoint_id):
                arm = self.arm("native-reference", 2)
                arm["provenance"]["checkpoint_id"] = checkpoint_id
                self.assert_arm_rejected(arm, "hf://owner/model")

    def test_public_model_alias_rejects_every_path_form_before_matrix_output(
        self,
    ) -> None:
        for model in (
            "/home/private/model",
            "home/private",
            "../model",
            "owner/model",
            r"C:\\Users\\private\\model",
        ):
            with self.subTest(model=model):
                arm = self.arm("native-reference", 2)
                arm["model"] = model
                arm["plan_sha256"] = matrix.canonical_hash(
                    {key: arm[key] for key in matrix.PLAN_KEYS}
                )
                self.assert_arm_rejected(arm, "path-free public API model alias")

    def test_nine_distinct_process_and_log_identities_are_required(self) -> None:
        arm = self.arm("native-reference", 2)
        arm["provenance"]["server_process"][
            "boot_id"
        ] = "00000000-0000-0000-0000-000000000000"
        self.assert_arm_rejected(arm, "boot ID is malformed")

        arms, public = self.valid_matrix()
        source = arms[("native-reference", 2)]["provenance"]["server_process"]
        candidate = arms[("native-optimized", 2)]
        candidate["provenance"]["server_process"] = copy.deepcopy(source)
        candidate["server_pid"] = source["pid"]
        with self.assertRaisesRegex(
            matrix.probe.ProbeError, "process identity is not fresh"
        ):
            matrix.validate_matrix(arms, public)

        arms, public = self.valid_matrix()
        source = arms[("native-reference", 2)]["provenance"]["server_process"]
        candidate_process = arms[("native-optimized", 2)]["provenance"][
            "server_process"
        ]
        candidate_process["log_device"] = source["log_device"]
        candidate_process["log_inode"] = source["log_inode"]
        with self.assertRaisesRegex(
            matrix.probe.ProbeError, "log device/inode identity"
        ):
            matrix.validate_matrix(arms, public)

    def test_cross_arm_provenance_and_reference_links_fail_closed(self) -> None:
        arms, public = self.valid_matrix()
        arms[("native-optimized", 8)]["provenance"]["source_bundle_sha256"] = self.sha(
            "different-source"
        )
        with self.assertRaisesRegex(matrix.probe.ProbeError, "R15 provenance differs"):
            matrix.validate_matrix(arms, public)

        arms, public = self.valid_matrix()
        arms[("native-optimized", 4)]["reference_receipt"]["sha256"] = self.sha(
            "wrong-reference"
        )
        with self.assertRaisesRegex(matrix.probe.ProbeError, "reference link differs"):
            matrix.validate_matrix(arms, public)

    def test_cross_arm_c1_semantic_drift_fails_closed(self) -> None:
        arms, public = self.valid_matrix()
        candidate = arms[("serial-control", 2)]
        mutated = copy.deepcopy(candidate["baseline_c1"][0]["token_ids"])
        mutated[0] += 1
        for request in [
            candidate["baseline_c1"][0],
            *[
                request
                for wave in candidate["waves"]
                for request in wave["requests"]
                if request["label"] == "row-00"
            ],
        ]:
            request["token_ids"] = copy.deepcopy(mutated)
            request["token_ids_sha256"] = matrix.canonical_hash(mutated)
        with self.assertRaisesRegex(
            matrix.probe.ProbeError, "C1 semantic evidence differs"
        ):
            matrix.validate_matrix(arms, public)

    def test_public_arm_receipts_must_be_distinct_and_path_free(self) -> None:
        arms, public = self.valid_matrix()
        public[("native-optimized", 2)]["sha256"] = public[("native-reference", 2)][
            "sha256"
        ]
        with self.assertRaisesRegex(matrix.probe.ProbeError, "not distinct"):
            matrix.validate_matrix(arms, public)
        arms, public = self.valid_matrix()
        public[("native-reference", 2)]["name"] = "/tmp/private.json"
        with self.assertRaisesRegex(
            matrix.probe.ProbeError, "basename without host path"
        ):
            matrix.validate_matrix(arms, public)


if __name__ == "__main__":
    unittest.main()
