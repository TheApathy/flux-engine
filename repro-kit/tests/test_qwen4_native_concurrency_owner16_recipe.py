# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise production recipe selection, arm merge, and live binding offline.

Only named production functions are loaded from AST. Docker, process files,
and descriptors are mocked. No model, compiler, server, GPU, or child runs.
These tests do not establish native parity, accuracy, or concurrency rates.
"""

import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "r15-gates/native-concurrency/native-concurrency-c2-c8"
RECIPE = ROOT / "recipes/flux-qwen4-native-concurrency-c8-owner16.env"
GENERIC = ROOT / "recipes/flux-qwen4-native-concurrency-c8.env"
ADDITIONS = {
    "ATLAS_FAST_START": "0",
    "ATLAS_FAST_LOAD_ROUTED_GROUPS": "1",
    "ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER": "1",
    "ATLAS_FAST_LOAD_DIRECT_IO_TENSOR_CAP": "20000",
    "ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS": "1",
    "ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB": "16",
    "FLUX_QWEN4_BOUNDED_TRANSPOSE_OWNERS": "1",
    "FLUX_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB": "16",
    "ATLAS_MEMAVAILABLE_FLOOR_MIB": "8192",
    "ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE": "0",
    "ATLAS_SCHED_BURST_GATHER_MS": "20",
    "ATLAS_MOE_T_LANES_WORKLIST_R8": "1",
}
FIXED_K5 = {prefix + suffix: "0"
            for prefix in ("ATLAS_QWEN4_", "FLUX_QWEN4_")
            for suffix in ("MTP_CONF", "MTP_CONF_EARLY", "MTP_VOCAB_PRUNE")}
LOG = (
    "info,atlas_bounded_transpose_allocation=debug,spark::scheduler::spec_accept_log=debug,"
    "flux_server::scheduler::spec_accept_log=debug,"
    "flux_cli::scheduler::spec_accept_log=debug"
)


def production(path, names, **injected):
    """Compile selected actual definitions without importing module side effects."""
    selected = []
    for node in ast.parse(path.read_text()).body:
        bound = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()
        if isinstance(node, ast.Assign):
            bound = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if bound & set(names):
            selected.append(node)
    found = set()
    for node in selected:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found.add(node.name)
        else:
            found.update(target.id for target in node.targets if isinstance(target, ast.Name))
    if set(names) - found:
        raise AssertionError("missing production definitions: " + repr(set(names) - found))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    namespace = {"re": re, "shlex": shlex, **injected}
    exec(compile(module, str(path), "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names}), namespace


SUITE, _ = production(ROOT / "r15-container-suite.py", ["_recipe_environment"])
PROBE, _ = production(
    ROOT / "qwen4_native_concurrency_probe.py",
    ["OPTIMIZATION_SELECTORS", "SELECTOR_PROFILES"],
)
D, NS = production(
    DRIVER,
    ["require", "LIVE_ENVIRONMENT_PREFIXES", "DIAGNOSTIC_ENVIRONMENT",
     "DIAGNOSTIC_LIMIT_KEYS", "FIXED_LAUNCH_FLAGS", "LABEL_PREFIX",
     "CONTAINER_MODELS", "live_environment_subset", "rust_log_emits_info",
     "arm_environment", "arm_argv", "bind_arm_process", "inspect_container"],
)


def parsed(path):
    return {key.decode(): value.decode() for key, value in SUITE._recipe_environment(path.read_bytes()).items()}


class Owner16RecipeTests(unittest.TestCase):
    def test_recipe_has_only_exact_owner_load_guard_and_logging_delta(self):
        generic, candidate = parsed(GENERIC), parsed(RECIPE)
        self.assertFalse(set(ADDITIONS) & set(generic))
        self.assertEqual(candidate, {**generic, **ADDITIONS, **FIXED_K5, "RUST_LOG": LOG})
        self.assertEqual(candidate["ATLAS_QWEN4_PREFILL_WIDE"], "0")
        self.assertNotIn("FLUX_QWEN4_PREFILL_WIDE", candidate)

    def test_literal_parser_refuses_duplicate_or_dynamic_owner_selection(self):
        for suffix in (b"ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB=64\n",
                       b"ATLAS_EXTRA=$DYNAMIC\n", b"ATLAS_EXTRA=`dynamic`\n"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                SUITE._recipe_environment(RECIPE.read_bytes() + suffix)

    def test_production_arm_merge_retains_owner16_for_all_plain_profiles(self):
        for profile in PROBE.SELECTOR_PROFILES:
            with self.subTest(profile=profile):
                overrides, live = D.arm_environment(
                    PROBE, parsed(RECIPE), {"PATH": "/usr/bin", "ATLAS_FAST_START": "1"},
                    "none", profile,
                )
                for key, value in ADDITIONS.items():
                    self.assertEqual(overrides[key], value)
                    self.assertEqual(live[key], value)
                for key, value in FIXED_K5.items():
                    self.assertEqual(overrides[key], value)
                    self.assertEqual(live[key], value)
                self.assertEqual(overrides["ATLAS_DISABLE_SPECULATION"], "1")
                self.assertEqual(overrides["RUST_LOG"], LOG)
                self.assertNotIn("PATH", live)
                for key, value in PROBE.SELECTOR_PROFILES[profile].items():
                    self.assertEqual(live[key], value)
                for key, value in D.DIAGNOSTIC_ENVIRONMENT.items():
                    self.assertEqual(live[key], value)

    def test_production_merge_refuses_recipe_or_image_qualification_key(self):
        reserved = (set(PROBE.SELECTOR_PROFILES["native-reference"])
                    | set(D.DIAGNOSTIC_ENVIRONMENT) | set(D.DIAGNOSTIC_LIMIT_KEYS))
        for key in reserved:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "pins a qualification selector"):
                    D.arm_environment(PROBE, {**parsed(RECIPE), key: "1"}, {}, "none", "native-reference")
                with self.assertRaisesRegex(ValueError, "pins a qualification selector"):
                    D.arm_environment(PROBE, parsed(RECIPE), {key: "1"}, "none", "native-reference")

    def test_production_plain_argv_has_exact_width_and_no_proposer(self):
        args = types.SimpleNamespace(container_port=8888)
        plan = {"in_container_binary_path": "/opt/flux/bin/flux-server-real"}
        for width in (2, 4, 8):
            argv = D.arm_argv(plan, args, "qad5500-owner16-plain", width, "none")
            self.assertEqual(argv[argv.index("--max-batch-size") + 1], str(width))
            self.assertEqual(argv[argv.index("--max-num-seqs") + 1], str(width))
            self.assertEqual(argv[argv.index("--max-seq-len") + 1], "2048")
            self.assertEqual(argv[argv.index("--max-prefill-tokens") + 1], "1024")
            for forbidden in ("--speculative", "--num-drafts", "--dflash", "--self-speculative"):
                self.assertNotIn(forbidden, argv)

    def bind(self, observed):
        _, expected = D.arm_environment(PROBE, parsed(RECIPE), {}, "none", "native-reference")
        manifest = {"argv": ["/opt/flux/bin/flux-server-real", "serve"],
                    "cwd": "/", "environment": expected}
        files = {
            "/proc/123/cmdline": b"\0".join(item.encode() for item in manifest["argv"]) + b"\0",
            "/proc/123/environ": b"\0".join((key + "=" + value).encode() for key, value in observed.items()) + b"\0",
            "/proc/sys/kernel/random/boot_id": "12345678-1234-1234-1234-123456789ab0",
        }
        class FakePath:
            def __init__(self, name):
                self.name = name
            def read_bytes(self):
                return files[self.name]
            def read_text(self, *args):
                return files[self.name]
        NS["Path"] = FakePath
        NS["os"] = types.SimpleNamespace(readlink=lambda path: "/")
        probe = types.SimpleNamespace(validate_log_binding=lambda log, pid: (7, 8),
                                      process_start_ticks=lambda pid: 90)
        return D.bind_arm_process(probe, 123, manifest, "/private/mock.log")

    def test_production_live_binding_accepts_exact_full_owner16_environment(self):
        _, expected = D.arm_environment(PROBE, parsed(RECIPE), {}, "none", "native-reference")
        self.assertEqual(self.bind(expected)["start_ticks"], 90)

    def test_production_live_binding_refuses_owner_or_loading_drift(self):
        _, expected = D.arm_environment(PROBE, parsed(RECIPE), {}, "none", "native-reference")
        for key in ADDITIONS:
            observed = dict(expected)
            observed.pop(key)
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "environment differs"):
                self.bind(observed)
        for changes in ({"ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB": "64"},
                        {"ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER": "0"},
                        {"ATLAS_HIDDEN_UNBOUND": "1"}, {"ATLAS_DISABLE_SPECULATION": "0"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "environment differs"):
                self.bind({**expected, **changes})

    def select(self, label_hash):
        raw = RECIPE.read_bytes()
        recipe_path = "/opt/flux/repro-kit/recipes/" + RECIPE.name
        labels = {D.LABEL_PREFIX + "binary-path": "/opt/flux/bin/flux-server-real",
                  D.LABEL_PREFIX + "recipe-path": recipe_path,
                  D.LABEL_PREFIX + "binary-sha256": "b" * 64,
                  D.LABEL_PREFIX + "recipe-sha256": label_hash}
        document = {"Id": "c" * 64, "Image": "sha256:" + "d" * 64,
                    "State": {"Running": True, "Pid": 123},
                    "HostConfig": {"ReadonlyRootfs": True},
                    "Config": {"Labels": labels, "Env": ["PATH=/usr/bin"]}}
        NS["docker"] = lambda argv: json.dumps(document).encode()
        NS["_container_mounts"] = lambda value: {"/results": {}, "/models": {}}
        NS["os"] = types.SimpleNamespace(open=lambda *args: 10, close=lambda fd: None,
                                         O_RDONLY=0, O_DIRECTORY=1, O_CLOEXEC=2)
        opened = []
        def open_recipe(root, path):
            opened.append(path)
            return 11
        suite = types.SimpleNamespace(
            strict_json_loads=lambda value, label: json.loads(value),
            normalized_absolute=lambda value, label: value,
            exact_hex=lambda *args: None, exact_int=lambda value, *args: value,
            _open_process_file=open_recipe,
            _hash_regular_fd=lambda *args: (hashlib.sha256(raw).hexdigest(), raw),
            _recipe_environment=SUITE._recipe_environment, MAX_PROGRAM_BYTES=1 << 20,
        )
        args = types.SimpleNamespace(container_id=document["Id"], image_id=document["Image"])
        result = D.inspect_container(args, suite)
        self.assertEqual(opened, [recipe_path])
        return result

    def test_production_image_selection_binds_dedicated_path_and_exact_bytes(self):
        digest = hashlib.sha256(RECIPE.read_bytes()).hexdigest()
        plan, recipe, _, _ = self.select(digest)
        self.assertEqual(plan["recipe_sha256"], digest)
        for key, value in ADDITIONS.items():
            self.assertEqual(recipe[key], value)

    def test_production_image_selection_refuses_mismatched_recipe_digest(self):
        with self.assertRaisesRegex(ValueError, "image recipe hash differs"):
            self.select(hashlib.sha256(GENERIC.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
