# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the production live-process boot-id predicate with mocked files."""

import ast
from pathlib import Path
import re
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "r15-gates/native-concurrency/native-concurrency-c2-c8"


def bind(boot_id):
    names = {"require", "bind_arm_process", "LIVE_ENVIRONMENT_PREFIXES"}
    nodes = []
    for node in ast.parse(DRIVER.read_text()).body:
        defined = {node.name} if isinstance(node, ast.FunctionDef) else set()
        if isinstance(node, ast.Assign):
            defined = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if defined & names:
            nodes.append(node)
    manifest = {"argv": ["/opt/flux/bin/flux-server-real", "serve"],
                "cwd": "/", "environment": {"ATLAS_QWEN4_NATIVE_BATCH": "1"}}
    files = {"/proc/123/cmdline": b"/opt/flux/bin/flux-server-real\0serve\0",
             "/proc/123/environ": b"ATLAS_QWEN4_NATIVE_BATCH=1\0",
             "/proc/sys/kernel/random/boot_id": boot_id}
    class FakePath:
        def __init__(self, path):
            self.path = path
        def read_bytes(self):
            return files[self.path]
        def read_text(self, *args):
            return files[self.path]
    namespace = {"re": re, "Path": FakePath,
                 "os": types.SimpleNamespace(readlink=lambda path: "/")}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(DRIVER), "exec"), namespace)
    probe = types.SimpleNamespace(validate_log_binding=lambda log, pid: (7, 8),
                                  process_start_ticks=lambda pid: 90)
    return namespace["bind_arm_process"](probe, 123, manifest, "/private/mock.log")


class BootIdTests(unittest.TestCase):
    def test_valid_nonzero_uuid_without_zero_character_is_accepted(self):
        value = "12345678-1234-1234-1234-123456789abc"
        self.assertNotIn("0", value)
        self.assertEqual(bind(value)["boot_id"], value)

    def test_valid_uuid_with_zero_character_is_accepted(self):
        value = "12345678-1234-1234-1234-123456789ab0"
        self.assertEqual(bind(value)["boot_id"], value)

    def test_all_zero_or_malformed_uuid_is_still_refused(self):
        for value in ("00000000-0000-0000-0000-000000000000", "", "-", "f" * 36,
                      "12345678-1234-1234-1234-123456789abg",
                      "12345678-1234-1234-1234-123456789abc-extra"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boot id is malformed"):
                bind(value)


if __name__ == "__main__":
    unittest.main()
