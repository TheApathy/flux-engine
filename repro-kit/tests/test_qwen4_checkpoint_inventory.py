# SPDX-License-Identifier: AGPL-3.0-only

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen4_checkpoint_inventory.py"
SPEC = importlib.util.spec_from_file_location("qwen4_checkpoint_inventory", MODULE_PATH)
assert SPEC and SPEC.loader
I = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(I)


class CheckpointInventoryTests(unittest.TestCase):
    def test_inventory_is_sorted_complete_and_content_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            (root / "nested").mkdir(parents=True)
            (root / "z.bin").write_bytes(b"z")
            (root / "nested/a.bin").write_bytes(b"alpha")
            document = I.inventory(root, "JMNI-Labs/QAD5500", "a" * 64)
            self.assertEqual(document["schema"], I.SCHEMA)
            self.assertEqual(
                [entry["path"] for entry in document["files"]],
                ["nested/a.bin", "z.bin"],
            )
            self.assertEqual(document["file_count"], 2)
            self.assertEqual(document["total_bytes"], 6)

    def test_main_writes_once_outside_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "model"
            root.mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            output = base / "inventory.json"
            args = [
                "--checkpoint-root", str(root),
                "--checkpoint-id", "JMNI-Labs/QAD5500",
                "--checkpoint-revision", "b" * 64,
                "--output", str(output),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(I.main(args), 0)
            value = json.loads(output.read_text(encoding="ascii"))
            self.assertEqual(value["file_count"], 1)
            with self.assertRaises(FileExistsError):
                with redirect_stdout(io.StringIO()):
                    I.main(args)

    def test_output_inside_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                I.main(
                    [
                        "--checkpoint-root", str(root),
                        "--checkpoint-id", "id",
                        "--checkpoint-revision", "c" * 40,
                        "--output", str(root / "inventory.json"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
