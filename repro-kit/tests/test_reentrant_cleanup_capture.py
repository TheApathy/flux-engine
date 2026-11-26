"""Docker-free regressions for repeated independent stopped-container capture."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import container_lifecycle_v3 as lifecycle
import r12_container_release_runner as release
from test_r12_container_release_runner import FakeDockerBackend, make_identity, CONTAINER_ID


class ReentrantCaptureTests(unittest.TestCase):
    def adapter(self, root):
        identity = make_identity(root)
        backend = FakeDockerBackend()
        backend.containers[CONTAINER_ID] = {
            "Id": CONTAINER_ID, "Name": "/" + identity.name,
            "Image": identity.image_id, "Config": {"Labels": dict(identity.ownership_labels)},
            "State": {"Running": False}}
        adapter = release.DockerLifecycleAdapter(identity, backend=backend,
            diagnostics_dir=root / "diagnostics", sleeper=lambda _seconds: None)
        return adapter, identity, backend

    def test_three_independent_cleanup_sweeps_reuse_exact_immutable_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, identity, _backend = self.adapter(root)
            first = None
            for _ in range(3):
                proof = lifecycle.reconcile_before_unlock(adapter, identity, lambda: None,
                                                          stable_sweeps=4, max_cycles=8)
                self.assertTrue(proof.lock_safe and proof.lock_released)
                self.assertEqual(proof.stable_stopped_sweeps, 4)
                self.assertEqual(proof.errors, [])
                captures = [(path.name, path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns,
                             path.stat().st_ctime_ns) for path in sorted((root / "diagnostics").iterdir())]
                if first is None:
                    first = captures
                self.assertEqual(captures, first)
            self.assertEqual(len(first), 2)

    def test_existing_different_bytes_are_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                release._capture_once(fd, "receipt", b"original")
                with self.assertRaisesRegex(RuntimeError, "capture bytes differ"):
                    release._capture_once(fd, "receipt", b"mutated!")
                self.assertEqual((root / "receipt").read_bytes(), b"original")
            finally:
                os.close(fd)

    def test_poisoned_existing_captures_are_refused_without_blocking(self):
        for poison in ("symlink", "hardlink", "fifo", "writable", "short"):
            with self.subTest(poison=poison), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "original").write_bytes(b"known")
                (root / "original").chmod(0o400)
                target = root / "capture"
                if poison == "symlink":
                    target.symlink_to(root / "original")
                elif poison == "hardlink":
                    os.link(root / "original", target)
                elif poison == "fifo":
                    os.mkfifo(target, 0o400)
                else:
                    target.write_bytes(b"known" if poison == "writable" else b"bad")
                    target.chmod(0o600 if poison == "writable" else 0o400)
                fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    with self.assertRaises((OSError, RuntimeError)):
                        release._capture_once(fd, "capture", b"known")
                finally:
                    os.close(fd)
                self.assertEqual((root / "original").read_bytes(), b"known")

    def test_partial_prior_capture_can_complete_on_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, identity, backend = self.adapter(root)
            record = lifecycle.ContainerRecord(CONTAINER_ID, "/" + identity.name,
                                              identity.image_id, identity.ownership_labels, False)
            backend.diagnostic_failure = True
            with self.assertRaisesRegex(RuntimeError, "diagnostic server.log"):
                adapter.capture_after_stop(record)
            backend.diagnostic_failure = False
            adapter.capture_after_stop(record)
            self.assertEqual(len(list((root / "diagnostics").iterdir())), 2)

    def test_existing_capture_mutation_during_read_is_rejected(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                release._capture_once(fd, "receipt", b"known")
                original = release.os.read
                def changed(descriptor, maximum):
                    raw = original(descriptor, maximum)
                    (root / "receipt").chmod(0o600)
                    return raw
                with mock.patch.object(release.os, "read", side_effect=changed):
                    with self.assertRaisesRegex(RuntimeError, "changed during comparison"):
                        release._capture_once(fd, "receipt", b"known")
            finally:
                os.close(fd)


if __name__ == "__main__":
    unittest.main()
